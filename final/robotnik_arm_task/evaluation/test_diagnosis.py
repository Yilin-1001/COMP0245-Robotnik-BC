"""Bottleneck diagnosis of a model family on the 64 test scenarios, outside the frozen test protocol.

This is a diagnostic peek at the test scenes, run at the user's request before DAgger is designed. It writes only to
--out and never to results/results.csv, so the frozen-protocol gate in evaluation/evaluate.py is untouched. Anything
designed after reading these results must be re-validated before it is reported on the same scenes.

Every rollout uses evaluation.evaluate.rollout (identical predicates: reach < 10 mm after 8 s, 3 s more, final 1 s
all < 10 mm, 40 s cap). A wrapper policy records, at every 1 kHz step, the true EE error, teacher A's reach gate and
base errors, and teacher A's label at the visited state, so action errors can be split by phase:
  nav       gate closed (the base is moving, the arm should hold its home pose)
  approach  gate open, true EE error >= 10 mm
  hold      gate open, true EE error < 10 mm
Modes: 'teacher' (teacher A; every model also predicts on its states, the expert-distribution error), 'student'
(full student), 'arm' (student arm, teacher wheels), 'base' (student wheels, teacher arm).
A second pass hands each failed full-student episode to teacher A at the 40 s cap for 15 s (can the teacher recover
from the states the student reaches, i.e. are DAgger labels there useful?).

Usage (from final/robotnik_arm_task/):
    python -m evaluation.test_diagnosis --models <s0.npz> <s1.npz> <s2.npz> --out results/diagnostics/<dir>
"""
import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse
import csv
import functools
import json
from pathlib import Path

import numpy as np
import yaml

from core.interfaces import Action, Policy

TASK_DIR = Path(__file__).resolve().parents[1]
TOL = 0.01
TRACE_EVERY = 20                  # saved traces at 50 Hz; statistics use every step
TAKEOVER_S = 15.0
PHASES = ("nav", "approach", "hold")


class Recorder(Policy):
    """Acts as the chosen mode and records teacher labels and model predictions at every visited state."""

    def __init__(self, mode: str, models: list[str]):
        from evaluation.evaluate import MarkovianReference, mlp_factory
        self.mode = mode
        self.expert = MarkovianReference().expert
        self.models = [mlp_factory(m) for m in models]  # 'teacher' mode: all models; otherwise exactly one
        self.meta = self.models[0].meta
        self.env = None

    def bind_env(self, env):
        self.env = env

    def reset(self):
        for m in self.models:
            m.reset()
        self.rows, self.pred = [], []
        self.last_raw = None

    def act(self, obs):
        wheels_t, gate = self.expert.base_command(obs)
        torque_t = self.expert.arm_torque(obs, gate)
        dist_err, heading_err = self.expert.base_errors(obs)
        err = float(np.linalg.norm(obs.target_ee_pos - self.env.ee_pos()))
        if self.mode == "teacher":
            raws = [m.predict(m.features(obs)) for m in self.models]
            action = Action(torque_t, wheels_t)
        else:
            a = self.models[0].act(obs)
            raws = [self.models[0].last_raw]
            self.last_raw = raws[0]
            action = Action(a.torque_arm if self.mode in ("student", "arm") else torque_t,
                            a.wheel_cmds if self.mode in ("student", "base") else wheels_t)
        self.rows.append(np.r_[self.env.t, err, float(gate), dist_err, heading_err, np.abs(obs.qd_mes).max(),
                               torque_t, wheels_t[:2], obs.base_pos[:2], obs.base_bearing])
        self.pred.append(np.stack(raws))
        return action


def _stats(rows: np.ndarray, pred: np.ndarray) -> dict:
    """Per-phase action errors (model clipped to the interface limits vs the teacher label) and timing markers."""
    t, err, gate, dist_err, heading_err, qd_max = rows[:, :6].T
    torque_t, wheel_t = rows[:, 6:13], rows[:, 13:15]
    phase = np.where(gate < 0.5, 0, np.where(err >= TOL, 1, 2))
    out = {}
    first = lambda mask: float(t[np.argmax(mask)]) if mask.any() else None
    opened = gate > 0.5
    out["t_gate_open"] = first(opened)
    out["gate_closings_after_open"] = int(np.sum((gate[1:] < 0.5) & (gate[:-1] > 0.5)))
    out["err_at_gate_open_mm"] = float(err[np.argmax(opened)] * 1000) if opened.any() else None
    for mm in (150, 50, 10):
        out[f"t_within_{mm}mm"] = first(err < mm / 1000)
    out["max_joint_speed"] = float(qd_max.max())
    out["final_dist_err_m"], out["final_heading_err_rad"] = float(dist_err[-1]), float(heading_err[-1])
    for i in range(pred.shape[1]):
        tau = np.clip(pred[:, i, :7], -30, 30) - torque_t
        wh = np.clip(pred[:, i, 7:9], -6, 6) - wheel_t
        sfx = "" if pred.shape[1] == 1 else f"_m{i}"
        for p, name in enumerate(PHASES):
            m = phase == p
            out[f"n_{name}"] = int(m.sum())
            out[f"arm_rmse_{name}{sfx}"] = float(np.sqrt(np.mean(tau[m] ** 2))) if m.any() else None
            out[f"wheel_rmse_{name}{sfx}"] = float(np.sqrt(np.mean(wh[m] ** 2))) if m.any() else None
        last = t >= t[-1] - 1.0  # final second: signed torque bias per joint, the error the teacher cannot absorb
        bias = tau[last].mean(axis=0)
        out[f"final1s_bias_max_abs_nm{sfx}"] = float(np.abs(bias).max())
        out[f"final1s_bias_nm{sfx}"] = json.dumps(np.round(bias, 3).tolist())
    return out


def _job(args):
    mode, models, scenario, config = args
    from evaluation.evaluate import make_robot_env, rollout
    rec = Recorder(mode, models)
    outcome = rollout(rec, scenario, config, functools.partial(make_robot_env, TOL))
    rows, pred = np.array(rec.rows), np.array(rec.pred)
    trace = {"rows": rows[::TRACE_EVERY].astype(np.float32), "pred": pred[::TRACE_EVERY].astype(np.float32)}
    return mode, models, outcome, _stats(rows, pred), trace


def _takeover(args):
    """Full student to the 40 s cap, then teacher A for TAKEOVER_S: does the teacher recover the student's state?"""
    model, scenario, config = args
    from evaluation.evaluate import MarkovianReference, make_robot_env, mlp_factory
    student, expert = mlp_factory(model), MarkovianReference().expert
    env = make_robot_env(TOL)
    try:
        obs = env.reset(scenario)
        student.reset()
        cap, extra = int(config["evaluation"]["episode_cap_s"] * 1000), int(TAKEOVER_S * 1000)
        errs = []
        for k in range(cap + extra):
            a = student.act(obs) if k < cap else expert.act(obs)
            obs, info = env.step(Action(np.clip(a.torque_arm, -30, 30), np.clip(a.wheel_cmds, -6, 6)))
            if k >= cap:
                errs.append(info["ee_err"])
            if not np.isfinite(info["ee_err"]):
                break
    finally:
        env.close()
    errs = np.array(errs)
    inside = errs < TOL
    # first time from which the error stays inside for the rest of the takeover
    stays = np.flip(np.cumprod(np.flip(inside))).astype(bool)
    return {"model": Path(model).stem, "scenario_id": scenario.scenario_id, "err_at_switch_mm": float(errs[0] * 1000),
            "final_err_mm": float(errs[-1] * 1000), "recovered_and_held": bool(stays.any() and errs.size >= 1000
                                                                              and stays[-1000:].all()),
            "t_to_settle_s": float(np.argmax(stays) / 1000) if stays.any() else None}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True, help="student NumPy exports (.npz)")
    p.add_argument("--modes", nargs="+", default=["teacher", "student", "arm", "base"])
    p.add_argument("--scenarios", nargs="*", help="test scenario ids (default all 64)")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    import multiprocessing as mp
    from evaluation.test_sets import get_split
    from experiments.ablation.run import _init_worker
    config = yaml.safe_load((Path(args.models[0]).parent / "config.yaml").read_text())
    scenarios = get_split("test", TASK_DIR / "experiments/splits/clean253.json", "A", allow_test=True)
    if args.scenarios:
        scenarios = [s for s in scenarios if s.scenario_id in set(args.scenarios)]
    jobs = [(mode, tuple(args.models) if mode == "teacher" else (m,), s, config)
            for mode in args.modes for m in (args.models[:1] if mode == "teacher" else args.models) for s in scenarios]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "traces").mkdir(exist_ok=True)
    print(f"{len(jobs)} rollouts", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(args.workers, _init_worker) as pool:
        for i, (mode, models, outcome, stats, trace) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1), 1):
            policy = "teacher-A" if mode == "teacher" else Path(models[0]).stem
            rows.append({"mode": mode, "policy": policy, **outcome, **stats})
            if mode in ("teacher", "student"):
                np.savez_compressed(args.out / "traces" / f"{mode}__{policy}__{outcome['scenario_id']}.npz", **trace)
            if i % 32 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)
        rows.sort(key=lambda r: (r["mode"], r["policy"], r["scenario_id"]))
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out / "rollouts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        by_id = {s.scenario_id: s for s in scenarios}
        failed = [(m, by_id[r["scenario_id"]], config) for r in rows if r["mode"] == "student"
                  and not r["sustained_hold_pass"] for m in args.models if Path(m).stem == r["policy"]]
        print(f"takeover: {len(failed)} failed student episodes", flush=True)
        takeovers = pool.map(_takeover, failed, chunksize=1) if failed else []
    if takeovers:
        with open(args.out / "takeover.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(takeovers[0]))
            w.writeheader()
            w.writerows(takeovers)
    (args.out / "run.json").write_text(json.dumps({"models": args.models, "modes": args.modes, "n_scenarios": len(scenarios),
                                                   "note": "diagnostic test peek; not in results/results.csv"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
