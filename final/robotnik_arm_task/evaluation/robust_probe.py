"""Robustness probe: teacher A and students under physics changes, sensor noise and external pushes.

In-range scenes (start x, y in +-0.5 m, any heading and azimuth, target 0.5-3 m away and 0.1-1.25 m high), fresh
development seeds 990000+ (disjoint from every frozen split and the other probes). Each condition changes one thing:

  nominal                 no change (control)
  friction x0.5 / x1.5    floor friction
  mass x0.9 / x1.1        every link's mass and inertia (teacher A's gravity model stays nominal)
  sensor noise x20        Gaussian noise of std 2e-3 on every observed quantity
  push base, driving      120 N for 0.2 s on the base at t = 3 s, random horizontal direction (episode cap 60 s)
  push base, at target    the same push 1 s after the end effector first enters 10 mm
  push hand, at target    20 N for 0.2 s on the end effector 1 s after it first enters 10 mm

Standard predicates (reach < 10 mm after 8 s, all of the final 1 s < 10 mm), except that the post-reach window is
10 s for the at-target pushes, so the push lands inside it and the final second judges the recovery. Teacher A runs
the same scenes as the reference; students are also scored on the scenes the teacher holds. Selects nothing.

Usage (from final/robotnik_arm_task/):
    python -m evaluation.robust_probe --models <a.npz> <b.npz> ... [--n 12] [--workers 22] [--out <dir>]
"""
import argparse
import copy
import csv
import functools
import json
from pathlib import Path

import numpy as np
import yaml

TASK_DIR = Path(__file__).resolve().parents[1]
SEED_BASE = 990_000
BASE_PUSH, EE_PUSH, PUSH_S = 120.0, 20.0, 0.2
# name -> (physics, push kind or None, evaluation overrides)
CONDITIONS = {
    "nominal": ({}, None, {}),
    "friction x0.5": ({"friction_scale": 0.5}, None, {}),
    "friction x1.5": ({"friction_scale": 1.5}, None, {}),
    "mass x0.9": ({"mass_scale": 0.9}, None, {}),
    "mass x1.1": ({"mass_scale": 1.1}, None, {}),
    "sensor noise x20": ({"sensor_noise_scale": 20.0}, None, {}),
    "push base, driving": ({}, "base_drive", {"episode_cap_s": 60.0}),
    "push base, at target": ({}, "base_hold", {"post_reach_s": 10.0}),
    "push hand, at target": ({}, "ee_hold", {"post_reach_s": 10.0}),
}


def scenes(condition: str, n: int) -> list:
    from data_collection.scenarios import Scenario
    index = list(CONDITIONS).index(condition)
    physics, push, _ = CONDITIONS[condition]
    out = []
    for i in range(n):
        rng = np.random.default_rng([SEED_BASE, index, i])
        start = rng.uniform(-0.5, 0.5, 2)
        yaw, azimuth = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
        d, z = rng.uniform(0.5, 3.0), rng.uniform(0.1, 1.25)
        target = np.r_[start + d * np.array([np.cos(yaw + azimuth), np.sin(yaw + azimuth)]), z]
        ph = dict(physics)
        if push:
            a = rng.uniform(-np.pi, np.pi)
            force = (BASE_PUSH if push.startswith("base") else EE_PUSH) * np.array([np.cos(a), np.sin(a), 0.0])
            ph["push"] = {"link": "base" if push.startswith("base") else "ee", "force": force.tolist(), "duration": PUSH_S,
                          **({"start": 3.0} if push == "base_drive" else {"after_reach": 1.0})}
        out.append(Scenario(seed=SEED_BASE + 1000 * index + i, target_ee_pos=target, base_start_pos=np.r_[start, 0.0],
                            base_start_bearing=float(yaw), physics=ph, scenario_id=f"{condition}-{i:02d}", split="probe"))
    return out


def _job(args):
    policy, condition, scenario, config = args
    from evaluation.evaluate import MarkovianReference, make_robot_env, mlp_factory, rollout
    p = MarkovianReference() if policy == "teacher-A" else mlp_factory(policy)
    out = rollout(p, scenario, config, functools.partial(make_robot_env, config["evaluation"]["reach_tolerance_m"]))
    return policy, condition, out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True, help="student NumPy exports (.npz)")
    p.add_argument("--n", type=int, default=12, help="scenes per condition")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--out", type=Path, default=TASK_DIR / "results/robust_probe/v1")
    args = p.parse_args(argv)
    import multiprocessing as mp
    from experiments.ablation.run import _init_worker
    base = yaml.safe_load((Path(args.models[0]).parent / "config.yaml").read_text())
    jobs = []
    for c, (_, _, overrides) in CONDITIONS.items():
        config = copy.deepcopy(base)
        config["evaluation"].update(overrides)
        jobs += [(pol, c, s, config) for s in scenes(c, args.n) for pol in ["teacher-A", *args.models]]
    print(f"{len(jobs)} rollouts", flush=True)
    with mp.get_context("spawn").Pool(args.workers, _init_worker) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [{"policy": Path(pol).stem if pol != "teacher-A" else pol, "condition": c, **o} for pol, c, o in results]
    with open(args.out / "rollouts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = ["teacher-A"] + [Path(m).stem for m in args.models]
    summary = {}
    for c in CONDITIONS:
        teacher = {r["scenario_id"]: r["sustained_hold_pass"] for r in rows if r["condition"] == c and r["policy"] == "teacher-A"}
        entry = {}
        for name in names:
            outs = [r for r in rows if r["condition"] == c and r["policy"] == name]
            feasible = [r for r in outs if teacher[r["scenario_id"]]]
            entry[name] = {"hold": f"{sum(r['sustained_hold_pass'] for r in outs)}/{len(outs)}",
                           "hold_where_teacher_holds": f"{sum(r['sustained_hold_pass'] for r in feasible)}/{len(feasible)}",
                           "median_final_err_mm": round(float(np.median([r["final_err_mm"] for r in outs])), 2)}
        summary[c] = entry
    (args.out / "summary.json").write_text(json.dumps({"n_per_condition": args.n, "conditions": summary}, indent=1))
    print(f"\n{args.n} scenes per condition; sustained hold")
    print(f"{'condition':<22} " + " ".join(f"{n[:22]:>23}" for n in names))
    for c, entry in summary.items():
        print(f"{c:<22} " + " ".join(f"{entry[n]['hold']:>23}" for n in names))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
