"""Closed-loop evaluation shared by every policy, and the compact versioned result index.

Predicate (docs/mlp_baseline_training_plan.md §6, identical for students and expert references):
- run at 1 kHz from the scenario's start; the evaluator alone decides termination (env.step()['done'] is ignored);
- task reach: true EE error below 10 mm at an observation with t > 8 s; then exactly 3 s more are executed;
- sustained hold: every observation in the final inclusive 1-second window is below 10 mm;
- 40 s cap. True EE error is read by this metric code only; policies receive the Observation alone.
Actions are clipped to the frozen interface limits (±30 N·m, ±6 rad/s) before RobotEnv.step, which applies
wheel speeds then arm torques. Workers return outcomes; only the parent process writes files.

Usage (from final/robotnik_arm_task/):
    python -m evaluation.evaluate --policy mlp --artifact checkpoints/.../inference.npz --split screening --record
    python -m evaluation.evaluate --policy expert --teacher A --split validation --record
"""
import os

if __name__ == "__main__":  # rollout processes: single-threaded BLAS before numerical imports
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse
import csv
import datetime
import functools
import json
import time
from pathlib import Path

import numpy as np

from core.interfaces import Action, Policy

TASK_DIR = Path(__file__).resolve().parents[1]
EVALUATION_ID = "rollout-v1"
RESULT_COLUMNS = ["run_id", "regime", "teacher", "objective", "cohort", "architecture", "lr", "seed", "fit_stage",
                  "config_hash", "split_sha256", "checkpoint_step", "checkpoint_path", "checkpoint_sha256",
                  "inference_path", "evaluation_split", "evaluation_id", "n", "sustained_hold_success",
                  "reach_success", "median_final_err_mm", "max_final1s_err_mm", "median_t_first_reach_s",
                  "median_first_arrival_s", "mean_saturation_frac", "mean_torque_variation_nm", "latency_p50_us",
                  "latency_p95_us", "latency_p99_us", "failures", "detail_path", "written_utc"]
KEY = ("run_id", "checkpoint_step", "evaluation_split", "evaluation_id")
LEGACY_COLUMNS = ["run_id", "policy", "ckpt", "split", "n", "success_10mm", "time_to_reach", "final_err", "path_len",
                  "torque_jerk", "finger_drift"]


# ---------------------------------------------------------------- rollout

def rollout(policy: Policy, scenario, config: dict, env_factory) -> dict:
    """One scenario in a fresh environment; never raises (failures are recorded) and always closes the env."""
    ev, itf = config["evaluation"], config["interface"]
    dt = 1.0 / itf["control_hz"]
    cap, post, hold = (int(round(ev[k] / dt)) for k in ("episode_cap_s", "post_reach_s", "hold_window_s"))
    tol, min_t = ev["reach_tolerance_m"], ev["min_reach_time_s"]
    tlim, wlim = float(itf["torque_limit"]), float(itf["wheel_limit"])
    target = np.asarray(scenario.target_ee_pos, dtype=float)
    errs, lat, dtau = [], [], []
    sat = T = 0
    first_reach = first_arrival = prev = failure = None
    env = None
    t_now = 0.0
    try:
        policy_itf = (getattr(policy, "meta", None) or {}).get("interface")
        if policy_itf is not None:
            for key in ("torque_limit", "wheel_limit", "control_hz"):
                if float(policy_itf[key]) != float(itf[key]):
                    raise ValueError(f"policy interface {key} {policy_itf[key]} != evaluation interface {itf[key]}")
        env = env_factory()
        obs = env.reset(scenario)
        if hasattr(policy, "bind_env"):
            policy.bind_env(env)
        policy.reset()
        errs.append(float(np.linalg.norm(target - env.ee_pos())))
        for k in range(cap):
            t_k = float(env.t)
            if first_arrival is None and errs[k] < tol:
                first_arrival = t_k
            if first_reach is None and errs[k] < tol and t_k > min_t:
                first_reach = k
            start = time.perf_counter_ns()
            action = policy.act(obs)
            lat.append((time.perf_counter_ns() - start) / 1000.0)
            torque, wheels = np.asarray(action.torque_arm, dtype=float), np.asarray(action.wheel_cmds, dtype=float)
            if torque.shape != (7,) or wheels.shape != (4,) or not (np.isfinite(torque).all() and np.isfinite(wheels).all()):
                raise FloatingPointError(f"invalid action at step {k}")
            raw = getattr(policy, "last_raw", None)
            pre_torque = np.asarray(raw[:7]) if raw is not None else torque
            sat += bool(np.any(np.abs(pre_torque) >= tlim))
            executed = Action(np.clip(torque, -tlim, tlim), np.clip(wheels, -wlim, wlim))
            if prev is not None:
                dtau.append(float(np.abs(executed.torque_arm - prev).mean()))
            prev = executed.torque_arm
            obs, info = env.step(executed)
            T = k + 1
            t_now = float(env.t)
            errs.append(float(info["ee_err"]))
            if not (np.isfinite(obs.q_mes).all() and np.isfinite(obs.base_pos).all() and np.isfinite(errs[-1])):
                raise FloatingPointError(f"non-finite observation after step {k}")
            if first_reach is not None and T - first_reach >= post:
                break
    except Exception as e:  # recorded, never silently dropped
        failure = f"{type(e).__name__}: {e}"
    finally:
        if env is not None:
            env.close()
    err = np.array(errs) if errs else np.array([np.inf])
    window = err[max(0, T - hold):T + 1]
    complete = failure is None and first_reach is not None and T - first_reach >= post
    termination = (f"failure: {failure}" if failure else "post_reach_window_complete" if complete else
                   "time_cap_post_reach_window_incomplete" if first_reach is not None else "time_cap_no_task_reach")
    pct = lambda q: float(np.percentile(lat, q)) if lat else None
    return {"scenario_id": scenario.scenario_id, "steps": T, "duration_s": t_now, "ever_reached": first_reach is not None,
            "first_reach_step": first_reach, "t_first_reach": None if first_reach is None else first_reach * dt,
            "first_arrival_t": first_arrival, "post_reach_window_complete": bool(complete),
            "sustained_hold_pass": bool(complete and np.all(window < tol)), "final_err_mm": float(err[-1] * 1000),
            "final1s_max_err_mm": float(window.max() * 1000), "final1s_mean_err_mm": float(window.mean() * 1000),
            "min_err_mm": float(err.min() * 1000), "saturation_frac": sat / T if T else None,
            "torque_variation_nm": float(np.mean(dtau)) if dtau else None,
            "latency_p50_us": pct(50), "latency_p95_us": pct(95), "latency_p99_us": pct(99),
            "failure": failure, "termination": termination}


# ---------------------------------------------------------------- environment and expert adapters

def make_robot_env(reach_tolerance: float = 0.01):
    from core.env import RobotEnv
    return RobotEnv(reach_tolerance=reach_tolerance)


class MarkovianReference(Policy):
    """Teacher A at its recorded lead time; a function of the raw observation."""
    name = "expert-A"

    def __init__(self, lead_time: float = 0.50):
        from data_collection.expert import MarkovianExpert, MarkovianExpertParams
        self.expert = MarkovianExpert(MarkovianExpertParams(lead_time=lead_time))

    def act(self, obs):
        return self.expert.act(obs)


class StatefulReference(Policy):
    """Teacher B (reference runs only): uses its genuine teacher-side state and privileged simulator inputs."""
    name = "expert-B"

    def __init__(self):
        self.env = self.teacher = None
        self._fresh = True

    def bind_env(self, env):
        from data_collection.template_teacher import TemplateTeacher
        if self.env is not env:
            self.env, self.teacher = env, TemplateTeacher(env)

    def reset(self):
        self._fresh = True

    def act(self, obs):
        if self._fresh:
            self.teacher.reset(target=self.env.scenario.target_ee_pos, q_hold=obs.q_mes)
            self._fresh = False
        return self.teacher.act(obs)[0]


def expert_factory(teacher: str):
    return MarkovianReference() if teacher == "A" else StatefulReference()


def mlp_factory(path: str, backend: str = "numpy", expected: dict | None = None):
    from training.common.policies import MLPPolicy
    return MLPPolicy.load(path, backend=backend, expected=expected)


# ---------------------------------------------------------------- parallel evaluation

_worker = {}


def _init_worker(policy_factory, env_factory):
    from training.common.config import pin_rollout_threads
    pin_rollout_threads()
    try:
        from threadpoolctl import threadpool_limits
        _worker["limits"] = threadpool_limits(1)
    except ImportError:
        pass
    import sys
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(1)
    _worker["policy"], _worker["env_factory"], _worker["config"] = policy_factory(), env_factory, None


def _run_one(args):
    scenario, config = args
    return rollout(_worker["policy"], scenario, config, _worker["env_factory"])


def evaluate_policy(policy_factory, scenarios: list, config: dict, env_factory=None, workers: int | None = None) -> list[dict]:
    """Per-scenario outcomes in scenario order. workers <= 1 runs in-process; otherwise spawned workers."""
    env_factory = env_factory or functools.partial(make_robot_env, config["evaluation"]["reach_tolerance_m"])
    workers = config["evaluation"].get("workers", 0) if workers is None else workers
    if not workers or workers <= 1 or len(scenarios) <= 1:
        policy = policy_factory()
        return [rollout(policy, s, config, env_factory) for s in scenarios]
    import multiprocessing as mp
    from training.common.config import pin_rollout_threads
    pin_rollout_threads()
    with mp.get_context("spawn").Pool(min(workers, len(scenarios)), _init_worker, (policy_factory, env_factory)) as pool:
        return pool.map(_run_one, [(s, config) for s in scenarios], chunksize=1)


def summarise(outcomes: list[dict]) -> dict:
    def med(key):
        v = [o[key] for o in outcomes if o[key] is not None]
        return float(np.median(v)) if v else None

    def mean(key):
        v = [o[key] for o in outcomes if o[key] is not None]
        return float(np.mean(v)) if v else None
    p99 = [o["latency_p99_us"] for o in outcomes if o["latency_p99_us"] is not None]
    return {"n": len(outcomes), "sustained_hold_success": float(np.mean([o["sustained_hold_pass"] for o in outcomes])),
            "reach_success": float(np.mean([o["ever_reached"] for o in outcomes])),
            "median_final_err_mm": med("final_err_mm"), "max_final1s_err_mm": max(o["final1s_max_err_mm"] for o in outcomes),
            "median_t_first_reach_s": med("t_first_reach"), "median_first_arrival_s": med("first_arrival_t"),
            "mean_saturation_frac": mean("saturation_frac"), "mean_torque_variation_nm": mean("torque_variation_nm"),
            "latency_p50_us": med("latency_p50_us"), "latency_p95_us": med("latency_p95_us"),
            "latency_p99_us": max(p99) if p99 else None, "failures": sum(o["failure"] is not None for o in outcomes)}


# ---------------------------------------------------------------- result index (parent process only)

def _read_index(path: Path) -> tuple[list, list[dict]]:
    if not path.exists() or not path.read_text().strip():
        return RESULT_COLUMNS, []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames or []
    return header, rows


def append_results(path: Path, rows: list[dict]) -> None:
    """Append rows keyed by (run_id, checkpoint_step, evaluation_split, evaluation_id); duplicates are refused.

    A header-only legacy index is upgraded to the current schema; a non-empty index with another schema needs an
    explicit migration and is never rewritten here.
    """
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + ".lock", "w") as lock:  # serialises concurrent parent-process writers
        fcntl.flock(lock, fcntl.LOCK_EX)
        _append_locked(path, rows)


def _append_locked(path: Path, rows: list[dict]) -> None:
    header, existing = _read_index(path)
    if header != RESULT_COLUMNS:
        if existing:
            raise ValueError(f"{path} has rows in an incompatible schema; explicit migration required")
        if header not in (LEGACY_COLUMNS, []):
            raise ValueError(f"{path} has an unknown header; explicit migration required")
    keys = {tuple(str(r[k]) for k in KEY) for r in existing}
    new = []
    for row in rows:
        unknown = set(row) - set(RESULT_COLUMNS)
        if unknown:
            raise ValueError(f"unknown result columns {sorted(unknown)}")
        key = tuple(str(row.get(k, "")) for k in KEY)
        if any(not v for v in key):
            raise ValueError(f"result row lacks key fields {KEY}: {key}")
        if key in keys:
            raise ValueError(f"duplicate evaluation {key}; refusing to overwrite")
        keys.add(key)
        new.append({c: row.get(c, "") for c in RESULT_COLUMNS})
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(existing + new)
    os.replace(tmp, path)


def write_details(path: Path, outcomes: list[dict], extra: dict | None = None) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; per-scenario details are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(outcomes[0]) + sorted(extra or {})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for o in outcomes:
            w.writerow({**o, **(extra or {})})


def result_row(identity: dict, summary: dict, **fields) -> dict:
    row = {k: identity.get(k, "") for k in ("run_id", "regime", "teacher", "objective", "cohort", "architecture",
                                            "seed", "fit_stage", "config_hash")}
    row.update({k: v for k, v in summary.items() if k in RESULT_COLUMNS})
    row.update(fields)
    row.setdefault("evaluation_id", EVALUATION_ID)
    row["written_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return {k: ("" if v is None else v) for k, v in row.items()}


# ---------------------------------------------------------------- frozen final protocol (test access)

def _file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_final_protocol(protocol: dict, index_path: Path | None = None) -> None:
    """The only gate to the 64 test scenarios: every declared artifact must equal the frozen validation choices.

    Architecture, LR, update budget, action loss and seed come from the protocol's earlier decisions; the
    checkpoint is best.pt for a selection artifact or final.pt for a refit artifact, with matching hashes and run
    config. Test rows already recorded under another protocol block a new one (no test-driven re-selection).
    """
    import yaml
    d = protocol.get("decisions") or {}
    final = d.get("final")
    if not final or not final.get("artifacts"):
        raise ValueError("no frozen final protocol: declare decisions.final.artifacts before any test evaluation")
    seeds = set(final.get("seeds") or [])
    for art in final["artifacts"]:
        t, arch = art["teacher"], art["architecture"]
        expected = {"architecture": d["selected"][t], "lr": d["lr"][t][arch] if arch in d["lr"].get(t, {}) else None,
                    "updates": d["budget"]["updates"], "loss": d["loss"]["action"]}
        for field, value in expected.items():
            if art[field] != value:
                raise ValueError(f"{art['label']}: {field} {art[field]!r} differs from the frozen choice {value!r}")
        if art["seed"] not in seeds:
            raise ValueError(f"{art['label']}: seed {art['seed']} is not in the frozen final seeds {sorted(seeds)}")
        kind = {"selection": "best.pt", "refit": "final.pt"}.get(art["fit_stage"])
        if kind is None or Path(art["checkpoint"]).name != kind:
            raise ValueError(f"{art['label']}: a {art['fit_stage']} artifact must be {kind}")
        for key in ("checkpoint", "inference"):
            if _file_sha(art[key]) != art[f"{key}_sha256"]:
                raise ValueError(f"{art['label']}: {key} sha256 differs from the frozen protocol")
        cfg = yaml.safe_load((Path(art["checkpoint"]).parent / "config.yaml").read_text())
        actual = {"architecture": cfg["identity"]["architecture"], "lr": cfg["training"]["lr"],
                  "updates": cfg["training"]["updates"], "loss": cfg["loss"]["action"], "seed": cfg["identity"]["seed"],
                  "fit_stage": cfg["identity"]["fit_stage"], "run_id": cfg["identity"]["run_id"],
                  "objective": cfg["identity"]["objective"], "teacher": cfg["identity"]["teacher"]}
        for field, value in actual.items():
            if art[field] != value:
                raise ValueError(f"{art['label']}: run config {field} {value!r} != protocol {art[field]!r}")
    if index_path is not None and Path(index_path).exists():
        own = f"test-{protocol['protocol_id']}"
        _, rows = _read_index(Path(index_path))
        other = [r for r in rows if r["evaluation_split"] == "test" and r["evaluation_id"] != own]
        if other:
            raise ValueError(f"test outcomes already observed under another protocol ({other[0]['evaluation_id']})")


def evaluate_final(protocol: dict, workers: int = 16) -> list[dict]:
    """Evaluate every declared artifact once on the 64 sealed test scenarios; returns the recorded rows."""
    import yaml
    from evaluation.test_sets import get_split
    index = TASK_DIR / "results" / "results.csv"
    validate_final_protocol(protocol, index_path=index)
    eval_id = f"test-{protocol['protocol_id']}"
    manifest = TASK_DIR / "experiments/splits/clean253.json"
    rows = []
    for art in protocol["decisions"]["final"]["artifacts"]:
        cfg = yaml.safe_load((Path(art["checkpoint"]).parent / "config.yaml").read_text())
        ident = dict(cfg["identity"])
        step = cfg["training"]["updates"] if art["fit_stage"] == "refit" else \
            json.loads((Path(art["checkpoint"]).parent / "selection.json").read_text())["selected_step"]
        _, existing = _read_index(index)
        if any((r["run_id"], r["checkpoint_step"], r["evaluation_split"], r["evaluation_id"]) ==
               (ident["run_id"], str(step), "test", eval_id) for r in existing):
            raise ValueError(f"{art['label']} was already evaluated on the test set; reruns are refused")
        cfg["evaluation"]["workers"] = workers
        factory = functools.partial(mlp_factory, art["inference"], "numpy", {"run_id": ident["run_id"]})
        outcomes = evaluate_policy(factory, get_split("test", manifest, art["teacher"], allow_test=True), cfg,
                                   workers=workers)
        detail = TASK_DIR / cfg["outputs"]["results_root"] / ident["regime"] / ident["teacher"] / ident["objective"] / \
            ident["run_id"] / f"rollouts_test_step{step}_{eval_id}.csv"
        write_details(detail, outcomes)
        row = result_row(ident, summarise(outcomes), lr=cfg["training"]["lr"], split_sha256=_file_sha(manifest),
                         checkpoint_step=step, checkpoint_path=art["checkpoint"],
                         checkpoint_sha256=art["checkpoint_sha256"], inference_path=art["inference"],
                         evaluation_split="test", evaluation_id=eval_id, detail_path=str(detail.relative_to(TASK_DIR)))
        append_results(index, [row])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- CLI

def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="closed-loop validation rollouts (test scenarios are sealed)")
    parser.add_argument("--policy", choices=["mlp", "expert"], required=True)
    parser.add_argument("--artifact", help="MLP inference export (.npz) or checkpoint (.pt with --backend torch)")
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--teacher", choices=["A", "B"])
    parser.add_argument("--split", choices=["screening", "validation"], default="screening")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--record", action="store_true", help="append the summary to results/results.csv")
    args = parser.parse_args(argv)
    from evaluation.test_sets import get_split
    from training.common.config import resolve_config
    split_manifest = TASK_DIR / "experiments/splits/clean253.json"
    if args.policy == "expert":
        if not args.teacher:
            parser.error("--teacher is required for expert references")
        cfg = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", args.teacher, {})
        teacher_version = cfg["data"]["teacher_version"]
        identity = {"run_id": f"expert-{args.teacher}-{teacher_version}", "regime": "clean", "teacher": args.teacher,
                    "objective": "expert", "cohort": "reference", "fit_stage": "reference"}
        factory = functools.partial(expert_factory, args.teacher)
        step, ckpt, ckpt_sha, inference = 0, teacher_version, "", ""
        detail_dir = TASK_DIR / "results/reference" / args.teacher
    else:
        import yaml
        artifact = Path(args.artifact).resolve()
        meta = None
        if artifact.suffix == ".npz":
            with np.load(artifact) as z:
                meta = json.loads(str(z["metadata_json"]))
        run_dir = artifact.parent
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        identity = dict(cfg["identity"], lr=cfg["training"]["lr"])
        if meta is None:
            from training.common.checkpoint import load_checkpoint
            payload = load_checkpoint(artifact)
            step, ckpt, ckpt_sha, inference = payload["step"], str(artifact), _sha256(artifact), ""
        else:
            step, ckpt, ckpt_sha, inference = meta["source_step"], meta["source_checkpoint"], meta["source_sha256"], str(artifact)
        expected = {"run_id": identity["run_id"], "teacher": identity["teacher"]}
        factory = functools.partial(mlp_factory, str(artifact), args.backend, expected)
        detail_dir = TASK_DIR / cfg["outputs"]["results_root"] / identity["regime"] / identity["teacher"] / \
            identity["objective"] / identity["run_id"]
        args.teacher = identity["teacher"]
    cfg["evaluation"]["workers"] = args.workers
    scenarios = get_split(args.split, split_manifest, args.teacher)
    outcomes = evaluate_policy(factory, scenarios, cfg, workers=args.workers)
    summary = summarise(outcomes)
    detail = detail_dir / f"rollouts_{args.split}_step{step}_{EVALUATION_ID}.csv"
    write_details(detail, outcomes)
    print(json.dumps(summary, indent=1))
    if args.record:
        row = result_row(identity, summary, lr=identity.get("lr", ""), split_sha256=_sha256(split_manifest),
                         checkpoint_step=step, checkpoint_path=ckpt, checkpoint_sha256=ckpt_sha, inference_path=inference,
                         evaluation_split=args.split, detail_path=str(detail.relative_to(TASK_DIR)))
        append_results(TASK_DIR / cfg["outputs"]["results_index"], [row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
