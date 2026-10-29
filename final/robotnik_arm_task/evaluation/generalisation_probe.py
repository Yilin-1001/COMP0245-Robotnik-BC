"""Out-of-range probe: how far do ablation models generalise beyond the training scene ranges?

Training scenes: start x, y within +-0.5 m, start-to-target distance 0.5-3.0 m, target height 0.1-1.25 m, any azimuth
and heading. Each condition below draws fresh random scenes (development seeds 960000+, disjoint from every frozen
split) and changes one range; C0 stays inside all ranges as the control. Teacher A runs the same scenes as the
reference, so a scene the teacher cannot finish is not counted against the student. Standard evaluator predicates.
This is a development probe; it selects nothing and is not the frozen test.

Usage (from final/robotnik_arm_task/):
    python -m evaluation.generalisation_probe --models <a.npz> <b.npz> ... [--n 12] [--out <dir>]
"""
import argparse
import csv
import functools
import json
from pathlib import Path

import numpy as np
import yaml

TASK_DIR = Path(__file__).resolve().parents[1]
SEED_BASE = 960_000
# name: (description, start radius range or None for the +-0.5 m box, distance range, height range)
CONDITIONS = {
    "C0-in-range": ("inside every training range (control)", None, (0.5, 3.0), (0.1, 1.25)),
    "C1-far-3-4m": ("target 3-4 m away", None, (3.0, 4.0), (0.1, 1.25)),
    "C2-far-4-6m": ("target 4-6 m away", None, (4.0, 6.0), (0.1, 1.25)),
    "C3-near-0.3-0.5m": ("target 0.3-0.5 m away", None, (0.3, 0.5), (0.1, 1.25)),
    "C4-start-1-3m": ("start 1-3 m from the origin", (1.0, 3.0), (0.5, 3.0), (0.1, 1.25)),
    "C5-high-1.25-1.4m": ("target 1.25-1.4 m high", None, (0.5, 3.0), (1.25, 1.4)),
    "C6-low-0.03-0.1m": ("target 0.03-0.1 m high", None, (0.5, 3.0), (0.03, 0.1)),
}


def scenes(condition: str, n: int) -> list:
    from data_collection.scenarios import Scenario
    _, start_r, dist, height = CONDITIONS[condition]
    index = list(CONDITIONS).index(condition)
    out = []
    for i in range(n):
        rng = np.random.default_rng([SEED_BASE, index, i])
        if start_r is None:
            start = rng.uniform(-0.5, 0.5, 2)
        else:
            r, a = rng.uniform(*start_r), rng.uniform(-np.pi, np.pi)
            start = r * np.array([np.cos(a), np.sin(a)])
        yaw, azimuth, d, z = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), rng.uniform(*dist), rng.uniform(*height)
        target = np.r_[start + d * np.array([np.cos(yaw + azimuth), np.sin(yaw + azimuth)]), z]
        out.append(Scenario(seed=SEED_BASE + 1000 * index + i, target_ee_pos=target, base_start_pos=np.r_[start, 0.0],
                            base_start_bearing=float(yaw), scenario_id=f"{condition}-{i:02d}", split="probe"))
    return out


def _job(args):
    policy, condition, scenario, config = args
    from evaluation.evaluate import make_robot_env, mlp_factory, MarkovianReference, rollout
    p = MarkovianReference() if policy == "teacher-A" else mlp_factory(policy)
    out = rollout(p, scenario, config, functools.partial(make_robot_env, config["evaluation"]["reach_tolerance_m"]))
    return policy, condition, out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True, help="student NumPy exports (.npz)")
    p.add_argument("--n", type=int, default=12, help="scenes per condition")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--out", type=Path, default=TASK_DIR / "results/ablation/2026-10-28__training-ablation-v1/generalisation")
    args = p.parse_args(argv)
    import multiprocessing as mp
    from experiments.ablation.run import _init_worker
    config = yaml.safe_load((Path(args.models[0]).parent / "config.yaml").read_text())
    jobs = [(policy, c, s, config) for c in CONDITIONS for s in scenes(c, args.n) for policy in ["teacher-A", *args.models]]
    print(f"{len(jobs)} rollouts", flush=True)
    with mp.get_context("spawn").Pool(args.workers, _init_worker) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "rollouts.csv", "w", newline="") as f:
        rows = [{"policy": Path(pol).stem if pol != "teacher-A" else pol, "condition": c, **o} for pol, c, o in results]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for c, (desc, *_) in CONDITIONS.items():
        teacher = {o["scenario_id"]: o["sustained_hold_pass"] for pol, cc, o in results if cc == c and pol == "teacher-A"}
        entry = {"description": desc, "teacher_hold": sum(teacher.values()), "n": len(teacher), "students": {}}
        for model in args.models:
            outs = [o for pol, cc, o in results if cc == c and pol == model]
            feasible = [o for o in outs if teacher[o["scenario_id"]]]
            entry["students"][Path(model).stem] = {
                "hold": sum(o["sustained_hold_pass"] for o in outs),
                "hold_where_teacher_holds": f"{sum(o['sustained_hold_pass'] for o in feasible)}/{len(feasible)}",
                "median_final_err_mm": float(np.median([o["final_err_mm"] for o in outs]))}
        summary[c] = entry
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
