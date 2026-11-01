"""Far-target probe: how far beyond the 0.5-3 m training range do teacher A and the students still reach and hold?

Distance bands (start-to-target, horizontal): 3-6, 6-9 and 9-12 m, plus an in-range 0.5-3 m control. Everything else
stays inside the training ranges (start x, y in +-0.5 m, any heading and azimuth, target height 0.1-1.25 m). Fresh
random scenes from development seeds 980000+ (disjoint from every frozen split and from generalisation_probe's 960000+).

Teacher A drives at about 0.22 m/s, so beyond about 9 m even the teacher cannot reach and hold within the task's 40 s
cap. This probe therefore raises the episode cap (default 80 s); every other predicate is the standard one (reach
< 10 mm after 8 s, then 3 s more, all of the final 1 s < 10 mm). Teacher A runs the same scenes as the reference, and
students are also scored only on scenes the teacher holds. A development probe: it selects nothing.

Usage (from final/robotnik_arm_task/):
    python -m evaluation.far_probe --models <a.npz> <b.npz> ... [--n 12] [--cap 80] [--workers 22] [--out <dir>]
"""
import argparse
import csv
import functools
import json
from pathlib import Path

import numpy as np
import yaml

TASK_DIR = Path(__file__).resolve().parents[1]
SEED_BASE = 980_000
BANDS = {"D0-in-range-0.5-3m": (0.5, 3.0), "D1-3-6m": (3.0, 6.0), "D2-6-9m": (6.0, 9.0), "D3-9-12m": (9.0, 12.0)}


def scenes(band: str, n: int) -> list:
    from data_collection.scenarios import Scenario
    index = list(BANDS).index(band)
    out = []
    for i in range(n):
        rng = np.random.default_rng([SEED_BASE, index, i])
        start = rng.uniform(-0.5, 0.5, 2)
        yaw, azimuth = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
        d, z = rng.uniform(*BANDS[band]), rng.uniform(0.1, 1.25)
        target = np.r_[start + d * np.array([np.cos(yaw + azimuth), np.sin(yaw + azimuth)]), z]
        out.append(Scenario(seed=SEED_BASE + 1000 * index + i, target_ee_pos=target, base_start_pos=np.r_[start, 0.0],
                            base_start_bearing=float(yaw), physics={}, scenario_id=f"{band}-{i:02d}", split="probe"))
    return out


def _job(args):
    policy, band, scenario, config = args
    from evaluation.evaluate import MarkovianReference, make_robot_env, mlp_factory, rollout
    p = MarkovianReference() if policy == "teacher-A" else mlp_factory(policy)
    out = rollout(p, scenario, config, functools.partial(make_robot_env, config["evaluation"]["reach_tolerance_m"]))
    d = float(np.linalg.norm(scenario.target_ee_pos[:2] - scenario.base_start_pos[:2]))
    return policy, band, {**out, "target_dist_m": d, "target_z_m": float(scenario.target_ee_pos[2])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True, help="student NumPy exports (.npz)")
    p.add_argument("--n", type=int, default=12, help="scenes per distance band")
    p.add_argument("--cap", type=float, default=80.0, help="episode cap in seconds (the task's is 40)")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--out", type=Path, default=TASK_DIR / "results/far_probe/v1")
    args = p.parse_args(argv)
    import multiprocessing as mp
    from experiments.ablation.run import _init_worker
    config = yaml.safe_load((Path(args.models[0]).parent / "config.yaml").read_text())
    config["evaluation"]["episode_cap_s"] = float(args.cap)
    jobs = [(pol, b, s, config) for b in BANDS for s in scenes(b, args.n) for pol in ["teacher-A", *args.models]]
    print(f"{len(jobs)} rollouts, cap {args.cap} s", flush=True)
    with mp.get_context("spawn").Pool(args.workers, _init_worker) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [{"policy": Path(pol).stem if pol != "teacher-A" else pol, "band": b, **o} for pol, b, o in results]
    with open(args.out / "rollouts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary, names = {}, ["teacher-A"] + [Path(m).stem for m in args.models]
    for b in BANDS:
        teacher = {r["scenario_id"]: r["sustained_hold_pass"] for r in rows if r["band"] == b and r["policy"] == "teacher-A"}
        entry = {}
        for name in names:
            outs = [r for r in rows if r["band"] == b and r["policy"] == name]
            feasible = [r for r in outs if teacher[r["scenario_id"]]]
            reach_t = [r["t_first_reach"] for r in outs if r["t_first_reach"] is not None]
            entry[name] = {"hold": f"{sum(r['sustained_hold_pass'] for r in outs)}/{len(outs)}",
                           "hold_where_teacher_holds": f"{sum(r['sustained_hold_pass'] for r in feasible)}/{len(feasible)}",
                           "median_final_err_mm": round(float(np.median([r["final_err_mm"] for r in outs])), 2),
                           "median_t_first_reach_s": round(float(np.median(reach_t)), 1) if reach_t else None}
        summary[b] = entry
    (args.out / "summary.json").write_text(json.dumps({"cap_s": args.cap, "n_per_band": args.n, "bands": summary}, indent=1))
    print(f"\ncap {args.cap} s, {args.n} scenes per band")
    print(f"{'band':<20} " + " ".join(f"{n[:28]:>30}" for n in names))
    for b, entry in summary.items():
        print(f"{b:<20} " + " ".join(f"{entry[n]['hold']:>12} ({entry[n]['median_final_err_mm']:>7} mm)    " for n in names))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
