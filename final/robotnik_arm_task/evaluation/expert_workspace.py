"""Geometric limits of a teacher (A by default, or B): which target heights and distances can it reach and hold?

Complements the disturbance limits in docs/reports/2026-10-26__report__expert-operating-limits.md. Three sweeps on
nominal physics, each grid value with random start pose, heading and relative azimuth (development seeds 970000+):
  height    target height 0.0-1.6 m at 1.5 m start-to-target distance
  near      start-to-target distance 0.05-0.5 m at random heights inside 0.1-1.25 m
  far       start-to-target distance 6-10 m, with the episode cap raised to 80 s (capability, not the 40 s task limit)
Standard evaluator predicates (reach after 8 s, 3 s post-reach window, final 1 s below 10 mm). Also prints the arm's
kinematic envelope from the teacher's model: shoulder height and the reach limit the teacher clamps to.

Usage (from final/robotnik_arm_task/): python -m evaluation.expert_workspace [--teacher A|B] [--n 8] [--out <dir>]
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
SEED_BASE = 970_000
SWEEPS = {
    "height": ("target_z", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6]),
    "near": ("distance", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
    "far": ("distance", [6.0, 8.0, 10.0]),
}


def scene(sweep: str, value: float, i: int):
    from data_collection.scenarios import Scenario
    rng = np.random.default_rng([SEED_BASE, list(SWEEPS).index(sweep), int(round(value * 1000)), i])
    start, yaw, azimuth = rng.uniform(-0.5, 0.5, 2), rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
    dist = value if sweep != "height" else 1.5
    z = value if sweep == "height" else rng.uniform(0.1, 1.25)
    target = np.r_[start + dist * np.array([np.cos(yaw + azimuth), np.sin(yaw + azimuth)]), z]
    return Scenario(seed=SEED_BASE + i, target_ee_pos=target, base_start_pos=np.r_[start, 0.0], base_start_bearing=float(yaw),
                    scenario_id=f"{sweep}-{value:g}-{i}", split="workspace")


def _job(args):
    teacher, sweep, value, scenario, config = args
    from evaluation.evaluate import expert_factory, make_robot_env, rollout
    out = rollout(expert_factory(teacher), scenario, config, functools.partial(make_robot_env, config["evaluation"]["reach_tolerance_m"]))
    return sweep, value, float(scenario.target_ee_pos[2]), out


def envelope() -> dict:
    """Shoulder height at the start pose and the teacher's reach clamp."""
    from core.interfaces import Observation
    from data_collection.expert import ARM_REACH_LIMIT, Q_HOLD_ARM, MarkovianExpert, MarkovianExpertParams
    import pinocchio as pin
    ex = MarkovianExpert(MarkovianExpertParams(lead_time=0.5))
    obs = Observation(np.zeros(3), 0.0, Q_HOLD_ARM.copy(), np.zeros(7), np.zeros(3))
    pin.framesForwardKinematics(ex.model, ex.data, ex.q_pin(obs))
    shoulder = ex.data.oMf[ex.shoulder_frame_id].translation
    return {"shoulder_height_m": float(shoulder[2]), "shoulder_forward_of_base_m": float(shoulder[0]),
            "teacher_reach_limit_m": ARM_REACH_LIMIT,
            "max_height_straight_up_m": float(shoulder[2] + ARM_REACH_LIMIT)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--teacher", choices=["A", "B"], default="A")
    p.add_argument("--n", type=int, default=8, help="random scenes per grid value")
    p.add_argument("--workers", type=int, default=22)
    p.add_argument("--out", type=Path, help="default results/diagnostics/2026-10-29__expert-workspace[-B]")
    args = p.parse_args(argv)
    args.out = args.out or TASK_DIR / ("results/diagnostics/2026-10-29__expert-workspace" + ("-B" if args.teacher == "B" else ""))
    import multiprocessing as mp
    from experiments.ablation.run import _init_worker
    base = yaml.safe_load((TASK_DIR / "experiments/configs/common.yaml").read_text())
    config = {"evaluation": base["evaluation"], "interface": base["interface"]}
    far = copy.deepcopy(config)
    far["evaluation"]["episode_cap_s"] = 80.0
    jobs = [(args.teacher, s, v, scene(s, v, i), far if s == "far" else config) for s, (_, values) in SWEEPS.items()
            for v in values for i in range(args.n)]
    print(f"{len(jobs)} teacher-{args.teacher} rollouts", flush=True)
    with mp.get_context("spawn").Pool(args.workers, _init_worker) as pool:
        results = pool.map(_job, jobs, chunksize=1)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "rollouts.csv", "w", newline="") as f:
        rows = [{"sweep": s, "value": v, "target_z": z, **o} for s, v, z, o in results]
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"teacher": args.teacher, "envelope": envelope(), "sweeps": {}}
    for s, (label, values) in SWEEPS.items():
        summary["sweeps"][s] = []
        for v in values:
            outs = [o for ss, vv, _, o in results if ss == s and vv == v]
            summary["sweeps"][s].append({
                label: v, "n": len(outs), "reach": sum(o["ever_reached"] for o in outs),
                "hold": sum(o["sustained_hold_pass"] for o in outs),
                "median_min_err_mm": float(np.median([o["min_err_mm"] for o in outs])),
                "median_final_err_mm": float(np.median([o["final_err_mm"] for o in outs])),
                "median_reach_t_s": float(np.median([o["t_first_reach"] for o in outs if o["t_first_reach"] is not None]))
                if any(o["t_first_reach"] is not None for o in outs) else None})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
