"""Template controller and Markovian expert under T3-style conditions, on development seeds (not the frozen T3 set).

The frozen T3 split runs once, after the training freeze. This script uses the same kinds of
perturbation on separate seeds (DEV_SEED_OFFSET + i), so the expert's robustness can be known now
without touching the test set. Perturbation conditions share the same 20 targets and start poses
as the nominal condition, so each is compared with nominal target by target, and both controllers
run the same episodes. The template is run as in compare_controllers._template_controller: it plans
its base goal once from the start pose and latches "base done".

Output: results/expert_t3_dev.csv and a summary table.

Usage: python -m evaluation.expert_t3_dev --n 20 --workers 20 [--controllers template expert]
"""
import argparse
import csv
import multiprocessing as mp
import os

import numpy as np
import yaml

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = yaml.safe_load(open(os.path.join(TASK_DIR, "settings.yaml")))
DEV_SEED_OFFSET = 800_000   # the frozen T1-T3 splits must use other seeds
MAX_TIME = 40.0             # s, the template's default step cap
MAX_TIME_PUSH = 60.0        # s; a sideways base push slides the robot for up to ~20 s
HOLD_TIME = 2.0             # s after the first reach
BASE_PUSH = 120.0           # N for 0.2 s: 24 N*s, about 0.4 m/s on the 60.6 kg robot
EE_PUSH = 20.0              # N for 0.2 s on the end-effector link
PUSH_DURATION = 0.2         # s
PUSH_AFTER_REACH = 1.0      # s after the first reach, for pushes while holding the target
RECOVERY_WINDOW = 25.0      # s after a push to get back within tolerance (and stay there to the end)
UNSTABLE_QD = 5.0           # rad/s; arm joint speed above this for more than UNSTABLE_STEPS counts as unstable.
UNSTABLE_STEPS = 50         # 50 ms; excludes the ~10 ms joint-6 snap when reaching starts (the initial pose is
                            # below joint 6's lower limit, so the clipped reference jumps; the template does the same)

# name -> (physics, target kind); "T3 spec" conditions come from docs/project_plan.md, the rest go beyond it
CONDITIONS = {
    "nominal":               ({}, "standard"),
    "friction x0.5":         ({"friction_scale": 0.5}, "standard"),
    "friction x1.5":         ({"friction_scale": 1.5}, "standard"),
    "mass x0.75":            ({"mass_scale": 0.75}, "standard"),
    "mass x1.25":            ({"mass_scale": 1.25}, "standard"),
    "sensor noise x2":       ({"sensor_noise_scale": 2.0}, "standard"),
    "push base, driving":    ({"push": "base_drive"}, "standard"),
    "push base, at target":  ({"push": "base_hold"}, "standard"),
    "push hand, at target":  ({"push": "ee_hold"}, "standard"),
    "far targets 3-3.5 m":   ({}, "far"),
    "unreachable (high)":    ({}, "unreachable"),
    # beyond the T3 spec
    "sensor noise x20":      ({"sensor_noise_scale": 20.0}, "standard"),
    "sensor noise x100":     ({"sensor_noise_scale": 100.0}, "standard"),
    "combined: fr x0.5, mass x1.25, noise x2": ({"friction_scale": 0.5, "mass_scale": 1.25, "sensor_noise_scale": 2.0}, "standard"),
}
CONTROLLERS = ("template", "expert")
COLUMNS = ["controller", "condition", "seed", "target_x", "target_y", "target_z", "dist", "height", "reached", "t_reach",
           "err_end_mm", "held", "push_peak_err_mm", "recover_s", "settled", "max_qd_last1s", "sat_frac_last5s",
           "unstable"]

_env = None
_expert = None


def make_scenario(condition: str, i: int):
    from data_collection.scenarios import Scenario
    physics, kind = CONDITIONS[condition]
    sc = SETTINGS["scenarios"]
    offset = {"standard": 0, "far": 10_000, "unreachable": 20_000}[kind]
    rng = np.random.default_rng(DEV_SEED_OFFSET + offset + i)
    dist = rng.uniform(*sc["target_dist"]) if kind == "standard" else rng.uniform(3.0, 3.5) if kind == "far" else rng.uniform(1.0, 2.5)
    az, height = rng.uniform(*sc["target_azimuth"]), rng.uniform(*sc["target_height"])
    if kind == "unreachable":
        height = rng.uniform(1.7, 2.0)
    start, bearing = np.array([*rng.uniform(-1, 1, 2), 0.0]), rng.uniform(-np.pi, np.pi)
    push_dir = rng.uniform(-np.pi, np.pi)
    target = np.array([start[0] + dist * np.cos(bearing + az), start[1] + dist * np.sin(bearing + az), height])

    physics = dict(physics)
    if "push" in physics:
        kind_push = physics.pop("push")
        force = (BASE_PUSH if kind_push.startswith("base") else EE_PUSH) * np.array([np.cos(push_dir), np.sin(push_dir), 0.0])
        physics["push"] = {"link": "base" if kind_push.startswith("base") else "ee", "force": force.tolist(),
                           "duration": PUSH_DURATION}
        physics["push"].update({"start": 3.0} if kind_push == "base_drive" else {"after_reach": PUSH_AFTER_REACH})
    return Scenario(seed=DEV_SEED_OFFSET + offset + i, target_ee_pos=target, base_start_pos=start,
                    base_start_bearing=bearing, physics=physics)


def _init_worker():
    global _env, _expert
    from core.env import RobotEnv
    from data_collection.expert import MarkovianExpert
    _env = RobotEnv(reach_tolerance=SETTINGS["reach_tolerance"])
    _expert = MarkovianExpert()


def run_episode(job):
    from evaluation.compare_controllers import _expert_controller, _template_controller
    controller, condition, i = job
    sc = make_scenario(condition, i)
    push = sc.physics.get("push")
    hold_push = push is not None and "after_reach" in push
    unreachable = CONDITIONS[condition][1] == "unreachable"
    obs = _env.reset(sc)
    step = _template_controller(_env, sc.target_ee_pos) if controller == "template" else _expert_controller(_expert)
    max_torque = _expert.params.arm.max_torque
    errs, qds, sats = [], [], []
    push_end = None
    for k in range(int((MAX_TIME_PUSH if push else MAX_TIME) / 0.001)):
        action, _ = step(obs)
        obs, info = _env.step(action)
        errs.append(info["ee_err"])
        qds.append(np.abs(np.array(_env.sim.GetMotorVelocities(0))[4:11]).max())  # true velocities
        sats.append(np.any(np.abs(action.torque_arm) >= max_torque - 1e-6))
        if info["pushing"]:
            push_end = info["t"] + 0.001
        t_reach = _env.first_reach_t
        if unreachable:
            if info["t"] >= 30.0:
                break
        elif hold_push:
            if push_end is not None and info["t"] >= push_end + RECOVERY_WINDOW:
                break
        elif push and t_reach is not None and info["t"] >= max(t_reach, push["start"] + push["duration"]) + HOLD_TIME:
            break
        elif t_reach is not None and info["t"] >= t_reach + HOLD_TIME:
            break

    errs = np.array(errs)
    tol = SETTINGS["reach_tolerance"]
    row = {"controller": controller, "condition": condition, "seed": sc.seed, "target_x": sc.target_ee_pos[0], "target_y": sc.target_ee_pos[1],
           "target_z": sc.target_ee_pos[2],
           "dist": np.linalg.norm(sc.target_ee_pos[:2] - sc.base_start_pos[:2]), "height": sc.target_ee_pos[2],
           "reached": _env.first_reach_t is not None, "t_reach": _env.first_reach_t,
           "err_end_mm": errs[-1] * 1000, "held": bool(errs[-1] < tol),
           "push_peak_err_mm": None, "recover_s": None,
           "settled": bool(max(qds[-1000:]) < 0.02), "max_qd_last1s": max(qds[-1000:]),
           "sat_frac_last5s": float(np.mean(sats[-5000:])), "unstable": bool(np.sum(np.array(qds) > UNSTABLE_QD) > UNSTABLE_STEPS)}
    if hold_push and push_end is not None:
        k_end = int(round(push_end / 0.001)) - 1
        after = errs[k_end:]
        row["push_peak_err_mm"] = after.max() * 1000
        # time until the end-effector is back within tolerance for good (0 if the push never moved it out)
        out = np.nonzero(after >= tol)[0]
        row["recover_s"] = ((out[-1] + 1) * 0.001 if len(out) else 0.0) if after[-1] < tol else None
    return row


def summarise(rows):
    for controller in CONTROLLERS:
        if any(r["controller"] == controller for r in rows):
            print(f"\n=== {controller} ===")
            summarise_controller([r for r in rows if r["controller"] == controller])
    if all(any(r["controller"] == c for r in rows) for c in CONTROLLERS):
        compare(rows)


def summarise_controller(rows):
    print(f"\n{'condition':42} {'reached':>8} {'held':>6} {'t_reach med':>11} {'end err med':>11} {'end err max':>11}  extra")
    for cond in CONDITIONS:
        rs = [r for r in rows if r["condition"] == cond]
        if not rs:
            continue
        n = len(rs)
        ok = [r for r in rs if r["reached"]]
        t_med = f"{np.median([r['t_reach'] for r in ok]):.1f}s" if ok else "-"
        errs = [r["err_end_mm"] for r in rs]
        extra = f"UNSTABLE arm {sum(r['unstable'] for r in rs)}/{n}; " if any(r["unstable"] for r in rs) else ""
        if rs[0]["push_peak_err_mm"] is not None or any(r["recover_s"] is not None for r in rs):
            peaks = [r["push_peak_err_mm"] for r in rs if r["push_peak_err_mm"] is not None]
            rec = [r["recover_s"] for r in rs if r["recover_s"] is not None]
            extra += (f"push peak err med {np.median(peaks):.0f}mm max {np.max(peaks):.0f}mm; "
                     f"recovered {len(rec)}/{len(peaks)}, time med {np.median(rec):.2f}s max {np.max(rec):.2f}s" if rec else
                     f"push peak err med {np.median(peaks):.0f}mm; recovered 0/{len(peaks)}")
        if CONDITIONS[cond][1] == "unreachable":
            extra += (f"settled {sum(r['settled'] for r in rs)}/{n}; torque at limit in last 5 s: "
                     f"median {np.median([r['sat_frac_last5s'] for r in rs]) * 100:.0f}% of steps")
        print(f"{cond:42} {len(ok):>4}/{n:<3} {sum(r['held'] for r in rs):>2}/{n:<3} {t_med:>11} "
              f"{np.median(errs):>9.1f}mm {np.max(errs):>9.1f}mm  {extra}")



def compare(rows):
    """Held-at-end counts and median final error side by side, per condition."""
    print(f"\n{'condition':42} {'held: template':>15} {'expert':>7} {'end err med: template':>22} {'expert':>9}")
    for cond in CONDITIONS:
        by = {c: [r for r in rows if r["condition"] == cond and r["controller"] == c] for c in CONTROLLERS}
        if not all(by.values()):
            continue
        held = [f"{sum(r['held'] for r in by[c])}/{len(by[c])}" for c in CONTROLLERS]
        err = [f"{np.median([r['err_end_mm'] for r in by[c]]):.1f}mm" for c in CONTROLLERS]
        print(f"{cond:42} {held[0]:>15} {held[1]:>7} {err[0]:>22} {err[1]:>9}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="targets per condition (10 for unreachable)")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--controllers", nargs="+", choices=CONTROLLERS, default=list(CONTROLLERS))
    parser.add_argument("--out", default="results/expert_t3_dev.csv")
    args = parser.parse_args()

    jobs = [(k, c, i) for k in args.controllers for c in CONDITIONS
            for i in range(args.n // 2 if CONDITIONS[c][1] == "unreachable" else args.n)]
    rows = []
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(run_episode, jobs):
            rows.append(r)
            if len(rows) % 20 == 0:
                print(f"[{len(rows)}/{len(jobs)}]", flush=True)
    rows.sort(key=lambda r: (CONTROLLERS.index(r["controller"]), list(CONDITIONS).index(r["condition"]), r["seed"]))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.out}")
    summarise(rows)


if __name__ == "__main__":
    main()
