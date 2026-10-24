"""Time to reach: template controller (robotnik_arm_control.py) vs Markovian expert, on the same targets.

Both controllers start at the origin facing +X, which the template assumes. Targets are drawn
uniformly in the robot's frame over the ranges in settings.yaml, plus the template's own
TARGET_EE_POS. Each episode runs until the end-effector is within reach_tolerance and HOLD_TIME
more seconds, or MAX_TIME. Times are simulated seconds; ms_per_step is wall-clock controller
compute (forward kinematics + control law, not the physics step).

Output: results/controller_timing.csv, results/figures/controller_timing.png

Usage: python -m evaluation.compare_controllers --n 30 --workers 12
"""
import argparse
import csv
import multiprocessing as mp
import os
import time

import numpy as np
import pinocchio as pin
import yaml

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = yaml.safe_load(open(os.path.join(TASK_DIR, "settings.yaml")))
SEED_OFFSET = 900_000           # away from the seeds used for training scenarios
TEMPLATE_TARGET = [0.0, 1.1, 0.5]  # TARGET_EE_POS in robotnik_arm_task.py, run as seed -1
HOLD_TIME = 2.0                 # s after the first reach, to check the target is held
MAX_TIME = 40.0                 # s, the template's default step cap (40000 steps)
COLUMNS = ["controller", "seed", "target_x", "target_y", "target_z", "dist", "azimuth", "height", "reached",
           "t_base", "t_reach", "t_arm", "err_after_hold_mm", "ms_per_step"]

_env = None
_expert = None


def sample_target(seed: int) -> np.ndarray:
    if seed < 0:
        return np.array(TEMPLATE_TARGET)
    sc = SETTINGS["scenarios"]
    rng = np.random.default_rng(SEED_OFFSET + seed)
    d, az, h = rng.uniform(*sc["target_dist"]), rng.uniform(*sc["target_azimuth"]), rng.uniform(*sc["target_height"])
    return np.array([d * np.cos(az), d * np.sin(az), h])


def _init_worker():
    global _env, _expert
    from core.env import RobotEnv
    from data_collection.expert import MarkovianExpert
    _env = RobotEnv(reach_tolerance=SETTINGS["reach_tolerance"])
    _expert = MarkovianExpert()


def _template_controller(env, target):
    """The template's controller and its state, as set up in robotnik_arm_task.py. Returns step(obs) -> (action, base_done)."""
    from core.interfaces import Action
    from robotnik_arm_control import ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller, base_controller

    dm = env.dyn_model
    shoulder_frame_id = dm.pin_model.getFrameId("robot_j2s7s300_link_2")
    lower, upper = env.sim.GetBotJointsLimit()
    arm_lower, arm_upper = np.array(lower[4:11]), np.array(upper[4:11])
    standoff = float(np.interp(target[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]))
    target_bearing = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    goal_pos = target[:2] - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=25.0)
    arm_params = ArmControllerParams(reach_limit=0.85)
    q_hold = env.init_motor_angles[4:11].copy()
    state = {"q_des": q_hold.copy()}

    def step(obs):
        q_all = np.concatenate((env.sim.GetBasePosition(), env.sim.GetBaseOrientation(), env.sim.GetMotorAngles(0)))
        q_pin = dm.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dm.pin_model, dm.pin_data, q_pin)
        pin.updateFramePlacements(dm.pin_model, dm.pin_data)
        ee_pos = dm.pin_data.oMf[env.ee_frame_id].translation.copy()
        wheel_cmds = base_controller(base_state, obs.base_pos, obs.base_bearing, goal_pos, target_bearing, env.t, base_params)
        torque_arm, state["q_des"], _ = arm_controller(
            dm, q_pin, obs.q_mes, obs.qd_mes, state["q_des"], q_hold, target, ee_pos, env.ee_frame_id,
            shoulder_frame_id, arm_lower, arm_upper, 0.001, arm_params, reach_active=base_state.done)
        return Action(torque_arm, wheel_cmds), base_state.done

    return step


def _expert_controller(expert):
    from core.interfaces import Action

    def step(obs):
        wheel_cmds, reach_active = expert.base_command(obs)
        return Action(expert.arm_torque(obs, reach_active), wheel_cmds), reach_active

    return step


def run_episode(job):
    controller, seed = job
    from data_collection.scenarios import Scenario
    target = sample_target(seed)
    obs = _env.reset(Scenario(seed=seed, target_ee_pos=target))
    step = _template_controller(_env, target) if controller == "template" else _expert_controller(_expert)
    t_base = t_reach = None
    compute = 0.0
    n_steps = int(MAX_TIME / 0.001)
    for k in range(n_steps):
        t0 = time.perf_counter()
        action, base_done = step(obs)
        compute += time.perf_counter() - t0
        if base_done and t_base is None:
            t_base = _env.t
        obs, info = _env.step(action)
        if info["done"] and t_reach is None:
            t_reach = info["t"]
        if t_reach is not None and info["t"] >= t_reach + HOLD_TIME:
            break
    return {
        "controller": controller, "seed": seed,
        "target_x": target[0], "target_y": target[1], "target_z": target[2],
        "dist": np.linalg.norm(target[:2]), "azimuth": np.arctan2(target[1], target[0]), "height": target[2],
        "reached": t_reach is not None, "t_base": t_base, "t_reach": t_reach,
        "t_arm": t_reach - t_base if t_reach is not None and t_base is not None else None,
        "err_after_hold_mm": info["ee_err"] * 1000, "ms_per_step": compute / (k + 1) * 1000,
    }


def summarise(rows):
    by = {c: {r["seed"]: r for r in rows if r["controller"] == c} for c in ("template", "expert")}
    seeds = sorted(by["template"])
    print(f"\n{'':10} {'reached':>8} {'t_base med':>11} {'t_arm med':>10} {'t_reach med':>12} {'t_reach mean':>13} "
          f"{'err after hold':>15} {'ms/step':>8}")
    for c in ("template", "expert"):
        rs = [by[c][s] for s in seeds]
        ok = [r for r in rs if r["reached"]]
        med = lambda key: np.median([r[key] for r in ok if r[key] is not None])
        print(f"{c:10} {len(ok):>4}/{len(rs):<3} {med('t_base'):>10.2f}s {med('t_arm'):>9.2f}s {med('t_reach'):>11.2f}s "
              f"{np.mean([r['t_reach'] for r in ok]):>12.2f}s {np.median([r['err_after_hold_mm'] for r in rs]):>12.1f}mm "
              f"{np.mean([r['ms_per_step'] for r in rs]):>8.3f}")
    both = [s for s in seeds if by["template"][s]["reached"] and by["expert"][s]["reached"]]
    diff = np.array([by["expert"][s]["t_reach"] - by["template"][s]["t_reach"] for s in both])
    print(f"\nPaired over {len(both)} targets reached by both: expert - template time to reach = "
          f"median {np.median(diff):+.2f}s, mean {diff.mean():+.2f}s, range [{diff.min():+.2f}, {diff.max():+.2f}]s; "
          f"expert faster on {int((diff < 0).sum())}/{len(both)}")
    d = by["template"][-1]["t_reach"], by["expert"][-1]["t_reach"]
    print(f"Template's default target {TEMPLATE_TARGET}: template {d[0]}s, expert {d[1]}s")
    return by, both


def plot(by, both, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})

    # (a) per-target time to reach; below the diagonal = Markovian expert faster
    tt = np.array([by["template"][s]["t_reach"] for s in both])
    te = np.array([by["expert"][s]["t_reach"] for s in both])
    hi = max(tt.max(), te.max()) * 1.05
    ax1.plot([0, hi], [0, hi], color=muted, lw=1, ls="--", zorder=1)
    ax1.scatter(tt, te, s=36, color=blue, edgecolor="white", linewidth=1.5, zorder=2)
    slow = [s for s in both if by["template"][s]["t_base"] >= 25.0 - 1e-6]  # template base hit nav_timeout
    for s in slow:
        x, y = by["template"][s]["t_reach"], by["expert"][s]["t_reach"]
        ax1.annotate("template base\nnav timed out (25 s)", (x, y), xytext=(x - 1, y + 5), ha="right", color=muted,
                     arrowprops={"arrowstyle": "-", "color": muted, "lw": 0.8})
    ax1.text(hi * 0.97, hi * 0.06, "Markovian faster", ha="right", color=muted)
    ax1.text(hi * 0.03, hi * 0.92, "template faster", ha="left", color=muted)
    ax1.set(xlim=(0, hi), ylim=(0, hi), xlabel="Template controller (s)", ylabel="Markovian expert (s)")
    ax1.set_title("(a) Time to reach 10 mm, per target", loc="left", color=ink)
    ax1.grid(color=grid, lw=0.6)
    ax1.set_axisbelow(True)

    # (b) median of each phase; medians are not additive, so each phase is labelled, not the total
    labels = ["Template", "Markovian"]
    base = [np.median([by[c][s]["t_base"] for s in both]) for c in ("template", "expert")]
    arm = [np.median([by[c][s]["t_arm"] for s in both]) for c in ("template", "expert")]
    y = np.arange(2)
    ax2.barh(y, base, height=0.5, color=blue, label="Base navigation", edgecolor="white", linewidth=2)
    ax2.barh(y, arm, left=base, height=0.5, color=orange, label="Arm reach", edgecolor="white", linewidth=2)
    for i in range(2):
        ax2.text(base[i] + arm[i] + 0.15, y[i], f"{base[i]:.1f} s + {arm[i]:.1f} s", va="center", color=ink)
    ax2.set(yticks=y, yticklabels=labels, xlabel="Median time per phase (s)", xlim=(0, max(np.add(base, arm)) * 1.45))
    ax2.invert_yaxis()
    ax2.set_title("(b) Where the time goes", loc="left", color=ink)
    ax2.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax2.grid(axis="x", color=grid, lw=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Figure: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="random targets, plus the template's default target")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default="results/controller_timing.csv")
    parser.add_argument("--fig", default="results/figures/controller_timing.png")
    args = parser.parse_args()

    jobs = [(c, s) for s in range(-1, args.n) for c in ("template", "expert")]
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        rows = []
        for r in pool.imap_unordered(run_episode, jobs):
            rows.append(r)
            print(f"[{len(rows)}/{len(jobs)}] {r['controller']:8} seed {r['seed']:>3} t_reach={r['t_reach']} "
                  f"err_after_hold={r['err_after_hold_mm']:.1f}mm", flush=True)
    rows.sort(key=lambda r: (r["seed"], r["controller"]))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.out}")
    by, both = summarise(rows)
    plot(by, both, args.fig)


if __name__ == "__main__":
    main()
