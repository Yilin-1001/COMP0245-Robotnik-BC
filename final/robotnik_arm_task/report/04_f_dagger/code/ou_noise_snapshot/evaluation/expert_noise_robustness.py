"""DART-style action-noise robustness of the two teachers, on development seeds.

Noise is added to the executed command, as in DART collection, and the teachers keep acting on the
resulting states. Sweeps the noise structure (white, Ornstein-Uhlenbeck with several correlation
times, a constant per-episode offset), its size, and the channel it enters (arm torques or wheel
speeds; wheel noise with and without gating once the teacher's navigation is done). Every condition
runs the same scenarios for both teachers.

Teachers: A = Markovian expert (data_collection/expert.py, lead_time 0.50 s, the collection setting);
B = template controller (data_collection/template_teacher.py, origin-based base goal).

Episode protocol:
  noisy phase: from t = 0 until first reach + HOLD_TIME, or NOISE_CAP s if the target is never reached
  clean tail:  noise off, run until the EE has stayed within tolerance for SETTLE_TIME, or TAIL_CAP s
Outcome: reached (under noise), hold_frac (share of the noisy hold window within tolerance), recovered
(within tolerance at the end of the clean tail).

Output: results/expert_noise_robustness.csv and a summary table.

Usage: python -m evaluation.expert_noise_robustness --n 16 --workers 22
"""
import argparse
import csv
import multiprocessing as mp
import os

import numpy as np
import yaml

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = yaml.safe_load(open(os.path.join(TASK_DIR, "settings.yaml")))
DEV_SEED_OFFSET = 870_000    # away from training, validation, test and earlier dev seeds
DT = 0.001
LEAD_TIME_A = 0.50           # s, the collection setting chosen on 26 Oct
HOLD_TIME = 3.0              # s of noisy hold after the first reach
NOISE_CAP = 30.0             # s; noise stops here if the target was never reached
SETTLE_TIME = 1.0            # s within tolerance to count as recovered in the clean tail
TAIL_CAP = 10.0              # s
MAX_TORQUE = 30.0            # N*m, executed torque limit (the controllers' own limit)
MAX_WHEEL = 6.0              # rad/s, executed wheel-speed limit (teachers command at most ~5.3)
UNSTABLE_QD, UNSTABLE_STEPS = 5.0, 50   # as in expert_t3_dev.py

STRUCTURES = {"white": 0.0, "ou_0.05": 0.05, "ou_0.2": 0.2, "ou_1.0": 1.0, "const": np.inf}  # correlation time (s)
ARM_SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0]      # N*m, per joint
WHEEL_SIGMAS = [0.1, 0.3, 1.0, 3.0]         # rad/s, per wheel
TEACHERS = ("A", "B")
COLUMNS = ["teacher", "condition", "channel", "structure", "sigma", "gated", "seed", "dist", "height",
           "reached", "t_reach", "hold_frac", "hold_rms_mm", "hold_max_mm", "recovered", "recover_s", "final_err_mm",
           "nav_done_t", "nav_timeout", "base_drift_after_nav_mm", "teacher_sat_frac", "exec_clip_frac", "unstable"]


def conditions():
    """name -> (channel, structure, sigma, gated)."""
    conds = {"clean": ("none", "none", 0.0, False)}
    for s in STRUCTURES:
        for sig in ARM_SIGMAS:
            conds[f"arm {s} {sig:g}Nm"] = ("arm", s, sig, False)
        for sig in WHEEL_SIGMAS:
            for gated in (False, True):
                conds[f"wheel {s} {sig:g}rad/s{' gated' if gated else ''}"] = ("wheel", s, sig, gated)
    return conds


CONDITIONS = conditions()


class ActionNoise:
    """Zero-mean noise with stationary std sigma per channel. tau = 0: white; inf: constant offset; else OU."""

    def __init__(self, sigma: np.ndarray, tau: float, rng: np.random.Generator):
        self.sigma, self.rng = sigma, rng
        self.a = 0.0 if tau == 0 else 1.0 if np.isinf(tau) else np.exp(-DT / tau)
        self.n = rng.normal(0.0, 1.0, sigma.shape) * sigma   # start in the stationary distribution

    def sample(self) -> np.ndarray:
        if self.a < 1.0:
            self.n = self.a * self.n + np.sqrt(1 - self.a**2) * self.rng.normal(0.0, 1.0, self.sigma.shape) * self.sigma
        return self.n


def make_scenario(i: int):
    from data_collection.scenarios import Scenario
    sc = SETTINGS["scenarios"]
    rng = np.random.default_rng(DEV_SEED_OFFSET + i)
    dist, az, height = rng.uniform(*sc["target_dist"]), rng.uniform(*sc["target_azimuth"]), rng.uniform(*sc["target_height"])
    start, bearing = np.array([*rng.uniform(-0.5, 0.5, 2), 0.0]), rng.uniform(-np.pi, np.pi)
    target = np.array([start[0] + dist * np.cos(bearing + az), start[1] + dist * np.sin(bearing + az), height])
    return Scenario(seed=DEV_SEED_OFFSET + i, target_ee_pos=target, base_start_pos=start, base_start_bearing=bearing)


_env = _expert = _template = None


def _init_worker():
    global _env, _expert, _template
    from core.env import RobotEnv
    from data_collection.expert import MarkovianExpert, MarkovianExpertParams
    from data_collection.template_teacher import TemplateTeacher
    _env = RobotEnv(reach_tolerance=SETTINGS["reach_tolerance"])
    _expert = MarkovianExpert(MarkovianExpertParams(lead_time=LEAD_TIME_A))
    _template = TemplateTeacher(_env)


def run_episode(job):
    from core.interfaces import Action
    teacher, condition, i = job
    channel, structure, sigma, gated = CONDITIONS[condition]
    sc = make_scenario(i)
    obs = _env.reset(sc)
    if teacher == "B":
        _template.reset()
    sig = np.zeros(11)
    if channel == "arm":
        sig[:7] = sigma
    elif channel == "wheel":
        sig[7:] = sigma
    # Same noise realisation for both teachers in a condition: seeded by scenario and condition, not teacher
    rng = np.random.default_rng([sc.seed, list(CONDITIONS).index(condition)])
    noise = ActionNoise(sig, STRUCTURES.get(structure, 0.0), rng)
    tol = SETTINGS["reach_tolerance"]

    t_reach = nav_done_t = nav_xy = None
    noisy_end = NOISE_CAP
    hold_errs, sats, clips, qd_over = [], [], [], 0
    base_drift = 0.0
    in_tol_since = recover_s = None
    noise_on = True
    k = 0
    while True:
        if teacher == "A":
            wheel_cmds, nav_done = _expert.base_command(obs)
            action = Action(_expert.arm_torque(obs, nav_done), wheel_cmds)
        else:
            action, state = _template.act(obs)
            nav_done = state["after"]["nav_done"]
        if nav_done and nav_done_t is None:
            nav_done_t, nav_xy = _env.t, obs.base_pos[:2].copy()
        n = noise.sample() if noise_on else np.zeros(11)
        if gated and nav_done:
            n = np.r_[n[:7], np.zeros(4)]
        torque = action.torque_arm + n[:7]
        wheels = action.wheel_cmds + n[7:]
        clipped = np.any(np.abs(torque) > MAX_TORQUE) or np.any(np.abs(wheels) > MAX_WHEEL)
        executed = Action(np.clip(torque, -MAX_TORQUE, MAX_TORQUE), np.clip(wheels, -MAX_WHEEL, MAX_WHEEL))
        obs, info = _env.step(executed)
        k += 1
        if np.abs(np.array(_env.sim.GetMotorVelocities(0))[4:11]).max() > UNSTABLE_QD:
            qd_over += 1
        if noise_on:
            sats.append(np.any(np.abs(action.torque_arm) >= MAX_TORQUE - 1e-6))
            clips.append(clipped)
            if nav_xy is not None:
                base_drift = max(base_drift, float(np.linalg.norm(obs.base_pos[:2] - nav_xy)))
        if info["done"] and t_reach is None:
            t_reach = info["t"]
            noisy_end = min(t_reach + HOLD_TIME, NOISE_CAP) if t_reach < NOISE_CAP else NOISE_CAP
        if noise_on:
            if t_reach is not None:
                hold_errs.append(info["ee_err"])
            if info["t"] >= noisy_end - 1e-9:
                noise_on, tail_start = False, info["t"]
        else:
            if info["ee_err"] < tol:
                in_tol_since = info["t"] if in_tol_since is None else in_tol_since
                if info["t"] - in_tol_since >= SETTLE_TIME:
                    recover_s = in_tol_since - tail_start
                    break
            else:
                in_tol_since = None
            if info["t"] - tail_start >= TAIL_CAP:
                break

    hold = np.array(hold_errs) if hold_errs else np.array([np.nan])
    return {"teacher": teacher, "condition": condition, "channel": channel, "structure": structure, "sigma": sigma,
            "gated": gated, "seed": sc.seed,
            "dist": float(np.linalg.norm(sc.target_ee_pos[:2] - sc.base_start_pos[:2])), "height": sc.target_ee_pos[2],
            "reached": t_reach is not None and t_reach <= NOISE_CAP, "t_reach": t_reach,
            "hold_frac": float(np.mean(hold < tol)) if hold_errs else 0.0,
            "hold_rms_mm": float(np.sqrt(np.mean(hold**2)) * 1000), "hold_max_mm": float(np.max(hold) * 1000),
            "recovered": recover_s is not None, "recover_s": recover_s, "final_err_mm": info["ee_err"] * 1000,
            "nav_done_t": nav_done_t,
            "nav_timeout": teacher == "B" and nav_done_t is not None and nav_done_t >= _template.base_params.nav_timeout - 1e-6,
            "base_drift_after_nav_mm": base_drift * 1000, "teacher_sat_frac": float(np.mean(sats)),
            "exec_clip_frac": float(np.mean(clips)), "unstable": qd_over > UNSTABLE_STEPS}


def summarise(rows):
    by = {}
    for r in rows:
        by.setdefault((r["teacher"], r["condition"]), []).append(r)
    print(f"\n{'condition':34} | {'A: reach hold>=90% recov':>25} {'hold rms':>8} | {'B: reach hold>=90% recov':>25} {'hold rms':>8} | B base drift med")
    for cond in CONDITIONS:
        if not any((t, cond) in by for t in TEACHERS):
            continue
        cells = []
        for t in TEACHERS:
            rs = by.get((t, cond), [])
            if not rs:
                cells.append(f"{'-':>25} {'-':>8}")
                continue
            n = len(rs)
            rms = np.nanmedian([r["hold_rms_mm"] for r in rs])
            cells.append(f"{sum(r['reached'] for r in rs):>6}/{n:<2}{sum(r['hold_frac'] >= 0.9 for r in rs):>6}/{n:<2}"
                         f"{sum(r['recovered'] for r in rs):>5}/{n:<2} {rms:>7.1f}mm")
        rs_b = by.get(("B", cond), [])
        drift = f"{np.median([r['base_drift_after_nav_mm'] for r in rs_b]):.0f}mm" if rs_b else "-"
        print(f"{cond:34} | {cells[0]} | {cells[1]} | {drift}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16, help="scenarios per condition")
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--teachers", nargs="+", choices=TEACHERS, default=list(TEACHERS))
    parser.add_argument("--conditions", nargs="*", default=None, help="substring filters on condition names")
    parser.add_argument("--out", default="results/expert_noise_robustness.csv")
    args = parser.parse_args()

    conds = [c for c in CONDITIONS if not args.conditions or c == "clean" or any(f in c for f in args.conditions)]
    jobs = [(t, c, i) for c in conds for i in range(args.n) for t in args.teachers]
    rows = []
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(run_episode, jobs, chunksize=2):
            rows.append(r)
            if len(rows) % 100 == 0:
                print(f"[{len(rows)}/{len(jobs)}]", flush=True)
    rows.sort(key=lambda r: (r["teacher"], list(CONDITIONS).index(r["condition"]), r["seed"]))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {args.out}")
    summarise(rows)


if __name__ == "__main__":
    main()
