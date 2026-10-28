"""Hold test: can a policy keep a target teacher A has just reached, and is its closed loop stable there?

Per scenario, teacher A drives the base and arm until the true end-effector error has stayed below 10 mm for 1 s.
The student then takes over for 10 s: the arm only, with the teacher's wheels ('arm'), or the whole robot ('full').
Reported: seconds until the error first leaves 10 mm (None = never), final and maximum error, maximum joint speed.

At the switch state, with qd set to 0, the arm closed loop M dq'' = (K - G) dq + D dqd is linearised by central
differences for student and teacher (K = dtau/dq, D = dtau/dqd, G = gravity Jacobian, M = arm inertia, all from the
teacher's Pinocchio model). It reports the largest real eigenvalue, the symmetric stiffness eigenvalue range and the
median damping diagonal. The student's arm-torque RMSE on the teacher's own approach and hold states up to the switch
is reported too. The test runs on fitting scenarios as a diagnostic; it never selects a model.

Usage (from final/robotnik_arm_task/): python -m evaluation.hold_test --artifact <model.npz> [--scenarios train-0001 ...]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
SCENARIOS = ["train-0001", "train-0035", "train-0067", "train-0108", "train-0142", "train-0178", "train-0216", "train-0255"]
TOL, CAP_S, SWITCH_AFTER_HOLD_S, STUDENT_S, FD_STEP = 0.01, 30.0, 1.0, 10.0, 1e-4


def load_scenario(scenario_id: str):
    from evaluation.test_sets import scenario_from_row
    with open(TASK_DIR / "data/Markovian/Clean_253/scenarios_train.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["scenario_id"] == scenario_id)
    return scenario_from_row(row, "train")


def _gravity(expert, obs):
    import pinocchio as pin
    return pin.computeGeneralizedGravity(expert.model, expert.data, expert.q_pin(obs))[expert.arm_idx_v].copy()


def linearise(expert, student, obs) -> dict:
    """Arm closed loop at obs with qd = 0, for teacher and student."""
    import pinocchio as pin
    from core.interfaces import Observation
    base = Observation(np.asarray(obs.base_pos, float).copy(), obs.base_bearing, np.asarray(obs.q_mes, float).copy(),
                       np.zeros(7), np.asarray(obs.target_ee_pos, float).copy())
    at = lambda q=None, qd=None: Observation(base.base_pos, base.base_bearing, base.q_mes if q is None else q,
                                             base.qd_mes if qd is None else qd, base.target_ee_pos)
    M = pin.crba(expert.model, expert.data, expert.q_pin(base))[np.ix_(expert.arm_idx_v, expert.arm_idx_v)]
    M = 0.5 * (M + M.T)
    unit = lambda j: np.eye(7)[j] * FD_STEP
    G = np.column_stack([(_gravity(expert, at(q=base.q_mes + unit(j))) - _gravity(expert, at(q=base.q_mes - unit(j))))
                         / (2 * FD_STEP) for j in range(7)])
    out = {}
    for name, f in (("expert", lambda o: expert.arm_torque(o, True)),
                    ("student", lambda o: student.predict(student.features(o))[:7])):
        K = np.column_stack([(f(at(q=base.q_mes + unit(j))) - f(at(q=base.q_mes - unit(j)))) / (2 * FD_STEP) for j in range(7)])
        D = np.column_stack([(f(at(qd=unit(j))) - f(at(qd=-unit(j)))) / (2 * FD_STEP) for j in range(7)])
        A = np.block([[np.zeros((7, 7)), np.eye(7)], [np.linalg.solve(M, K - G), np.linalg.solve(M, D)]])
        stiffness = np.linalg.eigvalsh(0.5 * ((K - G) + (K - G).T))
        out[name] = {"max_real_eig": float(np.linalg.eigvals(A).real.max()), "stiffness_eig_min": float(stiffness.min()),
                     "stiffness_eig_max": float(stiffness.max()), "damping_diag_median": float(np.median(np.diag(D)))}
    return out


def _episode(student, expert, env, scenario, mode: str):
    """Teacher until 1 s inside 10 mm, then the student for 10 s; returns the after-switch trace and diagnostics."""
    from core.interfaces import Action
    obs = env.reset(scenario)
    student.reset()
    err = float(np.linalg.norm(scenario.target_ee_pos - env.ee_pos()))
    inside_since = switch_k = switch_obs = None
    on_dist, after = {"approach": [], "hold": []}, []
    for k in range(int(CAP_S * 1000)):
        teacher = expert.act(obs)
        if switch_k is None and inside_since is not None and env.t - inside_since >= SWITCH_AFTER_HOLD_S:
            switch_k, switch_obs = k, obs
        if switch_k is None:
            if expert.base_command(obs)[1]:
                e = np.clip(student.predict(student.features(obs))[:7], -30, 30) - teacher.torque_arm
                on_dist["hold" if err < TOL else "approach"].append(e)
            action = teacher
        else:
            a = student.act(obs)
            action = Action(a.torque_arm, teacher.wheel_cmds if mode == "arm" else a.wheel_cmds)
            after.append((env.t, err, float(np.abs(obs.qd_mes).max())))
            if k - switch_k >= STUDENT_S * 1000:
                break
        obs, info = env.step(action)
        err = info["ee_err"]
        inside_since = (inside_since if inside_since is not None else env.t) if err < TOL else None
        if not np.isfinite(err) or err > 5.0:
            break
    return switch_k, switch_obs, np.array(after), on_dist


def hold_test(artifact: str, scenario_id: str) -> dict:
    """Both modes and the linearisation for one model and one scenario."""
    from core.env import RobotEnv
    from evaluation.evaluate import MarkovianReference
    from training.common.policies import MLPPolicy
    student, expert = MLPPolicy.load(artifact), MarkovianReference().expert
    env = RobotEnv(reach_tolerance=TOL)
    scenario = load_scenario(scenario_id)
    out = {"scenario_id": scenario_id}
    try:
        for mode in ("arm", "full"):
            switch_k, switch_obs, after, on_dist = _episode(student, expert, env, scenario, mode)
            if switch_k is None:
                out[mode] = None  # the teacher never held this target within the cap
                continue
            t, err, qd = after.T
            left = np.nonzero(err >= TOL)[0]
            out[mode] = {"switch_t_s": float(t[0]), "seconds_until_leaving_10mm": None if not len(left) else float(t[left[0]] - t[0]),
                         "final_err_mm": float(err[-1] * 1000), "max_err_mm": float(err.max() * 1000),
                         "max_joint_speed_rad_s": float(qd.max()), "stayed_within_10mm": not len(left),
                         "student_seconds": float(t[-1] - t[0])}
            if mode == "arm":
                out["linearised"] = linearise(expert, student, switch_obs)
                out["on_distribution_arm_rmse_nm"] = {k: float(np.sqrt(np.mean(np.square(v)))) if v else None
                                                      for k, v in on_dist.items()}
    finally:
        env.close()
    return out


def summarise(results: list[dict]) -> dict:
    arm = [r["arm"] for r in results if r.get("arm")]
    full = [r["full"] for r in results if r.get("full")]
    lin = [r["linearised"]["student"] for r in results if r.get("linearised")]
    med = lambda v: float(np.median(v)) if v else None
    return {"scenarios": len(results), "arm_stayed": sum(a["stayed_within_10mm"] for a in arm),
            "full_stayed": sum(f["stayed_within_10mm"] for f in full),
            "arm_median_final_err_mm": med([a["final_err_mm"] for a in arm]),
            "median_max_real_eig": med([x["max_real_eig"] for x in lin]),
            "median_stiffness_eig_max": med([x["stiffness_eig_max"] for x in lin]),
            "median_damping_diag": med([x["damping_diag_median"] for x in lin])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifact", required=True, help="MLP NumPy export (.npz)")
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    from training.common.config import pin_rollout_threads
    pin_rollout_threads()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(args.workers, len(args.scenarios))) as pool:
        results = pool.starmap(hold_test, [(args.artifact, s) for s in args.scenarios])
    print(json.dumps({"summary": summarise(results), "scenarios": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
