"""Clean expert-data collection (no action or sensor noise) for the Markovian and stateful teachers.

Runs every frozen scenario once per teacher at 1 kHz and stores the full trajectory, failures included.
Dataset layout (docs/expert_datasets.md, docs/clean_collection_report.md):
  <data>/<Markovian|Stateful>/Clean/{manifest.json, scenarios_*.csv, episodes.csv, train/, test/, qa/}

Per control step k (o_k = pre-action observation):
  1. save o_k (and simulator-truth diagnostics of the same instant)
  2. stateful only: snapshot the teacher memory before the call
  3. query the teacher once (stateful: base_controller, then arm_controller)
  4. save a_expert[k] = [torque_arm (7), wheel_cmds (4)] and the memory after the call
  5. a_executed[k] = a_expert[k] clipped to the execution limits (noise disabled)
  6. execute one 1 ms step and save o_(k+1)
Episode end: 3 s after the first task reach (true EE error < 10 mm, teacher reaching active, t > 8 s),
or the 40 s cap. The original stall stop is recorded as a diagnostic only.

Usage: python -m data_collection.collect --teachers markovian stateful --split train --stop 16   # pilot
       python -m data_collection.collect --teachers markovian stateful --split train test       # the rest
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import time

import numpy as np

from data_collection.scenarios import SETTINGS, TASK_DIR, read_manifest, scenario_dir, sha256, to_scenario

SCHEMA_VERSION = "clean-v1"
CFG = SETTINGS["clean_collection"]
DT = 0.001
CAP_STEPS = int(round(CFG["episode_cap_s"] / DT))
POST_STEPS = int(round(CFG["post_reach_record_s"] / DT))
HOLD_STEPS = int(round(CFG["hold_window_s"] / DT))
TOL = SETTINGS["reach_tolerance"]
TEACHERS = {
    "markovian": {"dir": "Markovian", "teacher_id": "markovian_expert", "teacher_version": "markovian-lt0.50-v1"},
    "stateful": {"dir": "Stateful", "teacher_id": "stateful_template", "teacher_version": "template-origin-goal-v1"},
}
SOURCE_FILES = ["data_collection/expert.py", "data_collection/template_teacher.py", "robotnik_arm_control.py",
                "robotnik_arm_task.py", "core/env.py", "core/interfaces.py", "data_collection/scenarios.py",
                "data_collection/collect.py", "settings.yaml", "configs/robotnik_arm.json",
                "models/summit_xl_arm_description/summit_xl_complete.urdf",
                "models/summit_xl_arm_description/summit_xl_complete_pin.urdf"]
OBS_FIELDS = {"obs_time": (), "obs_base_pos": (3,), "obs_base_yaw": (), "obs_q_mes": (7,), "obs_qd_mes": (7,)}
TRUTH_FIELDS = {"truth_ee_pos": (3,), "truth_ee_err": (), "truth_base_quat_xyzw": (4,), "truth_base_lin_vel": (3,),
                "truth_base_ang_vel": (3,), "truth_base_forward_speed": (), "truth_base_yaw_rate": (),
                "truth_finger_q": (6,), "truth_wheel_q": (4,), "truth_arm_qd": (7,)}


def dataset_dir(teacher: str) -> str:
    return os.path.join(TASK_DIR, SETTINGS["paths"]["data"], TEACHERS[teacher]["dir"], "Clean")


def episode_id(teacher: str, scenario_id: str) -> str:
    return f"{teacher}-clean-{scenario_id}"


def source_info() -> dict:
    """Git revision, working-tree state and SHA-256 of every file that determines the recorded behaviour."""
    import pinocchio
    import pybullet
    import scipy
    import simulation_and_control
    sim_dir = os.path.dirname(simulation_and_control.__file__)
    files = {f: sha256(os.path.join(TASK_DIR, f)) for f in SOURCE_FILES}
    for f in ("sim/pybullet_robot_interface.py", "controllers/servo_motor.py"):
        files[f"simulation_and_control/{f}"] = sha256(os.path.join(sim_dir, f))
    git = lambda *a: subprocess.run(["git", *a], cwd=TASK_DIR, capture_output=True, text=True).stdout.strip()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty_paths": git("status", "--porcelain").splitlines(),
            "source_sha256": files, "code_digest": digest,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                         "pybullet": getattr(pybullet, "__version__", "3.2.7 (requirements.txt)"),
                         "pinocchio": pinocchio.__version__}}


def teacher_parameters(teacher: str, worker_objs=None) -> dict:
    from dataclasses import asdict
    if teacher == "markovian":
        from data_collection.expert import MarkovianExpertParams
        p = asdict(MarkovianExpertParams(lead_time=CFG["markovian_lead_time"]))
        p["base"]["nav_timeout"] = "inf"
        return {"class": "data_collection.expert.MarkovianExpert", "params": p,
                "model": "summit_xl_complete_pin.urdf + 1 kg end-effector body (PyBullet default mass), fixed fingers 0.1 rad"}
    from data_collection import template_teacher as tt
    from robotnik_arm_control import ArmControllerParams, BaseControllerParams
    return {"class": "data_collection.template_teacher.TemplateTeacher (original arm_controller/base_controller)",
            "base": asdict(BaseControllerParams(nav_timeout=tt.BASE_NAV_TIMEOUT)),
            "arm": asdict(ArmControllerParams(reach_limit=tt.ARM_REACH_LIMIT)),
            "standoff_z_points": tt.STANDOFF_Z_POINTS, "standoff_points": tt.STANDOFF_POINTS,
            "goal_rule": "standoff point on the line from the world origin to the target, facing along it (robotnik_arm_task.py)",
            "call_order": "base_controller then arm_controller(reach_active = done)", "arm_dt": tt.CONTROL_DT,
            "privileged_inputs": "true base roll/pitch (simulator quaternion), measured finger and wheel angles in q_pin",
            "phase_codes": {name: i for i, name in enumerate(tt.PHASES)}}


# ---------------------------------------------------------------- worker

_env = _expert = _template = None


def _init_worker():
    global _env, _expert, _template
    from core.env import RobotEnv
    from data_collection.expert import MarkovianExpert, MarkovianExpertParams
    from data_collection.template_teacher import TemplateTeacher
    _env = RobotEnv(reach_tolerance=TOL)
    _expert = MarkovianExpert(MarkovianExpertParams(lead_time=CFG["markovian_lead_time"]))
    _template = TemplateTeacher(_env)


def _record_obs(buf, k, obs, env):
    buf["obs_time"][k] = env.t
    buf["obs_base_pos"][k] = obs.base_pos
    buf["obs_base_yaw"][k] = obs.base_bearing
    buf["obs_q_mes"][k] = obs.q_mes
    buf["obs_qd_mes"][k] = obs.qd_mes
    ee = env.ee_pos()
    buf["truth_ee_pos"][k] = ee
    buf["truth_ee_err"][k] = np.linalg.norm(env.scenario.target_ee_pos - ee)
    buf["truth_base_quat_xyzw"][k] = env.sim.GetBaseOrientation()
    lin, ang = env.client.getBaseVelocity(env.rid)
    buf["truth_base_lin_vel"][k], buf["truth_base_ang_vel"][k] = lin, ang
    buf["truth_base_forward_speed"][k] = np.cos(obs.base_bearing) * lin[0] + np.sin(obs.base_bearing) * lin[1]
    buf["truth_base_yaw_rate"][k] = ang[2]
    motors = np.array(env.sim.GetMotorAngles(0))
    buf["truth_wheel_q"][k], buf["truth_finger_q"][k] = motors[0:4], motors[11:17]
    buf["truth_arm_qd"][k] = np.array(env.sim.GetMotorVelocities(0))[4:11]


def run_episode(job):
    """Collect one episode; returns its summary row. Never raises: failures are recorded."""
    teacher, row, out_path = job
    from core.interfaces import Action
    from data_collection.template_teacher import PHASES
    from robotnik_arm_control import StopCriteria, check_stop
    from simulation_and_control import wrap_angle
    t_wall = time.time()
    sc = to_scenario(row)
    env = _env
    n = CAP_STEPS + 1
    buf = {k: np.full((n, *s), np.nan) for k, s in {**OBS_FIELDS, **TRUTH_FIELDS}.items()}
    tr = {"dt": np.full(CAP_STEPS, np.nan), "a_expert": np.full((CAP_STEPS, 11), np.nan),
          "a_executed": np.full((CAP_STEPS, 11), np.nan), "sampled_action_noise": np.zeros((CAP_STEPS, 11)),
          "clip_flags": np.zeros((CAP_STEPS, 11), bool), "teacher_reach_active": np.zeros(CAP_STEPS, bool),
          "teacher_ee_pos": np.full((CAP_STEPS, 3), np.nan)}
    if teacher == "markovian":
        tr.update({"teacher_dist_err": np.full(CAP_STEPS, np.nan), "teacher_heading_err": np.full(CAP_STEPS, np.nan)})
    else:
        tr.update({"q_des_before": np.full((CAP_STEPS, 7), np.nan), "q_des_after": np.full((CAP_STEPS, 7), np.nan),
                   "qd_des": np.full((CAP_STEPS, 7), np.nan), "phase_before": np.full(CAP_STEPS, -1, np.int8),
                   "phase_after": np.full(CAP_STEPS, -1, np.int8), "nav_done_before": np.zeros(CAP_STEPS, bool),
                   "nav_done_after": np.zeros(CAP_STEPS, bool)})
    lim = np.r_[np.full(7, CFG["exec_torque_limit"]), np.full(4, CFG["exec_wheel_limit"])]
    criteria = StopCriteria(reach_tolerance=TOL, max_steps=0, min_time=CFG["min_reach_time_s"])
    first_reach = original_stop = None
    error = None
    T = 0
    meta_teacher = {}
    try:
        obs = env.reset(sc)
        _record_obs(buf, 0, obs, env)
        if teacher == "stateful":
            _template.reset(target=sc.target_ee_pos, q_hold=obs.q_mes)
            meta_teacher = {"q_hold": _template.q_hold.tolist(), "base_goal_pos_W": _template.goal_pos.tolist(),
                            "base_goal_bearing": _template.goal_bearing, "time_scale_s": _template.base_params.nav_timeout,
                            "initial_phase": _template.base_state.phase, "initial_nav_done": _template.base_state.done,
                            "initial_q_des": _template.q_des.tolist()}
        for k in range(CAP_STEPS):
            t_k = buf["obs_time"][k]
            if teacher == "markovian":
                wheel_cmds, reach_active = _expert.base_command(obs)
                torque = _expert.arm_torque(obs, reach_active)
                tr["teacher_dist_err"][k], tr["teacher_heading_err"][k] = _expert.base_errors(obs)
                tr["teacher_ee_pos"][k] = _expert.ee_pos(obs)
            else:
                motors = np.zeros(17)
                motors[0:4], motors[4:11], motors[11:17] = buf["truth_wheel_q"][k], obs.q_mes, buf["truth_finger_q"][k]
                action, st = _template.act(obs, t=t_k, base_quat=buf["truth_base_quat_xyzw"][k], motor_angles=motors)
                wheel_cmds, torque, reach_active = action.wheel_cmds, action.torque_arm, st["reach_active"]
                tr["q_des_before"][k], tr["q_des_after"][k], tr["qd_des"][k] = st["before"]["q_des"], st["after"]["q_des"], st["qd_des"]
                tr["phase_before"][k], tr["phase_after"][k] = PHASES.index(st["before"]["phase"]), PHASES.index(st["after"]["phase"])
                tr["nav_done_before"][k], tr["nav_done_after"][k] = st["before"]["nav_done"], st["after"]["nav_done"]
                tr["teacher_ee_pos"][k] = st["teacher_ee_pos"]
            a_exp = np.concatenate((torque, wheel_cmds))
            a_exec = np.clip(a_exp, -lim, lim)
            tr["a_expert"][k], tr["a_executed"][k], tr["clip_flags"][k] = a_exp, a_exec, a_exec != a_exp
            tr["teacher_reach_active"][k] = reach_active
            err_k = buf["truth_ee_err"][k]
            if first_reach is None and err_k < TOL and reach_active and t_k > CFG["min_reach_time_s"]:
                first_reach = k
            if original_stop is None:
                msg = check_stop(criteria, t_k, DT, err_k, bool(reach_active))
                if msg:
                    original_stop = {"step": k, "t": float(t_k), "kind": "reached" if msg.startswith("Target reached") else "stall", "message": msg}
            obs, _ = env.step(Action(a_exec[:7].copy(), a_exec[7:].copy()))
            tr["dt"][k] = env.t - t_k
            _record_obs(buf, k + 1, obs, env)
            T = k + 1
            if not (np.isfinite(buf["obs_base_pos"][T]).all() and np.isfinite(buf["obs_q_mes"][T]).all()
                    and np.isfinite(buf["obs_qd_mes"][T]).all() and np.isfinite(a_exp).all()):
                error = "non-finite observation or action"
                break
            if first_reach is not None and T - first_reach >= POST_STEPS:
                break
    except Exception as e:  # recorded as a numerical/simulation failure, never silently dropped
        error = f"{type(e).__name__}: {e}"

    # ---- outcome
    err = buf["truth_ee_err"][:T + 1]
    window = err[max(0, T - HOLD_STEPS):T + 1]
    complete = first_reach is not None and T - first_reach >= POST_STEPS
    hold_pass = bool(complete and np.all(window < TOL))
    terminated = complete and error is None
    truncated = error is None and not complete and T >= CAP_STEPS
    reason = ("numerical_failure: " + error) if error else "post_reach_window_complete" if complete else (
        "time_cap_post_reach_window_incomplete" if first_reach is not None else "time_cap_no_task_reach")
    nav = {"nav_done_step": None, "nav_done_t": None, "nav_done_cause": None, "nav_timeout": False}
    if teacher == "stateful" and T > 0:
        done = tr["nav_done_after"][:T]
        if done.any():
            j = int(np.argmax(done))
            yaw_err = abs(wrap_angle(meta_teacher["base_goal_bearing"] - buf["obs_base_yaw"][j]))
            orient_ok = tr["phase_after"][j] == PHASES.index("orient") and yaw_err < _template.base_params.ang_tol
            t_j = float(buf["obs_time"][j])
            cause = "orient_complete" if orient_ok else "timeout" if t_j > _template.base_params.nav_timeout else "unknown"
            nav = {"nav_done_step": j, "nav_done_t": t_j, "nav_done_cause": cause, "nav_timeout": cause == "timeout"}
    elif teacher == "markovian" and T > 0 and tr["teacher_reach_active"][:T].any():
        j = int(np.argmax(tr["teacher_reach_active"][:T]))
        nav = {"nav_done_step": None, "nav_done_t": None, "nav_done_cause": "not_applicable (no latched done; no timeout)",
               "nav_timeout": False, "first_reach_active_step": j, "first_reach_active_t": float(buf["obs_time"][j])}
    qd = buf["truth_arm_qd"][:T + 1]
    outcome = {
        "T_actions": T, "duration_s": float(buf["obs_time"][T]) if T else 0.0,
        "ever_reached": first_reach is not None,
        "first_reach_step": first_reach, "t_first_reach": float(buf["obs_time"][first_reach]) if first_reach is not None else None,
        "post_reach_window_complete": bool(complete), "final_ee_err_mm": float(err[-1] * 1000),
        "final1s_max_err_mm": float(window.max() * 1000), "final1s_mean_err_mm": float(window.mean() * 1000),
        "final1s_frac_below_tol": float(np.mean(window < TOL)), "final1s_samples": int(len(window)),
        "sustained_hold_pass": hold_pass, "min_ee_err_mm": float(np.nanmin(err) * 1000),
        "ever_within_tol_any_time": bool(np.any(err < TOL)),
        **nav,
        "numerical_failure": error is not None, "failure_message": error,
        "terminated": bool(terminated), "truncated": bool(truncated), "termination_reason": reason,
        "original_stop_diagnostic": original_stop,
        "teacher_torque_saturation_frac": float(np.mean(np.any(np.abs(tr["a_expert"][:T, :7]) >= CFG["exec_torque_limit"] - 1e-9, axis=1))) if T else None,
        "exec_clip_steps": int(tr["clip_flags"][:T].any(axis=1).sum()),
        "max_true_arm_qd": float(np.nanmax(np.abs(qd))), "steps_true_arm_qd_gt5": int(np.sum(np.nanmax(np.abs(qd), axis=1) > 5.0)),
        "wall_time_s": time.time() - t_wall,
    }
    outcome["category"] = ("numerical_failure" if error else "success_sustained_hold" if hold_pass else
                           "reached_hold_failed" if complete else "reached_window_truncated" if first_reach is not None else
                           "not_reached_nav_timeout" if nav.get("nav_timeout") else "not_reached")

    term = np.zeros(T, bool)
    trunc = np.zeros(T, bool)
    if T:
        term[-1], trunc[-1] = terminated, truncated
    cfg_t = TEACHERS[teacher]
    meta = {
        "schema_version": SCHEMA_VERSION, "episode_id": episode_id(teacher, row["scenario_id"]), "scenario_id": row["scenario_id"],
        "split": row["split"], "source": "expert_clean", "teacher_id": cfg_t["teacher_id"], "teacher_version": cfg_t["teacher_version"],
        "random_seeds": {"scenario_seed": row["seed"], "sobol_seed": CFG["sobol_seed"][row["split"]], "action_noise_seed": None},
        "scenario": {k: row[k] for k in row},
        "target_ee_pos_W": sc.target_ee_pos.tolist(),
        "base_start_pose": {"x": float(sc.base_start_pos[0]), "y": float(sc.base_start_pos[1]), "yaw": float(sc.base_start_bearing)},
        "q_init": buf["obs_q_mes"][0].tolist(), "fixed_finger_angles": [0.1] * 6,
        "teacher_parameters": teacher_parameters(teacher), "teacher_initial_state": meta_teacher,
        "physics_parameters": {"physics": "nominal", "floor_friction": 1.0, "mass_scale": 1.0,
                               "wheel_contact": {"contactStiffness": 30000, "contactDamping": 1000, "anisotropicFriction": [1.0, 0.01, 1.0]},
                               "motor_gains": [400, 30], "external_push": None},
        "sensor_configuration": {"added_sensor_noise_std": 0.0, "measurement_delay": None,
                                 "base_velocity_source": "simulator_truth (PyBullet getBaseVelocity); not an encoder/odometry estimate"},
        "control_hz": 1000, "physics_dt": DT,
        "noise_configuration": {"enabled": False, "type": "none", "sampled_action_noise": "all zeros"},
        "execution_interface": {"order": "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]",
                                "limits": {"torque_Nm": CFG["exec_torque_limit"], "wheel_rad_s": CFG["exec_wheel_limit"]},
                                "execution_source": "expert (every step)"},
        "stopping_configuration": {"episode_cap_s": CFG["episode_cap_s"], "post_reach_record_s": CFG["post_reach_record_s"],
                                   "min_reach_time_s": CFG["min_reach_time_s"], "reach_tolerance_m": TOL,
                                   "hold_window_s": CFG["hold_window_s"], "stall_stop": "diagnostic only (not used to terminate)"},
        "outcome": outcome,
        "code_digest": None,  # filled by the parent process (same for every episode of one run)
    }
    arrays = {k: v[:T + 1] for k, v in buf.items()}
    arrays.update({k: v[:T] for k, v in tr.items()})
    arrays["terminated"], arrays["truncated"] = term, trunc
    return meta, arrays, out_path


def collect_job(job):
    """Worker entry point: run one episode and write its file; only the metadata returns to the parent."""
    teacher, row, out_path, code_digest = job
    meta, arrays, path = run_episode((teacher, row, out_path))
    return meta, path, write_episode(meta, arrays, path, code_digest)


# ---------------------------------------------------------------- parent

def write_episode(meta, arrays, out_path, code_digest):
    meta["code_digest"] = code_digest
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, metadata_json=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, out_path)
    return sha256(out_path)


def read_metadata(path: str) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["metadata_json"]))


EPISODE_COLUMNS = ["episode_id", "scenario_id", "split", "teacher_id", "teacher_version", "category", "ever_reached",
                   "t_first_reach", "post_reach_window_complete", "sustained_hold_pass", "final_ee_err_mm",
                   "final1s_max_err_mm", "final1s_mean_err_mm", "final1s_frac_below_tol", "min_ee_err_mm",
                   "nav_done_t", "nav_done_cause", "nav_timeout", "numerical_failure", "terminated", "truncated",
                   "termination_reason", "T_actions", "duration_s", "teacher_torque_saturation_frac", "exec_clip_steps",
                   "max_true_arm_qd", "steps_true_arm_qd_gt5", "original_stop_kind", "original_stop_t", "code_digest", "file"]


def write_episodes_csv(teacher: str):
    root = dataset_dir(teacher)
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "train", "*.npz")) + glob.glob(os.path.join(root, "test", "*.npz"))):
        m = read_metadata(path)
        o = m["outcome"]
        od = o.get("original_stop_diagnostic") or {}
        rows.append({**{k: m.get(k) for k in ("episode_id", "scenario_id", "split", "teacher_id", "teacher_version", "code_digest")},
                     **{k: o.get(k) for k in EPISODE_COLUMNS if k in o}, "original_stop_kind": od.get("kind"),
                     "original_stop_t": od.get("t"), "file": os.path.relpath(path, root)})
    order = {"train": 0, "test": 1}
    rows.sort(key=lambda r: (order[r["split"]], r["scenario_id"]))
    with open(os.path.join(root, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def write_dataset_manifest(teacher: str, info: dict, command: str):
    from core import features
    root = dataset_dir(teacher)
    src = scenario_dir()
    smf = json.load(open(os.path.join(src, "scenario_manifest.json")))
    is_b = teacher == "stateful"
    fields = {
        "metadata_json": "per-episode metadata (JSON string): ids, seeds, scenario, teacher parameters, configuration, outcome",
        **{k: f"observations [T+1{','+','.join(map(str, s)) if s else ''}]: deployable pre-action observation" for k, s in OBS_FIELDS.items()},
        **{k: f"diagnostics [T+1{','+','.join(map(str, s)) if s else ''}]: simulator truth at the same instant as obs index; NOT a student input" for k, s in TRUTH_FIELDS.items()},
        "dt": "transitions [T]: simulated time of each executed action (s)",
        "a_expert": "transitions [T,11]: clean teacher output on o_t = [torque_arm (N m, 7), wheel_cmds (rad/s, FR, FL, BL, BR)]",
        "a_executed": "transitions [T,11]: command sent to the simulator after execution limits",
        "sampled_action_noise": "transitions [T,11]: zeros (noise disabled)",
        "clip_flags": "transitions [T,11] bool: a_executed differs from a_expert",
        "terminated": "transitions [T] bool: true on the last action when the post-reach window completed",
        "truncated": "transitions [T] bool: true on the last action when the 40 s cap ended the episode",
        "teacher_reach_active": "teacher [T] bool: reaching enabled at this call (A: reach_active; B: done after the base call)",
        "teacher_ee_pos": "teacher [T,3]: the teacher's own EE estimate on o_t (A: fixed pitch FK; B: true orientation FK); NOT a student input",
    }
    if is_b:
        fields.update({k: f"teacher memory [T{',7' if 'des' in k else ''}]: label/diagnostic, NOT a student input" for k in
                       ("q_des_before", "q_des_after", "qd_des", "phase_before", "phase_after", "nav_done_before", "nav_done_after")})
    else:
        fields.update({"teacher_dist_err": "teacher diagnostic [T]: signed distance to the standoff point (m)",
                       "teacher_heading_err": "teacher diagnostic [T]: heading error to the target (rad)"})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": f"{TEACHERS[teacher]['dir']}/Clean", "teacher_id": TEACHERS[teacher]["teacher_id"],
        "teacher_version": TEACHERS[teacher]["teacher_version"], "teacher_parameters": teacher_parameters(teacher),
        "feature_definition_version": features.FEATURE_DEFINITION_VERSION,
        "default_student_input": {"dim": 26 if is_b else 21, "names": features.FEATURE_NAMES_B if is_b else features.FEATURE_NAMES_A,
                                  "builder": f"core.features.{'features_b' if is_b else 'features_a'}",
                                  "excluded": "body-frame gravity direction; teacher q_des/phase/done; simulator truth; base velocities (optional ablation only)",
                                  **({"task_context": "base_goal_pos_W, base_goal_bearing from metadata.teacher_initial_state; time_scale_s = 25.0 (nav_timeout)"} if is_b else {})},
        "action_labels": {"dim": 11, "field": "a_expert", "scales_note": "fit separate arm-torque and wheel-speed normalisers on train only"},
        "previous_action_convention": "prev_action[t] = a_executed[t-1]; prev_action[0] = zeros with validity mask False",
        "FK_configuration": features.FK_CONFIG,
        "scenario_split_definition": {"train": "scenarios_train.csv (256, Sobol m=8)", "test": "scenarios_test.csv (64, separate Sobol engine, m=6)",
                                      "sobol": smf["sobol"], "files_sha256": smf["files"], "no_validation_split": True,
                                      "test_use": "closed-loop evaluation only; not for tuning, selection or normalisation"},
        "collection_configuration": {k: CFG[k] for k in CFG}, "reach_tolerance_m": TOL,
        "outcome_definitions": {
            "task_reach": "first pre-action observation with true EE error < 10 mm, teacher reaching active on that step, and t > 8 s",
            "post_reach_window_complete": "3 s (3000 actions) recorded after the first task reach before the 40 s cap",
            "final1s": "observations T-1000..T (1001 samples, 1.0 s)",
            "sustained_hold_pass": "post-reach window complete and every final-1 s sample below 10 mm",
            "nav_timeout": "B only: done latched by the 25 s timeout, not by completing the orient phase",
            "initial_bc_eligible": "train split, valid data, post-reach window complete, sustained-hold pass"},
        "fields": fields, "source": info, "collection_command": command,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", choices=list(TEACHERS), default=list(TEACHERS))
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--start", type=int, default=0, help="first scenario index (per split)")
    parser.add_argument("--stop", type=int, default=None, help="one past the last scenario index (per split)")
    parser.add_argument("--workers", type=int, default=22)
    parser.add_argument("--supersede", action="store_true",
                        help="re-collect episodes that already exist; old files move to qa/superseded/ and are logged")
    args = parser.parse_args()

    info = source_info()
    command = "python -m data_collection.collect " + " ".join(
        [f"--teachers {' '.join(args.teachers)}", f"--split {' '.join(args.split)}", f"--start {args.start}"]
        + ([f"--stop {args.stop}"] if args.stop is not None else []) + (["--supersede"] if args.supersede else []))
    src = scenario_dir()
    smf = json.load(open(os.path.join(src, "scenario_manifest.json")))
    for f, h in smf["files"].items():
        if sha256(os.path.join(src, f)) != h:
            raise RuntimeError(f"frozen scenario manifest {f} changed since it was generated")
    rows = {s: read_manifest(os.path.join(src, f"scenarios_{s}.csv")) for s in args.split}

    jobs, attempt_ts = [], datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for teacher in args.teachers:
        root = dataset_dir(teacher)
        os.makedirs(os.path.join(root, "qa"), exist_ok=True)
        for f in ("scenarios_train.csv", "scenarios_test.csv"):
            shutil.copyfile(os.path.join(src, f), os.path.join(root, f))
        for s in args.split:
            for row in rows[s][args.start:args.stop]:
                path = os.path.join(root, s, episode_id(teacher, row["scenario_id"]) + ".npz")
                if os.path.exists(path):
                    old = read_metadata(path)
                    if not args.supersede:
                        if old["code_digest"] != info["code_digest"]:
                            raise RuntimeError(f"{path} was recorded with different code; re-run with --supersede")
                        continue
                    dest = os.path.join(root, "qa", "superseded", attempt_ts, s)
                    os.makedirs(dest, exist_ok=True)
                    shutil.move(path, os.path.join(dest, os.path.basename(path)))
                    with open(os.path.join(root, "qa", "attempts.jsonl"), "a") as f:
                        f.write(json.dumps({"event": "superseded", "episode_id": old["episode_id"], "moved_to": os.path.relpath(dest, root),
                                            "old_code_digest": old["code_digest"], "new_code_digest": info["code_digest"],
                                            "time": attempt_ts}) + "\n")
                jobs.append((teacher, row, path, info["code_digest"]))
    print(f"{len(jobs)} episodes to collect, code digest {info['code_digest'][:12]}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.Pool(args.workers, initializer=_init_worker) as pool:
            for i, (meta, path, digest) in enumerate(pool.imap_unordered(collect_job, jobs), 1):
                o = meta["outcome"]
                with open(os.path.join(os.path.dirname(os.path.dirname(path)), "qa", "attempts.jsonl"), "a") as f:
                    f.write(json.dumps({"event": "written", "episode_id": meta["episode_id"], "file": os.path.relpath(path, TASK_DIR),
                                        "sha256": digest, "code_digest": info["code_digest"], "category": o["category"],
                                        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                        "command": command}) + "\n")
                print(f"[{i}/{len(jobs)}] {meta['episode_id']:32} {o['category']:24} T={o['T_actions']:>5} "
                      f"t_reach={o['t_first_reach']} final1s_max={o['final1s_max_err_mm']:.2f}mm wall={o['wall_time_s']:.1f}s", flush=True)
    for teacher in args.teachers:
        write_dataset_manifest(teacher, info, command)
        rows_out = write_episodes_csv(teacher)
        cats = {}
        for r in rows_out:
            cats.setdefault(r["split"], {}).setdefault(r["category"], 0)
            cats[r["split"]][r["category"]] += 1
        print(f"{teacher}: {len(rows_out)} episodes on disk; {json.dumps(cats)}")
    print(f"wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
