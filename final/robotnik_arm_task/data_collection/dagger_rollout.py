"""DAgger block mixing with current-state expert labels and aligned failed prefixes."""
import json
import os
from pathlib import Path

import numpy as np

from core.interfaces import Action
from evaluation.evaluate import make_robot_env


def action_vector(action):
    v = np.r_[action.torque_arm, action.wheel_cmds].astype(float)
    if v.shape != (11,):
        raise ValueError("action must have 7 torques and 4 wheels")
    return v


def collect_episode(student, expert, scenario, config, beta, rng, env_factory=make_robot_env):
    if not np.isfinite(beta) or not 0 <= beta <= 1 or config["interface"]["control_hz"] != 1000:
        raise ValueError("DAgger uses beta in [0,1], 1kHz control, and ten-step blocks")
    ev, itf = config["evaluation"], config["interface"]
    cap = int(round(ev["episode_cap_s"]*1000))
    post, hold = int(round(ev["post_reach_s"]*1000)), int(round(ev["hold_window_s"]*1000))
    limits = np.r_[np.full(7, itf["torque_limit"]), np.full(4, itf["wheel_limit"])]
    obs_rows, actions, errs = [], [], []
    failure = first_reach = first_arrival = unexecuted = None
    env = None
    t = 0.
    start = __import__("time").monotonic()

    def record(obs):
        return (float(env.t), np.array(obs.base_pos, copy=True), float(obs.base_bearing),
                np.array(obs.q_mes, copy=True), np.array(obs.qd_mes, copy=True))

    try:
        env = env_factory()
        obs = env.reset(scenario)
        student.reset()
        expert.reset()
        obs_rows.append(record(obs))
        errs.append(float(np.linalg.norm(scenario.target_ee_pos-env.ee_pos())))
        for k in range(cap):
            t = float(env.t)
            if first_arrival is None and errs[k] < ev["reach_tolerance_m"]:
                first_arrival = t
            if first_reach is None and errs[k] < ev["reach_tolerance_m"] and t > ev["min_reach_time_s"]:
                first_reach = k
            if k % 10 == 0:
                chosen_expert = bool(rng.random() < beta)
            # Both controllers see the very same pre-action observation every millisecond.
            e, s = action_vector(expert.act(obs)), action_vector(student.act(obs))
            raw = np.array(student.last_raw, copy=True)
            active = bool(expert.expert.base_command(obs)[1])
            if raw.shape != (9,) or not all(np.isfinite(v).all() for v in (e, s, raw)):
                unexecuted = {"step": k, "expert": e.tolist(), "student": s.tolist(), "student_raw": raw.tolist()}
                raise FloatingPointError("invalid unexecuted action query")
            a = np.clip(e if chosen_expert else s, -limits, limits)
            # Append only after a real transition; unexecuted queries never enter the T rows.
            nxt, info = env.step(Action(a[:7], a[7:]))
            valid = all(np.isfinite(v).all() for v in (obs.base_pos, obs.base_bearing, obs.q_mes, obs.qd_mes, obs.target_ee_pos, e))
            phase = 2 if first_reach is not None else int(active)
            actions.append((e, raw, s, a, chosen_expert, k//10, valid, phase, errs[k]))
            obs_rows.append(record(nxt))
            errs.append(float(info["ee_err"]))
            obs, t = nxt, float(env.t)
            if not all(np.isfinite(v).all() for v in (obs.base_pos, obs.base_bearing, obs.q_mes, obs.qd_mes, errs[-1])):
                raise FloatingPointError(f"non-finite terminal observation after step {k}")
            if first_reach is not None and len(actions)-first_reach >= post:
                break
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
    T = len(actions)
    complete = failure is None and first_reach is not None and T-first_reach >= post
    outcome = {"scenario_id": scenario.scenario_id, "steps": T, "duration_s": t,
               "ever_reached": first_reach is not None, "first_reach_step": first_reach,
               "t_first_reach": None if first_reach is None else first_reach*.001,
               "first_arrival_t": first_arrival, "post_reach_window_complete": complete,
               "sustained_hold_pass": bool(complete and np.all(np.array(errs[max(0,T-hold):]) < ev["reach_tolerance_m"])),
               "final_err_mm": float(errs[-1]*1000) if errs else None, "failure": failure,
               "termination": "failure" if failure else "post_reach_window_complete" if complete else
                              "time_cap_post_reach_window_incomplete" if first_reach is not None else "time_cap_no_task_reach"}
    names = ("obs_time", "obs_base_pos", "obs_base_yaw", "obs_q_mes", "obs_qd_mes")
    widths = ((), (3,), (), (7,), (7,))
    arrays = {n: np.array([r[j] for r in obs_rows], dtype=float).reshape((-1,)+widths[j]) for j, n in enumerate(names)}
    names = ("expert_action", "student_raw", "student_action", "executed_action", "expert_executes", "block_id",
             "valid_supervision", "phase", "ee_err")
    widths = ((11,), (9,), (11,), (11,), (), (), (), (), ())
    for j, name in enumerate(names):
        dtype = bool if name in ("expert_executes", "valid_supervision") else int if name in ("block_id", "phase") else float
        arrays[name] = np.array([r[j] for r in actions], dtype=dtype).reshape((-1,)+widths[j])
    switches = int(np.count_nonzero(np.diff(arrays["expert_executes"].astype(int))))
    return arrays, {"outcome": outcome, "beta_requested": beta, "block_steps": 10,
                    "expert_execution_fraction": float(arrays["expert_executes"].mean()) if T else None,
                    "switches": switches, "unexecuted_query": unexecuted,
                    "wall_s": __import__("time").monotonic()-start}


def save_episode(path, arrays, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays, metadata_json=np.array(json.dumps(metadata)))
    os.replace(tmp, path)
