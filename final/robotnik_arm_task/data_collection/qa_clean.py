"""QA, initial-BC training manifests and trajectory coverage for the clean expert datasets.

Checks (each reported with its tolerance):
  inventory   one selected recording per teacher and scenario; identical frozen scenario files; no train/test overlap
  structure   shapes, T+1 observations for T actions, finite values, timestamps, zero noise, no clipping, end flags
  outcomes    every stored outcome flag recomputed independently from the stored arrays
  markovian   every stored action re-queried from its stored pre-action observation with the frozen expert
  stateful    teacher re-run from the recorded initial memory on the stored observations and teacher-side
              measurements (true q_des is never an input); actions and memory compared at every step;
              memory reset and continuity checked directly on the stored arrays
  alignment   simulator-truth EE recomputed from each stored pre-action observation (same index)
  features    default student inputs (21 / 26) built from stored raw data with the deployable FK estimator
  re-execution a few episodes re-simulated in fresh processes and compared array by array
Then writes training_manifest.csv (initial-BC selection) and trajectory coverage for each dataset.

Output: <dataset>/qa/{qa_report.json, episode_checks.csv, coverage_*.png}, <dataset>/training_manifest.csv

Usage: python -m data_collection.qa_clean [--expect pilot|full] [--reexec 4]
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os

import numpy as np

from data_collection.collect import (CFG, DT, HOLD_STEPS, POST_STEPS, TEACHERS, TOL, dataset_dir, episode_id,
                                     read_metadata, source_info)
from data_collection.scenarios import DIMENSIONS, read_manifest, scenario_dir, sha256

EXACT = 0.0          # tolerance for re-query / replay / recomputation from stored float64 values
TIME_TOL = 1e-9      # s, stored timestamps vs k * dt (the clock is an accumulated float, as in the template)
HOLD_CHATTER_SPEED = 0.5   # rad/s, median arm joint-speed norm after the first reach; settled holds are ~0.01
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

_fk = _expert = _env = _template = None


def _init():
    global _fk, _expert, _env, _template
    from core.env import RobotEnv
    from core.features import DeployableFK
    from data_collection.expert import MarkovianExpert, MarkovianExpertParams
    from data_collection.template_teacher import TemplateTeacher
    _fk = DeployableFK()
    _expert = MarkovianExpert(MarkovianExpertParams(lead_time=CFG["markovian_lead_time"]))
    _env = RobotEnv(reach_tolerance=TOL)   # used for its Pinocchio model only; never stepped here
    _template = TemplateTeacher(_env)


def load(path):
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d.pop("metadata_json")))
    return d


def recompute_outcome(d):
    """Independent recomputation of the outcome flags from stored arrays."""
    T = len(d["a_expert"])
    err, t, ra = d["truth_ee_err"], d["obs_time"], d["teacher_reach_active"]
    ok = (err[:T] < TOL) & ra & (t[:T] > CFG["min_reach_time_s"])
    first = int(np.argmax(ok)) if ok.any() else None
    complete = first is not None and T - first >= POST_STEPS
    w = err[T - HOLD_STEPS:T + 1] if T >= HOLD_STEPS else err
    return {"ever_reached": first is not None, "first_reach_step": first, "post_reach_window_complete": complete,
            "final_ee_err_mm": float(err[T] * 1000), "final1s_max_err_mm": float(w.max() * 1000),
            "final1s_mean_err_mm": float(w.mean() * 1000), "final1s_frac_below_tol": float(np.mean(w < TOL)),
            "sustained_hold_pass": bool(complete and np.all(w < TOL)),
            "ended_correctly": bool(complete and T - first == POST_STEPS) or (not complete and T == int(round(CFG["episode_cap_s"] / DT)))}


def check_episode(path):
    """All per-episode checks; returns a flat dict of results."""
    from core.features import features_a, features_b
    from core.interfaces import Observation
    from data_collection.template_teacher import PHASES
    import pinocchio as pin
    d = load(path)
    m, o = d["meta"], d["meta"]["outcome"]
    teacher = "markovian" if m["teacher_id"] == TEACHERS["markovian"]["teacher_id"] else "stateful"
    T = len(d["a_expert"])
    r = {"episode_id": m["episode_id"], "scenario_id": m["scenario_id"], "split": m["split"], "teacher": teacher, "T": T,
         "file_matches_id": os.path.basename(path) == m["episode_id"] + ".npz" and os.path.basename(os.path.dirname(path)) == m["split"]}
    problems = []

    # structure
    obs_keys = [k for k in d if k.startswith(("obs_", "truth_"))]
    tr_keys = [k for k in d if k not in obs_keys and k != "meta"]
    r["shapes_ok"] = all(len(d[k]) == T + 1 for k in obs_keys) and all(len(d[k]) == T for k in tr_keys) and T == o["T_actions"]
    r["obs_per_action"] = f"{T + 1}/{T}"
    r["finite_ok"] = all(np.isfinite(d[k]).all() for k in d if k != "meta" and np.asarray(d[k]).dtype.kind == "f")
    r["time_max_dev_s"] = float(np.max(np.abs(d["obs_time"] - np.arange(T + 1) * DT)))
    r["dt_max_dev_s"] = float(np.max(np.abs(d["dt"] - DT)))
    r["time_ok"] = d["obs_time"][0] == 0.0 and r["time_max_dev_s"] < TIME_TOL and r["dt_max_dev_s"] < TIME_TOL
    r["noise_zero"] = bool(np.all(d["sampled_action_noise"] == 0) and not m["noise_configuration"]["enabled"]
                           and m["sensor_configuration"]["added_sensor_noise_std"] == 0.0)
    r["executed_equals_expert"] = bool(np.array_equal(d["a_executed"], d["a_expert"]))
    r["clip_steps"] = int(d["clip_flags"].any(1).sum())
    r["end_flags_ok"] = bool(not d["terminated"][:-1].any() and not d["truncated"][:-1].any()
                             and d["terminated"][-1] == o["terminated"] and d["truncated"][-1] == o["truncated"]
                             and (d["terminated"][-1] != d["truncated"][-1]))
    r["ee_err_consistent"] = float(np.max(np.abs(np.linalg.norm(d["truth_ee_pos"] - np.array(m["target_ee_pos_W"]), axis=1) - d["truth_ee_err"])))

    # outcomes
    rec = recompute_outcome(d)
    mism = [k for k in rec if k in o and (rec[k] != o[k] if not isinstance(rec[k], float) else abs(rec[k] - o[k]) > 1e-9)]
    r["outcome_recomputed_ok"] = not mism and rec["ended_correctly"]
    r["outcome_mismatches"] = ";".join(mism) + ("" if rec["ended_correctly"] else ";end_step")

    # teacher re-query / replay
    obs = lambda k: Observation(base_pos=d["obs_base_pos"][k], base_bearing=float(d["obs_base_yaw"][k]), q_mes=d["obs_q_mes"][k],
                                qd_mes=d["obs_qd_mes"][k], target_ee_pos=np.array(m["target_ee_pos_W"]))
    act_diff = mem_diff = 0.0
    disc = 0
    if teacher == "markovian":
        for k in range(T):
            ob = obs(k)
            wheels, ra = _expert.base_command(ob)
            a = np.concatenate((_expert.arm_torque(ob, ra), wheels))
            act_diff = max(act_diff, float(np.max(np.abs(a - d["a_expert"][k]))))
            disc += int(ra != d["teacher_reach_active"][k])
            mem_diff = max(mem_diff, float(np.max(np.abs(_expert.ee_pos(ob) - d["teacher_ee_pos"][k]))))
        r["requery_max_action_diff"], r["requery_reach_flag_mismatch"], r["requery_teacher_ee_diff"] = act_diff, disc, mem_diff
        r["teacher_ok"] = act_diff <= EXACT and disc == 0 and mem_diff <= EXACT
    else:
        init = m["teacher_initial_state"]
        _template.reset(target=np.array(m["target_ee_pos_W"]), q_hold=np.array(init["q_hold"]))
        diffs = {k: 0.0 for k in ("a_expert", "q_des_before", "q_des_after", "qd_des", "teacher_ee_pos")}
        flags = {k: 0 for k in ("phase_before", "phase_after", "nav_done_before", "nav_done_after", "teacher_reach_active")}
        for k in range(T):
            motors = np.zeros(17)
            motors[0:4], motors[4:11], motors[11:17] = d["truth_wheel_q"][k], d["obs_q_mes"][k], d["truth_finger_q"][k]
            a, st = _template.act(obs(k), t=float(d["obs_time"][k]), base_quat=d["truth_base_quat_xyzw"][k], motor_angles=motors)
            vals = {"a_expert": np.concatenate((a.torque_arm, a.wheel_cmds)), "q_des_before": st["before"]["q_des"],
                    "q_des_after": st["after"]["q_des"], "qd_des": st["qd_des"], "teacher_ee_pos": st["teacher_ee_pos"]}
            for key, v in vals.items():
                diffs[key] = max(diffs[key], float(np.max(np.abs(v - d[key][k]))))
            fl = {"phase_before": PHASES.index(st["before"]["phase"]), "phase_after": PHASES.index(st["after"]["phase"]),
                  "nav_done_before": st["before"]["nav_done"], "nav_done_after": st["after"]["nav_done"], "teacher_reach_active": st["reach_active"]}
            for key, v in fl.items():
                flags[key] += int(v != d[key][k])
        r.update({f"replay_{k}_maxdiff": v for k, v in diffs.items()})
        r.update({f"replay_{k}_mismatch": v for k, v in flags.items()})
        # reset and continuity directly on stored memory
        qh = np.array(init["q_hold"])
        cont = {
            "reset_q_des": bool(np.array_equal(d["q_des_before"][0], qh) and np.array_equal(qh, d["obs_q_mes"][0])),
            "reset_q_hold_is_config": bool(np.allclose(qh, [0.0, 1.57, 0.0, 1.0, 0.0, 1.0, 0.0], atol=0, rtol=0)),
            "reset_phase_done": bool(d["phase_before"][0] == 0 and not d["nav_done_before"][0]),
            "q_des_continuity": bool(np.array_equal(d["q_des_before"][1:], d["q_des_after"][:-1])),
            "phase_continuity": bool(np.array_equal(d["phase_before"][1:], d["phase_after"][:-1])),
            "done_continuity": bool(np.array_equal(d["nav_done_before"][1:], d["nav_done_after"][:-1])),
            "done_latched": bool(np.all(np.diff(d["nav_done_after"].astype(int)) >= 0)),
            "phase_monotone": bool(np.all(np.diff(d["phase_after"].astype(int)) >= 0)),
            "reach_active_equals_done": bool(np.array_equal(d["teacher_reach_active"], d["nav_done_after"])),
            "hold_before_done": bool(np.all(d["q_des_after"][~d["nav_done_after"]] == qh) and np.all(d["qd_des"][~d["nav_done_after"]] == 0)),
        }
        r.update({f"mem_{k}": v for k, v in cont.items()})
        r["teacher_ok"] = all(v <= EXACT for v in diffs.values()) and not any(flags.values()) and all(cont.values())

    # alignment: truth EE recomputed from the observation and teacher-side measurements at the same index
    dm = _env.dyn_model
    idx = np.unique(np.r_[np.arange(0, T + 1, 97), np.arange(max(0, T - 5), T + 1)])
    same, shifted = 0.0, []
    for k in idx:
        motors = np.zeros(17)
        motors[0:4], motors[4:11], motors[11:17] = d["truth_wheel_q"][k], d["obs_q_mes"][k], d["truth_finger_q"][k]
        q_pin = dm.ReoderJoints2PinVec(np.concatenate((d["obs_base_pos"][k], d["truth_base_quat_xyzw"][k], motors)), "pos")
        pin.forwardKinematics(dm.pin_model, dm.pin_data, q_pin)
        pin.updateFramePlacements(dm.pin_model, dm.pin_data)
        ee = dm.pin_data.oMf[_env.ee_frame_id].translation
        same = max(same, float(np.max(np.abs(ee - d["truth_ee_pos"][k]))))
        if k < T:
            shifted.append(float(np.max(np.abs(ee - d["truth_ee_pos"][k + 1]))))
    r["align_truth_ee_maxdiff"], r["align_shifted_by_one_median"] = same, float(np.median(shifted))
    r["align_ok"] = same <= 1e-12

    # deployable features (inputs for the T pre-action observations)
    args = (d["obs_base_pos"][:T], d["obs_base_yaw"][:T], d["obs_q_mes"][:T], d["obs_qd_mes"][:T])
    if teacher == "markovian":
        X = features_a(*args, np.array(m["target_ee_pos_W"]), _fk)
    else:
        ti = m["teacher_initial_state"]
        X = features_b(*args, d["obs_time"][:T], np.array(m["target_ee_pos_W"]), np.array(ti["base_goal_pos_W"]),
                       ti["base_goal_bearing"], ti["time_scale_s"], _fk)
    fk_err = np.linalg.norm(_fk.ee_pos(d["obs_base_pos"][:T:10], d["obs_base_yaw"][:T:10], d["obs_q_mes"][:T:10]) - d["truth_ee_pos"][:T:10], axis=1)
    r["feature_dim"], r["features_finite"] = X.shape[1], bool(np.isfinite(X).all())
    r["fk_vs_truth_median_mm"], r["fk_vs_truth_max_mm"] = float(np.median(fk_err) * 1000), float(fk_err.max() * 1000)

    r["integrity_ok"] = bool(r["file_matches_id"] and r["shapes_ok"] and r["finite_ok"] and r["time_ok"] and r["noise_zero"]
                             and r["executed_equals_expert"] and r["end_flags_ok"] and r["ee_err_consistent"] < 1e-12
                             and r["outcome_recomputed_ok"] and r["teacher_ok"] and r["align_ok"] and r["features_finite"]
                             and r["feature_dim"] == (21 if teacher == "markovian" else 26) and not o["numerical_failure"])
    if not r["integrity_ok"]:
        problems = [k for k in ("file_matches_id", "shapes_ok", "finite_ok", "time_ok", "noise_zero", "executed_equals_expert",
                                "end_flags_ok", "outcome_recomputed_ok", "teacher_ok", "align_ok", "features_finite") if not r[k]]
    r["integrity_problems"] = ";".join(problems)

    # trajectory summary for coverage
    first = o["first_reach_step"]
    ra = d["teacher_reach_active"]
    seg = {"navigate": int(np.sum(~ra)), "approach": int(np.sum(ra[:first] if first is not None else ra)),
           "hold": int(T - first) if first is not None else 0}
    r.update({f"steps_{k}": v for k, v in seg.items()})
    if teacher == "stateful":
        pa, done = d["phase_after"], d["nav_done_after"]
        for i, name in enumerate(PHASES):
            r[f"steps_phase_{name}_pre_done"] = int(np.sum((pa == i) & ~done))
        trans = [(int(a), int(b)) for a, b in zip(pa[:-1], pa[1:]) if a != b]
        r["phase_sequence"] = "->".join([PHASES[pa[0]]] + [PHASES[b] for _, b in trans])
    else:
        bp = _expert.params.base
        de, he = np.abs(d["teacher_dist_err"]), np.abs(d["teacher_heading_err"])
        drive = de >= bp.pos_tol
        r["steps_base_turn_in_place"] = int(np.sum(drive & (he >= _expert.params.drive_heading_tol)))
        r["steps_base_drive"] = int(np.sum(drive & (he < _expert.params.drive_heading_tol)))
        r["steps_base_orient"] = int(np.sum(~drive & (he >= bp.ang_tol)))
        r["steps_base_stopped"] = int(np.sum(~drive & (he < bp.ang_tol)))
        r["reach_active_toggles"] = int(np.sum(np.diff(ra.astype(int)) != 0))
    # label quality in the post-reach window: period-2 (Nyquist) chatter with saturated torques
    if first is not None and T - first > 10:
        qd_h, a_h = d["obs_qd_mes"][first:T], d["a_expert"][first:T, :7]
        j = int(np.argmax(qd_h.std(0)))
        x = qd_h[:, j] - qd_h[:, j].mean()
        spec = np.abs(np.fft.rfft(x))
        r["hold_joint_speed_median"] = float(np.median(np.linalg.norm(qd_h, axis=1)))
        r["hold_sat_frac"] = float(np.mean(np.any(np.abs(a_h) >= CFG["exec_torque_limit"] - 1e-9, axis=1)))
        r["hold_dominant_joint"] = j + 1
        r["hold_dominant_freq_hz"] = float(np.fft.rfftfreq(len(x), DT)[int(np.argmax(spec[1:])) + 1])
        r["hold_ee_peak_dev_mm"] = float(np.abs(d["truth_ee_pos"][first:T + 1] - d["truth_ee_pos"][first:T + 1].mean(0)).max() * 1000)
    else:
        r.update({"hold_joint_speed_median": None, "hold_sat_frac": None, "hold_dominant_joint": None,
                  "hold_dominant_freq_hz": None, "hold_ee_peak_dev_mm": None})
    r["flag_hold_chatter"] = bool(r["hold_joint_speed_median"] is not None and r["hold_joint_speed_median"] > HOLD_CHATTER_SPEED)
    ra_idx = int(np.argmax(ra)) if ra.any() else None
    r["t_reach_active"] = float(d["obs_time"][ra_idx]) if ra_idx is not None else None
    r["err_at_reach_active_mm"] = float(d["truth_ee_err"][ra_idx] * 1000) if ra_idx is not None else None
    within = (d["truth_ee_err"][:T] < TOL) & ra
    r["t_first_within_tol_ungated"] = float(d["obs_time"][int(np.argmax(within))]) if within.any() else None
    q = d["obs_q_mes"]
    r.update({f"q{j}_min": float(q[:, j].min()) for j in range(7)})
    r.update({f"q{j}_max": float(q[:, j].max()) for j in range(7)})
    r["category"] = o["category"]
    for k in ("target_dist", "rel_azimuth", "target_z", "start_x", "start_y", "start_yaw"):
        r[k] = m["scenario"][k]
    return r


def reexec(job):
    """Re-simulate one episode in this (fresh) process and compare every stored array."""
    from data_collection import collect
    teacher, row, path = job
    collect._init_worker()
    meta, arrays, _ = collect.run_episode((teacher, row, path))
    stored = load(path)
    diffs = {k: float(np.max(np.abs(arrays[k].astype(float) - stored[k].astype(float)))) if arrays[k].shape == stored[k].shape else float("inf")
             for k in arrays}
    worst = max(diffs, key=diffs.get)
    return {"episode_id": stored["meta"]["episode_id"], "arrays": len(diffs), "max_abs_diff": diffs[worst], "worst_field": worst,
            "bitwise_identical": all(v == 0 for v in diffs.values())}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(c - h, 4), round(c + h, 4))


def coverage(rows, teacher, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    from data_collection.template_teacher import PHASES
    cov = {}
    for split in ("train", "test"):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        c = {"episodes": len(rs), "steps_total": int(sum(r["T"] for r in rs)),
             "steps_by_segment": {s: int(sum(r[f"steps_{s}"] for r in rs)) for s in ("navigate", "approach", "hold")}}
        if teacher == "stateful":
            c["steps_by_phase_pre_done"] = {p: int(sum(r[f"steps_phase_{p}_pre_done"] for r in rs)) for p in PHASES}
            c["episodes_visiting_phase"] = {p: int(sum(r[f"steps_phase_{p}_pre_done"] > 0 for r in rs)) for p in PHASES}
            seqs = {}
            for r in rs:
                seqs[r["phase_sequence"]] = seqs.get(r["phase_sequence"], 0) + 1
            c["phase_sequences"] = seqs
        else:
            c["steps_by_base_mode"] = {k: int(sum(r[f"steps_base_{k}"] for r in rs)) for k in ("turn_in_place", "drive", "orient", "stopped")}
            c["episodes_with_reach_active_toggles"] = int(sum(r["reach_active_toggles"] > 1 for r in rs))
        ra = [r["err_at_reach_active_mm"] for r in rs if r["err_at_reach_active_mm"] is not None]
        c["ee_err_at_reach_activation_mm"] = {"min": min(ra), "median": float(np.median(ra)), "max": max(ra)} if ra else None
        c["arm_joint_range_rad"] = {f"q{j}": [min(r[f"q{j}_min"] for r in rs), max(r[f"q{j}_max"] for r in rs)] for j in range(7)}
        fails = [r for r in rs if r["category"] != "success_sustained_hold"]
        bins = {"target_dist": [0.5, 1.33, 2.17, 3.0], "target_z": [0.1, 0.48, 0.87, 1.25],
                "rel_azimuth": [-np.pi, -3 * np.pi / 4, -np.pi / 4, np.pi / 4, 3 * np.pi / 4, np.pi], "start_yaw": [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]}
        c["failures_by_bin"] = {}
        for var, edges in bins.items():
            tot, _ = np.histogram([r[var] for r in rs], edges)
            fl, _ = np.histogram([r[var] for r in fails], edges)
            c["failures_by_bin"][var] = [{"bin": f"[{edges[i]:.2f},{edges[i+1]:.2f})", "episodes": int(tot[i]), "failures": int(fl[i])} for i in range(len(tot))]
        cov[split] = c

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ("train", "test")):
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        ok = [r for r in rs if r["category"] == "success_sustained_hold"]
        bad = [r for r in rs if r["category"] != "success_sustained_hold"]
        ax.scatter([r["rel_azimuth"] for r in ok], [r["target_dist"] for r in ok], s=16, color=BLUE, edgecolor="#fcfcfb", lw=0.8,
                   label=f"sustained hold ({len(ok)})")
        ax.scatter([r["rel_azimuth"] for r in bad], [r["target_dist"] for r in bad], s=46, marker="X", color=ORANGE, edgecolor=INK, lw=0.6,
                   label=f"other outcome ({len(bad)})")
        for r in bad:
            ax.annotate(f"{r['scenario_id']}\n{r['category']}", (r["rel_azimuth"], r["target_dist"]), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, color=MUTED)
        ax.set(xlabel="target azimuth relative to initial heading (rad)", ylabel="target distance (m)", xlim=(-np.pi, np.pi), ylim=(0.5, 3.0))
        ax.set_title(f"{TEACHERS[teacher]['dir']} {split} (n={len(rs)})", loc="left", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "coverage_outcomes.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    rs = [r for r in rows if r["split"] == "train"]
    if rs:
        t_rch = [r["t_first_within_tol_ungated"] for r in rs if r["t_first_within_tol_ungated"] is not None]
        t_task = [r["t_reach_active"] for r in rs if r["t_reach_active"] is not None]
        ax.hist(t_task, bins=np.arange(0, 41, 1), color=BLUE, alpha=0.9, label="reaching activated", edgecolor="#fcfcfb")
        ax.hist(t_rch, bins=np.arange(0, 41, 1), histtype="step", color=ORANGE, lw=2, label="first within 10 mm (no 8 s gate)")
        ax.axvline(CFG["min_reach_time_s"], color=MUTED, ls="--", lw=1)
        ax.text(CFG["min_reach_time_s"] + 0.3, ax.get_ylim()[1] * 0.9, "8 s task gate", color=MUTED)
        ax.set(xlabel="time (s)", ylabel="train episodes")
        ax.set_title(f"{TEACHERS[teacher]['dir']}: when reaching starts and first succeeds", loc="left", color=INK)
        ax.legend(frameon=False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "coverage_timing.png"), dpi=170)
    plt.close(fig)
    return cov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", choices=list(TEACHERS), default=list(TEACHERS))
    parser.add_argument("--expect", choices=["pilot", "full"], default="full")
    parser.add_argument("--reexec", type=int, default=4, help="episodes per teacher to re-simulate in fresh processes")
    parser.add_argument("--workers", type=int, default=22)
    args = parser.parse_args()

    src = scenario_dir()
    smf = json.load(open(os.path.join(src, "scenario_manifest.json")))
    rows = {s: read_manifest(os.path.join(src, f"scenarios_{s}.csv")) for s in ("train", "test")}
    expected = {"train": rows["train"][:CFG["pilot_scenarios"]] if args.expect == "pilot" else rows["train"],
                "test": [] if args.expect == "pilot" else rows["test"]}
    info = source_info()
    summary_all = {}
    for teacher in args.teachers:
        root = dataset_dir(teacher)
        qa = os.path.join(root, "qa")
        os.makedirs(qa, exist_ok=True)
        rep = {"teacher": teacher, "expect": args.expect, "tolerances": {"requery_replay_recompute": EXACT, "timestamps_s": TIME_TOL,
                                                                         "truth_ee_alignment_m": 1e-12}}
        # inventory
        files = sorted(glob.glob(os.path.join(root, "train", "*.npz")) + glob.glob(os.path.join(root, "test", "*.npz")))
        want = {episode_id(teacher, r["scenario_id"]): r["split"] for s in expected for r in expected[s]}
        have = {os.path.basename(f)[:-4]: os.path.basename(os.path.dirname(f)) for f in files}
        digests = {read_metadata(f)["code_digest"] for f in files}
        manifest = json.load(open(os.path.join(root, "manifest.json")))
        rep["inventory"] = {
            "expected": len(want), "found": len(have), "missing": sorted(set(want) - set(have)), "unexpected": sorted(set(have) - set(want)),
            "split_mismatch": sorted(k for k in want if k in have and have[k] != want[k]),
            "one_recording_per_scenario": len(have) == len(set(have)) and set(have) == set(want),
            "code_digests": sorted(digests), "single_code_digest": len(digests) == 1,
            "manifest_code_digest_matches": manifest["source"]["code_digest"] in digests and len(digests) == 1,
            "current_code_digest_matches": info["code_digest"] in digests,
            "scenario_files_match_frozen": all(sha256(os.path.join(root, f)) == h for f, h in smf["files"].items()),
            "train_test_overlap": smf["checks"]["train_test_identical_tuples"] + smf["checks"]["train_test_shared_ids"],
            "superseded_attempts": sum(1 for l in open(os.path.join(qa, "attempts.jsonl")) if '"superseded"' in l) if os.path.exists(os.path.join(qa, "attempts.jsonl")) else 0,
        }
        with mp.Pool(args.workers, initializer=_init) as pool:
            checks = list(pool.imap_unordered(check_episode, files, chunksize=1))
        checks.sort(key=lambda r: (r["split"] != "train", r["scenario_id"]))
        keys = sorted({k for r in checks for k in r}, key=lambda k: list(checks[0]).index(k) if k in checks[0] else 999)
        with open(os.path.join(qa, "episode_checks.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(checks)

        # deterministic re-execution in fresh processes
        if args.reexec:
            tr_rows = [r for r in checks if r["split"] == "train"]
            te_rows = [r for r in checks if r["split"] == "test"]
            n_te = min(len(te_rows), args.reexec // 2)
            n_tr = min(len(tr_rows), args.reexec - n_te)
            spaced = lambda rs, n: [rs[i] for i in np.unique(np.linspace(0, len(rs) - 1, n).astype(int))] if n else []
            pick = spaced(tr_rows, n_tr) + spaced(te_rows, n_te)
            jobs = [(teacher, next(x for s in rows for x in rows[s] if x["scenario_id"] == r["scenario_id"]),
                     os.path.join(root, r["split"], r["episode_id"] + ".npz")) for r in pick]
            with mp.get_context("spawn").Pool(len(jobs), maxtasksperchild=1) as pool:
                rep["reexecution"] = pool.map(reexec, jobs)

        # training manifest
        tm = []
        for r in checks:
            elig = r["split"] == "train" and r["integrity_ok"] and r["category"] == "success_sustained_hold"
            if r["split"] == "test":
                reason = "held-out test scenario (closed-loop evaluation only)"
            elif not r["integrity_ok"]:
                reason = f"excluded: data integrity ({r['integrity_problems']})"
            elif r["category"] != "success_sustained_hold":
                reason = f"excluded: outcome {r['category']}"
            else:
                reason = "included: valid data, post-reach window complete, sustained-hold pass"
            tm.append({"episode_id": r["episode_id"], "scenario_id": r["scenario_id"], "split": r["split"], "outcome": r["category"],
                       "integrity_status": "valid" if r["integrity_ok"] else f"invalid:{r['integrity_problems']}",
                       "eligible_initial_bc": elig, "reason": reason,
                       "offline_test_subset": r["split"] == "test" and r["integrity_ok"] and r["category"] == "success_sustained_hold",
                       "label_quality_flags": "hold_chatter" if r["flag_hold_chatter"] else "",
                       "file": f"{r['split']}/{r['episode_id']}.npz", "T_actions": r["T"]})
        with open(os.path.join(root, "training_manifest.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tm[0]))
            w.writeheader()
            w.writerows(tm)

        # summary
        def count(split, pred):
            return sum(1 for r in checks if r["split"] == split and pred(r))
        rep["checks"] = {k: sum(1 for r in checks if not r[k]) for k in
                         ("file_matches_id", "shapes_ok", "finite_ok", "time_ok", "noise_zero", "executed_equals_expert", "end_flags_ok",
                          "outcome_recomputed_ok", "teacher_ok", "align_ok", "features_finite", "integrity_ok")}
        rep["checks"] = {f"failing_{k}": v for k, v in rep["checks"].items()}
        rep["max_values"] = {k: max(r[k] for r in checks) for k in checks[0] if k.startswith(("time_max", "dt_max", "requery_", "replay_", "align_truth", "ee_err_consistent"))}
        rep["clip_steps_total"] = sum(r["clip_steps"] for r in checks)
        rep["feature_dims"] = sorted({r["feature_dim"] for r in checks})
        rep["fk_vs_truth_mm"] = {"median_of_medians": float(np.median([r["fk_vs_truth_median_mm"] for r in checks])),
                                 "max": float(max(r["fk_vs_truth_max_mm"] for r in checks))}
        if teacher == "stateful":
            rep["memory_checks_failing"] = {k: sum(1 for r in checks if not r[k]) for k in checks[0] if k.startswith("mem_")}
        rep["alignment_negative_control_median_shift_mm"] = float(np.median([r["align_shifted_by_one_median"] for r in checks]) * 1000)
        rep["outcomes"] = {}
        for s in ("train", "test"):
            n = count(s, lambda r: True)
            if not n:
                continue
            cats = {}
            for r in checks:
                if r["split"] == s:
                    cats[r["category"]] = cats.get(r["category"], 0) + 1
            hold = count(s, lambda r: r["category"] == "success_sustained_hold")
            rep["outcomes"][s] = {"episodes": n, "categories": cats, "sustained_hold_pass": hold, "wilson95": wilson(hold, n),
                                  "eligible_initial_bc": sum(1 for t in tm if t["split"] == s and t["eligible_initial_bc"]),
                                  "offline_test_subset": sum(1 for t in tm if t["split"] == s and t["offline_test_subset"])}
        chat = [r for r in checks if r["flag_hold_chatter"]]
        rep["label_quality"] = {
            "hold_chatter_rule": f"median arm joint-speed norm after the first task reach > {HOLD_CHATTER_SPEED} rad/s",
            "hold_chatter_episodes": {s: sum(1 for r in chat if r["split"] == s) for s in ("train", "test")},
            "hold_chatter_eligible_initial_bc": sum(1 for r in chat if r["split"] == "train" and r["integrity_ok"] and r["category"] == "success_sustained_hold"),
            "hold_chatter_dominant_freq_hz": sorted({round(r["hold_dominant_freq_hz"], 1) for r in chat}),
            "hold_chatter_dominant_joints": {j: sum(1 for r in chat if r["hold_dominant_joint"] == j) for j in sorted({r["hold_dominant_joint"] for r in chat})},
            "hold_chatter_sat_frac_median": float(np.median([r["hold_sat_frac"] for r in chat])) if chat else None,
            "hold_chatter_ee_peak_dev_mm_max": max((r["hold_ee_peak_dev_mm"] for r in chat), default=None),
            "other_holds_joint_speed_median": float(np.median([r["hold_joint_speed_median"] for r in checks if r["hold_joint_speed_median"] is not None and not r["flag_hold_chatter"]])),
            "note": "Flag only; the initial-BC selection rule is unchanged. Stored for a later, explicit selection decision.",
        }
        rep["coverage"] = coverage(checks, teacher, qa)
        with open(os.path.join(qa, "qa_report.json"), "w") as f:
            json.dump(rep, f, indent=2, default=str)
        summary_all[teacher] = rep
        print(f"\n=== {teacher} ===")
        print(json.dumps({k: rep[k] for k in ("inventory", "checks", "max_values", "outcomes", "feature_dims", "fk_vs_truth_mm",
                                               "alignment_negative_control_median_shift_mm", "clip_steps_total")}, indent=1, default=str))
        if teacher == "stateful":
            print(json.dumps(rep["memory_checks_failing"]))
        print(json.dumps(rep.get("reexecution"), default=str))
    # both datasets describe the same scenarios
    if len(summary_all) == 2:
        same = all(sha256(os.path.join(dataset_dir(t), f)) == h for t in TEACHERS for f, h in smf["files"].items())
        print(f"\nScenario files identical across teachers and frozen manifest: {same}")


if __name__ == "__main__":
    main()
