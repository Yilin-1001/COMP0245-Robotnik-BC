"""DAgger collection for teacher A: the student drives (beta = 0), teacher A labels every visited state.

One DAgger round = the current student policy rolled out alone on the 221 fitting scenarios plus fresh development
scenes biased towards low and near targets (where R4 fails, docs/reports/2026-10-29__report__r4-test-diagnosis.md),
with the clean teacher action a_expert stored as the label on every pre-action observation.

collect.run_episode is reused unchanged (stopping rule, outcome and arrays as in Clean/DART): it queries teacher A on
every observation and hands its action to env.step. The StudentEnv proxy below discards that action and executes the
student's own command on the same observation instead, so the label is the teacher's answer at a state the student
reached. Episodes run until 3 s after the first task reach or the 40 s cap; failed episodes are kept (they hold the
states DAgger is for).

Development scenes: ids dev-r<round>-<i>, numpy default_rng([seed, round, i]), inside the training ranges (start
x, y in +-0.5 m, any heading and azimuth); with probability low_frac the target height is drawn from the low band
and with probability near_frac the distance from the near band, otherwise from the full range. Validation and test
scenarios are never used.

Output: data/Markovian/DAgger/<YYYY-MM-DD>__collection__<round>-<digest12>/ (same layout as a DART collection, with
dataset_kind "dagger").

Usage (from final/robotnik_arm_task/):
    python -m data_collection.collect_dagger --student <model.npz> --round r1 [--dev-scenes 64] [--limit 4]
"""
import argparse
import datetime
import glob
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np

from data_collection import collect
from data_collection.scenarios import COLUMNS, TASK_DIR, read_manifest, sha256, write_manifest

SCHEMA_VERSION = "dagger-v1"
DATA_ROOT = os.path.join(TASK_DIR, collect.SETTINGS["paths"]["data"], "Markovian", "DAgger")
LIMITS = np.r_[np.full(7, collect.CFG["exec_torque_limit"]), np.full(4, collect.CFG["exec_wheel_limit"])]
SPLIT = os.path.join(TASK_DIR, "experiments/splits/clean253.json")
FITTING_SOURCE = os.path.join(TASK_DIR, "data/Markovian/Clean_253/scenarios_train.csv")
DEV = {"seed": 20261030, "low_frac": 0.7, "low_z": (0.1, 0.5), "near_frac": 0.5, "near_d": (0.5, 1.0),
       "full_z": (0.1, 1.25), "full_d": (0.5, 3.0)}


def fitting_rows() -> list[dict]:
    split = json.load(open(SPLIT))
    ids = set(split["fitting_ids"]["clean253"])
    if ids & set(split["validation_ids"]):
        raise ValueError("fitting and validation overlap")
    rows = [r for r in read_manifest(FITTING_SOURCE) if r["scenario_id"] in ids]
    if len(rows) != len(ids):
        raise ValueError(f"{len(ids) - len(rows)} fitting scenarios missing from {FITTING_SOURCE}")
    return rows


def dev_rows(round_name: str, n: int) -> list[dict]:
    r_index = int(round_name.lstrip("r"))
    rows = []
    for i in range(n):
        rng = np.random.default_rng([DEV["seed"], r_index, i])
        z = rng.uniform(*DEV["low_z"]) if rng.random() < DEV["low_frac"] else rng.uniform(*DEV["full_z"])
        d = rng.uniform(*DEV["near_d"]) if rng.random() < DEV["near_frac"] else rng.uniform(*DEV["full_d"])
        sx, sy = rng.uniform(-0.5, 0.5, 2)
        yaw, az = rng.uniform(-np.pi, np.pi, 2)
        row = {"scenario_id": f"dev-{round_name}-{i:04d}", "split": "train", "index": 900000 + 1000 * r_index + i,
               "seed": DEV["seed"] + 1000 * r_index + i, **{f"u_{k}": float("nan") for k in
                                                              ("target_dist", "rel_azimuth", "target_z", "start_x", "start_y", "start_yaw")},
               "target_dist": float(d), "rel_azimuth": float(az), "target_z": float(z), "start_x": float(sx),
               "start_y": float(sy), "start_yaw": float(yaw),
               "target_x": float(sx + d * np.cos(yaw + az)), "target_y": float(sy + d * np.sin(yaw + az))}
        rows.append({k: row[k] for k in COLUMNS})
    return rows


def episode_id(scenario_id: str, round_name: str) -> str:
    return f"markovian-dagger-{round_name}-{scenario_id}"


class StudentEnv:
    """RobotEnv proxy for collect.run_episode: executes the student's command instead of the teacher's."""

    def __init__(self, env, student):
        self._env, self._student = env, student

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, scenario):
        self._student.reset()
        self._obs = self._env.reset(scenario)
        self.student_raw, self.executed = [], []
        return self._obs

    def step(self, action):
        from core.interfaces import Action
        a = self._student.act(self._obs)
        raw = np.asarray(self._student.last_raw, dtype=float)
        executed = np.clip(np.r_[a.torque_arm, a.wheel_cmds], -LIMITS, LIMITS)
        if not np.isfinite(executed).all():
            raise FloatingPointError("non-finite student action")
        self.student_raw.append(raw)
        self.executed.append(executed)
        self._obs, info = self._env.step(Action(executed[:7].copy(), executed[7:].copy()))
        return self._obs, info


_student_env = None


def _init_worker(student_path: str):
    global _student_env
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from training.common.policies import MLPPolicy
    collect._init_worker()
    _student_env = StudentEnv(collect._env, MLPPolicy.load(student_path, backend="numpy"))
    collect._env = _student_env


def collect_job(job):
    row, round_name, out_path, digest, student_info = job
    meta, arrays, _ = collect.run_episode(("markovian", row, out_path))
    T = len(arrays["a_expert"])
    executed = np.array(_student_env.executed[:T]).reshape(T, 11)
    raw = np.array(_student_env.student_raw[:T]).reshape(T, 9)
    clean_exec = np.clip(arrays["a_expert"], -LIMITS, LIMITS)
    arrays["a_executed"], arrays["student_raw"] = executed, raw
    arrays["sampled_action_noise"] = executed - clean_exec  # student minus teacher command, for diagnostics only
    arrays["clip_flags"] = np.zeros((T, 11), bool)
    meta.update(schema_version=SCHEMA_VERSION, episode_id=episode_id(row["scenario_id"], round_name),
                source="expert_dagger", dagger_round=round_name)
    meta["noise_configuration"] = {"enabled": False, "type": "none (DAgger: the student executes, beta = 0)"}
    meta["dagger"] = {"round": round_name, "beta": 0.0, "student": student_info,
                      "scene_kind": "development" if row["scenario_id"].startswith("dev-") else "fitting",
                      "student_teacher_arm_rmse_nm": float(np.sqrt(np.mean((executed[:, :7] - clean_exec[:, :7]) ** 2))) if T else None}
    meta["execution_interface"]["execution_source"] = "student policy (beta = 0); label a_expert = teacher A on the same observation"
    return meta, out_path, collect.write_episode(meta, arrays, out_path, digest)


EPISODE_COLUMNS = ["episode_id", "scenario_id", "scene_kind", "category", "ever_reached", "sustained_hold_pass",
                   "t_first_reach", "final_ee_err_mm", "final1s_max_err_mm", "T_actions", "student_teacher_arm_rmse_nm",
                   "code_digest", "sha256", "file"]


def write_episodes_csv(root: str) -> list[dict]:
    import csv
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "train", "*.npz"))):
        m = collect.read_metadata(path)
        o = m["outcome"]
        rows.append({"episode_id": m["episode_id"], "scenario_id": m["scenario_id"], "scene_kind": m["dagger"]["scene_kind"],
                     "student_teacher_arm_rmse_nm": m["dagger"]["student_teacher_arm_rmse_nm"],
                     "code_digest": m["code_digest"], "sha256": sha256(path), "file": os.path.relpath(path, root),
                     **{k: o[k] for k in ("category", "ever_reached", "sustained_hold_pass", "t_first_reach",
                                          "final_ee_err_mm", "final1s_max_err_mm", "T_actions")}})
    with open(os.path.join(root, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True, help="student NumPy export (.npz) that drives this round")
    parser.add_argument("--round", required=True, help="round name r<k>, e.g. r1")
    parser.add_argument("--dev-scenes", type=int, default=64, help="fresh low/near-biased development scenes")
    parser.add_argument("--no-fitting", action="store_true", help="development scenes only")
    parser.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke runs)")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.round.startswith("r") or not args.round[1:].isdigit():
        raise ValueError("--round must look like r1, r2, ...")

    student = os.path.abspath(args.student)
    with np.load(student) as z:
        smeta = json.loads(str(z["metadata_json"]))
    if smeta.get("teacher") != "A" or (smeta.get("feature_definition") or "features-v1") not in ("features-v1", "features-v1-gate"):
        raise ValueError(f"{student}: DAgger here expects a teacher-A features-v1 or features-v1-gate student")
    student_info = {"path": os.path.relpath(student, TASK_DIR), "sha256": sha256(student), "run_id": smeta.get("run_id")}
    rows = ([] if args.no_fitting else fitting_rows()) + dev_rows(args.round, args.dev_scenes)
    split = json.load(open(SPLIT))
    held_out = set(split["validation_ids"])
    if any(r["scenario_id"] in held_out or r["scenario_id"].startswith("test-") for r in rows):
        raise ValueError("DAgger may not collect validation or test scenarios")

    info = collect.source_info()
    digest = hashlib.sha256(json.dumps({"collect": info["code_digest"], "student": student_info["sha256"], "round": args.round,
                                        "dev": [DEV, args.dev_scenes, args.no_fitting],
                                        "collect_dagger_py": sha256(os.path.abspath(__file__))},
                                       sort_keys=True).encode()).hexdigest()
    name = f"{args.round}-{digest[:12]}"
    existing = glob.glob(os.path.join(DATA_ROOT, f"*__collection__{name}"))
    root = existing[0] if existing else os.path.join(DATA_ROOT, f"{datetime.date.today()}__collection__{name}")
    os.makedirs(os.path.join(root, "qa"), exist_ok=True)
    write_manifest(rows, os.path.join(root, "scenarios_train.csv"))

    jobs = []
    for row in rows:
        path = os.path.join(root, "train", episode_id(row["scenario_id"], args.round) + ".npz")
        if os.path.exists(path) and collect.read_metadata(path)["code_digest"] == digest:
            continue
        jobs.append((row, args.round, path, digest, student_info))
    jobs = jobs[:args.limit] if args.limit else jobs
    command = f"python -m data_collection.collect_dagger --student {student_info['path']} --round {args.round} " \
              f"--dev-scenes {args.dev_scenes}" + (" --no-fitting" if args.no_fitting else "")
    print(f"{len(jobs)} episodes to collect into {os.path.relpath(root, TASK_DIR)}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.get_context("fork").Pool(args.workers, initializer=_init_worker, initargs=(student,)) as pool:
            for i, (meta, path, file_sha) in enumerate(pool.imap_unordered(collect_job, jobs), 1):
                with open(os.path.join(root, "qa", "attempts.jsonl"), "a") as f:
                    f.write(json.dumps({"event": "written", "episode_id": meta["episode_id"], "sha256": file_sha,
                                        "code_digest": digest, "category": meta["outcome"]["category"], "command": command,
                                        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}) + "\n")
                if i % 20 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)

    episodes = write_episodes_csv(root)
    kinds = {k: [e for e in episodes if e["scene_kind"] == k] for k in ("fitting", "development")}
    counts = {"train_episodes": len(episodes),
              **{f"{k}_episodes": len(v) for k, v in kinds.items()},
              **{f"{k}_student_hold": sum(e["sustained_hold_pass"] for e in v) for k, v in kinds.items()}}
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_id": f"Markovian/DAgger/{os.path.basename(root)}", "dataset_kind": "dagger",
        "teacher_id": collect.TEACHERS["markovian"]["teacher_id"], "teacher_version": collect.TEACHERS["markovian"]["teacher_version"],
        "teacher_parameters": collect.teacher_parameters("markovian"), "round": args.round, "beta": 0.0,
        "student": student_info, "development_scenes": {**DEV, "n": args.dev_scenes}, "fitting_scenes": not args.no_fitting,
        "code_digest": digest, "source": info, "collection_configuration": collect.CFG, "reach_tolerance_m": collect.TOL,
        "counts": counts,
        "fields": {"a_expert": "label: clean teacher-A output on o_t (the state the student reached)",
                   "a_executed": "the student's clipped command, sent to the simulator",
                   "student_raw": "the student's 9-dimensional prediction before limits",
                   "sampled_action_noise": "a_executed - clip(a_expert): student/teacher disagreement, diagnostics only",
                   "other fields": "as in Clean (see data/Markovian/Clean/manifest.json)"},
        "outcome_definitions": "as in Clean, for the student's own rollout",
        "collection_command": command,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps({"root": os.path.relpath(root, TASK_DIR), **counts, "wall_s": round(time.time() - t0, 1)}))
    return root


if __name__ == "__main__":
    main()
