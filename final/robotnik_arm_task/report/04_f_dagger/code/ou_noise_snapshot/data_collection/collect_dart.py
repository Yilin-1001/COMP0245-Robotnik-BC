"""DART collection for teacher A: the clean collector's episodes with gated OU noise on the executed action.

Labels stay the clean teacher output a_expert on the (perturbed) pre-action observation; only the simulator command
changes: a_executed = clip(clip(a_expert) + noise). The noise follows a protocol file
(experiments/configs/dart/<protocol>.yaml), one per noise level, so an ablation adds a protocol and never edits data.

collect.run_episode is reused unchanged, so stopping and outcome definitions equal the Clean data. It steps a
NoisyEnv, which adds the noise inside step() and logs it; the arrays and metadata are then marked as DART.

Output: data/Markovian/DART/<YYYY-MM-DD>__collection__<protocol>-<digest12>/
  manifest.json, scenarios_train.csv, episodes.csv, qa/attempts.jsonl, train/markovian-dart-<scenario>-d<draw>.npz

Usage: python -m data_collection.collect_dart --protocol experiments/configs/dart/d2-s075-x2.yaml [--limit 4]
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import multiprocessing as mp
import os
import time

import numpy as np
import yaml

from data_collection import collect
from data_collection.scenarios import TASK_DIR, read_manifest, sha256, write_manifest

SCHEMA_VERSION = "dart-v1"
DATA_ROOT = os.path.join(TASK_DIR, collect.SETTINGS["paths"]["data"], "Markovian", "DART")
NOISE_KEYS = {"type", "tau_s", "alpha", "arm_sigma_nm", "wheel_sigma_rad_s", "wheel_off", "all_off_after_reach_s"}
LIMITS = np.r_[np.full(7, collect.CFG["exec_torque_limit"]), np.full(4, collect.CFG["exec_wheel_limit"])]


def load_protocol(path: str) -> dict:
    protocol = yaml.safe_load(open(path))
    noise = protocol.get("noise") or {}
    if protocol.get("teacher") != "A" or set(noise) != NOISE_KEYS or noise["type"] != "ou" \
            or noise["wheel_off"] != "first_reach_active":
        raise ValueError(f"{path}: not a teacher-A OU protocol with keys {sorted(NOISE_KEYS)}")
    return protocol


def protocol_rows(protocol: dict) -> list[dict]:
    """The protocol's scenario rows, in source order; refuses any validation scenario."""
    sc = protocol["scenarios"]
    rows = read_manifest(os.path.join(TASK_DIR, sc["source"]))
    split = json.load(open(os.path.join(TASK_DIR, sc["split_manifest"])))
    ids = split
    for part in sc["ids"].split("."):
        ids = ids[part]
    ids = set(ids)
    if ids & set(split["validation_ids"]):
        raise ValueError("a DART protocol may not collect validation scenarios")
    chosen = [r for r in rows if r["scenario_id"] in ids]
    if len(chosen) != len(ids):
        raise ValueError(f"{len(ids) - len(chosen)} protocol scenarios are missing from {sc['source']}")
    return chosen


def episode_id(scenario_id: str, draw: int) -> str:
    return f"markovian-dart-{scenario_id}-d{draw}"


class NoisyEnv:
    """RobotEnv proxy for collect.run_episode: adds the protocol's gated OU noise to every executed action.

    Gates use the same rules as the episode loop: wheel noise latches off at the first reach-active step, and all
    noise stops all_off_after_reach_s after the first task reach (reach active, true EE error < tol, t > 8 s).
    """

    def __init__(self, env, expert, noise: dict):
        self._env, self._expert, self._cfg = env, expert, noise
        self._a = np.exp(-collect.DT / noise["tau_s"])
        self._b = np.sqrt(1.0 - self._a ** 2)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def configure(self, rng, alpha: float) -> None:
        self._rng, self.alpha = rng, float(alpha)
        self.sigma = alpha * np.r_[np.full(7, self._cfg["arm_sigma_nm"]), np.full(4, self._cfg["wheel_sigma_rad_s"])]

    def reset(self, scenario):
        self._obs = self._env.reset(scenario)
        self._state, self._k = np.zeros(11), 0
        self.wheel_off_step = self.reach_step = None
        self.noise, self.executed = [], []
        return self._obs

    def step(self, action):
        from core.interfaces import Action
        _, active = self._expert.base_command(self._obs)
        err = float(np.linalg.norm(self._env.scenario.target_ee_pos - self._env.ee_pos()))
        if self.reach_step is None and active and err < collect.TOL and self._env.t > collect.CFG["min_reach_time_s"]:
            self.reach_step = self._k
        if active and self.wheel_off_step is None:
            self.wheel_off_step = self._k
        self._state = self._a * self._state + self._b * self.sigma * self._rng.standard_normal(11)
        n = self._state.copy()
        if self.wheel_off_step is not None:
            n[7:] = 0.0
        if self.reach_step is not None and (self._k - self.reach_step) * collect.DT >= self._cfg["all_off_after_reach_s"]:
            n[:] = 0.0
        executed = np.clip(np.r_[action.torque_arm, action.wheel_cmds] + n, -LIMITS, LIMITS)
        self.noise.append(n)
        self.executed.append(executed)
        self._k += 1
        self._obs, info = self._env.step(Action(executed[:7].copy(), executed[7:].copy()))
        return self._obs, info


_noisy = None


def _init_worker(protocol: dict):
    global _noisy
    collect._init_worker()
    _noisy = NoisyEnv(collect._env, collect._expert, protocol["noise"])
    collect._env = _noisy


def collect_job(job):
    row, draw, out_path, digest, protocol, protocol_sha = job
    seed = [int(protocol["seed"]), int(row["index"]), int(draw)]
    rng = np.random.default_rng(seed)
    _noisy.configure(rng, rng.uniform(*protocol["noise"]["alpha"]))
    meta, arrays, _ = collect.run_episode(("markovian", row, out_path))
    T = len(arrays["a_expert"])
    noise, executed = np.array(_noisy.noise[:T]).reshape(T, 11), np.array(_noisy.executed[:T]).reshape(T, 11)
    clean_exec = np.clip(arrays["a_expert"], -LIMITS, LIMITS)
    arrays["a_executed"], arrays["sampled_action_noise"] = executed, noise
    arrays["clip_flags"] = executed != clean_exec + noise
    meta.update(schema_version=SCHEMA_VERSION, episode_id=episode_id(row["scenario_id"], draw), source="expert_dart",
                dart_draw=int(draw))
    meta["random_seeds"]["action_noise_seed"] = seed
    meta["noise_configuration"] = {
        "enabled": True, "protocol": protocol["protocol"], "protocol_sha256": protocol_sha, **protocol["noise"],
        "alpha_sampled": _noisy.alpha, "sigma": _noisy.sigma.tolist(), "rng": "numpy default_rng([seed, index, draw])",
        "order": "[torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]",
        "wheel_off_step": _noisy.wheel_off_step, "first_task_reach_step": _noisy.reach_step,
        "noisy_steps": int(np.any(noise != 0, axis=1).sum())}
    meta["execution_interface"]["execution_source"] = "expert + gated OU action noise: a_executed = clip(clip(a_expert) + noise)"
    meta["outcome"]["exec_clip_steps"] = int(arrays["clip_flags"].any(axis=1).sum())
    return meta, out_path, collect.write_episode(meta, arrays, out_path, digest)


EPISODE_COLUMNS = ["episode_id", "scenario_id", "draw", "alpha", "category", "ever_reached", "sustained_hold_pass",
                   "t_first_reach", "final_ee_err_mm", "final1s_max_err_mm", "T_actions", "noisy_steps",
                   "exec_clip_steps", "code_digest", "sha256", "file"]


def write_episodes_csv(root: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "train", "*.npz"))):
        m = collect.read_metadata(path)
        o, n = m["outcome"], m["noise_configuration"]
        rows.append({"episode_id": m["episode_id"], "scenario_id": m["scenario_id"], "draw": m["dart_draw"],
                     "alpha": n["alpha_sampled"], "noisy_steps": n["noisy_steps"], "code_digest": m["code_digest"],
                     "sha256": sha256(path), "file": os.path.relpath(path, root),
                     **{k: o[k] for k in ("category", "ever_reached", "sustained_hold_pass", "t_first_reach",
                                          "final_ee_err_mm", "final1s_max_err_mm", "T_actions", "exec_clip_steps")}})
    with open(os.path.join(root, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke runs)")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    protocol_sha = sha256(args.protocol)
    rows = protocol_rows(protocol)
    info = collect.source_info()
    digest = hashlib.sha256(json.dumps({"collect": info["code_digest"], "protocol": protocol_sha,
                                        "collect_dart_py": sha256(os.path.abspath(__file__))},
                                       sort_keys=True).encode()).hexdigest()
    name = f"{protocol['protocol']}-{digest[:12]}"
    existing = glob.glob(os.path.join(DATA_ROOT, f"*__collection__{name}"))
    root = existing[0] if existing else os.path.join(DATA_ROOT, f"{datetime.date.today()}__collection__{name}")
    os.makedirs(os.path.join(root, "qa"), exist_ok=True)
    write_manifest(rows, os.path.join(root, "scenarios_train.csv"))

    jobs = []
    for row in rows:
        for draw in range(protocol["draws_per_scenario"]):
            path = os.path.join(root, "train", episode_id(row["scenario_id"], draw) + ".npz")
            if os.path.exists(path) and collect.read_metadata(path)["code_digest"] == digest:
                continue
            jobs.append((row, draw, path, digest, protocol, protocol_sha))
    jobs = jobs[:args.limit] if args.limit else jobs
    command = f"python -m data_collection.collect_dart --protocol {args.protocol}" + (f" --limit {args.limit}" if args.limit else "")
    print(f"{len(jobs)} episodes to collect into {os.path.relpath(root, TASK_DIR)}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(protocol,)) as pool:
            for i, (meta, path, file_sha) in enumerate(pool.imap_unordered(collect_job, jobs), 1):
                o = meta["outcome"]
                with open(os.path.join(root, "qa", "attempts.jsonl"), "a") as f:
                    f.write(json.dumps({"event": "written", "episode_id": meta["episode_id"], "sha256": file_sha,
                                        "code_digest": digest, "category": o["category"], "command": command,
                                        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}) + "\n")
                if i % 20 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)

    episodes = write_episodes_csv(root)
    counts = {"scenarios": len(rows), "draws_per_scenario": protocol["draws_per_scenario"], "train_episodes": len(episodes),
              "sustained_hold": sum(e["sustained_hold_pass"] for e in episodes)}
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_id": f"Markovian/DART/{os.path.basename(root)}", "dataset_kind": "dart",
        "teacher_id": collect.TEACHERS["markovian"]["teacher_id"], "teacher_version": collect.TEACHERS["markovian"]["teacher_version"],
        "teacher_parameters": collect.teacher_parameters("markovian"), "protocol": protocol, "protocol_sha256": protocol_sha,
        "code_digest": digest, "source": info, "collection_configuration": collect.CFG, "reach_tolerance_m": collect.TOL,
        "counts": counts,
        "fields": {"a_expert": "label: clean teacher output on o_t, as in Clean",
                   "a_executed": "command sent to the simulator: clip(clip(a_expert) + sampled_action_noise)",
                   "sampled_action_noise": "applied noise after gating [T,11]",
                   "clip_flags": "execution limits changed the noisy command",
                   "other fields": "as in Clean (see data/Markovian/Clean/manifest.json)"},
        "outcome_definitions": "as in Clean; the sustained-hold check runs after the noise has stopped",
        "collection_command": command,
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps({"root": os.path.relpath(root, TASK_DIR), **counts, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
