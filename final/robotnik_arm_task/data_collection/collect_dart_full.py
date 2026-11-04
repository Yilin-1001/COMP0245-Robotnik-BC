"""Full DART collection for teacher A (Laskey et al., CoRL 2017): learner-fitted Gaussian action noise.

Unlike collect_dart (our DART-style scheme: fixed OU envelope, phase gated), the noise here is the paper's:
n_t ~ N(0, Sigma), drawn i.i.d. at every 1 kHz control step, for the whole episode, with a full 9 x 9 covariance over
[7 arm torques, right wheel, left wheel] (wheels expand to FR, FL, BL, BR = r, l, l, r). Sigma comes from the learner's
error on held-out demonstrations (experiments/dart_full.py); zero gives clean held-out demonstrations.
Labels stay the clean teacher output a_expert; only the simulator command changes: a_executed = clip(clip(a_expert) + n).

Scenes: 'fitting' = the 221 Clean_253 fitting scenarios (training data); 'heldout' = fresh scenes inside the training
ranges, ids dartho-k<k>-<i>, used only to estimate the next covariance (never trained on, never validation or test).

Output: data/Markovian/DART/<YYYY-MM-DD>__collection__dartfull-<scenes>-k<k>-<digest12>/ (same layout as collect_dart,
so training.datasets.build accepts the fitting collections).

Usage: python -m data_collection.collect_dart_full --covariance <npz|zero> --scenes fitting|heldout --round 1 [--draws 2]
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
from data_collection.collect_dart import DATA_ROOT, LIMITS, SCHEMA_VERSION, write_episodes_csv
from data_collection.collect_dagger import fitting_rows
from data_collection.scenarios import COLUMNS, sha256, write_manifest

HELDOUT = {"seed": 20261104, "z": (0.1, 1.25), "d": (0.5, 3.0), "index_base": 970000}


def heldout_rows(k: int, n: int) -> list[dict]:
    """Fresh held-out scenes for iteration k, uniform in the clean training ranges (distance uniform in radius)."""
    rows = []
    for i in range(n):
        rng = np.random.default_rng([HELDOUT["seed"], k, i])
        d, az, z = rng.uniform(*HELDOUT["d"]), rng.uniform(-np.pi, np.pi), rng.uniform(*HELDOUT["z"])
        sx, sy = rng.uniform(-0.5, 0.5, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        row = {"scenario_id": f"dartho-k{k}-{i:04d}", "split": "train", "index": HELDOUT["index_base"] + 1000 * k + i,
               "seed": HELDOUT["seed"] + 1000 * k + i,
               **{f"u_{c}": float("nan") for c in ("target_dist", "rel_azimuth", "target_z", "start_x", "start_y", "start_yaw")},
               "target_dist": float(d), "rel_azimuth": float(az), "target_z": float(z), "start_x": float(sx),
               "start_y": float(sy), "start_yaw": float(yaw),
               "target_x": float(sx + d * np.cos(yaw + az)), "target_y": float(sy + d * np.sin(yaw + az))}
        rows.append({c: row[c] for c in COLUMNS})
    return rows


def load_covariance(path: str) -> tuple[np.ndarray, str]:
    if path == "zero":
        return np.zeros((9, 9)), "zero"
    with np.load(path) as z:
        sigma = np.asarray(z["sigma"], dtype=float)
    if sigma.shape != (9, 9) or not np.allclose(sigma, sigma.T) or np.linalg.eigvalsh(sigma).min() < -1e-9:
        raise ValueError(f"{path}: covariance must be a symmetric PSD 9 x 9 matrix")
    return sigma, sha256(path)


class GaussianNoisyEnv:
    """RobotEnv proxy for collect.run_episode: adds i.i.d. N(0, Sigma) to every executed action, never gated."""

    def __init__(self, env, sigma: np.ndarray):
        self._env = env
        w, v = np.linalg.eigh(sigma)
        self._L = v * np.sqrt(np.clip(w, 0.0, None))  # L L^T = Sigma, robust to a singular Sigma

    def __getattr__(self, name):
        return getattr(self._env, name)

    def configure(self, rng) -> None:
        self._rng = rng

    def reset(self, scenario):
        self.noise, self.executed = [], []
        return self._env.reset(scenario)

    def step(self, action):
        from core.interfaces import Action
        n9 = self._L @ self._rng.standard_normal(9)
        n = np.r_[n9[:7], n9[7], n9[8], n9[8], n9[7]]  # FR, FL, BL, BR = r, l, l, r
        executed = np.clip(np.r_[action.torque_arm, action.wheel_cmds] + n, -LIMITS, LIMITS)
        self.noise.append(n)
        self.executed.append(executed)
        return self._env.step(Action(executed[:7].copy(), executed[7:].copy()))


_noisy = None


def _init_worker(sigma):
    global _noisy
    collect._init_worker()
    _noisy = GaussianNoisyEnv(collect._env, np.asarray(sigma))
    collect._env = _noisy


def episode_id(scenario_id: str, k: int, draw: int) -> str:
    return f"markovian-dartfull-k{k}-{scenario_id}-d{draw}"


def collect_job(job):
    row, draw, out_path, digest, protocol = job
    seed = [int(protocol["seed"]), int(protocol["round"]), int(row["index"]), int(draw)]
    _noisy.configure(np.random.default_rng(seed))
    meta, arrays, _ = collect.run_episode(("markovian", row, out_path))
    T = len(arrays["a_expert"])
    noise, executed = np.array(_noisy.noise[:T]).reshape(T, 11), np.array(_noisy.executed[:T]).reshape(T, 11)
    clean_exec = np.clip(arrays["a_expert"], -LIMITS, LIMITS)
    arrays["a_executed"], arrays["sampled_action_noise"] = executed, noise
    arrays["clip_flags"] = executed != clean_exec + noise
    meta.update(schema_version=SCHEMA_VERSION, episode_id=episode_id(row["scenario_id"], protocol["round"], draw),
                source="expert_dart", dart_draw=int(draw))
    meta["random_seeds"]["action_noise_seed"] = seed
    meta["noise_configuration"] = {
        "enabled": True, "type": "gaussian_full_iid", "protocol": protocol["protocol"], "round": protocol["round"],
        "covariance": protocol["covariance"], "covariance_sha256": protocol["covariance_sha256"],
        "alpha_sampled": 1.0, "sigma": np.sqrt(np.diag(np.asarray(protocol["sigma"]))).tolist(),
        "rng": "numpy default_rng([seed, round, index, draw])", "order": "[torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]",
        "noisy_steps": int(np.any(noise != 0, axis=1).sum())}
    meta["execution_interface"]["execution_source"] = "expert + i.i.d. Gaussian action noise (full DART): a_executed = clip(clip(a_expert) + noise)"
    meta["outcome"]["exec_clip_steps"] = int(arrays["clip_flags"].any(axis=1).sum())
    return meta, out_path, collect.write_episode(meta, arrays, out_path, digest)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--covariance", required=True, help="npz with 'sigma' (9 x 9), or 'zero'")
    p.add_argument("--scenes", choices=["fitting", "heldout"], required=True)
    p.add_argument("--round", type=int, required=True, help="DART iteration k (held-out scenes of round k are fresh)")
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--heldout-n", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args(argv)

    sigma, cov_sha = load_covariance(args.covariance)
    rows = fitting_rows() if args.scenes == "fitting" else heldout_rows(args.round, args.heldout_n)
    protocol = {"protocol": f"dartfull-{args.scenes}-k{args.round}", "teacher": "A", "round": args.round,
                "scenes": args.scenes, "draws_per_scenario": args.draws, "seed": args.seed,
                "covariance": args.covariance, "covariance_sha256": cov_sha, "sigma": sigma.tolist(),
                "noise": "i.i.d. N(0, Sigma) per 1 kHz step, whole episode, no gating (Laskey et al. 2017)",
                "heldout": HELDOUT if args.scenes == "heldout" else None}
    info = collect.source_info()
    digest = hashlib.sha256(json.dumps({"collect": info["code_digest"], "protocol": protocol,
                                        "collect_dart_full_py": sha256(os.path.abspath(__file__))},
                                       sort_keys=True).encode()).hexdigest()
    name = f"{protocol['protocol']}-{digest[:12]}"
    existing = glob.glob(os.path.join(DATA_ROOT, f"*__collection__{name}"))
    root = existing[0] if existing else os.path.join(DATA_ROOT, f"{datetime.date.today()}__collection__{name}")
    os.makedirs(os.path.join(root, "qa"), exist_ok=True)
    write_manifest(rows, os.path.join(root, "scenarios_train.csv"))
    jobs = []
    for row in rows:
        for draw in range(args.draws):
            path = os.path.join(root, "train", episode_id(row["scenario_id"], args.round, draw) + ".npz")
            if os.path.exists(path) and collect.read_metadata(path)["code_digest"] == digest:
                continue
            jobs.append((row, draw, path, digest, protocol))
    jobs = jobs[:args.limit] if args.limit else jobs
    print(f"{len(jobs)} episodes to collect into {root}", flush=True)
    t0 = time.time()
    if jobs:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(sigma.tolist(),)) as pool:
            for i, (meta, path, file_sha) in enumerate(pool.imap_unordered(collect_job, jobs), 1):
                with open(os.path.join(root, "qa", "attempts.jsonl"), "a") as f:
                    f.write(json.dumps({"event": "written", "episode_id": meta["episode_id"], "sha256": file_sha,
                                        "code_digest": digest, "category": meta["outcome"]["category"],
                                        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}) + "\n")
                if i % 40 == 0 or i == len(jobs):
                    print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    episodes = write_episodes_csv(root)
    counts = {"scenarios": len(rows), "draws_per_scenario": args.draws, "train_episodes": len(episodes),
              "sustained_hold": sum(e["sustained_hold_pass"] for e in episodes)}
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_id": f"Markovian/DART/{os.path.basename(root)}", "dataset_kind": "dart",
        "teacher_id": collect.TEACHERS["markovian"]["teacher_id"], "teacher_version": collect.TEACHERS["markovian"]["teacher_version"],
        "teacher_parameters": collect.teacher_parameters("markovian"), "protocol": protocol, "code_digest": digest,
        "source": info, "collection_configuration": collect.CFG, "reach_tolerance_m": collect.TOL, "counts": counts,
        "fields": {"a_expert": "label: clean teacher output on o_t, as in Clean",
                   "a_executed": "command sent to the simulator: clip(clip(a_expert) + sampled_action_noise)",
                   "sampled_action_noise": "i.i.d. N(0, Sigma) noise [T,11]"},
        "collection_command": "python -m data_collection.collect_dart_full " + " ".join(argv or []),
        "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(json.dumps({"root": os.path.relpath(root, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      **counts, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
