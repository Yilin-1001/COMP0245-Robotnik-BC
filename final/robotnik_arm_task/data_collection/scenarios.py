"""Scenario sampling: target, start pose and physics parameters.

Clean-collection manifests are scrambled Sobol samples in six dimensions, in this order:
  0 target horizontal distance from the initial base   [0.5, 3.0] m (uniform in radius, not in area)
  1 target azimuth relative to the initial base heading [-pi, pi]
  2 target world height                                 [0.1, 1.25] m
  3 initial base x                                      [-0.5, 0.5] m
  4 initial base y                                      [-0.5, 0.5] m
  5 initial base yaw                                    [-pi, pi]
The world target is start + distance * (cos(yaw + azimuth), sin(yaw + azimuth)), at the sampled height.
Training and test sets come from two Sobol engines with different scrambling seeds, so the test set
is not a prefix of the training sequence. Physics is nominal: no added noise, pushes or mass/friction changes.

Usage: python -m data_collection.scenarios   # writes the frozen manifests to <data>/scenarios/
"""
import csv
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = yaml.safe_load(open(os.path.join(TASK_DIR, "settings.yaml")))
DIMENSIONS = ["target_dist", "rel_azimuth", "target_z", "start_x", "start_y", "start_yaw"]
COLUMNS = (["scenario_id", "split", "index", "seed"] + [f"u_{d}" for d in DIMENSIONS] + DIMENSIONS
           + ["target_x", "target_y"])


@dataclass
class Scenario:
    seed: int
    target_ee_pos: np.ndarray
    base_start_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_start_bearing: float = 0.0
    physics: dict = field(default_factory=dict)  # friction / mass scales, noise, push
    scenario_id: str = ""
    split: str = ""


def sobol_scenarios(split: str) -> list[dict]:
    """Rows of one frozen manifest (train or test), in Sobol order."""
    from scipy.stats import qmc
    cfg = SETTINGS["clean_collection"]
    m = cfg["sobol_m"][split]
    u = qmc.Sobol(d=len(DIMENSIONS), scramble=True, seed=cfg["sobol_seed"][split]).random_base2(m=m)
    ranges = [cfg["ranges"][d] for d in DIMENSIONS]
    rows = []
    for i, ui in enumerate(u):
        v = {d: lo + ui[j] * (hi - lo) for j, (d, (lo, hi)) in enumerate(zip(DIMENSIONS, ranges))}
        heading = v["start_yaw"] + v["rel_azimuth"]
        row = {"scenario_id": f"{split}-{i:04d}", "split": split, "index": i, "seed": cfg["env_seed_offset"][split] + i}
        row.update({f"u_{d}": float(ui[j]) for j, d in enumerate(DIMENSIONS)})
        row.update({d: float(v[d]) for d in DIMENSIONS})
        row["target_x"] = float(v["start_x"] + v["target_dist"] * np.cos(heading))
        row["target_y"] = float(v["start_y"] + v["target_dist"] * np.sin(heading))
        rows.append(row)
    return rows


def write_manifest(rows: list[dict], path: str) -> str:
    """Write rows with repr() floats so they read back bit-exactly; returns the file's SHA-256."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: repr(v) if isinstance(v, float) else v for k, v in r.items()})
    return sha256(path)


def read_manifest(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["index"], r["seed"] = int(r["index"]), int(r["seed"])
        for k in COLUMNS[4:]:
            r[k] = float(r[k])
    return rows


def to_scenario(row: dict) -> Scenario:
    return Scenario(seed=row["seed"], target_ee_pos=np.array([row["target_x"], row["target_y"], row["target_z"]]),
                    base_start_pos=np.array([row["start_x"], row["start_y"], 0.0]),
                    base_start_bearing=row["start_yaw"], physics={},
                    scenario_id=row["scenario_id"], split=row["split"])


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def scenario_dir() -> str:
    return os.path.join(TASK_DIR, SETTINGS["paths"]["data"], "scenarios")


def check_manifests(train: list[dict], test: list[dict]) -> dict:
    """Duplicate and train/test overlap checks, in the normalised unit cube."""
    u = lambda rows: np.array([[r[f"u_{d}"] for d in DIMENSIONS] for r in rows])
    params = lambda rows: {tuple(r[d] for d in DIMENSIONS) for r in rows}
    ut, uv = u(train), u(test)
    nn = lambda a, b: np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    d_tt = nn(ut, ut) + np.eye(len(ut)) * 9
    d_vv = nn(uv, uv) + np.eye(len(uv)) * 9
    return {
        "train_count": len(train), "test_count": len(test),
        "train_duplicate_tuples": len(train) - len(params(train)),
        "test_duplicate_tuples": len(test) - len(params(test)),
        "train_test_identical_tuples": len(params(train) & params(test)),
        "train_test_shared_ids": len({r["scenario_id"] for r in train} & {r["scenario_id"] for r in test}),
        "train_test_shared_seeds": len({r["seed"] for r in train} & {r["seed"] for r in test}),
        "min_unit_distance_train_train": float(d_tt.min()),
        "min_unit_distance_test_test": float(d_vv.min()),
        "min_unit_distance_test_to_train": float(nn(uv, ut).min()),
        "median_unit_distance_test_to_nearest_train": float(np.median(nn(uv, ut).min(1))),
    }


def main():
    out = scenario_dir()
    os.makedirs(out, exist_ok=True)
    rows = {s: sobol_scenarios(s) for s in ("train", "test")}
    for s in rows:
        path = os.path.join(out, f"scenarios_{s}.csv")
        if os.path.exists(path):
            existing = read_manifest(path)
            if [r["scenario_id"] for r in existing] != [r["scenario_id"] for r in rows[s]] or any(
                    a[d] != b[d] for a, b in zip(existing, rows[s]) for d in DIMENSIONS):
                raise RuntimeError(f"{path} exists and differs from the regenerated manifest; refusing to overwrite a frozen manifest")
    import scipy
    manifest = {
        "description": "Frozen clean-collection scenario manifests (scrambled Sobol, robot-relative target sampling)",
        "dimensions": DIMENSIONS, "ranges": {d: SETTINGS["clean_collection"]["ranges"][d] for d in DIMENSIONS},
        "distance_distribution": "uniform in radius (not uniform in planar area); near targets are denser per m^2",
        "target_mapping": "target_xy = start_xy + target_dist * [cos(start_yaw + rel_azimuth), sin(start_yaw + rel_azimuth)], target_z = sampled height",
        "sobol": {s: {"engine": "scipy.stats.qmc.Sobol(d=6, scramble=True)", "seed": SETTINGS["clean_collection"]["sobol_seed"][s],
                      "call": f"random_base2(m={SETTINGS['clean_collection']['sobol_m'][s]})"} for s in rows},
        "env_seed": {s: f"{SETTINGS['clean_collection']['env_seed_offset'][s]} + index" for s in rows},
        "fixed_conditions": {"initial_arm_and_fingers": "fixed (configs/robotnik_arm.json, fingers 0.1 rad)",
                             "physics": "nominal masses and friction", "sensor_noise": 0.0, "action_noise": 0.0,
                             "external_push": None, "measurement_delay": None},
        "scipy_version": scipy.__version__,
        "checks": check_manifests(rows["train"], rows["test"]),
        "files": {},
    }
    for s in rows:
        manifest["files"][f"scenarios_{s}.csv"] = write_manifest(rows[s], os.path.join(out, f"scenarios_{s}.csv"))
    with open(os.path.join(out, "scenario_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["checks"], indent=2))
    print(f"Manifests: {out}")


if __name__ == "__main__":
    main()
