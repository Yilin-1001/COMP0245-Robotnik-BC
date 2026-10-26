"""Frozen scenario sets for closed-loop evaluation, read from the versioned split manifest.

screening:  the 8 frozen screening scenarios (one per relative-azimuth sector of the validation set)
validation: the 32 internal validation scenarios (whole episodes held out from fitting)
test:       the 64 original test scenarios; sealed unless an explicit frozen final protocol allows access

The archived T1–T3 generators of the earlier plan are not used by the MLP selection workflow.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

from data_collection.scenarios import Scenario

TASK_DIR = Path(__file__).resolve().parents[1]
TEACHER_DIRS = {"A": "Markovian", "B": "Stateful"}


def scenario_from_row(row: dict, split: str) -> Scenario:
    f = lambda k: float(row[k])
    return Scenario(seed=int(row["seed"]), target_ee_pos=np.array([f("target_x"), f("target_y"), f("target_z")]),
                    base_start_pos=np.array([f("start_x"), f("start_y"), 0.0]), base_start_bearing=f("start_yaw"),
                    physics={}, scenario_id=row["scenario_id"], split=split)


def get_split(name: str, split_manifest: Path, teacher: str, allow_test: bool = False) -> list[Scenario]:
    from training.common.dataset import load_split
    split = load_split(split_manifest)
    if name in ("screening", "validation"):
        ids = split["screening_ids"] if name == "screening" else split["validation_ids"]
        rows = {r["scenario_id"]: r for r in split["validation_scenarios"]}
        return [scenario_from_row(rows[i], "train") for i in ids]
    if name != "test":
        raise ValueError(f"unknown evaluation split {name!r}")
    if not allow_test:
        raise PermissionError("the 64 test scenarios are sealed: evaluate them only through a frozen final protocol")
    path = TASK_DIR / "data" / TEACHER_DIRS[teacher] / "Clean_253" / "scenarios_test.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != split["source_sha256"][f"{teacher}/clean253/scenarios_test.csv"]:
        raise ValueError(f"{path}: test scenario manifest changed since the split was frozen")
    with open(path, newline="") as f:
        return [scenario_from_row(r, "test") for r in csv.DictReader(f)]
