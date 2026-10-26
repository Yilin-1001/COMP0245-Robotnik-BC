"""Shared pytest setup: import the task package and register the real-data/simulator markers."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASK_DIR = REPO / "final" / "robotnik_arm_task"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "realdata: reads the recorded Clean_253/Clean_223 datasets")
    config.addinivalue_line("markers", "sim: constructs the PyBullet simulator")


# ---------------------------------------------------------------- tiny dataset fixtures

import csv
import hashlib
import json

import numpy as np
import pytest

TEACHER_VERSIONS = {"A": "markovian-lt0.50-v1", "B": "template-origin-goal-v1"}
TEACHER_DIRS = {"A": "Markovian", "B": "Stateful"}
ACTION_ORDER = "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]"


def _episode_arrays(T: int, rng, teacher: str) -> dict:
    a = rng.normal(0.0, 3.0, (T, 11))
    a[:, 10] = a[:, 7]  # BR = FR
    a[:, 9] = a[:, 8]   # BL = FL
    arrays = {"obs_time": np.arange(T + 1) * 0.001, "obs_base_pos": rng.normal(0, 0.3, (T + 1, 3)),
              "obs_base_yaw": rng.uniform(-3, 3, T + 1), "obs_q_mes": rng.normal(0, 1, (T + 1, 7)),
              "obs_qd_mes": rng.normal(0, 0.2, (T + 1, 7)), "truth_ee_err": rng.uniform(0, 0.2, T + 1),
              "a_expert": a, "a_executed": a.copy(), "clip_flags": np.zeros((T, 11), bool),
              "teacher_reach_active": np.arange(T) >= T // 2}
    if teacher == "B":
        arrays["nav_done_after"] = np.arange(T) >= T // 2
    return arrays


def write_fake_dataset(root, teacher="A", train=None, test=None, seed=0, mutate=None):
    """Tiny Clean_* style dataset: {scenario_id: T} per split. mutate(meta, arrays, split, sid) edits one episode."""
    train = {"train-0000": 5, "train-0001": 3} if train is None else train
    test = {"test-0000": 4} if test is None else test
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    rows_manifest, rows_prov = [], []
    for split, episodes in (("train", train), ("test", test)):
        (root / split).mkdir(exist_ok=True)
        with open(root / f"scenarios_{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario_id", "split", "index", "seed", "u_target_dist", "u_rel_azimuth", "u_target_z",
                        "target_x", "target_y", "target_z", "start_x", "start_y", "start_yaw"])
            for i, sid in enumerate(episodes):
                w.writerow([sid, split, i, 1000 + i, 0.5, (i + 0.5) / max(len(episodes), 1), 0.5, 1.0, 0.5, 0.6,
                            0.0, 0.0, 0.1])
        for sid, T in episodes.items():
            arrays = _episode_arrays(T, rng, teacher)
            eid = f"{TEACHER_DIRS[teacher].lower()}-clean-{sid}"
            meta = {"episode_id": eid, "scenario_id": sid, "split": split, "source": "expert_clean",
                    "teacher_version": TEACHER_VERSIONS[teacher], "target_ee_pos_W": [1.0, 0.5, 0.6],
                    "control_hz": 1000, "noise_configuration": {"enabled": False},
                    "execution_interface": {"order": ACTION_ORDER, "limits": {"torque_Nm": 30.0, "wheel_rad_s": 6.0}},
                    "teacher_initial_state": ({"base_goal_pos_W": [0.4, 0.2], "base_goal_bearing": 0.46,
                                               "time_scale_s": 25.0} if teacher == "B" else {}),
                    "outcome": {"T_actions": T, "first_reach_step": T - 2}}
            if mutate:
                mutate(meta, arrays, split, sid)
            path = root / split / f"{eid}.npz"
            np.savez_compressed(path, metadata_json=np.array(json.dumps(meta)), **arrays)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows_manifest.append([eid, sid, split, "success_sustained_hold", "valid", True, "included", False, "",
                                  f"{split}/{eid}.npz", T])
            rows_prov.append([eid, sid, split, f"{split}/{eid}.npz", f"../Clean/{split}/{eid}.npz", digest,
                              path.stat().st_size])
    with open(root / "training_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "scenario_id", "split", "outcome", "integrity_status", "eligible_initial_bc", "reason",
                    "offline_test_subset", "label_quality_flags", "file", "T_actions"])
        w.writerows(rows_manifest)
    with open(root / "episode_provenance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "scenario_id", "split", "file", "source_file", "sha256", "bytes"])
        w.writerows(rows_prov)
    manifest = {"teacher_version": TEACHER_VERSIONS[teacher], "dataset_kind": "paired clean trajectory subset",
                "counts": {"train_episodes": len(train), "test_episodes": len(test)},
                "collection_configuration": {"exec_torque_limit": 30.0, "exec_wheel_limit": 6.0}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


@pytest.fixture
def fake_dataset(tmp_path):
    return lambda teacher="A", **kw: write_fake_dataset(tmp_path / f"{TEACHER_DIRS[teacher]}_fake", teacher, **kw)


# ---------------------------------------------------------------- numerical fixtures

FEATURES_A = ([f"q_mes_{i}" for i in range(7)] + [f"qd_mes_{i}" for i in range(7)] + ["target_bx", "target_by", "target_bz"]
              + ["ee_err_bx", "ee_err_by", "ee_err_bz", "target_z_world"])
LABELS = [f"torque_{i}" for i in range(7)] + ["wheel_right", "wheel_left"]
FLOORS = {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05}


def make_normaliser(x_scale=None, y_scale=None):
    from training.common.normalisation import Normaliser
    return Normaliser(np.zeros(21), np.ones(21) if x_scale is None else x_scale, np.zeros(9),
                      np.ones(9) if y_scale is None else y_scale, FEATURES_A, LABELS, FLOORS)


@pytest.fixture
def unit_normaliser():
    return make_normaliser()


@pytest.fixture
def linear_pd_model():
    """Standardised linear map with torque_i = -20 * qd_i (zero intercept and cross terms)."""
    import torch
    m = torch.nn.Linear(21, 9)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
        m.weight[:7, 7:14] = -20.0 * torch.eye(7)
    return m
