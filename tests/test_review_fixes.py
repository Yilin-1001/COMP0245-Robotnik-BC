"""Regression tests for the final self-review findings (I1-I4)."""
import fcntl
import hashlib
import threading
import time

import numpy as np
import pytest
import torch

from conftest import TASK_DIR, make_normaliser


def test_feature_cache_key_covers_feature_code():
    from training.common.dataset import FeatureBuilder
    digest = hashlib.sha256((TASK_DIR / "core" / "features.py").read_bytes()).hexdigest()
    assert digest in FeatureBuilder("A").builder_id


def test_jacobian_targets_require_qd_columns(unit_normaliser):
    from evaluation.offline import torque_derivative_targets
    n = make_normaliser()
    n.feature_names = list(reversed(n.feature_names))
    with pytest.raises(ValueError, match="qd_mes"):
        torque_derivative_targets(torch.zeros(1, 7), n, 1e-4)


def test_result_index_writes_are_serialised(tmp_path):
    from evaluation.evaluate import RESULT_COLUMNS, append_results
    index = tmp_path / "results.csv"
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(run_id="r", checkpoint_step=1, evaluation_split="screening", evaluation_id="rollout-v1")
    lock = open(str(index) + ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    t = threading.Thread(target=append_results, args=(index, [row]))
    t.start()
    time.sleep(0.3)
    assert t.is_alive() and not index.exists()  # waits for the other writer
    fcntl.flock(lock, fcntl.LOCK_UN)
    t.join(5)
    assert not t.is_alive() and index.exists()


def test_policy_rejects_export_with_other_feature_order(tmp_path):
    from test_checkpoint import meta, normaliser, seeded_model
    from training.common.checkpoint import export_numpy, model_payload, save_checkpoint
    from training.common.policies import MLPPolicy
    m, n = seeded_model(), normaliser()
    n.feature_names = [f"f{i}" for i in range(21)]  # not the deployable A feature order
    md = meta(n=n)
    save_checkpoint(tmp_path / "best.pt", model_payload(m, n, md, step=1))
    export_numpy(tmp_path / "inference.npz", m, n, {**md, "source_checkpoint": str(tmp_path / "best.pt"), "source_step": 1})
    with pytest.raises(ValueError, match="feature"):
        MLPPolicy.load(str(tmp_path / "inference.npz"), backend="numpy", strict_features=True)


def test_rollout_rejects_policy_with_different_interface():
    from core.interfaces import Action, Observation, Policy
    from evaluation.evaluate import rollout

    class P(Policy):
        meta = {"interface": {"torque_limit": 30.0, "wheel_limit": 3.0, "control_hz": 1000}}

        def act(self, obs):
            return Action(np.zeros(7), np.zeros(4))
    cfg = {"evaluation": {"episode_cap_s": 0.01, "min_reach_time_s": 8.0, "post_reach_s": 3.0, "hold_window_s": 1.0,
                          "reach_tolerance_m": 0.01},
           "interface": {"torque_limit": 30.0, "wheel_limit": 6.0, "control_hz": 1000}}
    out = rollout(P(), type("S", (), {"scenario_id": "s", "target_ee_pos": np.zeros(3)})(), cfg, lambda: None)
    assert out["failure"] and "interface" in out["failure"]
