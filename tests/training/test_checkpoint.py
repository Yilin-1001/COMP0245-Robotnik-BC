import numpy as np
import pytest
import torch

from training.common.checkpoint import export_numpy, load_checkpoint, model_payload, save_checkpoint
from training.common.models import MLP
from training.common.normalisation import Normaliser

INTERFACE = {"action_order": "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]", "torque_limit": 30.0,
             "wheel_limit": 6.0, "control_hz": 1000}


def normaliser(dim=21, seed=0):
    rng = np.random.default_rng(seed)
    from core.features import FEATURE_NAMES_A, FEATURE_NAMES_B
    names = FEATURE_NAMES_A if dim == 21 else FEATURE_NAMES_B
    return Normaliser(rng.normal(size=dim), rng.uniform(0.5, 2, dim), rng.normal(size=9), rng.uniform(1, 5, 9),
                      names, [f"torque_{i}" for i in range(7)] + ["wheel_right", "wheel_left"],
                      {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05})


def seeded_model(dim=21, seed=0):
    torch.manual_seed(seed)
    return MLP(dim, [128, 128])


def meta(teacher="A", dim=21, n=None):
    ctx = {"goal_rule": "template-origin-standoff", "time_scale_s": 25.0} if teacher == "B" else None
    return {"teacher": teacher, "input_dim": dim, "feature_names": (n or normaliser(dim)).feature_names,
            "interface": dict(INTERFACE), "task_context": ctx, "run_id": "test-run", "objective": "bc"}


def test_round_trip_preserves_weights_and_meta(tmp_path):
    m, n = seeded_model(), normaliser()
    save_checkpoint(tmp_path / "best.pt", model_payload(m, n, meta(n=n), step=4))
    loaded = load_checkpoint(tmp_path / "best.pt", expected={"teacher": "A", "input_dim": 21})
    m2 = MLP(21, [128, 128])
    m2.load_state_dict(loaded["model_state"])
    x = torch.randn(5, 21)
    assert torch.equal(m(x), m2(x)) and loaded["step"] == 4
    assert Normaliser.from_dict(loaded["normaliser"]).sha256() == n.sha256()


@pytest.mark.parametrize("field,value", [("teacher", "B"), ("input_dim", 26), ("normaliser_sha256", "0" * 64),
                                         ("feature_names", ["other"] * 21),
                                         ("interface", {**INTERFACE, "wheel_limit": 3.0})])
def test_mismatched_expectations_are_rejected(tmp_path, field, value):
    m, n = seeded_model(), normaliser()
    save_checkpoint(tmp_path / "best.pt", model_payload(m, n, meta(n=n), step=1))
    with pytest.raises(ValueError, match=field):
        load_checkpoint(tmp_path / "best.pt", expected={field: value})


def test_save_is_atomic_and_records_hash(tmp_path):
    m, n = seeded_model(), normaliser()
    save_checkpoint(tmp_path / "step_2.pt", model_payload(m, n, meta(n=n), step=2))
    assert not list(tmp_path.glob("*.tmp"))
    info = export_numpy(tmp_path / "step_2.npz", m, n, {**meta(n=n), "source_checkpoint": str(tmp_path / "step_2.pt"),
                                                         "source_step": 2})
    with np.load(tmp_path / "step_2.npz") as z:
        assert "layer0_W" in z.files and "ln0_gamma" in z.files and "torque_W" in z.files and "wheel_b" in z.files
    assert info["source_sha256"] and info["source_step"] == 2
