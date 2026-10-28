"""Feature definitions: registry, reach gate versus teacher A, builder IDs and gate-aware deployment."""
import numpy as np
import pytest
import torch

from core.feature_sets import GATE_RULE, extend, feature_names, input_dim, reach_gate
from core.features import FEATURE_NAMES_A, STANDOFF_POINTS, STANDOFF_Z_POINTS
from core.interfaces import Observation


def test_registry_names_and_dimensions():
    assert input_dim("A") == input_dim("A", "features-v1") == 21
    assert input_dim("B", "features-v1") == 26
    assert feature_names("A", "features-v1-gate") == FEATURE_NAMES_A + ["reach_gate"]
    with pytest.raises(ValueError):
        feature_names("B", "features-v1-gate")
    with pytest.raises(ValueError):
        feature_names("A", "features-v9")


def test_gate_thresholds_are_the_teachers():
    from data_collection.expert import STANDOFF_POINTS as EXPERT_STANDOFF, STANDOFF_Z_POINTS as EXPERT_Z
    from data_collection.expert import MarkovianExpertParams
    p = MarkovianExpertParams()
    assert GATE_RULE["dist_tol_m"] == pytest.approx(p.reach_tol_scale * p.base.pos_tol)
    assert GATE_RULE["heading_tol_rad"] == pytest.approx(p.reach_tol_scale * p.base.ang_tol)
    assert GATE_RULE["standoff_points"] == EXPERT_STANDOFF == STANDOFF_POINTS
    assert GATE_RULE["standoff_z_points"] == EXPERT_Z == STANDOFF_Z_POINTS


def test_gate_matches_teacher_reach_flag_near_the_boundary():
    from core.features import DeployableFK, features_a
    from data_collection.expert import MarkovianExpert, MarkovianExpertParams
    expert, fk = MarkovianExpert(MarkovianExpertParams(lead_time=0.5)), DeployableFK()
    rng = np.random.default_rng(0)
    flags, gates = [], []
    for _ in range(400):
        target = np.array([*rng.uniform(-2, 2, 2), rng.uniform(0.1, 1.25)])
        standoff = np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS)
        d, h, yaw = standoff + rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15), rng.uniform(-np.pi, np.pi)
        base = np.r_[target[:2] - d * np.array([np.cos(yaw + h), np.sin(yaw + h)]), 0.0]
        obs = Observation(base, yaw, rng.normal(0, 0.5, 7), rng.normal(0, 0.2, 7), target)
        flags.append(expert.base_command(obs)[1])
        gates.append(reach_gate(features_a(base[None], np.array([yaw]), obs.q_mes[None], obs.qd_mes[None], target, fk))[0])
    assert 0.2 < np.mean(flags) < 0.8
    np.testing.assert_array_equal(np.array(gates).astype(bool), np.array(flags))


def test_extend_keeps_v1_and_appends_the_gate():
    x = np.random.default_rng(1).normal(size=(5, 21))
    assert extend("A", "features-v1", x) is x
    g = extend("A", "features-v1-gate", x)
    assert g.shape == (5, 22) and np.array_equal(g[:, :21], x) and np.array_equal(g[:, 21], reach_gate(x))


def test_v1_builder_id_is_unchanged_and_gate_differs():
    from training.common.dataset import FeatureBuilder
    v1, explicit, gate = FeatureBuilder("A"), FeatureBuilder("A", None, "features-v1"), FeatureBuilder("A", None, "features-v1-gate")
    assert v1.builder_id == explicit.builder_id and "definition" not in v1.builder_id
    assert gate.builder_id != v1.builder_id and gate.feature_names[-1] == "reach_gate"


INTERFACE = {"action_order": "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]", "torque_limit": 30.0,
             "wheel_limit": 6.0, "control_hz": 1000}


def _export(tmp_path, definition):
    from training.common.checkpoint import export_numpy, model_payload, save_checkpoint
    from training.common.models import MLP
    from training.common.normalisation import Normaliser
    names = feature_names("A", definition)
    rng = np.random.default_rng(0)
    n = Normaliser(rng.normal(size=22), np.ones(22), np.zeros(9), np.ones(9), names,
                   [f"torque_{i}" for i in range(7)] + ["wheel_right", "wheel_left"],
                   {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05})
    torch.manual_seed(0)
    m = MLP(22, [32])
    md = {"teacher": "A", "input_dim": 22, "feature_names": names, "interface": dict(INTERFACE), "task_context": None,
          "run_id": "test-run", "objective": "bc", "feature_definition": definition}
    save_checkpoint(tmp_path / "best.pt", model_payload(m, n, md, step=1))
    export_numpy(tmp_path / "inference.npz", m, n, {**md, "source_checkpoint": str(tmp_path / "best.pt"), "source_step": 1})
    return tmp_path / "inference.npz"


def test_gate_policy_loads_and_appends_the_gate(tmp_path):
    from training.common.policies import MLPPolicy
    policy = MLPPolicy.load(str(_export(tmp_path, "features-v1-gate")))
    obs = Observation(np.array([0.3, -0.2, 0.0]), 0.4, np.zeros(7), np.zeros(7), np.array([1.2, 0.3, 0.6]))
    x = policy.features(obs)
    assert x.shape == (22,) and x[21] == reach_gate(x[:21])
    assert np.isfinite(policy.act(obs).torque_arm).all()


def test_policy_refuses_names_that_disagree_with_the_definition(tmp_path):
    from training.common.policies import MLPPolicy
    path = _export(tmp_path, "features-v1-gate")
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    import json
    meta = json.loads(str(arrays["metadata_json"]))
    meta.pop("feature_definition")  # an unlabelled 22-input export would be read as features-v1
    arrays["metadata_json"] = np.array(json.dumps(meta))
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        MLPPolicy.load(str(path))
