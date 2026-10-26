import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import TASK_DIR
from core.interfaces import Observation
from test_checkpoint import meta, normaliser, seeded_model
from training.common.checkpoint import export_numpy, model_payload, save_checkpoint
from training.common.policies import MLPPolicy


def obs(seed=0):
    rng = np.random.default_rng(seed)
    return Observation(base_pos=rng.normal(0, 0.3, 3), base_bearing=float(rng.uniform(-3, 3)),
                       q_mes=rng.normal(0, 0.5, 7), qd_mes=rng.normal(0, 0.2, 7), target_ee_pos=np.array([1.5, 0.4, 0.7]))


def artifacts(tmp_path, teacher="A", scale=1.0):
    dim = 21 if teacher == "A" else 26
    m, n = seeded_model(dim), normaliser(dim)
    if scale != 1.0:
        with torch.no_grad():
            m.torque_head.weight.mul_(scale)
    md = meta(teacher, dim, n)
    save_checkpoint(tmp_path / "best.pt", model_payload(m, n, md, step=3))
    export_numpy(tmp_path / "inference.npz", m, n, {**md, "source_checkpoint": str(tmp_path / "best.pt"), "source_step": 3})
    return m, n


def test_numpy_backend_refuses_torch_format(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"not-a-numpy-export")
    with pytest.raises(ValueError):
        MLPPolicy.load(str(path), backend="numpy")


def test_torch_backend_refuses_numpy_export(tmp_path):
    artifacts(tmp_path)
    with pytest.raises(ValueError):
        MLPPolicy.load(str(tmp_path / "inference.npz"), backend="torch")


def test_numpy_and_torch_agree_and_actions_are_bounded(tmp_path):
    artifacts(tmp_path, scale=50.0)  # large torques so execution clipping is exercised
    np_policy = MLPPolicy.load(str(tmp_path / "inference.npz"), backend="numpy")
    pt_policy = MLPPolicy.load(str(tmp_path / "best.pt"), backend="torch")
    for seed in range(5):
        o = obs(seed)
        np_policy.reset(); pt_policy.reset()
        a_np, a_pt = np_policy.act(o), pt_policy.act(o)
        np.testing.assert_allclose(np_policy.last_raw, pt_policy.last_raw, rtol=1e-4, atol=1e-5)
        for a in (a_np, a_pt):
            assert a.torque_arm.shape == (7,) and a.wheel_cmds.shape == (4,)
            assert np.isfinite(a.torque_arm).all() and np.abs(a.torque_arm).max() <= 30.0
            assert np.abs(a.wheel_cmds).max() <= 6.0
            assert a.wheel_cmds[0] == a.wheel_cmds[3] and a.wheel_cmds[1] == a.wheel_cmds[2]
    assert np.abs(np_policy.last_raw[:7]).max() > 30.0


def test_stateful_context_is_causal_and_reset(tmp_path):
    artifacts(tmp_path, "B")
    p = MLPPolicy.load(str(tmp_path / "inference.npz"), backend="numpy")
    for _ in range(2):
        p.reset()
        p.act(obs(1))
        assert p.last_features[-1] == 0.0
        p.act(obs(2))
        assert np.isclose(p.last_features[-1], 0.001 / 25.0)


def test_stateful_goal_rule_matches_template_geometry():
    from core.features import stateful_goal
    goal, bearing = stateful_goal(np.array([3.0, 4.0, 0.5]), {"goal_rule": "template-origin-standoff"})
    assert np.isclose(bearing, np.arctan2(4.0, 3.0))
    np.testing.assert_allclose(goal, np.array([3.0, 4.0]) - 0.65 * np.array([0.6, 0.8]))
    with pytest.raises(ValueError):
        stateful_goal(np.zeros(3), {"goal_rule": "start-relative"})


@pytest.mark.realdata
def test_stateful_goal_matches_recorded_metadata():
    import json
    from core.features import stateful_goal
    paths = sorted((TASK_DIR / "data/Stateful/Clean_253/train").glob("*.npz"))[:20]
    for path in paths:
        with np.load(path) as z:
            m = json.loads(str(z["metadata_json"]))
        goal, bearing = stateful_goal(np.array(m["target_ee_pos_W"]), {"goal_rule": "template-origin-standoff"})
        np.testing.assert_allclose(goal, m["teacher_initial_state"]["base_goal_pos_W"], atol=1e-12)
        assert np.isclose(bearing, m["teacher_initial_state"]["base_goal_bearing"], atol=1e-12)


def test_numpy_inference_imports_no_torch_or_simulator(tmp_path):
    artifacts(tmp_path)
    code = ("import sys; from training.common.policies import MLPPolicy; from core.interfaces import Observation; "
            "import numpy as np; p = MLPPolicy.load(sys.argv[1], backend='numpy'); p.reset(); "
            "p.act(Observation(np.zeros(3), 0.0, np.zeros(7), np.zeros(7), np.array([1.0, 0.0, 0.5]))); "
            "bad = [m for m in sys.modules if m == 'torch' or m.startswith('training.jacobian') or m in "
            "('pybullet', 'core.env', 'data_collection.expert', 'data_collection.template_teacher')]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "inference.npz")], cwd=TASK_DIR,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
