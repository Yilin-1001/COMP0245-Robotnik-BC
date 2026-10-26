import numpy as np
import torch

from conftest import make_normaliser
from evaluation.offline import offline_metrics, torque_derivative_targets, velocity_jacobian
from training.common.dataset import EpisodeCache

CONFIG = {"loss": {"epsilon_tau": 1e-4}, "evaluation": {"hold_window_s": 1.0, "hold_bias_gate_nm": 0.5,
                                                        "jacobian_samples": 64},
          "interface": {"torque_limit": 30.0, "control_hz": 1000}}


def test_execution_clipping_is_not_internal_saturation(unit_normaliser):
    torque = torch.tensor([[0.0] * 7, [30.0] * 7])
    target, valid = torque_derivative_targets(torque, unit_normaliser, 1e-4)
    assert valid.shape == (2, 7)
    assert valid[0].all() and not valid[1].any()
    torch.testing.assert_close(target[0], -30 * torch.eye(7))


def test_near_limit_rows_are_omitted_without_preclip_metadata(unit_normaliser):
    torque = torch.tensor([[29.99995, -29.99995, 29.9998, 0.0, 0.0, 0.0, 0.0]])
    _, valid = torque_derivative_targets(torque, unit_normaliser, 1e-4)
    assert valid[0].tolist() == [False, False, True, True, True, True, True]


def test_verified_saturation_targets_zero_and_boundary_is_omitted(unit_normaliser):
    clipped = torch.tensor([[30.0, -30.0, 30.0, 5.0, 0.0, 0.0, 0.0]])
    preclip = torch.tensor([[35.0, -31.0, 30.00005, 5.0, 0.0, 0.0, 0.0]])
    target, valid = torque_derivative_targets(clipped, unit_normaliser, 1e-4, preclip_torque=preclip)
    assert valid[0].tolist() == [True, True, False, True, True, True, True]
    assert torch.all(target[0, 0] == 0) and torch.all(target[0, 1] == 0)
    assert target[0, 3, 3] == -30.0


def test_standardised_targets_use_sigma_qd_over_sigma_tau(unit_normaliser):
    x_scale = np.ones(21)
    x_scale[7:14] = np.arange(1, 8) * 0.1
    y_scale = np.ones(9)
    y_scale[:7] = np.arange(1, 8) * 2.0
    n = make_normaliser(x_scale, y_scale)
    target, valid = torque_derivative_targets(torch.zeros(1, 7), n, 1e-4)
    expected = torch.diag(torch.tensor(-30.0 * x_scale[7:14] / y_scale[:7], dtype=torch.float32))
    torch.testing.assert_close(target[0], expected)
    assert torch.count_nonzero(target[0] - torch.diag(torch.diagonal(target[0]))) == 0


def test_velocity_jacobian_matches_analytical_pd_map(linear_pd_model):
    x = torch.randn(4, 21, requires_grad=True)
    J = velocity_jacobian(linear_pd_model, x)
    assert J.shape == (4, 7, 7)
    torch.testing.assert_close(J, -20.0 * torch.eye(7).expand(4, 7, 7))
    J.sum().backward()  # differentiable for training use
    assert linear_pd_model.weight.grad[:7, 7:14].abs().sum() > 0


class Echo(torch.nn.Module):
    """Standardised prediction = first nine input columns (lets a test choose every prediction)."""
    def forward(self, x):
        return x[:, :9] * 1.0


def synthetic_cache(bias_per_episode, steps=1500):
    xs, ys, phases = [], [], []
    for bias in bias_per_episode:
        y = np.zeros((steps, 9), np.float32)
        y[:, :7] = 2.0
        x = np.zeros((steps, 21), np.float32)
        x[:, :9] = y
        x[:, :7] += bias  # constant torque bias of this episode
        x[:500, 7:9] += 0.3  # wheel error during navigation only
        phase = np.r_[np.zeros(500), np.ones(400), np.full(600, 2)].astype(np.int8)
        xs.append(x); ys.append(y); phases.append(phase)
    offsets = np.arange(len(bias_per_episode) + 1) * steps
    t = np.tile(np.arange(steps) * 0.001, len(bias_per_episode)).astype(np.float32)
    return EpisodeCache(np.concatenate(xs), np.concatenate(ys), offsets, [f"e{i}" for i in range(len(xs))],
                        {"phase": np.concatenate(phases), "t": t}, {})


def test_opposite_episode_biases_cannot_cancel(unit_normaliser):
    m = offline_metrics(Echo(), synthetic_cache([0.8, -0.8]), unit_normaliser, CONFIG)
    assert m["hold_bias"]["episodes"] == {"e0": [0.8] * 7, "e1": [-0.8] * 7} or \
        np.allclose(m["hold_bias"]["episodes"]["e0"], 0.8) and np.allclose(m["hold_bias"]["episodes"]["e1"], -0.8)
    assert np.isclose(m["hold_bias"]["max_abs_nm"], 0.8) and not m["hold_bias"]["gate_pass"]
    ok = offline_metrics(Echo(), synthetic_cache([0.2, -0.3]), unit_normaliser, CONFIG)
    assert ok["hold_bias"]["gate_pass"] and np.isclose(ok["hold_bias"]["max_abs_nm"], 0.3)


def test_phase_split_physical_errors(unit_normaliser):
    m = offline_metrics(Echo(), synthetic_cache([0.5]), unit_normaliser, CONFIG)
    assert np.isclose(m["arm_rmse_nm"], 0.5) and np.isclose(m["phase"]["hold"]["arm_rmse_nm"], 0.5)
    assert np.isclose(m["phase"]["navigation"]["wheel_rmse_rad_s"], 0.3)
    assert np.isclose(m["phase"]["approach"]["wheel_rmse_rad_s"], 0.0)
    assert m["phase"]["navigation"]["rows"] == 500 and m["phase"]["hold"]["rows"] == 600


def test_all_invalid_damping_rows_give_no_nan(unit_normaliser):
    cache = synthetic_cache([0.0])
    cache.y[:, :7] = 30.0  # every hold row at the torque limit -> no valid derivative rows
    cache.x[:, :7] = 30.0
    m = offline_metrics(Echo(), cache, unit_normaliser, CONFIG)
    d = m["damping"]
    assert d["valid_rows"] == 0 and d["omitted_rows"] == d["sampled_rows"] > 0
    assert d["median_diagonal"] is None and d["frac_below_minus33"] is None
    assert not any(isinstance(v, float) and np.isnan(v) for v in d.values())


def test_damping_diagnostic_reports_teacher_relative_slope(linear_pd_model, unit_normaliser):
    cache = synthetic_cache([0.0])
    m = offline_metrics(linear_pd_model, cache, unit_normaliser, CONFIG)
    d = m["damping"]
    assert np.isclose(d["median_diagonal"], -20.0) and d["frac_below_minus33"] == 0.0
    assert np.isclose(d["cross_joint_rms"], 0.0) and d["valid_rows"] == d["sampled_rows"] == 64
