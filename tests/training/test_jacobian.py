import numpy as np
import pytest
import torch

from conftest import make_normaliser
from training.baseline.loss import ActionObjective, action_loss
from training.jacobian.loss import JacobianObjective, calibrate_lambda

GEN = lambda s=1: torch.Generator().manual_seed(s)


def pd_model(slope=-20.0, cross=0.0):
    m = torch.nn.Linear(21, 9)
    with torch.no_grad():
        m.weight.zero_()
        m.bias.zero_()
        m.weight[:7, 7:14] = slope * torch.eye(7) + cross * (1 - torch.eye(7))
    return m


def test_derivative_loss_reaches_parameters(linear_pd_model, unit_normaliser):
    objective = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4)
    result = objective(linear_pd_model, torch.zeros(2, 21), torch.zeros(2, 9),
                       unit_normaliser, torch.Generator().manual_seed(1))
    assert result.total.item() > 0
    result.total.backward()
    assert linear_pd_model.weight.grad[:, 7:14].abs().sum().item() > 0


def test_zero_at_teacher_slope_positive_otherwise(unit_normaliser):
    obj = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4)
    x, y = torch.randn(8, 21), torch.zeros(8, 9)
    exact = obj(pd_model(-30.0), x * 0, y, unit_normaliser, GEN())
    assert exact.metrics["jacobian_loss"] == 0.0
    wrong = obj(pd_model(-20.0), x * 0, y, unit_normaliser, GEN())
    assert np.isclose(wrong.metrics["jacobian_loss"], 100.0 * 7 / 49)
    cross = obj(pd_model(-30.0, cross=1.0), x * 0, y, unit_normaliser, GEN())
    assert np.isclose(cross.metrics["jacobian_loss"], 42 / 49)


def test_gradient_step_moves_slope_toward_teacher(unit_normaliser):
    m = pd_model(-20.0)
    obj = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    obj(m, torch.zeros(4, 21), torch.zeros(4, 9), unit_normaliser, GEN()).total.backward()
    opt.step()
    diag = torch.diagonal(m.weight[:7, 7:14])
    assert torch.all(diag < -20.0) and torch.all(diag > -30.0)


def test_standardised_scales_define_the_target(unit_normaliser):
    x_scale, y_scale = np.ones(21), np.ones(9)
    x_scale[7:14], y_scale[:7] = 0.2, 4.0
    n = make_normaliser(x_scale, y_scale)
    m = pd_model(-30.0 * 0.2 / 4.0)  # the teacher's slope in standardised units
    obj = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4)
    assert obj(m, torch.zeros(3, 21), torch.zeros(3, 9), n, GEN()).metrics["jacobian_loss"] < 1e-12


def test_near_limit_rows_are_masked_conservatively(unit_normaliser):
    obj = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4)
    y = torch.zeros(4, 9)
    y[:, :7] = 30.0  # recorded torque at the limit: saturation cannot be verified without pre-clip torque
    m = pd_model(-20.0)
    r = obj(m, torch.zeros(4, 21), y, unit_normaliser, GEN())
    assert r.metrics["valid_entry_frac"] == 0.0 and r.metrics["jacobian_loss"] is None
    assert torch.isfinite(r.total) and torch.isclose(r.total, action_loss(m(torch.zeros(4, 21)), y, "mse"))
    y[0, 3] = 2.0  # one interior joint row is supervised
    r = obj(m, torch.zeros(4, 21), y, unit_normaliser, GEN())
    assert np.isclose(r.metrics["valid_entry_frac"], 7 / (4 * 49)) and np.isclose(r.metrics["jacobian_loss"], 100 / 7)


def test_zero_lambda_is_exactly_the_baseline_and_skips_derivatives(unit_normaliser, monkeypatch):
    import training.jacobian.loss as jl
    monkeypatch.setattr(jl, "velocity_jacobian", lambda *a, **k: pytest.fail("derivative computed at lambda 0"))
    m, x, y = pd_model(-20.0), torch.randn(16, 21), torch.randn(16, 9)
    a = ActionObjective("huber")(m, x, y, unit_normaliser, GEN())
    b = jl.JacobianObjective("huber", lambda_j=0.0, epsilon_tau=1e-4)(m, x, y, unit_normaliser, GEN())
    assert torch.equal(a.total, b.total)


def test_subbatch_uses_only_its_own_generator(unit_normaliser):
    obj = JacobianObjective("mse", lambda_j=1.0, epsilon_tau=1e-4, subbatch_size=4)
    g, m, x = GEN(5), pd_model(), torch.randn(32, 21)
    before = torch.get_rng_state()
    obj(m, x, torch.zeros(32, 9), unit_normaliser, g)
    assert torch.equal(before, torch.get_rng_state())
    assert not torch.equal(g.get_state(), GEN(5).get_state())


def test_calibration_sets_ten_percent_and_preserves_init(unit_normaliser):
    m = pd_model(-20.0)
    with torch.no_grad():
        m.bias[:7] = 1.0  # nonzero action loss
    before = {k: v.clone() for k, v in m.state_dict().items()}
    x, y = torch.zeros(64, 21), torch.zeros(64, 9)
    record = calibrate_lambda(m, x, y, unit_normaliser, {"kind": "mse", "huber_delta": 1.0, "epsilon_tau": 1e-4,
                                                          "kd": 30.0, "torque_limit": 30.0, "calibration_ratio": 0.1})
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    a, j = record["action_loss"], record["jacobian_loss"]
    assert np.isclose(record["lambda_j"] * j, 0.1 * a) and record["batch_sha256"] and record["init_sha256"]
    with pytest.raises(ValueError, match="denominator"):
        calibrate_lambda(pd_model(-30.0), x, y, unit_normaliser, {"kind": "mse", "huber_delta": 1.0, "epsilon_tau": 1e-4,
                                                                   "kd": 30.0, "torque_limit": 30.0, "calibration_ratio": 0.1})


def test_mlp_velocity_jacobian_matches_finite_differences():
    from evaluation.offline import velocity_jacobian
    from training.common.models import MLP
    torch.manual_seed(0)
    m = MLP(21, [64, 64]).double()
    x = torch.randn(3, 21, dtype=torch.float64)
    J = velocity_jacobian(m, x)
    h = 1e-6
    for j in range(7):
        e = torch.zeros(21, dtype=torch.float64)
        e[7 + j] = h
        fd = (m(x + e)[:, :7] - m(x - e)[:, :7]) / (2 * h)
        torch.testing.assert_close(J[:, :, j], fd, rtol=1e-6, atol=1e-8)
