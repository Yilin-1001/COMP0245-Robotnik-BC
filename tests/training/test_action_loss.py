import subprocess
import sys

import torch

from conftest import TASK_DIR
from training.baseline.loss import ActionObjective, action_loss
from training.common.engine import LossResult


def test_action_groups_have_equal_weight():
    pred = torch.cat((torch.ones(2, 7), torch.zeros(2, 2)), dim=1)
    assert action_loss(pred, torch.zeros_like(pred), "mse").item() == 0.5


def test_zero_residual_is_zero_loss():
    for kind in ("mse", "huber"):
        assert action_loss(torch.ones(3, 9), torch.ones(3, 9), kind).item() == 0.0


def test_huber_is_quadratic_inside_and_linear_outside_delta():
    pred = torch.zeros(1, 9)
    small, large = pred.clone(), pred.clone()
    small[0, :7], large[0, :7] = 0.5, 3.0
    assert torch.isclose(action_loss(small, pred, "huber"), torch.tensor(0.5 * 0.5 * 0.25))
    assert torch.isclose(action_loss(large, pred, "huber"), torch.tensor(0.5 * (3.0 - 0.5)))


def test_objective_returns_loss_result_with_gradients():
    model = torch.nn.Linear(21, 9)
    result = ActionObjective("huber")(model, torch.randn(8, 21), torch.randn(8, 9), None, torch.Generator())
    assert isinstance(result, LossResult) and set(result.metrics) >= {"action_loss", "arm_loss", "wheel_loss"}
    result.total.backward()
    assert model.weight.grad.abs().sum() > 0


def test_baseline_never_imports_or_calls_jacobian_helpers(monkeypatch):
    calls = []
    import torch.func
    monkeypatch.setattr(torch.func, "jacrev", lambda *a, **k: calls.append("jacrev"))
    monkeypatch.setattr(torch.func, "jacfwd", lambda *a, **k: calls.append("jacfwd"))
    ActionObjective("mse")(torch.nn.Linear(4, 9), torch.randn(2, 4), torch.randn(2, 9), None, torch.Generator())
    assert calls == []
    code = ("import sys, training.baseline.loss; "
            "print([m for m in sys.modules if m.startswith('training.jacobian') or m == 'evaluation.offline'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=TASK_DIR, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
