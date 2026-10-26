"""Action-only objective: 0.5 * mean arm loss + 0.5 * mean wheel loss, in standardised units.

Separate group means give the 7 torques and the 2 independent wheel speeds equal weight. Huber uses delta = 1
standardised unit. No derivative terms are computed here.
"""
import torch
import torch.nn.functional as F

from training.common.engine import LossResult

KINDS = ("huber", "mse")


def _elementwise(pred: torch.Tensor, target: torch.Tensor, kind: str, delta: float) -> torch.Tensor:
    if kind == "mse":
        return (pred - target) ** 2
    if kind == "huber":
        return F.huber_loss(pred, target, reduction="none", delta=delta)
    raise ValueError(f"action loss must be one of {KINDS}, got {kind!r}")


def action_loss_parts(pred: torch.Tensor, target: torch.Tensor, kind: str, delta: float = 1.0):
    e = _elementwise(pred, target, kind, delta)
    arm, wheel = e[..., :7].mean(), e[..., 7:9].mean()
    return 0.5 * arm + 0.5 * wheel, arm, wheel


def action_loss(pred: torch.Tensor, target: torch.Tensor, kind: str, delta: float = 1.0) -> torch.Tensor:
    return action_loss_parts(pred, target, kind, delta)[0]


class ActionObjective:
    def __init__(self, kind: str, delta: float = 1.0):
        if kind not in KINDS:
            raise ValueError(f"action loss must be one of {KINDS}, got {kind!r}")
        self.kind, self.delta = kind, float(delta)

    def __call__(self, model, x, y, normaliser, generator) -> LossResult:
        total, arm, wheel = action_loss_parts(model(x), y, self.kind, self.delta)
        return LossResult(total, {"action_loss": total.item(), "arm_loss": arm.item(), "wheel_loss": wheel.item()})
