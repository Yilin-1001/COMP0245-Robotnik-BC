"""Action-only objective: weighted group means in standardised units.

New run configs choose the weights explicitly; direct calls retain historical 0.5/0.5 defaults.
Huber uses delta = 1 standardised unit. No derivative terms are computed here.
"""
import torch
import torch.nn.functional as F

from training.common.engine import LossResult
from training.common.config import action_weights

KINDS = ("huber", "mse")


def _elementwise(pred: torch.Tensor, target: torch.Tensor, kind: str, delta: float) -> torch.Tensor:
    if kind == "mse":
        return (pred - target) ** 2
    if kind == "huber":
        return F.huber_loss(pred, target, reduction="none", delta=delta)
    raise ValueError(f"action loss must be one of {KINDS}, got {kind!r}")


def action_loss_parts(pred: torch.Tensor, target: torch.Tensor, kind: str, delta: float = 1.0,
                      *, arm_weight: float = 0.5, wheel_weight: float = 0.5):
    e = _elementwise(pred, target, kind, delta)
    arm, wheel = e[..., :7].mean(), e[..., 7:9].mean()
    return arm_weight * arm + wheel_weight * wheel, arm, wheel


def action_loss(pred: torch.Tensor, target: torch.Tensor, kind: str, delta: float = 1.0,
                *, arm_weight: float = 0.5, wheel_weight: float = 0.5) -> torch.Tensor:
    return action_loss_parts(pred, target, kind, delta, arm_weight=arm_weight, wheel_weight=wheel_weight)[0]


class ActionObjective:
    def __init__(self, kind: str, delta: float = 1.0, *, arm_weight: float = 0.5, wheel_weight: float = 0.5):
        if kind not in KINDS:
            raise ValueError(f"action loss must be one of {KINDS}, got {kind!r}")
        self.kind, self.delta = kind, float(delta)
        self.arm_weight, self.wheel_weight = action_weights({"arm_weight": arm_weight, "wheel_weight": wheel_weight})

    def __call__(self, model, x, y, normaliser, generator) -> LossResult:
        total, arm, wheel = action_loss_parts(model(x), y, self.kind, self.delta,
                                             arm_weight=self.arm_weight, wheel_weight=self.wheel_weight)
        return LossResult(total, {"action_loss": total.item(), "arm_loss": arm.item(), "wheel_loss": wheel.item()})
