"""Objective contract shared by every training objective; the update loop uses only this interface.

An Objective is called as objective(model, x, y, normaliser, generator) with standardised batches; generator is
the independent Jacobian sub-batch stream, which action-only objectives ignore.
"""
from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class LossResult:
    total: torch.Tensor
    metrics: dict = field(default_factory=dict)


class Objective(Protocol):
    def __call__(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, normaliser,
                 generator: torch.Generator) -> LossResult: ...
