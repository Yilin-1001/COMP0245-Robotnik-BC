"""The configurable MLP shared by every objective (M1–M6 in docs/mlp_baseline_training_plan.md §5).

[Linear -> LayerNorm -> SiLU] per hidden layer, then a linear 7-torque head and a linear 2-wheel (right, left)
head. Outputs are standardised; expand_wheels maps (right, left) to the [FR, FL, BL, BR] = [r, l, l, r] interface
after inverse scaling.
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], layernorm_eps: float = 1e-5):
        super().__init__()
        self.input_dim, self.hidden, self.layernorm_eps = int(input_dim), [int(h) for h in hidden], float(layernorm_eps)
        layers, d = [], self.input_dim
        for width in self.hidden:
            layers += [nn.Linear(d, width), nn.LayerNorm(width, eps=self.layernorm_eps), nn.SiLU()]
            d = width
        self.trunk = nn.Sequential(*layers)
        self.torque_head = nn.Linear(d, 7)
        self.wheel_head = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat((self.torque_head(h), self.wheel_head(h)), dim=-1)


def expand_wheels(y9):
    """[..., 7 torques, right, left] -> [..., 7 torques, FR=right, FL=left, BL=left, BR=right]."""
    if isinstance(y9, torch.Tensor):
        return torch.cat((y9[..., :7], y9[..., 7:8], y9[..., 8:9], y9[..., 8:9], y9[..., 7:8]), dim=-1)
    import numpy as np
    y9 = np.asarray(y9)
    return np.concatenate((y9[..., :7], y9[..., 7:8], y9[..., 8:9], y9[..., 8:9], y9[..., 7:8]), axis=-1)


def build_model(config: dict) -> MLP:
    m = config["model"]
    return MLP(m["input_dim"], m["architectures"][config["identity"]["architecture"]], m["layernorm_eps"])
