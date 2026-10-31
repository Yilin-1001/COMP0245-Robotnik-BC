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


class GatedMLP(nn.Module):
    """Two MLP branches of the same shape, switched per row by the reach_gate input (features-v1-gate).

    The gate column is standardised with the fitting mean m (0 < m < 1 whenever both gate values occur), so a raw 0
    maps to a negative and a raw 1 to a positive value: the switch is the sign of that column, with no extra
    parameter. Each branch only receives gradients from its own rows (navigation: gate 0, reaching: gate 1).
    """
    kind = "gated"

    def __init__(self, input_dim: int, hidden: list[int], layernorm_eps: float = 1e-5, gate_index: int = -1):
        super().__init__()
        self.nav = MLP(input_dim, hidden, layernorm_eps)
        self.reach = MLP(input_dim, hidden, layernorm_eps)
        self.input_dim, self.hidden, self.layernorm_eps = self.nav.input_dim, self.nav.hidden, self.nav.layernorm_eps
        self.gate_index = int(gate_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = (x[..., self.gate_index] > 0).unsqueeze(-1).to(x.dtype)
        return g * self.reach(x) + (1.0 - g) * self.nav(x)


def expand_wheels(y9):
    """[..., 7 torques, right, left] -> [..., 7 torques, FR=right, FL=left, BL=left, BR=right]."""
    if isinstance(y9, torch.Tensor):
        return torch.cat((y9[..., :7], y9[..., 7:8], y9[..., 8:9], y9[..., 8:9], y9[..., 7:8]), dim=-1)
    import numpy as np
    y9 = np.asarray(y9)
    return np.concatenate((y9[..., :7], y9[..., 7:8], y9[..., 8:9], y9[..., 8:9], y9[..., 7:8]), axis=-1)


def build_model(config: dict) -> nn.Module:
    """model.type: absent or 'mlp' (one network), or 'gated' (two networks switched by the last input, reach_gate)."""
    m = config["model"]
    hidden = m["architectures"][config["identity"]["architecture"]]
    kind = m.get("type", "mlp")
    if kind == "gated":
        if config.get("data", {}).get("feature_definition") != "features-v1-gate":
            raise ValueError("model.type 'gated' switches on reach_gate and needs feature_definition features-v1-gate")
        return GatedMLP(m["input_dim"], hidden, m["layernorm_eps"], gate_index=m["input_dim"] - 1)
    if kind != "mlp":
        raise ValueError(f"model.type must be 'mlp' or 'gated', got {kind!r}")
    return MLP(m["input_dim"], hidden, m["layernorm_eps"])


def model_from_spec(spec: dict) -> nn.Module:
    """Rebuild a network from a saved model spec (checkpoint metadata)."""
    if spec.get("type", "mlp") == "gated":
        return GatedMLP(spec["input_dim"], spec["hidden"], spec["layernorm_eps"], gate_index=spec["gate_index"])
    return MLP(spec["input_dim"], spec["hidden"], spec["layernorm_eps"])
