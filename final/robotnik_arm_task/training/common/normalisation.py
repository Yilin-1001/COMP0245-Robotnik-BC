"""Fitting-only input/output standardisation with recorded scale floors.

Statistics are step-weighted population moments (ddof=0) accumulated in float64 over the fitting rows only.
Transforms keep the caller's backend: NumPy in, NumPy out; a torch Tensor stays a Tensor on its device and dtype,
with gradients flowing through.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FLOOR_KEYS = ("input_scale_floor", "arm_scale_floor", "wheel_scale_floor")


@dataclass
class Normaliser:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    feature_names: list
    label_names: list
    floors: dict

    def __post_init__(self):
        for name in ("x_mean", "x_scale", "y_mean", "y_scale"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        if not (np.all(self.x_scale > 0) and np.all(self.y_scale > 0)):
            raise ValueError("normaliser scales must be positive")
        if len(self.y_mean) != 9 or len(self.x_mean) != len(self.feature_names):
            raise ValueError("normaliser dimensions disagree with its feature/label names")

    @staticmethod
    def _apply(v, shift, scale, forward: bool):
        if isinstance(v, np.ndarray) or np.isscalar(v) or isinstance(v, (list, tuple)):
            v = np.asarray(v, dtype=np.float64)
            return (v - shift) / scale if forward else v * scale + shift
        import torch
        s = torch.as_tensor(shift, dtype=v.dtype, device=v.device)
        c = torch.as_tensor(scale, dtype=v.dtype, device=v.device)
        return (v - s) / c if forward else v * c + s

    def transform_x(self, x):
        return self._apply(x, self.x_mean, self.x_scale, True)

    def transform_y(self, y):
        return self._apply(y, self.y_mean, self.y_scale, True)

    def inverse_y(self, y):
        return self._apply(y, self.y_mean, self.y_scale, False)

    def to_dict(self) -> dict:
        return {"x_mean": self.x_mean.tolist(), "x_scale": self.x_scale.tolist(), "y_mean": self.y_mean.tolist(),
                "y_scale": self.y_scale.tolist(), "feature_names": list(self.feature_names),
                "label_names": list(self.label_names), "floors": dict(self.floors)}

    @classmethod
    def from_dict(cls, d: dict) -> "Normaliser":
        return cls(**{k: d[k] for k in ("x_mean", "x_scale", "y_mean", "y_scale", "feature_names", "label_names",
                                        "floors")})

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        d = self.to_dict()
        np.savez(path, x_mean=self.x_mean, x_scale=self.x_scale, y_mean=self.y_mean, y_scale=self.y_scale,
                 names=np.array(json.dumps({k: d[k] for k in ("feature_names", "label_names", "floors")})))

    @classmethod
    def load(cls, path: Path) -> "Normaliser":
        with np.load(path) as z:
            names = json.loads(str(z["names"]))
            return cls(z["x_mean"], z["x_scale"], z["y_mean"], z["y_scale"], **names)


def fit_normaliser(cache, floors: dict) -> Normaliser:
    """Per-column mean and floored population std over the fitting cache rows (every step weighs the same)."""
    missing = [k for k in FLOOR_KEYS if k not in floors]
    if missing:
        raise ValueError(f"missing scale floors {missing}")
    x = np.asarray(cache.x, dtype=np.float64)
    y = np.asarray(cache.y, dtype=np.float64)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_std = np.sqrt(((x - x_mean) ** 2).mean(axis=0))
    y_std = np.sqrt(((y - y_mean) ** 2).mean(axis=0))
    y_floor = np.r_[np.full(7, floors["arm_scale_floor"]), np.full(2, floors["wheel_scale_floor"])]
    names = cache.metadata.get("feature_names") or [f"x{i}" for i in range(x.shape[1])]
    labels = cache.metadata.get("label_names") or [f"torque_{i}" for i in range(7)] + ["wheel_right", "wheel_left"]
    return Normaliser(x_mean, np.maximum(x_std, floors["input_scale_floor"]), y_mean, np.maximum(y_std, y_floor),
                      list(names), list(labels), {k: float(floors[k]) for k in FLOOR_KEYS})
