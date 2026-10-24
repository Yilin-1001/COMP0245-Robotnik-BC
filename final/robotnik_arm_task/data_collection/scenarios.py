"""Scenario sampling: target, start pose and physics parameters.

Targets are drawn in polar form: 0.5-3.0 m, azimuth -pi to pi, height 0.1-1.25 m.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    seed: int
    target_ee_pos: np.ndarray
    base_start_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_start_bearing: float = 0.0
    physics: dict = field(default_factory=dict)  # friction / mass scales, noise, push


def sample_scenario(seed: int, ranges: dict) -> Scenario:
    raise NotImplementedError
