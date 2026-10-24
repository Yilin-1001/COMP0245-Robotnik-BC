"""Derived network inputs, shared by training and inference so they cannot drift.

Raw state plus: target in the base frame, sin/cos of the bearing error,
end-effector error from forward kinematics, and a drive/reach phase flag.
"""
import numpy as np

from core.interfaces import Observation


def build_features(obs: Observation, ee_pos: np.ndarray, use_derived: bool = True) -> np.ndarray:
    raise NotImplementedError
