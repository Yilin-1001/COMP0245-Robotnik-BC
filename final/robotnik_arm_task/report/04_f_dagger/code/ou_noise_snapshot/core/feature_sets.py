"""Registered student feature definitions, looked up by (teacher, name) in training, dataset building and deployment.

features-v1      A: 21 inputs, B: 26 inputs, exactly core/features.py (left unchanged so existing cache IDs hold).
features-v1-gate A only: the 21 v1 inputs plus reach_gate (22), teacher A's reach-active rule recomputed from the
                 deployable v1 columns target_bx, target_by and target_z_world:
                     d = |target_b[:2]| - standoff(target_z),  h = atan2(target_by, target_bx)  (0 when |target_b[:2]| <= 1 mm)
                     reach_gate = 1 if |d| < 0.12 m and |h| < 0.075 rad, else 0
                 The thresholds are MarkovianExpertParams.reach_tol_scale (1.5) x BaseControllerParams pos_tol / ang_tol.
New definitions are appended here; a model stores its definition name, so it always gets the builder it was trained on.
"""
import numpy as np

from core import features

DEFAULT_DEFINITION = "features-v1"
GATE_RULE = {"dist_tol_m": 0.12, "heading_tol_rad": 0.075, "min_planar_dist_m": 1e-3,
             "standoff_z_points": list(features.STANDOFF_Z_POINTS), "standoff_points": list(features.STANDOFF_POINTS),
             "source": "MarkovianExpert.base_command: |dist_err| < 1.5 * 0.08 m and |heading_err| < 1.5 * 0.05 rad"}
_COL = {name: i for i, name in enumerate(features.FEATURE_NAMES_A)}
DEFINITIONS = {
    ("A", "features-v1"): list(features.FEATURE_NAMES_A),
    ("B", "features-v1"): list(features.FEATURE_NAMES_B),
    ("A", "features-v1-gate"): list(features.FEATURE_NAMES_A) + ["reach_gate"],
}


def feature_names(teacher: str, definition: str | None = None) -> list[str]:
    key = (teacher, definition or DEFAULT_DEFINITION)
    if key not in DEFINITIONS:
        known = sorted(d for t, d in DEFINITIONS if t == teacher)
        raise ValueError(f"unknown feature definition {key[1]!r} for teacher {teacher}; known: {known}")
    return list(DEFINITIONS[key])


def input_dim(teacher: str, definition: str | None = None) -> int:
    return len(feature_names(teacher, definition))


def reach_gate(x_v1: np.ndarray) -> np.ndarray:
    """(..., 21) teacher-A v1 features -> (...,) 0/1 reach gate, computed in float64."""
    x = np.asarray(x_v1, dtype=np.float64)
    bx, by, tz = x[..., _COL["target_bx"]], x[..., _COL["target_by"]], x[..., _COL["target_z_world"]]
    planar = np.hypot(bx, by)
    d = planar - np.interp(tz, GATE_RULE["standoff_z_points"], GATE_RULE["standoff_points"])
    h = np.where(planar > GATE_RULE["min_planar_dist_m"], np.arctan2(by, bx), 0.0)
    return ((np.abs(d) < GATE_RULE["dist_tol_m"]) & (np.abs(h) < GATE_RULE["heading_tol_rad"])).astype(np.float64)


def extend(teacher: str, definition: str | None, x_v1: np.ndarray) -> np.ndarray:
    """Definition features from the v1 features of the same rows ((N, d_v1) or (d_v1,))."""
    definition = definition or DEFAULT_DEFINITION
    feature_names(teacher, definition)
    if definition == "features-v1":
        return x_v1
    x = np.asarray(x_v1, dtype=np.float64)
    return np.concatenate((x, reach_gate(x)[..., None]), axis=-1)
