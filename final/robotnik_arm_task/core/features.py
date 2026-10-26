"""Deployable student inputs, built from stored raw observations (docs/expert_datasets.md).

Shared by training and inference so they cannot drift. Only deployment-available information is
used: the observation (base position and yaw, arm joints), the episode's target and, for dataset B,
the fixed navigation goal and a fixed time scale. Teacher memory (q_des, phase, done) and simulator
truth (EE position, roll/pitch, velocities) are never inputs.

Dataset A (21): q_mes (7), qd_mes (7), target in B_yaw (3), FK EE-to-target error in B_yaw (3), target z (1)
Dataset B (26): A's 21 + navigation goal in B_yaw (2), sin/cos(goal_bearing - yaw) (2), elapsed / time_scale (1)
B_yaw: origin at the base position, rotated by base yaw, world vertical axis kept.
"""
import os

import numpy as np
import pinocchio as pin

FEATURE_DEFINITION_VERSION = "features-v1"
FK_VERSION = "fk-v1"
TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(TASK_DIR, "models", "summit_xl_arm_description", "summit_xl_complete_pin.urdf")
EE_FRAME = "robot_j2s7s300_end_effector"
ARM_JOINTS = [f"robot_j2s7s300_joint_{i}" for i in range(1, 8)]
FINGER_JOINTS = [f"robot_j2s7s300_joint_finger{s}_{i}" for i in range(1, 4) for s in ("", "_tip")]
FK_CONFIG = {"version": FK_VERSION, "urdf": "models/summit_xl_arm_description/summit_xl_complete_pin.urdf",
             "ee_frame": EE_FRAME, "fixed_finger_angles": [0.1] * 6, "base_roll": 0.0, "base_pitch": 0.0022,
             "pitch_source": "median flat-ground pitch, docs/expert_model_accuracy.md (same value as MarkovianExpertParams.base_pitch)",
             "inputs": "obs base_pos (3), base_yaw, q_mes (7)"}
FEATURE_NAMES_A = ([f"q_mes_{i}" for i in range(7)] + [f"qd_mes_{i}" for i in range(7)] + ["target_bx", "target_by", "target_bz"]
                   + ["ee_err_bx", "ee_err_by", "ee_err_bz", "target_z_world"])
FEATURE_NAMES_B = FEATURE_NAMES_A + ["goal_bx", "goal_by", "sin_goal_heading_err", "cos_goal_heading_err", "elapsed_scaled"]


class DeployableFK:
    """EE position from the observation alone: observed yaw, fixed roll/pitch and fixed finger angles."""

    def __init__(self):
        self.model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(EE_FRAME)
        self.arm_idx = np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in ARM_JOINTS])
        self.q = pin.neutral(self.model)
        for name, angle in zip(FINGER_JOINTS, FK_CONFIG["fixed_finger_angles"]):
            self.q[self.model.joints[self.model.getJointId(name)].idx_q] = angle

    def ee_pos(self, base_pos, base_yaw, q_mes) -> np.ndarray:
        """Vectorised over leading dimension: base_pos (N,3), base_yaw (N,), q_mes (N,7) -> (N,3) world EE positions."""
        base_pos, base_yaw, q_mes = np.atleast_2d(base_pos), np.atleast_1d(base_yaw), np.atleast_2d(q_mes)
        out = np.empty((len(base_yaw), 3))
        q = self.q.copy()
        for i in range(len(base_yaw)):
            q[:3] = base_pos[i]
            q[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(FK_CONFIG["base_roll"], FK_CONFIG["base_pitch"], base_yaw[i])).coeffs()
            q[self.arm_idx] = q_mes[i]
            pin.framesForwardKinematics(self.model, self.data, q)
            out[i] = self.data.oMf[self.ee_id].translation
        return out


def to_base_yaw(vec_w: np.ndarray, base_yaw: np.ndarray) -> np.ndarray:
    """Rotate world-frame displacements (N,k>=2) into B_yaw: Rz(-yaw) applied to x, y; z unchanged."""
    c, s = np.cos(base_yaw), np.sin(base_yaw)
    out = np.array(vec_w, dtype=float, copy=True)
    out[:, 0] = c * vec_w[:, 0] + s * vec_w[:, 1]
    out[:, 1] = -s * vec_w[:, 0] + c * vec_w[:, 1]
    return out


def features_a(base_pos, base_yaw, q_mes, qd_mes, target_w, fk: DeployableFK) -> np.ndarray:
    """(N,21) dataset A inputs from N pre-action observations."""
    n = len(base_yaw)
    target = np.broadcast_to(np.asarray(target_w, dtype=float), (n, 3))
    target_b = to_base_yaw(target - base_pos, base_yaw)
    ee_err_b = to_base_yaw(target - fk.ee_pos(base_pos, base_yaw, q_mes), base_yaw)
    return np.concatenate((q_mes, qd_mes, target_b, ee_err_b, target[:, 2:3]), axis=1)


def features_b(base_pos, base_yaw, q_mes, qd_mes, t, target_w, goal_pos_w, goal_bearing, time_scale_s,
               fk: DeployableFK) -> np.ndarray:
    """(N,26) dataset B inputs: A's 21 plus the fixed navigation goal, final heading error and scaled time."""
    a = features_a(base_pos, base_yaw, q_mes, qd_mes, target_w, fk)
    goal = np.broadcast_to(np.asarray(goal_pos_w, dtype=float), (len(base_yaw), 2))
    goal_b = to_base_yaw(goal - base_pos[:, :2], base_yaw)
    err = goal_bearing - base_yaw
    return np.concatenate((a, goal_b, np.sin(err)[:, None], np.cos(err)[:, None], (np.asarray(t) / time_scale_s)[:, None]), axis=1)
