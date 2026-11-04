"""Observation/action layout and the Policy interface.

Fixed on day 1. Change it only in a commit of its own: the expert, the MLP and
ACT all implement Policy, so evaluate.py can score any of them.
"""
from dataclasses import dataclass

import numpy as np

N_ARM = 7
N_WHEELS = 4


@dataclass
class Observation:
    """Raw measurements at one control step, as read in robotnik_arm_task.py."""
    base_pos: np.ndarray       # (3,) wheeled base position, world frame
    base_bearing: float        # base yaw (rad)
    q_mes: np.ndarray          # (7,) arm joint angles (rad), q_mes[4:11] of the sim vector
    qd_mes: np.ndarray         # (7,) arm joint velocities (rad/s)
    target_ee_pos: np.ndarray  # (3,) TARGET_EE_POS, world frame


@dataclass
class Action:
    torque_arm: np.ndarray     # (7,) arm joint torques
    wheel_cmds: np.ndarray     # (4,) wheel angular velocities (rad/s)


class Policy:
    """Maps one Observation to one Action. Must not depend on hidden state
    unless reset() clears it at the start of each episode."""

    name = "policy"

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError
