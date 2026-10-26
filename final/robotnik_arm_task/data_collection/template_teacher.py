"""The template controller (robotnik_arm_control.py) as a stateful teacher, set up as in robotnik_arm_task.py.

Teacher for dataset B (docs/expert_datasets.md). It keeps the template's memory: the integrated
arm reference q_des_arm and the base navigation phase/done flag. Differences from
robotnik_arm_task.py are only in where measurements come from:
- Base position, yaw and arm joints come from the Observation, so collection noise reaches it
  as it reaches the Markovian expert. Roll and pitch, which the Observation lacks, come from the
  simulator (privileged teacher-side information), as do the finger and wheel angles in q_pin.
- The base goal uses the template's rule: the standoff point on the line from the world origin
  to the target, facing along that line. It does not depend on the start pose.

Call order per step matches robotnik_arm_task.py: base_controller first, then arm_controller with
reach_active = base_state.done. act() returns the action and before/after snapshots of the memory.

The privileged inputs (true base quaternion, all 17 motor angles) and the time can be passed to
act() explicitly, so a stored episode can be replayed without the simulator; by default they are
read from the environment.
"""
import numpy as np
import pinocchio as pin

from core.interfaces import Action, Observation
from robotnik_arm_control import (ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller,
                                  base_controller, quaternion2bearing)

# Values from robotnik_arm_task.py
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
PHASES = ("align", "drive", "orient")   # integer codes 0, 1, 2 in stored datasets
CONTROL_DT = 0.001                      # time_step passed to arm_controller, as sim.GetTimeStep() in the template


def template_base_goal(target: np.ndarray) -> tuple[np.ndarray, float]:
    """Base goal position and bearing, computed once per episode as in robotnik_arm_task.py."""
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = np.asarray(target[:2], dtype=float)
    bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    return target_xy - standoff * np.array([np.cos(bearing), np.sin(bearing)]), float(bearing)


class TemplateTeacher:
    name = "template"

    def __init__(self, env):
        self.env = env
        dm = env.dyn_model
        self.shoulder_frame_id = dm.pin_model.getFrameId("robot_j2s7s300_link_2")
        lower, upper = env.sim.GetBotJointsLimit()
        self.arm_lower, self.arm_upper = np.array(lower[4:11]), np.array(upper[4:11])
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)

    def reset(self, target: np.ndarray | None = None, q_hold: np.ndarray | None = None) -> None:
        """Call after env.reset(scenario): plans the base goal and initialises the memory.

        q_hold is the measured initial arm configuration (robotnik_arm_task.py uses the angles read
        after setup); it defaults to the configured initial angles, which env.reset sets exactly.
        """
        self.target = np.asarray(self.env.scenario.target_ee_pos if target is None else target, dtype=float).copy()
        self.goal_pos, self.goal_bearing = template_base_goal(self.target)
        self.base_state = BaseNavState()
        self.q_hold = (self.env.init_motor_angles[4:11] if q_hold is None else np.asarray(q_hold, dtype=float)).copy()
        self.q_des = self.q_hold.copy()
        self.qd_des = np.zeros(7)

    def privileged_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-side measurements the Observation lacks: true base quaternion (x, y, z, w) and all 17 motor angles."""
        return np.array(self.env.sim.GetBaseOrientation()), np.array(self.env.sim.GetMotorAngles(0))

    def act(self, obs: Observation, t: float | None = None, base_quat: np.ndarray | None = None,
            motor_angles: np.ndarray | None = None) -> tuple[Action, dict]:
        env, dm = self.env, self.env.dyn_model
        if base_quat is None or motor_angles is None:
            base_quat, motor_angles = self.privileged_inputs()
        t = env.t if t is None else t
        base_quat = np.asarray(base_quat, dtype=float)
        if quaternion2bearing(base_quat[3], base_quat[0], base_quat[1], base_quat[2]) == obs.base_bearing:
            # Observed yaw is the true yaw (no sensor noise): use the simulator quaternion as the template does.
            # Rebuilding it from roll/pitch/yaw changes it by ~1e-16, which the closed loop amplifies.
            quat = base_quat
        else:
            roll, pitch, _ = pin.rpy.matrixToRpy(pin.Quaternion(*np.roll(base_quat, 1)).matrix())
            quat = pin.Quaternion(pin.rpy.rpyToMatrix(roll, pitch, obs.base_bearing)).coeffs()  # x, y, z, w
        motor_angles = np.array(motor_angles, dtype=float)
        motor_angles[4:11] = obs.q_mes
        q_pin = dm.ReoderJoints2PinVec(np.concatenate((obs.base_pos, quat, motor_angles)), "pos")
        pin.forwardKinematics(dm.pin_model, dm.pin_data, q_pin)
        pin.updateFramePlacements(dm.pin_model, dm.pin_data)
        ee_pos = dm.pin_data.oMf[env.ee_frame_id].translation.copy()

        before = {"q_des": self.q_des.copy(), "phase": self.base_state.phase, "nav_done": self.base_state.done}
        wheel_cmds = base_controller(self.base_state, obs.base_pos, obs.base_bearing, self.goal_pos,
                                     self.goal_bearing, t, self.base_params)
        reach_active = self.base_state.done
        torque_arm, q_des, qd_des = arm_controller(
            dm, q_pin, obs.q_mes, obs.qd_mes, self.q_des, self.q_hold, self.target, ee_pos, env.ee_frame_id,
            self.shoulder_frame_id, self.arm_lower, self.arm_upper, CONTROL_DT, self.arm_params, reach_active=reach_active)
        self.q_des, self.qd_des = q_des.copy(), qd_des.copy()
        state = {"before": before,
                 "after": {"q_des": self.q_des.copy(), "phase": self.base_state.phase, "nav_done": self.base_state.done},
                 "qd_des": self.qd_des.copy(), "reach_active": reach_active, "teacher_ee_pos": ee_pos}
        return Action(np.array(torque_arm, dtype=float), np.array(wheel_cmds, dtype=float)), state
