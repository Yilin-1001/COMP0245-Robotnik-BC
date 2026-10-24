"""Markovian version of the controller in robotnik_arm_control.py.

The template keeps q_des_arm and a navigation phase between steps, so it cannot
label an arbitrary state. This expert computes its action from the current
state alone, which DART labelling and DAgger relabelling both need.

Changes from the template (robotnik_arm_control.py is left unmodified):
- Arm reference: the integrated q_des_arm becomes q + qd_des * lead_time, so the
  joint PD tracks a reference that is a function of the current state.
- Base: the align/drive/orient/done phases become a function of the distance and
  heading error to the target. The base goal is the standoff point on the line
  from the base to the target, so it does not depend on the start pose.
- Kinematics: built from the observation only (base yaw, arm angles and the fixed
  finger angles). Roll and pitch are not observed, so the base pitch is set to the
  value measured on flat ground.
"""
import os
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
from simulation_and_control import wrap_angle

from core.interfaces import Action, Observation, Policy
from robotnik_arm_control import ArmControllerParams, BaseControllerParams

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(TASK_DIR, "models", "summit_xl_arm_description", "summit_xl_complete_pin.urdf")
ARM_JOINTS = [f"robot_j2s7s300_joint_{i}" for i in range(1, 8)]
FINGER_JOINTS = [f"robot_j2s7s300_joint_finger{s}_{i}" for i in range(1, 4) for s in ("", "_tip")]
FIXED_FINGER_ANGLES = np.full(6, 0.1)
Q_HOLD_ARM = np.array([0.0, 1.57, 0.0, 1.0, 0.0, 1.0, 0.0])  # initial arm pose in configs/robotnik_arm.json

# Values from robotnik_arm_task.py
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]


@dataclass
class MarkovianExpertParams:
    arm: ArmControllerParams = field(default_factory=lambda: ArmControllerParams(reach_limit=ARM_REACH_LIMIT))
    base: BaseControllerParams = field(default_factory=lambda: BaseControllerParams(nav_timeout=np.inf))
    lead_time: float = 0.15          # s, q_des = q + qd_des * lead_time (max lead = max_joint_vel * lead_time)
    drive_heading_tol: float = 0.3   # rad, forward speed fades to zero as the heading error reaches this
    reach_tol_scale: float = 1.5     # the arm reaches while base errors are within this x the base stop tolerances
    base_pitch: float = 0.0022       # rad, median base pitch at the end of 20 random episodes (roll was ~0)


class MarkovianExpert(Policy):
    name = "expert"

    def __init__(self, params: MarkovianExpertParams | None = None):
        self.params = params or MarkovianExpertParams()
        self.model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
        self.ee_frame_id = self.model.getFrameId("robot_j2s7s300_end_effector")
        # The URDF gives the end-effector link no inertia, so PyBullet simulates it with its default
        # 1 kg at the link origin. The template's integrated reference hides the missing gravity
        # torque; a Markovian reference has no integrator, so the model must include it.
        ee_frame = self.model.frames[self.ee_frame_id]
        self.model.appendBodyToJoint(ee_frame.parentJoint, pin.Inertia(1.0, np.zeros(3), np.eye(3)), ee_frame.placement)
        self.data = self.model.createData()
        self.shoulder_frame_id = self.model.getFrameId("robot_j2s7s300_link_2")
        arm_ids = [self.model.getJointId(n) for n in ARM_JOINTS]
        self.arm_idx_q = np.array([self.model.joints[j].idx_q for j in arm_ids])
        self.arm_idx_v = np.array([self.model.joints[j].idx_v for j in arm_ids])
        self.arm_lower = self.model.lowerPositionLimit[self.arm_idx_q]
        self.arm_upper = self.model.upperPositionLimit[self.arm_idx_q]
        self.q_ref = pin.neutral(self.model)
        for name, angle in zip(FINGER_JOINTS, FIXED_FINGER_ANGLES):
            self.q_ref[self.model.joints[self.model.getJointId(name)].idx_q] = angle

    def act(self, obs: Observation) -> Action:
        wheel_cmds, reach_active = self.base_command(obs)
        return Action(torque_arm=self.arm_torque(obs, reach_active), wheel_cmds=wheel_cmds)

    def q_pin(self, obs: Observation) -> np.ndarray:
        q = self.q_ref.copy()
        q[:3] = obs.base_pos
        q[3:7] = pin.Quaternion(pin.rpy.rpyToMatrix(0.0, self.params.base_pitch, obs.base_bearing)).coeffs()  # x, y, z, w
        q[self.arm_idx_q] = obs.q_mes
        return q

    def ee_pos(self, obs: Observation) -> np.ndarray:
        """Forward kinematics from the observation alone."""
        pin.framesForwardKinematics(self.model, self.data, self.q_pin(obs))
        return self.data.oMf[self.ee_frame_id].translation.copy()

    def base_errors(self, obs: Observation) -> tuple[float, float]:
        """Signed distance to the standoff point and heading error towards the target."""
        to_target = obs.target_ee_pos[:2] - obs.base_pos[:2]
        dist = np.linalg.norm(to_target)
        standoff = float(np.interp(obs.target_ee_pos[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        heading_err = wrap_angle(np.arctan2(to_target[1], to_target[0]) - obs.base_bearing) if dist > 1e-3 else 0.0
        return dist - standoff, heading_err

    def base_command(self, obs: Observation) -> tuple[np.ndarray, bool]:
        """Skid-steer wheel speeds [front_right, front_left, back_left, back_right] and whether the arm may reach.

        Replaces the template's phases: turn in place while the heading error is large
        (align), drive forwards or backwards to the standoff distance while steering
        towards the target (drive), turn to face the target once at the standoff
        distance (orient), and stop once both errors are within tolerance (done).
        """
        bp = self.params.base
        dist_err, heading_err = self.base_errors(obs)
        v_cmd, w_cmd = 0.0, 0.0
        if abs(dist_err) >= bp.pos_tol:
            speed = np.clip(bp.kp_pos * abs(dist_err), bp.min_lin_vel, bp.max_lin_vel)
            v_cmd = np.sign(dist_err) * speed * np.clip(1.0 - abs(heading_err) / self.params.drive_heading_tol, 0.0, 1.0)
            w_cmd = np.clip(bp.kp_ori * heading_err, -bp.max_ang_vel, bp.max_ang_vel)
        elif abs(heading_err) >= bp.ang_tol:
            w_cmd = np.clip(bp.kp_ori * heading_err, -bp.max_ang_vel, bp.max_ang_vel)
        lw = (v_cmd - w_cmd * bp.wheel_base_width / 2) / bp.wheel_radius
        rw = (v_cmd + w_cmd * bp.wheel_base_width / 2) / bp.wheel_radius
        scale = self.params.reach_tol_scale
        reach_active = abs(dist_err) < scale * bp.pos_tol and abs(heading_err) < scale * bp.ang_tol
        return np.array([rw, lw, lw, rw]), reach_active

    def arm_torque(self, obs: Observation, reach_active: bool) -> np.ndarray:
        """Template arm_controller with the integrated joint reference replaced by q + qd_des * lead_time."""
        ap = self.params.arm
        q_arm, qd_arm = obs.q_mes, obs.qd_mes
        q_pin = self.q_pin(obs)
        pin.framesForwardKinematics(self.model, self.data, q_pin)
        if not reach_active:
            q_des_arm = Q_HOLD_ARM
            qd_des_arm = np.zeros(7)
        else:
            pin.computeJointJacobians(self.model, self.data, q_pin)
            J_lwa = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_arm = J_lwa[:3, self.arm_idx_v]

            # Increase damping near singular poses to keep joint velocities bounded.
            sigma_min = np.linalg.svd(J_arm, compute_uv=False)[-1]
            damping = np.interp(sigma_min, [0.0, ap.singularity_threshold], [ap.max_ik_damping, ap.min_ik_damping])

            # Proportional Cartesian velocity command towards the (reach-limited) target
            ee_pos = self.data.oMf[self.ee_frame_id].translation
            shoulder_pos = self.data.oMf[self.shoulder_frame_id].translation
            offset = obs.target_ee_pos - shoulder_pos
            if np.linalg.norm(offset) > ap.reach_limit:
                offset *= ap.reach_limit / np.linalg.norm(offset)
            v_cartesian = ap.cartesian_kp * (shoulder_pos + offset - ee_pos)
            cartesian_speed = np.linalg.norm(v_cartesian)
            if cartesian_speed > ap.max_cartesian_speed:
                v_cartesian *= ap.max_cartesian_speed / cartesian_speed

            # Saturation-aware solve: joints at a limit that the motion would push further into are frozen.
            has_limits = self.arm_lower < self.arm_upper
            free = np.ones(7, dtype=bool)
            qd_des_arm = np.zeros(7)
            for _ in range(7):
                J_free = J_arm[:, free]
                qd_des_arm = np.zeros(7)
                qd_des_arm[free] = J_free.T @ np.linalg.solve(J_free @ J_free.T + damping**2 * np.eye(3), v_cartesian)
                pushing_lower = has_limits & (q_arm <= self.arm_lower + ap.limit_margin) & (qd_des_arm < 0)
                pushing_upper = has_limits & (q_arm >= self.arm_upper - ap.limit_margin) & (qd_des_arm > 0)
                newly_blocked = (pushing_lower | pushing_upper) & free
                if not newly_blocked.any():
                    break
                free &= ~newly_blocked
                qd_des_arm[:] = 0.0
            max_joint_speed = np.max(np.abs(qd_des_arm))
            if max_joint_speed > ap.max_joint_vel:
                qd_des_arm *= ap.max_joint_vel / max_joint_speed

            # Markovian reference: a fixed lead along the desired velocity instead of an integrated q_des_arm
            q_des_arm = q_arm + qd_des_arm * self.params.lead_time
            q_des_arm = np.where(
                has_limits,
                np.clip(q_des_arm, self.arm_lower + ap.limit_margin, self.arm_upper - ap.limit_margin),
                q_des_arm,
            )

        # Joint-space PD law with gravity compensation
        torque_arm = ap.kp * (q_des_arm - q_arm) + ap.kd * (qd_des_arm - qd_arm)
        tau_g = pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        return np.clip(torque_arm + tau_g[self.arm_idx_v], -ap.max_torque, ap.max_torque)
