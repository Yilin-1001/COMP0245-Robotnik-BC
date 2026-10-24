"""Headless simulation wrapper around the setup in robotnik_arm_task.py."""
import os

import numpy as np
import pinocchio as pin
import pybullet as p
from simulation_and_control import MotorCommands, PinWrapper, pb

from core.interfaces import Action, Observation
from robotnik_arm_control import configure_gripper_fingers, quaternion2bearing

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF_FILE = "robotnik_arm.json"
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # as in robotnik_arm_task.py
EE_FRAME = "robot_j2s7s300_end_effector"
SIM_DT = 0.001
CONTROL_MODES = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6


class RobotEnv:
    def __init__(self, gui: bool = False, control_hz: int = 1000, reach_tolerance: float = 0.01):
        if round(1.0 / (SIM_DT * control_hz)) * SIM_DT * control_hz != 1.0:
            raise ValueError(f"control_hz must divide {1 / SIM_DT:.0f}")
        self.substeps = int(round(1.0 / (SIM_DT * control_hz)))
        self.reach_tolerance = reach_tolerance
        self.gui = gui

        # Same physics setup as robotnik_arm_task.py
        self.sim = pb.SimInterface(CONF_FILE, conf_file_path_ext=TASK_DIR, use_gui=gui)
        self.sim.SetFloorFriction(1.0)
        self.client = self.sim.pybullet_client
        self.rid = self.sim.bot[0].bot_pybullet
        for j in range(self.client.getNumJoints(self.rid)):
            if "wheel" in self.client.getJointInfo(self.rid, j)[1].decode():
                self.client.changeDynamics(self.rid, j, contactStiffness=30000, contactDamping=1000,
                                           anisotropicFriction=[1.0, 0.01, 1.0])
        self.sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        configure_gripper_fingers(self.sim, self.client, self.rid, FIXED_FINGER_ANGLES)

        # Ground-truth kinematics (full base orientation), used only for info["ee_pos"]
        ext_names = np.expand_dims(np.array(self.sim.getNameActiveJoints()), axis=0)
        self.dyn_model = PinWrapper(CONF_FILE, "pybullet", ext_names, ["pybullet"], False, 0, TASK_DIR)
        self.ee_frame_id = self.dyn_model.pin_model.getFrameId(EE_FRAME)

        self.init_motor_angles = np.array(self.sim.GetInitMotorAngles())
        self.init_motor_angles[11:17] = FIXED_FINGER_ANGLES
        self.cmd = MotorCommands()
        self.target_body = None
        self.scenario = None
        self.t = 0.0

    def reset(self, scenario) -> Observation:
        """Reset the robot, target and physics to the given scenario."""
        if scenario.physics:
            raise NotImplementedError("physics perturbations are not supported yet")
        self.scenario = scenario
        start = np.asarray(scenario.base_start_pos, dtype=float)
        quat = p.getQuaternionFromEuler([0.0, 0.0, scenario.base_start_bearing])
        self.client.resetBasePositionAndOrientation(self.rid, [start[0], start[1], 0.0], quat)
        self.client.resetBaseVelocity(self.rid, [0, 0, 0], [0, 0, 0])
        for i, jid in enumerate(self.sim.bot[0].active_joint_ids):
            self.client.resetJointState(self.rid, jid, targetValue=float(self.init_motor_angles[i]), targetVelocity=0.0)
        self.sim.ReceiveObservation()
        self.t = 0.0

        if self.gui:
            if self.target_body is not None:
                self.client.removeBody(self.target_body)
            shape = self.client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
            self.target_body = self.client.createMultiBody(
                baseMass=0, baseVisualShapeIndex=shape, basePosition=list(scenario.target_ee_pos))
        return self._observe()

    def step(self, action: Action) -> tuple[Observation, dict]:
        """Apply one action for one control period. info holds ee_pos, ee_err, done."""
        cmd_all = np.concatenate((action.wheel_cmds, action.torque_arm, np.zeros(6)))
        self.cmd.SetControlCmd(cmd_all, CONTROL_MODES)
        for _ in range(self.substeps):
            self.sim.Step(self.cmd, "torque")
        self.t += self.substeps * SIM_DT
        obs = self._observe()
        ee_pos = self.ee_pos()
        ee_err = float(np.linalg.norm(self.scenario.target_ee_pos - ee_pos))
        return obs, {"ee_pos": ee_pos, "ee_err": ee_err, "done": ee_err < self.reach_tolerance, "t": self.t}

    def ee_pos(self) -> np.ndarray:
        """End-effector position from the true base pose (including roll and pitch)."""
        q_all = np.concatenate((self.sim.GetBasePosition(), self.sim.GetBaseOrientation(), self.sim.GetMotorAngles(0)))
        q_pin = self.dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(self.dyn_model.pin_model, self.dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(self.dyn_model.pin_model, self.dyn_model.pin_data)
        return self.dyn_model.pin_data.oMf[self.ee_frame_id].translation.copy()

    def _observe(self) -> Observation:
        base_ori = np.array(self.sim.GetBaseOrientation())
        q_mes = np.array(self.sim.GetMotorAngles(0))
        qd_mes = np.array(self.sim.GetMotorVelocities(0))
        return Observation(
            base_pos=np.array(self.sim.GetBasePosition()),
            base_bearing=float(quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])),
            q_mes=q_mes[4:11],
            qd_mes=qd_mes[4:11],
            target_ee_pos=np.asarray(self.scenario.target_ee_pos, dtype=float).copy(),
        )

    def close(self) -> None:
        self.client.disconnect()
