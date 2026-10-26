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
SENSOR_NOISE_STD = 1e-4  # robot_noise std in configs/robotnik_arm.json (rad, rad/s, m, rad); physics["sensor_noise_scale"] multiplies it
PHYSICS_KEYS = {"friction_scale", "mass_scale", "sensor_noise_scale", "push"}
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

        # Nominal link masses and inertias, restored on every reset before any mass scaling
        self.links = [-1] + list(range(self.client.getNumJoints(self.rid)))
        self.nominal_dynamics = {j: self.client.getDynamicsInfo(self.rid, j)[:3:2] for j in self.links}
        self.ee_link = next(j for j in self.links[1:] if self.client.getJointInfo(self.rid, j)[12].decode() == EE_FRAME)

        self.init_motor_angles = np.array(self.sim.GetInitMotorAngles())
        self.init_motor_angles[11:17] = FIXED_FINGER_ANGLES
        self.cmd = MotorCommands()
        self.target_body = None
        self.scenario = None
        self.t = 0.0
        self.noise_std = 0.0
        self.rng = np.random.default_rng(0)
        self.push = None
        self.first_reach_t = None

    def reset(self, scenario) -> Observation:
        """Reset the robot, target and physics to the given scenario.

        scenario.physics may hold (all optional, nominal when absent):
          friction_scale:     floor friction = 1.0 * scale
          mass_scale:         every link's mass and inertia * scale
          sensor_noise_scale: Gaussian noise of std SENSOR_NOISE_STD * scale on every observed quantity
                              (the ground-truth info["ee_pos"] stays noise-free)
          push:               {"link": "base" | "ee", "force": [fx, fy, fz] (N, world frame), "duration": s,
                               and "start": s, or "after_reach": s after the end-effector first reaches the target}
        """
        physics = scenario.physics or {}
        unknown = set(physics) - PHYSICS_KEYS
        if unknown:
            raise ValueError(f"unknown physics keys: {sorted(unknown)}")
        self.sim.SetFloorFriction(1.0 * physics.get("friction_scale", 1.0))
        mass_scale = physics.get("mass_scale", 1.0)
        for j, (mass, inertia) in self.nominal_dynamics.items():
            self.client.changeDynamics(self.rid, j, mass=mass * mass_scale,
                                       localInertiaDiagonal=[i * mass_scale for i in inertia])
        self.noise_std = SENSOR_NOISE_STD * physics.get("sensor_noise_scale", 0.0)
        self.rng = np.random.default_rng(scenario.seed % 2**32)  # seeds may be negative (compare_controllers uses -1)
        self.push = physics.get("push")
        self.first_reach_t = None
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
        for k in range(self.substeps):
            self._apply_push(self.t + k * SIM_DT)
            self.sim.Step(self.cmd, "torque")
        self.t += self.substeps * SIM_DT
        obs = self._observe()
        ee_pos = self.ee_pos()
        ee_err = float(np.linalg.norm(self.scenario.target_ee_pos - ee_pos))
        done = ee_err < self.reach_tolerance
        if done and self.first_reach_t is None:
            self.first_reach_t = self.t
        return obs, {"ee_pos": ee_pos, "ee_err": ee_err, "done": done, "t": self.t, "pushing": self._push_active(self.t)}

    def _push_active(self, t: float) -> bool:
        if self.push is None:
            return False
        if "after_reach" in self.push:
            if self.first_reach_t is None:
                return False
            start = self.first_reach_t + self.push["after_reach"]
        else:
            start = self.push["start"]
        return start <= t < start + self.push["duration"]

    def _apply_push(self, t: float) -> None:
        """External force for the scenario's push, applied at the link's centre of mass (one physics step)."""
        if not self._push_active(t):
            return
        link = -1 if self.push["link"] == "base" else self.ee_link
        pos = self.client.getBasePositionAndOrientation(self.rid)[0] if link == -1 else self.client.getLinkState(self.rid, link)[0]
        self.client.applyExternalForce(self.rid, link, list(self.push["force"]), pos, p.WORLD_FRAME)

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
        noise = lambda n: self.rng.normal(0.0, self.noise_std, n) if self.noise_std else np.zeros(n)
        return Observation(
            base_pos=np.array(self.sim.GetBasePosition()) + noise(3),
            base_bearing=float(quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2]) + noise(1)[0]),
            q_mes=q_mes[4:11] + noise(7),
            qd_mes=qd_mes[4:11] + noise(7),
            target_ee_pos=np.asarray(self.scenario.target_ee_pos, dtype=float).copy(),
        )

    def close(self) -> None:
        self.client.disconnect()
