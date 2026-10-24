"""Headless simulation wrapper around the setup in robotnik_arm_task.py."""
from core.interfaces import Action, Observation


class RobotEnv:
    def __init__(self, gui: bool = False, control_hz: int = 100):
        raise NotImplementedError

    def reset(self, scenario) -> Observation:
        """Reset the robot, target and physics to the given scenario."""
        raise NotImplementedError

    def step(self, action: Action) -> tuple[Observation, dict]:
        """Apply one action for one control period. info holds ee_pos, ee_err, done."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
