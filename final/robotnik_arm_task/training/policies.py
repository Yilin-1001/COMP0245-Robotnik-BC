"""Learned policies wrapped in the Policy interface.

MLP: 3x256 with LayerNorm, separate arm-torque and wheel heads.
ACT (optional): action chunks with temporal ensembling.
"""
from core.interfaces import Action, Observation, Policy


class MLPPolicy(Policy):
    name = "mlp"

    @classmethod
    def load(cls, ckpt_path: str) -> "MLPPolicy":
        raise NotImplementedError

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError


class ACTPolicy(Policy):
    name = "act"

    @classmethod
    def load(cls, ckpt_path: str) -> "ACTPolicy":
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError
