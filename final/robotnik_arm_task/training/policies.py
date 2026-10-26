"""Learned policies wrapped in the Policy interface.

MLP: implemented in training.common.policies (re-exported here for existing imports).
ACT (optional, deferred): action chunks with temporal ensembling; outside the current MLP work.
"""
from core.interfaces import Action, Observation, Policy
from training.common.policies import MLPPolicy  # noqa: F401


class ACTPolicy(Policy):
    name = "act"

    @classmethod
    def load(cls, ckpt_path: str) -> "ACTPolicy":
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError
