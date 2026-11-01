import numpy as np

from core.interfaces import Action, Observation, Policy
from evaluation.residual_noise import NoisyExpert


class Zero(Policy):
    def act(self, obs):
        return Action(np.zeros(7), np.zeros(4))


def test_noise_is_added_per_channel_and_wheels_stay_mirrored():
    stats = {"mean": [0.5] * 7 + [0.1, -0.2], "std": [0.0] * 9}
    p = NoisyExpert(Zero(), stats, seed=3)
    p.reset()
    a = p.act(Observation(np.zeros(3), 0.0, np.zeros(7), np.zeros(7), np.zeros(3)))
    assert np.allclose(a.torque_arm, 0.5) and np.allclose(a.wheel_cmds, [0.1, -0.2, -0.2, 0.1])


def test_noise_stream_is_reproducible_per_episode():
    stats = {"mean": [0.0] * 9, "std": [1.0] * 9}
    o = Observation(np.zeros(3), 0.0, np.zeros(7), np.zeros(7), np.zeros(3))
    p, q = NoisyExpert(Zero(), stats, seed=7), NoisyExpert(Zero(), stats, seed=7)
    p.reset(); q.reset()
    assert np.array_equal(p.act(o).torque_arm, q.act(o).torque_arm)
