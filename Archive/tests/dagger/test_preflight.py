"""Catch recovery probes switching on elapsed reset time instead of reach activation."""
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from core.interfaces import Action, Observation, Policy


def module():
    assert importlib.util.find_spec("data_collection.dagger_preflight"), "DAgger recovery preflight is not implemented"
    from data_collection import dagger_preflight
    return dagger_preflight


class Controller(Policy):
    def __init__(self, value):
        self.value = value
        self.expert = self

    def base_command(self, obs):
        return np.zeros(4), bool(obs.base_pos[0] >= 1)

    def act(self, obs):
        return Action(np.full(7, self.value + obs.q_mes[0]), np.zeros(4))


def observation(x=0, q=0):
    return Observation(np.array([x, 0., 0.]), 0., np.full(7, q), np.zeros(7), np.ones(3))


def test_recovery_waits_for_reaching_and_recomputes_current_expert_action():
    p = module().RecoveryPolicy(Controller(2), Controller(10), "reach", .25)
    env = SimpleNamespace(t=0., ee_pos=lambda: np.zeros(3))
    p.bind_env(env)
    p.reset()
    env.t = 4.
    assert p.act(observation()).torque_arm[0] == 10
    env.t = 5.
    assert p.act(observation(1, 1)).torque_arm[0] == 3
    env.t = 5.249
    assert p.act(observation(1, 2)).torque_arm[0] == 4
    env.t = 5.25
    assert p.act(observation(1, 3)).torque_arm[0] == 13
    assert p.handoff_s == 5.
    assert p.recovery_s == 5.25
    p.reset()
    env.t = 0.
    assert p.act(observation()).torque_arm[0] == 10
    assert p.handoff_s is None


def test_reset_probe_uses_student_from_first_observation():
    p = module().RecoveryPolicy(Controller(2), Controller(10), "reset", 1.)
    env = SimpleNamespace(t=0., ee_pos=lambda: np.zeros(3))
    p.bind_env(env)
    p.reset()
    assert p.act(observation()).torque_arm[0] == 2
    env.t = 1.
    assert p.act(observation(q=4)).torque_arm[0] == 14


def test_gate_does_not_treat_diagnostic_or_reach_only_success_as_recovery():
    m = module()
    good = lambda n: [{"sustained_hold_pass": i < n, "ever_reached": True, "failure": None} for i in range(8)]
    outcomes = {"expert": good(8), "reach_0.25s": good(6), "reach_1s": good(6), "reach_3s": good(0)}
    assert m.recovery_gate(outcomes)["passed"]
    outcomes["reach_1s"] = good(5)
    assert not m.recovery_gate(outcomes)["passed"]
    outcomes["reach_1s"] = good(6)
    outcomes["expert"] = good(7)
    assert not m.recovery_gate(outcomes)["passed"]
    with pytest.raises(ValueError):
        m.recovery_gate({"expert": good(8)})
