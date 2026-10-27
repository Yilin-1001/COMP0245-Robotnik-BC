import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from core.interfaces import Action, Observation, Policy
from data_collection.scenarios import Scenario


def module():
    assert importlib.util.find_spec("data_collection.dagger_rollout"), "DAgger collector is not implemented"
    from data_collection import dagger_rollout
    return dagger_rollout


class Controller(Policy):
    def __init__(self, offset):
        self.offset = offset
        self.expert = self

    def base_command(self, obs):
        return np.zeros(4), bool(obs.q_mes[0] >= 10)

    def act(self, obs):
        value = obs.q_mes[0] + self.offset
        self.last_raw = np.r_[np.full(7, value), 8., 8.]
        return Action(np.full(7, value), np.full(4, 8.))


class ClockEnv:
    def __init__(self, fail_after=None):
        self.fail_after = fail_after

    def reset(self, scenario):
        self.scenario, self.t, self.k = scenario, 0., 0
        return self.obs()

    def obs(self):
        q = np.full(7, float(self.k))
        if self.fail_after == self.k:
            q[:] = np.nan
        return Observation(np.zeros(3), 0., q, np.zeros(7), self.scenario.target_ee_pos)

    def ee_pos(self):
        return np.zeros(3)

    def step(self, action):
        assert np.max(np.abs(action.torque_arm)) <= 30
        assert np.max(np.abs(action.wheel_cmds)) <= 6
        self.k += 1
        self.t = self.k * .001
        return self.obs(), dict(ee_err=1., t=self.t, done=False, ee_pos=np.zeros(3), pushing=False)

    def close(self):
        pass


def run(beta, env_factory=ClockEnv):
    cfg = {"interface": {"control_hz": 1000, "torque_limit": 30, "wheel_limit": 6},
           "evaluation": {"episode_cap_s": .023, "min_reach_time_s": 8, "post_reach_s": 3,
                          "hold_window_s": 1, "reach_tolerance_m": .01}}
    scene = Scenario(1, np.array([1., 0., 0.]), scenario_id="dagger-r1-0", split="dagger_train")
    return module().collect_episode(Controller(1), Controller(10), scene, cfg, beta, np.random.default_rng(0), env_factory)


@pytest.mark.parametrize("beta,expert", [(0., False), (1., True)])
def test_labels_are_current_expert_queries_even_when_student_executes(beta, expert):
    arrays, meta = run(beta)
    assert len(arrays["obs_time"]) == 24
    assert arrays["expert_action"].shape == (23, 11)
    np.testing.assert_array_equal(arrays["expert_action"][:, 0], np.arange(23) + 10)
    assert np.all(arrays["expert_action"][:, 7:] == 8)  # targets retain raw wheel output
    assert np.all(arrays["executed_action"][:, 7:] == 6)
    expected = np.minimum(np.arange(23) + (10 if expert else 1), 30)
    np.testing.assert_array_equal(arrays["executed_action"][:, 0], expected)
    assert np.all(arrays["expert_executes"] == expert)
    assert np.all(arrays["valid_supervision"])
    assert meta["outcome"]["steps"] == 23
    assert not meta["outcome"]["ever_reached"]


def test_block_holds_controller_identity_but_actions_recompute_each_millisecond():
    arrays, meta = run(.5)
    assert len(set(arrays["expert_executes"])) == 2
    for start, stop in [(0, 10), (10, 20), (20, 23)]:
        assert len(set(arrays["expert_executes"][start:stop])) == 1
    np.testing.assert_array_equal(np.diff(arrays["student_raw"][:, 0]), np.ones(22))
    assert meta["outcome"]["termination"] == "time_cap_no_task_reach"


def test_numerical_failure_keeps_valid_executed_prefix_and_terminal_observation():
    arrays, meta = run(0., lambda: ClockEnv(fail_after=3))
    assert arrays["expert_action"].shape == (3, 11)
    assert arrays["obs_q_mes"].shape == (4, 7)
    assert np.isnan(arrays["obs_q_mes"][-1]).all()
    assert arrays["valid_supervision"].all()
    assert meta["outcome"]["failure"]
    assert not meta["outcome"]["sustained_hold_pass"]


def test_invalid_unexecuted_query_is_diagnostic_not_an_extra_transition():
    class BadController(Controller):
        def act(self, obs):
            a = super().act(obs)
            a.torque_arm[:] = np.nan
            return a
    cfg = {"interface": {"control_hz": 1000, "torque_limit": 30, "wheel_limit": 6},
           "evaluation": {"episode_cap_s": .02, "min_reach_time_s": 8, "post_reach_s": 3,
                          "hold_window_s": 1, "reach_tolerance_m": .01}}
    scene = Scenario(1, np.ones(3))
    arrays, meta = module().collect_episode(BadController(1), Controller(10), scene, cfg, 1.,
                                           np.random.default_rng(0), ClockEnv)
    assert len(arrays["obs_time"]) == 1
    assert len(arrays["expert_action"]) == 0
    assert meta["unexecuted_query"]["step"] == 0
