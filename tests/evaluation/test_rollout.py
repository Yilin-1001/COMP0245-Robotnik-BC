import numpy as np
import pytest

from core.interfaces import Action, Observation, Policy
from evaluation.evaluate import evaluate_policy, rollout
from evaluation.test_sets import get_split

CONFIG = {"evaluation": {"episode_cap_s": 40.0, "min_reach_time_s": 8.0, "post_reach_s": 3.0, "hold_window_s": 1.0,
                         "reach_tolerance_m": 0.01, "workers": 0},
          "interface": {"torque_limit": 30.0, "wheel_limit": 6.0, "control_hz": 1000}}


class Scenario:
    def __init__(self, sid="s0"):
        self.scenario_id, self.target_ee_pos, self.seed = sid, np.array([1.0, 0.0, 0.5]), 1


class FakeEnv:
    """Scripted true EE error err_fn(k) for observation k; info['done'] mimics RobotEnv's tolerance flag."""
    def __init__(self, err_fn):
        self.err_fn, self.closed, self.actions, self.resets = err_fn, False, [], []

    def _obs(self):
        return Observation(np.zeros(3), 0.0, np.zeros(7), np.zeros(7), self.scenario.target_ee_pos.copy())

    def reset(self, scenario):
        self.scenario, self.k, self.t = scenario, 0, 0.0
        self.resets.append(scenario.scenario_id)
        return self._obs()

    def ee_pos(self):
        return self.scenario.target_ee_pos - np.array([self.err_fn(self.k), 0.0, 0.0])

    def step(self, action):
        self.actions.append(action)
        self.k += 1
        self.t = self.k * 0.001
        err = self.err_fn(self.k)
        return self._obs(), {"ee_err": err, "done": err < 0.01, "ee_pos": self.ee_pos(), "t": self.t}

    def close(self):
        self.closed = True


class ConstPolicy(Policy):
    def __init__(self, torque=1.0, wheel=0.5, fail_at=None):
        self.torque, self.wheel, self.fail_at, self.seen, self.resets = torque, wheel, fail_at, [], 0

    def reset(self):
        self.resets += 1

    def act(self, obs):
        self.seen.append(type(obs))
        if self.fail_at is not None and len(self.seen) > self.fail_at:
            raise RuntimeError("policy blew up")
        return Action(np.full(7, self.torque), np.full(4, self.wheel))


def run(err_fn, policy=None):
    env = FakeEnv(err_fn)
    out = rollout(policy or ConstPolicy(), Scenario(), CONFIG, lambda: env)
    return out, env


def test_done_flag_before_8s_does_not_terminate():
    out, env = run(lambda k: 0.0)  # within tolerance from the start, info['done'] always True
    assert out["first_reach_step"] == 8001 and np.isclose(out["t_first_reach"], 8.001)
    assert out["steps"] == 8001 + 3000 and out["post_reach_window_complete"] and out["sustained_hold_pass"]
    assert out["first_arrival_t"] == 0.0


def test_reach_after_8s_records_exactly_3s_more():
    out, _ = run(lambda k: 0.0 if k >= 9000 else 0.05)
    assert out["first_reach_step"] == 9000 and out["steps"] == 12000


def test_sustained_hold_uses_final_inclusive_second():
    end = 9000 + 3000
    bad, _ = run(lambda k: 0.02 if k == end - 1000 else (0.0 if k >= 9000 else 0.05))
    assert bad["post_reach_window_complete"] and not bad["sustained_hold_pass"]
    good, _ = run(lambda k: 0.02 if k == end - 1001 else (0.0 if k >= 9000 else 0.05))
    assert good["sustained_hold_pass"]


def test_episode_caps_at_40s():
    out, _ = run(lambda k: 0.05)
    assert out["steps"] == 40000 and not out["ever_reached"] and out["termination"] == "time_cap_no_task_reach"


def test_policy_sees_observations_only_and_limits_apply():
    policy = ConstPolicy(torque=50.0, wheel=-9.0)
    out, env = run(lambda k: 0.0 if k >= 8500 else 0.05, policy)
    assert set(policy.seen) == {Observation}
    assert np.all(env.actions[0].torque_arm == 30.0) and np.all(env.actions[0].wheel_cmds == -6.0)
    assert out["saturation_frac"] == 1.0


def test_robot_env_applies_wheels_then_arm():
    import inspect
    from core import env as core_env
    assert core_env.CONTROL_MODES[:4] == ["velocity"] * 4 and core_env.CONTROL_MODES[4:11] == ["torque"] * 7
    assert "np.concatenate((action.wheel_cmds, action.torque_arm, np.zeros(6)))" in inspect.getsource(core_env.RobotEnv.step)


def test_reset_per_scenario_and_close_on_exception():
    envs = []
    def factory():
        envs.append(FakeEnv(lambda k: 0.0 if k >= 8100 else 0.05))
        return envs[-1]
    policy = ConstPolicy()
    rows = evaluate_policy(lambda: policy, [Scenario("a"), Scenario("b")], CONFIG, env_factory=factory, workers=0)
    assert [r["scenario_id"] for r in rows] == ["a", "b"] and policy.resets == 2
    assert [e.resets for e in envs] == [["a"], ["b"]] and all(e.closed for e in envs)
    failing = FakeEnv(lambda k: 0.05)
    out = rollout(ConstPolicy(fail_at=5), Scenario(), CONFIG, lambda: failing)
    assert failing.closed and out["failure"].startswith("RuntimeError") and not out["sustained_hold_pass"]


def test_test_split_is_sealed_without_explicit_permission():
    from conftest import TASK_DIR
    manifest = TASK_DIR / "experiments" / "splits" / "clean253.json"
    with pytest.raises(PermissionError):
        get_split("test", manifest, "A")
    assert len(get_split("screening", manifest, "A")) == 8
    val = get_split("validation", manifest, "B")
    assert len(val) == 32 and val[0].scenario_id == "train-0143" and val[0].seed == 1000143
