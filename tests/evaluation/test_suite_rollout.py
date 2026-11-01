"""The new protocol must distinguish continuous hold, terminal precision and runtime failure."""
from types import SimpleNamespace
import numpy as np
import pytest

from core.interfaces import Action, Observation
from evaluation.suite_rollout import score_errors, rollout


def score(values, hold=3, failure=None):
    return score_errors(values, 10, hold, failure)


def test_success_from_zero_includes_full_hold_duration():
    r = score([.005] * 31)
    assert r['success'] and r['first_arrival_s'] == 0
    assert r['completion_time_s'] == 3 and r['longest_hold_s'] == 3
    assert not score([.005] * 30)['success']


def test_exact_threshold_resets_streak_and_recovery_can_succeed():
    errors = [.005] * 51
    errors[15] = .01
    r = score(errors)
    assert r['completion_time_s'] == 4.6
    assert not score(errors[:46])['success']


def test_t5_requires_twenty_seconds():
    assert not score([.005] * 200, 20)['success']
    assert score([.005] * 201, 20)['completion_time_s'] == 20


def test_later_drift_changes_grade_without_erasing_success():
    r = score([.005] * 31 + [.03] * 20)
    assert r['success'] and r['grade'] == 'D' and r['completion_time_s'] == 3


@pytest.mark.parametrize('error,grade',[(0,'A'),(.0049,'A'),(.005,'B'),(.01,'C'),(.02,'D'),(.05,'E')])
def test_grade_boundaries_are_strict_and_independent_of_reach(error, grade):
    r = score([error] * 41)
    assert r['grade'] == grade
    assert r['success'] == (error < .01)


def test_terminal_window_is_inclusive():
    errors = [.001] * 41
    errors[30] = .025
    assert score(errors)['grade'] == 'D'
    errors[30] = .001
    errors[29] = .025
    assert score(errors)['grade'] == 'A'


def test_exception_overrides_previously_completed_task():
    r = score([.001] * 41, failure='RuntimeError: simulator failed')
    assert r['grade'] == 'F' and not r['success'] and r['completion_time_s'] is None
    assert score([.001] * 30 + [np.nan])['grade'] == 'F'


class ScriptedEnv:
    def __init__(self, bad_at=None):
        self.k = 0
        self.bad_at = bad_at
        self.closed = False
        self.actions = []
    def reset(self, scenario):
        self.scenario = scenario
        return self.obs()
    def obs(self):
        return Observation(np.zeros(3), 0, np.zeros(7), np.zeros(7), self.scenario.target_ee_pos)
    def ee_pos(self):
        return np.zeros(3)
    def step(self, action):
        self.actions.append(action)
        self.k += 1
        if self.k == self.bad_at:
            raise RuntimeError('physics failure')
        return self.obs(), {'ee_err': 0, 'ee_pos': self.ee_pos(), 'done': True}
    def close(self):
        self.closed = True


class ConstantPolicy:
    def reset(self): pass
    def act(self, obs):
        assert isinstance(obs, Observation)
        return Action(np.full(7, 50.), np.full(4, -9.))


def test_rollout_runs_to_cap_ignoring_done_and_clips_actions():
    env = ScriptedEnv()
    scene = SimpleNamespace(target_ee_pos=np.zeros(3), scenario_id='development')
    r = rollout(ConstantPolicy(), scene, 5, 3, {'control_hz':10,'torque_limit':30,'wheel_limit':6}, lambda:env)
    assert r['steps'] == 50 and r['success'] and r['completion_time_s'] == 3
    assert env.closed and np.all(env.actions[0].torque_arm == 30)
    assert np.all(env.actions[0].wheel_cmds == -6)
    assert r['final_ee_pos_W'] == [0,0,0]


def test_rollout_failure_after_success_is_f_and_closes_environment():
    env = ScriptedEnv(bad_at=40)
    scene = SimpleNamespace(target_ee_pos=np.zeros(3), scenario_id='development')
    r = rollout(ConstantPolicy(), scene, 5, 3, {'control_hz':10,'torque_limit':30,'wheel_limit':6}, lambda:env)
    assert not r['success'] and r['grade'] == 'F' and env.closed
    assert r['failure_type'] == 'RuntimeError'


def test_simulator_construction_failure_is_infrastructure_not_an_f_rollout():
    def broken_factory():raise FileNotFoundError('robot URDF unavailable')
    scene=SimpleNamespace(target_ee_pos=np.zeros(3),scenario_id='development')
    with pytest.raises(FileNotFoundError):
        rollout(ConstantPolicy(),scene,5,3,{'control_hz':10,'torque_limit':30,'wheel_limit':6},broken_factory)
