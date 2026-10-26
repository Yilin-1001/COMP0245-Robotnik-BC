"""Harness check (a): the frozen experts, through the student's action encoding and the shared evaluator.

The expert acts on raw observations. Its 11 commands are encoded to the 9 trained channels, standardised,
inverted and expanded back, exactly as a student output would be. Student features are built in parallel only to
check their dimension and finiteness; they are not used to reconstruct the expert's input.
"""
import numpy as np
import pytest

from conftest import TASK_DIR
from core.interfaces import Action, Policy
from evaluation.evaluate import MarkovianReference, StatefulReference, rollout, make_robot_env
from evaluation.test_sets import get_split
from training.common.config import resolve_config
from training.common.models import expand_wheels

pytestmark = pytest.mark.sim


class RoundTripPolicy(Policy):
    def __init__(self, expert, teacher, normaliser):
        self.expert, self.teacher, self.normaliser = expert, teacher, normaliser
        self.max_round_trip_err, self.feature_dims, self.nonfinite = 0.0, set(), 0
        self._k, self._fk, self._goal = 0, None, None

    def bind_env(self, env):
        if hasattr(self.expert, "bind_env"):
            self.expert.bind_env(env)

    def reset(self):
        self.expert.reset()
        self._k, self._goal = 0, None

    def act(self, obs):
        from core.features import DeployableFK, features_a, features_b, stateful_goal
        a = self.expert.act(obs)
        raw = np.r_[a.torque_arm, a.wheel_cmds]
        y9 = np.r_[raw[:7], raw[7], raw[8]]
        back = expand_wheels(self.normaliser.inverse_y(self.normaliser.transform_y(y9)))
        self.max_round_trip_err = max(self.max_round_trip_err, float(np.abs(back - raw).max()))
        if self._fk is None:
            self._fk = DeployableFK()
        args = (obs.base_pos[None], np.array([obs.base_bearing]), obs.q_mes[None], obs.qd_mes[None])
        if self.teacher == "A":
            x = features_a(*args, obs.target_ee_pos, self._fk)
        else:
            if self._goal is None:
                self._goal = stateful_goal(obs.target_ee_pos, {"goal_rule": "template-origin-standoff"})
            x = features_b(*args, np.array([self._k * 0.001]), obs.target_ee_pos, *self._goal, 25.0, self._fk)
        self._k += 1
        self.feature_dims.add(x.shape[1])
        self.nonfinite += int(not np.isfinite(x).all())
        return Action(back[:7], back[7:])


@pytest.mark.parametrize("teacher,dim", [("A", 21), ("B", 26)])
def test_frozen_teacher_through_student_action_path(teacher, dim):
    from conftest import make_normaliser
    config = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", teacher, {})
    rng = np.random.default_rng(0)
    normaliser = make_normaliser(y_scale=rng.uniform(0.5, 6.0, 9))
    normaliser.y_mean = rng.normal(0, 2, 9)
    expert = MarkovianReference(lead_time=0.50) if teacher == "A" else StatefulReference()
    if teacher == "A":
        assert expert.expert.params.lead_time == 0.50
    policy = RoundTripPolicy(expert, teacher, normaliser)
    scenario = get_split("screening", TASK_DIR / "experiments/splits/clean253.json", teacher)[0]
    out = rollout(policy, scenario, config, make_robot_env)
    assert out["failure"] is None
    assert policy.max_round_trip_err < 1e-9
    assert policy.feature_dims == {dim} and policy.nonfinite == 0
    assert out["post_reach_window_complete"] and out["sustained_hold_pass"], out
