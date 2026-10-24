"""Markovian version of the controller in robotnik_arm_control.py.

The template keeps q_des_arm and a navigation phase between steps, so it cannot
label an arbitrary state. This expert computes its action from the current
state alone, which DART labelling and DAgger relabelling both need.
"""
from core.interfaces import Action, Observation, Policy


class MarkovianExpert(Policy):
    name = "expert"

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError
