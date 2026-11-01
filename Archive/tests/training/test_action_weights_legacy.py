"""Archived from tests/training/test_action_weights.py: needs the archived experiments/run.py."""


def test_frozen_legacy_experiment_keeps_its_original_action_weights():
    from experiments.run import build_jobs
    protocol = {'protocol_id': 'legacy', 'decisions': {
        'budget': {'updates': 42000, 'pilot_lr': .0003}, 'loss': {'action': 'huber'}}}
    jobs = build_jobs('screen', protocol)
    assert all(job['overrides'].get('loss.arm_weight') == .5 and
               job['overrides'].get('loss.wheel_weight') == .5 for job in jobs)
    protocol['decisions']['loss'].update(arm_weight=.8, wheel_weight=.2)
    assert all(job['overrides'].get('loss.arm_weight') == .8 and
               job['overrides'].get('loss.wheel_weight') == .2 for job in build_jobs('screen', protocol))
