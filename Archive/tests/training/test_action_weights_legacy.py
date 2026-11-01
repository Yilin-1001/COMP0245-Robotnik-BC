"""Archived from tests/training/test_action_weights.py: these tests need the archived experiments/run.py (2026-10-30)
and training/jacobian/train.py (2026-11-01)."""
import json

from conftest import TASK_DIR
from training.common.config import resolve_config


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


def test_old_calibration_is_refused_for_new_weights_but_accepted_for_legacy_config(tmp_path):
    from training.jacobian.train import apply_calibration
    cfg = resolve_config(TASK_DIR / 'experiments/configs/jacobian.yaml', 'A', {})
    record = {'teacher': 'A', 'architecture': 'm2', 'normaliser_sha256': 'x', 'lambda_j': .25}
    path = tmp_path / 'legacy.json'
    path.write_text(json.dumps(record))
    cfg['loss']['jacobian']['calibration'] = str(path)
    with pytest.raises(ValueError, match='weight'):
        apply_calibration(cfg, 'x')
    cfg['loss'].pop('arm_weight', None)
    cfg['loss'].pop('wheel_weight', None)
    assert apply_calibration(cfg, 'x')['loss']['jacobian']['lambda_j'] == .25
