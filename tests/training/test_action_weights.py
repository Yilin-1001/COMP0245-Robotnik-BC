"""Weighted objectives must affect gradients, identities and calibration, not just logs."""
import json

import pytest
import torch

from conftest import TASK_DIR
from training.baseline.loss import ActionObjective
from training.common.config import finalise_identity, resolve_config, validate_config
from training.jacobian.loss import calibrate_lambda
from training.jacobian.train import apply_calibration, objective_from
from experiments.run import _action_objective
from test_config import provenance, ready


@pytest.mark.parametrize('kind, expected', [('mse', 1.6), ('huber', 0.7)])
def test_weighted_action_loss_and_unweighted_group_metrics(kind, expected):
    pred = torch.tensor([[1.] * 7 + [2.] * 2], requires_grad=True)
    result = ActionObjective(kind, arm_weight=.8, wheel_weight=.2)(
        torch.nn.Identity(), pred, torch.zeros_like(pred), None, None)
    assert result.total.item() == pytest.approx(expected)
    assert result.metrics['arm_loss'] == pytest.approx(1. if kind == 'mse' else .5)
    assert result.metrics['wheel_loss'] == pytest.approx(4. if kind == 'mse' else 1.5)
    result.total.backward()
    assert pred.grad[0, 0].item() == pytest.approx(1.6 / 7 if kind == 'mse' else .8 / 7)
    assert pred.grad[0, 7].item() == pytest.approx(.4 if kind == 'mse' else .1)


@pytest.mark.parametrize('workflow', ['baseline.yaml', 'jacobian.yaml'])
def test_configuration_reaches_both_objective_factories(workflow):
    cfg = resolve_config(TASK_DIR / 'experiments/configs' / workflow, 'A', {})
    cfg['loss']['action'] = 'mse'
    assert cfg['loss'].get('arm_weight') == .8 and cfg['loss'].get('wheel_weight') == .2
    if workflow == 'jacobian.yaml':
        cfg['loss']['jacobian']['lambda_j'] = 0.
        objective = objective_from(cfg)
    else:
        objective = _action_objective(cfg)
    pred = torch.tensor([[1.] * 7 + [0.] * 2])
    result = objective(torch.nn.Identity(), pred, torch.zeros_like(pred), None, None)
    assert result.total.item() == pytest.approx(.8)


def test_weight_override_changes_scientific_run_identity():
    path = TASK_DIR / 'experiments/configs/baseline.yaml'
    old = ready(resolve_config(path, 'A', {'loss.arm_weight': .5, 'loss.wheel_weight': .5}))
    new = ready(resolve_config(path, 'A', {'loss.arm_weight': .8, 'loss.wheel_weight': .2}))
    assert finalise_identity(old, provenance())['identity']['run_id'] != \
        finalise_identity(new, provenance())['identity']['run_id']


@pytest.mark.parametrize('arm,wheel', [(-.1, 1.1), (float('nan'), .2), (True, 0.), (.8, .3), (0., 0.), ('bad', .2)])
def test_invalid_weights_are_rejected(arm, wheel):
    cfg = resolve_config(TASK_DIR / 'experiments/configs/baseline.yaml', 'A', {})
    cfg['loss'].update(arm_weight=arm, wheel_weight=wheel)
    with pytest.raises(ValueError, match='weight'):
        validate_config(cfg)


def test_calibration_uses_the_weighted_action_loss(unit_normaliser):
    model = torch.nn.Linear(21, 9)
    with torch.no_grad():
        model.weight.zero_()
        model.weight[:7, 7:14] = -20 * torch.eye(7)
        model.bias[:7] = 1.
        model.bias[7:] = 2.
    record = calibrate_lambda(model, torch.zeros(4, 21), torch.zeros(4, 9), unit_normaliser,
        {'kind': 'mse', 'huber_delta': 1., 'epsilon_tau': 1e-4, 'kd': 30., 'torque_limit': 30.,
         'calibration_ratio': .1, 'arm_weight': .8, 'wheel_weight': .2})
    assert record['action_loss'] == pytest.approx(1.6)
    assert record['lambda_j'] == pytest.approx(.0112)
    assert record['arm_weight'] == .8 and record['wheel_weight'] == .2


def test_old_calibration_is_refused_for_new_weights_but_accepted_for_legacy_config(tmp_path):
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
