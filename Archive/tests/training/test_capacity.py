"""Capacity experiments keep episode weighting and fitting-only scenario selection explicit."""
import itertools
import json

import numpy as np
import pytest
import torch

from training.common.dataset import EpisodeCache


def test_probe_errors_give_each_episode_equal_weight(unit_normaliser):
    from experiments.capacity import probe_metrics
    # Two zero-error rows in episode A; one arm-error=2 row in B. Row weighting would give MSE=4/3.
    x, y = np.zeros((3, 21)), np.zeros((3, 9))
    y[2, :7] = 2
    cache = EpisodeCache(x, y, np.array([0, 2, 3]), ['a', 'b'], {'phase': np.array([0, 0, 1])})
    model = torch.nn.Linear(21, 9)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    model.train()
    result = probe_metrics(model, cache, unit_normaliser, .8, .2)
    assert result['overall']['arm_rmse_nm'] == pytest.approx(2**.5)
    assert result['overall']['wheel_rmse_rad_s'] == 0
    assert result['overall']['arm_huber'] == pytest.approx(.75)
    assert result['overall']['action_huber'] == pytest.approx(.6)
    assert result['phase']['navigation']['arm_rmse_nm'] == 0
    assert result['phase']['approach']['arm_rmse_nm'] == 2
    assert result['phase']['hold'] is None
    assert result['sampled_rows'] == 3
    assert model.training


def test_probe_indices_include_endpoints_and_never_cross_episodes():
    from experiments.capacity import probe_indices
    cache = EpisodeCache(np.zeros((110, 21)), np.zeros((110, 9)), np.array([0, 100, 110]), ['a', 'b'])
    indices = probe_indices(cache)
    assert [len(x) for x in indices] == [64, 10]
    assert [indices[0][0], indices[0][-1], indices[1][0], indices[1][-1]] == [0, 99, 100, 109]
    assert len(np.unique(np.concatenate(indices))) == 74


def test_checkpoint_callback_records_actual_model_once(tmp_path, unit_normaliser):
    from experiments.capacity import probe_recorder
    cache = EpisodeCache(np.zeros((2, 21)), np.ones((2, 9)), np.array([0, 2]), ['a'])
    prep = {'train': cache, 'validation': cache, 'normaliser': unit_normaliser,
            'config': {'loss': {'arm_weight': .8, 'wheel_weight': .2}}}
    model = torch.nn.Linear(21, 9)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    path = tmp_path/'probe.jsonl'
    callback = probe_recorder(prep, path)
    # The engine's four arguments are update, .pt path, .npz path, model.
    callback(7000, tmp_path/'step_7000.pt', tmp_path/'step_7000.npz', model)
    callback(7000, tmp_path/'step_7000.pt', tmp_path/'step_7000.npz', model)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]['update'] == 7000
    assert records[0]['fitting']['overall']['action_huber'] == .5
    assert records[0]['validation']['overall']['arm_rmse_nm'] == 1


def test_fitting_diagnostic_uses_lexicographic_member_of_every_cell():
    from experiments.capacity import fitting_diagnostic_ids
    rows = []
    for i, (d, h, a) in enumerate(itertools.product(range(2), range(4), range(4))):
        for suffix in ('z', 'a'):
            rows.append({'scenario_id': f'{i:02d}-{suffix}', 'u_target_dist': (d+.5)/2,
                         'u_target_z': (h+.5)/4, 'u_rel_azimuth': (a+.5)/4})
    ids = [r['scenario_id'] for r in rows]
    selected = fitting_diagnostic_ids(rows, ids)
    assert selected == [f'{i:02d}-a' for i in range(32)]
    # A validation member with a smaller ID must never enter the diagnostic set.
    rows.append({**rows[0], 'scenario_id': '000-validation'})
    assert fitting_diagnostic_ids(rows, ids) == selected
    with pytest.raises(ValueError, match='empty fitting cell'):
        fitting_diagnostic_ids(rows, ids[2:])


@pytest.mark.parametrize('hidden', [[], [True], [0], [-1], [256.5]])
def test_invalid_capacity_architecture_is_rejected(hidden):
    from experiments.capacity import experiment_config
    with pytest.raises(ValueError, match='hidden'):
        experiment_config({'teacher': 'A', 'architecture': 'd08w0256', 'hidden': hidden})
