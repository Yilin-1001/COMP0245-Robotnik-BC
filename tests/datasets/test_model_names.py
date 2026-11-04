"""Report nomenclature, CLI aliases and legacy reproducibility must agree."""
import pytest

from conftest import TASK_DIR
from core.model_names import MODEL_NAMES, canonical_id, legacy_id, report_label
from training.datasets.train import resolve, run_name


def test_names_match_the_report_and_gate_input_contract():
    report = (TASK_DIR / 'report/README.md').read_text()
    for index in range(1, 9):
        row = f'R{index}'
        name, canonical, _ = MODEL_NAMES[row]
        assert f'{row} {name}' in report
        variant = canonical + ('-r3' if index >= 5 else '')
        config = resolve(variant, {})
        assert config['model']['input_dim'] == (22 if index % 2 == 0 else 21)
        assert ('Gate' in name) == (index % 2 == 0)
        assert 'DART' not in canonical
    assert MODEL_NAMES['R9'][0] == 'DART + Gate'


@pytest.mark.parametrize('variant', ['R1-baseline', 'R2-gate', 'R3-dart', 'R4-dart-gate',
    'R5-baseline-dagger-r3', 'R6-gate-dagger-r3', 'R7-dart-dagger-r3', 'R8-dart-gate-dagger-r3'])
def test_aliases_preserve_data_recipe_round_seed_and_original_run_names(variant):
    overrides = {'identity.version': 'v2', 'identity.date': '2026-11-04', 'identity.seed': 2}
    original = resolve(variant, overrides)
    renamed = resolve(canonical_id(variant), overrides)
    for key in ('data', 'model', 'loss', 'training', 'identity', 'template_sha256', 'layers'):
        assert original[key] == renamed[key]
    old_run = run_name(original)
    new_run = run_name(renamed)
    assert new_run == canonical_id(old_run)
    assert legacy_id(new_run) == old_run
    assert canonical_id(new_run) == new_run


def test_unknown_models_are_not_relabelled_and_rounds_stay_visible():
    assert canonical_id('unrelated-model') == 'unrelated-model'
    assert report_label('teacher A') == 'teacher A'
    assert report_label('R8-dart-gate-dagger-r3') == 'R8 f-DAgger + Gate (r3)'
    assert report_label('R9-dart-full-a6-r2') == 'R9 DART + Gate (a6-r2)'
    assert canonical_id('R9-dart-full-a6-r3') == 'R9-DART-Gate-a6-r3'
    assert legacy_id('R9-DART-Gate-a6-r3') == 'R9-dart-full-a6-r3'
