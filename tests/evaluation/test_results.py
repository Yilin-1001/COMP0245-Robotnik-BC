import csv

import pytest

from evaluation.evaluate import LEGACY_COLUMNS, RESULT_COLUMNS, append_results


@pytest.fixture
def complete_row():
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(run_id="clean-A-bc-clean253-m2-lr3e-4-s0-selection-abcdef123456", regime="clean", teacher="A",
               objective="bc", cohort="clean253", fit_stage="selection", seed=0, checkpoint_step=2000,
               evaluation_split="screening", evaluation_id="rollout-v1", n=8, sustained_hold_success=0.5)
    return row


def test_duplicate_evaluation_is_rejected(tmp_path, complete_row):
    path = tmp_path / "results.csv"
    append_results(path, [complete_row])
    with pytest.raises(ValueError):
        append_results(path, [complete_row])


def test_same_run_other_checkpoint_is_a_new_row(tmp_path, complete_row):
    path = tmp_path / "results.csv"
    append_results(path, [complete_row, dict(complete_row, checkpoint_step=4000)])
    assert len(list(csv.DictReader(open(path)))) == 2


def test_header_only_legacy_csv_is_upgraded_without_rows(tmp_path, complete_row):
    path = tmp_path / "results.csv"
    path.write_text(",".join(LEGACY_COLUMNS) + "\n")
    append_results(path, [])
    assert next(csv.reader(open(path))) == RESULT_COLUMNS
    assert list(csv.DictReader(open(path))) == []


def test_nonempty_incompatible_csv_requires_migration(tmp_path, complete_row):
    path = tmp_path / "results.csv"
    text = ",".join(LEGACY_COLUMNS) + "\nold,expert,,val,1,1,1,1,1,1,1\n"
    path.write_text(text)
    with pytest.raises(ValueError, match="migration"):
        append_results(path, [complete_row])
    assert path.read_text() == text


def test_missing_key_fields_are_rejected(tmp_path, complete_row):
    with pytest.raises(ValueError):
        append_results(tmp_path / "results.csv", [dict(complete_row, evaluation_split="")])
