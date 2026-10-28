"""Archived from tests/training/test_baseline_cli.py together with training/train_mlp.py (2026-10-28)."""
import pytest


def test_legacy_wrapper_points_to_new_entry_point(capsys):
    from training import train_mlp
    with pytest.raises(SystemExit) as e:
        train_mlp.main(["--data", "data/D0.npz"])
    assert e.value.code != 0
    assert "training.baseline.train" in capsys.readouterr().err
