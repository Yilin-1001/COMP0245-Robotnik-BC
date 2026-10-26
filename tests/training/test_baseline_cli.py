import json

import pytest

from conftest import TASK_DIR
from training.baseline.train import main
from training.common.config import resolve_config
from training.common.engine import prepare_run

READY = ["--set", "training.updates=10", "--set", "training.lr=1.0e-3", "--set", "loss.action=huber"]


def test_help_names_the_clean_baseline(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "action loss only" in capsys.readouterr().out


def test_objective_cannot_be_overridden():
    with pytest.raises(ValueError):
        main(["--teacher", "A", "--dry-run", "--set", "identity.objective=bcj"] + READY)


def test_unknown_override_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        main(["--teacher", "A", "--dry-run", "--set", "training.learning_rate=1"] + READY)


def test_jacobian_template_is_refused():
    with pytest.raises(ValueError, match="baseline"):
        main(["--teacher", "A", "--dry-run", "--config", str(TASK_DIR / "experiments/configs/jacobian.yaml")] + READY)


def test_dart_or_dagger_manifests_are_refused(fake_dataset):
    root = fake_dataset()
    manifest = json.loads((root / "manifest.json").read_text())
    for kind in ("DART noise collection", "DAgger round 1 aggregate"):
        (root / "manifest.json").write_text(json.dumps(dict(manifest, dataset_kind=kind)))
        config = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", "A", {})
        config["data"]["primary_root"] = str(root)
        with pytest.raises(ValueError, match="clean"):
            prepare_run(config)


def test_legacy_wrapper_points_to_new_entry_point(capsys):
    from training import train_mlp
    with pytest.raises(SystemExit) as e:
        train_mlp.main(["--data", "data/D0.npz"])
    assert e.value.code != 0
    assert "training.baseline.train" in capsys.readouterr().err


@pytest.mark.realdata
def test_dry_run_prepares_real_data_and_identity(capsys):
    assert main(["--teacher", "A", "--dry-run"] + READY) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"].startswith("clean-A-bc-clean253-m2-lr1e-3-s0-selection-")
    assert out["fitting_episodes"] == 221 and out["validation_episodes"] == 32
    assert out["input_dim"] == 21 and out["interface"]["wheel_limit"] == 6.0
