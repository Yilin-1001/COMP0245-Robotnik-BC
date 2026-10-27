import builtins

import pytest
import torch

from conftest import TASK_DIR
from test_engine import make_config, tiny_cache
from training.baseline.loss import ActionObjective
from training.common.engine import fit, init_model, select_checkpoint
from training.common.normalisation import fit_normaliser

FLOORS = {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05}


def refit_config(tmp_path, n):
    from training.common.config import finalise_identity, resolve_config
    c = make_config(tmp_path, n)
    c["identity"]["fit_stage"] = "refit"
    c = finalise_identity({k: v for k, v in c.items()}, c["provenance"])
    return c


@pytest.fixture
def refit_result(tmp_path):
    train = tiny_cache([40, 25, 61, 30], 0)
    n = fit_normaliser(train, FLOORS)
    c = refit_config(tmp_path, n)
    return fit(c, train, None, init_model(c), n, ActionObjective("mse"))


def test_refit_has_no_validation_selected_artifact(refit_result):
    assert (refit_result.run_dir / "final.pt").is_file()
    assert not (refit_result.run_dir / "best.pt").exists()


def test_refit_saves_final_at_fixed_update_with_matching_export(refit_result):
    payload = torch.load(refit_result.run_dir / "final.pt", weights_only=True)
    assert payload["step"] == 4 and refit_result.terminal_update == 4
    assert (refit_result.run_dir / "inference.npz").is_file()
    assert not list(refit_result.run_dir.glob("step_*.pt"))  # no intermediate candidates to select from
    with pytest.raises(ValueError, match="selection"):
        select_checkpoint(refit_result.run_dir, 4, {})


def test_refit_refuses_an_independent_validation_set(tmp_path):
    train, val = tiny_cache([40, 25], 0), tiny_cache([30], 1)
    n = fit_normaliser(train, FLOORS)
    c = refit_config(tmp_path, n)
    with pytest.raises(ValueError, match="refit"):
        fit(c, train, val, init_model(c), n, ActionObjective("mse"))


def test_refit_keeps_frozen_lambda_with_new_scales(tmp_path):
    import json
    from training.common.config import resolve_config
    from training.jacobian.train import apply_calibration
    c = resolve_config(TASK_DIR / "experiments/configs/jacobian.yaml", "A", {"identity.fit_stage": "refit"})
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"lambda_j": 0.7, "teacher": "A", "architecture": "m2", "normaliser_sha256": "old"}))
    c["loss"]["jacobian"]["calibration"] = str(path)
    out = apply_calibration(c, normaliser_sha256="refit-scales")
    assert out["loss"]["jacobian"]["lambda_j"] == 0.7


@pytest.mark.realdata
def test_real_refit_preparation_uses_all_training_episodes_and_no_test_arrays(monkeypatch):
    from training.common.config import resolve_config
    from training.common.engine import prepare_run
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    over = {"identity.fit_stage": "refit", "training.updates": 10, "training.lr": 1e-3, "loss.action": "huber"}
    refit = prepare_run(resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", "A", over))
    selection = prepare_run(resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", "A",
                                           dict(over, **{"identity.fit_stage": "selection"})))
    assert len(refit["train"].episode_ids) == 253 and refit["validation"] is None
    assert "train-0143" in refit["train"].episode_ids  # a former validation episode is now fitting data
    assert refit["normaliser"].sha256() != selection["normaliser"].sha256()
    assert refit["config"]["identity"]["run_id"] != selection["config"]["identity"]["run_id"]
    assert not any("/test/" in p for p in opened)
