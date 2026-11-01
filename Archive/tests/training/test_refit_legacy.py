"""Archived on 2026-11-01 from tests/training/test_refit.py: these tests need the archived
training/jacobian/train.py (apply_calibration) and the clean-pipeline prepare_run (engine_legacy.py)."""
import builtins

import pytest

from conftest import TASK_DIR
from test_engine import make_config


def test_refit_keeps_frozen_lambda_with_new_scales(tmp_path):
    import json
    from training.common.config import resolve_config
    from training.jacobian.train import apply_calibration
    c = resolve_config(TASK_DIR / "experiments/configs/jacobian.yaml", "A", {"identity.fit_stage": "refit"})
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"lambda_j": 0.7, "teacher": "A", "architecture": "m2", "normaliser_sha256": "old",
                               "arm_weight": 0.8, "wheel_weight": 0.2}))
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
