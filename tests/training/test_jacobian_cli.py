import json
import subprocess
import sys

import pytest
import torch

from conftest import TASK_DIR
from test_engine import make_config, tiny_cache
from training.baseline.loss import ActionObjective
from training.common.engine import fit, init_model
from training.common.normalisation import fit_normaliser
from training.jacobian.loss import JacobianObjective
from training.jacobian.train import apply_calibration, main

FLOORS = {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05}


def test_help_names_the_jacobian_objective(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "damping-Jacobian" in capsys.readouterr().out


def test_baseline_template_is_refused():
    with pytest.raises(ValueError, match="jacobian"):
        main(["--teacher", "A", "--dry-run", "--config", str(TASK_DIR / "experiments/configs/baseline.yaml")])


def test_training_requires_frozen_calibration():
    with pytest.raises(ValueError, match="calibration"):
        main(["--teacher", "A", "--dry-run", "--set", "training.updates=10", "--set", "training.lr=1.0e-3",
              "--set", "loss.action=huber"])


def test_calibration_must_match_teacher_architecture_and_scales(tmp_path):
    from training.common.config import resolve_config
    c = resolve_config(TASK_DIR / "experiments/configs/jacobian.yaml", "A", {})
    record = {"lambda_j": 0.25, "teacher": "B", "architecture": "m2", "normaliser_sha256": "x", "reference_seed": 0}
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(record))
    c["loss"]["jacobian"]["calibration"] = str(path)
    with pytest.raises(ValueError, match="teacher"):
        apply_calibration(c, normaliser_sha256="x")
    path.write_text(json.dumps(dict(record, teacher="A")))
    with pytest.raises(ValueError, match="normaliser"):
        apply_calibration(c, normaliser_sha256="y")
    out = apply_calibration(c, normaliser_sha256="x")
    assert out["loss"]["jacobian"]["lambda_j"] == 0.25 and out["loss"]["jacobian"]["calibration_record"]["sha256"]


def test_launcher_imports_no_collection_workflows():
    code = ("import sys, training.jacobian.train; "
            "print([m for m in sys.modules if m.startswith(('dart', 'dagger', 'data_collection')) or m == 'pybullet'])")
    out = subprocess.run([sys.executable, "-c", code], cwd=TASK_DIR, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_paired_runs_share_init_and_action_batches(tmp_path, monkeypatch):
    import training.common.engine as engine
    train, val = tiny_cache([40, 25, 61], 0), tiny_cache([30, 12], 1)
    n = fit_normaliser(train, FLOORS)
    seen = {}
    real = engine.sample_indices

    def spy(name):
        def f(*a, **k):
            idx = real(*a, **k)
            seen.setdefault(name, []).append(idx.clone())
            return idx
        return f
    bc_cfg = make_config(tmp_path / "bc", n)
    bcj_cfg = make_config(tmp_path / "bcj", n)
    m_bc, m_bcj = init_model(bc_cfg), init_model(bcj_cfg)
    assert all(torch.equal(a, b) for a, b in zip(m_bc.state_dict().values(), m_bcj.state_dict().values()))
    monkeypatch.setattr(engine, "sample_indices", spy("bc"))
    fit(bc_cfg, train, val, m_bc, n, ActionObjective("mse"))
    monkeypatch.setattr(engine, "sample_indices", spy("bcj"))
    fit(bcj_cfg, train, val, m_bcj, n, JacobianObjective("mse", lambda_j=0.5, epsilon_tau=1e-4, subbatch_size=4))
    assert len(seen["bc"]) == len(seen["bcj"]) == 4
    assert all(torch.equal(a, b) for a, b in zip(seen["bc"], seen["bcj"]))


@pytest.mark.realdata
def test_real_calibration_then_dry_run(tmp_path, capsys):
    common = ["--teacher", "A", "--set", "loss.action=huber", "--set", f"outputs.checkpoint_root={tmp_path}"]
    assert main(common + ["--calibrate"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["lambda_j"] > 0 and record["reference_seed"] == 0 and record["architecture"] == "m2"
    assert abs(record["lambda_j"] * record["jacobian_loss"] - 0.1 * record["action_loss"]) < 1e-9
    assert main(common + ["--dry-run", "--set", f"loss.jacobian.calibration={record['calibration']}",
                          "--set", "training.lr=1.0e-3", "--set", "training.updates=10"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"].startswith("clean-A-bcj-clean253-m2-lr1e-3-s0-selection-")
    assert out["lambda_j"] == record["lambda_j"]
