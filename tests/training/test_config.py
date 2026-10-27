from pathlib import Path

import pytest
import yaml

from conftest import TASK_DIR
from training.common.config import (create_run_dir, finalise_identity, format_lr, merge, parse_overrides,
                                    resolve_config, validate_config)

CONFIGS = TASK_DIR / "experiments" / "configs"
BASELINE = CONFIGS / "baseline.yaml"
JACOBIAN = CONFIGS / "jacobian.yaml"
INTERFACE = {"action_order": "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]", "torque_limit": 30.0, "wheel_limit": 6.0,
             "control_hz": 1000}


def provenance(**extra):
    p = {"source_sha256": {"train-0000": "ab" * 32}, "split_sha256": "cd" * 32, "feature_sha256": "ef" * 32,
         "normaliser_sha256": "12" * 32, "code_digest": "34" * 32, "dataset_kind": "clean_subset",
         "interface": dict(INTERFACE)}
    p.update(extra)
    return p


def ready(config, **training):
    config["training"].update({"lr": 3e-4, "updates": 4, **training})
    config["loss"]["action"] = "huber"
    return config


def test_layers_apply_in_documented_order():
    c = resolve_config(BASELINE, "A", {})
    assert c["identity"]["regime"] == "clean" and c["identity"]["objective"] == "bc"
    assert c["identity"]["teacher"] == "A" and c["model"]["input_dim"] == 21
    assert c["loss"]["jacobian"] is None
    b = resolve_config(JACOBIAN, "B", {"training.lr": 1e-3})
    assert b["identity"]["objective"] == "bcj" and b["model"]["input_dim"] == 26
    assert b["loss"]["jacobian"]["calibration_ratio"] == 0.1
    assert b["training"]["lr"] == 1e-3
    assert c["layers"] == ["common.yaml", "baseline.yaml", "objectives/bc.yaml", "teachers/A.yaml", "cli"]


def test_lists_are_replaced_not_concatenated():
    assert merge({"a": [1, 2], "b": {"c": 1}}, {"a": [3], "b": {"d": 2}}) == {"a": [3], "b": {"c": 1, "d": 2}}
    c = resolve_config(BASELINE, "A", parse_overrides(["training.lr_grid=[1.0e-3]"]))
    assert c["training"]["lr_grid"] == [1e-3]


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        resolve_config(BASELINE, "A", {"training.learning_rate": 1e-3})


def test_baseline_cannot_be_changed_to_bcj():
    with pytest.raises(ValueError):
        resolve_config(BASELINE,
                       "A", {"identity.objective": "bcj"})


def test_clean_entry_points_reject_dart_provenance():
    with pytest.raises(ValueError):
        resolve_config(BASELINE, "A", {"identity.regime": "dart"})
    c = ready(resolve_config(BASELINE, "A", {}))
    with pytest.raises(ValueError, match="clean"):
        finalise_identity(c, provenance(dataset_kind="dart"))


def test_wrong_input_dimension_is_rejected():
    with pytest.raises(ValueError):
        resolve_config(BASELINE, "A", {"model.input_dim": 26})


def test_null_training_fields_rejected_only_when_training():
    c = resolve_config(BASELINE, "A", {})
    validate_config(c)  # preparation: budget/LR/loss may still be open
    with pytest.raises(ValueError, match="updates"):
        validate_config(ready(c, updates=None), ready_to_train=True)


def test_wheel_limit_mismatch_with_manifest_is_rejected():
    c = ready(resolve_config(BASELINE, "A", {}))
    bad = provenance(interface={**INTERFACE, "wheel_limit": 3.0})
    with pytest.raises(ValueError, match="wheel_limit"):
        finalise_identity(c, bad)


def test_bcj_requires_calibrated_lambda_before_training():
    c = ready(resolve_config(JACOBIAN, "A", {}))
    with pytest.raises(ValueError, match="lambda_j"):
        validate_config(c, ready_to_train=True)


def test_distinct_scientific_choices_get_distinct_ids():
    base = finalise_identity(ready(resolve_config(BASELINE, "A", {})), provenance())
    other_ids = [
        finalise_identity(ready(resolve_config(BASELINE, "A", {"identity.cohort": "clean223"})), provenance()),
        finalise_identity(ready(resolve_config(BASELINE, "A", {"identity.fit_stage": "refit"})), provenance()),
    ]
    jc = ready(resolve_config(JACOBIAN, "A", {}))
    jc["loss"]["jacobian"]["lambda_j"] = 0.5
    other_ids.append(finalise_identity(jc, provenance(calibration_sha256="56" * 32)))
    ids = {base["identity"]["run_id"]} | {o["identity"]["run_id"] for o in other_ids}
    assert len(ids) == 4
    assert base["identity"]["run_id"].startswith("clean-A-bc-clean253-m2-lr3e-4-s0-selection-")
    assert len(base["identity"]["config_id"]) == 12


def test_operational_fields_do_not_change_identity():
    a = finalise_identity(ready(resolve_config(BASELINE, "A", {})), provenance())
    b = finalise_identity(ready(resolve_config(BASELINE, "A", {"training.device": "cpu",
                                                                "outputs.checkpoint_root": "/elsewhere"})), provenance())
    assert a["identity"]["run_id"] == b["identity"]["run_id"]


def test_existing_run_dir_needs_compatible_resume(tmp_path):
    c = finalise_identity(ready(resolve_config(BASELINE, "A", {"outputs.checkpoint_root": str(tmp_path)})), provenance())
    run_dir = create_run_dir(c)
    assert run_dir == tmp_path / "clean" / "A" / "bc" / c["identity"]["run_id"]
    assert yaml.safe_load((run_dir / "config.yaml").read_text())["identity"]["config_hash"] == c["identity"]["config_hash"]
    with pytest.raises(FileExistsError):
        create_run_dir(c)
    with pytest.raises(FileNotFoundError, match="latest.pt"):
        create_run_dir(c, resume=True)
    (run_dir / "latest.pt").write_bytes(b"x")
    assert create_run_dir(c, resume=True) == run_dir
    clash = dict(c, identity=dict(c["identity"], config_hash="0" * 64))
    with pytest.raises(ValueError, match="collision"):
        create_run_dir(clash, resume=True)


def test_lr_format_has_explicit_exponent_sign():
    assert format_lr(3e-4) == "3e-4" and format_lr(1e-3) == "1e-3" and format_lr(2.5e-4) == "2.5e-4"
