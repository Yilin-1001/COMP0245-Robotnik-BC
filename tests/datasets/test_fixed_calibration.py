"""A bcj run can copy lambda_J from a source run instead of calibrating, and refuses a source that does not match it."""
import pytest
import yaml

from conftest import TASK_DIR


def _config(**overrides):
    from training.datasets.train import resolve
    return resolve("R4-dart-gate", overrides)


def _source(tmp_path, **changes):
    config = _config()
    source = {"variant": "R4-dart-gate", "identity": {"objective": "bcj"},
              "data": {"datasets": list(config["data"]["datasets"])},
              "loss": {"jacobian": {"lambda_j": 0.0744}},
              "provenance": {"calibration": {"lambda_j": 0.0744, "ratio": 0.1}, "calibration_sha256": "abc",
                             "normaliser_sha256": "n1"}}
    for dotted, value in changes.items():
        node = source
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(source))
    return path


def test_copies_lambda_and_source_hash(tmp_path):
    from training.datasets.train import copy_calibration
    record, sha = copy_calibration(_config(), "n1", _source(tmp_path))
    assert record["lambda_j"] == 0.0744 and sha == "abc"
    assert record["copied_from"].endswith("config.yaml") and len(record["source_config_sha256"]) == 64


def test_rejects_other_variant(tmp_path):
    from training.datasets.train import copy_calibration
    with pytest.raises(ValueError, match="variant"):
        copy_calibration(_config(), "n1", _source(tmp_path, variant="R8-dart-gate-dagger-r3"))


def test_rejects_other_datasets(tmp_path):
    from training.datasets.train import copy_calibration
    with pytest.raises(ValueError, match="datasets"):
        copy_calibration(_config(), "n1", _source(tmp_path, **{"data.datasets": ["clean253__features-v1-gate"]}))


def test_rejects_other_normaliser(tmp_path):
    from training.datasets.train import copy_calibration
    with pytest.raises(ValueError, match="normaliser"):
        copy_calibration(_config(), "n2", _source(tmp_path))


def test_rejects_disagreeing_given_lambda(tmp_path):
    from training.datasets.train import copy_calibration
    with pytest.raises(ValueError, match="lambda_j"):
        copy_calibration(_config(**{"loss.jacobian.lambda_j": 0.07}), "n1", _source(tmp_path))


def test_source_must_be_bcj(tmp_path):
    from training.datasets.train import copy_calibration
    with pytest.raises(ValueError, match="bcj"):
        copy_calibration(_config(), "n1", _source(tmp_path, **{"identity.objective": "bc"}))


def test_source_path_may_be_relative_to_the_task_dir():
    from training.datasets.train import copy_calibration
    rel = "checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/config.yaml"
    if not (TASK_DIR / rel).exists():
        pytest.skip("R4 v1 checkpoint not present")
    source = yaml.safe_load((TASK_DIR / rel).read_text())
    record, sha = copy_calibration(_config(), source["provenance"]["normaliser_sha256"], rel)
    assert record["lambda_j"] == 0.07439793449003185 and sha == source["provenance"]["calibration_sha256"]
    assert record["copied_from"] == rel
