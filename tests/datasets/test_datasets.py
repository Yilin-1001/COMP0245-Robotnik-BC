"""Published feature datasets: build QA, hash-checked loading, scenario selection, mixes and the training resolver."""
import json

import numpy as np
import pytest
import torch

from training.datasets.build import build
from training.datasets.load import concat, load_dataset, select_scenarios


def test_v1_build_loads_and_selects(fake_dataset, tmp_path):
    src = fake_dataset("A", train={"train-0000": 5, "train-0001": 3, "train-0002": 4})
    out = tmp_path / "datasets"
    dest = build(str(src), "features-v1", "fake", out, workers=1)
    assert dest.name == "fake__features-v1" and json.loads((dest / "qa.json").read_text())["passed"]
    cache = load_dataset("fake__features-v1", out)
    assert cache.x.shape == (12, 21) and list(cache.offsets) == [0, 5, 8, 12]
    sub = select_scenarios(cache, ["train-0002", "train-0000"])
    assert sub.metadata["scenario_ids"] == ["train-0000", "train-0002"] and len(sub.x) == 9
    assert build(str(src), "features-v1", "fake", out, workers=1) == dest  # up to date, not rebuilt


def test_gate_qa_rejects_a_gate_that_disagrees_with_the_teacher_flag(fake_dataset, tmp_path):
    # The fake episodes' recorded reach flags are not derived from their random observations.
    src = fake_dataset("A")
    with pytest.raises(ValueError, match="reach_gate"):
        build(str(src), "features-v1-gate", "fake", tmp_path / "datasets", workers=1)
    assert not (tmp_path / "datasets" / "A" / "fake__features-v1-gate").exists()


def test_load_refuses_changed_files(fake_dataset, tmp_path):
    out = tmp_path / "datasets"
    dest = build(str(fake_dataset("A")), "features-v1", "fake", out, workers=1)
    x = np.load(dest / "x.npy")
    x[0, 0] += 1.0
    np.save(dest / "x.npy", x)
    with pytest.raises(ValueError, match="hash"):
        load_dataset("fake__features-v1", out)


def test_concat_shares_become_episode_weights(fake_dataset, tmp_path):
    out = tmp_path / "datasets"
    a = load_dataset(build(str(fake_dataset("A")), "features-v1", "fake", out, workers=1).name, out)
    b = select_scenarios(a, ["train-0000"])
    mixed = concat([a, b], [0.25, 0.75])
    assert len(mixed.episode_ids) == 3 and list(mixed.offsets) == [0, 5, 8, 13]
    np.testing.assert_allclose(mixed.episode_weights, [0.125, 0.125, 0.75])
    assert concat([a, b]).episode_weights is None
    with pytest.raises(ValueError):
        concat([a, b], [0.5, 0.6])


def test_sampling_without_weights_keeps_the_original_stream_and_weights_restrict_it():
    from training.common.dataset import EpisodeCache, sample_indices
    offsets = np.array([0, 4, 10])
    cache = EpisodeCache(np.zeros((10, 1)), np.zeros((10, 9)), offsets, ["a", "b"])
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    got = sample_indices(cache, 64, g1, torch.device("cpu"))
    e = torch.randint(2, (64,), generator=g2)
    u = torch.rand(64, generator=g2)
    lengths = torch.tensor([4, 6])
    assert torch.equal(got, torch.tensor([0, 4])[e] + torch.minimum((u * lengths[e]).long(), lengths[e] - 1))
    cache.episode_weights = np.array([0.0, 1.0])
    assert (sample_indices(cache, 64, torch.Generator().manual_seed(0), torch.device("cpu")) >= 4).all()


@pytest.mark.parametrize("variant,dim", [("R1-baseline", 21), ("R2-gate", 22), ("R3-dart", 21), ("R4-dart-gate", 22)])
def test_variants_resolve(variant, dim):
    from training.datasets.train import resolve
    config = resolve(variant, {"training.lr": 1e-3})
    assert config["model"]["input_dim"] == dim and config["identity"]["regime"] == "ablation"
    assert config["model"]["architectures"] == {"MLP01": [256, 256, 256]}  # m1-m6 are not carried over
    assert config["loss"]["jacobian"] is not None  # bcj by default
    assert resolve(variant, {"identity.objective": "bc"})["loss"]["jacobian"] is None


def test_resolver_rejects_mismatched_definitions_and_mixes():
    from training.datasets.train import resolve
    with pytest.raises(ValueError, match="feature definition"):
        resolve("R4-dart-gate", {"data.datasets": ["clean253__features-v1", "dart-d2s075x2__features-v1-gate"]})
    with pytest.raises(ValueError, match="mix"):
        resolve("R4-dart-gate", {"data.mix": [0.5, 0.4]})
    assert resolve("R4-dart-gate", {"data.mix": [0.5, 0.5]})["data"]["mix"] == [0.5, 0.5]


def test_run_names_follow_the_ablation_convention():
    from training.datasets.train import architecture_label, resolve, run_name, validate
    assert architecture_label("MLP01", [256, 256, 256]) == "MLP01-256x3"
    assert architecture_label("MLP02", [256, 128, 64]) == "MLP02-256-128-64"
    from training.common.config import parse_overrides
    config = resolve("R4-dart-gate", parse_overrides(["identity.version=v1", "identity.date=2026-10-28", "identity.seed=2"]))
    assert run_name(config) == "R4-dart-gate__MLP01-256x3__s2__v1__2026-10-28"  # CLI dates arrive as YAML dates
    with pytest.raises(ValueError, match="architecture"):
        resolve("R4-dart-gate", {"identity.architecture": "m6"})
    for bad in ({"identity.version": "1"}, {"identity.date": "28-10-2026"}):
        config = resolve("R4-dart-gate", {"identity.version": "v1", "identity.date": "2026-10-28", "training.lr": 1e-3,
                                          "training.updates": 10, "loss.action": "huber", **bad})
        config["loss"]["jacobian"]["lambda_j"] = 0.1
        with pytest.raises(ValueError):
            validate(config, ready_to_train=True)


def test_experiment_runs_use_the_experiment_directory(tmp_path):
    from training.common.config import run_dir_for
    config = {"identity": {"regime": "ablation", "teacher": "A", "objective": "bcj", "run_id": "R1"},
              "outputs": {"checkpoint_root": str(tmp_path), "experiment_dir": "2026-10-28__training-ablation-v1"}}
    assert run_dir_for(config) == tmp_path / "ablation" / "2026-10-28__training-ablation-v1" / "R1"
    del config["outputs"]["experiment_dir"]
    assert run_dir_for(config) == tmp_path / "ablation" / "A" / "bcj" / "R1"
