import builtins
import csv
import json

import numpy as np
import pytest
import torch

from training.common.dataset import (EpisodeCache, build_cache, load_cache, load_episode_refs, sample_indices)


def q_builder(arrays, meta):
    """Test feature builder: the first T measured joint angles (the real builders live in core.features)."""
    return arrays["obs_q_mes"]


def test_alignment_uses_first_T_observations_and_reduces_labels(fake_dataset, tmp_path):
    root = fake_dataset(train={"train-0000": 5, "train-0001": 3})
    refs = load_episode_refs(root, "A", "train")
    cache = build_cache(refs, q_builder, tmp_path / "cache")
    assert isinstance(cache, EpisodeCache)
    assert cache.offsets.tolist() == [0, 5, 8] and cache.episode_ids == ["train-0000", "train-0001"]
    assert cache.x.dtype == np.float32 and cache.y.dtype == np.float32 and cache.y.shape == (8, 9)
    for ref, (lo, hi) in zip(refs, zip(cache.offsets[:-1], cache.offsets[1:])):
        with np.load(ref.path) as z:
            np.testing.assert_array_equal(cache.x[lo:hi], z["obs_q_mes"][:-1].astype(np.float32))
            a = z["a_expert"]
            np.testing.assert_array_equal(cache.y[lo:hi], np.c_[a[:, :7], a[:, 7], a[:, 8]].astype(np.float32))
    assert cache.diagnostics["phase"].shape == (8,) and cache.diagnostics["t"].shape == (8,)


def _expect_reject(fake_dataset, tmp_path, match, **kw):
    root = fake_dataset(**kw)
    with pytest.raises(ValueError, match=match):
        build_cache(load_episode_refs(root, "A", "train"), q_builder, tmp_path / "cache")


def test_rejects_same_side_wheel_mismatch(fake_dataset, tmp_path):
    def bad(meta, arrays, split, sid):
        arrays["a_expert"][1, 10] += 1.0
    _expect_reject(fake_dataset, tmp_path, "wheel", mutate=bad)


def test_rejects_nonfinite_values(fake_dataset, tmp_path):
    def bad(meta, arrays, split, sid):
        arrays["obs_q_mes"][0, 2] = np.nan
    _expect_reject(fake_dataset, tmp_path, "non-finite", mutate=bad)


def test_rejects_wrong_observation_action_alignment(fake_dataset, tmp_path):
    def bad(meta, arrays, split, sid):
        arrays["obs_q_mes"] = arrays["obs_q_mes"][:-1]
    _expect_reject(fake_dataset, tmp_path, "T\\+1", mutate=bad)


def test_rejects_non_clean_provenance(fake_dataset, tmp_path):
    def bad(meta, arrays, split, sid):
        meta["noise_configuration"] = {"enabled": True}
    root = fake_dataset(mutate=bad)
    with pytest.raises(ValueError, match="clean"):
        load_episode_refs(root, "A", "train")


def test_rejects_wrong_action_order(fake_dataset, tmp_path):
    def bad(meta, arrays, split, sid):
        meta["execution_interface"]["order"] = "a = [wheel_cmds (4), torque_arm (7)]"
    root = fake_dataset(mutate=bad)
    with pytest.raises(ValueError, match="action order"):
        load_episode_refs(root, "A", "train")


def test_rejects_wrong_teacher_version(fake_dataset):
    root = fake_dataset()
    with pytest.raises(ValueError, match="teacher version"):
        load_episode_refs(root, "B", "train")


def test_rejects_wrong_file_hash(fake_dataset):
    root = fake_dataset()
    path = next((root / "train").glob("*.npz"))
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="hash"):
        load_episode_refs(root, "A", "train")


def _rewrite_csv(path, edit):
    rows = list(csv.reader(open(path)))
    edit(rows)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def test_rejects_duplicate_episode_ids(fake_dataset):
    root = fake_dataset()
    _rewrite_csv(root / "training_manifest.csv", lambda rows: rows.append(rows[1]))
    with pytest.raises(ValueError, match="duplicate"):
        load_episode_refs(root, "A", "train")


def test_rejects_unsafe_manifest_paths(fake_dataset):
    root = fake_dataset()
    def edit(rows):
        rows[1][3] = "../../outside.npz"
    _rewrite_csv(root / "episode_provenance.csv", edit)
    with pytest.raises(ValueError, match="unsafe"):
        load_episode_refs(root, "A", "train")


def test_rejects_count_disagreeing_with_derived_manifest(fake_dataset):
    root = fake_dataset()
    m = json.loads((root / "manifest.json").read_text())
    m["counts"]["train_episodes"] = 256  # e.g. the inherited original acquisition count
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="count"):
        load_episode_refs(root, "A", "train")


def test_cache_reuse_and_tamper_detection(fake_dataset, tmp_path):
    root = fake_dataset()
    refs = load_episode_refs(root, "A", "train")
    cache = build_cache(refs, q_builder, tmp_path / "cache")
    again = load_cache(tmp_path / "cache", refs)
    np.testing.assert_array_equal(again.x, cache.x)
    meta_path = tmp_path / "cache" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["feature_names"] = ["tampered"]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="tampered|hash"):
        load_cache(tmp_path / "cache", refs)


def test_partial_cache_without_completion_is_invalid(fake_dataset, tmp_path):
    root = fake_dataset()
    refs = load_episode_refs(root, "A", "train")
    build_cache(refs, q_builder, tmp_path / "cache")
    (tmp_path / "cache" / "COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        load_cache(tmp_path / "cache", refs)


def test_training_never_opens_test_arrays(fake_dataset, tmp_path, monkeypatch):
    root = fake_dataset()
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    refs = load_episode_refs(root, "A", "train")
    build_cache(refs, q_builder, tmp_path / "cache")
    assert opened and not any("/test/" in p for p in opened)


def test_sampling_is_episode_uniform_then_timestep_uniform():
    cache = EpisodeCache(x=np.zeros((1001, 1), np.float32), y=np.zeros((1001, 9), np.float32),
                         offsets=np.array([0, 1, 1001]), episode_ids=["short", "long"], diagnostics={}, metadata={})
    g = torch.Generator().manual_seed(7)
    idx = sample_indices(cache, 4000, g, torch.device("cpu"))
    ref = torch.Generator().manual_seed(7)
    e = torch.randint(2, (4000,), generator=ref)
    u = torch.rand(4000, generator=ref)
    lengths = torch.tensor([1, 1000])
    expected = torch.tensor([0, 1])[e] + torch.minimum((u * lengths[e]).long(), lengths[e] - 1)
    assert torch.equal(idx, expected)
    # one-step episode gets an episode's share (about half), not 1/1001 of the draws
    assert 0.45 < (idx == 0).float().mean().item() < 0.55
