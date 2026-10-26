import builtins
import csv
import json

import pytest

from conftest import TASK_DIR
from training.common.dataset import EXPECTED_VALIDATION_IDS, build_split, load_episode_refs

pytestmark = pytest.mark.realdata
DATA = TASK_DIR / "data"


@pytest.fixture
def primary_roots():
    return {"A": DATA / "Markovian" / "Clean_253", "B": DATA / "Stateful" / "Clean_253"}


@pytest.fixture
def reduced_roots():
    return {"A": DATA / "Markovian" / "Clean_223", "B": DATA / "Stateful" / "Clean_223"}


def test_real_split_contract(primary_roots, reduced_roots, monkeypatch):
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    split = build_split(primary_roots, reduced_roots)
    assert not any(p.endswith(".npz") for p in opened)  # scenario CSVs and manifests only
    assert len(split["validation_ids"]) == 32
    assert len(split["screening_ids"]) == 8
    assert len(split["fitting_ids"]["clean253"]) == 221
    assert len(split["fitting_ids"]["clean223"]) == 196
    assert split["validation_ids"] == EXPECTED_VALIDATION_IDS
    assert len(set(split["screening_ids"])) == 8 and set(split["screening_ids"]) <= set(split["validation_ids"])
    assert not set(split["validation_ids"]) & set(split["fitting_ids"]["clean253"])
    assert not set(split["validation_ids"]) & set(split["fitting_ids"]["clean223"])
    assert set(split["fitting_ids"]["clean223"]) <= set(split["fitting_ids"]["clean253"])
    heights = {r["scenario_id"]: float(r["target_z"]) for r in split["validation_scenarios"]}
    assert sum(h < 0.22753 for h in heights.values()) == 5


def test_teachers_are_paired(primary_roots):
    a = [r["scenario_id"] for r in csv.DictReader(open(primary_roots["A"] / "scenarios_train.csv"))]
    b = [r["scenario_id"] for r in csv.DictReader(open(primary_roots["B"] / "scenarios_train.csv"))]
    assert a == b and len(a) == 253


def test_committed_split_manifest_matches_rule(primary_roots, reduced_roots):
    split = build_split(primary_roots, reduced_roots)
    saved = json.loads((TASK_DIR / "experiments" / "splits" / "clean253.json").read_text())
    for key in ("validation_ids", "screening_ids", "fitting_ids", "source_sha256", "rule", "version"):
        assert saved[key] == split[key]


@pytest.mark.parametrize("teacher", ["A", "B"])
def test_real_training_refs_verify(primary_roots, teacher):
    refs = load_episode_refs(primary_roots[teacher], teacher, "train")
    assert len(refs) == 253 and len({r.scenario_id for r in refs}) == 253


@pytest.mark.parametrize("teacher,dim", [("A", 21), ("B", 26)])
def test_real_feature_builder_on_two_episodes(primary_roots, teacher, dim, tmp_path):
    import numpy as np
    from training.common.dataset import FeatureBuilder, build_cache
    refs = load_episode_refs(primary_roots[teacher], teacher, "train")[:2]
    ctx = {"goal_rule": "template-origin-standoff", "time_scale_s": 25.0} if teacher == "B" else None
    cache = build_cache(refs, FeatureBuilder(teacher, ctx), tmp_path / "cache")
    assert cache.x.shape[1] == dim and np.isfinite(cache.x).all()
    assert len(cache.x) == sum(r.metadata["outcome"]["T_actions"] for r in refs)
    assert set(np.unique(cache.diagnostics["phase"])) <= {0, 1, 2}
    if teacher == "B":
        assert cache.x[0, -1] == 0.0 and np.isclose(cache.x[1, -1], 0.001 / 25.0)
