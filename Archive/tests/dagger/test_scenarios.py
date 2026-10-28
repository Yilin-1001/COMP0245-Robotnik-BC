import importlib.util
from collections import Counter

import pytest


def module():
    assert importlib.util.find_spec("data_collection.dagger_scenarios"), "DAgger scene freezing is not implemented"
    from data_collection import dagger_scenarios
    return dagger_scenarios


def test_balanced_new_scenes_are_disjoint_and_reproducible():
    m = module()
    train = m.balanced_scenes(2026102701, 8, "dagger_train")
    test = m.balanced_scenes(2026102703, 2, "dagger_test")
    assert len(train) == 256 and len(test) == 64
    assert set(Counter(tuple(r["cell"]) for r in train if r["round"] == 1).values()) == {2}
    assert Counter(r["round"] for r in train) == {1: 64, 2: 64, 3: 64, 4: 64}
    assert train == m.balanced_scenes(2026102701, 8, "dagger_train")
    m.assert_disjoint(train + test, [])
    with pytest.raises(ValueError, match="geometry"):
        m.assert_disjoint(train, [dict(train[0], scenario_id="different-id")])


def test_frozen_content_cannot_be_overwritten(tmp_path):
    m = module()
    path = tmp_path / "manifest.json"
    first = m.freeze_json(path, {"round": 1})
    assert first == m.freeze_json(path, {"round": 1})
    with pytest.raises(ValueError):
        m.freeze_json(path, {"round": 2})
    assert path.read_text().find('1') >= 0
