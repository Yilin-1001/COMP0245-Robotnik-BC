import importlib.util
import json

import numpy as np
import pytest

from data_collection.dagger_preflight import sha
from data_collection.dagger_rollout import save_episode
from training.common.dataset import EpisodeCache
from test_collection import run, ClockEnv


def module():
    assert importlib.util.find_spec("training.dagger.dataset"), "DAgger source validation is missing"
    from training.dagger import dataset
    return dataset


def fixture(tmp_path, fail=False, change=None):
    arrays, meta = run(0., lambda: ClockEnv(fail_after=3 if fail else None))
    row = dict(scenario_id="dagger-r1-0", split="dagger_train", round=1, target_x=1., target_y=0., target_z=.5,
               start_x=0., start_y=0., start_yaw=0., seed=1)
    contract = dict(contract_sha256="a"*64, source="dagger_mixed", normaliser_content_sha256="b"*64,
                    parent_hashes={"inference.npz": "c"*64}, interface={"torque_limit": 30, "wheel_limit": 6})
    meta.update(scenario=row, scenario_id=row["scenario_id"], split="dagger_train", round=1, source="dagger_mixed",
                contract_sha256="a"*64, normaliser_sha256="b"*64, policy_sha256="c"*64,
                teacher_version="markovian-lt0.50-v1")
    if change:
        change(arrays, meta)
    path = tmp_path / "episode.npz"
    save_episode(path, arrays, meta)
    entry = dict(file=path.name, sha256=sha(path), scenario_id=row["scenario_id"], steps=len(arrays["expert_action"]),
                 valid_rows=int(arrays["valid_supervision"].sum()))
    return path, entry, contract, row


def builder(obs, meta):
    return np.repeat(obs["obs_q_mes"][:, :1], 21, axis=1)


def test_failed_trajectory_keeps_current_state_labels_and_excludes_only_terminal_obs(tmp_path):
    args = fixture(tmp_path, fail=True)
    x, y, diag, stats = module().episode_rows(*args, heldout=[], builder=builder)
    np.testing.assert_array_equal(x[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(y[:, 0], [10, 11, 12])
    np.testing.assert_array_equal(y[:, 7:], np.full((3, 2), 8))
    assert stats["valid_rows"] == 3 and stats["excluded_rows"] == 0
    assert stats["failure"]


@pytest.mark.parametrize("change,match", [
    (lambda a,m: a.update(obs_q_mes=a["obs_q_mes"][:-1]), "alignment"),
    (lambda a,m: m.update(normaliser_sha256="x"*64), "normalis"),
    (lambda a,m: m.update(source="dagger_expert_reference"), "source"),
    (lambda a,m: a["expert_action"].__setitem__((0,10), 9.), "wheel"),
])
def test_dagger_rejects_misalignment_or_incompatible_supervision(tmp_path, change, match):
    args = fixture(tmp_path, change=change)
    with pytest.raises(ValueError, match=match):
        module().episode_rows(*args, heldout=[], builder=builder)


def test_geometry_leakage_and_modified_file_are_rejected(tmp_path):
    path, entry, contract, row = fixture(tmp_path)
    with pytest.raises(ValueError, match="geometry"):
        module().episode_rows(path, entry, contract, row, heldout=[dict(row, scenario_id="validation-id")], builder=builder)
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="hash"):
        module().episode_rows(path, entry, contract, row, heldout=[], builder=builder)


def test_aggregate_rejects_duplicate_episode_identity():
    c = EpisodeCache(np.zeros((2,21)), np.zeros((2,9)), np.array([0,2]), ["same"], {"phase": np.zeros(2)})
    with pytest.raises(ValueError, match="duplicate"):
        module().combine([c, c])
