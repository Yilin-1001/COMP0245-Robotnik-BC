import copy
import importlib.util

import numpy as np
import pytest
import torch

from conftest import TASK_DIR
from training.baseline.train import objective_from
from training.common.config import resolve_config, validate_config as clean_validator, run_dir_for
from training.common.dataset import EpisodeCache
from training.common.engine import fit, init_model
from training.common.checkpoint import model_payload, save_checkpoint


def modules():
    assert importlib.util.find_spec("training.dagger"), "independent DAgger trainer is missing"
    from training.dagger import config, train
    return config, train


def tiny(tmp_path, normaliser):
    c = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", "A", {})
    c["identity"].update(regime="dagger", architecture="m1")
    c["training"].update(lr=1e-4, updates=4, warmup_updates=2, log_every=2, checkpoint_every=2,
                         device="cpu", batch_size=8)
    c["loss"]["action"] = "huber"
    c["outputs"]["checkpoint_root"] = str(tmp_path / "ckpts")
    c["evaluation"]["jacobian_samples"] = 2
    c["dagger"] = {"round": 1, "parent_sha256": "a"*64, "parent_normaliser_sha256": normaliser.sha256(),
                    "round_manifest_sha256": "b"*64, "protocol_sha256": "c"*64}
    c["provenance"] = {"dataset_kind": "dagger aggregate", "normaliser_sha256": normaliser.sha256(),
                        "source_sha256": {"e0": "d"*64}, "code_digest": "e"*64}
    config, _ = modules()
    return config.finalise(c)


def cache():
    rng = np.random.default_rng(2)
    return EpisodeCache(rng.normal(size=(24, 21)).astype(np.float32), rng.normal(size=(24, 9)).astype(np.float32),
                        np.array([0, 11, 24]), ["e0", "e1"], {"t": np.zeros(24), "phase": np.zeros(24, int)})


def test_dagger_uses_explicit_validator_and_clean_path_still_refuses_it(tmp_path, unit_normaliser):
    c = tiny(tmp_path, unit_normaliser)
    config, _ = modules()
    with pytest.raises(ValueError, match="clean"):
        clean_validator(c, ready_to_train=True)
    with pytest.raises(ValueError, match="clean"):
        fit(c, cache(), None, init_model(c), unit_normaliser, objective_from(c))
    result = fit(c, cache(), None, init_model(c), unit_normaliser, objective_from(c), validator=config.validate)
    assert result.terminal_update == 4
    state = torch.load(result.run_dir/"latest.pt", weights_only=True)
    assert all(int(v["step"]) == 4 for v in state["optimizer_state"]["state"].values())
    assert state["normaliser"] == unit_normaliser.to_dict()


def test_parent_weights_warm_start_but_optimizer_starts_at_zero(tmp_path, unit_normaliser):
    c = tiny(tmp_path, unit_normaliser)
    _, train = modules()
    parent = init_model(c)
    with torch.no_grad():
        for p in parent.parameters():
            p.add_(.25)
    path = tmp_path / "parent.pt"
    save_checkpoint(path, model_payload(parent, unit_normaliser, {"teacher": "A"}, 84000, config=c))
    student = train.warm_start(c, path, unit_normaliser)
    assert all(torch.equal(v, student.state_dict()[k]) for k, v in parent.state_dict().items())
    config, _ = modules()
    r = fit(c, cache(), None, student, unit_normaliser, objective_from(c), validator=config.validate)
    state = torch.load(r.run_dir/"latest.pt", weights_only=True)
    assert state["step"] == 4
    assert all(int(v["step"]) == 4 for v in state["optimizer_state"]["state"].values())
    altered = copy.deepcopy(unit_normaliser)
    altered.x_mean[0] = 1
    with pytest.raises(ValueError, match="normalis"):
        train.warm_start(c, path, altered)


def test_same_round_resume_matches_uninterrupted_and_data_change_is_refused(tmp_path, unit_normaliser):
    c = tiny(tmp_path / "full", unit_normaliser)
    config, _ = modules()
    full = init_model(c)
    fit(c, cache(), None, full, unit_normaliser, objective_from(c), validator=config.validate)
    split = tiny(tmp_path / "split", unit_normaliser)
    def stop(step, *_):
        if step == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        fit(split, cache(), None, init_model(split), unit_normaliser, objective_from(split), on_checkpoint=stop,
            validator=config.validate)
    resumed = init_model(split)
    latest = run_dir_for(split)/"latest.pt"
    fit(split, cache(), None, resumed, unit_normaliser, objective_from(split), resume=latest, validator=config.validate)
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in full.state_dict().items())
    changed = copy.deepcopy(split)
    changed["dagger"]["round_manifest_sha256"] = "f"*64
    changed = config.finalise(changed)
    with pytest.raises((ValueError, FileNotFoundError)):
        fit(changed, cache(), None, init_model(changed), unit_normaliser, objective_from(changed), resume=latest,
            validator=config.validate)
