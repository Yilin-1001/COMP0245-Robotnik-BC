import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import FEATURES_A, TASK_DIR
from training.baseline.loss import ActionObjective
from training.common.config import finalise_identity, resolve_config
from training.common.dataset import EpisodeCache
from training.common.engine import FitResult, fit, init_model, stream_seed
from training.common.normalisation import fit_normaliser

INTERFACE = {"action_order": "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]", "torque_limit": 30.0,
             "wheel_limit": 6.0, "control_hz": 1000}


def tiny_cache(lengths, seed):
    rng = np.random.default_rng(seed)
    n = sum(lengths)
    x = rng.normal(size=(n, 21)).astype(np.float32)
    W = np.random.default_rng(99).normal(size=(21, 9))
    y = (x @ W * 0.5).astype(np.float32)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    phase = np.zeros(n, np.int8)
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        phase[hi - min(5, hi - lo):hi] = 2
    return EpisodeCache(x, y, offsets, [f"e{i}" for i in range(len(lengths))],
                        {"phase": phase, "t": np.zeros(n, np.float32)},
                        {"feature_names": list(FEATURES_A)})


@pytest.fixture
def train_cache():
    return tiny_cache([40, 25, 61], 0)


@pytest.fixture
def validation_cache():
    return tiny_cache([30, 12], 1)


@pytest.fixture
def normaliser(train_cache):
    return fit_normaliser(train_cache, {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05})


def make_config(tmp_path, normaliser, **training):
    c = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", "A",
                       {"outputs.checkpoint_root": str(tmp_path / "ckpt"), "training.device": "cpu",
                        "identity.architecture": "m1"})
    c["training"].update({"lr": 3e-3, "updates": 4, "warmup_updates": 2, "log_every": 2, "checkpoint_every": 2,
                          "batch_size": 16, **training})
    c["loss"]["action"] = "mse"
    c["loss"].update(arm_weight=0.5, wheel_weight=0.5)  # fixtures use the legacy direct objective defaults
    c["evaluation"]["jacobian_samples"] = 8
    prov = {"source_sha256": {"e0": "a" * 64}, "split_sha256": "b" * 64, "feature_sha256": "c" * 64,
            "normaliser_sha256": normaliser.sha256(), "code_digest": "d" * 64, "dataset_kind": "clean subset",
            "interface": dict(INTERFACE)}
    return finalise_identity(c, prov)


@pytest.fixture
def config(tmp_path, normaliser):
    return make_config(tmp_path, normaliser)


@pytest.fixture
def model(config):
    return init_model(config)


@pytest.fixture
def action_objective():
    return ActionObjective("mse")


def test_engine_honours_update_count(config, train_cache, validation_cache, model, normaliser, action_objective):
    result = fit(config, train_cache, validation_cache, model, normaliser, action_objective)
    assert result.terminal_update == config["training"]["updates"] == 4
    assert isinstance(result, FitResult) and result.status == "completed"
    assert set(result.checkpoints) == {2, 4, "latest"}
    assert (result.run_dir / "step_4.pt").is_file() and (result.run_dir / "step_4.npz").is_file()
    assert (result.run_dir / "metrics.csv").is_file() and not (result.run_dir / "best.pt").exists()


def test_training_reduces_action_error(tmp_path, train_cache, validation_cache, normaliser, action_objective):
    c = make_config(tmp_path, normaliser, updates=300, log_every=100, checkpoint_every=300, warmup_updates=10)
    m = init_model(c)
    x = torch.as_tensor(normaliser.transform_x(train_cache.x), dtype=torch.float32)
    y = torch.as_tensor(normaliser.transform_y(train_cache.y), dtype=torch.float32)
    before = action_objective(m, x, y, normaliser, None).total.item()
    fit(c, train_cache, validation_cache, m, normaliser, action_objective)
    after = action_objective(m, x, y, normaliser, None).total.item()
    assert after < 0.5 * before


def test_optimiser_and_schedule(config, train_cache, validation_cache, model, normaliser, action_objective):
    result = fit(config, train_cache, validation_cache, model, normaliser, action_objective)
    ckpt = torch.load(result.run_dir / "latest.pt", weights_only=True)
    group = ckpt["optimizer_state"]["param_groups"][0]
    assert group["weight_decay"] == 1e-4 and group["initial_lr"] == 3e-3
    lrs = [float(r.split(",")[2]) for r in (result.run_dir / "metrics.csv").read_text().splitlines()[1:]]
    # update s (0-based) uses (s+1)/w during warm-up, then 0.5 * (1 + cos(pi * (s - w) / (U - w)))
    assert np.allclose(lrs, [3e-3 * 1.0, 3e-3 * 0.5], atol=1e-12)


def test_constant_schedule_after_warmup(tmp_path, train_cache, validation_cache, normaliser, action_objective):
    c = make_config(tmp_path, normaliser, schedule="constant")
    result = fit(c, train_cache, validation_cache, init_model(c), normaliser, action_objective)
    lrs = [float(r.split(",")[2]) for r in (result.run_dir / "metrics.csv").read_text().splitlines()[1:]]
    assert np.allclose(lrs, [3e-3, 3e-3])


def test_sampling_is_episodic(config, train_cache, validation_cache, model, normaliser, action_objective, monkeypatch):
    import training.common.engine as engine
    calls = []
    real = engine.sample_indices
    monkeypatch.setattr(engine, "sample_indices", lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
    fit(config, train_cache, validation_cache, model, normaliser, action_objective)
    assert calls == [16] * 4


def test_engine_imports_no_method_or_teacher_modules():
    code = ("import sys, training.common.engine, training.baseline.train; "
            "print([m for m in sys.modules if m.startswith(('training.jacobian', 'data_collection', 'dart', 'dagger'))"
            " or m in ('pybullet', 'core.env')])")
    out = subprocess.run([sys.executable, "-c", code], cwd=TASK_DIR, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def _weights(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def test_interrupted_resume_matches_uninterrupted(tmp_path, train_cache, validation_cache, normaliser,
                                                  action_objective):
    full_cfg = make_config(tmp_path / "full", normaliser)
    full = init_model(full_cfg)
    fit(full_cfg, train_cache, validation_cache, full, normaliser, action_objective)

    cfg = make_config(tmp_path / "split", normaliser)
    part = init_model(cfg)

    def interrupt(update, *_):
        if update == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        fit(cfg, train_cache, validation_cache, part, normaliser, action_objective, on_checkpoint=interrupt)
    resumed = init_model(cfg)  # a fresh process would rebuild the model, then restore it
    run_dir = tmp_path / "split" / "ckpt" / "clean" / "A" / "bc" / cfg["identity"]["run_id"]
    result = fit(cfg, train_cache, validation_cache, resumed, normaliser, action_objective, resume=run_dir / "latest.pt")
    assert result.terminal_update == 4
    assert torch.equal(_weights(resumed), _weights(full))
    a = torch.load(run_dir / "latest.pt", weights_only=True)
    b = torch.load(tmp_path / "full" / "ckpt" / "clean" / "A" / "bc" / full_cfg["identity"]["run_id"] / "latest.pt",
                   weights_only=True)
    assert a["scheduler_state"]["_last_lr"] == b["scheduler_state"]["_last_lr"]


def test_callback_cannot_change_later_batches(tmp_path, train_cache, validation_cache, normaliser, action_objective):
    plain_cfg = make_config(tmp_path / "plain", normaliser)
    plain = init_model(plain_cfg)
    fit(plain_cfg, train_cache, validation_cache, plain, normaliser, action_objective)
    noisy_cfg = make_config(tmp_path / "noisy", normaliser)
    noisy = init_model(noisy_cfg)

    def meddle(update, ckpt, export, model):
        torch.rand(100)
        np.random.rand(100)
        model.eval()
    fit(noisy_cfg, train_cache, validation_cache, noisy, normaliser, action_objective, on_checkpoint=meddle)
    assert torch.equal(_weights(plain), _weights(noisy))


def test_streams_are_independent_and_seed_derived():
    assert stream_seed(0, "batch") != stream_seed(0, "init") != stream_seed(0, "jacobian")
    assert stream_seed(0, "batch") == stream_seed(0, "batch") and stream_seed(1, "batch") != stream_seed(0, "batch")


def test_nonfinite_loss_is_rejected(config, train_cache, validation_cache, model, normaliser):
    from training.common.engine import LossResult

    def bad(model, x, y, n, g):
        return LossResult(model(x).sum() * float("nan"), {})
    with pytest.raises(FloatingPointError):
        fit(config, train_cache, validation_cache, model, normaliser, bad)


def test_training_requires_complete_recipe(tmp_path, train_cache, validation_cache, normaliser, action_objective):
    c = make_config(tmp_path, normaliser)
    c["training"]["updates"] = None
    with pytest.raises(ValueError):
        fit(c, train_cache, validation_cache, init_model(c), normaliser, action_objective)
