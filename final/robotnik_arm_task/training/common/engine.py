"""Objective contract shared by every training objective; the update loop uses only this interface.

An Objective is called as objective(model, x, y, normaliser, generator) with standardised batches; generator is
the independent Jacobian sub-batch stream, which action-only objectives ignore.
"""
from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class LossResult:
    total: torch.Tensor
    metrics: dict = field(default_factory=dict)


class Objective(Protocol):
    def __call__(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, normaliser,
                 generator: torch.Generator) -> LossResult: ...


# ---------------------------------------------------------------- update loop

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from training.common.checkpoint import export_numpy, load_checkpoint, model_payload, save_checkpoint
from training.common.dataset import sample_indices
from training.common.models import build_model

METRIC_COLUMNS = ["update", "elapsed_s", "lr", "train_loss", "train_arm_loss", "train_wheel_loss", "val_arm_rmse_nm",
                  "val_wheel_rmse_rad_s", "val_hold_arm_rmse_nm", "val_hold_bias_max_abs_nm", "val_hold_bias_gate_pass",
                  "val_damping_median_diag", "val_damping_frac_below_minus33", "extra"]


@dataclass
class FitResult:
    run_dir: Path
    terminal_update: int
    checkpoints: dict
    status: str


def stream_seed(seed: int, stream: str) -> int:
    """Independent RNG streams derived from the paired experiment seed (never from the objective-specific run ID)."""
    return int.from_bytes(hashlib.sha256(f"{int(seed)}:{stream}".encode()).digest()[:8], "little") % (2 ** 63)


def init_model(config: dict):
    """Model initialised from the paired seed's init stream; identical for BC and BC+J of one seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stream_seed(config["identity"]["seed"], "init"))
        return build_model(config)


def lr_factor(s: int, warmup: int, updates: int, schedule: str) -> float:
    """Multiplier for the s-th update (0-based): linear warm-up, then cosine to the budget or constant."""
    if s < warmup:
        return (s + 1) / warmup
    if schedule == "constant":
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, updates - warmup)))


def _rng_state(generators: dict) -> dict:
    state = {f"gen_{k}": g.get_state() for k, g in generators.items()}
    state["torch_cpu"] = torch.get_rng_state()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict, generators: dict) -> None:
    for k, g in generators.items():
        g.set_state(state[f"gen_{k}"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def policy_meta(config: dict) -> dict:
    ident = config["identity"]
    return {"teacher": ident["teacher"], "input_dim": config["model"]["input_dim"], "run_id": ident["run_id"],
            "objective": ident["objective"], "interface": dict(config["interface"]),
            "task_context": config["data"]["task_context"], "feature_names": None}


def offline_validation(model, validation, normaliser, config: dict) -> dict:
    """Offline proxy on the validation cache, with the model's mode and the global RNG left untouched."""
    from evaluation.offline import offline_metrics
    was_training = model.training
    with torch.random.fork_rng(devices=[]):
        m = offline_metrics(model, validation, normaliser, config)
    model.train(was_training)
    return m


def fit(config: dict, train, validation, model, normaliser, objective, on_checkpoint: Callable | None = None,
        resume: Path | None = None) -> FitResult:
    """Run config['training']['updates'] optimiser updates of objective on the fitting cache.

    Checkpoints step_<u>.pt (+ step_<u>.npz export) every checkpoint_every updates and at the terminal update;
    latest.pt is always resumable. Selection runs never write best.pt here: the closed-loop selection does.
    """
    from training.common.config import configure_runtime, create_run_dir, validate_config
    validate_config(config, ready_to_train=True)
    tr = config["training"]
    runtime = configure_runtime(tr["device"])
    device = torch.device(runtime["device"])
    run_dir = create_run_dir(config, resume=resume is not None)
    meta = dict(policy_meta(config), feature_names=list(normaliser.feature_names))
    normaliser.save(run_dir / "normaliser.npz")
    (run_dir / "provenance.json").write_text(json.dumps(config["provenance"], indent=1, default=str))

    x = torch.as_tensor(normaliser.transform_x(train.x), dtype=torch.float32, device=device)
    y = torch.as_tensor(normaliser.transform_y(train.y), dtype=torch.float32, device=device)
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise FloatingPointError("non-finite standardised fitting data")
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    U, w = tr["updates"], tr["warmup_updates"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, w, U, tr["schedule"]))
    seed = config["identity"]["seed"]
    gens = {name: torch.Generator(device=device).manual_seed(stream_seed(seed, name)) for name in ("batch", "jacobian")}
    start, elapsed0 = 0, 0.0
    checkpoints = {}
    if resume is not None:
        state = load_checkpoint(Path(resume), expected={"run_id": config["identity"]["run_id"]})
        if state["config"]["identity"]["config_hash"] != config["identity"]["config_hash"]:
            raise ValueError("resume checkpoint belongs to a different configuration")
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        sched.load_state_dict(state["scheduler_state"])
        _restore_rng(state["rng"], gens)
        start, elapsed0 = state["step"], state.get("elapsed_s", 0.0)
        checkpoints = {int(k): Path(v) for k, v in state.get("checkpoints", {}).items()}
        with open(run_dir / "continuations.jsonl", "a") as f:
            f.write(json.dumps({"resumed_at": start, "runtime": runtime}, default=str) + "\n")
    metrics_path = run_dir / "metrics.csv"
    if resume is None or not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(METRIC_COLUMNS)
    acc, n_acc, t0 = {}, 0, time.time()
    for s in range(start, U):
        idx = sample_indices(train, tr["batch_size"], gens["batch"], device)
        result = objective(model, x[idx], y[idx], normaliser, gens["jacobian"])
        if not torch.isfinite(result.total):
            raise FloatingPointError(f"non-finite loss at update {s + 1}")
        lr_used = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        result.total.backward()
        opt.step()
        sched.step()
        update = s + 1
        for k, v in result.metrics.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        n_acc += 1
        if update % tr["log_every"] == 0 or update == U:
            row = _metrics_row(update, elapsed0 + time.time() - t0, lr_used, {k: v / n_acc for k, v in acc.items()},
                               offline_validation(model, validation, normaliser, config) if validation is not None else None)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f, lineterminator="\n").writerow(row)
            acc, n_acc = {}, 0
        if update % tr["checkpoint_every"] == 0 or update == U:
            ckpt, export = run_dir / f"step_{update}.pt", run_dir / f"step_{update}.npz"
            payload = model_payload(model, normaliser, meta, update, config=config, provenance=config["provenance"],
                                    environment=runtime)
            save_checkpoint(ckpt, payload)
            export_numpy(export, model, normaliser, {**meta, "source_checkpoint": str(ckpt), "source_step": update})
            checkpoints[update] = ckpt
            save_checkpoint(run_dir / "latest.pt", model_payload(
                model, normaliser, meta, update, config=config, provenance=config["provenance"], environment=runtime,
                optimizer_state=opt.state_dict(), scheduler_state=sched.state_dict(), rng=_rng_state(gens),
                elapsed_s=elapsed0 + time.time() - t0, checkpoints={str(k): str(v) for k, v in checkpoints.items()}))
            if on_checkpoint is not None:
                with torch.random.fork_rng(devices=[]):
                    on_checkpoint(update, ckpt, export, model)
                model.train()
    return FitResult(run_dir, U, {**checkpoints, "latest": run_dir / "latest.pt"}, "completed")


def _metrics_row(update, elapsed, lr, train_metrics: dict, val: dict | None) -> list:
    v = val or {}
    hold = (v.get("phase") or {}).get("hold") or {}
    bias = v.get("hold_bias") or {}
    damping = v.get("damping") or {}
    extra = {k: val for k, val in train_metrics.items() if k not in ("action_loss", "arm_loss", "wheel_loss")}
    return [update, round(elapsed, 3), lr, train_metrics.get("action_loss"), train_metrics.get("arm_loss"),
            train_metrics.get("wheel_loss"), v.get("arm_rmse_nm"), v.get("wheel_rmse_rad_s"), hold.get("arm_rmse_nm"),
            bias.get("max_abs_nm"), bias.get("gate_pass"), damping.get("median_diagonal"),
            damping.get("frac_below_minus33"), json.dumps(extra) if extra else ""]


def read_metrics(run_dir: Path) -> list[dict]:
    with open(Path(run_dir) / "metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


def select_checkpoint(run_dir: Path, step: int, evidence: dict) -> dict:
    """Validation-selected artifact of a selection run: best.pt plus its matching inference.npz."""
    import shutil
    run_dir = Path(run_dir)
    src = run_dir / f"step_{step}.pt"
    payload = load_checkpoint(src)
    if payload["config"]["identity"]["fit_stage"] != "selection":
        raise ValueError("best.pt exists only for selection runs; refit runs save final.pt")
    for name in ("best.pt", "inference.npz", "selection.json"):
        if (run_dir / name).exists():
            raise FileExistsError(f"{run_dir / name} exists; selection is frozen")
    shutil.copyfile(src, run_dir / "best.pt")
    model = build_model(payload["config"])
    model.load_state_dict(payload["model_state"])
    from training.common.normalisation import Normaliser
    normaliser = Normaliser.from_dict(payload["normaliser"])
    meta = {k: v for k, v in payload["meta"].items() if k not in ("normaliser_sha256", "model")}
    info = export_numpy(run_dir / "inference.npz", model, normaliser,
                        {**meta, "source_checkpoint": str(run_dir / "best.pt"), "source_step": step})
    record = {"selected_step": step, **info, "evidence": evidence}
    (run_dir / "selection.json").write_text(json.dumps(record, indent=1, default=str))
    return record


# ---------------------------------------------------------------- run preparation (shared by clean launchers)

TASK_DIR = Path(__file__).resolve().parents[2]
CODE_GLOBS = ("training/**/*.py", "evaluation/offline.py", "evaluation/evaluate.py", "evaluation/test_sets.py",
              "core/features.py", "core/interfaces.py", "experiments/configs/**/*.yaml")


def code_digest() -> str:
    files = sorted({p for g in CODE_GLOBS for p in TASK_DIR.glob(g) if "__pycache__" not in p.parts})
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(TASK_DIR)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def _git(*args) -> str:
    import subprocess
    return subprocess.run(["git", *args], cwd=TASK_DIR, capture_output=True, text=True).stdout.strip()


def _root(config: dict, key: str) -> Path:
    p = Path(config["data"][key])
    return p if p.is_absolute() else TASK_DIR / p


def prepare_run(config: dict, fitting_ids: list[str] | None = None, workers: int = 16) -> dict:
    """Validate clean sources, build/reuse caches, fit fitting-only scales and finalise the run identity.

    Selection runs fit on the cohort's fitting IDs and validate on the 32 frozen IDs read from Clean_253.
    Refit runs fit on every training ID of the cohort and have no validation set.
    fitting_ids overrides the fitting set (the separately identified one-episode sanity run).
    """
    from training.common.config import finalise_identity
    from training.common.dataset import (FeatureBuilder, build_cache, cache_id, load_episode_refs, load_manifest,
                                         load_split, manifest_interface, select_refs, sha256_file)
    from training.common.normalisation import fit_normaliser
    ident, data = config["identity"], config["data"]
    teacher = ident["teacher"]
    primary, reduced = _root(config, "primary_root"), _root(config, "reduced_root")
    fit_root = primary if ident["cohort"] == "clean253" else reduced
    manifests = {r: load_manifest(r, teacher) for r in {fit_root, primary}}
    interfaces = {json.dumps(manifest_interface(m), sort_keys=True) for m in manifests.values()}
    if len(interfaces) != 1:
        raise ValueError("fitting and validation sources disagree on the execution interface")
    split_path = TASK_DIR / data["split_manifest"]
    split = load_split(split_path)
    fit_refs_all = load_episode_refs(fit_root, teacher, "train")
    if ident["fit_stage"] == "refit":
        ids, val_ids = [r.scenario_id for r in fit_refs_all], []
    else:
        ids, val_ids = split["fitting_ids"][ident["cohort"]], split["validation_ids"]
    if fitting_ids is not None:
        if not set(fitting_ids) <= set(ids):
            raise ValueError("sanity fitting IDs must come from the cohort's fitting set")
        ids = list(fitting_ids)
    fit_refs = select_refs(fit_refs_all, ids)
    val_refs = select_refs(load_episode_refs(primary, teacher, "train"), val_ids) if val_ids else []
    builder = FeatureBuilder(teacher, data["task_context"])
    cache_root = Path(data["cache_root"]) if Path(data["cache_root"]).is_absolute() else TASK_DIR / data["cache_root"]
    caches, ids_by = {}, {}
    for name, refs in (("fitting", fit_refs), ("validation", val_refs)):
        if not refs:
            caches[name] = None
            continue
        cid = cache_id(refs, builder.builder_id)
        caches[name] = build_cache(refs, builder, cache_root / teacher / cid, workers=workers)
        ids_by[name] = cid
    if caches["fitting"].x.shape[1] != config["model"]["input_dim"]:
        raise ValueError("cached feature dimension disagrees with the teacher's input dimension")
    floors = {k: data[k] for k in ("input_scale_floor", "arm_scale_floor", "wheel_scale_floor")}
    normaliser = fit_normaliser(caches["fitting"], floors)
    provenance = {
        "dataset_kind": next(iter(manifests.values()))["dataset_kind"],
        "teacher_version": data["teacher_version"], "interface": json.loads(interfaces.pop()),
        "source_sha256": {f"fit/{r.scenario_id}": r.sha256 for r in fit_refs} | {f"val/{r.scenario_id}": r.sha256 for r in val_refs},
        "fitting_ids": ids, "validation_ids": val_ids, "split_sha256": sha256_file(split_path),
        "feature_sha256": hashlib.sha256(builder.builder_id.encode()).hexdigest(), "cache_ids": ids_by,
        "normaliser_sha256": normaliser.sha256(), "code_digest": code_digest(),
        "operational": {"git_head": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    if config["identity"]["objective"] == "bcj":
        calib = config["loss"]["jacobian"].get("calibration_record")
        if calib:
            provenance["calibration_sha256"] = calib["sha256"]
    final = finalise_identity(config, provenance)
    return {"config": final, "train": caches["fitting"], "validation": caches["validation"], "normaliser": normaliser}
