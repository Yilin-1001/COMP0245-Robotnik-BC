"""Atomic checkpoints with compatibility checks, and explicit NumPy inference exports.

A checkpoint (.pt) holds the model, optional optimiser/scheduler/RNG state for resumption, the fitting-only
normaliser and the metadata a deployment must agree with (teacher, features, interface, task context).
A NumPy export (.npz) holds the same network as plain arrays, plus the hash and step of its source checkpoint.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

FORMAT = "mlp-checkpoint-v1"
EXPORT_FORMAT = "mlp-numpy-export-v1"
CHECKED_FIELDS = ("teacher", "input_dim", "feature_names", "interface", "normaliser_sha256", "objective", "run_id",
                  "task_context")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def model_spec(model) -> dict:
    return {"input_dim": model.input_dim, "hidden": list(model.hidden), "layernorm_eps": model.layernorm_eps}


def model_payload(model, normaliser, meta: dict, step: int, **state) -> dict:
    """Checkpoint contents; state may add optimizer_state, scheduler_state, rng, config, provenance, environment."""
    meta = dict(meta, normaliser_sha256=normaliser.sha256(), model=model_spec(model))
    return {"format": FORMAT, "step": int(step), "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "normaliser": normaliser.to_dict(), "meta": meta, **state}


def save_checkpoint(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def check_expected(meta: dict, expected: dict | None, what: str) -> None:
    for field, value in (expected or {}).items():
        if field not in CHECKED_FIELDS:
            raise ValueError(f"unknown compatibility field {field!r}")
        if meta.get(field) != value:
            raise ValueError(f"{what} {field} mismatch: stored {meta.get(field)!r}, expected {value!r}")


def load_checkpoint(path: Path, expected: dict | None = None) -> dict:
    path = Path(path)
    if path.suffix != ".pt":
        raise ValueError(f"{path}: checkpoints are .pt files")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} checkpoint")
    check_expected(payload["meta"], expected, "checkpoint")
    return payload


def export_numpy(path: Path, model, normaliser, metadata: dict) -> dict:
    """Write Linear/LayerNorm weights, epsilon and scales explicitly; metadata must name its source checkpoint."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("NumPy exports are .npz files")
    source = Path(metadata["source_checkpoint"])
    info = {"source_checkpoint": str(source), "source_step": int(metadata["source_step"]),
            "source_sha256": sha256_file(source)}
    meta = {k: v for k, v in metadata.items() if k not in ("source_checkpoint", "source_step")}
    meta.update(info, format=EXPORT_FORMAT, model=model_spec(model), normaliser=normaliser.to_dict(),
                normaliser_sha256=normaliser.sha256())
    arrays, i = {}, 0
    layers = list(model.trunk)
    for k in range(0, len(layers), 3):
        lin, ln = layers[k], layers[k + 1]
        arrays[f"layer{i}_W"] = lin.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"layer{i}_b"] = lin.bias.detach().cpu().numpy().astype(np.float64)
        arrays[f"ln{i}_gamma"] = ln.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"ln{i}_beta"] = ln.bias.detach().cpu().numpy().astype(np.float64)
        i += 1
    for head in ("torque", "wheel"):
        lin = getattr(model, f"{head}_head")
        arrays[f"{head}_W"] = lin.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"{head}_b"] = lin.bias.detach().cpu().numpy().astype(np.float64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, metadata_json=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    return info
