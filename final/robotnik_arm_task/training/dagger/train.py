"""Warm-start weights only; round optimizer and scheduler are owned by the shared engine."""
from training.common.checkpoint import load_checkpoint
from training.common.models import build_model
from training.common.normalisation import Normaliser


def warm_start(config, checkpoint, normaliser):
    payload = load_checkpoint(checkpoint, expected={"teacher": "A"})
    if Normaliser.from_dict(payload["normaliser"]).sha256() != normaliser.sha256() or normaliser.sha256() != config["dagger"]["parent_normaliser_sha256"]:
        raise ValueError("warm-start normaliser differs from the frozen parent")
    parent = payload["config"]
    architecture = config["identity"]["architecture"]
    if config["model"] != parent["model"] or architecture != parent["identity"]["architecture"]:
        raise ValueError("warm-start model architecture differs from its parent")
    model = build_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    return model
