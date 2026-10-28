"""DAgger-only validation and content identity without relaxing the clean launcher."""
import copy
import hashlib

from training.common.config import canonical_json, scientific_view, validate_config


def validate(config, ready_to_train=False):
    ident, d = config["identity"], config.get("dagger", {})
    if ident["regime"] != "dagger" or ident["teacher"] != "A" or ident["objective"] != "bc" or ident["fit_stage"] != "selection":
        raise ValueError("DAgger entry point accepts A plain BC selection only")
    if d.get("round") != 1:
        raise ValueError("only DAgger R1 is authorized")
    for key in ("parent_sha256", "parent_normaliser_sha256", "round_manifest_sha256", "protocol_sha256"):
        if len(str(d.get(key, ""))) != 64:
            raise ValueError(f"missing DAgger provenance {key}")
    if config["provenance"].get("normaliser_sha256") != d["parent_normaliser_sha256"]:
        raise ValueError("DAgger normaliser must remain frozen to its parent")
    if config["loss"]["action"] != "huber" or config["loss"]["huber_delta"] != 1 or (
            config["loss"].get("arm_weight"), config["loss"].get("wheel_weight")) != (.8, .2):
        raise ValueError("R1 uses Huber1 with arm/wheel .8/.2")
    clean = copy.deepcopy(config)
    clean["identity"]["regime"] = "clean"
    validate_config(clean, ready_to_train)
    if ready_to_train and not config["provenance"].get("source_sha256"):
        raise ValueError("DAgger source hashes are required")
    stored = ident.get("config_hash")
    if stored and stored != hashlib.sha256(canonical_json(scientific_view(config)).encode()).hexdigest():
        raise ValueError("DAgger config changed after identity freeze")


def finalise(config):
    config = copy.deepcopy(config)
    for key in ("config_hash", "config_id", "run_id"):
        config["identity"].pop(key, None)
    validate(config, ready_to_train=True)
    digest = hashlib.sha256(canonical_json(scientific_view(config)).encode()).hexdigest()
    config["identity"].update(config_hash=digest, config_id=digest[:12],
                              run_id=f"dagger-A-bc-r1-{config['identity']['architecture']}-s{config['identity']['seed']}-{digest[:12]}")
    return config
