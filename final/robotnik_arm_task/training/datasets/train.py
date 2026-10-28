"""Train an MLP on published feature datasets (the training-method ablation), through the shared engine.

Resolution: common.yaml -> objectives/<objective>.yaml -> datasets/base.yaml -> datasets/variants/<row>.yaml
-> --set overrides. Fitting rows: every listed dataset restricted to the frozen split's fitting scenarios (non-clean
datasets must hold fitting scenarios only); validation: the 32 frozen validation scenarios of data.validation_dataset.
Scales are fitted on the fitting rows, each row counted once whatever the mix. bcj calibrates lambda_J at the seed-0
initialisation on a fixed fitting batch (the clean launcher's rule), so every seed of one row shares it.

Runs are named <row>__<architecture>__s<seed>__<version>__<date> (architecture = registry name plus shape, e.g.
MLP01-256x3) and stored in checkpoints/ablation/<date>__<experiment>-<version>/; the terminal model is also saved as
<run>.npz and <run>.pt in the run directory.

Usage (from final/robotnik_arm_task/):
    python -m training.datasets.train --variant R4-dart-gate --set identity.version=v1 --set identity.date=2026-10-28 \
        --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber \
        --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5 [--set identity.seed=1] [--dry-run] [--resume]
"""
import argparse
import copy
import datetime
import hashlib
import json
import re
import shutil

import yaml

from training.common.config import (TASK_DIR, _set_dotted, action_weights, canonical_json, check_interface, merge,
                                    parse_overrides, run_dir_for, scientific_view)

CONFIG_DIR = TASK_DIR / "experiments/configs"
OBJECTIVES = ("bc", "bcj")


def resolve(variant: str, overrides: dict) -> dict:
    load = lambda p: yaml.safe_load(p.read_text()) or {}
    paths = {"base": CONFIG_DIR / "datasets/base.yaml", "variant": CONFIG_DIR / "datasets/variants" / f"{variant}.yaml"}
    if not paths["variant"].is_file():
        raise FileNotFoundError(f"unknown variant {variant!r}: {paths['variant']} does not exist")
    objective = overrides.get("identity.objective", merge(load(paths["base"]), load(paths["variant"]))["identity"]["objective"])
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    layers = [("common.yaml", CONFIG_DIR / "common.yaml"), (f"objectives/{objective}.yaml", CONFIG_DIR / "objectives" / f"{objective}.yaml"),
              ("datasets/base.yaml", paths["base"]), (f"datasets/variants/{variant}.yaml", paths["variant"])]
    config = {}
    for _, p in layers:
        config = merge(config, load(p))
    # The ablation's own architecture registry replaces the clean pipeline's m1-m6 names.
    config["model"]["architectures"] = copy.deepcopy(load(paths["base"])["model"]["architectures"])
    for key, value in overrides.items():
        _set_dotted(config, key, value)
    if config["identity"].get("date") is not None:  # YAML reads 2026-10-28 as a date; names and hashes use the string
        config["identity"]["date"] = str(config["identity"]["date"])
    from core.feature_sets import input_dim
    config["variant"] = variant
    config["model"]["input_dim"] = input_dim("A", config["data"]["feature_definition"])
    config["layers"] = [name for name, _ in layers] + ["cli"]
    config["template_sha256"] = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in layers}
    validate(config)
    return config


def validate(config: dict, ready_to_train: bool = False) -> None:
    from core.feature_sets import input_dim
    ident, data, loss, training = config["identity"], config["data"], config["loss"], config["training"]
    if ident["regime"] != "ablation" or ident["teacher"] != "A" or ident["fit_stage"] != "selection":
        raise ValueError("dataset runs are teacher-A selection runs of regime 'ablation'")
    if ident["objective"] not in OBJECTIVES or (loss.get("jacobian") is None) != (ident["objective"] == "bc"):
        raise ValueError("objective must be bc (no jacobian settings) or bcj (with them)")
    if not re.fullmatch(r"MLP\d{2}", str(ident["architecture"])) or ident["architecture"] not in config["model"]["architectures"]:
        raise ValueError(f"architecture must be a registered MLP<NN> name, got {ident['architecture']!r}")
    definition = data["feature_definition"]
    if config["model"]["input_dim"] != input_dim("A", definition):
        raise ValueError("model.input_dim disagrees with the feature definition")
    names = list(data["datasets"] or [])
    if not names or len(set(names)) != len(names):
        raise ValueError("data.datasets must list distinct published dataset names")
    for name in names + [data["validation_dataset"]]:
        if not isinstance(name, str) or not name.endswith(f"__{definition}"):
            raise ValueError(f"dataset {name!r} does not use the feature definition {definition!r}")
    if data["mix"] is not None and (len(data["mix"]) != len(names) or any(s <= 0 for s in data["mix"])
                                    or abs(sum(data["mix"]) - 1.0) > 1e-9):
        raise ValueError("data.mix must be null or one positive share per dataset, summing to 1")
    action_weights(loss)
    if not ready_to_train:
        return
    for key, value in (("training.lr", training["lr"]), ("training.updates", training["updates"]), ("loss.action", loss["action"]),
                       ("identity.version", ident.get("version")), ("identity.date", ident.get("date"))):
        if value is None:
            raise ValueError(f"{key} must be set (--set {key}=...) before training")
    if not re.fullmatch(r"v\d+", str(ident["version"])):
        raise ValueError(f"identity.version must look like v1, got {ident['version']!r}")
    datetime.date.fromisoformat(str(ident["date"]))
    if ident["objective"] == "bcj" and loss["jacobian"].get("lambda_j") is None:
        raise ValueError("bcj needs lambda_J; prepare() calibrates it")


def _ids(split: dict, dotted: str) -> list:
    node = split
    for part in dotted.split("."):
        node = node[part]
    return list(node)


def calibrate(config: dict, train, normaliser) -> dict:
    """lambda_J = ratio * L_action / L_J at the seed-0 initialisation on a fixed fitting batch."""
    import torch
    from training.common.dataset import sample_indices
    from training.common.engine import init_model, stream_seed
    from training.jacobian.loss import calibrate_lambda
    ref = copy.deepcopy(config)
    ref["identity"]["seed"] = 0
    g = torch.Generator().manual_seed(stream_seed(0, "calibration"))
    idx = sample_indices(train, config["training"]["batch_size"], g, torch.device("cpu")).numpy()
    x = torch.as_tensor(normaliser.transform_x(train.x[idx]), dtype=torch.float32)
    y = torch.as_tensor(normaliser.transform_y(train.y[idx]), dtype=torch.float32)
    loss, jac = config["loss"], config["loss"]["jacobian"]
    return calibrate_lambda(init_model(ref), x, y, normaliser, {
        "kind": loss["action"], "huber_delta": loss["huber_delta"], "epsilon_tau": loss["epsilon_tau"], "kd": jac["kd"],
        "torque_limit": config["interface"]["torque_limit"], "calibration_ratio": jac["calibration_ratio"],
        "arm_weight": loss["arm_weight"], "wheel_weight": loss["wheel_weight"]})


def prepare(config: dict) -> dict:
    """Load and select the datasets, fit scales, calibrate bcj and finalise the run identity."""
    import time
    from training.common.dataset import load_split, sha256_file
    from training.common.engine import _git, code_digest
    from training.common.normalisation import fit_normaliser
    from training.datasets.load import concat, load_dataset, select_scenarios
    config = copy.deepcopy(config)
    data = config["data"]
    split_path = TASK_DIR / data["split_manifest"]
    split = load_split(split_path)
    fit_ids, val_ids = _ids(split, data["fitting_ids"]), split["validation_ids"]
    parts, used = [], {}
    for name in data["datasets"]:
        full = load_dataset(name, data["dataset_root"])
        config["interface"] = check_interface(config["interface"], full.metadata["interface"])
        if full.metadata["source"]["kind"] != "clean" and not set(full.metadata["scenario_ids"]) <= set(fit_ids):
            raise ValueError(f"{name}: non-clean data outside the fitting scenarios")
        part = select_scenarios(full, fit_ids)
        if not part.episode_ids:
            raise ValueError(f"{name}: no fitting episodes")
        parts.append(part)
        used[name] = {"complete_sha256": full.metadata["complete_sha256"], "build_key": full.metadata["build_key"],
                      "episodes": len(part.episode_ids), "rows": int(len(part.x))}
    train = concat(parts, data["mix"])
    vfull = load_dataset(data["validation_dataset"], data["dataset_root"])
    validation = select_scenarios(vfull, val_ids)
    if sorted(validation.metadata["scenario_ids"]) != sorted(val_ids):
        raise ValueError(f"{data['validation_dataset']}: lacks some of the 32 validation scenarios")
    floors = {k: data[k] for k in ("input_scale_floor", "arm_scale_floor", "wheel_scale_floor")}
    normaliser = fit_normaliser(train, floors)
    provenance = {"datasets": used, "validation_dataset": {"name": data["validation_dataset"],
                                                          "complete_sha256": vfull.metadata["complete_sha256"]},
                  "fitting_ids": fit_ids, "validation_ids": val_ids, "split_sha256": sha256_file(split_path),
                  "normaliser_sha256": normaliser.sha256(), "code_digest": code_digest(),
                  "operational": {"git_head": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                                  "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    if config["identity"]["objective"] == "bcj":
        record = calibrate(config, train, normaliser)
        digest = hashlib.sha256(canonical_json(record).encode()).hexdigest()
        config["loss"]["jacobian"]["lambda_j"] = float(record["lambda_j"])
        config["loss"]["jacobian"]["calibration_record"] = {"path": "auto (seed-0 init, fitting batch)",
                                                           "lambda_j": float(record["lambda_j"]), "sha256": digest}
        provenance.update(calibration=record, calibration_sha256=digest)
    config["provenance"] = provenance
    validate(config, ready_to_train=True)
    full_hash = hashlib.sha256(canonical_json(scientific_view(config)).encode()).hexdigest()
    ident = config["identity"]
    ident["config_hash"], ident["config_id"] = full_hash, full_hash[:12]
    ident["run_id"] = run_name(config)
    config["outputs"]["experiment_dir"] = f"{ident['date']}__{config['outputs']['experiment']}-{ident['version']}"
    return {"config": config, "train": train, "validation": validation, "normaliser": normaliser}


def architecture_label(name: str, hidden: list) -> str:
    """MLP01 with [256, 256, 256] -> MLP01-256x3; unequal widths are listed, e.g. MLP02-256-128-64."""
    shape = f"{hidden[0]}x{len(hidden)}" if len(set(hidden)) == 1 else "-".join(str(w) for w in hidden)
    return f"{name}-{shape}"


def run_name(config: dict) -> str:
    """<row>__<architecture>__s<seed>__<version>__<date>, e.g. R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28."""
    ident = config["identity"]
    arch = architecture_label(ident["architecture"], config["model"]["architectures"][ident["architecture"]])
    return "__".join([config["variant"], arch, f"s{ident['seed']}", ident["version"], ident["date"]])


def summary(prep: dict) -> dict:
    c = prep["config"]
    return {"run_id": c["identity"]["run_id"], "run_dir": str(run_dir_for(c)), "variant": c["variant"],
            "feature_definition": c["data"]["feature_definition"], "input_dim": c["model"]["input_dim"],
            "datasets": c["provenance"]["datasets"], "mix": c["data"]["mix"],
            "fitting_episodes": len(prep["train"].episode_ids), "fitting_rows": int(len(prep["train"].x)),
            "validation_episodes": len(prep["validation"].episode_ids),
            "lambda_j": (c["loss"].get("jacobian") or {}).get("lambda_j"), "config_hash": c["identity"]["config_hash"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, help="a file name in experiments/configs/datasets/variants/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted overrides")
    parser.add_argument("--dry-run", action="store_true", help="load and check the data, print the identity only")
    parser.add_argument("--resume", action="store_true", help="continue this exact run from its latest.pt")
    args = parser.parse_args(argv)
    config = resolve(args.variant, parse_overrides(args.set))
    validate(config)
    prep = prepare(config)
    if args.dry_run:
        print(json.dumps(summary(prep), indent=1))
        return 0
    from training.baseline.train import objective_from as bc_objective
    from training.common.engine import fit, init_model
    from training.jacobian.train import objective_from as bcj_objective
    config = prep["config"]
    objective = (bcj_objective if config["identity"]["objective"] == "bcj" else bc_objective)(config)
    latest = run_dir_for(config) / "latest.pt" if args.resume else None
    result = fit(config, prep["train"], prep["validation"], init_model(config), prep["normaliser"], objective,
                 resume=latest, validator=validate)
    run_dir, run_id = run_dir_for(config), config["identity"]["run_id"]
    for ext in ("npz", "pt"):  # the terminal model under the run's own name
        shutil.copy2(run_dir / f"step_{result.terminal_update}.{ext}", run_dir / f"{run_id}.{ext}")
    print(json.dumps(dict(summary(prep), terminal_update=result.terminal_update, status=result.status,
                          final_model=str(run_dir / f"{run_id}.npz"),
                          checkpoints=sorted(k for k in result.checkpoints if isinstance(k, int))), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
