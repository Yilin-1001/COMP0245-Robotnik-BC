"""Clean-data BC+J: action loss plus damping-Jacobian loss (objective bcj), through the shared engine.

Two phases (research §4.4):
1. --calibrate: at the selected architecture's seed-0 initialisation, on a fixed fitting-only batch, set
   lambda_J = 0.1 * L_action / L_J and save the record (never overwritten).
2. training: --set loss.jacobian.calibration=<record> freezes that lambda_J for seeds 0-2 (and any refit).
Each BC+J run starts from the same seeded initial weights as its plain-BC pair; it never continues a trained
baseline checkpoint.

Usage (from final/robotnik_arm_task/):
    python -m training.jacobian.train --teacher A --set identity.architecture=m2 --calibrate ...
    python -m training.jacobian.train --teacher A --set loss.jacobian.calibration=<record.json> --set training.lr=... \
        --set training.updates=... --set loss.action=...
"""
import copy
import hashlib
import json
from pathlib import Path

from training.baseline.train import build_parser, run, summary
from training.common.config import TASK_DIR, parse_overrides, resolve_config, validate_config

DEFAULT_CONFIG = TASK_DIR / "experiments/configs/jacobian.yaml"
REFERENCE_SEED = 0


def apply_calibration(config: dict, normaliser_sha256: str) -> dict:
    """Attach the frozen calibration record; it must match the teacher, architecture and (selection) scales."""
    jac = config["loss"]["jacobian"]
    if not jac.get("calibration"):
        raise ValueError("BC+J training requires a frozen calibration record: run --calibrate first, then "
                         "--set loss.jacobian.calibration=<record.json>")
    path = Path(jac["calibration"])
    path = path if path.is_absolute() else TASK_DIR / path
    record = json.loads(path.read_text())
    ident = config["identity"]
    for field in ("teacher", "architecture"):
        if record[field] != ident[field]:
            raise ValueError(f"calibration {field} {record[field]!r} != run {field} {ident[field]!r}")
    if ident["fit_stage"] == "selection" and record["normaliser_sha256"] != normaliser_sha256:
        raise ValueError("calibration was made with different fitting-only normaliser scales")
    if jac.get("lambda_j") is not None and jac["lambda_j"] != record["lambda_j"]:
        raise ValueError("loss.jacobian.lambda_j disagrees with the frozen calibration record")
    out = copy.deepcopy(config)
    out["loss"]["jacobian"]["lambda_j"] = float(record["lambda_j"])
    out["loss"]["jacobian"]["calibration_record"] = {"path": str(path), "lambda_j": float(record["lambda_j"]),
                                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return out


def objective_from(config: dict):
    from training.jacobian.loss import JacobianObjective
    loss = config["loss"]
    jac = loss["jacobian"]
    return JacobianObjective(loss["action"], jac["lambda_j"], loss["epsilon_tau"], jac["subbatch_size"], jac["kd"],
                             config["interface"]["torque_limit"], loss["huber_delta"])


def calibrate(args) -> dict:
    import torch
    from training.common.engine import init_model, prepare_run, stream_seed
    from training.common.dataset import sample_indices
    from training.jacobian.loss import calibrate_lambda
    config = resolve_config(Path(args.config), args.teacher, parse_overrides(args.set))
    if config["identity"]["seed"] != REFERENCE_SEED or config["identity"]["fit_stage"] != "selection":
        raise ValueError("calibration uses the reference seed-0 initialisation of a selection run")
    if config["loss"]["action"] is None:
        raise ValueError("calibrate after freezing loss.action (the ratio uses the chosen action loss)")
    prep = prepare_run(config, workers=args.workers, finalise=False)
    normaliser, train = prep["normaliser"], prep["train"]
    g = torch.Generator().manual_seed(stream_seed(REFERENCE_SEED, "calibration"))
    idx = sample_indices(train, config["training"]["batch_size"], g, torch.device("cpu")).numpy()
    x = torch.as_tensor(normaliser.transform_x(train.x[idx]), dtype=torch.float32)
    y = torch.as_tensor(normaliser.transform_y(train.y[idx]), dtype=torch.float32)
    jac, loss = config["loss"]["jacobian"], config["loss"]
    record = calibrate_lambda(init_model(config), x, y, normaliser, {
        "kind": loss["action"], "huber_delta": loss["huber_delta"], "epsilon_tau": loss["epsilon_tau"], "kd": jac["kd"],
        "torque_limit": config["interface"]["torque_limit"], "calibration_ratio": jac["calibration_ratio"]})
    ident = config["identity"]
    record.update(teacher=ident["teacher"], architecture=ident["architecture"], cohort=ident["cohort"],
                  reference_seed=REFERENCE_SEED, action_kind=loss["action"], batch_rule="sample_indices with "
                  "stream 'calibration' of seed 0, batch_size rows of the fitting cache",
                  split_sha256=prep["provenance"]["split_sha256"], code_digest=prep["provenance"]["code_digest"])
    body = json.dumps(record, indent=1, sort_keys=True)
    root = Path(config["outputs"]["checkpoint_root"])
    root = root if root.is_absolute() else TASK_DIR / root
    path = root / "calibration" / ident["teacher"] / ident["architecture"] / \
        f"calibration-{hashlib.sha256(body.encode()).hexdigest()[:12]}.json"
    if path.exists():
        if path.read_text() != body:
            raise FileExistsError(f"{path} exists with different content")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return {"calibration": str(path.relative_to(TASK_DIR) if path.is_relative_to(TASK_DIR) else path), **record}


def prepare_from_args(args) -> dict:
    """Resolve, require the frozen calibration, prepare data and finalise the BC+J identity."""
    from training.common.engine import prepare_run
    if Path(args.config).name != "jacobian.yaml":
        raise ValueError(f"this entry point runs jacobian.yaml only; got {args.config}")
    config = resolve_config(Path(args.config), args.teacher, parse_overrides(args.set))
    if not config["loss"]["jacobian"].get("calibration"):
        apply_calibration(config, "")  # raises the explanatory calibration error
    probe = copy.deepcopy(config)
    probe["loss"]["jacobian"]["lambda_j"] = probe["loss"]["jacobian"].get("lambda_j") or 0.0
    validate_config(probe, ready_to_train=True)
    return prepare_run(config, workers=args.workers, before_finalise=lambda c, n: apply_calibration(c, n.sha256()))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Clean-data MLP BC+J: action loss plus the damping-Jacobian loss (objective bcj).",
                          DEFAULT_CONFIG)
    parser.add_argument("--calibrate", action="store_true", help="fitting-only lambda_J calibration at seed-0 init")
    args = parser.parse_args(argv)
    if Path(args.config).name != "jacobian.yaml":
        raise ValueError(f"this entry point runs jacobian.yaml only; got {args.config}")
    if args.sanity_episode:
        raise ValueError("the one-episode sanity run belongs to the baseline launcher")
    if args.calibrate:
        print(json.dumps(calibrate(args), indent=1))
        return 0
    prep = prepare_from_args(args)
    if args.dry_run:
        print(json.dumps(dict(summary(prep), lambda_j=prep["config"]["loss"]["jacobian"]["lambda_j"]), indent=1))
        return 0
    print(json.dumps(run(prep, objective_from(prep["config"]), args.resume), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
