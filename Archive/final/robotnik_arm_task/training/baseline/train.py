"""Clean-data plain BC: action loss only, through the shared engine.

Usage (from final/robotnik_arm_task/):
    python -m training.baseline.train --teacher A --set training.lr=3.0e-4 --set training.updates=30000 \
        --set loss.action=huber
    python -m training.baseline.train --teacher A ... --dry-run            # validate data and print the run ID
    python -m training.baseline.train --teacher A ... --resume             # continue the same run from latest.pt
    python -m training.baseline.train --teacher A ... --sanity-episode train-0001   # one-episode overfit check
"""
import argparse
import json
from pathlib import Path

from training.common.config import TASK_DIR, parse_overrides, resolve_config, run_dir_for, validate_config

DEFAULT_CONFIG = TASK_DIR / "experiments/configs/baseline.yaml"
SANITY_ROOT = "checkpoints/sanity"


def build_parser(description: str, default_config: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(default_config))
    p.add_argument("--teacher", choices=["A", "B"], required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="permitted dotted overrides")
    p.add_argument("--dry-run", action="store_true", help="validate sources/caches and print the identity only")
    p.add_argument("--resume", action="store_true", help="continue this exact run from its latest.pt")
    p.add_argument("--sanity-episode", help="separately identified overfit run on one fitting episode")
    p.add_argument("--workers", type=int, default=16, help="cache-building processes")
    return p


def prepare_from_args(args, workflow: str) -> dict:
    from training.common.engine import prepare_run
    if Path(args.config).name != workflow:
        raise ValueError(f"this entry point runs {workflow} only; got {args.config} "
                         f"(the baseline launcher accepts baseline.yaml, the Jacobian launcher jacobian.yaml)")
    overrides = parse_overrides(args.set)
    if args.sanity_episode:
        overrides.setdefault("outputs.checkpoint_root", SANITY_ROOT)
    config = resolve_config(Path(args.config), args.teacher, overrides)
    validate_config(config, ready_to_train=True)
    if config["identity"]["fit_stage"] == "refit" and args.sanity_episode:
        raise ValueError("a sanity run is a selection-stage diagnostic, not a refit")
    return prepare_run(config, [args.sanity_episode] if args.sanity_episode else None, workers=args.workers)


def summary(prep: dict) -> dict:
    c = prep["config"]
    return {"run_id": c["identity"]["run_id"], "run_dir": str(run_dir_for(c)), "fit_stage": c["identity"]["fit_stage"],
            "fitting_episodes": len(prep["train"].episode_ids),
            "validation_episodes": len(prep["validation"].episode_ids) if prep["validation"] is not None else 0,
            "fitting_rows": int(len(prep["train"].x)), "input_dim": int(prep["train"].x.shape[1]),
            "interface": c["interface"], "config_hash": c["identity"]["config_hash"]}


def run(prep: dict, objective, resume: bool) -> dict:
    from training.common.config import run_dir_for
    from training.common.engine import fit, init_model
    config = prep["config"]
    latest = run_dir_for(config) / "latest.pt" if resume else None
    result = fit(config, prep["train"], prep["validation"], init_model(config), prep["normaliser"], objective,
                 resume=latest)
    return dict(summary(prep), terminal_update=result.terminal_update, status=result.status,
                checkpoints=sorted(k for k in result.checkpoints if isinstance(k, int)))


def objective_from(config: dict):
    from training.baseline.loss import ActionObjective
    from training.common.config import action_weights
    loss = config["loss"]
    arm, wheel = action_weights(loss)
    return ActionObjective(loss["action"], loss["huber_delta"], arm_weight=arm, wheel_weight=wheel)


def main(argv: list[str] | None = None) -> int:
    args = build_parser("Clean-data MLP behaviour cloning, action loss only (objective bc).", DEFAULT_CONFIG).parse_args(argv)
    prep = prepare_from_args(args, "baseline.yaml")
    if args.dry_run:
        print(json.dumps(summary(prep), indent=1))
        return 0
    print(json.dumps(run(prep, objective_from(prep["config"]), args.resume), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
