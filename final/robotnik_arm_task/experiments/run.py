"""Reproducible clean-data MLP experiment stages over the existing launchers and the shared evaluator.

Stages (docs/mlp_training_research.md §4, docs/mlp_baseline_training_plan.md §5–7):
  pilot     two constant-LR M2-A Huber fits (1e-3, 3e-4), 50k updates, offline validation every 1k
  loss      matched Huber/MSE M2-A fits at the frozen budget and pilot LR (warm-up + cosine)
  baseline  first verified M2-A and M2-B baselines with 8- and 32-scenario rollouts
  screen    M1–M6 x {A, B} x 3 LRs (seed 0); offline LR choice; rollouts only for the chosen LR
  seeds     seeds 1–2 for the top two architectures per teacher
  jacobian  BC+J for each teacher's selected architecture, seeds 0–2, paired with plain BC
  compare223  optional reduced-data comparison of the selected architectures
  refit / test  the frozen final protocol (training.common refit mode, sealed test scenarios)

The runner resolves configurations and calls the launchers and evaluator; it never trains by itself. Numerical
decisions are stored as immutable experiments/protocols/<protocol_id>.yaml files with evidence hashes. A stage
that lacks a required decision stops with BlockedError rather than inventing a default.

Usage (from final/robotnik_arm_task/):
    python -m experiments.run --stage pilot --dry-run
    python -m experiments.run --stage screen --protocol experiments/protocols/<id>.yaml
    python -m experiments.run --freeze budget.updates=30000 --freeze budget.pilot_lr=3.0e-4 \
        --evidence checkpoints/.../metrics.csv --protocol experiments/protocols/<parent>.yaml
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml

TASK_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = TASK_DIR / "experiments" / "protocols"
ARCHITECTURES = ["m1", "m2", "m3", "m4", "m5", "m6"]
TEACHERS = ["A", "B"]
LR_GRID = [1e-3, 3e-4, 1e-4]
PILOT = {"updates": 50000, "lrs": [1e-3, 3e-4], "log_every": 1000, "checkpoint_every": 10000}
FINALIST_SEEDS = [1, 2]
PAIRED_SEEDS = [0, 1, 2]
STAGES = ("pilot", "loss", "baseline", "screen", "seeds", "jacobian", "compare223", "refit", "test")


class BlockedError(RuntimeError):
    """A stage needs a reviewed decision (or evidence) that the protocol does not contain yet."""


# ---------------------------------------------------------------- protocol

def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path: Path | None) -> dict:
    from training.common.config import validate_protocol
    if path is None:
        return {"protocol_id": None, "decisions": {}, "evidence": {}, "parent": None}
    protocol = yaml.safe_load(Path(path).read_text())
    validate_protocol(protocol)
    return protocol


def write_frozen_protocol(path: Path, decisions: dict, evidence: list[Path], parent: str | None = None) -> Path:
    """New immutable protocol file in directory path; an identical or existing protocol is never rewritten."""
    directory = Path(path)
    ev = {str(Path(p).resolve().relative_to(TASK_DIR)) if Path(p).resolve().is_relative_to(TASK_DIR) else str(p):
          sha256_file(p) for p in evidence}
    from training.common.config import protocol_id_for
    body = {"decisions": decisions, "evidence": ev, "parent": parent}
    pid = protocol_id_for(body)
    out = directory / f"{pid}.yaml"
    if out.exists():
        raise FileExistsError(f"{out} exists: frozen protocols are never rewritten")
    directory.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump({"protocol_id": pid, **body}, sort_keys=False))
    os.replace(tmp, out)
    return out


def extend_protocol(protocol: dict, updates: dict, evidence: list[Path]) -> Path:
    """New protocol = parent decisions plus updates (existing decisions may not be changed silently)."""
    decisions = json.loads(canonical(protocol.get("decisions") or {}))
    for dotted, value in updates.items():
        node, parts = decisions, dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if parts[-1] in node and node[parts[-1]] != value:
            raise ValueError(f"decision {dotted} is already frozen as {node[parts[-1]]!r}; start from its parent")
        node[parts[-1]] = value
    return write_frozen_protocol(PROTOCOL_DIR, decisions, evidence, parent=protocol.get("protocol_id"))


def _need(decisions: dict, *keys):
    node = decisions
    for k in keys:
        if not isinstance(node, dict) or node.get(k) is None:
            raise BlockedError(f"protocol lacks the reviewed decision {'.'.join(keys)}; freeze it with evidence first")
        node = node[k]
    return node


# ---------------------------------------------------------------- job matrices

def _job(stage, teacher, arch, seed, overrides, launcher="baseline", objective="bc", cohort="clean253"):
    over = {"identity.architecture": arch, "identity.seed": seed, "identity.cohort": cohort, **overrides}
    return {"stage": stage, "launcher": launcher, "teacher": teacher, "overrides": over,
            "identity": {"regime": "clean", "objective": objective, "teacher": teacher, "cohort": cohort,
                         "architecture": arch, "seed": seed, "fit_stage": "selection"}}


def build_jobs(stage: str, protocol: dict) -> list[dict]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; DART/DAgger are separate later workflows")
    d = protocol.get("decisions") or {}
    if stage == "pilot":
        return [_job(stage, "A", "m2", 0, {"training.lr": lr, "training.updates": PILOT["updates"],
                                           "training.schedule": "constant", "loss.action": "huber",
                                           "training.log_every": PILOT["log_every"],
                                           "training.checkpoint_every": PILOT["checkpoint_every"]})
                for lr in PILOT["lrs"]]
    updates, pilot_lr = _need(d, "budget", "updates"), _need(d, "budget", "pilot_lr")
    cosine = {"training.updates": updates, "training.schedule": "cosine"}
    if stage == "loss":
        return [_job(stage, "A", "m2", 0, {**cosine, "training.lr": pilot_lr, "loss.action": kind})
                for kind in ("huber", "mse")]
    action = _need(d, "loss", "action")
    base = {**cosine, "loss.action": action}
    if stage == "baseline":
        return [_job(stage, t, "m2", 0, {**base, "training.lr": pilot_lr}) for t in TEACHERS]
    if stage == "screen":
        return [_job(stage, t, a, 0, {**base, "training.lr": lr}) for t in TEACHERS for a in ARCHITECTURES for lr in LR_GRID]
    lr = lambda t, a: _need(d, "lr", t, a)
    if stage == "seeds":
        return [_job(stage, t, a, s, {**base, "training.lr": lr(t, a)})
                for t in TEACHERS for a in _need(d, "finalists", t) for s in FINALIST_SEEDS]
    if stage == "jacobian":
        return [_job(stage, t, _need(d, "selected", t), s,
                     {**base, "training.lr": lr(t, _need(d, "selected", t)),
                      "loss.jacobian.calibration": _need(d, "calibration", t)}, launcher="jacobian", objective="bcj")
                for t in TEACHERS for s in PAIRED_SEEDS]
    if stage == "compare223":
        return [_job(stage, t, _need(d, "selected", t), 0, {**base, "training.lr": lr(t, _need(d, "selected", t))},
                     cohort="clean223") for t in TEACHERS]
    if stage == "refit":
        refit = _need(d, "refit")
        jobs = []
        for t in TEACHERS:
            arch = _need(d, "selected", t)
            over = {**base, "training.lr": lr(t, arch), "identity.fit_stage": "refit"}
            for objective in _need(d, "refit", "objectives"):
                for seed in _need(d, "refit", "seeds"):
                    extra = {"loss.jacobian.calibration": _need(d, "calibration", t)} if objective == "bcj" else {}
                    job = _job(stage, t, arch, seed, {**over, **extra}, launcher="jacobian" if objective == "bcj"
                               else "baseline", objective=objective)
                    job["identity"]["fit_stage"] = "refit"
                    jobs.append(job)
        return jobs
    if not d.get("final"):
        raise BlockedError(f"stage {stage} requires a frozen final protocol (decisions.final)")
    from evaluation.evaluate import validate_final_protocol
    validate_final_protocol(protocol, index_path=TASK_DIR / "results" / "results.csv")
    return [{"stage": "test", "launcher": "evaluate", "teacher": a["teacher"], "artifact": a, "overrides": {},
             "identity": {"regime": "clean", "objective": a["objective"], "teacher": a["teacher"],
                          "architecture": a["architecture"], "seed": a["seed"], "fit_stage": a["fit_stage"]}}
            for a in d["final"]["artifacts"]]


def reuse_allowed(existing: dict, new: dict) -> bool:
    """An existing run is reused only for an identical full configuration hash."""
    return existing["identity"]["config_hash"] == new["identity"]["config_hash"]


# ---------------------------------------------------------------- ranking

def rank_offline(m: dict) -> tuple:
    """Lower is better: hold-bias gate failure, maximum absolute hold bias, arm RMSE, wheel RMSE."""
    bias = m["hold_bias"]
    inf = float("inf")
    return (0 if bias["gate_pass"] else 1, bias["max_abs_nm"] if bias["max_abs_nm"] is not None else inf,
            m["arm_rmse_nm"] if m["arm_rmse_nm"] is not None else inf,
            m["wheel_rmse_rad_s"] if m["wheel_rmse_rad_s"] is not None else inf)


def rank_rollout(s: dict) -> tuple:
    """Lower is better: sustained hold, reach, final error, saturation + command variation, earlier update."""
    f = lambda k: float(s[k]) if s.get(k) not in (None, "") else float("inf")
    return (-f("sustained_hold_success"), -f("reach_success"), f("median_final_err_mm"),
            f("mean_saturation_frac") + f("mean_torque_variation_nm"), int(s.get("checkpoint_step") or 0))


def offline_proxy_from_metrics(run_dir: Path) -> dict:
    """Terminal offline validation row of a run, in the shape rank_offline expects."""
    rows = list(csv.DictReader(open(Path(run_dir) / "metrics.csv")))
    last = rows[-1]
    num = lambda k: float(last[k]) if last[k] not in ("", "None") else None
    return {"update": int(last["update"]), "arm_rmse_nm": num("val_arm_rmse_nm"),
            "wheel_rmse_rad_s": num("val_wheel_rmse_rad_s"),
            "hold_bias": {"gate_pass": last["val_hold_bias_gate_pass"] == "True",
                          "max_abs_nm": num("val_hold_bias_max_abs_nm")}}


def plateau_update(rows: list[dict], key: str, tol: float = 0.05) -> int | None:
    """First logged update after which `key` stays within (1 + tol) of the curve's best value; None if never."""
    vals = [(int(r["update"]), float(r[key])) for r in rows if r.get(key) not in ("", "None", None)]
    if not vals:
        return None
    best = min(v for _, v in vals)
    for i, (u, _) in enumerate(vals):
        if all(v <= best * (1 + tol) for _, v in vals[i:]):
            return u if i < len(vals) - 1 else None  # the last point alone is not a plateau
    return None


def param_count(arch: str, input_dim: int) -> int:
    from training.common.models import MLP
    from training.common.config import resolve_config
    hidden = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", "A", {})["model"]["architectures"][arch]
    return sum(p.numel() for p in MLP(input_dim, hidden).parameters())


# ---------------------------------------------------------------- execution

def _launcher(job):
    if job["launcher"] == "jacobian":
        from training.jacobian import train as module
        prepare = module.prepare_from_args
    else:
        from training.baseline import train as module
        prepare = lambda args: module.prepare_from_args(args, "baseline.yaml")
    return module, prepare


def _args(job, workers):
    module, _ = _launcher(job)
    argv = ["--teacher", job["teacher"], "--workers", str(workers)]
    for k, v in job["overrides"].items():
        argv += ["--set", f"{k}={v}"]
    default = "jacobian.yaml" if job["launcher"] == "jacobian" else "baseline.yaml"
    parser = module.build_parser("", TASK_DIR / "experiments/configs" / default)
    if job["launcher"] == "jacobian":
        parser.add_argument("--calibrate", action="store_true")
    return parser.parse_args(argv)


def run_job(job: dict, workers: int = 16) -> dict:
    """Train a job, reusing a completed run of the identical configuration or resuming an interrupted one."""
    from training.common.config import run_dir_for
    module, prepare = _launcher(job)
    prep = prepare(_args(job, workers))
    config = prep["config"]
    run_dir = run_dir_for(config)
    terminal = run_dir / f"step_{config['training']['updates']}.pt"
    if (run_dir / "config.yaml").exists():
        existing = yaml.safe_load((run_dir / "config.yaml").read_text())
        if not reuse_allowed(existing, config):
            raise ValueError(f"{run_dir} holds a different configuration")
        if terminal.exists():
            return {"run_dir": run_dir, "config": config, "status": "reused"}
    objective = module.objective_from(config) if job["launcher"] == "jacobian" else _action_objective(config)
    out = module.run(prep, objective, resume=(run_dir / "latest.pt").exists())
    return {"run_dir": run_dir, "config": config, "status": out["status"]}


def _action_objective(config):
    from training.baseline.loss import ActionObjective
    return ActionObjective(config["loss"]["action"], config["loss"]["huber_delta"])


def _index_rows() -> list[dict]:
    path = TASK_DIR / "results" / "results.csv"
    return list(csv.DictReader(open(path))) if path.exists() else []


def evaluate_checkpoint(run_dir: Path, config: dict, step: int, split: str, workers: int) -> dict:
    """Closed-loop summary of one checkpoint on a selection split, recorded once in the result index."""
    import functools
    from evaluation.evaluate import (EVALUATION_ID, append_results, evaluate_policy, mlp_factory, result_row,
                                     summarise, write_details)
    from evaluation.test_sets import get_split
    ident = config["identity"]
    for row in _index_rows():
        if (row["run_id"], row["checkpoint_step"], row["evaluation_split"], row["evaluation_id"]) == \
                (ident["run_id"], str(step), split, EVALUATION_ID):
            return row
    export = Path(run_dir) / f"step_{step}.npz"
    manifest = TASK_DIR / config["data"]["split_manifest"]
    scenarios = get_split(split, manifest, ident["teacher"])
    cfg = json.loads(json.dumps(config))
    cfg["evaluation"]["workers"] = workers
    factory = functools.partial(mlp_factory, str(export), "numpy", {"run_id": ident["run_id"]})
    outcomes = evaluate_policy(factory, scenarios, cfg, workers=workers)
    summary = summarise(outcomes)
    detail = TASK_DIR / config["outputs"]["results_root"] / ident["regime"] / ident["teacher"] / ident["objective"] / \
        ident["run_id"] / f"rollouts_{split}_step{step}_{EVALUATION_ID}.csv"
    write_details(detail, outcomes)
    with np_load(export) as z:
        meta = json.loads(str(z["metadata_json"]))
    row = result_row(ident, summary, lr=config["training"]["lr"], split_sha256=sha256_file(manifest),
                     checkpoint_step=step, checkpoint_path=meta["source_checkpoint"],
                     checkpoint_sha256=meta["source_sha256"], inference_path=str(export.relative_to(TASK_DIR)),
                     evaluation_split=split, detail_path=str(detail.relative_to(TASK_DIR)))
    append_results(TASK_DIR / config["outputs"]["results_index"], [row])
    return row


def np_load(path):
    import numpy as np
    return np.load(path)


def select_run(run: dict, workers: int) -> dict:
    """8-scenario rollouts of every saved checkpoint, rank, then the chosen checkpoint on all 32 scenarios."""
    from training.common.engine import select_checkpoint
    run_dir, config = Path(run["run_dir"]), run["config"]
    steps = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("step_*.pt"))
    screening = [dict(evaluate_checkpoint(run_dir, config, s, "screening", workers), checkpoint_step=s) for s in steps]
    best = min(screening, key=rank_rollout)
    step = int(best["checkpoint_step"])
    validation = evaluate_checkpoint(run_dir, config, step, "validation", workers)
    if not (run_dir / "selection.json").exists():
        select_checkpoint(run_dir, step, {"screening": screening, "validation": validation,
                                          "rule": "rank_rollout on 8 screening scenarios, earlier update breaks ties"})
    return {"run_id": config["identity"]["run_id"], "step": step, "screening": best, "validation": validation,
            "run_dir": str(run_dir), "config": config}


def run_stage(stage: str, protocol: dict, workers: int, teachers=None) -> list[dict]:
    jobs = [j for j in build_jobs(stage, protocol) if teachers is None or j["teacher"] in teachers]
    runs = []
    for job in jobs:
        print(f"[{stage}] {job['launcher']} {job['teacher']} {job['overrides']}", flush=True)
        runs.append(dict(run_job(job, workers), job=job))
        print(f"    -> {runs[-1]['status']} {runs[-1]['config']['identity']['run_id']}", flush=True)
    return runs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="clean-data MLP experiment stages (no test access, no DART/DAgger)")
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--protocol", type=Path, help="frozen protocol with the decisions this stage needs")
    p.add_argument("--dry-run", action="store_true", help="print the job matrix without training")
    p.add_argument("--select", action="store_true", help="after training, run checkpoint rollouts and selection")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--teachers", nargs="+", choices=TEACHERS)
    p.add_argument("--freeze", action="append", default=[], metavar="DECISION=VALUE", help="write a new protocol")
    p.add_argument("--evidence", action="append", default=[], type=Path)
    args = p.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.freeze:
        updates = {k: yaml.safe_load(v) for k, v in (f.split("=", 1) for f in args.freeze)}
        if not args.evidence:
            p.error("--freeze requires --evidence files")
        print(extend_protocol(protocol, updates, args.evidence))
        return 0
    if not args.stage:
        p.error("--stage or --freeze is required")
    jobs = build_jobs(args.stage, protocol)
    if args.stage == "test" and not args.dry_run:
        from evaluation.evaluate import evaluate_final
        for row in evaluate_final(protocol, args.workers):
            print(json.dumps({k: row[k] for k in ("run_id", "teacher", "objective", "sustained_hold_success",
                                                   "reach_success", "median_final_err_mm", "failures")}))
        return 0
    if args.dry_run:
        for j in jobs:
            print(json.dumps({k: j[k] for k in ("launcher", "teacher", "identity", "overrides")}, default=str))
        print(f"{len(jobs)} jobs; outputs under checkpoints/clean/<teacher>/<objective>/<run_id>/")
        return 0
    runs = run_stage(args.stage, protocol, args.workers, args.teachers)
    if args.select:
        for r in runs:
            s = select_run(r, args.workers)
            print(json.dumps({"run_id": s["run_id"], "step": s["step"],
                              "screening_hold": s["screening"]["sustained_hold_success"],
                              "validation_hold": s["validation"]["sustained_hold_success"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
