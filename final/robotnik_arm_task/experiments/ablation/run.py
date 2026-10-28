"""Run one ablation manifest: train every (row, seed), evaluate on validation, hold-test, summarise.

Stages (each skips work that is already complete, so an interrupted run continues where it stopped):
  train     training.datasets.train per run, --parallel at a time (an existing latest.pt is resumed)
  validate  the terminal model on the 32 validation scenarios in each manifest mode: 'student' (the student drives
            everything; the primary metric) and 'arm' (teacher A drives the base); rows go to results/results.csv
  hold      evaluation.hold_test on the manifest's fitting scenarios
  summary   runs.csv (one row per model) and summary.csv / summary.md (per row: per-seed values, pooled Wilson CI)
  demo      experiments/demo_models/<experiment dir>.yaml, the hash-pinned models evaluation.live_compare loads
Outputs: checkpoints/ablation/<date>__<experiment>-<version>/<run>/ and results/ablation/<same>/.

Usage (from final/robotnik_arm_task/):
    python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml [--stage all]
"""
import argparse
import csv
import functools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TASK_DIR = Path(__file__).resolve().parents[2]
MODES = {"student": "rollout-v1", "arm": "rollout-v1-arm"}  # evaluation_id in results.csv


# ---------------------------------------------------------------- manifest and runs

def load_manifest(path: Path) -> dict:
    m = yaml.safe_load(Path(path).read_text())
    m["date"] = str(m["date"])
    m["path"] = str(Path(path))
    if set(m["evaluation"]["validation_modes"]) - set(MODES):
        raise ValueError(f"validation modes must be among {sorted(MODES)}")
    return m


def overrides(m: dict, seed: int) -> dict:
    return {"identity.version": m["version"], "identity.date": m["date"], "identity.architecture": m["architecture"],
            "identity.seed": seed, "outputs.experiment": m["experiment"], **m["recipe"]}


def runs(m: dict) -> list[dict]:
    """Every (row, seed) with its name and directory, resolved without loading data."""
    from training.common.config import run_dir_for
    from training.datasets.train import resolve, run_name
    out = []
    for row in m["rows"]:
        for seed in m["seeds"]:
            config = resolve(row, overrides(m, seed))
            config["identity"]["run_id"] = name = run_name(config)
            config["outputs"]["experiment_dir"] = experiment_dir(m)
            d = run_dir_for(config)
            out.append({"row": row, "seed": seed, "name": name, "dir": d, "model": d / f"{name}.npz"})
    return out


def rel(path) -> str:
    """Path relative to the task directory when inside it (as results.csv records paths), else absolute."""
    path = Path(path)
    return str(path.relative_to(TASK_DIR)) if path.is_relative_to(TASK_DIR) else str(path)


def experiment_dir(m: dict) -> str:
    return f"{m['date']}__{m['experiment']}-{m['version']}"


def results_dir(m: dict) -> Path:
    return TASK_DIR / "results/ablation" / experiment_dir(m)


# ---------------------------------------------------------------- train

def train(m: dict, parallel: int) -> None:
    pending = [r for r in runs(m) if not r["model"].exists()]
    logs = results_dir(m) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"train: {len(pending)} of {len(runs(m))} runs to train, {parallel} at a time", flush=True)
    active, failed = [], []
    while pending or active:
        while pending and len(active) < parallel:
            r = pending.pop(0)
            cmd = [sys.executable, "-m", "training.datasets.train", "--variant", r["row"]]
            for k, v in overrides(m, r["seed"]).items():
                cmd += ["--set", f"{k}={v}"]
            if (r["dir"] / "latest.pt").exists():
                cmd.append("--resume")
            log = open(logs / f"{r['name']}.train.log", "a")
            active.append((r, subprocess.Popen(cmd, cwd=TASK_DIR, stdout=log, stderr=subprocess.STDOUT), log, time.time()))
            print(f"  started {r['name']}", flush=True)
        time.sleep(5)
        for item in list(active):
            r, proc, log, t0 = item
            if proc.poll() is None:
                continue
            active.remove(item)
            log.close()
            ok = proc.returncode == 0 and r["model"].exists()
            print(f"  {'finished' if ok else 'FAILED'} {r['name']} ({(time.time() - t0) / 60:.1f} min)", flush=True)
            if not ok:
                failed.append(r["name"])
    if failed:
        raise RuntimeError(f"training failed for {failed}; see {logs}")


# ---------------------------------------------------------------- validate

def _policy(model: str, mode: str):
    from core.interfaces import Action, Policy
    from evaluation.evaluate import MarkovianReference, mlp_factory
    student = mlp_factory(model)
    if mode == "student":
        return student

    class ArmOnly(Policy):
        """The student's arm torques with teacher A's wheel commands."""
        meta = student.meta

        def __init__(self):
            self.expert = MarkovianReference().expert

        def reset(self):
            student.reset()

        def act(self, obs):
            a = student.act(obs)
            self.last_raw = student.last_raw
            return Action(a.torque_arm, self.expert.act(obs).wheel_cmds)
    return ArmOnly()


def _init_worker():
    from training.common.config import pin_rollout_threads
    pin_rollout_threads()


def _rollout(job):
    model, mode, scenario, config = job
    from evaluation.evaluate import make_robot_env, rollout
    out = rollout(_policy(model, mode), scenario, config, functools.partial(make_robot_env, config["evaluation"]["reach_tolerance_m"]))
    return model, mode, out


def validate(m: dict, workers: int) -> None:
    import hashlib
    import multiprocessing as mp
    from evaluation.evaluate import append_results, result_row, summarise, write_details
    from evaluation.test_sets import get_split
    split_path = TASK_DIR / "experiments/splits/clean253.json"
    scenarios = get_split("validation", split_path, "A")
    out_dir = results_dir(m) / "validation"
    todo, configs = [], {}
    for r in runs(m):
        if not r["model"].exists():
            raise FileNotFoundError(f"{r['name']} is not trained yet")
        configs[str(r["model"])] = (r, yaml.safe_load((r["dir"] / "config.yaml").read_text()))
        for mode in m["evaluation"]["validation_modes"]:
            if not (out_dir / f"{r['name']}__validation-{mode}.csv").exists():
                todo += [(str(r["model"]), mode, s, configs[str(r["model"])][1]) for s in scenarios]
    print(f"validate: {len(todo)} rollouts on {workers} workers", flush=True)
    if not todo:
        return
    grouped = {}
    with mp.get_context("spawn").Pool(workers, _init_worker) as pool:
        for i, (model, mode, outcome) in enumerate(pool.imap_unordered(_rollout, todo, chunksize=1), 1):
            grouped.setdefault((model, mode), []).append(outcome)
            if i % 64 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    details, rows = [], []
    for (model, mode), outcomes in sorted(grouped.items()):  # build everything first, then write
        r, config = configs[model]
        outcomes.sort(key=lambda o: o["scenario_id"])
        detail = out_dir / f"{r['name']}__validation-{mode}.csv"
        with np.load(model) as z:
            meta = json.loads(str(z["metadata_json"]))
        details.append((detail, outcomes))
        rows.append(result_row(dict(config["identity"]), summarise(outcomes), lr=config["training"]["lr"],
                               split_sha256=split_sha, checkpoint_step=meta["source_step"],
                               checkpoint_path=rel(meta["source_checkpoint"]), checkpoint_sha256=meta["source_sha256"],
                               inference_path=rel(model), evaluation_split="validation", evaluation_id=MODES[mode],
                               detail_path=rel(detail)))
    append_results(TASK_DIR / "results/results.csv", rows)
    for detail, outcomes in details:
        write_details(detail, outcomes)


# ---------------------------------------------------------------- hold test

def hold(m: dict, workers: int) -> None:
    import multiprocessing as mp
    from evaluation.hold_test import hold_test
    out_dir = results_dir(m) / "hold_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [r for r in runs(m) if not (out_dir / f"{r['name']}.json").exists()]
    jobs = [(str(r["model"]), s) for r in pending for s in m["evaluation"]["hold_test_scenarios"]]
    print(f"hold: {len(jobs)} scenario tests", flush=True)
    if not jobs:
        return
    with mp.get_context("spawn").Pool(workers, _init_worker) as pool:
        results = pool.starmap(hold_test, jobs, chunksize=1)
    by_model = {}
    for (model, _), res in zip(jobs, results):
        by_model.setdefault(model, []).append(res)
    for r in pending:
        (out_dir / f"{r['name']}.json").write_text(json.dumps(by_model[str(r["model"])], indent=1))


# ---------------------------------------------------------------- summary

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def summary(m: dict) -> None:
    from evaluation.hold_test import summarise as hold_summary
    out = results_dir(m)
    per_run = []
    for r in runs(m):
        row = {"row": r["row"], "seed": r["seed"], "run": r["name"], "model": rel(r["model"])}
        for mode in m["evaluation"]["validation_modes"]:
            with open(out / "validation" / f"{r['name']}__validation-{mode}.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            row[f"{mode}_n"] = len(rows)
            row[f"{mode}_hold"] = sum(x["sustained_hold_pass"] == "True" for x in rows)
            row[f"{mode}_reach"] = sum(x["ever_reached"] == "True" for x in rows)
            row[f"{mode}_median_final_err_mm"] = float(np.median([float(x["final_err_mm"]) for x in rows]))
        h = hold_summary(json.loads((out / "hold_test" / f"{r['name']}.json").read_text()))
        row.update({f"hold_test_{k}": v for k, v in h.items()})
        config = yaml.safe_load((r["dir"] / "config.yaml").read_text())
        row["lambda_j"] = config["loss"]["jacobian"]["lambda_j"] if config["loss"].get("jacobian") else None
        per_run.append(row)
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)
    table = []
    for row_name in m["rows"]:
        rs = [x for x in per_run if x["row"] == row_name]
        entry = {"row": row_name, "seeds": len(rs)}
        for mode in m["evaluation"]["validation_modes"]:
            k, n = sum(x[f"{mode}_hold"] for x in rs), sum(x[f"{mode}_n"] for x in rs)
            lo, hi = wilson(k, n)
            entry.update({f"{mode}_hold_per_seed": "/".join(str(x[f"{mode}_hold"]) for x in rs), f"{mode}_hold_rate": k / n,
                          f"{mode}_hold_ci95": f"{lo:.2f}-{hi:.2f}",
                          f"{mode}_reach_per_seed": "/".join(str(x[f"{mode}_reach"]) for x in rs),
                          f"{mode}_median_final_err_mm": float(np.median([x[f"{mode}_median_final_err_mm"] for x in rs]))})
        entry["hold_test_arm_stayed_per_seed"] = "/".join(str(x["hold_test_arm_stayed"]) for x in rs)
        entry["hold_test_median_stiffness_eig_max"] = float(np.median([x["hold_test_median_stiffness_eig_max"] for x in rs]))
        entry["hold_test_median_max_real_eig"] = float(np.median([x["hold_test_median_max_real_eig"] for x in rs]))
        table.append(entry)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    n = per_run[0]["student_n"]
    lines = [f"# {experiment_dir(m)}", "", f"Manifest: `{Path(m['path']).name}`; architecture {m['architecture']}; "
             f"seeds {m['seeds']}; validation {n} scenarios per model; terminal checkpoint only.", "",
             "| Row | Full student: hold per seed | Hold rate (Wilson 95%) | Reach per seed | Median final error | "
             "Arm only: hold per seed | Hold test (arm, 8 scenarios) per seed | Max stiffness eig (N·m/rad) |",
             "|---|---|---|---|---|---|---|---|"]
    for e in table:
        lines.append(f"| {e['row']} | {e['student_hold_per_seed']} (of {n}) | {e['student_hold_rate']:.2f} ({e['student_hold_ci95']}) | "
                     f"{e['student_reach_per_seed']} | {e['student_median_final_err_mm']:.1f} mm | "
                     f"{e.get('arm_hold_per_seed', '')} | {e['hold_test_arm_stayed_per_seed']} | "
                     f"{e['hold_test_median_stiffness_eig_max']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def demo(m: dict) -> Path:
    """experiments/demo_models/<experiment dir>.yaml: every terminal model, hash-pinned, for evaluation.live_compare."""
    import hashlib
    artifacts = []
    for r in runs(m):
        with open(results_dir(m) / "validation" / f"{r['name']}__validation-student.csv", newline="") as f:
            outcomes = list(csv.DictReader(f))
        artifacts.append({"label": r["name"], "row": r["row"], "teacher": "A", "objective": m["recipe"]["identity.objective"],
                          "seed": r["seed"], "run_id": r["name"], "inference": rel(r["model"]),
                          "inference_sha256": hashlib.sha256(r["model"].read_bytes()).hexdigest(),
                          "validation_full_student_hold": f"{sum(o['sustained_hold_pass'] == 'True' for o in outcomes)}/{len(outcomes)}"})
    path = TASK_DIR / "experiments/demo_models" / f"{experiment_dir(m)}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"kind": "demo-models", "experiment": experiment_dir(m), "manifest": rel(m["path"]),
                                    "note": "terminal models of the ablation; written by experiments.ablation.run --stage demo",
                                    "decisions": {"final": {"artifacts": artifacts}}}, sort_keys=False))
    print(f"demo manifest: {rel(path)}")
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--stage", choices=["train", "validate", "hold", "summary", "demo", "all"], default="all")
    p.add_argument("--parallel", type=int, default=3, help="training processes at a time")
    p.add_argument("--workers", type=int, default=22, help="simulation processes")
    args = p.parse_args(argv)
    m = load_manifest(TASK_DIR / args.manifest if not Path(args.manifest).is_absolute() else args.manifest)
    results_dir(m).mkdir(parents=True, exist_ok=True)
    (results_dir(m) / "manifest.yaml").write_text(Path(m["path"]).read_text())
    for stage, fn in (("train", lambda: train(m, args.parallel)), ("validate", lambda: validate(m, args.workers)),
                      ("hold", lambda: hold(m, args.workers)), ("summary", lambda: summary(m)), ("demo", lambda: demo(m))):
        if args.stage in (stage, "all"):
            fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
