"""DAgger rows R5-R8 of the training-method ablation v2, unattended: collect -> build -> train -> validate, k = 1..K.

Each DAgger row extends one v1 row with the same data and inputs: R5-baseline-dagger (R1-baseline), R6-gate-dagger
(R2-gate), R7-dart-dagger (R3-dart) and R8-dart-gate-dagger (R4-dart-gate).
Round k: the current student (round 0: the base row's v1 seed-0 model) drives alone (beta = 0) on the 221 fitting scenarios plus
fresh low/near-biased development scenes; teacher A labels every visited state (data_collection.collect_dagger).
The new student is trained from scratch, with the row's recipe and inputs, on the row's own datasets plus every DAgger
round so far (data aggregation), and is evaluated on the 32 frozen validation scenarios (full student, standard
predicates). It then drives round k + 1. Test scenarios are never touched.

Names: round k of row R<n> trains variant <row>-r<k> on datasets dagger-R<n>-<version>-r<i>__<features>, i <= k.
Models go to checkpoints/ablation/<date>__training-ablation-<version>/ and results to
results/ablation/<date>__training-ablation-<version>/dagger/<row>/ (summary.csv, one row per model; validation/; logs/).
Round 0 defaults to the base row's model in checkpoints/ablation/2026-10-28__training-ablation-v1/; --start replaces it.

Every step is skipped when its output already exists, so an interrupted run continues where it stopped.

Usage (from final/robotnik_arm_task/):
    python -m experiments.dagger_rounds --row R8-dart-gate-dagger --rounds 3 --date 2026-10-30 --final-seeds 1 2
    python -m experiments.dagger_rounds --row R5-baseline-dagger --rounds 3 --date 2026-10-30 \
        [--start <round-0 .npz>] [--final-seeds 1 2] [--dev-scenes 64] [--workers 22]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
ROWS = {  # DAgger row -> (base v1 row, feature definition, the base row's datasets)
    "R5-baseline-dagger": ("R1-baseline", "features-v1", ["clean253"]),
    "R6-gate-dagger": ("R2-gate", "features-v1-gate", ["clean253"]),
    "R7-dart-dagger": ("R3-dart", "features-v1", ["clean253", "dart-d2s075x2"]),
    "R8-dart-gate-dagger": ("R4-dart-gate", "features-v1-gate", ["clean253", "dart-d2s075x2"]),
}
VERSION = "v2"
V1_RUN = "{row}__MLP01-256x3__s0__v1__2026-10-28"  # round 0: the base row's terminal model of ablation v1
V1_DIR = "checkpoints/ablation/2026-10-28__training-ablation-v1"
RECIPE = {"identity.version": VERSION, "identity.objective": "bcj", "training.lr": "1.0e-3", "training.updates": "42000",
          "loss.action": "huber", "loss.arm_weight": "0.5", "loss.wheel_weight": "0.5"}
FIELDS = ["row", "round", "seed", "run", "collector", "sustained_hold", "reach", "n", "median_final_err_mm",
          "max_final1s_err_mm", "model", "written"]


def run(cmd: list[str], log: Path) -> str:
    """Run a step, tee its output to a log file, return stdout; stop the whole run on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    out = subprocess.run([sys.executable, *cmd], cwd=TASK_DIR, capture_output=True, text=True)
    log.write_text(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{out.stdout}\n--- stderr ---\n{out.stderr}")
    if out.returncode != 0:
        raise SystemExit(f"step failed ({out.returncode}) after {time.time() - t0:.0f} s; see {log}")
    print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return out.stdout


def last_json(text: str) -> dict:
    return json.loads(text[text.rindex("\n{") + 1:] if "\n{" in text else text[text.index("{"):])


def dataset_name(row: str, k: int) -> str:
    return f"dagger-{row.split('-')[0]}-{VERSION}-r{k}"


def write_variant(row: str, k: int) -> str:
    name = f"{row}-r{k}"
    base, definition, own = ROWS[row]
    datasets = [f"{d}__{definition}" for d in own] + [f"{dataset_name(row, i)}__{definition}" for i in range(1, k + 1)]
    path = TASK_DIR / "experiments/configs/datasets/variants" / f"{name}.yaml"
    text = (f"# {row}, DAgger round {k} ({VERSION}): {base}'s data ({', '.join(own)}) plus DAgger rounds 1..{k},\n"
            f"# inputs {definition}. Written by experiments/dagger_rounds.py.\n"
            f"data:\n  feature_definition: {definition}\n"
            f"  datasets: [{', '.join(datasets)}]\n  validation_dataset: clean253__{definition}\n")
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return name


def model_path(variant: str, seed: int, date: str) -> Path:
    run_id = f"{variant}__MLP01-256x3__s{seed}__{VERSION}__{date}"
    return TASK_DIR / f"checkpoints/ablation/{date}__training-ablation-{VERSION}/{run_id}/{run_id}.npz"


def train_and_validate(row, k, seed, date, collector, workers, out_dir, model=None) -> dict:
    """Train round k (k >= 1) unless its model exists, then validate it; round 0 passes the existing model."""
    if model is None:
        variant = write_variant(row, k)
        model = model_path(variant, seed, date)
    logs = out_dir / "logs"
    if not model.exists() and k == 0:
        raise SystemExit(f"round-0 model {model} not found; pass --start")
    if not model.exists():
        cmd = ["-m", "training.datasets.train", "--variant", variant, "--set", f"identity.date={date}",
               "--set", f"identity.seed={seed}"]
        for key, value in RECIPE.items():
            cmd += ["--set", f"{key}={value}"]
        run(cmd, logs / f"r{k}-s{seed}-train.log")
    cache = out_dir / "validation" / f"{model.stem}.json"
    if cache.exists():
        result = json.loads(cache.read_text())
    else:
        out = run(["-m", "evaluation.evaluate", "--policy", "mlp", "--split", "validation", "--workers", str(workers),
                   "--artifact", str(model.relative_to(TASK_DIR))], logs / f"r{k}-s{seed}-validate.log")
        result = last_json(out)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=1))
    n = result["n"]
    entry = {"row": row, "round": k, "seed": seed, "run": model.stem, "collector": collector,
             "sustained_hold": f"{round(result['sustained_hold_success'] * n)}/{n}",
             "reach": f"{round(result['reach_success'] * n)}/{n}", "n": n,
             "median_final_err_mm": round(result["median_final_err_mm"], 3),
             "max_final1s_err_mm": round(result["max_final1s_err_mm"], 3),
             "model": os.path.relpath(model, TASK_DIR), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    summary = out_dir / "summary.csv"
    rows = list(csv.DictReader(open(summary))) if summary.exists() else []
    rows = [{f: r.get(f) or (row if f == "row" else "") for f in FIELDS} for r in rows]  # older files lack the row column
    rows = [r for r in rows if not (r["round"] == str(k) and r["seed"] == str(seed))] + [entry]
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  {row} round {k} seed {seed}: validation sustained hold {entry['sustained_hold']}, reach {entry['reach']}, "
          f"median final error {entry['median_final_err_mm']} mm", flush=True)
    return {**entry, "hold": result["sustained_hold_success"], "model_path": model}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--row", default="R8-dart-gate-dagger", choices=list(ROWS))
    p.add_argument("--start", help="round-0 student (.npz); default: the base row's seed-0 model of ablation v1")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--date", required=True, help="experiment date YYYY-MM-DD, fixed for the whole experiment")
    p.add_argument("--dev-scenes", type=int, default=64)
    p.add_argument("--final-seeds", type=int, nargs="*", default=[], help="extra seeds for the best DAgger round, e.g. 1 2")
    p.add_argument("--workers", type=int, default=22)
    args = p.parse_args(argv)
    base, definition, _ = ROWS[args.row]
    out_dir = TASK_DIR / f"results/ablation/{args.date}__training-ablation-{VERSION}/dagger/{args.row}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = V1_RUN.format(row=base)
    start = Path(args.start).resolve() if args.start else TASK_DIR / V1_DIR / run_id / f"{run_id}.npz"
    print(f"\n===== {args.row} round 0 ({base}), seed 0: {start.name} =====", flush=True)
    student = train_and_validate(args.row, 0, 0, args.date, "-", args.workers, out_dir, model=start)["model_path"]
    results = []
    for k in range(1, args.rounds + 1):
        print(f"\n===== {args.row} DAgger round {k}: collector {student.name} =====", flush=True)
        out = run(["-m", "data_collection.collect_dagger", "--student", str(student), "--round", f"r{k}",
                   "--dev-scenes", str(args.dev_scenes), "--workers", str(args.workers)], out_dir / "logs" / f"r{k}-collect.log")
        collected = last_json(out)
        print(f"  collected: {json.dumps({x: collected[x] for x in collected if x != 'root'})}", flush=True)
        run(["-m", "training.datasets.build", "--source", collected["root"], "--name", dataset_name(args.row, k),
             "--features", definition, "--workers", str(args.workers)], out_dir / "logs" / f"r{k}-build.log")
        res = train_and_validate(args.row, k, 0, args.date, student.name, args.workers, out_dir)
        results.append(res)
        student = res["model_path"]
    if args.final_seeds:
        best = max(results, key=lambda r: (r["hold"], r["round"]))
        print(f"\n===== extra seeds for round {best['round']} (best seed-0 validation hold {best['sustained_hold']}) =====")
        for seed in args.final_seeds:
            train_and_validate(args.row, best["round"], seed, args.date, best["collector"], args.workers, out_dir)
    print(f"\nSummary: {out_dir / 'summary.csv'}")
    print((out_dir / "summary.csv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
