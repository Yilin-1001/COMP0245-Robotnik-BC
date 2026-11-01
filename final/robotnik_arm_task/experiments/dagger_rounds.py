"""DAgger rounds on top of an ablation row (R1-R4), unattended: collect -> build -> train -> validate, k = 1..K.

Round k: the current student (round 0: the row's own model) drives alone (beta = 0) on the 221 fitting scenarios plus
fresh low/near-biased development scenes; teacher A labels every visited state (data_collection.collect_dagger).
The new student is trained from scratch, with the row's recipe and inputs, on the row's own datasets plus every DAgger
round so far (data aggregation), and is evaluated on the 32 frozen validation scenarios (full student, standard
predicates). It then drives round k + 1. Test scenarios are never touched.

Rows (experiments/configs/datasets/variants/): R1-baseline (clean, 21 inputs), R2-gate (clean, 22 inputs),
R3-dart (clean + DART, 21 inputs), R4-dart-gate (clean + DART, 22 inputs). Without --start, the row's seed-0 model
is trained first (same recipe and date) and recorded as round 0.

Every step is skipped when its output already exists, so an interrupted run continues where it stopped.
Results: results/dagger/<tag>/summary.csv (one row per trained model) and logs/.

Usage (from final/robotnik_arm_task/):
    python -m experiments.dagger_rounds --row R4-dart-gate --start <R4 seed-0 .npz> --rounds 3 --tag v1 --date 2026-10-30
    python -m experiments.dagger_rounds --row R1-baseline --rounds 3 --tag R1v1 --date 2026-10-30 \
        [--baseline-seeds 1 2] [--final-seeds 1 2] [--dev-scenes 64] [--workers 22]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
ROWS = {  # row -> (feature definition, the row's own datasets)
    "R1-baseline": ("features-v1", ["clean253"]),
    "R2-gate": ("features-v1-gate", ["clean253"]),
    "R3-dart": ("features-v1", ["clean253", "dart-d2s075x2"]),
    "R4-dart-gate": ("features-v1-gate", ["clean253", "dart-d2s075x2"]),
}
RECIPE = {"identity.version": "v1", "identity.objective": "bcj", "training.lr": "1.0e-3", "training.updates": "42000",
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


def variant_name(row: str, tag: str, k: int) -> str:
    if k == 0:
        return row
    # The first run (R4, tag v1) used R6-dagger-<tag>-r<k>; keep it so its results and models stay addressable.
    return f"R6-dagger-{tag}-r{k}" if row == "R4-dart-gate" else f"{row.split('-')[0]}D-dagger-{tag}-r{k}"


def write_variant(row: str, tag: str, k: int) -> str:
    name = variant_name(row, tag, k)
    if k == 0:
        return name
    definition, own = ROWS[row]
    datasets = [f"{d}__{definition}" for d in own] + [f"dagger{tag}-r{i}__{definition}" for i in range(1, k + 1)]
    path = TASK_DIR / "experiments/configs/datasets/variants" / f"{name}.yaml"
    text = (f"# {row} + DAgger round {k} ({tag}): the row's data ({', '.join(own)}) plus DAgger rounds 1..{k},\n"
            f"# inputs {definition}. Written by experiments/dagger_rounds.py.\n"
            f"data:\n  feature_definition: {definition}\n"
            f"  datasets: [{', '.join(datasets)}]\n  validation_dataset: clean253__{definition}\n")
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return name


def model_path(variant: str, seed: int, date: str) -> Path:
    run_id = f"{variant}__MLP01-256x3__s{seed}__v1__{date}"
    return TASK_DIR / f"checkpoints/ablation/{date}__training-ablation-v1/{run_id}/{run_id}.npz"


def train_and_validate(row, tag, k, seed, date, collector, workers, out_dir) -> dict:
    variant = write_variant(row, tag, k)
    model = model_path(variant, seed, date)
    logs = out_dir / "logs"
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
             "model": str(model.relative_to(TASK_DIR)), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
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
    p.add_argument("--row", default="R4-dart-gate", choices=list(ROWS))
    p.add_argument("--start", help="round-0 student (.npz); default: train the row's seed-0 model first")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tag", default="v1", help="experiment tag in dataset and variant names (letters/digits), unique per row")
    p.add_argument("--date", required=True, help="experiment date YYYY-MM-DD, fixed for the whole run")
    p.add_argument("--dev-scenes", type=int, default=64)
    p.add_argument("--baseline-seeds", type=int, nargs="*", default=[], help="extra seeds of the row itself (round 0)")
    p.add_argument("--final-seeds", type=int, nargs="*", default=[], help="extra seeds for the best DAgger round, e.g. 1 2")
    p.add_argument("--workers", type=int, default=22)
    args = p.parse_args(argv)
    if not args.tag.isalnum():
        raise SystemExit("--tag must be letters and digits only")
    definition, _ = ROWS[args.row]
    out_dir = TASK_DIR / "results/dagger" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.start:
        student = Path(args.start).resolve()
    else:
        print(f"\n===== {args.row} round 0 (the row itself), seed 0 =====", flush=True)
        student = train_and_validate(args.row, args.tag, 0, 0, args.date, "-", args.workers, out_dir)["model_path"]
    for seed in args.baseline_seeds:
        train_and_validate(args.row, args.tag, 0, seed, args.date, "-", args.workers, out_dir)
    results = []
    for k in range(1, args.rounds + 1):
        print(f"\n===== {args.row} DAgger round {k}: collector {student.name} =====", flush=True)
        out = run(["-m", "data_collection.collect_dagger", "--student", str(student), "--round", f"r{k}",
                   "--dev-scenes", str(args.dev_scenes), "--workers", str(args.workers)], out_dir / "logs" / f"r{k}-collect.log")
        collected = last_json(out)
        print(f"  collected: {json.dumps({x: collected[x] for x in collected if x != 'root'})}", flush=True)
        run(["-m", "training.datasets.build", "--source", collected["root"], "--name", f"dagger{args.tag}-r{k}",
             "--features", definition, "--workers", str(args.workers)], out_dir / "logs" / f"r{k}-build.log")
        res = train_and_validate(args.row, args.tag, k, 0, args.date, student.name, args.workers, out_dir)
        results.append(res)
        student = res["model_path"]
    if args.final_seeds:
        best = max(results, key=lambda r: (r["hold"], r["round"]))
        print(f"\n===== extra seeds for round {best['round']} (best seed-0 validation hold {best['sustained_hold']}) =====")
        for seed in args.final_seeds:
            train_and_validate(args.row, args.tag, best["round"], seed, args.date, best["collector"], args.workers, out_dir)
    print(f"\nSummary: {out_dir / 'summary.csv'}")
    print((out_dir / "summary.csv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
