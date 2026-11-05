"""How many training updates does a small-network learner need? Retrain on a fixed DAgger aggregate with fewer updates.

Takes the final dataset of one experiments.dagger_arch_rounds chain (round k of architecture A) and retrains it with
shorter budgets (the cosine schedule then decays to zero at the new budget, so each run is a complete short training,
not an intermediate checkpoint). Everything else is the chain's recipe. Each model is evaluated on the 32 frozen
validation scenarios. Coarse scan with seed 0 first; then the smallest budget whose seed-0 hold is within --tolerance
of the 42,000-update reference gets the extra --confirm-seeds.

Only the final training is shortened: the DAgger data were collected by the 42,000-update students.

Results: results/ablation/<date>__small-net-dagger-v2/<arch>/steps.csv (one row per budget and seed), printed as a table
together with the 42,000-update reference from summary.csv / cost.csv.

Usage (from final/robotnik_arm_task/):
    python -m experiments.step_budget --arch MLP04 --round 3 --date 2026-11-04 [--budgets 21000 10500 5000]
        [--confirm-seeds 1 2] [--tolerance 2] [--workers 20]
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import yaml

from experiments import dagger_arch_rounds as ar
from experiments import dagger_rounds as dr

TASK_DIR = ar.TASK_DIR
FIELDS = ["arch", "round", "updates", "seed", "sustained_hold", "reach", "median_final_err_mm", "train_elapsed_s",
          "wall_s", "model", "written"]


def variant(arch: str, k: int, updates: int) -> str:
    """A copy of the chain's round-k variant under its own name, so run directories do not collide with 42,000."""
    src = TASK_DIR / "experiments/configs/datasets/variants" / f"{ar.write_variant(arch, k)}.yaml"
    name = f"{ar.variant_name(arch, k)}-u{updates}"
    dst = src.with_name(f"{name}.yaml")
    text = f"# {name}: identical to {src.stem}, trained with {updates} updates (experiments/step_budget.py).\n" + src.read_text()
    if not dst.exists() or dst.read_text() != text:
        dst.write_text(text)
    return name


def model_path(arch: str, k: int, updates: int, seed: int, date: str) -> Path:
    run_id = f"{ar.variant_name(arch, k)}-u{updates}__{ar.label(arch)}__s{seed}__{ar.VERSION}__{date}"
    return TASK_DIR / f"checkpoints/ablation/{date}__training-ablation-{ar.VERSION}/{run_id}/{run_id}.npz"


def run_one(arch, k, updates, seed, date, workers, out_dir) -> dict:
    path = out_dir / "steps.csv"
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    done = [r for r in rows if r["updates"] == str(updates) and r["seed"] == str(seed)]
    if done:
        return done[0]
    model = model_path(arch, k, updates, seed, date)
    t0 = time.time()
    if not model.exists():
        shutil.rmtree(model.parent, ignore_errors=True)
        cmd = ["-m", "training.datasets.train", "--variant", variant(arch, k, updates), "--set", f"identity.date={date}",
               "--set", f"identity.seed={seed}", "--set", f"identity.architecture={arch}"]
        for key, value in {**dr.RECIPE, "training.lr": ar.learning_rate(arch), "training.updates": str(updates)}.items():
            cmd += ["--set", f"{key}={value}"]
        dr.run(cmd, out_dir / "logs" / f"steps-u{updates}-s{seed}-train.log")
    metrics = list(csv.DictReader(open(model.parent / "metrics.csv")))
    out = dr.run(["-m", "evaluation.evaluate", "--policy", "mlp", "--split", "validation", "--workers", str(workers),
                  "--artifact", str(model.relative_to(TASK_DIR))], out_dir / "logs" / f"steps-u{updates}-s{seed}-validate.log")
    res = dr.last_json(out)
    n = res["n"]
    row = {"arch": arch, "round": k, "updates": updates, "seed": seed,
           "sustained_hold": f"{round(res['sustained_hold_success'] * n)}/{n}", "reach": f"{round(res['reach_success'] * n)}/{n}",
           "median_final_err_mm": round(res["median_final_err_mm"], 3), "train_elapsed_s": metrics[-1]["elapsed_s"],
           "wall_s": round(time.time() - t0, 1), "model": model.stem, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows + [row])
    print(f"  {arch} r{k} {updates} updates seed {seed}: hold {row['sustained_hold']}, "
          f"train {float(row['train_elapsed_s']) / 60:.1f} min", flush=True)
    return row


def hold(row: dict) -> int:
    return int(row["sustained_hold"].split("/")[0])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", required=True)
    p.add_argument("--round", type=int, required=True, help="the chain's chosen round, whose aggregate is retrained")
    p.add_argument("--date", required=True, help="date of the dagger_arch_rounds chain")
    p.add_argument("--budgets", type=int, nargs="+", default=[21000, 10500, 5000])
    p.add_argument("--confirm-seeds", type=int, nargs="*", default=[1, 2])
    p.add_argument("--tolerance", type=int, default=2, help="seed-0 hold may be this many scenes below the reference")
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args(argv)
    out_dir = TASK_DIR / f"results/ablation/{args.date}__small-net-dagger-v2/{args.arch}"
    summary = list(csv.DictReader(open(out_dir / "summary.csv")))
    ref = {int(r["seed"]): r for r in summary if int(r["round"]) == args.round}
    if 0 not in ref:
        raise SystemExit(f"no round-{args.round} seed-0 model in {out_dir / 'summary.csv'}")
    ref_hold = hold(ref[0])
    cost = [r for r in csv.DictReader(open(out_dir / "cost.csv")) if r["step"] == "train" and r["round"] == str(args.round)]
    ref_train = {int(r["seed"]): float(r["train_elapsed_s"]) for r in cost if r["train_elapsed_s"]}
    print(f"===== {args.arch} round {args.round}: reference 42000 updates, seed-0 hold {ref_hold}/32 =====", flush=True)
    scan = {b: run_one(args.arch, args.round, b, 0, args.date, args.workers, out_dir) for b in sorted(args.budgets, reverse=True)}
    ok = [b for b, r in scan.items() if hold(r) >= ref_hold - args.tolerance]
    chosen = min(ok) if ok else None
    if chosen is not None:
        print(f"===== smallest budget within {args.tolerance} of the reference: {chosen}; confirming seeds "
              f"{args.confirm_seeds} =====", flush=True)
        for seed in args.confirm_seeds:
            run_one(args.arch, args.round, chosen, seed, args.date, args.workers, out_dir)
    rows = list(csv.DictReader(open(out_dir / "steps.csv")))
    budgets = [42000] + sorted({int(r["updates"]) for r in rows}, reverse=True)
    lines = [f"| Updates | Train min (median) | Validation hold s0 / s1 / s2 | Pooled |", "|---|---|---|---|"]
    for b in budgets:
        if b == 42000:
            holds = {s: hold(r) for s, r in ref.items()}
            mins = sorted(ref_train.values())
        else:
            these = [r for r in rows if int(r["updates"]) == b]
            holds = {int(r["seed"]): hold(r) for r in these}
            mins = sorted(float(r["train_elapsed_s"]) for r in these)
        per = " / ".join(str(holds.get(s, "-")) for s in (0, 1, 2))
        pooled = f"{sum(holds.values())}/{32 * len(holds)}"
        lines.append(f"| {b} | {mins[len(mins) // 2] / 60:.1f} | {per} | {pooled} |" if mins else f"| {b} | - | {per} | {pooled} |")
    table = "\n".join([f"# {args.arch} round {args.round}: training budget", "", *lines, "",
                       f"Chosen: {chosen} (smallest budget with seed-0 hold >= {ref_hold - args.tolerance}/32)."])
    (out_dir / "steps.md").write_text(table + "\n")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
