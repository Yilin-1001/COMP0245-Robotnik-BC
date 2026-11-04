"""DART-style noise + few DAgger rounds vs DAgger only: unified-test success per round, and pipeline cost per round.

Two pipelines share the clean data and one 42k-update training run of the round-0 student:
  DAgger only   : R2-gate (round 0) -> R6-gate-dagger-r1..r3
  DART + DAgger : R4-dart-gate (round 0, adds the DART collection) -> R8-dart-gate-dagger-r1..r3
Success comes from unified-test rollouts.csv files (models are matched by run id, so several test runs can be merged).
Cost comes from the logs of the 2026-11-01 v2 DAgger run (collect, build, train and validate of each round's seed-0
model, i.e. what the pipeline must run in sequence) and the DART collection's per-episode write timestamps.

Usage (from final/robotnik_arm_task/):
    python -m experiments.dagger_low_round_report --tests results/test/2026-11-01_215340__R1-R8 results/test/<new run> \
        --out results/ablation/2026-11-03__training-ablation-v2/low-round
"""
import argparse
import csv
import datetime
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
V2_LOG = TASK_DIR / "results/ablation/2026-11-01__training-ablation-v2/dagger/run.log"
DART_DIR = TASK_DIR / "data/Markovian/DART/2026-10-28__collection__d2-s075-x2-58af5db87875"
DATASETS = TASK_DIR / "data/datasets/A"
PIPELINES = {"DAgger only": ("R2-gate", "R6-gate-dagger", "R6"), "DART + DAgger": ("R4-dart-gate", "R8-dart-gate-dagger", "R8")}
GROUPS = [("T0", None, "T0 core (in distribution)"), ("T1", "3-6m", "T1 far 3-6 m"), ("T1", "6-10m", "T1 far 6-10 m"),
          ("T1", "10-15m", "T1 far 10-15 m"), ("T2", None, "T2 near"), ("T3", "low", "T3 low"), ("T3", "high", "T3 high"),
          ("T4", None, "T4 translated start"), ("T5", None, "T5 long hold"), ("T1-T5", None, "T1-T5 pooled"),
          ("in-range", None, "In range (T0+T4+T5)"), ("edges", None, "Range edges (T2+T3)"), ("T1", None, "Far (T1, 3-15 m)"),
          ("all", None, "All 240 scenes")]
GROUP_SUITES = {"in-range": ("T0", "T4", "T5"), "edges": ("T2", "T3"), "T1-T5": ("T1", "T2", "T3", "T4", "T5")}
PANELS = ["In range (T0+T4+T5)", "Range edges (T2+T3)", "Far (T1, 3-15 m)"]
BOOTSTRAP_SEED = 1101031


def identify(model_id: str):
    """(pipeline, round, seed) from a run id, or None for rows outside this comparison."""
    seed = int(re.search(r"__s(\d+)__", model_id).group(1)) if "__s" in model_id else None
    for name, (base, dagger, _) in PIPELINES.items():
        if model_id.startswith(base + "__"):
            return name, 0, seed
        m = re.match(re.escape(dagger) + r"-r(\d)__", model_id)
        if m:
            return name, int(m.group(1)), seed
    return None


def load_rollouts(test_dirs):
    rows, seen = [], set()
    for d in test_dirs:
        path = Path(d) / "rollouts.csv"
        if not path.exists():  # an unfinished run: read the job records directly
            records = [json.load(open(p))["record"] for p in sorted((Path(d) / "jobs").glob("*.json"))]
        else:
            records = list(csv.DictReader(open(path)))
        for r in records:
            key = (r["model_id"], r["scenario_id"])
            ident = identify(r["model_id"])
            if ident is None or key in seen:
                continue
            seen.add(key)
            success = r["success"] in (True, "True")
            rows.append({"pipeline": ident[0], "round": ident[1], "seed": ident[2], "model_id": r["model_id"],
                         "scenario_id": r["scenario_id"], "suite": r["suite"], "band": r["band"], "success": success,
                         "final1s_max_err_mm": float(r["final1s_max_err_mm"]) if r["final1s_max_err_mm"] not in ("", None) else np.nan})
    return rows


def in_group(r, suite, band):
    if suite == "all":
        return True
    if suite in GROUP_SUITES:
        return r["suite"] in GROUP_SUITES[suite]
    return r["suite"] == suite and (band is None or r["band"] == band)


def success_table(rows):
    out = []
    for (pipe, rnd) in sorted({(r["pipeline"], r["round"]) for r in rows}):
        sel = [r for r in rows if r["pipeline"] == pipe and r["round"] == rnd]
        seeds = sorted({r["seed"] for r in sel})
        for suite, band, label in GROUPS:
            g = [r for r in sel if in_group(r, suite, band)]
            if not g:
                continue
            clusters = defaultdict(list)
            for r in g:
                clusters[r["scenario_id"]].append(float(r["success"]))
            values = np.array([np.mean(v) for v in clusters.values()])
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            means = values[rng.integers(0, len(values), (1000, len(values)))].mean(axis=1)
            lo, hi = np.quantile(means, [0.025, 0.975])
            per_seed = {s: sum(r["success"] for r in g if r["seed"] == s) for s in seeds}
            out.append({"pipeline": pipe, "round": rnd, "group": label, "seeds": " ".join(map(str, seeds)),
                        "scenes": len(clusters), "successes": sum(r["success"] for r in g), "n": len(g),
                        "rate": float(np.mean([r["success"] for r in g])), "ci_low": float(lo), "ci_high": float(hi),
                        "per_seed": " / ".join(f"{per_seed[s]}" for s in seeds),
                        "median_final1s_err_mm": float(np.nanmedian([r["final1s_max_err_mm"] for r in g]))})
    return out


def step_minutes():
    """{(row prefix, step, round): minutes} for the seed-0 chain of the 2026-11-01 v2 run."""
    out, row, step = {}, None, None
    for line in V2_LOG.read_text().splitlines():
        m = re.match(r"===== (R\d)-\S+ start", line)
        if m:
            row = m.group(1)
        if line.startswith("$ "):
            module = line.split(" -m ")[1].split()[0].split(".")[-1]
            k = re.search(r"(?:--round r|-r)(\d)\b", line) or re.search(r"-r(\d)__", line)
            s = re.search(r"identity\.seed=(\d)", line) or re.search(r"__s(\d)__", line)
            step = (row, module, int(k.group(1)) if k else 0, int(s.group(1)) if s else 0)
        m = re.search(r"done in ([\d.]+) min", line)
        if m and step and step[3] == 0:
            out[step[:3]] = float(m.group(1))
    return out


def dart_collection():
    times = [datetime.datetime.fromisoformat(json.loads(l)["time"]) for l in open(DART_DIR / "qa/attempts.jsonl")]
    steps = sum(int(r["T_actions"]) for r in csv.DictReader(open(DART_DIR / "episodes.csv")))
    return (max(times) - min(times)).total_seconds() / 60, len(times), steps


def rows_of(name):
    return json.load(open(DATASETS / name / "metadata.json"))["counts"]["rows"]


def cost_table():
    """Cumulative added cost per round, beyond the shared clean data and round-0 training."""
    minutes = step_minutes()
    dart_min, dart_eps, dart_steps = dart_collection()
    out = []
    for pipe, (_, _, row) in PIPELINES.items():
        wall, labels, cycles = (dart_min, dart_steps, 0) if pipe == "DART + DAgger" else (0.0, 0, 0)
        out.append({"pipeline": pipe, "round": 0, "added_wall_min": round(wall, 1), "added_expert_labels_M": round(labels / 1e6, 2),
                    "sequential_train_collect_cycles": 0,
                    "note": f"DART collection {dart_eps} episodes, first-to-last write" if wall else "-"})
        for k in (1, 2, 3):
            parts = {s: minutes[(row, s, k)] for s in ("collect_dagger", "build", "train", "evaluate")}
            wall += sum(parts.values())
            labels += rows_of(f"dagger-{row}-v2-r{k}__features-v1-gate")
            cycles += 1
            out.append({"pipeline": pipe, "round": k, "added_wall_min": round(wall, 1), "added_expert_labels_M": round(labels / 1e6, 2),
                        "sequential_train_collect_cycles": cycles,
                        "note": "round {}: collect {collect_dagger} + build {build} + train {train} + validate {evaluate} min".format(k, **parts)})
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(success, cost, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"DAgger only": "#3b6ea8", "DART + DAgger": "#c4572a"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    for ax, group in zip(axes, PANELS):
        for pipe in PIPELINES:
            pts = []
            for c in cost:
                if c["pipeline"] != pipe:
                    continue
                s = [r for r in success if r["pipeline"] == pipe and r["round"] == c["round"] and r["group"] == group]
                if s:
                    pts.append((c["added_wall_min"], s[0]["rate"], s[0]["ci_low"], s[0]["ci_high"], c["round"]))
            if not pts:
                continue
            x, y, lo, hi, k = map(np.array, zip(*pts))
            ax.errorbar(x, 100 * y, yerr=[100 * (y - lo), 100 * (hi - y)], color=colours[pipe], marker="o", capsize=3, label=pipe)
            for xi, yi, ki in zip(x, y, k):
                ax.annotate(f"r{ki}", (xi, 100 * yi), textcoords="offset points", xytext=(4, -10), fontsize=8, color=colours[pipe])
        ax.set_title(group, fontsize=10)
        ax.set_xlabel("added wall-clock (min)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tests", nargs="+", required=True, help="unified-test run directories")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rollouts(args.tests)
    success, cost = success_table(rows), cost_table()
    write_csv(args.out / "success_by_round.csv", success)
    write_csv(args.out / "cost_by_round.csv", cost)
    plot(success, cost, args.out / "cost_vs_success.png")
    for c in cost:
        print(c)
    for s in success:
        if s["group"] in PANELS + ["All 240 scenes"]:
            print(f"{s['pipeline']:14s} r{s['round']} {s['group']:26s} {s['successes']}/{s['n']} = {100 * s['rate']:.1f}% "
                  f"[{100 * s['ci_low']:.1f}, {100 * s['ci_high']:.1f}] seeds {s['per_seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
