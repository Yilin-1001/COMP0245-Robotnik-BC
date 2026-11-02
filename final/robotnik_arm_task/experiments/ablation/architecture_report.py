"""Architecture ablation report: test success, validation, hold test and latency per (row, architecture).

The unified test suite labels a model by its row only, so its own row-level tables pool every architecture of a
row. This report regroups the rollouts by (row, architecture), pools the 256x3 references from the earlier R1-R8
test run (after checking that both runs used identical code, scenes and protocol), and writes
architecture_summary.csv/.md and figures/architecture/*.png into the experiment's results directory.

Usage (from final/robotnik_arm_task/):
    python -m experiments.ablation.architecture_report --experiment results/ablation/<date>__architecture-ablation-v1 \
        --reference-test results/test/2026-11-01_215340__R1-R8 --tests results/test/<run> [...] \
        --latency results/ablation/<date>__architecture-ablation-v1/latency.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.ablation.run import rel, wilson

TASK_DIR = Path(__file__).resolve().parents[2]
SUITES = ["T0", "T1", "T2", "T3", "T4", "T5"]
REFERENCE = "MLP01-256x3"
ROW_STYLE = {"R4": ("#2a78d6", "o"), "R6": ("#eb6834", "s"), "R8": ("#1baf7a", "^")}  # dataviz slots 1-3, validated
WIDTH_LINE = ["MLP02-32x3", "MLP04-64x3", "MLP06-128x3", "MLP01-256x3", "MLP10-512x3"]
DEPTH_256 = ["MLP08-256x2", "MLP01-256x3", "MLP09-256x4"]
DEPTH_64 = ["MLP03-64x2", "MLP04-64x3", "MLP05-64x4"]


def split_model_id(model_id: str) -> tuple[str, str]:
    """('R4', 'MLP07-256-128-64') from 'R4-dart-gate__MLP07-256-128-64__s0__v1__2026-11-03'."""
    parts = model_id.split("__")
    return parts[0].split("-")[0], parts[1]


def check_comparable(reference_run: dict, new_run: dict) -> None:
    """Refuse to pool two test runs unless their scenes, protocol, interface and evaluated code are identical."""
    if reference_run["scene_manifest_sha256"] != new_run["scene_manifest_sha256"]:
        raise ValueError("scene manifest differs between the test runs")
    for key in ("protocol", "interface"):
        if reference_run["identity"][key] != new_run["identity"][key]:
            raise ValueError(f"{key} differs between the test runs")
    ref, new = reference_run["identity"]["code_sha256"], new_run["identity"]["code_sha256"]
    if ref != new:
        changed = sorted(k for k in set(ref) | set(new) if ref.get(k) != new.get(k))
        raise ValueError(f"evaluated code differs between the test runs: {changed}")


def _entry(group: list[dict]) -> dict:
    k, n = sum(r["success"] == "True" for r in group), len(group)
    lo, hi = wilson(k, n)
    times = [float(r["completion_time_s"]) for r in group if r["success"] == "True" and r["completion_time_s"]]
    return {"k": k, "n": n, "rate": k / n, "ci_lo": lo, "ci_hi": hi,
            **{f"grade_{g}": sum(r["grade"] == g for r in group) for g in "ABCDEF"},
            "median_completion_s": float(np.median(times)) if times else None}


def aggregate(rollouts: list[dict], rows: set[str]) -> list[dict]:
    """One entry per (row, architecture, suite) and per (row, architecture, 'all'), seeds pooled."""
    groups = {}
    for r in rollouts:
        if r["row"] == "expert":
            continue
        row, arch = split_model_id(r["model_id"])
        if row not in rows:
            continue
        groups.setdefault((row, arch, r["suite"]), []).append(r)
        groups.setdefault((row, arch, "all"), []).append(r)
    return [{"row": row, "architecture": arch, "suite": suite, **_entry(g)} for (row, arch, suite), g in sorted(groups.items())]


def lr_decision(holds: dict[str, dict[float, int]], tolerance: int = 2) -> str:
    """'keep' lr 1e-3 if, for every architecture, it is within `tolerance` holds of the best lr; else 'screen'."""
    return "keep" if all(h[1e-3] >= max(h.values()) - tolerance for h in holds.values()) else "screen"


def recommend(table: list[dict], margin: float = 0.03) -> dict[str, str]:
    """Per row, the smallest architecture whose all-suite rate is within `margin` of the 256x3 reference."""
    out = {}
    for row in sorted({e["row"] for e in table}):
        pooled = [e for e in table if e["row"] == row and e["suite"] == "all"]
        ref = next(e["rate"] for e in pooled if e["architecture"] == REFERENCE)
        ok = [e for e in pooled if e["rate"] >= ref - margin]
        out[row] = min(ok, key=lambda e: e["parameters"])["architecture"]
    return out


# ---------------------------------------------------------------- inputs

def _csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _seed(model_id: str) -> int:
    return int(model_id.split("__")[2][1:])


def load_tests(reference: Path, tests: list[Path], rows: set[str]) -> tuple[list[dict], dict]:
    """Rollouts of the new runs plus the reference run's 256x3 models of `rows`; expert agreement per new run."""
    ref_run = json.loads((reference / "run.json").read_text())
    ref_rollouts = _csv(reference / "rollouts.csv")
    ref_expert = {r["scenario_id"]: r["success"] for r in ref_rollouts if r["row"] == "expert"}
    rollouts = [r for r in ref_rollouts if r["row"] != "expert" and split_model_id(r["model_id"])[0] in rows]
    expert = {}
    for test in tests:
        run = json.loads((test / "run.json").read_text())
        check_comparable(ref_run, run)
        if run["status"] != "complete":
            raise ValueError(f"{test}: test run is not complete")
        new = _csv(test / "rollouts.csv")
        mine = {r["scenario_id"]: r["success"] for r in new if r["row"] == "expert"}
        expert[test.name] = (sum(mine[s] == ref_expert.get(s) for s in mine), len(mine))
        rollouts += [r for r in new if r["row"] != "expert"]
    return rollouts, expert


def per_seed_rates(rollouts: list[dict]) -> dict[tuple, float]:
    groups = {}
    for r in rollouts:
        row, arch = split_model_id(r["model_id"])
        groups.setdefault((row, arch, _seed(r["model_id"])), []).append(r["success"] == "True")
    return {k: float(np.mean(v)) for k, v in groups.items()}


def validation_and_hold(experiment: Path) -> dict[tuple, dict]:
    """(row, architecture) -> validation holds per seed, pooled CI, hold-test medians and per-seed arm RMSE."""
    out = {}
    for r in _csv(experiment / "runs.csv"):
        key = (r["row"].split("-")[0], r["architecture"])
        e = out.setdefault(key, {"holds": [], "n": 0, "stiffness": [], "damping": [], "max_re": [], "rmse": {}})
        e["holds"].append(int(r["student_hold"]))
        e["n"] += int(r["student_n"])
        e["stiffness"].append(float(r["hold_test_median_stiffness_eig_max"]))
        e["damping"].append(float(r["hold_test_median_damping_diag"]))
        e["max_re"].append(float(r["hold_test_median_max_real_eig"]))
        metrics = _csv(TASK_DIR / Path(r["model"]).parent / "metrics.csv")
        e["rmse"][int(r["seed"])] = float(metrics[-1]["val_arm_rmse_nm"])
    return out


# ---------------------------------------------------------------- outputs

def summary_rows(table, validation, latency) -> list[dict]:
    rows = []
    for e in (x for x in table if x["suite"] == "all"):
        key = (e["row"], e["architecture"])
        v, lat = validation[key], latency[e["architecture"]]
        k = sum(v["holds"])
        lo, hi = wilson(k, v["n"])
        suites = {s: next((x["rate"] for x in table if (x["row"], x["architecture"], x["suite"]) == (*key, s)), None)
                  for s in SUITES}
        rows.append({"row": e["row"], "architecture": e["architecture"], "parameters": lat["parameters"],
                     "flops": lat["flops"], "latency_median_us": lat["median_us"], "latency_p99_us": lat["p99_us"],
                     "validation_hold_per_seed": "/".join(map(str, v["holds"])), "validation_hold_rate": k / v["n"],
                     "validation_ci95": f"{lo:.2f}-{hi:.2f}", "test_k": e["k"], "test_n": e["n"], "test_rate": e["rate"],
                     "test_ci95": f"{e['ci_lo']:.3f}-{e['ci_hi']:.3f}", **{f"test_{s}": suites[s] for s in SUITES},
                     "test_median_completion_s": e["median_completion_s"],
                     "hold_stiffness_eig_max": float(np.median(v["stiffness"])),
                     "hold_damping_diag": float(np.median(v["damping"])),
                     "hold_max_real_eig": float(np.median(v["max_re"]))})
    return sorted(rows, key=lambda r: (r["row"], r["parameters"]))


def write_markdown(path: Path, rows: list[dict], choice: dict, expert: dict, latency: dict) -> None:
    fmt = lambda x: "–" if x is None else f"{100 * x:.1f}"
    lines = ["# Architecture ablation: summary", "",
             "Test = unified-test-v1 (240 scenes, continuous <10 mm hold), pooled over 3 seeds (720 rollouts per line). "
             "Validation = 32 frozen scenes, full student, sustained hold. Latency = isolated single-thread NumPy "
             "predict() on one seed's model. 256x3 test results come from the R1-R8 reference run.", "",
             "| Row | Architecture | Params | FLOPs | Latency µs (p50/p99) | Validation hold per seed | Test success % (95% CI) | "
             "T0 | T1 | T2 | T3 | T4 | T5 | Stiffness max | Damping | Max Re λ |",
             "|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lo, hi = (float(x) for x in r["test_ci95"].split("-"))
        lines.append(f"| {r['row']} | {r['architecture']} | {r['parameters']:,} | {r['flops']:,} | "
                     f"{r['latency_median_us']:.1f} / {r['latency_p99_us']:.1f} | {r['validation_hold_per_seed']} | "
                     f"{fmt(r['test_rate'])} ({100 * lo:.1f}–{100 * hi:.1f}) | "
                     + " | ".join(fmt(r[f"test_{s}"]) for s in SUITES)
                     + f" | {r['hold_stiffness_eig_max']:.1f} | {r['hold_damping_diag']:.1f} | {r['hold_max_real_eig']:.3f} |")
    ref_lat = latency[REFERENCE]
    lines += ["", "## Recommendation (smallest architecture within 3 percentage points of 256x3, all suites)", ""]
    for row, arch in choice.items():
        lat = latency[arch]
        lines.append(f"- {row}: **{arch}**: {ref_lat['parameters'] / lat['parameters']:.1f}x fewer parameters, "
                     f"{ref_lat['median_us'] / lat['median_us']:.2f}x faster predict() than 256x3.")
    ceiling = {r["row"]: r["test_T0"] for r in rows if r["architecture"] == REFERENCE}
    lines += ["", "T0 ceiling (256x3): " + ", ".join(f"{k} {fmt(v)}%" for k, v in ceiling.items())
              + "; most of the separation comes from T1-T3 and T5.",
              "Expert A determinism (per-scene success identical to the reference run): "
              + ", ".join(f"{name} {same}/{n}" for name, (same, n) in expert.items()) + "."]
    path.write_text("\n".join(lines) + "\n")


def _seed_band(seed_rates, row, arch):
    vals = [v for (r, a, _), v in seed_rates.items() if (r, a) == (row, arch)]
    return min(vals), max(vals)


def plot(out: Path, rows: list[dict], seed_rates: dict, validation: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3dd", "grid.linewidth": 0.6, "axes.axisbelow": True})
    by = {(r["row"], r["architecture"]): r for r in rows}
    row_names = sorted({r["row"] for r in rows})

    def scatter(ax, xkey, xlabel):
        for i, row in enumerate(row_names):
            colour, marker = ROW_STYLE[row]
            dodge = 1.07 ** (i - (len(row_names) - 1) / 2)  # rows share x; offset them slightly on the log axis
            pts = sorted((r[xkey], r["test_rate"], r["architecture"]) for r in rows if r["row"] == row)
            for x, y, arch in pts:
                lo, hi = _seed_band(seed_rates, row, arch)
                ax.errorbar(x * dodge, 100 * y, yerr=[[100 * (y - lo)], [100 * (hi - y)]], fmt=marker, color=colour, ms=6,
                            mec="white", mew=1, elinewidth=1, capsize=0, label=row if arch == pts[0][2] else None)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Unified test success (%), seeds pooled")
        ax.legend(frameon=False, loc="lower right")

    for xkey, xlabel, name in (("parameters", "Parameters (log scale)", "success_vs_parameters.png"),
                               ("latency_median_us", "Median predict() latency, µs (log scale)", "success_vs_latency.png")):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        scatter(ax, xkey, xlabel)
        ref = [r for r in rows if r["architecture"] == REFERENCE]
        if ref:
            ax.annotate("256x3 (current)", (ref[0][xkey], min(100 * r["test_rate"] for r in ref)),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7, color="#5f5e5a")
        fig.tight_layout()
        fig.savefig(out / name, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, (title, family) in zip(axes, (("Width (3 layers)", WIDTH_LINE), ("Depth at width 256", DEPTH_256),
                                          ("Depth at width 64", DEPTH_64))):
        for row in row_names:
            colour, marker = ROW_STYLE[row]
            pts = [by[(row, a)] for a in family if (row, a) in by]
            ax.plot([p["parameters"] for p in pts], [100 * p["test_rate"] for p in pts], color=colour, lw=2,
                    marker=marker, ms=6, mec="white", mew=1, label=row)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Parameters (log scale)")
        tick = [by[(row_names[0], a)] for a in family if (row_names[0], a) in by]
        ax.set_xticks([p["parameters"] for p in tick], [p["architecture"].split("-", 1)[1] for p in tick], fontsize=7)
        ax.minorticks_off()
    axes[0].set_ylabel("Unified test success (%)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "width_and_depth_lines.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for row in row_names:
        colour, marker = ROW_STYLE[row]
        xs, ys = [], []
        for (r, arch), v in validation.items():
            if r != row:
                continue
            for seed, rmse in v["rmse"].items():
                if (row, arch, seed) in seed_rates:
                    xs.append(rmse)
                    ys.append(100 * seed_rates[(row, arch, seed)])
        ax.plot(xs, ys, marker, color=colour, ms=5, mec="white", mew=0.8, label=row)
    ax.set_xlabel("Validation arm-torque RMSE, N·m (offline)")
    ax.set_ylabel("Unified test success (%), per seed")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "offline_rmse_vs_test_success.png", dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--reference-test", type=Path, required=True)
    p.add_argument("--tests", type=Path, nargs="+", required=True)
    p.add_argument("--latency", type=Path, required=True)
    p.add_argument("--rows", nargs="+", default=["R4", "R6", "R8"])
    args = p.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else TASK_DIR / path
    experiment, tests = resolve(args.experiment), [resolve(t) for t in args.tests]
    rollouts, expert = load_tests(resolve(args.reference_test), tests, set(args.rows))
    latency = {r["architecture"]: {k: (float(v) if k.endswith("_us") else int(v) if k in ("parameters", "flops") else v)
                                   for k, v in r.items()} for r in _csv(resolve(args.latency))}
    table = aggregate(rollouts, set(args.rows))
    for e in table:
        e["parameters"] = latency[e["architecture"]]["parameters"]
    validation = validation_and_hold(experiment)
    rows = summary_rows(table, validation, latency)
    choice = recommend(table)
    with open(experiment / "architecture_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_markdown(experiment / "architecture_summary.md", rows, choice, expert, latency)
    plot(experiment / "figures/architecture", rows, per_seed_rates(rollouts), validation)
    for test in tests:
        (test / "README.md").write_text(
            "# Architecture-ablation test run\n\nThe suite labels each model by its row only, so the row-level tables "
            "in this directory pool every architecture of a row. Use "
            f"`{rel(experiment)}/architecture_summary.md` for per-architecture results.\n")
    print((experiment / "architecture_summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
