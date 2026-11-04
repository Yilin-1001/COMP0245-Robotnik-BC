"""Training curves of the ablation models: losses and offline validation metrics per update, and DAgger rounds.

Reads checkpoints/ablation/<experiment>/<run>/metrics.csv (one line per 500 updates, written while a run trains, so
the figures can be redrawn mid-training) and, for DAgger experiments, results/ablation/<experiment>/dagger/<row>/
summary.csv. Writes, per experiment, results/ablation/<experiment>/figures/training/:
  <row>.png / .pdf      one figure per ablation row: training losses (top) and offline validation metrics (bottom).
                        v1 rows overlay seeds 0-2; DAgger rows overlay round 0 (the base row's v1 seed-0 model, grey)
                        and rounds 1-3 (seed 0 solid; the chosen round's seeds 1-2 dashed / dotted).
  dagger_rounds.png / .pdf   closed-loop validation sustained hold (of 32) against the DAgger round, one panel per row.
Offline metrics are measured on the expert's states; closed-loop success is the selection metric (pipeline Section 6).

Usage (from final/robotnik_arm_task/):
    python -m evaluation.plot_training_curves                                   # every experiment in checkpoints/ablation/
    python -m evaluation.plot_training_curves --experiments 2026-11-01__training-ablation-v2 [--rows R6-gate-dagger]
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from core.model_names import canonical_id, legacy_id, report_label

TASK_DIR = Path(__file__).resolve().parents[1]
RUN = re.compile(r"^(?P<variant>.+?)__(?P<arch>[^_]+)__s(?P<seed>\d+)__(?P<version>v\d+)__(?P<date>\d{4}-\d{2}-\d{2})$")
ROUND = re.compile(r"^(?P<row>.+)-r(?P<round>\d+)$")

# Categorical slots 1-3 of the validated reference palette (all-pairs CVD-safe); markers are the secondary encoding.
ROUND_STYLE = {0: ("#8a8986", "o"), 1: ("#2a78d6", "o"), 2: ("#eb6834", "s"), 3: ("#1baf7a", "^")}
SEED_DASH = {0: "-", 1: "--", 2: ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

TRAIN_PANELS = [("train_loss", "Total training loss", "standardised", True),
                ("train_arm_loss", "Arm action loss", "standardised", True),
                ("train_wheel_loss", "Wheel action loss", "standardised", True),
                ("jacobian_loss", "Damping-Jacobian loss", "standardised", True),
                ("lr", "Learning rate", "", False)]
VAL_PANELS = [("val_arm_rmse_nm", "Validation arm RMSE", "N·m", False),
              ("val_wheel_rmse_rad_s", "Validation wheel RMSE", "rad/s", False),
              ("val_hold_arm_rmse_nm", "Validation hold-phase arm RMSE", "N·m", False),
              ("val_hold_bias_max_abs_nm", "Hold torque bias, max |·| (dashed: 1 N·m)", "N·m", False),
              ("val_damping_median_diag", "Learned damping (dashed: −30)", "N·m·s/rad", False)]
REFERENCE = {"val_damping_median_diag": -30.0, "val_hold_bias_max_abs_nm": 1.0}


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    """metrics.csv -> column arrays (NaN where a value is missing); jacobian_loss comes from the JSON extra column."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key in rows[0] if rows else []:
        if key != "extra":
            out[key] = np.array([_num(r[key]) for r in rows])
    out["jacobian_loss"] = np.array([_num((json.loads(r["extra"]) if r.get("extra") else {}).get("jacobian_loss"))
                                     for r in rows])
    return out


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def discover(experiment: Path) -> list[dict]:
    """Every run of one experiment directory that has a metrics.csv, with its row, round and seed."""
    runs = []
    if not experiment.is_dir():  # e.g. a DAgger experiment whose models are on another machine: rounds figure only
        return runs
    for d in sorted(p for p in experiment.iterdir() if (p / "metrics.csv").is_file()):
        m = RUN.match(d.name)
        if m is None:
            continue
        variant, r = m["variant"], ROUND.match(m["variant"])
        row, rnd = (r["row"], int(r["round"])) if r and "dagger" in r["row"].lower() else (variant, None)
        runs.append({"dir": d, "name": d.name, "row": row, "round": rnd, "seed": int(m["seed"]), "arch": m["arch"]})
    return runs


def round_zero(row: str) -> Path | None:
    """The base row's v1 seed-0 run directory (round 0 of a DAgger row), if this machine has it."""
    from experiments.dagger_rounds import ROWS, V1_DIR, V1_RUN
    row = legacy_id(row)
    if row not in ROWS:
        return None
    d = TASK_DIR / V1_DIR / V1_RUN.format(row=ROWS[row][0])
    return d if (d / "metrics.csv").is_file() else None


def series_for_row(row: str, runs: list[dict]) -> list[dict]:
    """Curves to draw for one row: label, colour, marker, dash and metrics, in legend order."""
    series = []
    dagger = any(r["round"] is not None for r in runs)
    if dagger:
        base = round_zero(row)
        if base is not None:
            series.append({"label": f"round 0 = {report_label(base.name.split('__')[0])} (v1) s0", "colour": ROUND_STYLE[0][0],
                           "marker": ROUND_STYLE[0][1], "dash": "-", "metrics": read_metrics(base / "metrics.csv")})
    for r in sorted(runs, key=lambda r: (r["round"] or 0, r["seed"])):
        colour, marker = ROUND_STYLE[r["round"]] if dagger else (ROUND_STYLE[1][0], None)
        label = f"r{r['round']} s{r['seed']}" if dagger else f"s{r['seed']}"
        series.append({"label": label + (f" ({r['arch']})" if len({x['arch'] for x in runs}) > 1 else ""),
                       "colour": colour, "marker": marker, "dash": SEED_DASH.get(r["seed"], "-."),
                       "metrics": read_metrics(r["dir"] / "metrics.csv")})
    return series


def _style(ax, title: str, unit: str, log: bool) -> None:
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    if unit:
        ax.set_ylabel(unit, fontsize=7.5, color=MUTED)
    if log:
        from matplotlib.ticker import LogLocator, NullFormatter
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0e}".replace("e-0", "e-"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:g}k")


def plot_row(row: str, series: list[dict], out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), sharex=True)
    for ax, (key, title, unit, log) in zip(axes.flat, TRAIN_PANELS + VAL_PANELS):
        drawn = False
        for s in series:
            x, y = s["metrics"].get("update"), s["metrics"].get(key)
            if x is None or y is None or not np.isfinite(y).any() or (log and not (y[np.isfinite(y)] > 0).any()):
                continue
            ax.plot(x, y, color=s["colour"], linestyle=s["dash"], linewidth=1.4, marker=s["marker"], markersize=4.5,
                    markevery=max(1, len(x) // 6), markerfacecolor=s["colour"], markeredgecolor="white",
                    markeredgewidth=0.8, label=s["label"])
            drawn = True
        if key in REFERENCE:
            ax.axhline(REFERENCE[key], color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        _style(ax, title, unit, log and drawn)
        if not drawn:
            ax.text(0.5, 0.5, "not logged", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("update", fontsize=7.5, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{report_label(row)}: training losses (top) and offline validation metrics on the expert's states (bottom)",
                 fontsize=10, color=INK, x=0.01, y=0.99, ha="left")
    fig.legend(handles, labels, loc="upper left", ncol=min(len(labels), 8), frameon=False, fontsize=8,
               bbox_to_anchor=(0.005, 0.965), labelcolor=INK, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return _save(fig, out / canonical_id(row))


def read_dagger_summary(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [dict(r, round=int(r["round"]), seed=int(r["seed"]), hold=int(r["sustained_hold"].split("/")[0]),
                     n=int(r["n"])) for r in csv.DictReader(f)]


def plot_rounds(summaries: dict[str, list[dict]], out: Path) -> list[Path]:
    """Closed-loop validation hold against the DAgger round, one panel per row (small multiples, shared y)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(summaries)
    fig, axes = plt.subplots(1, len(rows), figsize=(max(3.3 * len(rows), 4.8), 3.3), sharey=True, squeeze=False)
    colour = ROUND_STYLE[1][0]
    for ax, row in zip(axes[0], rows):
        entries = summaries[row]
        n = max(e["n"] for e in entries)
        s0 = sorted((e for e in entries if e["seed"] == 0), key=lambda e: e["round"])
        ax.plot([e["round"] for e in s0], [e["hold"] for e in s0], color=colour, linewidth=1.6, marker="o",
                markersize=6, markeredgecolor="white", markeredgewidth=1.0, label="seed 0", zorder=3)
        for e in s0:  # labels below the points near the ceiling (never crossing the reference line), right of round 0
            below = e["hold"] > n - 5
            ax.annotate(f"{e['hold']}", (e["round"], e["hold"]), xytext=((9 if e["round"] == 0 else -9) if below else 0, -12 if below else 6),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        extra = [e for e in entries if e["seed"] != 0 and e["round"] > 0]
        if extra:
            ax.scatter([e["round"] + 0.14 * e["seed"] for e in extra], [e["hold"] for e in extra], s=30,
                       facecolor="white", edgecolor=colour, linewidth=1.2, label="seeds 1–2 (chosen round)", zorder=4)
        ax.axhline(n, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(f"all {n}", (0, n), xycoords=("axes fraction", "data"), xytext=(3, 3),
                    textcoords="offset points", ha="left", fontsize=7, color=MUTED)
        _style(ax, report_label(row), "validation sustained hold" if ax is axes[0, 0] else "", False)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}")
        ax.set_xticks(range(0, max(e["round"] for e in entries) + 1))
        ax.set_xlabel("DAgger round (0 = v1 base model)", fontsize=7.5, color=MUTED)
        ax.set_ylim(-1, n + 4)
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes[0]), key=lambda hl: len(hl[1]))
    fig.suptitle("Closed-loop validation sustained hold per DAgger round", fontsize=10, color=INK, x=0.01, y=0.98,
                 ha="left")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 0.915), ncol=2, frameon=False,
               fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, out / "dagger_rounds")


def _save(fig, stem: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=200, facecolor="#fcfcfb")
    fig.savefig(paths[1], facecolor="#fcfcfb")
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiments", nargs="*", help="names in checkpoints/ablation/ (default: all of them)")
    p.add_argument("--rows", nargs="*", type=legacy_id, help="report model IDs or historical aliases, e.g. R4-OU-Gate R6-DAgger-Gate")
    args = p.parse_args(argv)
    root = TASK_DIR / "checkpoints/ablation"
    names = args.experiments or sorted(d.name for d in root.iterdir() if d.is_dir())
    for name in names:
        runs = [r for r in discover(root / name) if not args.rows or legacy_id(r["row"]) in args.rows]
        out = TASK_DIR / "results/ablation" / name / "figures/training"
        written = []
        for row in sorted({r["row"] for r in runs}):
            written += plot_row(row, series_for_row(row, [r for r in runs if r["row"] == row]), out)
        dagger_dir = TASK_DIR / "results/ablation" / name / "dagger"
        summaries = {d.name: read_dagger_summary(d / "summary.csv") for d in sorted(dagger_dir.glob("*"))
                     if (d / "summary.csv").is_file() and (not args.rows or legacy_id(d.name) in args.rows)}
        if summaries:
            written += plot_rounds(summaries, out)
        print(f"{name}: {len(runs)} runs, {len(written) // 2} figures -> {out.relative_to(TASK_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
