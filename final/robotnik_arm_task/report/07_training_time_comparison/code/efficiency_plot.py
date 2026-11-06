"""Efficiency figure and table: training time vs T0 (in-distribution) test success, rounds 1-3 of each method.

Reads success_by_round.csv and cost_by_round.csv written by make_comparison.py.
Names ("+ Gate" = the 22-input models, as in ../../06_ablation/; every model here has the Gate):
  - f-DAgger + Gate (ours): phase-gated OU noise, then DAgger (R4 -> R8).
  - DAgger + Gate: DAgger from clean data (R2 -> R6).
  - DART + Gate: full DART with noise optimisation, alpha = 6 (R2 -> R9-dart-full-a6).
Pipeline time is reconstructed from stage timings. It adds each method's own time (OU-noise collection; every
train -> collect cycle) to the shared clean collection and round-0 training (SHARED_MIN, the same for every method).
The figure shows rounds 1-3. Curves are monotone PCHIP interpolation between the measured rounds; bands are 95%
scene-bootstrap CIs. The table keeps round 0 as well.

Usage (from final/robotnik_arm_task/):
    python report/07_training_time_comparison/code/efficiency_plot.py --tables report/07_training_time_comparison/tables \
        --fig-dir report/07_training_time_comparison/figures
Writes efficiency_T0.{pdf,png} to --fig-dir and efficiency_table.{csv,tex} to --tables.
"""
import argparse
import csv
from pathlib import Path

SHARED_MIN = 9.2  # clean collection 5.3 min + one 42k-update round-0 training ~3.9 min (measured, 20 workers)
SERIES = [  # (label, pipeline key in the CSVs, colour slot, marker); fixed categorical order
    ("DAgger + Gate", "DAgger + Gate", "#2a78d6", "o"),
    ("f-DAgger + Gate (ours)", "f-DAgger + Gate", "#eb6834", "s"),
    ("DART + Gate", "DART + Gate", "#1baf7a", "^"),  # full DART with noise optimisation, alpha = 6
]
GROUP = "T0 core (in distribution)"
TEXT, MUTED, GRID, BC_GREY = "#0b0b0b", "#52514e", "#e8e7e3", "#8a8984"


def load(comp: Path):
    succ = {(r["pipeline"], int(r["round"])): r for r in csv.DictReader(open(comp / "success_by_round.csv"))
            if r["group"] == GROUP}
    cost = {(c["pipeline"], int(c["round"])): c for c in csv.DictReader(open(comp / "cost_by_round.csv"))}
    rows = []
    for label, key, colour, marker in SERIES:
        for k in range(4):
            if (key, k) in succ and (key, k) in cost:
                s, c = succ[(key, k)], cost[(key, k)]
                rows.append({"method": label, "round": k, "added_min": float(c["added_wall_min"]),
                             "total_min": round(SHARED_MIN + float(c["added_wall_min"]), 1),
                             "cycles": int(c["sequential_train_collect_cycles"]),
                             "t0_success": 100 * float(s["rate"]), "ci_low": 100 * float(s["ci_low"]),
                             "ci_high": 100 * float(s["ci_high"]), "successes": f"{s['successes']}/{s['n']}"})
    return rows


def plot(rows, out: Path):
    """Smooth monotone (PCHIP) curves through measured rounds 1-3 (round 0 omitted), with 95% CI bands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import PchipInterpolator
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for label, _, colour, marker in SERIES:
        pts = sorted((r for r in rows if r["method"] == label and r["round"] >= 1), key=lambda r: r["total_min"])
        x = np.array([r["total_min"] for r in pts])
        y, lo, hi = (np.array([r[k] for r in pts]) for k in ("t0_success", "ci_low", "ci_high"))
        xs = np.linspace(x[0], x[-1], 200)
        ax.fill_between(xs, np.clip(PchipInterpolator(x, lo)(xs), 0, 100), np.clip(PchipInterpolator(x, hi)(xs), 0, 100),
                        color=colour, alpha=0.13, lw=0, zorder=1)
        ax.plot(xs, PchipInterpolator(x, y)(xs), color=colour, lw=1.6, zorder=2)
        ax.plot(x, y, ls="none", marker=marker, ms=4.5, color=colour, mec="white", mew=0.7, zorder=3)
    ax.set_xlabel("Cumulative pipeline time (min)")
    ax.set_ylabel("T0 success rate (%)")
    ax.set_xlim(16, 39)
    ax.set_ylim(50, 101)
    ax.set_yticks(range(50, 101, 10))
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    from matplotlib.lines import Line2D
    order = [SERIES[1], SERIES[0], SERIES[2]]  # ours first in the legend; colours keep their fixed slots
    handles = [Line2D([], [], color=c, lw=1.6, marker=m, ms=4.5, mec="white", mew=0.7) for _, _, c, m in order]
    ax.legend(handles, [lab for lab, *_ in order], loc="lower right", frameon=False, handlelength=1.8, labelcolor=TEXT)
    fig.tight_layout(pad=0.25)
    fig.savefig(out / "efficiency_T0.pdf")
    fig.savefig(out / "efficiency_T0.png", dpi=300)


def write_table(rows, out: Path):
    with open(out / "efficiency_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Method & Round & Time (min) & Cycles & T0 success (\%) & 95\% CI \\", r"\midrule"]
    for label, *_ in SERIES:
        for r in [r for r in rows if r["method"] == label]:
            lines.append(f"{label} & r{r['round']} & {r['total_min']:.1f} & {r['cycles']} & {r['t0_success']:.1f} & "
                         f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}] \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (out / "efficiency_table.tex").write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tables", type=Path, required=True, help="directory with success_by_round.csv, cost_by_round.csv")
    p.add_argument("--fig-dir", type=Path, help="default: --tables")
    args = p.parse_args(argv)
    fig_dir = args.fig_dir or args.tables
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = load(args.tables)
    plot(rows, fig_dir)
    write_table(rows, args.tables)
    for r in rows:
        print(f"{r['method']:28s} r{r['round']} {r['total_min']:5.1f} min  T0 {r['t0_success']:5.1f}% "
              f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]  cycles {r['cycles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
