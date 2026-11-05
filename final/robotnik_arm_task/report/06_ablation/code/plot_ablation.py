"""Training-method ablation R1-R8: report figures drawn from ../tables and ../data (no simulation here).

  figures/ablation_overview.{pdf,png}   validation hold, test T0 and test over all 240 scenes for the four methods,
                                        each without (R1/R3/R5/R7) and with (R2/R4/R6/R8) the gate; 95% intervals
  figures/test_suites.{pdf,png}         success per test suite T0-T5 and over all 240 scenes, R1-R8 and the expert
  figures/dagger_rounds.{pdf,png}       validation hold per DAgger round for DAgger and f-DAgger, without / with gate
  figures/terminal_error_ecdf.{pdf,png} T0 terminal error (max over the final second), one panel per method
  figures/gate_effect.{pdf,png}         with-minus-without Gate difference per method, with 95% paired scene-bootstrap CIs

Colour encodes the gate in every figure (orange = without, blue = with); hatching or line style repeats it, so it
never rests on colour alone. Run make_tables.py first.

Usage (from final/robotnik_arm_task/; needs matplotlib):
    python report/06_ablation/code/plot_ablation.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from test_data import load_rollouts

ROOT = Path(__file__).resolve().parents[1]  # report/06_ablation
DATA, TAB, FIG = ROOT / "data", ROOT / "tables", ROOT / "figures"
# Reference palette, categorical slots 1-2 (validated pair, CVD-safe) and the text/grid neutrals.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.dpi": 300, "hatch.linewidth": 0.6})
METHODS = ["Baseline", "OU", "DAgger", "f-DAgger"]
PAIRS = {"Baseline": ("R1", "R2"), "OU": ("R3", "R4"), "DAgger": ("R5", "R6"), "f-DAgger": ("R7", "R8")}
SUITES = ["T0", "T1", "T2", "T3", "T4", "T5"]
SUITE_LABEL = {"T0": "T0\nin range", "T1": "T1\nfar", "T2": "T2\nnear", "T3": "T3\nheight\nedges",
               "T4": "T4\nshifted\nstart", "T5": "T5\nlong\nhold", "all": "All\n240"}


def read(name):
    with open(TAB / name, newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figures/{name}.{{pdf,png}}")


def overview():
    val = {r["row"]: r for r in read("validation.csv")}
    test = {(r["row"], r["suite"]): r for r in read("test_by_band.csv") if r["band"] == "all"}
    panels = [
        ("Validation hold (32 scenes)",
         lambda k: tuple(100 * float(val[k][c]) for c in ("rate", "scene_boot95_low", "scene_boot95_high"))),
        ("Test T0, in range (64 scenes)",
         lambda k: tuple(float(test[(k, "T0")][c]) for c in ("rate_pct", "scene_boot95_low_pct", "scene_boot95_high_pct"))),
        ("Test, all 240 scenes (T0–T5)",
         lambda k: tuple(float(test[(k, "all")][c]) for c in ("rate_pct", "scene_boot95_low_pct", "scene_boot95_high_pct"))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    x, w = np.arange(len(METHODS)), 0.36
    for ax, (title, get) in zip(axes, panels):
        for j, (colour, hatch, label) in enumerate([(ORANGE, "////", "without Gate"), (BLUE, None, "with Gate")]):
            rows = [PAIRS[m][j] for m in METHODS]
            vals = np.array([get(k) for k in rows])
            xs = x + (j - 0.5) * (w + 0.03)
            ax.bar(xs, vals[:, 0], w, color=colour if hatch is None else "white", edgecolor=colour, hatch=hatch,
                   linewidth=0.9, label=label)
            ax.errorbar(xs, vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="none",
                        ecolor=INK, elinewidth=0.6, capsize=1.5)
            for xi, v, top, k in zip(xs, vals[:, 0], vals[:, 2], rows):
                ax.annotate(f"{v:.0f}", (xi, top), xytext=(0, 1.5), textcoords="offset points", ha="center",
                            va="bottom", fontsize=5.5, color=INK)
                ax.annotate(k, (xi, 0), xytext=(0, -1.5), textcoords="offset points", ha="center", va="top",
                            fontsize=5, color=MUTED)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xticks(x, METHODS)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Success (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, handlelength=1.4,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.8, rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.005, "Bars: success rate, 3 seeds pooled. Error bars: 95% percentile bootstrap over scenes (1,000 resamples; "
             "a scene's 3 seed rollouts resampled together). Intervals are conditional on the 3 trained models.",
             fontsize=5.8, color=MUTED, ha="left", va="bottom")
    save(fig, "ablation_overview")


def test_suites():
    rows = read("test_by_suite.csv")
    order = [r for r in rows if r["row"] != "expert"] + [r for r in rows if r["row"] == "expert"]
    cols = [f"{s}_pct" for s in SUITES] + ["all240_pct"]
    m = np.array([[float(r[c]) for c in cols] for r in order])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if m[i, j] >= 60 else INK, fontweight="bold" if j == m.shape[1] - 1 else None)
    ax.set_xticks(range(len(cols)), [SUITE_LABEL[s] for s in SUITES] + [SUITE_LABEL["all"]], fontsize=5.8)
    ax.set_yticks(range(len(order)), [f"{r['row']}  {r['report_name']}" if r["row"] != "expert" else "Expert A"
                                      for r in order])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for y in (1.5, 3.5, 5.5, 7.5):
        ax.axhline(y, color="white", linewidth=1.5)
    ax.axvline(5.5, color="white", linewidth=1.5)
    ax.set_title("Test success (%), 3 seeds pooled; colour = success", loc="left", color=INK, fontsize=6.5, pad=30)
    fig.tight_layout()
    save(fig, "test_suites")


def dagger_rounds():
    rows = read("validation_by_round.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style = {"R5": (ORANGE, "--", "o"), "R6": (BLUE, "-", "o"), "R7": (ORANGE, "--", "s"), "R8": (BLUE, "-", "s")}
    ends, LABEL_GAP = [], 11.0  # LABEL_GAP in percentage points
    for k, (colour, ls, marker) in style.items():
        g = sorted((r for r in rows if r["row"] == k), key=lambda r: int(r["round"]))
        rnd = np.array([int(r["round"]) for r in g])
        num = [list(map(int, r["pooled"].split("/"))) for r in g]
        rate = np.array([100 * a / b for a, b in num])
        three = np.array([r["seeds"] == "3" for r in g])
        ax.plot(rnd, rate, color=colour, linestyle=ls, linewidth=1.3)
        ax.scatter(rnd[three], rate[three], color=colour, marker=marker, s=22, zorder=3, edgecolor="white",
                   linewidth=0.8)
        ax.scatter(rnd[~three], rate[~three], facecolor="white", edgecolor=colour, marker=marker, s=22, zorder=3,
                   linewidth=1.0)
        ends.append((rate[-1], f"{k} {g[0]['report_name']}"))
    # End labels, top to bottom by final value, at least LABEL_GAP points apart so close rows do not overlap.
    y_prev = None
    for y, text in sorted(ends, reverse=True):
        y_label = y if y_prev is None else min(y, y_prev - LABEL_GAP)
        ax.annotate(text, (3, y), xytext=(3.2, y_label), textcoords="data", va="center", fontsize=6, color=INK,
                    annotation_clip=False)
        y_prev = y_label
    ax.set_xticks(range(4), ["0\n(no DAgger)", "1", "2", "3"])
    ax.set_xlim(-0.15, 3.15)
    ax.set_ylim(0, 104)
    ax.set_xlabel("DAgger round")
    ax.set_ylabel("Validation sustained hold (%)")
    ax.set_title("Validation hold per DAgger round", loc="left", color=INK)
    ax.plot([], [], color=ORANGE, linestyle="--", label="without Gate")
    ax.plot([], [], color=BLUE, linestyle="-", label="with Gate")
    ax.scatter([], [], color=MUTED, marker="o", s=14, label="DAgger (from clean)")
    ax.scatter([], [], color=MUTED, marker="s", s=14, label="f-DAgger (from OU)")
    ax.scatter([], [], color=MUTED, marker="o", s=14, label="filled: 3 seeds")
    ax.scatter([], [], facecolor="white", edgecolor=MUTED, marker="o", s=14, label="open: seed 0 only")
    ax.legend(loc="upper center", frameon=False, ncol=2, columnspacing=1.0, handlelength=1.8,
              bbox_to_anchor=(0.5, -0.40))
    fig.tight_layout()
    fig.subplots_adjust(right=0.74)
    save(fig, "dagger_rounds")


def terminal_error_ecdf():
    rolls = [r for r in load_rollouts() if r["suite"] == "T0"]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.85), sharey=True)
    for ax, m in zip(axes, METHODS):
        for k, colour, ls, label in [(PAIRS[m][0], ORANGE, "--", "without Gate"), (PAIRS[m][1], BLUE, "-", "with Gate")]:
            err = np.sort([float(r["final1s_max_err_mm"]) for r in rolls if r["row"] == k])
            ax.step(err, np.arange(1, len(err) + 1) / len(err) * 100, where="post", color=colour, linestyle=ls,
                    linewidth=1.2, label=f"{k} {label}")
        expert = np.sort([float(r["final1s_max_err_mm"]) for r in rolls if r["row"] == "expert"])
        ax.step(expert, np.arange(1, len(expert) + 1) / len(expert) * 100, where="post", color=MUTED,
                linestyle=":", linewidth=1.0, label="Expert A")
        ax.axvline(10, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.3, 3000)
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.set_ylim(0, 101)
        ax.set_title(f"{m}: {PAIRS[m][0]} / {PAIRS[m][1]}", loc="left", color=INK)
        ax.set_xlabel("Final-second max error (mm)")
    axes[0].set_ylabel("T0 rollouts (%)")
    axes[0].annotate("10 mm", (10, 96), xytext=(2, 0), textcoords="offset points", fontsize=5.5, color=MUTED, va="top")
    handles = [plt.Line2D([], [], color=ORANGE, linestyle="--", linewidth=1.2, label="without Gate"),
               plt.Line2D([], [], color=BLUE, linestyle="-", linewidth=1.2, label="with Gate"),
               plt.Line2D([], [], color=MUTED, linestyle=":", linewidth=1.0, label="Expert A")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save(fig, "terminal_error_ecdf")


def gate_effect():
    """Forest plot: Gate effect (with - without, percentage points) per method and metric, 95% paired bootstrap CI."""
    comp = [r for r in read("comparisons.csv") if r["factor"] == "Gate"]
    metrics = ["validation", "T0", "T1", "T2", "T3", "T4", "T5", "all 240"]
    label = {"validation": "Validation", "T0": "T0 in range", "T1": "T1 far", "T2": "T2 near", "T3": "T3 height edges",
             "T4": "T4 shifted start", "T5": "T5 long hold", "all 240": "All 240 (test)"}
    pairs = [("R1", "R2"), ("R3", "R4"), ("R5", "R6"), ("R7", "R8")]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.5), sharey=True)
    for ax, (a, b), m in zip(axes, pairs, METHODS):
        for i, metric in enumerate(metrics):
            r = next(c for c in comp if c["without_row"] == a and c["metric"] == metric)
            d, lo, hi = (float(r[k]) for k in ("diff_pp", "paired_boot95_low_pp", "paired_boot95_high_pp"))
            sig = r["significant_holm_0.05"] == "True"
            ax.plot([lo, hi], [i, i], color=BLUE, linewidth=1.2, solid_capstyle="butt")
            ax.plot(d, i, marker="o", ms=4.5, color=BLUE if sig else "white", markeredgecolor=BLUE, markeredgewidth=1.0)
        ax.axvline(0, color=MUTED, linewidth=0.7)
        ax.set_title(f"{m}: {b} − {a}", loc="left", color=INK)
        ax.set_xlim(-60, 100)
        ax.set_xticks([-50, 0, 50, 100])
        ax.set_xlabel("Gate effect (pp)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(metrics)), [label[m] for m in metrics])
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.005, "Dot: with − without Gate, same scenes, 3 seeds pooled. Line: 95% paired bootstrap CI over scenes "
             "(10,000 resamples). Filled: Holm-adjusted sign-flip permutation p < 0.05 over all 112 comparisons; open: not.",
             fontsize=5.8, color=MUTED, ha="left", va="bottom")
    save(fig, "gate_effect")


if __name__ == "__main__":
    overview()
    gate_effect()
    test_suites()
    dagger_rounds()
    terminal_error_ecdf()
