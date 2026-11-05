"""Ablation results grouped as Accuracy and Generalization (report figures; no simulation here).

  figures/ablation_accuracy.{pdf,png}        in the training range: success on T0 (in range), T4 (shifted start) and
                                             T5 (20 s hold), and the in-range terminal error (T0 + T4 + T5)
  figures/ablation_generalization.{pdf,png}  outside the training range: success on T1 per distance band (3-6, 6-10,
                                             10-15 m; training stops at 3.1 m), T2 (near, 0.30-0.49 m) and T3 per height
                                             band (low 0.05-0.10 m, high 1.25-1.30 m)

Success panels: the four methods, each without (R1/R3/R5/R7, hatched orange) and with (R2/R4/R6/R8, solid blue) the
Gate, as in ablation_overview; 3 seeds pooled; error bars are the 95% scene-cluster bootstrap from
tables/test_by_band.csv; the dashed line is the expert on the same scenes. Terminal-error panel: median and
interquartile range of the final-second maximum error over all in-range rollouts (from the test records). R5-R8 are
ablation v3. Run make_tables.py first.

Usage (from final/robotnik_arm_task/; needs matplotlib):
    python report/06_ablation/code/plot_accuracy_generalization.py
"""
import csv

import numpy as np

from plot_ablation import BLUE, INK, METHODS, MUTED, ORANGE, PAIRS, plt, read, save
from test_data import load_rollouts

ROWS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
GATE = {"R2", "R4", "R6", "R8"}
OURS = "R8"


def band_table():
    return {(r["row"], r["suite"], r["band"]): r for r in read("test_by_band.csv")}


def success_panel(ax, cell, suite, band, title, show_ylabel):
    """Grouped bars: method on x, without / with Gate side by side, 95% CI, expert as a dashed reference."""
    x, w = np.arange(len(METHODS)), 0.36
    for j, (colour, hatch, label) in enumerate([(ORANGE, "////", "without Gate"), (BLUE, None, "with Gate")]):
        rows = [PAIRS[m][j] for m in METHODS]
        vals = np.array([[float(cell[(k, suite, band)][c]) for c in ("rate_pct", "scene_boot95_low_pct",
                                                                      "scene_boot95_high_pct")] for k in rows])
        xs = x + (j - 0.5) * (w + 0.03)
        ax.bar(xs, vals[:, 0], w, color=colour if hatch is None else "white", edgecolor=colour, hatch=hatch,
               linewidth=0.9, label=label)
        ax.errorbar(xs, vals[:, 0], yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="none",
                    ecolor=INK, elinewidth=0.6, capsize=1.5)
        for xi, v, top, k in zip(xs, vals[:, 0], vals[:, 2], rows):
            ax.annotate(f"{v:.0f}", (xi, top), xytext=(0, 1.5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=5.3, color=INK, fontweight="bold" if k == OURS else None)
            ax.annotate(k, (xi, 0), xytext=(0, -1.5), textcoords="offset points", ha="center", va="top",
                        fontsize=5, color=INK if k == OURS else MUTED, fontweight="bold" if k == OURS else None)
    expert = float(cell[("expert", suite, band)]["rate_pct"])
    ax.axhline(expert, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
    ax.annotate(f"Expert {expert:.0f}", (-0.45, expert), xytext=(0, 1.5), textcoords="offset points",
                ha="left", va="bottom", fontsize=5.3, color=MUTED, annotation_clip=False)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xticks(x, METHODS)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    if show_ylabel:
        ax.set_ylabel("Success (%)")


def terminal_error_panel(ax):
    """Median and IQR of the final-second max error over T0 + T4 + T5, working controllers (R4-R8) and the expert."""
    rolls = [r for r in load_rollouts() if r["suite"] in ("T0", "T4", "T5")]
    order = ["expert", "R4", "R5", "R6", "R7", "R8"]
    names = {"expert": "Expert", "R4": "R4 OU + Gate", "R5": "R5 DAgger", "R6": "R6 DAgger + Gate",
             "R7": "R7 f-DAgger", "R8": "R8 f-DAgger + Gate"}
    for i, k in enumerate(order):
        e = np.array([float(r["final1s_max_err_mm"]) for r in rolls if r["row"] == k])
        q1, med, q3 = np.quantile(e, [0.25, 0.5, 0.75])
        colour = MUTED if k == "expert" else (BLUE if k in GATE else ORANGE)
        ax.plot([q1, q3], [i, i], color=colour, linewidth=1.6, solid_capstyle="butt")
        filled = k == "expert" or k in GATE
        ax.plot(med, i, marker="o", ms=4.5, color=colour if filled else "white", markeredgecolor=colour,
                markeredgewidth=1.1)
        ax.annotate(f"{med:.1f}", (q3, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=5.3,
                    color=INK, fontweight="bold" if k == OURS else None)
    ax.axvline(10, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.annotate("10 mm (success threshold)", (10, -0.45), xytext=(3, 0), textcoords="offset points", fontsize=5.3,
                color=MUTED, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1500)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100], ["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_yticks(range(len(order)), [names[k] for k in order])
    for t in ax.get_yticklabels():
        if t.get_text().startswith(OURS):
            t.set_fontweight("bold")
            t.set_color(INK)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_xlabel("Final-second max error (mm, log)")
    ax.set_title("(d) Terminal precision in range: final-second max error over T0 + T4 + T5 (336 rollouts per row)",
                 loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ratio = next(r for r in csv.DictReader(open(ROOT_TABLES / "precision.csv"))
                 if r["group"] == "T0+T4+T5" and r["row"] == "R8/R4")
    r6 = next(r for r in csv.DictReader(open(ROOT_TABLES / "precision.csv"))
              if r["group"] == "T0+T4+T5" and r["row"] == "R8/R6")
    ax.text(0.99, 0.97, "Paired per-scene error ratio (geometric mean)\n"
                       f"R8 / R4: {float(ratio['error_ratio']):.2f}× [{float(ratio['ratio_boot95_low']):.2f}, "
                       f"{float(ratio['ratio_boot95_high']):.2f}]\n"
                       f"R8 / R6: {float(r6['error_ratio']):.2f}× [{float(r6['ratio_boot95_low']):.2f}, "
                       f"{float(r6['ratio_boot95_high']):.2f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=INK, linespacing=1.5)


def caption(fig, text):
    """Figure note wrapped to the figure width (so bbox_inches='tight' does not widen the figure)."""
    import textwrap
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(text, 175)), fontsize=5.6, color=MUTED, ha="left", va="bottom",
             linespacing=1.3)


def legend(fig, ax, y):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, handlelength=1.4,
               bbox_to_anchor=(0.5, y))


def accuracy(cell):
    fig = plt.figure(figsize=(7.16, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 0.85])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)] + [fig.add_subplot(gs[1, :])]
    success_panel(axes[0], cell, "T0", "all", "(a) T0 in range (64 scenes)", True)
    success_panel(axes[1], cell, "T4", "all", "(b) T4 shifted start (24)", False)
    success_panel(axes[2], cell, "T5", "all", "(c) T5 long hold, 20 s (24)", False)
    for ax in axes[1:3]:
        ax.set_yticklabels([])
    terminal_error_panel(axes[3])
    legend(fig, axes[0], 1.05)
    fig.tight_layout(w_pad=0.7, h_pad=1.0, rect=(0, 0.06, 1, 1))
    caption(fig, "Accuracy: scenes inside the training range. (a)-(c) success, 3 seeds pooled; error bars 95% "
             "scene bootstrap; dashed line: expert. (d) dot: median, line: interquartile range over all in-range "
             "rollouts (R1–R3 omitted: their median error is 0.28 m or more); ratio: per-scene geometric mean, 95% scene bootstrap (secondary analysis).")
    save(fig, "ablation_accuracy")


def generalization(cell):
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.1), sharey=True)
    panels = [("T1", "3-6m", "(a) T1 far, 3–6 m (24 scenes)"), ("T1", "6-10m", "(b) T1 far, 6–10 m (24)"),
              ("T1", "10-15m", "(c) T1 far, 10–15 m (24)"), ("T2", "all", "(d) T2 near, 0.30–0.49 m (24)"),
              ("T3", "low", "(e) T3 low, 0.05–0.10 m (16)"), ("T3", "high", "(f) T3 high, 1.25–1.30 m (16)")]
    for i, (ax, (suite, band, title)) in enumerate(zip(axes.flat, panels)):
        success_panel(ax, cell, suite, band, title, i % 3 == 0)
    legend(fig, axes[0, 0], 1.04)
    fig.tight_layout(w_pad=0.7, h_pad=1.2, rect=(0, 0.06, 1, 1))
    caption(fig, "Generalization: scenes outside the training range (base-to-target distance 0.5–3.1 m, "
             "target height 0.1–1.25 m). Success, 3 seeds pooled; error bars 95% scene bootstrap; dashed line: expert "
             "on the same scenes (it holds 10 of the 16 T3 low targets).")
    save(fig, "ablation_generalization")


from plot_ablation import TAB as ROOT_TABLES  # noqa: E402

if __name__ == "__main__":
    cell = band_table()
    accuracy(cell)
    generalization(cell)
