"""Figure of the frozen scenario split: fitting (221), validation (32) and sealed test (64) scenarios.

Four panels over the sampled scenario dimensions. The dashed lines in the first two panels are the strata of the
validation rule (2 distance bins x 4 height bins x 4 azimuth sectors, one validation scenario per occupied cell).
The test scenarios come from a separate Sobol engine and are shown for coverage only; they never enter selection.

Usage (from final/robotnik_arm_task/): python -m evaluation.plot_scenario_split [--out <png>]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
# Categorical slots 1-3 in fixed order (validated all-pairs); shapes give a second, colour-free encoding.
GROUPS = [("fitting", "#2a78d6", "o", 18), ("validation", "#eb6834", "D", 42), ("test", "#1baf7a", "^", 34)]
PANELS = [("target_dist", "target_z", "Target distance from start (m)", "Target height (m)"),
          ("rel_azimuth", "target_dist", "Target azimuth relative to start heading (rad)", "Target distance from start (m)"),
          ("rel_azimuth", "start_yaw", "Target azimuth relative to start heading (rad)", "Start heading (rad)"),
          ("start_x", "start_y", "Start x (m)", "Start y (m)")]


def load_groups() -> dict:
    split = json.loads((TASK_DIR / "experiments/splits/clean253.json").read_text())
    root = TASK_DIR / "data/Markovian/Clean_253"
    with open(root / "scenarios_train.csv", newline="") as f:
        train = {r["scenario_id"]: r for r in csv.DictReader(f)}
    with open(root / "scenarios_test.csv", newline="") as f:
        test = list(csv.DictReader(f))
    rows = {"fitting": [train[i] for i in split["fitting_ids"]["clean253"]],
            "validation": [train[i] for i in split["validation_ids"]], "test": test}
    return {k: {c: np.array([float(r[c]) for r in v]) for c in ("target_dist", "target_z", "rel_azimuth", "start_yaw",
                                                                 "start_x", "start_y")} for k, v in rows.items()}


def plot(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = load_groups()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), facecolor=SURFACE)
    for ax, (x, y, xl, yl) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for name, color, marker, size in GROUPS:
            g = groups[name]
            ax.scatter(g[x], g[y], s=size, c=color, marker=marker, edgecolors=SURFACE, linewidths=0.8,
                       zorder={"fitting": 2, "test": 3, "validation": 4}[name])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    # Validation strata: distance split at its range midpoint, height and azimuth at quarters of their ranges.
    strata = {"target_dist": [1.75], "target_z": [0.1 + q * 1.15 for q in (0.25, 0.5, 0.75)],
              "rel_azimuth": [-np.pi / 2, 0.0, np.pi / 2]}
    for ax, (x, y, *_) in zip(axes.flat[:2], PANELS[:2]):
        for v in strata.get(x, []):
            ax.axvline(v, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        for v in strata.get(y, []):
            ax.axhline(v, color=INK_2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    handles = [plt.Line2D([], [], linestyle="", marker=m, markersize=8 if m == "o" else 9, markerfacecolor=c,
                          markeredgecolor=SURFACE, label=f"{name} ({len(groups[name]['target_z'])})")
               for name, c, m, _ in GROUPS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.955), ncol=3, frameon=False,
               labelcolor=INK_2, handletextpad=0.3, columnspacing=1.6)
    fig.suptitle("Scenario split: fitting, validation and sealed test scenarios", x=0.06, y=0.995, ha="left",
                 fontsize=13, color=INK)
    fig.text(0.06, 0.012, "Dashed lines: validation strata (2 distance x 4 height x 4 azimuth bins; one validation "
             "scenario per cell). Test scenarios come from a separate Sobol engine.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path,
                   default=TASK_DIR / "results/figures/2026-10-28/scenario-split/2026-10-28__figure__scenario-split.png")
    print(plot(p.parse_args(argv).out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
