"""Figures for docs/2026-10-30__guide__project-pipeline.md (teacher A only).

collection.png      253 clean teacher-A episodes: base paths, end-effector error over time, time to first reach
dart.png            one scenario recorded clean and with two DART noise draws (executed vs labelled torque)
split.png           the frozen fitting / validation / test scenario split (evaluation.plot_scenario_split)
state_coverage.png  training rows of the clean and DART datasets: which states each covers
features.png        the base-yaw frame, the relative-position inputs and the reach gate

Reads data/Markovian/{Clean_253,DART}/ and data/datasets/A/; writes nothing else.
Usage (from final/robotnik_arm_task/): python -m evaluation.plot_pipeline_figures [--out <dir>]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from evaluation import plot_scenario_split
from evaluation.plot_scenario_split import GRID, INK, INK_2, MUTED, SURFACE

TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = TASK_DIR.parents[1] / "docs/figures/pipeline"
CLEAN = TASK_DIR / "data/Markovian/Clean_253"
DART = next((TASK_DIR / "data/Markovian/DART").glob("*__collection__d2-s075-x2-*"), None)
DATASETS = TASK_DIR / "data/datasets/A"
# Categorical slots 1-3 of the validated palette; every series also has its own marker or line style and a label.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
EXAMPLE = "train-0102"           # a DART scenario whose first draw has the largest noise scale (alpha ~ 1)
TOL_MM, MIN_REACH_S = 10.0, 8.0
STANDOFF_Z, STANDOFF = [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]


def style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 11, "axes.titlelocation": "left",
                         "axes.titlecolor": INK})


def frame(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def finish(fig, out: Path, title: str, note: str | None = None) -> Path:
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=13, color=INK)
    if note:
        fig.text(0.02, 0.01, note, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    return out


def load(path: Path) -> dict:
    z = np.load(path)
    meta = json.loads(str(z["metadata_json"]))
    err_mm = 1000 * np.linalg.norm(z["teacher_ee_pos"] - np.asarray(meta["target_ee_pos_W"]), axis=1)
    return {"t": z["obs_time"][:-1], "base": z["obs_base_pos"][:, :2], "err_mm": err_mm, "target": meta["target_ee_pos_W"],
            "a_expert": z["a_expert"], "a_executed": z["a_executed"], "reach_active": z["teacher_reach_active"],
            "base_pos": z["obs_base_pos"][:-1], "yaw": z["obs_base_yaw"][:-1]}


def rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collection(plt, out: Path) -> Path:
    episodes = [r for r in rows(CLEAN / "episodes.csv") if r["split"] == "train"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.05, 1.25, 0.9]})
    for ax in axes:
        frame(ax)
    a, b, c = axes
    for r in episodes:
        e = load(CLEAN / r["file"])
        a.plot(e["base"][::200, 0], e["base"][::200, 1], color=BLUE, linewidth=0.6, alpha=0.35)
        a.plot(*e["base"][0], marker="o", markersize=1.6, color=INK_2, alpha=0.6, linestyle="")
        a.plot(e["target"][0], e["target"][1], marker="^", markersize=4, color=ORANGE, markeredgecolor=SURFACE,
               markeredgewidth=0.4, linestyle="")
        b.plot(e["t"][::50], e["err_mm"][::50], color=BLUE, linewidth=0.5, alpha=0.25)
    a.set_aspect("equal")
    a.set_xlabel("World x (m)")
    a.set_ylabel("World y (m)")
    a.set_title("(a) Base paths: start (dot) to target (triangle)")
    b.set_yscale("log")
    b.set_ylim(0.01, 6000)
    b.axhline(TOL_MM, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    b.axvline(MIN_REACH_S, color=INK_2, linewidth=1, linestyle=(0, (1, 2)))
    b.text(MIN_REACH_S + 0.2, 2000, "success window opens (8 s)", color=INK_2, fontsize=9)
    b.text(0.3, TOL_MM * 1.25, "10 mm tolerance", color=INK_2, fontsize=9)
    b.set_xlabel("Time (s)")
    b.set_ylabel("End-effector to target distance (mm)")
    b.set_title("(b) Navigate, reach, then hold")
    t_reach = np.array([float(r["t_first_reach"]) for r in episodes])
    c.hist(t_reach, bins=np.arange(8, t_reach.max() + 0.5, 0.5), color=BLUE, edgecolor=SURFACE, linewidth=1)
    c.set_xlabel("Time of first reach (s)")
    c.set_ylabel("Episodes")
    c.set_title("(c) Time to first reach")
    return finish(fig, out, f"Clean teacher-A demonstrations: {len(episodes)} training scenarios at 1 kHz",
                  "Each episode ends 3 s after the first reach (40 s cap). A reach before 8 s counts from 8 s, as in "
                  "the course task.")


def dart(plt, out: Path) -> Path:
    clean = load(CLEAN / f"train/markovian-clean-{EXAMPLE}.npz")
    draws = [load(DART / f"train/markovian-dart-{EXAMPLE}-d{d}.npz") for d in (0, 1)]
    alphas = {r["draw"]: float(r["alpha"]) for r in rows(DART / "episodes.csv") if r["scenario_id"] == EXAMPLE}
    series = [("clean", clean, BLUE, "-"), (f"DART draw 0 (alpha {alphas['0']:.2f})", draws[0], ORANGE, "-"),
              (f"DART draw 1 (alpha {alphas['1']:.2f})", draws[1], AQUA, (0, (5, 2)))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax in axes:
        frame(ax)
    a, b, c = axes
    for label, e, color, ls in series:
        a.plot(e["base"][::20, 0], e["base"][::20, 1], color=color, linestyle=ls, linewidth=2, label=label)
        b.plot(e["t"][::10], np.maximum(e["err_mm"][::10], 0.1), color=color, linestyle=ls, linewidth=1.4, label=label)
    a.plot(*clean["base"][0], marker="o", color=INK_2, linestyle="")
    a.plot(clean["target"][0], clean["target"][1], marker="^", markersize=9, color=ORANGE, markeredgecolor=SURFACE,
           linestyle="")
    a.annotate("target", clean["target"][:2], textcoords="offset points", xytext=(6, -12), color=INK_2, fontsize=9)
    a.annotate("start", clean["base"][0], textcoords="offset points", xytext=(6, 4), color=INK_2, fontsize=9)
    a.set_aspect("equal")
    a.set_xlabel("World x (m)")
    a.set_ylabel("World y (m)")
    a.set_title("(a) Base path")
    a.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    b.set_yscale("log")
    b.axhline(TOL_MM, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    b.set_xlabel("Time (s)")
    b.set_ylabel("End-effector to target distance (mm)")
    b.set_title("(b) The noisy runs still reach and hold")
    e, j = draws[0], 1
    window = (e["t"] >= 8) & (e["t"] <= 12)
    c.plot(e["t"][window], e["a_executed"][window, j], color=ORANGE, linewidth=0.8, label="executed (expert + OU noise)")
    c.plot(e["t"][window], e["a_expert"][window, j], color=INK, linewidth=1.6, label="training label (expert action)")
    c.set_xlabel("Time (s)")
    c.set_ylabel(f"Joint {j + 1} torque (N·m)")
    c.set_title("(c) Noise is executed, never learned")
    c.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    return finish(fig, out, f"DART-style noise injection on scenario {EXAMPLE}",
                  "OU noise (tau 0.2 s): arm sigma 0.75 N·m, wheels 0.15 rad/s, both scaled by alpha ~ U[0.25, 1] per "
                  "episode; wheel noise stops when the reach gate first opens.")


def dataset_rows(name: str, fitting: set, n: int, rng) -> np.ndarray:
    root = DATASETS / name
    x = np.load(root / "x.npy", mmap_mode="r")
    keep = np.concatenate([np.arange(int(r["row_start"]), int(r["row_end"])) for r in rows(root / "episodes.csv")
                           if r["scenario_id"] in fitting])
    return np.asarray(x[np.sort(rng.choice(keep, size=min(n, len(keep)), replace=False))], dtype=np.float64)


def state_coverage(plt, out: Path) -> Path:
    split = json.loads((TASK_DIR / "experiments/splits/clean253.json").read_text())
    fitting = set(split["fitting_ids"]["clean253"])
    rng = np.random.default_rng(0)
    sets = [("clean (221 episodes)", dataset_rows("clean253__features-v1-gate", fitting, 400_000, rng), BLUE, "-"),
            ("DART (441 episodes)", dataset_rows("dart-d2s075x2__features-v1-gate", fitting, 400_000, rng), ORANGE,
             (0, (5, 2)))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)
    for ax in axes:
        frame(ax)
    a, b, c = axes
    err_bins, qd_bins = np.logspace(-1, 3.7, 60), np.logspace(-4, 1, 60)
    for label, x, color, ls in sets:
        err = 1000 * np.linalg.norm(x[:, 17:20], axis=1)
        qd = np.linalg.norm(x[:, 7:14], axis=1)
        near = (x[:, 21] > 0.5) & (err < 20)
        a.hist(err, bins=err_bins, density=True, histtype="step", color=color, linestyle=ls, linewidth=1.8, label=label)
        b.hist(qd, bins=qd_bins, density=True, histtype="step", color=color, linestyle=ls, linewidth=1.8, label=label)
        c.hist(qd[near], bins=qd_bins, density=True, histtype="step", color=color, linestyle=ls, linewidth=1.8,
               label=label)
    for ax, xl, title in ((a, "End-effector to target distance (mm)", "(a) All rows: distance to target"),
                          (b, "Joint speed |qd| (rad/s)", "(b) All rows: joint speed"),
                          (c, "Joint speed |qd| (rad/s)", "(c) Holding rows (gate open, < 20 mm)")):
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Density")
        ax.set_title(title)
    a.axvline(TOL_MM, color=INK_2, linewidth=1, linestyle=(0, (1, 2)))
    a.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    return finish(fig, out, "Training-row coverage: DART adds perturbed states around the expert's path",
                  "400k rows sampled from the fitting scenarios of each dataset. Clean holding rows sit almost still; "
                  "DART rows show the arm moving near the target, where the expert's corrective torque is the label.")


def features(plt, out: Path) -> Path:
    from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Wedge
    fig = plt.figure(figsize=(15, 5.6), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35], hspace=0.35)
    a = fig.add_subplot(grid[:, 0])
    frame(a)
    a.grid(False)
    z = 0.457
    standoff = float(np.interp(z, STANDOFF_Z, STANDOFF))
    h_tol, d_tol = 0.075, 0.12
    # Gate region in B_yaw: targets at standoff +- 0.12 m and within +- 0.075 rad of the base heading.
    a.add_patch(Wedge((0, 0), standoff + d_tol, -np.degrees(h_tol), np.degrees(h_tol), width=2 * d_tol,
                      facecolor=ORANGE, alpha=0.25, edgecolor=ORANGE, linewidth=1.2))
    a.add_patch(Rectangle((-0.33, -0.25), 0.66, 0.5, facecolor=GRID, edgecolor=INK_2, linewidth=1.2))
    a.add_patch(Polygon([[0.18, 0.08], [0.32, 0], [0.18, -0.08]], closed=True, facecolor=INK_2))
    target, ee = np.array([1.45, 0.75]), np.array([0.55, -0.2])
    for start, end, color, text, off in (((0, 0), target, BLUE, "target_b = target - base\n(rotated by -yaw)", (-120, 14)),
                                         (ee, target, AQUA, "ee_err_b = target - FK(q)", (14, 4))):
        a.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color, linewidth=2))
        mid = (np.asarray(start) + end) / 2
        a.annotate(text, mid, textcoords="offset points", xytext=off, color=INK_2, fontsize=9)
    a.plot(*target, marker="^", markersize=11, color=ORANGE, markeredgecolor=SURFACE, linestyle="")
    a.plot(*ee, marker="o", markersize=7, color=AQUA, markeredgecolor=SURFACE, linestyle="")
    a.annotate("end effector (FK)", ee, textcoords="offset points", xytext=(-20, -16), color=INK_2, fontsize=9)
    a.annotate("target", target, textcoords="offset points", xytext=(8, 4), color=INK_2, fontsize=9)
    a.annotate(f"reach_gate = 1 when the target lies here:\n|d| < {d_tol} m, |h| < {h_tol} rad\n(standoff {standoff:.2f} m at z = {z} m)",
               (standoff, -0.06), textcoords="offset points", xytext=(30, -70), color=INK_2, fontsize=9,
               arrowprops={"arrowstyle": "-", "color": MUTED})
    a.annotate("", (1.9, 0), (0, 0), arrowprops={"arrowstyle": "->", "color": MUTED, "linestyle": "--"})
    a.text(1.6, 0.03, "x_b (heading)", color=MUTED, fontsize=9, va="bottom")
    a.set_xlim(-0.5, 2.0)
    a.set_ylim(-0.75, 1.1)
    a.set_aspect("equal")
    a.set_xlabel("x in base-yaw frame (m)")
    a.set_ylabel("y in base-yaw frame (m)")
    a.set_title("(a) Inputs are expressed relative to the base")

    e = load(CLEAN / f"train/markovian-clean-{EXAMPLE}.npz")
    rel = np.asarray(e["target"])[:2] - e["base_pos"][:, :2]
    c, s = np.cos(e["yaw"]), np.sin(e["yaw"])
    bx, by = c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]
    d = np.hypot(bx, by) - standoff
    h = np.arctan2(by, bx)
    gate = (np.abs(d) < d_tol) & (np.abs(h) < h_tol)
    t = e["t"]
    panels = [(np.abs(d), d_tol, "|d| (m)", "(b) Distance to the standoff point"),
              (np.abs(h), h_tol, "|h| (rad)", "(c) Heading error to the target"),
              (None, None, "gate", "(d) reach_gate input vs the expert's internal reach flag")]
    for row, (y, tol, yl, title) in enumerate(panels):
        ax = fig.add_subplot(grid[row, 1])
        frame(ax)
        if y is not None:
            ax.plot(t, y, color=BLUE, linewidth=1.6)
            ax.axhline(tol, color=ORANGE, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(t[-1], tol, f"threshold {tol}", color=INK_2, fontsize=9, ha="right", va="bottom")
            ax.set_yscale("log")
        else:
            ax.step(t, gate.astype(float), where="post", color=BLUE, linewidth=2, label="reach_gate (from inputs)")
            ax.step(t, e["reach_active"].astype(float) * 0.92, where="post", color=ORANGE, linewidth=1.4,
                    linestyle=(0, (4, 2)), label="expert reach_active (scaled 0.92 to show overlap)")
            ax.set_yticks([0, 1])
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, 0.95))
            ax.set_xlabel("Time (s)")
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=10)
    agree = np.mean(gate == e["reach_active"])
    return finish(fig, out, "Feature engineering: relative coordinates and the reach gate",
                  f"Scenario {EXAMPLE} (clean). The gate recomputed from the inputs matches the expert's own reach "
                  f"flag on {100 * agree:.2f}% of the {len(t):,} steps.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    out = p.parse_args(argv).out
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    for name, fn in (("collection", collection), ("dart", dart), ("state_coverage", state_coverage),
                     ("features", features)):
        print(fn(plt, out / f"{name}.png"))
        plt.close("all")
    print(plot_scenario_split.plot(out / "split.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
