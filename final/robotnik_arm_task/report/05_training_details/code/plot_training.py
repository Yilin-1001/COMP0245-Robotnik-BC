"""Training details: report figures drawn from the copied run logs in ../data (no training runs here).

  figures/training_curves.{pdf,png}   OU + Gate (R4-dart-gate, f-DAgger + Gate round 0) and f-DAgger + Gate round 3 (R8-dart-gate-dagger-r3), 3 seeds each:
                                      training losses and learning rate (top), offline validation metrics on the
                                      expert's states (bottom). Line = seed mean, band = seed min-max.
  figures/jacobian_effect.{pdf,png}   archived matched pair on clean data (M6 256-128-64): BC vs BC+J, 3 seeds each.
Both are IEEE double-column width (7.16 in). Training losses are means over each 500-update logging window.

Usage (from final/robotnik_arm_task/; needs matplotlib):
    python report/05_training_details/code/plot_training.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parents[1]  # report/05_training_details
DATA, FIG = ROOT / "data", ROOT / "figures"
# Reference palette: categorical slots 1-2 (validated all-pairs, CVD-safe) plus its neutral grey; line style is the
# secondary encoding, so identity never rests on colour alone.
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8986"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def read_metrics(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {k: np.array([float(r[k]) if r[k] not in ("", "False", "True") else np.nan for r in rows])
           for k in rows[0] if k not in ("extra", "val_hold_bias_gate_pass")}
    out["jacobian_loss"] = np.array([json.loads(r["extra"] or "{}").get("jacobian_loss", np.nan) for r in rows], float)
    return out


def runs(pattern: str) -> list[dict]:
    found = [read_metrics(p / "metrics.csv") for p in sorted(DATA.glob(pattern))]
    assert len(found) == 3, (pattern, len(found))
    return found


def band(ax, group: list[dict], key: str, colour: str, style: str, label: str, scale: float = 1.0):
    u = group[0]["update"] / 1000
    v = np.vstack([g[key] for g in group]) * scale
    ax.fill_between(u, np.nanmin(v, 0), np.nanmax(v, 0), color=colour, alpha=0.18, linewidth=0)
    ax.plot(u, np.nanmean(v, 0), color=colour, linestyle=style, linewidth=1.2, label=label)


def sci_label(v: float, _pos) -> str:
    """2e-3 -> 2×10⁻³, 1e-2 -> 10⁻²."""
    e = int(np.floor(np.log10(v) + 1e-9))
    m = round(v / 10 ** e)
    sup = str(e).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"10{sup}" if m == 1 else f"{m}×10{sup}"


def panel(ax, title: str, unit: str, log: bool = False, ref: float | None = None, ref_label: str = ""):
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylabel(unit)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(sci_label))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    if ref is not None:
        ax.axhline(ref, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
        ax.annotate(ref_label, (1.0, ref), xycoords=("axes fraction", "data"), xytext=(-2, -3),
                    textcoords="offset points", ha="right", va="top", fontsize=6, color=MUTED)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"figures/{name}.pdf, .png")


def training_curves():
    r4 = runs("runs/2026-10-28__training-ablation-v1/R4-dart-gate__*")
    r8 = runs("runs/2026-11-01__training-ablation-v2/R8-dart-gate-dagger-r3__*")
    groups = [(r4, BLUE, "-", "R4 OU + Gate"), (r8, ORANGE, "--", "R8 f-DAgger + Gate (r3)")]
    panels = [("train_loss", "(a) Total training loss", "standardised", True, None, ""),
              ("train_arm_loss", "(b) Arm action loss (Huber)", "standardised", True, None, ""),
              ("train_wheel_loss", "(c) Wheel action loss (Huber)", "standardised", True, None, ""),
              ("jacobian_loss", "(d) Damping-Jacobian loss", "standardised", True, None, ""),
              ("lr", "(e) Learning rate", "× 10⁻³", False, None, ""),
              ("val_arm_rmse_nm", "(f) Validation arm RMSE", "N·m", False, None, ""),
              ("val_hold_arm_rmse_nm", "(g) Val. hold-phase arm RMSE", "N·m", False, None, ""),
              ("val_damping_median_diag", "(h) Learned damping", "N·m·s/rad", False, -30.0, "expert −30")]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.5), sharex=True)
    for ax, (key, title, unit, log, ref, ref_label) in zip(axes.flat, panels):
        for group, colour, style, label in groups:
            band(ax, group, key, colour, style, label, 1e3 if key == "lr" else 1.0)
        panel(ax, title, unit, log, ref, ref_label)
    for ax in axes[1]:
        ax.set_xlabel("update (thousands)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.0, 1.06))
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, "training_curves")


def jacobian_effect():
    bc = runs("jacobian_effect/clean-A-bc-clean253-m6-*")
    bcj = runs("jacobian_effect/clean-A-bcj-clean253-m6-*")
    groups = [(bc, GREY, "--", "BC (action loss only)"), (bcj, BLUE, "-", "BC+J (plus damping-Jacobian loss)")]
    panels = [("val_damping_median_diag", "(a) Learned damping", "N·m·s/rad", -30.0, "expert −30"),
              ("val_arm_rmse_nm", "(b) Validation arm RMSE", "N·m", None, ""),
              ("train_arm_loss", "(c) Arm action loss (training)", "standardised", None, "")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.9), sharex=True)
    for ax, (key, title, unit, ref, ref_label) in zip(axes, panels):
        for group, colour, style, label in groups:
            band(ax, group, key, colour, style, label)
        panel(ax, title, unit, key == "train_arm_loss", ref, ref_label)
        if ref is not None:
            ax.set_ylim(-33.5, None)  # room for the reference label under the line
        ax.set_xlabel("update (thousands)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.0, 1.12))
    fig.tight_layout(w_pad=0.8)
    save(fig, "jacobian_effect")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    training_curves()
    jacobian_effect()
