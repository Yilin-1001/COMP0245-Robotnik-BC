"""Report figures and summaries derived only from the recorded result index. Output: results/figures/ (local).

validation_hold_by_architecture.png  32-scenario sustained-hold success per architecture, teacher and objective
validation_summary.csv               the plotted numbers (mean over recorded runs, run count, failures)
test_summary.csv / test_hold.png      only when the frozen final protocol has recorded test rows

Usage: python -m evaluation.plots [--index results/results.csv] [--out results/figures]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
SERIES = {"A": "#2a78d6", "B": "#eb6834"}  # categorical slots 1-2 (validated light-mode palette)
INK, MUTED, GRID, SURFACE = "#1d1d1b", "#6b6b66", "#e6e6e2", "#fcfcfb"


def _rows(index: Path) -> list[dict]:
    with open(index, newline="") as f:
        return list(csv.DictReader(f))


def _summarise(rows: list[dict], split: str) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        if r["evaluation_split"] == split and r["objective"] in ("bc", "bcj") and r["sustained_hold_success"] != "":
            groups[(r["teacher"], r["objective"], r["architecture"], r["fit_stage"])].append(r)
    out = []
    for (t, obj, arch, stage), rs in sorted(groups.items()):
        holds = [float(r["sustained_hold_success"]) for r in rs]
        out.append({"teacher": t, "objective": obj, "architecture": arch, "fit_stage": stage, "runs": len(rs),
                    "mean_sustained_hold": round(sum(holds) / len(holds), 6), "min_sustained_hold": min(holds),
                    "max_sustained_hold": max(holds),
                    "failures": sum(int(r["failures"] or 0) for r in rs), "run_ids": " ".join(r["run_id"] for r in rs)})
    return out


def _write(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _bar_figure(summary: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sorted({(s["objective"], s["architecture"]) for s in summary})
    teachers = [t for t in SERIES if any(s["teacher"] == t for s in summary)]
    value = {(s["teacher"], s["objective"], s["architecture"]): s for s in summary}
    width = 0.8 / max(1, len(teachers))
    fig, ax = plt.subplots(figsize=(max(3.5, 0.55 * len(labels) + 1.5), 2.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for k, t in enumerate(teachers):
        xs, ys = [], []
        for i, lab in enumerate(labels):
            s = value.get((t, *lab))
            if s is not None:
                xs.append(i + (k - (len(teachers) - 1) / 2) * width)
                ys.append(100 * s["mean_sustained_hold"])
        ax.bar(xs, ys, width=width * 0.92, color=SERIES[t], label=f"teacher {t}", edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(labels)), [f"{a.upper()}{'+J' if o == 'bcj' else ''}" for o, a in labels],
                  color=INK, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("sustained hold (%)", color=INK, fontsize=7)
    ax.set_title(title, color=INK, fontsize=8, loc="left")
    ax.tick_params(axis="y", colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if len(teachers) > 1:
        ax.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--index", type=Path, default=TASK_DIR / "results" / "results.csv")
    p.add_argument("--out", type=Path, default=TASK_DIR / "results" / "figures")
    args = p.parse_args(argv)
    rows = _rows(args.index)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, stem, title in (("validation", "validation_hold_by_architecture", "32 validation scenarios"),
                               ("test", "test_hold", "64 test scenarios (frozen final protocol)")):
        summary = _summarise(rows, split)
        if summary:
            _write(args.out / f"{split}_summary.csv", summary)
            _bar_figure(summary, args.out / f"{stem}.png", title)
            print(f"wrote {args.out / (stem + '.png')} ({len(summary)} groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
