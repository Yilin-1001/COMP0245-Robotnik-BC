"""Coverage audit of the frozen clean-collection scenario manifests, run before any rollout.

Reports marginal histograms, pairwise scatter plots, coarse joint-cell counts, boundary and
combination coverage, centred L2 discrepancy against i.i.d. uniform sets of the same size, the
template (dataset B) navigation geometry, and distance to the development scenarios used for
expert tuning and noise pilots. Marginal coverage does not prove six-dimensional coverage.

Output: <data>/scenarios/coverage/{coverage_report.md, coverage.json, *.png}

A filtered subset (e.g. data/Markovian/Clean_223) is audited the same way with --train/--test/--out;
--reference adds filtering.png and a removed-scenario table comparing the subset with the full manifest.

Usage: python -m data_collection.scenario_coverage
       python -m data_collection.scenario_coverage --label Clean_223 --train data/Markovian/Clean_223/scenarios_train.csv \
           --test data/Markovian/Clean_223/scenarios_test.csv --reference data/scenarios/scenarios_train.csv --out <dir>
"""
import argparse
import json
import os

import numpy as np

from data_collection.scenarios import DIMENSIONS, SETTINGS, read_manifest, scenario_dir
from data_collection.template_teacher import template_base_goal

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"train": "#2a78d6", "test": "#eb6834"}   # validated pair (dataviz validator, light surface)
UNITS = {"target_dist": "m", "rel_azimuth": "rad", "target_z": "m", "start_x": "m", "start_y": "m", "start_yaw": "rad"}
HIST_BINS = {"train": 8, "test": 4}                # 32 and 16 expected points per bin
SECTORS = [("front", -np.pi / 4, np.pi / 4), ("left", np.pi / 4, 3 * np.pi / 4),
           ("right", -3 * np.pi / 4, -np.pi / 4), ("rear", None, None)]


def sector(az):
    for name, lo, hi in SECTORS[:3]:
        if lo <= az < hi:
            return name
    return "rear"


def arrays(rows):
    return {d: np.array([r[d] for r in rows]) for d in DIMENSIONS + ["target_x", "target_y"]} | {
        "u": np.array([[r[f"u_{d}"] for d in DIMENSIONS] for r in rows]), "ids": np.array([r["scenario_id"] for r in rows])}


def cell_counts(a, n_dist, n_height):
    lo, hi = SETTINGS["clean_collection"]["ranges"]["target_dist"]
    zlo, zhi = SETTINGS["clean_collection"]["ranges"]["target_z"]
    d_edges, z_edges = np.linspace(lo, hi, n_dist + 1), np.linspace(zlo, zhi, n_height + 1)
    di = np.clip(np.digitize(a["target_dist"], d_edges) - 1, 0, n_dist - 1)
    zi = np.clip(np.digitize(a["target_z"], z_edges) - 1, 0, n_height - 1)
    secs = [s[0] for s in SECTORS]
    counts = {}
    for d in range(n_dist):
        for z in range(n_height):
            for s in secs:
                key = f"dist[{d_edges[d]:.2f},{d_edges[d+1]:.2f}) x {s} x z[{z_edges[z]:.2f},{z_edges[z+1]:.2f})"
                counts[key] = int(np.sum((di == d) & (zi == z) & (np.array([sector(x) for x in a["rel_azimuth"]]) == s)))
    return counts


def combos(a):
    near, far = a["target_dist"] < 1.0, a["target_dist"] > 2.5
    low, high = a["target_z"] < 0.4, a["target_z"] > 1.0
    secs = np.array([sector(x) for x in a["rel_azimuth"]])
    out = {}
    for dn, dm in (("near<1.0m", near), ("far>2.5m", far)):
        for hn, hm in (("low<0.4m", low), ("high>1.0m", high)):
            for s in ("front", "left", "right", "rear"):
                out[f"{dn} & {hn} & {s}"] = int(np.sum(dm & hm & (secs == s)))
    return out


def boundary(a):
    rng = SETTINGS["clean_collection"]["ranges"]
    out = {}
    for d in DIMENSIONS:
        lo, hi = rng[d]
        w = hi - lo
        out[d] = {"range": [lo, hi], "min": float(a[d].min()), "max": float(a[d].max()),
                  "gap_to_lower_frac": float((a[d].min() - lo) / w), "gap_to_upper_frac": float((hi - a[d].max()) / w),
                  "within_5pct_of_lower": int(np.sum(a[d] < lo + 0.05 * w)),
                  "within_5pct_of_upper": int(np.sum(a[d] > hi - 0.05 * w))}
    return out


def template_geometry(a):
    """Dataset B: the origin-based goal, the initial turn to the goal and the final heading change it implies."""
    from simulation_and_control import wrap_angle
    out = []
    for i in range(len(a["target_dist"])):
        target = np.array([a["target_x"][i], a["target_y"][i], a["target_z"][i]])
        goal, goal_bearing = template_base_goal(target)
        start = np.array([a["start_x"][i], a["start_y"][i]])
        to_goal = goal - start
        out.append({"start_to_goal_m": float(np.linalg.norm(to_goal)),
                    "initial_turn_to_goal_rad": float(abs(wrap_angle(np.arctan2(to_goal[1], to_goal[0]) - a["start_yaw"][i]))),
                    "final_heading_change_rad": float(abs(wrap_angle(goal_bearing - a["start_yaw"][i]))),
                    "target_xy_from_origin_m": float(np.linalg.norm(target[:2]))})
    return {k: np.array([o[k] for o in out]) for k in out[0]}


def discrepancy(u, n_ref=200, seed=0):
    from scipy.stats import qmc
    rng = np.random.default_rng(seed)
    ref = np.array([qmc.discrepancy(rng.random(u.shape), method="CD") for _ in range(n_ref)])
    return {"sobol_CD": float(qmc.discrepancy(u, method="CD")), "iid_uniform_CD_median": float(np.median(ref)),
            "iid_uniform_CD_p05": float(np.percentile(ref, 5)), "iid_uniform_CD_p95": float(np.percentile(ref, 95)), "n_ref_sets": n_ref}


def dev_scenarios():
    """Development scenarios reproducible from the repo's scripts, as 6-D parameter tuples."""
    from evaluation import expert_noise_robustness, expert_t3_dev
    from evaluation.compare_controllers import sample_target
    out = []

    def add(src, sc):
        s, t = np.asarray(sc.base_start_pos, dtype=float), np.asarray(sc.target_ee_pos, dtype=float)
        dist = np.linalg.norm(t[:2] - s[:2])
        az = np.arctan2(t[1] - s[1], t[0] - s[0]) - sc.base_start_bearing
        out.append((src, [dist, (az + np.pi) % (2 * np.pi) - np.pi, t[2], s[0], s[1], sc.base_start_bearing]))

    for i in range(16):
        add("expert_noise_robustness 870000+", expert_noise_robustness.make_scenario(i))
    for cond in ("nominal", "far targets 3-3.5 m", "unreachable (high)"):
        for i in range(20):
            add("expert_t3_dev 800000+", expert_t3_dev.make_scenario(cond, i))
    from data_collection.scenarios import Scenario
    for i in range(-1, 30):
        add("compare_controllers 900000+", Scenario(seed=i, target_ee_pos=sample_target(i)))
    return out


def dev_distance(a, dev):
    rng = SETTINGS["clean_collection"]["ranges"]
    lo = np.array([rng[d][0] for d in DIMENSIONS])
    w = np.array([rng[d][1] - rng[d][0] for d in DIMENSIONS])
    p = np.array([v for _, v in dev])
    up = (p - lo) / w
    ua = a["u"]
    diff = ua[:, None, :] - up[None, :, :]
    for j in (1, 5):  # angles are periodic
        diff[..., j] = (diff[..., j] + 0.5) % 1.0 - 0.5
    dist = np.sqrt((diff ** 2).sum(-1))
    exact = int(np.sum(np.all(np.isclose(a["u"][:, None, :], up[None, :, :], atol=1e-12), axis=-1)))
    return {"n_dev_scenarios": len(dev), "dev_sources": sorted({s for s, _ in dev}), "identical_tuples": exact,
            "min_unit_distance_to_dev": float(dist.min()), "median_unit_distance_to_nearest_dev": float(np.median(dist.min(1)))}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    return matplotlib, plt


def plot_filtering(ref, kept, out, label):
    """Subset vs the full training manifest: per-bin counts kept/removed and where the removed scenarios lie."""
    _, plt = _style()
    rng = SETTINGS["clean_collection"]["ranges"]
    keep = np.isin(ref["ids"], kept["ids"])
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, d in zip(axes.flat, DIMENSIONS):
        edges = np.linspace(*rng[d], HIST_BINS["train"] + 1)
        full, _ = np.histogram(ref[d], edges)
        sub, _ = np.histogram(ref[d][keep], edges)
        ax.stairs(full, edges, color=MUTED, lw=1.5, ls="--", label=f"full train (n={len(keep)})")
        ax.stairs(sub, edges, color=COLORS["train"], lw=2, fill=True, alpha=0.35)
        ax.stairs(sub, edges, color=COLORS["train"], lw=2, label=f"{label} train (n={int(keep.sum())})")
        for i in np.nonzero(full - sub)[0]:
            ax.text((edges[i] + edges[i + 1]) / 2, full[i] + 0.6, f"-{full[i] - sub[i]}", ha="center", color=MUTED, fontsize=7.5)
        ax.set(xlabel=f"{d} ({UNITS[d]})", ylabel="scenarios per bin", ylim=(0, max(full) * 1.25))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"{label}: training scenarios kept vs the full 256 (labels: removed per bin)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "filtering_marginals.png"), dpi=170)
    plt.close(fig)

    pairs = [("target_dist", "target_z"), ("rel_azimuth", "target_dist"), ("rel_azimuth", "start_yaw"), ("start_x", "start_y")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for ax, (x, y) in zip(axes.flat, pairs):
        ax.scatter(ref[x][keep], ref[y][keep], s=16, color=COLORS["train"], edgecolor="#fcfcfb", linewidth=0.8, label=f"kept ({int(keep.sum())})")
        ax.scatter(ref[x][~keep], ref[y][~keep], s=42, marker="X", color=COLORS["test"], edgecolor=INK, linewidth=0.5,
                   label=f"removed ({int((~keep).sum())})", zorder=3)
        ax.set(xlabel=f"{x} ({UNITS[x]})", ylabel=f"{y} ({UNITS[y]})", xlim=rng[x], ylim=rng[y])
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.suptitle(f"{label}: where the removed training scenarios lie", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "filtering_pairs.png"), dpi=170)
    plt.close(fig)


def plot(sets, out, label=""):
    matplotlib, plt = _style()
    rng = SETTINGS["clean_collection"]["ranges"]
    prefix = f"{label}: " if label else ""

    # Marginals: share of each set per bin (different bin counts for 256 and 64 points)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
    for ax, d in zip(axes.flat, DIMENSIONS):
        for s, a in sets.items():
            edges = np.linspace(*rng[d], HIST_BINS[s] + 1)
            counts, _ = np.histogram(a[d], edges)
            ax.stairs(counts / len(a[d]), edges, color=COLORS[s], lw=2, label=f"{s} (n={len(a[d])}, {HIST_BINS[s]} bins)",
                      fill=(s == "train"), alpha=0.35 if s == "train" else 1.0)
            if s == "test":
                ax.stairs(counts / len(a[d]), edges, color=COLORS[s], lw=2)
        ax.set_xlabel(f"{d} ({UNITS[d]})")
        ax.set_ylabel("share of set per bin")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, loc="lower left")
    fig.suptitle(prefix + "Scenario marginals: each bin's share of its set (uniform = 1/bins)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "marginals.png"), dpi=170)
    plt.close(fig)

    # Pairwise coverage
    pairs = [("target_dist", "target_z"), ("rel_azimuth", "target_dist"), ("rel_azimuth", "start_yaw"), ("start_x", "start_y")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for ax, (x, y) in zip(axes.flat, pairs):
        for s, a in sets.items():
            ax.scatter(a[x], a[y], s=16 if s == "train" else 30, marker="o" if s == "train" else "^", color=COLORS[s],
                       edgecolor="#fcfcfb", linewidth=0.8, label=f"{s} (n={len(a[x])})", zorder=2 if s == "train" else 3)
        ax.set(xlabel=f"{x} ({UNITS[x]})", ylabel=f"{y} ({UNITS[y]})", xlim=rng[x], ylim=rng[y])
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes.flat[0].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.suptitle(prefix + "Pairwise scenario coverage (axes span the full sampling ranges)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pairs.png"), dpi=170)
    plt.close(fig)

    # World-frame view: start positions and targets
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for s, a in sets.items():
        ax.scatter(a["target_x"], a["target_y"], s=14 if s == "train" else 26, marker="o" if s == "train" else "^",
                   color=COLORS[s], edgecolor="#fcfcfb", linewidth=0.8, label=f"{s} targets")
    ax.add_patch(matplotlib.patches.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=MUTED, lw=1, ls="--"))
    ax.text(0.52, 0.52, "start box", color=MUTED)
    ax.set(xlabel="world x (m)", ylabel="world y (m)", aspect="equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.set_title(prefix + "World-frame targets", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "world_targets.png"), dpi=170)
    plt.close(fig)


def table(d: dict, header=("item", "count")) -> str:
    lines = [f"| {header[0]} | {header[1]} |", "| --- | ---: |"]
    lines += [f"| {k} | {v} |" for k, v in d.items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None, help="train manifest (default: the frozen one)")
    parser.add_argument("--test", default=None, help="test manifest (default: the frozen one)")
    parser.add_argument("--reference", default=None, help="full train manifest the subset was filtered from")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    src = scenario_dir()
    out = args.out or os.path.join(src, "coverage")
    os.makedirs(out, exist_ok=True)
    paths = {"train": args.train or os.path.join(src, "scenarios_train.csv"), "test": args.test or os.path.join(src, "scenarios_test.csv")}
    sets = {s: arrays(read_manifest(paths[s])) for s in ("train", "test")}
    dev = dev_scenarios()
    report = {"note": "Marginal and pairwise coverage do not prove six-dimensional coverage. Sobol does not place points exactly on range endpoints.",
              "label": args.label, "inputs": paths}
    if args.reference:
        ref = arrays(read_manifest(args.reference))
        removed = ~np.isin(ref["ids"], sets["train"]["ids"])
        if not np.all(np.isin(sets["train"]["ids"], ref["ids"])):
            raise ValueError("the subset contains scenarios that are not in the reference manifest")
        rng = SETTINGS["clean_collection"]["ranges"]
        report["filtering"] = {
            "reference": args.reference, "reference_n": len(removed), "kept": int((~removed).sum()), "removed": int(removed.sum()),
            "removed_ids": ref["ids"][removed].tolist(),
            "removed_per_bin": {d: (np.histogram(ref[d], np.linspace(*rng[d], HIST_BINS["train"] + 1))[0]
                                    - np.histogram(ref[d][~removed], np.linspace(*rng[d], HIST_BINS["train"] + 1))[0]).tolist() for d in DIMENSIONS},
            "full_cells_now_empty": [k for k, v in cell_counts(sets["train"], 3, 3).items() if v == 0],
        }
        plot_filtering(ref, sets["train"], out, args.label or "subset")
    for s, a in sets.items():
        geo = template_geometry(a)
        rng = SETTINGS["clean_collection"]["ranges"]
        report[s] = {
            "n": len(a["target_dist"]),
            "histograms": {d: np.histogram(a[d], np.linspace(*rng[d], HIST_BINS[s] + 1))[0].tolist() for d in DIMENSIONS},
            "cells": cell_counts(a, 3, 3) if s == "train" else cell_counts(a, 2, 2),
            "combinations": combos(a),
            "boundary": boundary(a),
            "discrepancy": discrepancy(a["u"]),
            "rel_azimuth_sectors": {name: int(np.sum([sector(x) == name for x in a["rel_azimuth"]])) for name, *_ in SECTORS},
            "large_initial_heading_error": {"|rel_azimuth|>2.36 rad (rear sector)": int(np.sum(np.abs(a["rel_azimuth"]) > 3 * np.pi / 4)),
                                            "|rel_azimuth|>2.8 rad": int(np.sum(np.abs(a["rel_azimuth"]) > 2.8))},
            "start_quadrants": {f"x{'>' if sx else '<'}0,y{'>' if sy else '<'}0": int(np.sum(((a['start_x'] > 0) == sx) & ((a['start_y'] > 0) == sy)))
                                for sx in (True, False) for sy in (True, False)},
            "template_B_geometry": {k: {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} for k, v in geo.items()}
                                   | {"initial_turn_to_goal>2.36rad": int(np.sum(geo["initial_turn_to_goal_rad"] > 3 * np.pi / 4)),
                                      "start_to_goal<0.08m (already within pos_tol)": int(np.sum(geo["start_to_goal_m"] < 0.08)),
                                      "target_xy_within_0.8m_of_origin": int(np.sum(geo["target_xy_from_origin_m"] < 0.8))},
            "development_overlap": dev_distance(a, dev),
        }
        cells = np.array(list(report[s]["cells"].values()))
        report[s]["cell_summary"] = {"cells": len(cells), "expected_per_cell": len(a["target_dist"]) / len(cells),
                                     "empty": int(np.sum(cells == 0)), "min": int(cells.min()), "max": int(cells.max()),
                                     "cells_with_<=2": int(np.sum(cells <= 2))}
    with open(os.path.join(out, "coverage.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot(sets, out, args.label)

    title = f"# Scenario coverage audit: {args.label}" if args.label else "# Scenario coverage audit (before rollout)"
    md = [title, "", f"Generated by `python -m data_collection.scenario_coverage` from `{paths['train']}` and `{paths['test']}`. " + report["note"], ""]
    if "filtering" in report:
        f = report["filtering"]
        md += [f"## Filtering against `{f['reference']}`", "",
               f"Kept {f['kept']} of {f['reference_n']} training scenarios; removed {f['removed']}. "
               f"Coarse train cells (3 distance x 4 sectors x 3 heights) left empty by the filter: {f['full_cells_now_empty'] or 'none'}. "
               "See `filtering_marginals.png` and `filtering_pairs.png`.", "",
               "| variable | removed per bin (8 equal bins, low to high) |", "| --- | --- |"]
        md += [f"| {d} | {v} |" for d, v in f["removed_per_bin"].items()]
        md += ["", f"Removed scenario IDs: {', '.join(f['removed_ids'])}", ""]
    for s in ("train", "test"):
        r = report[s]
        md += [f"## {s} (n = {r['n']})", "",
               f"Histogram counts ({HIST_BINS[s]} equal bins per variable, expected {r['n'] // HIST_BINS[s]} per bin):", ""]
        md += ["| variable | counts | min | max |", "| --- | --- | ---: | ---: |"]
        md += [f"| {d} | {r['histograms'][d]} | {r['boundary'][d]['min']:.3f} | {r['boundary'][d]['max']:.3f} |" for d in DIMENSIONS]
        cs = r["cell_summary"]
        md += ["", f"Coarse cells (distance x azimuth sector x height): {cs['cells']} cells, {cs['expected_per_cell']:.1f} expected each; "
               f"empty {cs['empty']}, min {cs['min']}, max {cs['max']}, cells with <= 2 points {cs['cells_with_<=2']}.", "",
               table(r["cells"], ("cell", "scenarios")), "",
               "Near/far x low/high x direction combinations:", "", table(r["combinations"], ("combination", "scenarios")), "",
               f"Azimuth sectors: {r['rel_azimuth_sectors']}. Large initial heading errors: {r['large_initial_heading_error']}. "
               f"Start quadrants: {r['start_quadrants']}.", "",
               f"Discrepancy (centred L2, unit cube): Sobol {r['discrepancy']['sobol_CD']:.5f}; i.i.d. uniform sets of the same size: "
               f"median {r['discrepancy']['iid_uniform_CD_median']:.5f} (5-95%: {r['discrepancy']['iid_uniform_CD_p05']:.5f}-{r['discrepancy']['iid_uniform_CD_p95']:.5f}).", "",
               "Boundary gaps (fraction of each range between the nearest sample and the endpoint):", "",
               "| variable | gap to lower | gap to upper | within 5% of lower | within 5% of upper |", "| --- | ---: | ---: | ---: | ---: |"]
        md += [f"| {d} | {b['gap_to_lower_frac']:.4f} | {b['gap_to_upper_frac']:.4f} | {b['within_5pct_of_lower']} | {b['within_5pct_of_upper']} |"
               for d, b in r["boundary"].items()]
        md += ["", f"Template (dataset B) geometry: `{json.dumps(r['template_B_geometry'])}`", "",
               f"Development-scenario separation: `{json.dumps(r['development_overlap'])}`", ""]
    with open(os.path.join(out, "coverage_report.md"), "w") as f:
        f.write("\n".join(md))
    for s in ("train", "test"):
        print(s, json.dumps(report[s]["cell_summary"]), json.dumps(report[s]["discrepancy"]),
              json.dumps(report[s]["development_overlap"]))
    print(f"Coverage report: {out}")


if __name__ == "__main__":
    main()
