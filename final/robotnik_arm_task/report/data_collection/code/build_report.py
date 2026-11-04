"""Rebuild tables and figures from this report's copied evidence; no simulation.

Run from final/robotnik_arm_task/: python report/data_collection/code/build_report.py
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DIMS = ["target_dist", "rel_azimuth", "target_z", "start_x", "start_y", "start_yaw"]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (ROOT / "tables" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unit(rows):
    return np.array([[float(r[f"u_{d}"]) for d in DIMS] for r in rows])


def save(fig, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def main():
    for folder in ("tables", "figures"):
        (ROOT / folder).mkdir(exist_ok=True)
    inventory = json.loads((DATA / "source_inventory.json").read_text())
    for entry in inventory:
        assert hashlib.sha256((ROOT / entry["copy"]).read_bytes()).hexdigest() == entry["sha256"]
    manifest = json.loads((DATA / "scenarios/scenario_manifest.json").read_text())
    split = json.loads((DATA / "clean253_split.json").read_text())
    scenes = {s: read_csv(DATA / f"scenarios/scenarios_{s}.csv") for s in ("train", "test")}
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((DATA / "scenarios" / name).read_bytes()).hexdigest() == digest
    fit_ids, val_ids = set(split["fitting_ids"]["clean253"]), set(split["validation_ids"])
    assert not fit_ids & val_ids
    assert (len(fit_ids), len(val_ids)) == (221, 32)
    groups = {"train256": scenes["train"],
              "clean253": [r for r in scenes["train"] if r["scenario_id"] in fit_ids | val_ids],
              "fitting221": [r for r in scenes["train"] if r["scenario_id"] in fit_ids],
              "validation32": [r for r in scenes["train"] if r["scenario_id"] in val_ids],
              "test64": scenes["test"]}
    assert {r["scenario_id"] for r in groups["clean253"]} == fit_ids | val_ids
    train, test = unit(scenes["train"]), unit(scenes["test"])
    assert (len(train), len(test)) == (256, 64)
    for s, u in (("train", train), ("test", test)):
        assert len(np.unique(u, axis=0)) == len(u)
        assert np.all((u >= 0) & (u < 1))
        assert len({r["scenario_id"] for r in scenes[s]}) == len(u)
    assert not {r["scenario_id"] for r in scenes["train"]} & {r["scenario_id"] for r in scenes["test"]}
    assert not {tuple(r) for r in train} & {tuple(r) for r in test}
    assert not {r["seed"] for r in scenes["train"]} & {r["seed"] for r in scenes["test"]}
    marginal_rows, cell_rows, coverage_rows = [], [], []
    validation_cells = None
    for name, rows in groups.items():
        u = unit(rows)
        cells = np.minimum((u[:, [0, 2, 1]] * [2, 4, 4]).astype(int), [1, 3, 3])
        counts = [int(np.sum(np.all(cells == [d, z, a], axis=1)))
                  for d in range(2) for z in range(4) for a in range(4)]
        if name == "validation32":
            validation_cells = counts
        coverage_rows.append({"group": name, "scenarios": len(rows),
                              "CD_6d": qmc.discrepancy(u, method="CD"),
                              "empty_cells_2dist_4height_4azimuth": counts.count(0),
                              "min_cell_count": min(counts), "max_cell_count": max(counts)})
        for j, d in enumerate(DIMS):
            for b, count in enumerate(np.histogram(u[:, j], bins=np.linspace(0, 1, 9))[0]):
                marginal_rows.append({"group": name, "dimension": d, "bin_of_8": b, "count": int(count)})
        for (d, z, a), count in zip(np.ndindex(2, 4, 4), counts):
            cell_rows.append({"group": name, "distance_bin": d, "height_bin": z,
                              "azimuth_bin": a, "count": count})
    assert validation_cells == [1] * 32
    write_csv("coverage.csv", coverage_rows)
    write_csv("marginals.csv", marginal_rows)
    write_csv("split_cells.csv", cell_rows)
    outcomes = []
    for teacher in ("markovian", "stateful"):
        episodes = read_csv(DATA / f"clean/{teacher}_episodes.csv")
        for s in ("train", "test"):
            rows = [r for r in episodes if r["split"] == s]
            outcomes.append({"teacher": teacher, "split": s, "episodes": len(rows),
                             "sustained_holds": sum(r["sustained_hold_pass"] == "True" for r in rows),
                             "actions_1khz": sum(int(r["T_actions"]) for r in rows),
                             "execution_clip_steps": sum(int(r["exec_clip_steps"]) for r in rows)})
    write_csv("collection.csv", outcomes)

    # Repeat the historical audit's 200 IID reference sets, using RNG seed 0 for each split.
    comparison = []
    audit = json.loads((DATA / "scenarios/coverage/coverage.json").read_text())
    for s, u in (("train", train), ("test", test)):
        rng = np.random.default_rng(0)
        cd = np.array([qmc.discrepancy(rng.random(u.shape), method="CD") for _ in range(200)])
        sobol_cd = qmc.discrepancy(u, method="CD")
        assert np.isclose(sobol_cd, audit[s]["discrepancy"]["sobol_CD"], rtol=1e-8)
        assert np.isclose(np.median(cd), audit[s]["discrepancy"]["iid_uniform_CD_median"], rtol=1e-8)
        comparison.append({"split": s, "n": len(u), "sobol_CD": sobol_cd,
                           "iid_CD_median": np.median(cd), "iid_CD_p05": np.percentile(cd, 5),
                           "iid_CD_p95": np.percentile(cd, 95), "iid_over_sobol": np.median(cd) / sobol_cd,
                           "iid_sets": 200, "iid_seed": 0})
    write_csv("sobol_vs_iid.csv", comparison)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, color, label in (("fitting221", "#2a78d6", "Fitting (221)"),
                               ("validation32", "#df6a32", "Validation (32)"),
                               ("test64", "#258675", "Test (64)")):
        rows = groups[name]
        for ax, x, y in zip(axes, ["target_dist", "rel_azimuth", "start_x"],
                            ["target_z", "start_yaw", "start_y"]):
            ax.scatter([float(r[x]) for r in rows], [float(r[y]) for r in rows],
                       s=18, alpha=.75, color=color, label=label)
    for ax, xlabel, ylabel in zip(axes, ["Target distance (m)", "Relative target azimuth (rad)", "Start x (m)"],
                                 ["Target height (m)", "Start yaw (rad)", "Start y (m)"]):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen scene splits: projections of six sampled parameters")
    save(fig, "scenario_coverage")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    iid = np.random.default_rng(0).random(train.shape)
    for ax, points, title in zip(axes[:2], (train, iid), ("Scrambled Sobol (256)", "IID uniform (256; seed 0 example)")):
        ax.scatter(points[:, 0], points[:, 2], s=14, color="#2a78d6")
        ax.set(xlabel="Normalised target distance", ylabel="Normalised target height",
               xlim=(0, 1), ylim=(0, 1), title=title)
        ax.set_xticks(np.linspace(0, 1, 5)); ax.set_yticks(np.linspace(0, 1, 5))
        ax.grid(alpha=.2)
    for i, row in enumerate(comparison):
        med = row["iid_CD_median"]
        axes[2].errorbar(i + .09, med, yerr=[[med - row["iid_CD_p05"]], [row["iid_CD_p95"] - med]],
                         fmt="o", capsize=5, color="#df6a32", label="IID median; 5–95%" if i == 0 else None)
        axes[2].scatter(i - .09, row["sobol_CD"], marker="D", color="#2a78d6", label="Frozen Sobol" if i == 0 else None)
    axes[2].set(yscale="log", xticks=[0, 1], xticklabels=["Train (256)", "Test (64)"],
                ylabel="6-D centred discrepancy (lower is better)", title="200 IID sets per split")
    axes[2].legend(frameon=False, fontsize=8)
    save(fig, "sobol_vs_iid")
    print("Verified 11 evidence hashes, split separation and 32 validation cells;")
    print("reproduced historical discrepancy audit; wrote 5 tables and 2 PNG/PDF figure pairs.")


if __name__ == "__main__":
    main()
