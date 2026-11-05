"""DAgger without development scenes (v3, 2026-11-05) vs with them (v2, 2026-11-01): rebuild ../tables from ../data.

v2 collected, in every DAgger round, the 221 fitting scenarios plus 64 fresh development scenes (285 episodes). Those
development scenes were biased to low and near targets after a diagnosis that had looked at the T0 test scenes, so the
report uses v3, which collects only the 221 fitting scenarios per round (--dev-scenes 0) and is otherwise identical
(same round-0 students, recipe, 3 rounds, round 3 selected, seeds 0-2, same unified test scenes).

Tables:
  collections.csv       per version, row and round: collector, episodes (fitting / development), student holds
  rates.csv             success % per row: validation (32 scenes), T0..T5, T1-T5 and all 240 test scenes, 3 seeds pooled,
                        95% scene-cluster bootstrap (1,000 resamples, seed 1101031); R1-R4 (v1) shown for reference
  dev_scene_effect.csv  v3 - v2 for each of R5-R8 and metric: paired per-scene difference, 95% paired bootstrap
                        (10,000), two-sided sign-flip permutation p (20,000), seed 20261104, Holm over this table (36 p)
  summary.md            the two tables above as Markdown

These statistics are the same as report/06_ablation/code/make_tables.py. They are conditional on one seed-0 collection
chain per version, so the effect of the development scenes cannot be separated from that chain's randomness.
No simulation or training runs here.

Usage (from final/robotnik_arm_task/):
    python report/08_dev_scene_removal/code/make_tables.py
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DATA, OUT = HERE / "data", HERE / "tables"
V2_TEST = DATA / "test/v2_2026-11-01_rollouts.csv"
V3_TEST = DATA / "test/v3_2026-11-05_rollouts.csv"
VAL = DATA / "validation/per_scene"
BOOT_SEED, BOOT_N = 1101031, 1000
PAIRED_N, PERM_N, STATS_SEED = 10_000, 20_000, 20261104
METRICS = ["validation", "T0", "T1", "T2", "T3", "T4", "T5", "T1-T5", "all 240"]
V1_RUN = {"R1": "R1-baseline", "R2": "R2-gate", "R3": "R3-dart", "R4": "R4-dart-gate"}
DAGGER = {"R5": "R5-baseline-dagger", "R6": "R6-gate-dagger", "R7": "R7-dart-dagger", "R8": "R8-dart-gate-dagger"}
NAMES = {"R1": "Baseline", "R2": "Baseline + Gate", "R3": "OU", "R4": "OU + Gate", "R5": "DAgger", "R6": "DAgger + Gate",
         "R7": "f-DAgger", "R8": "f-DAgger + Gate"}


def label(row: str, version: str) -> str:
    return row if row in V1_RUN else f"{row}-{version}"


def test_records() -> list[dict]:
    out = []
    for r in csv.DictReader(open(V2_TEST)):
        if r["row"] != "expert":
            out.append({"row": label(r["row"], "v2"), "seed": r["seed"], "suite": r["suite"], "scene": r["scenario_id"],
                        "success": r["success"] == "True"})
    for r in csv.DictReader(open(V3_TEST)):
        m = re.match(r"(R[5-8])-.*-nodev-r3__", r["model_id"])
        if m:
            out.append({"row": label(m[1], "v3"), "seed": r["seed"], "suite": r["suite"], "scene": r["scenario_id"],
                        "success": r["success"] == "True"})
    return out


def val_records() -> list[dict]:
    files = {row: [VAL / f"{run}__MLP01-256x3__s{s}__v1__2026-10-28.csv" for s in range(3)] for row, run in V1_RUN.items()}
    for row, run in DAGGER.items():
        files[label(row, "v2")] = [next(VAL.glob(f"{run}-r3__MLP01-256x3__s{s}__v2__*.csv")) for s in range(3)]
        files[label(row, "v3")] = [VAL / f"{run}-nodev-r3__MLP01-256x3__s{s}__v3__2026-11-05.csv" for s in range(3)]
    return [{"row": row, "seed": str(s), "scene": r["scenario_id"], "success": r["sustained_hold_pass"] == "True"}
            for row, paths in files.items() for s, path in enumerate(paths) for r in csv.DictReader(open(path))]


def select(test, val, metric):
    if metric == "validation":
        return val
    if metric == "all 240":
        return test
    if metric == "T1-T5":
        return [r for r in test if r["suite"] != "T0"]
    return [r for r in test if r["suite"] == metric]


def per_scene(records):
    d = defaultdict(lambda: defaultdict(list))
    for r in records:
        d[r["row"]][r["scene"]].append(float(r["success"]))
    return ({k: {s: float(np.mean(v)) for s, v in x.items()} for k, x in d.items()},
            {k: sum(len(v) for v in x.values()) for k, x in d.items()})


def boot_ci(scene_means):
    v = np.array(list(scene_means.values()))
    rng = np.random.default_rng(BOOT_SEED)
    lo, hi = np.quantile(v[rng.integers(0, len(v), (BOOT_N, len(v)))].mean(axis=1), [0.025, 0.975])
    return 100 * v.mean(), 100 * lo, 100 * hi


def paired(a, b, rng):
    keys = sorted(a)
    assert keys == sorted(b), "scene sets differ"
    d = np.array([b[k] - a[k] for k in keys])
    lo, hi = np.quantile(d[rng.integers(0, len(d), (PAIRED_N, len(d)))].mean(axis=1), [0.025, 0.975])
    flips = rng.choice([-1.0, 1.0], size=(PERM_N, len(d)))
    p = (1 + np.sum(np.abs((flips * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)) / (1 + PERM_N)
    return 100 * d.mean(), 100 * lo, 100 * hi, p


def holm(ps):
    order, adj, running = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def write(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def collections() -> list[dict]:
    out = []
    for path in sorted((DATA / "collections").glob("v*_r*_manifest.json")):
        m = json.loads(path.read_text())
        student = m["student"]["run_id"]
        n = int(student[1])  # round 1 is collected by the base row R1-R4, later rounds by the DAgger row itself
        row = f"R{n + 4}" if n <= 4 else f"R{n}"
        c = m["counts"]
        out.append({"version": path.name[:2], "row": row, "name": NAMES[row], "round": int(m["round"][1:]),
                    "collector": student, "episodes": c["train_episodes"], "fitting": c["fitting_episodes"],
                    "development": c["development_episodes"], "fitting_student_hold": c["fitting_student_hold"],
                    "development_student_hold": c["development_student_hold"],
                    "collection": m["dataset_id"].split("/")[-1], "written": m["written"]})
    return sorted(out, key=lambda r: (r["version"], r["row"], r["round"]))


def main():
    OUT.mkdir(exist_ok=True)
    coll = collections()
    assert len(coll) == 24 and {(r["version"], r["episodes"]) for r in coll} == {("v2", 285), ("v3", 221)}
    write("collections.csv", coll)

    test, val = test_records(), val_records()
    tables = {m: per_scene(select(test, val, m)) for m in METRICS}
    rows = list(V1_RUN) + [label(r, v) for r in DAGGER for v in ("v2", "v3")]
    rates = []
    for row in rows:
        entry = {"row": row, "name": NAMES[row[:2]]}
        for m in METRICS:
            scene, n = tables[m]
            rate, lo, hi = boot_ci(scene[row])
            entry[m] = f"{rate:.1f} [{lo:.1f}, {hi:.1f}]"
            entry[f"{m} n"] = n[row]
        rates.append(entry)
    write("rates.csv", rates)

    rng = np.random.default_rng(STATS_SEED)
    effect = []
    for row in DAGGER:
        a, b = label(row, "v2"), label(row, "v3")
        for m in METRICS:
            scene, _ = tables[m]
            diff, lo, hi, p = paired(scene[a], scene[b], rng)
            effect.append({"row": row, "name": NAMES[row], "metric": m,
                           "v2_pct": round(100 * np.mean(list(scene[a].values())), 1),
                           "v3_pct": round(100 * np.mean(list(scene[b].values())), 1), "diff_pp": round(diff, 1),
                           "ci_low": round(lo, 1), "ci_high": round(hi, 1), "p": float(f"{p:.2g}")})
    for o, padj in zip(effect, holm([o["p"] for o in effect])):
        o["p_holm"] = float(f"{padj:.2g}")
        o["sig_holm_0.05"] = padj < 0.05
    write("dev_scene_effect.csv", effect)

    md = ["# DAgger without development scenes (v3) vs with them (v2)", "",
          "Generated by `code/make_tables.py` from `data/`. Success % pooled over 3 seeds [95% scene-cluster bootstrap]. "
          "T0 was viewed during development; T1-T5 were not.", "",
          "## Collection per DAgger round", "", "| version | episodes per round | fitting | development |", "|---|---|---|---|"]
    for v in ("v2", "v3"):
        r = next(c for c in coll if c["version"] == v)
        md.append(f"| {v} | {r['episodes']} | {r['fitting']} | {r['development']} |")
    md += ["", "## Success rates", "", "| row | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    md += ["| " + r["row"] + " | " + " | ".join(r[m] for m in METRICS) + " |" for r in rates]
    md += ["", "## v3 - v2 (Holm over the 36 p values; * = significant at 0.05)", "",
           "| row | metric | v2 % | v3 % | v3-v2 pp [95% CI] | Holm p |", "|---|---|---|---|---|---|"]
    for o in effect:
        star = " *" if o["sig_holm_0.05"] else ""
        md.append(f"| {o['row']} {o['name']} | {o['metric']} | {o['v2_pct']} | {o['v3_pct']} | "
                  f"{o['diff_pp']} [{o['ci_low']}, {o['ci_high']}] | {o['p_holm']}{star} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print(OUT / "summary.md")


if __name__ == "__main__":
    main()
