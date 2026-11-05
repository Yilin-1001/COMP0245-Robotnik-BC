"""Training-method ablation R1-R8: rebuild every table in ../tables from the copied data in ../data.

The eight rows are four data-collection methods, each trained without and with the reach-gate input:

    R1 Baseline         R2 Baseline + Gate      clean expert data only
    R3 OU               R4 OU + Gate            clean + phase-gated OU-noise data (called "DART" in the run ids)
    R5 DAgger           R6 DAgger + Gate        R1 / R2 data + 3 DAgger rounds
    R7 f-DAgger         R8 f-DAgger + Gate      R3 / R4 data + 3 DAgger rounds (OU noise, then DAgger)

R1-R4 are ablation v1 (2026-10-28). R5-R8 are ablation v3 (2026-11-05): every DAgger round collects the 221 fitting
scenarios only (no development scenes; v3 replaces v2, see ../08_dev_scene_removal/). v3 run ids carry "-nodev".
Test records: R1-R4 from the 2026-11-01 unified test, R5-R8 from the 2026-11-05 one (same frozen scenes and code;
code/test_data.py merges them and checks that the expert's rollouts are identical).
The row numbers and run ids (R1-baseline, ..., R8-dart-gate-dagger) are unchanged; only the report names are new.
No simulation or training runs here. Tables:

  naming.csv              row -> report name, run id, factors, inputs, training data
  validation.csv          32 validation scenes x 3 seeds, sustained hold (DAgger rows at round 3), with the 95% scene-cluster
                          bootstrap interval (primary) and the Wilson 95% interval (assumes 96 independent trials; reference)
  validation_by_round.csv R5-R8 per DAgger round (round 0 = the base row's seed-0 model), per seed
  test_by_suite.csv       unified 240-scene test, success % per suite T0-T5 and over all 240 scenes, 3 seeds pooled
  test_by_band.csv        the same per suite and band (T1 distance bands, T3 low/high), with the 95% scene-cluster bootstrap
  comparisons.csv         all 12 one-factor contrasts of the 2x2x2 design (Gate, OU, DAgger), plus DAgger vs OU (R3/R5,
                          R4/R6), on validation and every test suite: paired difference, 95% paired scene-cluster bootstrap CI, sign-flip permutation p, Holm p,
                          and how many of the 9 seed pairings agree in sign
  gate_effect.csv         the Gate contrasts of comparisons.csv, per method
  failure_modes.csv       test failures split into "never within 10 mm" and "reached but did not hold"
  terminal_grades.csv     terminal precision grade (final-second max error) over all 240 scenes
  hold_test.csv           R1-R4 hold test: start at the expert's hold state, does the student stay, linearised stiffness
  cost.csv                training rows, added expert labels, added wall time, sequential train->collect cycles
  efficiency.csv          inputs, parameters, inference latency per control step (from the test rollouts)
  precision.csv           SECONDARY, added after the v3 results were seen: in-range terminal precision (final-second max
                          error, the suite's predefined grade metric) per row on T0 and on T0+T4+T5, and paired ratios of
                          the per-scene geometric-mean error between rows (95% scene bootstrap, 10,000, seed 20261105)

Statistics (see the README section "How uncertainty is reported"):
  - The unit of resampling is the scene. Each row is run on every scene with 3 training seeds; a scene's 3 rollouts are
    kept together (cluster), so the intervals do not treat them as independent.
  - Per-row rates: 95% percentile bootstrap over scenes, 1,000 resamples, seed 1101031 (the test suite's own method).
  - Differences between two rows (same scenes): paired bootstrap of the per-scene difference in success, 10,000 resamples,
    and a two-sided sign-flip permutation test of the mean per-scene difference (20,000 draws; p = (1 + #extreme) / (1 + N)).
    Holm's step-down correction is applied over every p value in comparisons.csv.
  - All intervals are conditional on the 3 trained models of each row; 3 seeds are too few for an interval over training runs,
    so the seed spread is reported descriptively (per-seed counts, sign agreement over the 9 seed pairings).

Usage (from final/robotnik_arm_task/):
    python report/06_ablation/code/make_tables.py
"""
import csv
import datetime
import glob
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from test_data import load_rollouts, load_summary

ROOT = Path(__file__).resolve().parents[1]  # report/06_ablation
DATA, OUT = ROOT / "data", ROOT / "tables"
BOOTSTRAP_SEED, BOOTSTRAP_N = 1101031, 1000  # same as the test suite's own report (data/test/report.json)
PAIRED_N, PERM_N, STATS_SEED = 10_000, 20_000, 20261104  # paired-difference bootstrap and sign-flip permutation test
SUITES = ["T0", "T1", "T2", "T3", "T4", "T5"]
SUITE_NAMES = {"T0": "in range", "T1": "far 3-15 m", "T2": "near", "T3": "height edges", "T4": "translated start",
               "T5": "long hold (20 s)"}

# row -> (run id, report name, method, gate, OU noise, DAgger, base row whose data a DAgger row starts from)
ROWS = {
    "R1": ("R1-baseline", "Baseline", "Baseline", 0, 0, 0, None),
    "R2": ("R2-gate", "Baseline + Gate", "Baseline", 1, 0, 0, None),
    "R3": ("R3-dart", "OU", "OU", 0, 1, 0, None),
    "R4": ("R4-dart-gate", "OU + Gate", "OU", 1, 1, 0, None),
    "R5": ("R5-baseline-dagger", "DAgger", "DAgger", 0, 0, 1, "R1"),
    "R6": ("R6-gate-dagger", "DAgger + Gate", "DAgger", 1, 0, 1, "R2"),
    "R7": ("R7-dart-dagger", "f-DAgger", "f-DAgger", 0, 1, 1, "R3"),
    "R8": ("R8-dart-gate-dagger", "f-DAgger + Gate", "f-DAgger", 1, 1, 1, "R4"),
}
METHODS = ["Baseline", "OU", "DAgger", "f-DAgger"]
DAGGER_TAG, DAGGER_SUFFIX = "-nodev", "__v3__2026-11-05"  # ablation v3: <row>-nodev-r<k>__MLP01-256x3__s<seed>__v3__2026-11-05
RUN_ID = {k: v[0] for k, v in ROWS.items()}
NAME = {k: v[1] for k, v in ROWS.items()}
ROW_OF_RUN = {v[0]: k for k, v in ROWS.items()}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def scene_bootstrap(rollouts):
    """95% interval of the success rate, resampling scenes (the seeds of a scene stay together)."""
    clusters = defaultdict(list)
    for r in rollouts:
        clusters[r["scenario_id"]].append(r["success"])
    values = np.array([np.mean(v) for v in clusters.values()])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = values[rng.integers(0, len(values), (BOOTSTRAP_N, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------------------------------- inputs
def rollouts():
    out = []
    for r in load_rollouts():
        r["success"] = r["success"] == "True"
        out.append(r)
    return out


def dagger_validation():
    """{(row, round, seed): (hold, n)} from the v3 summaries (2026-11-05 run, including the extra seeds of rounds 1-2)."""
    out = {}
    for path in sorted(glob.glob(str(DATA / "validation/v3_*.csv"))):
        for r in read_csv(path):
            k, n = (int(x) for x in r["sustained_hold"].split("/"))
            out[(ROW_OF_RUN[r["row"]], int(r["round"]), int(r["seed"]))] = (k, n, float(r["median_final_err_mm"]))
    return out


def dataset_rows(name, fitting=None):
    eps = read_csv(DATA / f"datasets/{name}_episodes.csv")
    if fitting is not None:
        eps = [e for e in eps if e["scenario_id"] in fitting]
    return len(eps), sum(int(e["rows"]) for e in eps)


# ---------------------------------------------------------------------------------------------------- tables
def naming():
    data = {"R1": "221 clean expert episodes", "R2": "221 clean expert episodes",
            "R3": "221 clean + 441 OU-noise episodes", "R4": "221 clean + 441 OU-noise episodes",
            "R5": "R1 data + DAgger rounds 1-3", "R6": "R2 data + DAgger rounds 1-3",
            "R7": "R3 data + DAgger rounds 1-3", "R8": "R4 data + DAgger rounds 1-3"}
    write_csv("naming.csv", [{
        "row": k, "report_name": NAME[k], "run_id": RUN_ID[k], "method": v[2], "gate": v[3], "ou_noise": v[4],
        "dagger": v[5], "inputs": "features-v1-gate (22)" if v[3] else "features-v1 (21)", "training_data": data[k],
        "model_tested": f"{RUN_ID[k]}{DAGGER_TAG}-r3 (round 3, v3)" if v[5] else RUN_ID[k]} for k, v in ROWS.items()])


def validation_per_scene(dag):
    """[{"row", "seed", "scenario_id", "success"}] for the tested model of each row (round 3 for R5-R8), 32 scenes x 3 seeds.

    R1-R4: the v1 per-seed validation records; R5-R8: the round-3 step-42000 rollouts. Counts are checked against the
    summaries the tables report."""
    v1 = {ROW_OF_RUN[r["row"]]: [int(x) for x in r["student_hold_per_seed"].split("/")]
          for r in read_csv(DATA / "validation/v1_R1-R4_summary.csv")}
    out, scenes = [], None
    for k, v in ROWS.items():
        for seed in range(3):
            run = (f"{RUN_ID[k]}__MLP01-256x3__s{seed}__v1__2026-10-28" if not v[5]
                   else f"{RUN_ID[k]}{DAGGER_TAG}-r3__MLP01-256x3__s{seed}{DAGGER_SUFFIX}")
            recs = read_csv(DATA / f"validation/per_scene/{run}.csv")
            ids = sorted(r["scenario_id"] for r in recs)
            scenes = scenes or ids
            assert ids == scenes and len(ids) == 32, run  # the same 32 validation scenes for every model
            hold = sum(r["sustained_hold_pass"] == "True" for r in recs)
            assert hold == (dag[(k, 3, seed)][0] if v[5] else v1[k][seed]), (run, hold)
            out += [{"row": k, "seed": str(seed), "scenario_id": r["scenario_id"],
                     "success": r["sustained_hold_pass"] == "True"} for r in recs]
    return out


def validation(dag, val_scene):
    v1 = {ROW_OF_RUN[r["row"]]: r for r in read_csv(DATA / "validation/v1_R1-R4_summary.csv")}
    table = []
    for k, v in ROWS.items():
        if v[5]:
            cells = [dag[(k, 3, s)] for s in range(3)]
            per_seed, err = [c[0] for c in cells], statistics.median(c[2] for c in cells)
        else:
            per_seed = [int(x) for x in v1[k]["student_hold_per_seed"].split("/")]
            err = float(v1[k]["student_median_final_err_mm"])
        hold, n = sum(per_seed), 96
        lo, hi = wilson(hold, n)
        blo, bhi = scene_bootstrap([r for r in val_scene if r["row"] == k])
        table.append({"row": k, "report_name": NAME[k], "run_id": RUN_ID[k], "gate": v[3], "ou_noise": v[4], "dagger": v[5],
                      "round": 3 if v[5] else 0, "hold_per_seed": "/".join(map(str, per_seed)),
                      "pooled": f"{hold}/{n}", "rate": round(hold / n, 3),
                      "scene_boot95_low": round(blo, 3), "scene_boot95_high": round(bhi, 3),
                      "wilson95_low_iid": round(lo, 3), "wilson95_high_iid": round(hi, 3),
                      "median_final_err_mm": round(err, 1)})
    write_csv("validation.csv", table)
    return {r["row"]: r for r in table}


def validation_by_round(dag):
    table = []
    for k in ("R5", "R6", "R7", "R8"):
        for rnd in range(4):
            cells = {s: dag.get((k, rnd, s)) for s in range(3)}
            have = [c for c in cells.values() if c]
            table.append({"row": k, "report_name": NAME[k], "round": rnd,
                          "model": RUN_ID[ROWS[k][6]] if rnd == 0 else f"{RUN_ID[k]}{DAGGER_TAG}-r{rnd}",
                          **{f"seed{s}": f"{c[0]}/{c[1]}" if c else "" for s, c in cells.items()},
                          "pooled": f"{sum(c[0] for c in have)}/{sum(c[1] for c in have)}",
                          "seeds": len(have)})
    # Round 0 is the v1 model: fill seeds 1-2 from the v1 summary so every round-0 cell has all 3 seeds.
    v1 = {ROW_OF_RUN[r["row"]]: [int(x) for x in r["student_hold_per_seed"].split("/")]
          for r in read_csv(DATA / "validation/v1_R1-R4_summary.csv")}
    for t in table:
        if t["round"] == 0:
            seeds = v1[ROWS[t["row"]][6]]
            assert t["seed0"] == f"{seeds[0]}/32", (t, seeds)
            t.update({f"seed{s}": f"{seeds[s]}/32" for s in range(3)}, pooled=f"{sum(seeds)}/96", seeds=3)
    write_csv("validation_by_round.csv", table)


def test_tables(rolls):
    by = defaultdict(list)
    for r in rolls:
        by[(r["row"], r["suite"], r["band"])].append(r)
        by[(r["row"], r["suite"], "all")].append(r)
        by[(r["row"], "all", "all")].append(r)
    summary = {(r["row"], r["suite"], r["band"]): r for r in load_summary()
               if r["level"] == "row" and r["subset"] == "all"}
    order = ["expert"] + list(ROWS)
    bands = sorted({(s, b) for (_, s, b) in by if s != "all"}, key=lambda x: (x[0], x[1] != "all", x[1]))

    long = []
    for row in order:
        for suite, band in bands + [("all", "all")]:
            g = by[(row, suite, band)]
            k, n = sum(r["success"] for r in g), len(g)
            if (row, suite, band) in summary:  # cross-check against the suite's own report
                s = summary[(row, suite, band)]
                assert (int(s["successes"]), int(s["n"])) == (k, n), (row, suite, band)
                lo, hi = float(s["ci_low"]), float(s["ci_high"])
            else:
                lo, hi = scene_bootstrap(g)
            done = [float(r["completion_time_s"]) for r in g if r["success"]]
            per_seed = Counter(r["seed"] for r in g if r["success"])
            long.append({"row": row, "report_name": NAME.get(row, "Expert A (teacher)"), "suite": suite, "band": band,
                         "successes": k, "n": n, "rate_pct": round(100 * k / n, 1), "scene_boot95_low_pct": round(100 * lo, 1),
                         "scene_boot95_high_pct": round(100 * hi, 1),
                         "per_seed": "" if row == "expert" else "/".join(str(per_seed.get(str(s), 0)) for s in range(3)),
                         "completion_median_s": round(statistics.median(done), 2) if done else "",
                         "final1s_max_err_median_mm": round(statistics.median(float(r["final1s_max_err_mm"]) for r in g), 2)})
    write_csv("test_by_band.csv", long)

    cell = {(r["row"], r["suite"], r["band"]): r for r in long}
    wide = []
    for row in order:
        entry = {"row": row, "report_name": NAME.get(row, "Expert A (teacher)")}
        for s in SUITES:
            entry[f"{s}_pct"] = cell[(row, s, "all")]["rate_pct"]
        a = cell[(row, "all", "all")]
        entry.update({"all240_pct": a["rate_pct"], "all240_scene_boot95": f"{a['scene_boot95_low_pct']}-{a['scene_boot95_high_pct']}",
                      "all240_successes": f"{a['successes']}/{a['n']}"})
        wide.append(entry)
    write_csv("test_by_suite.csv", wide)
    return cell


# The 12 one-factor contrasts of the 2 x 2 x 2 design, plus 2 cross-method ones: (factor, row without, row with, held fixed).
CONTRASTS = [("Gate", "R1", "R2", "no OU, no DAgger"), ("Gate", "R3", "R4", "OU, no DAgger"),
             ("Gate", "R5", "R6", "DAgger, no OU"), ("Gate", "R7", "R8", "OU + DAgger (f-DAgger)"),
             ("OU", "R1", "R3", "no Gate, no DAgger"), ("OU", "R2", "R4", "Gate, no DAgger"),
             ("OU", "R5", "R7", "no Gate, DAgger"), ("OU", "R6", "R8", "Gate, DAgger"),
             ("DAgger", "R1", "R5", "no Gate, no OU"), ("DAgger", "R2", "R6", "Gate, no OU"),
             ("DAgger", "R3", "R7", "no Gate, OU"), ("DAgger", "R4", "R8", "Gate, OU"),
             # two cross-method contrasts used in the README: OU data instead of 3 DAgger rounds ("without" = OU row)
             ("DAgger instead of OU", "R3", "R5", "no Gate"), ("DAgger instead of OU", "R4", "R6", "Gate")]
METRICS = [("validation", None), ("T0", "T0"), ("T1", "T1"), ("T2", "T2"), ("T3", "T3"), ("T4", "T4"), ("T5", "T5"),
           ("all 240", "all")]


def per_scene_table(records):
    """{row: {scene: mean success over its seeds}}, {row: {seed: [success...]}}."""
    scene, seed = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in records:
        scene[r["row"]][r["scenario_id"]].append(float(r["success"]))
        seed[r["row"]][r["seed"]].append(float(r["success"]))
    return ({k: {s: np.mean(v) for s, v in d.items()} for k, d in scene.items()},
            {k: {s: np.mean(v) for s, v in d.items()} for k, d in seed.items()})


def paired(a, b, rng):
    """Paired per-scene difference b - a: mean, 95% bootstrap CI over scenes, two-sided sign-flip permutation p."""
    keys = sorted(a)
    assert keys == sorted(b)
    d = np.array([b[k] - a[k] for k in keys])
    boot = d[rng.integers(0, len(d), (PAIRED_N, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    flips = rng.choice([-1.0, 1.0], size=(PERM_N, len(d)))
    extreme = np.sum(np.abs((flips * d).mean(axis=1)) >= abs(d.mean()) - 1e-12)
    return d.mean(), lo, hi, (1 + extreme) / (1 + PERM_N), len(d)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def comparisons(val_scene, rolls):
    rng = np.random.default_rng(STATS_SEED)
    sources = {"validation": per_scene_table(val_scene)}
    for name, suite in METRICS[1:]:
        sources[name] = per_scene_table([r for r in rolls if r["row"] in ROWS and (suite == "all" or r["suite"] == suite)])
    table = []
    for factor, a, b, fixed in CONTRASTS:
        for metric, _ in METRICS:
            scene, seed = sources[metric]
            diff, lo, hi, p, n = paired(scene[a], scene[b], rng)
            pairs = [seed[b][j] - seed[a][i] for i in seed[a] for j in seed[b]]
            same = sum(np.sign(x) == np.sign(diff) for x in pairs) if diff != 0 else sum(x == 0 for x in pairs)
            table.append({"factor": factor, "held_fixed": fixed, "without_row": a, "without": NAME[a], "with_row": b,
                          "with": NAME[b], "metric": metric, "scenes": n, "seeds_per_scene": 3,
                          "rate_without_pct": round(100 * np.mean(list(scene[a].values())), 1),
                          "rate_with_pct": round(100 * np.mean(list(scene[b].values())), 1),
                          "diff_pp": round(100 * diff, 1), "paired_boot95_low_pp": round(100 * lo, 1),
                          "paired_boot95_high_pp": round(100 * hi, 1), "p_signflip": p,
                          "seed_pairings_same_sign": f"{same}/9",
                          "seed_pairing_diff_min_pp": round(100 * min(pairs), 1),
                          "seed_pairing_diff_max_pp": round(100 * max(pairs), 1)})
    for t, padj in zip(table, holm([t["p_signflip"] for t in table])):
        t["p_signflip"] = float(f"{t['p_signflip']:.2g}")
        t["p_holm"] = float(f"{padj:.2g}")
        t["significant_holm_0.05"] = padj < 0.05
    write_csv("comparisons.csv", table)
    return table


def gate_effect(comp):
    keep = {"validation": "validation hold", "T0": "test T0 in range", "T2": "test T2 near",
            "T3": "test T3 height edges", "all 240": "test all 240"}
    method = {"R1": "Baseline", "R3": "OU", "R5": "DAgger", "R7": "f-DAgger"}
    write_csv("gate_effect.csv", [{
        "method": method[c["without_row"]], "metric": keep[c["metric"]], "without_gate_row": c["without_row"],
        "without_gate_pct": c["rate_without_pct"], "with_gate_row": c["with_row"], "with_gate_pct": c["rate_with_pct"],
        "gate_minus_no_gate_pp": c["diff_pp"], "paired_boot95_low_pp": c["paired_boot95_low_pp"],
        "paired_boot95_high_pp": c["paired_boot95_high_pp"], "p_holm": c["p_holm"],
        "seed_pairings_same_sign": c["seed_pairings_same_sign"]}
        for c in comp if c["factor"] == "Gate" and c["metric"] in keep])


def failure_modes(rolls):
    table = []
    for row in ["expert"] + list(ROWS):
        for suite in SUITES + ["all"]:
            g = [r for r in rolls if r["row"] == row and (suite == "all" or r["suite"] == suite)]
            fail = [r for r in g if not r["success"]]
            never = sum(float(r["min_err_mm"]) >= 10 for r in fail)
            table.append({"row": row, "report_name": NAME.get(row, "Expert A (teacher)"), "suite": suite, "n": len(g),
                          "failures": len(fail), "never_within_10mm": never, "reached_not_held": len(fail) - never,
                          "exceptions": sum(bool(r["failure"]) for r in g),
                          "median_min_err_of_failures_mm": round(statistics.median(float(r["min_err_mm"]) for r in fail), 1)
                          if fail else ""})
    write_csv("failure_modes.csv", table)


def terminal_grades(rolls):
    table = []
    for row in ["expert"] + list(ROWS):
        g = [r for r in rolls if r["row"] == row]
        c = Counter(r["grade"] for r in g)
        table.append({"row": row, "report_name": NAME.get(row, "Expert A (teacher)"), "n": len(g),
                      **{f"grade_{x}": c.get(x, 0) for x in "ABCDEF"},
                      "share_below_10mm_pct": round(100 * (c["A"] + c["B"]) / len(g), 1),
                      "median_final1s_max_err_mm": round(statistics.median(float(r["final1s_max_err_mm"]) for r in g), 2)})
    write_csv("terminal_grades.csv", table)


def hold_test():
    summary = {ROW_OF_RUN[r["row"]]: r for r in read_csv(DATA / "validation/v1_R1-R4_summary.csv")}
    table = []
    for k in ("R1", "R2", "R3", "R4"):
        files = sorted(glob.glob(str(DATA / f"hold_test/{RUN_ID[k]}__*.json")))
        tests = [json.loads(Path(p).read_text()) for p in files]
        stayed = [sum(t["arm"]["stayed_within_10mm"] for t in ts) for ts in tests]
        flat = [t for ts in tests for t in ts]
        stiff = statistics.median(t["linearised"]["student"]["stiffness_eig_max"] for t in flat)
        assert "/".join(map(str, stayed)) == summary[k]["hold_test_arm_stayed_per_seed"], k
        table.append({"row": k, "report_name": NAME[k], "scenes_per_seed": len(tests[0]),
                      "arm_stayed_per_seed": "/".join(map(str, stayed)), "arm_stayed": f"{sum(stayed)}/{len(flat)}",
                      "full_stayed": f"{sum(t['full']['stayed_within_10mm'] for t in flat)}/{len(flat)}",
                      "median_stiffness_eig_max_nm_per_rad": round(stiff, 2),
                      "median_max_real_eig": round(statistics.median(t["linearised"]["student"]["max_real_eig"] for t in flat), 3),
                      "median_damping_diag": round(statistics.median(t["linearised"]["student"]["damping_diag_median"] for t in flat), 2),
                      "expert_median_max_real_eig": round(statistics.median(t["linearised"]["expert"]["max_real_eig"] for t in flat), 3)})
    write_csv("hold_test.csv", table)


def step_minutes():
    """{(row, module, round): minutes} for the seed-0 chain of the 2026-11-05 v3 DAgger run."""
    out, row, step = {}, None, None
    for line in (DATA / "logs/v3_2026-11-05_dagger_run_log.txt").read_text().splitlines():
        m = re.match(r"===== (R\d)-\S+ start", line)
        if m:
            row = m.group(1)
        if line.startswith("$ "):
            module = line.split(" -m ")[1].split()[0].split(".")[-1]
            k = re.search(r"(?:--round r|-r)(\d)\b", line) or re.search(r"-r(\d)__", line)
            s = re.search(r"identity\.seed=(\d)", line) or re.search(r"__s(\d)__", line)
            step = (row, module, int(k.group(1)) if k else 0, int(s.group(1)) if s else 0)
        m = re.search(r"done in ([\d.]+) min", line)
        if m and step and step[3] == 0:
            out[step[:3]] = float(m.group(1))
    return out


def cost():
    fitting = set(json.loads((DATA / "datasets/clean253_split.json").read_text())["fitting_ids"]["clean253"])
    clean = dataset_rows("clean253__features-v1", fitting)
    ou = dataset_rows("dart-d2s075x2__features-v1")
    times = [datetime.datetime.fromisoformat(json.loads(l)["time"])
             for l in open(DATA / "logs/ou_collection_attempts.jsonl")]
    ou_min = (max(times) - min(times)).total_seconds() / 60
    v1_train = [float(m) for m in re.findall(r"finished \S+ \(([\d.]+) min\)",
                                              (DATA / "logs/v1_2026-10-28_run_log.txt").read_text())]
    minutes = step_minutes()
    table = []
    for k, v in ROWS.items():
        base = ROWS[k][6] or k
        eps, rows = clean
        added_labels, added_min, cycles, note = 0, 0.0, 0, []
        if ROWS[base][4]:
            eps, rows = eps + ou[0], rows + ou[1]
            added_labels += ou[1]
            added_min += ou_min
            note.append(f"OU collection {ou_min:.1f}")
        if v[5]:
            feats = "features-v1-gate" if v[3] else "features-v1"
            for rnd in (1, 2, 3):
                e, r = dataset_rows(f"dagger-{k}-v3-r{rnd}__{feats}")
                eps, rows, added_labels = eps + e, rows + r, added_labels + r
                parts = [minutes[(k, s, rnd)] for s in ("collect_dagger", "build", "train", "evaluate")]
                added_min += sum(parts)
                cycles += 1
            note.append("3 DAgger rounds (collect + build + train + validate; seed-0 chain) "
                        f"{sum(minutes[(k, s, r)] for s in ('collect_dagger', 'build', 'train', 'evaluate') for r in (1, 2, 3)):.1f}")
        table.append({"row": k, "report_name": NAME[k], "training_episodes": eps, "training_rows_M": round(rows / 1e6, 2),
                      "added_expert_labels_M": round(added_labels / 1e6, 2), "added_wall_min": round(added_min, 1),
                      "sequential_train_collect_cycles": cycles, "note": " + ".join(note) + (" min" if note else "-")})
    write_csv("cost.csv", table)
    print(f"  clean fitting {clean}, OU {ou}, v1 training run minutes {sorted(set(v1_train))}")


PRECISION_PAIRS = [("R8", "R4"), ("R8", "R6"), ("R8", "R7"), ("R6", "R4"), ("R5", "R4")]


def precision(rolls):
    """Terminal precision in range (secondary analysis; see the module docstring). No Holm correction: descriptive CIs."""
    groups = {"T0": ("T0",), "T0+T4+T5": ("T0", "T4", "T5")}
    table = []
    for gname, suites in groups.items():
        per_scene = {}
        for k in ["expert"] + list(ROWS):
            g = [r for r in rolls if r["row"] == k and r["suite"] in suites]
            e = np.array([float(r["final1s_max_err_mm"]) for r in g])
            scenes = defaultdict(list)
            for r in g:
                scenes[r["scenario_id"]].append(math.log(max(float(r["final1s_max_err_mm"]), 1e-3)))
            per_scene[k] = {sc: float(np.mean(v)) for sc, v in scenes.items()}
            table.append({"group": gname, "row": k, "report_name": NAME.get(k, "Expert A (teacher)"), "n": len(g),
                          "median_final1s_max_err_mm": round(float(np.median(e)), 2),
                          "share_below_5mm_pct": round(100 * float(np.mean(e < 5)), 1),
                          "share_below_10mm_pct": round(100 * float(np.mean(e < 10)), 1),
                          "comparison": "", "error_ratio": "", "ratio_boot95_low": "", "ratio_boot95_high": ""})
        rng = np.random.default_rng(20261105)
        for a, b in PRECISION_PAIRS:
            keys = sorted(per_scene[a])
            d = np.array([per_scene[a][sc] - per_scene[b][sc] for sc in keys])
            lo, hi = np.exp(np.quantile(d[rng.integers(0, len(d), (PAIRED_N, len(d)))].mean(axis=1), [0.025, 0.975]))
            table.append({"group": gname, "row": f"{a}/{b}", "report_name": f"{NAME[a]} / {NAME[b]}", "n": len(keys),
                          "median_final1s_max_err_mm": "", "share_below_5mm_pct": "", "share_below_10mm_pct": "",
                          "comparison": "geometric-mean error ratio, paired by scene",
                          "error_ratio": round(float(np.exp(d.mean())), 3), "ratio_boot95_low": round(float(lo), 3),
                          "ratio_boot95_high": round(float(hi), 3)})
    write_csv("precision.csv", table)


def efficiency(rolls):
    table = []
    for k, v in ROWS.items():
        n_in = 22 if v[3] else 21
        params = (n_in * 256 + 256 + 512) + 2 * (256 * 256 + 256 + 512) + (256 * 7 + 7) + (256 * 2 + 2)
        g = [r for r in rolls if r["row"] == k and r["latency_p50_us"]]
        table.append({"row": k, "report_name": NAME[k], "inputs": n_in, "parameters": params,
                      "float32_weights_kib": round(params * 4 / 1024, 1),
                      "latency_p50_us_median": round(statistics.median(float(r["latency_p50_us"]) for r in g), 1),
                      "latency_p99_us_median": round(statistics.median(float(r["latency_p99_us"]) for r in g), 1),
                      "rollouts": len(g)})
    write_csv("efficiency.csv", table)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rolls = rollouts()
    dag = dagger_validation()
    naming()
    val_scene = validation_per_scene(dag)
    validation(dag, val_scene)
    validation_by_round(dag)
    test_tables(rolls)
    gate_effect(comparisons(val_scene, rolls))
    failure_modes(rolls)
    terminal_grades(rolls)
    hold_test()
    cost()
    efficiency(rolls)
    precision(rolls)
