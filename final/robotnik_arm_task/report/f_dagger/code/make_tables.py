"""f-DAgger (OU noise injection + DAgger): rebuild every table in ../tables from ../data.

No simulation or training runs here; the inputs were produced by the pipeline commands listed in ../README.md. Tables:
  ou_pilot.csv               OU-scale pilot of 2026-10-26: reach / sustained hold per cohort, teacher and noise scale
  ou_collection.csv          the OU-noise collection used for training (d2-s075-x2): episodes, holds, alpha, rows
  dagger_collections.csv     every DAgger round of 2026-11-01: collector, student holds while collecting, rows
  factorial_validation.csv   R1-R8 (gate x OU noise x DAgger) on the 32 validation scenes, 3 seeds, Wilson 95%
  dagger_rounds.csv          R5-R8 per DAgger round: validation hold per seed (rounds 1-2 seeds 1-2 from 2026-11-03)
  factorial_test.csv         R1-R8 and teacher A on the unified 240-scene test (T0-T5), 3 seeds pooled
  noise_strength_controls.csv R4 noise-scale and data-volume controls (teammate's run, transcribed; see README)

Usage (from final/robotnik_arm_task/):
    python report/f_dagger/code/make_tables.py
"""
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # report/f_dagger
DATA, OUT = ROOT / "data", ROOT / "tables"
ROWS = {  # row -> (gate, OU noise, DAgger)
    "R1-baseline": (0, 0, 0), "R2-gate": (1, 0, 0), "R3-dart": (0, 1, 0), "R4-dart-gate": (1, 1, 0),
    "R5-baseline-dagger": (0, 0, 1), "R6-gate-dagger": (1, 0, 1), "R7-dart-dagger": (0, 1, 1), "R8-dart-gate-dagger": (1, 1, 1),
}
BASE_ROW = {"R1-baseline": "R5-baseline-dagger", "R2-gate": "R6-gate-dagger", "R3-dart": "R7-dart-dagger",
            "R4-dart-gate": "R8-dart-gate-dagger"}  # round-0 model -> the DAgger row it starts
SUITES = ["T0", "T1", "T2", "T3", "T4", "T5"]


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
    return f"{centre - half:.2f}-{centre + half:.2f}"


def ou_pilot():
    groups = defaultdict(list)
    for cohort in ("tuning", "confirmation", "d2_sampled"):
        for r in read_csv(DATA / f"ou_noise/pilot/{cohort}_results.csv"):
            scale = "alpha ~ U[0.25, 1] x (0.75, 0.15)" if cohort == "d2_sampled" else f"{float(r['sigma_t']):g} / {float(r['sigma_w']):g}"
            groups[(cohort, r["teacher"], scale)].append(r)
    rows = []
    for (cohort, teacher, scale), eps in groups.items():
        rows.append({"cohort": cohort, "teacher": "A (lead 0.15 s)" if teacher == "Markovian" else "B (template)",
                     "arm_sigma_nm / wheel_sigma_rad_s": scale, "n": len(eps),
                     "reached": sum(e["reached"] == "True" for e in eps),
                     "sustained_hold": sum(e["held_last1s"] == "True" for e in eps),
                     "median_final_err_mm": round(statistics.median(float(e["final_error_mm"]) for e in eps), 2)})
    write_csv("ou_pilot.csv", rows)


def ou_collection():
    eps = read_csv(DATA / "collections/ou_d2-s075-x2_episodes.csv")
    alpha = [float(e["alpha"]) for e in eps]
    steps = sum(int(e["T_actions"]) for e in eps)
    held = [e for e in eps if e["sustained_hold_pass"] == "True"]
    write_csv("ou_collection.csv", [{
        "protocol": "d2-s075-x2", "scenarios": len({e["scenario_id"] for e in eps}), "draws": len({e["draw"] for e in eps}),
        "episodes": len(eps), "sustained_hold": len(held), "rows_all": steps,
        "rows_used_for_training": sum(int(e["T_actions"]) for e in held),
        "noisy_step_share": round(sum(int(e["noisy_steps"]) for e in eps) / steps, 3),
        "exec_clip_step_share": round(sum(int(e["exec_clip_steps"]) for e in eps) / steps, 5),
        "alpha_min": round(min(alpha), 3), "alpha_median": round(statistics.median(alpha), 3), "alpha_max": round(max(alpha), 3),
        "median_final_err_mm": round(statistics.median(float(e["final_ee_err_mm"]) for e in eps), 2)}])


def dagger_collections():
    rows = []
    for path in sorted(glob.glob(str(DATA / "collections/dagger/*_manifest.json"))):
        m = json.loads(Path(path).read_text())
        eps = read_csv(path.replace("_manifest.json", "_episodes.csv"))
        collector = m["student"]["run_id"].split("__")[0]
        row = BASE_ROW.get(collector) or collector.rsplit("-r", 1)[0]
        rmse = [float(e["student_teacher_arm_rmse_nm"]) for e in eps if e["student_teacher_arm_rmse_nm"]]
        rows.append({"row": row, "round": m["round"], "collector": collector, "episodes": m["counts"]["train_episodes"],
                     "fitting_student_hold": f"{m['counts']['fitting_student_hold']}/{m['counts']['fitting_episodes']}",
                     "dev_student_hold": f"{m['counts']['development_student_hold']}/{m['counts']['development_episodes']}",
                     "rows": sum(int(e["T_actions"]) for e in eps),
                     "median_student_teacher_arm_rmse_nm": round(statistics.median(rmse), 3),
                     "collection": Path(path).name.replace("_manifest.json", "")})
    rows.sort(key=lambda r: (list(ROWS).index(r["row"]), r["round"]))
    write_csv("dagger_collections.csv", rows)


def dagger_validation():
    """{(row, round, seed): (hold, n)} from the v2 summaries (2026-11-01 run plus the 2026-11-03 extra seeds)."""
    out = {}
    for path in sorted(glob.glob(str(DATA / "validation/v2_*.csv"))):
        for r in read_csv(path):
            k, n = (int(x) for x in r["sustained_hold"].split("/"))
            out[(r["row"], int(r["round"]), int(r["seed"]))] = (k, n)
    return out


def factorial_validation(v2):
    rows = []
    for r in read_csv(DATA / "validation/v1_R1-R4_summary.csv"):
        per_seed = [int(x) for x in r["student_hold_per_seed"].split("/")]
        rows.append((r["row"], "-", per_seed, 32))
    for row in list(ROWS)[4:]:
        per_seed = [v2[(row, 3, s)][0] for s in range(3)]
        rows.append((row, "r3", per_seed, 32))
    table = []
    for row, rnd, per_seed, n in rows:
        gate, ou, dagger = ROWS[row]
        k = sum(per_seed)
        table.append({"row": row, "gate": gate, "ou_noise": ou, "dagger": dagger, "round": rnd,
                      "hold_per_seed": "/".join(map(str, per_seed)), "pooled": f"{k}/{3 * n}", "rate": round(k / (3 * n), 3),
                      "wilson95": wilson(k, 3 * n)})
    write_csv("factorial_validation.csv", table)


def dagger_rounds(v2):
    table = []
    for row in list(ROWS)[4:]:
        for rnd in range(4):
            cells = {s: v2.get((row, rnd, s)) for s in range(3)}
            table.append({"row": row, "round": rnd,
                          **{f"seed{s}": f"{c[0]}/{c[1]}" if c else "" for s, c in cells.items()},
                          "pooled": f"{sum(c[0] for c in cells.values() if c)}/{sum(c[1] for c in cells.values() if c)}"})
    write_csv("dagger_rounds.csv", table)


def factorial_test():
    cells = {}
    for r in read_csv(DATA / "test/R1-R8_2026-11-01_summary.csv"):
        if r["level"] == "row" and r["subset"] == "all" and r["band"] == "all":
            cells[(r["row"], r["suite"])] = (int(r["successes"]), int(r["n"]))
    table = []
    for short, full in [("expert", "teacher A")] + [(f"R{i + 1}", name) for i, name in enumerate(ROWS)]:
        entry = {"row": full, "round": "r3" if short in ("R5", "R6", "R7", "R8") else "-"}
        for s in SUITES:
            k, n = cells[(short, s)]
            entry[s] = f"{100 * k / n:.1f}"
        k = sum(cells[(short, s)][0] for s in SUITES)
        n = sum(cells[(short, s)][1] for s in SUITES)
        entry["all_240"] = f"{100 * k / n:.1f}"
        entry["successes"] = f"{k}/{n}"
        table.append(entry)
    write_csv("factorial_test.csv", table)


def noise_strength_controls():
    # Transcribed from docs/reports/2026-10-31__report__dagger-ablation-and-final-model.md, Section 4 (teammate's
    # machine, 2026-10-30, validation 3 seeds x 32). Per-seed counts and the checkpoints are not on this machine.
    rows = [("R2-gate", "clean only", "-", "-", "10/96", "this machine (v1)"),
            ("R4-dart-gate-s0375", "clean + OU", "0.375 / 0.075", 442, "69/96", "teammate, 2026-10-30"),
            ("R4-dart-gate", "clean + OU", "0.75 / 0.15", 442, "83/96", "this machine (v1)"),
            ("R4-dart-gate-s150", "clean + OU", "1.5 / 0.3", 442, "84/96", "teammate, 2026-10-30"),
            ("R4-dart-gate-x4", "clean + OU", "0.75 / 0.15", 884, "82/96", "teammate, 2026-10-30")]
    write_csv("noise_strength_controls.csv", [
        {"variant": v, "data": d, "arm_sigma_nm / wheel_sigma_rad_s": s, "ou_episodes": e, "validation_hold": h, "source": src}
        for v, d, s, e, h, src in rows])


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    ou_pilot()
    ou_collection()
    dagger_collections()
    v2 = dagger_validation()
    factorial_validation(v2)
    dagger_rounds(v2)
    factorial_test()
    noise_strength_controls()
