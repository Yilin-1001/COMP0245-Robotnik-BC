"""Markovian expert A vs the course template controller B: rebuild every table in ../tables from ../data.

No simulation runs here; the rollouts were produced by the scripts listed in ../README.md. Tables:
  hidden_state_probe.csv     B's action at one fixed observation when only its memory changes; A re-queried
  geometry.csv               sustained-hold counts over the target-height and start-distance grids (8 scenes per point)
  near_comparison.csv        near targets (0-0.5 m): success, first arrival, terminal error, base path, reversing
  disturbance_thresholds.csv largest tested noise sigma with >= 15/16 episodes completing / holding, per teacher
  physics_push.csv           changed physics and pushes (20 scenes per condition): A (lead 0.50 and 0.15) vs B

Usage (from final/robotnik_arm_task/):
    python report/01_markovian_expert/code/make_tables.py
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # report/01_markovian_expert
DATA, OUT = ROOT / "data", ROOT / "tables"
PASS = 15  # of 16 scenes per disturbance condition, as in docs/reports/2026-10-26__report__expert-operating-limits.md


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def hidden_state_probe():
    probe = json.loads((DATA / "hidden_state/controlled_intervention.json").read_text())
    rows = []
    for kind in ("reference_probe", "navigation_probe"):
        for p in probe[kind]:
            rows.append({"probe": kind, "q_des_offset_rad": p["reference_offset_rad"], "nav_done": p["done_before"],
                         **{f"torque_j{i + 1}_nm": round(t, 4) for i, t in enumerate(p["torque_arm"])},
                         **{f"wheel_{w}_rad_s": round(v, 4) for w, v in zip(("fr", "fl", "bl", "br"), p["wheel_cmds"])}})
    rows.append({"probe": "A_requery", "q_des_offset_rad": "", "nav_done": "",
                 "torque_j1_nm": probe["A_requery_max_torque_difference_nm"],
                 "wheel_fr_rad_s": probe["A_requery_max_wheel_difference_rad_s"]})
    write_csv("hidden_state_probe.csv", [{k: r.get(k, "") for k in rows[0]} for r in rows])


def geometry():
    cells = defaultdict(dict)
    for r in read_csv(DATA / "geometry/combined_summary.csv"):
        if r["sweep"] not in ("height", "far", "near"):
            continue
        teacher = "B_no_nav_timeout" if r["extended_B_timeout"] == "True" else r["teacher"]
        cells[(r["sweep"], float(r["value"]))][teacher] = f"{r['hold']}/{r['n']}"
    rows = [{"sweep": s, "value": v, "A": c.get("A", ""), "B": c.get("B", ""), "B_no_nav_timeout": c.get("B_no_nav_timeout", "")}
            for (s, v), c in sorted(cells.items())]
    write_csv("geometry.csv", rows)


def near_comparison():
    rows = [{"distance_m": r["distance_m"], "teacher": r["teacher"], "holds": f"{r['holds']}/{r['n']}",
             "median_first_arrival_s": round(float(r["median_first_arrival_t"]), 2),
             "median_final_err_mm": round(float(r["median_final_err_mm"]), 3),
             "median_base_path_m": round(float(r["median_base_path_m"]), 3),
             "episodes_reversing": f"{r['episodes_with_reverse_commands']}/{r['n']}"}
            for r in read_csv(DATA / "near/summary.csv")]
    write_csv("near_comparison.csv", rows)


def disturbance_thresholds():
    groups = defaultdict(list)
    for r in read_csv(DATA / "disturbance/expert_noise_robustness.csv"):
        if r["channel"] != "none":
            groups[(r["channel"], r["structure"], r["gated"], float(r["sigma"]), r["teacher"])].append(r)
    best = {}
    for (channel, structure, gated, sigma, teacher), eps in groups.items():
        assert len(eps) == 16, (channel, structure, gated, sigma, teacher)
        completes = sum(e["reached"] == "True" and e["recovered"] == "True" for e in eps)
        holds = sum(e["reached"] == "True" and float(e["hold_frac"] or 0) >= 0.9 for e in eps)
        key = (channel, structure, gated)
        for metric, n in (("completes", completes), ("holds", holds)):
            b = best.setdefault(key, {}).setdefault((teacher, metric), {"pass": [], "grid": []})
            b["grid"].append(sigma)
            if n >= PASS:
                b["pass"].append(sigma)

    def label(b):
        if not b["pass"]:
            return "none"
        top = max(b["pass"])
        return f">={top:g}" if top == max(b["grid"]) else f"{top:g}"

    order = {"white": 0, "ou_0.05": 1, "ou_0.2": 2, "ou_1.0": 3, "const": 4}
    rows = []
    for (channel, structure, gated), b in sorted(best.items(), key=lambda kv: (kv[0][0], kv[0][2], order[kv[0][1]])):
        rows.append({"channel": channel, "unit": "N*m" if channel == "arm" else "rad/s", "structure": structure,
                     "gated": gated, **{f"{t}_{m}": label(b[(t, m)]) for t in ("A", "B") for m in ("completes", "holds")}})
    write_csv("disturbance_thresholds.csv", rows)


def physics_push():
    by = defaultdict(list)
    for r in read_csv(DATA / "physics/expert_t3_dev_A_lead050.csv"):
        by[(r["condition"], "A_lead050")].append(r)
    for r in read_csv(DATA / "physics/expert_t3_dev_lead015_and_template.csv"):
        by[(r["condition"], "B_template" if r["controller"] == "template" else "A_lead015")].append(r)
    conditions = list(dict.fromkeys(c for c, _ in by))
    rows = []
    for c in conditions:
        row = {"condition": c}
        seeds = {k: sorted(e["seed"] for e in by[(c, k)]) for k in ("A_lead050", "A_lead015", "B_template")}
        assert len({tuple(s) for s in seeds.values()}) == 1, c  # all three controllers ran the same scenes
        for k in ("A_lead050", "B_template", "A_lead015"):
            eps = by[(c, k)]
            row[f"{k}_held"] = f"{sum(e['held'] == 'True' for e in eps)}/{len(eps)}"
            row[f"{k}_median_err_mm"] = round(statistics.median(float(e["err_end_mm"]) for e in eps), 1)
        rows.append(row)
    write_csv("physics_push.csv", rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    hidden_state_probe()
    geometry()
    near_comparison()
    disturbance_thresholds()
    physics_push()
