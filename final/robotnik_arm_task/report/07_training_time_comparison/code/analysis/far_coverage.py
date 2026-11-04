"""Why DAgger + Gate wins far targets (T1) and f-DAgger + Gate rounds lose them: data coverage and failure geometry.

1. Coverage. For each training dataset: rows, driving-phase share (reach gate closed), and base-to-target distance
   coverage (max, rows beyond 3 m and 4 m). R6's round-1 data contain far states because its BC student drove away.
2. Failure geometry. For failed T1 rollouts: progress along the start->target line and lateral offset of the final
   end effector. A navigation failure ends short or off-line; an arm failure ends at the target.

Usage (from final/robotnik_arm_task/; needs data/datasets and results/test, which are git-ignored):
    python report/07_training_time_comparison/code/analysis/far_coverage.py
Writes data_coverage.csv and t1_failure_geometry.csv to report/07_training_time_comparison/data/analysis/.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[4]
OUT = TASK_DIR / "report/07_training_time_comparison/data/analysis"
DATASETS = ["clean253", "dart-d2s075x2"] + [f"dagger-R{r}-v2-r{k}" for r in (6, 8) for k in (1, 2, 3)] + \
           [f"dartfull-a6-k{k}" for k in (1, 2, 3)]
TESTS = ["2026-11-01_215340__R1-R8", "2026-11-03_231207__dart-low-round-dagger",
         "2026-11-04_134912__dart-full-a6", "2026-11-04_151122__dart-full-a6-r2r3"]
NAMES = [("R4-dart-gate__", "f-DAgger + Gate r0"), ("R8-dart-gate-dagger-r", "f-DAgger + Gate r"), ("R6-gate-dagger-r", "DAgger + Gate r"),
         ("R9-dart-full-a6-r", "DART + Gate r"), ("R2-gate__", "Baseline + Gate (r0)")]


def coverage():
    rows = []
    for name in DATASETS:
        d = TASK_DIR / "data/datasets/A" / f"{name}__features-v1-gate"
        x = np.load(d / "x.npy", mmap_mode="r")
        s = np.asarray(x[::50, :])  # every 50th row
        dist = np.hypot(s[:, 14], s[:, 15])
        n = len(x)
        rows.append({"dataset": name, "rows": n, "driving_share_pct": round(100 * float((s[:, 21] < 0.5).mean()), 1),
                     "max_target_distance_m": round(float(dist.max()), 2),
                     "rows_beyond_3m": int(round(n * float((dist > 3).mean()))),
                     "rows_beyond_4m": int(round(n * float((dist > 4).mean())))})
        print(rows[-1], flush=True)
    return rows


def label(model_id):
    for prefix, name in NAMES:
        if model_id.startswith(prefix):
            return name if name.endswith("0)") or name.endswith("r0") else name + model_id[len(prefix)]
    return None


def failure_geometry():
    stats = defaultdict(list)
    for t in TESTS:
        for r in csv.DictReader(open(TASK_DIR / "results/test" / t / "rollouts.csv")):
            name = label(r["model_id"])
            if name is None or r["suite"] != "T1" or r["success"] == "True":
                continue
            p = np.array(json.loads(r["final_ee_pos_W"])[:2])
            s0 = np.array([float(r["start_x"]), float(r["start_y"])])
            g = np.array([float(r["target_x"]), float(r["target_y"])])
            u = (g - s0) / np.linalg.norm(g - s0)
            v = p - s0
            stats[name].append((v @ u / np.linalg.norm(g - s0), abs(v[0] * u[1] - v[1] * u[0])))
    rows = []
    for name in sorted(stats):
        a = np.array(stats[name])
        rows.append({"method_round": name, "t1_failures": len(a),
                     "progress_median": round(float(np.median(a[:, 0])), 2),
                     "progress_q25": round(float(np.quantile(a[:, 0], 0.25)), 2),
                     "progress_q75": round(float(np.quantile(a[:, 0], 0.75)), 2),
                     "overshoot_pct": round(100 * float((a[:, 0] > 1.05).mean())),
                     "backwards_pct": round(100 * float((a[:, 0] < 0).mean())),
                     "lateral_median_m": round(float(np.median(a[:, 1])), 2)})
        print(rows[-1])
    return rows


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "data_coverage.csv", coverage())
    write(OUT / "t1_failure_geometry.csv", failure_geometry())
