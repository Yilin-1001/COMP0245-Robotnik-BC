"""Structure of the learner's error on expert states: what DART's first noise estimate sees, and what it cannot represent.

For Baseline + Gate (R2-gate) and OU + Gate (R4-dart-gate, f-DAgger + Gate round 0), seed 0, on the 32 clean validation episodes, this script reports:
  - per-channel error std (7 arm torques in N*m; right and left wheel in rad/s) and bias;
  - error std with the reach gate closed (driving) and open (reaching and holding);
  - autocorrelation of the arm error at lags 1 ms to 1 s (DART's i.i.d. Gaussian assumes 0 beyond 1 ms);
  - off-diagonal / diagonal energy of the arm error covariance (cross-joint correlation).

Usage (from final/robotnik_arm_task/; needs data/datasets and checkpoints, which are git-ignored):
    python report/07_training_time_comparison/code/analysis/learner_error_structure.py
Writes report/07_training_time_comparison/data/noise/learner_error_structure.csv.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TASK_DIR))
from training.common.policies import MLPPolicy  # noqa: E402

OUT = TASK_DIR / "report/07_training_time_comparison/data/noise/learner_error_structure.csv"
V1 = TASK_DIR / "checkpoints/ablation/2026-10-28__training-ablation-v1"
MODELS = {"Baseline + Gate (R2-gate s0)": V1 / "R2-gate__MLP01-256x3__s0__v1__2026-10-28/step_42000.npz",
          "OU + Gate (R4-dart-gate s0)": V1 / "R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/step_42000.npz"}
LAGS_MS = (1, 10, 100, 200, 1000)


def main():
    val = set(json.load(open(TASK_DIR / "experiments/splits/clean253.json"))["validation_ids"])
    ds = TASK_DIR / "data/datasets/A/clean253__features-v1-gate"
    eps = [r for r in csv.DictReader(open(ds / "episodes.csv")) if r["scenario_id"] in val]
    x, y = np.load(ds / "x.npy", mmap_mode="r"), np.load(ds / "y.npy", mmap_mode="r")
    rows = []
    for name, path in MODELS.items():
        pol = MLPPolicy.load(str(path))
        errors, gates = [], []
        for r in eps:
            a, b = int(r["row_start"]), int(r["row_end"])
            X, Y = np.asarray(x[a:b], float), np.asarray(y[a:b], float)
            errors.append(pol.net((X - pol.x_mean) / pol.x_scale) * pol.y_scale + pol.y_mean - Y)
            gates.append(X[:, 21])
        e, g = np.concatenate(errors), np.concatenate(gates)
        cov = np.cov(e.T)
        row = {"model": name, "episodes": len(eps), "rows": len(e),
               **{f"std_{c}": round(float(v), 3) for c, v in zip([f"arm{j}" for j in range(7)] + ["wheel_r", "wheel_l"],
                                                                  np.sqrt(np.diag(cov)))},
               "arm_std_driving_nm": round(float(e[g < 0.5, :7].std()), 3),
               "arm_std_reach_hold_nm": round(float(e[g > 0.5, :7].std()), 3),
               "wheel_std_driving_rad_s": round(float(e[g < 0.5, 7:].std()), 3),
               "wheel_std_reach_hold_rad_s": round(float(e[g > 0.5, 7:].std()), 3)}
        for lag in LAGS_MS:
            num = den = 0.0
            for ep in errors:
                ea = ep[:, :7] - ep[:, :7].mean(0)
                if len(ea) > lag:
                    num += (ea[lag:] * ea[:-lag]).sum()
                    den += (ea * ea).sum()
            row[f"arm_autocorr_{lag}ms"] = round(num / den, 2)
        arm = cov[:7, :7]
        row["arm_offdiag_over_diag"] = round(float(np.abs(arm - np.diag(np.diag(arm))).sum() / np.trace(arm)), 2)
        rows.append(row)
        print(row, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
