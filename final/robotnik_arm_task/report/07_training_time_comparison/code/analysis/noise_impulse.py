"""Applied action noise, measured from the collected episodes: OU (the R3/R4 OU-noise data) vs full DART (alpha = 1, 6).

The arm's state deviation is the time integral of the torque perturbation. So, besides the per-step RMS, this script
reports the RMS impulse (N*m*s) over 10/100/500 ms windows. It also reports the share of steps with noise, and the
wheel RMS. Every 8th episode of each collection is used.

Usage (from final/robotnik_arm_task/; needs data/Markovian/DART, which is git-ignored):
    python report/07_training_time_comparison/code/analysis/noise_impulse.py
Writes report/07_training_time_comparison/data/noise/applied_noise.csv.
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[4]
OUT = TASK_DIR / "report/07_training_time_comparison/data/noise/applied_noise.csv"
DT = 1e-3
WINDOWS_MS = (10, 100, 500)


def collections():
    root = TASK_DIR / "data/Markovian/DART"
    out = {"OU (R3/R4 data)": root / "2026-10-28__collection__d2-s075-x2-58af5db87875"}
    for d in sorted(glob.glob(str(root / "*__collection__dartfull-fitting-k1-*"))):
        cov = json.load(open(Path(d) / "manifest.json"))["protocol"]["covariance"]
        out["DART alpha = 6 (iteration 1)" if "a6" in cov else "DART alpha = 1 (iteration 1)"] = Path(d)
    return out


def measure(root: Path) -> dict:
    files = sorted(glob.glob(str(root / "train/*.npz")))[::8]
    arm, wheel, noisy, imp = [], [], [], {w: [] for w in WINDOWS_MS}
    for f in files:
        n = np.load(f)["sampled_action_noise"]
        on = np.any(n != 0, axis=1)
        noisy.append(on.mean())
        arm.append(n[on, :7])
        wheel.append(n[on, 7:])
        c = np.cumsum(np.vstack([np.zeros((1, 7)), n[:, :7]]), axis=0) * DT
        for w in WINDOWS_MS:
            seg = c[w:] - c[:-w]
            imp[w].append(seg[on[:len(seg)]])
    a, wh = np.concatenate(arm), np.concatenate(wheel)
    per_joint = np.sqrt((a ** 2).mean(0))
    row = {"episodes_used": len(files), "noisy_steps_pct": round(100 * float(np.mean(noisy)), 1),
           "arm_rms_nm": round(float(np.sqrt((a ** 2).mean())), 3),
           "arm_rms_per_joint_min_nm": round(float(per_joint.min()), 2), "arm_rms_per_joint_max_nm": round(float(per_joint.max()), 2)}
    for w in WINDOWS_MS:
        row[f"arm_impulse_rms_{w}ms_nms"] = round(float(np.sqrt((np.concatenate(imp[w]) ** 2).mean())), 4)
    row["wheel_rms_rad_s"] = round(float(np.sqrt((wh ** 2).mean())), 3)
    return row


if __name__ == "__main__":
    rows = []
    for name, root in collections().items():
        rows.append({"noise": name, "collection": root.name, **measure(root)})
        print(rows[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
