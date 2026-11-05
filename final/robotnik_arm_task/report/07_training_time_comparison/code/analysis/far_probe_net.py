"""Distance-extrapolation probe of the learned wheel commands (explains the far-target regression of f-DAgger + Gate rounds).

Takes clean driving-phase states (gate closed, target 1-3.1 m away), moves the target along its own bearing to
distance D (the end-effector error features shift with it), and compares each model's wheel output with the expert's
label at the original state. The expert's wheel command saturates beyond about 0.25 m, so the label does not change
with D. Also reports each model's input scale for target_bx: a 10 m target in std units shows how far outside the
training inputs it lies.

Usage (from final/robotnik_arm_task/; needs data/datasets and checkpoints, which are git-ignored):
    python report/07_training_time_comparison/code/analysis/far_probe_net.py
Writes report/07_training_time_comparison/data/analysis/far_probe_net.csv.
"""
import csv
import glob
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TASK_DIR))
from training.common.policies import MLPPolicy  # noqa: E402

OUT = TASK_DIR / "report/07_training_time_comparison/data/analysis/far_probe_net.csv"
DISTANCES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]


def models():
    out = [("f-DAgger + Gate r0 (R4)", k, TASK_DIR / f"checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s{k}__v1__2026-10-28/step_42000.npz")
           for k in range(3)]
    for row, name in [("R8-dart-gate-dagger", "f-DAgger + Gate"), ("R6-gate-dagger", "DAgger + Gate")]:
        for r in (1, 2, 3):
            for k in range(3):
                path = glob.glob(str(TASK_DIR / f"checkpoints/ablation/2026-11-05__training-ablation-v3/{row}-nodev-r{r}__MLP01-256x3__s{k}__v3__2026-11-05/step_42000.npz"))[0]
                out.append((f"{name} r{r}", k, Path(path)))
    return out


def main():
    ds = TASK_DIR / "data/datasets/A/clean253__features-v1-gate"
    x, y = np.load(ds / "x.npy", mmap_mode="r"), np.load(ds / "y.npy", mmap_mode="r")
    s, yy = np.asarray(x[::997], float), np.asarray(y[::997], float)
    d0 = np.hypot(s[:, 14], s[:, 15])
    keep = (s[:, 21] < 0.5) & (d0 >= 1.0)
    s, yy, d0 = s[keep], yy[keep], d0[keep]

    def moved(dist):
        z = s.copy()
        new = z[:, 14:16] / d0[:, None] * dist
        z[:, 17:19] += new - z[:, 14:16]  # ee_err = target - ee, so it shifts with the target
        z[:, 14:16] = new
        return z

    rows = []
    for name, seed, path in models():
        pol = MLPPolicy.load(str(path))
        errs = []
        for dist in DISTANCES:
            pred = np.array([pol.predict(v) for v in moved(dist)])
            e = np.sqrt(np.mean((np.clip(pred[:, 7:9], -pol.wheel_limit, pol.wheel_limit) - yy[:, 7:9]) ** 2, axis=1))
            errs.append(round(float(np.median(e)), 3))
        rows.append([name, seed] + errs + [round(float(pol.x_scale[14]), 3), round(float(10 / pol.x_scale[14]), 1)])
        print(name, seed, errs, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "seed"] + [f"median_wheel_rmse_rad_s_at_{d}m" for d in DISTANCES]
                   + ["x_scale_target_bx_m", "target_10m_in_std_units"])
        w.writerows(rows)
    print(f"{len(s)} probe states; wrote {OUT.relative_to(TASK_DIR)}")


if __name__ == "__main__":
    main()
