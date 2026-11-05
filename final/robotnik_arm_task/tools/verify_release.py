"""Check that the released weights load and run in PyTorch, and agree with the NumPy exports used by the demo.

Run from final/robotnik_arm_task/ after tools/fetch_release.py:
    python tools/verify_release.py          # the 27 final models (R1-R9 x seeds 0-2)
    python tools/verify_release.py --all    # also the 22 round-history models

For every model it:
  1. loads the .pt with plain torch.load(weights_only=True) and rebuilds the network from the saved spec
     (load_state_dict(strict=True)), then counts parameters;
  2. predicts 9 actions (7 arm torques in N*m, right/left wheel speeds in rad/s) for 256 probe states
     drawn around the saved input normaliser, in PyTorch (float32);
  3. predicts the same states with the .npz through MLPPolicy (NumPy, float64, the live-demo/evaluation path);
  4. checks the .npz SHA-256 against model_index.csv and reports the largest |PyTorch - NumPy| difference.
It also times one single-state forward pass on CPU. Closed-loop checks: README sections 5 and 6.
"""
import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
from training.common.models import model_from_spec  # noqa: E402
from training.common.policies import MLPPolicy  # noqa: E402

RELEASE = TASK / "checkpoints/2026-11-05final"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torch_policy(pt_path: Path):
    """Everything needed to run a released model with PyTorch alone."""
    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    net = model_from_spec(payload["meta"]["model"])
    net.load_state_dict(payload["model_state"], strict=True)
    net.eval()
    n = {k: torch.tensor(payload["normaliser"][k], dtype=torch.float32)
         for k in ("x_mean", "x_scale", "y_mean", "y_scale")}

    def predict(x: torch.Tensor) -> torch.Tensor:  # physical units, before the +-30 N*m / +-6 rad/s limits
        with torch.no_grad():
            return net((x - n["x_mean"]) / n["x_scale"]) * n["y_scale"] + n["y_mean"]

    return payload, net, n, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="include the 22 round-history models")
    parser.add_argument("--tol", type=float, default=1e-3, help="max allowed |PyTorch - NumPy| (N*m or rad/s)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [r for r in csv.DictReader((RELEASE / "model_index.csv").open()) if args.all or r["group"] == "final"]
    if not (RELEASE / rows[0]["checkpoint"]).exists():
        sys.exit("weights not found: run  python tools/fetch_release.py  first")

    print(f"torch {torch.__version__}, CPU, 1 thread\n")
    print(f"{'model':<44}{'inputs':>7}{'params':>9}{'max|pt-npz|':>13}{'ms/step':>9}  sha256")
    failures = 0
    for r in rows:
        pt, npz = RELEASE / r["checkpoint"], RELEASE / r["inference"]
        payload, net, n, predict = torch_policy(pt)
        rng = np.random.default_rng(0)
        x = n["x_mean"].numpy() + n["x_scale"].numpy() * rng.standard_normal((256, int(r["input_dim"])))
        y_torch = predict(torch.tensor(x, dtype=torch.float32)).double().numpy()
        y_numpy = MLPPolicy.load(str(npz), backend="numpy").predict(x)
        diff = float(np.abs(y_torch - y_numpy).max())
        one = torch.tensor(x[:1], dtype=torch.float32)
        for _ in range(20):
            predict(one)
        t0 = time.perf_counter()
        for _ in range(200):
            predict(one)
        ms = (time.perf_counter() - t0) / 200 * 1e3
        params = sum(p.numel() for p in net.parameters())
        hash_ok = sha256(npz) == r["inference_sha256"]
        ok = diff < args.tol and hash_ok and payload["step"] == 42000 and y_torch.shape == (256, 9)
        failures += not ok
        name = f"{r['model_id']} s{r['seed']} r{r['round']}"
        print(f"{name:<44}{r['input_dim']:>7}{params:>9,}{diff:>13.2e}{ms:>9.3f}  {'ok' if hash_ok else 'MISMATCH'}"
              f"{'' if ok else '   <-- FAIL'}")
    print(f"\n{len(rows) - failures}/{len(rows)} models passed (tolerance {args.tol:g}).")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
