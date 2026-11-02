"""Isolated single-thread NumPy latency of exported MLP policies, with parameter and FLOP counts.

One call = MLPPolicy.predict on one feature row (input normalisation, network, output scaling); the forward-kinematics
features (~6 us, identical for every architecture) are excluded. Run on an otherwise idle machine.

Usage (from final/robotnik_arm_task/):
    python -m evaluation.latency_bench --models <model.npz> [...] --out results/ablation/<experiment>/latency.csv
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def mlp_cost(input_dim: int, hidden: list[int], outputs: int = 9) -> dict:
    """Parameters (weights, biases, LayerNorm gamma/beta) and FLOPs (2 x multiply-adds of the linear layers)."""
    params, flops, width = 0, 0, int(input_dim)
    for h in hidden:
        params += width * h + h + 2 * h
        flops += 2 * width * h
        width = h
    return {"parameters": params + width * outputs + outputs, "flops": flops + 2 * width * outputs}


def time_predict(model: str, calls: int = 20000, warmup: int = 2000, repeats: int = 3) -> dict:
    """Median (of per-repeat medians) and pooled p99 microseconds per predict() call."""
    from training.common.policies import MLPPolicy
    policy = MLPPolicy.load(model)
    x = np.array(policy.x_mean, dtype=float)
    for _ in range(warmup):
        policy.predict(x)
    medians, pooled = [], []
    for _ in range(repeats):
        times = np.empty(calls)
        for i in range(calls):
            start = time.perf_counter_ns()
            policy.predict(x)
            times[i] = (time.perf_counter_ns() - start) / 1000.0
        medians.append(float(np.median(times)))
        pooled.append(times)
    return {"median_us": float(np.median(medians)), "p99_us": float(np.percentile(np.concatenate(pooled), 99)),
            "repeat_medians_us": medians}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calls", type=int, default=20000)
    args = p.parse_args(argv)
    rows = []
    for model in args.models:
        with np.load(model) as z:
            meta = json.loads(str(z["metadata_json"]))
        hidden, run_id = meta["model"]["hidden"], meta["run_id"]
        t = time_predict(model, calls=args.calls)
        rows.append({"model": model, "architecture": run_id.split("__")[1], **mlp_cost(meta["input_dim"], hidden),
                     "npz_bytes": Path(model).stat().st_size, "median_us": t["median_us"], "p99_us": t["p99_us"]})
        print(f"{rows[-1]['architecture']}: {t['median_us']:.1f} us median, {t['p99_us']:.1f} us p99", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
