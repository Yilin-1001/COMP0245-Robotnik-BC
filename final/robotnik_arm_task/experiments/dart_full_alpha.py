"""Full DART, one iteration with a scaled noise prior alpha (paper's grasping experiment used 3x and 6x T tr(Sigma_hat)).

Sigma = scale * Sigma_hat_1, where Sigma_hat_1 is iteration 1's learner-error estimate from experiments/dart_full.py
(R2-gate on clean held-out demonstrations). Same noise RNG as the alpha = 1 run (common random numbers), 221 fitting
scenarios x 2 draws, then clean253 + this collection, same recipe, seeds as given.

Models: checkpoints/ablation/<date>__training-ablation-v2/R9-dart-full-a<scale>-r1__MLP01-256x3__s<seed>__v2__<date>/
Results: results/ablation/<date>__dart-full/ (summary.csv, timing.csv rows tagged a<scale>)

Usage (from final/robotnik_arm_task/):
    python -m experiments.dart_full_alpha --date 2026-11-04 --scale 6 --seeds 0 1 2 --workers 20
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments import dart_full as d
from experiments.dagger_rounds import RECIPE, TASK_DIR, run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True)
    p.add_argument("--scale", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args(argv)
    out_dir = TASK_DIR / f"results/ablation/{args.date}__dart-full"
    tag = f"a{args.scale}"
    row = f"{d.ROW}-{tag}"
    base = out_dir / "covariance" / "sigma_k1.npz"
    with np.load(base) as z:
        sigma_hat = z["sigma_hat"]
    cov = out_dir / "covariance" / f"sigma_k1_{tag}.npz"
    np.savez(cov, sigma=args.scale * sigma_hat, sigma_hat=sigma_hat, scale=args.scale)
    cov.with_suffix(".json").write_text(json.dumps({"source": str(base.relative_to(TASK_DIR)), "scale": args.scale,
                                                    "alpha_rule": f"alpha = {args.scale} T tr(Sigma_hat)",
                                                    "std": np.sqrt(np.diag(args.scale * sigma_hat)).round(4).tolist()}, indent=1))
    print(f"Sigma std ({tag}): {np.sqrt(np.diag(args.scale * sigma_hat)).round(3).tolist()}", flush=True)

    timing_path = out_dir / "timing.csv"
    timing = list(csv.DictReader(open(timing_path))) if timing_path.exists() else []

    def timed(step, fn):
        t0 = time.time()
        value = fn()
        timing.append({"iteration": f"{tag}-1", "step": step, "minutes": round((time.time() - t0) / 60, 2),
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(timing_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(timing[0]))
            w.writeheader()
            w.writerows(timing)
        print(f"  [{step} {tag}] {timing[-1]['minutes']} min", flush=True)
        return value

    root = timed("collect_noisy", lambda: d.collect(str(cov), "fitting", 1, 2, args.workers, out_dir / "logs" / f"{tag}-collect.log"))
    name = f"dartfull-{tag}-k1"
    timed("build", lambda: d.build(root, name, args.workers, out_dir / "logs" / f"{tag}-build.log"))
    variant = f"{row}-r1"
    text = (f"# {d.ROW}, alpha = {args.scale} T tr(Sigma_hat), one DART iteration: clean253 plus {name},\n"
            f"# inputs {d.FEATURES}. Written by experiments/dart_full_alpha.py.\n"
            f"data:\n  feature_definition: {d.FEATURES}\n  datasets: [clean253__{d.FEATURES}, {name}__{d.FEATURES}]\n"
            f"  validation_dataset: clean253__{d.FEATURES}\n")
    (TASK_DIR / "experiments/configs/datasets/variants" / f"{variant}.yaml").write_text(text)
    for seed in args.seeds:
        run_id = f"{variant}__MLP01-256x3__s{seed}__v2__{args.date}"
        model = TASK_DIR / f"checkpoints/ablation/{args.date}__training-ablation-v2/{run_id}/step_42000.npz"

        def train():
            if not model.exists():
                cmd = ["-m", "training.datasets.train", "--variant", variant, "--set", f"identity.date={args.date}",
                       "--set", f"identity.seed={seed}"]
                for key, value in RECIPE.items():
                    cmd += ["--set", f"{key}={value}"]
                run(cmd, out_dir / "logs" / f"{tag}-r1-s{seed}-train.log")

        if seed == 0:
            timed("train", train)
            entry = timed("validate", lambda: d.validate(model, f"{tag}-1", seed, out_dir, args.workers))
        else:
            train()
            entry = d.validate(model, f"{tag}-1", seed, out_dir, args.workers)
        print(f"  {variant} s{seed}: {entry['sustained_hold']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
