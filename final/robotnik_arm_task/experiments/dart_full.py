"""Full DART (Laskey et al., CoRL 2017) with noise optimisation, unattended, for comparison with DART-style + DAgger.

Row R9-dart-full, inputs features-v1-gate, recipe identical to R4/R8 (BC+J, Huber, 42k updates, lr 1e-3, 0.5/0.5).
Round 0 is R2-gate seed 0 (BC on the clean data with the same inputs). Iteration k = 1..K:
  1. held-out demonstrations H_{k-1}: 32 fresh scenes collected with the previous noise (k = 1: no noise);
  2. Sigma_hat_k = (1/T) E[sum_t e_t e_t^T], e_t = pi_{k-1}(x_t) - pi*(x_t) on H_{k-1} (paper eq. 3, held-out per its
     appendix), with alpha = T tr(Sigma_hat) (the paper's MuJoCo choice, no extra knowledge), so Sigma_k = Sigma_hat_k;
  3. D_k: 221 fitting scenarios x 2 draws with i.i.d. N(0, Sigma_k) noise at every step (collect_dart_full);
  4. pi_k trained from scratch on clean253 + D_1..D_k (aggregation), then validated on the 32 validation scenarios.
Every step's wall time is logged to timing.csv (the seed-0 chain is the pipeline cost). Extra seeds are trained last.

Models: checkpoints/ablation/<date>__training-ablation-v2/R9-dart-full-r<k>__MLP01-256x3__s<seed>__v2__<date>/
Results: results/ablation/<date>__dart-full/ (summary.csv, timing.csv, covariance/, logs/, validation/)

Usage (from final/robotnik_arm_task/):
    python -m experiments.dart_full --date 2026-11-04 --iterations 3 --extra-seeds 1 2 --workers 20
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.dagger_rounds import RECIPE, TASK_DIR, last_json, run

ROW = "R9-dart-full"
FEATURES = "features-v1-gate"
START = "checkpoints/ablation/2026-10-28__training-ablation-v1/R2-gate__MLP01-256x3__s0__v1__2026-10-28/step_42000.npz"


def timed(log_rows, out_dir, k, step, fn):
    t0 = time.time()
    value = fn()
    minutes = (time.time() - t0) / 60
    log_rows.append({"iteration": k, "step": step, "minutes": round(minutes, 2),
                     "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(out_dir / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)
    print(f"  [{step} k={k}] {minutes:.1f} min", flush=True)
    return value


def collect(cov: str, scenes: str, k: int, draws: int, workers: int, log: Path) -> str:
    out = run(["-m", "data_collection.collect_dart_full", "--covariance", cov, "--scenes", scenes, "--round", str(k),
               "--draws", str(draws), "--workers", str(workers)], log)
    res = last_json(out)
    print(f"  collected {scenes} k={k}: {res}", flush=True)
    return res["root"]


def names(tag: str = "") -> dict:
    """Row, dataset, covariance and round-label names; tag '' is the alpha = 1 run, e.g. 'a6' the alpha = 6 run."""
    t = f"-{tag}" if tag else ""
    return {"row": f"{ROW}{t}", "dataset": lambda k: f"dartfull{t}-k{k}", "cov": lambda k: f"sigma_k{k}{'_' + tag if tag else ''}.npz",
            "round": lambda k: f"{tag}-{k}" if tag else k}


def estimate(student: str, heldout_root: str, out: Path, scale: float = 1.0) -> dict:
    """Paper eq. 3 on every held-out episode: second moment of the learner's raw 9-d error w.r.t. the clean labels."""
    from data_collection.collect import read_metadata
    from training.common.dataset import EpisodeRef, FeatureBuilder, _episode_rows, sha256_file
    from training.common.policies import MLPPolicy
    pol = MLPPolicy.load(str(TASK_DIR / student))
    builder = FeatureBuilder("A", None, FEATURES)
    root = TASK_DIR / heldout_root
    rows = list(csv.DictReader(open(root / "episodes.csv")))
    second, n, per_ep = np.zeros((9, 9)), 0, []
    for r in rows:
        path = root / r["file"]
        meta = read_metadata(str(path))
        x, y, _ = _episode_rows(EpisodeRef(path, r["scenario_id"], "A", sha256_file(path), meta), builder)
        e = pol.net((x - pol.x_mean) / pol.x_scale) * pol.y_scale + pol.y_mean - y
        second += e.T @ e
        n += len(e)
        per_ep.append(len(e))
    sigma_hat = second / n
    sigma = scale * sigma_hat  # alpha = scale T tr(Sigma_hat): Sigma = alpha / (T tr(Sigma_hat)) * Sigma_hat
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, sigma=sigma, sigma_hat=sigma_hat, rows=n, episodes=len(rows))
    info = {"student": student, "heldout": heldout_root, "episodes": len(rows), "rows": n,
            "std": np.sqrt(np.diag(sigma)).round(4).tolist(), "trace": float(np.trace(sigma)),
            "alpha_rule": f"alpha = {scale} T tr(Sigma_hat)" + (" (no extra knowledge)" if scale == 1 else "")}
    out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    print(f"  Sigma_k std (arm N*m x7, wheels rad/s x2): {info['std']}", flush=True)
    return info


def build(root: str, name: str, workers: int, log: Path) -> None:
    run(["-m", "training.datasets.build", "--source", root, "--name", name, "--features", FEATURES,
         "--workers", str(workers)], log)


def write_variant(k: int, tag: str = "") -> str:
    nm = names(tag)
    name = f"{nm['row']}-r{k}"
    datasets = [f"clean253__{FEATURES}"] + [f"{nm['dataset'](i)}__{FEATURES}" for i in range(1, k + 1)]
    text = (f"# {nm['row']}, DART iteration {k}: clean253 plus full-DART (learner-fitted Gaussian noise) iterations 1..{k},\n"
            f"# inputs {FEATURES}. Written by experiments/dart_full.py.\n"
            f"data:\n  feature_definition: {FEATURES}\n  datasets: [{', '.join(datasets)}]\n"
            f"  validation_dataset: clean253__{FEATURES}\n")
    path = TASK_DIR / "experiments/configs/datasets/variants" / f"{name}.yaml"
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return name


def model_path(k: int, seed: int, date: str, tag: str = "") -> Path:
    run_id = f"{names(tag)['row']}-r{k}__MLP01-256x3__s{seed}__v2__{date}"
    return TASK_DIR / f"checkpoints/ablation/{date}__training-ablation-v2/{run_id}/step_42000.npz"


def train(k: int, seed: int, date: str, log: Path, tag: str = "") -> Path:
    model = model_path(k, seed, date, tag)
    if not model.exists():
        cmd = ["-m", "training.datasets.train", "--variant", write_variant(k, tag), "--set", f"identity.date={date}",
               "--set", f"identity.seed={seed}"]
        for key, value in RECIPE.items():
            cmd += ["--set", f"{key}={value}"]
        run(cmd, log)
    return model


def validate(model: Path, k, seed: int, out_dir: Path, workers: int) -> dict:
    cache = out_dir / "validation" / f"{model.parent.name}.json"
    if cache.exists():
        res = json.loads(cache.read_text())
    else:
        out = run(["-m", "evaluation.evaluate", "--policy", "mlp", "--split", "validation", "--workers", str(workers),
                   "--artifact", str(model.relative_to(TASK_DIR))], out_dir / "logs" / f"r{k}-s{seed}-validate.log")
        res = last_json(out)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(res, indent=1))
    n = res["n"]
    entry = {"row": ROW, "round": k, "seed": seed, "run": model.parent.name,
             "sustained_hold": f"{round(res['sustained_hold_success'] * n)}/{n}", "reach": f"{round(res['reach_success'] * n)}/{n}",
             "median_final_err_mm": round(res["median_final_err_mm"], 3), "model": str(model.relative_to(TASK_DIR))}
    summary = out_dir / "summary.csv"
    rows = list(csv.DictReader(open(summary))) if summary.exists() else []
    rows = [r for r in rows if not (r["round"] == str(k) and r["seed"] == str(seed))] + [entry]
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(entry))
        w.writeheader()
        w.writerows(rows)
    print(f"  {ROW} r{k} s{seed}: validation hold {entry['sustained_hold']}, reach {entry['reach']}, "
          f"median final error {entry['median_final_err_mm']} mm", flush=True)
    return entry


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--draws", type=int, default=2, help="noisy draws per fitting scenario per iteration (2 = R4's 442)")
    p.add_argument("--extra-seeds", type=int, nargs="*", default=[])
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--scale", type=int, default=1, help="alpha = scale T tr(Sigma_hat); 1 is the paper's MuJoCo choice")
    p.add_argument("--start-iteration", type=int, default=1,
                   help="continue from iteration k: the seed-0 model and covariance of iteration k-1 must exist")
    args = p.parse_args(argv)
    out_dir = TASK_DIR / f"results/ablation/{args.date}__dart-full"
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    tag = f"a{args.scale}" if args.scale != 1 else ""
    nm = names(tag)
    timing_path = out_dir / "timing.csv"
    timing = list(csv.DictReader(open(timing_path))) if timing_path.exists() else []  # append, never overwrite
    pre = f"{tag}-" if tag else ""  # log-file prefix
    student, cov = START, "zero"
    if args.start_iteration > 1:
        prev = args.start_iteration - 1
        model = model_path(prev, 0, args.date, tag)
        cov = str(out_dir / "covariance" / nm["cov"](prev))
        if not model.exists() or not Path(cov).exists():
            raise SystemExit(f"iteration {prev} of {nm['row']} is incomplete: need {model} and {cov}")
        student = str(model.relative_to(TASK_DIR))
    for k in range(args.start_iteration, args.iterations + 1):
        rk = nm["round"](k)
        print(f"\n===== {nm['row']} iteration {k} ({time.strftime('%H:%M:%S')}), learner {Path(student).parent.name} =====", flush=True)
        held = timed(timing, out_dir, rk, "collect_heldout", lambda: collect(cov, "heldout", k - 1, 1, args.workers,
                                                                              out_dir / "logs" / f"{pre}k{k - 1}-heldout.log"))
        cov_path = out_dir / "covariance" / nm["cov"](k)
        timed(timing, out_dir, rk, "estimate_noise", lambda: estimate(student, held, cov_path, args.scale))
        cov = str(cov_path)
        root = timed(timing, out_dir, rk, "collect_noisy", lambda: collect(cov, "fitting", k, args.draws, args.workers,
                                                                           out_dir / "logs" / f"{pre}k{k}-collect.log"))
        timed(timing, out_dir, rk, "build", lambda: build(root, nm["dataset"](k), args.workers, out_dir / "logs" / f"{pre}k{k}-build.log"))
        model = timed(timing, out_dir, rk, "train", lambda: train(k, 0, args.date, out_dir / "logs" / f"{pre}r{k}-s0-train.log", tag))
        timed(timing, out_dir, rk, "validate", lambda: validate(model, rk, 0, out_dir, args.workers))
        student = str(model.relative_to(TASK_DIR))
    for k in range(args.start_iteration, args.iterations + 1):
        for seed in args.extra_seeds:
            print(f"\n===== extra seed {seed} for iteration {k} ({time.strftime('%H:%M:%S')}) =====", flush=True)
            model = train(k, seed, args.date, out_dir / "logs" / f"{pre}r{k}-s{seed}-train.log", tag)
            validate(model, nm["round"](k), seed, out_dir, args.workers)
    print((out_dir / "summary.csv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
