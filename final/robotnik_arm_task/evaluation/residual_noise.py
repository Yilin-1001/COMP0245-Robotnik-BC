"""Harness check (c), a separately labelled diagnostic: the frozen expert under noise matched to a fitted
student's validation residuals (prediction minus expert label), per channel, i.i.d. per 1 ms step.

Matching a residual marginal does not reproduce the student's state-dependent errors, so a different closed-loop
outcome is diagnostic evidence, not a harness failure. Outputs stay local under results/diagnostics/.

Usage (from final/robotnik_arm_task/):
    python -m evaluation.residual_noise --run-dir checkpoints/clean/A/bc/<run_id> --step <k> --split validation
"""
import os

if __name__ == "__main__":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"

import argparse
import functools
import json
from pathlib import Path

import numpy as np

from core.interfaces import Action, Policy

TASK_DIR = Path(__file__).resolve().parents[1]


class NoisyExpert(Policy):
    """Expert action plus N(mean, std) noise per channel (7 torques, right, left wheel; wheels mirrored)."""

    def __init__(self, expert: Policy, stats: dict, seed: int):
        self.expert, self.mean, self.std, self.seed = expert, np.asarray(stats["mean"]), np.asarray(stats["std"]), seed
        self.episode = 0
        self.rng = None

    def bind_env(self, env):
        if hasattr(self.expert, "bind_env"):
            self.expert.bind_env(env)

    def reset(self):
        self.expert.reset()
        self.rng = np.random.default_rng([self.seed, self.episode])
        self.episode += 1

    def act(self, obs):
        a = self.expert.act(obs)
        n = self.mean + self.std * self.rng.standard_normal(9)
        r, l = n[7], n[8]
        return Action(np.asarray(a.torque_arm) + n[:7], np.asarray(a.wheel_cmds) + np.array([r, l, l, r]))


def residual_stats(run_dir: Path, step: int) -> dict:
    """Physical-unit residual mean/std per trained channel on the run's validation cache."""
    import yaml
    from evaluation.offline import _predict
    from training.common.checkpoint import load_checkpoint
    from training.common.dataset import load_cache, load_episode_refs, load_split, select_refs
    from training.common.models import build_model
    from training.common.normalisation import Normaliser
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    n = Normaliser.load(run_dir / "normaliser.npz")
    split = load_split(TASK_DIR / cfg["data"]["split_manifest"])
    refs = select_refs(load_episode_refs(TASK_DIR / cfg["data"]["primary_root"], cfg["identity"]["teacher"], "train"),
                       split["validation_ids"])
    val = load_cache(TASK_DIR / cfg["data"]["cache_root"] / cfg["identity"]["teacher"] /
                     cfg["provenance"]["cache_ids"]["validation"], refs)
    model = build_model(cfg)
    model.load_state_dict(load_checkpoint(run_dir / f"step_{step}.pt")["model_state"])
    err = n.inverse_y(_predict(model, n.transform_x(val.x.astype(np.float64)))) - val.y
    return {"mean": err.mean(axis=0).tolist(), "std": err.std(axis=0).tolist(), "rows": int(len(err)),
            "run_id": cfg["identity"]["run_id"], "step": step, "teacher": cfg["identity"]["teacher"]}


def noisy_expert_factory(teacher: str, stats: dict, seed: int):
    from evaluation.evaluate import expert_factory
    return NoisyExpert(expert_factory(teacher), stats, seed)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="expert under student-residual-matched action noise (diagnostic)")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--split", choices=["screening", "validation"], default="validation")
    p.add_argument("--seed", type=int, default=870100)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)
    from evaluation.evaluate import evaluate_policy, summarise, write_details
    from evaluation.test_sets import get_split
    from training.common.config import resolve_config
    run_dir = args.run_dir if args.run_dir.is_absolute() else TASK_DIR / args.run_dir
    stats = residual_stats(run_dir, args.step)
    cfg = resolve_config(TASK_DIR / "experiments/configs/baseline.yaml", stats["teacher"], {})
    scenarios = get_split(args.split, TASK_DIR / "experiments/splits/clean253.json", stats["teacher"])
    outcomes = evaluate_policy(functools.partial(noisy_expert_factory, stats["teacher"], stats, args.seed), scenarios,
                               cfg, workers=args.workers)
    out = TASK_DIR / "results/diagnostics" / f"expert_residual_noise_{stats['run_id']}_step{args.step}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    write_details(out / "rollouts.csv", outcomes)
    summary = summarise(outcomes)
    (out / "summary.json").write_text(json.dumps({"stats": stats, "seed": args.seed, "summary": summary}, indent=1))
    print(json.dumps({"residual_mean": np.round(stats["mean"], 4).tolist(), "residual_std": np.round(stats["std"], 4).tolist(),
                      **{k: summary[k] for k in ("n", "sustained_hold_success", "reach_success", "median_final_err_mm",
                                                 "max_final1s_err_mm", "failures")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
