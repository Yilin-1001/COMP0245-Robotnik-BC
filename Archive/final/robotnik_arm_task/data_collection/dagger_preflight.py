"""Gated, fitting-only recovery probes for the frozen D8 DAgger parent.

This is diagnostic execution, not DAgger data. It uses the unchanged task evaluator.
Run with ``python -m data_collection.dagger_preflight --workers 8``.
"""
import os

if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_key] = "1"

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import yaml

from core.interfaces import Policy

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path("checkpoints/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/clean/A/bc/"
              "clean-A-bc-clean253-d08w0256-lr3e-4-s0-selection-ca837b8fa585")
EXPECTED = {"best.pt": "a439ce195692874c6a712c05d2d0de6a13c22e8061ac5af2783805e5034dcd35",
            "inference.npz": "2300123d9cd98e74f1db7db554789226f72e91fa8b8b377ff1166abd4e5520ad",
            "normaliser.npz": "d69994cf4ac00b84213113aabc6ab51560df3ec6bcd430795c7cfc8357f1ea53"}
MODES = {"expert": ("expert", 0.), "reach_0.25s": ("reach", .25), "reach_1s": ("reach", 1.),
         "reach_3s": ("reach", 3.), "reset_1s": ("reset", 1.), "reset_3s": ("reset", 3.)}
WORKER = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecoveryPolicy(Policy):
    """Whole-controller intervention; both policies recompute on each current observation."""
    def __init__(self, student, expert, trigger, duration):
        if trigger not in ("expert", "reach", "reset") or duration < 0:
            raise ValueError("invalid recovery probe")
        self.student, self.expert = student, expert
        self.trigger, self.duration = trigger, duration
        self.meta = getattr(student, "meta", {})

    def bind_env(self, env):
        self.env = env

    def reset(self):
        self.student.reset()
        self.expert.reset()
        self.handoff_s = self.recovery_s = None
        self.trace = []
        self.k = 0

    def act(self, obs):
        student, expert = self.student.act(obs), self.expert.act(obs)
        active = bool(self.expert.expert.base_command(obs)[1])
        t = float(self.env.t)
        if self.handoff_s is None and (self.trigger == "reset" or active):
            self.handoff_s = t
        use_student = self.trigger != "expert" and self.handoff_s is not None and t - self.handoff_s < self.duration - 1e-10
        if self.trigger != "expert" and self.handoff_s is not None and not use_student and self.recovery_s is None:
            self.recovery_s = t
        action = student if use_student else expert
        if self.k % 100 == 0:
            self.trace.append({"t": t, "ee_err_m": float(np.linalg.norm(obs.target_ee_pos - self.env.ee_pos())),
                               "teacher_reach_active": active, "student_executes": use_student,
                               "base_pos": np.asarray(obs.base_pos).tolist(),
                               "qd_norm": float(np.linalg.norm(obs.qd_mes))})
        self.k += 1
        self.last_raw = None
        return action


def recovery_gate(outcomes):
    required = {"expert": 8, "reach_0.25s": 6, "reach_1s": 6}
    checks = {}
    for mode, threshold in required.items():
        if mode not in outcomes or len(outcomes[mode]) != 8:
            raise ValueError(f"gate needs exactly eight outcomes for {mode}")
        count = sum(bool(o["sustained_hold_pass"]) and o["failure"] is None for o in outcomes[mode])
        checks[mode] = {"held": count, "required": threshold, "passed": count >= threshold}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "rule": "Expert 8/8, reaching perturbations .25s and 1s each >=6/8; others diagnostic only."}


def _init(artifact):
    from evaluation.evaluate import mlp_factory, expert_factory
    from threadpoolctl import threadpool_limits
    WORKER["limits"] = threadpool_limits(1)
    WORKER["student"] = mlp_factory(artifact)
    WORKER["expert"] = expert_factory("A")


def _one(job):
    from evaluation.evaluate import rollout, make_robot_env
    name, scenario, cfg = job
    trigger, duration = MODES[name]
    policy = RecoveryPolicy(WORKER["student"], WORKER["expert"], trigger, duration)
    outcome = rollout(policy, scenario, cfg, make_robot_env)
    outcome.update(probe=name, handoff_s=policy.handoff_s, recovery_s=policy.recovery_s, trace=policy.trace)
    return outcome


def main():
    from evaluation.evaluate import summarise
    from evaluation.test_sets import scenario_from_row
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dagger/2026-10-27__preflight__d08w0256-a")
    args = parser.parse_args()
    parent = ROOT / PARENT
    for name, expected in EXPECTED.items():
        if sha(parent / name) != expected:
            raise ValueError(f"parent artifact changed: {name}")
    cfg = yaml.safe_load((parent / "config.yaml").read_text())
    ids = sorted(cfg["provenance"]["fitting_ids"])
    ids = [ids[i] for i in np.linspace(0, len(ids)-1, 8, dtype=int)]
    if set(ids) & set(cfg["provenance"]["validation_ids"]):
        raise ValueError("fitting probe overlaps validation")
    source = ROOT / cfg["data"]["primary_root"] / "scenarios_train.csv"
    with source.open() as f:
        rows = {r["scenario_id"]: r for r in csv.DictReader(f)}
    scenarios = [scenario_from_row(rows[i], "train") for i in ids]
    provenance = {"parent": str(PARENT), "parent_hashes": EXPECTED, "teacher": "markovian-lt0.50-v1",
                  "scenario_ids": ids, "scenario_source_sha256": sha(source),
                  "script_sha256": sha(__file__), "evaluator_sha256": sha(ROOT / "evaluation/evaluate.py"),
                  "expert_sha256": sha(ROOT / "data_collection/expert.py"),
                  "modes": MODES, "evaluation": cfg["evaluation"], "interface": cfg["interface"]}
    args.output.mkdir(parents=True, exist_ok=True)
    contract = args.output / "contract.json"
    encoded = json.dumps(provenance, indent=2, sort_keys=True)
    if contract.exists() and contract.read_text() != encoded:
        raise ValueError("immutable preflight contract differs")
    if not contract.exists():
        contract.write_text(encoded)
    results_file = args.output / "outcomes.jsonl"
    outcomes = [json.loads(line) for line in results_file.read_text().splitlines()] if results_file.exists() else []
    done = {(o["probe"], o["scenario_id"]) for o in outcomes}
    expected_keys = {(name, s.scenario_id) for name in MODES for s in scenarios}
    if len(done) != len(outcomes) or not done <= expected_keys:
        raise ValueError("duplicate or unknown saved preflight result")
    jobs = [(name, s, cfg) for name in MODES for s in scenarios if (name, s.scenario_id) not in done]
    start = time.monotonic()
    from training.common.config import pin_rollout_threads
    pin_rollout_threads()
    with mp.get_context("spawn").Pool(max(1, min(args.workers, len(jobs) or 1)), _init,
                                     (str(parent / "inference.npz"),)) as pool, results_file.open("a") as stream:
        for o in pool.imap_unordered(_one, jobs, chunksize=1):
            stream.write(json.dumps(o) + "\n")
            stream.flush()
            outcomes.append(o)
            print(json.dumps({"completed": len(outcomes), "total": 48, "probe": o["probe"],
                              "scenario": o["scenario_id"], "hold": o["sustained_hold_pass"],
                              "elapsed_s": round(time.monotonic()-start, 1)}), flush=True)
    grouped = {mode: [o for o in outcomes if o["probe"] == mode] for mode in MODES}
    report = {"contract_sha256": sha(contract), "outcomes_sha256": sha(results_file),
              "gate": recovery_gate(grouped),
              "summaries": {k: dict(summarise(v), held=sum(o["sustained_hold_pass"] for o in v),
                                    reached=sum(o["ever_reached"] for o in v)) for k, v in grouped.items()}}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
