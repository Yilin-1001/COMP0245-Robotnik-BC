"""Frozen A/R1 collection: freeze, smoke, collect --kind reference/mixed.

Independent test geometry is frozen here but never rolled out by this entry point.
"""
import os
if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_key] = "1"
import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import yaml

from data_collection.dagger_preflight import ROOT, PARENT, EXPECTED, sha, recovery_gate
from data_collection.dagger_scenarios import balanced_scenes, assert_disjoint, freeze_json
from data_collection.dagger_rollout import collect_episode, save_episode
from data_collection.scenarios import read_manifest, to_scenario

DEFAULT_OUTPUT = ROOT / "results/dagger/2026-10-27__experiment__d08w0256-a-r1"
PREFLIGHT = ROOT / "results/dagger/2026-10-27__preflight__d08w0256-a"
WORKER = {}


def collection_code():
    paths = {p for glob in ("data_collection/dagger*.py", "data_collection/expert.py", "core/**/*.py",
                            "controllers/**/*.py", "training/common/policies.py", "settings.yaml", "configs/robotnik_arm.json")
             for p in ROOT.glob(glob) if "__pycache__" not in p.parts}
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def parent_config():
    for name, digest in EXPECTED.items():
        if sha(ROOT / PARENT / name) != digest:
            raise ValueError(f"R0 changed: {name}")
    return yaml.safe_load((ROOT / PARENT / "config.yaml").read_text())


def freeze(output):
    output = Path(output)
    cfg = parent_config()
    pre = json.loads((PREFLIGHT / "summary.json").read_text())
    if sha(PREFLIGHT / "outcomes.jsonl") != pre["outcomes_sha256"] or sha(PREFLIGHT / "contract.json") != pre["contract_sha256"]:
        raise ValueError("preflight evidence hash mismatch")
    outcomes = [json.loads(line) for line in (PREFLIGHT / "outcomes.jsonl").read_text().splitlines()]
    gate = recovery_gate({k: [o for o in outcomes if o["probe"] == k] for k in ("expert", "reach_0.25s", "reach_1s")})
    if not gate["passed"]:
        raise ValueError("D8 recovery gate failed; R1 cannot proceed")
    train = balanced_scenes(2026102701, 8, "dagger_train")
    test = balanced_scenes(2026102703, 2, "dagger_test")
    primary = ROOT / cfg["data"]["primary_root"]
    old_files = [primary / f"scenarios_{split}.csv" for split in ("train", "test")]
    old = [r for p in old_files for r in read_manifest(p)]
    assert_disjoint(train + test, old)
    scene_sha = freeze_json(output / "scenes.json", {"train": train, "sealed_test": test,
                    "seeds": {"train": 2026102701, "test": 2026102703}, "pool_size": 4096,
                    "overlap_tolerance": 1e-9, "old_geometry_sources": {str(p.relative_to(ROOT)): sha(p) for p in old_files}})
    protocol = {"version": "dagger-a-r1-v1", "user_approved_date": "2026-10-27", "round": 1,
                "parent": str(PARENT), "parent_hashes": EXPECTED, "teacher_version": "markovian-lt0.50-v1",
                "teacher_lead_time": .50, "normaliser_content_sha256": cfg["provenance"]["normaliser_sha256"],
                "parent_config_sha256": sha(ROOT / PARENT / "config.yaml"), "scenes_sha256": scene_sha,
                "preflight_summary_sha256": sha(PREFLIGHT / "summary.json"), "recovery_gate": gate,
                "source_code": collection_code(), "beta": .75, "block_steps": 10, "mixture_seed": 2026102702,
                "interface": cfg["interface"], "evaluation": cfg["evaluation"],
                "loss": {"action": "huber", "huber_delta": 1., "arm_weight": .8, "wheel_weight": .2},
                "training": {"optimizer": "adamw", "lr": 1e-4, "weight_decay": 1e-4,
                             "adam_betas": [.9, .999], "adam_eps": 1e-8, "batch_size": 1024,
                             "warmup_updates": 500, "schedule": "cosine", "updates": 20000,
                             "log_every": 2000, "checkpoint_every": 5000, "device": "cuda"},
                "sampling": "uniform episode then uniform valid row; D0=221 + D1=64; exclude reference data",
                "screening_steps": [5000, 10000, 15000, 20000],
                "selection": "screening winner gets validation32; compare parent by hold/reach/median-error; ties retain parent",
                "fitting_diagnostic": "same fixed balanced32 as R0; no selection", "sealed_test_evaluation": False,
                "controls": "C0/C1 deferred; no causal attribution before matched controls"}
    freeze_json(output / "protocol.json", protocol)
    return protocol


def load_contract(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    parent_config()
    if collection_code() != protocol["source_code"] or sha(output / "scenes.json") != protocol["scenes_sha256"]:
        raise ValueError("collection code/scenes changed after freezing")
    return protocol


def smoke(output):
    from core.interfaces import Observation
    from evaluation.evaluate import mlp_factory, expert_factory
    from data_collection.dagger_rollout import action_vector
    output = Path(output)
    load_contract(output)
    cfg = copy.deepcopy(parent_config())
    cfg["evaluation"]["episode_cap_s"] = .020
    row = json.loads((output / "scenes.json").read_text())["train"][0]
    scenario = to_scenario(row)
    checks = []
    for beta in (0, 1):
        expert = expert_factory("A")
        arrays, meta = collect_episode(mlp_factory(str(ROOT/PARENT/"inference.npz")), expert, scenario, cfg,
                                       float(beta), np.random.default_rng(0))
        if len(arrays["expert_action"]) != 20 or not arrays["valid_supervision"].all() or meta["outcome"]["failure"]:
            raise ValueError(f"real simulator smoke failed: {meta}")
        errors = []
        for k in (0, 1, 9, 19):
            obs = Observation(arrays["obs_base_pos"][k], arrays["obs_base_yaw"][k], arrays["obs_q_mes"][k],
                              arrays["obs_qd_mes"][k], scenario.target_ee_pos)
            errors.append(float(np.max(np.abs(action_vector(expert.act(obs))-arrays["expert_action"][k]))))
        if max(errors) > 1e-10 or not np.all(arrays["expert_executes"] == bool(beta)):
            raise ValueError("current-state expert label/beta endpoint mismatch")
        path = output / f"smoke_beta{beta}.npz"
        if not path.exists():
            save_episode(path, arrays, meta)
        checks.append({"beta": beta, "max_label_recompute_error": max(errors), "steps": 20})
    report = {"passed": True, "checks": checks, "protocol_sha256": sha(output / "protocol.json")}
    freeze_json(output / "smoke.json", report)
    return report


def _init_worker(artifact, cfg, contract, destination):
    from evaluation.evaluate import mlp_factory, expert_factory
    from threadpoolctl import threadpool_limits
    WORKER.update(limits=threadpool_limits(1), student=mlp_factory(artifact), expert=expert_factory("A"),
                  config=cfg, contract=contract, destination=Path(destination))


def _entry(path, row, contract):
    with np.load(path) as z:
        meta = json.loads(str(z["metadata_json"]))
        if meta["contract_sha256"] != contract["contract_sha256"] or meta["scenario"] != row:
            raise ValueError(f"foreign completed episode: {path}")
        return {"file": path.name, "sha256": sha(path), "scenario_id": row["scenario_id"],
                "valid_rows": int(z["valid_supervision"].sum()), "steps": len(z["expert_action"]),
                "outcome": meta["outcome"], "expert_execution_fraction": meta["expert_execution_fraction"], "wall_s": meta["wall_s"]}


def _collect(row):
    contract = WORKER["contract"]
    seed = int(contract["mixture_seed"] + row["index"])
    arrays, meta = collect_episode(WORKER["student"], WORKER["expert"], to_scenario(row), WORKER["config"],
                                   contract["beta"], np.random.default_rng(seed))
    meta.update(contract_sha256=contract["contract_sha256"], scenario=row, scenario_id=row["scenario_id"],
                split="dagger_train", round=1, teacher_version="markovian-lt0.50-v1", mixture_seed=seed,
                source=contract["source"], normaliser_sha256=contract["normaliser_content_sha256"],
                policy_sha256=EXPECTED["inference.npz"])
    path = WORKER["destination"] / f"{row['scenario_id']}.npz"
    save_episode(path, arrays, meta)
    return _entry(path, row, contract)


def collect(output, kind, workers=8):
    output = Path(output)
    protocol = load_contract(output)
    if not json.loads((output / "smoke.json").read_text())["passed"]:
        raise ValueError("real simulator smoke is required")
    if kind not in ("reference", "mixed"):
        raise ValueError("only R1 mixed and its expert reference are authorized")
    protocol_sha = sha(output / "protocol.json")
    dest = ROOT / "data/Markovian/DAgger" / ("2026-10-27__collection__a-r1-" + protocol_sha[:12]) / kind / "round_001"
    dest.mkdir(parents=True, exist_ok=True)
    contract = {**protocol, "protocol_sha256": protocol_sha, "kind": kind,
                "beta": 1. if kind == "reference" else protocol["beta"],
                "source": "dagger_expert_reference" if kind == "reference" else "dagger_mixed"}
    contract["contract_sha256"] = freeze_json(dest / "collection_contract.json", contract)
    rows = [r for r in json.loads((output / "scenes.json").read_text())["train"] if r["round"] == 1]
    entries, jobs = [], []
    for row in rows:
        path = dest / f"{row['scenario_id']}.npz"
        if path.exists():
            entries.append(_entry(path, row, contract))
        else:
            jobs.append(row)
    from training.common.config import pin_rollout_threads
    pin_rollout_threads()
    start = time.monotonic()
    with mp.get_context("spawn").Pool(max(1, min(workers, len(jobs) or 1)), _init_worker,
             (str(ROOT/PARENT/"inference.npz"), parent_config(), contract, str(dest))) as pool:
        for entry in pool.imap_unordered(_collect, jobs, chunksize=1):
            entries.append(entry)
            print(json.dumps({"kind": kind, "completed": len(entries), "total": 64, "scenario": entry["scenario_id"],
                              "hold": entry["outcome"]["sustained_hold_pass"], "elapsed_s": round(time.monotonic()-start, 1)}), flush=True)
    if len(entries) != 64:
        raise ValueError("round is incomplete")
    manifest = {"contract_sha256": contract["contract_sha256"], "source": contract["source"],
                "episodes": sorted(entries, key=lambda e: e["scenario_id"])}
    manifest_sha = freeze_json(dest / "round_manifest.json", manifest)
    freeze_json(output / f"{kind}_collection.json", {"directory": str(dest.relative_to(ROOT)), "manifest_sha256": manifest_sha})
    print(json.dumps({"kind": kind, "directory": str(dest), "valid_rows": sum(e["valid_rows"] for e in entries),
                      "held": sum(e["outcome"]["sustained_hold_pass"] for e in entries), "n": 64}), flush=True)
    return dest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "smoke", "collect"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--kind", choices=("reference", "mixed"), default="mixed")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    if args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "smoke":
        print(json.dumps(smoke(args.output)), flush=True)
    else:
        collect(args.output, args.kind, args.workers)


if __name__ == "__main__":
    main()
