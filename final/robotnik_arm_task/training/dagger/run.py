"""Execute the approved frozen D8 A/R1 only; all evaluations are pure student.

python -m training.dagger.run --stage smoke
python -m training.dagger.run --stage all --workers 8
"""
import os
if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_key] = "1"
import argparse
import copy
import functools
import json
from pathlib import Path
import resource
import time

import numpy as np

from data_collection.dagger import DEFAULT_OUTPUT, load_contract, parent_config
from data_collection.dagger_preflight import ROOT, PARENT, EXPECTED, sha
from data_collection.dagger_scenarios import freeze_json
from training.dagger.config import finalise, validate

R0_RESULTS = ROOT / "results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256"


def score(summary):
    return (-summary["sustained_hold_success"], -summary["reach_success"], summary["median_final_err_mm"])


def choose_candidate(rows):
    if not rows:
        raise ValueError("no screened candidates")
    return min(rows, key=lambda r: (*score(r["summary"]), r["step"]))


def promotes(candidate, parent):
    return score(candidate) < score(parent)


def recipe(output, protocol, parent, manifest_sha, source_sha):
    from training.common.engine import code_digest
    cfg = copy.deepcopy(parent)
    cfg["identity"]["regime"] = "dagger"
    cfg["training"].update(protocol["training"])
    cfg["loss"].update(protocol["loss"])
    cfg["outputs"].update(checkpoint_root="checkpoints", results_root=str(output), results_index=str(output/"results.csv"))
    cfg["dagger"] = {"round": 1, "parent_sha256": EXPECTED["best.pt"],
                      "parent_normaliser_sha256": protocol["normaliser_content_sha256"],
                      "round_manifest_sha256": manifest_sha, "protocol_sha256": sha(output/"protocol.json")}
    cfg["provenance"].update(dataset_kind="dagger aggregate D0 fitting221 plus R1 mixed64", code_digest=code_digest(),
                             clean_fitting_ids=parent["provenance"]["fitting_ids"],
                             parent_checkpoint_sha256=EXPECTED["best.pt"], dagger_protocol_sha256=sha(output/"protocol.json"),
                             round_manifest_sha256=manifest_sha)
    cfg["provenance"]["source_sha256"].update({f"D1/{k}": v for k,v in source_sha.items()})
    return finalise(cfg)


def prepare(output):
    from training.dagger.dataset import original_data, round_data, combine
    from experiments.capacity import probe_indices
    protocol, parent = load_contract(output), parent_config()
    # The teacher-only reference is evidence, never an aggregate source.
    reference = json.loads((output/"reference_collection.json").read_text())
    if sha(ROOT/reference["directory"]/"round_manifest.json") != reference["manifest_sha256"]:
        raise ValueError("expert reference manifest changed")
    d0, val, norm = original_data(parent)
    d1, stats, digest, sources = round_data(output, protocol, parent)
    cfg = recipe(output, protocol, parent, digest, sources)
    aggregate = combine([d0, d1])
    cfg["provenance"]["fitting_ids"] = aggregate.episode_ids
    cfg = finalise(cfg)
    binding = json.loads((R0_RESULTS/"probe_indices.json").read_text())
    for key,cache in (("fitting",d0),("validation",val)):
        actual = {"episode_ids": cache.episode_ids, "indices": [p.tolist() for p in probe_indices(cache)]}
        if actual != binding["probes"][key]:
            raise ValueError("fixed offline probes differ from R0")
    report = {"original_episodes": len(d0.episode_ids), "new_episodes": len(d1.episode_ids),
              "original_rows": len(d0.x), "new_rows": len(d1.x), "aggregate_rows": len(aggregate.x),
              "new_episode_sampling_mass": len(d1.episode_ids)/len(aggregate.episode_ids),
              "new_row_fraction": len(d1.x)/len(aggregate.x),
              "float32_gpu_data_bytes": int(len(aggregate.x)*(21+9)*4),
              "normaliser_sha256": norm.sha256(), "episodes": stats,
              "reference_manifest_sha256": reference["manifest_sha256"], "round_manifest_sha256": digest}
    freeze_json(output/"data_inventory.json", report)
    freeze_json(output/"probe_indices.json", binding)
    freeze_json(output/"run_config.json", cfg)
    return {"config": cfg, "d0": d0, "d1": d1, "train": aggregate, "validation": val, "normaliser": norm}


def smoke(output, device="cuda"):
    import torch
    from training.common.config import configure_runtime, run_dir_for
    from training.common.dataset import EpisodeCache
    from training.common.engine import fit
    from training.baseline.train import objective_from
    from training.dagger.train import warm_start
    from training.dagger.dataset import original_data
    if (output/"training_smoke.json").exists():
        return json.loads((output/"training_smoke.json").read_text())
    torch.set_num_threads(1)
    runtime = configure_runtime(device)
    protocol, parent = load_contract(output), parent_config()
    d0, _, norm = original_data(parent)
    small = EpisodeCache(d0.x[:64], d0.y[:64], np.array([0,64]), ["smoke-only-original-prefix"])
    cfg = recipe(output, protocol, parent, "0"*64, {})
    cfg["dagger"]["smoke_only"] = True
    cfg["training"].update(updates=4, warmup_updates=2, log_every=2, checkpoint_every=2, batch_size=32, device=device)
    cfg["outputs"]["checkpoint_root"] = str(output/"smoke_full")
    cfg = finalise(cfg)
    full = warm_start(cfg,ROOT/PARENT/"best.pt",norm)
    fit(cfg,small,None,full,norm,objective_from(cfg),validator=validate)
    cfg["outputs"]["checkpoint_root"] = str(output/"smoke_resumed")
    part = warm_start(cfg,ROOT/PARENT/"best.pt",norm)
    def stop(step, *_):
        if step == 2:
            raise InterruptedError("intentional smoke interruption after durable checkpoint")
    try:
        fit(cfg,small,None,part,norm,objective_from(cfg),on_checkpoint=stop,validator=validate)
    except InterruptedError:
        pass
    resumed = warm_start(cfg,ROOT/PARENT/"best.pt",norm)
    fit(cfg,small,None,resumed,norm,objective_from(cfg),resume=run_dir_for(cfg)/"latest.pt",validator=validate)
    diff = max(float((v-resumed.state_dict()[k]).abs().max()) for k,v in full.state_dict().items())
    state = torch.load(run_dir_for(cfg)/"latest.pt",map_location="cpu",weights_only=True)
    steps = sorted({int(v["step"]) for v in state["optimizer_state"]["state"].values()})
    if diff != 0 or steps != [4]:
        raise ValueError("GPU warm-start/resume smoke mismatch")
    report = {"passed": True,"runtime": runtime,"resume_max_abs_weight_difference":diff,
              "new_optimizer_steps":steps,"normaliser_sha256":norm.sha256(),"protocol_sha256":sha(output/"protocol.json")}
    freeze_json(output/"training_smoke.json",report)
    print(json.dumps({"training_smoke":report}),flush=True)
    return report


def evaluate_artifact(output, artifact, config, split, step, workers):
    from evaluation.evaluate import evaluate_policy, mlp_factory, summarise, write_details
    from evaluation.test_sets import get_split, scenario_from_row
    import csv
    name = output/f"rollouts__{split}__step{step}.json"
    if name.exists():
        record = json.loads(name.read_text())
        if record["inference_sha256"] != sha(artifact):
            raise ValueError("saved evaluation belongs to another artifact")
        return record
    if split == "fitting_diagnostic":
        ids = json.loads((output/"probe_indices.json").read_text())["fitting_diagnostic_ids"]
        with (ROOT/config["data"]["primary_root"]/"scenarios_train.csv").open() as f:
            rows = {r["scenario_id"]:r for r in csv.DictReader(f)}
        scenarios = [scenario_from_row(rows[i],"train") for i in ids]
    elif split in ("screening","validation"):
        scenarios = get_split(split,ROOT/config["data"]["split_manifest"],"A")
    else:
        raise ValueError("R1 development never evaluates test scenes")
    factory = functools.partial(mlp_factory,str(artifact),"numpy",{"teacher":"A","run_id":config["identity"]["run_id"]})
    start = time.monotonic()
    outcomes = evaluate_policy(factory,scenarios,config,workers=workers)
    record = {"step":step,"split":split,"inference_sha256":sha(artifact),"policy":"pure student; no expert queried",
              "evaluation_code_sha256":sha(ROOT/"evaluation/evaluate.py"),"summary":summarise(outcomes),
              "outcomes":outcomes,"elapsed_s":time.monotonic()-start}
    write_details(name.with_suffix(".csv"),outcomes)
    freeze_json(name,record)
    print(json.dumps({"evaluation":split,"step":step,"summary":record["summary"]}),flush=True)
    return record


def train(output, workers, resume=False):
    import torch
    from training.common.config import configure_runtime, run_dir_for
    from training.common.engine import fit
    from training.baseline.train import objective_from
    from training.dagger.train import warm_start
    from experiments.capacity import probe_metrics
    torch.set_num_threads(1)
    runtime = configure_runtime("cuda")
    if not json.loads((output/"training_smoke.json").read_text())["passed"]:
        raise ValueError("training/resume smoke is required")
    prep = prepare(output)
    cfg,norm = prep["config"],prep["normaliser"]
    directory = run_dir_for(cfg)
    model = warm_start(cfg,ROOT/PARENT/"best.pt",norm)
    metrics = output/"probe_metrics.jsonl"
    recorded = {json.loads(line)["update"] for line in metrics.read_text().splitlines()} if metrics.exists() else set()
    def probe(step, model):
        if step in recorded:
            return
        row = {"update":step, **{name:probe_metrics(model,prep[key],norm,.8,.2)
              for name,key in (("fitting","d0"),("validation","validation"),("dagger_r1","d1"))}}
        with metrics.open("a") as f:
            f.write(json.dumps(row)+"\n")
        recorded.add(step)
        print(json.dumps({"probe_update":step,"fitting_huber":row["fitting"]["overall"]["action_huber"],
                          "validation_huber":row["validation"]["overall"]["action_huber"]}),flush=True)
    def checkpoint(step,ckpt,export,model):
        probe(step,model)
        evaluate_artifact(output,export,cfg,"screening",step,workers)
    if not resume:
        probe(0,model)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = fit(cfg,prep["train"],prep["validation"],model,norm,objective_from(cfg),on_checkpoint=checkpoint,
                 resume=directory/"latest.pt" if resume else None,validator=validate)
    report = {"run_id":cfg["identity"]["run_id"],"run_dir":str(directory.relative_to(ROOT)),
              "terminal_update":result.terminal_update,"runtime":runtime,"wall_s_including_screening":time.monotonic()-start,
              "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"cpu_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    freeze_json(output/"training_complete.json",report)
    print(json.dumps({"training_complete":report}),flush=True)
    return cfg


def baseline(output):
    from evaluation.evaluate import summarise
    # The original step export and promoted export carry different provenance metadata but identical numerical arrays.
    with np.load(ROOT/PARENT/"inference.npz") as a, np.load(ROOT/PARENT/"step_84000.npz") as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files if k != "metadata_json"):
            raise ValueError("R0 evaluation export differs numerically from promoted parent")
    ids = set(json.loads((output/"probe_indices.json").read_text())["fitting_diagnostic_ids"])
    records = {split:json.loads((R0_RESULTS/f"rollouts__{split}__step84000.json").read_text()) for split in ("fitting","validation")}
    for record in records.values():
        if record["inference_sha256"] != sha(ROOT/PARENT/"step_84000.npz") or record["evaluation_script_sha256"] != sha(ROOT/"evaluation/evaluate.py"):
            raise ValueError("R0 evaluation evidence mismatch")
    return {"validation":records["validation"]["summary"],
            "fitting_diagnostic":summarise([o for o in records["fitting"]["outcomes"] if o["scenario_id"] in ids]),
            "reuse_evidence":{str(p.relative_to(ROOT)):sha(p) for p in [R0_RESULTS/"rollouts__fitting__step84000.json",R0_RESULTS/"rollouts__validation__step84000.json"]}}


def evaluate(output, workers):
    from training.common.config import run_dir_for
    from training.common.engine import select_checkpoint
    cfg = json.loads((output/"run_config.json").read_text())
    validate(cfg,True)
    directory = run_dir_for(cfg)
    screened = [evaluate_artifact(output,directory/f"step_{step}.npz",cfg,"screening",step,workers)
                for step in (5000,10000,15000,20000)]
    chosen = choose_candidate(screened)
    step = chosen["step"]
    val = evaluate_artifact(output,directory/f"step_{step}.npz",cfg,"validation",step,workers)
    fitting = evaluate_artifact(output,directory/f"step_{step}.npz",cfg,"fitting_diagnostic",step,workers)
    prior = baseline(output)
    promoted = promotes(val["summary"],prior["validation"])
    evidence = {"screening_steps":[r["step"] for r in screened],"candidate_step":step,
                "validation":val["summary"],"parent_validation":prior["validation"],"promoted":promoted,
                "criterion":"hold, reach, median final error; exact tie keeps parent; no fitting/test selection"}
    if not (directory/"selection.json").exists():
        select_checkpoint(directory,step,evidence)
    report = {"run_id":cfg["identity"]["run_id"],"candidate_step":step,"promoted":promoted,"R0":prior,
              "R1":{"validation":val["summary"],"fitting_diagnostic":fitting["summary"]},
              "candidate_checkpoint":str((directory/"best.pt").relative_to(ROOT)),
              "candidate_checkpoint_sha256":sha(directory/"best.pt"),
              "candidate_inference":str((directory/"inference.npz").relative_to(ROOT)),
              "candidate_inference_sha256":sha(directory/"inference.npz"),
              "collection_parent_next":str((directory/"inference.npz" if promoted else ROOT/PARENT/"inference.npz").relative_to(ROOT)),
              "screening":[{"step":r["step"],"summary":r["summary"]} for r in screened],
              "new_test_evaluated":False,"controls_completed":False}
    freeze_json(output/"evaluation_complete.json",report)
    print(json.dumps({"evaluation_complete":report}),flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","smoke","train","evaluate","all"),default="all")
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "smoke":
        smoke(args.output)
    elif args.stage == "prepare":
        p = prepare(args.output)
        print(json.dumps({"run_id":p["config"]["identity"]["run_id"],"rows":len(p["train"].x)}),flush=True)
    elif args.stage == "evaluate":
        evaluate(args.output,args.workers)
    else:
        if args.stage == "all":
            smoke(args.output)
        train(args.output,args.workers,args.resume)
        if args.stage == "all":
            evaluate(args.output,args.workers)


if __name__ == "__main__":
    main()
