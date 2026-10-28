"""Export fixed-probe learning curves, learning rates and R0/R1 closed-loop comparisons."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_collection.dagger import DEFAULT_OUTPUT
from data_collection.dagger_preflight import ROOT, sha
from data_collection.dagger_scenarios import freeze_json


def curve_series(rows):
    updates = [r["update"] for r in rows]
    if not updates or updates != sorted(set(updates)):
        raise ValueError("fixed-probe updates must be unique and increasing")
    fields = ("action_huber","arm_rmse_nm","wheel_rmse_rad_s")
    series = {name:{key:[r[name]["overall"][key] for r in rows] for key in fields}
              for name in ("fitting","validation")}
    if any(not np.isfinite(v).all() for metrics in series.values() for v in metrics.values()):
        raise ValueError("non-finite learning curve")
    return {"updates":updates,**series}


def sampling_audit(config, offsets, phases, original_rows, expected_rng, device="cuda"):
    import torch
    from training.common.engine import stream_seed
    from training.common.dataset import EpisodeCache, sample_indices
    cache = EpisodeCache(None,None,np.asarray(offsets),[])
    generator = torch.Generator(device=device).manual_seed(stream_seed(config["identity"]["seed"],"batch"))
    phase = torch.as_tensor(phases,dtype=torch.int64,device=device)
    counts = torch.zeros(6,dtype=torch.int64,device=device)
    for _ in range(config["training"]["updates"]):
        idx = sample_indices(cache,config["training"]["batch_size"],generator,device)
        counts += torch.bincount((idx >= original_rows).long()*3+phase[idx],minlength=6)
    if not torch.equal(generator.get_state().cpu(),expected_rng.cpu()):
        raise ValueError("replayed batch RNG differs from the terminal checkpoint")
    counts = counts.reshape(2,3).cpu().numpy()
    return {"source_phase_counts":counts.tolist(),"total_gradient_rows":int(counts.sum()),
            "actual_new_source_fraction":float(counts[1].sum()/counts.sum()),"rng_matches_checkpoint":True,
            "source_order":["original221","dagger64"],"phase_order":["navigation","approach","hold"]}


def audit_sampling(output):
    import torch
    from training.common.config import configure_runtime, run_dir_for
    from training.common.dataset import sha256_file
    cfg = json.loads((output/"run_config.json").read_text())
    configure_runtime("cuda")
    cache = ROOT/cfg["data"]["cache_root"]/"A"/cfg["provenance"]["cache_ids"]["fitting"]
    files = json.loads((cache/"COMPLETE.json").read_text())["files"]
    for name in ("offsets.npy","diagnostics.npz"):
        if sha256_file(cache/name) != files[name]:
            raise ValueError("original cache changed")
    offsets = np.load(cache/"offsets.npy")
    with np.load(cache/"diagnostics.npz") as z:
        phases = [z["phase"]]
    original_rows = int(offsets[-1])
    lengths = np.diff(offsets).tolist()
    record = json.loads((output/"mixed_collection.json").read_text())
    dest = ROOT/record["directory"]
    if sha(dest/"round_manifest.json") != record["manifest_sha256"]:
        raise ValueError("round manifest changed")
    for entry in json.loads((dest/"round_manifest.json").read_text())["episodes"]:
        path = dest/entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("round source changed")
        with np.load(path) as z:
            selected = z["phase"][z["valid_supervision"]]
            phases.append(selected)
            lengths.append(len(selected))
    latest = run_dir_for(cfg)/"latest.pt"
    state = torch.load(latest,map_location="cpu",weights_only=True)
    if state["config"]["identity"]["config_hash"] != cfg["identity"]["config_hash"] or state["step"] != cfg["training"]["updates"]:
        raise ValueError("sampling audit requires the matching terminal checkpoint")
    report = sampling_audit(cfg,np.r_[0,np.cumsum(lengths)],np.concatenate(phases),original_rows,state["rng"]["gen_batch"])
    report.update(latest_sha256=sha(latest),source_sha256=sha(__file__))
    freeze_json(output/"sampling_audit.json",report)
    print(json.dumps({"sampling_audit":report}),flush=True)
    return report


def make_report(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "dagger-a-r1"
    from training.common.engine import lr_factor
    output = Path(output)
    rows = [json.loads(s) for s in (output/"probe_metrics.jsonl").read_text().splitlines()]
    curves = curve_series(rows)
    cfg = json.loads((output/"run_config.json").read_text())
    train = json.loads((output/"training_complete.json").read_text())
    evaluation = json.loads((output/"evaluation_complete.json").read_text())
    inventory = json.loads((output/"data_inventory.json").read_text())
    audit = json.loads((output/"collection_audit.json").read_text())
    with (ROOT/train["run_dir"]/"metrics.csv").open() as f:
        logged = list(csv.DictReader(f))
    out = output/"plots"
    out.mkdir(exist_ok=True)
    colors = {"fitting":"#b68d19","validation":"#35a5dd"}
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for ax,key,label in zip(axes.flat[:3],("action_huber","arm_rmse_nm","wheel_rmse_rad_s"),
                            ("Weighted standardized Huber","Arm RMSE (N m)","Wheel RMSE (rad/s)")):
        for name in ("fitting","validation"):
            ax.plot(curves["updates"],curves[name][key],"o-",label=f"{name} (fixed probes)",color=colors[name])
        ax.set(xlabel="R1 optimizer updates",ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    ax = axes[1,1]
    tr = cfg["training"]
    updates = np.arange(1,tr["updates"]+1)
    rates = [tr["lr"]*lr_factor(int(u)-1,tr["warmup_updates"],tr["updates"],tr["schedule"]) for u in updates]
    ax.plot(updates,rates,color="#4b657b",label="Configured/used schedule")
    ax.scatter([int(r["update"]) for r in logged],[float(r["lr"]) for r in logged],s=15,color="#b24b3b",label="Logged LR")
    ax.set(xlabel="R1 optimizer updates",ylabel="Learning rate")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle("DAgger R1 / D8-W256: fixed original fitting vs validation\n0 = frozen R0; episode-mean probes, identical units on both splits")
    for suffix in ("png","pdf","svg"):
        metadata = {"CreationDate":None,"ModDate":None} if suffix == "pdf" else {"Date":None} if suffix == "svg" else None
        fig.savefig(out/f"2026-10-27__learning-curves__dagger-a-r1.{suffix}",dpi=180,metadata=metadata)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
    names = ["R0","5k","10k","15k","20k"]
    baseline_screen = json.loads((ROOT/"results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/verification.json").read_text())["evaluation"]["screening"][-1]["summary"]
    points = [baseline_screen]+[r["summary"] for r in evaluation["screening"]]
    for ax,key,label in zip(axes,("reach_success","sustained_hold_success","median_final_err_mm"),
                            ("Screening reach fraction","Screening hold fraction","Median final error (mm)")):
        ax.plot(names,[p[key] for p in points],"o-",color="#35a5dd")
        ax.set(ylabel=label,xlabel="Pure-student checkpoint")
        ax.grid(alpha=.2)
        if "success" in key:
            ax.set_ylim(-.03,1.03)
    fig.savefig(out/"2026-10-27__closed-loop__dagger-a-r1.png",dpi=180)
    plt.close(fig)
    # Source and phase coverage: distinguish row fractions from gradient sampling mass.
    source = {"original_rows":inventory["original_rows"],"new_rows":inventory["new_rows"],
              "new_episode_sampling_mass":inventory["new_episode_sampling_mass"],
              "new_row_fraction":inventory["new_row_fraction"],"new_phase_rows":audit["new_phase_rows"],
              "conditional_new_episode_phase_sampling_mass":audit["conditional_new_episode_phase_sampling_mass"]}
    source["actual_gradient_sampling"] = json.loads((output/"sampling_audit.json").read_text())
    report = {"run_id":cfg["identity"]["run_id"],"curve_series":curves,"source_coverage":source,
              "R0":evaluation["R0"],"R1":evaluation["R1"],"candidate_step":evaluation["candidate_step"],
              "promoted":evaluation["promoted"],"label_audit_rows":audit["queried_rows"],
              "source_sha256":sha(__file__),"plots":{p.name:sha(p) for p in sorted(out.iterdir())},
              "notes":["Collection success includes expert execution and is not student success.",
                       "C0/C1 not completed: no causal attribution to DAgger.",
                       "Independent new test remains sealed; no R2-R4 launched."]}
    freeze_json(output/"report.json",report)
    print(json.dumps({"report":str(output/"report.json"),"plots":list(report["plots"])}),flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-audit",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    args = parser.parse_args()
    if args.sampling_audit:
        audit_sampling(args.output)
    if not args.audit_only:
        make_report(args.output)


if __name__ == "__main__":
    main()
