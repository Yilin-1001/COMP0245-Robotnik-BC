"""Validate DAgger sources separately from clean data; retain valid failed prefixes."""
import hashlib
import json
from pathlib import Path

import numpy as np

from data_collection.dagger_preflight import ROOT, PARENT, sha
from data_collection.dagger_scenarios import assert_disjoint
from training.common.dataset import EpisodeCache, FeatureBuilder, OBS_FIELDS, LABEL_NAMES


def episode_rows(path, entry, contract, scenario, heldout, builder=None):
    if sha(path) != entry["sha256"]:
        raise ValueError("DAgger episode hash mismatch")
    assert_disjoint([scenario], heldout)
    with np.load(path) as z:
        a = {k: z[k] for k in z.files if k != "metadata_json"}
        meta = json.loads(str(z["metadata_json"]))
    if meta.get("source") != "dagger_mixed" or contract["source"] != "dagger_mixed":
        raise ValueError("main aggregate excludes reference/unknown source")
    if meta.get("normaliser_sha256") != contract["normaliser_content_sha256"]:
        raise ValueError("DAgger normaliser mismatch")
    if (meta.get("contract_sha256") != contract["contract_sha256"] or meta.get("scenario") != scenario or
            meta.get("scenario_id") != entry["scenario_id"] or meta.get("split") != "dagger_train" or
            meta.get("teacher_version") != "markovian-lt0.50-v1" or meta.get("round") != 1 or
            meta.get("policy_sha256") != contract["parent_hashes"]["inference.npz"]):
        raise ValueError("foreign DAgger identity, scene or teacher")
    T = len(a["expert_action"])
    shapes = {"expert_action": (T,11), "student_raw": (T,9), "student_action": (T,11), "executed_action": (T,11),
              "expert_executes": (T,), "block_id": (T,), "valid_supervision": (T,), "phase": (T,), "ee_err": (T,),
              "obs_time": (T+1,), "obs_base_pos": (T+1,3), "obs_base_yaw": (T+1,), "obs_q_mes": (T+1,7), "obs_qd_mes": (T+1,7)}
    if any(a[k].shape != shape for k, shape in shapes.items()) or entry["steps"] != T or meta["outcome"]["steps"] != T:
        raise ValueError("DAgger T+1/T alignment mismatch")
    e = a["expert_action"]
    if not (np.array_equal(e[:,7], e[:,10]) and np.array_equal(e[:,8], e[:,9])):
        raise ValueError("same-side wheel labels differ")
    if not np.allclose(a["obs_time"], np.arange(T+1)*.001, atol=1e-9, rtol=0):
        raise ValueError("DAgger time alignment differs from 1kHz")
    if not np.array_equal(a["block_id"], np.arange(T)//10) or any(
            len(set(a["expert_executes"][i:i+10])) != 1 for i in range(0,T,10)):
        raise ValueError("DAgger controller identities changed within a block")
    limits = np.r_[np.full(7,contract["interface"]["torque_limit"]), np.full(4,contract["interface"]["wheel_limit"])]
    selected = np.where(a["expert_executes"][:,None], e, a["student_action"])
    if not np.array_equal(a["executed_action"], np.clip(selected,-limits,limits)):
        raise ValueError("executed action differs from its selected/clipped controller")
    # A non-finite next observation is allowed; only the T pre-action observations supervise.
    valid = np.isfinite(e).all(axis=1)
    for key in OBS_FIELDS:
        valid &= np.isfinite(a[key][:T]).reshape(T,-1).all(axis=1) if T else np.ones(0,bool)
    if not np.array_equal(valid, a["valid_supervision"]) or entry["valid_rows"] != int(valid.sum()):
        raise ValueError("supervision mask/row inventory mismatch")
    builder = builder or FeatureBuilder("A", None)
    obs = {k: a[k][:T][valid] for k in OBS_FIELDS}
    feature_meta = {"target_ee_pos_W": [scenario[k] for k in ("target_x","target_y","target_z")], "scenario_id": scenario["scenario_id"]}
    x = np.asarray(builder(obs, feature_meta), dtype=np.float32)
    y = np.c_[e[valid,:7], e[valid,7], e[valid,8]].astype(np.float32)
    if x.shape != (int(valid.sum()),21) or not np.isfinite(x).all():
        raise ValueError("DAgger feature shape or finiteness mismatch")
    diag = {"t": a["obs_time"][:T][valid].astype(np.float32), "phase": a["phase"][valid].astype(np.int8),
            "ee_err": a["ee_err"][valid].astype(np.float32)}
    stats = {"scenario_id": scenario["scenario_id"], "steps": T, "valid_rows": int(valid.sum()),
             "excluded_rows": int(T-valid.sum()), "failure": meta["outcome"]["failure"],
             "phase_rows": {name: int(np.sum(diag["phase"] == i)) for i,name in enumerate(("navigation","approach","hold"))},
             "expert_execution_fraction": meta["expert_execution_fraction"], "outcome": meta["outcome"]}
    return x, y, diag, stats


def combine(caches):
    ids = [i for c in caches for i in c.episode_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate aggregate episode identity")
    lengths = np.concatenate([c.lengths for c in caches])
    if np.any(lengths <= 0):
        raise ValueError("zero-length episode needs explicit diagnosis before aggregation")
    keys = set.intersection(*(set(c.diagnostics) for c in caches))
    return EpisodeCache(np.concatenate([c.x for c in caches]), np.concatenate([c.y for c in caches]),
                        np.r_[0,np.cumsum(lengths)].astype(np.int64), ids,
                        {k: np.concatenate([c.diagnostics[k] for c in caches]) for k in keys}, dict(caches[0].metadata))


def original_data(parent):
    from training.common.dataset import load_episode_refs, select_refs, load_cache, load_split
    from training.common.normalisation import Normaliser
    cfg, prov = parent, parent["provenance"]
    split = load_split(ROOT / cfg["data"]["split_manifest"])
    if prov["fitting_ids"] != split["fitting_ids"]["clean253"] or prov["validation_ids"] != split["validation_ids"]:
        raise ValueError("R0 data split changed")
    refs = load_episode_refs(ROOT / cfg["data"]["primary_root"], "A", "train")
    caches = {}
    for name,prefix in (("fitting","fit"),("validation","val")):
        chosen = select_refs(refs,prov[f"{name}_ids"])
        if any(prov["source_sha256"][f"{prefix}/{r.scenario_id}"] != r.sha256 for r in chosen):
            raise ValueError("original source hashes changed from R0")
        caches[name] = load_cache(ROOT / cfg["data"]["cache_root"] / "A" / prov["cache_ids"][name], chosen)
    normaliser = Normaliser.load(ROOT/PARENT/"normaliser.npz")
    if normaliser.sha256() != prov["normaliser_sha256"]:
        raise ValueError("frozen R0 normaliser changed")
    return caches["fitting"], caches["validation"], normaliser


def round_data(output, protocol, parent):
    from data_collection.scenarios import read_manifest
    record = json.loads((output/"mixed_collection.json").read_text())
    dest = ROOT / record["directory"]
    manifest_path = dest/"round_manifest.json"
    if sha(manifest_path) != record["manifest_sha256"]:
        raise ValueError("round manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    contract_path = dest/"collection_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["contract_sha256"] = sha(contract_path)
    if manifest["contract_sha256"] != contract["contract_sha256"] or contract["protocol_sha256"] != sha(output/"protocol.json"):
        raise ValueError("round belongs to another collection contract")
    scene_manifest = json.loads((output/"scenes.json").read_text())
    allowed = {r["scenario_id"]: r for r in scene_manifest["train"] if r["round"] == 1}
    entries = manifest["episodes"]
    if len(entries) != 64 or len({e["scenario_id"] for e in entries}) != 64 or {e["scenario_id"] for e in entries} != set(allowed):
        raise ValueError("round does not contain the 64 frozen identities")
    primary = ROOT / parent["data"]["primary_root"]
    excluded = [r for split in ("train","test") for r in read_manifest(primary/f"scenarios_{split}.csv")] + scene_manifest["sealed_test"]
    assert_disjoint(list(allowed.values()),excluded)
    builder = FeatureBuilder("A",None)
    if hashlib.sha256(builder.builder_id.encode()).hexdigest() != parent["provenance"]["feature_sha256"]:
        raise ValueError("feature definition changed from R0")
    parts, stats = [], []
    for entry in entries:
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("unsafe episode path")
        x,y,diag,report = episode_rows(dest/entry["file"], entry, contract, allowed[entry["scenario_id"]], [], builder)
        stats.append(report)
        parts.append(EpisodeCache(x,y,np.array([0,len(x)]),[entry["scenario_id"]],diag,
                                 {"feature_names": builder.feature_names, "label_names": LABEL_NAMES}))
    return combine(parts), stats, record["manifest_sha256"], {e["scenario_id"]: e["sha256"] for e in entries}
