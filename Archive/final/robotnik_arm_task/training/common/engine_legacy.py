"""Archived on 2026-11-01 from training/common/engine.py: data preparation of the clean M1-M6 launchers (training.baseline.train / training.jacobian.train). It built feature caches under data/cache/.

Kept for reference only; it is not importable as is (its imports refer to the original module)."""

def _root(config: dict, key: str) -> Path:
    p = Path(config["data"][key])
    return p if p.is_absolute() else TASK_DIR / p


def prepare_run(config: dict, fitting_ids: list[str] | None = None, workers: int = 16, finalise: bool = True,
                before_finalise: Callable | None = None) -> dict:
    """Validate clean sources, build/reuse caches, fit fitting-only scales and finalise the run identity.

    Selection runs fit on the cohort's fitting IDs and validate on the 32 frozen IDs read from Clean_253.
    Refit runs fit on every training ID of the cohort and have no validation set.
    fitting_ids overrides the fitting set (the separately identified one-episode sanity run).
    before_finalise(config, normaliser) -> config runs after the scales exist (BC+J attaches its frozen calibration);
    finalise=False returns the prepared data and provenance draft without a run identity (calibration itself).
    """
    from training.common.config import finalise_identity
    from training.common.dataset import (FeatureBuilder, build_cache, cache_id, load_episode_refs, load_manifest,
                                         load_split, manifest_interface, select_refs, sha256_file)
    from training.common.normalisation import fit_normaliser
    ident, data = config["identity"], config["data"]
    teacher = ident["teacher"]
    primary, reduced = _root(config, "primary_root"), _root(config, "reduced_root")
    fit_root = primary if ident["cohort"] == "clean253" else reduced
    manifests = {r: load_manifest(r, teacher) for r in {fit_root, primary}}
    interfaces = {json.dumps(manifest_interface(m), sort_keys=True) for m in manifests.values()}
    if len(interfaces) != 1:
        raise ValueError("fitting and validation sources disagree on the execution interface")
    split_path = TASK_DIR / data["split_manifest"]
    split = load_split(split_path)
    fit_refs_all = load_episode_refs(fit_root, teacher, "train")
    if ident["fit_stage"] == "refit":
        ids, val_ids = [r.scenario_id for r in fit_refs_all], []
    else:
        ids, val_ids = split["fitting_ids"][ident["cohort"]], split["validation_ids"]
    if fitting_ids is not None:
        if not set(fitting_ids) <= set(ids):
            raise ValueError("sanity fitting IDs must come from the cohort's fitting set")
        ids = list(fitting_ids)
    fit_refs = select_refs(fit_refs_all, ids)
    val_refs = select_refs(load_episode_refs(primary, teacher, "train"), val_ids) if val_ids else []
    builder = FeatureBuilder(teacher, data["task_context"])
    cache_root = Path(data["cache_root"]) if Path(data["cache_root"]).is_absolute() else TASK_DIR / data["cache_root"]
    caches, ids_by = {}, {}
    for name, refs in (("fitting", fit_refs), ("validation", val_refs)):
        if not refs:
            caches[name] = None
            continue
        cid = cache_id(refs, builder.builder_id)
        caches[name] = build_cache(refs, builder, cache_root / teacher / cid, workers=workers)
        ids_by[name] = cid
    if caches["fitting"].x.shape[1] != config["model"]["input_dim"]:
        raise ValueError("cached feature dimension disagrees with the teacher's input dimension")
    floors = {k: data[k] for k in ("input_scale_floor", "arm_scale_floor", "wheel_scale_floor")}
    normaliser = fit_normaliser(caches["fitting"], floors)
    provenance = {
        "dataset_kind": next(iter(manifests.values()))["dataset_kind"],
        "teacher_version": data["teacher_version"], "interface": json.loads(interfaces.pop()),
        "source_sha256": {f"fit/{r.scenario_id}": r.sha256 for r in fit_refs} | {f"val/{r.scenario_id}": r.sha256 for r in val_refs},
        "fitting_ids": ids, "validation_ids": val_ids, "split_sha256": sha256_file(split_path),
        "feature_sha256": hashlib.sha256(builder.builder_id.encode()).hexdigest(), "cache_ids": ids_by,
        "normaliser_sha256": normaliser.sha256(), "code_digest": code_digest(),
        "operational": {"git_head": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    if before_finalise is not None:
        config = before_finalise(config, normaliser)
    out = {"train": caches["fitting"], "validation": caches["validation"], "normaliser": normaliser,
           "provenance": provenance}
    if not finalise:
        return dict(out, config=config)
    if config["identity"]["objective"] == "bcj":
        calib = config["loss"]["jacobian"].get("calibration_record")
        if calib:
            provenance["calibration_sha256"] = calib["sha256"]
    return dict(out, config=finalise_identity(config, provenance))
