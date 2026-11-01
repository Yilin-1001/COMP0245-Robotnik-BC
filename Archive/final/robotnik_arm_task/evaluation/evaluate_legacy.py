"""Archived on 2026-11-01 from evaluation/evaluate.py: the clean pipeline's frozen final-protocol gate to the 64 test scenarios. Only the archived experiments/run.py called it; the 64 test scenarios are now run by evaluation.test_diagnosis.

Kept for reference only; it is not importable as is (its imports refer to the original module)."""

# ---------------------------------------------------------------- frozen final protocol (test access)

def _file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_final_protocol(protocol: dict, index_path: Path | None = None) -> None:
    """The only gate to the 64 test scenarios: every declared artifact must equal the frozen validation choices.

    Architecture, LR, update budget, action loss and seed come from the protocol's earlier decisions; the
    checkpoint is best.pt for a selection artifact or final.pt for a refit artifact, with matching hashes and run
    config. Test rows already recorded under another protocol block a new one (no test-driven re-selection).
    """
    import yaml
    d = protocol.get("decisions") or {}
    final = d.get("final")
    if not final or not final.get("artifacts"):
        raise ValueError("no frozen final protocol: declare decisions.final.artifacts before any test evaluation")
    seeds = set(final.get("seeds") or [])
    for art in final["artifacts"]:
        t, arch = art["teacher"], art["architecture"]
        expected = {"architecture": d["selected"][t], "lr": d["lr"][t][arch] if arch in d["lr"].get(t, {}) else None,
                    "updates": d["budget"]["updates"], "loss": d["loss"]["action"]}
        for field, value in expected.items():
            if art[field] != value:
                raise ValueError(f"{art['label']}: {field} {art[field]!r} differs from the frozen choice {value!r}")
        if art["seed"] not in seeds:
            raise ValueError(f"{art['label']}: seed {art['seed']} is not in the frozen final seeds {sorted(seeds)}")
        kind = {"selection": "best.pt", "refit": "final.pt"}.get(art["fit_stage"])
        if kind is None or Path(art["checkpoint"]).name != kind:
            raise ValueError(f"{art['label']}: a {art['fit_stage']} artifact must be {kind}")
        for key in ("checkpoint", "inference"):
            if _file_sha(art[key]) != art[f"{key}_sha256"]:
                raise ValueError(f"{art['label']}: {key} sha256 differs from the frozen protocol")
        cfg = yaml.safe_load((Path(art["checkpoint"]).parent / "config.yaml").read_text())
        actual = {"architecture": cfg["identity"]["architecture"], "lr": cfg["training"]["lr"],
                  "updates": cfg["training"]["updates"], "loss": cfg["loss"]["action"], "seed": cfg["identity"]["seed"],
                  "fit_stage": cfg["identity"]["fit_stage"], "run_id": cfg["identity"]["run_id"],
                  "objective": cfg["identity"]["objective"], "teacher": cfg["identity"]["teacher"]}
        for field, value in actual.items():
            if art[field] != value:
                raise ValueError(f"{art['label']}: run config {field} {value!r} != protocol {art[field]!r}")
    if index_path is not None and Path(index_path).exists():
        own = f"test-{protocol['protocol_id']}"
        _, rows = _read_index(Path(index_path))
        other = [r for r in rows if r["evaluation_split"] == "test" and r["evaluation_id"] != own]
        if other:
            raise ValueError(f"test outcomes already observed under another protocol ({other[0]['evaluation_id']})")


def evaluate_final(protocol: dict, workers: int = 16) -> list[dict]:
    """Evaluate every declared artifact once on the 64 sealed test scenarios; returns the recorded rows."""
    import yaml
    from evaluation.test_sets import get_split
    index = TASK_DIR / "results" / "results.csv"
    validate_final_protocol(protocol, index_path=index)
    eval_id = f"test-{protocol['protocol_id']}"
    manifest = TASK_DIR / "experiments/splits/clean253.json"
    rows = []
    for art in protocol["decisions"]["final"]["artifacts"]:
        cfg = yaml.safe_load((Path(art["checkpoint"]).parent / "config.yaml").read_text())
        ident = dict(cfg["identity"])
        step = cfg["training"]["updates"] if art["fit_stage"] == "refit" else \
            json.loads((Path(art["checkpoint"]).parent / "selection.json").read_text())["selected_step"]
        _, existing = _read_index(index)
        if any((r["run_id"], r["checkpoint_step"], r["evaluation_split"], r["evaluation_id"]) ==
               (ident["run_id"], str(step), "test", eval_id) for r in existing):
            raise ValueError(f"{art['label']} was already evaluated on the test set; reruns are refused")
        cfg["evaluation"]["workers"] = workers
        factory = functools.partial(mlp_factory, art["inference"], "numpy", {"run_id": ident["run_id"]})
        outcomes = evaluate_policy(factory, get_split("test", manifest, art["teacher"], allow_test=True), cfg,
                                   workers=workers)
        detail = TASK_DIR / cfg["outputs"]["results_root"] / ident["regime"] / ident["teacher"] / ident["objective"] / \
            ident["run_id"] / f"rollouts_test_step{step}_{eval_id}.csv"
        write_details(detail, outcomes)
        row = result_row(ident, summarise(outcomes), lr=cfg["training"]["lr"], split_sha256=_file_sha(manifest),
                         checkpoint_step=step, checkpoint_path=art["checkpoint"],
                         checkpoint_sha256=art["checkpoint_sha256"], inference_path=art["inference"],
                         evaluation_split="test", evaluation_id=eval_id, detail_path=str(detail.relative_to(TASK_DIR)))
        append_results(index, [row])
        rows.append(row)
    return rows
