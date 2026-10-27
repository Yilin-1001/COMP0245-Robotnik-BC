import copy
import hashlib

import pytest
import yaml

from evaluation.evaluate import append_results, validate_final_protocol

DECISIONS = {"budget": {"updates": 30000, "pilot_lr": 3e-4}, "loss": {"action": "huber"},
             "lr": {"A": {"m2": 1e-3}}, "selected": {"A": "m2"}}


def make_artifact(tmp_path, name="best.pt", fit_stage="selection", arch="m2", lr=1e-3, seed=0, objective="bc"):
    run = tmp_path / f"run-{fit_stage}-{seed}-{objective}"
    run.mkdir()
    config = {"identity": {"run_id": run.name, "teacher": "A", "objective": objective, "architecture": arch,
                           "seed": seed, "fit_stage": fit_stage, "config_hash": "c" * 64},
              "training": {"lr": lr, "updates": 30000, "schedule": "cosine"}, "loss": {"action": "huber"}}
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    (run / name).write_bytes(b"weights")
    (run / "inference.npz").write_bytes(b"export")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {"label": f"A-{objective}-s{seed}", "teacher": "A", "objective": objective, "architecture": arch, "lr": lr,
            "updates": 30000, "loss": "huber", "seed": seed, "fit_stage": fit_stage, "run_id": run.name,
            "checkpoint": str(run / name), "checkpoint_sha256": sha(run / name),
            "inference": str(run / "inference.npz"), "inference_sha256": sha(run / "inference.npz")}


def protocol(artifacts, seeds=(0,)):
    return {"protocol_id": "p-final", "decisions": dict(DECISIONS, final={"artifacts": artifacts, "seeds": list(seeds),
                                                                          "evaluation_id": "rollout-v1"})}


def test_valid_final_protocol_passes(tmp_path):
    validate_final_protocol(protocol([make_artifact(tmp_path)]))


@pytest.mark.parametrize("field,value", [("architecture", "m3"), ("lr", 3e-4), ("updates", 10000), ("loss", "mse"),
                                         ("seed", 5)])
def test_final_choices_must_match_frozen_selection(tmp_path, field, value):
    art = make_artifact(tmp_path)
    art[field] = value
    with pytest.raises(ValueError):
        validate_final_protocol(protocol([art]))


def test_checkpoint_hash_and_artifact_kind_are_checked(tmp_path):
    art = make_artifact(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        validate_final_protocol(protocol([dict(art, checkpoint_sha256="0" * 64)]))
    wrong = make_artifact(tmp_path, name="final.pt", fit_stage="selection", seed=0, objective="bcj")
    with pytest.raises(ValueError, match="best.pt"):
        validate_final_protocol(protocol([wrong]))
    refit = make_artifact(tmp_path, name="best.pt", fit_stage="refit")
    with pytest.raises(ValueError, match="final.pt"):
        validate_final_protocol(protocol([refit]))


def test_test_results_from_another_protocol_block_a_new_one(tmp_path):
    index = tmp_path / "results.csv"
    from evaluation.evaluate import RESULT_COLUMNS
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(run_id="earlier", teacher="A", checkpoint_step=1, evaluation_split="test", evaluation_id="test-p-old")
    append_results(index, [row])
    with pytest.raises(ValueError, match="already observed"):
        validate_final_protocol(protocol([make_artifact(tmp_path)]), index_path=index)


def test_duplicate_test_evaluation_is_refused(tmp_path):
    from evaluation.evaluate import RESULT_COLUMNS
    index = tmp_path / "results.csv"
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(run_id="r", teacher="A", checkpoint_step=30000, evaluation_split="test", evaluation_id="test-p-final")
    append_results(index, [row])
    with pytest.raises(ValueError):
        append_results(index, [copy.deepcopy(row)])
