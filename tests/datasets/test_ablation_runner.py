"""The ablation manifest resolves to the intended runs, names and directories, and the interval helper is correct."""
import pytest

from conftest import TASK_DIR

MANIFEST = TASK_DIR / "experiments/ablation/2026-10-28__training-ablation-v1.yaml"


def test_manifest_resolves_to_twelve_named_runs():
    from experiments.ablation.run import experiment_dir, load_manifest, runs
    m = load_manifest(MANIFEST)
    rs = runs(m)
    assert len(rs) == 12 and len({r["name"] for r in rs}) == 12
    assert experiment_dir(m) == "2026-10-28__training-ablation-v1"
    r = next(r for r in rs if r["row"] == "R4-dart-gate" and r["seed"] == 2)
    assert r["name"] == "R4-dart-gate__MLP01-256x3__s2__v1__2026-10-28"
    assert r["dir"] == TASK_DIR / "checkpoints/ablation/2026-10-28__training-ablation-v1" / r["name"]
    assert r["model"] == r["dir"] / f"{r['name']}.npz"


def test_manifest_recipe_is_the_agreed_one():
    from experiments.ablation.run import load_manifest, overrides
    m = load_manifest(MANIFEST)
    o = overrides(m, 1)
    assert o["identity.seed"] == 1 and o["identity.architecture"] == "MLP01" and o["identity.objective"] == "bcj"
    assert (o["training.lr"], o["training.updates"], o["loss.action"]) == (1e-3, 42000, "huber")
    assert (o["loss.arm_weight"], o["loss.wheel_weight"]) == (0.5, 0.5)
    assert m["rows"] == ["R1-baseline", "R2-gate", "R3-dart", "R4-dart-gate"] and m["seeds"] == [0, 1, 2]


def test_wilson_interval():
    from experiments.ablation.run import wilson
    lo, hi = wilson(23, 32)
    assert lo == pytest.approx(0.5463, abs=1e-4) and hi == pytest.approx(0.8444, abs=1e-4)  # 23/32, z = 1.96
    assert wilson(0, 32)[0] == 0.0 and wilson(32, 32)[1] == 1.0


# ---------------------------------------------------------------- architecture grids and references

R4_V1 = ("checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s{seed}__v1__2026-10-28/"
         "R4-dart-gate__MLP01-256x3__s{seed}__v1__2026-10-28.npz")
R4_SOURCE = "checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/config.yaml"


def _grid(tmp_path, **changes):
    import yaml
    from experiments.ablation.run import load_manifest
    m = yaml.safe_load(MANIFEST.read_text())
    m.pop("architecture")
    m.update({"experiment": "architecture-ablation", "date": "2026-11-02", "architectures": ["MLP02", "MLP07"],
              "rows": ["R4-dart-gate"], "seeds": [0, 1], "lambda_j_source": {"R4-dart-gate": R4_SOURCE},
              "references": {"R4-dart-gate": R4_V1}, **changes})
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(m))
    return load_manifest(path)


def test_grid_manifest_resolves_rows_x_architectures_x_seeds_plus_references(tmp_path):
    from experiments.ablation.run import runs
    if not (TASK_DIR / R4_V1.format(seed=0)).exists():
        pytest.skip("R4 v1 checkpoints not present")
    rs = runs(_grid(tmp_path))
    assert len(rs) == 4 + 2 and len({r["name"] for r in rs}) == 6
    assert any(r["name"] == "R4-dart-gate__MLP07-256-128-64__s1__v1__2026-11-02" for r in rs)
    refs = [r for r in rs if r["reference"]]
    assert len(refs) == 2 and all(r["architecture"] == "MLP01-256x3" and r["parameters"] == 141321 for r in refs)
    assert {r["architecture"] for r in rs if not r["reference"]} == {"MLP02-32x3", "MLP07-256-128-64"}


def test_overrides_carry_architecture_and_lambda_source(tmp_path):
    from experiments.ablation.run import overrides
    o = overrides(_grid(tmp_path), 0, "MLP02", "R4-dart-gate")
    assert o["loss.jacobian.calibration"] == R4_SOURCE and o["identity.architecture"] == "MLP02"
    assert o["identity.seed"] == 0 and o["training.lr"] == 1e-3


def test_missing_reference_is_an_error_not_a_training_job(tmp_path, monkeypatch):
    import subprocess
    from experiments.ablation import run
    monkeypatch.setattr(run, "results_dir", lambda m: tmp_path / "results")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("no training may start"))
    m = _grid(tmp_path, references={"R4-dart-gate": str(tmp_path / "R4-dart-gate__MLP01-256x3__s{seed}__v1__2026-10-28.npz")})
    with pytest.raises(FileNotFoundError, match="reference"):
        run.train(m, 1)


def test_reference_validation_skips_results_index(tmp_path, monkeypatch):
    import json
    import multiprocessing
    import numpy as np
    import yaml
    import evaluation.evaluate as ev
    import evaluation.test_sets as ts
    from experiments.ablation import run
    ref = "R4-dart-gate__MLP01-256x3__s{seed}__v1__2026-10-28"
    for seed in (0, 1):
        d = tmp_path / ref.format(seed=seed)
        d.mkdir()
        meta = {"source_step": 42000, "source_checkpoint": str(d / "step_42000.pt"), "source_sha256": "x"}
        np.savez(d / f"{d.name}.npz", metadata_json=np.array(json.dumps(meta)))
        (d / "config.yaml").write_text(yaml.safe_dump({"identity": {"run_id": d.name}, "training": {"lr": 1e-3}}))
    m = _grid(tmp_path, architectures=[], references={"R4-dart-gate": str(tmp_path / ref / f"{ref}.npz")})

    class Pool:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def imap_unordered(self, fn, jobs, chunksize=1): return ((j[0], j[1], {"scenario_id": j[2]}) for j in jobs)

    monkeypatch.setattr(run, "results_dir", lambda m: tmp_path / "results")
    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: type("C", (), {"Pool": Pool}))
    monkeypatch.setattr(ts, "get_split", lambda *a: ["val-0", "val-1"])
    monkeypatch.setattr(ev, "summarise", lambda outcomes: {})
    monkeypatch.setattr(ev, "result_row", lambda ident, summary, **kw: {"run_id": ident["run_id"]})
    monkeypatch.setattr(ev, "write_details", lambda path, outcomes: (path.parent.mkdir(parents=True, exist_ok=True),
                                                                 path.write_text("scenario_id\n")))
    appended = []
    monkeypatch.setattr(ev, "append_results", lambda path, rows: appended.extend(rows))
    run.validate(m, 1)
    assert appended == []
    for seed in (0, 1):
        assert (tmp_path / "results/validation" / f"{ref.format(seed=seed)}__validation-student.csv").exists()


def test_unindexed_manifest_never_appends_to_results_index(tmp_path, monkeypatch):
    import json
    import multiprocessing
    import numpy as np
    import yaml
    import evaluation.evaluate as ev
    import evaluation.test_sets as ts
    from experiments.ablation import run
    m = _grid(tmp_path, architectures=["MLP02"], seeds=[0], references={})
    m["evaluation"]["index_results"] = False
    (r,) = run.runs(m)
    monkeypatch.setattr(run, "runs", lambda m: [{**r, "dir": tmp_path / "trained", "model": tmp_path / "trained/m.npz"}])
    (tmp_path / "trained").mkdir()
    meta = {"source_step": 42000, "source_checkpoint": "x.pt", "source_sha256": "x"}
    np.savez(tmp_path / "trained/m.npz", metadata_json=np.array(json.dumps(meta)))
    (tmp_path / "trained/config.yaml").write_text(yaml.safe_dump({"identity": {"run_id": r["name"]}, "training": {"lr": 3e-4}}))

    class Pool:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def imap_unordered(self, fn, jobs, chunksize=1): return ((j[0], j[1], {"scenario_id": j[2]}) for j in jobs)

    monkeypatch.setattr(run, "results_dir", lambda m: tmp_path / "results")
    monkeypatch.setattr(multiprocessing, "get_context", lambda kind: type("C", (), {"Pool": Pool}))
    monkeypatch.setattr(ts, "get_split", lambda *a: ["val-0"])
    monkeypatch.setattr(ev, "summarise", lambda outcomes: {})
    monkeypatch.setattr(ev, "result_row", lambda ident, summary, **kw: {"run_id": ident["run_id"]})
    monkeypatch.setattr(ev, "write_details", lambda path, outcomes: (path.parent.mkdir(parents=True, exist_ok=True),
                                                                 path.write_text("scenario_id\n")))
    monkeypatch.setattr(ev, "append_results", lambda path, rows: pytest.fail("unindexed manifest wrote results.csv"))
    run.validate(m, 1)
    assert (tmp_path / "results/validation" / f"{r['name']}__validation-student.csv").exists()


def test_rows_filter_rejects_unknown_row(tmp_path):
    import yaml
    from experiments.ablation.run import main
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in _grid(tmp_path).items() if k != "path"}))
    with pytest.raises(SystemExit, match="R9"):
        main(["--manifest", str(path), "--rows", "R9", "--stage", "summary"])
