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
