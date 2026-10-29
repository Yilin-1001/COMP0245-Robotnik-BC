"""Exercise independent expert/student physics processes through the demo CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

from conftest import TASK_DIR


DEMO_MANIFEST = TASK_DIR / "experiments/demo_models/2026-10-28__training-ablation-v1.yaml"


def require_demo_artifact(row):
    """Skip when the seed-0 model of this ablation row is absent (checkpoints are git-ignored)."""
    protocol = yaml.safe_load(DEMO_MANIFEST.read_text())
    artifact = next(a for a in protocol["decisions"]["final"]["artifacts"] if (a["row"], a["seed"]) == (row, 0))
    if not (TASK_DIR / artifact["inference"]).exists():
        pytest.skip(f"trained checkpoint not present locally: {artifact['inference']}")
    return artifact


@pytest.mark.sim
@pytest.mark.parametrize("row", ["R4-dart-gate", "R1-baseline"])
def test_live_comparison_uses_two_independent_simulators_at_the_same_time(tmp_path, row):
    artifact = require_demo_artifact(row)
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.live_compare", "--headless", "--row", row,
         "--seconds", "0.1", "--output", str(output)],
        cwd=TASK_DIR, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())["runs"]
    assert {r["role"] for r in rows} == {"expert", "student"}
    assert len({r["pid"] for r in rows}) == 2
    assert all(r["steps"] == 100 and r["scenario_id"] == "train-0186" for r in rows)  # the default demo scenario
    assert all(np.isclose(r["simulation_time_s"], .1) for r in rows)
    assert all(r["failure"] is None and np.isfinite(r["final_error_mm"]) for r in rows)
    np.testing.assert_allclose(rows[0]["initial_base_pos"], rows[1]["initial_base_pos"])
    np.testing.assert_allclose(rows[0]["initial_arm_q"], rows[1]["initial_arm_q"])
    assert not np.allclose(rows[0]["final_arm_q"], rows[1]["final_arm_q"], atol=1e-7)
    assert json.loads(output.read_text())["artifact"]["run_id"] == artifact["run_id"]  # --row picked this model


@pytest.mark.sim
def test_custom_target_is_used_by_both_simulators(tmp_path):
    require_demo_artifact("R4-dart-gate")
    output = tmp_path / "custom.json"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.live_compare", "--headless", "--seconds", "0.02",
         "--target", "0.2", "1.1", "0.5", "--output", str(output)],
        cwd=TASK_DIR, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())["runs"]
    assert len(rows) == 2
    for row in rows:
        np.testing.assert_allclose(row["target_ee_pos"], [.2, 1.1, .5])
        assert row["failure"] is None and row["steps"] == 20
    np.testing.assert_allclose(rows[0]["initial_base_pos"], rows[1]["initial_base_pos"])


@pytest.mark.sim
def test_custom_start_and_bearing_are_used_by_both_simulators(tmp_path):
    require_demo_artifact("R4-dart-gate")
    output = tmp_path / "start.json"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.live_compare", "--headless", "--seconds", "0.02",
         "--start", "0.3", "-0.2", "--bearing", "1.2", "--target", "2.0", "1.5", "0.6", "--output", str(output)],
        cwd=TASK_DIR, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARNING" not in result.stdout  # 2.27 m away, 0.6 m high, start within +-0.5 m: inside the training ranges
    rows = json.loads(output.read_text())["runs"]
    for row in rows:
        np.testing.assert_allclose(row["initial_base_pos"][:2], [.3, -.2], atol=1e-3)
        assert row["scenario_id"].endswith("/custom-target-start-bearing") and row["failure"] is None


def test_scenes_outside_the_training_ranges_are_flagged():
    from data_collection.scenarios import Scenario
    from evaluation.live_compare import outside_training_range
    inside = Scenario(seed=0, target_ee_pos=np.array([1.5, 0.0, 0.6]), base_start_pos=np.zeros(3))
    far_high = Scenario(seed=0, target_ee_pos=np.array([4.5, 0.0, 1.5]), base_start_pos=np.array([1.0, 0.0, 0.0]))
    edge = Scenario(seed=0, target_ee_pos=np.array([3.0, 0.0, 1.25]), base_start_pos=np.zeros(3))
    assert outside_training_range(inside) == [] and outside_training_range(edge) == []  # range ends are inside
    assert len(outside_training_range(far_high)) == 3  # 3.5 m away, 1.5 m high, start x 1.0 m
