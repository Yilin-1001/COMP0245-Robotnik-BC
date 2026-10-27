"""Exercise independent expert/student physics processes through the demo CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import TASK_DIR


@pytest.mark.sim
@pytest.mark.parametrize("teacher", ["A", "B"])
def test_live_comparison_uses_two_independent_simulators_at_the_same_time(tmp_path, teacher):
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.live_compare", "--headless", "--teacher", teacher,
         "--seconds", "0.1", "--output", str(output)],
        cwd=TASK_DIR, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())["runs"]
    assert {r["role"] for r in rows} == {"expert", "student"}
    assert len({r["pid"] for r in rows}) == 2
    assert all(r["steps"] == 100 and r["scenario_id"] == "train-0143" for r in rows)
    assert all(np.isclose(r["simulation_time_s"], .1) for r in rows)
    assert all(r["failure"] is None and np.isfinite(r["final_error_mm"]) for r in rows)
    np.testing.assert_allclose(rows[0]["initial_base_pos"], rows[1]["initial_base_pos"])
    np.testing.assert_allclose(rows[0]["initial_arm_q"], rows[1]["initial_arm_q"])
    assert not np.allclose(rows[0]["final_arm_q"], rows[1]["final_arm_q"], atol=1e-7)
