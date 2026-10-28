from pathlib import Path
import json

import numpy as np
import pytest


@pytest.mark.sim
@pytest.mark.realdata
def test_freeze_and_real_pybullet_smoke_preserve_raw_current_state_labels(tmp_path):
    from data_collection import dagger
    if not (dagger.PREFLIGHT / "summary.json").exists() or not (dagger.ROOT / dagger.PARENT / "best.pt").exists():
        pytest.skip("requires the approved R0 artifacts and completed recovery evidence")
    assert hasattr(dagger, "freeze"), "DAgger pipeline still has only a placeholder"
    contract = dagger.freeze(tmp_path)
    assert contract["training"]["updates"] == 20000
    assert len(json.loads((tmp_path / "scenes.json").read_text())["train"]) == 256
    result = dagger.smoke(tmp_path)
    assert result["passed"]
    for beta in (0, 1):
        with np.load(tmp_path / f"smoke_beta{beta}.npz") as z:
            assert z["expert_action"].shape == (20, 11)
            assert z["obs_q_mes"].shape == (21, 7)
            assert np.all(z["expert_executes"] == bool(beta))
            assert np.all(z["valid_supervision"])
