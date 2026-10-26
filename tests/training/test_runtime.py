import os
import subprocess
import sys

import pytest

from conftest import TASK_DIR


def test_import_has_no_simulator_or_training_side_effects():
    code = ("import sys; import training.common.config; "
            "bad = [m for m in ('pybullet', 'core.env', 'torch', 'simulation_and_control') if m in sys.modules]; "
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], cwd=TASK_DIR, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_cpu_device_is_explicit():
    from training.common.config import configure_runtime
    report = configure_runtime("cpu")
    assert report["device"] == "cpu"
    assert report["tensor_check"] == 6.0
    assert report["dtype"] == "float32"
    assert report["python"] == sys.version.split()[0]
    assert "torch" in report and "cuda_arch_list" in report


def test_unavailable_requested_cuda_is_reported(monkeypatch):
    import torch
    from training.common.config import configure_runtime
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        configure_runtime("cuda")
    report = configure_runtime("auto")
    assert report["device"] == "cpu"
    assert "CUDA unavailable" in report["fallback_reason"]


def test_rollout_threads_are_one():
    from training.common.config import configure_runtime
    report = configure_runtime("cpu", rollout=True)
    assert report["device"] == "cpu"
    assert report["torch_threads"] == 1
    assert all(os.environ[v] == "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"))
