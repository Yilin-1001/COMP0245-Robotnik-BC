"""Runtime checks, configuration resolution, validation and run identity (structure design §5–6).

Importing this module has no side effects: torch is imported inside functions only.
"""
import os
import platform

ROLLOUT_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def pin_rollout_threads() -> None:
    """Single-threaded BLAS for simulation/rollout workers; call before numerical imports where possible."""
    for var in ROLLOUT_THREAD_VARS:
        os.environ[var] = "1"


def configure_runtime(device: str, rollout: bool = False) -> dict:
    """Select and verify the compute device; returns versions, CUDA architectures and thread settings.

    device: "cpu", "cuda" (fails if unusable) or "auto" (CUDA when a kernel runs, else CPU with the reason).
    rollout: pin BLAS and torch to one thread, as every simulation/evaluation worker must.
    """
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError(f"device must be cpu, cuda or auto, got {device!r}")
    if rollout:
        pin_rollout_threads()
    import torch
    if rollout:
        torch.set_num_threads(1)
    report = {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "cuda_arch_list": [], "gpu": None, "dtype": "float32", "amp": False, "fallback_reason": None,
              "rollout": rollout, "torch_threads": torch.get_num_threads(),
              "thread_env": {v: os.environ.get(v) for v in ROLLOUT_THREAD_VARS}}
    selected = "cpu"
    if device in ("cuda", "auto"):
        reason = _cuda_problem(torch)
        if reason is None:
            selected = "cuda"
            report["cuda_arch_list"] = torch.cuda.get_arch_list()
            report["gpu"] = torch.cuda.get_device_name(0)
        elif device == "cuda":
            raise RuntimeError(f"CUDA requested but unavailable: {reason}")
        else:
            report["fallback_reason"] = reason
    check = torch.arange(4, dtype=torch.float32, device=selected).sum().item()
    report.update(device=selected, tensor_check=float(check))
    return report


def _cuda_problem(torch) -> str | None:
    """None when a CUDA kernel actually runs on this build; otherwise a description of why not."""
    if not torch.cuda.is_available():
        return "CUDA unavailable (torch.cuda.is_available() is False)"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            return f"CUDA unavailable: {arch} not in this build's architectures {torch.cuda.get_arch_list()}"
        x = torch.ones(8, device="cuda")
        if float((x * 2).sum().item()) != 16.0:
            return "CUDA unavailable: kernel check returned a wrong result"
    except Exception as e:  # driver/kernel failures surface as runtime errors
        return f"CUDA unavailable: {type(e).__name__}: {e}"
    return None
