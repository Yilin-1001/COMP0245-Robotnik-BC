"""Runtime checks, configuration resolution, validation and run identity (structure design §5–6).

Importing this module has no side effects: torch is imported inside functions only.
"""
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import yaml

TASK_DIR = Path(__file__).resolve().parents[2]
ROLLOUT_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
TEACHER_INPUT_DIMS = {"A": 21, "B": 26}
WORKFLOW_OBJECTIVES = {"baseline.yaml": "bc", "jacobian.yaml": "bcj"}
COHORTS = ("clean253", "clean223")
FIT_STAGES = ("selection", "refit")
ACTION_LOSSES = ("huber", "mse")
SCHEDULES = ("cosine", "constant")
INTERFACE_KEYS = ("action_order", "torque_limit", "wheel_limit", "control_hz")

# Every permitted key, and the one layer that may define it. Values at the "opaque" paths are mappings or
# lists owned as a whole (their inner keys are validated by the code that consumes them).
OPAQUE = ("model.architectures", "loss.jacobian", "data.task_context", "interface", "provenance")
OWNERS = {
    "common": ["identity.cohort", "identity.architecture", "identity.seed", "identity.fit_stage", "identity.replicate",
               "model.architectures", "model.layernorm_eps",
               "training.optimizer", "training.lr", "training.lr_grid", "training.weight_decay", "training.batch_size",
               "training.warmup_updates", "training.schedule", "training.updates", "training.log_every",
               "training.checkpoint_every", "training.device",
               "loss.action", "loss.action_candidates", "loss.huber_delta", "loss.epsilon_tau",
               "loss.arm_weight", "loss.wheel_weight",
               "data.split_manifest", "data.cache_root", "data.input_scale_floor", "data.arm_scale_floor",
               "data.wheel_scale_floor",
               "evaluation.workers", "evaluation.episode_cap_s", "evaluation.min_reach_time_s", "evaluation.post_reach_s",
               "evaluation.hold_window_s", "evaluation.reach_tolerance_m", "evaluation.hold_bias_gate_nm",
               "evaluation.jacobian_samples",
               "outputs.checkpoint_root", "outputs.results_root", "outputs.results_index", "interface", "provenance"],
    "workflow": ["identity.regime", "identity.objective"],
    "objective": ["loss.jacobian"],
    "teacher": ["identity.teacher", "model.input_dim", "data.teacher_dir", "data.teacher_version",
                "data.feature_definition", "data.primary_root", "data.reduced_root", "data.task_context"],
}
KNOWN = {k for keys in OWNERS.values() for k in keys}
CLI_PERMITTED = {"identity.cohort", "identity.architecture", "identity.seed", "identity.fit_stage", "identity.replicate",
                 "training.lr", "training.lr_grid", "training.updates", "training.schedule", "training.warmup_updates",
                 "training.log_every", "training.checkpoint_every", "training.device", "loss.action",
                 "loss.arm_weight", "loss.wheel_weight",
                 "loss.jacobian.lambda_j", "loss.jacobian.calibration", "data.cache_root", "evaluation.workers",
                 "outputs.checkpoint_root", "outputs.results_root", "outputs.results_index"}
# Recorded but excluded from scientific identity: moving artifacts or changing device is not a new experiment.
OPERATIONAL = {"training.device", "evaluation.workers", "data.cache_root", "data.primary_root", "data.reduced_root",
               "data.split_manifest", "outputs.checkpoint_root", "outputs.results_root", "outputs.results_index",
               "identity.run_id", "identity.config_id", "identity.config_hash", "provenance.operational"}


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
    report = {"python": platform.python_version(), "torch": str(torch.__version__), "cuda_build": str(torch.version.cuda),
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


# ---------------------------------------------------------------- resolution

def merge(base: dict, update: dict) -> dict:
    """Recursive mapping merge; scalars and lists in update replace base values (lists are never concatenated)."""
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and v:
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_overrides(items: list[str]) -> dict:
    """CLI --set key=value pairs; values are parsed as YAML (numbers, lists, null)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if key in OPAQUE or not isinstance(v, dict):
            out[key] = v
        else:
            out.update(_flatten(v, key + "."))
    return out


def _check_layer(name: str, layer: str, data: dict) -> None:
    for key in _flatten(data):
        if key not in KNOWN:
            raise ValueError(f"unknown config key {key!r} in {name}")
        if key not in OWNERS[layer]:
            raise ValueError(f"{name} may not define {key!r} (owned by another layer)")


def _set_dotted(config: dict, key: str, value) -> None:
    node = config
    parts = key.split(".")
    for p in parts[:-1]:
        if node.get(p) is None:
            raise ValueError(f"override {key!r} addresses an unset mapping")
        node = node[p]
    node[parts[-1]] = value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_config(path: Path, teacher: str, overrides: dict) -> dict:
    """common.yaml -> workflow (path) -> objectives/<objective>.yaml -> teachers/<teacher>.yaml -> overrides."""
    path = Path(path)
    if path.name not in WORKFLOW_OBJECTIVES:
        raise ValueError(f"unknown workflow template {path.name}; clean workflows are {sorted(WORKFLOW_OBJECTIVES)}")
    if teacher not in TEACHER_INPUT_DIMS:
        raise ValueError(f"teacher must be A or B, got {teacher!r}")
    root = path.parent
    load = lambda p: yaml.safe_load(p.read_text()) or {}
    layers = [("common.yaml", "common", root / "common.yaml"), (path.name, "workflow", path)]
    workflow = load(path)
    objective = workflow.get("identity", {}).get("objective")
    if objective != WORKFLOW_OBJECTIVES[path.name]:
        raise ValueError(f"{path.name} must pin objective {WORKFLOW_OBJECTIVES[path.name]!r}, found {objective!r}")
    layers += [(f"objectives/{objective}.yaml", "objective", root / "objectives" / f"{objective}.yaml"),
               (f"teachers/{teacher}.yaml", "teacher", root / "teachers" / f"{teacher}.yaml")]
    config, hashes = {}, {}
    for name, layer, p in layers:
        data = load(p)
        _check_layer(name, layer, data)
        config = merge(config, data)
        hashes[name] = _sha256_file(p)
    for key, value in (overrides or {}).items():
        top = key if key in KNOWN else next((k for k in OPAQUE if key.startswith(k + ".")), None)
        if top is None:
            raise ValueError(f"unknown config key {key!r} in overrides")
        if key not in CLI_PERMITTED:
            raise ValueError(f"override of {key!r} is not permitted")
        _set_dotted(config, key, value)
    config["layers"] = [name for name, _, _ in layers] + ["cli"]
    config["template_sha256"] = hashes
    validate_config(config)
    return config


# ---------------------------------------------------------------- validation

def action_weights(loss: dict) -> tuple[float, float]:
    """Read explicit action-group weights; checkpoints predating the fields used 0.5/0.5."""
    values = (loss.get("arm_weight", 0.5), loss.get("wheel_weight", 0.5))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in values) or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("action weights must be finite non-negative numbers summing to 1")
    return float(values[0]), float(values[1])


def validate_config(config: dict, ready_to_train: bool = False) -> None:
    """Invariants of every clean config; ready_to_train additionally requires a complete training recipe."""
    ident, model, training, loss = config["identity"], config["model"], config["training"], config["loss"]
    workflow = config["layers"][1]
    if ident["regime"] != "clean":
        raise ValueError(f"clean entry points accept regime 'clean' only, got {ident['regime']!r}")
    if ident["objective"] != WORKFLOW_OBJECTIVES[workflow]:
        raise ValueError(f"{workflow} is pinned to objective {WORKFLOW_OBJECTIVES[workflow]!r}")
    if (loss.get("jacobian") is None) != (ident["objective"] == "bc"):
        raise ValueError("objective bc has no jacobian loss settings; bcj requires them")
    if model["input_dim"] != TEACHER_INPUT_DIMS[ident["teacher"]]:
        raise ValueError(f"teacher {ident['teacher']} has {TEACHER_INPUT_DIMS[ident['teacher']]} inputs, "
                         f"config says {model['input_dim']}")
    if ident["cohort"] not in COHORTS or ident["fit_stage"] not in FIT_STAGES:
        raise ValueError(f"cohort must be one of {COHORTS} and fit_stage one of {FIT_STAGES}")
    if ident["architecture"] not in model["architectures"]:
        raise ValueError(f"unknown architecture {ident['architecture']!r}")
    if training["schedule"] not in SCHEDULES or training["optimizer"] != "adamw":
        raise ValueError(f"schedule must be one of {SCHEDULES}; optimizer must be adamw")
    if loss["action"] is not None and loss["action"] not in ACTION_LOSSES:
        raise ValueError(f"action loss must be one of {ACTION_LOSSES}")
    action_weights(loss)
    for key in ("seed", "replicate", "batch_size", "warmup_updates", "log_every", "checkpoint_every"):
        value = ident.get(key, training.get(key))
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if not ready_to_train:
        return
    if training["updates"] is None or not isinstance(training["updates"], int) or training["updates"] <= 0:
        raise ValueError("training.updates must be a positive integer before the first optimiser update")
    if training["lr"] is None or not (training["lr"] > 0):
        raise ValueError("training.lr must be set before the first optimiser update")
    if loss["action"] is None:
        raise ValueError("loss.action must be chosen before the first optimiser update")
    if ident["objective"] == "bcj":
        lam = loss["jacobian"].get("lambda_j")
        if lam is None or not math.isfinite(lam) or lam < 0:
            raise ValueError("loss.jacobian.lambda_j must be a frozen finite calibration before training")


def check_interface(config_interface: dict, manifest_interface: dict) -> dict:
    """The runtime interface comes from the source manifests; any disagreement fails rather than overriding."""
    for key in INTERFACE_KEYS:
        if key not in manifest_interface:
            raise ValueError(f"source manifest interface lacks {key!r}")
        if key in config_interface and config_interface[key] != manifest_interface[key]:
            raise ValueError(f"interface {key} mismatch: config {config_interface[key]!r} vs manifest "
                             f"{manifest_interface[key]!r}")
    return {k: manifest_interface[k] for k in INTERFACE_KEYS}


# ---------------------------------------------------------------- identity

def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def format_lr(lr: float) -> str:
    mantissa, exponent = f"{lr:e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+d}".replace("e+", "e")


def scientific_view(config: dict) -> dict:
    """The fields that define the experiment: everything except operational fields and resolution metadata."""
    flat = _flatten({k: v for k, v in config.items() if k not in ("layers", "template_sha256")})
    prov = config.get("provenance") or {}
    view = {k: v for k, v in flat.items() if k not in OPERATIONAL and k != "provenance"}
    view["provenance"] = {k: v for k, v in prov.items() if k != "operational"}
    return view


def finalise_identity(config: dict, provenance: dict) -> dict:
    """Attach provenance (source/split/feature/normaliser/calibration hashes, manifest interface) and the run ID."""
    config = copy.deepcopy(config)
    validate_config(config, ready_to_train=True)
    kind = str(provenance.get("dataset_kind", ""))
    if any(word in kind.lower() for word in ("dart", "dagger", "noise")):
        raise ValueError(f"clean entry points require clean source data, got dataset_kind {kind!r}")
    for key in ("source_sha256", "split_sha256", "feature_sha256", "normaliser_sha256", "code_digest", "interface"):
        if not provenance.get(key):
            raise ValueError(f"provenance lacks {key!r}")
    if config["identity"]["objective"] == "bcj" and not provenance.get("calibration_sha256"):
        raise ValueError("bcj identity requires the frozen calibration hash")
    config["interface"] = check_interface(config.get("interface") or {}, provenance["interface"])
    config["provenance"] = copy.deepcopy(provenance)
    full = hashlib.sha256(canonical_json(scientific_view(config)).encode()).hexdigest()
    ident = config["identity"]
    ident["config_hash"], ident["config_id"] = full, full[:12]
    ident["run_id"] = "-".join([ident["regime"], ident["teacher"], ident["objective"], ident["cohort"],
                                ident["architecture"], "lr" + format_lr(config["training"]["lr"]),
                                f"s{ident['seed']}", ident["fit_stage"], full[:12]])
    return config


def run_dir_for(config: dict) -> Path:
    """<root>/<regime>/<teacher>/<objective>/<run_id>, or <root>/<regime>/<experiment dir>/<run_id> for experiment runs."""
    ident = config["identity"]
    root = Path(config["outputs"]["checkpoint_root"])
    root = root if root.is_absolute() else TASK_DIR / root
    if config["outputs"].get("experiment_dir"):
        return root / ident["regime"] / config["outputs"]["experiment_dir"] / ident["run_id"]
    return root / ident["regime"] / ident["teacher"] / ident["objective"] / ident["run_id"]


def create_run_dir(config: dict, resume: bool = False) -> Path:
    """New run directory with the immutable resolved config; an existing one is reopened only for exact resume."""
    run_dir = run_dir_for(config)
    stored = run_dir / "config.yaml"
    if run_dir.exists():
        old = yaml.safe_load(stored.read_text()) if stored.exists() else None
        if old is None or old["identity"]["config_hash"] != config["identity"]["config_hash"]:
            raise ValueError(f"run ID collision or foreign directory at {run_dir}: config hash differs")
        if not resume:
            raise FileExistsError(f"{run_dir} exists; pass resume to continue it")
        if not (run_dir / "latest.pt").is_file():
            raise FileNotFoundError(f"cannot resume {run_dir}: latest.pt is missing")
        return run_dir
    if resume:
        raise FileNotFoundError(f"cannot resume {run_dir}: it does not exist (no latest.pt)")
    run_dir.mkdir(parents=True)
    tmp = stored.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(config, sort_keys=False))
    os.replace(tmp, stored)
    return run_dir


# ---------------------------------------------------------------- frozen protocols

PROTOCOL_DECISIONS = {"budget", "loss", "lr", "finalists", "selected", "calibration", "refit", "final", "notes"}


def protocol_id_for(body: dict) -> str:
    """Content-derived ID of a frozen protocol (decisions, evidence hashes and parent)."""
    return "p-" + hashlib.sha256(canonical_json(body).encode()).hexdigest()[:12]


def validate_protocol(protocol: dict) -> None:
    body = {k: protocol.get(k) for k in ("decisions", "evidence", "parent")}
    if protocol.get("protocol_id") != protocol_id_for(body):
        raise ValueError("protocol content does not match its ID: tampered or edited in place")
    unknown = set(protocol.get("decisions") or {}) - PROTOCOL_DECISIONS
    if unknown:
        raise ValueError(f"unknown decision groups {sorted(unknown)}")
