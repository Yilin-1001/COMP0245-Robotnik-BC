"""Offline control diagnostics for any MLP, independent of its training objective (research §5).

- Physical-unit arm/wheel errors overall and per diagnostic phase (navigation, approach, hold).
- Signed per-episode torque bias over the final hold window; the gate uses the maximum absolute episode bias.
- Learned damping: the student's d tau / d qd against the teachers' exact -kd * I on valid rows.

Derivative targets come from the clean expert torque labels. Execution clip_flags are never a saturation mask:
without pre-clip torques, rows within epsilon_tau of the limit are omitted rather than labelled saturated.
"""
import numpy as np
import torch

PHASE_NAMES = {0: "navigation", 1: "approach", 2: "hold"}
QD_FEATURES = [f"qd_mes_{i}" for i in range(7)]  # velocity_jacobian differentiates these input columns
DIAGNOSTIC_SEED = 0  # fixed draw of hold rows for the damping diagnostic, identical for every model


def torque_derivative_targets(raw_torque: torch.Tensor, normaliser, epsilon_tau: float,
                              preclip_torque: torch.Tensor | None = None, kd: float = 30.0,
                              torque_limit: float = 30.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardised 7x7 targets of d tau_std / d qd_std and the valid-row mask [B,7].

    Physical target J_ij = -kd * delta_ij for an unclipped joint i and 0 for a verified strictly saturated one;
    standardised J_ij * sigma_qd_j / sigma_tau_i. Boundary rows are omitted (mask False).
    """
    if epsilon_tau <= 0:
        raise ValueError("epsilon_tau must be positive")
    if list(normaliser.feature_names[7:14]) != QD_FEATURES:
        raise ValueError(f"feature columns 7:14 must be {QD_FEATURES}, got {normaliser.feature_names[7:14]}")
    raw = torch.as_tensor(raw_torque)
    dtype, device = (raw.dtype if raw.is_floating_point() else torch.float32), raw.device
    if preclip_torque is None:
        unclipped = raw.abs() < torque_limit - epsilon_tau
        saturated = torch.zeros_like(unclipped)
    else:
        pre = torch.as_tensor(preclip_torque, device=device)
        unclipped = pre.abs() < torque_limit - epsilon_tau
        saturated = pre.abs() > torque_limit + epsilon_tau
    valid = unclipped | saturated
    sigma_qd = torch.as_tensor(normaliser.x_scale[7:14], dtype=dtype, device=device)
    sigma_tau = torch.as_tensor(normaliser.y_scale[:7], dtype=dtype, device=device)
    factor = sigma_qd[None, :] / sigma_tau[:, None]                      # [i, j] = sigma_qd_j / sigma_tau_i
    eye = torch.eye(7, dtype=dtype, device=device)
    target = -kd * eye * factor * unclipped.to(dtype)[:, :, None]       # row i scaled by its unclipped flag
    return target, valid


def velocity_jacobian(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-sample d (standardised torque outputs) / d (standardised qd inputs, columns 7:14): [B,7,7].

    Differentiable with respect to the model parameters (used by the BC+J loss); callers detach for diagnostics.
    """
    from torch.func import jacrev, vmap

    def torque(row):
        return model(row.unsqueeze(0))[0, :7]

    return vmap(jacrev(torque))(x)[:, :, 7:14]


def _predict(model, x_std: np.ndarray, chunk: int = 65536) -> np.ndarray:
    device = next(iter(model.parameters()), torch.zeros(())).device
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_std), chunk):
            out.append(model(torch.as_tensor(x_std[i:i + chunk], dtype=torch.float32, device=device)).double().cpu().numpy())
    model.train(was_training)
    return np.concatenate(out) if out else np.zeros((0, 9))


def _err_summary(err: np.ndarray) -> dict:
    if len(err) == 0:
        return {"rows": 0, "arm_rmse_nm": None, "arm_mae_nm": None, "wheel_rmse_rad_s": None, "wheel_mae_rad_s": None}
    return {"rows": int(len(err)), "arm_rmse_nm": float(np.sqrt((err[:, :7] ** 2).mean())),
            "arm_mae_nm": float(np.abs(err[:, :7]).mean()), "wheel_rmse_rad_s": float(np.sqrt((err[:, 7:] ** 2).mean())),
            "wheel_mae_rad_s": float(np.abs(err[:, 7:]).mean())}


def _median(v) -> float | None:
    return float(np.median(v)) if len(v) else None


def offline_metrics(model: torch.nn.Module, cache, normaliser, config: dict) -> dict:
    """Errors, hold bias and damping diagnostics on one cache (normally the 32 validation episodes)."""
    ev, limit = config["evaluation"], float(config["interface"]["torque_limit"])
    hold_steps = int(round(ev["hold_window_s"] * config["interface"]["control_hz"]))
    y = np.asarray(cache.y, dtype=np.float64)
    pred = np.asarray(normaliser.inverse_y(_predict(model, normaliser.transform_x(np.asarray(cache.x, np.float64)))))
    err = pred - y
    out = _err_summary(err)
    out["arm_rmse_per_joint_nm"] = np.sqrt((err[:, :7] ** 2).mean(axis=0)).tolist() if len(err) else []
    phase = cache.diagnostics.get("phase")
    out["phase"] = {name: _err_summary(err[phase == code]) for code, name in PHASE_NAMES.items()} if phase is not None else {}

    episodes, biases = {}, []
    for eid, lo, hi in zip(cache.episode_ids, cache.offsets[:-1], cache.offsets[1:]):
        b = err[max(lo, hi - hold_steps):hi, :7].mean(axis=0)
        episodes[eid] = b.tolist()
        biases.append(b)
    abs_b = np.abs(np.array(biases)) if biases else np.zeros((0, 7))
    max_abs = float(abs_b.max()) if abs_b.size else None
    out["hold_bias"] = {"episodes": episodes, "window_steps": hold_steps,
                        "median_abs_per_joint_nm": np.median(abs_b, axis=0).tolist() if abs_b.size else [],
                        "max_abs_per_joint_nm": abs_b.max(axis=0).tolist() if abs_b.size else [],
                        "max_abs_nm": max_abs, "gate_nm": ev["hold_bias_gate_nm"],
                        "gate_pass": bool(max_abs is not None and max_abs <= ev["hold_bias_gate_nm"])}
    out["damping"] = damping_diagnostics(model, cache, normaliser, config, phase)
    return out


def damping_diagnostics(model, cache, normaliser, config: dict, phase=None) -> dict:
    ev, eps = config["evaluation"], float(config["loss"]["epsilon_tau"])
    limit = float(config["interface"]["torque_limit"])
    rows = np.flatnonzero(phase == 2) if phase is not None else np.arange(len(cache.x))
    rng = np.random.default_rng(DIAGNOSTIC_SEED)
    if len(rows) > ev["jacobian_samples"]:
        rows = np.sort(rng.choice(rows, ev["jacobian_samples"], replace=False))
    report = {"sampled_rows": int(len(rows)), "valid_rows": 0, "omitted_rows": int(len(rows)), "valid_entries": 0,
              "saturated_entries": 0, "preclip_available": False, "median_diagonal": None,
              "median_diagonal_per_joint": [None] * 7, "frac_below_minus33": None, "cross_joint_rms": None,
              "diagonal_rmse_vs_teacher": None, "teacher_diagonal": -30.0}
    if not len(rows):
        return report
    device = next(iter(model.parameters()), torch.zeros(())).device
    x = torch.as_tensor(normaliser.transform_x(np.asarray(cache.x[rows], np.float64)), dtype=torch.float32, device=device)
    J_std = velocity_jacobian(model, x).detach().double().cpu().numpy()
    factor = normaliser.y_scale[:7][:, None] / normaliser.x_scale[7:14][None, :]   # back to N*m*s/rad
    J = J_std * factor[None]
    _, valid = torque_derivative_targets(torch.as_tensor(cache.y[rows, :7], dtype=torch.float64), normaliser, eps,
                                         torque_limit=limit)
    valid = valid.numpy()
    diag = np.diagonal(J, axis1=1, axis2=2)                                   # [B,7]
    off = J[~np.eye(7, dtype=bool)[None].repeat(len(J), 0)].reshape(len(J), 7, 6)
    vd = diag[valid]
    report.update(valid_rows=int(valid.any(axis=1).sum()), omitted_rows=int((~valid.any(axis=1)).sum()),
                  valid_entries=int(valid.sum()))
    if vd.size:
        report.update(median_diagonal=float(np.median(vd)),
                      median_diagonal_per_joint=[_median(diag[valid[:, j], j]) for j in range(7)],
                      frac_below_minus33=float(np.mean(vd < -33.0)),
                      cross_joint_rms=float(np.sqrt((off[valid] ** 2).mean())),
                      diagonal_rmse_vs_teacher=float(np.sqrt(((vd + 30.0) ** 2).mean())))
    return report
