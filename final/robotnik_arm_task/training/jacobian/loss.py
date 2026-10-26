"""BC+J objective: the baseline action loss plus lambda_J times the mean squared error of the student's
standardised 7x7 torque-versus-velocity Jacobian against the teachers' exact -kd * I (research §3.3, §4.4).

Valid rows come from the clean expert torque labels with a positive boundary tolerance; execution clip_flags are
never used. Wheels receive action loss only. The Jacobian is differentiated through to the weights.
"""
import hashlib

import torch

from evaluation.offline import torque_derivative_targets, velocity_jacobian
from training.baseline.loss import action_loss_parts
from training.common.engine import LossResult


def jacobian_term(model, x, y, normaliser, epsilon_tau: float, kd: float, torque_limit: float):
    """(mean squared error over valid entries or None, valid-entry fraction) on the given standardised rows."""
    raw = normaliser.inverse_y(y)[:, :7]
    target, valid = torque_derivative_targets(raw, normaliser, epsilon_tau, kd=kd, torque_limit=torque_limit)
    mask = valid[:, :, None].expand(-1, -1, 7)
    frac = mask.float().mean().item()
    if not mask.any():
        return None, frac
    J = velocity_jacobian(model, x)
    return ((J - target.to(J.dtype)) ** 2)[mask].mean(), frac


class JacobianObjective:
    def __init__(self, kind: str, lambda_j: float, epsilon_tau: float, subbatch_size: int = 256, kd: float = 30.0,
                 torque_limit: float = 30.0, huber_delta: float = 1.0):
        if lambda_j is None or lambda_j < 0:
            raise ValueError("lambda_j must be a frozen non-negative calibration")
        self.kind, self.lambda_j, self.epsilon_tau = kind, float(lambda_j), float(epsilon_tau)
        self.subbatch_size, self.kd, self.torque_limit, self.delta = int(subbatch_size), float(kd), float(torque_limit), huber_delta

    def __call__(self, model, x, y, normaliser, generator) -> LossResult:
        action, arm, wheel = action_loss_parts(model(x), y, self.kind, self.delta)
        metrics = {"action_loss": action.item(), "arm_loss": arm.item(), "wheel_loss": wheel.item()}
        if self.lambda_j == 0.0:
            return LossResult(action, metrics)
        n = min(self.subbatch_size, len(x))
        idx = torch.randperm(len(x), generator=generator, device=generator.device)[:n].to(x.device)
        jac, frac = jacobian_term(model, x[idx], y[idx], normaliser, self.epsilon_tau, self.kd, self.torque_limit)
        metrics.update(jacobian_loss=None if jac is None else jac.item(), valid_entry_frac=frac,
                       weighted_jacobian=None if jac is None else self.lambda_j * jac.item())
        total = action if jac is None else action + self.lambda_j * jac
        return LossResult(total, metrics)


def _tensor_sha(*tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def calibrate_lambda(model, x, y, normaliser, config: dict) -> dict:
    """lambda_J = ratio * L_action / L_J at this (seed-0) initialisation on a fixed fitting-only batch.

    config: kind, huber_delta, epsilon_tau, kd, torque_limit, calibration_ratio. The model is left unchanged.
    """
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        action = action_loss_parts(model(x), y, config["kind"], config["huber_delta"])[0].item()
        jac, frac = jacobian_term(model, x, y, normaliser, config["epsilon_tau"], config["kd"], config["torque_limit"])
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    j = None if jac is None else jac.item()
    if j is None or not (j > 0) or not torch.isfinite(torch.tensor(j)) or not (action > 0):
        raise ValueError(f"calibration denominator is zero or non-finite (action {action}, jacobian {j}); diagnose first")
    return {"lambda_j": config["calibration_ratio"] * action / j, "action_loss": action, "jacobian_loss": j,
            "ratio": config["calibration_ratio"], "valid_entry_frac": frac, "batch_rows": int(len(x)),
            "batch_sha256": _tensor_sha(x, y), "init_sha256": _tensor_sha(*state.values()),
            "normaliser_sha256": normaliser.sha256()}
