import numpy as np


def regression_metrics(tau, tau_pred, regressor):
    """Return training-fit errors and rank-aware goodness-of-fit metrics."""
    tau = np.asarray(tau).reshape(-1)
    tau_pred = np.asarray(tau_pred).reshape(-1)
    regressor = np.asarray(regressor)
    if tau.size == 0 or tau.shape != tau_pred.shape:
        raise ValueError("Measured and predicted torques must be nonempty and equally sized.")
    if regressor.ndim != 2 or regressor.shape[0] != tau.size:
        raise ValueError("The regressor must have one row per torque measurement.")

    error = tau - tau_pred
    sse = float(error @ error)
    centered = tau - tau.mean()
    sst = float(centered @ centered)
    rank = int(np.linalg.matrix_rank(regressor))
    r_squared = 1 - sse / sst if sst > 0 else np.nan
    # No intercept is fitted; dependent columns count only through matrix rank.
    residual_dof = tau.size - rank
    adjusted = 1 - (1 - r_squared) * (tau.size - 1) / residual_dof if residual_dof > 0 else np.nan
    return {
        "rmse": float(np.sqrt(sse / tau.size)),
        "mae": float(np.mean(np.abs(error))),
        "r_squared": float(r_squared),
        "adjusted_r_squared": float(adjusted),
        "rank": rank,
        "parameters": regressor.shape[1],
    }
