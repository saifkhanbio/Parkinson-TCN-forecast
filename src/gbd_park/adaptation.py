"""Convex two-parameter calibration with a frozen source predictor."""

import numpy as np
from scipy.optimize import minimize_scalar


def regularized_location(residuals, penalty):
    """Exact minimizer of mean(abs(residuals - b)) + penalty*b*b."""
    ordered = np.sort(np.asarray(residuals, dtype=float))
    n = len(ordered)
    right_derivative = 2 * penalty * ordered + (2 * np.arange(1, n + 1) - n) / n
    index = int(np.searchsorted(right_derivative, 0))
    if index == n:
        return -1 / (2 * penalty)
    left_derivative = 2 * penalty * ordered[index] + (2 * index - n) / n
    return float(ordered[index] if left_derivative <= 0 else (n - 2 * index) / (2 * penalty * n))


def fit_adaptation(observed_changes, predicted_changes, penalty):
    """Equal-window/age/horizon MAE plus penalty*(b0**2 + b1**2)."""
    error = np.asarray(observed_changes, dtype=float) - np.asarray(predicted_changes, dtype=float)
    if error.ndim != 2 or error.shape[1] != 5 or not len(error) or not np.isfinite(error).all() or penalty <= 0:
        raise ValueError("Adaptation requires finite five-horizon errors and positive penalty")
    residual = error.ravel()
    horizon = np.tile(np.arange(1, 6) / 5, len(error))
    initial = float(np.mean(np.abs(residual)))
    if initial == 0:
        return {"b0": 0.0, "b1": 0.0, "objective": 0.0, "unadapted_mae": 0.0, "penalty": penalty, "status": "ok"}
    bound = float(np.sqrt(initial / penalty))

    def objective(b1):
        b0 = regularized_location(residual - horizon * b1, penalty)
        return float(np.mean(np.abs(residual - b0 - horizon * b1)) + penalty * (b0*b0 + b1*b1))

    result = minimize_scalar(objective, bounds=(-bound, bound), method="bounded",
                             options={"xatol": 1e-12, "maxiter": 500})
    if not result.success:
        raise RuntimeError("Profile adaptation optimizer did not converge")
    b1 = min([float(result.x), 0.0, -bound, bound], key=objective)
    b0 = regularized_location(residual - horizon * b1, penalty)
    final = objective(b1)
    if not np.isfinite(final) or final > initial + 1e-10:
        raise RuntimeError("Adaptation objective did not improve on zero correction")
    return {"b0": b0, "b1": b1, "objective": final, "unadapted_mae": initial,
            "penalty": penalty, "status": "ok"}


def correction(calibration):
    return calibration["b0"] + calibration["b1"] * np.arange(1, 6) / 5
