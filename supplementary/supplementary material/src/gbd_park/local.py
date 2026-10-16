"""Saudi-only log-rate baselines. All fitting is restricted to supplied history."""

import itertools
import json
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def settings_grid(config):
    result = []
    for family in config["models"]["local_order"]:
        if family == "log_trend":
            settings = [{"window": w} for w in config["models"]["log_trend_windows"]]
        elif family == "age_smooth_trend":
            settings = [{"window": w, "penalty": p} for w, p in itertools.product(
                config["models"]["age_smooth_windows"], config["models"]["age_smooth_penalties"])]
        else:
            settings = [{}]
        for setting in settings:
            ident = family + "__" + "__".join(f"{k}={v}" for k, v in setting.items())
            result.append({"family": family, "setting_id": ident.rstrip("_"),
                           "setting": setting, "grid_order": len(result)})
    return result


def select_history(panel, config, origin, sex, target="Saudi Arabia", outcome="prevalence"):
    history = panel.loc[
        panel.location_name.eq(target) & panel.sex.eq(sex) & panel.outcome.eq(outcome)
        & panel.year.between(config["calendar"]["history_start"], origin)
    ].copy()
    if history.duplicated(["year", "age"]).any():
        raise ValueError("Duplicate history keys")
    matrix = history.pivot(index="year", columns="age", values="rate").reindex(
        index=range(config["calendar"]["history_start"], origin + 1), columns=config["ages"])
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing, nonfinite or nonpositive history")
    if len(values) < 8:
        raise ValueError("At least eight history years required")
    return matrix.index.to_numpy(), np.log(values)


def warning_text(records):
    return " | ".join(sorted({f"{w.category.__name__}: {w.message}" for w in records}))


def arima_forecast(values, horizon, grid):
    """Training-AICc order selection, retaining every candidate's disposition."""
    audit = []
    best = None
    for p, d, q in itertools.product(grid["p"], grid["d"], grid["q"]):
        if p + q > grid["max_p_plus_q"]:
            continue
        row = {"p": p, "d": d, "q": q, "trend": "c" if d == 0 else "t",
               "status": "failed", "aicc": None, "reason": "", "warnings": ""}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted = ARIMA(values, order=(p, d, q), trend=row["trend"],
                               enforce_stationarity=True, enforce_invertibility=True).fit(
                    method="statespace", method_kwargs={"maxiter": 200, "disp": 0})
                prediction = np.asarray(fitted.forecast(horizon), dtype=float)
            row["warnings"] = warning_text(caught)
            row["converged"] = bool(fitted.mle_retvals.get("converged", False))
            aicc = float(fitted.aicc)
            row["aicc"] = aicc if np.isfinite(aicc) else None
            if not row["converged"]:
                raise ValueError("Optimizer did not converge")
            if not np.isfinite(aicc) or not np.isfinite(prediction).all():
                raise ValueError("Nonfinite AICc or forecast")
            if np.any(np.abs(prediction) > 700):
                raise ValueError("Forecast outside safe exponential range")
            row["status"] = "eligible"
            key = (aicc, len(fitted.params), p, d, q)
            if best is None or key < best[0]:
                best = (key, prediction, fitted, len(audit))
        except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError) as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        audit.append(row)
    if best is None:
        return np.repeat(values[-1], horizon), {
            "status": "fallback", "reason": "No converged finite ARIMA candidate",
            "parameter_count": 0, "fitted_parameters": {}}, audit
    _, prediction, fitted, winner = best
    audit[winner]["status"] = "selected"
    return prediction, {"status": "ok", "reason": "", "parameter_count": len(fitted.params),
                        "fitted_parameters": dict(zip(fitted.param_names, map(float, fitted.params))),
                        "order": [audit[winner][k] for k in ["p", "d", "q"]],
                        "trend": audit[winner]["trend"], "aicc": audit[winner]["aicc"],
                        "warnings": audit[winner]["warnings"]}, audit


def age_smooth_forecast(years, values, horizons, penalty):
    """Mean squared log error + penalty * sum(second differences of slopes)^2."""
    n, ages = values.shape
    time = (years - years[-1]) / 10.0
    identity = np.eye(ages)
    design = np.column_stack([np.tile(identity, (n, 1)), np.kron(time[:, None], identity)])
    # The last column is the open-ended oldest age: no artificial numeric midpoint.
    differences = np.zeros((max(0, ages - 3), ages))
    for i in range(len(differences)):
        differences[i, i:i + 3] = [1, -2, 1]
    regularizer = np.column_stack([np.zeros_like(differences), differences])
    augmented = np.vstack([design / np.sqrt(n * ages), np.sqrt(penalty) * regularizer])
    response = np.r_[values.ravel() / np.sqrt(n * ages), np.zeros(len(differences))]
    coefficients = np.linalg.lstsq(augmented, response, rcond=None)[0]
    prediction = coefficients[:ages, None] + coefficients[ages:, None] * np.asarray(horizons)[None, :] / 10
    return prediction, coefficients


def forecast_setting(panel, config, origin, sex, spec, target="Saudi Arabia"):
    years, all_values = select_history(panel, config, origin, sex, target)
    values = all_values
    window = spec["setting"].get("window", "all")
    if window != "all":
        values, years = values[-window:], years[-window:]
    horizons = config["calendar"]["horizons"]
    family = spec["family"]
    ages = len(config["ages"])
    predictions = np.empty((ages, len(horizons)))
    metadata, audits = [], []
    if family == "age_smooth_trend":
        try:
            predictions, coefficients = age_smooth_forecast(years, values, horizons, spec["setting"]["penalty"])
            if not np.isfinite(predictions).all() or np.any(np.abs(predictions) > 700):
                raise ValueError("Nonfinite or unsafe age-smoothed forecast")
            metadata = [{"status": "ok", "reason": "", "parameter_count": 2,
                         "fitted_parameters": {"intercept": float(coefficients[a]),
                                               "slope_per_decade": float(coefficients[ages + a])}}
                        for a in range(ages)]
        except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError) as exc:
            predictions = np.repeat(all_values[-1, :, None], len(horizons), axis=1)
            metadata = [{"status": "fallback", "reason": f"{type(exc).__name__}: {exc}",
                         "parameter_count": 0, "fitted_parameters": {}} for _ in range(ages)]
    else:
        for a, age in enumerate(config["ages"]):
            y = values[:, a]
            info = {"status": "ok", "reason": "", "fitted_parameters": {}}
            try:
                if family == "persistence":
                    prediction = np.repeat(y[-1], len(horizons))
                    info["parameter_count"] = 0
                elif family == "log_trend":
                    matrix = np.column_stack([np.ones(len(years)), years - origin])
                    intercept, slope = np.linalg.lstsq(matrix, y, rcond=None)[0]
                    prediction = intercept + slope * np.asarray(horizons)
                    info.update(parameter_count=2, fitted_parameters={"intercept": float(intercept), "slope": float(slope)})
                elif family == "damped_ets":
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        fitted = ExponentialSmoothing(y, trend="add", damped_trend=True,
                            seasonal=None, initialization_method="estimated").fit(
                                optimized=True, remove_bias=False, use_brute=True, method="L-BFGS-B")
                        prediction = np.asarray(fitted.forecast(len(horizons)), dtype=float)
                    if not fitted.mle_retvals.get("success", False):
                        raise ValueError("ETS optimizer did not converge")
                    info.update(parameter_count=5, warnings=warning_text(caught), fitted_parameters={
                        key: float(fitted.params[key]) for key in ["smoothing_level", "smoothing_trend", "damping_trend", "initial_level", "initial_trend"]})
                elif family == "arima":
                    prediction, info, candidates = arima_forecast(y, len(horizons), config["models"]["arima"])
                    audits.extend(dict(item, age=age) for item in candidates)
                else:
                    raise KeyError(family)
                if not np.isfinite(prediction).all() or np.any(np.abs(prediction) > 700):
                    raise ValueError("Nonfinite or unsafe log forecast")
            except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError) as exc:
                prediction = np.repeat(all_values[-1, a], len(horizons))
                info.update(status="fallback", reason=f"{type(exc).__name__}: {exc}", parameter_count=0)
            predictions[a] = prediction
            metadata.append(info)
    if not np.isfinite(predictions).all():
        raise ValueError("Nonfinite forecast matrix")
    rows = []
    for a, age in enumerate(config["ages"]):
        for j, horizon in enumerate(horizons):
            rows.append({"target": target, "sex": sex, "age": age, "outcome": "prevalence",
                         "origin": origin, "horizon": horizon, "forecast_year": origin + horizon,
                         "family": family, "setting_id": spec["setting_id"], "grid_order": spec["grid_order"],
                         "log_prediction": float(predictions[a, j]), "prediction": float(np.exp(predictions[a, j])),
                         "history_start": int(years[0]), "history_end": origin,
                         "parameter_count": metadata[a]["parameter_count"],
                         "status": metadata[a]["status"], "fallback_reason": metadata[a]["reason"],
                         "setting_json": json.dumps(spec["setting"], sort_keys=True)})
    fits = [dict(meta, age=age, origin=origin, sex=sex, family=family, setting_id=spec["setting_id"])
            for age, meta in zip(config["ages"], metadata)]
    return rows, fits, [dict(a, origin=origin, sex=sex) for a in audits]
