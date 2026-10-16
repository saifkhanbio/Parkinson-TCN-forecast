"""Scoring and time-restricted local-family selection; no model fitting."""

import numpy as np
import pandas as pd


KEYS = ["target", "sex", "age", "outcome", "forecast_year"]


def score_forecasts(forecasts, truth, ages, horizons, maximum_verification_year):
    if "observed_rate" in forecasts:
        raise ValueError("Prediction ledger must not already contain verification values")
    if forecasts.forecast_year.max() > maximum_verification_year:
        raise ValueError("Forecast crosses the permitted scoring cutoff")
    if not forecasts.forecast_year.eq(forecasts.origin + forecasts.horizon).all():
        raise ValueError("Forecast year does not equal origin plus horizon")
    keys = ["target", "outcome", "origin", "sex", "family", "setting_id", "age", "horizon"]
    if forecasts.duplicated(keys).any():
        raise ValueError("Duplicate forecast keys")
    for _, group in forecasts.groupby(["target", "outcome", "origin", "sex", "family", "setting_id"]):
        if set(zip(group.age, group.horizon)) != {(a, h) for a in ages for h in horizons}:
            raise ValueError("Missing or unexpected age/horizon forecast cells")
    allowed_truth = truth[truth.forecast_year.le(maximum_verification_year)]
    if allowed_truth.duplicated(KEYS).any():
        raise ValueError("Duplicate truth keys")
    scored = forecasts.merge(allowed_truth[KEYS + ["observed_rate"]], on=KEYS, how="left", validate="many_to_one")
    if scored.observed_rate.isna().any():
        raise ValueError("Unmatched verification values")
    if not np.isfinite(scored[["prediction", "observed_rate"]]).all().all() or not (scored[["prediction", "observed_rate"]] > 0).all().all():
        raise ValueError("Scoring requires finite positive rates")
    scored["absolute_log_error"] = np.abs(np.log(scored.prediction) - np.log(scored.observed_rate))
    scored["absolute_rate_error"] = np.abs(scored.prediction - scored.observed_rate)
    return scored


def select_settings(scored, config, origin, sex, family):
    eligible = scored.loc[(scored.sex == sex) & (scored.family == family)
                          & scored.origin.ge(config["calendar"]["inner_first_origin"])
                          & (scored.origin + 5).le(origin) & scored.horizon.eq(5)]
    if eligible.empty:
        raise ValueError("No completed inner validation windows")
    expected_origins = set(range(config["calendar"]["inner_first_origin"], origin - 4))
    for _, group in eligible.groupby("setting_id"):
        if set(group.origin) != expected_origins or len(group) != len(expected_origins) * len(config["ages"]):
            raise ValueError("Incomplete inner validation grid")
    losses = eligible.groupby("setting_id", as_index=False).agg(
        loss=("absolute_log_error", "mean"), parameter_count=("parameter_count", "mean"), grid_order=("grid_order", "first"))
    winner = losses.sort_values(["loss", "parameter_count", "grid_order"]).iloc[0]
    return str(winner.setting_id), float(winner.loss), sorted(expected_origins)


def select_family(development_scores, config, fit_origin, sex):
    allowed = [o for o in config["calendar"]["selection_origins"] if o + 5 <= fit_origin]
    rows = development_scores.loc[development_scores.origin.isin(allowed)
                                   & development_scores.sex.eq(sex) & development_scores.horizon.eq(5)]
    if not allowed:
        raise ValueError("No completed family-selection origins")
    for family in config["models"]["local_order"]:
        group = rows[rows.family.eq(family)]
        if len(group) != len(allowed) * len(config["ages"]):
            raise ValueError("Incomplete family selection evidence")
    losses = rows.groupby("family", as_index=False).agg(
        loss=("absolute_log_error", "mean"), parameter_count=("parameter_count", "mean"))
    losses["family_order"] = losses.family.map({f: i for i, f in enumerate(config["models"]["local_order"])})
    winner = losses.sort_values(["loss", "parameter_count", "family_order"]).iloc[0]
    return {"fit_origin": fit_origin, "sex": sex, "selected_family": winner.family,
            "selection_loss": float(winner.loss), "selection_origins": "|".join(map(str, allowed)),
            "last_selection_target_year": max(allowed) + 5}
