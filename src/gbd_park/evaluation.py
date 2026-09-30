"""Frozen-comparator roles and descriptive Saudi primary/reliability metrics.

Age cells, sexes, countries, and overlapping forecast windows are not treated
as independent participants. No hypothesis tests are implemented here.
"""

import numpy as np
import pandas as pd


COORDS = ["sex", "age", "horizon"]
GROUP_KEYS = ["target", "outcome", "origin", "family"]
WEIGHT_KEYS = ["target", "outcome", "origin", "sex", "age"]
COMPARATOR_ROLES = ["local_champion", "nonneural_champion"]


def _check_values(frame, scored=False):
    fields = ["prediction"] + (["observed_rate", "absolute_log_error", "absolute_rate_error"] if scored else [])
    if not set(fields).issubset(frame.columns):
        raise ValueError("Forecast values or required error columns are missing")
    if not np.isfinite(frame[fields].to_numpy(dtype=float)).all() or not frame.prediction.gt(0).all():
        raise ValueError("Forecast rates and errors must be finite, with positive point rates")
    if "log_prediction" in frame:
        logs = frame.log_prediction.to_numpy(dtype=float)
        if not np.isfinite(logs).all() or not np.allclose(logs, np.log(frame.prediction), atol=1e-10, rtol=1e-12):
            raise ValueError("Log predictions do not match positive point predictions")
    if scored:
        if not frame.observed_rate.gt(0).all():
            raise ValueError("Verification rates must be positive")
        expected_log = np.abs(np.log(frame.prediction.to_numpy()) - np.log(frame.observed_rate.to_numpy()))
        expected_rate = np.abs(frame.prediction.to_numpy() - frame.observed_rate.to_numpy())
        if not np.allclose(frame.absolute_log_error, expected_log, rtol=1e-12, atol=1e-12):
            raise ValueError("Absolute log errors do not match forecast and verification rates")
        if not np.allclose(frame.absolute_rate_error, expected_rate, rtol=1e-12, atol=1e-12):
            raise ValueError("Rate errors do not match forecast and verification rates")


def _check_grid(frame, config, scored=False, horizons=None):
    required = set(GROUP_KEYS + COORDS + ["forecast_year"])
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("A nonempty forecast ledger with all coordinate keys is required")
    keys = GROUP_KEYS + COORDS
    if frame[keys + ["forecast_year"]].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Missing or duplicate forecast keys")
    if not frame.forecast_year.eq(frame.origin + frame.horizon).all():
        raise ValueError("Forecast year must equal origin plus horizon")
    if not frame.target.eq(config["primary_target"]).all() or not frame.outcome.eq("prevalence").all():
        raise ValueError("Stage-5 evaluation is restricted to target-country prevalence")
    if not frame.forecast_year.le(config["calendar"]["history_end"]).all():
        raise ValueError("Forecast verification exceeds the available outcome calendar")
    horizons = config["calendar"]["horizons"] if horizons is None else horizons
    expected = {(sex, age, horizon) for sex in config["sexes"] for age in config["ages"] for horizon in horizons}
    for _, group in frame.groupby(GROUP_KEYS, sort=False):
        if len(group) != len(expected) or set(zip(group.sex, group.age, group.horizon)) != expected:
            raise ValueError("Incomplete or unexpected age-sex-horizon forecast grid")
    _check_values(frame, scored=scored)


def champion_forecasts(predictions, mappings, config):
    """Create two comparison-role views from saved, origin-specific family choices.

    Return only champion rows, for explicit concatenation with the underlying
    family forecasts. Values and historical setting IDs remain unchanged.
    A role is a duplicate view for comparison, not an additional forecast fit.
    """
    if "observed_rate" in predictions:
        raise ValueError("Construct comparison roles before joining verification outcomes")
    if predictions.family.isin(COMPARATOR_ROLES).any():
        raise ValueError("Champion roles already exist in the supplied forecast ledger")
    _check_grid(predictions, config)
    required = {"fit_origin", "role", "sex", "source_family", "last_selection_target_year"}
    if not required.issubset(mappings.columns):
        raise ValueError("Saved champion mappings are incomplete")
    origins = sorted(predictions.origin.unique())
    selected = mappings.loc[mappings.fit_origin.isin(origins)].copy()
    keys = ["fit_origin", "role", "sex"]
    expected = {(origin, role, sex) for origin in origins for role in COMPARATOR_ROLES for sex in config["sexes"]}
    if (selected[keys + ["source_family", "last_selection_target_year"]].isna().any().any()
            or selected.duplicated(keys).any() or len(selected) != len(expected)
            or set(map(tuple, selected[keys].to_numpy())) != expected):
        raise ValueError("Champion mappings must specify each role and both sexes at every origin")
    if not selected.last_selection_target_year.le(selected.fit_origin).all():
        raise ValueError("Champion mapping used outcomes after its fit origin")
    rows = []
    expected_cells = {(age, horizon) for age in config["ages"] for horizon in config["calendar"]["horizons"]}
    for mapping in selected.itertuples():
        allowed = config["models"]["local_order" if mapping.role == "local_champion" else "nonneural_order"]
        if mapping.source_family not in allowed:
            raise ValueError("Champion source family is outside its declared comparator group")
        part = predictions.loc[predictions.origin.eq(mapping.fit_origin) & predictions.sex.eq(mapping.sex)
                               & predictions.family.eq(mapping.source_family)].copy()
        if len(part) != len(expected_cells) or set(zip(part.age, part.horizon)) != expected_cells:
            raise ValueError("Selected champion source forecasts are incomplete")
        if "setting_id" in part and part.setting_id.nunique(dropna=False) != 1:
            raise ValueError("Champion source contains multiple settings for the same sex and origin")
        part["source_family"] = mapping.source_family
        part["family"] = mapping.role
        part["last_family_selection_target_year"] = mapping.last_selection_target_year
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    _check_grid(result, config)
    return result


def origin_population_weights(panel, config, origins):
    """Infer origin-year 45+ populations from prevalence Number/Rate values.

    Weights sum to one over the eleven ages within each sex and origin. They
    are identical across methods and horizons and use no later population
    observations. The denominator remains a modeled GBD-vintage quantity.
    """
    origins = list(origins)
    if (not origins or len(set(origins)) != len(origins)
            or any(isinstance(origin, bool) or not isinstance(origin, (int, np.integer)) for origin in origins)):
        raise ValueError("Population weighting needs distinct integer fit origins")
    required = {"location_name", "outcome", "year", "sex", "age", "count", "rate"}
    if not required.issubset(panel.columns):
        raise ValueError("Population source must include prevalence Number and Rate columns")
    source = panel.loc[panel.location_name.eq(config["primary_target"]) & panel.outcome.eq("prevalence")
                       & panel.year.isin(origins)].copy()
    keys = ["year", "sex", "age"]
    expected = {(origin, sex, age) for origin in origins for sex in config["sexes"] for age in config["ages"]}
    if source.duplicated(keys).any() or len(source) != len(expected) or set(map(tuple, source[keys].to_numpy())) != expected:
        raise ValueError("Origin population cells are missing, duplicated, or unexpected")
    if not np.isfinite(source[["count", "rate"]]).all().all() or not source[["count", "rate"]].gt(0).all().all():
        raise ValueError("Population inference requires positive finite Number and Rate values")
    population = source["count"].to_numpy(dtype=float) / source.rate.to_numpy(dtype=float) * 100000
    if not np.isfinite(population).all() or (population <= 0).any():
        raise ValueError("Invalid count/rate-implied population")
    if "implied_population" in source:
        existing = source.implied_population.to_numpy(dtype=float)
        if not np.isfinite(existing).all() or not np.allclose(existing, population, rtol=1e-10, atol=1e-7):
            raise ValueError("Stored implied population disagrees with the prevalence Number/Rate denominator")
    result = source[["location_name", "outcome", "year", "sex", "age"]].rename(
        columns={"location_name": "target", "year": "origin"})
    result["origin_population"] = population
    totals = result.groupby(["target", "outcome", "origin", "sex"]).origin_population.transform("sum")
    result["origin_population_weight"] = result.origin_population / totals
    result["population_source"] = "origin_year_GBD_prevalence_Number_divided_by_Rate_times_100000"
    return result.sort_values(["origin", "sex", "age"]).reset_index(drop=True)


def summarize_scores(scored, weights, config):
    """Keep origin/horizon results, five-origin reliability, and primary views separate."""
    _check_grid(scored, config, scored=True)
    expected_origins = set(config["calendar"]["reliability_origins"])
    for _, group in scored.groupby(["target", "outcome", "family"], sort=False):
        if set(group.origin) != expected_origins:
            raise ValueError("Reliability summaries require each prescribed origin exactly once")
    required = set(WEIGHT_KEYS + ["origin_population", "origin_population_weight"])
    if not required.issubset(weights.columns):
        raise ValueError("Origin population weights are missing required columns")
    matched_weights = weights.loc[weights.origin.isin(expected_origins)
                                  & weights.target.eq(config["primary_target"]) & weights.outcome.eq("prevalence")].copy()
    if matched_weights.duplicated(WEIGHT_KEYS).any():
        raise ValueError("Duplicate origin population weights")
    expected_weight_keys = scored[WEIGHT_KEYS].drop_duplicates()
    if set(map(tuple, matched_weights[WEIGHT_KEYS].to_numpy())) != set(map(tuple, expected_weight_keys.to_numpy())):
        raise ValueError("Population weights do not cover the complete scoring grid")
    numbers = matched_weights[["origin_population", "origin_population_weight"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers <= 0).any():
        raise ValueError("Origin populations and weights must be positive and finite")
    sums = matched_weights.groupby(["target", "outcome", "origin", "sex"]).origin_population_weight.sum()
    if not np.allclose(sums, 1, atol=1e-12, rtol=1e-12):
        raise ValueError("Origin age weights must sum to one separately for each sex")
    denominator = matched_weights.groupby(["target", "outcome", "origin", "sex"]).origin_population.transform("sum")
    if not np.allclose(matched_weights.origin_population_weight, matched_weights.origin_population / denominator,
                       atol=1e-12, rtol=1e-12):
        raise ValueError("Weights disagree with their common origin-population definition")
    merged = scored.merge(matched_weights[WEIGHT_KEYS + ["origin_population", "origin_population_weight"]],
                          on=WEIGHT_KEYS, how="left", validate="many_to_one")
    merged["weighted_absolute_log_error"] = merged.absolute_log_error * merged.origin_population_weight
    merged["fallback"] = merged.status.eq("fallback").astype(int) if "status" in merged else 0
    by_origin = merged.groupby(["target", "outcome", "origin", "sex", "family", "horizon"], as_index=False).agg(
        mean_age_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        population_weighted_absolute_log_error=("weighted_absolute_log_error", "sum"),
        fallback_cells=("fallback", "sum"), age_cells=("absolute_log_error", "size"))
    by_origin_all = by_origin.groupby(["target", "outcome", "origin", "sex", "family"], as_index=False).agg(
        mean_absolute_log_error=("mean_age_absolute_log_error", "mean"), mean_rate_mae=("rate_mae", "mean"),
        mean_population_weighted_absolute_log_error=("population_weighted_absolute_log_error", "mean"),
        fallback_cells=("fallback_cells", "sum"), horizons=("horizon", "nunique"))
    reliability = by_origin.groupby(["target", "outcome", "sex", "family", "horizon"], as_index=False).agg(
        mean_absolute_log_error=("mean_age_absolute_log_error", "mean"), mean_rate_mae=("rate_mae", "mean"),
        mean_population_weighted_absolute_log_error=("population_weighted_absolute_log_error", "mean"),
        n_origins=("origin", "nunique"), fallback_cells=("fallback_cells", "sum"))
    reliability_all = by_origin_all.groupby(["target", "outcome", "sex", "family"], as_index=False).agg(
        mean_absolute_log_error=("mean_absolute_log_error", "mean"), mean_rate_mae=("mean_rate_mae", "mean"),
        mean_population_weighted_absolute_log_error=("mean_population_weighted_absolute_log_error", "mean"),
        n_origins=("origin", "nunique"), fallback_cells=("fallback_cells", "sum"))
    origin, horizon = config["calendar"]["primary_origin"], config["calendar"]["primary_horizon"]
    return {"by_origin": by_origin, "by_origin_all_horizons": by_origin_all,
            "reliability_by_horizon": reliability, "reliability_all_horizons": reliability_all,
            "primary_by_family": by_origin.loc[by_origin.origin.eq(origin) & by_origin.horizon.eq(horizon)].copy(),
            "primary_age_scores": merged.loc[merged.origin.eq(origin) & merged.horizon.eq(horizon)].copy()}


def primary_contrasts(scored, config):
    """Evaluate four strict endpoint contrasts; retain zero-denominator cases."""
    origin, horizon = config["calendar"]["primary_origin"], config["calendar"]["primary_horizon"]
    endpoint = scored.loc[scored.origin.eq(origin) & scored.horizon.eq(horizon)
                          & scored.family.isin(["tcn_adapted"] + COMPARATOR_ROLES)].copy()
    if set(endpoint.family) != {"tcn_adapted", *COMPARATOR_ROLES}:
        raise ValueError("Primary evaluation requires the TCN and both champion roles")
    _check_grid(endpoint, config, scored=True, horizons=[horizon])
    contrasts, age_tables, success_by_sex = [], [], {}
    for sex in config["sexes"]:
        neural = endpoint.loc[endpoint.sex.eq(sex) & endpoint.family.eq("tcn_adapted")].set_index("age").reindex(config["ages"])
        tcn_error = float(neural.absolute_log_error.mean())
        successes = []
        for role in COMPARATOR_ROLES:
            comparator = endpoint.loc[endpoint.sex.eq(sex) & endpoint.family.eq(role)].set_index("age").reindex(config["ages"])
            if not np.allclose(neural.observed_rate, comparator.observed_rate, atol=0, rtol=1e-12):
                raise ValueError("Primary comparators must share the same verification rates")
            source_family = role
            if "source_family" in comparator:
                if comparator.source_family.isna().any() or comparator.source_family.nunique() != 1:
                    raise ValueError("Primary champion must retain one source family per sex")
                source_family = str(comparator.source_family.iloc[0])
            comparator_error = float(comparator.absolute_log_error.mean())
            difference = tcn_error - comparator_error
            relative = (comparator_error - tcn_error) / comparator_error if comparator_error > 0 else np.nan
            lower = bool(tcn_error < comparator_error)
            successes.append(lower)
            contrasts.append({"target": config["primary_target"], "outcome": "prevalence", "origin": origin,
                              "horizon": horizon, "forecast_year": origin + horizon, "sex": sex,
                              "comparator": role, "comparator_source_family": source_family,
                              "tcn_error": tcn_error, "comparator_error": comparator_error,
                              "absolute_loss_difference": difference, "relative_improvement": relative,
                              "relative_improvement_percent": relative * 100, "strictly_lower": lower})
            ages = pd.DataFrame({"age": config["ages"], "observed_rate": neural.observed_rate.to_numpy(),
                                 "tcn_prediction": neural.prediction.to_numpy(), "comparator_prediction": comparator.prediction.to_numpy(),
                                 "tcn_absolute_log_error": neural.absolute_log_error.to_numpy(),
                                 "comparator_absolute_log_error": comparator.absolute_log_error.to_numpy()})
            ages["origin"], ages["horizon"], ages["sex"] = origin, horizon, sex
            ages["comparator"], ages["comparator_source_family"] = role, source_family
            ages["absolute_loss_difference"] = ages.tcn_absolute_log_error - ages.comparator_absolute_log_error
            ages["harmed"] = ages.absolute_loss_difference > 0
            age_tables.append(ages)
        success_by_sex[sex] = all(successes)
    verdict = {"origin": origin, "horizon": horizon, "forecast_year": origin + horizon,
               "joint_success": all(success_by_sex.values()), "success_by_sex": success_by_sex,
               "required_contrasts": len(config["sexes"]) * len(COMPARATOR_ROLES),
               "criterion": "Strictly lower equal-age absolute log error for each sex against both frozen champions",
               "statistical_significance_claim": False, "clinical_significance_claim": False}
    return pd.DataFrame(contrasts), pd.concat(age_tables, ignore_index=True), verdict


def matched_tcn_harm(scored, config):
    """Compare correction against the identical unadapted neural ensemble."""
    selected = scored.loc[scored.family.isin(["tcn_adapted", "tcn_unadapted"])].copy()
    if set(selected.family) != {"tcn_adapted", "tcn_unadapted"}:
        raise ValueError("Matched adaptation requires adapted and unadapted TCN forecasts")
    _check_grid(selected, config, scored=True)
    if "ensemble_fingerprint" not in selected or selected.ensemble_fingerprint.isna().any():
        raise ValueError("Matched adaptation requires saved ensemble fingerprints")
    keys = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]
    adapted = selected.loc[selected.family.eq("tcn_adapted"), keys + ["ensemble_fingerprint", "observed_rate", "absolute_log_error"]]
    unadapted = selected.loc[selected.family.eq("tcn_unadapted"), keys + ["ensemble_fingerprint", "observed_rate", "absolute_log_error"]]
    paired = adapted.merge(unadapted, on=keys, how="outer", suffixes=("_adapted", "_unadapted"), validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Matched adaptation is missing a counterpart forecast")
    if not paired.ensemble_fingerprint_adapted.eq(paired.ensemble_fingerprint_unadapted).all():
        raise ValueError("Adaptation comparison uses different source ensembles")
    if not np.allclose(paired.observed_rate_adapted, paired.observed_rate_unadapted, rtol=1e-12, atol=0):
        raise ValueError("Adaptation counterparts use different verification rates")
    pairs = paired[keys].copy()
    pairs["ensemble_fingerprint"] = paired.ensemble_fingerprint_adapted
    pairs["adapted_error"] = paired.absolute_log_error_adapted
    pairs["matched_unadapted_error"] = paired.absolute_log_error_unadapted
    pairs["adaptation_loss_change"] = pairs.adapted_error - pairs.matched_unadapted_error
    pairs["harmed"] = pairs.adaptation_loss_change > 0
    summary = pairs.groupby(["target", "outcome", "origin", "sex", "horizon"], as_index=False).agg(
        mean_loss_change=("adaptation_loss_change", "mean"), harmed_cells=("harmed", "sum"), cells=("harmed", "size"))
    summary["harm_fraction"] = summary.harmed_cells / summary.cells
    return pairs, summary
