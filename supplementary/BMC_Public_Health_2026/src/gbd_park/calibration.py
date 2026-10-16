"""Bounded exploratory dispersion calibration for frozen joint residual banks.

Only a positive, sex-specific multiplier is transferred or selected. Target
residual shapes, historical block identities, point forecasts and population
residuals are preserved. Selection uses completed development forecasts only.
"""

import copy

import numpy as np
import pandas as pd

from gbd_park.intervals import apply_bank, build_residual_bank, weighted_interval_score


COORDINATES = ["sex", "age", "horizon"]
UNDERLYING_FAMILIES = (
    "persistence", "log_trend", "damped_ets", "arima", "age_smooth_trend",
    "pooled_ridge", "pooled_boosting", "donor_ridge_unadapted", "donor_ridge_adapted",
    "donor_boosting_unadapted", "donor_boosting_adapted",
    "tcn_adapted", "tcn_unadapted", "tcn_intercept",
)
FACTORS = (1., 1.25, 1.5, 2., 3.)
GCC_COUNTRIES = {"Saudi Arabia", "Bahrain", "Kuwait", "Oman", "Qatar", "United Arab Emirates"}


def _amendment(amendment):
    factors = np.asarray(amendment["factor_grid"], dtype=float)
    if not np.array_equal(factors, np.asarray(FACTORS)):
        raise ValueError("Calibration candidates must be the frozen factors 1, 1.25, 1.5, 2, 3")
    if amendment["selection_horizon"] != 5 or amendment["selection_levels"] != [.5, .8]:
        raise ValueError("Calibration selection requires horizon-five log WIS at 50/80 levels")
    if amendment["development_origins"] != [2012, 2013] or amendment["cold_start_factor"] != 1:
        raise ValueError("Unexpected calibration development calendar or cold-start factor")
    if (amendment["penalty"] != .05 or amendment["normalization_floor"] != 1e-6
            or amendment["target_weight"] != .5):
        raise ValueError("The penalty, normalization floor and 50/50 borrowing weight are frozen")
    countries = list(amendment["countries"])
    if (len(countries) != 6 or set(countries) != GCC_COUNTRIES
            or set(amendment["outcomes"]) != {"prevalence", "incidence"}):
        raise ValueError("Calibration requires the six GCC countries and two specified outcomes")
    return factors


def scale_bank(bank, scales_by_sex, config):
    """Return an explicitly transformed copy; never change the original bank.

    Raw residuals, coordinate means and historical origin labels remain exact
    copies. ``unscaled_centered_residuals`` records the original residuals,
    while ``centered_residuals`` is the positive sex-wise scaled matrix used
    by the immutable ``apply_bank`` function. An already transformed bank is
    rejected to prevent silently compounding factors.
    """
    if set(scales_by_sex) != set(config["sexes"]):
        raise ValueError("Supply one scale for every configured sex")
    factors = {}
    for sex, value in scales_by_sex.items():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Width scales must be positive finite numbers")
        try:
            number = float(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Width scales must be positive finite numbers") from exc
        if not np.isfinite(number) or number <= 0:
            raise ValueError("Width scales must be positive finite numbers")
        factors[sex] = number
    if "unscaled_centered_residuals" in bank or "width_scales_by_sex" in bank:
        raise ValueError("Scale the original bank, not an already transformed bank")
    coords = pd.MultiIndex.from_product(
        [config["sexes"], config["ages"], config["calendar"]["horizons"]],
        names=COORDINATES).to_frame(index=False)
    if not coords.equals(bank["coords"]):
        raise ValueError("Bank coordinates disagree with the configured complete joint grid")
    n_blocks = int(bank["n_blocks"])
    origins = bank["origins"]
    if (len(origins) != n_blocks or len(set(origins)) != n_blocks
            or any(origin+max(config["calendar"]["horizons"]) > bank["fit_origin"] for origin in origins)):
        raise ValueError("Bank contains duplicate or unavailable residual origins")
    centered = np.asarray(bank["centered_residuals"], dtype=float)
    raw = np.asarray(bank["raw_residuals"], dtype=float)
    center = np.asarray(bank["center"], dtype=float)
    if (centered.shape != (n_blocks, len(coords)) or raw.shape != centered.shape
            or center.shape != (len(coords),) or not np.isfinite(centered).all()
            or not np.isfinite(raw).all()):
        raise ValueError("Invalid original bank residual dimensions or values")
    if n_blocks and (not np.isfinite(center).all()
                     or not np.allclose(raw.mean(axis=0), center, atol=1e-12, rtol=1e-12)
                     or not np.allclose(raw-center, centered, atol=1e-12, rtol=1e-12)):
        raise ValueError("Original bank does not use coordinate-mean centered residuals")
    copied = copy.deepcopy(bank)
    copied["unscaled_centered_residuals"] = centered.copy()
    copied["centered_residuals"] = centered * coords.sex.map(factors).to_numpy(dtype=float)[None, :]
    if not np.isfinite(copied["centered_residuals"]).all():
        raise ValueError("Scaling produced nonfinite residuals")
    copied["width_scales_by_sex"] = factors
    copied["residual_transform"] = "sex_specific_positive_scale_of_coordinate_centered_log_residuals"
    copied["point_forecasts_unchanged"] = True
    copied["population_residuals_rescaled"] = False
    return copied


def development_losses(history, config, amendment):
    """Reconstruct past banks and the five fixed loss curves for each supplied case.

    Input is the saved, selected prequential history of all fourteen underlying
    families, with actual ``target`` and ``outcome`` labels. One or multiple
    GCC/outcome cases may be supplied. Rows after the last development origin
    are excluded before value/family validation. Development verification is
    labeled by ``last_label_year`` and is not automatically eligible for an
    earlier evaluation origin.
    """
    factors = _amendment(amendment)
    keys = {"target", "outcome", "origin", "family", *COORDINATES,
            "forecast_year", "prediction", "log_prediction", "observed_rate"}
    if not keys.issubset(history.columns):
        raise ValueError("Prequential history is missing required forecast/verification columns")
    retained = history.loc[history.origin.le(max(amendment["development_origins"]))].copy()
    if retained.empty or retained[["target", "outcome", "origin", "family", *COORDINATES]].isna().any().any():
        raise ValueError("Require nonempty identified historical forecast cells")
    if (not retained.target.isin(amendment["countries"]).all()
            or not retained.outcome.isin(amendment["outcomes"]).all()):
        raise ValueError("Historical case is outside the specified GCC/outcome scope")
    families = config["models"]["local_order"] + config["models"]["nonneural_order"] + ["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
    if set(families) != set(UNDERLYING_FAMILIES) or len(families) != 14:
        raise ValueError("Use the fourteen unchanged underlying model families")
    expected_coords = pd.MultiIndex.from_product(
        [config["sexes"], config["ages"], config["calendar"]["horizons"]], names=COORDINATES)
    output = []
    for (country, outcome), case in retained.groupby(["target", "outcome"], sort=False):
        if set(case.family) != set(families):
            raise ValueError("Each case requires all fourteen underlying families, without champion aliases")
        for family in families:
            source = case.loc[case.family.eq(family)]
            for origin in amendment["development_origins"]:
                bank = build_residual_bank(source, config, origin, family)
                if bank["status"] != "ok":
                    raise ValueError("Development forecast has insufficient completed residual blocks")
                current = source.loc[source.origin.eq(origin)].copy()
                if (len(current) != len(expected_coords) or current.duplicated(COORDINATES).any()
                        or set(pd.MultiIndex.from_frame(current[COORDINATES])) != set(expected_coords)):
                    raise ValueError("Incomplete development age/sex/horizon forecast grid")
                if (not current.forecast_year.eq(current.origin+current.horizon).all()
                        or not np.isfinite(current.observed_rate).all() or not current.observed_rate.gt(0).all()):
                    raise ValueError("Invalid development verification rates or forecast years")
                normalizers = {}
                for sex in config["sexes"]:
                    take = bank["coords"].sex.eq(sex) & bank["coords"].horizon.eq(amendment["selection_horizon"])
                    normalizers[sex] = max(float(np.abs(bank["raw_residuals"][:, take.to_numpy()]).mean()),
                                           amendment["normalization_floor"])
                for factor in factors:
                    transformed = scale_bank(bank, {sex: factor for sex in config["sexes"]}, config)
                    # The current prediction ledger contains no verification values.
                    point_columns = ["target", "outcome", "origin", "family", *COORDINATES,
                                     "forecast_year", "prediction", "log_prediction"]
                    point_columns += [name for name in ["status", "fallback_reason"] if name in current]
                    intervals, _ = apply_bank(current[point_columns], transformed, config)
                    for sex in config["sexes"]:
                        part = intervals.loc[intervals.sex.eq(sex) & intervals.horizon.eq(amendment["selection_horizon"])
                                             & intervals.scale.eq("log_rate") & intervals.level.isin(amendment["selection_levels"])]
                        lower = part.pivot(index="age", columns="level", values="lower").reindex(
                            index=config["ages"], columns=amendment["selection_levels"])
                        upper = part.pivot(index="age", columns="level", values="upper").reindex_like(lower)
                        medians = part.pivot(index="age", columns="level", values="median").reindex_like(lower)
                        observed = current.loc[current.sex.eq(sex) & current.horizon.eq(amendment["selection_horizon"])].set_index("age").reindex(config["ages"])
                        values = weighted_interval_score(np.log(observed.observed_rate.to_numpy()), medians.iloc[:, 0].to_numpy(),
                                                         lower.to_numpy(), upper.to_numpy(), amendment["selection_levels"])
                        loss = float(np.mean(values))
                        output.append({"country": country, "outcome": outcome, "family": family, "sex": sex,
                                       "development_origin": int(origin), "factor": float(factor),
                                       "normalizer": normalizers[sex], "mean_log_wis": loss,
                                       "normalized_wis": loss/normalizers[sex], "last_label_year": int(origin+5),
                                       "bank_n_blocks": bank["n_blocks"],
                                       "last_residual_label_year": int(max(bank["origins"])+5)})
    return pd.DataFrame(output)


def select_factor(losses, amendment, target, outcome, source_family, sex, fit_origin, policy):
    """Select one scalar with fixed shrinkage toward one and chronological guards.

    GCC borrowing averages country means (excluding the target in the donor
    component), then weights target and donor components 50/50. The penalty is
    added once. Only eligible, matching rows are validated; future, different
    outcome/sex/family and unused-country losses cannot affect a choice.
    """
    factors = _amendment(amendment)
    if target not in amendment["countries"] or outcome not in amendment["outcomes"]:
        raise ValueError("Unknown target country or outcome")
    if source_family not in UNDERLYING_FAMILIES or sex not in {"Male", "Female"}:
        raise ValueError("Use a genuine underlying family and configured sex")
    if isinstance(fit_origin, (bool, np.bool_)) or not isinstance(fit_origin, (int, np.integer)):
        raise ValueError("The calibration fit origin must be an integer")
    if fit_origin not in amendment["evaluation_origins"]:
        raise ValueError("Fit origin is outside this bounded evaluation calendar")
    if policy not in {"target_only", "gcc_assisted"}:
        raise ValueError("Unknown width-selection policy")
    eligible = [origin for origin in amendment["development_origins"] if origin+5 <= fit_origin]
    donors = [country for country in amendment["countries"] if country != target] if policy == "gcc_assisted" else []
    weights = ({target: amendment["target_weight"], **{country: (1-amendment["target_weight"])/len(donors) for country in donors}}
               if donors else {target: 1.})
    result = {"target": target, "outcome": outcome, "source_family": source_family, "sex": sex,
              "fit_origin": int(fit_origin), "policy": policy, "eligible_origins": eligible,
              "last_label_year": max(eligible)+5 if eligible else None, "donor_countries": donors,
              "country_weights": weights, "factor": 1., "target_loss": None, "donor_loss": None,
              "unpenalized_loss": None, "penalty": 0., "objective": None,
              "candidate_objectives": [], "status": "cold_start_factor_one"}
    if not eligible:
        return result
    identifiers = {"country", "outcome", "family", "sex", "development_origin"}
    if not identifiers.issubset(losses.columns):
        raise ValueError("Calibration loss identifiers are missing")
    countries = list(weights)
    part = losses.loc[losses.country.isin(countries) & losses.outcome.eq(outcome)
                      & losses.family.eq(source_family) & losses.sex.eq(sex)
                      & losses.development_origin.isin(eligible)].copy()
    required = {"factor", "normalizer", "mean_log_wis", "normalized_wis", "last_label_year"}
    if not required.issubset(part.columns):
        raise ValueError("Eligible loss curves lack normalization, score or label-year fields")
    keys = ["country", "development_origin", "factor"]
    expected = {(country, origin, float(factor)) for country in countries for origin in eligible for factor in factors}
    if (part.duplicated(keys).any() or len(part) != len(expected)
            or set(map(tuple, part[keys].to_numpy())) != expected):
        raise ValueError("Incomplete or duplicated eligible country/origin/factor calibration grid")
    numeric = part[["normalizer", "mean_log_wis", "normalized_wis", "last_label_year"]].to_numpy(dtype=float)
    if (not np.isfinite(numeric).all() or part.normalizer.lt(amendment["normalization_floor"]).any()
            or part.mean_log_wis.lt(0).any() or part.normalized_wis.lt(0).any()):
        raise ValueError("Eligible calibration losses must be finite and nonnegative with positive normalization")
    if (not part.last_label_year.eq(part.development_origin+5).all()
            or not part.last_label_year.le(fit_origin).all()):
        raise ValueError("Eligible calibration loss has inconsistent or future labels")
    if not np.allclose(part.normalized_wis, part.mean_log_wis/part.normalizer, rtol=1e-12, atol=1e-12):
        raise ValueError("Normalized calibration loss disagrees with its recorded raw loss and normalizer")
    for _, group in part.groupby(["country", "development_origin"]):
        if group.normalizer.nunique() != 1:
            raise ValueError("The past-only normalizer must be identical across factor candidates")
    for factor in factors:
        current = part.loc[part.factor.eq(factor)]
        country_means = current.groupby("country").normalized_wis.mean().to_dict()
        target_loss = float(country_means[target])
        donor_loss = float(np.mean([country_means[country] for country in donors])) if donors else None
        unpenalized = (amendment["target_weight"]*target_loss+(1-amendment["target_weight"])*donor_loss
                       if donors else target_loss)
        penalty = float(amendment["penalty"]*np.log(float(factor))**2)
        objective = float(unpenalized+penalty)
        result["candidate_objectives"].append({"factor": float(factor), "country_losses": country_means,
                                               "target_loss": target_loss, "donor_loss": donor_loss,
                                               "unpenalized_loss": float(unpenalized), "penalty": penalty,
                                               "objective": objective})
    winner = min(result["candidate_objectives"], key=lambda row: (row["objective"], row["factor"]))
    result.update({key: winner[key] for key in ["factor", "target_loss", "donor_loss", "unpenalized_loss", "penalty", "objective"]})
    result["status"] = "selected_completed_development"
    return result
