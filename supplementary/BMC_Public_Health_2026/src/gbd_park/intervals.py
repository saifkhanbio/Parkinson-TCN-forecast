"""Joint historical residual blocks and empirical predictive interval scores.

Intervals describe forecast errors of the GBD point estimates. Source bounds
and neural-seed dispersion are not inputs. The finite, overlapping historical
blocks do not provide an exact distribution-free coverage guarantee.
"""

import numpy as np
import pandas as pd


COORDINATES = ["sex", "age", "horizon"]


def _coordinates(config):
    return pd.MultiIndex.from_product(
        [config["sexes"], config["ages"], config["calendar"]["horizons"]],
        names=COORDINATES,
    ).to_frame(index=False)


def _single_value(frame, name, default):
    if name not in frame or frame.empty:
        return default
    if frame[name].isna().any() or frame[name].nunique() != 1:
        raise ValueError(f"A residual bank requires exactly one {name}")
    return frame[name].iloc[0]


def _validate_rates(frame, observed=False):
    fields = ["prediction", "log_prediction"] + (["observed_rate"] if observed else [])
    values = frame[fields].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame.prediction.to_numpy(dtype=float) <= 0).any():
        raise ValueError("Forecasts require finite log predictions and positive finite rates")
    if observed and (frame.observed_rate.to_numpy(dtype=float) <= 0).any():
        raise ValueError("Verification rates must be positive and finite")
    if not np.allclose(np.log(frame.prediction.to_numpy(dtype=float)),
                       frame.log_prediction.to_numpy(dtype=float), rtol=1e-12, atol=1e-10):
        raise ValueError("Prediction and log_prediction disagree")


def build_residual_bank(scored, config, fit_origin, family):
    """Build complete origin blocks, then center each coordinate by its mean.

    The caller supplies one chronologically generated forecast per coordinate
    and family, not the candidate tuning grid. Every eligible origin from the
    configured first residual origin through ``fit_origin - max(horizons)``
    must be present and complete. Future rows are excluded before validating
    outcomes, so unknown future verification values cannot affect this bank.
    """
    if isinstance(fit_origin, bool) or not isinstance(fit_origin, (int, np.integer)):
        raise ValueError("The fit origin must be an integer")
    if not isinstance(family, str) or not family:
        raise ValueError("A nonempty method-family identifier is required")
    required = {"family", "origin", "prediction", "log_prediction", "observed_rate", *COORDINATES}
    if not required.issubset(scored.columns):
        raise ValueError(f"Missing residual source columns: {sorted(required - set(scored.columns))}")
    horizons = config["calendar"]["horizons"]
    first_origin = int(config["intervals"]["residual_first_origin"])
    origins = list(range(first_origin, int(fit_origin) - max(horizons) + 1))
    coords = _coordinates(config)
    selected = scored.loc[scored.family.eq(family) & scored.origin.isin(origins)].copy()
    keys = ["origin"] + COORDINATES
    if selected.duplicated(keys).any():
        raise ValueError("Duplicate residual-bank origin/sex/age/horizon cells")
    expected = pd.MultiIndex.from_product(
        [origins, config["sexes"], config["ages"], horizons], names=keys)
    actual = pd.MultiIndex.from_frame(selected[keys])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Incomplete or unexpected eligible joint residual blocks")
    selected = selected.set_index(keys).reindex(expected).reset_index()
    target = _single_value(selected, "target", config["primary_target"])
    outcome = _single_value(selected, "outcome", "prevalence")
    _validate_rates(selected, observed=True)
    expected_year = selected.origin + selected.horizon
    if "forecast_year" in selected and not selected.forecast_year.eq(expected_year).all():
        raise ValueError("Residual verification year disagrees with origin plus horizon")
    if not expected_year.le(fit_origin).all():
        raise ValueError("Residual verification includes a future label")
    residual = (np.log(selected.observed_rate.to_numpy(dtype=float))
                - selected.log_prediction.to_numpy(dtype=float)).reshape(len(origins), len(coords))
    center = residual.mean(axis=0) if len(origins) else np.full(len(coords), np.nan)
    centered = residual - center
    minimum = int(config["intervals"]["minimum_blocks_to_emit"])
    status = "ok" if len(origins) >= minimum else "insufficient_blocks"
    return {"status": status, "n_blocks": len(origins), "origins": origins,
            "coords": coords, "raw_residuals": residual, "centered_residuals": centered,
            "center": center, "centering": "coordinate_arithmetic_mean",
            "fit_origin": int(fit_origin), "family": family, "target": target, "outcome": outcome,
            "minimum_blocks": minimum,
            "reason": "" if status == "ok" else f"Only {len(origins)} eligible joint blocks; require {minimum}"}


def apply_bank(point_frame, bank, config):
    """Add complete centered blocks to current log points and summarize each scale.

    Returns ``(interval_frame, draw_frame)``. Interval ``point_prediction`` is
    the original point on the named scale, while ``median`` is the predictive
    median. Each draw retains ``residual_origin`` across all coordinates for
    later joint ratios, age shares, and population-conditional count sums.
    With fewer than the required blocks, interval bounds/medians are NaN and
    the draw frame is empty; point forecasts and block counts remain explicit.
    """
    required = {"origin", "family", "prediction", "log_prediction", *COORDINATES}
    if not required.issubset(point_frame.columns):
        raise ValueError(f"Missing point forecast columns: {sorted(required - set(point_frame.columns))}")
    if point_frame.empty or not point_frame.origin.eq(bank["fit_origin"]).all():
        raise ValueError("Point forecasts must match the bank fit origin")
    if not point_frame.family.eq(bank["family"]).all():
        raise ValueError("Point forecasts must match the bank family")
    coords = _coordinates(config)
    if not coords.equals(bank["coords"]):
        raise ValueError("Bank coordinate order disagrees with the configuration")
    if point_frame.duplicated(COORDINATES).any():
        raise ValueError("Duplicate point forecast coordinates")
    expected = pd.MultiIndex.from_frame(coords)
    actual = pd.MultiIndex.from_frame(point_frame[COORDINATES])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Point forecasts require the complete joint coordinate grid")
    points = point_frame.set_index(COORDINATES).reindex(expected).reset_index()
    _validate_rates(points)
    for name in ["target", "outcome"]:
        if _single_value(points, name, bank[name]) != bank[name]:
            raise ValueError(f"Point forecast {name} disagrees with the residual bank")
    if "forecast_year" in points and not points.forecast_year.eq(points.origin + points.horizon).all():
        raise ValueError("Point forecast year disagrees with origin plus horizon")
    n_blocks = int(bank["n_blocks"])
    origins = bank["origins"]
    residuals = np.asarray(bank["centered_residuals"], dtype=float)
    if n_blocks != len(origins) or residuals.shape != (n_blocks, len(coords)):
        raise ValueError("Bank residual dimensions disagree with its coordinates/origins")
    if len(set(origins)) != len(origins) or any(origin + max(config["calendar"]["horizons"]) > bank["fit_origin"] for origin in origins):
        raise ValueError("Bank origins include duplicates or incomplete future blocks")
    if not np.isfinite(residuals).all():
        raise ValueError("Bank residuals must be finite")
    levels = np.asarray(config["intervals"]["central_levels"], dtype=float)
    _validate_levels(levels)
    minimum = int(config["intervals"]["minimum_blocks_to_emit"])
    available = n_blocks >= minimum
    if bank["status"] != ("ok" if available else "insufficient_blocks"):
        raise ValueError("Bank availability status disagrees with its block count")
    draw_columns = ["target", "outcome", "origin", "family", "residual_origin", *COORDINATES,
                    "forecast_year", "log_prediction", "prediction", "point_status",
                    "point_fallback_reason", "log_draw", "rate_draw"]
    intervals, draws = [], []
    log_points = points.log_prediction.to_numpy(dtype=float)
    rate_points = points.prediction.to_numpy(dtype=float)
    point_status = points["status"].to_numpy() if "status" in points else np.repeat("unspecified", len(points))
    point_reason = points["fallback_reason"].fillna("").to_numpy() if "fallback_reason" in points else np.repeat("", len(points))
    log_draws = log_points[None, :] + residuals
    if available:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                rate_draws = np.exp(log_draws)
            except FloatingPointError as exc:
                raise ValueError("Residual bank produces nonfinite rate draws") from exc
        if not np.isfinite(rate_draws).all() or (rate_draws <= 0).any():
            raise ValueError("Residual bank produces nonpositive or nonfinite rate draws")
        for block_index, residual_origin in enumerate(origins):
            block = coords.copy()
            block["target"], block["outcome"] = bank["target"], bank["outcome"]
            block["origin"], block["family"] = bank["fit_origin"], bank["family"]
            block["residual_origin"] = residual_origin
            block["forecast_year"] = bank["fit_origin"] + block.horizon
            block["log_prediction"], block["prediction"] = log_points, rate_points
            block["point_status"], block["point_fallback_reason"] = point_status, point_reason
            block["log_draw"], block["rate_draw"] = log_draws[block_index], rate_draws[block_index]
            draws.append(block[draw_columns])
    else:
        rate_draws = np.empty((0, len(coords)))
    for scale, values, point in [("log_rate", log_draws, log_points), ("rate", rate_draws, rate_points)]:
        for level in levels:
            if available:
                # Interpolate after transforming to the stated scale. In
                # particular exp(quantile(log_draws)) is not a rate quantile.
                lower, median, upper = np.quantile(
                    values, [(1 - level) / 2, 0.5, (1 + level) / 2], axis=0, method="linear")
            else:
                lower = median = upper = np.full(len(coords), np.nan)
            table = coords.copy()
            table["target"], table["outcome"] = bank["target"], bank["outcome"]
            table["origin"], table["family"] = bank["fit_origin"], bank["family"]
            table["forecast_year"] = bank["fit_origin"] + table.horizon
            table["scale"], table["level"] = scale, float(level)
            table["lower"], table["median"], table["upper"] = lower, median, upper
            table["point_prediction"] = point
            table["point_status"], table["point_fallback_reason"] = point_status, point_reason
            table["n_blocks"], table["status"] = n_blocks, bank["status"]
            table["unavailable_reason"] = bank.get("reason", "")
            intervals.append(table)
    draw_frame = pd.concat(draws, ignore_index=True) if draws else pd.DataFrame(columns=draw_columns)
    return pd.concat(intervals, ignore_index=True), draw_frame


def bank_quantiles(bank, config):
    """Summarize residual offsets/multipliers without inventing current forecasts.

    ``scale`` is ``log_offset`` for centered log residuals and
    ``rate_multiplier`` for their exponentials. These are calibration
    quantities, with a fit origin and no forecast year or point prediction.
    """
    coords = _coordinates(config)
    if not coords.equals(bank["coords"]):
        raise ValueError("Bank coordinate order disagrees with the configuration")
    n_blocks = int(bank["n_blocks"])
    residuals = np.asarray(bank["centered_residuals"], dtype=float)
    origins = bank["origins"]
    if residuals.shape != (n_blocks, len(coords)) or len(origins) != n_blocks:
        raise ValueError("Bank residual dimensions disagree with its coordinates/origins")
    if len(set(origins)) != len(origins) or any(origin + max(config["calendar"]["horizons"]) > bank["fit_origin"] for origin in origins):
        raise ValueError("Bank origins include duplicates or incomplete future blocks")
    if not np.isfinite(residuals).all():
        raise ValueError("Bank residuals must be finite")
    minimum = int(config["intervals"]["minimum_blocks_to_emit"])
    available = n_blocks >= minimum
    if bank["status"] != ("ok" if available else "insufficient_blocks"):
        raise ValueError("Bank availability status disagrees with its block count")
    levels = np.asarray(config["intervals"]["central_levels"], dtype=float)
    _validate_levels(levels)
    if available:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                multipliers = np.exp(residuals)
            except FloatingPointError as exc:
                raise ValueError("Residual bank produces nonfinite rate multipliers") from exc
        if not np.isfinite(multipliers).all() or (multipliers <= 0).any():
            raise ValueError("Residual bank produces nonpositive or nonfinite rate multipliers")
    else:
        multipliers = np.empty((0, len(coords)))
    tables = []
    for scale, values in [("log_offset", residuals), ("rate_multiplier", multipliers)]:
        for level in levels:
            if available:
                lower, median, upper = np.quantile(
                    values, [(1 - level) / 2, 0.5, (1 + level) / 2], axis=0, method="linear")
            else:
                lower = median = upper = np.full(len(coords), np.nan)
            table = coords.copy()
            for name in ["fit_origin", "family", "target", "outcome", "n_blocks", "status"]:
                table[name] = bank[name]
            table["scale"], table["level"] = scale, float(level)
            table["lower"], table["median"], table["upper"] = lower, median, upper
            table["unavailable_reason"] = bank.get("reason", "")
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _validate_levels(levels):
    if (levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or (levels <= 0).any() or (levels >= 1).any() or len(np.unique(levels)) != len(levels)):
        raise ValueError("Central interval levels must be unique finite values strictly between zero and one")


def interval_score(observed, lower, upper, alpha):
    """Central interval score for miscoverage alpha; supports NumPy broadcasting."""
    observed, lower, upper, alpha = np.broadcast_arrays(
        np.asarray(observed, dtype=float), np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float), np.asarray(alpha, dtype=float))
    if not all(np.isfinite(value).all() for value in [observed, lower, upper, alpha]):
        raise ValueError("Interval scoring requires finite values")
    if (lower > upper).any() or (alpha <= 0).any() or (alpha >= 1).any():
        raise ValueError("Invalid interval bounds or miscoverage alpha")
    result = upper - lower + (2 / alpha) * (np.maximum(lower - observed, 0) + np.maximum(observed - upper, 0))
    return float(result) if result.ndim == 0 else result


def weighted_interval_score(observed, median, lowers, uppers, levels):
    """WIS with a median weight of 0.5 and interval weights alpha/2.

    Bounds use the last axis for interval levels; all preceding dimensions
    broadcast against observed and median. For the study's principal WIS use
    levels [0.5, 0.8]; adding 0.95 gives the supplementary sparse-tail score.
    """
    levels = np.asarray(levels, dtype=float)
    _validate_levels(levels)
    lower, upper = np.broadcast_arrays(np.asarray(lowers, dtype=float), np.asarray(uppers, dtype=float))
    if lower.ndim < 1 or lower.shape[-1] != len(levels):
        raise ValueError("The last bounds axis must match the interval levels")
    observed, median, _ = np.broadcast_arrays(np.asarray(observed, dtype=float),
                                              np.asarray(median, dtype=float),
                                              np.empty(lower.shape[:-1]))
    if not np.isfinite(observed).all() or not np.isfinite(median).all():
        raise ValueError("WIS observations and medians must be finite")
    lower = np.broadcast_to(lower, observed.shape + (len(levels),))
    upper = np.broadcast_to(upper, observed.shape + (len(levels),))
    alpha = 1 - levels
    score = interval_score(observed[..., None], lower, upper, alpha)
    result = (0.5 * np.abs(observed - median) + np.sum((alpha / 2) * score, axis=-1)) / (len(levels) + 0.5)
    return float(result) if result.ndim == 0 else result


def score_intervals(interval_frame, truth, config, maximum_verification_year):
    """Score complete joint interval grids on their stated scales.

    Return all interval cells with inclusive coverage, width, and interval
    score, plus one coordinate/scale row with principal 50/80 WIS,
    supplementary 50/80/95 WIS, and predictive-median absolute error.
    Unavailable rows remain present with NaN scores. Source lower/upper bounds
    are ignored; only the separately supplied GBD point estimates verify the
    intervals. No row beyond the permitted verification cutoff is scored.
    """
    if isinstance(maximum_verification_year, bool) or not isinstance(maximum_verification_year, (int, np.integer)):
        raise ValueError("The verification cutoff must be an integer")
    group_keys = ["target", "outcome", "origin", "family"]
    interval_keys = group_keys + COORDINATES + ["scale", "level"]
    required = {*interval_keys, "forecast_year", "lower", "median", "upper", "status", "n_blocks"}
    if not required.issubset(interval_frame.columns):
        raise ValueError(f"Missing interval columns: {sorted(required - set(interval_frame.columns))}")
    if interval_frame.empty or interval_frame[interval_keys + ["forecast_year", "status", "n_blocks"]].isna().any().any():
        raise ValueError("Interval scoring needs a nonempty ledger with complete coordinate/status keys")
    if interval_frame.duplicated(interval_keys).any():
        raise ValueError("Duplicate interval forecast cells")
    if not interval_frame.forecast_year.eq(interval_frame.origin + interval_frame.horizon).all():
        raise ValueError("Interval forecast year disagrees with origin plus horizon")
    if not interval_frame.forecast_year.le(maximum_verification_year).all():
        raise ValueError("Interval forecast crosses the permitted verification cutoff")
    levels = np.sort(np.asarray(config["intervals"]["central_levels"], dtype=float))
    _validate_levels(levels)
    if not {0.5, 0.8, 0.95}.issubset(set(levels)):
        raise ValueError("Study interval scoring requires central levels 0.5, 0.8, and 0.95")
    expected = set(pd.MultiIndex.from_product(
        [config["sexes"], config["ages"], config["calendar"]["horizons"], ["log_rate", "rate"], levels]))
    minimum = int(config["intervals"]["minimum_blocks_to_emit"])
    for _, group in interval_frame.groupby(group_keys, sort=False):
        actual = set(zip(group.sex, group.age, group.horizon, group.scale, group.level))
        if len(group) != len(expected) or actual != expected:
            raise ValueError("Interval scoring requires the complete joint coordinate/scale/level grid")
        if group.status.nunique() != 1 or group.n_blocks.nunique() != 1:
            raise ValueError("Bank availability/count must agree across a complete family-origin grid")
        n_blocks = float(group.n_blocks.iloc[0])
        if not np.isfinite(n_blocks) or n_blocks < 0 or n_blocks != int(n_blocks):
            raise ValueError("Residual block count must be a nonnegative integer")
        status = group.status.iloc[0]
        if status != ("ok" if n_blocks >= minimum else "insufficient_blocks"):
            raise ValueError("Interval status disagrees with residual-block count")
        bounds = group[["lower", "median", "upper"]].to_numpy(dtype=float)
        if status == "ok":
            if not np.isfinite(bounds).all() or (bounds[:, 0] > bounds[:, 1]).any() or (bounds[:, 1] > bounds[:, 2]).any():
                raise ValueError("Available intervals require finite, ordered lower/median/upper values")
            if (group.loc[group.scale.eq("rate"), ["lower", "median", "upper"]].to_numpy(dtype=float) <= 0).any():
                raise ValueError("Rate-scale predictive quantiles must be positive")
        elif not np.isnan(bounds).all():
            raise ValueError("Unavailable interval bounds and medians must remain NaN")
    truth_keys = ["target", "outcome", "sex", "age", "forecast_year"]
    if not {*truth_keys, "observed_rate"}.issubset(truth.columns):
        raise ValueError("Verification data lack the required rate and coordinate columns")
    allowed_truth = truth.loc[truth.forecast_year.le(maximum_verification_year), truth_keys + ["observed_rate"]]
    if allowed_truth.duplicated(truth_keys).any():
        raise ValueError("Duplicate verification keys")
    if "observed_rate" in interval_frame or "observed_value" in interval_frame:
        raise ValueError("Unscored interval ledger must not already contain verification values")
    cells = interval_frame.merge(allowed_truth, on=truth_keys, how="left", validate="many_to_one")
    observed_rate = cells.observed_rate.to_numpy(dtype=float)
    if not np.isfinite(observed_rate).all() or (observed_rate <= 0).any():
        raise ValueError("Each interval needs a matched positive finite verification rate")
    cells["observed_value"] = observed_rate
    log_scale = cells.scale.eq("log_rate")
    cells.loc[log_scale, "observed_value"] = np.log(observed_rate[log_scale])
    available = cells.status.eq("ok")
    for field in ["covered", "width", "interval_score"]:
        cells[field] = np.nan
    subset = cells.loc[available]
    cells.loc[available, "covered"] = ((subset.observed_value >= subset.lower)
                                       & (subset.observed_value <= subset.upper)).astype(float)
    cells.loc[available, "width"] = subset.upper - subset.lower
    cells.loc[available, "interval_score"] = interval_score(
        subset.observed_value.to_numpy(), subset.lower.to_numpy(), subset.upper.to_numpy(),
        1 - subset.level.to_numpy())
    wis_keys = group_keys + COORDINATES + ["forecast_year", "scale"]
    auxiliary = [name for name in ["point_prediction", "point_status", "point_fallback_reason", "unavailable_reason"] if name in cells]
    wis = cells.drop_duplicates(wis_keys)[wis_keys + ["median", "observed_value", "n_blocks", "status"] + auxiliary].copy()
    wis = wis.reset_index(drop=True)
    index = pd.MultiIndex.from_frame(wis[wis_keys])
    arrays = {}
    for name in ["lower", "median", "upper"]:
        arrays[name] = cells.pivot(index=wis_keys, columns="level", values=name).reindex(index=index, columns=levels).to_numpy(dtype=float)
    if not np.allclose(arrays["median"], wis["median"].to_numpy(dtype=float)[:, None], rtol=1e-12, atol=1e-12, equal_nan=True):
        raise ValueError("Predictive medians disagree between interval levels")
    available = wis.status.eq("ok").to_numpy()
    for field in ["wis_50_80", "wis_50_80_95", "median_absolute_error"]:
        wis[field] = np.nan
    wis.loc[available, "median_absolute_error"] = np.abs(
        wis.loc[available, "observed_value"].to_numpy() - wis.loc[available, "median"].to_numpy())
    for field, included in [("wis_50_80", [0.5, 0.8]), ("wis_50_80_95", [0.5, 0.8, 0.95])]:
        positions = [int(np.where(levels == level)[0][0]) for level in included]
        wis.loc[available, field] = weighted_interval_score(
            wis.loc[available, "observed_value"].to_numpy(), wis.loc[available, "median"].to_numpy(),
            arrays["lower"][available][:, positions], arrays["upper"][available][:, positions], included)
    return cells, wis
