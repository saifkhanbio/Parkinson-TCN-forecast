"""Fixed, target-only age/horizon calibration of frozen joint residual draws.

This exploratory postprocessor changes predictive distributions, never point
forecasts. Positive piecewise slopes retain each coordinate's historical-origin
ranks. Partially matured forecast errors inform parameters; the joint draw bank
continues to use its original complete historical-origin blocks.
"""

from copy import deepcopy
import json

import numpy as np
import pandas as pd


SETTINGS = {
    "quantiles": [.2, .5, .8],
    "horizon_kernel_center": 2.,
    "horizon_kernel_adjacent": 1.,
    "group_location_shrinkage_origins": 20.,
    "tail_scale_shrinkage_origins": 8.,
    "reference_semispread_floor": 1e-6,
    "raw_tail_ratio_bounds": [.5, 3.],
    "origin_count_role": "normalized_kernel_weighted_mean_of_distinct_origins_not_effective_sample_size",
    "quantile_order": "within_horizon_then_kernel_average_then_semispread_ratio",
    "location_formula": "a*a*median_group + a*(1-a)*median_sex_pool",
    "tail_formula": "exp(b*(a*log(group_ratio)+(1-a)*log(sex_pool_ratio)))",
    "transform": "location + left_scale*min(z,0) + right_scale*max(z,0)",
}
COORDINATES = ["sex", "age", "horizon"]


def _original_bank(bank, config, source_family_by_sex):
    if set(source_family_by_sex) != set(config["sexes"]):
        raise ValueError("Supply the current underlying family for each configured sex")
    allowed = set(config["models"]["local_order"] + config["models"]["nonneural_order"]
                  + ["tcn_adapted", "tcn_intercept", "tcn_unadapted"])
    if not set(source_family_by_sex.values()).issubset(allowed):
        raise ValueError("Use genuine underlying families, not historical champion aliases")
    if bank.get("source_family_by_sex", source_family_by_sex) != source_family_by_sex:
        raise ValueError("The frozen bank and current underlying-family mapping disagree")
    if any(key in bank for key in ["residual_transform", "unscaled_centered_residuals",
                                   "structured_original_centered_residuals"]):
        raise ValueError("Transform the original frozen bank only, without compounding postprocessors")
    origin = bank["fit_origin"]
    if isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)):
        raise ValueError("The bank fit origin must be an integer")
    first = int(config["intervals"]["residual_first_origin"])
    horizons = list(config["calendar"]["horizons"])
    if horizons != [1, 2, 3, 4, 5]:
        raise ValueError("The fixed postprocessor requires horizons one through five")
    expected_origins = list(range(first, int(origin)-max(horizons)+1))
    if (bank["origins"] != expected_origins or bank["n_blocks"] != len(expected_origins)
            or len(expected_origins) < config["intervals"]["minimum_blocks_to_emit"] or bank["status"] != "ok"):
        raise ValueError("Require the complete original bank with at least five completed joint blocks")
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"], horizons],
                                       names=COORDINATES).to_frame(index=False)
    if not bank["coords"].equals(coords):
        raise ValueError("Original bank coordinate order differs from the complete configured grid")
    raw, center, centered = (np.asarray(bank[key], dtype=float)
                             for key in ["raw_residuals", "center", "centered_residuals"])
    if (raw.shape != (len(expected_origins), len(coords)) or centered.shape != raw.shape
            or center.shape != (len(coords),) or not np.isfinite(raw).all()
            or not np.isfinite(center).all() or not np.isfinite(centered).all()):
        raise ValueError("Original residual bank has invalid dimensions or values")
    if (not np.allclose(raw.mean(axis=0), center, atol=1e-12, rtol=1e-12)
            or not np.allclose(raw-center, centered, atol=1e-12, rtol=1e-12)):
        raise ValueError("Original residuals must retain coordinate arithmetic-mean centering")
    groups = {}
    for name, starts in config["age_groups"].items():
        groups[name] = [age for age in config["ages"] if int(age.split("-")[0].rstrip("+")) in starts]
    if (set(groups) != {"45-64", "65-79", "80+"}
            or groups["80+"] != ["80-84", "85-89", "90-94", "95+"]
            or sorted(age for ages in groups.values() for age in ages) != sorted(config["ages"])):
        raise ValueError("Use the three disjoint locked age groups and all eleven ages including four 80+ bands")
    return coords, groups


def _eligible_history(history, bank, config, source_family_by_sex):
    required = {"target", "outcome", "family", "origin", "forecast_year", "prediction",
                "log_prediction", "observed_rate", *COORDINATES}
    if not required.issubset(history.columns):
        raise ValueError("Historical forecasts lack required identity, prediction or verification columns")
    origin, first = int(bank["fit_origin"]), int(config["intervals"]["residual_first_origin"])
    case = history.loc[history.target.eq(bank["target"]) & history.outcome.eq(bank["outcome"])].copy()
    pieces = []
    for sex in config["sexes"]:
        family = source_family_by_sex[sex]
        selected = case.loc[case.sex.eq(sex) & case.family.eq(family)
                            & case.origin.ge(first) & case.origin.lt(origin)
                            & case.horizon.isin(config["calendar"]["horizons"])
                            & (case.origin+case.horizon).le(origin)
                            & case.forecast_year.le(origin)].copy()
        keys = ["horizon", "origin", "age"]
        expected = pd.MultiIndex.from_tuples(
            [(h, u, age) for h in config["calendar"]["horizons"]
             for u in range(first, origin-h+1) for age in config["ages"]], names=keys)
        if (selected.duplicated(keys).any() or len(selected) != len(expected)
                or set(pd.MultiIndex.from_frame(selected[keys])) != set(expected)):
            raise ValueError("Missing or duplicate eligible partially matured age/origin/horizon cells")
        selected = selected.set_index(keys).reindex(expected).reset_index()
        values = selected[["prediction", "log_prediction", "observed_rate"]].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or not selected.prediction.gt(0).all()
                or not selected.observed_rate.gt(0).all()
                or not selected.forecast_year.eq(selected.origin+selected.horizon).all()):
            raise ValueError("Eligible residual labels must be finite, positive, and correctly dated")
        if not np.allclose(np.log(selected.prediction), selected.log_prediction, rtol=1e-12, atol=1e-10):
            raise ValueError("Eligible prediction and log prediction disagree")
        selected["raw_log_error"] = np.log(selected.observed_rate)-selected.log_prediction
        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


def _semispread_ratios(observed_quantiles, reference_quantiles):
    observed = np.array([observed_quantiles[1]-observed_quantiles[0],
                         observed_quantiles[2]-observed_quantiles[1]])
    reference = np.array([reference_quantiles[1]-reference_quantiles[0],
                          reference_quantiles[2]-reference_quantiles[1]])
    if (observed < 0).any() or (reference < 0).any():
        raise ValueError("Smoothed quantiles must be ordered")
    degenerate = reference <= SETTINGS["reference_semispread_floor"]
    ratios = np.ones(2)
    ratios[~degenerate] = np.clip(observed[~degenerate]/reference[~degenerate],
                                 *SETTINGS["raw_tail_ratio_bounds"])
    return ratios, degenerate, observed, reference


def structured_bank(history, bank, config, source_family_by_sex):
    """Return a transformed bank copy and thirty sex/group/horizon audit rows.

    ``history`` includes the saved underlying-family forecasts at origins 2003
    onward, including recent partially matured windows. All eligible cells are
    mandatory. Other families/cases and future labels are filtered before value
    checks. Each horizon contributes its own empirical quantiles, then adjacent
    horizons are smoothed with normalized 1/2/1 weights; ratios are formed last.

    The count used for shrinkage is a weighted MEAN of distinct origin counts,
    never the number of ages or a claim of independent effective sample size.
    The existing ``centered_residuals`` field is the transport interface required
    by ``apply_bank``; after this distribution calibration it need not have mean
    zero. Its original values and all original bank metadata remain available.
    """
    coords, groups = _original_bank(bank, config, source_family_by_sex)
    eligible = _eligible_history(history, bank, config, source_family_by_sex)
    quantiles = SETTINGS["quantiles"]
    original = np.asarray(bank["centered_residuals"], dtype=float)
    transformed = original.copy()
    records = []
    for sex in config["sexes"]:
        source = eligible.loc[eligible.sex.eq(sex)]
        counts = source.groupby("horizon").origin.nunique().to_dict()
        latest_labels = source.groupby("horizon").forecast_year.max().astype(int).to_dict()
        observed, reference = {}, {}
        for group, ages in {"sex_pool": config["ages"], **groups}.items():
            for h in config["calendar"]["horizons"]:
                values = source.loc[source.horizon.eq(h) & source.age.isin(ages), "raw_log_error"].to_numpy()
                take = coords.sex.eq(sex) & coords.horizon.eq(h) & coords.age.isin(ages)
                observed[(group, h)] = np.quantile(values, quantiles, method="linear")
                reference[(group, h)] = np.quantile(original[:, take.to_numpy()], quantiles, method="linear")
        for h in config["calendar"]["horizons"]:
            weights = {j: (SETTINGS["horizon_kernel_center"] if h == j else SETTINGS["horizon_kernel_adjacent"])
                       for j in config["calendar"]["horizons"] if abs(j-h) <= 1}
            total = sum(weights.values())
            weights = {j: value/total for j, value in weights.items()}
            n = float(sum(weights[j]*counts[j] for j in weights))
            a = n/(n+SETTINGS["group_location_shrinkage_origins"])
            b = n/(n+SETTINGS["tail_scale_shrinkage_origins"])
            pool_q = sum(weights[j]*observed[("sex_pool", j)] for j in weights)
            pool_reference_q = sum(weights[j]*reference[("sex_pool", j)] for j in weights)
            pool_ratios, pool_degenerate, pool_spreads, pool_reference_spreads = _semispread_ratios(pool_q, pool_reference_q)
            for group, ages in groups.items():
                group_q = sum(weights[j]*observed[(group, j)] for j in weights)
                group_reference_q = sum(weights[j]*reference[(group, j)] for j in weights)
                group_ratios, group_degenerate, group_spreads, group_reference_spreads = _semispread_ratios(group_q, group_reference_q)
                location = float(a*a*group_q[1]+a*(1-a)*pool_q[1])
                scales = np.exp(b*(a*np.log(group_ratios)+(1-a)*np.log(pool_ratios)))
                take = coords.sex.eq(sex) & coords.horizon.eq(h) & coords.age.isin(ages)
                values = original[:, take.to_numpy()]
                transformed[:, take.to_numpy()] = location+scales[0]*np.minimum(values, 0)+scales[1]*np.maximum(values, 0)
                record = dict(target=bank["target"], outcome=bank["outcome"], family=bank["family"],
                    source_family=source_family_by_sex[sex], fit_origin=int(bank["fit_origin"]), sex=sex,
                    age_group=group, horizon=int(h), age_bands="|".join(ages), n_age_bands=len(ages),
                    original_joint_blocks=int(bank["n_blocks"]), weighted_distinct_origin_count=n,
                    distinct_origins_at_horizon=int(counts[h]), last_label_year=int(latest_labels[h]),
                    origin_counts_by_horizon=json.dumps({str(j): int(counts[j]) for j in counts}, sort_keys=True),
                    last_label_year_by_horizon=json.dumps({str(j): latest_labels[j] for j in latest_labels}, sort_keys=True),
                    horizon_kernel_weights=json.dumps({str(j): float(weights[j]) for j in weights}, sort_keys=True),
                    count_is_effective_sample_size=False, group_location_shrinkage=a, scale_shrinkage=b,
                    location_group_weight=a*a, location_pool_weight=a*(1-a), location_zero_weight=1-a,
                    location=location, left_scale=float(scales[0]), right_scale=float(scales[1]),
                    original_mean_offset=float(values.mean()), transformed_mean_offset=float(transformed[:, take.to_numpy()].mean()))
                for prefix, data in [("group_observed", group_q), ("pool_observed", pool_q),
                                     ("group_reference", group_reference_q), ("pool_reference", pool_reference_q)]:
                    record.update({prefix+"_q"+str(int(q*100)): float(value) for q, value in zip(quantiles, data)})
                for side, index in [("left", 0), ("right", 1)]:
                    record.update({"group_"+side+"_ratio": float(group_ratios[index]),
                                   "pool_"+side+"_ratio": float(pool_ratios[index]),
                                   "group_"+side+"_zero_reference": bool(group_degenerate[index]),
                                   "pool_"+side+"_zero_reference": bool(pool_degenerate[index]),
                                   "group_"+side+"_observed_semispread": float(group_spreads[index]),
                                   "pool_"+side+"_observed_semispread": float(pool_spreads[index]),
                                   "group_"+side+"_reference_semispread": float(group_reference_spreads[index]),
                                   "pool_"+side+"_reference_semispread": float(pool_reference_spreads[index])})
                records.append(record)
    if not np.isfinite(transformed).all():
        raise ValueError("Structured calibration produced nonfinite residual offsets")
    copied = deepcopy(bank)
    copied["structured_original_centered_residuals"] = original.copy()
    copied["centered_residuals"] = transformed
    copied["residual_transform"] = "fixed_target_age_horizon_asymmetric_distribution_calibration"
    copied["residual_field_role"] = "distribution_offsets_not_necessarily_coordinate_mean_zero"
    copied["structured_settings"] = deepcopy(SETTINGS)
    copied["structured_source_family_by_sex"] = dict(source_family_by_sex)
    copied["point_forecasts_unchanged"] = True
    copied["population_residuals_rescaled"] = False
    return copied, pd.DataFrame(records)
