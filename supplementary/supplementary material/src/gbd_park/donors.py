"""Prespecified, origin-restricted regional donor-country selection."""

import hashlib
import json

import numpy as np
import pandas as pd


TRAJECTORY_NAMES = ["mean_age_log_level", "mean_age_log_slope_8years",
                    "mean_age_sd_log_change_8years", "mean_age_log_change_last3"]
STRATEGIES = {"all_six_other_regional_countries", "other_five_gcc",
              "three_similar", "three_random"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _country_features(panel, config, origin, countries, sex, outcome):
    required = {"location_name", "sex", "age", "year", "outcome", "rate"}
    if not required.issubset(panel.columns):
        raise ValueError("Donor similarity needs complete country/sex/age/year outcome rates")
    years = np.arange(origin - 7, origin + 1)
    names = [country["name"] for country in countries]
    selected = panel.loc[panel.location_name.isin(names) & panel.sex.eq(sex)
                         & panel.outcome.eq(outcome) & panel.year.isin(years)].copy()
    keys = ["location_name", "age", "year"]
    expected = pd.MultiIndex.from_product([names, config["ages"], years], names=keys)
    actual = pd.MultiIndex.from_frame(selected[keys])
    if selected.duplicated(keys).any() or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Similarity requires a unique complete last-eight-year country/age grid")
    rates = selected.set_index(keys).reindex(expected).rate.to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Similarity rates must be positive and finite")
    logs = np.log(rates).reshape(len(countries), len(config["ages"]), 8)
    time = np.arange(8, dtype=float) - 3.5
    final = logs[:, :, -1]
    slopes = (logs * time).sum(axis=2) / np.dot(time, time)
    trajectory = np.column_stack([
        final.mean(axis=1), slopes.mean(axis=1),
        np.diff(logs, axis=2).std(axis=2, ddof=1).mean(axis=1),
        (logs[:, :, -1] - logs[:, :, -4]).mean(axis=1),
    ])
    profile = final - final.mean(axis=1, keepdims=True)
    return trajectory, profile, years.tolist()


def _scaled_block(features, donor_indices, target_index):
    donors = features[donor_indices]
    median = np.median(donors, axis=0)
    quartiles = np.quantile(donors, [.25, .75], axis=0, method="linear")
    iqr = quartiles[1] - quartiles[0]
    # Treat floating-point cancellation around a mathematically constant
    # feature as zero variation, without changing substantive donor variation.
    zero_tolerance = 32 * np.finfo(float).eps * np.maximum(1., np.abs(median))
    retained = iqr > zero_tolerance
    if retained.any():
        donor_scaled = (donors[:, retained] - median[retained]) / iqr[retained]
        target_scaled = (features[target_index, retained] - median[retained]) / iqr[retained]
        distance = np.mean((donor_scaled - target_scaled) ** 2, axis=1)
    else:
        distance = np.zeros(len(donor_indices), dtype=float)
    return distance, {"median": median.tolist(), "iqr": iqr.tolist(),
                      "zero_tolerance": zero_tolerance.tolist(), "retained": retained.tolist(),
                      "retained_components": int(retained.sum()), "weight": .5}


def select_donors(panel, config, origin, target, sex, strategy, seed=None, outcome="prevalence"):
    """Select countries without any post-origin rates or forecast-error feedback.

    Similarity uses the target sex; every selected donor contributes both sexes
    to later source fitting. This function chooses countries and records the
    decision, but does not fit or adapt a predictor. Fixed/random pools do not
    inspect outcome values. Random selection accepts only the five locked seeds.
    """
    if isinstance(origin, bool) or not isinstance(origin, (int, np.integer)):
        raise ValueError("Donor selection needs an integer forecast origin")
    if sex not in config["sexes"] or outcome not in set(config["outcomes"].values()):
        raise ValueError("Unknown target sex or outcome")
    if strategy not in STRATEGIES:
        raise ValueError("Unknown prespecified donor strategy")
    countries = sorted(config["countries"], key=lambda country: country["gbd_id"])
    names = [country["name"] for country in countries]
    ids = [country["gbd_id"] for country in countries]
    if len(names) != 7 or len(set(names)) != 7 or len(set(ids)) != 7 or target not in names:
        raise ValueError("Selection requires the seven unique locked regional countries and a known target")
    target_index = names.index(target)
    donor_indices = [index for index, name in enumerate(names) if name != target]
    pool = [countries[index] for index in donor_indices]
    if strategy != "three_random" and seed is not None:
        raise ValueError("Random donor seeds apply only to the three_random strategy")
    audit = {"origin": int(origin), "target": target, "sex": sex, "outcome": outcome,
             "strategy": strategy, "seed": None if seed is None else int(seed),
             "candidate_countries": [country["name"] for country in pool],
             "contributing_sexes": list(config["sexes"]), "target_excluded_all_sexes": True,
             "minimum_feature_year": None, "maximum_feature_year": None,
             "distance_table": [], "feature_audit": {}}
    if strategy == "all_six_other_regional_countries":
        selected = pool
    elif strategy == "other_five_gcc":
        if not countries[target_index]["gcc"]:
            raise ValueError("The other_five_gcc pool requires a GCC target")
        selected = [country for country in pool if country["gcc"]]
        if len(selected) != 5:
            raise ValueError("The locked GCC donor pool must contain five other GCC countries")
    elif strategy == "three_random":
        if (isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                or seed not in config["donors"]["random_seeds"]):
            raise ValueError("Use one of the five prespecified random-donor seeds")
        indices = np.random.default_rng(int(seed)).choice(len(pool), size=3, replace=False)
        selected = [pool[index] for index in sorted(indices)]
        audit["random_pool_indices"] = sorted(map(int, indices))
    else:
        trajectory, profile, years = _country_features(panel, config, int(origin), countries, sex, outcome)
        trajectory_distance, trajectory_audit = _scaled_block(trajectory, donor_indices, target_index)
        profile_distance, profile_audit = _scaled_block(profile, donor_indices, target_index)
        distance = .5 * trajectory_distance + .5 * profile_distance
        if not np.isfinite(distance).all():
            raise ValueError("Donor similarity produced a nonfinite distance")
        ranked = sorted(range(len(pool)), key=lambda index: (float(distance[index]), pool[index]["gbd_id"]))
        selected_indices = set(ranked[:3])
        selected = [pool[index] for index in sorted(selected_indices)]
        audit["distance_table"] = [
            {"country": pool[index]["name"], "gbd_id": int(pool[index]["gbd_id"]),
             "rank": rank + 1, "distance": float(distance[index]),
             "trajectory_mse": float(trajectory_distance[index]),
             "age_profile_mse": float(profile_distance[index]), "selected": index in selected_indices}
            for rank, index in enumerate(ranked)]
        feature_rows = [{"country": country["name"], "gbd_id": int(country["gbd_id"]),
                         "is_target": index == target_index,
                         "trajectory": dict(zip(TRAJECTORY_NAMES, trajectory[index].tolist())),
                         "centered_log_age_profile": dict(zip(config["ages"], profile[index].tolist()))}
                        for index, country in enumerate(countries)]
        audit["feature_audit"] = {"years": years, "trajectory_names": TRAJECTORY_NAMES,
                                  "age_order": list(config["ages"]), "rows": feature_rows,
                                  "trajectory_scaling": trajectory_audit, "profile_scaling": profile_audit,
                                  "distance_definition": "half_trajectory_MSE_plus_half_profile_MSE",
                                  "scaling_population": "six_donors_only",
                                  "variability_ddof": 1, "recent_change_years": 3}
        audit["minimum_feature_year"], audit["maximum_feature_year"] = min(years), max(years)
        audit["feature_sha256"] = _digest(audit["feature_audit"])
    audit["countries"] = [country["name"] for country in selected]
    audit["donor_list_sha256"] = hashlib.sha256(json.dumps(audit["countries"]).encode()).hexdigest()
    audit["selection_sha256"] = _digest(audit)
    return audit
