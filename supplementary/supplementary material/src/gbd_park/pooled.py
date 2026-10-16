"""Country-balanced non-neural models with completed-window training."""

import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from gbd_park.adaptation import correction, fit_adaptation


def base_grid(config):
    grid = [{"algorithm": "ridge", "alpha": a} for a in config["models"]["ridge_penalties"]]
    boost = config["models"]["boosting"]
    grid.extend({"algorithm": "boosting", "depth": d, "trees": t, "min_leaf": leaf}
                for d, t, leaf in itertools.product(boost["depth"], boost["trees"], boost["min_leaf"]))
    return grid


def setting_id(family, base, penalty=None):
    value = family + "__" + "__".join(f"{k}={v}" for k, v in base.items() if k != "algorithm")
    return value if penalty is None else value + f"__adaptation_penalty={penalty}"


def settings_grid(config):
    result = []
    for family in config["models"]["nonneural_order"]:
        algorithm = "ridge" if "ridge" in family else "boosting"
        penalties = config["adaptation"]["penalties"] if family.endswith("_adapted") else [None]
        for base in [g for g in base_grid(config) if g["algorithm"] == algorithm]:
            for penalty in penalties:
                result.append({"family": family, "base": base, "penalty": penalty,
                               "setting_id": setting_id(family, base, penalty), "grid_order": len(result)})
    return result


def country_pool(config, target, mode):
    if mode not in ["pooled", "donor"]:
        raise ValueError("Unknown pooling mode")
    names = [c["name"] for c in config["countries"]]
    if target not in names:
        raise ValueError("Unknown target")
    return [n for n in names if mode == "pooled" or n != target]


def features(log_history, sex, age_index, age_count):
    age = np.zeros(age_count)
    age[age_index] = 1
    return np.r_[log_history - log_history[-1], log_history[-1], float(sex == "Male"), age]


def series_values(panel, config, cutoff, countries):
    data = panel.loc[panel.year.between(config["calendar"]["history_start"], cutoff)
                     & panel.location_name.isin(countries) & panel.outcome.eq("prevalence")]
    if data.duplicated(["location_name", "sex", "age", "year"]).any():
        raise ValueError("Duplicate source history")
    years = np.arange(config["calendar"]["history_start"], cutoff + 1)
    for country in countries:
        for sex in config["sexes"]:
            values = data.loc[data.location_name.eq(country) & data.sex.eq(sex)].pivot(
                index="year", columns="age", values="rate").reindex(index=years, columns=config["ages"]).to_numpy()
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"Incomplete or invalid source history: {country}, {sex}")
            for a, age in enumerate(config["ages"]):
                yield country, sex, age, a, years, np.log(values[:, a])


def build_examples(panel, config, cutoff, countries):
    xs, ys, rows = [], [], []
    width = config["calendar"]["window"]
    for country, sex, age, a, years, values in series_values(panel, config, cutoff, countries):
        for index in range(width - 1, len(years) - 5):
            xs.append(features(values[index-width+1:index+1], sex, a, len(config["ages"])))
            ys.append(values[index+1:index+6] - values[index])
            rows.append({"country": country, "sex": sex, "age": age, "window_origin": int(years[index]),
                         "input_start": int(years[index-width+1]), "input_end": int(years[index]),
                         "label_start": int(years[index+1]), "label_end": int(years[index+5])})
    if not rows:
        raise ValueError("No complete training windows")
    meta = pd.DataFrame(rows)
    assert meta.label_end.le(cutoff).all()
    return np.asarray(xs), np.asarray(ys), meta


def target_inputs(panel, config, cutoff, target):
    xs, levels, rows = [], [], []
    for country, sex, age, a, years, values in series_values(panel, config, cutoff, [target]):
        xs.append(features(values[-8:], sex, a, len(config["ages"])))
        levels.append(values[-1])
        rows.append({"country": country, "sex": sex, "age": age})
    return np.asarray(xs), np.asarray(levels), pd.DataFrame(rows)


def balanced_weights(meta):
    ncountry = meta.country.nunique()
    nsex = meta.groupby("country").sex.transform("nunique")
    nage = meta.groupby(["country", "sex"]).age.transform("nunique")
    nwindow = meta.groupby(["country", "sex", "age"]).age.transform("size")
    weights = len(meta) / (ncountry * nsex * nage * nwindow)
    np.testing.assert_allclose(weights.sum(), len(meta), rtol=1e-12)
    return weights.to_numpy()


def fit_base(x, y, meta, base, config):
    weights = balanced_weights(meta)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    transformed = scaler.transform(x)
    warning_messages = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if base["algorithm"] == "ridge":
            model = Ridge(alpha=base["alpha"], solver="svd", fit_intercept=True).fit(transformed, y, sample_weight=weights)
            models = [model]
            parameter_count = int(model.coef_.size + model.intercept_.size)
        else:
            models = [GradientBoostingRegressor(
                loss="absolute_error", learning_rate=config["models"]["boosting"]["learning_rate"],
                n_estimators=base["trees"], max_depth=base["depth"], min_samples_leaf=base["min_leaf"],
                random_state=11, subsample=1.0, max_features=None, n_iter_no_change=None
            ).fit(transformed, y[:, h], sample_weight=weights) for h in range(5)]
            # One initial value plus leaf values and split thresholds per tree.
            parameter_count = sum(1 + sum(tree.tree_.node_count for tree in m.estimators_[:, 0]) for m in models)
        warning_messages = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    fitted = {"scaler": scaler, "models": models, "algorithm": base["algorithm"], "base": base,
              "parameter_count": parameter_count, "warnings": warning_messages}
    return fitted


def predict_changes(fitted, x):
    transformed = fitted["scaler"].transform(x)
    if fitted["algorithm"] == "ridge":
        prediction = fitted["models"][0].predict(transformed)
    else:
        prediction = np.column_stack([m.predict(transformed) for m in fitted["models"]])
    if prediction.shape != (len(x), 5) or not np.isfinite(prediction).all():
        raise ValueError("Invalid base predictions")
    return prediction


def forecast_origin(panel, config, origin, checkpoint_dir=None, selected_bases=None):
    target = config["primary_target"]
    current_x, levels, current_meta = target_inputs(panel, config, origin, target)
    target_x, target_y, target_meta = build_examples(panel, config, origin, [target])
    specs = settings_grid(config)
    selected_bases = base_grid(config) if selected_bases is None else selected_bases
    forecasts, fits, calibrations, training_metadata = [], [], [], []
    for mode in ["pooled", "donor"]:
        countries = country_pool(config, target, mode)
        x, y, meta = build_examples(panel, config, origin, countries)
        weights = balanced_weights(meta)
        training_metadata.extend(meta.assign(fit_origin=origin, pool=mode, sample_weight=weights).to_dict("records"))
        for base in selected_bases:
            model_id = f"origin{origin}__{mode}__" + "__".join(f"{k}={v}" for k, v in base.items())
            fit_status, reason, fitted = "ok", "", None
            try:
                fitted = fit_base(x, y, meta, base, config)
                fingerprint = joblib.hash(fitted, hash_name="sha1")
                unadapted = predict_changes(fitted, current_x)
                target_prediction = predict_changes(fitted, target_x) if mode == "donor" else None
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
                fit_status, reason = "fallback", f"{type(exc).__name__}: {exc}"
                unadapted = np.zeros((len(current_x), 5))
                fingerprint = "unavailable"
            if checkpoint_dir is not None and fit_status == "ok":
                joblib.dump(fitted, checkpoint_dir / f"{model_id}.joblib", compress=3)
            fit_record = {"model_id": model_id, "origin": origin, "pool": mode, "base": base,
                          "countries": countries, "training_windows": len(meta),
                          "minimum_input_year": int(meta.input_start.min()), "maximum_input_year": int(meta.input_end.max()),
                          "maximum_label_year": int(meta.label_end.max()), "status": fit_status, "reason": reason,
                          "base_fingerprint_before": fingerprint, "feature_mean": fitted["scaler"].mean_.tolist() if fitted is not None else [],
                          "feature_scale": fitted["scaler"].scale_.tolist() if fitted is not None else [],
                          "warnings": fitted["warnings"] if fitted is not None else []}
            for spec in [s for s in specs if s["base"] == base and s["family"].startswith(mode + "_")]:
                for sex in config["sexes"]:
                    indices = current_meta.sex.eq(sex).to_numpy()
                    prediction = unadapted[indices].copy()
                    status, fallback_reason, cal = fit_status, reason, None
                    if spec["penalty"] is not None:
                        eligible = target_meta.sex.eq(sex).to_numpy()
                        if fit_status == "ok":
                            try:
                                cal = fit_adaptation(target_y[eligible], target_prediction[eligible], spec["penalty"])
                                prediction += correction(cal)
                            except (ValueError, RuntimeError, FloatingPointError) as exc:
                                status, fallback_reason = "fallback", f"Adaptation: {type(exc).__name__}: {exc}"
                                prediction[:] = 0
                        calibrations.append({"model_id": model_id, "origin": origin, "sex": sex,
                                             "setting_id": spec["setting_id"], "penalty": spec["penalty"],
                                             "target_windows": int(eligible.sum()), "last_target_label_year": int(target_meta.loc[eligible, "label_end"].max()),
                                             "status": status, "reason": fallback_reason,
                                             **({} if cal is None else cal)})
                    logs = levels[indices, None] + prediction
                    invalid = ~np.isfinite(logs).all(axis=1) | (np.abs(logs) > 700).any(axis=1)
                    for a, (_, row) in enumerate(current_meta.loc[indices].iterrows()):
                        row_status = "fallback" if invalid[a] else status
                        row_reason = "Unsafe log forecast" if invalid[a] else fallback_reason
                        output = np.repeat(levels[indices][a], 5) if invalid[a] else logs[a]
                        parameters = 0 if fitted is None else fitted["parameter_count"] + (2 if cal is not None else 0)
                        for h in range(1, 6):
                            forecasts.append({"target": target, "sex": sex, "age": row.age, "outcome": "prevalence",
                                "origin": origin, "horizon": h, "forecast_year": origin + h, "family": spec["family"],
                                "setting_id": spec["setting_id"], "grid_order": spec["grid_order"], "model_id": model_id,
                                "base_fingerprint": fingerprint, "log_prediction": float(output[h-1]), "prediction": float(np.exp(output[h-1])),
                                "parameter_count": parameters, "status": row_status, "fallback_reason": row_reason,
                                "donor_strategy": "all_six_other_regional_countries", "target_in_base_fit": mode == "pooled",
                                "adaptation": "two_parameter" if spec["penalty"] is not None else "none",
                                "setting_json": json.dumps({**base, "adaptation_penalty": spec["penalty"]}, sort_keys=True)})
            after = joblib.hash(fitted, hash_name="sha1") if fit_status == "ok" else "unavailable"
            if after != fingerprint:
                raise AssertionError("Source predictor changed during adaptation/inference")
            fit_record["base_fingerprint_after"] = after
            fits.append(fit_record)
    return forecasts, fits, calibrations, training_metadata
