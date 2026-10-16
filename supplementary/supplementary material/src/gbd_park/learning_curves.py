"""Fixed-setting Saudi information budgets with identical shared donor fits."""

import copy
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from gbd_park.adaptation import correction, fit_adaptation
from gbd_park.local import forecast_setting, settings_grid
from gbd_park.pooled import (balanced_weights, build_examples, fit_base,
                             predict_changes, target_inputs)
from gbd_park.tcn import (fit_tcn, predict_changes as tcn_predict,
                          state_fingerprint, save_checkpoint)
from gbd_park.tcn_forecasting import make_forecasts, parameter_count

OUTCOMES = ("prevalence", "incidence")
ROLE = "prespecified_supporting_target_history_learning_curves"
KEYS = ["outcome", "history_years", "family", "sex", "age", "horizon"]


def families(config):
    return config["models"]["local_order"] + config["models"]["nonneural_order"] + [
        "tcn_unadapted", "tcn_intercept", "tcn_adapted"]


def tasks(config):
    origin = config["learning_curve"]["origin"]
    return [{"id": f"SAU_{outcome}_history{years}", "outcome": outcome,
             "history_years": years, "target_history_start": origin-years+1,
             "origin": origin, "target": config["primary_target"]}
            for outcome in OUTCOMES for years in config["learning_curve"]["history_years"]]


def context(panel, config, outcome, years):
    """Drop unavailable Saudi and future rows before any feature construction."""
    if outcome not in OUTCOMES or years not in config["learning_curve"]["history_years"]:
        raise ValueError("Unknown learning-curve outcome or history budget")
    origin = config["learning_curve"]["origin"]
    start = origin-years+1
    countries = [country["name"] for country in config["countries"]]
    selected = (panel.outcome.eq(outcome) & panel.location_name.isin(countries)
                & panel.sex.isin(config["sexes"]) & panel.age.isin(config["ages"])
                & panel.year.between(config["calendar"]["history_start"], origin)
                & (~panel.location_name.eq(config["primary_target"]) | panel.year.ge(start)))
    work = panel.loc[selected].copy()
    keys = ["location_name", "sex", "age", "year"]
    if work.duplicated(keys).any():
        raise ValueError("Duplicate learning-curve history keys")
    expected = {(country, sex, age, year) for country in countries
                for sex in config["sexes"] for age in config["ages"]
                for year in range(start if country == config["primary_target"] else
                                  config["calendar"]["history_start"], origin+1)}
    if set(work[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Incomplete learning-curve country/sex/age/year grid")
    if not np.isfinite(work.rate).all() or work.rate.le(0).any():
        raise ValueError("Learning-curve history requires finite positive rates")
    work["source_outcome"] = outcome
    work["outcome"] = "prevalence"
    return work.sort_values(keys).reset_index(drop=True)


def budget_config(config, years):
    cfg = copy.deepcopy(config)
    cfg["calendar"]["history_start"] = config["learning_curve"]["origin"]-years+1
    return cfg


def training_examples(work, config, years, mode):
    """Build each country's complete windows, preserving unequal history lengths."""
    if mode not in {"pooled", "donor", "target"}:
        raise ValueError("Unknown training pool")
    target = config["primary_target"]
    names = [c["name"] for c in config["countries"] if
             mode == "pooled" or (c["name"] == target) == (mode == "target")]
    arrays = []
    for country in names:
        cfg = budget_config(config, years) if country == target else config
        arrays.append(build_examples(work, cfg, config["learning_curve"]["origin"], [country]))
    return (np.concatenate([a[0] for a in arrays]), np.concatenate([a[1] for a in arrays]),
            pd.concat([a[2] for a in arrays], ignore_index=True))


def target_arrays(work, config, years):
    cfg = budget_config(config, years)
    x, levels, meta = target_inputs(work, cfg, config["learning_curve"]["origin"], config["primary_target"])
    target_x, target_y, training = training_examples(work, config, years, "target")
    return x, levels, meta, target_x, target_y, training


def fixed_base(config, kind):
    defaults = config["cold_start_defaults"]
    if kind == "tcn":
        return {"channels": defaults["tcn_channels"], "weight_decay": defaults["tcn_weight_decay"],
                "epochs": defaults["tcn_epochs"]}
    if kind == "ridge":
        return {"algorithm": "ridge", "alpha": defaults["ridge_penalty"]}
    if kind == "boosting":
        return {"algorithm": "boosting", "depth": defaults["boosting_depth"],
                "trees": defaults["boosting_trees"], "min_leaf": defaults["boosting_min_leaf"]}
    raise ValueError("Unknown base algorithm")


def local_specs(config):
    defaults = config["cold_start_defaults"]
    keep = {"log_trend": {"window": defaults["trend_years"]},
            "age_smooth_trend": {"window": defaults["age_smooth_years"],
                                 "penalty": defaults["age_smooth_penalty"]}}
    return [spec for spec in settings_grid(config) if spec["setting"] == keep.get(spec["family"], {})]


def source_fingerprint(fitted, kind):
    """Hash learned numeric state without unstable sklearn tree pickle padding."""
    if kind == "tcn":
        return state_fingerprint(fitted)
    digest = hashlib.sha256(json.dumps(fitted["base"], sort_keys=True).encode())

    def add(name, value):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(json.dumps({"name": name, "shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode())
        digest.update(array.tobytes())

    for field in ["mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_"]:
        add("scaler."+field, getattr(fitted["scaler"], field))
    for index, model in enumerate(fitted["models"]):
        if kind == "ridge":
            add(f"model{index}.coef", model.coef_)
            add(f"model{index}.intercept", model.intercept_)
        else:
            add(f"model{index}.init", model.init_.constant_)
            add(f"model{index}.learning_rate", model.learning_rate)
            for ti, estimator in enumerate(model.estimators_[:, 0]):
                for field in ["children_left", "children_right", "feature", "threshold", "value"]:
                    add(f"model{index}.tree{ti}.{field}", getattr(estimator.tree_, field))
    return digest.hexdigest()


def fit_local(panel, config, outcome, years, sex):
    work = context(panel, config, outcome, years)
    cfg = budget_config(config, years)
    rows, fits, arima = [], [], []
    for spec in local_specs(config):
        r, f, a = forecast_setting(work, cfg, config["learning_curve"]["origin"], sex,
                                    spec, config["primary_target"])
        rows.extend(r)
        fits.extend(f)
        arima.extend(a)
    return {"forecasts": rows, "fits": fits, "arima": arima}


def fit_source(panel, config, outcome, kind, mode="donor", years=None, seed=None,
               device="cpu", checkpoint_path=None):
    """One donor fit is evaluated on every target budget without refitting."""
    if mode not in {"donor", "pooled"} or (kind == "tcn" and mode != "donor"):
        raise ValueError("Unsupported source model/pool")
    if mode == "donor" and years is not None:
        raise ValueError("A shared donor fit must not be identified by a target budget")
    if mode == "pooled" and years not in config["learning_curve"]["history_years"]:
        raise ValueError("Pooled fit requires an explicit target budget")
    budgets = config["learning_curve"]["history_years"] if mode == "donor" else [years]
    # The smallest budget verifies that source fitting never needs older Saudi rows.
    work = context(panel, config, outcome, min(budgets))
    x, y, meta = training_examples(work, config, min(budgets), mode)
    base = fixed_base(config, kind)
    weights = balanced_weights(meta)
    fitted, before, status, reason = None, "unavailable", "ok", ""
    if kind == "tcn" and seed not in config["models"]["tcn"]["ensemble_seeds"]:
        raise ValueError("Unexpected TCN ensemble seed")
    try:
        fitted = (fit_tcn(x, y, meta, base, config, seed, device) if kind == "tcn"
                  else fit_base(x, y, meta, base, config))
        before = source_fingerprint(fitted, kind)
        if checkpoint_path is not None:
            if kind == "tcn":
                save_checkpoint(fitted, checkpoint_path)
            else:
                joblib.dump(fitted, checkpoint_path, compress=3)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        status, reason = "fallback", f"{type(exc).__name__}: {exc}"
    audit = {"actual_outcome": outcome, "kind": kind, "pool": mode, "base": base,
             "seed": seed, "device": device, "status": status, "reason": reason,
             "source_countries": list(meta.country.unique()), "training_windows": len(meta),
             "minimum_input_year": int(meta.input_start.min()), "maximum_label_year": int(meta.label_end.max()),
             "target_in_base_fit": bool(meta.country.eq(config["primary_target"]).any()),
             "fingerprint_before": before, "parameter_count": fitted["parameter_count"] if fitted else 0,
             "feature_mean": fitted["scaler"].mean_.tolist() if fitted else [],
             "feature_scale": fitted["scaler"].scale_.tolist() if fitted else [],
             "feature_variance": fitted["scaler"].var_.tolist() if fitted else [],
             "training_losses": fitted.get("training_losses", []) if fitted else []}
    payloads = {}
    for budget in budgets:
        restricted = context(panel, config, outcome, budget)
        current_x, levels, current_meta, target_x, target_y, target_meta = target_arrays(restricted, config, budget)
        predictor = tcn_predict if kind == "tcn" else predict_changes
        budget_status, budget_reason = status, reason
        try:
            if fitted is None or status != "ok":
                raise RuntimeError(reason)
            current = predictor(fitted, current_x)
            historical = predictor(fitted, target_x) if mode == "donor" else np.zeros_like(target_y)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            current, historical = np.zeros((len(current_x), 5)), np.zeros_like(target_y)
            budget_status, budget_reason = "fallback", f"{type(exc).__name__}: {exc}"
        payloads[budget] = {"origin": config["learning_curve"]["origin"], "base": base,
            "seed": seed, "current_changes": current, "target_changes": historical,
            "levels": levels, "current_meta": current_meta, "target_y": target_y, "target_meta": target_meta,
            "audit": {**audit, "status": budget_status, "reason": budget_reason,
                      "history_years": budget, "target_windows": len(target_meta)}}
    after = source_fingerprint(fitted, kind) if fitted else "unavailable"
    if after != before:
        raise ValueError("Frozen source predictor changed during target inference")
    audit["fingerprint_after"] = after
    for payload in payloads.values():
        payload["audit"]["fingerprint_after"] = after
    return {"audit": audit, "payloads": payloads, "training_meta": meta.assign(sample_weight=weights)}


def nonneural_forecasts(payload, config, mode, kind):
    """Matched frozen-base zero/two-coefficient controls, using only eligible labels."""
    if mode not in {"donor", "pooled"} or kind not in {"ridge", "boosting"}:
        raise ValueError("Unknown non-neural family")
    modes = [False, True] if mode == "donor" else [False]
    rows, adaptations = [], []
    for adapted in modes:
        family = f"{mode}_{kind}" + ("_adapted" if adapted else "_unadapted") if mode == "donor" else f"pooled_{kind}"
        for sex in config["sexes"]:
            take = payload["current_meta"].sex.eq(sex).to_numpy()
            eligible = payload["target_meta"].sex.eq(sex).to_numpy()
            changes = payload["current_changes"][take].copy()
            status, reason = payload["audit"]["status"], payload["audit"]["reason"]
            calibration = None
            if adapted and status == "ok":
                try:
                    calibration = fit_adaptation(payload["target_y"][eligible], payload["target_changes"][eligible],
                                                 config["cold_start_defaults"]["adaptation_penalty"])
                    changes += correction(calibration)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    status, reason = "fallback", f"Adaptation: {type(exc).__name__}: {exc}"
                    changes[:] = 0
            if adapted:
                adaptations.append({"family": family, "sex": sex, "origin": payload["origin"],
                    "target_windows": int(eligible.sum()), "windows_per_age": int(eligible.sum()/len(config["ages"])),
                    "first_target_input_year": int(payload["target_meta"].input_start.min()),
                    "last_target_label_year": int(payload["target_meta"].label_end.max()),
                    "status": status, "reason": reason, **(calibration or {})})
            logs = payload["levels"][take, None] + changes
            for i, (_, row) in enumerate(payload["current_meta"].loc[take].iterrows()):
                row_status, row_reason = status, reason
                if not np.isfinite(logs[i]).all() or (np.abs(logs[i]) > 700).any():
                    logs[i] = payload["levels"][take][i]
                    row_status, row_reason = "fallback", "Unsafe log forecast"
                for h in config["calendar"]["horizons"]:
                    rows.append({"target": config["primary_target"], "outcome": "prevalence", "sex": sex,
                        "age": row.age, "origin": payload["origin"], "horizon": h,
                        "forecast_year": payload["origin"]+h, "family": family,
                        "setting_id": family+"__fixed_defaults", "log_prediction": float(logs[i, h-1]),
                        "prediction": float(np.exp(logs[i, h-1])), "status": row_status, "fallback_reason": row_reason,
                        "parameter_count": payload["audit"]["parameter_count"]+(2 if adapted else 0),
                        "source_fingerprint": payload["audit"]["fingerprint_before"],
                        "target_in_base_fit": mode == "pooled", "adaptation": "two_parameter" if adapted else "none"})
    return rows, adaptations


def check_ledger(frame, config):
    expected = {(task["outcome"], task["history_years"], family, sex, age, h)
                for task in tasks(config) for family in families(config)
                for sex in config["sexes"] for age in config["ages"] for h in config["calendar"]["horizons"]}
    if frame.duplicated(KEYS).any() or set(frame[KEYS].itertuples(index=False, name=None)) != expected:
        raise ValueError("Learning-curve ledger must contain every complete arm and family")
    if not frame.target.eq(config["primary_target"]).all() or not frame.origin.eq(config["learning_curve"]["origin"]).all():
        raise ValueError("Unexpected target or origin")
    if not frame.forecast_year.eq(frame.origin+frame.horizon).all():
        raise ValueError("Invalid forecast year")
    if (not np.isfinite(frame[["prediction", "log_prediction"]]).all().all()
            or frame.prediction.le(0).any() or not frame.status.isin(["ok", "fallback"]).all()):
        raise ValueError("Unsafe forecast values or missing fit status")
    np.testing.assert_allclose(np.log(frame.prediction), frame.log_prediction, atol=2e-14, rtol=2e-14)
    if not frame.target_history_start.eq(frame.origin-frame.history_years+1).all():
        raise ValueError("Budget year metadata mismatch")
    for family in ["donor_ridge_unadapted", "donor_boosting_unadapted", "tcn_unadapted"]:
        wide = frame.loc[frame.family.eq(family)].pivot(index=["outcome", "sex", "age", "horizon"],
                                                       columns="history_years", values="log_prediction")
        np.testing.assert_array_equal(wide.to_numpy(), np.repeat(wide.iloc[:, :1].to_numpy(), len(wide.columns), axis=1))
    return {"prediction_rows": len(frame), "arms": len(tasks(config)),
            "shared_unadapted_predictions_identical": True}


def summarize(scored, config):
    tables = []
    for band, ages in [("45+", config["ages"]), ("80+", config["ages"][-4:])]:
        part = scored.loc[scored.age.isin(ages)]
        table = part.groupby(["target", "outcome", "history_years", "sex", "family", "horizon"], as_index=False).agg(
            mean_absolute_log_error=("absolute_log_error", "mean"), mean_absolute_rate_error=("absolute_rate_error", "mean"),
            age_cells=("age", "size"), fallback_cells=("status", lambda v: int(v.eq("fallback").sum())))
        table["age_group"] = band
        tables.append(table)
    summary = pd.concat(tables, ignore_index=True)
    keys = ["target", "outcome", "sex", "family", "horizon", "age_group"]
    complete = max(config["learning_curve"]["history_years"])
    paired = summary.merge(summary.loc[summary.history_years.eq(complete), keys+["mean_absolute_log_error"]],
                           on=keys, validate="many_to_one", suffixes=("", "_full_history"))
    paired["absolute_log_error_change_vs_full_history"] = paired.mean_absolute_log_error-paired.mean_absolute_log_error_full_history
    paired["relative_error_increase_vs_full_history_percent"] = np.where(
        paired.mean_absolute_log_error_full_history.gt(0),
        100*paired.absolute_log_error_change_vs_full_history/paired.mean_absolute_log_error_full_history, np.nan)
    comparisons = []
    keys = ["target", "outcome", "history_years", "sex", "horizon", "age_group"]
    for prefix in ["tcn", "donor_ridge", "donor_boosting"]:
        item = summary.loc[summary.family.eq(prefix+"_adapted")].merge(
            summary.loc[summary.family.eq(prefix+"_unadapted")], on=keys, validate="one_to_one",
            suffixes=("_adapted", "_unadapted"))
        item["base_family"] = prefix
        item["absolute_log_error_change_from_adaptation"] = item.mean_absolute_log_error_adapted-item.mean_absolute_log_error_unadapted
        comparisons.append(item)
    return summary, paired, pd.concat(comparisons, ignore_index=True)
