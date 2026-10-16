"""Post-result exploratory corrections with explicitly chronological selection."""

import copy
import json

import numpy as np
import pandas as pd

from gbd_park.adaptation import regularized_location
from gbd_park.intervals import apply_bank


def age_setting(penalty):
    return "age_penalty=" + ("none" if penalty is None else str(penalty))


def group_labels(meta, config):
    lookup = {int(age): group for group, ages in config["age_groups"].items() for age in ages}
    starts = meta.age.map(lambda age: int(str(age).split("-")[0].rstrip("+")))
    labels = starts.map(lookup)
    if labels.isna().any():
        raise ValueError("Age correction contains an unknown broad-age group")
    return labels.to_numpy()


def fit_group_offsets(residuals, meta, penalty, config):
    """Minimize overall mean absolute error plus lambda times three squared offsets."""
    residuals = np.asarray(residuals, dtype=float)
    if residuals.shape != (len(meta), 5) or not len(meta) or not np.isfinite(residuals).all():
        raise ValueError("Age residuals require aligned finite five-horizon cells")
    groups = group_labels(meta, config)
    if set(groups) != set(config["age_groups"]):
        raise ValueError("All three broad-age groups are required")
    if penalty is not None and (not np.isfinite(penalty) or penalty <= 0):
        raise ValueError("A fitted age penalty must be positive and finite")
    result = {}
    for group in config["age_groups"]:
        values = residuals[groups == group].ravel()
        fraction = len(values) / residuals.size
        offset = 0.0 if penalty is None else regularized_location(values, penalty / fraction)
        bound = 0.0 if penalty is None else fraction / (2 * penalty)
        if not np.isfinite(offset) or abs(offset) > bound + 1e-12:
            raise ValueError("Age correction violates its regularization bound")
        result[group] = {"offset": float(offset), "cell_fraction": float(fraction),
                         "absolute_offset_bound": float(bound), "cells": len(values)}
    adjustments = np.asarray([result[group]["offset"] for group in groups])[:, None]
    objective = float(np.mean(np.abs(residuals - adjustments)))
    if penalty is not None:
        objective += penalty * sum(row["offset"] ** 2 for row in result.values())
    return result, objective


def candidate_forecasts(payloads, calibration_rows, baseline, config, amendment):
    """Reuse original ensemble/sex correction, fitting only small new age offsets."""
    payloads = sorted(payloads, key=lambda item: item["seed"])
    if [p["seed"] for p in payloads] != config["models"]["tcn"]["ensemble_seeds"]:
        raise ValueError("All original ensemble seeds must be retained")
    first = payloads[0]
    origin = first["origin"]
    for payload in payloads:
        if payload["origin"] != origin or payload["base"] != first["base"]:
            raise ValueError("Original ensemble origin or architecture differs")
        for name in ["target_meta", "current_meta"]:
            if not payload[name].equals(first[name]):
                raise ValueError("Original ensemble metadata differs")
        for name in ["target_y", "levels"]:
            np.testing.assert_array_equal(payload[name], first[name])
        if payload["audit"]["fingerprint_before"] != payload["audit"]["fingerprint_after"]:
            raise ValueError("Original neural source state was not frozen")
    if not first["target_meta"].label_end.le(origin).all():
        raise ValueError("Age adaptation cannot use future target labels")
    if not baseline.origin.eq(origin).all() or len(baseline) != 110:
        raise ValueError("Original point forecast has the wrong complete-grid origin")
    historical = np.mean([p["target_changes"] for p in payloads], axis=0)
    current = np.mean([p["current_changes"] for p in payloads], axis=0)
    records, audits = [], []
    for sex in config["sexes"]:
        target_mask = first["target_meta"].sex.eq(sex).to_numpy()
        current_mask = first["current_meta"].sex.eq(sex).to_numpy()
        original = baseline[baseline.sex.eq(sex)].copy()
        if original.duplicated(["age", "horizon"]).any():
            raise ValueError("Original point forecast has duplicate age/horizon cells")
        chosen = [c for c in calibration_rows if c["origin"] == origin and c["sex"] == sex and c["family"] == "tcn_adapted"]
        if len(chosen) != 1:
            raise ValueError("Exactly one original sex correction is required")
        cal = chosen[0]
        if cal["last_target_label_year"] > origin:
            raise ValueError("Original correction includes a future label")
        success = cal["status"] == "ok" and all(p["audit"]["status"] == "ok" for p in payloads)
        shift = (cal["b0"] + cal["b1"] * np.arange(1, 6) / 5) if success else np.zeros(5)
        ordered_ages = first["current_meta"].loc[current_mask, "age"].tolist()
        expected = original.pivot(index="age", columns="horizon", values="log_prediction").reindex(
            index=ordered_ages, columns=range(1, 6)).to_numpy()
        reconstructed = first["levels"][current_mask, None] + (current[current_mask] + shift if success else 0)
        np.testing.assert_allclose(reconstructed, expected, rtol=0, atol=1e-12)
        residuals = first["target_y"][target_mask] - historical[target_mask] - shift
        target_meta = first["target_meta"].loc[target_mask].copy()
        for index, penalty in enumerate(amendment["age_candidates"]):
            status, reason = "ok", ""
            try:
                if not success:
                    raise ValueError("Original ensemble/correction used a fallback")
                offsets, objective = fit_group_offsets(residuals, target_meta, penalty, config)
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                offsets = {g: {"offset": 0.0, "cell_fraction": None,
                               "absolute_offset_bound": None, "cells": 0} for g in config["age_groups"]}
                objective, status, reason = None, "fallback", str(exc)
            frame = original.copy()
            frame["original_setting_id"] = frame.setting_id
            frame["original_log_prediction"] = frame.log_prediction
            labels = group_labels(frame, config)
            frame["age_group"] = labels
            frame["age_offset"] = [offsets[g]["offset"] for g in labels]
            frame["log_prediction"] = frame.log_prediction + frame.age_offset
            frame["prediction"] = np.exp(frame.log_prediction)
            if not np.isfinite(frame.prediction).all() or not frame.prediction.gt(0).all():
                raise ValueError("Age correction generated invalid rates")
            frame["family"] = "tcn_age_candidate"
            frame["setting_id"], frame["grid_order"] = age_setting(penalty), index
            frame["age_penalty"] = penalty
            frame["parameter_count"] = frame.parameter_count + (0 if penalty is None else 3)
            frame["status"], frame["fallback_reason"] = status, reason
            records.extend(frame.to_dict("records"))
            audits.append({"origin": origin, "sex": sex, "setting_id": age_setting(penalty),
                           "penalty": penalty, "offsets": offsets, "objective": objective,
                           "original_b0": cal.get("b0"), "original_b1": cal.get("b1"),
                           "original_correction_refitted": False,
                           "last_target_label_year": int(target_meta.label_end.max()),
                           "target_windows": len(target_meta), "status": status, "reason": reason})
    return records, audits


def select_age(scores, config, amendment, origin, sex):
    origins = list(range(config["calendar"]["inner_first_origin"], origin - 4))
    if not origins:
        penalty = amendment["age_cold_start_penalty"]
        return {"origin": origin, "sex": sex, "setting_id": age_setting(penalty), "penalty": penalty,
                "inner_origins": [], "last_label_year": None, "loss": None, "status": "cold_start"}
    eligible = scores[scores.origin.isin(origins) & scores.sex.eq(sex) & scores.horizon.eq(5)]
    options = []
    for index, penalty in enumerate(amendment["age_candidates"]):
        part = eligible[eligible.setting_id.eq(age_setting(penalty))]
        expected = {(o, a) for o in origins for a in config["ages"]}
        if len(part) != len(expected) or set(zip(part.origin, part.age)) != expected:
            raise ValueError("Incomplete eligible age-correction tuning grid")
        if not np.isfinite(part.absolute_log_error).all() or part.absolute_log_error.lt(0).any():
            raise ValueError("Invalid eligible age-correction error")
        options.append((float(part.absolute_log_error.mean()), index, penalty))
    loss, _, penalty = min(options, key=lambda option: option[:2])
    return {"origin": origin, "sex": sex, "setting_id": age_setting(penalty), "penalty": penalty,
            "inner_origins": origins, "last_label_year": max(origins)+5, "loss": loss, "status": "tuned"}


def retention_family(point_family, retention):
    return f"{point_family}__retention={retention:g}"


def apply_retention(points, bank, config, retention_by_sex, output_family):
    if set(retention_by_sex) != set(config["sexes"]) or any(r not in [0, 0.5, 1] for r in retention_by_sex.values()):
        raise ValueError("Bias retention must assign a prescribed fraction to each sex")
    adjusted = copy.deepcopy(bank)
    fractions = adjusted["coords"].sex.map(retention_by_sex).to_numpy(dtype=float)
    adjusted["centered_residuals"] = bank["centered_residuals"] + fractions[None, :] * bank["center"][None, :]
    adjusted["centering"] = "residual_minus_one_minus_retention_times_historical_mean"
    adjusted["family"] = output_family
    frame = points.copy()
    original_family = frame.family.unique()
    if len(original_family) != 1 or original_family[0] != bank["family"]:
        raise ValueError("Point family must match its own prequential residual bank")
    frame["family"] = output_family
    intervals, draws = apply_bank(frame, adjusted, config)
    for output in [intervals, draws]:
        output["source_point_family"] = original_family[0]
        output["bias_retention"] = output.sex.map(retention_by_sex)
    return intervals, draws


def select_retention(wis, config, amendment, origin, sex, point_family):
    origins = [o for o in amendment["interval_development_origins"] if o + 5 <= origin]
    if not origins:
        return {"origin": origin, "sex": sex, "point_family": point_family,
                "retention": amendment["bias_retention_cold_start"], "inner_origins": [],
                "last_label_year": None, "loss": None, "status": "cold_start"}
    options = []
    for retention in amendment["bias_retention_candidates"]:
        part = wis[wis.origin.isin(origins) & wis.sex.eq(sex) & wis.horizon.eq(5)
                   & wis.scale.eq("log_rate") & wis.family.eq(retention_family(point_family, retention))]
        expected = {(o, a) for o in origins for a in config["ages"]}
        if len(part) != len(expected) or set(zip(part.origin, part.age)) != expected:
            raise ValueError("Incomplete completed interval-tuning evidence")
        if not np.isfinite(part.wis_50_80).all() or part.wis_50_80.lt(0).any():
            raise ValueError("Invalid completed interval-tuning WIS")
        options.append((float(part.wis_50_80.mean()), retention))
    loss, retention = min(options)
    return {"origin": origin, "sex": sex, "point_family": point_family, "retention": retention,
            "inner_origins": origins, "last_label_year": max(origins)+5, "loss": loss, "status": "tuned"}
