"""Chronological TCN fitting, ensemble-first target adaptation, and selection."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from gbd_park.adaptation import correction, fit_adaptation, regularized_location
from gbd_park.pooled import balanced_weights, build_examples, country_pool, target_inputs
from gbd_park.tcn import base_grid, fit_tcn, predict_changes, save_checkpoint, state_fingerprint


def base_id(base):
    return "__".join(f"{key}={base[key]}" for key in ["channels", "weight_decay", "epochs"])


def parameter_count(base):
    channels = int(base["channels"])
    return 6 * channels * channels + 11 * channels + 70


def fit_seed(panel, config, origin, base, seed, device="cpu", checkpoint_path=None):
    """Fit one source model and return frozen predictions for later ensembling."""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; execution policy must not silently change")
    started = time.perf_counter()
    target = config["primary_target"]
    countries = country_pool(config, target, "donor")
    x, y, meta = build_examples(panel, config, origin, countries)
    current_x, levels, current_meta = target_inputs(panel, config, origin, target)
    target_x, target_y, target_meta = build_examples(panel, config, origin, [target])
    weights = balanced_weights(meta)
    assert not meta.country.eq(target).any() and meta.label_end.le(origin).all()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    audit = {"origin": origin, "base": dict(base), "base_id": base_id(base), "seed": seed,
             "model_id": f"origin{origin}__{base_id(base)}__seed={seed}",
             "device": str(device), "status": "ok", "reason": "", "countries": countries,
             "training_windows": len(meta), "target_windows": len(target_meta),
             "maximum_input_year": int(meta.input_end.max()), "maximum_label_year": int(meta.label_end.max()),
             "last_target_label_year": int(target_meta.label_end.max()),
             "parameter_count": parameter_count(base), "target_in_base_fit": False}
    try:
        fitted = fit_tcn(x, y, meta, base, config, seed, device)
        before = state_fingerprint(fitted)
        current_changes = predict_changes(fitted, current_x)
        target_changes = predict_changes(fitted, target_x)
        after = state_fingerprint(fitted)
        assert before == after, "TCN state changed during inference"
        assert fitted["parameter_count"] == parameter_count(base)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(fitted, checkpoint_path)
        audit.update(fingerprint_before=before, fingerprint_after=after,
                     training_losses=fitted["training_losses"],
                     feature_mean=fitted["scaler"].mean_.tolist(), feature_scale=fitted["scaler"].scale_.tolist())
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        current_changes = np.zeros((len(current_x), 5))
        target_changes = np.zeros_like(target_y)
        audit.update(status="fallback", reason=f"{type(exc).__name__}: {exc}",
                     fingerprint_before="unavailable", fingerprint_after="unavailable")
    audit["elapsed_seconds"] = time.perf_counter() - started
    audit["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated()) if str(device).startswith("cuda") else 0
    return {"origin": origin, "base": dict(base), "seed": int(seed), "current_changes": current_changes,
            "target_changes": target_changes, "target_y": target_y, "levels": levels,
            "current_meta": current_meta, "target_meta": target_meta, "audit": audit,
            "training_meta": meta.assign(sample_weight=weights, fit_origin=origin)}


def make_forecasts(payloads, config, penalties, include_intercept=False):
    """Average seed log changes, then fit one correction per sex and penalty."""
    if not payloads:
        raise ValueError("An ensemble cannot be empty")
    payloads = sorted(payloads, key=lambda item: item["seed"])
    first = payloads[0]
    seeds = [p["seed"] for p in payloads]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate ensemble seeds")
    for p in payloads:
        if p["origin"] != first["origin"] or p["base"] != first["base"]:
            raise ValueError("Ensemble base setting/origin mismatch")
        for key in ["current_meta", "target_meta"]:
            if not p[key].equals(first[key]):
                raise ValueError("Ensemble metadata mismatch")
        for key in ["levels", "target_y"]:
            np.testing.assert_array_equal(p[key], first[key])
    failed = [p for p in payloads if p["audit"]["status"] != "ok"]
    # A failed seed remains in the ledger; the complete ensemble falls back.
    current = np.mean([p["current_changes"] for p in payloads], axis=0)
    historical = np.mean([p["target_changes"] for p in payloads], axis=0)
    fingerprint = hashlib.sha256(json.dumps(
        [(p["seed"], p["audit"]["fingerprint_before"]) for p in payloads]).encode()).hexdigest()
    origin, base = first["origin"], first["base"]
    meta, training = first["current_meta"], first["target_meta"]
    grid = base_grid(config)
    grid_index = grid.index(base) if base in grid else 0
    records, calibrations = [], []
    for sex in config["sexes"]:
        sex_penalties = [penalties[sex]] if isinstance(penalties, dict) else penalties
        modes = [("tcn_unadapted", None)] + [("tcn_adapted", penalty) for penalty in sex_penalties]
        if include_intercept:
            modes += [("tcn_intercept", penalty) for penalty in sex_penalties]
        take = meta.sex.eq(sex).to_numpy()
        eligible = training.sex.eq(sex).to_numpy()
        for family, penalty in modes:
            changes = current[take].copy()
            cal, status, reason = None, "ok", ""
            if failed:
                status = "fallback"
                reason = "Failed ensemble seed(s): " + "; ".join(f"{p['seed']}: {p['audit']['reason']}" for p in failed)
                changes[:] = 0
            elif penalty is not None:
                try:
                    if family == "tcn_adapted":
                        cal = fit_adaptation(first["target_y"][eligible], historical[eligible], penalty)
                    else:
                        residual = (first["target_y"][eligible] - historical[eligible]).ravel()
                        b0 = regularized_location(residual, penalty)
                        cal = {"b0": b0, "b1": 0.0, "penalty": penalty, "status": "ok",
                               "objective": float(np.mean(np.abs(residual - b0)) + penalty*b0*b0),
                               "unadapted_mae": float(np.mean(np.abs(residual)))}
                    changes += correction(cal)
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    status, reason = "fallback", f"Adaptation: {type(exc).__name__}: {exc}"
                    changes[:] = 0
            ident = family + "__" + base_id(base) + ("" if penalty is None else f"__adaptation_penalty={penalty}")
            if penalty is not None:
                calibrations.append({"origin": origin, "sex": sex, "family": family, "setting_id": ident,
                                     "base_id": base_id(base), "seed_or_ensemble": "|".join(map(str, seeds)),
                                     "ensemble_fingerprint": fingerprint, "penalty": penalty,
                                     "target_windows": int(eligible.sum()),
                                     "last_target_label_year": int(training.loc[eligible, "label_end"].max()),
                                     **({} if cal is None else cal), "status": status, "reason": reason})
            log_rates = first["levels"][take, None] + changes
            for a, (_, row) in enumerate(meta.loc[take].iterrows()):
                row_status, row_reason = status, reason
                if not np.isfinite(log_rates[a]).all() or (np.abs(log_rates[a]) > 700).any():
                    log_rates[a] = first["levels"][take][a]
                    row_status, row_reason = "fallback", "Unsafe log forecast"
                for h in range(1, 6):
                    records.append({"target": config["primary_target"], "sex": sex, "age": row.age,
                        "outcome": "prevalence", "origin": origin, "horizon": h, "forecast_year": origin+h,
                        "family": family, "setting_id": ident, "base_id": base_id(base),
                        "grid_order": grid_index*4 + (config["adaptation"]["penalties"].index(penalty) if penalty is not None else 0),
                        "base_json": json.dumps(base, sort_keys=True), "adaptation_penalty": penalty,
                        "log_prediction": float(log_rates[a, h-1]), "prediction": float(np.exp(log_rates[a, h-1])),
                        "parameter_count": parameter_count(base) + (2 if family == "tcn_adapted" else int(penalty is not None)),
                        "fitted_seed_parameters": len(seeds)*parameter_count(base),
                        "status": row_status, "fallback_reason": row_reason,
                        "seed_or_ensemble": "|".join(map(str, seeds)), "ensemble_fingerprint": fingerprint,
                        "donor_strategy": config["donors"]["primary"], "target_in_base_fit": False,
                        "adaptation": {"tcn_unadapted": "none", "tcn_adapted": "two_parameter", "tcn_intercept": "intercept_only"}[family]})
    return records, calibrations


def select_tcn_settings(scores, config, origin):
    """Select each sex's penalty within each base, then an equal-sex base."""
    origins = list(range(config["calendar"]["inner_first_origin"], origin - 4))
    if not origins:
        defaults = config["cold_start_defaults"]
        return {"fit_origin": origin, "base": {"channels": defaults["tcn_channels"],
                    "weight_decay": defaults["tcn_weight_decay"], "epochs": defaults["tcn_epochs"]},
                "penalties": {sex: defaults["adaptation_penalty"] for sex in config["sexes"]},
                "inner_origins": [], "last_inner_label_year": None, "loss": None,
                "status": "cold_start_defaults"}
    rows = scores.loc[scores.family.eq("tcn_adapted") & scores.origin.isin(origins) & scores.horizon.eq(5)]
    candidates = []
    for index, base in enumerate(base_grid(config)):
        penalties, losses = {}, {}
        for sex in config["sexes"]:
            choices = []
            for penalty_index, penalty in enumerate(config["adaptation"]["penalties"]):
                part = rows[rows.base_id.eq(base_id(base)) & rows.sex.eq(sex) & rows.adaptation_penalty.eq(penalty)]
                if set(zip(part.origin, part.age)) != {(o, a) for o in origins for a in config["ages"]} or len(part) != len(origins)*len(config["ages"]):
                    raise ValueError("Incomplete chronological TCN tuning grid")
                loss = float(part.absolute_log_error.mean())
                if not np.isfinite(loss):
                    raise ValueError("Nonfinite TCN selection loss")
                choices.append((loss, penalty_index, penalty))
            loss, _, penalty = min(choices)
            penalties[sex], losses[sex] = penalty, loss
        candidates.append({"fit_origin": origin, "base": base, "penalties": penalties,
                           "inner_origins": origins, "last_inner_label_year": max(origins)+5,
                           "loss": float(np.mean(list(losses.values()))), "loss_by_sex": losses,
                           "parameter_count": parameter_count(base), "grid_order": index, "status": "tuned"})
    return min(candidates, key=lambda item: (item["loss"], item["parameter_count"], item["grid_order"]))
