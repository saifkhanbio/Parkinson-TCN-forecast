"""Replay final-origin non-neural forecasts from frozen checkpoints, without fits."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
import pandas as pd

from gbd_park.pooled import predict_changes, target_inputs
from run_local_baselines import check_lock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_run(directory):
    record = json.loads((directory / "run_manifest.json").read_text())
    if record["status"] != "complete" or not record["final_period_scored"]:
        raise ValueError("Checkpoint replay requires the completed frozen primary evaluation")
    for name, expected in record["output_sha256"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Changed run artifact: {name}")
    for name, expected in record["code_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed evaluation source: {name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/primary_v1")
    parser.add_argument("--output", default="work/primary-validation/replay_nonneural.json")
    args = parser.parse_args()
    directory, output = ROOT / args.run, ROOT / args.output
    if output.exists():
        raise FileExistsError("Refusing to overwrite prior independent replay evidence")
    check_lock()
    manifest = verify_run(directory)
    manifest_hash = sha(directory / "run_manifest.json")
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    origin = config["calendar"]["primary_origin"]
    families = config["models"]["nonneural_order"]
    predictions = pd.read_csv(directory / "predictions.csv", float_precision="round_trip")
    if "observed_rate" in predictions:
        raise ValueError("Replay must read predictions without verification values")
    selected = predictions.loc[predictions.origin.eq(origin) & predictions.family.isin(families)].copy()
    keys = ["family", "sex", "age", "horizon"]
    expected_keys = {(f, s, a, h) for f in families for s in config["sexes"]
                     for a in config["ages"] for h in config["calendar"]["horizons"]}
    if selected.duplicated(keys).any() or set(map(tuple, selected[keys].to_numpy())) != expected_keys:
        raise ValueError("The six-family final-origin forecast grid is incomplete")
    if len(selected) != 660 or not selected.target.eq(config["primary_target"]).all():
        raise ValueError("Unexpected final-origin replay scope")

    # Match the original feature loader; no outcome after issuance enters replay.
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")].copy()
    x, levels, meta = target_inputs(panel, config, origin, config["primary_target"])
    fits = [row for row in json_lines(directory / "nonneural_fit_audit.jsonl") if row["origin"] == origin]
    corrections = [row for row in json_lines(directory / "nonneural_adaptation_audit.jsonl") if row["origin"] == origin]
    groups, seen_checkpoints = [], set()
    maximum_rate_difference = maximum_log_difference = 0.0
    total_cells = checkpoint_cells = fallback_cells = 0
    for (family, sex, setting_id, model_id), part in selected.groupby(
            ["family", "sex", "setting_id", "model_id"], dropna=False):
        fit_rows = [row for row in fits if row["model_id"] == model_id]
        if len(fit_rows) != 1:
            raise ValueError("Selected forecast must match exactly one base-fit audit")
        audit = fit_rows[0]
        take = meta.sex.eq(sex).to_numpy()
        selected_ages = meta.loc[take, "age"].tolist()
        fitted = None
        checkpoint_hash = None
        audit_fingerprint = audit["base_fingerprint_before"]
        before = audit_fingerprint
        if audit_fingerprint != audit["base_fingerprint_after"]:
            raise ValueError("Original source estimator changed during fitting-time inference/adaptation")
        if not part.base_fingerprint.eq(audit_fingerprint).all():
            raise ValueError("Prediction ledger fingerprint differs from the original fit audit")
        if audit["status"] == "ok":
            checkpoint = directory / "nonneural_checkpoints" / str(origin) / f"{model_id}.joblib"
            checkpoint_hash = sha(checkpoint)
            if manifest["output_sha256"][str(checkpoint.relative_to(directory))] != checkpoint_hash:
                raise ValueError("Checkpoint file does not match the committed run")
            fitted = joblib.load(checkpoint)
            before = joblib.hash(fitted, hash_name="sha1")
            # Generic joblib hashes need not survive serialization of sklearn
            # boosting objects. The committed file SHA protects the saved
            # checkpoint; a separately loaded object must have the same hash.
            if joblib.hash(joblib.load(checkpoint), hash_name="sha1") != before:
                raise ValueError("Repeated loads produce inconsistent checkpoint state hashes")
            np.testing.assert_allclose(fitted["scaler"].mean_, audit["feature_mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(fitted["scaler"].scale_, audit["feature_scale"], rtol=0, atol=1e-12)
            if fitted["base"] != audit["base"]:
                raise ValueError("Checkpoint base settings differ from the original fit audit")
            # Predict all source-order target inputs, exactly as the fitting routine did.
            changes = predict_changes(fitted, x)[take].copy()
            seen_checkpoints.add(str(checkpoint.relative_to(directory)))
        else:
            if not part.status.eq("fallback").all() or before != "unavailable":
                raise ValueError("Failed source fit lacks the prescribed persistence fallback")
            changes = np.zeros((len(selected_ages), len(config["calendar"]["horizons"])))

        calibration = None
        if part.adaptation.eq("two_parameter").all():
            matches = [row for row in corrections if row["sex"] == sex
                       and row["setting_id"] == setting_id and row["model_id"] == model_id]
            if len(matches) != 1:
                raise ValueError("Adapted forecast must match exactly one frozen correction record")
            calibration = matches[0]
            if calibration["last_target_label_year"] > origin:
                raise ValueError("Saved adaptation used a label after issuance")
            if calibration["status"] == "ok":
                coefficients = np.asarray([calibration["b0"], calibration["b1"]], dtype=float)
                if not np.isfinite(coefficients).all() or calibration["penalty"] <= 0:
                    raise ValueError("Invalid saved adaptation coefficients")
                # Apply saved coefficients directly; do not call the adaptation optimizer.
                changes += calibration["b0"] + calibration["b1"] * np.arange(1, 6) / 5
            else:
                changes[:] = 0
        elif not part.adaptation.eq("none").all():
            raise ValueError("Unexpected or mixed adaptation type in a selected forecast")

        logs = levels[take, None] + changes
        unsafe = ~np.isfinite(logs).all(axis=1) | (np.abs(logs) > 700).any(axis=1)
        logs[unsafe] = levels[take][unsafe, None]
        columns = config["calendar"]["horizons"]
        expected_rate = part.pivot(index="age", columns="horizon", values="prediction").reindex(
            index=selected_ages, columns=columns).to_numpy(dtype=float)
        expected_log = part.pivot(index="age", columns="horizon", values="log_prediction").reindex(
            index=selected_ages, columns=columns).to_numpy(dtype=float)
        reproduced_rate = np.exp(logs)
        np.testing.assert_allclose(logs, expected_log, rtol=0, atol=1e-12)
        np.testing.assert_allclose(reproduced_rate, expected_rate, rtol=1e-12, atol=1e-12)
        if fitted is not None and joblib.hash(fitted, hash_name="sha1") != before:
            raise ValueError("Inference/correction mutated a saved base estimator")
        group_fallbacks = int(part.status.eq("fallback").sum())
        if group_fallbacks:
            statuses = part.pivot(index="age", columns="horizon", values="status").reindex(
                index=selected_ages, columns=columns).to_numpy()
            expected_persistence = np.broadcast_to(levels[take, None], logs.shape)
            np.testing.assert_allclose(logs[statuses == "fallback"],
                                       expected_persistence[statuses == "fallback"], rtol=0, atol=1e-12)
        total_cells += expected_rate.size
        fallback_cells += group_fallbacks
        checkpoint_cells += expected_rate.size - group_fallbacks
        maximum_rate_difference = max(maximum_rate_difference, float(np.max(np.abs(reproduced_rate - expected_rate))))
        maximum_log_difference = max(maximum_log_difference, float(np.max(np.abs(logs - expected_log))))
        groups.append({"family": family, "sex": sex, "setting_id": setting_id,
                       "model_id": model_id, "cells": int(expected_rate.size),
                       "fallback_cells": group_fallbacks,
                       "audit_base_fingerprint": audit_fingerprint, "loaded_base_fingerprint": before,
                       "audit_and_loaded_fingerprints_equal": audit_fingerprint == before,
                       "checkpoint_sha256": checkpoint_hash,
                       "adaptation": "none" if calibration is None else "saved_two_parameter_correction",
                       "b0": None if calibration is None else calibration.get("b0"),
                       "b1": None if calibration is None else calibration.get("b1")})

    if total_cells != 660 or len(groups) != 12:
        raise ValueError("Replay did not preserve all twelve family/sex grids")
    if sha(directory / "run_manifest.json") != manifest_hash:
        raise ValueError("Run manifest changed during replay")
    verify_run(directory)
    check_lock()
    report = {"passed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": directory.name, "origin": origin, "families": families,
              "replayed_cells": total_cells, "checkpoint_replayed_nonfallback_cells": checkpoint_cells,
              "verified_persistence_fallback_cells": fallback_cells, "family_sex_groups": len(groups),
              "unique_checkpoints_loaded": len(seen_checkpoints),
              "maximum_absolute_rate_difference": maximum_rate_difference,
              "maximum_absolute_log_difference": maximum_log_difference,
              "run_manifest_sha256": manifest_hash, "script_sha256": sha(Path(__file__)),
              "models_refitted": False, "adaptation_refitted": False,
              "evaluation_scores_parsed": False, "post_origin_outcomes_used": False,
              "frozen_checkpoint_and_correction_verified": True, "model_states_unchanged": True,
              "original_in_memory_hash_matches_loaded_groups": sum(g["audit_and_loaded_fingerprints_equal"] for g in groups),
              "serialization_hash_note": "Initial replay required original in-memory joblib hashes to equal loaded hashes. "
                  "Boosting objects did not satisfy that assumption although checkpoint file hashes were intact. "
                  "Replay instead verifies committed file SHA-256, original before/after audit hashes, repeated-load "
                  "and loaded before/after state hashes, saved scaler moments, and all predicted cells. "
                  "Both original and loaded fingerprints are retained; no model or forecast artifact changed.",
              "groups": groups}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
