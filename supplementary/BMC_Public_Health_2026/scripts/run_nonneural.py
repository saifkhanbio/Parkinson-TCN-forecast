"""Stage 2: pooled and donor-only non-neural historical development."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import scipy
import sklearn
import joblib
from gbd_park.pooled import forecast_origin, settings_grid
from gbd_park.scoring import score_forecasts, select_settings, select_family
from run_local_baselines import check_lock, sha, now, json_lines


def verify_previous():
    directory = ROOT / "results/local_baselines_v1"
    record = json.loads((directory / "run_manifest.json").read_text())
    for name, expected in record["output_sha256"].items():
        assert sha(directory/name) == expected, name
    for name, expected in record["code_sha256"].items():
        assert sha(ROOT/name) == expected, name
    return sha(directory/"run_manifest.json")


def fit_job(origin, config, out):
    started = time.perf_counter()
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")].copy()
    checkpoints = out / "checkpoints" / str(origin)
    checkpoints.mkdir(parents=True)
    forecasts, fits, calibrations, training = forecast_origin(panel, config, origin, checkpoints)
    return {"origin": origin, "forecasts": forecasts, "fits": fits, "calibrations": calibrations,
            "training": training, "elapsed_seconds": time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/nonneural_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four workers")
    out = ROOT/args.output
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    check_lock()
    previous_hash = verify_previous()
    config_file = ROOT/"study_design/locked_v1/design.json"
    config = json.loads(config_file.read_text())
    tests = json.loads((ROOT/"work/nonneural-validation/tests.json").read_text())
    assert tests["passed"], "Run the validation tests first"
    for name, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT/name) == expected, f"Changed since testing: {name}"
    code = dict(tests["tested_code_sha256"])
    for name in ["scripts/run_local_baselines.py", "src/gbd_park/__init__.py", "src/gbd_park/local.py"]:
        code[name] = sha(ROOT/name)
    out.mkdir(parents=True)
    manifest = {"run_id": out.name, "protocol_version": config["version"], "created_utc": now(),
                "status": "running", "config_sha256": sha(config_file), "code_sha256": code,
                "source_lock_hash": sha(ROOT/"study_design/locked_v1/lock_manifest.json"),
                "stage1_manifest_sha256": previous_hash, "test_report_sha256": sha(ROOT/"work/nonneural-validation/tests.json"),
                "model_origins": list(range(2003, 2014)), "maximum_scored_year": 2018,
                "final_period_scored": False, "workers": args.workers,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    (out/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    events = []

    def event(name, **details):
        events.append({"time_utc": now(), "event": name, **details})
        (out/"events.json").write_text(json.dumps(events, indent=2)+"\n")

    started = time.perf_counter()
    event("fit_started")
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_job, origin, config, out) for origin in manifest["model_origins"]]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"Completed {len(completed)}/11: origin {result['origin']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    completed.sort(key=lambda item: item["origin"])
    forecasts = pd.DataFrame([r for result in completed for r in result["forecasts"]])
    forecasts["run_id"] = out.name
    forecasts["protocol_version"] = config["version"]
    forecasts["config_sha256"] = manifest["config_sha256"]
    forecasts["source_lock_hash"] = manifest["source_lock_hash"]
    forecasts["seed_or_ensemble"] = "11_for_boosting;ridge_deterministic"
    forecasts = forecasts.sort_values(["origin", "sex", "grid_order", "age", "horizon"]).reset_index(drop=True)
    assert len(forecasts) == 11*2*11*5*len(settings_grid(config)) == 94380
    assert forecasts.forecast_year.max() == 2018
    forecasts.to_csv(out/"candidate_predictions.csv", index=False)
    prediction_hash = sha(out/"candidate_predictions.csv")
    fits = [r for result in completed for r in result["fits"]]
    calibrations = [r for result in completed for r in result["calibrations"]]
    json_lines(out/"base_fit_audit.jsonl", fits)
    json_lines(out/"adaptation_audit.jsonl", calibrations)
    training = pd.DataFrame([r for result in completed for r in result["training"]])
    training.to_csv(out/"training_window_audit.csv", index=False)
    pd.DataFrame([{k: x[k] for k in ["origin", "elapsed_seconds"]} for x in completed]).to_csv(out/"fit_timings.csv", index=False)
    assert training.label_end.le(training.fit_origin).all()
    assert not training.loc[training.pool.eq("donor"), "country"].eq(config["primary_target"]).any()
    for _, group in training.groupby(["fit_origin", "pool"]):
        np.testing.assert_allclose(group.groupby("country").sample_weight.sum(), len(group)/group.country.nunique(), rtol=1e-12)
    assert all(f["base_fingerprint_before"] == f["base_fingerprint_after"] for f in fits)
    assert all(c["last_target_label_year"] <= c["origin"] for c in calibrations)
    event("candidate_predictions_committed", sha256=prediction_hash, rows=len(forecasts))
    event("development_scoring_started", maximum_verification_year=2018)
    panel = pd.read_csv(ROOT/"data/processed/design_v1/regional_outcomes.csv")
    truth = panel.loc[panel.location_name.eq(config["primary_target"]) & panel.outcome.eq("prevalence") & panel.year.le(2018)].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    saved = pd.read_csv(out/"candidate_predictions.csv")
    scores = score_forecasts(saved, truth, config["ages"], config["calendar"]["horizons"], 2018)
    scores.to_csv(out/"candidate_scores.csv", index=False)
    tuning, selections = [], []
    for origin in config["calendar"]["selection_origins"]:
        for sex in config["sexes"]:
            for family in config["models"]["nonneural_order"]:
                ident, loss, inner = select_settings(scores, config, origin, sex, family)
                tuning.append({"origin": origin, "sex": sex, "family": family, "setting_id": ident,
                               "inner_loss": loss, "inner_origins": "|".join(map(str, inner)), "last_inner_label_year": max(inner)+5})
                selections.append(saved.loc[saved.origin.eq(origin) & saved.sex.eq(sex) & saved.setting_id.eq(ident)])
    development = pd.concat(selections, ignore_index=True)
    assert len(development) == 3300
    development.to_csv(out/"development_predictions.csv", index=False)
    event("selected_development_predictions_committed", sha256=sha(out/"development_predictions.csv"))
    development_scores = score_forecasts(pd.read_csv(out/"development_predictions.csv"), truth,
                                        config["ages"], config["calendar"]["horizons"], 2018)
    development_scores.to_csv(out/"development_scores.csv", index=False)
    pd.DataFrame(tuning).to_csv(out/"tuning_decisions.csv", index=False)
    by_origin = development_scores.groupby(["origin", "sex", "family", "horizon"], as_index=False).agg(
        mean_age_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        fallback_cells=("status", lambda x: int(x.eq("fallback").sum())))
    by_origin.to_csv(out/"scores_by_origin.csv", index=False)
    summary = by_origin.groupby(["sex", "family", "horizon"], as_index=False).agg(
        mean_absolute_log_error=("mean_age_absolute_log_error", "mean"), mean_rate_mae=("rate_mae", "mean"),
        fallback_cells=("fallback_cells", "sum"))
    summary.to_csv(out/"development_summary.csv", index=False)
    selector_config = copy.deepcopy(config)
    selector_config["models"]["local_order"] = config["models"]["nonneural_order"]
    champions = []
    for origin in config["calendar"]["reliability_origins"]:
        for sex in config["sexes"]:
            winner = select_family(development_scores, selector_config, origin, sex)
            ident, loss, inner = select_settings(scores, config, origin, sex, winner["selected_family"])
            winner.update(setting_id=ident, inner_tuning_loss=loss, last_inner_label_year=max(inner)+5)
            champions.append(winner)
    pd.DataFrame(champions).to_csv(out/"nonneural_champions_for_later_evaluation.csv", index=False)
    adapted = development_scores[development_scores.family.str.endswith("_adapted")].copy()
    adapted["matched_family"] = adapted.family.str.replace("_adapted", "_unadapted", regex=False)
    match_keys = ["model_id", "sex", "age", "horizon", "matched_family"]
    unadapted = scores[scores.family.str.endswith("_unadapted")].rename(columns={
        "family": "matched_family", "base_fingerprint": "matched_fingerprint", "absolute_log_error": "matched_unadapted_error"})
    pairs = adapted.merge(unadapted[match_keys+["matched_fingerprint", "matched_unadapted_error"]],
                          on=match_keys, how="left", validate="one_to_one")
    assert pairs.matched_unadapted_error.notna().all()
    assert pairs.base_fingerprint.eq(pairs.matched_fingerprint).all()
    pairs["adaptation_loss_change"] = pairs.absolute_log_error - pairs.matched_unadapted_error
    pairs = pairs.rename(columns={"absolute_log_error": "adapted_error"})
    pairs[["origin", "sex", "age", "horizon", "family", "model_id", "adapted_error", "matched_unadapted_error", "adaptation_loss_change"]].to_csv(out/"matched_adaptation_comparisons.csv", index=False)
    assert sha(out/"candidate_predictions.csv") == prediction_hash
    assert all(t["last_inner_label_year"] <= t["origin"] for t in tuning)
    assert all(c["last_inner_label_year"] <= c["fit_origin"] and c["last_selection_target_year"] <= c["fit_origin"] for c in champions)
    check_lock()
    assert verify_previous() == previous_hash
    assert all(sha(ROOT/name) == digest for name, digest in code.items())
    validation = {"passed": True, "unit_tests": tests["tests_run"], "candidate_predictions": len(saved),
                  "selected_development_predictions": len(development), "base_fits": len(fits), "adaptations": len(calibrations),
                  "base_fit_failures": sum(f["status"] != "ok" for f in fits),
                  "selected_forecast_fallback_cells": int(development.status.eq("fallback").sum()),
                  "completed_labels_only": True, "donor_fit_excludes_saudi": True, "balanced_country_weights": True,
                  "base_models_unchanged_by_adaptation": True, "saved_predictions_before_scoring": True,
                  "stage1_and_locked_artifacts_unchanged": True, "maximum_scored_year": 2018,
                  "final_2019_2023_scores_produced": False, "elapsed_seconds": time.perf_counter()-started}
    (out/"validation_report.json").write_text(json.dumps(validation, indent=2)+"\n")
    event("development_complete", elapsed_seconds=validation["elapsed_seconds"])
    manifest.update(status="complete", completed_utc=now())
    manifest["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run_manifest.json"}
    (out/"run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
