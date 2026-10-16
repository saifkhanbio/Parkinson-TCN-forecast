"""Stage 1: forecast first, then score Saudi local historical development only."""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
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
import statsmodels

from gbd_park.local import forecast_setting, settings_grid
from gbd_park.scoring import score_forecasts, select_family, select_settings


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def check_lock():
    lock = json.loads((ROOT / "study_design/locked_v1/lock_manifest.json").read_text())
    for section in ["source_sha256", "design_sha256", "output_sha256"]:
        for name, expected in lock[section].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"Locked artifact changed: {name}")
    return lock


def fit_job(origin, sex, config):
    # Keep scoring truth out of the worker's fitting panel.
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.location_name.eq(config["primary_target"])
                      & panel.outcome.eq("prevalence") & panel.sex.eq(sex)].copy()
    forecasts, fits, arima = [], [], []
    started = time.perf_counter()
    for spec in settings_grid(config):
        rows, metadata, audit = forecast_setting(panel, config, origin, sex, spec)
        forecasts.extend(rows)
        fits.extend(metadata)
        arima.extend(audit)
    return {"origin": origin, "sex": sex, "forecasts": forecasts, "fits": fits, "arima": arima,
            "elapsed_seconds": time.perf_counter() - started}


def json_lines(path, records):
    with path.open("x") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/local_baselines_v1")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("Use one through four CPU workers")
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing run directory")
    lock = check_lock()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    code = sorted((ROOT / "src/gbd_park").glob("*.py")) + [Path(__file__).resolve()]
    code += [ROOT / "tests/test_local_baselines.py", ROOT / "study_design/local_baselines_implementation.md"]
    code_hashes = {str(p.relative_to(ROOT)): sha(p) for p in code}
    test_path = ROOT / "work/local-baselines-validation/tests.json"
    tests = json.loads(test_path.read_text())
    if not tests["passed"]:
        raise ValueError("Baseline validation suite has not passed")
    for path, expected in tests["tested_code_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Code changed after tests: {path}")
    out.mkdir(parents=True)
    events = []

    def event(name, **details):
        events.append({"time_utc": now(), "event": name, **details})
        (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")

    manifest = {"run_id": out.name, "protocol_version": config["version"], "created_utc": now(),
                "scope": "Saudi prevalence local baselines; historical development only",
                "model_origins": list(range(2003, 2014)), "maximum_verification_year": 2018,
                "final_period_scored": False, "source_lock_hash": sha(ROOT / "study_design/locked_v1/lock_manifest.json"),
                "config_sha256": sha(config_path), "code_sha256": code_hashes,
                "test_report_sha256": sha(test_path), "workers": args.workers,
                "versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
                "status": "running"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    event("fit_started")
    started = time.perf_counter()
    jobs = [(origin, sex) for origin in manifest["model_origins"] for sex in config["sexes"]]
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_job, origin, sex, config) for origin, sex in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f"Completed {len(completed)}/{len(jobs)}: {result['origin']} {result['sex']} ({result['elapsed_seconds']:.1f}s)", flush=True)
    completed.sort(key=lambda x: (x["origin"], x["sex"]))
    forecasts = pd.DataFrame([row for result in completed for row in result["forecasts"]])
    forecasts["run_id"] = out.name
    forecasts["protocol_version"] = config["version"]
    forecasts["config_sha256"] = manifest["config_sha256"]
    forecasts["source_lock_hash"] = manifest["source_lock_hash"]
    forecasts["donor_strategy"] = "target_only"
    forecasts["adaptation"] = "none"
    forecasts["seed_or_ensemble"] = "deterministic_statistical"
    expected = 11 * 2 * len(settings_grid(config)) * 11 * 5
    assert len(forecasts) == expected
    assert forecasts.forecast_year.max() == 2018
    forecasts = forecasts.sort_values(["origin", "sex", "grid_order", "age", "horizon"]).reset_index(drop=True)
    prediction_path = out / "candidate_predictions.csv"
    forecasts.to_csv(prediction_path, index=False)
    json_lines(out / "fitted_parameters.jsonl", [r for x in completed for r in x["fits"]])
    arima = pd.DataFrame([row for result in completed for row in result["arima"]])
    arima.to_csv(out / "arima_candidate_audit.csv", index=False)
    pd.DataFrame([{k: result[k] for k in ["origin", "sex", "elapsed_seconds"]} for result in completed]).to_csv(out / "fit_timings.csv", index=False)
    prediction_hash = sha(prediction_path)
    event("candidate_predictions_committed", sha256=prediction_hash, rows=len(forecasts))

    # Scoring reads back the saved predictions, then separately obtains verification values.
    event("development_scoring_started", maximum_verification_year=2018)
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = panel.loc[panel.location_name.eq(config["primary_target"]) & panel.outcome.eq("prevalence") & panel.year.le(2018)].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    saved = pd.read_csv(prediction_path)
    candidate_scores = score_forecasts(saved, truth, config["ages"], config["calendar"]["horizons"], 2018)
    candidate_scores.to_csv(out / "candidate_scores.csv", index=False)
    tuning, chosen = [], []
    for origin in config["calendar"]["selection_origins"]:
        for sex in config["sexes"]:
            for family in config["models"]["local_order"]:
                setting_id, loss, inner = select_settings(candidate_scores, config, origin, sex, family)
                tuning.append({"origin": origin, "sex": sex, "family": family, "setting_id": setting_id,
                               "inner_loss": loss, "inner_origins": "|".join(map(str, inner)),
                               "last_inner_label_year": max(inner) + 5})
                chosen.append(saved.loc[saved.origin.eq(origin) & saved.sex.eq(sex) & saved.setting_id.eq(setting_id)])
    development = pd.concat(chosen, ignore_index=True)
    development.to_csv(out / "development_predictions.csv", index=False)
    event("selected_development_predictions_committed", sha256=sha(out / "development_predictions.csv"), rows=len(development))
    development_scores = score_forecasts(pd.read_csv(out / "development_predictions.csv"), truth,
                                        config["ages"], config["calendar"]["horizons"], 2018)
    development_scores.to_csv(out / "development_scores.csv", index=False)
    pd.DataFrame(tuning).to_csv(out / "tuning_decisions.csv", index=False)
    scores_by_origin = development_scores.groupby(["origin", "sex", "family", "horizon"], as_index=False).agg(
        mean_age_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        fallback_cells=("status", lambda x: int(x.eq("fallback").sum())))
    scores_by_origin.to_csv(out / "scores_by_origin.csv", index=False)
    summary = scores_by_origin.groupby(["sex", "family", "horizon"], as_index=False).agg(
        mean_absolute_log_error=("mean_age_absolute_log_error", "mean"), mean_rate_mae=("rate_mae", "mean"),
        fallback_cells=("fallback_cells", "sum"))
    summary.to_csv(out / "development_summary.csv", index=False)
    champions = []
    for origin in config["calendar"]["reliability_origins"]:
        for sex in config["sexes"]:
            champion = select_family(development_scores, config, origin, sex)
            ident, loss, inner = select_settings(candidate_scores, config, origin, sex, champion["selected_family"])
            champion.update(setting_id=ident, inner_tuning_loss=loss, last_inner_label_year=max(inner) + 5)
            champions.append(champion)
    pd.DataFrame(champions).to_csv(out / "local_champions_for_later_evaluation.csv", index=False)
    assert len(development) == 2750
    assert all(t["last_inner_label_year"] <= t["origin"] for t in tuning)
    assert all(t["last_inner_label_year"] <= t["fit_origin"] and t["last_selection_target_year"] <= t["fit_origin"] for t in champions)
    assert sha(prediction_path) == prediction_hash
    assert code_hashes == {str(p.relative_to(ROOT)): sha(p) for p in code}
    check_lock()
    validation = {"passed": True, "candidate_prediction_rows": len(saved), "selected_development_rows": len(development),
                  "development_origins": config["calendar"]["selection_origins"], "all_forecast_cells_present": True,
                  "training_and_tuning_time_cutoffs_passed": True, "predictions_saved_before_scoring": True,
                  "prediction_ledger_unchanged_after_scoring": True, "locked_inputs_unchanged": True,
                  "maximum_scored_year": 2018, "final_2019_2023_scores_produced": False,
                  "selected_forecast_fallback_cells": int(development.status.eq("fallback").sum()),
                  "arima_candidate_status_counts": arima.status.value_counts().to_dict(),
                  "unit_tests": tests["tests_run"], "elapsed_seconds": time.perf_counter() - started}
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    event("development_complete", elapsed_seconds=validation["elapsed_seconds"])
    manifest["status"] = "complete"
    manifest["completed_utc"] = now()
    manifest["output_sha256"] = {p.name: sha(p) for p in sorted(out.glob("*")) if p.name != "run_manifest.json"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
