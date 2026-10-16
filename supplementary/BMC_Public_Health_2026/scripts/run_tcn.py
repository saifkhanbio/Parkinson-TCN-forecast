"""Stage 3: fixed-grid TCN development with a five-seed reported ensemble."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from gbd_park.pooled import country_pool
from gbd_park.scoring import score_forecasts
from gbd_park.tcn import base_grid
from gbd_park.tcn_forecasting import base_id, fit_seed, make_forecasts, select_tcn_settings
from run_local_baselines import check_lock, sha, now, json_lines


def previous_manifests():
    result = {}
    for name in ["local_baselines_v1", "nonneural_v1"]:
        directory = ROOT / "results" / name
        record = json.loads((directory / "run_manifest.json").read_text())
        assert record["status"] == "complete"
        for filename, expected in record["output_sha256"].items():
            assert sha(directory / filename) == expected, filename
        for filename, expected in record["code_sha256"].items():
            assert sha(ROOT / filename) == expected, filename
        result[name] = sha(directory / "run_manifest.json")
    return result


def fit_job(config, origin, base, seed, device, out):
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel[panel.year.le(origin) & panel.outcome.eq("prevalence")].copy()
    ident = f"origin{origin}__{base_id(base)}__seed={seed}"
    payload = fit_seed(panel, config, origin, base, seed, device, out / "checkpoints" / f"{ident}.joblib")
    joblib.dump(payload, out / "payloads" / f"{ident}.joblib", compress=3)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/tcn_v1")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda:0"], default="auto")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing run")
    check_lock()
    previous = previous_manifests()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    test_path = ROOT / "work/tcn-validation/tests.json"
    tests = json.loads(test_path.read_text())
    assert tests["passed"], "Current passing neural tests are required"
    for filename, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT / filename) == expected, f"Changed since tests: {filename}"
    code_files = ["src/gbd_park/tcn.py", "src/gbd_park/tcn_forecasting.py", "src/gbd_park/pooled.py",
                  "src/gbd_park/adaptation.py", "src/gbd_park/scoring.py", "scripts/run_tcn.py",
                  "scripts/run_local_baselines.py", "tests/test_tcn.py", "study_design/tcn_implementation.md"]
    for required in ["src/gbd_park/tcn.py", "src/gbd_park/tcn_forecasting.py", "scripts/run_tcn.py",
                     "study_design/tcn_implementation.md"]:
        assert required in tests["tested_code_sha256"], f"Tests must cover current file: {required}"
    benchmark_path = ROOT / "work/tcn-benchmark/report.json"
    benchmark = json.loads(benchmark_path.read_text())
    assert benchmark["status"] == "complete" and benchmark["cuda_reproducibility"]["status"] == "ok"
    for filename, expected in benchmark["input_sha256"].items():
        assert sha(ROOT / filename) == expected, f"Changed since benchmark: {filename}"
    device = benchmark["recommendation"]["device"] if args.device == "auto" else args.device
    workers = args.workers if args.workers is not None else benchmark["recommendation"]["workers"]
    if not 1 <= workers <= (2 if device.startswith("cuda") else 8):
        raise ValueError("Worker count exceeds the stage-3 resource limit")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; do not silently switch devices")
    out.mkdir(parents=True)
    (out / "checkpoints").mkdir()
    (out / "payloads").mkdir()
    (out / "tests_at_run.json").write_bytes(test_path.read_bytes())
    (out / "benchmark_at_run.json").write_bytes(benchmark_path.read_bytes())
    code = {name: sha(ROOT / name) for name in code_files}
    manifest = {"run_id": out.name, "protocol_version": config["version"], "status": "running",
                "created_utc": now(), "config_sha256": sha(config_path), "code_sha256": code,
                "source_lock_hash": sha(ROOT / "study_design/locked_v1/lock_manifest.json"),
                "prior_manifests_sha256": previous, "test_report_sha256": sha(test_path),
                "benchmark_sha256": sha(benchmark_path), "device": device, "workers": workers,
                "candidate_origins": list(range(2003, 2014)), "development_origins": config["calendar"]["selection_origins"],
                "ensemble_seeds": config["models"]["tcn"]["ensemble_seeds"],
                "maximum_scored_year": 2018, "final_period_scored": False,
                "donor_countries": country_pool(config, config["primary_target"], "donor"),
                "versions": {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
                             "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "scipy": scipy.__version__, "joblib": joblib.__version__}}
    manifest["donor_list_sha256"] = __import__("hashlib").sha256(json.dumps(manifest["donor_countries"]).encode()).hexdigest()
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started, events = time.perf_counter(), []

    def event(name, **details):
        events.append({"time_utc": now(), "event": name, **details})
        (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")

    def execute(jobs, label):
        completed = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(fit_job, config, origin, base, seed, device, out) for origin, base, seed in jobs]
            for future in as_completed(futures):
                payload = future.result()
                completed.append(payload)
                if len(completed) % 4 == 0 or len(completed) == len(jobs):
                    print(f"{label}: {len(completed)}/{len(jobs)} fits; {time.perf_counter()-started:.1f}s elapsed", flush=True)
        return sorted(completed, key=lambda p: (p["origin"], base_id(p["base"]), p["seed"]))

    def annotated(records):
        frame = pd.DataFrame(records)
        frame["run_id"] = out.name
        frame["protocol_version"] = config["version"]
        frame["config_sha256"] = manifest["config_sha256"]
        frame["source_lock_hash"] = manifest["source_lock_hash"]
        frame["donor_list_sha256"] = manifest["donor_list_sha256"]
        return frame.sort_values(["origin", "sex", "family", "setting_id", "age", "horizon"]).reset_index(drop=True)

    event("candidate_fitting_started", device=device, workers=workers)
    seed = config["models"]["tcn"]["tuning_seed"]
    jobs = [(origin, base, seed) for origin in manifest["candidate_origins"] for base in base_grid(config)]
    candidates = execute(jobs, "Seed-11 grid")
    candidate_records, candidate_calibrations = [], []
    for payload in candidates:
        rows, corrections = make_forecasts([payload], config, config["adaptation"]["penalties"])
        candidate_records.extend(rows)
        candidate_calibrations.extend(corrections)
    candidate_predictions = annotated(candidate_records)
    assert len(candidate_predictions) == 48400
    assert candidate_predictions.seed_or_ensemble.eq("11").all()
    assert candidate_predictions.forecast_year.max() == 2018
    candidate_predictions.to_csv(out / "candidate_predictions.csv", index=False)
    prediction_hash = sha(out / "candidate_predictions.csv")
    json_lines(out / "candidate_adaptation_audit.jsonl", candidate_calibrations)
    json_lines(out / "candidate_fit_audit.jsonl", [p["audit"] for p in candidates])
    training = pd.concat([next(p["training_meta"] for p in candidates if p["origin"] == o)
                          for o in manifest["candidate_origins"]], ignore_index=True)
    training.to_csv(out / "training_window_audit.csv", index=False)
    assert training.label_end.le(training.fit_origin).all()
    assert not training.country.eq(config["primary_target"]).any()
    for _, group in training.groupby("fit_origin"):
        np.testing.assert_allclose(group.groupby("country").sample_weight.sum(), len(group)/6, rtol=1e-12)
    event("candidate_predictions_committed", rows=len(candidate_predictions), sha256=prediction_hash)
    event("candidate_scoring_started", maximum_verification_year=2018)
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = panel[panel.location_name.eq(config["primary_target"]) & panel.outcome.eq("prevalence") & panel.year.le(2018)].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    candidate_scores = score_forecasts(pd.read_csv(out / "candidate_predictions.csv"), truth,
                                      config["ages"], config["calendar"]["horizons"], 2018)
    assert candidate_scores.seed_or_ensemble.astype(str).eq("11").all()
    candidate_scores.to_csv(out / "candidate_scores.csv", index=False)
    choices = [select_tcn_settings(candidate_scores, config, origin) for origin in manifest["development_origins"]]
    later = [select_tcn_settings(candidate_scores, config, origin) for origin in config["calendar"]["reliability_origins"]]
    (out / "tuning_decisions.json").write_text(json.dumps(choices, indent=2) + "\n")
    (out / "choices_for_later_evaluation.json").write_text(json.dumps(later, indent=2) + "\n")
    assert all(choice["last_inner_label_year"] <= choice["fit_origin"] for choice in choices + later)
    event("outer_settings_selected", origins=manifest["development_origins"])
    jobs = [(choice["fit_origin"], choice["base"], s) for choice in choices
            for s in manifest["ensemble_seeds"] if s != seed]
    extras = execute(jobs, "Additional ensemble seeds")
    development_records, development_calibrations, seed_records = [], [], []
    for choice in choices:
        pool = [p for p in candidates + extras if p["origin"] == choice["fit_origin"] and p["base"] == choice["base"]]
        assert sorted(p["seed"] for p in pool) == sorted(manifest["ensemble_seeds"])
        records, corrections = make_forecasts(pool, config, choice["penalties"], include_intercept=True)
        development_records.extend(records)
        development_calibrations.extend(corrections)
        for payload in pool:
            records, _ = make_forecasts([payload], config, [])
            seed_records.extend(records)
    development = annotated(development_records)
    assert len(development) == 1650
    development.to_csv(out / "development_predictions.csv", index=False)
    annotated(seed_records).to_csv(out / "seed_predictions.csv", index=False)
    json_lines(out / "development_adaptation_audit.jsonl", development_calibrations)
    all_fits = [p["audit"] for p in candidates + extras]
    json_lines(out / "base_fit_audit.jsonl", all_fits)
    event("development_predictions_committed", rows=len(development), sha256=sha(out / "development_predictions.csv"))
    event("development_scoring_started", maximum_verification_year=2018)
    scores = score_forecasts(pd.read_csv(out / "development_predictions.csv"), truth,
                            config["ages"], config["calendar"]["horizons"], 2018)
    scores.to_csv(out / "development_scores.csv", index=False)
    by_origin = scores.groupby(["origin", "sex", "family", "horizon"], as_index=False).agg(
        mean_age_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        fallback_cells=("status", lambda x: int(x.eq("fallback").sum())))
    by_origin.to_csv(out / "scores_by_origin.csv", index=False)
    by_origin.groupby(["sex", "family", "horizon"], as_index=False).agg(
        mean_absolute_log_error=("mean_age_absolute_log_error", "mean"), mean_rate_mae=("rate_mae", "mean"),
        fallback_cells=("fallback_cells", "sum")).to_csv(out / "development_summary.csv", index=False)
    keys = ["origin", "sex", "age", "horizon", "ensemble_fingerprint"]
    reference = scores[scores.family.eq("tcn_unadapted")][keys + ["absolute_log_error"]].rename(
        columns={"absolute_log_error": "matched_unadapted_error"})
    pairs = scores[~scores.family.eq("tcn_unadapted")].merge(reference, on=keys, validate="many_to_one")
    assert len(pairs) == 1100 and pairs.matched_unadapted_error.notna().all()
    pairs["adaptation_loss_change"] = pairs.absolute_log_error - pairs.matched_unadapted_error
    pairs.rename(columns={"absolute_log_error": "adapted_error"})[
        keys + ["family", "adapted_error", "matched_unadapted_error", "adaptation_loss_change"]].to_csv(
            out / "matched_adaptation_comparisons.csv", index=False)
    assert len(all_fits) == 108
    assert all(row["fingerprint_before"] == row["fingerprint_after"] for row in all_fits)
    assert all(row["maximum_label_year"] <= row["origin"] for row in all_fits)
    assert all(row["last_target_label_year"] <= row["origin"] for row in candidate_calibrations + development_calibrations)
    assert sha(out / "candidate_predictions.csv") == prediction_hash
    check_lock()
    assert previous_manifests() == previous
    assert all(sha(ROOT / filename) == digest for filename, digest in code.items())
    validation = {"passed": True, "unit_tests": tests["tests_run"], "candidate_predictions": len(candidate_predictions),
                  "development_predictions": len(development), "seed_predictions": len(seed_records),
                  "base_fits": len(all_fits), "failures": sum(f["status"] != "ok" for f in all_fits),
                  "candidate_adaptations": len(candidate_calibrations), "development_adaptations": len(development_calibrations),
                  "adaptation_failures": sum(c["status"] != "ok" for c in candidate_calibrations + development_calibrations),
                  "selected_fallback_cells": int(development.status.eq("fallback").sum()),
                  "parameter_counts": sorted(set(f["parameter_count"] for f in all_fits)),
                  "device": device, "workers": workers, "completed_labels_only": True,
                  "donor_fit_excludes_saudi_both_sexes": True, "frozen_base_states": True,
                  "mean_log_ensemble_before_single_sex_correction": True, "all_five_seeds_retained": True,
                  "prediction_before_scoring": True, "maximum_scored_year": 2018,
                  "final_period_scored": False, "interval_banks_complete": False,
                  "prior_stages_and_lock_unchanged": True, "elapsed_seconds": time.perf_counter()-started}
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    event("development_complete", elapsed_seconds=validation["elapsed_seconds"])
    manifest.update(status="complete", completed_utc=now())
    manifest["output_sha256"] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                 if p.is_file() and p.name != "run_manifest.json"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
