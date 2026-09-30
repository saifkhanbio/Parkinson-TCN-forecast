"""Bounded post-result exploratory correction and interval-bias evaluation."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
import scipy

from gbd_park.improvements import candidate_forecasts, select_age, select_retention, apply_retention, retention_family
from gbd_park.intervals import build_residual_bank, score_intervals
from gbd_park.scoring import score_forecasts
from run_local_baselines import check_lock, sha, now, json_lines


def verified_prior(amendment):
    hashes = {}
    for run in amendment["original_runs"]:
        directory = ROOT / "results" / run
        record = json.loads((directory / "run_manifest.json").read_text())
        assert record["status"] == "complete"
        for name, expected in record["output_sha256"].items():
            assert sha(directory / name) == expected, name
        for name, expected in record["code_sha256"].items():
            assert sha(ROOT / name) == expected, name
        hashes[run] = sha(directory / "run_manifest.json")
    return hashes


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def original_origin(config, origin):
    if origin <= 2008:
        directory = ROOT / "results/intervals_v1"
        references = [r for r in read_lines(directory / "early_seed_references.jsonl") if r["origin"] == origin]
        paths = [ROOT / r["payload_path"] for r in references]
        calibration_path = directory / "early_tcn_adaptation_audit.jsonl"
    else:
        directory = ROOT / "results" / ("tcn_v1" if origin <= 2013 else "primary_v1")
        choices = json.loads((directory / ("tuning_decisions.json" if origin <= 2013 else "tcn_choices.json")).read_text())
        choice = next(row for row in choices if row["fit_origin"] == origin)
        base = choice["base"]
        ident = "__".join(f"{name}={base[name]}" for name in ["channels", "weight_decay", "epochs"])
        paths = [directory / "payloads" / f"origin{origin}__{ident}__seed={seed}.joblib"
                 for seed in config["models"]["tcn"]["ensemble_seeds"]]
        calibration_path = directory / ("development_adaptation_audit.jsonl" if origin <= 2013 else "tcn_adaptation_audit.jsonl")
    payloads = [joblib.load(path) for path in paths]
    corrections = [r for r in read_lines(calibration_path) if r["origin"] == origin]
    return payloads, corrections, paths, calibration_path


def correction_job(origin, baseline, config, amendment):
    payloads, corrections, paths, calibration_path = original_origin(config, origin)
    before = [joblib.hash(payload) for payload in payloads]
    rows, audit = candidate_forecasts(payloads, corrections, baseline, config, amendment)
    assert before == [joblib.hash(payload) for payload in payloads]
    sources = [{"origin": origin, "path": str(path.relative_to(ROOT)), "sha256": sha(path)}
               for path in paths + [calibration_path]]
    return rows, audit, sources


def truth_frame(panel, maximum_year):
    return panel.loc[panel.location_name.eq("Saudi Arabia") & panel.outcome.eq("prevalence")
                     & panel.year.le(maximum_year)].rename(
                         columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})


def baseline_bank(history, config, origin, family):
    if family == "tcn_v1_unchanged" and origin >= 2014:
        bank = joblib.load(ROOT / "results/intervals_v1/banks" / f"origin{origin}__tcn_adapted.joblib")
        bank["family"] = family
        return bank
    return build_residual_bank(history, config, origin, family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/improvements_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reviewed", action="store_true", help="Coordinator completed the pre-run review")
    args = parser.parse_args()
    if not args.reviewed:
        raise ValueError("Coordinator review is required before exploratory execution")
    if not 1 <= args.workers <= 8:
        raise ValueError("Use one through eight CPU correction workers")
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Existing exploratory results cannot be overwritten")
    config_path = ROOT / "study_design/locked_v1/design.json"
    amendment_path = ROOT / "study_design/exploratory_v1_1.json"
    config, amendment = json.loads(config_path.read_text()), json.loads(amendment_path.read_text())
    lock_path = ROOT / "study_design/exploratory_v1_1.lock.json"
    amendment_lock = json.loads(lock_path.read_text())
    for name, expected in amendment_lock["document_sha256"].items():
        assert sha(ROOT / name) == expected, "Exploratory amendment changed after registration"
    check_lock()
    prior_hashes = verified_prior(amendment)
    assert prior_hashes == amendment_lock["prior_manifest_sha256"]
    tests_path = ROOT / "work/improvements-validation/tests.json"
    tests = json.loads(tests_path.read_text())
    assert tests["passed"]
    for name, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT / name) == expected, f"Changed since tests: {name}"
    required = ["src/gbd_park/improvements.py", "scripts/run_improvements.py", "tests/test_improvements.py",
                "study_design/exploratory_v1_1.md", "study_design/exploratory_v1_1.json"]
    assert set(required).issubset(tests["tested_code_sha256"])
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
    previous = json.loads((ROOT / "results/primary_v1/run_manifest.json").read_text())
    assert all(previous["versions"][name] == value for name, value in versions.items())
    out.mkdir(parents=True)
    (out / "tests_at_run.json").write_bytes(tests_path.read_bytes())
    manifest = {"run_id": out.name, "status": "running", "created_utc": now(),
                "role": "exploratory_after_primary_results_viewed", "primary_verdict_replaced": False,
                "amendment_sha256": sha(amendment_path), "amendment_lock_sha256": sha(lock_path),
                "prior_manifest_sha256": prior_hashes, "code_sha256": tests["tested_code_sha256"],
                "source_models_retrained": False, "original_correction_refitted": False,
                "workers": args.workers, "versions": versions, "final_period_scored": False}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started, events = time.perf_counter(), []

    def event(name, **details):
        events.append({"event": name, "time_utc": now(), **copy.deepcopy(details)})
        (out / "events.json").write_text(json.dumps(events, indent=2) + "\n")

    historical_origins, evaluation_origins = amendment["historical_origins"], amendment["evaluation_origins"]
    origins = historical_origins + evaluation_origins
    old_history = pd.read_csv(ROOT / "results/intervals_v1/prequential_predictions.csv", float_precision="round_trip")
    old_evaluation = pd.read_csv(ROOT / "results/primary_v1/predictions.csv", float_precision="round_trip")
    original = pd.concat([old_history[old_history.family.eq("tcn_adapted")],
                          old_evaluation[old_evaluation.family.eq("tcn_adapted")]], ignore_index=True)
    assert len(original) == 1760 and set(original.origin) == set(origins)
    event("new_correction_fitting_started", original_source_retraining=False)
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(correction_job, origin, original[original.origin.eq(origin)].copy(), config, amendment)
                   for origin in origins]
        for index, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            print(f"Frozen-source correction batches: {index}/{len(origins)}", flush=True)
    candidates = pd.DataFrame([row for result in completed for row in result[0]]).sort_values(
        ["origin", "sex", "grid_order", "age", "horizon"]).reset_index(drop=True)
    audits = [row for result in completed for row in result[1]]
    json_lines(out / "age_correction_audit.jsonl", audits)
    json_lines(out / "frozen_source_references.jsonl", [row for result in completed for row in result[2]])
    assert len(candidates) == 7040
    candidates.to_csv(out / "age_candidate_predictions.csv", index=False)
    event("age_candidates_committed", rows=len(candidates), sha256=sha(out / "age_candidate_predictions.csv"))
    # This first scoring stage is restricted to the historical development years.
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    historical_truth = truth_frame(panel, 2018).copy()
    del panel
    event("historical_candidate_scoring_started", maximum_verification_year=2018)
    candidate_scores = score_forecasts(candidates[candidates.origin.isin(historical_origins)], historical_truth,
                                      config["ages"], config["calendar"]["horizons"], 2018)
    candidate_scores.to_csv(out / "historical_age_candidate_scores.csv", index=False)
    age_choices = [select_age(candidate_scores, config, amendment, origin, sex) for origin in origins for sex in config["sexes"]]
    json_lines(out / "age_setting_choices.jsonl", age_choices)
    chosen = []
    for choice in age_choices:
        part = candidates[candidates.origin.eq(choice["origin"]) & candidates.sex.eq(choice["sex"])
                          & candidates.setting_id.eq(choice["setting_id"])].copy()
        part["last_inner_label_year"] = choice["last_label_year"]
        part["family"] = "tcn_age_group_correction"
        chosen.append(part)
    baseline = original.copy()
    baseline["family"] = "tcn_v1_unchanged"
    points = pd.concat([baseline] + chosen, ignore_index=True).sort_values(
        ["origin", "family", "sex", "age", "horizon"]).reset_index(drop=True)
    assert len(points) == 3520
    points.to_csv(out / "point_predictions.csv", index=False)
    event("point_predictions_committed", sha256=sha(out / "point_predictions.csv"))
    historical_scores = score_forecasts(points[points.origin.isin(historical_origins)], historical_truth,
                                       config["ages"], config["calendar"]["horizons"], 2018)
    # Retain exact source residual inputs for unchanged-v1 interval reconstruction.
    old_scores = pd.read_csv(ROOT / "results/intervals_v1/prequential_scores.csv", float_precision="round_trip")
    old_scores = old_scores[old_scores.family.eq("tcn_adapted")].copy()
    old_scores["family"] = "tcn_v1_unchanged"
    historical_scores = pd.concat([old_scores, historical_scores[historical_scores.family.eq("tcn_age_group_correction")]], ignore_index=True)
    historical_scores.to_csv(out / "historical_selected_scores.csv", index=False)
    development = []
    for origin in amendment["interval_development_origins"]:
        for family in amendment["point_families"]:
            bank = baseline_bank(historical_scores, config, origin, family)
            current = points[points.origin.eq(origin) & points.family.eq(family)]
            for retention in amendment["bias_retention_candidates"]:
                frame, _ = apply_retention(current, bank, config, dict.fromkeys(config["sexes"], retention),
                                           retention_family(family, retention))
                development.append(frame)
    development = pd.concat(development, ignore_index=True)
    assert len(development) == 7920
    development.to_csv(out / "development_interval_predictions.csv", index=False)
    event("development_intervals_committed", sha256=sha(out / "development_interval_predictions.csv"))
    dev_cells, dev_wis = score_intervals(development, historical_truth, config, 2018)
    dev_cells.to_csv(out / "development_interval_scores.csv", index=False)
    dev_wis.to_csv(out / "development_wis_scores.csv", index=False)
    retention_choices = [select_retention(dev_wis, config, amendment, origin, sex, family)
                         for origin in evaluation_origins for sex in config["sexes"] for family in amendment["point_families"]]
    json_lines(out / "retention_choices.jsonl", retention_choices)
    intervals, draws, bank_audit = [], [], []
    for origin in evaluation_origins:
        for family in amendment["point_families"]:
            bank = baseline_bank(historical_scores, config, origin, family)
            current = points[points.origin.eq(origin) & points.family.eq(family)]
            for index, residual_origin in enumerate(bank["origins"]):
                frame = bank["coords"].copy()
                frame["origin"], frame["family"], frame["residual_origin"] = origin, family, residual_origin
                frame["raw_log_residual"], frame["historical_mean"] = bank["raw_residuals"][index], bank["center"]
                bank_audit.append(frame)
            variants = [(retention_family(family, r), dict.fromkeys(config["sexes"], r)) for r in amendment["bias_retention_candidates"]]
            selected = {row["sex"]: row["retention"] for row in retention_choices if row["origin"] == origin and row["point_family"] == family}
            variants.append((family + "__retention=selected", selected))
            for label, mapping in variants:
                frame, joint = apply_retention(current, bank, config, mapping, label)
                intervals.append(frame)
                draws.append(joint)
    intervals, draws = pd.concat(intervals, ignore_index=True), pd.concat(draws, ignore_index=True)
    assert len(intervals) == 26400 and len(draws) == 39600
    intervals.to_csv(out / "interval_predictions.csv", index=False)
    draws.to_csv(out / "joint_draws.csv", index=False)
    pd.concat(bank_audit, ignore_index=True).to_csv(out / "residual_bank_audit.csv", index=False)
    original_intervals = pd.read_csv(ROOT / "results/primary_v1/intervals.csv", float_precision="round_trip")
    old = original_intervals[original_intervals.family.eq("tcn_adapted")]
    unchanged = intervals[intervals.family.eq(retention_family("tcn_v1_unchanged", 0))]
    keys = ["origin", "sex", "age", "horizon", "scale", "level"]
    old_values = old.set_index(keys).sort_index()[["lower", "median", "upper", "point_prediction"]]
    new_values = unchanged.set_index(keys).sort_index()[["lower", "median", "upper", "point_prediction"]]
    assert old_values.index.equals(new_values.index)
    np.testing.assert_allclose(new_values, old_values, rtol=1e-12, atol=1e-12)
    zero_candidates = candidates[candidates.setting_id.eq("age_penalty=none")].set_index(["origin", "sex", "age", "horizon"]).sort_index()
    original_indexed = original.set_index(["origin", "sex", "age", "horizon"]).sort_index()
    np.testing.assert_array_equal(zero_candidates.log_prediction, original_indexed.log_prediction)
    commits = {name: sha(out / name) for name in ["age_setting_choices.jsonl", "retention_choices.jsonl", "point_predictions.csv",
                                                "interval_predictions.csv", "joint_draws.csv", "residual_bank_audit.csv"]}
    (out / "pre_score_commit.json").write_text(json.dumps({"committed_utc": now(), "sha256": commits}, indent=2) + "\n")
    event("evaluation_predictions_and_choices_committed", hashes=commits)
    event("exploratory_evaluation_scoring_started", maximum_verification_year=2023)
    # Post-2018 outcomes cannot change correction or interval-retention choices.
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = truth_frame(panel, 2023)
    scored = score_forecasts(points[points.origin.isin(evaluation_origins)], truth, config["ages"], config["calendar"]["horizons"], 2023)
    scored["signed_log_error"] = scored.log_prediction - np.log(scored.observed_rate)
    scored["signed_rate_error"] = scored.prediction - scored.observed_rate
    scored.to_csv(out / "evaluation_point_scores.csv", index=False)
    cells, wis = score_intervals(intervals, truth, config, 2023)
    cells.to_csv(out / "evaluation_interval_scores.csv", index=False)
    wis.to_csv(out / "evaluation_wis_scores.csv", index=False)
    scored.groupby(["origin", "family", "sex", "horizon"], as_index=False).agg(
        mean_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        mean_signed_log_error=("signed_log_error", "mean"), mean_signed_rate_error=("signed_rate_error", "mean")).to_csv(out / "point_summary.csv", index=False)
    cells.groupby(["origin", "family", "sex", "horizon", "scale", "level"], as_index=False).agg(
        coverage=("covered", "mean"), width=("width", "mean"), interval_score=("interval_score", "mean"), n_blocks=("n_blocks", "first")).to_csv(out / "interval_summary.csv", index=False)
    wis.groupby(["origin", "family", "sex", "horizon", "scale"], as_index=False).agg(
        wis_50_80=("wis_50_80", "mean"), wis_50_80_95=("wis_50_80_95", "mean"),
        median_absolute_error=("median_absolute_error", "mean")).to_csv(out / "wis_summary.csv", index=False)
    scored.groupby(["family", "sex", "age", "horizon"], as_index=False).agg(
        mean_signed_log_error=("signed_log_error", "mean"), mean_absolute_log_error=("absolute_log_error", "mean"),
        origins=("origin", "nunique")).to_csv(out / "signed_error_by_age.csv", index=False)
    bounds = truth[truth.forecast_year.between(2019, 2023)][["target", "sex", "age", "forecast_year", "observed_rate", "rate_lower", "rate_upper"]].copy()
    bounds["role"] = "source_estimate_bounds_not_forecast_intervals_or_calibration_targets"
    bounds.to_csv(out / "source_bounds_separate.csv", index=False)
    assert all(sha(out / name) == digest for name, digest in commits.items())
    assert verified_prior(amendment) == prior_hashes
    assert all(sha(ROOT / name) == digest for name, digest in tests["tested_code_sha256"].items())
    validation = {"passed": True, "tests": tests["tests_run"], "age_candidate_cells": len(candidates),
                  "selected_point_cells": len(points), "evaluation_point_cells": len(scored),
                  "evaluation_interval_cells": len(cells), "joint_draw_cells": len(draws),
                  "new_source_fits": 0, "original_correction_refits": 0,
                  "correction_fallbacks": sum(row["status"] != "ok" for row in audits),
                  "zero_age_offset_reproduces_original_log_points": True,
                  "zero_bias_retention_reproduces_original_intervals": True,
                  "maximum_original_interval_difference": float(np.max(np.abs(new_values.to_numpy() - old_values.to_numpy()))),
                  "selection_uses_completed_history_only": True, "original_primary_verdict_preserved": True,
                  "elapsed_seconds": time.perf_counter() - started}
    (out / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
    event("exploratory_evaluation_complete", elapsed_seconds=validation["elapsed_seconds"])
    manifest.update(status="complete", final_period_scored=True, completed_utc=now())
    manifest["output_sha256"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "run_manifest.json"}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
