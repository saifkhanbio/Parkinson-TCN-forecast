"""Fixed-setting mortality history sensitivity, with matched numerical devices.

The 1980 and 1990 starts change both donor training and target adaptation
history. This is not the target-only learning curve or a replacement for the
common-period supporting comparison. No interval procedure is tuned here.
"""

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
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
import statsmodels
import torch
from gbd_park.local import forecast_setting, settings_grid as local_grid
from gbd_park.secondary import context_config, restore_outcome, score_actual, job_fingerprint
from gbd_park.tcn_forecasting import fit_seed, make_forecasts
from run_local_baselines import check_lock, sha, now
from run_secondary import (source_panel, write_json, hash_files, verify_hashes,
                           phase_complete, commit_phase)
from run_supporting import verify_prior

ROLE = "prespecified_supporting_mortality_available_history_sensitivity"
OUTCOMES = ("deaths", "ylls")
START_YEARS = (1980, 1990)
REQUIRED = ["scripts/run_mortality_history_sensitivity.py", "tests/test_mortality_history.py",
            "scripts/run_supporting.py", "src/gbd_park/supporting.py",
            "study_design/supporting_outcomes_implementation.md"]


def history_tasks(config):
    return [{"id": f"{country['iso3']}_{outcome}_history{start}",
             "target": country["name"], "outcome": outcome, "history_start": start}
            for country in config["countries"] if country["gcc"]
            for outcome in OUTCOMES for start in START_YEARS]


def history_context(panel, config, target, outcome, history_start, origin=2018):
    """Select actual mortality outcome and complete arm history before aliasing."""
    if outcome not in OUTCOMES or history_start not in START_YEARS:
        raise ValueError("Mortality history requires deaths/ylls and a 1980/1990 start")
    if not history_start <= origin <= config["calendar"]["history_end"]:
        raise ValueError("Invalid mortality history origin")
    copied = context_config(config, target)
    copied["calendar"]["history_start"] = int(history_start)
    countries = [country["name"] for country in copied["countries"]]
    work = panel.loc[panel.outcome.eq(outcome) & panel.location_name.isin(countries)
                     & panel.sex.isin(copied["sexes"]) & panel.age.isin(copied["ages"])
                     & panel.year.between(history_start, origin)].copy()
    keys = ["location_name", "sex", "age", "year"]
    expected = pd.MultiIndex.from_product([countries, copied["sexes"], copied["ages"],
                                          range(history_start, origin+1)], names=keys)
    actual = pd.MultiIndex.from_frame(work[keys])
    if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
        raise ValueError("Mortality history is not a complete country/sex/age/year grid")
    if not np.isfinite(work.rate).all() or work.rate.le(0).any():
        raise ValueError("Mortality history rates must be finite and positive")
    work["source_outcome"] = outcome
    work["outcome"] = "prevalence"
    return work, copied


def fixed_base(config):
    defaults = config["cold_start_defaults"]
    return {"channels": defaults["tcn_channels"],
            "weight_decay": defaults["tcn_weight_decay"], "epochs": defaults["tcn_epochs"]}


def job_specs(config, device):
    jobs = []
    for task in history_tasks(config):
        common = {"trial_id": task["id"], "target": task["target"], "outcome": task["outcome"],
                  "history_start": task["history_start"], "origin": 2018}
        jobs.extend({**common, "kind": "tcn", "base": fixed_base(config), "seed": seed, "device": device}
                    for seed in config["models"]["tcn"]["ensemble_seeds"])
        jobs.extend({**common, "kind": "local", "sex": sex, "device": "cpu"}
                    for sex in config["sexes"])
    return jobs


def result_path(out, job):
    return Path(out) / "jobs" / job["trial_id"] / job_fingerprint(job) / "result.joblib"


def load_history_result(out, job):
    path = result_path(out, job)
    record = json.loads((path.parent / "complete.json").read_text())
    if record["job"] != job or record["job_sha256"] != job_fingerprint(job):
        raise ValueError("Mortality history cache identity mismatch")
    verify_hashes(path.parent, record["artifact_sha256"])
    return joblib.load(path)


def fit_history_job(config, job, out):
    path = result_path(out, job)
    directory, marker = path.parent, path.parent / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["config_sha256"] != job_fingerprint(config):
            raise ValueError("Mortality history configuration mismatch")
        load_history_result(out, job)
        return str(path)
    directory.mkdir(parents=True, exist_ok=True)
    panel, cfg = history_context(source_panel(), config, job["target"], job["outcome"],
                                 job["history_start"], job["origin"])
    started = time.perf_counter()
    if job["kind"] == "tcn":
        result = fit_seed(panel, cfg, job["origin"], job["base"], job["seed"],
                          job["device"], directory / "checkpoint.joblib")
        result["audit"].update(actual_target=job["target"], actual_outcome=job["outcome"],
                                history_start=job["history_start"],
                                target_and_donor_history_years=job["origin"]-job["history_start"]+1)
    elif job["kind"] == "local":
        spec = next(item for item in local_grid(cfg) if item["family"] == "damped_ets")
        rows, fits, arima = forecast_setting(panel, cfg, job["origin"], job["sex"], spec,
                                            target=job["target"])
        result = {"forecasts": restore_outcome(pd.DataFrame(rows), job["outcome"]).to_dict("records"),
                  "fits": fits, "arima": arima}
    else:
        raise ValueError("Unknown mortality history job kind")
    result.update(job=job, actual_target=job["target"], actual_outcome=job["outcome"],
                  history_start=job["history_start"], elapsed_seconds=time.perf_counter()-started)
    joblib.dump(result, path, compress=3)
    write_json(marker, {"job": job, "job_sha256": job_fingerprint(job),
                        "config_sha256": job_fingerprint(config), "completed_utc": now(),
                        "artifact_sha256": hash_files(directory, [p for p in directory.rglob("*")
                                                                   if p.is_file() and p != marker])})
    return str(path)


def run_jobs(jobs, config, out, cpu_workers, gpu_workers, device):
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cpu_workers, mp_context=context) as cpu:
        gpu = ProcessPoolExecutor(max_workers=gpu_workers, mp_context=context) if device.startswith("cuda") else None
        try:
            futures = {(gpu if gpu is not None and job["kind"] == "tcn" else cpu).submit(
                fit_history_job, config, job, out): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 8 == 0 or index == len(futures):
                    print(f"Mortality history fits: {index}/{len(futures)}; "
                          f"{time.perf_counter()-started:.1f}s", flush=True)
        finally:
            if gpu is not None:
                gpu.shutdown(wait=True, cancel_futures=True)


def assemble_issued(config, out, device):
    if phase_complete(out, "issued_commit"):
        return
    rows, adaptation, seeds, fits = [], [], [], []
    jobs = job_specs(config, device)
    for task in history_tasks(config):
        _, cfg = history_context(source_panel(), config, task["target"], task["outcome"], task["history_start"])
        task_jobs = [job for job in jobs if job["trial_id"] == task["id"]]
        payloads = [load_history_result(out, job) for job in task_jobs if job["kind"] == "tcn"]
        if [p["seed"] for p in payloads] != config["models"]["tcn"]["ensemble_seeds"]:
            raise ValueError("Mortality history must retain all five fixed ensemble seeds")
        neural, corrections = make_forecasts(payloads, cfg, [config["cold_start_defaults"]["adaptation_penalty"]])
        predictions = restore_outcome(pd.DataFrame(neural), task["outcome"])
        local = [row for job in task_jobs if job["kind"] == "local"
                 for row in load_history_result(out, job)["forecasts"]]
        predictions = pd.concat([predictions, pd.DataFrame(local)], ignore_index=True)
        predictions["history_start"] = task["history_start"]
        predictions["trial_id"] = task["id"]
        predictions["analysis_role"] = ROLE
        predictions["tcn_device"] = device
        predictions["history_role"] = "both_donor_training_and_target_adaptation"
        if len(predictions) != 330 or predictions.duplicated(["family", "sex", "age", "horizon"]).any():
            raise ValueError("Mortality history issued ledger is incomplete")
        if set(predictions.family) != {"tcn_adapted", "tcn_unadapted", "damped_ets"}:
            raise ValueError("Mortality history issued families disagree with specification")
        rows.append(predictions)
        adaptation.extend({**row, "trial_id": task["id"], "outcome": task["outcome"],
                           "history_start": task["history_start"]} for row in corrections)
        for job, payload in zip([job for job in task_jobs if job["kind"] == "tcn"], payloads):
            seeds.append({"trial_id": task["id"], "outcome": task["outcome"],
                          "history_start": task["history_start"], "seed": job["seed"],
                          "payload_path": str(result_path(out, job).relative_to(out)),
                          "checkpoint_path": str((result_path(out, job).parent / "checkpoint.joblib").relative_to(out)),
                          "audit": payload["audit"]})
        fits.extend({"trial_id": task["id"], "sex": job["sex"], "fits": load_history_result(out, job)["fits"]}
                    for job in task_jobs if job["kind"] == "local")
    predictions = pd.concat(rows, ignore_index=True)
    if len(predictions) != 7920:
        raise ValueError("Expected all 24 mortality history arms")
    predictions.to_csv(out / "predictions.csv", index=False)
    write_json(out / "adaptation_audit.json", adaptation)
    write_json(out / "seed_audit.json", seeds)
    write_json(out / "local_fit_audit.json", fits)
    files = [out / name for name in ["predictions.csv", "adaptation_audit.json", "seed_audit.json", "local_fit_audit.json"]]
    commit_phase(out, "issued_commit", files, final_period_scored=False, prediction_rows=len(predictions),
                 tcn_source_fits=120, all_24_history_arms_issued_before_scoring=True,
                 history_starts=list(START_YEARS), role=ROLE)


def score_all(config, out):
    if not phase_complete(out, "issued_commit"):
        raise ValueError("All mortality history arms must be committed before scoring")
    if phase_complete(out, "scoring_complete"):
        return
    points = pd.read_csv(out / "predictions.csv")
    expected = {task["id"] for task in history_tasks(config)}
    if set(points.trial_id) != expected or len(points) != 7920:
        raise ValueError("Scoring requires all 24 complete mortality history arms")
    scores = []
    for task in history_tasks(config):
        cfg = context_config(config, task["target"])
        scored = score_actual(points.loc[points.trial_id.eq(task["id"])], source_panel(), cfg,
                              task["target"], task["outcome"], 2023)
        scores.append(scored)
    scored = pd.concat(scores, ignore_index=True)
    scored.to_csv(out / "point_scores.csv", index=False)
    grouped = []
    for band, subset in [("45+", scored), ("80+", scored.loc[scored.age.isin(config["ages"][-4:])])]:
        table = subset.groupby(["target", "outcome", "history_start", "sex", "family", "horizon"], as_index=False).agg(
            mean_absolute_log_error=("absolute_log_error", "mean"), mean_absolute_rate_error=("absolute_rate_error", "mean"),
            age_cells=("age", "size"), fallback_cells=("status", lambda values: int(values.eq("fallback").sum())))
        table["age_group"] = band
        grouped.append(table)
    summary = pd.concat(grouped, ignore_index=True)
    summary.to_csv(out / "summary.csv", index=False)
    keys = ["target", "outcome", "sex", "family", "horizon", "age_group"]
    paired = summary.loc[summary.history_start.eq(1980)].merge(
        summary.loc[summary.history_start.eq(1990)], on=keys, validate="one_to_one", suffixes=("_1980", "_1990"))
    paired["absolute_log_error_change_1980_minus_1990"] = paired.mean_absolute_log_error_1980-paired.mean_absolute_log_error_1990
    paired["relative_improvement_percent"] = np.where(
        paired.mean_absolute_log_error_1990.gt(0),
        -100*paired.absolute_log_error_change_1980_minus_1990/paired.mean_absolute_log_error_1990, np.nan)
    paired["within_device_history_comparison"] = True
    paired.to_csv(out / "history_comparisons.csv", index=False)
    validation = {"passed": True, "role": ROLE, "history_arms": 24, "tcn_source_fits": 120,
                  "ets_sex_fits": 48, "prediction_rows": len(points), "point_score_rows": len(scored),
                  "summary_rows": len(summary), "history_comparison_rows": len(paired),
                  "fallback_cells": int(scored.status.eq("fallback").sum()),
                  "all_arms_committed_before_scoring": True, "intervals_fitted": False,
                  "hyperparameters_selected": False, "primary_result_replaced": False,
                  "target_and_donor_history_both_changed": True,
                  "cross_device_accuracy_claim": False}
    write_json(out / "validation_report.json", validation)
    commit_phase(out, "scoring_complete", [out / name for name in ["point_scores.csv", "summary.csv",
                 "history_comparisons.csv", "validation_report.json"]], final_period_scored=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/mortality_history_v1")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--gpu-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cpu_workers <= 16 or not 1 <= args.gpu_workers <= 4:
        raise ValueError("Mortality history allows 1..16 CPU and 1..4 GPU workers")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent CPU fallback")
    check_lock()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    tests_path = ROOT / "work/supporting-validation/history_tests.json"
    tests = json.loads(tests_path.read_text())
    if not tests["passed"] or not set(REQUIRED).issubset(tests["tested_code_sha256"]):
        raise ValueError("Current mortality history script and specification must pass tests")
    verify_hashes(ROOT, tests["tested_code_sha256"])
    prior, frozen_code = verify_prior()
    code = {**frozen_code, **{name: sha(ROOT / name) for name in REQUIRED}}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}
    identity = {"code_sha256": code, "prior_manifest_sha256": prior, "config_sha256": sha(config_path),
                "source_sha256": sha(ROOT / "data/processed/design_v1/regional_outcomes.csv"),
                "test_report_sha256": sha(tests_path), "versions": versions, "device": args.device,
                "cpu_workers": args.cpu_workers, "gpu_workers": args.gpu_workers,
                "jobs": job_specs(config, args.device)}
    out = ROOT / args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError("Existing mortality history run requires --resume")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Mortality history resume identity mismatch")
        if manifest["status"] == "complete":
            verify_hashes(out, manifest["output_sha256"])
            print("Completed mortality history run verified; no refitting", flush=True)
            return
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a nonexistent mortality history run")
        out.mkdir(parents=True)
        manifest = {"status": "running", "created_utc": now(), "identity": identity,
                    "role": ROLE, "code_sha256": code, "final_period_scored": False}
        write_json(out / "run_manifest.json", manifest)
        (out / "tests_at_run.json").write_bytes(tests_path.read_bytes())
    started = time.perf_counter()
    run_jobs(identity["jobs"], config, out, args.cpu_workers, args.gpu_workers, args.device)
    assemble_issued(config, out, args.device)
    score_all(config, out)
    check_lock()
    if verify_prior()[0] != prior:
        raise ValueError("A completed prior run changed")
    verify_hashes(ROOT, code)
    manifest.update(status="complete", completed_utc=now(), final_period_scored=True,
                    elapsed_seconds=time.perf_counter()-started,
                    output_sha256=hash_files(out, [path for path in out.rglob("*")
                            if path.is_file() and path != out / "run_manifest.json"]))
    write_json(out / "run_manifest.json", manifest)
    print(f"Completed mortality history sensitivity in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
