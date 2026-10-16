"""Frozen-model mortality/disability comparisons with a global forecast commitment.

All model implementations, grids, chronological selection and residual-bank
construction are reused without changing the completed primary or secondary runs.
"""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
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
from gbd_park.local import forecast_setting
from gbd_park.pooled import forecast_origin
from gbd_park.tcn_forecasting import fit_seed
from gbd_park.intervals import score_intervals
from gbd_park.secondary import (context_config, restore_outcome, actual_truth,
                               score_actual, job_fingerprint)
from gbd_park.supporting import supporting_tasks, working_context, actual_evaluation, OUTCOMES, ROLE
from run_local_baselines import check_lock, sha, now
from run_secondary import (write_json, hash_files, verify_hashes, phase_complete,
                           commit_phase, source_panel, job_spec, candidate_jobs,
                           result_path, load_result, prepare_choices, selected_jobs,
                           assemble_issued as frozen_assemble_issued)

REQUIRED = ["src/gbd_park/supporting.py", "scripts/run_supporting.py",
            "tests/test_supporting.py", "study_design/supporting_outcomes_implementation.md"]
PRIOR = ["local_baselines_v1", "nonneural_v1", "tcn_v1", "intervals_v1", "primary_v1",
         "secondary_v1", "donor_comparisons_gpu_v1", "demography_v1", "count_coherence_v1",
         "improvements_v1", "calibration_v1_2", "reliability_v1_3",
         "population_sensitivity_v1", "release_sensitivity_v1"]


def fit_job(config, job, out):
    """Public reusable fitting primitive; only origin-restricted actual-outcome data enter models."""
    if job["outcome"] not in OUTCOMES:
        raise ValueError("Unknown supporting fit outcome")
    ident = job_fingerprint(job)
    directory = Path(out) / "jobs" / job["trial_id"] / ident
    marker = directory / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if (record["job"] != job or record["job_sha256"] != ident
                or record["config_sha256"] != job_fingerprint(config)):
            raise ValueError("Job identity mismatch")
        verify_hashes(directory, record["artifact_sha256"])
        return str(directory / "result.joblib")
    directory.mkdir(parents=True, exist_ok=True)
    panel, cfg = working_context(source_panel(), config, job["target"], job["outcome"],
                                 job.get("donor_countries"), job["origin"])
    started = time.perf_counter()
    if job["kind"] == "tcn":
        result = fit_seed(panel, cfg, job["origin"], job["base"], job["seed"], job["device"],
                          directory / "checkpoint.joblib")
        result["audit"].update(target=job["target"], actual_outcome=job["outcome"])
    elif job["kind"] == "local":
        forecasts, fits, arima = [], [], []
        for spec in job["specs"]:
            p, f, a = forecast_setting(panel, cfg, job["origin"], job["sex"], spec, target=job["target"])
            forecasts.extend(p)
            fits.extend(f)
            arima.extend(a)
        result = {"origin": job["origin"], "sex": job["sex"], "forecasts": forecasts,
                  "fits": fits, "arima": arima}
    elif job["kind"] == "nonneural":
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        forecasts, fits, calibrations, training = forecast_origin(
            panel, cfg, job["origin"], checkpoints, job.get("selected_bases"))
        result = {"origin": job["origin"], "forecasts": forecasts, "fits": fits,
                  "calibrations": calibrations, "training": training}
    else:
        raise ValueError("Unknown fit job kind")
    if "forecasts" in result:
        result["forecasts"] = restore_outcome(pd.DataFrame(result["forecasts"]), job["outcome"]).to_dict("records")
    result.update(actual_target=job["target"], actual_outcome=job["outcome"], job=job,
                  elapsed_seconds=time.perf_counter()-started)
    joblib.dump(result, directory / "result.joblib", compress=3)
    artifacts = [p for p in directory.rglob("*") if p.is_file() and p.name != "complete.json"]
    write_json(marker, {"job": job, "job_sha256": ident, "config_sha256": job_fingerprint(config), "completed_utc": now(),
                        "artifact_sha256": hash_files(directory, artifacts)})
    return str(directory / "result.joblib")



def run_queue(jobs, config, out, workers, neural_workers, device, phase):
    """One global CPU pool, with a separate CUDA queue only when explicitly requested."""
    unique = {job_fingerprint(job): job for job in jobs}
    pending = []
    for job in unique.values():
        directory = result_path(out, job).parent
        if (directory / "complete.json").exists():
            record = json.loads((directory / "complete.json").read_text())
            if (record["job"] != job or record["job_sha256"] != job_fingerprint(job)
                    or record["config_sha256"] != job_fingerprint(config)):
                raise ValueError("Cached job identity mismatch")
            verify_hashes(directory, record["artifact_sha256"])
        else:
            pending.append(job)
    print(f"{phase}: {len(pending)} new jobs, {len(unique)-len(pending)} verified cached jobs", flush=True)
    if not pending:
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as cpu:
        gpu = ProcessPoolExecutor(max_workers=neural_workers, mp_context=context) if device.startswith("cuda") else None
        try:
            futures = {(gpu if gpu is not None and job["kind"] == "tcn" else cpu).submit(
                fit_job, config, job, out): job for job in pending}
            start = time.perf_counter()
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 12 == 0 or index == len(futures):
                    print(f"{phase}: {index}/{len(futures)} jobs; {time.perf_counter()-start:.1f}s", flush=True)
        finally:
            if gpu is not None:
                gpu.shutdown(wait=True, cancel_futures=True)



def assemble_issued(task, config, out, device):
    """Restart uncommitted JSONL writes before calling immutable assembly."""
    directory = out / "trials" / task["id"]
    if phase_complete(directory, "issued_commit"):
        return
    if not phase_complete(directory, "choices_frozen"):
        raise ValueError("Choices must be committed before issuance")
    # These files belong exclusively to the uncommitted issuance phase.
    # The immutable helper regenerates them from verified fit payloads.
    for name in ["tcn_adaptation_audit.jsonl", "seed_references.jsonl",
                 "local_fit_audit.jsonl", "nonneural_fit_audit.jsonl",
                 "nonneural_adaptation_audit.jsonl"]:
        path = directory / name
        if path.exists():
            path.unlink()
    frozen_assemble_issued(task, config, out, device)


def run_trial_phase(tasks, config, out, device, workers, phase):
    """Independent case phases share a bounded pool, with a barrier on return."""
    functions = {"choices": prepare_choices, "issuance": assemble_issued,
                 "scoring": score_trial}
    if phase not in functions:
        raise ValueError("Unknown supporting case phase")
    function = functions[phase]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = {}
        for task in tasks:
            arguments = (task, config, out) if phase == "scoring" else (task, config, out, device)
            jobs[pool.submit(function, *arguments)] = task
        for index, future in enumerate(as_completed(jobs), 1):
            future.result()
            print(f"{phase}: {index}/{len(tasks)} cases; {jobs[future]['id']}; "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
    elapsed = time.perf_counter()-started
    print(f"{phase} phase complete: {elapsed:.1f}s", flush=True)
    return elapsed


def score_trial(task, config, out):
    directory = out / "trials" / task["id"]
    verify_global_commit(out, config)
    if phase_complete(directory, "scoring_complete"):
        return
    if not phase_complete(directory, "issued_commit"):
        raise ValueError("Issued points and intervals must be committed before final scoring")
    cfg = context_config(config, task["target"])
    points = pd.read_csv(directory / "predictions.csv")
    scores = score_actual(points, source_panel(), cfg, task["target"], task["outcome"], 2023)
    scores.to_csv(directory / "point_scores.csv", index=False)
    truth = actual_truth(source_panel(), task["target"], task["outcome"], 2023)
    cells, wis = score_intervals(pd.read_csv(directory / "intervals.csv"), truth, cfg, 2023)
    cells.to_csv(directory / "interval_scores.csv", index=False)
    wis.to_csv(directory / "wis_scores.csv", index=False)
    tables, verdict = actual_evaluation(scores, pd.read_csv(directory / "origin_population_weights.csv"), cfg, task["outcome"])
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    write_json(directory / "endpoint_verdict.json", verdict)
    cells.groupby(["origin", "sex", "family", "horizon", "scale", "level"], as_index=False).agg(
        coverage=("covered", "mean"), mean_width=("width", "mean"), mean_interval_score=("interval_score", "mean"),
        age_cells=("covered", "size"), n_blocks=("n_blocks", "first")).to_csv(directory / "interval_summary_by_origin.csv", index=False)
    wis.groupby(["origin", "sex", "family", "horizon", "scale"], as_index=False).agg(
        mean_wis_50_80=("wis_50_80", "mean"), mean_wis_50_80_95=("wis_50_80_95", "mean"),
        n_blocks=("n_blocks", "first")).to_csv(directory / "wis_summary_by_origin.csv", index=False)
    assert len(scores) == 8800 and len(cells) == 52800 and len(wis) == 17600
    validation = {"passed": True, "target": task["target"], "outcome": task["outcome"],
                  "prediction_rows": len(scores), "underlying_predictions": 7700, "champion_view_predictions": 1100,
                  "interval_rows": len(cells), "wis_rows": len(wis), "joint_draw_rows": 79200,
                  "prequential_rows": 16940, "residual_banks": 80, "tcn_fits": 157,
                  "selected_fallback_cells": int(scores.status.eq("fallback").sum()),
                  "endpoint_origin_fitted_once": True, "analysis_role": ROLE, "all_trials_committed_before_any_final_scoring": True,
                  "maximum_verification_year": 2023, "final_period_scored": True,
                  "history_start": 1990, "saudi_prevalence_primary_result_replaced": False}
    write_json(directory / "validation_report.json", validation)
    phase_complete(directory, "issued_commit")
    paths = [p for p in directory.rglob("*") if p.is_file()]
    commit_phase(directory, "scoring_complete", paths, final_period_scored=True)



def verify_global_commit(out, config):
    """Scoring requires all 24 issued cases, not merely the current trial."""
    marker = out / "global_issued_commit.json"
    if not phase_complete(out, "global_issued_commit"):
        raise ValueError("All supporting cases must be committed before final scoring")
    record = json.loads(marker.read_text())
    expected = {f"trials/{task['id']}/issued_commit.json" for task in supporting_tasks(config)}
    if len(expected) != 24 or set(record["artifact_sha256"]) != expected:
        raise ValueError("Global commitment must contain exactly all 24 supporting cases")
    for task in supporting_tasks(config):
        if not phase_complete(out / "trials" / task["id"], "issued_commit"):
            raise ValueError("Missing issued supporting ledger")
    return record


def verify_prior():
    """Verify every completed study run under its original manifest conventions."""
    manifests, code = {}, {}
    for run in PRIOR:
        directory = ROOT / "results" / run
        manifest_path = directory / "run_manifest.json"
        record = json.loads(manifest_path.read_text())
        if record["status"] != "complete":
            raise ValueError("Prior study stage incomplete")
        if "output_sha256" in record:
            verify_hashes(directory, record["output_sha256"])
        elif "artifact_sha256" in record:
            verify_hashes(ROOT, record["artifact_sha256"])
        else:
            raise ValueError("Prior manifest has no output hashes")
        for section in ["code_sha256", "protected_sha256", "source_sha256"]:
            if isinstance(record.get(section), dict):
                verify_hashes(ROOT, record[section])
        for name, expected in record.get("code_sha256", {}).items():
            if name in code and code[name] != expected:
                raise ValueError("Prior runs disagree on frozen code identity")
            code[name] = expected
        manifests[run] = sha(manifest_path)
    # The release diagnostic keeps its script/test identities under source_sha256.
    release = json.loads((ROOT / "results/release_sensitivity_v1/run_manifest.json").read_text())
    for name, expected in release["source_sha256"].items():
        if name.endswith(".py"):
            code[name] = expected
    return manifests, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/supporting_v1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--phase-workers", type=int, default=4)
    parser.add_argument("--neural-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.workers <= min(32, os.cpu_count() or 1)
            or not 1 <= args.neural_workers <= 4 or not 1 <= args.phase_workers <= 6):
        raise ValueError("Allowed workers: CPU 1..min(32,logical CPUs); CUDA 1..4; case phases 1..6")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent CPU fallback")
    check_lock()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    tasks = supporting_tasks(config)
    if len(tasks) != 24:
        raise ValueError("Expected six GCC countries and four supporting outcomes")
    tests_path = ROOT / "work/supporting-validation/tests.json"
    tests = json.loads(tests_path.read_text())
    if not tests["passed"] or not set(REQUIRED).issubset(tests["tested_code_sha256"]):
        raise ValueError("Current runner, module, tests, and specification must pass tests first")
    verify_hashes(ROOT, tests["tested_code_sha256"])
    prior, old_code = verify_prior()
    code = {**old_code, **{name: sha(ROOT / name) for name in REQUIRED}}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}
    primary = json.loads((ROOT / "results/primary_v1/run_manifest.json").read_text())
    for name, version in versions.items():
        if primary["versions"].get(name, version) != version:
            raise ValueError(f"Numerical environment differs from Saudi primary: {name}")
    identity = {"code_sha256": code, "config_sha256": sha(config_path), "source_sha256": sha(ROOT / "data/processed/design_v1/regional_outcomes.csv"),
                "prior_manifests_sha256": prior, "test_report_sha256": sha(tests_path), "versions": versions,
                "tasks": tasks, "device": args.device, "workers": args.workers,
                "neural_workers": args.neural_workers, "phase_workers": args.phase_workers}
    out = ROOT / args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError("Existing run requires explicit --resume")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Resume identity mismatch: code/config/source/device/resources/tests must remain identical")
        if manifest["status"] == "complete":
            verify_hashes(out, manifest["output_sha256"])
            print("Existing complete run verified; nothing refitted", flush=True)
            return
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a nonexistent run")
        out.mkdir(parents=True)
        manifest = {"run_id": out.name, "created_utc": now(), "status": "running", "identity": identity,
                    "role": ROLE,
                    "code_sha256": code, "prior_manifests_sha256": prior,
                    "final_period_scored": False, "maximum_scored_year": 2023}
        write_json(out / "run_manifest.json", manifest)
        (out / "tests_at_run.json").write_bytes(tests_path.read_bytes())
        benchmark = ROOT / "work/parallel-benchmark/report.json"
        if benchmark.exists():
            (out / "benchmark_at_run.json").write_bytes(benchmark.read_bytes())
    started = time.perf_counter()
    events = json.loads((out / "events.json").read_text()) if (out / "events.json").exists() else []
    def event(name, **details):
        if name not in [row["event"] for row in events]:
            events.append({"event": name, "time_utc": now(), **copy.deepcopy(details)})
            write_json(out / "events.json", events)
    event("candidate_fitting_started")
    jobs = [job for task in tasks for job in candidate_jobs(task, config, args.device)]
    run_queue(jobs, config, out, args.workers, args.neural_workers, args.device, "Candidate grids")
    phase_seconds = run_trial_phase(tasks, config, out, args.device, args.phase_workers, "choices")
    event("all_choices_frozen", phase_elapsed_seconds=phase_seconds)
    event("selected_fitting_started")
    selected = [job for task in tasks for job in selected_jobs(task, config, out, args.device)]
    run_queue(selected, config, out, args.workers, args.neural_workers, args.device, "Selected ensembles and evaluation")
    phase_seconds = run_trial_phase(tasks, config, out, args.device, args.phase_workers, "issuance")
    commits = {task["id"]: sha(out / "trials" / task["id"] / "issued_commit.json") for task in tasks}
    event("all_trials_issued_before_final_scoring", issued_commit_sha256=commits,
          phase_elapsed_seconds=phase_seconds)
    if not phase_complete(out, "global_issued_commit"):
        commit_phase(out, "global_issued_commit", [out / "trials" / task["id"] / "issued_commit.json" for task in tasks])
    event("evaluation_scoring_started")
    phase_seconds = run_trial_phase(tasks, config, out, args.device, args.phase_workers, "scoring")
    event("all_scoring_complete", phase_elapsed_seconds=phase_seconds)
    check_lock()
    assert verify_prior()[0] == prior
    verify_hashes(ROOT, code)
    validations = [json.loads((out / "trials" / task["id"] / "validation_report.json").read_text()) for task in tasks]
    write_json(out / "validation_report.json", {"passed": all(row["passed"] for row in validations),
               "trials": validations, "trial_count": len(tasks), "all_trials_committed_before_any_final_scoring": True,
               "saudi_prevalence_primary_unchanged": True, "workers": args.workers, "device": args.device,
               "phase_workers": args.phase_workers,
               "this_invocation_elapsed_seconds": time.perf_counter()-started})
    event("complete")
    manifest.update(status="complete", final_period_scored=True, completed_utc=now())
    manifest["output_sha256"] = hash_files(out, [p for p in out.rglob("*") if p.is_file() and p != out / "run_manifest.json"])
    write_json(out / "run_manifest.json", manifest)
    print(f"Completed {len(tasks)} supporting trials; Saudi prevalence primary preserved", flush=True)


if __name__ == "__main__":
    main()
