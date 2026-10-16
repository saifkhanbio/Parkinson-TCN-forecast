"""Identical locked incidence/GCC replication, with resumable immutable fit jobs."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
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
from gbd_park.pooled import forecast_origin, settings_grid as nonneural_grid, country_pool
from gbd_park.tcn import base_grid
from gbd_park.tcn_forecasting import fit_seed, make_forecasts, select_tcn_settings
from gbd_park.prequential import select_prequential, champion_history
from gbd_park.intervals import build_residual_bank, bank_quantiles, apply_bank, score_intervals
from gbd_park.secondary import (secondary_tasks, context_config, working_context, restore_outcome,
                               internal_outcome, actual_truth, score_actual, baseline_choices,
                               family_mappings, actual_champions, actual_population_weights,
                               actual_evaluation, job_fingerprint)
from run_local_baselines import check_lock, sha, now, json_lines
from run_primary import validate_forecast_ledger
from run_intervals import bank_long

REQUIRED = ["src/gbd_park/secondary.py", "scripts/run_secondary.py", "tests/test_secondary.py",
            "study_design/secondary_implementation.md"]
PRIOR = ["local_baselines_v1", "nonneural_v1", "tcn_v1", "intervals_v1", "primary_v1"]


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def hash_files(directory, paths):
    return {str(Path(path).relative_to(directory)): sha(path) for path in sorted(paths)
            if not Path(path).name.endswith(".tmp")}


def verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Missing or changed artifact: {path}")


def phase_complete(directory, name):
    marker = directory / f"{name}.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record["status"] != "complete":
        raise ValueError("Invalid phase completion marker")
    verify_hashes(directory, record["artifact_sha256"])
    return True


def commit_phase(directory, name, paths, **details):
    if (directory / f"{name}.json").exists():
        raise FileExistsError("Phase is already committed")
    write_json(directory / f"{name}.json", {"status": "complete", "committed_utc": now(),
               "artifact_sha256": hash_files(directory, paths), **details})


@lru_cache(maxsize=1)
def source_panel():
    return pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")


def fit_job(config, job, out):
    """Public reusable fitting primitive; only origin-restricted actual-outcome data enter models."""
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


def job_spec(task, kind, origin, **kwargs):
    return {"trial_id": task["id"], "target": task["target"], "outcome": task["outcome"],
            "kind": kind, "origin": int(origin), **kwargs}


def candidate_jobs(task, config, device):
    jobs = []
    for origin in range(2003, 2014):
        jobs.extend(job_spec(task, "local", origin, sex=sex, specs=local_grid(config)) for sex in config["sexes"])
        jobs.append(job_spec(task, "nonneural", origin))
        jobs.extend(job_spec(task, "tcn", origin, base=base, seed=config["models"]["tcn"]["tuning_seed"],
                             device=device) for base in base_grid(config))
    return jobs


def result_path(out, job):
    return out / "jobs" / job["trial_id"] / job_fingerprint(job) / "result.joblib"


def load_result(out, job):
    directory = result_path(out, job).parent
    record = json.loads((directory / "complete.json").read_text())
    if record["job"] != job:
        raise ValueError("Cached job identity mismatch")
    verify_hashes(directory, record["artifact_sha256"])
    return joblib.load(directory / "result.joblib")


def run_queue(jobs, config, out, workers, neural_workers, device, phase):
    """One global CPU pool, with a separate CUDA queue only when explicitly requested."""
    unique = {job_fingerprint(job): job for job in jobs}
    pending = []
    for job in unique.values():
        directory = result_path(out, job).parent
        if (directory / "complete.json").exists():
            record = json.loads((directory / "complete.json").read_text())
            if record["job"] != job:
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


def prepare_choices(task, config, out, device):
    directory = out / "trials" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if phase_complete(directory, "choices_frozen"):
        return
    cfg = context_config(config, task["target"])
    jobs = candidate_jobs(task, config, device)
    paths, scores, histories, decisions = [], {}, [], []
    for kind in ["local", "nonneural", "tcn"]:
        records = []
        for job in jobs:
            if job["kind"] != kind:
                continue
            result = load_result(out, job)
            if kind == "tcn":
                rows, _ = make_forecasts([result], cfg, cfg["adaptation"]["penalties"])
                records.extend(rows)
            else:
                records.extend(result["forecasts"])
        predictions = restore_outcome(pd.DataFrame(records), task["outcome"])
        prediction_path = directory / f"candidate_{kind}_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        # Commit the unscored candidate ledger before its restricted truth join.
        commit_path = directory / f"candidate_{kind}_ledger.json"
        write_json(commit_path, {"committed_utc": now(), "sha256": sha(prediction_path), "maximum_label_year": 2018})
        scored = score_actual(predictions, source_panel(), cfg, task["target"], task["outcome"], 2018)
        score_path = directory / f"candidate_{kind}_scores.csv"
        scored.to_csv(score_path, index=False)
        scores[kind] = scored
        paths.extend([prediction_path, commit_path, score_path])
        if kind != "tcn":
            for origin in range(2003, 2014):
                picked, picked_decisions = select_prequential(predictions, scored, cfg, origin, kind)
                histories.append(picked)
                decisions.extend(picked_decisions)
    historical = pd.concat(histories, ignore_index=True)
    historical.to_csv(directory / "historical_baseline_predictions.csv", index=False)
    pd.DataFrame(decisions).to_csv(directory / "historical_setting_decisions.csv", index=False)
    historical_scores = score_actual(historical, source_panel(), cfg, task["target"], task["outcome"], 2018)
    mapping = family_mappings(historical_scores, cfg, cfg["calendar"]["reliability_origins"])
    mapping.to_csv(directory / "champion_family_mappings.csv", index=False)
    choices = [select_tcn_settings(scores["tcn"], cfg, origin) for origin in range(2003, 2019)]
    write_json(directory / "tcn_choices.json", choices)
    final_decisions = pd.concat([baseline_choices(scores[group], cfg, cfg["calendar"]["reliability_origins"], group)
                                for group in ["local", "nonneural"]], ignore_index=True)
    final_decisions.to_csv(directory / "settings_decisions.csv", index=False)
    actual_population_weights(source_panel(), cfg, cfg["calendar"]["reliability_origins"], task["outcome"]).to_csv(
        directory / "origin_population_weights.csv", index=False)
    names = ["historical_baseline_predictions.csv", "historical_setting_decisions.csv", "champion_family_mappings.csv",
             "tcn_choices.json", "settings_decisions.csv", "origin_population_weights.csv"]
    commit_phase(directory, "choices_frozen", paths+[directory / name for name in names],
                 actual_target=task["target"], actual_outcome=task["outcome"], maximum_selection_year=2018)


def selected_jobs(task, config, out, device):
    directory = out / "trials" / task["id"]
    if not phase_complete(directory, "choices_frozen"):
        raise ValueError("Freeze choices before selected fits")
    choices = json.loads((directory / "tcn_choices.json").read_text())
    jobs = [job_spec(task, "tcn", choice["fit_origin"], base=choice["base"], seed=seed, device=device)
            for choice in choices for seed in config["models"]["tcn"]["ensemble_seeds"]]
    decisions = pd.read_csv(directory / "settings_decisions.csv")
    local = {s["setting_id"]: s for s in local_grid(config)}
    nonneural = {s["setting_id"]: s for s in nonneural_grid(config)}
    for origin in config["calendar"]["reliability_origins"]:
        for sex in config["sexes"]:
            selected = decisions.loc[decisions.origin.eq(origin) & decisions.sex.eq(sex)
                                      & decisions.family.isin(config["models"]["local_order"])]
            jobs.append(job_spec(task, "local", origin, sex=sex, specs=[local[ident] for ident in selected.setting_id]))
        selected = decisions.loc[decisions.origin.eq(origin) & decisions.family.isin(config["models"]["nonneural_order"])]
        bases = []
        for ident in selected.setting_id:
            base = nonneural[ident]["base"]
            if base not in bases:
                bases.append(base)
        jobs.append(job_spec(task, "nonneural", origin, selected_bases=bases))
    return jobs


def assemble_issued(task, config, out, device):
    directory = out / "trials" / task["id"]
    if phase_complete(directory, "issued_commit"):
        return
    cfg = context_config(config, task["target"])
    families = cfg["models"]["local_order"] + cfg["models"]["nonneural_order"] + ["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
    mapping = pd.read_csv(directory / "champion_family_mappings.csv")
    choices = json.loads((directory / "tcn_choices.json").read_text())
    neural_rows, corrections, references = [], [], []
    for choice in choices:
        jobs = [job_spec(task, "tcn", choice["fit_origin"], base=choice["base"], seed=seed, device=device)
                for seed in cfg["models"]["tcn"]["ensemble_seeds"]]
        payloads = [load_result(out, job) for job in jobs]
        rows, calibration = make_forecasts(payloads, cfg, choice["penalties"], include_intercept=True)
        for row in rows:
            row.update(last_inner_label_year=choice["last_inner_label_year"], selection_status=choice["status"])
        neural_rows.extend(rows)
        corrections.extend(calibration)
        for job, payload in zip(jobs, payloads):
            references.append({"origin": job["origin"], "base": job["base"], "seed": job["seed"],
                               "payload_path": str(result_path(out, job).relative_to(ROOT)),
                               "checkpoint_path": str((result_path(out, job).parent / "checkpoint.joblib").relative_to(ROOT)),
                               "status": payload["audit"]["status"], "state_fingerprint": payload["audit"]["fingerprint_before"]})
    neural = restore_outcome(pd.DataFrame(neural_rows), task["outcome"])
    json_lines(directory / "tcn_adaptation_audit.jsonl", corrections)
    json_lines(directory / "seed_references.jsonl", references)
    historical = pd.concat([pd.read_csv(directory / "historical_baseline_predictions.csv"),
                            neural.loc[neural.origin.le(2013)]], ignore_index=True)
    assert len(historical) == 16940 and not historical.duplicated(["origin", "family", "sex", "age", "horizon"]).any()
    historical.to_csv(directory / "prequential_predictions.csv", index=False)
    write_json(directory / "prequential_prediction_commit.json", {"committed_utc": now(),
               "sha256": sha(directory / "prequential_predictions.csv"), "maximum_verification_year": 2018})
    scored_history = score_actual(historical, source_panel(), cfg, task["target"], task["outcome"], 2018)
    scored_history["log_residual"] = np.log(scored_history.observed_rate)-scored_history.log_prediction
    scored_history.to_csv(directory / "prequential_scores.csv", index=False)
    banks, summaries, blocks, quantiles = {}, [], [], []
    (directory / "banks").mkdir(exist_ok=True)
    for origin in cfg["calendar"]["reliability_origins"]:
        sources = {family: (scored_history, {sex: family for sex in cfg["sexes"]}) for family in families}
        for role in ["local_champion", "nonneural_champion"]:
            selected = mapping.loc[mapping.fit_origin.eq(origin) & mapping.role.eq(role)]
            by_sex = selected.set_index("sex").source_family.to_dict()
            sources[role] = (champion_history(scored_history, cfg, by_sex, role), by_sex)
        for family, (history, by_sex) in sources.items():
            bank = build_residual_bank(history, cfg, origin, family)
            bank["source_family_by_sex"] = by_sex
            assert bank["status"] == "ok" and bank["n_blocks"] == origin-2007
            joblib.dump(bank, directory / "banks" / f"origin{origin}__{family}.joblib", compress=3)
            banks[(origin, family)] = bank
            blocks.append(bank_long(bank))
            quantiles.append(bank_quantiles(bank, cfg))
            summaries.append({"fit_origin": origin, "family": family, "n_blocks": bank["n_blocks"],
                              "last_residual_label_year": max(bank["origins"])+5,
                              "source_family_by_sex": json.dumps(by_sex, sort_keys=True), "status": bank["status"]})
    pd.DataFrame(summaries).to_csv(directory / "bank_summary.csv", index=False)
    pd.concat(blocks, ignore_index=True).to_csv(directory / "residual_blocks.csv", index=False)
    pd.concat(quantiles, ignore_index=True).to_csv(directory / "calibration_quantiles.csv", index=False)
    jobs = selected_jobs(task, config, out, device)
    baseline_rows, local_audit, nn_audit, nn_calibrations = [], [], [], []
    for job in jobs:
        if job["kind"] == "tcn":
            continue
        result = load_result(out, job)
        baseline_rows.extend(result["forecasts"])
        if job["kind"] == "local":
            local_audit.extend(result["fits"])
        else:
            nn_audit.extend(result["fits"])
            nn_calibrations.extend(result["calibrations"])
    json_lines(directory / "local_fit_audit.jsonl", local_audit)
    json_lines(directory / "nonneural_fit_audit.jsonl", nn_audit)
    json_lines(directory / "nonneural_adaptation_audit.jsonl", nn_calibrations)
    decisions = pd.read_csv(directory / "settings_decisions.csv")
    selected = pd.DataFrame(baseline_rows).merge(decisions, on=["origin", "sex", "family", "setting_id"],
                                                how="inner", validate="many_to_one")
    selected["seed_or_ensemble"] = np.where(selected.family.isin(cfg["models"]["local_order"]),
                                              "deterministic_statistical", "11_for_boosting;ridge_deterministic")
    points = pd.concat([selected, neural.loc[neural.origin.ge(2014)]], ignore_index=True)
    points["source_family"] = points.family
    points["run_id"], points["protocol_version"] = out.name, config["version"]
    points["population_scenario"] = "not_applicable_rate_forecast"
    points["residual_block_count"] = points.origin-2007
    validate_forecast_ledger(internal_outcome(points, task["outcome"]), cfg, families)
    points = pd.concat([points, actual_champions(points, mapping, cfg, task["outcome"])], ignore_index=True)
    points = points.sort_values(["origin", "family", "sex", "age", "horizon"]).reset_index(drop=True)
    validate_forecast_ledger(internal_outcome(points, task["outcome"]), cfg, families+["local_champion", "nonneural_champion"])
    points.to_csv(directory / "predictions.csv", index=False)
    interval_frames, draw_frames = [], []
    for (origin, family), part in points.groupby(["origin", "family"]):
        bank = banks[(origin, family)]
        intervals, draws = apply_bank(part, bank, cfg)
        for frame in [intervals, draws]:
            frame["source_family"] = frame.sex.map(bank["source_family_by_sex"])
            frame["run_id"] = out.name
        interval_frames.append(intervals)
        draw_frames.append(draws)
    intervals, draws = pd.concat(interval_frames, ignore_index=True), pd.concat(draw_frames, ignore_index=True)
    assert len(points) == 8800 and len(intervals) == 52800 and len(draws) == 79200 and len(banks) == 80
    intervals.to_csv(directory / "intervals.csv", index=False)
    draws.to_csv(directory / "joint_draws.csv", index=False)
    paths = [path for path in directory.rglob("*") if path.is_file()]
    commit_phase(directory, "issued_commit", paths, final_period_scored=False,
                 prediction_rows=len(points), interval_rows=len(intervals), joint_draw_rows=len(draws),
                 historical_maximum_verification_year=2018)


def score_trial(task, config, out):
    directory = out / "trials" / task["id"]
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
                  "primary_origin_fitted_once": True, "all_trials_committed_before_any_final_scoring": True,
                  "maximum_verification_year": 2023, "final_period_scored": True}
    write_json(directory / "validation_report.json", validation)
    phase_complete(directory, "issued_commit")
    paths = [p for p in directory.rglob("*") if p.is_file()]
    commit_phase(directory, "scoring_complete", paths, final_period_scored=True)


def verify_prior():
    manifests, code = {}, {}
    for run in PRIOR:
        directory = ROOT / "results" / run
        record = json.loads((directory / "run_manifest.json").read_text())
        if record["status"] != "complete":
            raise ValueError("Prior study stage incomplete")
        verify_hashes(directory, record["output_sha256"])
        verify_hashes(ROOT, record["code_sha256"])
        code.update(record["code_sha256"])
        manifests[run] = sha(directory / "run_manifest.json")
    return manifests, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/secondary_v1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--neural-workers", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12 or not 1 <= args.neural_workers <= 4:
        raise ValueError("Allowed workers: CPU 1..12; CUDA 1..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no silent CPU fallback")
    check_lock()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    tasks = secondary_tasks(config)
    tests_path = ROOT / "work/secondary-validation/tests.json"
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
                "tasks": tasks, "device": args.device, "workers": args.workers, "neural_workers": args.neural_workers}
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
                    "role": "prespecified incidence/GCC replication; does not replace Saudi prevalence primary",
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
    for task in tasks:
        prepare_choices(task, config, out, args.device)
        print(f"Frozen choices: {task['id']}", flush=True)
    event("all_choices_frozen")
    event("selected_fitting_started")
    selected = [job for task in tasks for job in selected_jobs(task, config, out, args.device)]
    run_queue(selected, config, out, args.workers, args.neural_workers, args.device, "Selected ensembles and evaluation")
    for task in tasks:
        assemble_issued(task, config, out, args.device)
        print(f"Committed issued ledgers: {task['id']}", flush=True)
    commits = {task["id"]: sha(out / "trials" / task["id"] / "issued_commit.json") for task in tasks}
    event("all_trials_issued_before_final_scoring", issued_commit_sha256=commits)
    if not phase_complete(out, "global_issued_commit"):
        commit_phase(out, "global_issued_commit", [out / "trials" / task["id"] / "issued_commit.json" for task in tasks])
    event("evaluation_scoring_started")
    for task in tasks:
        score_trial(task, config, out)
        print(f"Completed secondary verification: {task['id']}", flush=True)
    check_lock()
    assert verify_prior()[0] == prior
    verify_hashes(ROOT, code)
    validations = [json.loads((out / "trials" / task["id"] / "validation_report.json").read_text()) for task in tasks]
    write_json(out / "validation_report.json", {"passed": all(row["passed"] for row in validations),
               "trials": validations, "trial_count": len(tasks), "all_trials_committed_before_any_final_scoring": True,
               "saudi_prevalence_primary_unchanged": True, "workers": args.workers, "device": args.device,
               "this_invocation_elapsed_seconds": time.perf_counter()-started})
    event("complete")
    manifest.update(status="complete", final_period_scored=True, completed_utc=now())
    manifest["output_sha256"] = hash_files(out, [p for p in out.rglob("*") if p.is_file() and p != out / "run_manifest.json"])
    write_json(out / "run_manifest.json", manifest)
    print(f"Completed {len(tasks)} secondary trials; Saudi prevalence primary preserved", flush=True)


if __name__ == "__main__":
    main()
