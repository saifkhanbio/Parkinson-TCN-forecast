"""Project 2024–2028 from the supplied 2023-ending regional GBD data."""

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
from gbd_park.pooled import forecast_origin, settings_grid as nonneural_grid, target_inputs, predict_changes as pooled_predict
from gbd_park.tcn import base_grid, load_checkpoint, predict_changes as neural_predict, state_fingerprint
from gbd_park.tcn_forecasting import fit_seed, make_forecasts, select_tcn_settings
from gbd_park.prequential import champion_history
from gbd_park.intervals import build_residual_bank, apply_bank
from gbd_park.secondary import (context_config, restore_outcome, score_actual,
                               baseline_choices, family_mappings, job_fingerprint)
from gbd_park.projections import (projection_tasks, underlying_families, working_context,
                                 validate_rate_grid, historical_ledger, champion_views,
                                 population_scenarios, conditional_statistics, baseline_statistics,
                                 CUTOFF, ROLES, SCENARIOS)
from run_local_baselines import check_lock, sha, now
from run_secondary import (write_json, hash_files, verify_hashes, phase_complete, commit_phase,
                           source_panel, job_spec, result_path, load_result)

REQUIRED = ["src/gbd_park/projections.py", "scripts/run_projections.py", "tests/test_projections.py",
            "study_design/projections_implementation.md"]
PRIOR = ["local_baselines_v1", "nonneural_v1", "tcn_v1", "intervals_v1", "primary_v1",
         "secondary_v1", "population_sensitivity_v1"]
HANDOFF = "results/population_sensitivity_v1/population_scenarios_2024_2028.csv"
ROLE = "locked_2023_cutoff_scenario_projections_no_future_verification"


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def source_paths(task):
    if task["id"] == "SAU_prevalence":
        return {"candidate_local": "results/local_baselines_v1/candidate_predictions.csv",
                "candidate_nonneural": "results/nonneural_v1/candidate_predictions.csv",
                "candidate_tcn": "results/tcn_v1/candidate_predictions.csv",
                "early": "results/intervals_v1/prequential_predictions.csv",
                "issued": "results/primary_v1/predictions.csv",
                "mapping": "results/primary_v1/champion_family_mappings.csv"}
    prefix = "results/secondary_v1/trials/"+task["id"]+"/"
    return {**{"candidate_"+kind: prefix+"candidate_"+kind+"_predictions.csv"
               for kind in ["local", "nonneural", "tcn"]},
            "early": prefix+"prequential_predictions.csv", "issued": prefix+"predictions.csv",
            "mapping": prefix+"champion_family_mappings.csv"}


def fit_job(config, job, out):
    if job["kind"] == "tcn" and job.get("device") != "cpu":
        raise ValueError("Projection ensembles must use the frozen CPU device")
    ident = job_fingerprint(job)
    directory = Path(out) / "jobs" / job["trial_id"] / ident
    marker = directory / "complete.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if (record["job"] != job or record["job_sha256"] != ident
                or record["config_sha256"] != job_fingerprint(config)):
            raise ValueError("Projection fit-job identity mismatch")
        verify_hashes(directory, record["artifact_sha256"])
        return str(directory / "result.joblib")
    directory.mkdir(parents=True, exist_ok=True)
    panel, cfg = working_context(source_panel(), config, job["target"], job["outcome"], job["origin"])
    started = time.perf_counter()
    if job["kind"] == "tcn":
        result = fit_seed(panel, cfg, job["origin"], job["base"], job["seed"], "cpu", directory / "checkpoint.joblib")
        result["audit"].update(target=job["target"], actual_outcome=job["outcome"])
    elif job["kind"] == "local":
        forecasts, fits, arima = [], [], []
        for spec in job["specs"]:
            p, f, a = forecast_setting(panel, cfg, job["origin"], job["sex"], spec, target=job["target"])
            forecasts.extend(p)
            fits.extend(f)
            arima.extend(a)
        result = {"origin": job["origin"], "sex": job["sex"], "forecasts": forecasts, "fits": fits, "arima": arima}
    elif job["kind"] == "nonneural":
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        forecasts, fits, calibrations, training = forecast_origin(panel, cfg, job["origin"], checkpoints,
                                                                  job.get("selected_bases"))
        result = {"origin": job["origin"], "forecasts": forecasts, "fits": fits,
                  "calibrations": calibrations, "training": training}
    else:
        raise ValueError("Unknown projection fit job kind")
    if "forecasts" in result:
        result["forecasts"] = restore_outcome(pd.DataFrame(result["forecasts"]), job["outcome"]).to_dict("records")
    result.update(actual_target=job["target"], actual_outcome=job["outcome"], job=job,
                  elapsed_seconds=time.perf_counter()-started)
    joblib.dump(result, directory / "result.joblib", compress=3)
    artifacts = [p for p in directory.rglob("*") if p.is_file() and p != marker]
    write_json(marker, {"job": job, "job_sha256": ident, "config_sha256": job_fingerprint(config),
                        "completed_utc": now(), "artifact_sha256": hash_files(directory, artifacts)})
    return str(directory / "result.joblib")


def candidate_jobs(task, config):
    jobs = []
    for origin in range(2014, 2019):
        jobs.extend(job_spec(task, "local", origin, sex=sex, specs=local_grid(config)) for sex in config["sexes"])
        jobs.append(job_spec(task, "nonneural", origin))
        jobs.extend(job_spec(task, "tcn", origin, base=base, seed=config["models"]["tcn"]["tuning_seed"],
                             device="cpu") for base in base_grid(config))
    return jobs


def run_queue(jobs, config, out, workers, label):
    unique = {job_fingerprint(job): job for job in jobs}
    pending = []
    for job in unique.values():
        path = result_path(out, job)
        if (path.parent / "complete.json").exists():
            record = json.loads((path.parent / "complete.json").read_text())
            if record["config_sha256"] != job_fingerprint(config):
                raise ValueError("Cached projection configuration mismatch")
            load_result(out, job)
        else:
            pending.append(job)
    print(f"{label}: {len(pending)} new jobs, {len(unique)-len(pending)} verified cached jobs", flush=True)
    if not pending:
        return
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(fit_job, config, job, out): job for job in pending}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 12 == 0 or index == len(futures):
                print(f"{label}: {index}/{len(futures)}; {time.perf_counter()-start:.1f}s", flush=True)


def prepare_choices(task, config, out):
    directory = out / "trials" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    if phase_complete(directory, "choices_frozen"):
        return
    cfg = context_config(config, task["target"])
    sources = source_paths(task)
    consumed = {name: sha(ROOT / name) for name in sources.values()}
    write_json(directory / "consumed_sources.json", consumed)
    scores = {}
    jobs = candidate_jobs(task, config)
    paths = [directory / "consumed_sources.json"]
    for kind in ["local", "nonneural", "tcn"]:
        old = read_csv(ROOT / sources["candidate_"+kind])
        if set(old.origin) != set(range(2003, 2014)) or not old.outcome.eq(task["outcome"]).all():
            raise ValueError("Unexpected old candidate source")
        records = []
        for job in jobs:
            if job["kind"] != kind:
                continue
            result = load_result(out, job)
            rows = make_forecasts([result], cfg, cfg["adaptation"]["penalties"])[0] if kind == "tcn" else result["forecasts"]
            records.extend(rows)
        recent = restore_outcome(pd.DataFrame(records), task["outcome"])
        candidate = pd.concat([old, recent], ignore_index=True)
        prediction_path = directory / f"historical_candidate_{kind}_predictions.csv"
        candidate.to_csv(prediction_path, index=False)
        marker = directory / f"historical_candidate_{kind}_ledger.json"
        write_json(marker, {"committed_utc": now(), "sha256": sha(prediction_path),
                            "role": "historical_settings_selection_only", "maximum_verification_year": CUTOFF})
        scored = score_actual(candidate, source_panel(), cfg, task["target"], task["outcome"], CUTOFF)
        score_path = directory / f"historical_candidate_{kind}_scores.csv"
        scored.to_csv(score_path, index=False)
        scores[kind] = scored
        paths.extend([prediction_path, marker, score_path])
    decisions = pd.concat([baseline_choices(scores[group], cfg, [CUTOFF], group)
                           for group in ["local", "nonneural"]], ignore_index=True)
    decisions.to_csv(directory / "settings_decisions.csv", index=False)
    choice = select_tcn_settings(scores["tcn"], cfg, CUTOFF)
    write_json(directory / "tcn_choice.json", choice)
    history = historical_ledger(read_csv(ROOT / sources["early"]), read_csv(ROOT / sources["issued"]),
                                cfg, task["target"], task["outcome"])
    history.to_csv(directory / "original_historical_predictions.csv", index=False)
    history_scores = score_actual(history, source_panel(), cfg, task["target"], task["outcome"], CUTOFF)
    history_scores.to_csv(directory / "original_historical_scores.csv", index=False)
    mapping = family_mappings(history_scores, cfg, [CUTOFF])
    prior_mapping = read_csv(ROOT / sources["mapping"]).loc[lambda d: d.fit_origin.eq(2018)]
    left = mapping.set_index(["role", "sex"]).source_family.sort_index()
    right = prior_mapping.set_index(["role", "sex"]).source_family.sort_index()
    pd.testing.assert_series_equal(left, right, check_names=False)
    mapping.to_csv(directory / "champion_family_mappings.csv", index=False)
    names = ["settings_decisions.csv", "tcn_choice.json", "original_historical_predictions.csv",
             "original_historical_scores.csv", "champion_family_mappings.csv"]
    commit_phase(directory, "choices_frozen", paths+[directory / name for name in names],
                 settings_origins=list(range(2003, 2019)), family_selection_origins=list(range(2009, 2014)),
                 maximum_settings_label_year=CUTOFF, maximum_family_selection_label_year=2018,
                 future_projection_outcomes_used=False)


def selected_jobs(task, config, out):
    directory = out / "trials" / task["id"]
    if not phase_complete(directory, "choices_frozen"):
        raise ValueError("Freeze projection choices before selected fits")
    choice = json.loads((directory / "tcn_choice.json").read_text())
    jobs = [job_spec(task, "tcn", CUTOFF, base=choice["base"], seed=seed, device="cpu")
            for seed in config["models"]["tcn"]["ensemble_seeds"]]
    decisions = read_csv(directory / "settings_decisions.csv")
    local = {spec["setting_id"]: spec for spec in local_grid(config)}
    nonneural = {spec["setting_id"]: spec for spec in nonneural_grid(config)}
    for sex in config["sexes"]:
        part = decisions.loc[decisions.sex.eq(sex) & decisions.family.isin(config["models"]["local_order"])]
        jobs.append(job_spec(task, "local", CUTOFF, sex=sex, specs=[local[key] for key in part.setting_id]))
    bases = []
    for ident in decisions.loc[decisions.family.isin(config["models"]["nonneural_order"]), "setting_id"]:
        base = nonneural[ident]["base"]
        if base not in bases:
            bases.append(base)
    jobs.append(job_spec(task, "nonneural", CUTOFF, selected_bases=bases))
    return jobs


def assemble_case(task, config, out):
    directory = out / "trials" / task["id"]
    if phase_complete(directory, "issued_commit"):
        return
    cfg = context_config(config, task["target"])
    jobs = selected_jobs(task, config, out)
    payloads = [load_result(out, job) for job in jobs if job["kind"] == "tcn"]
    choice = json.loads((directory / "tcn_choice.json").read_text())
    neural, adaptation = make_forecasts(payloads, cfg, choice["penalties"], include_intercept=True)
    neural = restore_outcome(pd.DataFrame(neural), task["outcome"])
    neural["last_inner_label_year"] = choice["last_inner_label_year"]
    neural["selection_status"] = choice["status"]
    baseline_rows, audits, references = [], [], []
    for job in jobs:
        result = load_result(out, job)
        if job["kind"] == "tcn":
            references.append({"seed": job["seed"], "origin": CUTOFF, "base": job["base"],
                               "payload_path": str(result_path(out, job).relative_to(out)),
                               "checkpoint_path": str((result_path(out, job).parent / "checkpoint.joblib").relative_to(out)),
                               "audit": result["audit"]})
        else:
            baseline_rows.extend(result["forecasts"])
            audits.append({"job": job, "fits": result["fits"], "calibrations": result.get("calibrations", [])})
    decisions = read_csv(directory / "settings_decisions.csv")
    baseline = pd.DataFrame(baseline_rows).merge(decisions, on=["origin", "sex", "family", "setting_id"],
                                               how="inner", validate="many_to_one")
    points = pd.concat([baseline, neural], ignore_index=True)
    points["source_family"] = points.family
    points = pd.concat([points, champion_views(points, read_csv(directory / "champion_family_mappings.csv"),
                                               cfg, task["target"], task["outcome"])], ignore_index=True)
    points["data_cutoff"], points["gbd_release"], points["analysis_role"] = CUTOFF, config["gbd_release"], ROLE
    points["residual_block_count"] = 16
    points["actual_computation_utc"] = now()
    families = underlying_families(config)+["local_champion", "nonneural_champion"]
    validate_rate_grid(points, cfg, task["target"], task["outcome"], [CUTOFF], families)
    if "observed_rate" in points:
        raise ValueError("Future verification values cannot enter the projection ledger")
    points.to_csv(directory / "predictions.csv", index=False)
    write_json(directory / "seed_references.json", references)
    write_json(directory / "tcn_adaptation_audit.json", adaptation)
    write_json(directory / "baseline_fit_audit.json", audits)
    history = read_csv(directory / "original_historical_scores.csv")
    mapping = read_csv(directory / "champion_family_mappings.csv")
    intervals, draws, bank_summary = [], [], []
    (directory / "banks").mkdir(exist_ok=True)
    for family in families:
        if family in ["local_champion", "nonneural_champion"]:
            by_sex = mapping.loc[mapping.role.eq(family)].set_index("sex").source_family.to_dict()
            residual_history = champion_history(history, cfg, by_sex, family)
        else:
            by_sex = {sex: family for sex in cfg["sexes"]}
            residual_history = history
        bank = build_residual_bank(residual_history, cfg, CUTOFF, family)
        if bank["n_blocks"] != 16 or bank["origins"] != list(range(2003, 2019)):
            raise ValueError("Projection intervals require exactly sixteen completed historical blocks")
        bank["source_family_by_sex"] = by_sex
        joblib.dump(bank, directory / "banks" / (family+".joblib"), compress=3)
        interval, draw = apply_bank(points.loc[points.family.eq(family)], bank, cfg)
        for frame in [interval, draw]:
            frame["source_family"] = frame.sex.map(by_sex)
            frame["data_cutoff"] = CUTOFF
        intervals.append(interval)
        draws.append(draw)
        bank_summary.append({"family": family, "n_blocks": 16, "first_origin": 2003, "last_origin": 2018,
                             "maximum_label_year": CUTOFF, "source_family_by_sex": json.dumps(by_sex, sort_keys=True)})
    intervals, draws = pd.concat(intervals, ignore_index=True), pd.concat(draws, ignore_index=True)
    intervals.to_csv(directory / "intervals.csv", index=False)
    draws.to_csv(directory / "joint_draws.csv.gz", index=False, compression="gzip")
    pd.DataFrame(bank_summary).to_csv(directory / "bank_summary.csv", index=False)
    populations = population_scenarios(read_csv(ROOT / HANDOFF), cfg, task["target"], task["outcome"])
    populations.to_csv(directory / "population_scenarios.csv", index=False)
    burden_points, burden_intervals, burden_draws = conditional_statistics(points, draws, populations, cfg)
    burden_points.to_csv(directory / "burden_predictions.csv", index=False)
    burden_intervals.to_csv(directory / "burden_intervals.csv", index=False)
    burden_draws.to_csv(directory / "burden_joint_statistics.csv.gz", index=False, compression="gzip")
    baseline_statistics(source_panel(), populations, cfg, task["target"], task["outcome"]).to_csv(
        directory / "baseline_2023_burden.csv", index=False)
    base_rates = source_panel().loc[lambda d: d.location_name.eq(task["target"]) & d.outcome.eq(task["outcome"])
                                    & d.year.eq(CUTOFF)].copy()
    base_rates.to_csv(directory / "baseline_2023_age_rates.csv", index=False)
    validation = {"passed": True, "target": task["target"], "outcome": task["outcome"],
                  "role": ROLE, "data_cutoff": CUTOFF, "years": list(range(2024, 2029)),
                  "rate_points": len(points), "rate_intervals": len(intervals), "rate_draws": len(draws),
                  "burden_points": len(burden_points), "burden_intervals": len(burden_intervals),
                  "burden_draw_statistics": len(burden_draws), "n_blocks": 16,
                  "family_selection_origins": list(range(2009, 2014)),
                  "settings_selection_origins": list(range(2003, 2019)),
                  "selected_fallback_cells": int(points.status.eq("fallback").sum()),
                  "future_verification_used": False, "interval_calibration_claim": False,
                  "population_uncertainty_propagated": False, "full_GBD_uncertainty_propagated": False}
    if (len(points), len(intervals), len(draws), len(burden_points), len(burden_intervals), len(burden_draws)) != (
            1760, 10560, 28160, 1215, 3645, 19440):
        raise ValueError("Unexpected projection output dimensions")
    write_json(directory / "validation_report.json", validation)
    commit_phase(directory, "issued_commit", [p for p in directory.rglob("*") if p.is_file()],
                 actual_computation_utc=now(), data_cutoff=CUTOFF, future_verification_used=False)


def run_case_phase(tasks, config, out, workers, function, name):
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(function, task, config, out): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"{name}: {index}/{len(tasks)} {futures[future]['id']}; {time.perf_counter()-start:.1f}s", flush=True)


def verify_sources(tasks):
    manifests, code = {}, {}
    for name in PRIOR:
        directory = ROOT / "results" / name
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("Projection prerequisite run is incomplete")
        verify_hashes(directory, manifest["output_sha256"])
        verify_hashes(ROOT, manifest["code_sha256"])
        manifests[name] = sha(directory / "run_manifest.json")
        code.update(manifest["code_sha256"])
    consumed = {name: sha(ROOT / name) for task in tasks for name in source_paths(task).values()}
    consumed[HANDOFF] = sha(ROOT / HANDOFF)
    return manifests, code, consumed


def compare_tables(actual, expected, keys, values):
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    if left.index.has_duplicates or right.index.has_duplicates or not left.index.equals(right.index):
        raise ValueError("Projection audit table keys disagree")
    for column in values:
        np.testing.assert_allclose(left[column], right[column], rtol=2e-11, atol=2e-10, equal_nan=True)


def verify_case(task, config, out):
    """No fits: reconstruct decisions, issued distributions, and checkpoint inference."""
    directory = out / "trials" / task["id"]
    if not phase_complete(directory, "issued_commit"):
        raise ValueError("Cannot verify an uncommitted projection case")
    cfg = context_config(config, task["target"])
    sources = source_paths(task)
    verify_hashes(ROOT, json.loads((directory / "consumed_sources.json").read_text()))
    source_history = historical_ledger(read_csv(ROOT / sources["early"]), read_csv(ROOT / sources["issued"]),
                                       cfg, task["target"], task["outcome"])
    history = read_csv(directory / "original_historical_scores.csv")
    fresh_history = score_actual(source_history, source_panel(), cfg, task["target"], task["outcome"], CUTOFF)
    keys = ["origin", "family", "sex", "age", "horizon"]
    compare_tables(history, fresh_history, keys, ["prediction", "observed_rate", "absolute_log_error"])
    mapping = family_mappings(fresh_history, cfg, [CUTOFF])
    saved_mapping = read_csv(directory / "champion_family_mappings.csv")
    pd.testing.assert_frame_equal(mapping.sort_values(["role", "sex"]).reset_index(drop=True),
                                  saved_mapping.sort_values(["role", "sex"]).reset_index(drop=True),
                                  check_dtype=False, atol=2e-12, rtol=2e-12)
    baseline_decisions = []
    for kind in ["local", "nonneural", "tcn"]:
        predictions = read_csv(directory / f"historical_candidate_{kind}_predictions.csv")
        scores = read_csv(directory / f"historical_candidate_{kind}_scores.csv")
        fresh = score_actual(predictions, source_panel(), cfg, task["target"], task["outcome"], CUTOFF)
        compare_tables(scores, fresh, keys+["setting_id"], ["prediction", "observed_rate", "absolute_log_error"])
        if kind == "tcn":
            expected = select_tcn_settings(fresh, cfg, CUTOFF)
            saved = json.loads((directory / "tcn_choice.json").read_text())
            if expected["base"] != saved["base"] or expected["penalties"] != saved["penalties"]:
                raise ValueError("Projection TCN choice replay differs")
            np.testing.assert_allclose(expected["loss"], saved["loss"], rtol=1e-12, atol=1e-12)
        else:
            baseline_decisions.append(baseline_choices(fresh, cfg, [CUTOFF], kind))
    pd.testing.assert_frame_equal(pd.concat(baseline_decisions, ignore_index=True),
                                  read_csv(directory / "settings_decisions.csv"), check_dtype=False,
                                  atol=2e-12, rtol=2e-12)
    points = read_csv(directory / "predictions.csv")
    families = underlying_families(cfg)+["local_champion", "nonneural_champion"]
    validate_rate_grid(points, cfg, task["target"], task["outcome"], [CUTOFF], families)
    current_jobs = selected_jobs(task, config, out)
    neural_payloads = [load_result(out, job) for job in current_jobs if job["kind"] == "tcn"]
    choice = json.loads((directory / "tcn_choice.json").read_text())
    neural_rows, _ = make_forecasts(neural_payloads, cfg, choice["penalties"], include_intercept=True)
    neural_rows = restore_outcome(pd.DataFrame(neural_rows), task["outcome"])
    baseline_rows = [row for job in current_jobs if job["kind"] != "tcn"
                     for row in load_result(out, job)["forecasts"]]
    baseline_rows = pd.DataFrame(baseline_rows).merge(read_csv(directory / "settings_decisions.csv"),
        on=["origin", "sex", "family", "setting_id"], how="inner", validate="many_to_one")
    replay_points = pd.concat([baseline_rows, neural_rows], ignore_index=True)
    replay_points = pd.concat([replay_points, champion_views(replay_points, mapping, cfg,
        task["target"], task["outcome"])], ignore_index=True)
    compare_tables(points, replay_points, keys, ["prediction", "log_prediction"])
    intervals, draws = [], []
    for family in families:
        selected_history = fresh_history
        if family in ROLES[1:]:
            by_sex = mapping.loc[mapping.role.eq(family)].set_index("sex").source_family.to_dict()
            selected_history = champion_history(fresh_history, cfg, by_sex, family)
        bank = build_residual_bank(selected_history, cfg, CUTOFF, family)
        saved_bank = joblib.load(directory / "banks" / (family+".joblib"))
        np.testing.assert_allclose(saved_bank["raw_residuals"], bank["raw_residuals"], rtol=1e-12, atol=1e-12)
        interval, draw = apply_bank(points.loc[points.family.eq(family)], bank, cfg)
        intervals.append(interval)
        draws.append(draw)
    intervals, draws = pd.concat(intervals, ignore_index=True), pd.concat(draws, ignore_index=True)
    compare_tables(read_csv(directory / "intervals.csv"), intervals, keys+["scale", "level"], ["lower", "median", "upper"])
    compare_tables(read_csv(directory / "joint_draws.csv.gz"), draws, keys+["residual_origin"], ["log_draw", "rate_draw"])
    populations = population_scenarios(read_csv(ROOT / HANDOFF), cfg, task["target"], task["outcome"])
    bp, bi, bd = conditional_statistics(points, draws, populations, cfg)
    from gbd_park.population_sensitivity import CONTEXT, NODE
    compare_tables(read_csv(directory / "burden_predictions.csv"), bp, CONTEXT+NODE, ["value"])
    compare_tables(read_csv(directory / "burden_intervals.csv"), bi, CONTEXT+NODE+["level"], ["lower", "median", "upper"])
    compare_tables(read_csv(directory / "burden_joint_statistics.csv.gz"), bd, CONTEXT+NODE+["residual_origin"], ["value"])
    compare_tables(read_csv(directory / "baseline_2023_burden.csv"),
                   baseline_statistics(source_panel(), populations, cfg, task["target"], task["outcome"]), CONTEXT+NODE, ["value"])
    checkpoint_replays = []
    work, _ = working_context(source_panel(), cfg, task["target"], task["outcome"], CUTOFF)
    current_x, levels, metadata = target_inputs(work, cfg, CUTOFF, task["target"])
    references = json.loads((directory / "seed_references.json").read_text())
    seed = next(item for item in references if item["seed"] == 11)
    payload = joblib.load(out / seed["payload_path"])
    if payload["audit"]["status"] == "ok":
        fitted = load_checkpoint(out / seed["checkpoint_path"])
        if state_fingerprint(fitted) != payload["audit"]["fingerprint_before"]:
            raise ValueError("Projection neural checkpoint fingerprint differs")
        np.testing.assert_allclose(neural_predict(fitted, current_x), payload["current_changes"], atol=1e-12, rtol=1e-12)
        checkpoint_replays.append("tcn_seed11")
    nn_job = next(job for job in selected_jobs(task, config, out) if job["kind"] == "nonneural")
    nn = load_result(out, nn_job)
    nn_rows = pd.DataFrame(nn["forecasts"])
    for pool, family in [("pooled", "pooled_ridge"), ("donor", "donor_ridge_unadapted")]:
        fit = next(item for item in nn["fits"] if item["pool"] == pool and item["base"]["algorithm"] == "ridge")
        if fit["status"] != "ok":
            continue
        fitted = joblib.load(result_path(out, nn_job).parent / "checkpoints" / (fit["model_id"]+".joblib"))
        changes = pooled_predict(fitted, current_x)
        expected_rows = []
        for index, row in metadata.iterrows():
            for h in range(1, 6):
                expected_rows.append({"sex": row.sex, "age": row.age, "horizon": h,
                                      "log_prediction": levels[index]+changes[index, h-1]})
        actual = nn_rows.loc[nn_rows.family.eq(family) & nn_rows.model_id.eq(fit["model_id"])]
        compare_tables(actual, pd.DataFrame(expected_rows), ["sex", "age", "horizon"], ["log_prediction"])
        checkpoint_replays.append(pool+"_ridge")
    return {"case": task["id"], "passed": True, "checkpoint_replays": checkpoint_replays,
            "rate_points": len(points), "rate_intervals": len(intervals), "rate_draws": len(draws),
            "burden_statistics": len(bp), "future_verification_used": False}


def write_report(tasks, config, out, audit):
    """Summary artifacts are descriptions of projections, never future scores."""
    report_dir = ROOT / "reports" / out.name
    report_dir.mkdir(parents=True, exist_ok=True)
    future, baseline, rates, rate_intervals = [], [], [], []
    for task in tasks:
        directory = out / "trials" / task["id"]
        point = read_csv(directory / "burden_predictions.csv")
        interval = read_csv(directory / "burden_intervals.csv")
        from gbd_park.population_sensitivity import CONTEXT, NODE
        bound = interval.loc[interval.level.eq(.8), CONTEXT+NODE+["lower", "median", "upper", "n_blocks", "uncertainty_scope"]]
        future.append(point.merge(bound, on=CONTEXT+NODE, validate="one_to_one"))
        baseline.append(read_csv(directory / "baseline_2023_burden.csv"))
        rates.append(read_csv(directory / "predictions.csv"))
        rate_intervals.append(read_csv(directory / "intervals.csv"))
    future, baseline = pd.concat(future, ignore_index=True), pd.concat(baseline, ignore_index=True)
    matching = ["target", "outcome", "scenario", "measure", "node", "sex", "age_group", "unit", "hierarchy_level"]
    scenario_base = baseline.loc[baseline.scenario.isin(SCENARIOS), matching+["value"]].rename(columns={"value": "scenario_baseline_2023"})
    summary = future.merge(scenario_base, on=matching, how="left", validate="many_to_one")
    native_keys = [column for column in matching if column != "scenario"]
    native = baseline.loc[baseline.scenario.eq("native_gbd_2023"), native_keys+["value"]].rename(columns={"value": "native_GBD_2023"})
    summary = summary.merge(native, on=native_keys, how="left", validate="many_to_one")
    summary["change_from_scenario_baseline"] = summary.value-summary.scenario_baseline_2023
    summary["change_percent_from_scenario_baseline"] = 100*summary.change_from_scenario_baseline/summary.scenario_baseline_2023
    summary.to_csv(report_dir / "projection_summary_all_years.csv", index=False)
    summary.loc[summary.forecast_year.eq(2028)].to_csv(report_dir / "projection_summary_2028.csv", index=False)
    baseline.to_csv(report_dir / "baseline_2023_burden.csv", index=False)
    pd.concat(rates, ignore_index=True).to_csv(report_dir / "all_family_age_specific_rates.csv", index=False)
    pd.concat(rate_intervals, ignore_index=True).to_csv(report_dir / "all_family_age_specific_intervals.csv.gz", index=False, compression="gzip")
    saudi = summary.loc[summary.target.eq("Saudi Arabia") & summary.forecast_year.eq(2028)
                        & summary.measure.eq("count") & summary.age_group.eq("45+")].copy()
    saudi.to_csv(report_dir / "saudi_2028_totals.csv", index=False)
    lines = ["# Parkinson’s projections from the 2023 data cutoff", "",
             f"Computed {now()}. Source vintage: {config['gbd_release']} as labeled in the supplied export. "
             "The disease data end in 2023; the modeled period is 2024–2028. These were not forecasts issued in calendar 2023. "
             "No 2024–2028 disease observations were used or scored, including the calendar years already elapsed at computation.", "",
             "## Procedures and uncertainty", "",
             "All twelve GCC prevalence/incidence cases retain fourteen families and two champion views. Settings use completed "
             "2003–2018 inner blocks, while comparator-family selection retains the locked 2009–2013 development origins. "
             "The original model grids and CPU device are unchanged; no winner was chosen from the inspected final endpoint.", "",
             "Rate intervals use sixteen original historical residual blocks, spanning forecast origins 2003–2018. "
             "They preserve sex, age and horizon dependence. The 50%/80% intervals and sparse-tail 95% sensitivity are empirical "
             "method outputs; additional blocks do not establish nominal coverage or resolve the previously observed oldest-age failures. "
             "The original primary and all evaluated results remain unchanged.", "",
             "Counts and age shares below use fixed UN WPP 2024 population scenarios. Their intervals include rate-error variation "
             "conditional on each population path; demographic uncertainty and full GBD estimation uncertainty are excluded. "
             "Population reference-date comparability remains unverified. Age-standardized rates were not multiplied by populations.", "",
             "## Saudi 2028 45+ totals", "",
             "Values are modeled prevalent cases or annual incident cases. Brackets show nominal 80% conditional intervals. "
             "Each scenario’s 2023 baseline is shown separately from native GBD 2023; an unaligned-source difference is not disease growth.", "",
             "| Outcome | Sex | View | Population scenario | Native 2023 | Scenario 2023 | Projected 2028 [80%] |", "|---|---|---|---|---:|---:|---:|"]
    for row in saudi.sort_values(["outcome", "sex", "family", "scenario"]).itertuples():
        lines.append(f"| {row.outcome} | {row.sex} | {row.family} | {row.scenario} | {row.native_GBD_2023:.1f} | "
                     f"{row.scenario_baseline_2023:.1f} | {row.value:.1f} [{row.lower:.1f}, {row.upper:.1f}] |")
    lines += ["", "## Output and validation", "",
              "The CSV tables retain all five projection years, all six GCC countries, all family-specific age rates, conditional "
              "count nodes, sex-specific 65+/80+ shares and paired male/female rate ratios. Shares are percentages within age 45+, "
              "not all-age population burden. Point totals remain distinct from predictive medians.", "",
              f"All {len(audit)} case audits passed without refitting. Verification reconstructs historical choices, sixteen-block "
              "banks, rate and transformed quantiles, population formulas and representative checkpoint inference. "
              "There are no future accuracy scores, future coverage estimates or new primary-success verdicts.", ""]
    (report_dir / "report.md").write_text("\n".join(lines))
    files = [p for p in report_dir.iterdir() if p.is_file() and p.name != "validation.json"]
    write_json(report_dir / "validation.json", {"passed": True, "cases": audit, "future_verification_used": False,
               "source_run_manifest_sha256": sha(out / "run_manifest.json"),
               "artifact_sha256": hash_files(report_dir, files)})


def verify_run(tasks, config, out, workers):
    manifest = json.loads((out / "run_manifest.json").read_text())
    if manifest["status"] == "complete":
        verify_hashes(out, manifest["output_sha256"])
    if not phase_complete(out, "global_issued_commit"):
        raise ValueError("All twelve projection cases must be issued before verification/reporting")
    marker = json.loads((out / "global_issued_commit.json").read_text())
    expected = {f"trials/{task['id']}/issued_commit.json" for task in tasks}
    if len(expected) != 12 or set(marker["artifact_sha256"]) != expected:
        raise ValueError("Projection global commitment must contain exactly twelve cases")
    audits = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(verify_case, task, config, out): task for task in tasks}
        for future in as_completed(futures):
            audits.append(future.result())
            print("Verified projection case: "+futures[future]["id"], flush=True)
    return sorted(audits, key=lambda item: item["case"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/projections_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--phase-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Audit an existing committed run without model fitting")
    args = parser.parse_args()
    if not 1 <= args.workers <= min(32, os.cpu_count() or 1) or not 1 <= args.phase_workers <= 6:
        raise ValueError("Invalid bounded projection worker counts")
    check_lock()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    tasks = projection_tasks(config)
    if len(tasks) != 12:
        raise ValueError("Projection scope must be twelve main GCC cases")
    tests_path = ROOT / "work/projections-validation/tests.json"
    tests = json.loads(tests_path.read_text())
    if not tests["passed"] or not set(REQUIRED).issubset(tests["tested_code_sha256"]):
        raise ValueError("Current projection code, tests and specification must pass first")
    verify_hashes(ROOT, tests["tested_code_sha256"])
    prior, old_code, consumed = verify_sources(tasks)
    code = {**old_code, **{name: sha(ROOT / name) for name in REQUIRED}}
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}
    primary = json.loads((ROOT / "results/primary_v1/run_manifest.json").read_text())
    if versions != primary["versions"]:
        raise ValueError("Projection numerical environment must match the primary CPU experiment")
    identity = {"code_sha256": code, "source_sha256": {**consumed,
                    "data/processed/design_v1/regional_outcomes.csv": sha(ROOT / "data/processed/design_v1/regional_outcomes.csv")},
                "config_sha256": sha(config_path), "prior_manifest_sha256": prior,
                "test_report_sha256": sha(tests_path), "versions": versions, "device": "cpu",
                "workers": args.workers, "phase_workers": args.phase_workers, "tasks": tasks}
    out = ROOT / args.output
    if out.exists():
        if not args.resume and not args.verify:
            raise FileExistsError("Existing projection run requires explicit --resume or --verify")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Projection resume identity mismatch")
        if args.verify:
            audit = verify_run(tasks, config, out, args.phase_workers)
            destination = ROOT / "work/projections-validation/reverification.json"
            write_json(destination, {"passed": True, "created_utc": now(), "source_manifest_sha256": sha(out / "run_manifest.json"),
                                     "cases": audit, "fits": 0})
            print("Projection verification complete; no fits or projection outputs changed", flush=True)
            return
        if manifest["status"] == "complete":
            verify_hashes(out, manifest["output_sha256"])
            print("Completed projection run verified; nothing refitted", flush=True)
            return
    else:
        if args.resume or args.verify:
            raise FileNotFoundError("Requested projection run does not exist")
        out.mkdir(parents=True)
        manifest = {"status": "running", "created_utc": now(), "identity": identity,
                    "code_sha256": code, "role": ROLE, "data_cutoff": CUTOFF,
                    "future_verification_used": False, "gbd_release": config["gbd_release"]}
        write_json(out / "run_manifest.json", manifest)
        (out / "tests_at_run.json").write_bytes(tests_path.read_bytes())
    started = time.perf_counter()
    run_queue([job for task in tasks for job in candidate_jobs(task, config)], config, out, args.workers,
              "Historical candidate extension")
    run_case_phase(tasks, config, out, args.phase_workers, prepare_choices, "Projection choices")
    jobs = [job for task in tasks for job in selected_jobs(task, config, out)]
    run_queue(jobs, config, out, args.workers, "Selected 2023-cutoff models")
    run_case_phase(tasks, config, out, args.phase_workers, assemble_case, "Projection issuance")
    if not phase_complete(out, "global_issued_commit"):
        commit_phase(out, "global_issued_commit", [out / "trials" / task["id"] / "issued_commit.json" for task in tasks],
                     data_cutoff=CUTOFF, future_verification_used=False, actual_computation_utc=now())
    audit = verify_run(tasks, config, out, args.phase_workers)
    check_lock()
    verify_hashes(ROOT, identity["source_sha256"])
    verify_hashes(ROOT, code)
    if verify_sources(tasks)[0] != prior:
        raise ValueError("A projection source run changed")
    write_json(out / "validation_report.json", {"passed": True, "cases": audit, "case_count": 12,
               "fits": {"new_candidate_jobs": 660, "selected_jobs": len(jobs), "unique_neural_fits": 540},
               "future_verification_used": False, "original_primary_unchanged": True,
               "data_cutoff": CUTOFF, "elapsed_seconds": time.perf_counter()-started})
    manifest.update(status="complete", completed_utc=now(), elapsed_seconds=time.perf_counter()-started,
                    output_sha256=hash_files(out, [p for p in out.rglob("*")
                                                   if p.is_file() and p != out / "run_manifest.json"]))
    write_json(out / "run_manifest.json", manifest)
    write_report(tasks, config, out, audit)
    print(f"Completed twelve projection cases in {time.perf_counter()-started:.1f}s; no future outcomes scored", flush=True)


if __name__ == "__main__":
    main()
