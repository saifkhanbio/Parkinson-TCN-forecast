"""Bounded, separately identified global ASR benchmark with frozen issuance."""

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
import torch
import scipy
import sklearn
import statsmodels

from gbd_park import global_asr as asr
from gbd_park.pooled import balanced_weights, build_examples, target_inputs, predict_changes as pooled_predict
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
from run_local_baselines import check_lock, now

SOURCES = ["age_standard/parkinsons_ml_ready_wide.csv", "age_standard/parkinsons_gbd2023_dataset.xlsx",
           "supporting_data/2026-09-26/raw/WPP2024_PopulationByAge5GroupSex_Medium.csv.gz"]
REQUIRED = ["src/gbd_park/global_asr.py", "scripts/run_global_asr.py", "tests/test_global_asr.py",
            "study_design/global_asr_implementation.md"]
DEPENDENCIES = ["src/gbd_park/local.py", "src/gbd_park/pooled.py", "src/gbd_park/adaptation.py",
                "src/gbd_park/tcn.py", "src/gbd_park/tcn_forecasting.py", "scripts/run_local_baselines.py",
                "study_design/locked_v1/design.json"]
TESTS = "work/global-asr-validation/tests.json"


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def hashes(root, paths):
    return {str(Path(path).relative_to(root)): asr.digest(path) for path in sorted(map(Path, paths))}


def verify_hashes(root, expected):
    for name, fingerprint in expected.items():
        if asr.digest(Path(root) / name) != fingerprint:
            raise ValueError(f"Changed artifact: {name}")


def jobs(config, device):
    result = []
    for task in asr.tasks(config):
        result.append({**task, "kind": "local", "scope": "regional", "device": "cpu"})
        for scope in asr.SCOPES:
            result.append({**task, "kind": "nonneural", "scope": scope, "device": "cpu"})
            result.extend({**task, "kind": "tcn", "scope": scope, "device": device, "seed": seed}
                          for seed in config["models"]["tcn"]["ensemble_seeds"])
    return result


def job_path(out, job):
    import hashlib
    key = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:20]
    return Path(out) / "jobs" / job["id"] / key


def load_job(out, job):
    folder = job_path(out, job)
    record = json.loads((folder / "complete.json").read_text())
    if record["job"] != job:
        raise ValueError("Job identity changed")
    verify_hashes(folder, record["output_sha256"])
    return joblib.load(folder / "result.joblib")


def fit_job(out, config, job):
    folder = job_path(out, job)
    if (folder / "complete.json").exists():
        load_job(out, job)
        return job
    folder.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(Path(out) / "source_panel.csv.gz", float_precision="round_trip")
    registry = pd.read_csv(Path(out) / "location_registry.csv")
    work, cfg = asr.context(panel, config, registry, job["target"], job["outcome"], job["scope"])
    if job["kind"] == "local":
        value = asr.local_forecasts(work, cfg, job["outcome"])
    elif job["kind"] == "nonneural":
        value = asr.nonneural_forecasts(work, cfg, job["outcome"], job["scope"], folder)
    else:
        value = asr.fit_seed(work, cfg, job["seed"], job["device"], folder / "checkpoint.joblib")
    joblib.dump(value, folder / "result.joblib", compress=3)
    write_json(folder / "complete.json", {"job": job, "completed_utc": now(),
        "output_sha256": hashes(folder, [path for path in folder.iterdir() if path.is_file() and path.name != "complete.json"])})
    return job


def assemble(out, config, job_specs):
    if (out / "issued_commit.json").exists():
        record = json.loads((out / "issued_commit.json").read_text())
        if record["cases"] != [task["id"] for task in asr.tasks(config)]:
            raise ValueError("Saved ASR global issuance is incomplete")
        verify_hashes(out, record["output_sha256"])
        return
    panel = pd.read_csv(out / "source_panel.csv.gz", float_precision="round_trip")
    registry = pd.read_csv(out / "location_registry.csv")
    records, audits, adaptations = [], [], []
    for task in asr.tasks(config):
        parts = []
        selected = [job for job in job_specs if job["id"] == task["id"]]
        for job in selected:
            if job["kind"] == "tcn":
                continue
            value = load_job(out, job)
            parts.append(value["predictions"])
        for scope in asr.SCOPES:
            payloads = [load_job(out, job) for job in selected if job["kind"] == "tcn" and job["scope"] == scope]
            _, cfg = asr.context(panel, config, registry, task["target"], task["outcome"], scope)
            forecast, calibration = asr.ensemble_forecasts(payloads, cfg, task["outcome"], scope)
            parts.append(forecast)
            audits.extend(dict(item["audit"], target=task["target"], outcome=task["outcome"], scope=scope) for item in payloads)
            adaptations.extend(dict(item, target=task["target"], outcome=task["outcome"], scope=scope) for item in calibration)
        points = pd.concat(parts, ignore_index=True)
        asr.check_forecasts(points, config, single_case=True)
        directory = out / "cases" / task["id"]
        directory.mkdir(parents=True, exist_ok=True)
        points.to_csv(directory / "predictions.csv", index=False, float_format="%.17g")
        write_json(directory / "issued_commit.json", {"issued_utc": now(), "task": task,
            "final_period_scored": False, "output_sha256": hashes(directory, [directory / "predictions.csv"])})
        records.append(points)
    all_points = pd.concat(records, ignore_index=True)
    asr.check_forecasts(all_points, config)
    all_points.to_csv(out / "predictions.csv", index=False, float_format="%.17g")
    write_json(out / "seed_audit.json", audits)
    write_json(out / "adaptation_audit.json", adaptations)
    paths = [path for path in out.rglob("*") if path.is_file() and path.name not in ["run_manifest.json", "issued_commit.json"]]
    paths += list((out / "cases").glob("*/issued_commit.json"))
    write_json(out / "issued_commit.json", {"issued_utc": now(), "final_period_scored": False,
        "cases": [task["id"] for task in asr.tasks(config)], "forecast_rows": len(all_points),
        "output_sha256": hashes(out, paths)})


def score_all(out, config):
    commit = json.loads((out / "issued_commit.json").read_text())
    if commit["final_period_scored"] or commit["cases"] != [task["id"] for task in asr.tasks(config)]:
        raise ValueError("Complete immutable ASR issuance is required before scoring")
    verify_hashes(out, commit["output_sha256"])
    panel = pd.read_csv(out / "source_panel.csv.gz", float_precision="round_trip")
    points = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
    scored = asr.score_forecasts(points, panel, config)
    scored.to_csv(out / "point_scores.csv", index=False, float_format="%.17g")
    grouping = ["target", "outcome", "donor_scope", "family", "sex"]
    summary = scored.groupby(grouping + ["horizon"], as_index=False).agg(
        absolute_log_error=("absolute_log_error", "mean"), absolute_rate_error=("absolute_rate_error", "mean"),
        fallback_cells=("status", lambda values: int(values.eq("fallback").sum())))
    summary.to_csv(out / "summary.csv", index=False)
    means = scored.groupby(grouping, as_index=False).agg(
        mean_absolute_log_error=("absolute_log_error", "mean"), mean_absolute_rate_error=("absolute_rate_error", "mean"))
    means.to_csv(out / "five_horizon_means.csv", index=False)
    compare_keys = ["target", "outcome", "family", "sex", "horizon"]
    contrasts = summary.loc[summary.donor_scope.eq("global")].merge(
        summary.loc[summary.donor_scope.eq("regional")], on=compare_keys, suffixes=("_global", "_regional"), validate="one_to_one")
    contrasts["absolute_log_error_change_global_minus_regional"] = contrasts.absolute_log_error_global - contrasts.absolute_log_error_regional
    contrasts["relative_improvement_percent"] = np.where(contrasts.absolute_log_error_regional.gt(0),
        100 * (1 - contrasts.absolute_log_error_global / contrasts.absolute_log_error_regional), np.nan)
    contrasts.to_csv(out / "donor_scope_comparisons.csv", index=False)
    adaptation = []
    pair_keys = ["target", "outcome", "donor_scope", "sex", "horizon"]
    for adapted, unadapted in [("donor_ridge_adapted", "donor_ridge_unadapted"),
                               ("donor_boosting_adapted", "donor_boosting_unadapted"),
                               ("tcn_adapted", "tcn_unadapted"), ("tcn_intercept", "tcn_unadapted")]:
        matched = summary.loc[summary.family.eq(adapted)].merge(summary.loc[summary.family.eq(unadapted)],
            on=pair_keys, suffixes=("_adapted", "_unadapted"), validate="one_to_one")
        matched["absolute_log_error_change_adapted_minus_unadapted"] = matched.absolute_log_error_adapted - matched.absolute_log_error_unadapted
        adaptation.append(matched)
    pd.concat(adaptation, ignore_index=True).to_csv(out / "adaptation_comparisons.csv", index=False)
    validation = {"passed": True, "role": asr.ROLE, "forecast_rows": len(points), "cases": len(asr.tasks(config)),
                  "source_locations": panel.location_id.nunique(), "tcn_source_fits": 120, "all_cases_issued_before_scoring": True,
                  "intervals_fitted": False, "champions_selected": False, "primary_verdict_replaced": False,
                  "fallback_cells": int(scored.status.eq("fallback").sum())}
    write_json(out / "validation_report.json", validation)
    write_json(out / "scoring_complete.json", {"scored_utc": now(), "issued_commit_sha256": asr.digest(out / "issued_commit.json"),
        "output_sha256": hashes(out, [out / name for name in ["point_scores.csv", "summary.csv", "five_horizon_means.csv",
            "donor_scope_comparisons.csv", "adaptation_comparisons.csv", "validation_report.json"]])})


def audit(out, config):
    """No-refit replay of sources, completed windows, scalers, checkpoints and scores."""
    manifest = json.loads((out / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed global ASR run can be audited")
    verify_hashes(out, manifest["output_sha256"])
    verify_hashes(ROOT, manifest["identity"]["code_sha256"])
    verify_hashes(ROOT, manifest["identity"]["source_sha256"])
    commit = json.loads((out / "issued_commit.json").read_text())
    verify_hashes(out, commit["output_sha256"])
    if commit["cases"] != [task["id"] for task in asr.tasks(config)]:
        raise ValueError("Global ASR issuance case list incomplete")
    panel = asr.asr_panel(pd.read_csv(ROOT / SOURCES[0]))
    saved_panel = pd.read_csv(out / "source_panel.csv.gz", float_precision="round_trip")
    pd.testing.assert_frame_equal(panel, saved_panel, check_dtype=False, atol=1e-12, rtol=1e-12)
    registry = asr.location_registry(pd.read_csv(ROOT / SOURCES[0]), asr.read_un_registry(ROOT / SOURCES[2]))
    pd.testing.assert_frame_equal(registry, pd.read_csv(out / "location_registry.csv"), check_dtype=False)
    replay = asr.replay_workbook(panel, ROOT / SOURCES[1])
    all_recreated, job_specs, seed_count, checkpoint_count = [], manifest["identity"]["jobs"], 0, 0
    for task in asr.tasks(config):
        task_jobs = [job for job in job_specs if job["id"] == task["id"]]
        local = next(job for job in task_jobs if job["kind"] == "local")
        all_recreated.append(load_job(out, local)["predictions"])
        for scope in asr.SCOPES:
            work, cfg = asr.context(panel, config, registry, task["target"], task["outcome"], scope)
            donors = [country["name"] for country in cfg["countries"] if country["name"] != task["target"]]
            x, y, meta = build_examples(work, cfg, 2018, donors)
            cx, _, _ = target_inputs(work, cfg, 2018, task["target"])
            tx, ty, tm = build_examples(work, cfg, 2018, [task["target"]])
            expected_mean = np.average(x, axis=0, weights=balanced_weights(meta))
            expected_var = np.average((x-expected_mean)**2, axis=0, weights=balanced_weights(meta))
            seed_jobs = [job for job in task_jobs if job["kind"] == "tcn" and job["scope"] == scope]
            payloads = []
            for job in seed_jobs:
                payload = load_job(out, job)
                payloads.append(payload)
                seed_count += 1
                if payload["audit"]["countries"] != donors or task["target"] in payload["audit"]["countries"]:
                    raise ValueError("Source target exclusion failed")
                pd.testing.assert_frame_equal(payload["training_meta"].drop(columns=["sample_weight", "fit_origin"]), meta)
                np.testing.assert_allclose(payload["training_meta"].sample_weight, balanced_weights(meta), atol=1e-12)
                np.testing.assert_allclose(payload["target_y"], ty, rtol=1e-13, atol=1e-13)
                pd.testing.assert_frame_equal(payload["target_meta"], tm)
                if payload["audit"]["status"] == "ok":
                    fitted = load_checkpoint(job_path(out, job) / "checkpoint.joblib")
                    if state_fingerprint(fitted) != payload["audit"]["fingerprint_before"]:
                        raise ValueError("Checkpoint fingerprint mismatch")
                    np.testing.assert_allclose(fitted["scaler"].mean_, expected_mean, rtol=1e-11, atol=1e-11)
                    np.testing.assert_allclose(fitted["scaler"].var_, expected_var, rtol=1e-11, atol=1e-11)
                    np.testing.assert_allclose(predict_changes(fitted, cx), payload["current_changes"], rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(predict_changes(fitted, tx), payload["target_changes"], rtol=2e-6, atol=2e-7)
                    checkpoint_count += 1
            forecast, _ = asr.ensemble_forecasts(payloads, cfg, task["outcome"], scope)
            all_recreated.append(forecast)
            njob = next(job for job in task_jobs if job["kind"] == "nonneural" and job["scope"] == scope)
            nonneural = load_job(out, njob)
            all_recreated.append(nonneural["predictions"])
            for fit in nonneural["fits"]:
                expected_countries = donors + [task["target"]] if fit["pool"] == "pooled" else donors
                if set(fit["countries"]) != set(expected_countries) or fit["maximum_label_year"] > 2018:
                    raise ValueError("Non-neural source country/calendar mismatch")
                if fit["status"] != "ok":
                    continue
                fitted = joblib.load(job_path(out, njob) / (fit["model_id"] + ".joblib"))
                nx, _, nm = build_examples(work, cfg, 2018, fit["countries"])
                np.testing.assert_allclose(fitted["scaler"].mean_, np.average(nx, axis=0, weights=balanced_weights(nm)), rtol=1e-11, atol=1e-11)
                base_changes = pooled_predict(fitted, cx)
                check = nonneural["predictions"]
                family = f"{fit['pool']}_{fit['base']['algorithm']}" + ("_unadapted" if fit["pool"] == "donor" else "")
                for index, sex in enumerate(cfg["sexes"]):
                    actual = check.loc[check.family.eq(family) & check.sex.eq(sex)].sort_values("horizon")
                    np.testing.assert_allclose(actual.log_prediction.to_numpy(), base_changes[index] + cx[index, 8], rtol=1e-10, atol=1e-10)
                checkpoint_count += 1
    recreated = pd.concat(all_recreated, ignore_index=True)
    saved = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
    keys = ["target", "outcome", "donor_scope", "family", "sex", "horizon"]
    compare = recreated.merge(saved, on=keys, suffixes=("_replay", "_saved"), validate="one_to_one")
    asr.check_forecasts(saved, config)
    np.testing.assert_allclose(compare.prediction_replay, compare.prediction_saved, rtol=1e-12, atol=1e-12)
    rescored = asr.score_forecasts(saved, panel, config)
    scored = pd.read_csv(out / "point_scores.csv", float_precision="round_trip")
    for field in ["observed_rate", "absolute_log_error", "absolute_rate_error"]:
        np.testing.assert_allclose(rescored[field], scored[field], rtol=1e-10, atol=1e-12)
    result = {"passed": True, "audit_utc": now(), "forecast_rows": len(saved), "seed_payloads": seed_count,
              "checkpoint_replays": checkpoint_count, "workbook_asr_cells": replay["matched_asr_cells"],
              "registry_locations": len(registry), "run_manifest_sha256": asr.digest(out / "run_manifest.json"),
              "audit_code_sha256": asr.digest(Path(__file__)), "local_fit_refit": False,
              "source_hierarchy_independently_verified": False}
    write_json(ROOT / "work/global-asr-validation/audit.json", result)
    return result


def report(out, report_dir, audit_result):
    report_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(out / "point_scores.csv")
    contrasts = pd.read_csv(out / "donor_scope_comparisons.csv")
    saudi = points.loc[points.target.eq("Saudi Arabia") & points.horizon.eq(5)]
    table = ["| Outcome | Sex | Family | Scope | H5 absolute log error |", "|---|---|---|---|---:|"]
    for row in saudi.sort_values(["outcome", "sex", "donor_scope", "family"]).itertuples():
        table.append(f"| {row.outcome} | {row.sex} | {row.family} | {row.donor_scope} | {row.absolute_log_error:.6f} |")
    tcn = contrasts.loc[contrasts.family.eq("tcn_adapted") & contrasts.horizon.eq(5)]
    detail = ["| Target | Outcome | Sex | Global vs regional error change | Relative improvement |",
              "|---|---|---|---:|---:|"]
    for row in tcn.itertuples():
        detail.append(f"| {row.target} | {row.outcome} | {row.sex} | {row.absolute_log_error_change_global_minus_regional:.6f} | {row.relative_improvement_percent:.1f}% |")
    text = "# Separate global ASR benchmark\n\n" + (
        "Supporting point-forecast comparison; origin 2018, verification 2019–2023, Saudi Arabia primary within this supporting analysis. "
        "Full-age age-standardized rates are distinct from the regional age-45+ primary estimand.\n\n"
        f"All {len(points):,} point forecasts were issued before final scoring. The no-refit computational audit passed, replaying "
        f"{audit_result['checkpoint_replays']} checkpoints and {audit_result['workbook_asr_cells']:,} source ASRs. "
        f"There were {int(points.status.eq('fallback').sum())} fallback forecast cells.\n\n"
        "## Saudi five-year endpoint\n\n") + "\n".join(table) + "\n\n## Global versus regional TCN\n\n" + (
        "Negative error changes favor global donors. Relative improvement is undefined when regional error is zero.\n\n") + "\n".join(detail) + (
        "\n\n## Interpretation limits\n\n"
        "The source supplies 203 countries/territories; every label matches a UN country/area record, but the native GBD hierarchy, exact boundaries, standard weights and extraction settings remain unverified. "
        "Workbook agreement is an internal consistency check. No independent-country inference, predictive-interval calibration or conversion of ASRs to counts is supported. "
        "Settings and donor pools were fixed before these fits, but the evaluation years had already been inspected in earlier study work. "
        "No age-smoothed method or post-result champion is constructed. These results do not replace the regional primary verdict.\n\n"
        "All families, every horizon, five-horizon means, donor-scope contrasts and adaptation contrasts are saved in the results directory.\n")
    (report_dir / "report.md").write_text(text)
    for name in ["summary.csv", "five_horizon_means.csv", "donor_scope_comparisons.csv", "adaptation_comparisons.csv"]:
        (report_dir / name).write_bytes((out / name).read_bytes())
    write_json(report_dir / "validation.json", {"passed": True, "run_manifest_sha256": asr.digest(out / "run_manifest.json"),
        "audit_sha256": asr.digest(ROOT / "work/global-asr-validation/audit.json"),
        "output_sha256": hashes(report_dir, [path for path in report_dir.iterdir() if path.name != "validation.json"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/global_asr_v1")
    parser.add_argument("--reports", default="reports/global_asr_v1")
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--gpu-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    out = ROOT / args.output
    if args.verify:
        result = audit(out, config)
        report(out, ROOT / args.reports, result)
        print(json.dumps(result), flush=True)
        return
    if not 1 <= args.cpu_workers <= 16 or not 1 <= args.gpu_workers <= 4:
        raise ValueError("Worker bounds are CPU 1–16 and GPU 1–4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    check_lock()
    test_path = ROOT / TESTS
    gate = json.loads(test_path.read_text())
    if not gate["passed"] or not set(REQUIRED).issubset(gate["tested_code_sha256"]):
        raise ValueError("Global ASR code/specification must pass its current test gate")
    verify_hashes(ROOT, gate["tested_code_sha256"])
    code = {name: asr.digest(ROOT / name) for name in REQUIRED + DEPENDENCIES}
    identity = {"code_sha256": code, "source_sha256": {name: asr.digest(ROOT / name) for name in SOURCES},
        "test_report_sha256": asr.digest(test_path), "device": args.device, "cpu_workers": args.cpu_workers,
        "gpu_workers": args.gpu_workers, "jobs": jobs(config, args.device),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                     "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}}
    if out.exists():
        if not args.resume:
            raise FileExistsError("Existing global ASR run requires --resume")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Global ASR resume identity changed")
        if manifest["status"] == "complete":
            result = audit(out, config); report(out, ROOT / args.reports, result)
            print("Completed global ASR run verified without fitting", flush=True)
            return
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume absent global ASR run")
        out.mkdir(parents=True)
        manifest = {"status": "running", "created_utc": now(), "identity": identity, "code_sha256": code,
                    "role": asr.ROLE, "final_period_scored": False}
        write_json(out / "run_manifest.json", manifest)
        wide = pd.read_csv(ROOT / SOURCES[0])
        panel = asr.asr_panel(wide)
        registry = asr.location_registry(wide, asr.read_un_registry(ROOT / SOURCES[2]))
        if len(registry) != 203:
            raise ValueError("The preserved global source must contain its recorded 203 locations")
        registry.to_csv(out / "location_registry.csv", index=False)
        panel.to_csv(out / "source_panel.csv.gz", index=False, float_format="%.17g")
        write_json(out / "source_validation.json", asr.replay_workbook(panel, ROOT / SOURCES[1]))
        (out / "tests_at_run.json").write_bytes(test_path.read_bytes())
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=context) as cpu, \
            ProcessPoolExecutor(max_workers=args.gpu_workers, mp_context=context) as neural:
        futures = [(neural if job["kind"] == "tcn" else cpu).submit(fit_job, out, config, job) for job in identity["jobs"]]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 4 == 0 or completed == len(futures):
                print(f"Global ASR fits: {completed}/{len(futures)}; {time.perf_counter()-started:.1f}s", flush=True)
    assemble(out, config, identity["jobs"])
    score_all(out, config)
    check_lock(); verify_hashes(ROOT, code); verify_hashes(ROOT, identity["source_sha256"])
    manifest.update(status="complete", completed_utc=now(), final_period_scored=True,
        elapsed_seconds=time.perf_counter()-started, output_sha256=hashes(out,
        [path for path in out.rglob("*") if path.is_file() and path != out / "run_manifest.json"]))
    write_json(out / "run_manifest.json", manifest)
    result = audit(out, config)
    report(out, ROOT / args.reports, result)
    print(f"Global ASR complete and audited: {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
