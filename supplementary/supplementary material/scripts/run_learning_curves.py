"""Run fixed Saudi target-history budgets, then verify and report without retuning."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work/learning-curves-validation/matplotlib"))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import torch
from gbd_park.learning_curves import (OUTCOMES, ROLE, KEYS, tasks, context, target_arrays,
    training_examples, fixed_base, fit_source, fit_local, nonneural_forecasts, check_ledger,
    summarize, families, source_fingerprint)
from gbd_park.pooled import balanced_weights, predict_changes
from gbd_park.tcn import load_checkpoint, state_fingerprint, predict_changes as tcn_predict
from gbd_park.tcn_forecasting import make_forecasts
from gbd_park.secondary import score_actual, job_fingerprint
from run_local_baselines import check_lock, sha, now
from run_secondary import (source_panel, write_json, hash_files, verify_hashes, phase_complete, commit_phase)
from run_supporting import verify_prior as verify_original_prior

REQUIRED = ["src/gbd_park/learning_curves.py", "scripts/run_learning_curves.py",
            "tests/test_learning_curves.py", "study_design/learning_curves_implementation.md"]
EXTRA_PRIOR = ["supporting_v1", "mortality_history_v1", "disability_components_v1"]


def verify_prior():
    manifests, code = verify_original_prior()
    for name in EXTRA_PRIOR:
        out = ROOT / "results" / name
        path = out / "run_manifest.json"
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            raise ValueError("A preceding study run is incomplete")
        verify_hashes(out, record["output_sha256"])
        verify_hashes(ROOT, record["code_sha256"])
        for key, value in record["code_sha256"].items():
            if key in code and code[key] != value:
                raise ValueError("Previous runs disagree on code identity")
            code[key] = value
        manifests[name] = sha(path)
    return manifests, code


def job_specs(config):
    jobs = []
    for outcome in OUTCOMES:
        jobs.extend({"outcome": outcome, "kind": "tcn", "mode": "donor", "seed": seed,
                     "device": "cpu", "base": fixed_base(config, "tcn")}
                    for seed in config["models"]["tcn"]["ensemble_seeds"])
        jobs.extend({"outcome": outcome, "kind": kind, "mode": "donor", "device": "cpu",
                     "base": fixed_base(config, kind)} for kind in ["ridge", "boosting"])
        for years in config["learning_curve"]["history_years"]:
            jobs.extend({"outcome": outcome, "kind": kind, "mode": "pooled", "history_years": years,
                         "device": "cpu", "base": fixed_base(config, kind)} for kind in ["ridge", "boosting"])
            jobs.extend({"outcome": outcome, "kind": "local", "history_years": years,
                         "sex": sex, "device": "cpu"} for sex in config["sexes"])
    return jobs


def result_path(out, job):
    return Path(out) / "jobs" / job_fingerprint(job) / "result.joblib"


def load_result(out, job, config=None):
    path = result_path(out, job)
    record = json.loads((path.parent / "complete.json").read_text())
    if record["job"] != job or record["job_sha256"] != job_fingerprint(job):
        raise ValueError("Learning-curve job identity mismatch")
    if config is not None and record["config_sha256"] != job_fingerprint(config):
        raise ValueError("Learning-curve cached configuration mismatch")
    verify_hashes(path.parent, record["artifact_sha256"])
    return joblib.load(path)


def fit_job(config, job, out):
    path = result_path(out, job)
    marker = path.parent / "complete.json"
    if marker.exists():
        load_result(out, job, config)
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if job["kind"] == "local":
        result = fit_local(source_panel(), config, job["outcome"], job["history_years"], job["sex"])
    else:
        result = fit_source(source_panel(), config, job["outcome"], job["kind"], job["mode"],
                            job.get("history_years"), job.get("seed"), "cpu", path.parent / "checkpoint.joblib")
    result.update(job=job, elapsed_seconds=time.perf_counter()-started)
    joblib.dump(result, path, compress=3)
    write_json(marker, {"job": job, "job_sha256": job_fingerprint(job),
                        "config_sha256": job_fingerprint(config), "completed_utc": now(),
                        "artifact_sha256": hash_files(path.parent, [p for p in path.parent.rglob("*")
                                                                   if p.is_file() and p != marker])})
    return str(path)


def run_jobs(config, out, workers):
    jobs = job_specs(config)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(fit_job, config, job, out): job for job in jobs}
        for index, future in enumerate(as_completed(pending), 1):
            future.result()
            if index % 4 == 0 or index == len(jobs):
                print(f"Learning-curve fits: {index}/{len(jobs)}; {time.perf_counter()-started:.1f}s", flush=True)


def build_issued_tables(config, out):
    """Reconstruct every issued family from cached fits and frozen source outputs."""
    rows, adaptations, fit_audits, source_audits = [], [], [], []
    jobs = job_specs(config)
    loaded = {job_fingerprint(job): load_result(out, job, config) for job in jobs}
    for job in jobs:
        result = loaded[job_fingerprint(job)]
        if job["kind"] == "local":
            fit_audits.append({"job": job, "fits": result["fits"], "arima": result["arima"]})
        else:
            path = result_path(out, job)
            source_audits.append({"job": job, "audit": result["audit"],
                                 "result_path": str(path.relative_to(out)),
                                 "checkpoint_path": str((path.parent / "checkpoint.joblib").relative_to(out)),
                                 "budgets": sorted(result["payloads"])})
    for task in tasks(config):
        outcome, years = task["outcome"], task["history_years"]
        forecasts, corrections = [], []
        selected = [job for job in jobs if job["outcome"] == outcome and
                    ("history_years" not in job or job["history_years"] == years)]
        neural = [loaded[job_fingerprint(job)]["payloads"][years] for job in selected if job["kind"] == "tcn"]
        if [payload["seed"] for payload in neural] != config["models"]["tcn"]["ensemble_seeds"]:
            raise ValueError("All fixed TCN ensemble seeds are required")
        p, a = make_forecasts(neural, config, [config["cold_start_defaults"]["adaptation_penalty"]], include_intercept=True)
        forecasts.extend(p)
        corrections.extend(a)
        for job in selected:
            result = loaded[job_fingerprint(job)]
            if job["kind"] == "local":
                forecasts.extend(result["forecasts"])
            elif job["kind"] != "tcn":
                p, a = nonneural_forecasts(result["payloads"][years], config, job["mode"], job["kind"])
                forecasts.extend(p)
                corrections.extend(a)
        frame = pd.DataFrame(forecasts)
        frame["outcome"], frame["history_years"] = outcome, years
        frame["target_history_start"] = task["target_history_start"]
        frame["donor_history_start"] = config["calendar"]["history_start"]
        frame["trial_id"], frame["analysis_role"], frame["device"] = task["id"], ROLE, "cpu"
        frame["target_windows_per_age_sex"] = years-12
        frame["donor_windows_per_age_sex"] = task["origin"]-config["calendar"]["history_start"]+1-12
        rows.append(frame)
        adaptations.extend({**row, "trial_id": task["id"], "outcome": outcome, "history_years": years,
                            "windows_per_age": years-12, "first_target_input_year": task["target_history_start"]}
                           for row in corrections)
    predictions = pd.concat(rows, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    return predictions, adaptations, source_audits, fit_audits


def assemble_issued(config, out):
    if phase_complete(out, "issued_commit"):
        return
    predictions, adaptations, source_audits, fit_audits = build_issued_tables(config, out)
    validation = check_ledger(predictions, config)
    predictions.to_csv(out / "predictions.csv", index=False)
    write_json(out / "adaptation_audit.json", adaptations)
    write_json(out / "source_fit_audit.json", source_audits)
    write_json(out / "local_fit_audit.json", fit_audits)
    paths = [out / name for name in ["predictions.csv", "adaptation_audit.json", "source_fit_audit.json", "local_fit_audit.json"]]
    # Bind all nested source/checkpoint commits into the global issuance ledger.
    paths.extend(p for p in (out / "jobs").rglob("*") if p.is_file())
    commit_phase(out, "issued_commit", paths, **validation, all_six_arms_issued_before_scoring=True,
                 final_period_scored=False, role=ROLE)


def score_all(config, out):
    if not phase_complete(out, "issued_commit"):
        raise ValueError("All six complete learning-curve arms must be committed before scoring")
    if phase_complete(out, "scoring_complete"):
        return
    points = pd.read_csv(out / "predictions.csv")
    validation = check_ledger(points, config)
    scores = []
    for task in tasks(config):
        subset = points.loc[points.trial_id.eq(task["id"])]
        scores.append(score_actual(subset, source_panel(), config, config["primary_target"], task["outcome"], 2023))
    scored = pd.concat(scores, ignore_index=True)
    summary, paired, adaptation = summarize(scored, config)
    scored.to_csv(out / "point_scores.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "history_comparisons.csv", index=False)
    adaptation.to_csv(out / "adaptation_comparisons.csv", index=False)
    validation.update(passed=True, role=ROLE, point_score_rows=len(scored), summary_rows=len(summary),
        shared_tcn_source_fits=2*len(config["models"]["tcn"]["ensemble_seeds"]),
        shared_nonneural_source_fits=4, pooled_source_fits=12,
        fallback_cells=int(scored.status.eq("fallback").sum()),
        all_six_arms_committed_before_scoring=True, target_history_only_changed=True,
        intervals_fitted=False, champions_selected=False, target_validation_hyperparameters_selected=False,
        arima_order_selected_on_training_aicc_only=True, primary_result_replaced=False)
    write_json(out / "validation_report.json", validation)
    commit_phase(out, "scoring_complete", [out / name for name in ["point_scores.csv", "summary.csv",
        "history_comparisons.csv", "adaptation_comparisons.csv", "validation_report.json"]], final_period_scored=True)


def audit_run(config, out):
    """Read-only replay of every source checkpoint and derived score; no fitting."""
    manifest = json.loads((out / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Audit requires a completed learning-curve run")
    verify_hashes(out, manifest["output_sha256"])
    verify_hashes(ROOT, manifest["code_sha256"])
    if manifest["identity"]["source_sha256"] != sha(ROOT / "data/processed/design_v1/regional_outcomes.csv"):
        raise ValueError("Learning-curve source changed")
    if not phase_complete(out, "issued_commit") or not phase_complete(out, "scoring_complete"):
        raise ValueError("Missing committed learning-curve phase")
    panel = source_panel()
    points = pd.read_csv(out / "predictions.csv")
    validation = check_ledger(points, config)
    replayed = []
    for job in job_specs(config):
        result = load_result(out, job, config)
        if job["kind"] == "local":
            # Local coefficient/candidate metadata are hashed; no statistical refit.
            continue
        record = result["audit"]
        budget = job.get("history_years", min(config["learning_curve"]["history_years"]))
        work = context(panel, config, job["outcome"], budget)
        x, y, meta = training_examples(work, config, budget, job["mode"])
        weights = balanced_weights(meta)
        pd.testing.assert_frame_equal(result["training_meta"], meta.assign(sample_weight=weights))
        if record["status"] != "ok":
            if record["status"] != "fallback" or not record["reason"]:
                raise ValueError("Unexplained source fit failure")
            replayed.append({"job": job, "status": "documented_fallback"})
            continue
        path = result_path(out, job).parent / "checkpoint.joblib"
        neural = job["kind"] == "tcn"
        fitted = load_checkpoint(path) if neural else joblib.load(path)
        fingerprint = source_fingerprint(fitted, job["kind"])
        if not record["fingerprint_before"] == record["fingerprint_after"] == fingerprint:
            raise ValueError("Source checkpoint fingerprint changed")
        mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x-mean)**2, axis=0, weights=weights)
        np.testing.assert_allclose(fitted["scaler"].mean_, mean, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(fitted["scaler"].var_, variance, atol=1e-12, rtol=1e-12)
        country_mass = meta.assign(weight=weights).groupby("country").weight.sum().to_numpy()
        np.testing.assert_allclose(country_mass, country_mass[0], rtol=1e-12)
        predictor = tcn_predict if neural else predict_changes
        for years, payload in result["payloads"].items():
            work = context(panel, config, job["outcome"], years)
            current, levels, cm, historical, labels, tm = target_arrays(work, config, years)
            np.testing.assert_array_equal(payload["target_y"], labels)
            np.testing.assert_array_equal(payload["levels"], levels)
            pd.testing.assert_frame_equal(payload["target_meta"], tm)
            pd.testing.assert_frame_equal(payload["current_meta"], cm)
            if payload["audit"]["status"] == "ok":
                np.testing.assert_array_equal(predictor(fitted, current), payload["current_changes"])
                if job["mode"] == "donor":
                    np.testing.assert_array_equal(predictor(fitted, historical), payload["target_changes"])
            if not tm.input_start.ge(2019-years).all() or not tm.label_end.le(2018).all():
                raise ValueError("Target adaptation used unavailable history")
        replayed.append({"job": job, "status": "checkpoint_replayed", "payloads": len(result["payloads"])})
    reconstructed, adaptation, sources, local = build_issued_tables(config, out)
    order = KEYS + ["prediction", "log_prediction", "status", "target_history_start", "target_windows_per_age_sex"]
    pd.testing.assert_frame_equal(reconstructed[order], points[order], check_dtype=False, atol=1e-12, rtol=1e-12)
    for name, expected in [("adaptation_audit", adaptation), ("source_fit_audit", sources), ("local_fit_audit", local)]:
        if json.loads((out / f"{name}.json").read_text()) != expected:
            raise ValueError(f"Reconstructed {name} disagrees with saved issuance")
    rescored = []
    for task in tasks(config):
        subset = points.loc[points.trial_id.eq(task["id"])]
        rescored.append(score_actual(subset, panel, config, config["primary_target"], task["outcome"], 2023))
    scored = pd.concat(rescored, ignore_index=True)
    saved = pd.read_csv(out / "point_scores.csv")
    for column in ["absolute_log_error", "absolute_rate_error"]:
        np.testing.assert_allclose(scored[column], saved[column], atol=1e-12, rtol=1e-12)
    for name, frame in zip(["summary", "history_comparisons", "adaptation_comparisons"], summarize(scored, config)):
        pd.testing.assert_frame_equal(frame, pd.read_csv(out / f"{name}.csv"), check_dtype=False,
                                      atol=1e-10, rtol=1e-10)
    return {"passed": True, "completed_utc": now(), "run_manifest_sha256": sha(out / "run_manifest.json"),
            "audit_code_sha256": sha(Path(__file__)), **validation,
            "source_checkpoint_replays": replayed, "no_source_model_refitting": True,
            "all_issued_families_and_corrections_reconstructed": True}


def report(config, out, destination, audit):
    if not audit["passed"]:
        raise ValueError("Report requires successful replay validation")
    if destination.exists():
        raise FileExistsError("Existing learning-curve report must be preserved")
    destination.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = pd.read_csv(out / "summary.csv")
    endpoint = summary.loc[summary.horizon.eq(5)].copy()
    endpoint.to_csv(destination / "endpoint_summary.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, outcome in enumerate(OUTCOMES):
        for col, sex in enumerate(config["sexes"]):
            ax = axes[row, col]
            subset = endpoint.loc[endpoint.outcome.eq(outcome) & endpoint.sex.eq(sex) & endpoint.age_group.eq("45+")]
            for family in ["tcn_adapted", "tcn_unadapted", "damped_ets", "pooled_ridge", "pooled_boosting"]:
                part = subset.loc[subset.family.eq(family)].sort_values("history_years")
                ax.plot(part.history_years, part.mean_absolute_log_error, marker="o", label=family)
            ax.set(title=f"{outcome.title()} — {sex}", xlabel="Saudi history years", ylabel="Mean absolute log error")
            ax.set_xticks(config["learning_curve"]["history_years"])
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(destination / "learning_curves.png", dpi=180)
    fig.savefig(destination / "learning_curves.svg")
    plt.close(fig)
    lines = ["# Saudi target-history learning curves", "", "Completed fixed-setting supporting analysis. "
             "Forecast origin 2018; five-year endpoint 2023. Both outcomes retain ages 45–49 through 95+. "
             "Donor histories remain 1990–2018 for every Saudi budget.", "",
             "The full 9,240-row forecast ledger was committed before scoring. Every saved source checkpoint "
             "was replayed without refitting; shared unadapted donor predictions are identical across budgets.", "",
             "## Five-year endpoint, ages 45+", "", "Mean age-specific absolute log error; smaller values are better. "
             "All fourteen families are retained in the accompanying CSVs; this display emphasizes matched transfer "
             "controls and representative statistical/pooled comparisons.", "",
             "| Outcome | Sex | Family | 15 years | 20 years | 29 years |", "|---|---|---|---:|---:|---:|"]
    for outcome in OUTCOMES:
        for sex in config["sexes"]:
            for family in ["tcn_unadapted", "tcn_intercept", "tcn_adapted", "damped_ets", "arima", "pooled_ridge", "pooled_boosting"]:
                part = endpoint.loc[endpoint.outcome.eq(outcome) & endpoint.sex.eq(sex) & endpoint.family.eq(family)
                                    & endpoint.age_group.eq("45+")].set_index("history_years")
                values = " | ".join(f"{part.loc[y, 'mean_absolute_log_error']:.5f}" for y in config["learning_curve"]["history_years"])
                lines.append(f"| {outcome} | {sex} | {family} | {values} |")
    lines.extend(["", "## Oldest-age checks", "", "| Outcome | Sex | Adapted TCN, 15 years | 20 years | 29 years |",
                  "|---|---|---:|---:|---:|"])
    for outcome in OUTCOMES:
        for sex in config["sexes"]:
            part = endpoint.loc[endpoint.outcome.eq(outcome) & endpoint.sex.eq(sex) & endpoint.family.eq("tcn_adapted")
                                & endpoint.age_group.eq("80+")].set_index("history_years")
            values = " | ".join(f"{part.loc[y, 'mean_absolute_log_error']:.5f}" for y in config["learning_curve"]["history_years"])
            lines.append(f"| {outcome} | {sex} | {values} |")
    validation = json.loads((out / "validation_report.json").read_text())
    lines.extend(["", "## Interpretation limits", "", "This sensitivity isolates available Saudi history while holding "
        "donor information, source checkpoints, settings, seeds, and device fixed. Its three budget arms share the same "
        "origin and evaluation labels. Differences are descriptive, and do not establish repeated-sample reliability. "
        "The 15-, 20-, and 29-year budgets provide only 3, 8, and 17 fully labeled adaptation windows per sex–age stratum. "
        "ARIMA selects orders by training AICc within each budget; no family champions or validation hyperparameters "
        "are selected. Intervals are not estimated. GBD outcomes are modeled estimates; the original primary conclusion is unchanged.",
        "", f"Fallback forecast cells: {validation['fallback_cells']:,}. Inspect fit audits for any fitting failures.",
        "", "## Artifacts", "", "- `results/learning_curves_v1/summary.csv`: all families, sexes, outcomes, "
        "horizons and 45+/80+ groups.", "- `history_comparisons.csv`: each family's error relative to its own 29-year arm.",
        "- `adaptation_comparisons.csv`: matched adapted versus unadapted donor comparisons.",
        "- `predictions.csv`, source checkpoints and issuance/scoring commits: full reproducible evidence.",
        "- `work/learning-curves-validation/audit.json`: checkpoint replay and score verification.", ""])
    (destination / "report.md").write_text("\n".join(lines))
    write_json(destination / "validation.json", {"passed": True, "run_manifest_sha256": sha(out / "run_manifest.json"),
                "audit": audit, "report_sha256": hash_files(destination, [p for p in destination.iterdir() if p.is_file()])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/learning_curves_v1")
    parser.add_argument("--report", default="reports/learning_curves_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Replay existing checkpoints and scores without fitting")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("Use 1–16 CPU workers")
    check_lock()
    config_path = ROOT / "study_design/locked_v1/design.json"
    config = json.loads(config_path.read_text())
    out = ROOT / args.output
    if args.verify:
        verified = audit_run(config, out)
        write_json(ROOT / "work/learning-curves-validation/audit.json", verified)
        print(f"Verified {verified['prediction_rows']} learning-curve forecasts without refitting", flush=True)
        return
    gate_path = ROOT / "work/learning-curves-validation/tests.json"
    gate = json.loads(gate_path.read_text())
    if not gate["passed"] or not set(REQUIRED).issubset(gate["tested_code_sha256"]):
        raise ValueError("Current learning-curve implementation requires its passed test gate")
    verify_hashes(ROOT, gate["tested_code_sha256"])
    prior, code = verify_prior()
    code.update({name: sha(ROOT / name) for name in REQUIRED})
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}
    identity = {"code_sha256": code, "prior_manifest_sha256": prior, "config_sha256": sha(config_path),
                "source_sha256": sha(ROOT / "data/processed/design_v1/regional_outcomes.csv"),
                "test_report_sha256": sha(gate_path), "versions": versions, "device": "cpu",
                "workers": args.workers, "jobs": job_specs(config)}
    if out.exists():
        if not args.resume:
            raise FileExistsError("Existing learning-curve output requires --resume")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Learning-curve resume identity mismatch")
        if manifest["status"] == "complete":
            verify_hashes(out, manifest["output_sha256"])
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume a nonexistent learning-curve run")
        out.mkdir(parents=True)
        manifest = {"status": "running", "created_utc": now(), "identity": identity,
                    "code_sha256": code, "role": ROLE, "final_period_scored": False}
        write_json(out / "run_manifest.json", manifest)
        (out / "tests_at_run.json").write_bytes(gate_path.read_bytes())
    started = time.perf_counter()
    if manifest["status"] != "complete":
        run_jobs(config, out, args.workers)
        assemble_issued(config, out)
        score_all(config, out)
        check_lock()
        if verify_prior()[0] != prior:
            raise ValueError("A completed prior study run changed")
        verify_hashes(ROOT, code)
        if sha(config_path) != identity["config_sha256"] or sha(ROOT / "data/processed/design_v1/regional_outcomes.csv") != identity["source_sha256"]:
            raise ValueError("Learning-curve source or design changed during execution")
        manifest.update(status="complete", completed_utc=now(), final_period_scored=True,
                        elapsed_seconds=time.perf_counter()-started,
                        output_sha256=hash_files(out, [p for p in out.rglob("*") if p.is_file() and p != out / "run_manifest.json"]))
        write_json(out / "run_manifest.json", manifest)
    audit = audit_run(config, out)
    write_json(ROOT / "work/learning-curves-validation/audit.json", audit)
    destination = ROOT / args.report
    if not destination.exists():
        report(config, out, destination, audit)
    else:
        record = json.loads((destination / "validation.json").read_text())
        if record["run_manifest_sha256"] != sha(out / "run_manifest.json"):
            raise ValueError("Existing learning-curve report references a different run")
        verify_hashes(destination, record["report_sha256"])
    print(f"Learning curves complete, replayed and reported in {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
