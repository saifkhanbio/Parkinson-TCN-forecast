"""One fixed-design CPU recovery; preserve the original failed GPU artifacts."""
import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import pandas as pd
import run_global_asr as original
from gbd_park import global_asr as asr

OUTPUT = ROOT / "results/global_asr_cpu_recovery_v1"
REPORTS = ROOT / "reports/global_asr_cpu_recovery_v1"
WORK = ROOT / "work/global-asr-cpu-recovery-validation"
AUDIT_PATH = WORK / "independent_audit.json"
ORIGINAL = ROOT / "results/global_asr_v1"
SNAPSHOT = WORK / "preserved_sha256.json"
AUDITOR = "work/learning-curves-validation/audit_global_asr.py"
NEW_FILES = ["scripts/run_global_asr_cpu_recovery.py", "tests/test_global_asr_cpu_recovery.py",
             "study_design/global_asr_cpu_recovery_2026-09-30.md"]
TESTS = WORK / "tests.json"


def write(path, data):
    original.write_json(path, data)


def preservation():
    snapshot = json.loads(SNAPSHOT.read_text())
    original.verify_hashes(ROOT, snapshot["sha256"])
    manifest = json.loads((ORIGINAL / "run_manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The failed original execution must remain a completed preserved ledger")
    original.verify_hashes(ORIGINAL, manifest["output_sha256"])
    original.verify_hashes(ROOT, manifest["code_sha256"])
    return {"snapshot_sha256": asr.digest(SNAPSHOT), "original_run_manifest_sha256": asr.digest(ORIGINAL / "run_manifest.json"),
            "preserved_identities": len(snapshot["sha256"])}


def source_gate(audits, config):
    """A failed seed is an execution failure, never a source-selection option."""
    required = {(task["target"], task["outcome"], scope, seed) for task in asr.tasks(config)
                for scope in asr.SCOPES for seed in config["models"]["tcn"]["ensemble_seeds"]}
    found = [(row["target"], row["outcome"], row["scope"], row["seed"]) for row in audits]
    if len(audits) != 120 or len(set(found)) != 120 or set(found) != required:
        raise ValueError("Recovery requires all 120 prespecified source fits and all 24 complete ensembles")
    failed = [row for row in audits if row["status"] != "ok" or row["device"] != "cpu"]
    if failed:
        raise ValueError(f"Recovery source gate failed: {len(failed)} unsuccessful or non-CPU source fits; no automatic retry")
    for row in audits:
        expected_count = 202 if row["scope"] == "global" else 6
        regional = {country["name"] for country in config["countries"] if country["name"] != row["target"]}
        if (row["base"] != asr.fixed_base(config) or row["parameter_count"] != 1732
                or row["fingerprint_before"] != row["fingerprint_after"]
                or row["target"] in row["countries"] or row["maximum_label_year"] != 2018
                or len(set(row["countries"])) != expected_count or len(row["countries"]) != expected_count
                or (row["scope"] == "regional" and set(row["countries"]) != regional)):
            raise ValueError("CPU recovery changed source settings, exclusion, weights or calendar")
    return {"successful_source_fits": 120, "complete_five_seed_ensembles": 24, "all_scopes_same_device": "cpu"}


def execution_gate(out, config, require_points):
    audits = []
    for job in original.jobs(config, "cpu"):
        if job["kind"] != "tcn":
            continue
        payload = original.load_job(out, job)
        audits.append(dict(payload["audit"], target=job["target"], outcome=job["outcome"], scope=job["scope"]))
    result = source_gate(audits, config)
    if require_points:
        points = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
        asr.check_forecasts(points, config)
        neural = points.loc[points.family.str.startswith("tcn_")]
        if len(neural) != 720 or not neural.status.eq("ok").all():
            raise ValueError("Recovery requires 720 actual TCN forecasts; fallback cells cannot support the comparison")
        if not points.status.eq("ok").all():
            raise ValueError("Recovery report requires all 2,640 forecasts successful; retain baseline failure evidence and stop")
        result.update(forecast_rows=len(points), successful_neural_forecast_cells=len(neural), fallback_cells=0)
    return dict(result, passed=True, predictive_settings_changed=False, automatic_retries=0)


def audit_and_gate(out, config):
    """Load the immutable independent auditor; avoid original hardcoded writers."""
    spec = importlib.util.spec_from_file_location("global_asr_recovery_independent_auditor", ROOT / AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(out)
    if (not result["passed"] or result["forecast_rows"] != 2640
            or result["source_checkpoint_replays"] != 216 or len(result["cases"]) != 12):
        raise ValueError("CPU recovery requires independent replay of every 120 neural and 96 non-neural checkpoint")
    execution_gate(out, config, True)
    write(AUDIT_PATH, result)
    return result


def verify_existing(out, config):
    """Verify frozen recovery witnesses without changing their timestamps/hashes."""
    result = json.loads(AUDIT_PATH.read_text())
    report = json.loads((REPORTS / "validation.json").read_text())
    fingerprint = asr.digest(out / "run_manifest.json")
    if (not result["passed"] or result["forecast_rows"] != 2640 or result["source_checkpoint_replays"] != 216
            or result["run_manifest_sha256"] != fingerprint or result["audit_code_sha256"] != asr.digest(ROOT / AUDITOR)
            or not report["passed"] or report["run_manifest_sha256"] != fingerprint
            or report["audit_sha256"] != asr.digest(AUDIT_PATH) or report["fallback_cells"] != 0):
        raise ValueError("Recovery audit/report witness binding changed")
    original.verify_hashes(REPORTS, report["output_sha256"])
    execution_gate(out, config, True)
    return result


def write_report(out, audit_result):
    """Recovery report has its own audit binding and never writes old report paths."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(out / "point_scores.csv")
    contrasts = pd.read_csv(out / "donor_scope_comparisons.csv")
    saudi = points.loc[points.target.eq("Saudi Arabia") & points.horizon.eq(5)]
    table = ["| Outcome | Sex | Family | Scope | H5 absolute log error |", "|---|---|---|---|---:|"]
    for row in saudi.sort_values(["outcome", "sex", "donor_scope", "family"]).itertuples():
        table.append(f"| {row.outcome} | {row.sex} | {row.family} | {row.donor_scope} | {row.absolute_log_error:.6f} |")
    selected = contrasts.loc[contrasts.family.eq("tcn_adapted") & contrasts.horizon.eq(5)]
    comparisons = ["| Target | Outcome | Sex | Global minus regional log error | Relative improvement |", "|---|---|---|---:|---:|"]
    for row in selected.itertuples():
        comparisons.append(f"| {row.target} | {row.outcome} | {row.sex} | {row.absolute_log_error_change_global_minus_regional:.6f} | {row.relative_improvement_percent:.1f}% |")
    text = ("# Separate global ASR benchmark — CPU execution recovery\n\n"
        "This is the fixed-design CPU recovery of the original CUDA-failed execution. The original run retained 110 failed and ten successful source fits; "
        "every five-seed ensemble failed, so all 720 original TCN cells were persistence fallbacks. Those original files remain preserved and cannot establish a neural accuracy comparison.\n\n"
        "Both donor scopes were rerun on CPU with the same countries, origin, settings and all five seeds. "
        "All 120 source fits and all 2,640 forecasts succeeded, including 720 TCN forecasts. The independent audit replayed all 216 neural/non-neural checkpoints. "
        "All forecast ledgers preceded final-period scoring. No predictive setting was changed during recovery.\n\n"
        "## Saudi five-year endpoint\n\n") + "\n".join(table) + "\n\n## Global versus regional adapted TCN\n\n" + (
        "Negative differences favor global donors. This is a separate full-age ASR supporting comparison, with no replacement of the regional primary verdict.\n\n") + "\n".join(comparisons) + (
        "\n\n## Interpretation limits\n\n"
        "The source contains modeled male/female prevalence and incidence ASRs for 203 supplied countries/territories. Workbook agreement and UN geographic-label matching do not independently establish native GBD provenance, exact location boundaries or standard weights. "
        "The retrospective 2018-origin endpoint was previously inspected. No interval-calibration, count-conversion, causal biological or independent-country inference follows from this point-only benchmark. "
        "The original failed execution, recovery rationale, successful CPU ledger and independent audit are all retained.\n")
    (REPORTS / "report.md").write_text(text)
    for name in ["summary.csv", "five_horizon_means.csv", "donor_scope_comparisons.csv", "adaptation_comparisons.csv"]:
        (REPORTS / name).write_bytes((out / name).read_bytes())
    write(REPORTS / "validation.json", {"passed": True, "run_manifest_sha256": asr.digest(out / "run_manifest.json"),
        "audit_sha256": asr.digest(AUDIT_PATH), "audit_path": str(AUDIT_PATH.relative_to(ROOT)),
        "source_checkpoint_replays": audit_result["source_checkpoint_replays"], "fallback_cells": 0,
        "output_sha256": original.hashes(REPORTS, [path for path in REPORTS.iterdir() if path.name != "validation.json"])})


def identity(config, cpu_workers, neural_workers):
    original_gate = json.loads((ROOT / original.TESTS).read_text())
    if not original_gate["passed"]:
        raise ValueError("Original fixed-design tests were not passed")
    original.verify_hashes(ROOT, original_gate["tested_code_sha256"])
    gate = json.loads(TESTS.read_text())
    if not gate["passed"] or not set(NEW_FILES).issubset(gate["tested_code_sha256"]):
        raise ValueError("CPU recovery wrapper requires its current passing gate")
    original.verify_hashes(ROOT, gate["tested_code_sha256"])
    code_names = list(dict.fromkeys(original.REQUIRED + original.DEPENDENCIES + NEW_FILES + [AUDITOR]))
    return {"code_sha256": {name: asr.digest(ROOT / name) for name in code_names},
        "source_sha256": {name: asr.digest(ROOT / name) for name in original.SOURCES},
        "test_report_sha256": asr.digest(TESTS), "original_test_report_sha256": asr.digest(ROOT / original.TESTS),
        "preservation": preservation(), "device": "cpu", "cpu_workers": cpu_workers, "neural_workers": neural_workers,
        "jobs": original.jobs(config, "cpu"), "recovery_reason": "all_original_GPU_ensembles_failed_due_to_CUDA_launch_errors",
        "fixed_predictive_settings_unchanged": True, "automatic_retries": 0,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": original.torch.__version__, "scipy": original.scipy.__version__, "sklearn": original.sklearn.__version__,
            "statsmodels": original.statsmodels.__version__, "joblib": original.joblib.__version__}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--neural-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cpu_workers <= 8 or not 1 <= args.neural_workers <= 8:
        raise ValueError("Recovery CPU queues allow one through eight workers each")
    original.check_lock()
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    current_identity = identity(config, args.cpu_workers, args.neural_workers)
    if OUTPUT.exists():
        if not args.resume and not args.verify:
            raise FileExistsError("Recovery exists; explicit verified resume is required")
        manifest = json.loads((OUTPUT / "run_manifest.json").read_text())
        if manifest["identity"] != current_identity:
            raise ValueError("Recovery identity changed")
        if manifest["status"] == "failed_execution_gate":
            raise ValueError("The single CPU recovery failed its gate; automatic retry is prohibited")
        if args.verify or manifest["status"] == "complete":
            if manifest["status"] != "complete":
                raise ValueError("Cannot verify an incomplete recovery")
            original.verify_hashes(OUTPUT, manifest["output_sha256"])
            if args.verify or (AUDIT_PATH.exists() and (REPORTS / "validation.json").exists()):
                verify_existing(OUTPUT, config)
            else:
                audited = audit_and_gate(OUTPUT, config)
                write_report(OUTPUT, audited)
            preservation()
            print("Completed CPU recovery verified without refitting", flush=True)
            return
    else:
        if args.resume or args.verify:
            raise FileNotFoundError("Recovery does not exist")
        OUTPUT.mkdir(parents=True)
        manifest = {"status": "running", "created_utc": original.now(), "identity": current_identity,
            "code_sha256": current_identity["code_sha256"], "role": asr.ROLE,
            "execution_role": "single_CPU_recovery_of_failed_GPU_ensembles", "final_period_scored": False}
        write(OUTPUT / "run_manifest.json", manifest)
        for name in ["source_panel.csv.gz", "location_registry.csv", "source_validation.json"]:
            (OUTPUT / name).write_bytes((ORIGINAL / name).read_bytes())
        (OUTPUT / "tests_at_run.json").write_bytes(TESTS.read_bytes())
    started = time.perf_counter()
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=context) as local, \
                ProcessPoolExecutor(max_workers=args.neural_workers, mp_context=context) as neural:
            futures = [(neural if job["kind"] == "tcn" else local).submit(original.fit_job, OUTPUT, config, job)
                       for job in current_identity["jobs"]]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 4 == 0 or count == len(futures):
                    print(f"CPU ASR recovery: {count}/{len(futures)} fits; {time.perf_counter()-started:.1f}s", flush=True)
        try:
            execution_gate(OUTPUT, config, False)
            original.assemble(OUTPUT, config, current_identity["jobs"])
            gate = execution_gate(OUTPUT, config, True)
        except ValueError as exc:
            manifest.update(status="failed_execution_gate", completed_utc=original.now(), failure_reason=str(exc))
            write(OUTPUT / "run_manifest.json", manifest)
            write(OUTPUT / "execution_gate.json", {"passed": False, "reason": str(exc), "automatic_retries": 0})
            raise
        write(OUTPUT / "execution_gate.json", gate)
        original.score_all(OUTPUT, config)
        original.verify_hashes(ROOT, current_identity["code_sha256"])
        original.verify_hashes(ROOT, current_identity["source_sha256"])
        original.check_lock(); preservation()
        manifest.update(status="complete", completed_utc=original.now(), final_period_scored=True,
            elapsed_seconds=time.perf_counter()-started, output_sha256=original.hashes(OUTPUT,
            [path for path in OUTPUT.rglob("*") if path.is_file() and path != OUTPUT / "run_manifest.json"]))
        write(OUTPUT / "run_manifest.json", manifest)
        audited = audit_and_gate(OUTPUT, config)
        write_report(OUTPUT, audited)
        print(f"CPU recovery completed and audited in {time.perf_counter()-started:.1f}s", flush=True)
    finally:
        preservation()


if __name__ == "__main__":
    main()
