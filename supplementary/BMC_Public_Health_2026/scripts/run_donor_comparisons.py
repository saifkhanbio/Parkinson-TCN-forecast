"""Prespecified Saudi donor-pool comparisons with a fresh common-device reference."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
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
from gbd_park.donors import select_donors
from gbd_park.evaluation import matched_tcn_harm, primary_contrasts
from gbd_park.intervals import apply_bank, build_residual_bank, score_intervals
from gbd_park.scoring import score_forecasts
from gbd_park.tcn import base_grid
from gbd_park.tcn_forecasting import base_id, fit_seed, make_forecasts, select_tcn_settings
from run_local_baselines import check_lock, now, sha


def arms(config):
    return [{"arm": "all_six", "strategy": "all_six_other_regional_countries", "seed": None},
            {"arm": "gcc_five", "strategy": "other_five_gcc", "seed": None},
            {"arm": "similar_three", "strategy": "three_similar", "seed": None}] + [
        {"arm": f"random_three_{seed}", "strategy": "three_random", "seed": seed}
        for seed in config["donors"]["random_seeds"]]


def fit_job_description(origin, countries, base, seed, device):
    description = {"origin": int(origin), "countries": sorted(countries), "base": dict(base),
                   "seed": int(seed), "device": device, "target": "Saudi Arabia", "outcome": "prevalence"}
    description["job_id"] = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    return description


def source_context(panel, config, job):
    copied = deepcopy(config)
    names = set(job["countries"])
    target = config["primary_target"]
    if target in names or len(names) != len(job["countries"]) or not names:
        raise ValueError("Cached donor pool must exclude the target and contain unique countries")
    if not names.issubset({country["name"] for country in config["countries"]}):
        raise ValueError("Unknown donor in cached source pool")
    # Preserve the original model training-row order while the cache key is
    # invariant to the ordering used to describe membership of the pool.
    copied["countries"] = [country for country in copied["countries"] if country["name"] in names | {target}]
    working = panel.loc[panel.outcome.eq("prevalence") & panel.year.le(job["origin"])
                        & panel.location_name.isin(names | {target})].copy()
    return working, copied


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_cached_job(out, job):
    directory = Path(out) / "cache"
    ready = directory / f"{job['job_id']}.ready.json"
    if not ready.exists():
        return None
    record = json.loads(ready.read_text())
    if record["job"] != job:
        raise ValueError("Cached fit identity changed")
    payload_path = directory / f"{job['job_id']}.joblib"
    if not payload_path.is_file() or sha(payload_path) != record["payload_sha256"]:
        raise ValueError("Committed cached payload changed or is missing")
    if record["checkpoint_sha256"] is not None:
        checkpoint = Path(out) / "checkpoints" / f"{job['job_id']}.joblib"
        if not checkpoint.is_file() or sha(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("Committed cached checkpoint changed or is missing")
    return joblib.load(payload_path)


def cached_fit_job(config, job, out):
    cached = load_cached_job(out, job)
    if cached is not None:
        return cached
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    working, copied = source_context(panel, config, job)
    checkpoint = Path(out) / "checkpoints" / f"{job['job_id']}.joblib"
    payload = fit_seed(working, copied, job["origin"], job["base"], job["seed"], job["device"], checkpoint)
    if set(payload["audit"]["countries"]) != set(job["countries"]):
        raise ValueError("Fitted source countries disagree with the cached donor plan")
    if not payload["training_meta"].sex.isin(config["sexes"]).all() or set(payload["training_meta"].sex) != set(config["sexes"]):
        raise ValueError("A selected donor pool must train on both sexes")
    payload["cache_job"] = job
    path = Path(out) / "cache" / f"{job['job_id']}.joblib"
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    joblib.dump(payload, temporary, compress=3)
    temporary.replace(path)
    _atomic_json(path.with_suffix(".ready.json"), {
        "job": job, "payload_sha256": sha(path),
        "checkpoint_sha256": sha(checkpoint) if payload["audit"]["status"] == "ok" else None})
    return payload


def compose_forecasts(config, plans, payloads, origin, arm, base, seeds, penalties, include_intercept):
    """Pair sex-specific source pools before shared-architecture selection."""
    rows, calibrations = [], []
    for sex in config["sexes"]:
        plan = plans[(origin, arm, sex)]
        group = []
        for seed in seeds:
            key = fit_job_description(origin, plan["countries"], base, seed, plan["device"])["job_id"]
            if key not in payloads:
                raise ValueError("Missing cached source fit for a required ensemble seed")
            group.append(payloads[key])
        if sorted(payload["seed"] for payload in group) != sorted(seeds):
            raise ValueError("Incomplete seed ensemble")
        predicted, adapted = make_forecasts(group, config, penalties, include_intercept=include_intercept)
        metadata = {"arm": arm, "donor_strategy": plan["strategy"], "random_donor_seed": plan["seed"],
                    "donor_list_sha256": plan["donor_list_sha256"],
                    "donor_countries": "|".join(plan["countries"]),
                    "source_cache_keys": "|".join(p["cache_job"]["job_id"] for p in group),
                    "device": plan["device"], "experiment_role": "prespecified_secondary_donor_comparison"}
        rows.extend({**row, **metadata} for row in predicted if row["sex"] == sex)
        calibrations.extend({**row, **metadata} for row in adapted if row["sex"] == sex)
    return rows, calibrations


def score_arms(predictions, truth, config, maximum_year):
    return pd.concat([score_forecasts(group, truth, config["ages"], config["calendar"]["horizons"], maximum_year)
                      for _, group in predictions.groupby("arm", sort=True)], ignore_index=True)


def write_frame(path, frame):
    """A resumed phase must reproduce previously committed table bytes exactly."""
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(temporary, index=False)
    if path.exists():
        if sha(path) != sha(temporary):
            raise ValueError(f"Resumed computation changed an existing committed table: {path.name}")
        temporary.unlink()
    else:
        temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"Resumed computation changed committed metadata: {path.name}")
    else:
        _atomic_json(path, value)


def validate_reported_grid(frame, config, definitions, origins):
    keys = ["arm", "origin", "family", "sex", "age", "horizon"]
    expected = {(arm["arm"], origin, family, sex, age, horizon)
                for arm in definitions for origin in origins
                for family in ["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
                for sex in config["sexes"] for age in config["ages"]
                for horizon in config["calendar"]["horizons"]}
    if "observed_rate" in frame or len(frame) != len(expected) or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError("Reported donor forecast ledger must have complete unique unverified coordinates")
    if not frame.forecast_year.eq(frame.origin + frame.horizon).all():
        raise ValueError("Donor forecast years disagree with their horizons")
    seeds = "|".join(map(str, sorted(config["models"]["tcn"]["ensemble_seeds"])))
    if not frame.seed_or_ensemble.eq(seeds).all():
        raise ValueError("All five fixed seeds are required in every reported donor forecast")
    for _, group in frame.groupby(["arm", "origin"]):
        if group.base_json.nunique() != 1:
            raise ValueError("Each donor arm must use one shared architecture for both target sexes")


def _reference_manifest():
    directory = ROOT / "results/primary_v1"
    record = json.loads((directory / "run_manifest.json").read_text())
    if record["status"] != "complete" or not record["final_period_scored"]:
        raise ValueError("Complete immutable Saudi primary evaluation is required")
    for path, digest in record["output_sha256"].items():
        if sha(directory / path) != digest:
            raise ValueError(f"Primary reference changed: {path}")
    for path, digest in record["code_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Primary source changed: {path}")
    return sha(directory / "run_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/donor_comparisons_gpu_v1")
    parser.add_argument("--device", choices=["cuda:0"], default="cuda:0")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    check_lock()
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("The common-device donor sensitivity requires CUDA; no CPU substitution")
    benchmark_path = ROOT / "work/parallel-benchmark/report.json"
    benchmark = json.loads(benchmark_path.read_text())
    workers = args.workers if args.workers is not None else benchmark["recommendation"]["gpu_workers"]
    if not 1 <= workers <= 8:
        raise ValueError("Use one through eight benchmarked GPU workers")
    test_path = ROOT / "work/donor-comparison-validation/tests.json"
    tests = json.loads(test_path.read_text())
    if not tests["passed"]:
        raise ValueError("Passing current donor-comparison tests are required")
    for path, digest in tests["tested_code_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Source changed after tests: {path}")
    required = ["src/gbd_park/donors.py", "scripts/run_donor_comparisons.py", "tests/test_donor_comparisons.py",
                "study_design/donor_implementation.md", "study_design/donor_comparisons_implementation.md"]
    if not set(required).issubset(tests["tested_code_sha256"]):
        raise ValueError("Tests must cover the donor runner and numerical specifications")
    reference_hash = _reference_manifest()
    code = dict(tests["tested_code_sha256"])
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "torch": torch.__version__, "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "scipy": scipy.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}
    source_hash = sha(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    out = ROOT / args.output
    started = time.perf_counter()
    if out.exists():
        if not args.resume:
            raise FileExistsError("Use --resume to continue this exact documented experiment")
        manifest = json.loads((out / "run_manifest.json").read_text())
        if (manifest["code_sha256"] != code or manifest["config_sha256"] != sha(ROOT / "study_design/locked_v1/design.json")
                or manifest["primary_reference_sha256"] != reference_hash or manifest["device"] != args.device
                or manifest["source_sha256"] != source_hash or manifest["versions"] != versions):
            raise ValueError("Resume requires unchanged source, design, primary reference, and device")
        if manifest["status"] == "complete":
            for path, digest in manifest["output_sha256"].items():
                if sha(out / path) != digest:
                    raise ValueError("Completed donor run changed")
            print("Donor comparison is already complete and verified.", flush=True)
            return
        events = json.loads((out / "events.json").read_text()) if (out / "events.json").exists() else []
    else:
        out.mkdir(parents=True)
        for name in ["cache", "checkpoints", "banks"]:
            (out / name).mkdir()
        manifest = {"status": "running", "created_utc": now(), "run_id": out.name,
                    "code_sha256": code, "config_sha256": sha(ROOT / "study_design/locked_v1/design.json"),
                    "primary_reference_sha256": reference_hash, "device": args.device, "workers": workers,
                    "benchmark_sha256": sha(benchmark_path), "test_report_sha256": sha(test_path),
                    "source_sha256": source_hash, "source_lock_sha256": sha(ROOT / "study_design/locked_v1/lock_manifest.json"),
                    "versions": versions,
                    "role": "prespecified_secondary_GPU_donor_sensitivity_with_fresh_GPU_all_six_reference",
                    "CPU_primary_result_replaced": False, "final_period_scored": False}
        (out / "tests_at_run.json").write_bytes(test_path.read_bytes())
        (out / "benchmark_at_run.json").write_bytes(benchmark_path.read_bytes())
        events = []
        _atomic_json(out / "run_manifest.json", manifest)

    def event(name, **details):
        if any(row["event"] == name for row in events):
            return
        events.append({"event": name, "time_utc": now(), **deepcopy(details)})
        _atomic_json(out / "events.json", events)

    payloads = {}
    fit_jobs = {}

    def execute(jobs, phase):
        unique = {job["job_id"]: job for job in jobs}
        fit_jobs.update(unique)
        pending = []
        for key, job in unique.items():
            if key in payloads:
                continue
            cached = load_cached_job(out, job)
            if cached is None:
                pending.append(job)
            else:
                payloads[key] = cached
        event(phase + "_started", unique_jobs=len(unique))
        print(f"{phase}: {len(pending)} fits remaining; {len(unique)-len(pending)} cached", flush=True)
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(cached_fit_job, config, job, out): job for job in pending}
                for index, future in enumerate(as_completed(futures), 1):
                    payload = future.result()
                    payloads[payload["cache_job"]["job_id"]] = payload
                    if index % 4 == 0 or index == len(pending):
                        print(f"{phase}: {index}/{len(pending)} new fits; {time.perf_counter()-started:.1f}s", flush=True)
        event(phase + "_complete", unique_jobs=len(unique))

    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel_for_planning = panel.loc[panel.year.le(2018)].copy()
    del panel
    definitions = arms(config)
    origins = list(range(2003, 2019))
    plans, audit = {}, []
    for origin in origins:
        for arm in definitions:
            for sex in config["sexes"]:
                decision = select_donors(panel_for_planning, config, origin, config["primary_target"], sex,
                                         arm["strategy"], seed=arm["seed"])
                decision.update(arm=arm["arm"], device=args.device)
                plans[(origin, arm["arm"], sex)] = decision
                audit.append(decision)
    write_json(out / "donor_decisions.json", audit)
    event("donor_plans_frozen", decisions=len(audit), sha256=sha(out / "donor_decisions.json"))
    candidates = [fit_job_description(origin, plans[(origin, arm["arm"], sex)]["countries"], base, 11, args.device)
                  for origin in range(2003, 2014) for arm in definitions for sex in config["sexes"] for base in base_grid(config)]
    execute(candidates, "candidate_fits")
    candidate_rows, candidate_cal = [], []
    for origin in range(2003, 2014):
        for arm in definitions:
            for base in base_grid(config):
                rows, corrections = compose_forecasts(config, plans, payloads, origin, arm["arm"], base, [11],
                                                      config["adaptation"]["penalties"], False)
                candidate_rows.extend(rows)
                candidate_cal.extend(corrections)
    order = ["arm", "origin", "family", "setting_id", "sex", "age", "horizon"]
    candidate = pd.DataFrame(candidate_rows).sort_values(order).reset_index(drop=True)
    if len(candidate) != len(definitions) * 48400:
        raise ValueError("Incomplete donor candidate forecast grid")
    write_frame(out / "candidate_predictions.csv", candidate)
    event("candidate_predictions_committed", sha256=sha(out / "candidate_predictions.csv"))
    truth_history = panel_for_planning.loc[panel_for_planning.location_name.eq(config["primary_target"])
                                         & panel_for_planning.outcome.eq("prevalence")].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    candidate_scores = score_arms(candidate, truth_history, config, 2018)
    write_frame(out / "candidate_scores.csv", candidate_scores)
    choices = [{"arm": arm["arm"], **select_tcn_settings(candidate_scores.loc[candidate_scores.arm.eq(arm["arm"])], config, origin)}
               for arm in definitions for origin in origins]
    write_json(out / "settings_decisions.json", choices)
    event("outer_settings_frozen", choices=len(choices), sha256=sha(out / "settings_decisions.json"))
    extras = [fit_job_description(choice["fit_origin"], plans[(choice["fit_origin"], choice["arm"], sex)]["countries"],
                                  choice["base"], seed, args.device)
              for choice in choices for sex in config["sexes"] for seed in config["models"]["tcn"]["ensemble_seeds"]]
    execute(extras, "reported_ensemble_fits")
    rows, corrections = [], []
    for choice in choices:
        predicted, calibrated = compose_forecasts(config, plans, payloads, choice["fit_origin"], choice["arm"],
                                                  choice["base"], config["models"]["tcn"]["ensemble_seeds"],
                                                  choice["penalties"], True)
        rows.extend(predicted)
        corrections.extend(calibrated)
    all_points = pd.DataFrame(rows).sort_values(order).reset_index(drop=True)
    validate_reported_grid(all_points, config, definitions, origins)
    if len(all_points) != len(definitions) * 16 * 330:
        raise ValueError("Incomplete five-seed donor forecast grid")
    historical = all_points.loc[all_points.origin.le(2013)].copy()
    evaluation = all_points.loc[all_points.origin.ge(2014)].copy()
    write_frame(out / "historical_predictions.csv", historical)
    write_frame(out / "predictions.csv", evaluation)
    write_frame(out / "adaptation_audit.csv", pd.DataFrame(candidate_cal + corrections).sort_values(["arm", "origin", "sex", "setting_id"]))
    write_json(out / "fit_jobs.json", sorted(fit_jobs.values(), key=lambda item: item["job_id"]))
    write_json(out / "fit_audit.json", [{"job_id": key, **payloads[key]["audit"]} for key in sorted(payloads)])
    event("reported_predictions_committed", sha256=sha(out / "predictions.csv"))
    historical_scores = score_arms(historical, truth_history, config, 2018)
    write_frame(out / "historical_scores.csv", historical_scores)
    interval_tables, draw_tables, bank_audit = [], [], []
    for arm in definitions:
        scores = historical_scores.loc[historical_scores.arm.eq(arm["arm"])]
        for origin in config["calendar"]["reliability_origins"]:
            for family in ["tcn_adapted", "tcn_unadapted", "tcn_intercept"]:
                bank = build_residual_bank(scores, config, origin, family)
                path = out / "banks" / f"{arm['arm']}__{origin}__{family}.joblib"
                if not path.exists():
                    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
                    joblib.dump(bank, temporary, compress=3)
                    temporary.replace(path)
                elif joblib.hash(joblib.load(path)) != joblib.hash(bank):
                    raise ValueError("Resumed residual bank differs from its committed contents")
                points = evaluation.loc[evaluation.arm.eq(arm["arm"]) & evaluation.origin.eq(origin) & evaluation.family.eq(family)]
                intervals, draws = apply_bank(points, bank, config)
                intervals["arm"], draws["arm"] = arm["arm"], arm["arm"]
                interval_tables.append(intervals)
                draw_tables.append(draws)
                bank_audit.append({"arm": arm["arm"], "origin": origin, "family": family,
                                   "n_blocks": bank["n_blocks"], "sha256": sha(path)})
    intervals, draws = pd.concat(interval_tables, ignore_index=True), pd.concat(draw_tables, ignore_index=True)
    write_frame(out / "intervals.csv", intervals)
    write_frame(out / "joint_draws.csv", draws)
    write_frame(out / "bank_references.csv", pd.DataFrame(bank_audit))
    committed = {name: sha(out / name) for name in ["predictions.csv", "intervals.csv", "joint_draws.csv",
                                                   "settings_decisions.json", "donor_decisions.json"]}
    write_json(out / "pre_score_commit.json", committed)
    event("intervals_committed", hashes=committed)
    if any(sha(out / name) != digest for name, digest in committed.items()):
        raise ValueError("Forecast or interval ledger changed before final verification")
    event("final_scoring_started", maximum_year=2023)
    full_panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = full_panel.loc[full_panel.location_name.eq(config["primary_target"]) & full_panel.outcome.eq("prevalence")].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    scored = score_arms(evaluation, truth, config, 2023)
    write_frame(out / "point_scores.csv", scored)
    cell_tables, wis_tables = [], []
    for _, part in intervals.groupby("arm", sort=True):
        cells, wis = score_intervals(part, truth, config, 2023)
        wis["arm"] = part.arm.iloc[0]
        cell_tables.append(cells)
        wis_tables.append(wis)
    write_frame(out / "interval_scores.csv", pd.concat(cell_tables, ignore_index=True))
    write_frame(out / "wis_scores.csv", pd.concat(wis_tables, ignore_index=True))
    summary = scored.groupby(["arm", "origin", "sex", "family", "horizon"], as_index=False).agg(
        mean_age_absolute_log_error=("absolute_log_error", "mean"), rate_mae=("absolute_rate_error", "mean"),
        fallback_cells=("status", lambda values: int(values.eq("fallback").sum())))
    write_frame(out / "summary_by_origin.csv", summary)
    primary_reference = pd.read_csv(ROOT / "results/primary_v1/point_scores.csv")
    comparators = primary_reference.loc[primary_reference.family.isin(["local_champion", "nonneural_champion"])]
    contrasts, harm_cells, harm_summaries, verdicts = [], [], [], []
    for arm, part in scored.groupby("arm", sort=True):
        contrast, _, verdict = primary_contrasts(pd.concat([part, comparators], ignore_index=True), config)
        contrast["arm"] = arm
        contrasts.append(contrast)
        verdict.update(arm=arm, role="secondary_donor_sensitivity", CPU_primary_result_replaced=False)
        verdicts.append(verdict)
        cells, summary = matched_tcn_harm(part, config)
        cells["arm"], summary["arm"] = arm, arm
        harm_cells.append(cells)
        harm_summaries.append(summary)
    write_frame(out / "endpoint_contrasts.csv", pd.concat(contrasts, ignore_index=True))
    write_frame(out / "matched_adaptation_cells.csv", pd.concat(harm_cells, ignore_index=True))
    write_frame(out / "matched_adaptation_summary.csv", pd.concat(harm_summaries, ignore_index=True))
    write_json(out / "endpoint_verdicts.json", verdicts)
    keys = ["origin", "sex", "age", "horizon", "family"]
    reference = scored.loc[scored.arm.eq("all_six"), keys + ["absolute_log_error"]].rename(
        columns={"absolute_log_error": "all_six_gpu_error"})
    donor_harm = scored.loc[~scored.arm.eq("all_six")].merge(reference, on=keys, validate="many_to_one")
    donor_harm["donor_loss_change"] = donor_harm.absolute_log_error - donor_harm.all_six_gpu_error
    donor_harm["harmed"] = donor_harm.donor_loss_change > 0
    write_frame(out / "donor_harm_cells.csv", donor_harm)
    write_frame(out / "donor_harm_summary.csv", donor_harm.groupby(
        ["arm", "origin", "sex", "family", "horizon"], as_index=False).agg(
            mean_loss_change=("donor_loss_change", "mean"), harm_fraction=("harmed", "mean")))
    comparator_keys = ["origin", "sex", "age", "horizon"]
    comparison = comparators[comparator_keys + ["family", "source_family", "absolute_log_error"]].rename(
        columns={"family": "comparator", "source_family": "comparator_source_family",
                 "absolute_log_error": "comparator_error"})
    borrowing = scored.loc[scored.family.eq("tcn_adapted")].merge(comparison, on=comparator_keys, validate="many_to_many")
    if len(borrowing) != len(definitions) * 550 * 2:
        raise ValueError("Incomplete fixed-comparator borrowing-harm grid")
    borrowing["borrowing_loss_change"] = borrowing.absolute_log_error - borrowing.comparator_error
    borrowing["harmed"] = borrowing.borrowing_loss_change > 0
    write_frame(out / "borrowing_harm_cells.csv", borrowing)
    cpu = primary_reference.loc[primary_reference.family.isin(["tcn_adapted", "tcn_unadapted", "tcn_intercept"]),
                                keys + ["prediction", "absolute_log_error"]].rename(
        columns={"prediction": "cpu_primary_prediction", "absolute_log_error": "cpu_primary_error"})
    device_reference = scored.loc[scored.arm.eq("all_six")].merge(cpu, on=keys, validate="one_to_one")
    device_reference["device_procedure_loss_change"] = device_reference.absolute_log_error - device_reference.cpu_primary_error
    write_frame(out / "gpu_reference_vs_cpu_primary.csv", device_reference)
    for path, digest in code.items():
        if sha(ROOT / path) != digest:
            raise ValueError("Source changed during donor experiment")
    if _reference_manifest() != reference_hash:
        raise ValueError("CPU primary reference changed")
    check_lock()
    validation = {"passed": True, "unit_tests": tests["tests_run"], "arms": len(definitions),
                  "unique_source_fits": len(payloads), "source_failures": sum(p["audit"]["status"] != "ok" for p in payloads.values()),
                  "candidate_forecasts": len(candidate), "historical_forecasts": len(historical),
                  "evaluation_forecasts": len(evaluation), "interval_rows": len(intervals), "joint_draw_rows": len(draws),
                  "five_seed_ensembles": True, "shared_sex_base_settings": True, "historical_pool_selection": True,
                  "device": args.device, "workers": workers, "CPU_primary_result_replaced": False,
                  "final_period_scored": True, "elapsed_seconds_this_invocation": time.perf_counter()-started}
    _atomic_json(out / "validation_report.json", validation)
    event("complete", unique_fits=len(payloads))
    manifest.update(status="complete", completed_utc=now(), final_period_scored=True,
                    output_sha256={str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*"))
                                   if path.is_file() and path.name != "run_manifest.json"})
    _atomic_json(out / "run_manifest.json", manifest)
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
