"""Bounded, post-result interval calibration using immutable point forecasts."""
import os
for _name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
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
from gbd_park.calibration import development_losses, scale_bank, select_factor
from gbd_park.demography import joint_count_draws
from gbd_park.intervals import apply_bank, score_intervals
from run_demography import STAT, burden_statistics, ratio_statistics, distribution_summary, score_distributions
from run_local_baselines import check_lock, now, sha

AMENDMENT = ROOT / "study_design/interval_calibration_v1_2.json"
LOCK = ROOT / "study_design/interval_calibration_v1_2.lock.json"
TESTS = ROOT / "work/calibration-validation/tests.json"
NUMERIC_TOLERANCE = dict(rtol=2e-12, atol=1e-10)


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cases(config, amendment):
    result = []
    for country in config["countries"]:
        if country["name"] not in amendment["countries"]:
            continue
        for outcome in amendment["outcomes"]:
            identifier = country["iso3"] + "_" + outcome
            primary = identifier == "SAU_prevalence"
            source = "results/primary_v1" if primary else "results/secondary_v1/trials/" + identifier
            history = "results/intervals_v1" if primary else source
            result.append(dict(id=identifier, target=country["name"], outcome=outcome,
                               source=source, history=history, banks=history,
                               demography="results/demography_v1/" + identifier))
    assert len(result) == 12
    return result


def require_source(frame, case):
    if set(frame.target) != {case["target"]} or set(frame.outcome) != {case["outcome"]}:
        raise ValueError("Wrong target/outcome source ledger")


def underlying_family(mappings, role, sex, origin):
    if role == "tcn_adapted":
        return role
    rows = mappings.loc[mappings.fit_origin.eq(origin) & mappings.role.eq(role) & mappings.sex.eq(sex)]
    if len(rows) != 1 or rows.source_family.isna().any():
        raise ValueError("Need one frozen underlying family per current origin/role/sex")
    if "last_selection_target_year" in rows and not rows.last_selection_target_year.le(origin).all():
        raise ValueError("Family mapping uses a future selection label")
    return rows.source_family.iloc[0]


def labels(frame):
    result = frame.copy()
    parts = result.family.str.split("__", n=1, expand=True)
    if parts.shape[1] != 2 or parts.isna().any().any():
        raise ValueError("Calibration family must encode role and variant")
    result["role"], result["variant"] = parts[0], parts[1]
    return result


def match_numeric(new, old, keys, fields):
    if new.duplicated(keys).any() or old.duplicated(keys).any():
        raise ValueError("Duplicate comparison keys")
    a, b = new.set_index(keys).sort_index(), old.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Reference and new ledger keys disagree")
    np.testing.assert_allclose(a[fields].to_numpy(float), b[fields].to_numpy(float), **NUMERIC_TOLERANCE)
    return float(np.max(np.abs(a[fields].to_numpy(float) - b[fields].to_numpy(float))))


def prepare_development(case, config, amendment):
    history = read_csv(ROOT / case["history"] / "prequential_scores.csv")
    require_source(history, case)
    return development_losses(history, config, amendment)


def issue_case(case, config, amendment, loss_path, output):
    directory = Path(output) / case["id"]
    directory.mkdir()
    source, demographic_source = ROOT / case["source"], ROOT / case["demography"]
    all_points = read_csv(source / "predictions.csv")
    points = all_points.loc[all_points.family.isin(amendment["roles"])].copy()
    require_source(points, case)
    assert len(points) == 1650
    points.to_csv(directory / "original_point_predictions.csv", index=False)
    mappings = read_csv(source / "champion_family_mappings.csv")
    losses = read_csv(loss_path)
    populations = read_csv(demographic_source / "population_forecasts.csv")
    population_errors = read_csv(demographic_source / "population_residuals.csv")
    require_source(populations, case)
    require_source(population_errors, case)
    old_burden_points = read_csv(demographic_source / "predictions.csv")
    old_burden_points = old_burden_points[old_burden_points.family.isin(amendment["roles"])].copy()
    burden_points = []
    for variant in amendment["main_variants"]:
        part = old_burden_points.copy()
        part["family"] = part.family + "__" + variant
        burden_points.append(part)
    burden_points = labels(pd.concat(burden_points, ignore_index=True))

    interval_frames, draw_frames, derived_frames, decisions, bank_refs = [], [], [], [], []
    for origin in amendment["evaluation_origins"]:
        for role in amendment["roles"]:
            bank_path = ROOT / case["banks"] / "banks" / f"origin{origin}__{role}.joblib"
            bank = joblib.load(bank_path)
            fingerprint = joblib.hash(bank)
            if bank["target"] != case["target"] or bank["outcome"] != case["outcome"]:
                raise ValueError("Wrong residual bank target/outcome")
            if bank["origins"] != list(range(2003, origin - 4)):
                raise ValueError("Residual bank has wrong complete historical origins")
            current = points.loc[points.origin.eq(origin) & points.family.eq(role)].copy()
            by_sex = {sex: underlying_family(mappings, role, sex, origin) for sex in config["sexes"]}
            if bank.get("source_family_by_sex", by_sex) != by_sex:
                raise ValueError("Bank and current frozen champion mapping disagree")
            if "source_family" in current:
                for sex in config["sexes"]:
                    if not current.loc[current.sex.eq(sex), "source_family"].eq(by_sex[sex]).all():
                        raise ValueError("Current point family differs from frozen mapping")
            factors = {variant: dict.fromkeys(config["sexes"], factor)
                       for variant, factor in zip(amendment["fixed_variants"], amendment["factor_grid"])}
            for policy in ["target_only", "gcc_assisted"]:
                factors[policy] = {}
                for sex in config["sexes"]:
                    decision = select_factor(losses, amendment, case["target"], case["outcome"],
                                             by_sex[sex], sex, origin, policy)
                    decision["role"], decision["case_id"] = role, case["id"]
                    decisions.append(decision)
                    factors[policy][sex] = decision["factor"]
            bank_refs.append(dict(origin=origin, role=role, path=str(bank_path.relative_to(ROOT)),
                                  sha256=sha(bank_path), source_family_by_sex=by_sex, n_blocks=bank["n_blocks"]))
            for variant, scales in factors.items():
                family = role + "__" + variant
                calibrated = scale_bank(bank, scales, config)
                calibrated["family"] = family
                current_variant = current.copy()
                current_variant["family"] = family
                intervals, draws = apply_bank(current_variant, calibrated, config)
                for frame in [intervals, draws]:
                    frame["role"], frame["variant"] = role, variant
                    frame["factor"] = frame.sex.map(scales)
                    frame["source_family"] = frame.sex.map(by_sex)
                interval_frames.append(intervals)
                draw_frames.append(draws)
                if variant in amendment["main_variants"]:
                    derived_frames.append(ratio_statistics(draws, config, True))
                    for method in amendment["population_methods"]:
                        pop = populations[populations.origin.eq(origin) & populations.population_method.eq(method)]
                        errors = population_errors[population_errors.fit_origin.eq(origin)
                                                   & population_errors.population_method.eq(method)]
                        joint = joint_count_draws(draws, pop, errors)
                        derived_frames.append(burden_statistics(joint, config, True))
            assert joblib.hash(bank) == fingerprint

    intervals = pd.concat(interval_frames, ignore_index=True)
    draws = pd.concat(draw_frames, ignore_index=True)
    derived = labels(pd.concat(derived_frames, ignore_index=True))
    burden_intervals = labels(distribution_summary(derived))
    original = intervals[intervals.variant.eq("original")].copy()
    original["family"] = original.role
    old_intervals = read_csv(source / "intervals.csv")
    old_intervals = old_intervals[old_intervals.family.isin(amendment["roles"])]
    rate_diff = match_numeric(original, old_intervals,
                              ["target", "outcome", "origin", "family", "sex", "age", "horizon", "scale", "level"],
                              ["lower", "median", "upper", "point_prediction"])
    original = burden_intervals[burden_intervals.variant.eq("original")].copy()
    original["family"] = original.role
    old_intervals = read_csv(demographic_source / "intervals.csv")
    old_intervals = old_intervals[old_intervals.family.isin(amendment["roles"])]
    burden_diff = match_numeric(original, old_intervals, STAT + ["level"], ["lower", "median", "upper"])
    for variant in amendment["main_variants"]:
        current = burden_points[burden_points.variant.eq(variant)].copy()
        current["family"] = current.role
        assert match_numeric(current, old_burden_points, STAT, ["value"]) == 0
    intervals.to_csv(directory / "rate_intervals.csv", index=False)
    draws.to_csv(directory / "rate_draws.csv.gz", index=False, compression="gzip")
    derived.to_csv(directory / "burden_draws.csv.gz", index=False, compression="gzip")
    burden_points.to_csv(directory / "burden_predictions.csv", index=False)
    burden_intervals.to_csv(directory / "burden_intervals.csv", index=False)
    write_json(directory / "factor_choices.json", decisions)
    write_json(directory / "bank_references.json", bank_refs)
    source_refs = {str((source / name).relative_to(ROOT)): sha(source / name)
                   for name in ["predictions.csv", "intervals.csv", "champion_family_mappings.csv"]}
    source_refs.update({str((demographic_source / name).relative_to(ROOT)): sha(demographic_source / name)
                        for name in ["predictions.csv", "intervals.csv", "population_forecasts.csv", "population_residuals.csv"]})
    write_json(directory / "source_references.json", source_refs)
    checks = dict(passed=True, case=case["id"], original_point_rows=len(points),
                  rate_interval_rows=len(intervals), rate_draw_rows=len(draws),
                  burden_interval_rows=len(burden_intervals), burden_draw_rows=len(derived),
                  burden_point_rows=len(burden_points), factor_choices=len(decisions),
                  original_rate_max_difference=rate_diff, original_burden_max_difference=burden_diff,
                  original_points_unchanged=True, population_forecasts_unchanged=True)
    write_json(directory / "issuance_validation.json", checks)
    hashes = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.is_file()}
    write_json(directory / "issued_commit.json", dict(committed_utc=now(), final_period_scored=False,
                                                       artifact_sha256=hashes))
    return checks


def rate_summaries(frame, wis, config):
    summaries, wis_summaries = [], []
    scopes = {"45+": config["ages"]}
    scopes.update({name: [a for a in config["ages"] if int(a.split("-")[0].rstrip("+")) in starts]
                   for name, starts in config["age_groups"].items()})
    scopes.update({"age:" + age: [age] for age in config["ages"]})
    group = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "scale"]
    for scope, ages in scopes.items():
        part = frame[frame.age.isin(ages)].copy()
        part["lower_miss"] = part.observed_value.lt(part.lower)
        part["upper_miss"] = part.observed_value.gt(part.upper)
        result = part.groupby(group + ["level"], as_index=False).agg(
            coverage=("covered", "mean"), mean_width=("width", "mean"),
            lower_miss_rate=("lower_miss", "mean"), upper_miss_rate=("upper_miss", "mean"),
            mean_interval_score=("interval_score", "mean"), age_cells=("age", "size"))
        result["age_scope"] = scope
        summaries.append(result)
        result = wis[wis.age.isin(ages)].groupby(group, as_index=False).agg(
            mean_wis_50_80=("wis_50_80", "mean"), mean_wis_50_80_95=("wis_50_80_95", "mean"))
        result["age_scope"] = scope
        wis_summaries.append(result)
    return pd.concat(summaries, ignore_index=True), pd.concat(wis_summaries, ignore_index=True)


def score_case(case, config, amendment, output):
    directory = Path(output) / case["id"]
    global_commit = json.loads((Path(output) / "global_issued_commit.json").read_text())
    assert len(global_commit["case_commit_sha256"]) == 12
    for name, digest in global_commit["case_commit_sha256"].items():
        assert sha(Path(output) / name) == digest
    own_commit = json.loads((directory / "issued_commit.json").read_text())
    assert all(sha(directory / name) == digest for name, digest in own_commit["artifact_sha256"].items())
    write_json(directory / "scoring_event.json", dict(scoring_started_utc=now(),
                                                       global_commit_sha256=sha(Path(output) / "global_issued_commit.json")))
    # All twelve cases have committed their distributions before this truth read.
    panel = read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    truth = panel.loc[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    cells, wis = score_intervals(read_csv(directory / "rate_intervals.csv"), truth, config, 2023)
    cells, wis = labels(cells), labels(wis)
    cells.to_csv(directory / "rate_interval_scores.csv.gz", index=False, compression="gzip")
    wis.to_csv(directory / "rate_wis_scores.csv", index=False)
    summary, wis_summary = rate_summaries(cells, wis, config)
    summary.to_csv(directory / "rate_summary.csv", index=False)
    wis_summary.to_csv(directory / "rate_wis_summary.csv", index=False)
    original_burden_truth = read_csv(ROOT / case["demography"] / "point_scores.csv")
    original_burden_truth = original_burden_truth[original_burden_truth.family.isin(amendment["roles"])]
    burden_truth = []
    for variant in amendment["main_variants"]:
        part = original_burden_truth[STAT + ["observed"]].copy()
        part["family"] = part.family + "__" + variant
        burden_truth.append(part)
    burden_truth = pd.concat(burden_truth, ignore_index=True)
    burden_cells, burden_wis = score_distributions(read_csv(directory / "burden_intervals.csv"), burden_truth)
    burden_cells, burden_wis = labels(burden_cells), labels(burden_wis)
    burden_cells["lower_miss"] = burden_cells.observed.lt(burden_cells.lower)
    burden_cells["upper_miss"] = burden_cells.observed.gt(burden_cells.upper)
    burden_cells.to_csv(directory / "burden_interval_scores.csv.gz", index=False, compression="gzip")
    burden_wis.to_csv(directory / "burden_wis_scores.csv", index=False)
    points = read_csv(directory / "burden_predictions.csv").merge(burden_truth, on=STAT, validate="one_to_one")
    points["absolute_error"] = abs(points.value - points.observed)
    points.to_csv(directory / "burden_point_scores.csv", index=False)
    assert all(sha(directory / name) == digest for name, digest in own_commit["artifact_sha256"].items())
    validation = dict(passed=True, case=case["id"], rate_score_rows=len(cells), rate_wis_rows=len(wis),
                      burden_score_rows=len(burden_cells), burden_wis_rows=len(burden_wis),
                      forecast_commit_unchanged=True, scoring_complete_utc=now())
    write_json(directory / "validation_report.json", validation)
    return validation


def verify_prior(amendment, lock):
    result = {}
    for run in amendment["prior_runs"]:
        directory = ROOT / "results" / run
        manifest = json.loads((directory / "run_manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("A required original run is incomplete")
        for name, expected in manifest["output_sha256"].items():
            assert sha(directory / name) == expected, (run, name)
        for name, expected in manifest["code_sha256"].items():
            assert sha(ROOT / name) == expected, name
        result[run] = sha(directory / "run_manifest.json")
    assert result == lock["prior_manifest_sha256"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/calibration_v1_2")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    amendment, lock = json.loads(AMENDMENT.read_text()), json.loads(LOCK.read_text())
    if not 1 <= args.workers <= amendment["max_workers"]:
        raise ValueError("Invalid CPU worker count")
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Existing calibration results cannot be overwritten")
    check_lock()
    for name, expected in lock["document_sha256"].items():
        assert sha(ROOT / name) == expected, "Exploratory protocol changed after lock"
    tests = json.loads(TESTS.read_text())
    assert tests["passed"] and tests["implementation_review_passed"]
    for name, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    required = {"src/gbd_park/calibration.py", "scripts/run_calibration.py", "tests/test_calibration.py"}
    assert required.issubset(tests["tested_code_sha256"])
    print("Verifying immutable prior outputs and source code...", flush=True)
    priors = verify_prior(amendment, lock)
    started = time.perf_counter()
    out.mkdir(parents=True)
    write_json(out / "tests_at_run.json", tests)
    tasks = cases(config, amendment)
    write_json(out / "cases.json", tasks)
    manifest = dict(status="running", created_utc=now(), run_id=out.name,
                    role=amendment["status"], prior_manifest_sha256=priors,
                    amendment_lock_sha256=sha(LOCK), code_sha256=tests["tested_code_sha256"],
                    workers=args.workers, numerical_threads_per_worker=1,
                    versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                  scipy=scipy.__version__, joblib=joblib.__version__),
                    original_primary_preserved=True, point_models_refitted=False,
                    population_refitted=False, final_period_scored=False)
    write_json(out / "run_manifest.json", manifest)
    events = []

    def event(name, **kwargs):
        events.append(dict(event=name, time_utc=now(), **kwargs))
        write_json(out / "events.json", events)

    context = multiprocessing.get_context("spawn")
    event("development_calibration_started", maximum_development_label_year=2018)
    development = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(prepare_development, task, config, amendment): task for task in tasks}
        for future in as_completed(futures):
            development.append(future.result())
            print("Development losses complete: " + futures[future]["id"], flush=True)
    losses = pd.concat(development, ignore_index=True)
    losses.to_csv(out / "development_losses.csv", index=False)
    event("development_losses_committed", sha256=sha(out / "development_losses.csv"), rows=len(losses))
    issued = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(issue_case, task, config, amendment, out / "development_losses.csv", out): task for task in tasks}
        for future in as_completed(futures):
            issued.append(future.result())
            print("Calibrated forecasts committed: " + futures[future]["id"], flush=True)
    commits = {task["id"] + "/issued_commit.json": sha(out / task["id"] / "issued_commit.json") for task in tasks}
    write_json(out / "global_issued_commit.json", dict(committed_utc=now(), case_commit_sha256=commits,
                                                       development_losses_sha256=sha(out / "development_losses.csv")))
    event("all_cases_committed_before_scoring", cases=len(tasks), sha256=sha(out / "global_issued_commit.json"))
    event("evaluation_scoring_started")
    validation = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(score_case, task, config, amendment, out): task for task in tasks}
        for future in as_completed(futures):
            validation.append(future.result())
            print("Calibration evaluation complete: " + futures[future]["id"], flush=True)
    assert verify_prior(amendment, lock) == priors
    for name, expected in tests["tested_code_sha256"].items():
        assert sha(ROOT / name) == expected
    write_json(out / "validation_report.json", dict(passed=True, cases=validation, issuance=issued,
               original_primary_preserved=True, all_cases_committed_before_scoring=True,
               elapsed_seconds=time.perf_counter() - started))
    event("complete")
    manifest.update(status="complete", completed_utc=now(), final_period_scored=True,
                    output_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                   if p.is_file() and p.name != "run_manifest.json"})
    write_json(out / "run_manifest.json", manifest)
    print("All 12 exploratory calibration cases complete; original primary unchanged.", flush=True)


if __name__ == "__main__":
    main()
