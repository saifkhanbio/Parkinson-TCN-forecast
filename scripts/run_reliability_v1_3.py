"""Bounded exploratory age/horizon calibration and joint dynamic forecasting."""
import os
for _name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
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
from gbd_park.demography import hierarchy, joint_count_draws
from gbd_park.intervals import apply_bank, interval_score, weighted_interval_score
from run_calibration import cases, underlying_family, labels, match_numeric, rate_summaries
from run_demography import STAT, burden_statistics, ratio_statistics
from run_local_baselines import check_lock, now, sha

SPEC = ROOT / "study_design/reliability_v1_3.json"
LOCK = ROOT / "study_design/reliability_v1_3.lock.json"
TESTS = ROOT / "work/reliability-validation/tests.json"
RATE_KEYS = ["target", "outcome", "origin", "family", "sex", "age", "horizon", "forecast_year", "scale"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def tensor(frame, config, value, draw=False):
    """Canonical axes: [draw,] horizon, sex/age; reject missing coordinates."""
    columns = ["horizon", "sex", "age"]
    axes = [config["calendar"]["horizons"], config["sexes"], config["ages"]]
    if draw:
        columns.insert(0, "residual_origin")
        axes.insert(0, sorted(frame.residual_origin.unique()))
    index = pd.MultiIndex.from_product(axes, names=columns)
    if frame.duplicated(columns).any() or len(frame) != len(index):
        raise ValueError("Missing or duplicate tensor coordinates")
    result = frame.set_index(columns).reindex(index)[value].to_numpy(float)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Nonpositive or missing tensor values")
    shape = ([len(axes[0])] if draw else []) + [len(config["calendar"]["horizons"]), len(config["sexes"])*len(config["ages"])]
    return result.reshape(shape)


def rate_ledgers(arrays, config, context, n_blocks, kind):
    point, draws = arrays["point_rates"], arrays["rate_draws"]
    coords = pd.MultiIndex.from_product([config["calendar"]["horizons"], config["sexes"], config["ages"]],
                                       names=["horizon", "sex", "age"]).to_frame(index=False)
    if draws.shape[1:] != point.shape or point.shape != (5, 22):
        raise ValueError("Unexpected full-grid rate array shape")
    if not np.isfinite(draws).all() or not np.isfinite(point).all() or (draws <= 0).any() or (point <= 0).any():
        raise ValueError("Invalid rate draws or points")
    points = coords.copy()
    for key, value in context.items():
        points[key] = value
    points["forecast_year"] = points.origin + points.horizon
    points["prediction"], points["log_prediction"] = point.ravel(), np.log(point).ravel()
    points["status"] = "ok"
    intervals = []
    for scale, values, point_values in [("rate", draws, point), ("log_rate", np.log(draws), np.log(point))]:
        for level in config["intervals"]["central_levels"]:
            bounds = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear")
            part = points.drop(columns=["prediction", "log_prediction"]).copy()
            part["scale"], part["level"] = scale, level
            part["lower"], part["median"], part["upper"] = [x.ravel() for x in bounds]
            part["point_prediction"] = point_values.ravel()
            part["n_blocks"], part["n_draws"], part["draw_kind"] = n_blocks, len(draws), kind
            intervals.append(part)
    return points, pd.concat(intervals, ignore_index=True)


def derived_values(rates, counts, config, ratios=False):
    """Vectorized exact transformations of each complete joint trajectory."""
    if ratios:
        n_age = len(config["ages"])
        male = config["sexes"].index("Male")*n_age
        female = config["sexes"].index("Female")*n_age
        values = rates[..., male:male+n_age] / rates[..., female:female+n_age]
        metadata = pd.DataFrame([dict(measure="sex_rate_ratio", node="Male_Female__"+age,
            sex="Male/Female", age_group=age, unit="rate_ratio", hierarchy_level="age_sex_ratio") for age in config["ages"]])
        return values, metadata
    matrix, nodes, bottom = hierarchy(config)
    values = [np.einsum("...c,kc->...k", counts, matrix)]
    metadata = nodes.rename(columns={"level": "hierarchy_level"}).copy()
    metadata["measure"], metadata["unit"] = "count", "modeled_number"
    extra = []
    for sex in config["sexes"]:
        selector = np.array([s == sex for s, age in bottom])
        total = counts[..., selector].sum(axis=-1)
        for threshold in [65, 80]:
            selected = np.array([s == sex and int(age.split("-")[0].rstrip("+")) >= threshold for s, age in bottom])
            values.append((100*counts[..., selected].sum(axis=-1)/total)[..., None])
            age_group = f"{threshold}+_within_45+"
            extra.append(dict(measure="age_share", node=sex+"__"+age_group, sex=sex,
                              age_group=age_group, unit="percent", hierarchy_level="sex_total"))
    return np.concatenate(values, axis=-1), pd.concat([metadata, pd.DataFrame(extra)], ignore_index=True)


def burden_ledgers(arrays, methods, config, context, n_blocks, kind):
    intervals, points = [], []
    for method in methods + ["not_applicable"]:
        ratios = method == "not_applicable"
        values, metadata = derived_values(arrays["rate_draws"], None if ratios else arrays["count_draws__"+method], config, ratios)
        point_values, _ = derived_values(arrays["point_rates"], None if ratios else arrays["point_counts__"+method], config, ratios)
        rows = []
        for horizon in config["calendar"]["horizons"]:
            part = metadata.copy()
            part["horizon"] = horizon
            rows.append(part)
        base = pd.concat(rows, ignore_index=True)
        for key, value in context.items():
            base[key] = value
        base["forecast_year"] = base.origin + base.horizon
        base["population_method"] = method
        point = base.copy()
        point["value"] = point_values.ravel()
        points.append(point)
        for level in config["intervals"]["central_levels"]:
            bounds = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear")
            part = base.copy()
            part["level"] = level
            part["lower"], part["median"], part["upper"] = [x.ravel() for x in bounds]
            part["n_blocks"], part["n_draws"], part["draw_kind"] = n_blocks, len(values), kind
            intervals.append(part)
    return pd.concat(points, ignore_index=True), pd.concat(intervals, ignore_index=True)


def completed_history(prequential, point_source, panel, case, origin):
    """No unavailable verification enters a marginal calibration fitting frame."""
    fields = ["target", "outcome", "origin", "family", "sex", "age", "horizon", "forecast_year", "prediction", "log_prediction", "observed_rate"]
    old = prequential.loc[prequential.origin.lt(origin) & prequential.forecast_year.le(origin), fields].copy()
    current = point_source.loc[point_source.origin.lt(origin) & point_source.forecast_year.le(origin)
                               & ~point_source.family.isin(["local_champion", "nonneural_champion"])].copy()
    truth = panel.loc[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])
                      & panel.year.le(origin), ["sex", "age", "year", "rate"]].rename(columns={"year": "forecast_year", "rate": "observed_rate"})
    current = current.merge(truth, on=["sex", "age", "forecast_year"], how="left", validate="many_to_one")
    result = pd.concat([old, current[fields]], ignore_index=True)
    if result.duplicated(["origin", "family", "sex", "age", "horizon"]).any():
        raise ValueError("Overlapping historical forecast sources")
    if not np.isfinite(result.observed_rate).all() or not result.forecast_year.le(origin).all():
        raise ValueError("Unavailable historical calibration label")
    return result


def score_bounds(intervals, truth, keys, observed_name):
    """Common numerical scoring without equating Monte Carlo draws with blocks."""
    if intervals.duplicated(keys+["level"]).any() or truth.duplicated(keys).any():
        raise ValueError("Duplicate interval or truth coordinates")
    cells = intervals.merge(truth[keys+[observed_name]], on=keys, how="left", validate="many_to_one")
    values = cells[["lower", "median", "upper", observed_name]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
        raise ValueError("Nonfinite verification or unordered interval")
    cells["covered"] = (cells[observed_name] >= cells.lower) & (cells[observed_name] <= cells.upper)
    cells["width"] = cells.upper-cells.lower
    cells["lower_miss"] = cells[observed_name] < cells.lower
    cells["upper_miss"] = cells[observed_name] > cells.upper
    cells["interval_score"] = interval_score(cells[observed_name], cells.lower, cells.upper, 1-cells.level)
    levels = [.5, .8, .95]
    lower = cells.pivot(index=keys, columns="level", values="lower")[levels]
    upper = cells.pivot(index=keys, columns="level", values="upper")[levels]
    base = cells.drop_duplicates(keys).set_index(keys).reindex(lower.index)
    medians = cells.pivot(index=keys, columns="level", values="median")[levels]
    np.testing.assert_allclose(medians, np.broadcast_to(base["median"].to_numpy()[:, None], medians.shape))
    wis = base[[observed_name, "median", "n_blocks", "n_draws", "draw_kind"]].copy()
    wis["wis_50_80"] = weighted_interval_score(base[observed_name], base["median"], lower.iloc[:, :2], upper.iloc[:, :2], levels[:2])
    wis["wis_50_80_95"] = weighted_interval_score(base[observed_name], base["median"], lower, upper, levels)
    return cells, wis.reset_index()


def issue_case(case, config, spec, output):
    from gbd_park.structured_calibration import structured_bank
    from gbd_park.dynamic_joint import fit_dynamic_joint
    directory = Path(output)/case["id"]
    directory.mkdir()
    (directory/"draws").mkdir()
    source, prior = ROOT/case["source"], ROOT/"results/calibration_v1_2"/case["id"]
    demographic = ROOT/case["demography"]
    all_points = read(source/"predictions.csv")
    mappings = read(source/"champion_family_mappings.csv")
    past_scores = read(ROOT/case["history"]/"prequential_scores.csv")
    panel = read(ROOT/"data/processed/design_v1/regional_outcomes.csv")
    # Filter before any model receives numeric values; full panel truth is never
    # passed to the dynamic model or a calibration module.
    panel = panel.loc[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])
                      & panel.year.le(max(spec["evaluation_origins"]))].copy()
    populations = read(demographic/"population_forecasts.csv")
    population_errors = read(demographic/"population_residuals.csv")
    original_burden_points = read(demographic/"predictions.csv")
    rate_intervals = read(prior/"rate_intervals.csv")
    rate_intervals = rate_intervals[rate_intervals.variant.isin(spec["reference_variants"])].copy()
    burden_intervals = read(prior/"burden_intervals.csv")
    burden_points = read(prior/"burden_predictions.csv")
    for frame in [burden_intervals, burden_points]:
        if set(frame.variant) != set(spec["reference_variants"]):
            raise ValueError("Unexpected reference procedures")
    point_frames = []
    originals = all_points[all_points.family.isin(spec["roles"])].copy()
    originals.to_csv(directory/"original_point_predictions.csv", index=False)
    for variant in spec["reference_variants"]:
        points = originals.copy()
        points["family"] = points.family+"__"+variant
        point_frames.append(points)
    for frame in [rate_intervals, burden_intervals]:
        frame["n_draws"], frame["draw_kind"] = frame.n_blocks, "historical_joint_blocks"
    rate_frames, burden_frames, burden_point_frames = [rate_intervals], [burden_intervals], [burden_points]
    decisions, draw_index, diagnostics, source_references = [], [], [], {}
    for origin in spec["evaluation_origins"]:
        eligible = completed_history(past_scores, all_points, panel, case, origin)
        historical = panel.loc[panel.year.le(origin)].copy()
        historical = historical[["location_name", "outcome", "year", "sex", "age", "rate", "count", "implied_population"]]
        for role in spec["roles"]:
            original_bank_path = ROOT/case["banks"]/"banks"/f"origin{origin}__{role}.joblib"
            bank = joblib.load(original_bank_path)
            fingerprint = joblib.hash(bank)
            families = {sex: underlying_family(mappings, role, sex, origin) for sex in config["sexes"]}
            transformed, details = structured_bank(eligible, bank, config, families)
            family = role+"__structured"
            transformed["family"] = family
            current = originals.loc[originals.origin.eq(origin) & originals.family.eq(role)].copy()
            current["family"] = family
            intervals, draws = apply_bank(current, transformed, config)
            arrays = dict(point_rates=tensor(current, config, "prediction"), rate_draws=tensor(draws, config, "rate_draw", True))
            for method in spec["population_methods"]:
                pop = populations[populations.origin.eq(origin) & populations.population_method.eq(method)]
                errors = population_errors[population_errors.fit_origin.eq(origin) & population_errors.population_method.eq(method)]
                joint = joint_count_draws(draws, pop, errors)
                arrays["point_populations__"+method] = tensor(pop, config, "population")
                arrays["point_counts__"+method] = arrays["point_rates"]*arrays["point_populations__"+method]/100000
                arrays["population_draws__"+method] = tensor(joint, config, "population_draw", True)
                arrays["count_draws__"+method] = tensor(joint, config, "count", True)
            context = dict(target=case["target"], outcome=case["outcome"], origin=origin, family=family)
            new_points, new_intervals = rate_ledgers(arrays, config, context, bank["n_blocks"], "transformed_historical_joint_blocks")
            status_keys = ["sex", "age", "horizon"]
            status = current[status_keys+["status"]].copy().rename(columns={"status": "point_status"})
            status["point_fallback_reason"] = current["fallback_reason"].fillna("").to_numpy() if "fallback_reason" in current else ""
            new_points = new_points.merge(status, on=status_keys, validate="one_to_one")
            new_points["status"] = new_points.point_status
            new_points["fallback_reason"] = new_points.point_fallback_reason
            new_intervals = new_intervals.merge(status, on=status_keys, validate="many_to_one")
            match_numeric(new_intervals, intervals, RATE_KEYS+["level"], ["lower", "median", "upper", "point_prediction"])
            b_points, b_intervals = burden_ledgers(arrays, spec["population_methods"], config, context, bank["n_blocks"], "transformed_historical_joint_blocks")
            old_b = original_burden_points[original_burden_points.origin.eq(origin) & original_burden_points.family.eq(role)].copy()
            old_b["family"] = family
            match_numeric(b_points, old_b, STAT, ["value"])
            match_numeric(new_points, current, RATE_KEYS[:-1], ["prediction", "log_prediction"])
            point_frames.append(new_points)
            rate_frames.append(new_intervals)
            burden_point_frames.append(b_points)
            burden_frames.append(b_intervals)
            path = f"draws/origin{origin}__{family}.npz"
            np.savez_compressed(directory/path, **arrays)
            details = details.copy()
            details["role"], details["origin"] = role, origin
            decisions.append(details)
            draw_index.append(dict(path=path, family=family, origin=origin, kind="structured",
                                   n_blocks=bank["n_blocks"], n_draws=bank["n_blocks"],
                                   residual_origins=bank["origins"], population_methods=spec["population_methods"],
                                   original_bank=str(original_bank_path.relative_to(ROOT)), source_family_by_sex=families))
            source_references[str(original_bank_path.relative_to(ROOT))] = sha(original_bank_path)
            assert joblib.hash(bank) == fingerprint
        seed = spec["seed"] + 1000*case["case_index"] + origin
        model = fit_dynamic_joint(historical, config, origin, case["target"], case["outcome"], seed=seed, draws=spec["dynamic_draws"])
        method, family = "dynamic_joint", "robust_dynamic__joint"
        arrays = {name: model[name] for name in ["point_rates", "rate_draws"]}
        for name in ["point_counts", "point_populations", "count_draws", "population_draws"]:
            arrays[name+"__"+method] = model[name]
        np.testing.assert_allclose(arrays["point_rates"]*model["point_populations"]/100000, model["point_counts"], rtol=1e-12)
        np.testing.assert_allclose(arrays["rate_draws"]*model["population_draws"]/100000, model["count_draws"], rtol=1e-12)
        context = dict(target=case["target"], outcome=case["outcome"], origin=origin, family=family)
        new_points, new_intervals = rate_ledgers(arrays, config, context, 0, "model_monte_carlo")
        b_points, b_intervals = burden_ledgers(arrays, [method], config, context, 0, "model_monte_carlo")
        point_frames.append(new_points)
        rate_frames.append(new_intervals)
        burden_point_frames.append(b_points)
        burden_frames.append(b_intervals)
        path = f"draws/origin{origin}__{family}.npz"
        np.savez_compressed(directory/path, **arrays)
        model_path = f"draws/origin{origin}__dynamic_state.joblib"
        joblib.dump(model, directory/model_path, compress=3)
        assert model["diagnostics"].get("origin", origin) == origin
        assert model["diagnostics"].get("seed", seed) == seed
        diagnostics.append({**model["diagnostics"], "origin": origin, "seed": seed})
        draw_index.append(dict(path=path, family=family, origin=origin, kind="dynamic_joint",
                               n_blocks=0, n_draws=spec["dynamic_draws"], population_methods=[method],
                               fitted_state=model_path, seed=seed, training_years=sorted(historical.year.unique().tolist())))
    for name, parts in [("rate_predictions.csv", point_frames), ("rate_intervals.csv", rate_frames),
                        ("burden_predictions.csv", burden_point_frames), ("burden_intervals.csv", burden_frames)]:
        frame = labels(pd.concat(parts, ignore_index=True))
        assert frame.family.nunique() == 13
        frame.to_csv(directory/name, index=False)
    pd.concat(decisions, ignore_index=True).to_csv(directory/"structured_parameters.csv", index=False)
    write_json(directory/"dynamic_diagnostics.json", diagnostics)
    write_json(directory/"draw_index.json", draw_index)
    for path in [source/"predictions.csv", source/"champion_family_mappings.csv", ROOT/case["history"]/"prequential_scores.csv",
                 demographic/"population_forecasts.csv", demographic/"population_residuals.csv", demographic/"predictions.csv",
                 prior/"rate_intervals.csv", prior/"burden_intervals.csv", prior/"burden_predictions.csv",
                 ROOT/"data/processed/design_v1/regional_outcomes.csv"]:
        source_references[str(path.relative_to(ROOT))] = sha(path)
    write_json(directory/"source_references.json", source_references)
    validation = dict(passed=True, case=case["id"], procedures=13, original_points_unchanged=True,
                      structured_population_unchanged=True, dynamic_model_separate=True, issued_draw_files=len(draw_index))
    write_json(directory/"issuance_validation.json", validation)
    hashes = {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    write_json(directory/"issued_commit.json", dict(committed_utc=now(), final_period_scored=False, artifact_sha256=hashes))
    return validation


def score_case(case, config, spec, output):
    out, directory = Path(output), Path(output)/case["id"]
    global_commit = json.loads((out/"global_issued_commit.json").read_text())
    assert len(global_commit["case_commit_sha256"]) == 12
    for name, digest in global_commit["case_commit_sha256"].items():
        assert sha(out/name) == digest
    issued = json.loads((directory/"issued_commit.json").read_text())
    assert all(sha(directory/name) == digest for name, digest in issued["artifact_sha256"].items())
    write_json(directory/"scoring_event.json", dict(scoring_started_utc=now(), global_commit_sha256=sha(out/"global_issued_commit.json")))
    panel = read(ROOT/"data/processed/design_v1/regional_outcomes.csv")
    actual = panel.loc[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate"})
    points, intervals = read(directory/"rate_predictions.csv"), read(directory/"rate_intervals.csv")
    truth_keys = ["target", "outcome", "sex", "age", "forecast_year"]
    truth = intervals[RATE_KEYS].drop_duplicates().merge(actual[truth_keys+["observed_rate"]], on=truth_keys, validate="many_to_one")
    truth["observed_value"] = np.where(truth.scale.eq("rate"), truth.observed_rate, np.log(truth.observed_rate))
    cells, wis = score_bounds(intervals, truth, RATE_KEYS, "observed_value")
    cells, wis = labels(cells), labels(wis)
    cells.to_csv(directory/"rate_interval_scores.csv.gz", index=False, compression="gzip")
    wis.to_csv(directory/"rate_wis_scores.csv", index=False)
    summary, ws = rate_summaries(cells, wis, config)
    summary.to_csv(directory/"rate_summary.csv", index=False)
    ws.to_csv(directory/"rate_wis_summary.csv", index=False)
    scored_points = points.merge(actual[truth_keys+["observed_rate", "count"]], on=truth_keys, validate="many_to_one")
    scored_points["absolute_log_error"] = abs(scored_points.log_prediction-np.log(scored_points.observed_rate))
    scored_points["signed_log_error"] = scored_points.log_prediction-np.log(scored_points.observed_rate)
    scored_points.to_csv(directory/"rate_point_scores.csv", index=False)
    verified = scored_points.copy()
    verified["prediction"] = verified.observed_rate
    truths = [ratio_statistics(verified, config)]
    for family, part in verified.groupby("family"):
        methods = ["dynamic_joint"] if family == "robust_dynamic__joint" else spec["population_methods"]
        for method in methods:
            part = part.copy()
            part["population_method"] = method
            truths.append(burden_statistics(part, config))
    burden_truth = pd.concat(truths, ignore_index=True)[STAT+["value"]].rename(columns={"value": "observed"})
    b_cells, b_wis = score_bounds(read(directory/"burden_intervals.csv"), burden_truth, STAT, "observed")
    labels(b_cells).to_csv(directory/"burden_interval_scores.csv.gz", index=False, compression="gzip")
    labels(b_wis).to_csv(directory/"burden_wis_scores.csv", index=False)
    b_points = read(directory/"burden_predictions.csv").merge(burden_truth, on=STAT, validate="one_to_one")
    b_points["absolute_error"] = abs(b_points.value-b_points.observed)
    b_points["absolute_log_error"] = abs(np.log(b_points.value/b_points.observed))
    b_points.to_csv(directory/"burden_point_scores.csv", index=False)
    # Evaluate population assumptions separately, without treating them as the
    # source of any burden difference established by this bundled comparison.
    population_scores = []
    bottom = pd.MultiIndex.from_product([config["sexes"], config["ages"]], names=["sex", "age"])
    for item in json.loads((directory/"draw_index.json").read_text()):
        if item["kind"] != "dynamic_joint":
            continue
        data = np.load(directory/item["path"])
        for h in config["calendar"]["horizons"]:
            obs = actual[actual.forecast_year.eq(item["origin"]+h)].set_index(["sex", "age"]).reindex(bottom)
            pop = (obs["count"]/obs.observed_rate*100000).to_numpy()
            draws = data["population_draws__dynamic_joint"][:, h-1]
            bounds = np.quantile(draws, [.025, .1, .25, .5, .75, .9, .975], axis=0)
            for j, (sex, age) in enumerate(bottom):
                prediction = data["point_populations__dynamic_joint"][h-1, j]
                row = dict(target=case["target"], outcome=case["outcome"], origin=item["origin"], horizon=h,
                           sex=sex, age=age, prediction=prediction, observed=pop[j],
                           absolute_log_error=float(abs(np.log(prediction/pop[j]))))
                row["coverage_80"] = bool(bounds[1, j] <= pop[j] <= bounds[5, j])
                row["width_80"] = float(bounds[5, j]-bounds[1, j])
                population_scores.append(row)
    pd.DataFrame(population_scores).to_csv(directory/"dynamic_population_scores.csv", index=False)
    assert all(sha(directory/name) == digest for name, digest in issued["artifact_sha256"].items())
    validation = dict(passed=True, case=case["id"], procedures=int(points.family.nunique()),
                      rate_interval_rows=len(cells), burden_interval_rows=len(b_cells),
                      point_rows=len(points), forecast_commit_unchanged=True, scoring_complete_utc=now())
    write_json(directory/"validation_report.json", validation)
    return validation


def verify_sources(spec, lock):
    check_lock()
    for name, expected in lock["document_sha256"].items():
        assert sha(ROOT/name) == expected, name
    for run in spec["prior_runs"]:
        folder = ROOT/"results"/run
        assert sha(folder/"run_manifest.json") == lock["prior_manifest_sha256"][run]
        manifest = json.loads((folder/"run_manifest.json").read_text())
        assert manifest["status"] == "complete"
        for name, expected in manifest["output_sha256"].items():
            assert sha(folder/name) == expected, (run, name)
        for name, expected in manifest["code_sha256"].items():
            assert sha(ROOT/name) == expected, name


def main():
    from gbd_park.structured_calibration import SETTINGS as STRUCTURED_SETTINGS
    from gbd_park.dynamic_joint import SETTINGS as DYNAMIC_SETTINGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/reliability_v1_3")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    spec, lock, tests = [json.loads(p.read_text()) for p in [SPEC, LOCK, TESTS]]
    config = json.loads((ROOT/"study_design/locked_v1/design.json").read_text())
    assert 1 <= args.workers <= 12
    assert spec["structured_settings"] == STRUCTURED_SETTINGS
    assert spec["dynamic_settings"] == DYNAMIC_SETTINGS
    assert spec["ages"] == config["ages"] and spec["horizons"] == config["calendar"]["horizons"]
    assert spec["dynamic_draws"] == DYNAMIC_SETTINGS["draws"]
    assert tests["passed"] and tests["implementation_review_passed"]
    assert all(sha(ROOT/name) == digest for name, digest in tests["tested_code_sha256"].items())
    verify_sources(spec, lock)
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tasks = cases(config, spec)
    for i, task in enumerate(tasks):
        task["case_index"] = i
    write_json(out/"cases.json", tasks)
    write_json(out/"tests_at_run.json", tests)
    manifest = dict(status="running", run_id=out.name, created_utc=now(), workers=args.workers,
                    numerical_threads_per_worker=1, original_primary_preserved=True,
                    original_models_refitted=False, new_dynamic_models=True, final_period_scored=False,
                    amendment_lock_sha256=sha(LOCK), code_sha256=tests["tested_code_sha256"],
                    prior_manifest_sha256=lock["prior_manifest_sha256"], role=spec["status"],
                    versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                  scipy=scipy.__version__, joblib=joblib.__version__))
    write_json(out/"run_manifest.json", manifest)
    events = []
    def event(name, **kwargs):
        events.append(dict(event=name, time_utc=now(), **kwargs))
        write_json(out/"events.json", events)
    event("issuance_started")
    context = multiprocessing.get_context("spawn")
    issuance = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(issue_case, task, config, spec, out): task for task in tasks}
        for future in as_completed(futures):
            issuance.append(future.result())
            print("Issued and committed: "+futures[future]["id"], flush=True)
    commits = {task["id"]+"/issued_commit.json": sha(out/task["id"]/"issued_commit.json") for task in tasks}
    write_json(out/"global_issued_commit.json", dict(committed_utc=now(), case_commit_sha256=commits))
    event("all_cases_committed_before_scoring", cases=len(tasks), sha256=sha(out/"global_issued_commit.json"))
    event("evaluation_scoring_started")
    validations = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(score_case, task, config, spec, out): task for task in tasks}
        for future in as_completed(futures):
            validations.append(future.result())
            print("Scored: "+futures[future]["id"], flush=True)
    verify_sources(spec, lock)
    assert all(sha(ROOT/name) == digest for name, digest in tests["tested_code_sha256"].items())
    write_json(out/"validation_report.json", dict(passed=True, cases=validations, issuance=issuance,
               original_primary_preserved=True, all_cases_committed_before_scoring=True,
               elapsed_seconds=time.perf_counter()-started))
    event("complete")
    manifest.update(status="complete", final_period_scored=True, completed_utc=now(),
        output_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run_manifest.json"})
    write_json(out/"run_manifest.json", manifest)
    print("All 12 bounded reliability cases completed.", flush=True)


if __name__ == "__main__":
    main()
