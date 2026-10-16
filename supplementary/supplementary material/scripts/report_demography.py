"""Independent audit and reporting of operational burden and native-count studies."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_local_baselines import sha, check_lock

RUN = ROOT / "results/demography_v1"
COUNT = ROOT / "results/count_coherence_v1"
OUT = ROOT / "reports/demography_v1"
CONTEXT = ["target", "outcome", "origin", "family", "horizon", "forecast_year", "population_method"]
STAT = CONTEXT + ["measure", "node", "sex", "age_group", "unit"]
ROLES = ["tcn_adapted", "local_champion", "nonneural_champion"]
LABELS = {"tcn_adapted": "Adapted TCN", "local_champion": "Local champion", "nonneural_champion": "Non-neural champion"}
COLORS = {"observed": "#222222", "tcn_adapted": "#1479a6", "local_champion": "#7d8997", "nonneural_champion": "#cb7a29"}
METHODS = ["log_trend_last8", "persistence"]


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def verify_run(directory):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["output_sha256"].items():
        assert sha(directory / name) == expected, str(directory / name)
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    assert "tests/test_demography_runner.py" in manifest["code_sha256"]
    return manifest


def verify_events(directory, commit_name, start="scoring_started"):
    events = json.loads((directory / "events.json").read_text())
    names = [row["event"] for row in events]
    order = [commit_name, start, "scoring_complete"]
    assert all(names.count(name) == 1 for name in order)
    indices = [names.index(name) for name in order]
    assert indices == sorted(indices)
    times = [datetime.fromisoformat(events[index]["time_utc"]) for index in indices]
    assert times == sorted(times)
    commits = events[indices[0]]["sha256"]
    for name, expected in commits.items():
        assert sha(directory / name) == expected, name
    if (directory / "pre_score_commit.json").exists():
        before = json.loads((directory / "pre_score_commit.json").read_text())
        assert before["sha256"] == commits
        assert datetime.fromisoformat(before["time_utc"]) <= times[1]


def node_definitions(config):
    bottom = [(sex, age) for sex in config["sexes"] for age in config["ages"]]
    definitions = []
    for sex, age in bottom:
        definitions.append((f"{sex}__{age}", sex, age, "age_sex", [(sex, age)]))
    for sex in config["sexes"]:
        for group, starts in config["age_groups"].items():
            definitions.append((f"{sex}__{group}", sex, group, "broad_age_sex",
                                [(sex, age) for age in config["ages"] if int(age.split("-")[0].rstrip("+")) in starts]))
    definitions.extend((f"{sex}__45+", sex, "45+", "sex_total", [(sex, age) for age in config["ages"]]) for sex in config["sexes"])
    definitions.append(("Both__45+", "Both", "45+", "grand_total", bottom))
    matrix = np.asarray([[float(pair in members) for pair in bottom] for _, _, _, _, members in definitions])
    assert matrix.shape == (31, 22)
    return bottom, definitions, matrix


def expected_statistics(cells, rate_rows, config, draw=False):
    """Vectorized independent sums/shares, preserving each original block ID."""
    bottom, definitions, matrix = node_definitions(config)
    ids = CONTEXT + (["residual_origin"] if draw else [])
    pivot = cells.pivot(index=ids, columns=["sex", "age"], values="count").reindex(columns=pd.MultiIndex.from_tuples(bottom))
    assert np.isfinite(pivot.to_numpy()).all()
    summed = pivot.to_numpy() @ matrix.T
    frames = []
    for index, (node, sex, group, level, _) in enumerate(definitions):
        part = pivot.index.to_frame(index=False)
        part["measure"], part["node"], part["sex"], part["age_group"], part["unit"] = "count", node, sex, group, "modeled_number"
        part["value"], part["hierarchy_level"] = summed[:, index], level
        frames.append(part)
    for sex in config["sexes"]:
        denominator = pivot.loc[:, sex].sum(axis=1).to_numpy()
        for threshold in [65, 80]:
            ages = [age for age in config["ages"] if int(age.split("-")[0].rstrip("+")) >= threshold]
            numerator = pivot.loc[:, [(sex, age) for age in ages]].sum(axis=1).to_numpy()
            part = pivot.index.to_frame(index=False)
            group = f"{threshold}+_within_45+"
            part["measure"], part["node"], part["sex"], part["age_group"], part["unit"] = "age_share", f"{sex}__{group}", sex, group, "percent"
            part["value"], part["hierarchy_level"] = 100*numerator/denominator, "sex_total"
            frames.append(part)
    ratio_ids = [name for name in CONTEXT if name != "population_method"] + ["age"] + (["residual_origin"] if draw else [])
    ratios = rate_rows.pivot(index=ratio_ids, columns="sex", values="rate_draw" if draw else "prediction")
    part = ratios.index.to_frame(index=False).rename(columns={"age": "age_group"})
    part["value"] = (ratios["Male"]/ratios["Female"]).to_numpy()
    part["population_method"], part["measure"], part["unit"], part["sex"] = "not_applicable", "sex_rate_ratio", "rate_ratio", "Male/Female"
    part["node"], part["hierarchy_level"] = "Male_Female__" + part.age_group, "age_sex_ratio"
    frames.append(part)
    return pd.concat(frames, ignore_index=True)[STAT + (["residual_origin"] if draw else []) + ["value", "hierarchy_level"]]


def independent_population(panel, config, target, outcome, origin, method):
    selected = panel[panel.location_name.eq(target) & panel.outcome.eq(outcome) & panel.year.between(origin-7, origin)]
    grid = selected.pivot(index=["sex", "age"], columns="year", values="count") / selected.pivot(index=["sex", "age"], columns="year", values="rate") * 100000
    grid = grid.reindex(index=pd.MultiIndex.from_product([config["sexes"], config["ages"]]), columns=range(origin-7, origin+1))
    assert grid.shape == (22, 8) and np.isfinite(grid.to_numpy()).all()
    logs = np.log(grid.to_numpy())
    time = np.arange(-7, 1, dtype=float)
    slopes = ((time-time.mean())[None, :] * (logs-logs.mean(axis=1)[:, None])).sum(axis=1) / np.sum((time-time.mean())**2)
    if method == "persistence":
        slopes, intercepts = np.zeros(22), logs[:, -1]
    else:
        intercepts = logs.mean(axis=1) - slopes*time.mean()
    frame = grid.index.to_frame(index=False)
    frame.columns = ["sex", "age"]
    records = []
    for horizon in range(1, 6):
        current = frame.copy()
        current["horizon"] = horizon
        current["log_population"] = intercepts + slopes*horizon
        current["population"] = np.exp(current.log_population)
        records.append(current)
    return pd.concat(records, ignore_index=True)


def audit_populations(directory, panel, config, target, outcome):
    populations, errors = read(directory / "population_forecasts.csv"), read(directory / "population_residuals.csv")
    source = panel[panel.location_name.eq(target) & panel.outcome.eq(outcome)].copy()
    source["population"] = source["count"]/source.rate*100000
    truth = source.set_index(["year", "sex", "age"]).population
    for (origin, method), current in populations.groupby(["origin", "population_method"]):
        expected = independent_population(panel, config, target, outcome, origin, method).set_index(["sex", "age", "horizon"]).sort_index()
        actual = current.set_index(["sex", "age", "horizon"]).sort_index()
        np.testing.assert_allclose(actual[["log_population", "population"]], expected[["log_population", "population"]], atol=1e-12, rtol=1e-12)
        assert actual.last_population_input_year.eq(origin).all()
        selected = errors[errors.fit_origin.eq(origin) & errors.population_method.eq(method)]
        assert set(selected.residual_origin) == set(range(2003, origin-4))
        for residual_origin, block in selected.groupby("residual_origin"):
            historical = independent_population(panel, config, target, outcome, residual_origin, method).set_index(["sex", "age", "horizon"])
            block = block.set_index(["sex", "age", "horizon"]).reindex(historical.index)
            observed = np.asarray([truth.loc[(residual_origin+h, s, a)] for s, a, h in historical.index])
            np.testing.assert_allclose(block.raw_population_log_error, np.log(observed)-historical.log_population, atol=1e-12, rtol=1e-12)
            assert (block.forecast_year <= origin).all()
        center = selected.groupby(["sex", "age", "horizon"]).raw_population_log_error.transform("mean")
        np.testing.assert_allclose(selected.mean_population_log_error, center, atol=1e-13)
        np.testing.assert_allclose(selected.centered_population_log_error, selected.raw_population_log_error-center, atol=1e-13)
    return populations, errors


def audit_intervals(directory, statistics):
    intervals, scored, wis = read(directory / "intervals.csv"), read(directory / "interval_scores.csv.gz"), read(directory / "wis_scores.csv")
    quantile_frames, count_frames = [], []
    probabilities = [.025, .1, .25, .5, .75, .9, .975]
    for origin, part in statistics.groupby("origin"):
        block = part.pivot(index="residual_origin", columns=STAT, values="value")
        assert set(block.index) == set(range(2003, origin-4)) and np.isfinite(block.to_numpy()).all()
        values = np.quantile(block.to_numpy(), probabilities, axis=0, method="linear").T
        quantile_frames.append(pd.DataFrame(values, index=block.columns, columns=probabilities))
        count_frames.append(pd.Series(len(block), index=block.columns))
    quantiles, count = pd.concat(quantile_frames), pd.concat(count_frames)
    for level, lower, upper in [(.5, .25, .75), (.8, .1, .9), (.95, .025, .975)]:
        part = intervals[intervals.level.eq(level)].set_index(STAT).reindex(quantiles.index)
        np.testing.assert_allclose(part[["lower", "median", "upper"]], quantiles[[lower, .5, upper]], atol=1e-10, rtol=1e-12)
        np.testing.assert_array_equal(part.n_blocks, count)
    observed = scored.observed.to_numpy()
    lower, upper, alpha = scored.lower.to_numpy(), scored.upper.to_numpy(), 1-scored.level.to_numpy()
    width = upper-lower
    score = width + 2/alpha*(np.maximum(lower-observed, 0)+np.maximum(observed-upper, 0))
    np.testing.assert_allclose(scored.width, width, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(scored.interval_score, score, atol=1e-9, rtol=1e-12)
    np.testing.assert_array_equal(scored.covered, (observed >= lower) & (observed <= upper))
    weighted = scored[STAT + ["level"]].copy()
    weighted["score"] = alpha*score/2
    weighted = weighted.pivot(index=STAT, columns="level", values="score")
    index = wis.set_index(STAT).reindex(weighted.index)
    for field, levels in [("wis_50_80", [.5, .8]), ("wis_50_80_95", [.5, .8, .95])]:
        expected = (.5*abs(index.observed-index["median"]) + weighted[levels].sum(axis=1))/(len(levels)+.5)
        np.testing.assert_allclose(index[field], expected, atol=1e-9, rtol=1e-12)
    point_truth = read(directory / "point_scores.csv").set_index(STAT).observed
    np.testing.assert_array_equal(index.observed, point_truth.reindex(index.index))
    return intervals, scored, wis


def shapley_independent(n0, r0, n1, r1):
    old, new = [n0.sum(), n0/n0.sum(), r0], [n1.sum(), n1/n1.sum(), r1]
    def value(state):
        return state[0] * np.sum(state[1] * state[2]) / 100000
    result = np.zeros(3)
    for order in itertools.permutations(range(3)):
        state = list(old)
        before = value(state)
        for index in order:
            state[index] = new[index]
            after = value(state)
            result[index] += (after-before)/6
            before = after
    return np.r_[value(new)-value(old), result]


def audit_case(case, config):
    directory = RUN / case["id"]
    verify_events(directory, "forecasts_committed")
    source = ROOT / case["source"]
    source_root = ROOT / "results" / ("primary_v1" if case["id"] == "SAU_prevalence" else "secondary_v1")
    manifest = json.loads((source_root / "run_manifest.json").read_text())
    for name in ["predictions.csv", "joint_draws.csv"]:
        assert sha(source / name) == manifest["output_sha256"][str((source/name).relative_to(source_root))]
    points, rate_draws = read(source / "predictions.csv"), read(source / "joint_draws.csv")
    families = set(config["models"]["local_order"] + config["models"]["nonneural_order"]
                   + ["tcn_unadapted", "tcn_intercept", "tcn_adapted", "local_champion", "nonneural_champion"])
    assert set(points.family) == families and set(rate_draws.family) == families
    assert set(points.origin) == set(range(2014, 2019)) and len(points) == 8800 and len(rate_draws) == 79200
    panel = read(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    populations, errors = audit_populations(directory, panel, config, case["target"], case["outcome"])
    keys = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]
    cells = points.merge(populations[keys + ["population_method", "population"]], on=keys, how="left")
    assert len(cells) == 2*len(points)
    cells["count"] = cells.prediction*cells.population/100000
    expected_points = expected_statistics(cells, points, config).set_index(STAT).sort_index()
    saved_points = read(directory / "predictions.csv").set_index(STAT).sort_index()
    assert len(saved_points) == 32400
    assert expected_points.index.equals(saved_points.index)
    np.testing.assert_allclose(saved_points.value, expected_points.value, atol=1e-10, rtol=1e-12)
    joint = rate_draws.merge(populations[keys + ["population_method", "log_population"]], on=keys)
    assert len(joint) == 2*len(rate_draws)
    error_keys = ["target", "outcome", "sex", "age", "horizon", "residual_origin", "population_method"]
    joint = joint.merge(errors[["fit_origin"] + error_keys + ["centered_population_log_error"]].rename(columns={"fit_origin": "origin"}),
                        on=["origin"]+error_keys, how="left", validate="many_to_one")
    assert np.isfinite(joint.centered_population_log_error).all()
    joint["count"] = np.exp(joint.log_draw)*np.exp(joint.log_population+joint.centered_population_log_error)/100000
    expected_draws = expected_statistics(joint, rate_draws, config, True).set_index(STAT+["residual_origin"]).sort_index()
    saved_draws = read(directory / "joint_statistics.csv.gz").set_index(STAT+["residual_origin"]).sort_index()
    assert len(saved_draws) == 291600
    assert expected_draws.index.equals(saved_draws.index)
    np.testing.assert_allclose(saved_draws.value, expected_draws.value, atol=1e-9, rtol=1e-12)
    intervals, interval_scores, wis = audit_intervals(directory, saved_draws.reset_index())
    actual = panel[panel.location_name.eq(case["target"]) & panel.outcome.eq(case["outcome"])].rename(
        columns={"location_name": "target", "year": "forecast_year", "rate": "observed_rate", "count": "observed_count"})
    truth_keys = ["target", "outcome", "sex", "age", "forecast_year"]
    observed_rates = points.merge(actual[truth_keys+["observed_rate", "observed_count"]], on=truth_keys, validate="many_to_one")
    observed_rates["prediction"] = observed_rates.observed_rate
    observed_cells = pd.concat([observed_rates.assign(population_method=method, count=observed_rates.observed_count) for method in METHODS])
    expected_truth = expected_statistics(observed_cells, observed_rates, config).set_index(STAT).sort_index()
    scores = read(directory / "point_scores.csv").set_index(STAT).sort_index()
    np.testing.assert_allclose(scores.observed, expected_truth.value, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(scores.value, saved_points.value, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(scores.absolute_error, abs(scores.value-scores.observed), atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(scores.absolute_log_error, abs(np.log(scores.value)-np.log(scores.observed)), atol=1e-12, rtol=1e-12)
    decomposition = read(directory / "accounting_decomposition_2018_2023.csv")
    selected_truth = actual.set_index(["forecast_year", "sex", "age"]).sort_index()
    for row in decomposition.itertuples():
        start = selected_truth.loc[(2018, row.sex)].reindex(config["ages"])
        n0, r0 = (start.observed_count/start.observed_rate*100000).to_numpy(), start.observed_rate.to_numpy()
        if row.family == "observed_accounting":
            end = selected_truth.loc[(2023, row.sex)].reindex(config["ages"])
            n1, r1 = (end.observed_count/end.observed_rate*100000).to_numpy(), end.observed_rate.to_numpy()
        else:
            end = points[points.origin.eq(2018) & points.horizon.eq(5) & points.sex.eq(row.sex) & points.family.eq(row.family)].set_index("age").reindex(config["ages"])
            pop = populations[populations.origin.eq(2018) & populations.horizon.eq(5) & populations.sex.eq(row.sex)
                              & populations.population_method.eq(row.population_method)].set_index("age").reindex(config["ages"])
            n1, r1 = pop.population.to_numpy(), end.prediction.to_numpy()
        expected = shapley_independent(n0, r0, n1, r1)
        np.testing.assert_allclose([row.total_change, row.population_size, row.age_composition, row.rate_component], expected, atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(row.total_change, row.population_size+row.age_composition+row.rate_component, atol=1e-9, rtol=1e-12)
    validation = {"case": case["id"], "passed": True, "point_statistics_reconstructed": len(scores),
                  "joint_statistics_reconstructed": len(saved_draws), "interval_quantiles_reconstructed": len(intervals),
                  "wis_rows_recalculated": len(wis), "shapley_rows_recalculated": len(decomposition),
                  "population_forecasts_reconstructed": len(populations), "population_residuals_reconstructed": len(errors)}
    scores = scores.reset_index()
    groups = ["target", "outcome", "origin", "family", "horizon", "population_method", "measure", "sex", "age_group", "hierarchy_level"]
    summary = scores.groupby(groups, as_index=False).agg(mean_absolute_error=("absolute_error", "mean"), mean_absolute_log_error=("absolute_log_error", "mean"))
    selected = scores[scores.horizon.eq(5) & scores.family.isin(ROLES)].copy()
    selected_intervals = interval_scores[interval_scores.horizon.eq(5) & interval_scores.family.isin(ROLES) & interval_scores.level.eq(.8)].copy()
    selected_wis = wis[wis.horizon.eq(5) & wis.family.isin(ROLES)].copy()
    return validation, summary, selected, selected_intervals, selected_wis, decomposition


def audit_count_coherence(config, panel):
    verify_events(COUNT, "reconciled_forecasts_committed")
    independent, predictions, weights, scored = [read(COUNT / name) for name in ["independent_predictions.csv", "predictions.csv", "weights.csv", "point_scores.csv"]]
    bottom, definitions, matrix = node_definitions(config)
    node_order = [row[0] for row in definitions]
    checked = 0
    for (target, outcome), source in panel[panel.location_name.isin([c["name"] for c in config["countries"] if c["gcc"]])
                                          & panel.outcome.isin(["prevalence", "incidence"])].groupby(["location_name", "outcome"]):
        history = source.pivot(index="year", columns=["sex", "age"], values="count").reindex(columns=pd.MultiIndex.from_tuples(bottom))
        history = pd.DataFrame(history.to_numpy()@matrix.T, index=history.index, columns=node_order)
        fit = independent[independent.target.eq(target) & independent.outcome.eq(outcome)]
        for origin in range(2014, 2019):
            floor = 1e-8*history.loc[1990:origin].mean().to_numpy()**2
            for horizon in range(1, 6):
                residual = []
                for earlier in range(2003, origin-4):
                    old = fit[fit.origin.eq(earlier) & fit.horizon.eq(horizon)].set_index("node").reindex(node_order)
                    residual.append(history.loc[earlier+horizon].to_numpy()-old.count_prediction.to_numpy())
                residual = np.asarray(residual)
                variance = np.maximum(np.var(residual, axis=0, ddof=0), floor)
                actual_weights = weights[weights.target.eq(target) & weights.outcome.eq(outcome) & weights.origin.eq(origin)
                                         & weights.horizon.eq(horizon)].set_index("node").reindex(node_order)
                np.testing.assert_allclose(actual_weights.variance, variance, atol=1e-10, rtol=1e-12)
                assert actual_weights.residual_blocks.eq(len(residual)).all() and actual_weights.last_residual_label_year.le(origin).all()
                current = predictions[predictions.target.eq(target) & predictions.outcome.eq(outcome) & predictions.origin.eq(origin) & predictions.horizon.eq(horizon)]
                base = fit[fit.origin.eq(origin) & fit.horizon.eq(horizon)].set_index("node").reindex(node_order).count_prediction.to_numpy()
                for family, values in current.groupby("family"):
                    values = values.set_index("node").reindex(node_order)
                    counts = values.count_prediction.to_numpy()
                    discrepancy = counts-matrix@counts[:22]
                    np.testing.assert_allclose(values.coherence_discrepancy, discrepancy, atol=1e-9, rtol=1e-12)
                    if family == "independent":
                        np.testing.assert_array_equal(counts, base)
                    elif family == "bottom_up":
                        np.testing.assert_allclose(counts, matrix@base[:22], atol=1e-9, rtol=1e-12)
                    else:
                        assert family == "nonnegative_diagonal_wls" and (counts >= 0).all()
                        design, response = matrix/np.sqrt(variance[:, None]), base/np.sqrt(variance)
                        residual_wls = design@counts[:22]-response
                        gradient = design.T@residual_wls
                        tolerance = 1e-7*(1+np.linalg.norm(design)*np.linalg.norm(residual_wls))
                        assert (gradient >= -tolerance).all()
                        assert np.max(abs(gradient[counts[:22] > 1e-8]), initial=0) <= tolerance
                        np.testing.assert_allclose(discrepancy, 0, atol=1e-8)
                    selected = scored[scored.target.eq(target) & scored.outcome.eq(outcome) & scored.origin.eq(origin)
                                      & scored.horizon.eq(horizon) & scored.family.eq(family)].set_index("node").reindex(node_order)
                    np.testing.assert_allclose(selected.observed_count, history.loc[origin+horizon], atol=1e-9, rtol=1e-12)
                    np.testing.assert_array_equal(selected.count_prediction, counts)
                    np.testing.assert_allclose(selected.absolute_count_error, abs(counts-history.loc[origin+horizon].to_numpy()), atol=1e-9, rtol=1e-12)
                    checked += len(counts)
    assert checked == len(scored) == 27900
    groups = ["target", "outcome", "origin", "horizon", "family", "level"]
    summary = scored.groupby(groups, as_index=False).agg(count_mae=("absolute_count_error", "mean"),
                                                        max_coherence_discrepancy=("coherence_discrepancy", lambda x: abs(x).max()))
    saved = read(COUNT / "summary.csv").set_index(groups).sort_index()
    expected = summary.set_index(groups).sort_index()
    np.testing.assert_allclose(saved[expected.columns], expected, atol=1e-9, rtol=1e-12)
    return {"passed": True, "native_count_cells_recalculated": checked, "variance_rows_recalculated": len(weights),
            "nonnegative_wls_optimality_checked": True}, summary, scored


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def count_report(summary, scored):
    audit = json.loads((COUNT / "fit_audit.json").read_text())
    failures = [row for row in audit if row["status"] != "ok"]
    failure_rows = []
    for row in failures:
        node = row.get("age", row.get("node", ""))
        origin = row["origin"]
        used = sorted({fit for fit in range(2014, 2019) if origin <= 2013 and origin+5 <= fit})
        failure_rows.append([row["target"], row["outcome"], origin, node,
                             "Yes" if 2014 <= origin <= 2018 else "No", ", ".join(map(str, used)) or "None"])
    table = markdown_table(["Country", "Outcome", "Origin", "Node", "Evaluation point", "Weight-bank origins using this fallback"], failure_rows)
    rows = []
    for (outcome, level), part in summary[summary.target.eq("Saudi Arabia") & summary.origin.eq(2018) & summary.horizon.eq(5)].groupby(["outcome", "level"]):
        values = part.set_index("family")
        rows.append([outcome, level, *[f"{values.loc[f, 'count_mae']:.2f}" for f in ["independent", "bottom_up", "nonnegative_diagonal_wls"]]])
    saudi = markdown_table(["Outcome", "Hierarchy level", "Independent MAE", "Bottom-up MAE", "Nonnegative WLS MAE"], rows)
    rows = []
    for (target, outcome), part in summary[summary.origin.eq(2018) & summary.horizon.eq(5) & summary.level.eq("grand_total")].groupby(["target", "outcome"]):
        values = part.set_index("family")
        rows.append([target, outcome, *[f"{values.loc[f, 'count_mae']:.2f}" for f in ["independent", "bottom_up", "nonnegative_diagonal_wls"]]])
    gulf = markdown_table(["Country", "Outcome", "Independent absolute error", "Bottom-up absolute error", "Nonnegative WLS absolute error"], rows)
    coherence = scored.groupby("family").coherence_discrepancy.agg(lambda x: float(abs(x).max()))
    text = f"""# Native-count hierarchy experiment

This secondary experiment forecasts native GBD modeled counts for ages 45+ independently of the population-conditioned rate forecasts. It does not alter the Saudi primary rate result. Independent damped ETS forecasts at 31 hierarchy nodes are compared with bottom-up aggregation and nonnegative diagonal weighted least squares (WLS). All methods use the same issued native-count forecasts and pre-origin residual evidence.

## Saudi Arabia: origin 2018, horizon 5

Errors are in modeled-number units. Age–sex, broad-age–sex, sex-total and grand-total rows average over 22, six, two and one nodes respectively. Their magnitudes should not be averaged into a single unweighted clinical performance measure.

{saudi}

## Gulf benchmark: 45+ grand totals in 2023

Each cell is the absolute error of one both-sex 45+ total; it is not a sample mean over independent participants.

{gulf}

## Accounting consistency and failures

Maximum absolute count-sum discrepancy over all targets, outcomes, origins and horizons: independent={coherence['independent']:.9g}; bottom-up={coherence['bottom_up']:.9g}; nonnegative WLS={coherence['nonnegative_diagonal_wls']:.9g}. The coherent methods enforce accounting identities. Their accuracy must be assessed separately from this arithmetic property; the tables retain cases in which reconciliation increases error.

{table}

Every failed ETS fit retains the locked persistence forecast. It is kept in evaluated forecasts or historical residual banks whenever temporally eligible. There are no dropped nodes. Variances use centered complete historical count-error blocks, denominator n, with the declared training-mean floor. WLS is solved with nonnegativity constraints; it is not an unconstrained forecast clipped afterward. This report independently checks variance construction, all 27,900 evaluated cells, both summing identities, and constrained optimality conditions. No ETS model is refitted for the audit.

The [full native-count ledger](../../results/count_coherence_v1/point_scores.csv) and [origin/horizon summaries](../../results/count_coherence_v1/summary.csv) retain all five horizons and evaluation origins. Count coherence does not imply reliable forecast intervals, which are not constructed for this native-count experiment. These modeled counts are not clinical registries or validated service demand.
"""
    (OUT / "native_count_report.md").write_text(text)
    return failures, coherence.to_dict()


def plot_burden(points, intervals, config):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    current = points[points.target.eq("Saudi Arabia") & points.origin.eq(2018)]
    bands = intervals[intervals.target.eq("Saudi Arabia") & intervals.origin.eq(2018)]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, outcome in enumerate(["prevalence", "incidence"]):
        for col, sex in enumerate(["Male", "Female", "Both"]):
            part = current[current.outcome.eq(outcome) & current.measure.eq("count") & current.population_method.eq("log_trend_last8")
                           & current.sex.eq(sex) & current.age_group.eq("45+")].set_index("family")
            interval = bands[bands.outcome.eq(outcome) & bands.measure.eq("count") & bands.population_method.eq("log_trend_last8")
                             & bands.sex.eq(sex) & bands.age_group.eq("45+")].set_index("family")
            observed = part.observed.iloc[0]
            assert np.allclose(part.observed, observed)
            ax = axs[row, col]
            values = [observed] + [part.loc[family, "value"] for family in ROLES]
            ax.bar(range(4), values, color=[COLORS["observed"]]+[COLORS[f] for f in ROLES], alpha=.75)
            for index, family in enumerate(ROLES, 1):
                item = interval.loc[family]
                ax.vlines(index, item.lower, item.upper, color="black", linewidth=1.7)
                ax.scatter([index], [item["median"]], marker="_", color="black", s=80)
            ax.set(title=f"{outcome.title()} — {sex}", ylabel="Modeled number, ages 45+", xticks=range(4),
                   xticklabels=["Observed", "TCN", "Local", "Non-neural"])
    fig.suptitle("Saudi 2023 burden from origin 2018\nBars: points; black lines: 80% predictive intervals; ticks: predictive medians", fontsize=14)
    fig.savefig(OUT / "saudi_count_forecasts.png", dpi=180)
    fig.savefig(OUT / "saudi_count_forecasts.svg")
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, outcome in enumerate(["prevalence", "incidence"]):
        for col, sex in enumerate(config["sexes"]):
            ax = axs[row, col]
            for index, family in enumerate(["observed"]+ROLES):
                role = ROLES[0] if family == "observed" else family
                part = current[current.outcome.eq(outcome) & current.measure.eq("age_share") & current.population_method.eq("log_trend_last8")
                               & current.sex.eq(sex) & current.family.eq(role)].set_index("age_group")
                groups = ["65+_within_45+", "80+_within_45+"]
                x = np.arange(2)+(index-1.5)*.18
                values = part.reindex(groups).observed if family == "observed" else part.reindex(groups).value
                ax.bar(x, values, width=.17, label="Observed" if family == "observed" else LABELS[family], color=COLORS[family], alpha=.8)
                if family != "observed":
                    interval = bands[bands.outcome.eq(outcome) & bands.measure.eq("age_share") & bands.population_method.eq("log_trend_last8")
                                     & bands.sex.eq(sex) & bands.family.eq(family)].set_index("age_group").reindex(groups)
                    ax.vlines(x, interval.lower, interval.upper, color="black", linewidth=1.3)
            ax.set(title=f"{outcome.title()} — {sex}", xticks=range(2), xticklabels=["65+ within 45+", "80+ within 45+"], ylabel="Share of modeled 45+ burden (%)", ylim=(0, 100))
    axs[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Saudi 2023 age composition: points and 80% intervals\nOperational population: preceding eight-year log trend", fontsize=14)
    fig.savefig(OUT / "saudi_age_shares.png", dpi=180)
    fig.savefig(OUT / "saudi_age_shares.svg")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, outcome in zip(axs, ["prevalence", "incidence"]):
        for family in ["observed"]+ROLES:
            role = ROLES[0] if family == "observed" else family
            part = current[current.outcome.eq(outcome) & current.measure.eq("sex_rate_ratio") & current.family.eq(role)].set_index("age_group").reindex(config["ages"])
            values = part.observed if family == "observed" else part.value
            ax.plot(range(11), values, color=COLORS[family], marker="o", markersize=3,
                    label="Observed" if family == "observed" else LABELS[family])
        interval = bands[bands.outcome.eq(outcome) & bands.measure.eq("sex_rate_ratio") & bands.family.eq("tcn_adapted")].set_index("age_group").reindex(config["ages"])
        ax.fill_between(range(11), interval.lower, interval.upper, color=COLORS["tcn_adapted"], alpha=.16, label="TCN 80% interval")
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(title=outcome.title(), ylabel="Male / female age-specific rate", xticks=range(11), xticklabels=config["ages"])
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("Saudi 2023 sex-rate ratios from origin 2018\nRatios use rates, without sex-population weighting", fontsize=14)
    fig.savefig(OUT / "saudi_sex_rate_ratios.png", dpi=180)
    fig.savefig(OUT / "saudi_sex_rate_ratios.svg")
    plt.close(fig)


def burden_report(points, interval_scores, wis, decomposition, validations):
    current = points[points.origin.eq(2018)]
    saudi_totals = points[points.target.eq("Saudi Arabia") & points.measure.eq("count")
                          & points.node.eq("Both__45+") & points.population_method.eq("log_trend_last8")]
    saudi_count_loss = saudi_totals.groupby(["outcome", "family"]).absolute_error.mean()
    saudi_endpoint = saudi_totals[saudi_totals.origin.eq(2018)].set_index(["outcome", "family"])
    saudi_ratio_loss = points[points.target.eq("Saudi Arabia") & points.measure.eq("sex_rate_ratio")].groupby(
        ["outcome", "family"]).absolute_log_error.mean()
    oldest_male = current[current.target.eq("Saudi Arabia") & current.outcome.eq("prevalence")
                          & current.measure.eq("age_share") & current.sex.eq("Male")
                          & current.age_group.eq("80+_within_45+") & current.population_method.eq("log_trend_last8")
                          & current.family.eq("tcn_adapted")].iloc[0]
    female_total = current[current.target.eq("Saudi Arabia") & current.outcome.eq("prevalence")
                           & current.measure.eq("count") & current.node.eq("Female__45+")
                           & current.population_method.eq("log_trend_last8") & current.family.eq("tcn_adapted")].iloc[0]
    female_components = decomposition[decomposition.target.eq("Saudi Arabia") & decomposition.outcome.eq("prevalence")
                                       & decomposition.sex.eq("Female")
                                       & decomposition.population_method.isin(["log_trend_last8", "implied_gbd"])].set_index("family")
    gulf_count_loss = points[points.measure.eq("count") & points.node.eq("Both__45+")
                            & points.population_method.eq("log_trend_last8")].groupby(["target", "outcome", "family"]).absolute_error.mean().unstack()
    gulf_ratio_loss = points[points.measure.eq("sex_rate_ratio")].groupby(["target", "outcome", "family"]).absolute_log_error.mean().unstack()
    count_wins = (gulf_count_loss.tcn_adapted < gulf_count_loss.local_champion) & (gulf_count_loss.tcn_adapted < gulf_count_loss.nonneural_champion)
    ratio_wins = (gulf_ratio_loss.tcn_adapted < gulf_ratio_loss.local_champion) & (gulf_ratio_loss.tcn_adapted < gulf_ratio_loss.nonneural_champion)
    rows = []
    for outcome in ["prevalence", "incidence"]:
        for sex in ["Male", "Female", "Both"]:
            for family in ROLES:
                mask = current.target.eq("Saudi Arabia") & current.outcome.eq(outcome) & current.sex.eq(sex) & current.family.eq(family)
                selected = current[mask & current.measure.eq("count") & current.age_group.eq("45+")].set_index("population_method")
                trend, persistent = selected.loc["log_trend_last8"], selected.loc["persistence"]
                rows.append([outcome, sex, LABELS[family], f"{trend.observed:.1f}", f"{trend.value:.1f}", f"{trend.absolute_error:.1f}", f"{persistent.absolute_error:.1f}"])
    saudi_counts = markdown_table(["Outcome", "Sex", "Rate procedure", "Observed 45+ count", "Operational count point", "Log-trend population absolute error", "Persistence population absolute error"], rows)
    rows = []
    for (outcome, sex, group), part in current[current.target.eq("Saudi Arabia") & current.measure.eq("age_share")
                                             & current.population_method.eq("log_trend_last8")].groupby(["outcome", "sex", "age_group"]):
        part = part.set_index("family")
        rows.append([outcome, sex, group.split("_")[0], f"{part.observed.iloc[0]:.2f}",
                     *[f"{part.loc[family, 'value']:.2f} ({part.loc[family, 'absolute_error']:.2f})" for family in ROLES]])
    shares = markdown_table(["Outcome", "Sex", "Age threshold", "Observed share (%)", "TCN share (pp error)", "Local share (pp error)", "Non-neural share (pp error)"], rows)
    rows = []
    for outcome in ["prevalence", "incidence"]:
        cases = [("count", "Both", "45+", "Modeled number")]
        cases += [("age_share", sex, group, "Percentage points") for sex in ["Male", "Female"] for group in ["65+_within_45+", "80+_within_45+"]]
        cases += [("sex_rate_ratio", "Male/Female", None, "Rate-ratio scale")]
        for measure, sex, group, unit in cases:
            selected = interval_scores[interval_scores.target.eq("Saudi Arabia") & interval_scores.outcome.eq(outcome)
                                       & interval_scores.family.eq("tcn_adapted") & interval_scores.measure.eq(measure)
                                       & interval_scores.sex.eq(sex) & interval_scores.population_method.isin(["log_trend_last8", "not_applicable"])]
            proper = wis[wis.target.eq("Saudi Arabia") & wis.outcome.eq(outcome) & wis.family.eq("tcn_adapted")
                         & wis.measure.eq(measure) & wis.sex.eq(sex) & wis.population_method.isin(["log_trend_last8", "not_applicable"])]
            if group is not None:
                selected, proper = selected[selected.age_group.eq(group)], proper[proper.age_group.eq(group)]
            rows.append([outcome, measure, sex, group or "11 ages", unit,
                         f"{100*selected.covered.mean():.1f}%", f"{selected.width.mean():.3f}", f"{proper.wis_50_80.mean():.3f}"])
    saudi_intervals = markdown_table(["Outcome", "Endpoint", "Sex", "Age definition", "Width/WIS units", "80% coverage", "Mean width", "WIS (50/80)"], rows)
    rows = []
    for (target, outcome, family), part in points.groupby(["target", "outcome", "family"]):
        count = part[part.measure.eq("count") & part.node.eq("Both__45+") & part.population_method.eq("log_trend_last8")]
        share = part[part.measure.eq("age_share") & part.population_method.eq("log_trend_last8")]
        ratio = part[part.measure.eq("sex_rate_ratio")]
        cells = interval_scores[interval_scores.target.eq(target) & interval_scores.outcome.eq(outcome) & interval_scores.family.eq(family)
                                & interval_scores.measure.eq("count") & interval_scores.node.eq("Both__45+") & interval_scores.population_method.eq("log_trend_last8")]
        rows.append([target, outcome, LABELS[family], f"{count.absolute_error.mean():.2f}", f"{share.absolute_error.mean():.3f}",
                     f"{ratio.absolute_log_error.mean():.5f}", f"{100*cells.covered.mean():.0f}%"])
    gcc = markdown_table(["Country", "Outcome", "Rate procedure", "45+ total count MAE", "Mean share pp error", "Mean absolute log-ratio error", "Total-count 80% coverage"], rows)
    rows = []
    for (outcome, sex), part in decomposition[decomposition.target.eq("Saudi Arabia")].groupby(["outcome", "sex"]):
        for family in ["observed_accounting", "tcn_adapted"]:
            selected = part[part.family.eq(family) & (part.population_method.eq("log_trend_last8") | part.population_method.eq("implied_gbd"))].iloc[0]
            rows.append([outcome, sex, "Observed accounting" if family == "observed_accounting" else "TCN operational forecast", *[f"{selected[field]:+.2f}" for field in ["population_size", "age_composition", "rate_component", "total_change"]]])
    accounting = markdown_table(["Outcome", "Sex", "Endpoint", "Population-size component", "Age-composition component", "Rate component", "Total count change"], rows)
    total_points = sum(row["point_statistics_reconstructed"] for row in validations)
    total_draws = sum(row["joint_statistics_reconstructed"] for row in validations)
    total_intervals = sum(row["interval_quantiles_reconstructed"] for row in validations)
    text = f"""# Operational burden, age composition and count consistency

These secondary analyses translate all 14 rate-model families and both frozen comparator roles into modeled Parkinson’s burden for ages 45+ in the six GCC countries. Prevalence and incidence are evaluated separately. **The original Saudi primary rate conclusion remains unchanged.** All displayed forecasts are retrospective five-year predictions; these are not new forecasts issued from 2023.

**Saudi total-count accuracy favors the local comparator, while age composition and interval reliability reveal weaknesses hidden by aggregate totals.** Across the five origins, TCN versus local-comparator total-count MAE is {saudi_count_loss.loc['prevalence','tcn_adapted']:.2f} versus {saudi_count_loss.loc['prevalence','local_champion']:.2f} for prevalence, and {saudi_count_loss.loc['incidence','tcn_adapted']:.2f} versus {saudi_count_loss.loc['incidence','local_champion']:.2f} for incidence. The corresponding final-endpoint absolute errors are {saudi_endpoint.loc[('prevalence','tcn_adapted'),'absolute_error']:.2f} versus {saudi_endpoint.loc[('prevalence','local_champion'),'absolute_error']:.2f}, and {saudi_endpoint.loc[('incidence','tcn_adapted'),'absolute_error']:.2f} versus {saudi_endpoint.loc[('incidence','local_champion'),'absolute_error']:.2f}. These are separate secondary metrics, not a replacement primary endpoint.

## Operational meaning and uncertainty

Age–sex counts equal forecast age-specific rates × forecast populations / 100,000. Populations are inferred from native GBD Number/Rate and forecast from the preceding eight annual log values at each issuance origin; population persistence is the fixed sensitivity. No realized future population or later UN vintage is used as an operational predictor. The observed comparator is the native GBD modeled count. The modeled population denominators are inferred separately by outcome to preserve source accounting. No age-standardized rate is multiplied by a population.

Each joint draw combines the rate residual and population residual from the same completed historical origin. Whole blocks preserve dependence across age, sex and horizon. Counts, sums, shares and rate ratios are transformed before direct empirical quantiles are calculated. Point sums and predictive medians remain distinct. Source-estimate bounds and neural-seed dispersion are not treated as predictive uncertainty. The 7–11 overlapping historical blocks provide no exact coverage guarantee.

## Saudi Arabia: 2023 endpoint from origin 2018

Counts are restricted to ages 45+. Prevalence counts represent modeled prevalent cases; incidence counts represent modeled new cases during the year. Rate families and comparator roles retain the settings selected before issuance. Differences between sex populations therefore affect burden counts without being interpreted as disease susceptibility.

{saudi_counts}

![Saudi count forecasts](saudi_count_forecasts.png)

## Age composition and sex-rate ratios

Both age-share denominators are the same-sex modeled burden at ages 45+: the 65+ or 80+ numerator is divided by that 45+ total. The thresholds are nested and are not complementary categories. Errors are percentage points, not relative percentages.

{shares}

The TCN predicts the Saudi male 80+ prevalence share at {oldest_male.value:.2f}% versus {oldest_male.observed:.2f}% observed, an underestimate of {oldest_male.absolute_error:.2f} percentage points. Both comparator procedures show a similar shortfall. Good prediction of the total burden therefore does not establish accurate prediction of its oldest-age distribution.

![Saudi age shares](saudi_age_shares.png)

Male/female ratios use paired age-specific rates directly. Unequal male and female population sizes do not enter the rate ratios. The descriptive age profiles do not identify hormonal, genetic, occupational, migration or diagnostic mechanisms.

TCN ratio accuracy is modestly better than both comparator roles in Saudi Arabia: five-origin mean absolute log-ratio error is {saudi_ratio_loss.loc['prevalence','tcn_adapted']:.5f} for prevalence (local {saudi_ratio_loss.loc['prevalence','local_champion']:.5f}; non-neural {saudi_ratio_loss.loc['prevalence','nonneural_champion']:.5f}) and {saudi_ratio_loss.loc['incidence','tcn_adapted']:.5f} for incidence (local {saudi_ratio_loss.loc['incidence','local_champion']:.5f}; non-neural {saudi_ratio_loss.loc['incidence','nonneural_champion']:.5f}). This point-accuracy result should be read alongside the roughly 50% coverage of nominal 80% ratio intervals below.

![Saudi sex-rate ratios](saudi_sex_rate_ratios.png)

## Saudi adapted-TCN interval reliability

These horizon-5 summaries average origins 2014–2018. Count and each share endpoint have five dependent origin observations; the rate-ratio row averages 11 ages across those origins. WIS uses 50% and 80% intervals and the predictive median. Units remain separate; a numerical WIS from one endpoint cannot be ranked against another endpoint. The [selected-role interval ledger](selected_role_interval_summary.csv) retains all three roles and individual origins.

{saudi_intervals}

The 100% coverage of both-sex total-count intervals represents only five overlapping origin observations for each outcome. It coexists with 20% coverage of the male 80+ share intervals and much lower than nominal ratio coverage. Aggregate coverage therefore does not demonstrate reliable age-specific or sex-comparison uncertainty.

## Gulf benchmarking across five origins

The table reports horizon 5 over origins 2014–2018, including 2018 once. Count MAE averages the five both-sex 45+ totals. Share error averages the two sex-specific thresholds across the five origins; separate 65+ and 80+ endpoints remain in each case’s `age_share_summary.csv`. Ratio error averages 11 ages and five origins. The three metrics have different units and are not pooled into a composite score. Total-count coverage has only five dependent origin observations and changes in 20-percentage-point steps.

{gcc}

For total-count MAE, the TCN beats both frozen comparator roles in {int(count_wins.xs('prevalence',level='outcome').sum())} of six prevalence analyses and {int(count_wins.xs('incidence',level='outcome').sum())} of six incidence analyses. For rate-ratio error it does so in {int(ratio_wins.xs('prevalence',level='outcome').sum())} of six prevalence analyses and {int(ratio_wins.xs('incidence',level='outcome').sum())} of six incidence analyses. These are descriptive within-country comparisons, not independent country-level significance tests. Qatar's total-count intervals cover none of the five verified totals for either outcome under any of the three displayed procedures, showing that Saudi aggregate coverage does not generalize uniformly across the Gulf.

All families, ages, horizons, sex totals, grand totals, population methods and uncertainty levels remain in the [full results](../../results/demography_v1/) and [independently checked endpoint summary](endpoint_summary.csv). The 95% intervals are supplementary sparse-tail summaries. Joint block variation does not represent sampling from independent patients.

## Descriptive 2018–2023 count-change accounting

The symmetric Shapley calculation averages all six replacement orders for population size, population age composition within ages 45+, and age-specific rates. The signed components sum to the total count change. The observed endpoint and the operational TCN endpoint answer different descriptive questions; neither gives causal biological attribution. Components can be negative even when total burden increases.

{accounting}

The near-exact female prevalence total ({female_total.value:.1f} predicted versus {female_total.observed:.1f} observed; error {female_total.absolute_error:.1f}) masks compensating component errors. Its forecast population-size contribution is {female_components.loc['tcn_adapted','population_size']:+.2f}, compared with {female_components.loc['observed_accounting','population_size']:+.2f} observed, while its population-age-composition contribution is {female_components.loc['tcn_adapted','age_composition']:+.2f}, compared with {female_components.loc['observed_accounting','age_composition']:+.2f} observed. The opposite composition signs mean that an accurate total cannot be interpreted as an accurately forecast demographic trajectory.

## Native-count reconciliation

The separate [native-count experiment](native_count_report.md) forecasts the 31 hierarchy nodes directly with unchanged damped ETS, then compares independent, bottom-up and nonnegative diagonal WLS predictions. Accounting coherence is reported separately from forecast accuracy. These native-count models do not replace the population-conditioned rate-model counts above.

## Validation and limitations

The independent audit reconstructed {total_points:,} point statistics, {total_draws:,} paired transformed draws, and {total_intervals:,} interval rows across all 12 country–outcome cases. It independently reconstructs historical population forecasts and residuals, rate ratios, count sums, nested shares, quantiles, WIS, observed verification values and all Shapley components. Source files and shared test/code/specification hashes are checked, and forecast commits precede scoring-start events. The native-count audit additionally recalculates all 27,900 evaluation values and pre-origin variance weights and checks nonnegative WLS optimality. See [validation](validation.json).

These are forecasts of retrospectively modeled GBD point estimates, not clinical event predictions, survival estimates, causal effects, staffing requirements or proof of adequate service capacity. Population uncertainty is represented only by paired historical operational errors, not a full demographic uncertainty model. The separate [primary report](../primary_v1/report.md) remains the source of the prespecified primary conclusion.

This report completes the operational-population and native-count components covered here. The separately locked UN population-source sensitivity and projection scenarios remain pending; the entire demographic workstream is not claimed complete.
"""
    (OUT / "report.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four independent report workers")
    check_lock()
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    panel = read(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    count_manifest = verify_run(COUNT)
    assert sha(ROOT / "data/processed/design_v1/regional_outcomes.csv") == count_manifest["source_sha256"]
    assert sha(ROOT / "study_design/locked_v1/design.json") == count_manifest["config_sha256"]
    count_validation, count_summary, count_scores = audit_count_coherence(config, panel)
    OUT.mkdir(parents=True, exist_ok=True)
    failures, discrepancies = count_report(count_summary, count_scores)
    count_validation.update(report_code_sha256=sha(Path(__file__)), source_manifest_sha256=sha(COUNT / "run_manifest.json"),
                            fit_fallbacks=failures, maximum_coherence_discrepancy=discrepancies)
    (OUT / "count_validation.json").write_text(json.dumps(count_validation, indent=2)+"\n")
    if args.count_only:
        print(json.dumps(count_validation, indent=2))
        return
    manifest = verify_run(RUN)
    for run, expected in manifest["source_manifest_sha256"].items():
        assert sha(ROOT / "results" / run / "run_manifest.json") == expected
    cases = [dict(id=country["iso3"]+"_"+outcome, target=country["name"], outcome=outcome,
                  source="results/primary_v1" if country["iso3"] == "SAU" and outcome == "prevalence" else "results/secondary_v1/trials/"+country["iso3"]+"_"+outcome)
             for country in config["countries"] if country["gcc"] for outcome in ["prevalence", "incidence"]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(audit_case, case, config) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Independent burden audit complete: {result[0]['case']}", flush=True)
    validations = [result[0] for result in results]
    summary, points, intervals, wis, decomposition = [pd.concat([result[index] for result in results], ignore_index=True) for index in range(1, 6)]
    summary.sort_values(["target", "outcome", "origin", "family", "horizon"]).to_csv(OUT / "endpoint_summary.csv", index=False)
    # Keep proper-score units separate for counts, percentage shares and ratios.
    group = ["target", "outcome", "origin", "family", "population_method", "measure", "sex", "age_group", "unit"]
    joined = intervals.groupby(group, as_index=False).agg(coverage_80=("covered", "mean"), width_80=("width", "mean"))
    joined = joined.merge(wis.groupby(group, as_index=False).agg(wis_50_80=("wis_50_80", "mean"), wis_50_80_95=("wis_50_80_95", "mean")), on=group, validate="one_to_one")
    joined.to_csv(OUT / "selected_role_interval_summary.csv", index=False)
    plot_burden(points, intervals, config)
    burden_report(points, intervals, wis, decomposition, validations)
    validation = {"passed": True, "role": "independent_secondary_demography_and_native_count_audit",
                  "report_code_sha256": sha(Path(__file__)), "demography_manifest_sha256": sha(RUN / "run_manifest.json"),
                  "count_manifest_sha256": sha(COUNT / "run_manifest.json"), "case_validations": validations,
                  "count_validation": count_validation, "original_primary_conclusion_preserved": True}
    validation["report_artifact_sha256"] = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({"passed": True, "cases": len(validations), "report": str(OUT / 'report.md')}, indent=2))


if __name__ == "__main__":
    main()
