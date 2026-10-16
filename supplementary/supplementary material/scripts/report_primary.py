"""Independent audit and reporting of the frozen Saudi primary rate evaluation."""

import json
import os
from datetime import datetime
from pathlib import Path
import sys

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gbd_park_matplotlib")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gbd_park.pooled import target_inputs
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
from gbd_park.tcn_forecasting import base_id
from run_local_baselines import check_lock, sha

RUN = ROOT / "results/primary_v1"
OUT = ROOT / "reports/primary_v1"
COORDS = ["sex", "age", "horizon"]
ROLES = ["tcn_adapted", "local_champion", "nonneural_champion"]
LABELS = {
    "persistence": "Persistence", "log_trend": "Log-linear trend", "damped_ets": "Damped ETS",
    "arima": "ARIMA", "age_smooth_trend": "Age-smoothed trend", "pooled_ridge": "Pooled ridge",
    "pooled_boosting": "Pooled boosting", "donor_ridge_unadapted": "Donor ridge, unadapted",
    "donor_ridge_adapted": "Donor ridge, adapted", "donor_boosting_unadapted": "Donor boosting, unadapted",
    "donor_boosting_adapted": "Donor boosting, adapted", "tcn_adapted": "Adapted TCN",
    "tcn_unadapted": "TCN, no adaptation", "tcn_intercept": "TCN, intercept adaptation",
    "local_champion": "Local champion", "nonneural_champion": "Non-neural champion",
}
COLORS = {"tcn_adapted": "#146ca4", "local_champion": "#64748b", "nonneural_champion": "#c26d19"}


def verify_run(directory, final_period=False):
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["final_period_scored"] is final_period
    for name, expected in manifest["output_sha256"].items():
        assert sha(directory / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    return manifest


def verify_commit_order():
    commit = json.loads((RUN / "pre_score_commit.json").read_text())
    for name, expected in commit["artifact_sha256"].items():
        assert sha(RUN / name) == expected
    assert {"predictions.csv", "intervals.csv", "joint_draws.csv", "origin_population_weights.csv"}.issubset(commit["artifact_sha256"])
    events = json.loads((RUN / "events.json").read_text())
    names = [event["event"] for event in events]
    order = ["settings_frozen", "forecast_fitting_started", "predictions_committed", "intervals_committed", "evaluation_scoring_started"]
    assert all(names.count(name) == 1 for name in order)
    assert [names.index(name) for name in order] == sorted(names.index(name) for name in order)
    times = [datetime.fromisoformat(events[names.index(name)]["time_utc"]) for name in order]
    assert times == sorted(times)
    assert datetime.fromisoformat(commit["committed_utc"]) <= times[-1]
    for name in ["settings_decisions.csv", "tcn_choices.json", "champion_family_mappings.csv", "origin_population_weights.csv"]:
        assert name in events[names.index("settings_frozen")]["hashes"]


def verify_roles(points, intervals, mappings):
    for role in ["local_champion", "nonneural_champion"]:
        for row in mappings.loc[mappings.role.eq(role)].itertuples():
            assert row.last_selection_target_year <= row.fit_origin
            keys = ["age", "horizon"]
            chosen = points.loc[points.origin.eq(row.fit_origin) & points.sex.eq(row.sex) & points.family.eq(role)].set_index(keys).sort_index()
            source = points.loc[points.origin.eq(row.fit_origin) & points.sex.eq(row.sex) & points.family.eq(row.source_family)].set_index(keys).sort_index()
            assert chosen.index.equals(source.index) and len(chosen) == 55
            assert chosen.source_family.eq(row.source_family).all() and chosen.setting_id.eq(source.setting_id).all()
            np.testing.assert_array_equal(chosen.prediction, source.prediction)
            np.testing.assert_array_equal(chosen.log_prediction, source.log_prediction)
            interval_keys = keys + ["scale", "level"]
            chosen = intervals.loc[intervals.origin.eq(row.fit_origin) & intervals.sex.eq(row.sex) & intervals.family.eq(role)].set_index(interval_keys).sort_index()
            source = intervals.loc[intervals.origin.eq(row.fit_origin) & intervals.sex.eq(row.sex) & intervals.family.eq(row.source_family)].set_index(interval_keys).sort_index()
            assert chosen.index.equals(source.index) and len(chosen) == 330
            np.testing.assert_allclose(chosen[["lower", "median", "upper", "point_prediction"]],
                                       source[["lower", "median", "upper", "point_prediction"]], atol=1e-13, rtol=1e-12)


def verify_banks_and_quantiles(points, intervals, draws, config):
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"], config["calendar"]["horizons"]], names=COORDS)
    references = pd.read_csv(RUN / "bank_references.csv")
    count = 0
    assert len(references) == 80
    for row in references.itertuples():
        path = ROOT / row.path
        assert sha(path) == row.sha256
        bank = joblib.load(path)
        assert bank["fit_origin"] == row.origin and bank["family"] == row.family and bank["n_blocks"] == row.origin - 2007
        assert max(bank["origins"]) + 5 <= row.origin
        assert list(map(tuple, bank["coords"].to_numpy())) == list(coords)
        point = points.loc[points.origin.eq(row.origin) & points.family.eq(row.family)].set_index(COORDS).reindex(coords)
        log_draws = point.log_prediction.to_numpy()[None, :] + bank["centered_residuals"]
        rate_draws = np.exp(log_draws)
        saved_draws = draws.loc[draws.origin.eq(row.origin) & draws.family.eq(row.family)]
        assert len(saved_draws) == bank["n_blocks"] * len(coords)
        for scale, values, field, point_field in [("log_rate", log_draws, "log_draw", "log_prediction"),
                                                   ("rate", rate_draws, "rate_draw", "prediction")]:
            actual_draws = saved_draws.pivot(index="residual_origin", columns=COORDS, values=field).reindex(
                index=bank["origins"], columns=coords).to_numpy()
            np.testing.assert_allclose(actual_draws, values, rtol=1e-12, atol=1e-13)
            for level in config["intervals"]["central_levels"]:
                part = intervals.loc[intervals.origin.eq(row.origin) & intervals.family.eq(row.family)
                                     & intervals.scale.eq(scale) & intervals.level.eq(level)].set_index(COORDS).reindex(coords)
                expected = np.quantile(values, [(1-level)/2, 0.5, (1+level)/2], axis=0, method="linear").T
                np.testing.assert_allclose(part[["lower", "median", "upper"]], expected, rtol=1e-12, atol=1e-13)
                np.testing.assert_allclose(part.point_prediction, point[point_field], rtol=1e-12, atol=1e-13)
                count += len(part)
    assert count == len(intervals) == 52800
    return count


def verify_summaries(scores, weights, config):
    keys = ["target", "outcome", "origin", "sex", "age"]
    joined = scores.merge(weights[keys + ["origin_population_weight"]], on=keys, how="left", validate="many_to_one")
    joined["weighted_error"] = joined.absolute_log_error * joined.origin_population_weight
    group_keys = ["target", "outcome", "origin", "sex", "family", "horizon"]
    expected = joined.groupby(group_keys).agg(mean_age_absolute_log_error=("absolute_log_error", "mean"),
                                             rate_mae=("absolute_rate_error", "mean"),
                                             population_weighted_absolute_log_error=("weighted_error", "sum"))
    saved = pd.read_csv(RUN / "by_origin.csv").set_index(group_keys).sort_index()
    assert expected.index.equals(saved.index)
    np.testing.assert_allclose(saved[expected.columns], expected, rtol=1e-12, atol=1e-12)
    group_keys = ["target", "outcome", "sex", "family", "horizon"]
    reliability = expected.reset_index().groupby(group_keys).mean_age_absolute_log_error.mean()
    saved = pd.read_csv(RUN / "reliability_by_horizon.csv").set_index(group_keys).sort_index()
    np.testing.assert_allclose(saved.mean_absolute_log_error, reliability, rtol=1e-12, atol=1e-13)
    assert saved.n_origins.eq(5).all()
    all_horizons = scores.groupby(["target", "outcome", "sex", "family"]).absolute_log_error.mean()
    saved_all = pd.read_csv(RUN / "reliability_all_horizons.csv").set_index(["target", "outcome", "sex", "family"]).sort_index()
    np.testing.assert_allclose(saved_all.mean_absolute_log_error, all_horizons, rtol=1e-12, atol=1e-13)
    return expected.reset_index()


def verify_interval_scores(cells, wis):
    """Recalculate IS/WIS with magnitude-aware floating-point audit bounds.

    CSV parse/write round trips can shift a rate by an ulp. Subtraction of
    similar rates and the 2/alpha tail multiplier amplify that rounding; an
    absolute tolerance depending only on the final small score is unsuitable.
    These bounds affect audit comparisons only, never reported metric values
    or the strict primary success criterion.
    """
    eps = np.finfo(float).eps
    rate = cells.observed_rate.to_numpy()
    observed = np.where(cells.scale.eq("log_rate"), np.log(rate), rate)
    np.testing.assert_allclose(cells.observed_value, observed, rtol=1e-13, atol=1e-13)
    lower, upper, alpha = cells.lower.to_numpy(), cells.upper.to_numpy(), 1-cells.level.to_numpy()
    width = upper-lower
    calculated = width + np.where(observed < lower, (lower-observed)*2/alpha, 0)
    calculated += np.where(observed > upper, (observed-upper)*2/alpha, 0)
    width_bound = 16*eps*(np.abs(upper)+np.abs(lower)+1)
    score_bound = width_bound + 16*eps*(2/alpha)*(np.abs(observed)+np.abs(lower)+np.abs(upper)+1)
    width_difference = np.abs(cells.width.to_numpy()-width)
    score_difference = np.abs(cells.interval_score.to_numpy()-calculated)
    assert (width_difference <= width_bound).all(), "Width discrepancy exceeds floating-point round-trip bound"
    assert (score_difference <= score_bound).all(), "Interval-score discrepancy exceeds floating-point round-trip bound"
    np.testing.assert_array_equal(cells.covered, ((observed >= lower) & (observed <= upper)).astype(float))
    keys = ["target", "outcome", "origin", "family"] + COORDS + ["forecast_year", "scale"]
    index = wis.set_index(keys)
    terms = cells[keys + ["level", "median"]].copy()
    terms["weighted_score"] = alpha*calculated/2
    terms["weighted_bound"] = alpha*score_bound/2
    scores = terms.pivot(index=keys, columns="level", values="weighted_score").reindex(index=index.index)
    bounds = terms.pivot(index=keys, columns="level", values="weighted_bound").reindex(index=index.index)
    medians = terms.pivot(index=keys, columns="level", values="median").reindex(index=index.index)
    np.testing.assert_allclose(medians.to_numpy(), np.broadcast_to(index["median"].to_numpy()[:, None], medians.shape), atol=1e-13)
    median_error = np.abs(index.observed_value-index["median"])
    median_bound = 16*eps*(np.abs(index.observed_value)+np.abs(index["median"])+1)
    assert (np.abs(index.median_absolute_error-median_error) <= median_bound).all()
    maxima = {}
    for field, levels in [("wis_50_80", [0.5, 0.8]), ("wis_50_80_95", [0.5, 0.8, 0.95])]:
        expected = (0.5*median_error+scores[levels].sum(axis=1))/(len(levels)+0.5)
        tolerance = (0.5*median_bound+bounds[levels].sum(axis=1))/(len(levels)+0.5) + 16*eps*(np.abs(expected)+1)
        difference = np.abs(index[field]-expected)
        assert (difference <= tolerance).all(), f"{field} discrepancy exceeds floating-point round-trip bound"
        maxima[field + "_max_absolute_difference"] = float(difference.max())
    return len(wis), {"interval_score_max_absolute_difference": float(score_difference.max()),
                      "width_max_absolute_difference": float(width_difference.max()), **maxima,
                      "audit_roundoff_rule": "16*float64_epsilon times input magnitudes, propagated through subtraction and 2/alpha penalties; no metric or criterion changes"}


def verify_primary_contrasts(scores, contrasts, age_contrasts, verdict, config):
    endpoint = scores.loc[scores.origin.eq(2018) & scores.horizon.eq(5)]
    means = endpoint.groupby(["sex", "family"]).absolute_log_error.mean()
    expected_success = {}
    for sex in config["sexes"]:
        successful = []
        for comparator in ["local_champion", "nonneural_champion"]:
            row = contrasts.loc[contrasts.sex.eq(sex) & contrasts.comparator.eq(comparator)].iloc[0]
            neural, reference = float(means.loc[sex, "tcn_adapted"]), float(means.loc[sex, comparator])
            np.testing.assert_allclose([row.tcn_error, row.comparator_error, row.absolute_loss_difference],
                                       [neural, reference, neural-reference], rtol=1e-12, atol=1e-13)
            relative = (reference-neural)/reference if reference > 0 else np.nan
            np.testing.assert_allclose(row.relative_improvement, relative, equal_nan=True, atol=1e-13)
            np.testing.assert_allclose(row.relative_improvement_percent, relative*100, equal_nan=True, atol=1e-11)
            assert bool(row.strictly_lower) == (neural < reference)
            successful.append(neural < reference)
            ages = age_contrasts.loc[age_contrasts.sex.eq(sex) & age_contrasts.comparator.eq(comparator)].set_index("age").reindex(config["ages"])
            tcn = endpoint.loc[endpoint.sex.eq(sex) & endpoint.family.eq("tcn_adapted")].set_index("age").reindex(config["ages"])
            other = endpoint.loc[endpoint.sex.eq(sex) & endpoint.family.eq(comparator)].set_index("age").reindex(config["ages"])
            np.testing.assert_allclose(ages.absolute_loss_difference, tcn.absolute_log_error - other.absolute_log_error, atol=1e-13)
        expected_success[sex] = all(successful)
    assert len(contrasts) == 4 and len(age_contrasts) == 44
    assert verdict["success_by_sex"] == expected_success and verdict["joint_success"] == all(expected_success.values())
    return means


def replay_primary_tcn(scores, config):
    origin = 2018
    choices = json.loads((RUN / "tcn_choices.json").read_text())
    choice = next(row for row in choices if row["fit_origin"] == origin)
    panel = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    panel = panel.loc[panel.year.le(origin) & panel.outcome.eq("prevalence")]
    x, levels, meta = target_inputs(panel, config, origin, config["primary_target"])
    changes, failed, replayed_checkpoints = [], False, 0
    for seed in config["models"]["tcn"]["ensemble_seeds"]:
        name = f"origin{origin}__{base_id(choice['base'])}__seed={seed}.joblib"
        payload = joblib.load(RUN / "payloads" / name)
        assert payload["origin"] == origin and payload["seed"] == seed and payload["base"] == choice["base"]
        assert meta.equals(payload["current_meta"])
        np.testing.assert_array_equal(levels, payload["levels"])
        if payload["audit"]["status"] != "ok":
            failed = True
            changes.append(np.zeros((len(meta), 5)))
            continue
        fitted = load_checkpoint(RUN / "checkpoints" / name)
        before = state_fingerprint(fitted)
        prediction = predict_changes(fitted, x)
        np.testing.assert_array_equal(prediction, payload["current_changes"])
        assert before == state_fingerprint(fitted) == payload["audit"]["fingerprint_before"] == payload["audit"]["fingerprint_after"]
        changes.append(prediction)
        replayed_checkpoints += 1
    ensemble = np.mean(changes, axis=0)
    calibrations = pd.read_json(RUN / "tcn_adaptation_audit.jsonl", lines=True)
    cells = 0
    selected = scores.loc[scores.origin.eq(origin) & scores.family.str.startswith("tcn_")]
    for (family, sex), part in selected.groupby(["family", "sex"]):
        indices = meta.sex.eq(sex).to_numpy()
        forecast_changes = ensemble[indices].copy()
        if failed:
            forecast_changes[:] = 0
        elif family != "tcn_unadapted":
            cal = calibrations.loc[calibrations.origin.eq(origin) & calibrations.family.eq(family) & calibrations.sex.eq(sex)]
            assert len(cal) == 1
            row = cal.iloc[0]
            if row.status == "ok":
                forecast_changes += row.b0 + row.b1 * np.arange(1, 6)/5
            else:
                forecast_changes[:] = 0
        logs = levels[indices, None] + forecast_changes
        invalid = ~np.isfinite(logs).all(axis=1) | (np.abs(logs) > 700).any(axis=1)
        logs[invalid] = levels[indices][invalid, None]
        expected = part.pivot(index="age", columns="horizon", values="log_prediction").reindex(
            index=config["ages"], columns=config["calendar"]["horizons"]).to_numpy()
        np.testing.assert_allclose(logs, expected, rtol=0, atol=2e-14)
        cells += expected.size
    return {"primary_tcn_replayed_cells": cells, "frozen_checkpoints_replayed": replayed_checkpoints}


def main():
    check_lock()
    manifest = verify_run(RUN, final_period=True)
    for name, expected in manifest["prior_manifests_sha256"].items():
        prior = ROOT / "results" / name
        verify_run(prior)
        assert sha(prior / "run_manifest.json") == expected
    verify_commit_order()
    config = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    validation = json.loads((RUN / "validation_report.json").read_text())
    tests_path = RUN / "tests_at_run.json"
    assert sha(tests_path) == manifest["test_report_sha256"]
    tests = json.loads(tests_path.read_text())
    points = pd.read_csv(RUN / "predictions.csv")
    scores = pd.read_csv(RUN / "point_scores.csv")
    intervals = pd.read_csv(RUN / "intervals.csv")
    draws = pd.read_csv(RUN / "joint_draws.csv")
    cells = pd.read_csv(RUN / "interval_scores.csv")
    wis = pd.read_csv(RUN / "wis_scores.csv")
    weights = pd.read_csv(RUN / "origin_population_weights.csv")
    mappings = pd.read_csv(RUN / "champion_family_mappings.csv")
    contrasts = pd.read_csv(RUN / "primary_contrasts.csv")
    age_contrasts = pd.read_csv(RUN / "primary_age_contrasts.csv")
    verdict = json.loads((RUN / "primary_verdict.json").read_text())
    assert "observed_rate" not in points and "observed_rate" not in intervals
    assert len(points) == 8800 and len(cells) == 52800 and len(wis) == 17600
    assert not points.duplicated(["origin", "family"] + COORDS).any()
    assert set(points.origin) == set(config["calendar"]["reliability_origins"])
    assert points.forecast_year.max() == scores.forecast_year.max() == 2023
    point_keys = ["origin", "family"] + COORDS
    saved_predictions = points.set_index(point_keys).sort_index()
    scored_predictions = scores.set_index(point_keys).sort_index()
    prediction_roundtrip_difference = float(np.max(np.abs(saved_predictions.prediction - scored_predictions.prediction)))
    np.testing.assert_allclose(saved_predictions.prediction, scored_predictions.prediction, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(scores.absolute_log_error, np.abs(np.log(scores.prediction) - np.log(scores.observed_rate)), atol=1e-13)
    np.testing.assert_allclose(scores.absolute_rate_error, np.abs(scores.prediction - scores.observed_rate), atol=1e-11)
    source = pd.read_csv(ROOT / "data/processed/design_v1/regional_outcomes.csv")
    source = source.loc[source.location_name.eq(config["primary_target"]) & source.outcome.eq("prevalence")
                        & source.year.le(2023)].rename(columns={"location_name": "target", "year": "forecast_year", "rate": "source_rate"})
    truth_keys = ["target", "outcome", "sex", "age", "forecast_year"]
    joined = scores.merge(source[truth_keys + ["source_rate"]], on=truth_keys, how="left", validate="many_to_one")
    np.testing.assert_allclose(joined.observed_rate, joined.source_rate, rtol=1e-12)
    population = weights.merge(source.rename(columns={"forecast_year": "origin"})[
        ["target", "outcome", "sex", "age", "origin", "count", "source_rate", "implied_population"]],
        on=["target", "outcome", "sex", "age", "origin"], how="left", validate="one_to_one")
    expected_population = population["count"] / population.source_rate * 100000
    np.testing.assert_allclose(population.origin_population, expected_population, rtol=1e-10)
    np.testing.assert_allclose(population.origin_population, population.implied_population, rtol=1e-10)
    totals = population.groupby(["origin", "sex"]).origin_population.transform("sum")
    np.testing.assert_allclose(population.origin_population_weight, population.origin_population/totals, rtol=1e-12)
    verify_roles(points, intervals, mappings)
    interval_count = verify_banks_and_quantiles(points, intervals, draws, config)
    wis_count, interval_roundoff = verify_interval_scores(cells, wis)
    by_origin = verify_summaries(scores, weights, config)
    primary_means = verify_primary_contrasts(scores, contrasts, age_contrasts, verdict, config)
    replay = replay_primary_tcn(scores, config)
    nonneural_replay_path = ROOT / "work/primary-validation/replay_nonneural.json"
    nonneural_replay = json.loads(nonneural_replay_path.read_text())
    assert nonneural_replay["passed"] and not nonneural_replay["models_refitted"] and not nonneural_replay["adaptation_refitted"]
    assert nonneural_replay["run_manifest_sha256"] == sha(RUN / "run_manifest.json")
    assert nonneural_replay["script_sha256"] == sha(ROOT / "work/primary-validation/replay_nonneural.py")
    assert nonneural_replay["replayed_cells"] == 660 and nonneural_replay["unique_checkpoints_loaded"] == 7
    harm = pd.read_csv(RUN / "matched_adaptation_cells.csv")
    np.testing.assert_allclose(harm.adaptation_loss_change, harm.adapted_error-harm.matched_unadapted_error, atol=1e-13)
    original_pairs = scores.loc[scores.family.isin(["tcn_adapted", "tcn_unadapted"])].pivot(
        index=["origin", "sex", "age", "horizon", "ensemble_fingerprint"], columns="family", values="absolute_log_error")
    indexed_harm = harm.set_index(["origin", "sex", "age", "horizon", "ensemble_fingerprint"]).sort_index()
    np.testing.assert_allclose(indexed_harm.adaptation_loss_change,
                               original_pairs.tcn_adapted-original_pairs.tcn_unadapted, atol=1e-13)
    assert len(original_pairs) == 550 and original_pairs.notna().all().all()

    primary_interval = cells.loc[cells.origin.eq(2018) & cells.horizon.eq(5) & cells.scale.eq("rate")].groupby(
        ["sex", "family", "level"], as_index=False).agg(coverage=("covered", "mean"), mean_width=("width", "mean"),
                                                       covered_cells=("covered", "sum"), age_cells=("covered", "size"))
    reliability_interval = cells.loc[cells.horizon.eq(5) & cells.scale.eq("rate")].groupby(
        ["sex", "family", "level"], as_index=False).agg(coverage=("covered", "mean"), mean_width=("width", "mean"),
                                                       covered_cells=("covered", "sum"), age_origin_cells=("covered", "size"))
    primary_wis = wis.loc[wis.origin.eq(2018) & wis.horizon.eq(5) & wis.scale.eq("rate")].groupby(
        ["sex", "family"], as_index=False).agg(wis_50_80=("wis_50_80", "mean"), wis_50_80_95=("wis_50_80_95", "mean"))
    reliability_wis = wis.loc[wis.horizon.eq(5) & wis.scale.eq("rate")].groupby(
        ["sex", "family"], as_index=False).agg(wis_50_80=("wis_50_80", "mean"), wis_50_80_95=("wis_50_80_95", "mean"))
    reliability = pd.read_csv(RUN / "reliability_by_horizon.csv")
    reliability_all = pd.read_csv(RUN / "reliability_all_horizons.csv")
    primary_summary = pd.read_csv(RUN / "primary_by_family.csv")
    adaptation_primary = harm.loc[harm.origin.eq(2018) & harm.horizon.eq(5)].groupby("sex", as_index=False).agg(
        mean_loss_change=("adaptation_loss_change", "mean"), harmed_cells=("harmed", "sum"), age_cells=("harmed", "size"))
    adaptation_reliability = harm.loc[harm.horizon.eq(5)].groupby("sex", as_index=False).agg(
        mean_loss_change=("adaptation_loss_change", "mean"), harmed_cells=("harmed", "sum"), age_origin_cells=("harmed", "size"))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [("primary_interval_diagnostics", primary_interval.merge(primary_wis, on=["sex", "family"])),
                        ("reliability_interval_diagnostics", reliability_interval.merge(reliability_wis, on=["sex", "family"])),
                        ("primary_adaptation_diagnostic", adaptation_primary),
                        ("reliability_adaptation_diagnostic", adaptation_reliability)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)
    (OUT / "tests_at_run.json").write_bytes(tests_path.read_bytes())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, sex in enumerate(config["sexes"]):
        values = [primary_means.loc[sex, family] for family in ROLES]
        axes[0, column].barh(np.arange(3), values, color=[COLORS[family] for family in ROLES])
        axes[0, column].set_yticks(np.arange(3), [LABELS[family] for family in ROLES])
        axes[0, column].invert_yaxis()
        for index, value in enumerate(values):
            axes[0, column].text(value + max(values)*0.025, index, f"{value:.5f}", va="center", fontsize=9)
        axes[0, column].set_xlim(0, max(values)*1.35 if max(values) else 0.01)
        axes[0, column].set_xlabel("Mean absolute log error across eleven ages")
        axes[0, column].set_title(sex)
        coverage = primary_interval.loc[primary_interval.sex.eq(sex) & primary_interval.level.eq(0.8)].set_index("family")
        values = [coverage.loc[family, "coverage"] for family in ROLES]
        axes[1, column].barh(np.arange(3), values, color=[COLORS[family] for family in ROLES])
        axes[1, column].set_yticks(np.arange(3), [LABELS[family] for family in ROLES])
        axes[1, column].invert_yaxis()
        axes[1, column].axvline(0.8, linestyle="--", color="#9f1239", linewidth=1.6)
        for index, value in enumerate(values):
            axes[1, column].text(value+0.015, index, f"{value:.1%}", va="center", fontsize=9)
        axes[1, column].set_xlim(0, 1.13)
        axes[1, column].set_xticks(np.arange(0, 1.01, 0.2))
        axes[1, column].xaxis.set_major_formatter(PercentFormatter(1))
        axes[1, column].set_xlabel("Central 80% rate interval coverage (11 ages)")
        for axis in axes[:, column]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="x", alpha=0.15)
            axis.set_axisbelow(True)
    common_error_limit = max(primary_means.loc[sex, family] for sex in config["sexes"] for family in ROLES) * 1.35
    for axis in axes[0]:
        axis.set_xlim(0, common_error_limit if common_error_limit else 0.01)
    fig.suptitle("Saudi primary endpoint: 2018-origin forecasts of 2023 prevalence", fontsize=14)
    fig.text(0.5, 0.015, "Dashed line: nominal 80% interval level. Age cells are dependent.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    fig.savefig(OUT / "primary_summary.png", dpi=180)
    fig.savefig(OUT / "primary_summary.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    largest = 0
    for axis, sex in zip(axes, config["sexes"]):
        for family in ROLES:
            part = scores.loc[scores.origin.eq(2018) & scores.horizon.eq(5) & scores.sex.eq(sex) & scores.family.eq(family)].set_index("age").reindex(config["ages"])
            axis.plot(np.arange(len(config["ages"])), part.absolute_log_error, color=COLORS[family], marker="o", markersize=3, label=LABELS[family])
            largest = max(largest, part.absolute_log_error.max())
        axis.set_title(sex)
        axis.set_xticks(np.arange(len(config["ages"])), config["ages"], rotation=45, ha="right")
        axis.set_xlabel("Age band")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, largest*1.08 if largest else 0.01)
    axes[0].set_ylabel("Absolute log error at the 2023 endpoint")
    fig.suptitle("Saudi primary endpoint: age-specific forecast errors", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(OUT / "primary_age_errors.png", dpi=180)
    fig.savefig(OUT / "primary_age_errors.svg")
    plt.close(fig)

    primary_lines = ["| Sex | Frozen comparator | Adapted TCN error | Comparator error | TCN − comparator | Relative improvement |",
                     "|---|---|---:|---:|---:|---:|"]
    for row in contrasts.itertuples():
        relative = "Undefined: zero comparator error" if pd.isna(row.relative_improvement_percent) else f"{row.relative_improvement_percent:+.2f}%"
        primary_lines.append(f"| {row.sex} | {LABELS[row.comparator_source_family]} | {row.tcn_error:.6f} | {row.comparator_error:.6f} | {row.absolute_loss_difference:+.6f} | {relative} |")
    all_lines = ["| Underlying family | Male primary error | Female primary error |", "|---|---:|---:|"]
    for family in manifest["families"]:
        all_lines.append(f"| {LABELS[family]} | {primary_means.loc['Male', family]:.6f} | {primary_means.loc['Female', family]:.6f} |")
    reliability_lines = ["| Sex | Procedure | 2014 | 2015 | 2016 | 2017 | 2018 | Five-origin mean, h5 | Mean over origins and h1–5 |",
                         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for sex in config["sexes"]:
        for family in ROLES:
            values = by_origin.loc[by_origin.sex.eq(sex) & by_origin.family.eq(family) & by_origin.horizon.eq(5)].set_index("origin").mean_age_absolute_log_error
            average = reliability.loc[reliability.sex.eq(sex) & reliability.family.eq(family) & reliability.horizon.eq(5), "mean_absolute_log_error"].item()
            average_all = reliability_all.loc[reliability_all.sex.eq(sex) & reliability_all.family.eq(family), "mean_absolute_log_error"].item()
            reliability_lines.append(f"| {sex} | {LABELS[family]} | " + " | ".join(f"{values.loc[origin]:.6f}" for origin in range(2014, 2019)) + f" | {average:.6f} | {average_all:.6f} |")
    interval_lines = ["| Sex | Procedure | Primary 50% / 80% / 95% coverage | Primary 80% width | Primary 50/80 WIS | Reliability 80% coverage | Reliability 50/80 WIS |",
                      "|---|---|---|---:|---:|---:|---:|"]
    for sex in config["sexes"]:
        for family in ROLES:
            primary_ci = primary_interval.loc[primary_interval.sex.eq(sex) & primary_interval.family.eq(family)].set_index("level")
            reliability_ci = reliability_interval.loc[reliability_interval.sex.eq(sex) & reliability_interval.family.eq(family) & reliability_interval.level.eq(0.8)].iloc[0]
            primary_score = primary_wis.loc[primary_wis.sex.eq(sex) & primary_wis.family.eq(family), "wis_50_80"].item()
            reliability_score = reliability_wis.loc[reliability_wis.sex.eq(sex) & reliability_wis.family.eq(family), "wis_50_80"].item()
            coverage = " / ".join(f"{primary_ci.loc[level, 'coverage']:.1%}" for level in [0.5, 0.8, 0.95])
            interval_lines.append(f"| {sex} | {LABELS[family]} | {coverage} | {primary_ci.loc[0.8, 'mean_width']:.3f} | {primary_score:.3f} | {reliability_ci.coverage:.1%} | {reliability_score:.3f} |")
    weight_lines = ["| Sex | Procedure | Equal-age primary error | Origin-population-weighted error |", "|---|---|---:|---:|"]
    for sex in config["sexes"]:
        for family in ROLES:
            row = primary_summary.loc[primary_summary.sex.eq(sex) & primary_summary.family.eq(family)].iloc[0]
            weight_lines.append(f"| {sex} | {LABELS[family]} | {row.mean_age_absolute_log_error:.6f} | {row.population_weighted_absolute_log_error:.6f} |")
    harm_lines = ["| Sex | Primary mean error change | Primary harmed ages | Reliability mean error change | Reliability harmed cells |",
                  "|---|---:|---:|---:|---:|"]
    for sex in config["sexes"]:
        primary_harm = adaptation_primary.loc[adaptation_primary.sex.eq(sex)].iloc[0]
        overall_harm = adaptation_reliability.loc[adaptation_reliability.sex.eq(sex)].iloc[0]
        harm_lines.append(f"| {sex} | {primary_harm.mean_loss_change:+.6f} | {int(primary_harm.harmed_cells)}/11 | {overall_harm.mean_loss_change:+.6f} | {int(overall_harm.harmed_cells)}/55 |")
    success_text = "met" if verdict["joint_success"] else "not met"
    successful = [sex.lower() + "s" for sex in config["sexes"] if verdict["success_by_sex"][sex]]
    sex_text = " and ".join(successful) if successful else "neither sex"
    sex_result = f"The adapted TCN had strictly lower error against both frozen comparator families for {sex_text}." if successful else "Neither sex had strictly lower adapted-TCN error against both frozen comparators."
    female_comparisons = contrasts.loc[contrasts.sex.eq("Female")].set_index("comparator")
    female_local = female_comparisons.loc["local_champion"]
    female_nonneural = female_comparisons.loc["nonneural_champion"]
    if female_local.strictly_lower and not female_nonneural.strictly_lower and female_nonneural.relative_improvement_percent < 0:
        natural_name = {"donor_ridge_adapted": "adapted donor ridge"}.get(
            female_nonneural.comparator_source_family, LABELS[female_nonneural.comparator_source_family])
        sex_result += (f" For females, it beat {LABELS[female_local.comparator_source_family]} but had "
                       f"{-female_nonneural.relative_improvement_percent:.2f}% higher error than {natural_name}.")
    interval_under = primary_interval.loc[primary_interval.family.isin(ROLES) & primary_interval.level.eq(0.8), "coverage"].lt(0.8).sum()
    report = f"""# Stage 5 — Saudi primary and nested reliability evaluation

The prespecified joint primary criterion was **{success_text}**. {sex_result} This is a comparison of forecast errors, not a claim of statistical or clinical significance.

The primary endpoint is Saudi GBD-estimated prevalence in **2023**, forecast from origin **2018**, evaluated separately for males and females across eleven age bands from 45–49 to 95+. Lower mean absolute log error is better. Both local and non-neural comparator families and all settings were frozen using eligible historical evidence before these forecasts were scored.

## Primary result

{chr(10).join(primary_lines)}

Negative absolute differences and positive relative improvements favor the adapted TCN. Relative improvement divides the absolute improvement by comparator error; a zero comparator error makes that percentage undefined. The primary criterion requires all four absolute differences to be strictly negative. Ties do not satisfy it. The method remains the all-six-donor, five-seed TCN with two target coefficients per sex, regardless of which baseline performed best after evaluation.

![Primary point error and interval coverage](primary_summary.png)

All fourteen underlying families are reported below; the champion roles above are views of selected family forecasts and are not additional fits.

{chr(10).join(all_lines)}

![Primary age-specific errors](primary_age_errors.png)

Every age-specific prediction and error is retained in [primary age scores](../../results/primary_v1/primary_age_scores.csv); the [paired age contrasts](../../results/primary_v1/primary_age_contrasts.csv) show where transfer increased or reduced error. Age cells are dependent and do not constitute independent participants.

## Nested reliability, kept separate from the primary endpoint

The five reliability origins are 2014–2018, with horizon-five endpoints 2019–2023. At each origin, settings and comparator families use only historical blocks completed by that origin. The origin-2018 forecast appears **once** in this sequence and supplies the primary endpoint above; it was not refitted as a second independent test.

{chr(10).join(reliability_lines)}

The five-origin mean and the mean over horizons 1–5 are secondary summaries. They do not replace a failed primary contrast or add independent replications. [All origins/horizons](../../results/primary_v1/by_origin.csv), [all-family reliability means](../../results/primary_v1/reliability_by_horizon.csv), and [original-rate errors](../../results/primary_v1/primary_by_family.csv) remain available.

## Interval reliability

These intervals use the frozen, jointly paired historical residual banks: 7–11 complete blocks at origins 2014–2018, including eleven blocks for the primary forecast. Points and intervals were saved before verification outcomes were joined. No interval widening or method tuning followed the evaluation.

{chr(10).join(interval_lines)}

Primary coverage is over **11 dependent age cells per sex/procedure**; pooled reliability coverage is over **55 dependent age–origin cells**. Width and WIS above are on the rate scale, per 100,000 population. Of the six primary sex/procedure summaries, **{int(interval_under)} were below nominal 80% coverage**. Reported coverage must be read alongside interval width and WIS; point accuracy alone does not demonstrate calibration.

The principal WIS includes central 50% and 80% intervals and their stated-scale predictive median. The 95% intervals and 50/80/95 WIS are supplementary because tail information is sparse. Log-rate and rate quantiles are interpolated separately; a rate quantile is not obtained by exponentiating an interpolated log quantile. Source GBD bounds and seed spread are not substituted for forecast uncertainty. See [all interval diagnostics](primary_interval_diagnostics.csv), [reliability diagnostics](reliability_interval_diagnostics.csv), and the [WIS methodology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618) with its [correction](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010592).

## Secondary weighting and matched adaptation

Population weights use the **issuance-year prevalence Number/Rate-implied GBD denominator**, cross-checked against the prepared implied population. They sum to one over ages 45+ within each sex and remain identical across methods and horizons. Later population observations do not enter these weights. This is a modeled GBD-vintage denominator, not an independently observed population or a count forecast; the primary estimand remains equal-age error.

{chr(10).join(weight_lines)}

Adaptation harm is compared against the **same five-seed neural ensemble without correction**, verified by its shared fingerprint. Negative changes mean that correction reduced error.

{chr(10).join(harm_lines)}

Harm fractions describe dependent cells. Independently tuned non-neural adapted and unadapted families are not presented as matched checkpoint experiments here. Further donor selection and demographic analyses remain separate.

## Validation and limitations

- All **{tests['tests_run']} stage-5 tests passed**. The checks cover complete chronological forecast grids, frozen choices, role/source equality, strict/tied/zero-comparator primary cases, consistent origin population weights, exact matched ensembles, and committing point/interval artifacts before scoring.
- Saved **{validation['unique_model_forecasts']:,} underlying forecast cells** and **{validation['champion_view_forecasts']:,} explicit champion-view cells**, **{validation['interval_rows']:,} interval rows**, and **{validation['joint_draw_rows']:,} joint draw rows**. Origin 2018 was fitted once; all five prescribed neural seeds are retained.
- Selected underlying persistence-fallback cells: **{validation['selected_fallback_cells']}**. TCN fit failures: **{validation['tcn_fit_failures']}**; non-neural fit failures: **{validation['nonneural_fit_failures']}**; local age-fit fallbacks: **{validation['local_age_fit_fallbacks']}**; TCN correction failures: **{validation['tcn_adaptation_failures']}**. Failed fits retain their status and forecasts.
- Independent reporting reproduced the four primary contrasts and strict verdict, role/source equality, population weights, all **{interval_count:,} interval quantiles**, all **{wis_count:,} WIS rows**, and **{replay['primary_tcn_replayed_cells']} primary-origin TCN forecasts** from **{replay['frozen_checkpoints_replayed']} frozen checkpoints** without refitting. A [separate non-neural replay](../../work/primary-validation/replay_nonneural.json) reproduced **660 forecast cells** from seven saved checkpoints. Raw/design/code hashes and all four earlier run manifests remained unchanged. Report verification records negligible CSV round-trip differences using explicit machine-precision bounds; these checks do not alter metrics or the strict primary criterion.
- Four CPU workers in `agpu` completed the run in **{validation['elapsed_seconds']:.1f} seconds**, using the stage-3 numerical device and software. Settings, family mappings, weights, point forecasts, and interval draws are protected by the [pre-score artifact commitment](../../results/primary_v1/pre_score_commit.json).

This is a retrospective forecast evaluation against revised GBD estimates. Descriptive 2023 data were inspected during study design, so the final evaluation is not described as blinded. The short, overlapping history limits independent information. Residual centering removes historical average error and may miss later bias; sparse empirical intervals have no exact coverage guarantee. There are no age-cell t-tests, seed-based epidemiological significance tests, causal biological claims, survival analyses, or individual patient predictions.

## Next stage and artifacts

Stage 6 evaluates incidence, alternative donor pools, other GCC targets, demographic/population scenarios, and count aggregation under their prespecified rules. The Saudi prevalence primary result above remains fixed; secondary analyses will not replace it.

- [Issued point forecasts](../../results/primary_v1/predictions.csv), [point scores](../../results/primary_v1/point_scores.csv), [primary contrasts](../../results/primary_v1/primary_contrasts.csv), and [strict criterion verdict](../../results/primary_v1/primary_verdict.json).
- [Intervals](../../results/primary_v1/intervals.csv), [joint draws](../../results/primary_v1/joint_draws.csv), [interval scores](../../results/primary_v1/interval_scores.csv), [WIS scores](../../results/primary_v1/wis_scores.csv), and [bank references](../../results/primary_v1/bank_references.csv).
- [Frozen baseline settings](../../results/primary_v1/settings_decisions.csv), [TCN settings](../../results/primary_v1/tcn_choices.json), [champion mappings](../../results/primary_v1/champion_family_mappings.csv), [population weights](../../results/primary_v1/origin_population_weights.csv), and [matched adaptation](../../results/primary_v1/matched_adaptation_summary.csv).
- [Run validation](../../results/primary_v1/validation_report.json), [immutable manifest](../../results/primary_v1/run_manifest.json), [test evidence](tests_at_run.json), [report verification](report_validation.json), [implementation specification](../../study_design/primary_evaluation_implementation.md), and [independent review](../../work/primary-validation/review.md).

```bash
/home/saif/agpu_env/bin/python tests/test_primary.py
/home/saif/agpu_env/bin/python scripts/run_primary.py --output results/primary_reproduction
/home/saif/agpu_env/bin/python scripts/report_primary.py
```

The runner refuses an existing result directory. The report reads the original `primary_v1` artifacts without fitting or changing results.
"""
    (OUT / "report.md").write_text(report)
    verification = {"passed": True, "all_run_code_source_and_prior_hashes_verified": True,
                    "settings_points_intervals_committed_before_scoring": True,
                    "role_point_and_interval_equality_verified": True,
                    "independent_primary_contrasts_and_strict_verdict": True,
                    "origin_population_weights_verified": True,
                    "point_prediction_csv_roundtrip_max_absolute_difference": prediction_roundtrip_difference,
                    "interval_score_csv_roundtrip_audit": interval_roundoff,
                    "independent_interval_quantile_rows": interval_count, "independent_wis_rows": wis_count,
                    "maximum_scored_year": 2023, **replay,
                    "nonneural_replay_passed": nonneural_replay["passed"],
                    "nonneural_replayed_cells": nonneural_replay["replayed_cells"],
                    "nonneural_replay_sha256": sha(nonneural_replay_path),
                    "run_manifest_sha256": sha(RUN / "run_manifest.json"),
                    "report_script_sha256": sha(Path(__file__)),
                    "output_sha256": {path.name: sha(path) for path in sorted(OUT.iterdir())
                                      if path.is_file() and path.name != "report_validation.json"}}
    (OUT / "report_validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
