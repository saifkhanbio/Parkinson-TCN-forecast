"""Independent reconstruction and reporting of exploratory interval calibration."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
import os
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
from run_local_baselines import sha, check_lock
from report_demography import expected_statistics, STAT

RUN = ROOT / "results/calibration_v1_2"
OUT = ROOT / "reports/calibration_v1_2"
COORDS = ["sex", "age", "horizon"]
RATE_KEYS = ["target", "outcome", "origin", "family"] + COORDS + ["forecast_year", "scale"]
MAIN = ["original", "target_only", "gcc_assisted"]
ROLES = ["tcn_adapted", "local_champion", "nonneural_champion"]
LABELS = {"tcn_adapted": "Adapted TCN", "local_champion": "Local comparator", "nonneural_champion": "Non-neural comparator",
          "original": "Original", "target_only": "Target only", "gcc_assisted": "GCC assisted"}
COLORS = {"original": "#778899", "target_only": "#1780a4", "gcc_assisted": "#c47932"}


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def json_read(path):
    return json.loads(Path(path).read_text())


def compare(new, old, keys, fields, exact=False):
    assert not new.duplicated(keys).any() and not old.duplicated(keys).any()
    a, b = new.set_index(keys).sort_index(), old.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    if exact:
        np.testing.assert_array_equal(a[fields], b[fields])
    else:
        np.testing.assert_allclose(a[fields], b[fields], rtol=2e-12, atol=1e-9)
    return float(np.max(abs(a[fields].to_numpy()-b[fields].to_numpy())))


def verify_integrity(amendment):
    manifest = json_read(RUN / "run_manifest.json")
    assert manifest["status"] == "complete" and manifest["final_period_scored"]
    assert not manifest["point_models_refitted"] and not manifest["population_refitted"]
    for name, expected in manifest["output_sha256"].items():
        assert sha(RUN / name) == expected, name
    for name, expected in manifest["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    lock = ROOT / "study_design/interval_calibration_v1_2.lock.json"
    assert sha(lock) == manifest["amendment_lock_sha256"]
    document = json_read(lock)
    for name, expected in document["document_sha256"].items():
        assert sha(ROOT / name) == expected
    for run in amendment["prior_runs"]:
        directory = ROOT / "results" / run
        assert sha(directory / "run_manifest.json") == manifest["prior_manifest_sha256"][run] == document["prior_manifest_sha256"][run]
        prior = json_read(directory / "run_manifest.json")
        for name, expected in prior["output_sha256"].items():
            assert sha(directory / name) == expected, (run, name)
        for name, expected in prior["code_sha256"].items():
            assert sha(ROOT / name) == expected, name
    assert json_read(ROOT / "results/primary_v1/primary_verdict.json")["joint_success"] is False
    events = json_read(RUN / "events.json")
    names = [row["event"] for row in events]
    order = ["development_losses_committed", "all_cases_committed_before_scoring", "evaluation_scoring_started", "complete"]
    assert all(names.count(name) == 1 for name in order)
    indices = [names.index(name) for name in order]
    assert indices == sorted(indices)
    times = [datetime.fromisoformat(events[index]["time_utc"]) for index in indices]
    assert times == sorted(times)
    global_commit = json_read(RUN / "global_issued_commit.json")
    assert len(global_commit["case_commit_sha256"]) == 12
    assert global_commit["development_losses_sha256"] == sha(RUN / "development_losses.csv")
    assert sha(RUN / "global_issued_commit.json") == events[indices[1]]["sha256"]
    global_time = datetime.fromisoformat(global_commit["committed_utc"])
    assert global_time <= times[2]
    for name, expected in global_commit["case_commit_sha256"].items():
        path = RUN / name
        assert sha(path) == expected
        commit = json_read(path)
        assert not commit["final_period_scored"]
        assert datetime.fromisoformat(commit["committed_utc"]) <= global_time
        for filename, digest in commit["artifact_sha256"].items():
            assert sha(path.parent / filename) == digest
        scored = json_read(path.parent / "scoring_event.json")
        assert scored["global_commit_sha256"] == sha(RUN / "global_issued_commit.json")
        assert global_time <= datetime.fromisoformat(scored["scoring_started_utc"])
    return manifest


def wis_from_draws(draws, observed):
    lower50, upper50, lower80, upper80, median = np.quantile(draws, [.25, .75, .1, .9, .5], axis=0, method="linear")
    is50 = upper50-lower50+4*(np.maximum(lower50-observed, 0)+np.maximum(observed-upper50, 0))
    is80 = upper80-lower80+10*(np.maximum(lower80-observed, 0)+np.maximum(observed-upper80, 0))
    return (.5*abs(observed-median)+.25*is50+.1*is80)/2.5


def development_curves(history, case, config, amendment):
    records = []
    families = config["models"]["local_order"]+config["models"]["nonneural_order"]+["tcn_adapted", "tcn_unadapted", "tcn_intercept"]
    for family in families:
        source = history[history.family.eq(family)]
        for origin in amendment["development_origins"]:
            origins = list(range(2003, origin-4))
            for sex in config["sexes"]:
                past = source[source.sex.eq(sex) & source.origin.isin(origins)]
                expected = pd.MultiIndex.from_product([origins, config["ages"], range(1, 6)], names=["origin", "age", "horizon"])
                assert len(past) == len(expected) and not past.duplicated(["origin", "age", "horizon"]).any()
                past = past.set_index(["origin", "age", "horizon"]).reindex(expected)
                assert (past.forecast_year <= origin).all()
                raw = (np.log(past.observed_rate.to_numpy())-past.log_prediction.to_numpy()).reshape(len(origins), 11, 5)[:, :, 4]
                normalizer = max(float(abs(raw).mean()), amendment["normalization_floor"])
                centered = raw-raw.mean(axis=0)
                current = source[source.origin.eq(origin) & source.sex.eq(sex) & source.horizon.eq(5)].set_index("age").reindex(config["ages"])
                assert current.forecast_year.eq(origin+5).all() and len(current) == 11
                observed = np.log(current.observed_rate.to_numpy())
                for factor in amendment["factor_grid"]:
                    draws = current.log_prediction.to_numpy()[None, :]+factor*centered
                    loss = float(wis_from_draws(draws, observed).mean())
                    records.append(dict(country=case["target"], outcome=case["outcome"], family=family, sex=sex,
                                        development_origin=origin, factor=factor, normalizer=normalizer,
                                        mean_log_wis=loss, normalized_wis=loss/normalizer,
                                        bank_n_blocks=len(origins), last_label_year=origin+5,
                                        last_residual_label_year=max(origins)+5))
    return pd.DataFrame(records)


def verify_choices(choices, losses, mappings, config, amendment):
    assert len(choices) == 60
    seen = set()
    for choice in choices:
        target, outcome, family, sex, origin, policy = [choice[key] for key in ["target", "outcome", "source_family", "sex", "fit_origin", "policy"]]
        key = (origin, choice["role"], sex, policy)
        assert key not in seen
        seen.add(key)
        source_family = "tcn_adapted"
        if choice["role"] != "tcn_adapted":
            mapping = mappings[mappings.fit_origin.eq(origin) & mappings.role.eq(choice["role"]) & mappings.sex.eq(sex)]
            assert len(mapping) == 1 and mapping.last_selection_target_year.le(origin).all()
            source_family = mapping.source_family.iloc[0]
        assert family == source_family
        origins = [u for u in amendment["development_origins"] if u+5 <= origin]
        assert origins == choice["eligible_origins"]
        donors = [country for country in amendment["countries"] if country != target] if policy == "gcc_assisted" else []
        assert choice["donor_countries"] == donors
        weights = {target: .5, **{country: .1 for country in donors}} if donors else {target: 1.}
        assert choice["country_weights"] == weights and target not in donors and "Jordan" not in donors
        if not origins:
            assert choice["factor"] == 1 and choice["last_label_year"] is None and not choice["candidate_objectives"]
            continue
        assert choice["last_label_year"] == max(origins)+5 <= origin
        expected_choices = []
        for factor in amendment["factor_grid"]:
            part = losses[losses.country.isin(weights) & losses.outcome.eq(outcome) & losses.family.eq(family)
                          & losses.sex.eq(sex) & losses.development_origin.isin(origins) & losses.factor.eq(factor)]
            assert len(part) == len(weights)*len(origins)
            country_means = part.groupby("country").normalized_wis.mean().to_dict()
            target_loss = country_means[target]
            donor_loss = float(np.mean([country_means[country] for country in donors])) if donors else None
            unpenalized = .5*target_loss+.5*donor_loss if donors else target_loss
            penalty = amendment["penalty"]*np.log(factor)**2
            objective = unpenalized+penalty
            recorded = next(row for row in choice["candidate_objectives"] if row["factor"] == factor)
            np.testing.assert_allclose([recorded["target_loss"], recorded["unpenalized_loss"], recorded["penalty"], recorded["objective"]],
                                       [target_loss, unpenalized, penalty, objective], atol=1e-12, rtol=1e-12)
            if donors:
                np.testing.assert_allclose(recorded["donor_loss"], donor_loss, atol=1e-12)
            for country in weights:
                np.testing.assert_allclose(recorded["country_losses"][country], country_means[country], atol=1e-12)
            expected_choices.append((objective, factor))
        objective, factor = min(expected_choices)
        assert choice["factor"] == factor
        np.testing.assert_allclose(choice["objective"], objective, atol=1e-12)


def score_audit(cells, wis, keys, observed_field):
    observed = cells[observed_field].to_numpy()
    lower, upper, alpha = cells.lower.to_numpy(), cells.upper.to_numpy(), 1-cells.level.to_numpy()
    width = upper-lower
    score = width+2/alpha*(np.maximum(lower-observed, 0)+np.maximum(observed-upper, 0))
    np.testing.assert_allclose(cells.width, width, atol=1e-9, rtol=2e-12)
    np.testing.assert_allclose(cells.interval_score, score, atol=1e-8, rtol=2e-12)
    np.testing.assert_array_equal(cells.covered.astype(bool), (observed >= lower) & (observed <= upper))
    temporary = cells[keys+['level']].copy()
    temporary["weighted_score"] = alpha*score/2
    terms = temporary.pivot(index=keys, columns="level", values="weighted_score")
    index = wis.set_index(keys).reindex(terms.index)
    truths = cells.drop_duplicates(keys).set_index(keys).reindex(index.index)
    np.testing.assert_array_equal(index[observed_field], truths[observed_field])
    for name, levels in [("wis_50_80", [.5, .8]), ("wis_50_80_95", [.5, .8, .95])]:
        expected = (.5*abs(index[observed_field]-index["median"])+terms[levels].sum(axis=1))/(len(levels)+.5)
        np.testing.assert_allclose(index[name], expected, atol=1e-8, rtol=2e-12)


def rate_summary(cells, wis, config):
    summaries = []
    scopes = {"45+": config["ages"], **{group: [age for age in config["ages"] if int(age.split("-")[0].rstrip("+")) in starts]
                                        for group, starts in config["age_groups"].items()},
              **{"age:"+age: [age] for age in config["ages"]}}
    groups = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "scale"]
    for scope, ages in scopes.items():
        frame = cells[cells.age.isin(ages)].copy()
        frame["lower_miss"] = frame.observed_value < frame.lower
        frame["upper_miss"] = frame.observed_value > frame.upper
        summary = frame.groupby(groups+["level"], as_index=False).agg(coverage=("covered", "mean"), width=("width", "mean"),
                lower_miss=("lower_miss", "mean"), upper_miss=("upper_miss", "mean"), interval_score=("interval_score", "mean"))
        proper = wis[wis.age.isin(ages)].groupby(groups, as_index=False).agg(wis_50_80=("wis_50_80", "mean"), wis_50_80_95=("wis_50_80_95", "mean"))
        summary = summary.merge(proper, on=groups, validate="many_to_one")
        summary["age_scope"] = scope
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def audit_case(case, config, amendment):
    directory = RUN / case["id"]
    for name, digest in json_read(directory / "source_references.json").items():
        assert sha(ROOT / name) == digest
    history = read(ROOT / case["history"] / "prequential_scores.csv")
    recalculated = development_curves(history, case, config, amendment)
    all_losses = read(RUN / "development_losses.csv")
    loss = all_losses[all_losses.country.eq(case["target"]) & all_losses.outcome.eq(case["outcome"])]
    keys = ["country", "outcome", "family", "sex", "development_origin", "factor"]
    compare(recalculated, loss, keys, ["normalizer", "mean_log_wis", "normalized_wis", "bank_n_blocks", "last_label_year", "last_residual_label_year"])
    choices = json_read(directory / "factor_choices.json")
    mappings = read(ROOT / case["source"] / "champion_family_mappings.csv")
    verify_choices(choices, all_losses, mappings, config, amendment)
    points = read(directory / "original_point_predictions.csv")
    original_points = read(ROOT / case["source"] / "predictions.csv")
    original_points = original_points[original_points.family.isin(ROLES)]
    point_keys = ["target", "outcome", "origin", "family"]+COORDS+["forecast_year"]
    compare(points, original_points, point_keys, ["prediction", "log_prediction"], exact=True)
    intervals, draws = read(directory / "rate_intervals.csv"), read(directory / "rate_draws.csv.gz")
    assert len(points) == 1650 and len(intervals) == 69300 and len(draws) == 103950
    for frame in [intervals, draws]:
        assert frame.family.eq(frame.role+"__"+frame.variant).all()
        assert set(frame.variant) == set(amendment["fixed_variants"]+["target_only", "gcc_assisted"])
    coords = pd.MultiIndex.from_product([config["sexes"], config["ages"], range(1, 6)], names=COORDS)
    references = {(row["origin"], row["role"]): row for row in json_read(directory / "bank_references.json")}
    assert set(references) == {(origin, role) for origin in amendment["evaluation_origins"] for role in ROLES}
    for (origin, role), reference in references.items():
        assert sha(ROOT / reference["path"]) == reference["sha256"]
        bank = joblib.load(ROOT / reference["path"])
        residual_origins = list(range(2003, origin-4))
        assert bank["origins"] == residual_origins and bank["n_blocks"] == len(residual_origins)
        assert list(map(tuple, bank["coords"].to_numpy())) == list(coords)
        raw_sexes = []
        for sex in config["sexes"]:
            source_family = reference["source_family_by_sex"][sex]
            source = history[history.family.eq(source_family) & history.sex.eq(sex) & history.origin.isin(residual_origins)]
            index = pd.MultiIndex.from_product([residual_origins, config["ages"], range(1, 6)], names=["origin", "age", "horizon"])
            assert len(source) == len(index) and not source.duplicated(["origin", "age", "horizon"]).any()
            source = source.set_index(["origin", "age", "horizon"]).reindex(index)
            assert source.forecast_year.le(origin).all()
            raw_sexes.append((np.log(source.observed_rate)-source.log_prediction).to_numpy().reshape(len(residual_origins), 55))
        raw = np.concatenate(raw_sexes, axis=1)
        center = raw.mean(axis=0)
        np.testing.assert_allclose(bank["raw_residuals"], raw, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(bank["centered_residuals"], raw-center, atol=1e-12, rtol=1e-12)
        current = points[points.origin.eq(origin) & points.family.eq(role)].set_index(COORDS).reindex(coords)
        for variant in amendment["fixed_variants"]+["target_only", "gcc_assisted"]:
            if variant in amendment["fixed_variants"]:
                fixed = amendment["factor_grid"][amendment["fixed_variants"].index(variant)]
                factors = dict.fromkeys(config["sexes"], fixed)
            else:
                factors = {row["sex"]: row["factor"] for row in choices if row["fit_origin"] == origin and row["role"] == role and row["policy"] == variant}
            factor_vector = np.asarray([factors[sex] for sex, _, _ in coords])
            log_draws = current.log_prediction.to_numpy()[None, :]+(raw-center)*factor_vector[None, :]
            selected = draws[draws.origin.eq(origin) & draws.role.eq(role) & draws.variant.eq(variant)]
            np.testing.assert_array_equal(selected.factor, selected.sex.map(factors))
            for scale, values, field, point_field in [("log_rate", log_draws, "log_draw", "log_prediction"), ("rate", np.exp(log_draws), "rate_draw", "prediction")]:
                actual = selected.pivot(index="residual_origin", columns=COORDS, values=field).reindex(index=residual_origins, columns=coords)
                np.testing.assert_allclose(actual, values, atol=1e-10, rtol=2e-12)
                for level in [.5, .8, .95]:
                    part = intervals[intervals.origin.eq(origin) & intervals.role.eq(role) & intervals.variant.eq(variant)
                                     & intervals.scale.eq(scale) & intervals.level.eq(level)].set_index(COORDS).reindex(coords)
                    expected = np.quantile(values, [(1-level)/2, .5, (1+level)/2], axis=0, method="linear").T
                    np.testing.assert_allclose(part[["lower", "median", "upper"]], expected, atol=1e-10, rtol=2e-12)
                    np.testing.assert_array_equal(part.point_prediction, current[point_field])
    original = intervals[intervals.variant.eq("original")].copy()
    original["family"] = original.role
    previous = read(ROOT / case["source"] / "intervals.csv")
    previous = previous[previous.family.isin(ROLES)]
    rate_difference = compare(original, previous, RATE_KEYS+["level"], ["lower", "median", "upper", "point_prediction"])
    rate_cells, rate_wis = read(directory / "rate_interval_scores.csv.gz"), read(directory / "rate_wis_scores.csv")
    score_audit(rate_cells, rate_wis, RATE_KEYS, "observed_value")
    # Check verification against immutable source scores, not merely internal score arithmetic.
    old_truth = read(ROOT / case["source"] / "point_scores.csv")
    old_truth = old_truth[old_truth.family.isin(ROLES)].rename(columns={"family": "role"})
    truth_keys = ["target", "outcome", "origin", "role"]+COORDS+["forecast_year"]
    observed = rate_cells.merge(old_truth[truth_keys+["observed_rate"]].rename(columns={"observed_rate": "reference_observed"}), on=truth_keys, validate="many_to_one")
    np.testing.assert_allclose(observed.observed_rate, observed.reference_observed, atol=1e-10, rtol=2e-12)
    np.testing.assert_allclose(rate_cells.observed_value, np.where(rate_cells.scale.eq("log_rate"), np.log(rate_cells.observed_rate), rate_cells.observed_rate), atol=1e-12)
    # Rebuild all derived statistics with the unchanged operational population draws.
    demographic = ROOT / case["demography"]
    population, errors = read(demographic / "population_forecasts.csv"), read(demographic / "population_residuals.csv")
    selected_draws = draws[draws.variant.isin(MAIN)]
    population_keys = ["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]
    joint = selected_draws.merge(population[population_keys+["population_method", "log_population"]], on=population_keys)
    assert len(joint) == 2*len(selected_draws)
    error_keys = ["target", "outcome", "sex", "age", "horizon", "residual_origin", "population_method"]
    joint = joint.merge(errors[["fit_origin"]+error_keys+["centered_population_log_error"]].rename(columns={"fit_origin": "origin"}),
                        on=["origin"]+error_keys, validate="many_to_one", how="left")
    assert np.isfinite(joint.centered_population_log_error).all()
    joint["count"] = np.exp(joint.log_draw)*np.exp(joint.log_population+joint.centered_population_log_error)/100000
    rebuilt = expected_statistics(joint, selected_draws, config, True)
    burden_draws = read(directory / "burden_draws.csv.gz")
    assert len(burden_draws) == 164025
    compare(burden_draws, rebuilt, STAT+["residual_origin"], ["value"])
    burden_intervals, burden_points = read(directory / "burden_intervals.csv"), read(directory / "burden_predictions.csv")
    assert len(burden_intervals) == 54675 and len(burden_points) == 18225
    for origin, block in burden_draws.groupby("origin"):
        matrix = block.pivot(index="residual_origin", columns=STAT, values="value")
        assert set(matrix.index) == set(range(2003, origin-4)) and np.isfinite(matrix.to_numpy()).all()
        for level in [.5, .8, .95]:
            current = burden_intervals[burden_intervals.origin.eq(origin) & burden_intervals.level.eq(level)].set_index(STAT).reindex(matrix.columns)
            expected = np.quantile(matrix.to_numpy(), [(1-level)/2, .5, (1+level)/2], axis=0, method="linear").T
            np.testing.assert_allclose(current[["lower", "median", "upper"]], expected, atol=1e-9, rtol=2e-12)
    old_burden_points = read(demographic / "predictions.csv")
    old_burden_points = old_burden_points[old_burden_points.family.isin(ROLES)]
    for variant in MAIN:
        selected = burden_points[burden_points.variant.eq(variant)].copy()
        selected["family"] = selected.role
        compare(selected, old_burden_points, STAT, ["value"], exact=True)
    original = burden_intervals[burden_intervals.variant.eq("original")].copy()
    original["family"] = original.role
    previous = read(demographic / "intervals.csv")
    previous = previous[previous.family.isin(ROLES)]
    burden_difference = compare(original, previous, STAT+["level"], ["lower", "median", "upper"])
    burden_cells, burden_wis = read(directory / "burden_interval_scores.csv.gz"), read(directory / "burden_wis_scores.csv")
    score_audit(burden_cells, burden_wis, STAT, "observed")
    old_truth = read(demographic / "point_scores.csv")
    old_truth = old_truth[old_truth.family.isin(ROLES)].rename(columns={"family": "role"})
    truth_keys = [key if key != "family" else "role" for key in STAT]
    observed = burden_cells.merge(old_truth[truth_keys+["observed"]].rename(columns={"observed": "reference_observed"}), on=truth_keys, validate="many_to_one")
    np.testing.assert_allclose(observed.observed, observed.reference_observed, atol=1e-9, rtol=2e-12)
    rates = rate_summary(rate_cells, rate_wis, config)
    summary_keys = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "scale", "level", "age_scope"]
    emitted = read(directory / "rate_summary.csv").rename(columns={"mean_width": "width", "lower_miss_rate": "lower_miss",
                         "upper_miss_rate": "upper_miss", "mean_interval_score": "interval_score"})
    compare(rates, emitted, summary_keys, ["coverage", "width", "lower_miss", "upper_miss", "interval_score"])
    proper = read(directory / "rate_wis_summary.csv").rename(columns={"mean_wis_50_80": "wis_50_80", "mean_wis_50_80_95": "wis_50_80_95"})
    compare(rates[rates.level.eq(.8)], proper, [key for key in summary_keys if key != "level"], ["wis_50_80", "wis_50_80_95"])
    burden_keys = ["target", "outcome", "origin", "role", "variant", "horizon", "population_method", "measure", "node", "sex", "age_group", "unit"]
    burden = burden_cells.groupby(burden_keys+["level"], as_index=False).agg(coverage=("covered", "mean"), width=("width", "mean"),
                    lower_miss=("lower_miss", "mean"), upper_miss=("upper_miss", "mean"), interval_score=("interval_score", "mean"))
    proper = burden_wis.groupby(burden_keys, as_index=False).agg(wis_50_80=("wis_50_80", "mean"), wis_50_80_95=("wis_50_80_95", "mean"))
    burden = burden.merge(proper, on=burden_keys, validate="many_to_one")
    validation = dict(passed=True, case=case["id"], development_losses_reconstructed=len(recalculated), choices_replayed=len(choices),
                      rate_draws_reconstructed=len(draws), rate_interval_rows_reconstructed=len(intervals),
                      burden_draws_reconstructed=len(burden_draws), burden_interval_rows_reconstructed=len(burden_intervals),
                      rate_wis_rows_recalculated=len(rate_wis), burden_wis_rows_recalculated=len(burden_wis),
                      original_rate_max_difference=rate_difference, original_burden_max_difference=burden_difference,
                      original_points_exact=True, population_inputs_unchanged=True)
    return validation, rates, burden, pd.DataFrame(choices)


METRICS = ["coverage", "width", "lower_miss", "upper_miss", "interval_score", "wis_50_80", "wis_50_80_95"]


def across_origins(frame, keys):
    """Equal-origin descriptive averages, without treating origins as independent."""
    return frame.groupby(keys, as_index=False)[METRICS].mean()


def burden_endpoints(frame):
    subsets = []
    definitions = [("Both-sex 45+ count", frame.measure.eq("count") & frame.node.eq("Both__45+")),
                   ("Male 65+ share", frame.measure.eq("age_share") & frame.node.eq("Male__65+_within_45+")),
                   ("Female 65+ share", frame.measure.eq("age_share") & frame.node.eq("Female__65+_within_45+")),
                   ("Male 80+ share", frame.measure.eq("age_share") & frame.node.eq("Male__80+_within_45+")),
                   ("Female 80+ share", frame.measure.eq("age_share") & frame.node.eq("Female__80+_within_45+")),
                   ("Age-specific M/F ratio", frame.measure.eq("sex_rate_ratio"))]
    for endpoint, mask in definitions:
        part = frame[mask].copy()
        part["endpoint"] = endpoint
        keys = ["target", "outcome", "origin", "role", "variant", "horizon", "population_method", "level", "endpoint", "unit"]
        subsets.append(across_origins(part, keys))
    return pd.concat(subsets, ignore_index=True)


def paired_changes(frame, keys):
    original = frame[frame.variant.eq("original")][keys+METRICS]
    changes = frame.merge(original, on=keys, suffixes=("", "_original"), validate="many_to_one")
    changes["coverage_change_pp"] = 100*(changes.coverage-changes.coverage_original)
    changes["width_ratio"] = changes.width/changes.width_original
    changes["wis_change_percent"] = 100*(changes.wis_50_80/changes.wis_50_80_original-1)
    return changes


def markdown(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"]*len(headers)) + " |\n" + "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def plot_rates(frame, scope):
    part = frame[frame.target.eq("Saudi Arabia") & frame.horizon.eq(5) & frame.level.eq(.8)
                 & frame.scale.eq("rate") & frame.age_scope.eq(scope) & frame.variant.isin(MAIN)]
    mean = across_origins(part, ["outcome", "sex", "role", "variant"])
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 7), sharey=True, constrained_layout=True)
    x = np.arange(3)
    for row, outcome in enumerate(["prevalence", "incidence"]):
        for col, sex in enumerate(["Male", "Female"]):
            ax = axes[row, col]
            selected = mean[mean.outcome.eq(outcome) & mean.sex.eq(sex)].set_index(["role", "variant"])
            for i, variant in enumerate(MAIN):
                heights = [selected.loc[(role, variant), "coverage"] for role in ROLES]
                bars = ax.bar(x+(i-1)*.25, heights, width=.24, color=COLORS[variant], label=LABELS[variant])
                ax.bar_label(bars, labels=[f"{value:.0%}" for value in heights], fontsize=8, padding=3)
            ax.axhline(.8, color="#222222", linestyle="--", linewidth=1)
            ax.set_xticks(x, ["TCN", "Local", "Non-neural"])
            ax.set_ylim(0, 1.09)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(f"{outcome.capitalize()} · {sex.lower()}", fontsize=11)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel("80% interval coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    ages = 11 if scope == "45+" else 4
    fig.suptitle(f"Saudi age-specific rates, ages {scope}: five-year interval coverage\n"
                 f"Five overlapping origins × {ages} dependent age cells per sex; dashed line = 80%", fontsize=12)
    filename = "saudi_rate_coverage_"+("45plus" if scope == "45+" else "80plus")
    for extension in ["png", "svg"]:
        fig.savefig(OUT / f"{filename}.{extension}", dpi=170)
    plt.close(fig)


def plot_burdens(frame):
    endpoints = ["Both-sex 45+ count", "Male 80+ share", "Female 80+ share", "Age-specific M/F ratio"]
    part = frame[frame.target.eq("Saudi Arabia") & frame.horizon.eq(5) & frame.level.eq(.8)
                 & frame.endpoint.isin(endpoints) & frame.population_method.isin(["log_trend_last8", "not_applicable"])]
    mean = across_origins(part, ["outcome", "endpoint", "role", "variant"])
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), sharey=True, constrained_layout=True)
    x = np.arange(3)
    for row, outcome in enumerate(["prevalence", "incidence"]):
        for col, endpoint in enumerate(endpoints):
            ax = axes[row, col]
            selected = mean[mean.outcome.eq(outcome) & mean.endpoint.eq(endpoint)].set_index(["role", "variant"])
            for i, variant in enumerate(MAIN):
                heights = [selected.loc[(role, variant), "coverage"] for role in ROLES]
                bars = ax.bar(x+(i-1)*.25, heights, width=.24, color=COLORS[variant], label=LABELS[variant])
                ax.bar_label(bars, labels=[f"{value:.0%}" for value in heights], fontsize=7, padding=2, rotation=90)
            ax.axhline(.8, color="#222222", linestyle="--", linewidth=1)
            ax.set_xticks(x, ["TCN", "Local", "Non-neural"], fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_title(endpoint, fontsize=10)
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(f"{outcome.capitalize()}\n80% interval coverage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Saudi derived burden: unchanged points and population models\n"
                 "Five dependent totals/shares; 55 dependent age–origin ratio cells", fontsize=12)
    for extension in ["png", "svg"]:
        fig.savefig(OUT / f"saudi_derived_coverage.{extension}", dpi=170)
    plt.close(fig)


def rate_table(frame, scope, final=False, gcc=False):
    part = frame[frame.horizon.eq(5) & frame.level.eq(.8) & frame.age_scope.eq(scope) & frame.variant.isin(MAIN)]
    if final:
        part = part[part.origin.eq(2018)]
    if gcc:
        part = part[part.role.eq("tcn_adapted")]
        grouping = ["target", "outcome", "variant", "scale"]
        rows = []
        mean = across_origins(part, grouping)
        for (target, outcome), block in mean.groupby(["target", "outcome"]):
            rate, log = block[block.scale.eq("rate")].set_index("variant"), block[block.scale.eq("log_rate")].set_index("variant")
            cov = " / ".join(f"{100*rate.loc[v,'coverage']:.1f}" for v in MAIN)
            widths = " / ".join(f"{rate.loc[v,'width']/rate.loc['original','width']:.2f}×" for v in MAIN[1:])
            loss = " / ".join(f"{100*(log.loc[v,'wis_50_80']/log.loc['original','wis_50_80']-1):+.1f}%" for v in MAIN[1:])
            rows.append([target, outcome, cov, widths, loss])
        return markdown(["Country", "Outcome", "Coverage O / T / G (%)", "Width T/O / G/O", "Log WIS change T / G"], rows)
    part = part[part.target.eq("Saudi Arabia")]
    mean = across_origins(part, ["outcome", "sex", "role", "variant", "scale"])
    rows = []
    for outcome in ["prevalence", "incidence"]:
        for sex in ["Male", "Female"]:
            for role in ROLES:
                block = mean[mean.outcome.eq(outcome) & mean.sex.eq(sex) & mean.role.eq(role)]
                rate, log = block[block.scale.eq("rate")].set_index("variant"), block[block.scale.eq("log_rate")].set_index("variant")
                cov = " / ".join(f"{100*rate.loc[v,'coverage']:.1f}" for v in MAIN)
                widths = " / ".join(f"{rate.loc[v,'width']/rate.loc['original','width']:.2f}×" for v in MAIN[1:])
                loss = " / ".join(f"{100*(log.loc[v,'wis_50_80']/log.loc['original','wis_50_80']-1):+.1f}%" for v in MAIN[1:])
                rows.append([outcome, sex, LABELS[role], cov, widths, loss])
    return markdown(["Outcome", "Sex", "Procedure", "Coverage O / T / G (%)", "Width T/O / G/O", "Log WIS change T / G"], rows)


def report(rates, endpoints, decisions, validations, amendment):
    final_decisions = decisions[decisions.target.eq("Saudi Arabia") & decisions.fit_origin.isin([2017, 2018])]
    rows = []
    for outcome in ["prevalence", "incidence"]:
        for sex in ["Male", "Female"]:
            for role in ROLES:
                selected = final_decisions[final_decisions.outcome.eq(outcome) & final_decisions.sex.eq(sex) & final_decisions.role.eq(role)]
                lookup = selected.set_index(["fit_origin", "policy"]).factor
                final_family = selected[selected.fit_origin.eq(2018)].source_family.unique()
                assert len(final_family) == 1
                rows.append([outcome, sex, LABELS[role], final_family[0],
                             " / ".join(f"{lookup.loc[(2017,p)]:g}" for p in MAIN[1:]),
                             " / ".join(f"{lookup.loc[(2018,p)]:g}" for p in MAIN[1:])])
    factors = markdown(["Outcome", "Sex", "Procedure", "2018 underlying family", "2017 T / G", "2018 T / G"], rows)
    selected = endpoints[endpoints.target.eq("Saudi Arabia") & endpoints.horizon.eq(5) & endpoints.level.eq(.8)
                         & endpoints.role.eq("tcn_adapted") & endpoints.population_method.isin(["log_trend_last8", "not_applicable"])]
    mean = across_origins(selected, ["outcome", "endpoint", "unit", "variant"])
    rows = []
    for (outcome, endpoint, unit), block in mean.groupby(["outcome", "endpoint", "unit"]):
        block = block.set_index("variant")
        cov = " / ".join(f"{100*block.loc[v,'coverage']:.1f}" for v in MAIN)
        widths = " / ".join(f"{block.loc[v,'width']:.3g}" for v in MAIN)
        wis = " / ".join(f"{block.loc[v,'wis_50_80']:.3g}" for v in MAIN)
        rows.append([outcome, endpoint, unit, cov, widths, wis])
    derived = markdown(["Outcome", "Endpoint", "Unit", "Coverage O / T / G (%)", "Width O / T / G", "WIS O / T / G"], rows)
    fixed = rates[rates.target.eq("Saudi Arabia") & rates.horizon.eq(5) & rates.level.eq(.8) & rates.role.eq("tcn_adapted")
                  & rates.age_scope.isin(["45+", "80+"]) & rates.variant.isin(amendment["fixed_variants"])]
    fixed = across_origins(fixed, ["outcome", "sex", "age_scope", "variant", "scale"])
    rows = []
    for (outcome, sex, scope), block in fixed.groupby(["outcome", "sex", "age_scope"]):
        cov = block[block.scale.eq("rate")].set_index("variant").coverage
        loss = block[block.scale.eq("log_rate")].set_index("variant").wis_50_80
        coverage_text = " / ".join(f"{100*cov[v]:.1f}" for v in amendment["fixed_variants"])
        loss_text = " / ".join(f"{loss[v]:.4f}" for v in amendment["fixed_variants"])
        rows.append([outcome, sex, scope, coverage_text, loss_text])
    sensitivity = markdown(["Outcome", "Sex", "Ages", "Coverage at c=1 / 1.25 / 1.5 / 2 / 3 (%)", "Log WIS at same five factors"], rows)
    totals = {key: sum(row[key] for row in validations) for key in ["development_losses_reconstructed", "choices_replayed", "rate_draws_reconstructed",
              "rate_interval_rows_reconstructed", "burden_draws_reconstructed", "burden_interval_rows_reconstructed"]}
    # Narrative counts describe prespecified policies; they never change choices.
    means = across_origins(rates[rates.level.eq(.8) & rates.horizon.eq(5) & rates.age_scope.eq("45+")
                                & rates.scale.eq("log_rate") & rates.variant.isin(MAIN)], ["target", "outcome", "role", "sex", "variant"])
    contrasts = paired_changes(means, ["target", "outcome", "role", "sex"])
    saudi = contrasts[contrasts.target.eq("Saudi Arabia")]
    counts = {policy: (int((saudi[saudi.variant.eq(policy)].wis_change_percent < -1e-10).sum()),
                       int((saudi[saudi.variant.eq(policy)].wis_change_percent > 1e-10).sum())) for policy in MAIN[1:]}
    selected_rates = rates[rates.target.eq("Saudi Arabia") & rates.role.eq("tcn_adapted") & rates.horizon.eq(5)
                           & rates.level.eq(.8) & rates.scale.eq("rate") & rates.variant.isin(MAIN)]
    selected_mean = across_origins(selected_rates, ["outcome", "sex", "age_scope", "variant"]).set_index(["outcome", "sex", "age_scope", "variant"])
    pm0 = selected_mean.loc[("prevalence", "Male", "45+", "original")]
    pmt = selected_mean.loc[("prevalence", "Male", "45+", "target_only")]
    im0 = selected_mean.loc[("incidence", "Male", "45+", "original")]
    imt = selected_mean.loc[("incidence", "Male", "45+", "target_only")]
    img = selected_mean.loc[("incidence", "Male", "45+", "gcc_assisted")]
    old_male = selected_mean.loc[("prevalence", "Male", "80+", "target_only")]
    old_female = selected_mean.loc[("prevalence", "Female", "80+", "target_only")]
    final_old_female = selected_rates[selected_rates.origin.eq(2018) & selected_rates.outcome.eq("prevalence")
                                      & selected_rates.sex.eq("Female") & selected_rates.age_scope.eq("80+") & selected_rates.variant.eq("target_only")].iloc[0]
    rate_contrasts = across_origins(rates[rates.role.eq("tcn_adapted") & rates.horizon.eq(5) & rates.level.eq(.8)
                                         & rates.scale.eq("rate") & rates.age_scope.eq("45+") & rates.variant.isin(MAIN)],
                                   ["target", "outcome", "sex", "variant"])
    rate_contrasts = paired_changes(rate_contrasts, ["target", "outcome", "sex"])
    gcc_counts = {}
    for policy in MAIN:
        block = rate_contrasts[rate_contrasts.variant.eq(policy)]
        gcc_counts[policy] = dict(below=int((block.coverage < .8-1e-10).sum()),
             coverage_improved=int((block.coverage_change_pp > 1e-10).sum()),
             wis_improved=int((block.wis_50_80 < block.wis_50_80_original-1e-10).sum()),
             wis_worse=int((block.wis_50_80 > block.wis_50_80_original+1e-10).sum()))
    report_text = f"""# Exploratory interval-calibration experiment v1.2

The original primary conclusion is unchanged: the adapted TCN did not meet the joint male–female primary success criterion. This amendment was designed after viewing 2019–2023 performance. Its results are exploratory even though every factor decision uses only information available at the forecast origin. Point predictions, age coverage, population models and source-family selections remain fixed.

Across the 12 Saudi outcome–sex–procedure summaries, target-only calibration lowers five-origin log WIS in {counts['target_only'][0]} and raises it in {counts['target_only'][1]}; GCC-assisted calibration lowers it in {counts['gcc_assisted'][0]} and raises it in {counts['gcc_assisted'][1]}. These are descriptive comparisons of the two prespecified policies, not a new selection rule. Full coverage, width and directional misses must be read alongside WIS.

The adapted TCN shows limited, sex-specific gains. For Saudi male prevalence, both policies increase 80% coverage from {100*pm0.coverage:.1f}% to {100*pmt.coverage:.1f}% and reduce rate-scale WIS from {pm0.wis_50_80:.4f} to {pmt.wis_50_80:.4f}, with mean width increasing {100*(pmt.width/pm0.width-1):.1f}%. Male incidence coverage rises from {100*im0.coverage:.1f}% to {100*imt.coverage:.1f}%, but rate-scale WIS worsens by {100*(imt.wis_50_80/im0.wis_50_80-1):.1f}% under target-only selection and {100*(img.wis_50_80/im0.wis_50_80-1):.1f}% under GCC assistance; widths increase {100*(imt.width/im0.width-1):.1f}% and {100*(img.width/im0.width-1):.1f}%, respectively. Female TCN factors remain 1 for both outcomes. GCC assistance therefore does not provide a general Saudi calibration improvement.

## What was changed and what was preserved

One sex-specific multiplier scales the original centered joint log-residual blocks, shared across all 11 ages and five horizons. The fixed grid is 1, 1.25, 1.5, 2 and 3. Target-only selection uses the target's normalized past log WIS; GCC-assisted selection gives the target half the weight and the other five GCC countries one tenth each. Both add the locked penalty 0.05 × log(c)². Donors contribute same-underlying-family development losses, never residual draws or additional effective observations. Jordan is excluded. The current target's frozen underlying family is used for each champion role.

Calibration at 2014–2016 defaults to 1 because no eligible development interval is fully verified. Origin 2017 uses the interval issued in 2012, and 2018 uses those issued in 2012 and 2013. Each development normalizer uses only that issuance's historical bank. The final banks still contain only 7–11 dependent, overlapping residual-origin blocks. Family selection and interval selection share limited historical evidence; exact finite-sample or conformal coverage is not claimed.

Population forecasts and historical population residuals are reused unchanged. Rate and population draws retain their original residual-origin pairing before counts, age shares and sex ratios are transformed. There is no model retraining and no population refitting. A fixed point forecast is distinct from the empirical predictive median, which can move when widths change. Empirical quantiles may lie on the same side of the point forecast, so wider scalar multipliers need not produce nested endpoints or monotone coverage.

## Saudi factor decisions

T denotes target-only selection; G denotes GCC-assisted selection. All 2014–2016 factors equal 1. The full [decision ledger](factor_decisions.csv) retains every country, underlying family, origin, sex and candidate objective.

{factors}

## Saudi age-specific rates across five origins

The following tables evaluate horizon 5 across origins 2014–2018, counting 2018 once. O / T / G means original / target-only / GCC-assisted. Coverage and mean width use the rate scale; width ratios compare each policy with the original. WIS changes use log-rate WIS with 50% and 80% intervals; negative values are better. Numerical WIS values from different scales or derived units should not be combined. All 50%, 80% and supplementary 95% results, horizons, directional misses and individual ages are retained in the [independently reconstructed summaries](rate_summary.csv.gz).

### All ages 45+

Each sex has 55 dependent age–origin cells, not 55 independent replications.

{rate_table(rates, '45+')}

![Saudi age-45+ rate interval coverage](saudi_rate_coverage_45plus.png)

### Explicit age-80+ assessment

Each sex has 20 dependent cells: four age groups (80–84, 85–89, 90–94 and 95+) and five origins. Every age remains in the model and ledger; no weakly calibrated age group is excluded.

The Saudi TCN prevalence intervals remain poorly calibrated at age 80+: male coverage stays {100*old_male.coverage:.0f}% and female coverage {100*old_female.coverage:.0f}% under both policies. At the 2018→2023 endpoint, female coverage is {100*final_old_female.coverage:.0f}% across four ages; {int(round(4*final_old_female.lower_miss))} of the four observed rates lie below the lower bound. This directional failure is preserved in the analysis rather than hidden by aggregate coverage.

{rate_table(rates, '80+')}

![Saudi age-80+ rate interval coverage](saudi_rate_coverage_80plus.png)

### The 2018 five-year forecast, evaluated at 2023

This is also one observation within the preceding five-origin summaries. It is displayed separately to expose final-period behavior, without treating the two displays as independent evidence.

{rate_table(rates, '45+', final=True)}

The corresponding age-80+ endpoint table, all single-age rows and lower-versus-upper misses are available in [final endpoint summaries](final_2018_rate_summary.csv).

## Fixed-factor sensitivity

All five fixed multipliers are displayed regardless of final performance. The table shows the adapted TCN; [complete fixed-grid sensitivities](fixed_factor_summary.csv) include every role and GCC country. These curves describe trade-offs and are not used to choose a new post-result winner. Original factors are included as c=1. Fixed-factor derived burden variants were not part of the locked scope.

{sensitivity}

## Derived burden with unchanged population models

The adapted-TCN table uses the operational last-eight-year log-trend population forecast. Persistence results and every comparator role remain in the [derived endpoint ledger](burden_endpoint_summary.csv). Count and each share endpoint have five dependent observations; the sex-rate ratio averages 11 ages across those origins. The share unit is percentage points; counts are modeled numbers, not patient records. Sex-rate ratios use no population forecast.

{derived}

![Saudi derived-burden coverage](saudi_derived_coverage.png)

Improved rate intervals cannot be assumed to improve a total count, an age share or a sex ratio: each depends on the joint transformed distribution. Neither the point count errors nor the previously observed demographic-component cancellation can improve in this experiment, because those points and population trajectories are unchanged. The separate native-count reconciliation results are likewise unchanged. Population-source sensitivity and projection scenarios remain separate unfinished work.

For both Saudi outcomes, all three procedures still cover only one of five observed male 80+ shares (20%); the remaining four observations exceed the upper bound. Meanwhile, both-sex total-count coverage remains 100% across only five overlapping origins. The unchanged contrast between these endpoints shows why total-burden coverage cannot establish adequate age-composition uncertainty. Adapted-TCN sex-ratio coverage improves modestly, but stays below 80% for both outcomes.

## GCC benchmarking

The table applies the same horizon-5 evaluation to all six GCC targets, showing the adapted TCN with equal weight to male and female strata. Saudi Arabia remains the primary target. These within-country descriptive comparisons do not establish independent country-level significance. Other roles, age-80+ results and all directional misses are retained in the downloadable ledgers.

{rate_table(rates, '45+', gcc=True)}

Across 24 TCN country–outcome–sex summaries, {gcc_counts['original']['below']} are below 80% coverage originally and {gcc_counts['target_only']['below']} under either selected policy. Target-only calibration improves coverage in {gcc_counts['target_only']['coverage_improved']} summaries and GCC assistance in {gcc_counts['gcc_assisted']['coverage_improved']}, with no coverage deterioration in these data. Rate-scale WIS improves in {gcc_counts['target_only']['wis_improved']} summaries for each policy, while worsening in {gcc_counts['target_only']['wis_worse']} under target-only calibration and {gcc_counts['gcc_assisted']['wis_worse']} under GCC assistance. These mixed trade-offs do not support selecting one policy as a universal replacement.

## Independent validation and limits

The audit independently reconstructed {totals['development_losses_reconstructed']:,} development loss cells and replayed {totals['choices_replayed']:,} policy choices. It reconstructed {totals['rate_draws_reconstructed']:,} rate draws, {totals['rate_interval_rows_reconstructed']:,} rate interval rows, {totals['burden_draws_reconstructed']:,} derived draws and {totals['burden_interval_rows_reconstructed']:,} derived interval rows. It recomputed interval scores and WIS, checked factor-1 reproduction, unchanged point values and population inputs, and verified all original and new code/output hashes. All 12 cases committed their distributions before any new final scoring. See [validation](validation.json).

This is retrospective forecasting of modeled GBD point estimates. Historical intervals describe forecast-error variation, not the original GBD source-estimate uncertainty intervals, patient uncertainty or causal biological effects. Calibration uses only one or two completed development interval origins and is weakly identified; the 95% tails are especially sparse. Scaling the same blocks cannot create new temporal information or repair systematic point bias. The [original primary report](../primary_v1/report.md) remains authoritative for the prespecified hypothesis.
"""
    (OUT / "report.md").write_text(report_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one through four independent report workers")
    check_lock()
    config = json_read(ROOT / "study_design/locked_v1/design.json")
    amendment = json_read(ROOT / "study_design/interval_calibration_v1_2.json")
    verify_integrity(amendment)
    cases = json_read(RUN / "cases.json")
    assert len(cases) == 12
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(audit_case, case, config, amendment) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Independent calibration audit complete: {result[0]['case']}", flush=True)
    validations = sorted([result[0] for result in results], key=lambda row: row["case"])
    rates, burdens, decisions = [pd.concat([result[index] for result in results], ignore_index=True) for index in range(1, 4)]
    rate_keys = ["target", "outcome", "origin", "role", "variant", "sex", "horizon", "scale", "level", "age_scope"]
    burden_keys = ["target", "outcome", "origin", "role", "variant", "horizon", "population_method", "measure", "node", "sex", "age_group", "unit", "level"]
    rates = rates.sort_values(rate_keys)
    burdens = burdens.sort_values(burden_keys)
    endpoints = burden_endpoints(burdens)
    OUT.mkdir(parents=True, exist_ok=True)
    rates.to_csv(OUT / "rate_summary.csv.gz", index=False, compression="gzip")
    burdens.to_csv(OUT / "burden_node_summary.csv.gz", index=False, compression="gzip")
    endpoints.to_csv(OUT / "burden_endpoint_summary.csv", index=False)
    decisions.sort_values(["target", "outcome", "fit_origin", "role", "sex", "policy"]).to_csv(OUT / "factor_decisions.csv", index=False)
    fixed = rates[rates.variant.isin(amendment["fixed_variants"]) & rates.horizon.eq(5)]
    across_origins(fixed, [key for key in rate_keys if key not in ["origin", "horizon"]]).to_csv(OUT / "fixed_factor_summary.csv", index=False)
    rates[rates.origin.eq(2018) & rates.horizon.eq(5)].to_csv(OUT / "final_2018_rate_summary.csv", index=False)
    means = across_origins(rates[rates.horizon.eq(5)], [key for key in rate_keys if key not in ["origin", "horizon"]])
    paired_changes(means, [key for key in rate_keys if key not in ["origin", "horizon", "variant"]]).to_csv(OUT / "rate_policy_contrasts.csv", index=False)
    means = across_origins(endpoints[endpoints.horizon.eq(5)], ["target", "outcome", "role", "variant", "population_method", "level", "endpoint", "unit"])
    paired_changes(means, ["target", "outcome", "role", "population_method", "level", "endpoint", "unit"]).to_csv(OUT / "burden_policy_contrasts.csv", index=False)
    plot_rates(rates, "45+")
    plot_rates(rates, "80+")
    plot_burdens(endpoints)
    report(rates, endpoints, decisions, validations, amendment)
    validation = dict(passed=True, created_utc=datetime.now().astimezone().isoformat(), role="independent_exploratory_calibration_audit",
                      report_code_sha256=sha(Path(__file__)), independent_aggregation_helper_sha256=sha(ROOT / "scripts/report_demography.py"),
                      source_manifest_sha256=sha(RUN / "run_manifest.json"), original_primary_preserved=True,
                      all_case_commits_precede_scoring=True, source_refits=False, case_validations=validations)
    validation["artifact_sha256"] = {path.name: sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "validation.json"}
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps(dict(passed=True, cases=len(validations), report=str(OUT / "report.md")), indent=2))


if __name__ == "__main__":
    main()
