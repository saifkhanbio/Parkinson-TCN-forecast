"""Analytic accounting, historical-ratio and paired-block uncertainty tests."""

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
from gbd_park import disability_components as components
import run_disability_components as runner

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def source_fixture():
    rows = []
    for ci, country in enumerate(["Saudi Arabia", "Jordan"]):
        for year in range(1980, 2024):
            for si, sex in enumerate(CONFIG["sexes"]):
                for ai, age in enumerate(CONFIG["ages"]):
                    elapsed = year-1980
                    prevalence = 20+.7*elapsed+ai+3*si+ci
                    ratio = .05+.002*elapsed+.007*si+.001*ai
                    ylds, ylls = prevalence*ratio, 15+.2*elapsed+.5*ai+si
                    population = 1000+100*ai+50*si+10*elapsed
                    rates = dict(prevalence=prevalence, incidence=.01*prevalence,
                                 deaths=.1*ylls, ylds=ylds, ylls=ylls, dalys=ylds+ylls)
                    for outcome, rate in rates.items():
                        rows.append(dict(location_name=country, year=year, sex=sex, age=age,
                                         outcome=outcome, rate=rate, count=rate*population/100000,
                                         implied_population=population,
                                         rate_lower=0., rate_upper=999999., count_lower=-1., count_upper=12345.))
    return pd.DataFrame(rows)


def points_fixture(outcome, value, origin=2014, family="local_champion"):
    rows = []
    for sex in CONFIG["sexes"]:
        for age in CONFIG["ages"]:
            for horizon in range(1, 6):
                rows.append(dict(target="Saudi Arabia", origin=origin, family=family, sex=sex,
                                 age=age, horizon=horizon, forecast_year=origin+horizon,
                                 outcome=outcome, prediction=value, log_prediction=np.log(value),
                                 setting_id=family+"_historically_selected", status="ok"))
    return pd.DataFrame(rows)


def draws_fixture(points, values):
    draws = []
    for residual_origin, value in zip(range(2003, 2010), values):
        frame = points.copy()
        frame["residual_origin"] = residual_origin
        frame["rate_draw"], frame["log_draw"] = value, np.log(value)
        draws.append(frame)
    return pd.concat(draws, ignore_index=True)


def ratio_inputs(panel):
    families = ["tcn_adapted", "persistence", "log_trend", "pooled_ridge", "donor_ridge_adapted"]
    history = []
    for origin in range(2003, 2014):
        for fi, family in enumerate(families):
            frame = points_fixture("prevalence", 20+fi+2*(origin-2003), origin=origin, family=family)
            frame["setting_id"] = family+"__origin"+str(origin)
            history.append(frame)
    mappings = pd.DataFrame([
        dict(fit_origin=2014, role="local_champion", sex="Male", source_family="log_trend", last_selection_target_year=2014),
        dict(fit_origin=2014, role="local_champion", sex="Female", source_family="persistence", last_selection_target_year=2014),
        dict(fit_origin=2014, role="nonneural_champion", sex="Male", source_family="donor_ridge_adapted", last_selection_target_year=2014),
        dict(fit_origin=2014, role="nonneural_champion", sex="Female", source_family="pooled_ridge", last_selection_target_year=2014),
    ])
    points = pd.concat([points_fixture("prevalence", 40+i, family=role)
                        for i, role in enumerate(components.ROLES)], ignore_index=True)
    config = deepcopy(CONFIG)
    config["calendar"]["reliability_origins"] = [2014]
    return pd.concat(history, ignore_index=True), points, mappings, config


def population_inputs(config=CONFIG, origins=(2014,)):
    populations, errors = [], []
    for origin in origins:
        for method, scale in [("log_trend_last8", 1.), ("persistence", 2.)]:
            template = points_fixture("prevalence", 1., origin=origin)
            template = template[["target", "origin", "sex", "age", "horizon", "forecast_year"]].copy()
            ai = template.age.map({age: i+1 for i, age in enumerate(config["ages"])}).to_numpy()
            si = np.where(template.sex.eq("Male"), 1., 2.)
            template["population_method"] = method
            template["population"] = 1000*ai*si*scale
            template["log_population"] = np.log(template.population)
            populations.append(template)
            blocks = list(range(2003, origin-4))
            for bi, residual_origin in enumerate(blocks):
                error = template.copy()
                error["fit_origin"], error["origin"], error["residual_origin"] = origin, residual_origin, residual_origin
                error["centered_population_log_error"] = (bi-(len(blocks)-1)/2)*.06*ai/11
                errors.append(error)
    return pd.concat(populations, ignore_index=True), pd.concat(errors, ignore_index=True)


def burden_fixture():
    points, draws = [], []
    x = np.array([30., 20., 15., 10., 4., 2., 1.])
    for outcome, point, values in [("ylds", 7., x), ("ylls", 14., 2*x),
                                    ("deaths", 2., x/10), ("dalys", 22., 3.2*x)]:
        frame = points_fixture(outcome, point, family="tcn_adapted")
        frame["procedure"] = "direct"
        points.append(frame)
        draws.append(draws_fixture(frame, values))
    points.append(components.sum_daly(points[0], points[1], CONFIG))
    draws.append(components.sum_daly(draws[0], draws[1], CONFIG, draw=True))
    population, errors = population_inputs()
    return pd.concat(points, ignore_index=True), pd.concat(draws, ignore_index=True), population, errors


def write_full_synthetic_sources(root, panel):
    """A complete case for the production issuance/scoring paths, without fits."""
    core = root / "core"
    families = CONFIG["models"]["local_order"] + CONFIG["models"]["nonneural_order"] + [
        "tcn_adapted", "tcn_unadapted", "tcn_intercept", "local_champion", "nonneural_champion"]
    for outcome, value in [("deaths", 2.), ("ylds", 7.), ("ylls", 14.), ("dalys", 22.)]:
        points, draws = [], []
        for origin in CONFIG["calendar"]["reliability_origins"]:
            residual_origins = list(range(2003, origin-4))
            for fi, family in enumerate(families):
                point = points_fixture(outcome, value*(1+.001*fi), origin=origin, family=family)
                point["source_family"] = ("persistence" if family == "local_champion" else
                                           "pooled_ridge" if family == "nonneural_champion" else np.nan)
                points.append(point)
                for bi, residual_origin in enumerate(residual_origins):
                    draw = point.copy()
                    draw["residual_origin"] = residual_origin
                    draw["rate_draw"] = draw.prediction*np.exp(.015*(bi-(len(residual_origins)-1)/2))
                    draw["log_draw"] = np.log(draw.rate_draw)
                    draws.append(draw)
        directory = core / "trials" / ("SAU_"+outcome)
        directory.mkdir(parents=True)
        pd.concat(points, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
        pd.concat(draws, ignore_index=True).to_csv(directory / "joint_draws.csv", index=False)
    history, _, mapping, _ = ratio_inputs(panel)
    historical = root / "results/intervals_v1"
    historical.mkdir(parents=True)
    history.to_csv(historical / "prequential_predictions.csv", index=False)
    source = root / "results/primary_v1"
    source.mkdir(parents=True)
    points, maps = [], []
    for origin in CONFIG["calendar"]["reliability_origins"]:
        for i, role in enumerate(components.ROLES):
            points.append(points_fixture("prevalence", 40+i, origin=origin, family=role))
        current = mapping.copy()
        current["fit_origin"] = origin
        current["last_selection_target_year"] = origin
        maps.append(current)
    pd.concat(points, ignore_index=True).to_csv(source / "predictions.csv", index=False)
    pd.concat(maps, ignore_index=True).to_csv(source / "champion_family_mappings.csv", index=False)
    population, errors = population_inputs(origins=CONFIG["calendar"]["reliability_origins"])
    demo = root / "results/demography_v1/SAU_prevalence"
    demo.mkdir(parents=True)
    population.to_csv(demo / "population_forecasts.csv", index=False)
    errors.to_csv(demo / "population_residuals.csv", index=False)
    data = root / "data/processed/design_v1"
    data.mkdir(parents=True)
    panel.to_csv(data / "regional_outcomes.csv", index=False)
    return core


class DisabilityComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = source_fixture()

    def test_training_ratio_is_ratio_of_sums_over_exact_last_eight_years(self):
        ratios = components.training_ratios(self.panel, CONFIG, "Saudi Arabia", 2014)
        self.assertEqual(len(ratios), 22)
        self.assertTrue(ratios.ratio_history_start.eq(2007).all())
        self.assertTrue(ratios.ratio_history_end.eq(2014).all())
        for row in ratios.itertuples():
            part = self.panel[self.panel.location_name.eq("Saudi Arabia") & self.panel.sex.eq(row.sex)
                              & self.panel.age.eq(row.age) & self.panel.year.between(2007, 2014)]
            values = part.pivot(index="year", columns="outcome", values="rate")
            expected = values.ylds.sum()/values.prevalence.sum()
            self.assertAlmostEqual(row.training_ratio, expected, places=14)
            self.assertGreater(abs(row.training_ratio-(values.ylds/values.prevalence).mean()), 1e-6)
        sexes = ratios.pivot(index="age", columns="sex", values="training_ratio")
        self.assertTrue((sexes.Female > sexes.Male).all())

    def test_ratio_excludes_future_older_other_outcome_and_other_country_rows(self):
        expected = components.training_ratios(self.panel, CONFIG, "Saudi Arabia", 2014)
        changed = self.panel.copy()
        irrelevant = (changed.location_name.ne("Saudi Arabia") | ~changed.outcome.isin(["prevalence", "ylds"])
                      | ~changed.year.between(2007, 2014))
        changed.loc[irrelevant, "rate"] = np.nan
        pd.testing.assert_frame_equal(expected, components.training_ratios(changed, CONFIG, "Saudi Arabia", 2014))

    def test_ratio_rejects_missing_duplicate_nonpositive_and_nonfinite_cells(self):
        chosen = self.panel.index[self.panel.location_name.eq("Saudi Arabia") & self.panel.outcome.eq("ylds")
                                  & self.panel.year.eq(2007) & self.panel.age.eq("95+")][0]
        invalid = [self.panel.drop(chosen), pd.concat([self.panel, self.panel.loc[[chosen]]], ignore_index=True)]
        for value in [0., -1., np.nan, np.inf]:
            changed = self.panel.copy()
            changed.loc[chosen, "rate"] = value
            invalid.append(changed)
        for frame in invalid:
            with self.assertRaises(ValueError):
                components.training_ratios(frame, CONFIG, "Saudi Arabia", 2014)

    def test_baseline_multiplies_points_keeps_sex_age_ratios_and_restores_ylds(self):
        points = points_fixture("prevalence", 100.)
        baseline = components.prevalence_yld_predictions(points, self.panel, CONFIG, "Saudi Arabia")
        self.assertTrue(baseline.outcome.eq("ylds").all())
        np.testing.assert_allclose(baseline.prediction, baseline.training_ratio*100.)
        np.testing.assert_allclose(baseline.log_prediction, np.log(baseline.prediction))
        self.assertTrue(baseline.ratio_history_end.eq(baseline.origin).all())
        self.assertEqual(len(baseline), 110)
        verified = points.assign(observed_rate=2.)
        with self.assertRaises(ValueError):
            components.prevalence_yld_predictions(verified, self.panel, CONFIG, "Saudi Arabia")
        with self.assertRaises(ValueError):
            components.prevalence_yld_predictions(points.iloc[:-1], self.panel, CONFIG, "Saudi Arabia")

    def test_component_point_sum_uses_rate_scale_and_retains_distinct_champions(self):
        ylds = points_fixture("ylds", 7.)
        ylls = points_fixture("ylls", 14.)
        ylds["source_family"] = "arima"
        ylls["source_family"] = "log_trend"
        result = components.sum_daly(ylds, ylls.sample(frac=1, random_state=3), CONFIG)
        np.testing.assert_array_equal(result.prediction, 21.)
        np.testing.assert_allclose(result.log_prediction, np.log(21.))
        self.assertTrue(result.outcome.eq("dalys").all())
        self.assertEqual(set(result.source_family_yld), {"arima"})
        self.assertEqual(set(result.source_family_yll), {"log_trend"})
        self.assertFalse(np.isclose(result.log_prediction.iloc[0], np.log(7.)+np.log(14.)))
        population = np.linspace(1000, 12000, len(result))
        np.testing.assert_allclose(result.prediction*population/100000,
                                   ylds.prediction*population/100000+ylls.prediction*population/100000)
        underlying = ylds.assign(family="tcn_adapted", source_family=np.nan)
        underlying_other = ylls.assign(family="tcn_adapted", source_family=np.nan)
        restored = components.sum_daly(underlying, underlying_other, CONFIG)
        self.assertEqual(set(restored.source_family_yld), {"tcn_adapted"})
        self.assertEqual(set(restored.source_family_yll), {"tcn_adapted"})

    def test_joint_component_sum_quantiles_are_not_summed_marginal_endpoints(self):
        left = points_fixture("ylds", 7.)
        right = points_fixture("ylls", 14.)
        x = np.array([1., 2., 4., 10., 15., 20., 30.])
        y = np.array([30., 20., 15., 10., 4., 2., 1.])
        points = components.sum_daly(left, right, CONFIG)
        draws = components.sum_daly(draws_fixture(left, x),
                                    draws_fixture(right, y).sample(frac=1, random_state=8), CONFIG, draw=True)
        intervals = components.intervals_from_draws(points, draws, CONFIG)
        self.assertEqual(len(intervals), 660)
        self.assertEqual(set(intervals.n_blocks), {7})
        for row in intervals.itertuples():
            values = x+y if row.scale == "rate" else np.log(x+y)
            expected = np.quantile(values, [(1-row.level)/2, .5, (1+row.level)/2], method="linear")
            np.testing.assert_allclose([row.lower, row.median, row.upper], expected)
            self.assertAlmostEqual(row.point_prediction, 21. if row.scale == "rate" else np.log(21.))
        central80 = intervals[intervals.scale.eq("rate") & intervals.level.eq(.8)].iloc[0]
        self.assertGreater(abs(central80.lower-(np.quantile(x, .1)+np.quantile(y, .1))), 1.)
        central50 = intervals[intervals.scale.eq("rate") & intervals.level.eq(.5)].iloc[0]
        log50 = intervals[intervals.scale.eq("log_rate") & intervals.level.eq(.5)].iloc[0]
        self.assertGreater(abs(central50.lower-np.exp(log50.lower)), .001)

    def test_component_pairing_rejects_missing_oldest_age_and_residual_origin(self):
        left = points_fixture("ylds", 7.)
        right = points_fixture("ylls", 14.)
        with self.assertRaises(ValueError):
            components.sum_daly(left, right[right.age.ne("95+")], CONFIG)
        ldraw = draws_fixture(left, np.arange(1., 8.))
        rdraw = draws_fixture(right, np.arange(11., 18.))
        bad_frames = [rdraw[rdraw.residual_origin.ne(2009)], rdraw.iloc[:-1],
                      pd.concat([rdraw, rdraw.iloc[[0]]], ignore_index=True),
                      rdraw.replace({"residual_origin": {2009: 2010}})]
        for bad in bad_frames:
            with self.assertRaises(ValueError):
                components.sum_daly(ldraw, bad, CONFIG, draw=True)
        with self.assertRaises(ValueError):
            components.sum_daly(left, right.assign(outcome="deaths"), CONFIG)

    def test_ratio_baseline_uses_own_historical_origin_ratios_and_outer_champion(self):
        history, points, mappings, config = ratio_inputs(self.panel)
        issued, intervals, draws, banks, historical = components.issue_ratio_forecasts(
            history, points, mappings, self.panel, config, "Saudi Arabia")
        self.assertEqual(len(issued), 330)
        self.assertEqual(len(intervals), 1980)
        self.assertEqual(len(draws), 2310)
        self.assertEqual(set(banks), {(2014, role) for role in components.ROLES})
        self.assertTrue(historical.ratio_history_end.eq(historical.origin).all())
        source_index = history.set_index(["origin", "family", "sex", "age", "horizon"]).prediction
        target_panel = self.panel[self.panel.location_name.eq("Saudi Arabia")]
        observed_index = target_panel[target_panel.outcome.eq("ylds")].set_index(["sex", "age", "year"]).rate
        expected_ratios = {}
        for past_origin in range(2003, 2010):
            selected = target_panel[target_panel.year.between(past_origin-7, past_origin)]
            total = selected.groupby(["sex", "age", "outcome"]).rate.sum().unstack()
            expected_ratios[past_origin] = total.ylds/total.prevalence
        for role in components.ROLES:
            bank = banks[(2014, role)]
            self.assertEqual(bank["origins"], list(range(2003, 2010)))
            np.testing.assert_allclose(bank["centered_residuals"].mean(axis=0), 0., atol=1e-14)
            by_sex = bank["prevalence_source_family_by_sex"]
            if role == "local_champion":
                self.assertEqual(by_sex, {"Male": "log_trend", "Female": "persistence"})
            if role == "nonneural_champion":
                self.assertEqual(by_sex, {"Male": "donor_ridge_adapted", "Female": "pooled_ridge"})
            for bi, past_origin in enumerate(bank["origins"]):
                for ci, coord in bank["coords"].iterrows():
                    source = source_index.loc[(past_origin, by_sex[coord.sex], coord.sex, coord.age, coord.horizon)]
                    ratio = expected_ratios[past_origin].loc[(coord.sex, coord.age)]
                    observed = observed_index.loc[(coord.sex, coord.age, past_origin+coord.horizon)]
                    expected = np.log(observed)-np.log(source*ratio)
                    self.assertAlmostEqual(bank["raw_residuals"][bi, ci], expected, places=12)

    def test_ratio_baseline_bank_excludes_future_truth_and_future_selected_family(self):
        history, points, mappings, config = ratio_inputs(self.panel)
        first = components.issue_ratio_forecasts(history, points, mappings, self.panel, config, "Saudi Arabia")
        altered = self.panel.copy()
        altered.loc[altered.year.gt(2014) & altered.outcome.eq("ylds"), "rate"] *= 1000
        second = components.issue_ratio_forecasts(history, points, mappings, altered, config, "Saudi Arabia")
        for left, right in zip(first[:3], second[:3]):
            pd.testing.assert_frame_equal(left, right)
        for key in first[3]:
            np.testing.assert_array_equal(first[3][key]["raw_residuals"], second[3][key]["raw_residuals"])
        future_map = mappings.copy()
        future_map.loc[0, "last_selection_target_year"] = 2015
        with self.assertRaises(ValueError):
            components.issue_ratio_forecasts(history, points, future_map, self.panel, config, "Saudi Arabia")

    def test_source_accounting_checks_both_point_scales_and_common_population(self):
        audited = components.source_accounting(self.panel, CONFIG)
        self.assertEqual(len(audited), 2*34*22)
        self.assertLess(audited.rate_relative_identity_error.abs().max(), 1e-14)
        self.assertLess(audited.count_relative_identity_error.abs().max(), 1e-14)
        self.assertLess(audited.maximum_relative_population_difference.max(), 1e-14)
        # Arbitrary uncertainty endpoints are not substituted for source point identities.
        changed = self.panel.copy()
        changed[["rate_lower", "rate_upper", "count_lower", "count_upper"]] = np.nan
        pd.testing.assert_frame_equal(audited, components.source_accounting(changed, CONFIG))
        row = self.panel.index[self.panel.outcome.eq("dalys") & self.panel.year.eq(2010)][0]
        for field in ["rate", "count"]:
            bad = self.panel.copy()
            bad.loc[row, field] *= 1.01
            with self.assertRaises((AssertionError, ValueError)):
                components.source_accounting(bad, CONFIG)
        # Preserve DALY accounting while violating the prevalence denominator.
        bad = self.panel.copy()
        bad.loc[bad.outcome.eq("deaths") & bad.year.eq(2010), "count"] *= 1.01
        with self.assertRaisesRegex(ValueError, "denominators"):
            components.source_accounting(bad, CONFIG)

    def test_population_errors_pair_by_historical_origin_before_counts_and_shares(self):
        points, draws, population, errors = burden_fixture()
        predicted, intervals, checks = runner.convert_burden(points, draws, population,
                                                              errors.sample(frac=1, random_state=91), CONFIG)
        self.assertEqual(len(predicted), 5*5*2*35)
        self.assertEqual(len(intervals), 3*len(predicted))
        self.assertTrue(all(check["maximum_absolute_count_identity_error"] < 1e-10 for check in checks))
        self.assertEqual({check["draw"] for check in checks}, {False, True})
        self.assertEqual(set(predicted.loc[predicted.measure.eq("count") & predicted.outcome.eq("deaths"), "unit"]), {"modeled_events"})
        self.assertEqual(set(predicted.loc[predicted.measure.eq("count") & predicted.outcome.ne("deaths"), "unit"]), {"modeled_burden_years"})
        self.assertEqual(set(predicted.loc[predicted.measure.eq("age_share"), "unit"]), {"percent"})
        ai = np.arange(1., 12.)
        yld_draw = np.array([30., 20., 15., 10., 4., 2., 1.])
        total_draws, share_draws = [], []
        for bi, value in enumerate(yld_draw):
            population_draw = 1000*ai*np.exp((bi-3)*.06*ai/11)
            male_counts = 3*value*population_draw/100000
            total_draws.append(3*male_counts.sum())  # Female denominator is twice Male.
            share_draws.append(100*male_counts[-4:].sum()/male_counts.sum())
        subset = intervals[intervals.procedure.eq("component_sum") & intervals.population_method.eq("log_trend_last8")
                           & intervals.horizon.eq(1) & intervals.level.eq(.8)]
        count = subset[subset.node.eq("Both__45+")].iloc[0]
        share = subset[subset.node.eq("Male__80+_within_45+")].iloc[0]
        np.testing.assert_allclose([count.lower, count["median"], count.upper], np.quantile(total_draws, [.1, .5, .9]))
        np.testing.assert_allclose([share.lower, share["median"], share.upper], np.quantile(share_draws, [.1, .5, .9]))
        base_point = predicted[predicted.procedure.eq("component_sum") & predicted.population_method.eq("log_trend_last8")
                               & predicted.horizon.eq(1) & predicted.node.eq("Both__45+")].value.iloc[0]
        self.assertAlmostEqual(base_point, 21.*3*1000*ai.sum()/100000)
        # A different pairing leaves marginals unchanged but changes aggregate intervals.
        reversed_errors = errors.copy()
        reversed_errors["residual_origin"] = 4012-reversed_errors.residual_origin
        _, changed, _ = runner.convert_burden(points, draws, population, reversed_errors, CONFIG)
        compared = changed[changed.procedure.eq("component_sum") & changed.population_method.eq("log_trend_last8")
                           & changed.horizon.eq(1) & changed.level.eq(.8) & changed.node.eq("Both__45+")].iloc[0]
        self.assertGreater(abs(compared.upper-count.upper), .1)

    def test_burden_score_matches_analytic_interval_and_wis_arithmetic(self):
        points, draws, population, errors = burden_fixture()
        predicted, intervals, _ = runner.convert_burden(points, draws, population, errors, CONFIG)
        truth = predicted.rename(columns={"value": "observed"})
        scores, wis = runner.score_burden(intervals, truth)
        self.assertEqual(len(scores), len(intervals))
        self.assertEqual(len(wis), len(predicted))
        np.testing.assert_array_equal(scores.covered, ~scores.below_lower & ~scores.above_upper)
        target = scores[(scores.procedure == "component_sum") & (scores.population_method == "log_trend_last8")
                        & (scores.horizon == 1) & (scores.node == "Both__45+")].set_index("level")
        y, median = target.observed.iloc[0], target["median"].iloc[0]
        terms = []
        for level in [.5, .8]:
            row = target.loc[level]
            alpha = 1-level
            score = row.upper-row.lower+2/alpha*max(row.lower-y, 0)+2/alpha*max(y-row.upper, 0)
            self.assertAlmostEqual(row.interval_score, score)
            terms.append(alpha/2*score)
        expected = (.5*abs(y-median)+sum(terms))/2.5
        actual = wis[(wis.procedure == "component_sum") & (wis.population_method == "log_trend_last8")
                     & (wis.horizon == 1) & (wis.node == "Both__45+")].wis_50_80.iloc[0]
        self.assertAlmostEqual(actual, expected)

    def test_burden_rejects_missing_malformed_or_unpaired_coordinates(self):
        points, draws, population, errors = burden_fixture()
        with self.assertRaises(ValueError):
            runner.convert_burden(points.iloc[:-1], draws, population, errors, CONFIG)
        with self.assertRaises(ValueError):
            runner.convert_burden(points, draws.iloc[:-1], population, errors, CONFIG)
        wrong_blocks = errors.copy()
        wrong_blocks.loc[wrong_blocks.residual_origin.eq(2009), "residual_origin"] = 2010
        with self.assertRaises(ValueError):
            runner.convert_burden(points, draws, population, wrong_blocks, CONFIG)
        bad = draws.copy()
        bad.loc[0, "rate_draw"] = np.nan
        with self.assertRaises(ValueError):
            runner.convert_burden(points, bad, population, errors, CONFIG)
        # An entire absent horizon is not detected by within-horizon age completeness alone.
        missing_horizon = draws[~(draws.outcome.eq("deaths") & draws.horizon.eq(5))]
        with self.assertRaises(ValueError):
            runner.convert_burden(points, missing_horizon, population, errors, CONFIG)

    def test_full_synthetic_country_issuance_then_committed_scoring(self):
        country = next(country for country in CONFIG["countries"] if country["iso3"] == "SAU")
        with tempfile.TemporaryDirectory(prefix="disability-synthetic-") as temp:
            root = Path(temp)
            core = write_full_synthetic_sources(root, self.panel)
            destination = root / "derived"
            destination.mkdir()
            with patch.object(runner, "ROOT", root):
                result = runner.issue_country(country, CONFIG, core, destination)
                self.assertFalse(result["cached"])
                directory = destination / "SAU"
                self.assertTrue(runner.phase_complete(directory, "issued_commit"))
                self.assertFalse((directory / "rate_point_scores.csv").exists())
                with self.assertRaises((AssertionError, ValueError, FileNotFoundError)):
                    runner.score_country(country, CONFIG, destination)
                self.assertFalse((directory / "rate_point_scores.csv").exists())
                # Other country markers are integrity-only fixtures; this smoke scores Saudi only.
                commits = [directory / "issued_commit.json"]
                for other in CONFIG["countries"]:
                    if not other["gcc"] or other["iso3"] == "SAU":
                        continue
                    other_dir = destination / other["iso3"]
                    other_dir.mkdir()
                    ledger = other_dir / "synthetic_issuance_marker.txt"
                    ledger.write_text("Synthetic commitment fixture; no score or numerical claim.\n")
                    runner.commit_phase(other_dir, "issued_commit", [ledger])
                    commits.append(other_dir / "issued_commit.json")
                runner.commit_phase(destination, "global_issued_commit", commits)
                runner.score_country(country, CONFIG, destination)
                validation = json.loads((directory / "validation_report.json").read_text())
                self.assertTrue(validation["passed"])
                self.assertEqual(validation["derived_rate_points"], 10450)
                self.assertEqual(validation["derived_rate_intervals"], 62700)
                self.assertEqual(validation["burden_point_statistics"], 31500)
                self.assertEqual(validation["burden_interval_rows"], 94500)
                self.assertTrue(runner.phase_complete(directory, "scoring_complete"))
                rates = runner.read(directory / "rate_point_scores.csv")
                np.testing.assert_allclose(rates.absolute_log_error, np.abs(np.log(rates.prediction/rates.observed_rate)), atol=1e-14)
                burdens = runner.read(directory / "burden_point_scores.csv")
                np.testing.assert_allclose(burdens.absolute_error, np.abs(burdens.value-burdens.observed))
                with patch.object(runner, "convert_burden", side_effect=AssertionError("Unexpected recomputation")):
                    cached = runner.issue_country(country, CONFIG, core, destination)
                self.assertTrue(cached["cached"])

    def test_scoring_gate_rechecks_nested_issued_artifact_hashes(self):
        country = next(country for country in CONFIG["countries"] if country["iso3"] == "SAU")
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            markers = []
            for item in CONFIG["countries"]:
                if not item["gcc"]:
                    continue
                directory = destination / item["iso3"]
                directory.mkdir()
                issued = directory / "rate_predictions.csv"
                issued.write_text("synthetic committed artifact\n")
                runner.commit_phase(directory, "issued_commit", [issued])
                markers.append(directory / "issued_commit.json")
            runner.commit_phase(destination, "global_issued_commit", markers)
            (destination / "SAU/rate_predictions.csv").write_text("changed after commitment\n")
            with patch.object(runner, "read", side_effect=AssertionError("Integrity must be checked before reading outcomes")):
                with self.assertRaisesRegex(ValueError, "Missing or changed artifact"):
                    runner.score_country(country, CONFIG, destination)


if __name__ == "__main__":
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DisabilityComponentTests))
    required = ["src/gbd_park/disability_components.py", "scripts/run_disability_components.py", "tests/test_disability_components.py",
                "study_design/supporting_outcomes_implementation.md"]
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  elapsed_seconds=time.perf_counter()-start,
                  fixture_type="synthetic_analytic_accounting_no_production_forecast_scores_read",
                  tested_code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in required})
    path = ROOT / "work/supporting-validation/component_tests.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
