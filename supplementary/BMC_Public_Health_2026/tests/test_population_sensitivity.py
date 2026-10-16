"""Synthetic scientific checks for fixed population-source sensitivity scenarios."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbd_park.population_sensitivity import (
    CELL, CONTEXT, NODE, SCENARIOS, aggregate, conditional_intervals,
    error_accounting, populations, projection_populations,
)


CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
TARGET = "Saudi Arabia"
OUTCOME = "prevalence"


def fixture(full_calendar=False):
    """Different source trends expose accidental anchoring to the final year."""
    config = copy.deepcopy(CONFIG)
    if not full_calendar:
        config["calendar"]["reliability_origins"] = [2014, 2018]
        config["calendar"]["horizons"] = [1, 5]
    years = list(range(2014, 2029))
    panel_rows, un_rows = [], []
    for country_index, country in enumerate(config["countries"]):
        for sex_index, sex in enumerate(config["sexes"]):
            for age_index, age in enumerate(config["ages"]):
                base = 100000 * (country_index + 2) + 7000 * sex_index + 1300 * age_index
                for year in years:
                    delta = year - 2014
                    gbd = base * (1 + .017 * delta + .0007 * delta**2)
                    un = base * (.8 + .02 * age_index) * (1 + .023 * delta + .0013 * delta**2)
                    un_rows.append(dict(location_name=country["name"], sex=sex,
                                        age=age, year=year, population_persons=un))
                    for outcome_index, outcome in enumerate(["prevalence", "incidence"]):
                        rate = float(30 + 2 * age_index + 4 * sex_index + 3 * outcome_index)
                        panel_rows.append(dict(location_name=country["name"], sex=sex, age=age,
                                               year=year, outcome=outcome, rate=rate,
                                               count=rate * gbd * (1 + .003 * outcome_index) / 100000))
    panel, un = pd.DataFrame(panel_rows), pd.DataFrame(un_rows)
    operational_rows = []
    source = panel.loc[panel.location_name.eq(TARGET) & panel.outcome.eq(OUTCOME)]
    for origin in config["calendar"]["reliability_origins"]:
        for row in source.loc[source.year.eq(origin)].itertuples():
            for horizon in config["calendar"]["horizons"]:
                for method in SCENARIOS[:2]:
                    pop = row.count / row.rate * 100000
                    if method == "log_trend_last8":
                        pop *= np.exp(.011 * horizon)
                    operational_rows.append(dict(target=TARGET, outcome=OUTCOME, origin=origin,
                                                 horizon=horizon, forecast_year=origin + horizon,
                                                 sex=row.sex, age=row.age, population_method=method,
                                                 population=pop))
    return config, panel, un, pd.DataFrame(operational_rows)


def cell_counts(**context):
    base = dict(target=TARGET, outcome=OUTCOME, origin=2014, horizon=5,
                forecast_year=2019, family="tcn_adapted", scenario="persistence")
    base.update(context)
    return pd.DataFrame([{**base, "sex": sex, "age": age, "count": 1.}
                         for sex in CONFIG["sexes"] for age in CONFIG["ages"]])


def paired_draws():
    """Opposing tails make median(total) differ from sum(marginal medians)."""
    frames = []
    young = [32., 16., 8., 4., 2., 1.]
    old = list(reversed(young))
    for block, (a, b) in enumerate(zip(young, old), start=2003):
        frame = cell_counts(residual_origin=block)
        frame.loc[frame.sex.eq("Male") & frame.age.eq("45-49"), "count"] = a
        frame.loc[frame.sex.eq("Male") & frame.age.eq("95+"), "count"] = b
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class HistoricalPopulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.panel, cls.un, cls.operational = fixture()

    def run_populations(self, panel=None, un=None, operational=None):
        return populations(self.panel if panel is None else panel,
                           self.un if un is None else un,
                           self.operational if operational is None else operational,
                           self.config, TARGET, OUTCOME)

    def test_alignment_uses_each_historical_origin_and_preserves_95_plus(self):
        actual = self.run_populations()
        aligned = actual.loc[actual.scenario.eq("un_2024_origin_aligned")]
        self.assertEqual(len(aligned), 2 * 2 * 2 * 11)
        self.assertEqual(set(aligned.age), set(CONFIG["ages"]))
        for row in aligned.itertuples():
            gbd = self.panel.loc[self.panel.location_name.eq(TARGET) &
                                 self.panel.outcome.eq(OUTCOME) & self.panel.sex.eq(row.sex) &
                                 self.panel.age.eq(row.age)].set_index("year")
            un = self.un.loc[self.un.location_name.eq(TARGET) & self.un.sex.eq(row.sex) &
                             self.un.age.eq(row.age)].set_index("year").population_persons
            at_origin = gbd.loc[row.origin, "count"] / gbd.loc[row.origin, "rate"] * 100000
            expected = at_origin * un.loc[row.forecast_year] / un.loc[row.origin]
            at_2023 = gbd.loc[2023, "count"] / gbd.loc[2023, "rate"] * 100000
            wrong_2023_anchor = at_2023 * un.loc[row.forecast_year] / un.loc[2023]
            self.assertAlmostEqual(row.population, expected, places=8)
            self.assertGreater(abs(row.population - wrong_2023_anchor), 1000)
            self.assertEqual(row.alignment_year, row.origin)

    def test_operational_values_and_explicit_future_information_labels(self):
        result = self.run_populations()
        expected_roles = {
            "log_trend_last8": "historical_operational",
            "persistence": "historical_operational",
            "gbd_realized_oracle": "realized_population_oracle",
            "un_2024_unaligned": "later_vintage_UN_source_scenario",
            "un_2024_origin_aligned": "later_vintage_UN_growth_scenario",
        }
        for scenario, role in expected_roles.items():
            part = result.loc[result.scenario.eq(scenario)]
            self.assertEqual(set(part.population_information_role), {role})
            self.assertEqual(set(part.uses_future_population_information),
                             {scenario not in SCENARIOS[:2]})
            if scenario != "un_2024_origin_aligned":
                self.assertTrue(part.alignment_year.isna().all())
            if scenario in SCENARIOS[:2]:
                saved = self.operational.loc[self.operational.population_method.eq(scenario)]
                merged = part.merge(saved, on=CELL, suffixes=("", "_saved"), validate="one_to_one")
                np.testing.assert_array_equal(merged.population, merged.population_saved)

    def test_diagnostic_inputs_can_use_future_population_and_operational_values_do_not_change(self):
        baseline = self.run_populations()
        modified_panel, modified_un = self.panel.copy(), self.un.copy()
        modified_panel.loc[modified_panel.year.eq(2023), "count"] *= 3
        modified_un.loc[modified_un.year.eq(2023), "population_persons"] *= 2
        changed = self.run_populations(modified_panel, modified_un)
        key = CELL + ["scenario"]
        compared = baseline.merge(changed, on=key, suffixes=("_old", "_new"), validate="one_to_one")
        operational = compared.scenario.isin(SCENARIOS[:2])
        np.testing.assert_array_equal(compared.loc[operational, "population_old"],
                                      compared.loc[operational, "population_new"])
        oracle = compared.scenario.eq("gbd_realized_oracle") & compared.forecast_year.eq(2023)
        np.testing.assert_allclose(compared.loc[oracle, "population_new"],
                                   3 * compared.loc[oracle, "population_old"])
        diagnostic = compared.scenario.isin(SCENARIOS[3:]) & compared.forecast_year.eq(2023)
        np.testing.assert_allclose(compared.loc[diagnostic, "population_new"],
                                   2 * compared.loc[diagnostic, "population_old"])

    def test_implied_population_and_rate_to_count_units_are_persons_and_per_100000(self):
        panel, un = self.panel.copy(), self.un.copy()
        panel["rate"], panel["count"] = 200., 600.
        un["population_persons"] = 450000.
        scenarios = self.run_populations(panel, un)
        oracle = scenarios.loc[scenarios.scenario.eq("gbd_realized_oracle")]
        np.testing.assert_array_equal(oracle.population, 300000.)
        unaligned = scenarios.loc[scenarios.scenario.eq("un_2024_unaligned")]
        np.testing.assert_array_equal(unaligned.population, 450000.)
        values = oracle.loc[oracle.origin.eq(2014) & oracle.horizon.eq(5)].copy()
        values["family"], values["count"] = "tcn_adapted", 200. * values.population / 100000
        nodes = aggregate(values, self.config)
        self.assertEqual(nodes.loc[nodes.node.eq("Both__45+"), "value"].item(), 22 * 600.)
        self.assertEqual(set(nodes.loc[nodes.measure.eq("count"), "unit"]), {"modeled_number"})
        self.assertEqual(set(nodes.loc[nodes.measure.eq("age_share"), "unit"]), {"percent"})

    def test_missing_origin_or_verification_population_is_rejected(self):
        for source in ["panel", "un"]:
            for year in [2014, 2023]:
                with self.subTest(source=source, year=year):
                    data = getattr(self, source)
                    absent = data.location_name.eq(TARGET) & data.sex.eq("Female") & data.age.eq("95+") & data.year.eq(year)
                    if source == "panel":
                        absent &= data.outcome.eq(OUTCOME)
                    with self.assertRaises(ValueError):
                        self.run_populations(**{source: data.loc[~absent]})

    def test_duplicate_or_nonpositive_source_population_is_rejected(self):
        for source in ["panel", "un"]:
            data = getattr(self, source)
            row = data.loc[data.location_name.eq(TARGET)].iloc[[0]]
            with self.subTest(source=source, defect="duplicate"):
                with self.assertRaises(ValueError):
                    self.run_populations(**{source: pd.concat([data, row], ignore_index=True)})
        for source, column in [("panel", "rate"), ("panel", "count"), ("un", "population_persons")]:
            for invalid in [0., -1., np.nan, np.inf]:
                with self.subTest(source=source, column=column, invalid=invalid):
                    data = getattr(self, source).copy()
                    data.loc[data.location_name.eq(TARGET), column] = invalid
                    with self.assertRaises(ValueError):
                        self.run_populations(**{source: data})

    def test_missing_duplicate_and_misdated_operational_cells_are_rejected(self):
        wrong_year = self.operational.copy()
        wrong_year.loc[0, "forecast_year"] += 1
        for data in [self.operational.iloc[1:],
                     pd.concat([self.operational, self.operational.iloc[[0]]], ignore_index=True), wrong_year]:
            with self.subTest(rows=len(data)):
                with self.assertRaises(ValueError):
                    self.run_populations(operational=data)


class AggregationAndConditionalIntervalTests(unittest.TestCase):
    def test_all_count_nodes_and_oldest_age_shares_are_retained(self):
        cells = cell_counts()
        cells.loc[cells.sex.eq("Male") & cells.age.eq("95+"), "count"] = 17.
        cells.loc[cells.sex.eq("Female") & cells.age.eq("95+"), "count"] = 23.
        actual = aggregate(cells.sample(frac=1, random_state=1), CONFIG)
        self.assertEqual(len(actual.loc[actual.measure.eq("count")]), 31)
        self.assertEqual(len(actual.loc[actual.measure.eq("age_share")]), 4)
        for sex, oldest, total in [("Male", 17., 27.), ("Female", 23., 33.)]:
            by_node = actual.loc[actual.sex.eq(sex)].set_index("node").value
            self.assertEqual(by_node.loc[f"{sex}__95+"], oldest)
            self.assertEqual(by_node.loc[f"{sex}__80+"], oldest + 3)
            self.assertAlmostEqual(by_node.loc[f"{sex}__80+_within_45+"], 100 * (oldest + 3) / total)
            self.assertAlmostEqual(by_node.loc[f"{sex}__65+_within_45+"], 100 * (oldest + 6) / total)
        self.assertEqual(actual.loc[actual.node.eq("Both__45+"), "value"].item(), 60.)

    def test_each_context_needs_unique_complete_cells_not_only_complete_union(self):
        first, second = cell_counts(), cell_counts(horizon=1, forecast_year=2015)
        incomplete = pd.concat([first, second.iloc[:-1]], ignore_index=True)
        duplicate = pd.concat([first, first.iloc[[0]]], ignore_index=True)
        wrong_age = first.copy()
        wrong_age.loc[wrong_age.age.eq("95+"), "age"] = "100+"
        for cells in [incomplete, duplicate, wrong_age]:
            with self.subTest(rows=len(cells)):
                with self.assertRaises(ValueError):
                    aggregate(cells, CONFIG)

    def test_joint_draw_aggregation_precedes_quantiles_and_preserves_pairing(self):
        cells = paired_draws()
        stats = aggregate(cells.sample(frac=1, random_state=4), CONFIG,
                          identifiers=CONTEXT + ["residual_origin"])
        intervals = conditional_intervals(stats.sample(frac=1, random_state=5))
        whole = intervals.loc[intervals.node.eq("Both__45+") & intervals.level.eq(.8)].iloc[0]
        expected_draws = np.array([53., 38., 32., 32., 38., 53.])
        np.testing.assert_allclose([whole.lower, whole["median"], whole.upper],
                                   np.quantile(expected_draws, [.1, .5, .9]))
        marginal_median_sum = intervals.loc[intervals.hierarchy_level.eq("age_sex") &
                                           intervals.level.eq(.8), "median"].sum()
        self.assertEqual(whole["median"], 38.)
        self.assertEqual(marginal_median_sum, 32.)
        share = intervals.loc[intervals.node.eq("Male__80+_within_45+") & intervals.level.eq(.8)].iloc[0]
        expected_shares = 100 * (np.array([1., 2., 4., 8., 16., 32.]) + 3) / (expected_draws - 11)
        np.testing.assert_allclose([share.lower, share["median"], share.upper],
                                   np.quantile(expected_shares, [.1, .5, .9]))
        self.assertEqual(set(intervals.n_blocks), {6})
        self.assertEqual(set(intervals.uncertainty_scope), {"rate_conditional_fixed_population"})
        self.assertEqual(set(intervals.level), {.5, .8, .95})

    def test_row_order_does_not_change_conditional_intervals(self):
        cells = paired_draws()
        first = aggregate(cells, CONFIG, identifiers=CONTEXT + ["residual_origin"])
        second = aggregate(cells.sample(frac=1, random_state=23), CONFIG,
                           identifiers=CONTEXT + ["residual_origin"])
        a, b = conditional_intervals(first), conditional_intervals(second)
        keys = CONTEXT + NODE + ["level"]
        pd.testing.assert_frame_equal(a.sort_values(keys).reset_index(drop=True),
                                      b.sort_values(keys).reset_index(drop=True))

    def test_duplicate_or_too_few_conditional_blocks_are_rejected(self):
        stats = aggregate(paired_draws(), CONFIG, identifiers=CONTEXT + ["residual_origin"])
        for invalid in [pd.concat([stats, stats.iloc[[0]]], ignore_index=True),
                        stats.loc[stats.residual_origin.lt(2007)]]:
            with self.assertRaises(ValueError):
                conditional_intervals(invalid)

    def test_missing_block_at_one_node_is_rejected_even_when_five_remain(self):
        stats = aggregate(paired_draws(), CONFIG, identifiers=CONTEXT + ["residual_origin"])
        missing = stats.node.eq("Both__45+") & stats.residual_origin.eq(2008)
        with self.assertRaises(ValueError):
            conditional_intervals(stats.loc[~missing])

    def test_inconsistent_block_identities_are_rejected_even_with_equal_counts(self):
        stats = aggregate(paired_draws(), CONFIG, identifiers=CONTEXT + ["residual_origin"])
        changed = stats.node.eq("Male__95+") & stats.residual_origin.eq(2008)
        stats.loc[changed, "residual_origin"] = 1999
        with self.assertRaises(ValueError):
            conditional_intervals(stats)


class ErrorAccountingTests(unittest.TestCase):
    def test_equal_and_opposite_rate_population_errors_can_cancel_exactly(self):
        values = pd.DataFrame(dict(prediction=[200.], observed_rate=[100.],
                                   population=[100000.], observed_population=[200000.]))
        result = error_accounting(values).iloc[0]
        self.assertEqual(result.predicted_count, 200.)
        self.assertEqual(result.observed_count, 200.)
        self.assertAlmostEqual(result.log_rate_error, np.log(2))
        self.assertAlmostEqual(result.log_population_error, -np.log(2))
        self.assertEqual(result.log_count_error, 0.)
        self.assertEqual(result.rate_effect, 150.)
        self.assertEqual(result.population_effect, -150.)

    def test_pure_rate_and_population_errors_have_the_expected_contributions(self):
        values = pd.DataFrame(dict(prediction=[250., 100.], observed_rate=[100., 100.],
                                   population=[200000., 400000.], observed_population=[200000., 200000.]))
        result = error_accounting(values)
        np.testing.assert_array_equal(result.rate_effect, [300., 0.])
        np.testing.assert_array_equal(result.population_effect, [0., 200.])
        np.testing.assert_allclose(result.log_count_error, [np.log(2.5), np.log(2)])

    def test_signed_effects_add_at_every_count_node_and_reverse_symmetrically(self):
        cells = cell_counts()
        rng = np.random.default_rng(214)
        cells["observed_rate"] = rng.uniform(10., 1000., len(cells))
        cells["observed_population"] = rng.uniform(1000., 1000000., len(cells))
        cells["prediction"] = cells.observed_rate * rng.uniform(.5, 2., len(cells))
        cells["population"] = cells.observed_population * rng.uniform(.5, 2., len(cells))
        accounted = error_accounting(cells)
        np.testing.assert_allclose(accounted.log_count_error,
                                   accounted.log_rate_error + accounted.log_population_error, atol=1e-13)
        nodes = {}
        for value in ["rate_effect", "population_effect", "predicted_count", "observed_count"]:
            nodes[value] = aggregate(accounted, CONFIG, value=value, allow_signed=True).set_index("node").value
        self.assertEqual(len(nodes["rate_effect"]), 31)
        np.testing.assert_allclose(nodes["rate_effect"] + nodes["population_effect"],
                                   nodes["predicted_count"] - nodes["observed_count"], atol=1e-10)
        reversed_values = cells.copy()
        reversed_values["prediction"], reversed_values["observed_rate"] = cells.observed_rate, cells.prediction
        reversed_values["population"], reversed_values["observed_population"] = cells.observed_population, cells.population
        backward = error_accounting(reversed_values)
        for field in ["rate_effect", "population_effect", "log_count_error"]:
            np.testing.assert_allclose(backward[field], -accounted[field], atol=1e-12)

    def test_signed_zero_effects_are_allowed_without_constructing_age_shares(self):
        cells = cell_counts()
        cells["rate_effect"] = 0.
        actual = aggregate(cells, CONFIG, value="rate_effect", allow_signed=True)
        self.assertEqual(set(actual.measure), {"count"})
        np.testing.assert_array_equal(actual.value, 0.)
        with self.assertRaises(ValueError):
            aggregate(cells, CONFIG, value="rate_effect")

    def test_nonpositive_and_nonfinite_accounting_inputs_are_rejected(self):
        values = pd.DataFrame(dict(prediction=[200.], observed_rate=[100.],
                                   population=[100000.], observed_population=[200000.]))
        for column in values:
            for invalid in [0., -1., np.nan, np.inf]:
                bad = values.copy()
                bad[column] = invalid
                with self.subTest(column=column, invalid=invalid):
                    with self.assertRaises(ValueError):
                        error_accounting(bad)


class ProjectionHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.panel, cls.un, _ = fixture()

    def test_projection_handoff_is_complete_2023_aligned_and_population_only(self):
        result = projection_populations(self.panel, self.un, self.config)
        countries = {item["name"] for item in self.config["countries"] if item["gcc"]}
        self.assertEqual(len(result), 6 * 2 * 2 * 11 * 5 * 2)
        self.assertEqual(set(result.location_name), countries)
        self.assertEqual(set(result.year), set(range(2024, 2029)))
        self.assertEqual(set(result.age), set(CONFIG["ages"]))
        self.assertEqual(set(result.origin), {2023})
        self.assertEqual(set(result.unit), {"persons"})
        self.assertEqual(set(result.role), {"population_only_handoff_not_disease_forecast"})
        self.assertEqual(set(result.scenario), {"un_medium_unaligned", "gbd_2023_aligned_un_growth"})
        self.assertNotIn("prediction", result.columns)
        self.assertNotIn("predicted_count", result.columns)
        keys = ["location_name", "outcome", "sex", "age", "year", "scenario"]
        self.assertFalse(result.duplicated(keys).any())
        for row in result.itertuples():
            un = self.un.loc[self.un.location_name.eq(row.location_name) &
                             self.un.sex.eq(row.sex) & self.un.age.eq(row.age)].set_index("year").population_persons
            if row.scenario == "un_medium_unaligned":
                expected = un.loc[row.year]
            else:
                baseline = self.panel.loc[self.panel.location_name.eq(row.location_name) &
                                           self.panel.sex.eq(row.sex) & self.panel.age.eq(row.age) &
                                           self.panel.outcome.eq(row.outcome) & self.panel.year.eq(2023)].iloc[0]
                expected = baseline["count"] / baseline.rate * 100000 * un.loc[row.year] / un.loc[2023]
            self.assertAlmostEqual(row.population, expected, places=7)

    def test_each_outcome_retains_its_own_implied_gbd_baseline(self):
        result = projection_populations(self.panel, self.un, self.config)
        keys = ["location_name", "sex", "age", "year", "scenario"]
        both = result.loc[result.outcome.eq("prevalence")].merge(
            result.loc[result.outcome.eq("incidence")], on=keys, suffixes=("_prev", "_inc"), validate="one_to_one")
        aligned = both.scenario.eq("gbd_2023_aligned_un_growth")
        np.testing.assert_allclose(both.loc[aligned, "population_inc"] / both.loc[aligned, "population_prev"], 1.003)
        np.testing.assert_array_equal(both.loc[~aligned, "population_inc"], both.loc[~aligned, "population_prev"])

    def test_missing_baseline_or_future_cells_are_rejected(self):
        for source, year in [("panel", 2023), ("un", 2023), ("un", 2028)]:
            data = getattr(self, source)
            absent = data.location_name.eq(TARGET) & data.sex.eq("Female") & data.age.eq("95+") & data.year.eq(year)
            with self.subTest(source=source, year=year):
                kwargs = {"panel": self.panel, "un": self.un, "config": self.config}
                kwargs[source] = data.loc[~absent]
                with self.assertRaises(ValueError):
                    projection_populations(**kwargs)

    def test_duplicate_projection_cells_are_rejected(self):
        for source in ["panel", "un"]:
            data = getattr(self, source)
            row = data.loc[data.year.eq(2023) & data.location_name.eq(TARGET)].iloc[[0]]
            kwargs = {"panel": self.panel, "un": self.un, "config": self.config}
            kwargs[source] = pd.concat([data, row], ignore_index=True)
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    projection_populations(**kwargs)

    def test_same_row_count_with_wrong_oldest_age_does_not_pass_completeness(self):
        panel, un = self.panel.copy(), self.un.copy()
        panel.loc[panel.age.eq("95+"), "age"] = "100+"
        un.loc[un.age.eq("95+"), "age"] = "100+"
        with self.assertRaises(ValueError):
            projection_populations(panel, un, self.config)

    def test_negative_rate_and_count_cannot_cancel_into_valid_population(self):
        panel = self.panel.copy()
        at_baseline = panel.year.eq(2023) & panel.location_name.eq(TARGET)
        panel.loc[at_baseline, ["rate", "count"]] *= -1
        with self.assertRaises(ValueError):
            projection_populations(panel, self.un, self.config)


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("population_sensitivity_runner_test",
                                                      ROOT / "scripts/run_population_sensitivity.py")
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_conditional_scores_use_interval_median_direction_and_50_80_wis(self):
        common = dict(target=TARGET, outcome=OUTCOME, origin=2018, horizon=5, forecast_year=2023,
                      family="tcn_adapted", scenario="persistence", measure="count", node="Both__45+",
                      sex="Both", age_group="45+", unit="modeled_number", hierarchy_level="grand_total")
        intervals = pd.DataFrame([{**common, "level": level, "lower": low, "upper": high,
                                   "median": 100., "n_blocks": 7,
                                   "uncertainty_scope": "rate_conditional_fixed_population"}
                                  for level, low, high in [(.5, 90., 110.), (.8, 80., 115.), (.95, 70., 130.)]])
        truth = pd.DataFrame([{**common, "observed": 120.}])
        scored, wis = self.runner.score(intervals, truth)
        np.testing.assert_allclose(scored.interval_score, [60., 85., 60.])
        np.testing.assert_array_equal(scored.covered, [False, False, True])
        np.testing.assert_array_equal(scored.above_upper, [True, True, False])
        self.assertFalse(scored.below_lower.any())
        self.assertAlmostEqual(wis.wis_50_80.item(), 13.4)
        self.assertAlmostEqual(wis.wis_50_80_95.item(), 10.)
        self.assertEqual(wis["median"].item(), 100.)

    def test_complete_case_reuses_inputs_commits_before_scoring_and_separates_uncertainty_scope(self):
        config, panel, un, operational = fixture(full_calendar=True)
        point_frames, draw_frames = [], []
        source = panel.loc[panel.location_name.eq(TARGET) & panel.outcome.eq(OUTCOME)]
        for origin in config["calendar"]["reliability_origins"]:
            native = source.loc[source.year.eq(origin)].set_index(["sex", "age"]).rate
            for family_index, family in enumerate(["tcn_adapted", "local_champion", "nonneural_champion"]):
                points = operational.loc[operational.origin.eq(origin) &
                                          operational.population_method.eq("persistence"), CELL].copy()
                points["family"] = family
                points["prediction"] = [native.loc[(sex, age)] * (1.05 + .01 * family_index)
                                         for sex, age in zip(points.sex, points.age)]
                point_frames.append(points)
                residual_origins = list(range(2003, origin - 4))
                for block in residual_origins:
                    draws = points.drop(columns="prediction").copy()
                    draws["residual_origin"] = block
                    direction = np.where(draws.sex.eq("Male"), 1., -1.)
                    scale = .015 * (block - np.mean(residual_origins))
                    draws["rate_draw"] = points.prediction * np.exp(direction * scale)
                    draw_frames.append(draws)
        points = pd.concat(point_frames, ignore_index=True)
        draws = pd.concat(draw_frames, ignore_index=True)
        # Only the shape and preserved numerical content of old reference intervals
        # matter here; their intentionally distinct width exposes accidental reuse.
        reference_cells = points.merge(operational.rename(columns={"population_method": "scenario"}),
                                        on=CELL, validate="many_to_many")
        reference_cells["count"] = reference_cells.prediction * reference_cells.population / 100000
        reference_points = aggregate(reference_cells, config)
        reference_frames = []
        for level in [.5, .8, .95]:
            part = reference_points.copy()
            part["level"], part["observed"], part["median"] = level, part.value * 1.01, part.value * 1.03
            part["lower"], part["upper"], part["width"] = part.value * .5, part.value * 2., part.value * 1.5
            part["covered"], part["n_blocks"] = True, part.origin - 2007
            reference_frames.append(part)
        reference = pd.concat(reference_frames, ignore_index=True).rename(columns={"scenario": "population_method"})
        reference_wis = reference_points.rename(columns={"scenario": "population_method"}).copy()
        reference_wis["wis_50_80"], reference_wis["wis_50_80_95"] = .123, .456
        with tempfile.TemporaryDirectory(prefix="population-sensitivity-test-") as directory:
            root = Path(directory)
            prepared = root / "data/processed/design_v1"
            old = root / "results/demography_v1/SYN_prevalence"
            saved = root / "synthetic_source"
            destination = root / "new_results"
            for folder in [prepared, old, saved, destination]:
                folder.mkdir(parents=True)
            panel.to_csv(prepared / "regional_outcomes.csv", index=False)
            un.to_csv(prepared / "un_population_1990_2028.csv", index=False)
            operational.to_csv(old / "population_forecasts.csv", index=False)
            points.to_csv(saved / "predictions.csv", index=False)
            draws.to_csv(saved / "joint_draws.csv", index=False)
            reference.to_csv(old / "interval_scores.csv.gz", index=False, compression="gzip")
            reference_wis.to_csv(old / "wis_scores.csv", index=False)
            inputs = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in root.rglob("*") if path.is_file()}
            case = dict(id="SYN_prevalence", target=TARGET, outcome=OUTCOME, source="synthetic_source")
            with patch.object(self.runner, "ROOT", root):
                validation = self.runner.case_job(case, config, destination)
                with self.assertRaises(FileExistsError):
                    self.runner.case_job(case, config, destination)
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["fitted_models"], 0)
            self.assertEqual(validation["minimum_blocks"], 7)
            self.assertEqual(validation["maximum_blocks"], 11)
            self.assertEqual(validation["conditional_interval_rows"], 3 * 5 * 5 * 5 * 35 * 3)
            for path, digest in inputs.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            out = destination / case["id"]
            commit = json.loads((out / "transformation_commit.json").read_text())
            self.assertLessEqual(commit["created_utc"], validation["scoring_started_utc"])
            self.assertTrue(commit["future_population_used_in_diagnostics"])
            self.assertEqual(commit["role"], "retrospective_sensitivity_not_new_holdout")
            for name, digest in commit["sha256"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), digest)
            conditional = pd.read_csv(out / "intervals.csv", float_precision="round_trip")
            joint = pd.read_csv(out / "original_joint_reference.csv.gz", float_precision="round_trip")
            joint_wis = pd.read_csv(out / "original_joint_wis_reference.csv", float_precision="round_trip")
            self.assertEqual(set(conditional.uncertainty_scope), {"rate_conditional_fixed_population"})
            self.assertEqual(set(joint.uncertainty_scope), {"joint_rate_population_original"})
            self.assertEqual(set(joint_wis.uncertainty_scope), {"joint_rate_population_original"})
            self.assertEqual(set(conditional.scenario), set(SCENARIOS))
            self.assertEqual(set(joint.scenario), set(SCENARIOS[:2]))
            original_reference = pd.read_csv(old / "interval_scores.csv.gz", float_precision="round_trip")
            np.testing.assert_array_equal(joint[["lower", "upper", "median"]],
                                          original_reference[["lower", "upper", "median"]])
            # Independent arithmetic reconstruction for all five population inputs
            # at the final endpoint; no aggregate()/conditional_intervals() oracle.
            pops = pd.read_csv(out / "population_inputs.csv", float_precision="round_trip")
            endpoints = draws.loc[draws.origin.eq(2018) & draws.horizon.eq(5) & draws.family.eq("tcn_adapted")]
            issued = pd.read_csv(out / "predictions.csv", float_precision="round_trip")
            frozen = points.loc[points.origin.eq(2018) & points.horizon.eq(5) & points.family.eq("tcn_adapted")]
            for scenario in SCENARIOS:
                pop = pops.loc[pops.origin.eq(2018) & pops.horizon.eq(5) & pops.scenario.eq(scenario)]
                joined = endpoints.merge(pop[["sex", "age", "population"]], on=["sex", "age"], validate="many_to_one")
                joined["count"] = joined.rate_draw * joined.population / 100000
                totals = joined.groupby("residual_origin")["count"].sum().to_numpy()
                interval = conditional.loc[conditional.origin.eq(2018) & conditional.horizon.eq(5) &
                                             conditional.family.eq("tcn_adapted") & conditional.scenario.eq(scenario) &
                                             conditional.node.eq("Both__45+") & conditional.level.eq(.8)].iloc[0]
                np.testing.assert_allclose([interval.lower, interval["median"], interval.upper],
                                           np.quantile(totals, [.1, .5, .9]), rtol=1e-12)
                point_cells = frozen.merge(pop[["sex", "age", "population"]], on=["sex", "age"], validate="one_to_one")
                expected_point = (point_cells.prediction * point_cells.population / 100000).sum()
                point = issued.loc[issued.origin.eq(2018) & issued.horizon.eq(5) & issued.family.eq("tcn_adapted") &
                                    issued.scenario.eq(scenario) & issued.node.eq("Both__45+"), "value"].item()
                self.assertAlmostEqual(point, expected_point, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
