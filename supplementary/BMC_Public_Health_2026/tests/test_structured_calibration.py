"""Synthetic leakage and analytic checks for fixed structured calibration."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from gbd_park.demography import count_and_share_views, joint_count_draws
from gbd_park.intervals import apply_bank, build_residual_bank
from gbd_park.structured_calibration import SETTINGS, structured_bank

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
SOURCES = {"Male": "damped_ets", "Female": "arima"}
ROLE = "local_champion"


def history_fixture():
    shocks = [-.08, .02, .05, -.04, .14, .01, -.02, .12, -.03, .18, .1, -.08, .16, .06, -.05, .11]
    records = []
    for origin in range(2003, 2019):
        for sex_index, sex in enumerate(CONFIG["sexes"]):
            for age_index, age in enumerate(CONFIG["ages"]):
                group = 0 if age_index < 4 else 1 if age_index < 7 else 2
                for horizon in CONFIG["calendar"]["horizons"]:
                    prediction = 100 + 5*age_index + 10*sex_index
                    shock = shocks[origin-2003]
                    residual = (.03*(sex_index-.5) + .01*group + .005*horizon
                                + shock*(1+.04*age_index)*(1 if shock < 0 else 1.5))
                    records.append(dict(target="Saudi Arabia", outcome="prevalence", family=SOURCES[sex],
                        origin=origin, sex=sex, age=age, horizon=horizon, forecast_year=origin+horizon,
                        prediction=prediction, log_prediction=np.log(prediction), observed_rate=prediction*np.exp(residual),
                        setting_id=f"{SOURCES[sex]}__historical_setting_at_{origin}", status="ok"))
    return pd.DataFrame(records)


def bank_fixture(history=None, fit_origin=2018):
    history = history_fixture() if history is None else history
    bank = build_residual_bank(history.assign(family=ROLE), CONFIG, fit_origin, ROLE)
    bank["source_family_by_sex"] = dict(SOURCES)
    bank["custom_preserved_metadata"] = {"proof": [1, 2, 3]}
    return bank


class StructuredCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = history_fixture()
        cls.bank = bank_fixture(cls.history)
        cls.transformed, cls.diagnostics = structured_bank(cls.history, cls.bank, CONFIG, SOURCES)

    def test_exact_constants_and_closed_form_horizon_counts_and_shrinkage(self):
        self.assertEqual(SETTINGS["quantiles"], [.2, .5, .8])
        self.assertEqual(SETTINGS["raw_tail_ratio_bounds"], [.5, 3.])
        self.assertEqual(len(self.diagnostics), 2*3*5)
        self.assertTrue(self.diagnostics.count_is_effective_sample_size.eq(False).all())
        expected_counts = {"1": 15, "2": 14, "3": 13, "4": 12, "5": 11}
        for row in self.diagnostics.itertuples():
            self.assertEqual(json.loads(row.origin_counts_by_horizon), expected_counts)
            self.assertEqual(row.distinct_origins_at_horizon, 16-row.horizon)
            self.assertEqual(row.last_label_year, 2018)
            weights = json.loads(row.horizon_kernel_weights)
            self.assertAlmostEqual(sum(weights.values()), 1)
            expected_n = sum(weight*expected_counts[h] for h, weight in weights.items())
            self.assertAlmostEqual(row.weighted_distinct_origin_count, expected_n)
            self.assertAlmostEqual(row.group_location_shrinkage, expected_n/(expected_n+20))
            self.assertAlmostEqual(row.scale_shrinkage, expected_n/(expected_n+8))
            self.assertAlmostEqual(row.location_group_weight+row.location_pool_weight+row.location_zero_weight, 1)
            self.assertAlmostEqual(row.location_zero_weight, 1-row.group_location_shrinkage)
        h1 = self.diagnostics.loc[self.diagnostics.horizon.eq(1)].iloc[0]
        h5 = self.diagnostics.loc[self.diagnostics.horizon.eq(5)].iloc[0]
        self.assertEqual(json.loads(h1.horizon_kernel_weights), {"1": 2/3, "2": 1/3})
        self.assertEqual(json.loads(h5.horizon_kernel_weights), {"4": 1/3, "5": 2/3})
        self.assertAlmostEqual(h1.weighted_distinct_origin_count, 44/3)
        self.assertAlmostEqual(h5.weighted_distinct_origin_count, 34/3)

    def test_quantiles_smoothed_before_ratios_and_fixed_formula_reconstructed(self):
        sex, group, horizon = "Female", "80+", 4
        ages = ["80-84", "85-89", "90-94", "95+"]
        weights = {3: .25, 4: .5, 5: .25}
        row = self.diagnostics.loc[self.diagnostics.sex.eq(sex) & self.diagnostics.age_group.eq(group)
                                   & self.diagnostics.horizon.eq(horizon)].iloc[0]
        q, ref = {}, {}
        for name, selected_ages in [("group", ages), ("pool", CONFIG["ages"])]:
            q[name], ref[name] = np.zeros(3), np.zeros(3)
            for h, weight in weights.items():
                part = self.history.loc[self.history.sex.eq(sex) & self.history.horizon.eq(h)
                    & self.history.origin.between(2003, 2018-h) & self.history.age.isin(selected_ages)]
                errors = np.log(part.observed_rate)-part.log_prediction
                q[name] += weight*np.quantile(errors, [.2, .5, .8], method="linear")
                take = self.bank["coords"].sex.eq(sex) & self.bank["coords"].horizon.eq(h) & self.bank["coords"].age.isin(selected_ages)
                ref[name] += weight*np.quantile(self.bank["centered_residuals"][:, take], [.2, .5, .8], method="linear")
            np.testing.assert_allclose(row[[name+"_observed_q20", name+"_observed_q50", name+"_observed_q80"]].to_numpy(float), q[name], atol=1e-15)
            np.testing.assert_allclose(row[[name+"_reference_q20", name+"_reference_q50", name+"_reference_q80"]].to_numpy(float), ref[name], atol=1e-15)
        n = .25*13+.5*12+.25*11
        a, b = n/(n+20), n/(n+8)
        self.assertAlmostEqual(row.location, a*a*q["group"][1]+a*(1-a)*q["pool"][1], places=14)
        ratios = {name: np.clip(np.diff(q[name])/np.diff(ref[name]), .5, 3.) for name in q}
        expected_scales = np.exp(b*(a*np.log(ratios["group"])+(1-a)*np.log(ratios["pool"])))
        np.testing.assert_allclose([row.left_scale, row.right_scale], expected_scales, atol=1e-14)

    def test_unavailable_fifth_horizon_ignored_but_matured_first_horizon_used(self):
        changed = self.history.copy()
        future = changed.origin.eq(2017) & changed.horizon.eq(5)
        changed.loc[future, ["prediction", "log_prediction", "observed_rate"]] = np.nan
        transformed, diagnostics = structured_bank(changed, self.bank, CONFIG, SOURCES)
        np.testing.assert_array_equal(transformed["centered_residuals"], self.transformed["centered_residuals"])
        pd.testing.assert_frame_equal(diagnostics, self.diagnostics)
        matured = changed.origin.eq(2017) & changed.horizon.eq(1) & changed.sex.eq("Male")
        changed.loc[matured, "observed_rate"] *= 3
        transformed, diagnostics = structured_bank(changed, self.bank, CONFIG, SOURCES)
        self.assertFalse(np.array_equal(transformed["centered_residuals"], self.transformed["centered_residuals"]))
        female = diagnostics.sex.eq("Female")
        pd.testing.assert_frame_equal(diagnostics.loc[female].reset_index(drop=True),
                                      self.diagnostics.loc[female].reset_index(drop=True))
        np.testing.assert_array_equal(self.bank["centered_residuals"], bank_fixture(self.history)["centered_residuals"])

    def test_all_future_and_unrelated_cells_and_source_bounds_do_not_influence_fit(self):
        changed = self.history.copy()
        future = (changed.origin+changed.horizon).gt(2018)
        changed.loc[future, ["prediction", "log_prediction", "observed_rate"]] = np.nan
        unrelated = pd.concat([self.history.assign(target="Jordan"), self.history.assign(outcome="incidence"),
                               self.history.assign(family="pooled_ridge")], ignore_index=True)
        unrelated[["prediction", "log_prediction", "observed_rate"]] = np.nan
        changed = pd.concat([changed, unrelated, changed.loc[future].iloc[:1]], ignore_index=True)
        changed["rate_lower"], changed["rate_upper"] = np.nan, np.inf
        actual, audit = structured_bank(changed.sample(frac=1, random_state=5), self.bank, CONFIG, SOURCES)
        np.testing.assert_array_equal(actual["centered_residuals"], self.transformed["centered_residuals"])
        pd.testing.assert_frame_equal(audit, self.diagnostics)

    def test_missing_duplicate_mistimed_eligible_cells_and_wrong_family_are_rejected(self):
        index = self.history.index[self.history.origin.eq(2017) & self.history.horizon.eq(1)][0]
        for kind in ["missing", "duplicate", "year", "nonpositive"]:
            with self.subTest(kind=kind):
                altered = self.history.copy()
                if kind == "missing":
                    altered = altered.drop(index=index)
                elif kind == "duplicate":
                    altered = pd.concat([altered, altered.loc[[index]]], ignore_index=True)
                elif kind == "year":
                    altered.loc[index, "forecast_year"] = 2019
                else:
                    altered.loc[index, "observed_rate"] = 0
                with self.assertRaises(ValueError):
                    structured_bank(altered, self.bank, CONFIG, SOURCES)
        with self.assertRaises(ValueError):
            structured_bank(self.history, self.bank, CONFIG, {"Male": "local_champion", "Female": "arima"})
        with self.assertRaises(ValueError):
            structured_bank(self.history, self.bank, CONFIG, {"Male": "arima", "Female": "arima"})

    def test_original_bank_history_and_point_forecasts_remain_unchanged(self):
        bank, history = deepcopy(self.bank), self.history.copy(deep=True)
        fingerprint = joblib.hash(bank)
        actual, diagnostics = structured_bank(history, bank, CONFIG, SOURCES)
        self.assertEqual(joblib.hash(bank), fingerprint)
        pd.testing.assert_frame_equal(history, self.history)
        for field in ["raw_residuals", "center"]:
            np.testing.assert_array_equal(actual[field], bank[field])
            self.assertIsNot(actual[field], bank[field])
        np.testing.assert_array_equal(actual["structured_original_centered_residuals"], bank["centered_residuals"])
        self.assertEqual(actual["custom_preserved_metadata"], bank["custom_preserved_metadata"])
        self.assertEqual(actual["origins"], bank["origins"])
        self.assertIn("not_necessarily_coordinate_mean_zero", actual["residual_field_role"])
        points = self.history.loc[self.history.origin.eq(2018)].drop(columns="observed_rate").assign(family=ROLE)
        point_copy = points.copy(deep=True)
        original_intervals, _ = apply_bank(points, bank, CONFIG)
        intervals, _ = apply_bank(points, actual, CONFIG)
        np.testing.assert_array_equal(intervals.point_prediction, original_intervals.point_prediction)
        pd.testing.assert_frame_equal(points, point_copy)
        with self.assertRaises(ValueError):
            structured_bank(history, actual, CONFIG, SOURCES)

    def test_positive_asymmetric_piecewise_map_preserves_each_coordinate_rank(self):
        old = self.bank["centered_residuals"]
        new = self.transformed["centered_residuals"]
        self.assertTrue((self.diagnostics[["left_scale", "right_scale"]] > 0).all().all())
        self.assertTrue((abs(self.diagnostics.left_scale-self.diagnostics.right_scale) > 1e-6).any())
        for column in range(old.shape[1]):
            np.testing.assert_array_equal(np.argsort(new[:, column], kind="stable"), np.argsort(old[:, column], kind="stable"))
        for row in self.diagnostics.itertuples():
            take = (self.bank["coords"].sex.eq(row.sex) & self.bank["coords"].horizon.eq(row.horizon)
                    & self.bank["coords"].age.isin(row.age_bands.split("|")))
            z = old[:, take]
            expected = row.location+row.left_scale*np.minimum(z, 0)+row.right_scale*np.maximum(z, 0)
            np.testing.assert_array_equal(new[:, take], expected)
            self.assertLess(row.location-row.left_scale*1e-6, row.location)
            self.assertGreater(row.location+row.right_scale*1e-6, row.location)

    def test_age_groups_preserve_all_bands_and_80_boundary(self):
        for group, ages in {"45-64": ["45-49", "50-54", "55-59", "60-64"],
                            "65-79": ["65-69", "70-74", "75-79"],
                            "80+": ["80-84", "85-89", "90-94", "95+"]}.items():
            rows = self.diagnostics.loc[self.diagnostics.age_group.eq(group)]
            self.assertEqual(len(rows), 10)
            self.assertEqual(set(rows.age_bands), {"|".join(ages)})
            self.assertEqual(set(rows.n_age_bands), {len(ages)})
        config = deepcopy(CONFIG)
        config["age_groups"]["80+"] = [80, 85, 90]
        with self.assertRaises(ValueError):
            structured_bank(self.history, self.bank, config, SOURCES)

    def test_zero_reference_spread_falls_back_to_one_with_explicit_flags(self):
        history = self.history.copy()
        history["observed_rate"] = history.prediction*np.exp(.1)
        bank = bank_fixture(history)
        actual, audit = structured_bank(history, bank, CONFIG, SOURCES)
        for prefix in ["group", "pool"]:
            for side in ["left", "right"]:
                self.assertTrue(audit[prefix+"_"+side+"_zero_reference"].all())
                np.testing.assert_array_equal(audit[prefix+"_"+side+"_ratio"], 1)
        np.testing.assert_array_equal(audit.left_scale, 1)
        np.testing.assert_array_equal(audit.right_scale, 1)
        np.testing.assert_allclose(audit.location, .1*audit.group_location_shrinkage, atol=1e-15)
        self.assertTrue(np.isfinite(actual["centered_residuals"]).all())

    def test_joint_population_pairing_and_count_distribution_remain_whole_blocks(self):
        points = self.history.loc[self.history.origin.eq(2018)].drop(columns="observed_rate").assign(family=ROLE)
        _, draws = apply_bank(points, self.transformed, CONFIG)
        self.assertEqual(sorted(draws.residual_origin.unique()), self.bank["origins"])
        population = points[["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]].copy()
        population["population_method"] = "persistence"
        population["log_population"] = np.log(10000.)
        errors = []
        for residual_origin in self.bank["origins"]:
            block = population.drop(columns=["forecast_year", "log_population"]).copy()
            block["origin"], block["residual_origin"], block["fit_origin"] = residual_origin, residual_origin, 2018
            block["centered_population_log_error"] = .03*(residual_origin-2008)
            errors.append(block)
        errors = pd.concat(errors, ignore_index=True)
        fingerprint = joblib.hash((population, errors))
        joint = joint_count_draws(draws.sample(frac=1, random_state=8), population,
                                 errors.sample(frac=1, random_state=9))
        self.assertEqual(joblib.hash((population, errors)), fingerprint)
        expected = np.exp(joint.log_draw+np.log(10000.)+.03*(joint.residual_origin-2008))/100000
        np.testing.assert_allclose(joint["count"], expected, rtol=1e-13)
        counts, _ = count_and_share_views(joint, CONFIG, identifiers=["residual_origin", "horizon"])
        total = joint.loc[joint.horizon.eq(5)].groupby("residual_origin")["count"].sum()
        aggregate = counts.loc[counts.horizon.eq(5) & counts.node.eq("Both__45+")].set_index("residual_origin")["count"]
        np.testing.assert_allclose(aggregate.reindex(total.index), total, rtol=1e-13)
        np.testing.assert_allclose(np.quantile(aggregate, [.1, .5, .9], method="linear"),
                                   np.quantile(total, [.1, .5, .9], method="linear"), rtol=1e-13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
