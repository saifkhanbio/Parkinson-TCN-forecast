"""Synthetic chronology, selection and joint-transformation tests for amendment v1.2."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
import pandas as pd
from gbd_park.calibration import development_losses, scale_bank, select_factor
from gbd_park.demography import count_and_share_views, joint_count_draws, rate_ratio_views
from gbd_park.intervals import apply_bank, build_residual_bank
from run_calibration import labels, rate_summaries, require_source, underlying_family
from run_demography import burden_statistics, distribution_summary

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
AMENDMENT = json.loads((ROOT / "study_design/interval_calibration_v1_2.json").read_text())
FAMILIES = CONFIG["models"]["local_order"] + CONFIG["models"]["nonneural_order"] + [
    "tcn_adapted", "tcn_intercept", "tcn_unadapted"]
FAMILY = "tcn_adapted"
COORDS = ["sex", "age", "horizon"]


def history_fixture():
    rows = []
    for origin in range(2003, 2014):
        for sex_index, sex in enumerate(CONFIG["sexes"]):
            for age_index, age in enumerate(CONFIG["ages"]):
                for horizon in CONFIG["calendar"]["horizons"]:
                    prediction = 100 + age_index * 3 + sex_index * 2
                    residual = .2 + sex_index * .1 + .02 * (origin-2005) * (1+age_index/10) * horizon/5
                    rows.append(dict(target="Saudi Arabia", outcome="prevalence", origin=origin,
                        sex=sex, age=age, horizon=horizon, forecast_year=origin+horizon,
                        prediction=prediction, log_prediction=np.log(prediction), observed_rate=prediction*np.exp(residual),
                        setting_id=f"synthetic_at_{origin}", status="ok", fallback_reason=""))
    source = pd.DataFrame(rows)
    return pd.concat([source.assign(family=family) for family in FAMILIES], ignore_index=True)


def loss_fixture():
    return pd.DataFrame([dict(country=country, outcome="prevalence", family=FAMILY, sex="Male",
        development_origin=origin, factor=factor, normalizer=.2, mean_log_wis=.2, normalized_wis=1.,
        last_label_year=origin+5, bank_n_blocks=origin-2007, last_residual_label_year=origin)
        for country in AMENDMENT["countries"] for origin in AMENDMENT["development_origins"]
        for factor in AMENDMENT["factor_grid"]])


def manual_wis(observed, draws):
    median = np.quantile(draws, .5, axis=0, method="linear")
    total = .5 * abs(observed-median)
    for level in [.5, .8]:
        alpha = 1-level
        lower, upper = np.quantile(draws, [alpha/2, 1-alpha/2], axis=0, method="linear")
        interval_score = upper-lower + 2/alpha*np.maximum(lower-observed, 0) + 2/alpha*np.maximum(observed-upper, 0)
        total += alpha/2*interval_score
    return total/2.5


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = history_fixture()
        cls.losses = development_losses(cls.history, CONFIG, AMENDMENT)

    def test_development_grid_and_direct_raw_normalization_and_wis(self):
        self.assertEqual(len(self.losses), 14*2*2*5)
        self.assertEqual(set(self.losses.family), set(FAMILIES))
        for origin in [2012, 2013]:
            bank = build_residual_bank(self.history, CONFIG, origin, FAMILY)
            points = self.history.loc[self.history.family.eq(FAMILY) & self.history.origin.eq(origin)]
            for sex in CONFIG["sexes"]:
                coord = bank["coords"].sex.eq(sex) & bank["coords"].horizon.eq(5)
                raw = bank["raw_residuals"][:, coord]
                normalizer = max(np.mean(abs(raw)), 1e-6)
                centered = raw-raw.mean(axis=0)
                current = points.loc[points.sex.eq(sex) & points.horizon.eq(5)].set_index("age").reindex(CONFIG["ages"])
                self.assertGreater(abs(normalizer-np.mean(abs(centered))), .1)
                for factor in AMENDMENT["factor_grid"]:
                    draws = current.log_prediction.to_numpy()[None, :] + factor*centered
                    expected_wis = manual_wis(np.log(current.observed_rate.to_numpy()), draws).mean()
                    actual = self.losses.loc[self.losses.family.eq(FAMILY) & self.losses.sex.eq(sex)
                        & self.losses.development_origin.eq(origin) & self.losses.factor.eq(factor)].iloc[0]
                    self.assertAlmostEqual(actual.normalizer, normalizer, places=13)
                    self.assertAlmostEqual(actual.mean_log_wis, expected_wis, places=13)
                    self.assertAlmostEqual(actual.normalized_wis, expected_wis/normalizer, places=13)
                    self.assertEqual(actual.bank_n_blocks, origin-2007)
                    self.assertEqual(actual.last_residual_label_year, origin)
                    self.assertEqual(actual.last_label_year, origin+5)

    def test_future_rows_source_bounds_and_input_order_do_not_change_losses(self):
        future = self.history.loc[self.history.origin.eq(2013)].copy()
        future["origin"], future["forecast_year"], future["family"] = 2014, 9999, "unknown_future_family"
        future[["prediction", "log_prediction", "observed_rate"]] = np.nan
        changed = pd.concat([self.history, future, future.iloc[:1]], ignore_index=True)
        changed["rate_lower"], changed["rate_upper"] = np.nan, np.inf
        changed = changed.sample(frac=1, random_state=14)
        actual = development_losses(changed, CONFIG, AMENDMENT)
        order = ["country", "outcome", "family", "sex", "development_origin", "factor"]
        pd.testing.assert_frame_equal(actual.sort_values(order).reset_index(drop=True),
                                      self.losses.sort_values(order).reset_index(drop=True))

    def test_development_truth_changes_loss_but_not_past_only_normalizer(self):
        changed = self.history.copy()
        mask = changed.origin.eq(2012) & changed.sex.eq("Male")
        changed.loc[mask, "observed_rate"] *= 3
        actual = development_losses(changed, CONFIG, AMENDMENT)
        order = ["country", "outcome", "family", "sex", "development_origin", "factor"]
        actual = actual.sort_values(order).reset_index(drop=True)
        expected = self.losses.sort_values(order).reset_index(drop=True)
        np.testing.assert_array_equal(actual.normalizer, expected.normalizer)
        unaffected = actual.sex.eq("Female") | actual.development_origin.eq(2013)
        pd.testing.assert_frame_equal(actual.loc[unaffected].reset_index(drop=True), expected.loc[unaffected].reset_index(drop=True))
        self.assertTrue((actual.loc[~unaffected, "mean_log_wis"] > expected.loc[~unaffected, "mean_log_wis"]).all())

    def test_zero_residual_normalization_floor(self):
        history = self.history.copy()
        history["observed_rate"] = history.prediction
        losses = development_losses(history, CONFIG, AMENDMENT)
        np.testing.assert_array_equal(losses.normalizer, AMENDMENT["normalization_floor"])
        np.testing.assert_allclose(losses.normalized_wis, 0, atol=1e-8)

    def test_incomplete_duplicate_family_and_mistimed_cells_are_rejected(self):
        eligible = self.history.index[self.history.family.eq(FAMILY) & self.history.origin.eq(2003)][0]
        for kind in ["missing", "duplicate", "family", "year"]:
            with self.subTest(kind=kind):
                altered = self.history.copy()
                if kind == "missing":
                    altered = altered.drop(index=eligible)
                elif kind == "duplicate":
                    altered = pd.concat([altered, altered.loc[[eligible]]], ignore_index=True)
                elif kind == "family":
                    altered = altered.loc[~altered.family.eq(FAMILY)]
                else:
                    altered.loc[eligible, "forecast_year"] += 1
                with self.assertRaises(ValueError):
                    development_losses(altered, CONFIG, AMENDMENT)


class SelectionTests(unittest.TestCase):
    def choose(self, losses=None, **kwargs):
        settings = dict(target="Saudi Arabia", outcome="prevalence", source_family=FAMILY,
                        sex="Male", fit_origin=2018, policy="target_only")
        settings.update(kwargs)
        supplied = (loss_fixture() if losses is None else losses).copy()
        supplied["mean_log_wis"] = supplied.normalizer*supplied.normalized_wis
        return select_factor(supplied, AMENDMENT, **settings)

    def test_cold_selection_ignores_all_unavailable_values(self):
        losses = loss_fixture()
        losses[["normalized_wis", "normalizer", "mean_log_wis"]] = np.nan
        losses = pd.concat([losses, losses.iloc[:1]], ignore_index=True)
        for origin in [2014, 2015, 2016]:
            result = self.choose(losses, fit_origin=origin, policy="gcc_assisted")
            self.assertEqual(result["factor"], 1.)
            self.assertEqual(result["eligible_origins"], [])
            self.assertEqual(result["candidate_objectives"], [])
            self.assertIsNone(result["last_label_year"])

    def test_completed_development_origins_only_and_future_nan_invariance(self):
        losses = loss_fixture()
        target = losses.country.eq("Saudi Arabia")
        losses.loc[target & losses.development_origin.eq(2012) & losses.factor.eq(1), "normalized_wis"] = 0
        losses.loc[target & losses.development_origin.eq(2013) & losses.factor.eq(1), "normalized_wis"] = 10
        losses.loc[target & losses.development_origin.eq(2013) & losses.factor.eq(2), "normalized_wis"] = 0
        at2017 = self.choose(losses, fit_origin=2017)
        self.assertEqual(at2017["factor"], 1.)
        self.assertEqual(at2017["eligible_origins"], [2012])
        self.assertEqual(at2017["last_label_year"], 2017)
        changed = losses.copy()
        changed.loc[changed.development_origin.eq(2013), "normalized_wis"] = np.nan
        changed = pd.concat([changed, changed.loc[changed.development_origin.eq(2013)].iloc[:1]])
        self.assertEqual(at2017, self.choose(changed, fit_origin=2017))
        at2018 = self.choose(losses)
        self.assertEqual(at2018["factor"], 2.)
        self.assertEqual(at2018["eligible_origins"], [2012, 2013])
        self.assertEqual(at2018["last_label_year"], 2018)

    def test_half_target_half_equal_five_donor_country_weights(self):
        losses = loss_fixture()
        losses["normalized_wis"] = 20.
        for country in AMENDMENT["countries"]:
            values = (1., .8) if country == "Saudi Arabia" else (0., 10.) if country == "Bahrain" else (10., 0.)
            for factor, value in zip([1., 2.], values):
                losses.loc[losses.country.eq(country) & losses.factor.eq(factor), "normalized_wis"] = value
        result = self.choose(losses, policy="gcc_assisted")
        self.assertEqual(result["factor"], 2.)
        self.assertAlmostEqual(result["target_loss"], .8)
        self.assertAlmostEqual(result["donor_loss"], 2.)
        self.assertAlmostEqual(result["unpenalized_loss"], 1.4)
        self.assertAlmostEqual(result["penalty"], .05*np.log(2)**2)
        self.assertAlmostEqual(result["objective"], 1.4+.05*np.log(2)**2)
        self.assertNotIn("Saudi Arabia", result["donor_countries"])
        self.assertEqual(len(result["donor_countries"]), 5)
        self.assertEqual(result["country_weights"]["Saudi Arabia"], .5)
        for country in result["donor_countries"]:
            self.assertAlmostEqual(result["country_weights"][country], .1)
        self.assertEqual(result, self.choose(losses.sample(frac=1, random_state=23), policy="gcc_assisted"))

    def test_country_outcome_sex_and_current_source_family_filtering(self):
        losses = loss_fixture()
        losses.loc[losses.country.eq("Saudi Arabia") & losses.factor.eq(1.5), "normalized_wis"] = 0
        expected = self.choose(losses)
        self.assertEqual(expected["factor"], 1.5)
        unrelated = pd.concat([losses.assign(country="Jordan"), losses.assign(outcome="incidence"),
                               losses.assign(sex="Female"), losses.assign(family="arima")], ignore_index=True)
        unrelated["normalized_wis"] = np.nan
        self.assertEqual(expected, self.choose(pd.concat([losses, unrelated], ignore_index=True)))
        target_only = losses.loc[losses.country.eq("Saudi Arabia")]
        self.assertEqual(expected, self.choose(target_only))
        with self.assertRaises(ValueError):
            self.choose(target_only, policy="gcc_assisted")
        with self.assertRaises(ValueError):
            self.choose(losses, source_family="local_champion")

    def test_penalty_and_exact_tie_choose_smaller_fixed_factor(self):
        losses = loss_fixture()
        self.assertEqual(self.choose(losses)["factor"], 1.)
        losses.loc[losses.factor.eq(2), "normalized_wis"] = 1.-.05*np.log(2)**2
        result = self.choose(losses)
        self.assertEqual(result["factor"], 1.)
        self.assertEqual(sorted(row["factor"] for row in result["candidate_objectives"]), AMENDMENT["factor_grid"])

    def test_inconsistent_normalization_is_rejected(self):
        losses = loss_fixture()
        losses.loc[0, "mean_log_wis"] = .9
        with self.assertRaises(ValueError):
            select_factor(losses, AMENDMENT, "Saudi Arabia", "prevalence", FAMILY, "Male", 2018, "target_only")

    def test_missing_duplicate_nonfinite_or_extra_factor_grid_is_rejected(self):
        for kind in ["cell", "country", "duplicate", "nan", "extra", "year"]:
            with self.subTest(kind=kind):
                losses = loss_fixture()
                if kind == "cell":
                    losses = losses.iloc[1:]
                elif kind == "country":
                    losses = losses.loc[~losses.country.eq("Bahrain")]
                elif kind == "duplicate":
                    losses = pd.concat([losses, losses.iloc[:1]], ignore_index=True)
                elif kind == "nan":
                    losses.loc[0, "normalized_wis"] = np.nan
                elif kind == "extra":
                    losses = pd.concat([losses, losses.loc[losses.factor.eq(1)].assign(factor=4)], ignore_index=True)
                else:
                    losses.loc[0, "last_label_year"] = 2019
                with self.assertRaises(ValueError):
                    self.choose(losses, policy="gcc_assisted")


class JointTransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = history_fixture()

    def bank(self):
        return build_residual_bank(self.history, CONFIG, 2012, FAMILY)

    def points(self):
        return self.history.loc[self.history.family.eq(FAMILY) & self.history.origin.eq(2012)].drop(columns="observed_rate")

    def test_factor_one_exact_control_deep_copy_and_shuffled_point_order(self):
        bank = self.bank()
        before = joblib.hash(bank)
        scaled = scale_bank(bank, {"Male": 1., "Female": 1.}, CONFIG)
        self.assertEqual(joblib.hash(bank), before)
        for field in ["raw_residuals", "centered_residuals", "center"]:
            np.testing.assert_array_equal(scaled[field], bank[field])
            self.assertIsNot(scaled[field], bank[field])
        original_intervals, original_draws = apply_bank(self.points(), bank, CONFIG)
        intervals, draws = apply_bank(self.points().sample(frac=1, random_state=5), scaled, CONFIG)
        pd.testing.assert_frame_equal(intervals, original_intervals)
        pd.testing.assert_frame_equal(draws, original_draws)
        scaled["coords"].loc[0, "age"] = "altered copy"
        scaled["raw_residuals"][0, 0] = 999
        self.assertEqual(joblib.hash(bank), before)

    def test_sex_scales_preserve_raw_center_and_complete_joint_covariance(self):
        bank = self.bank()
        scaled = scale_bank(bank, {"Male": 3., "Female": 1.25}, CONFIG)
        factors = bank["coords"].sex.map({"Male": 3., "Female": 1.25}).to_numpy()
        np.testing.assert_array_equal(scaled["centered_residuals"], bank["centered_residuals"]*factors)
        np.testing.assert_array_equal(scaled["raw_residuals"], bank["raw_residuals"])
        np.testing.assert_array_equal(scaled["center"], bank["center"])
        np.testing.assert_allclose(scaled["centered_residuals"].mean(axis=0), 0, atol=2e-15)
        expected_covariance = np.cov(bank["centered_residuals"], rowvar=False)*np.outer(factors, factors)
        np.testing.assert_allclose(np.cov(scaled["centered_residuals"], rowvar=False), expected_covariance, atol=1e-16)
        self.assertEqual(scaled["origins"], bank["origins"])
        with self.assertRaises(ValueError):
            scale_bank(scaled, {"Male": 1., "Female": 1.}, CONFIG)
        for mapping in [{"Male": 1}, {"Male": 0, "Female": 1}, {"Male": np.inf, "Female": 1}]:
            with self.assertRaises(ValueError):
                scale_bank(bank, mapping, CONFIG)

    def test_asymmetric_residuals_shift_median_without_point_clipping_or_nested_bounds(self):
        bank = self.bank()
        values = np.array([-.5, -.4, -.3, -.2, 1.4])
        bank["centered_residuals"][:] = values[:, None]
        bank["raw_residuals"] = bank["centered_residuals"] + bank["center"]
        original, _ = apply_bank(self.points(), bank, CONFIG)
        wider, _ = apply_bank(self.points(), scale_bank(bank, {"Male": 3., "Female": 3.}, CONFIG), CONFIG)
        keep = original.scale.eq("log_rate") & original.level.eq(.5)
        self.assertTrue((wider.loc[keep, "upper"] < original.loc[keep, "upper"]).all())
        self.assertTrue((wider.loc[keep, "median"] < wider.loc[keep, "point_prediction"]).all())
        np.testing.assert_array_equal(wider.point_prediction, original.point_prediction)

    def test_paired_counts_shares_ratios_and_direct_transformed_quantiles(self):
        bank = self.bank()
        # Opposite sex loadings prevent a sum of marginal quantiles from
        # accidentally agreeing with a correctly paired total distribution.
        female = bank["coords"].sex.eq("Female").to_numpy()
        bank["centered_residuals"][:, female] *= -1
        bank["raw_residuals"] = bank["center"] + bank["centered_residuals"]
        scaled = scale_bank(bank, {"Male": 2., "Female": 1.25}, CONFIG)
        points = self.points()
        intervals, draws = apply_bank(points, scaled, CONFIG)
        population = points[["target", "outcome", "origin", "sex", "age", "horizon", "forecast_year"]].copy()
        population["population_method"] = "log_trend_last8"
        population["log_population"] = np.log(10000.)
        errors = []
        for block_index, residual_origin in enumerate(bank["origins"]):
            part = population.drop(columns=["log_population", "forecast_year"]).copy()
            part["origin"], part["fit_origin"], part["residual_origin"] = residual_origin, 2012, residual_origin
            part["centered_population_log_error"] = .03*(block_index-2)
            errors.append(part)
        errors = pd.concat(errors, ignore_index=True)
        joint = joint_count_draws(draws.sample(frac=1, random_state=10), population,
                                  errors.sample(frac=1, random_state=11))
        expected_count = np.exp(joint.log_draw + np.log(10000.) + .03*(joint.residual_origin-2005))/100000
        np.testing.assert_allclose(joint["count"], expected_count, rtol=1e-13)
        counts, shares = count_and_share_views(joint, CONFIG, identifiers=["residual_origin", "horizon"])
        for (residual_origin, horizon), part in joint.groupby(["residual_origin", "horizon"]):
            observed = counts.loc[counts.residual_origin.eq(residual_origin) & counts.horizon.eq(horizon)]
            self.assertAlmostEqual(observed.loc[observed.node.eq("Both__45+"), "count"].iloc[0], part["count"].sum())
            for sex in CONFIG["sexes"]:
                same_sex = part.loc[part.sex.eq(sex)]
                expected_share = same_sex.loc[same_sex.age.isin(["80-84", "85-89", "90-94", "95+"]), "count"].sum()/same_sex["count"].sum()
                actual_share = shares.loc[shares.residual_origin.eq(residual_origin) & shares.horizon.eq(horizon)
                                          & shares.sex.eq(sex) & shares.threshold.eq(80), "share"].iloc[0]
                self.assertAlmostEqual(actual_share, expected_share)
        transformed = burden_statistics(joint, CONFIG, True)
        distribution = distribution_summary(transformed)
        h5 = joint.loc[joint.horizon.eq(5)]
        total_draws = h5.groupby("residual_origin")["count"].sum()
        total_interval = distribution.loc[distribution.node.eq("Both__45+") & distribution.horizon.eq(5)
                                          & distribution.level.eq(.8)].iloc[0]
        np.testing.assert_allclose(total_interval[["lower", "median", "upper"]].to_numpy(float),
                                   np.quantile(total_draws, [.1, .5, .9], method="linear"), rtol=1e-13)
        marginal_lower_sum = h5.groupby(["sex", "age"])["count"].quantile(.1, interpolation="linear").sum()
        self.assertGreater(abs(total_interval.lower-marginal_lower_sum), .05)
        female_h5 = h5.loc[h5.sex.eq("Female")]
        older = female_h5.loc[female_h5.age.isin(["80-84", "85-89", "90-94", "95+"])]
        share_draws = 100*older.groupby("residual_origin")["count"].sum()/female_h5.groupby("residual_origin")["count"].sum()
        share_interval = distribution.loc[distribution.measure.eq("age_share") & distribution.sex.eq("Female")
            & distribution.age_group.eq("80+_within_45+") & distribution.horizon.eq(5) & distribution.level.eq(.8)].iloc[0]
        np.testing.assert_allclose(share_interval[["lower", "median", "upper"]].to_numpy(float),
                                   np.quantile(share_draws, [.1, .5, .9], method="linear"), rtol=1e-13)
        ratios = rate_ratio_views(draws, CONFIG, value="rate_draw", identifiers=["residual_origin", "age", "horizon"])
        one = draws.loc[draws.age.eq("80-84") & draws.horizon.eq(5)].pivot(index="residual_origin", columns="sex", values="rate_draw")
        direct = one.Male/one.Female
        actual = ratios.loc[ratios.age.eq("80-84") & ratios.horizon.eq(5)].set_index("residual_origin").male_female_rate_ratio
        np.testing.assert_allclose(actual.reindex(direct.index), direct, rtol=1e-13)
        lower = np.quantile(direct, .1, method="linear")
        ratio_of_marginal_bounds = np.quantile(one.Male, .1, method="linear")/np.quantile(one.Female, .9, method="linear")
        self.assertGreater(abs(lower-ratio_of_marginal_bounds), 1e-5)
        rate_cell = intervals.loc[intervals.sex.eq("Male") & intervals.age.eq("80-84") & intervals.horizon.eq(5)
                                   & intervals.scale.eq("rate") & intervals.level.eq(.8)].iloc[0]
        np.testing.assert_allclose([rate_cell.lower, rate_cell["median"], rate_cell.upper],
                                   np.quantile(one.Male, [.1, .5, .9], method="linear"), rtol=1e-13)
        log_cell = intervals.loc[intervals.sex.eq("Male") & intervals.age.eq("80-84") & intervals.horizon.eq(5)
                                  & intervals.scale.eq("log_rate") & intervals.level.eq(.8)].iloc[0]
        self.assertNotAlmostEqual(rate_cell.lower, np.exp(log_cell.lower), places=6)


class RunnerGuardTests(unittest.TestCase):
    def test_current_frozen_family_mapping_ignores_future_rows_and_rejects_future_labels(self):
        mapping = pd.DataFrame([dict(fit_origin=2014, role="local_champion", sex="Male",
                                     source_family="arima", last_selection_target_year=2014),
                                dict(fit_origin=2018, role="local_champion", sex="Male",
                                     source_family="damped_ets", last_selection_target_year=2018)])
        self.assertEqual(underlying_family(mapping, "local_champion", "Male", 2014), "arima")
        changed = mapping.copy()
        changed.loc[changed.fit_origin.eq(2018), ["source_family", "last_selection_target_year"]] = [None, 9999]
        self.assertEqual(underlying_family(changed, "local_champion", "Male", 2014), "arima")
        self.assertEqual(underlying_family(pd.DataFrame(), FAMILY, "Male", 2014), FAMILY)
        for altered in [mapping.iloc[1:], pd.concat([mapping, mapping.iloc[:1]], ignore_index=True),
                        mapping.assign(last_selection_target_year=2023)]:
            with self.assertRaises(ValueError):
                underlying_family(altered, "local_champion", "Male", 2014)

    def test_80plus_scope_has_all_four_bands_and_individual_oldest_outputs(self):
        interval_rows, wis_rows = [], []
        for sex in CONFIG["sexes"]:
            for index, age in enumerate(CONFIG["ages"]):
                for scale in ["rate", "log_rate"]:
                    context = dict(target="Saudi Arabia", outcome="prevalence", origin=2018, role=FAMILY,
                                   variant="original", sex=sex, age=age, horizon=5, scale=scale)
                    wis_rows.append(dict(**context, wis_50_80=index+1., wis_50_80_95=index+2.))
                    for level in [.5, .8]:
                        interval_rows.append(dict(**context, level=level, lower=0., upper=1., width=1.,
                            observed_value=2. if index>=7 else .5, covered=index<7, interval_score=index+1.))
        summaries, wis = rate_summaries(pd.DataFrame(interval_rows), pd.DataFrame(wis_rows), CONFIG)
        older = summaries.loc[summaries.age_scope.eq("80+")]
        np.testing.assert_array_equal(older.age_cells, 4)
        np.testing.assert_array_equal(older.coverage, 0)
        np.testing.assert_array_equal(older.upper_miss_rate, 1)
        np.testing.assert_array_equal(older.lower_miss_rate, 0)
        np.testing.assert_array_equal(wis.loc[wis.age_scope.eq("80+"), "mean_wis_50_80"], 9.5)
        np.testing.assert_array_equal(summaries.loc[summaries.age_scope.eq("65-79"), "coverage"], 1)
        np.testing.assert_array_equal(summaries.loc[summaries.age_scope.eq("45+"), "age_cells"], 11)
        for age in ["80-84", "85-89", "90-94", "95+"]:
            rows = summaries.loc[summaries.age_scope.eq("age:"+age)]
            self.assertEqual(len(rows), 8)
            np.testing.assert_array_equal(rows.age_cells, 1)

    def test_role_variant_encoding_and_source_identity(self):
        frame = pd.DataFrame({"family": ["local_champion__gcc_assisted", "tcn_adapted__fixed_1.25"]})
        encoded = labels(frame)
        self.assertEqual(encoded.role.tolist(), ["local_champion", "tcn_adapted"])
        self.assertEqual(encoded.variant.tolist(), ["gcc_assisted", "fixed_1.25"])
        self.assertNotIn("role", frame)
        with self.assertRaises(ValueError):
            labels(pd.DataFrame({"family": [FAMILY]}))
        identity = pd.DataFrame({"target": ["Saudi Arabia"], "outcome": ["prevalence"]})
        case = dict(target="Saudi Arabia", outcome="prevalence")
        require_source(identity, case)
        for altered in [identity.assign(target="Oman"), identity.assign(outcome="incidence")]:
            with self.assertRaises(ValueError):
                require_source(altered, case)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        paths = ["src/gbd_park/calibration.py", "tests/test_calibration.py", "study_design/interval_calibration_v1_2.md",
                 "study_design/interval_calibration_v1_2.json", "src/gbd_park/intervals.py", "src/gbd_park/demography.py",
                 "study_design/locked_v1/design.json", "scripts/run_calibration.py", "scripts/run_demography.py"]
        report = dict(passed=True, tests_run=result.testsRun, elapsed_seconds=time.perf_counter()-started,
                      synthetic_data_only=True, final_evaluation_scores_read=False,
                      tested_code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths})
        directory = ROOT / "work/calibration-validation"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "unit_tests.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
