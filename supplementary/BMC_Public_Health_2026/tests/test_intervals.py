"""Analytic checks for chronological joint residual prediction intervals."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from gbd_park.intervals import (apply_bank, bank_quantiles, build_residual_bank,
                                interval_score, score_intervals, weighted_interval_score)
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nonneural_grid
from gbd_park.prequential import champion_history, select_prequential

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
FAMILY = "damped_ets"


def coordinates():
    return pd.DataFrame([{"sex": sex, "age": age, "horizon": horizon}
                         for sex in CONFIG["sexes"] for age in CONFIG["ages"]
                         for horizon in CONFIG["calendar"]["horizons"]])


def scored_fixture(shocks=None, common_shock=False):
    """Construct exact synthetic log residuals; do not read final outcomes."""
    shocks = np.arange(11) - 5 if shocks is None else np.asarray(shocks, dtype=float)
    rows = []
    for index, shock in enumerate(shocks):
        origin = 2003 + index
        for position, row in coordinates().iterrows():
            bias = .002 * position
            loading = .1 if common_shock else .01 * (position + 1)
            residual = bias + shock * loading
            prediction = 100 + position
            rows.append({"target": CONFIG["primary_target"], "outcome": "prevalence",
                         "family": FAMILY, "setting_id": "synthetic", "origin": origin,
                         "sex": row.sex, "age": row.age, "horizon": row.horizon,
                         "forecast_year": origin + row.horizon, "prediction": prediction,
                         "log_prediction": np.log(prediction),
                         "observed_rate": np.exp(np.log(prediction) + residual),
                         "expected_residual": residual, "lower": 0., "upper": 999999.})
    return pd.DataFrame(rows)


def point_fixture(origin=2012, equal_rates=False):
    frame = coordinates()
    frame["target"] = CONFIG["primary_target"]
    frame["outcome"] = "prevalence"
    frame["family"] = FAMILY
    frame["setting_id"] = "synthetic_current"
    frame["origin"] = origin
    frame["forecast_year"] = origin + frame.horizon
    frame["prediction"] = 100. if equal_rates else 100. + np.arange(len(frame))
    frame["log_prediction"] = np.log(frame.prediction)
    return frame


def candidate_fixture(group, origin):
    grid = local_grid(CONFIG) if group == "local" else nonneural_grid(CONFIG)
    predictions, scores, winners = [], [], {}
    for family in CONFIG["models"][group + "_order"]:
        specs = [spec for spec in grid if spec["family"] == family]
        for sex in CONFIG["sexes"]:
            winner = specs[-1 if sex == "Male" else 0]["setting_id"]
            winners[(sex, family)] = winner
            for index, spec in enumerate(specs):
                for age in CONFIG["ages"]:
                    for horizon in CONFIG["calendar"]["horizons"]:
                        predictions.append({"target": CONFIG["primary_target"], "outcome": "prevalence",
                                            "origin": origin, "sex": sex, "age": age, "horizon": horizon,
                                            "forecast_year": origin + horizon, "family": family,
                                            "setting_id": spec["setting_id"], "prediction": 100 + index,
                                            "log_prediction": np.log(100 + index)})
                    for previous in range(2003, 2014):
                        loss = .01 if spec["setting_id"] == winner else 1 + index * .01
                        if previous > 2004:
                            loss = 100 - loss
                        scores.append({"origin": previous, "sex": sex, "age": age, "horizon": 5,
                                       "family": family, "setting_id": spec["setting_id"],
                                       "absolute_log_error": loss, "parameter_count": 1,
                                       "grid_order": spec["grid_order"]})
    return pd.DataFrame(predictions), pd.DataFrame(scores), winners


def scoring_fixture():
    bank = build_residual_bank(scored_fixture(), CONFIG, 2012, FAMILY)
    point = point_fixture(equal_rates=True)
    intervals, _ = apply_bank(point, bank, CONFIG)
    for level, bounds in [(.5, [8., 9., 10.]), (.8, [6., 9., 12.]), (.95, [4., 9., 14.])]:
        for scale in ["rate", "log_rate"]:
            mask = intervals.level.eq(level) & intervals.scale.eq(scale)
            intervals.loc[mask, ["lower", "median", "upper"]] = bounds if scale == "rate" else np.log(bounds)
    truth = point[["target", "outcome", "sex", "age", "forecast_year"]].copy()
    truth["observed_rate"] = np.where(truth.sex.eq("Male"), 10., 13.)
    return intervals, truth


class IntervalTests(unittest.TestCase):
    def test_residual_sign_centering_and_joint_covariance(self):
        scored = scored_fixture()
        bank = build_residual_bank(scored, CONFIG, 2012, FAMILY)
        expected = scored.loc[scored.origin.le(2007), "expected_residual"].to_numpy().reshape(5, 110)
        np.testing.assert_allclose(bank["raw_residuals"], expected, atol=2e-14)
        np.testing.assert_allclose(bank["center"], expected.mean(axis=0), atol=2e-14)
        np.testing.assert_allclose(bank["centered_residuals"], expected - expected.mean(axis=0), atol=2e-14)
        np.testing.assert_allclose(bank["centered_residuals"].mean(axis=0), 0, atol=2e-14)
        np.testing.assert_allclose(np.cov(bank["centered_residuals"], rowvar=False),
                                   np.cov(expected, rowvar=False), atol=2e-14)
        self.assertEqual(bank["status"], "ok")
        self.assertEqual(bank["n_blocks"], 5)
        self.assertEqual(bank["origins"], list(range(2003, 2008)))
        pd.testing.assert_frame_equal(bank["coords"].reset_index(drop=True), coordinates())

    def test_full_horizon_eligibility_and_future_nan_invariance(self):
        original = scored_fixture()
        bank = build_residual_bank(original, CONFIG, 2014, FAMILY)
        altered = original.copy()
        future = altered.origin.gt(2009)
        altered.loc[future, ["prediction", "log_prediction", "observed_rate"]] = np.nan
        altered = pd.concat([altered, altered.loc[future].iloc[:1]], ignore_index=True)
        other = build_residual_bank(altered, CONFIG, 2014, FAMILY)
        self.assertEqual(bank["origins"], list(range(2003, 2010)))
        self.assertEqual(bank["n_blocks"], 7)
        np.testing.assert_array_equal(bank["raw_residuals"], other["raw_residuals"])
        np.testing.assert_array_equal(bank["centered_residuals"], other["centered_residuals"])
        self.assertNotIn(2010, bank["origins"])  # Its early horizons exist, its fifth is incomplete.
        self.assertEqual(build_residual_bank(original, CONFIG, 2018, FAMILY)["n_blocks"], 11)

    def test_missing_age_sex_horizon_or_origin_is_rejected(self):
        scored = scored_fixture()
        masks = [scored.age.ne(CONFIG["ages"][0]), scored.sex.ne("Female"),
                 scored.horizon.ne(5), scored.origin.ne(2005)]
        for mask in masks:
            with self.subTest(removed=int((~mask).sum())):
                with self.assertRaises(ValueError):
                    build_residual_bank(scored.loc[mask], CONFIG, 2012, FAMILY)
        with self.assertRaises(ValueError):
            build_residual_bank(scored.drop(index=0), CONFIG, 2012, FAMILY)

    def test_duplicate_joint_cell_is_rejected(self):
        scored = scored_fixture()
        duplicated = pd.concat([scored, scored.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            build_residual_bank(duplicated, CONFIG, 2012, FAMILY)

    def test_four_blocks_are_unavailable_without_fabricated_draws(self):
        bank = build_residual_bank(scored_fixture(), CONFIG, 2011, FAMILY)
        self.assertEqual(bank["status"], "insufficient_blocks")
        self.assertEqual(bank["n_blocks"], 4)
        intervals, draws = apply_bank(point_fixture(2011), bank, CONFIG)
        self.assertTrue(draws.empty)
        self.assertTrue(intervals[["lower", "median", "upper"]].isna().all().all())
        self.assertEqual(set(intervals.n_blocks), {4})
        self.assertEqual(set(intervals.status), {"insufficient_blocks"})

    def test_linear_quantiles_are_computed_separately_on_log_and_rate_scales(self):
        bank = build_residual_bank(scored_fixture(shocks=[-2, -1, 0, 1, 2], common_shock=True),
                                   CONFIG, 2012, FAMILY)
        point = point_fixture()
        intervals, draws = apply_bank(point, bank, CONFIG)
        cell = (intervals.sex == "Male") & (intervals.age == CONFIG["ages"][0]) & (intervals.horizon == 1)
        cell_draws = draws.loc[(draws.sex == "Male") & (draws.age == CONFIG["ages"][0]) & (draws.horizon == 1)]
        for _, interval in intervals.loc[cell].iterrows():
            alpha = 1 - interval.level
            values = cell_draws["log_draw" if interval.scale == "log_rate" else "rate_draw"].to_numpy()
            expected = np.quantile(values, [alpha / 2, .5, 1 - alpha / 2], method="linear")
            np.testing.assert_allclose([interval.lower, interval["median"], interval.upper], expected, atol=1e-12)
        rate = intervals.loc[cell & intervals.scale.eq("rate") & intervals.level.eq(.8)].iloc[0]
        log = intervals.loc[cell & intervals.scale.eq("log_rate") & intervals.level.eq(.8)].iloc[0]
        self.assertGreater(abs(rate.lower - np.exp(log.lower)), .01)
        self.assertEqual(len(draws), 5 * 110)
        self.assertEqual(set(draws.residual_origin), set(range(2003, 2008)))

    def test_predictive_median_can_differ_from_point_forecast(self):
        bank = build_residual_bank(scored_fixture(shocks=[-2, -1, 0, 1, 8], common_shock=True),
                                   CONFIG, 2012, FAMILY)
        intervals, _ = apply_bank(point_fixture(equal_rates=True), bank, CONFIG)
        logged = intervals.loc[intervals.scale.eq("log_rate")]
        rates = intervals.loc[intervals.scale.eq("rate")]
        np.testing.assert_allclose(logged["median"], np.log(100) - .12, atol=2e-14)
        np.testing.assert_allclose(rates["median"], 100 * np.exp(-.12), atol=2e-12)
        self.assertTrue((rates["median"] < rates.point_prediction).all())

    def test_joint_draw_keys_preserve_sex_ratios_and_count_totals(self):
        bank = build_residual_bank(scored_fixture(shocks=[-2, -1, 0, 1, 8], common_shock=True),
                                   CONFIG, 2012, FAMILY)
        point = point_fixture(equal_rates=True)
        point.loc[point.sex.eq("Male"), "prediction"] = 200.
        point["log_prediction"] = np.log(point.prediction)
        _, draws = apply_bank(point.sample(frac=1, random_state=11), bank, CONFIG)
        self.assertFalse(draws.duplicated(["residual_origin", "sex", "age", "horizon"]).any())
        ratios = draws.pivot(index=["residual_origin", "age", "horizon"], columns="sex", values="rate_draw")
        np.testing.assert_allclose(ratios.Male / ratios.Female, 2., atol=2e-14)
        # Fixed population of100,000 per stratum makes rate equal conditional count.
        totals = draws.groupby(["residual_origin", "horizon"]).rate_draw.sum()
        multiplier = draws.loc[draws.sex.eq("Female") & draws.age.eq(CONFIG["ages"][0]),
                               ["residual_origin", "horizon", "rate_draw"]].set_index(["residual_origin", "horizon"]).rate_draw / 100
        np.testing.assert_allclose(totals, 11 * (200 + 100) * multiplier.reindex(totals.index), atol=2e-11)
        self.assertEqual(len(totals), 5 * 5)

    def test_source_uncertainty_bounds_do_not_enter_calibration(self):
        original = scored_fixture()
        altered = original.copy()
        for name in ["lower", "upper", "rate_lower", "rate_upper", "count_lower", "count_upper"]:
            altered[name] = np.nan if "lower" in name else np.inf
        first = build_residual_bank(original, CONFIG, 2012, FAMILY)
        second = build_residual_bank(altered, CONFIG, 2012, FAMILY)
        np.testing.assert_array_equal(first["centered_residuals"], second["centered_residuals"])
        first_intervals, first_draws = apply_bank(point_fixture(), first, CONFIG)
        second_intervals, second_draws = apply_bank(point_fixture(), second, CONFIG)
        pd.testing.assert_frame_equal(first_intervals, second_intervals)
        pd.testing.assert_frame_equal(first_draws, second_draws)

    def test_interval_score_hand_examples_and_boundary_behavior(self):
        actual = interval_score(np.array([8, 9, 10, 7, 13]), 8, 10, .5)
        np.testing.assert_allclose(actual, [2, 2, 2, 6, 14])
        np.testing.assert_allclose(interval_score(13, 6, 12, .2), 16)

    def test_weighted_interval_score_uses_standard_weights_and_median(self):
        y = np.array([10., 13.])
        median = np.array([9., 9.])
        lower = np.array([[8., 6.], [8., 6.]])
        upper = np.array([[10., 12.], [10., 12.]])
        score = weighted_interval_score(y, median, lower, upper, [.5, .8])
        np.testing.assert_allclose(score, [.64, 2.84])
        supplemental = weighted_interval_score(13., 9., [8., 6., 4.], [10., 12., 14.], [.5, .8, .95])
        self.assertAlmostEqual(float(supplemental), 2.1)

    def test_bank_offset_quantiles_need_no_current_forecast(self):
        bank = build_residual_bank(scored_fixture(), CONFIG, 2012, FAMILY)
        offsets = bank_quantiles(bank, CONFIG)
        self.assertEqual(len(offsets), 110 * 2 * 3)
        self.assertEqual(set(offsets.scale), {"log_offset", "rate_multiplier"})
        self.assertFalse({"forecast_year", "origin", "point_prediction", "prediction"} & set(offsets.columns))
        self.assertEqual(set(offsets.fit_origin), {2012})
        point = point_fixture(equal_rates=True)
        point["prediction"], point["log_prediction"] = 1., 0.
        intervals, _ = apply_bank(point, bank, CONFIG)
        expected = intervals.copy()
        expected["scale"] = expected.scale.map({"log_rate": "log_offset", "rate": "rate_multiplier"})
        keys = ["sex", "age", "horizon", "scale", "level"]
        comparison = offsets.merge(expected, on=keys, suffixes=("_offset", "_point"), validate="one_to_one")
        for value in ["lower", "median", "upper"]:
            np.testing.assert_allclose(comparison[value + "_offset"], comparison[value + "_point"], atol=1e-12)
        early = bank_quantiles(build_residual_bank(scored_fixture(), CONFIG, 2011, FAMILY), CONFIG)
        self.assertTrue(early[["lower", "median", "upper"]].isna().all().all())
        self.assertEqual(set(early.status), {"insufficient_blocks"})

    def test_scoring_coverage_boundaries_and_hand_wis_on_each_scale(self):
        intervals, truth = scoring_fixture()
        cells, wis = score_intervals(intervals, truth, CONFIG, 2017)
        self.assertEqual(len(cells), 110 * 2 * 3)
        self.assertEqual(len(wis), 110 * 2)
        rate = cells.loc[cells.scale.eq("rate")]
        self.assertTrue(rate.loc[rate.sex.eq("Male"), "covered"].eq(1).all())
        self.assertTrue(rate.loc[rate.sex.eq("Female") & rate.level.lt(.95), "covered"].eq(0).all())
        self.assertTrue(rate.loc[rate.sex.eq("Female") & rate.level.eq(.95), "covered"].eq(1).all())
        self.assertEqual(set(rate.loc[rate.level.eq(.5), "width"]), {2.})
        self.assertEqual(set(rate.loc[rate.level.eq(.8), "width"]), {6.})
        rate_wis = wis.loc[wis.scale.eq("rate")]
        np.testing.assert_allclose(rate_wis.loc[rate_wis.sex.eq("Male"), "wis_50_80"], .64)
        np.testing.assert_allclose(rate_wis.loc[rate_wis.sex.eq("Female"), "wis_50_80"], 2.84)
        np.testing.assert_allclose(rate_wis.loc[rate_wis.sex.eq("Female"), "wis_50_80_95"], 2.1)
        for sex, observed in [("Male", 10.), ("Female", 13.)]:
            y, median = np.log(observed), np.log(9.)
            contribution = .5 * abs(y - median)
            for level, lower, upper in [(.5, 8., 10.), (.8, 6., 12.)]:
                alpha, lower, upper = 1 - level, np.log(lower), np.log(upper)
                score = upper - lower
                if y < lower:
                    score += 2 * (lower - y) / alpha
                if y > upper:
                    score += 2 * (y - upper) / alpha
                contribution += alpha * score / 2
            mask = wis.sex.eq(sex) & wis.scale.eq("log_rate")
            np.testing.assert_allclose(wis.loc[mask, "wis_50_80"], contribution / 2.5, atol=1e-14)
        # Both endpoints count as covered, including the lower boundary.
        truth["observed_rate"] = 8.
        boundary, _ = score_intervals(intervals, truth, CONFIG, 2017)
        self.assertTrue(boundary.loc[boundary.scale.eq("rate") & boundary.level.eq(.5), "covered"].eq(1).all())

    def test_interval_scoring_rejects_missing_duplicate_and_future_grids(self):
        intervals, truth = scoring_fixture()
        with self.assertRaisesRegex(ValueError, "complete joint"):
            score_intervals(intervals.drop(index=0), truth, CONFIG, 2017)
        with self.assertRaisesRegex(ValueError, "Duplicate interval"):
            score_intervals(pd.concat([intervals, intervals.iloc[:1]], ignore_index=True), truth, CONFIG, 2017)
        with self.assertRaisesRegex(ValueError, "verification cutoff"):
            score_intervals(intervals, truth, CONFIG, 2016)
        inconsistent = intervals.copy()
        index = inconsistent.index[inconsistent.scale.eq("rate")][0]
        inconsistent.loc[index, "median"] += .01
        with self.assertRaisesRegex(ValueError, "medians disagree"):
            score_intervals(inconsistent, truth, CONFIG, 2017)
        future = truth.copy()
        future["forecast_year"] += 10
        future["observed_rate"] = np.nan
        extended = pd.concat([truth, future, future], ignore_index=True)
        expected_cells, expected_wis = score_intervals(intervals, truth, CONFIG, 2017)
        actual_cells, actual_wis = score_intervals(intervals, extended, CONFIG, 2017)
        pd.testing.assert_frame_equal(actual_cells, expected_cells)
        pd.testing.assert_frame_equal(actual_wis, expected_wis)

    def test_unavailable_intervals_retain_nan_scores_and_all_cells(self):
        bank = build_residual_bank(scored_fixture(), CONFIG, 2011, FAMILY)
        point = point_fixture(2011)
        intervals, _ = apply_bank(point, bank, CONFIG)
        truth = point[["target", "outcome", "sex", "age", "forecast_year"]].copy()
        truth["observed_rate"] = 10.
        cells, wis = score_intervals(intervals, truth, CONFIG, 2016)
        self.assertEqual(len(cells), 110 * 2 * 3)
        self.assertEqual(len(wis), 110 * 2)
        self.assertTrue(cells[["covered", "width", "interval_score"]].isna().all().all())
        self.assertTrue(wis[["wis_50_80", "wis_50_80_95", "median_absolute_error"]].isna().all().all())
        self.assertEqual(set(wis.n_blocks), {4})

    def test_prequential_cold_defaults_before_completed_inner_blocks(self):
        expected = {"persistence": "persistence", "damped_ets": "damped_ets", "arima": "arima",
                    "log_trend": "log_trend__window=8",
                    "age_smooth_trend": "age_smooth_trend__window=all__penalty=1",
                    "pooled_ridge": "pooled_ridge__alpha=1",
                    "pooled_boosting": "pooled_boosting__depth=1__trees=100__min_leaf=5",
                    "donor_ridge_unadapted": "donor_ridge_unadapted__alpha=1",
                    "donor_ridge_adapted": "donor_ridge_adapted__alpha=1__adaptation_penalty=1",
                    "donor_boosting_unadapted": "donor_boosting_unadapted__depth=1__trees=100__min_leaf=5",
                    "donor_boosting_adapted": "donor_boosting_adapted__depth=1__trees=100__min_leaf=5__adaptation_penalty=1"}
        for group in ["local", "nonneural"]:
            for origin in [2003, 2007]:
                with self.subTest(group=group, origin=origin):
                    predictions, scores, _ = candidate_fixture(group, origin)
                    scores["absolute_log_error"] = np.nan
                    selected, decisions = select_prequential(predictions, scores, CONFIG, origin, group)
                    self.assertEqual(len(selected), len(CONFIG["models"][group + "_order"]) * 110)
                    for decision in decisions:
                        self.assertEqual(decision["setting_id"], expected[decision["family"]])
                        self.assertEqual(decision["selection_status"], "cold_start_defaults")
                        self.assertEqual(decision["inner_origins"], "")
                        self.assertIsNone(decision["last_inner_label_year"])
                        self.assertIsNone(decision["inner_loss"])

    def test_prequential_selection_uses_completed_blocks_and_ignores_future_nan(self):
        for group in ["local", "nonneural"]:
            with self.subTest(group=group):
                predictions, scores, expected = candidate_fixture(group, 2009)
                chosen, decisions = select_prequential(predictions, scores, CONFIG, 2009, group)
                altered = scores.copy()
                altered.loc[altered.origin.gt(2004), "absolute_log_error"] = np.nan
                other, other_decisions = select_prequential(predictions, altered, CONFIG, 2009, group)
                pd.testing.assert_frame_equal(chosen, other)
                self.assertEqual(decisions, other_decisions)
                for decision in decisions:
                    self.assertEqual(decision["setting_id"], expected[(decision["sex"], decision["family"])])
                    self.assertEqual(decision["inner_origins"], "2003|2004")
                    self.assertEqual(decision["last_inner_label_year"], 2009)
                    self.assertEqual(decision["selection_status"], "tuned_completed_blocks")
                # 2008 is the first origin with a completed five-year inner block.
                earlier, earlier_scores, _ = candidate_fixture(group, 2008)
                _, first_decisions = select_prequential(earlier, earlier_scores, CONFIG, 2008, group)
                self.assertTrue(all(d["inner_origins"] == "2003" for d in first_decisions))

    def test_prequential_rejects_missing_forecast_and_entire_candidate_setting(self):
        for group in ["local", "nonneural"]:
            with self.subTest(group=group):
                predictions, scores, expected = candidate_fixture(group, 2009)
                family = "log_trend" if group == "local" else "pooled_ridge"
                chosen = predictions.sex.eq("Male") & predictions.family.eq(family)
                chosen &= predictions.setting_id.eq(expected[("Male", family)])
                missing = predictions.drop(index=predictions.index[chosen][0])
                with self.assertRaisesRegex(ValueError, "Incomplete selected historical"):
                    select_prequential(missing, scores, CONFIG, 2009, group)
                grid = local_grid(CONFIG) if group == "local" else nonneural_grid(CONFIG)
                ident = next(spec["setting_id"] for spec in grid if spec["family"] == family)
                missing_setting = scores.loc[~(scores.family.eq(family) & scores.setting_id.eq(ident))]
                with self.assertRaisesRegex(ValueError, "Incomplete historical candidate"):
                    select_prequential(predictions, missing_setting, CONFIG, 2009, group)
                predictions["observed_rate"] = 100.
                with self.assertRaisesRegex(ValueError, "separate from verification"):
                    select_prequential(predictions, scores, CONFIG, 2009, group)

    def test_champion_pairs_source_families_and_preserves_historical_settings(self):
        sources = []
        for family in ["damped_ets", "arima", "log_trend"]:
            frame = scored_fixture()
            frame["family"] = family
            frame["setting_id"] = [f"{family}_historical_{origin}" for origin in frame.origin]
            sources.append(frame)
        all_sources = pd.concat(sources, ignore_index=True)
        mapping = {"Male": "damped_ets", "Female": "arima"}
        champion = champion_history(all_sources, CONFIG, mapping, "local_champion")
        self.assertEqual(len(champion), 11 * 110)
        self.assertEqual(set(champion.family), {"local_champion"})
        for sex, family in mapping.items():
            rows = champion.loc[champion.sex.eq(sex)]
            self.assertEqual(set(rows.source_family), {family})
            self.assertEqual(list(rows.setting_id), [f"{family}_historical_{origin}" for origin in rows.origin])
        bank = build_residual_bank(champion, CONFIG, 2014, "local_champion")
        self.assertEqual(bank["origins"], list(range(2003, 2010)))
        for block_index, origin in enumerate(bank["origins"]):
            selected = champion.loc[champion.origin.eq(origin)].set_index(["sex", "age", "horizon"])
            expected = selected.reindex(pd.MultiIndex.from_frame(coordinates())).expected_residual.to_numpy()
            np.testing.assert_allclose(bank["raw_residuals"][block_index], expected, atol=2e-14)
        with self.assertRaises(ValueError):
            champion_history(all_sources, CONFIG, {"Male": "damped_ets"}, "local_champion")
        with self.assertRaises(ValueError):
            champion_history(all_sources, CONFIG, {"Male": "missing", "Female": "arima"}, "local_champion")
        with self.assertRaises(ValueError):
            build_residual_bank(champion.drop(index=0), CONFIG, 2014, "local_champion")


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(IntervalTests))
    candidates = ["src/gbd_park/intervals.py", "src/gbd_park/prequential.py", "tests/test_intervals.py",
                  "scripts/run_intervals.py", "study_design/interval_implementation.md",
                  "study_design/intervals_implementation.md", "src/gbd_park/scoring.py",
                  "src/gbd_park/local.py", "src/gbd_park/pooled.py", "src/gbd_park/adaptation.py",
                  "src/gbd_park/tcn.py", "src/gbd_park/tcn_forecasting.py"]
    files = [ROOT / name for name in candidates if (ROOT / name).exists()]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "device": "cpu",
              "tested_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in files}}
    output = ROOT / "work/interval-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
