"""Synthetic checks for the primary endpoint and locked evaluation workflow."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from gbd_park.evaluation import (champion_forecasts, matched_tcn_harm,
                                 origin_population_weights, primary_contrasts,
                                 summarize_scores)
from gbd_park.intervals import apply_bank, build_residual_bank
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import settings_grid as nonneural_grid
from gbd_park.prequential import champion_history
from run_primary import (evaluation_origins, select_baseline_settings,
                         validate_forecast_ledger, verify_committed_outputs,
                         verify_scoring_order)

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def scored_fixture():
    coefficients = {"tcn_adapted": .01, "tcn_unadapted": .03, "tcn_intercept": .04,
                    "local_champion": .02, "nonneural_champion": .03}
    rows = []
    for origin in CONFIG["calendar"]["reliability_origins"]:
        for family, coefficient in coefficients.items():
            for sex in CONFIG["sexes"]:
                for age_index, age in enumerate(CONFIG["ages"]):
                    for horizon in CONFIG["calendar"]["horizons"]:
                        error = coefficient * (age_index + 1) * horizon / 5
                        error *= 1 if sex == "Male" else 1.2
                        error += .001 * (origin - 2014)
                        sign = 1 if age_index % 2 else -1
                        prediction = 100 * np.exp(sign * error)
                        rows.append({"target": CONFIG["primary_target"], "outcome": "prevalence",
                                     "origin": origin, "sex": sex, "age": age, "horizon": horizon,
                                     "forecast_year": origin + horizon, "family": family,
                                     "setting_id": family + "__synthetic", "prediction": prediction,
                                     "log_prediction": np.log(prediction), "observed_rate": 100.,
                                     "absolute_log_error": error, "absolute_rate_error": abs(prediction - 100),
                                     "status": "ok", "fallback_reason": "", "parameter_count": 1,
                                     "source_family": ("damped_ets" if family == "local_champion" else
                                                       "pooled_ridge" if family == "nonneural_champion" else family),
                                     "ensemble_fingerprint": f"origin{origin}__shared_neural_base",
                                     "seed_or_ensemble": "11|23|37|53|71"})
    return pd.DataFrame(rows)


def population_fixture():
    rows = []
    for country in [CONFIG["primary_target"], "Jordan"]:
        for year in range(2010, 2024):
            for sex in CONFIG["sexes"]:
                for age_index, age in enumerate(CONFIG["ages"]):
                    population = 1000 * (age_index + 1) * (1 if sex == "Male" else 2) * (1 + .01 * (year - 2010))
                    rate = 20 + age_index
                    rows.append({"location_name": country, "outcome": "prevalence", "year": year,
                                 "sex": sex, "age": age, "rate": rate,
                                 "count": rate * population / 100000,
                                 "implied_population": population})
    return pd.DataFrame(rows)


def update_loss(frame, mask, value):
    frame.loc[mask, "absolute_log_error"] = value
    frame.loc[mask, "prediction"] = 100 * np.exp(frame.loc[mask, "absolute_log_error"])
    frame.loc[mask, "log_prediction"] = np.log(frame.loc[mask, "prediction"])
    frame.loc[mask, "absolute_rate_error"] = np.abs(frame.loc[mask, "prediction"] - 100)


def baseline_scores(group):
    grid = local_grid(CONFIG) if group == "local" else nonneural_grid(CONFIG)
    rows, expected = [], {}
    for family in CONFIG["models"][group + "_order"]:
        specs = [spec for spec in grid if spec["family"] == family]
        for sex in CONFIG["sexes"]:
            winner = specs[-1 if sex == "Male" else 0]["setting_id"]
            expected[(sex, family)] = winner
            for index, spec in enumerate(specs):
                for origin in range(2003, 2014):
                    loss = .01 if spec["setting_id"] == winner else 1 + index * .01
                    if origin > 2009:
                        loss = 100 - loss
                    for age in CONFIG["ages"]:
                        rows.append({"family": family, "sex": sex, "origin": origin, "horizon": 5,
                                     "age": age, "setting_id": spec["setting_id"],
                                     "absolute_log_error": loss, "parameter_count": 1,
                                     "grid_order": spec["grid_order"]})
    return pd.DataFrame(rows), expected


def champion_fixture():
    template = scored_fixture().query("family == 'tcn_adapted'").drop(
        columns=["observed_rate", "absolute_log_error", "absolute_rate_error", "source_family"])
    families = CONFIG["models"]["local_order"] + CONFIG["models"]["nonneural_order"]
    frames, mappings = [], []
    for index, family in enumerate(families):
        frame = template.copy()
        frame["family"] = family
        frame["setting_id"] = [f"{family}_origin{origin}" for origin in frame.origin]
        frame["log_prediction"] += index * .001
        frame["prediction"] = np.exp(frame.log_prediction)
        frames.append(frame)
    for origin in CONFIG["calendar"]["reliability_origins"]:
        for role, by_sex in [("local_champion", {"Male": "damped_ets", "Female": "arima"}),
                             ("nonneural_champion", {"Male": "pooled_boosting", "Female": "donor_ridge_adapted"})]:
            for sex, family in by_sex.items():
                mappings.append({"fit_origin": origin, "role": role, "sex": sex,
                                 "source_family": family, "last_selection_target_year": origin})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(mappings)


class PrimaryTests(unittest.TestCase):
    def test_primary_requires_strict_improvement_for_both_sexes_and_comparators(self):
        scored = scored_fixture()
        contrasts, age_contrasts, success = primary_contrasts(scored, CONFIG)
        self.assertEqual(len(contrasts), 4)
        self.assertEqual(len(age_contrasts), 4 * 11)
        self.assertTrue(success["joint_success"])
        self.assertEqual(success["success_by_sex"], {"Male": True, "Female": True})
        self.assertTrue(contrasts.strictly_lower.all())
        for _, row in contrasts.iterrows():
            self.assertAlmostEqual(row.absolute_loss_difference, row.tcn_error - row.comparator_error)
            self.assertAlmostEqual(row.relative_improvement, (row.comparator_error - row.tcn_error) / row.comparator_error)
            self.assertAlmostEqual(row.relative_improvement_percent, 100 * row.relative_improvement)
        adapted = scored.loc[scored.origin.eq(2018) & scored.horizon.eq(5)
                            & scored.sex.eq("Male") & scored.family.eq("tcn_adapted"), "absolute_log_error"].to_numpy()
        tied = scored.origin.eq(2018) & scored.horizon.eq(5) & scored.sex.eq("Male") & scored.family.eq("local_champion")
        update_loss(scored, tied, adapted)
        _, _, verdict = primary_contrasts(scored, CONFIG)
        self.assertFalse(verdict["joint_success"])
        self.assertFalse(verdict["success_by_sex"]["Male"])
        self.assertTrue(verdict["success_by_sex"]["Female"])

    def test_zero_comparator_loss_has_no_relative_improvement(self):
        scored = scored_fixture()
        zero = scored.origin.eq(2018) & scored.horizon.eq(5) & scored.family.eq("nonneural_champion")
        update_loss(scored, zero, 0.)
        contrasts, _, success = primary_contrasts(scored, CONFIG)
        rows = contrasts.loc[contrasts.comparator_error.eq(0)]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows.relative_improvement.isna().all())
        self.assertTrue(rows.relative_improvement_percent.isna().all())
        np.testing.assert_allclose(rows.absolute_loss_difference, rows.tcn_error)
        self.assertFalse(success["joint_success"])

    def test_population_weights_use_exact_origin_target_population_only(self):
        panel = population_fixture()
        expected = origin_population_weights(panel, CONFIG, [2014])
        changed = panel.copy()
        changed.loc[changed.year.gt(2014) | changed.location_name.ne(CONFIG["primary_target"]),
                    ["count", "rate", "implied_population"]] = np.nan
        actual = origin_population_weights(changed, CONFIG, [2014])
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(len(expected), 22)
        self.assertEqual(set(expected.origin), {2014})
        for sex in CONFIG["sexes"]:
            rows = expected.loc[expected.sex.eq(sex)].set_index("age").reindex(CONFIG["ages"])
            np.testing.assert_allclose(rows.origin_population_weight, np.arange(1, 12) / 66)
            self.assertAlmostEqual(rows.origin_population_weight.sum(), 1.)
            np.testing.assert_allclose(rows.origin_population,
                                       1000 * np.arange(1, 12) * (1 if sex == "Male" else 2) * 1.04)
        missing = panel.drop(panel.index[panel.year.eq(2014) & panel.location_name.eq(CONFIG["primary_target"])][0])
        with self.assertRaises(ValueError):
            origin_population_weights(missing, CONFIG, [2014])

    def test_equal_age_primary_and_origin_weighted_secondary_remain_distinct(self):
        scored = scored_fixture()
        weights = origin_population_weights(population_fixture(), CONFIG, CONFIG["calendar"]["reliability_origins"])
        tables = summarize_scores(scored, weights, CONFIG)
        by_origin = tables["by_origin"]
        row = by_origin.loc[by_origin.origin.eq(2018) & by_origin.sex.eq("Male")
                            & by_origin.family.eq("tcn_adapted") & by_origin.horizon.eq(5)].iloc[0]
        errors = .01 * np.arange(1, 12) + .004
        self.assertAlmostEqual(row.mean_age_absolute_log_error, errors.mean())
        self.assertAlmostEqual(row.population_weighted_absolute_log_error,
                               np.average(errors, weights=np.arange(1, 12)))
        self.assertNotEqual(row.mean_age_absolute_log_error, row.population_weighted_absolute_log_error)
        reliability = tables["reliability_by_horizon"]
        result = reliability.loc[reliability.sex.eq("Male") & reliability.family.eq("tcn_adapted")
                                 & reliability.horizon.eq(5)].iloc[0]
        self.assertEqual(result.n_origins, 5)
        self.assertAlmostEqual(result.mean_absolute_log_error, .06 + .002)
        self.assertEqual(set(tables["primary_age_scores"].origin), {2018})
        self.assertEqual(set(tables["primary_age_scores"].horizon), {5})

    def test_primary_rejects_duplicate_and_missing_age_evidence(self):
        scored = scored_fixture()
        index = scored.index[scored.origin.eq(2018) & scored.horizon.eq(5)
                             & scored.family.eq("tcn_adapted")][0]
        with self.assertRaises(ValueError):
            primary_contrasts(scored.drop(index=index), CONFIG)
        with self.assertRaises(ValueError):
            primary_contrasts(pd.concat([scored, scored.loc[[index]]], ignore_index=True), CONFIG)

    def test_champion_roles_preserve_their_frozen_source_forecasts(self):
        forecasts, mappings = champion_fixture()
        original = forecasts.copy(deep=True)
        champions = champion_forecasts(forecasts, mappings, CONFIG)
        pd.testing.assert_frame_equal(forecasts, original)
        self.assertEqual(len(champions), 5 * 2 * 110)
        self.assertEqual(set(champions.family), {"local_champion", "nonneural_champion"})
        keys = ["age", "horizon"]
        for mapping in mappings.itertuples():
            actual = champions.loc[champions.origin.eq(mapping.fit_origin) & champions.sex.eq(mapping.sex)
                                   & champions.family.eq(mapping.role)].set_index(keys).sort_index()
            expected = forecasts.loc[forecasts.origin.eq(mapping.fit_origin) & forecasts.sex.eq(mapping.sex)
                                     & forecasts.family.eq(mapping.source_family)].set_index(keys).sort_index()
            np.testing.assert_array_equal(actual.prediction, expected.prediction)
            np.testing.assert_array_equal(actual.log_prediction, expected.log_prediction)
            self.assertEqual(actual.setting_id.tolist(), expected.setting_id.tolist())
            self.assertEqual(set(actual.source_family), {mapping.source_family})
        future = mappings.copy()
        future.loc[0, "last_selection_target_year"] += 1
        with self.assertRaises(ValueError):
            champion_forecasts(forecasts, future, CONFIG)
        with self.assertRaises(ValueError):
            champion_forecasts(forecasts, mappings.drop(index=0), CONFIG)
        with self.assertRaises(ValueError):
            champion_forecasts(forecasts, pd.concat([mappings, mappings.iloc[:1]], ignore_index=True), CONFIG)
        wrong_group = mappings.copy()
        wrong_group.loc[0, "source_family"] = "pooled_ridge"
        with self.assertRaises(ValueError):
            champion_forecasts(forecasts, wrong_group, CONFIG)

    def test_champion_intervals_equal_corresponding_source_family_coordinates(self):
        forecasts, mappings = champion_fixture()
        forecasts = forecasts.loc[forecasts.origin.eq(2014)]
        champions = champion_forecasts(forecasts, mappings, CONFIG)
        selected = mappings.loc[mappings.fit_origin.eq(2014)]
        historical = []
        for family_index, family in enumerate(selected.source_family.unique()):
            template = forecasts.loc[forecasts.family.eq(family)].copy()
            for origin in range(2003, 2010):
                block = template.copy()
                block["origin"] = origin
                block["forecast_year"] = origin + block.horizon
                residual = .01 * (family_index + 1) * (origin - 2006) * block.horizon
                block["observed_rate"] = np.exp(block.log_prediction + residual)
                historical.append(block)
        history = pd.concat(historical, ignore_index=True)
        for role in ["local_champion", "nonneural_champion"]:
            mapping = selected.loc[selected.role.eq(role)].set_index("sex").source_family.to_dict()
            paired_history = champion_history(history, CONFIG, mapping, role)
            role_bank = build_residual_bank(paired_history, CONFIG, 2014, role)
            actual, _ = apply_bank(champions.loc[champions.family.eq(role)], role_bank, CONFIG)
            for sex, family in mapping.items():
                source_bank = build_residual_bank(history, CONFIG, 2014, family)
                expected, _ = apply_bank(forecasts.loc[forecasts.family.eq(family)], source_bank, CONFIG)
                keys = ["age", "horizon", "scale", "level"]
                left = actual.loc[actual.sex.eq(sex)].set_index(keys).sort_index()
                right = expected.loc[expected.sex.eq(sex)].set_index(keys).sort_index()
                for name in ["lower", "median", "upper", "point_prediction"]:
                    np.testing.assert_array_equal(left[name], right[name])

    def test_adaptation_harm_uses_identical_ensemble_and_preserves_ties(self):
        scored = scored_fixture()
        pairs, summary = matched_tcn_harm(scored, CONFIG)
        self.assertEqual(len(pairs), 5 * 110)
        self.assertTrue(pairs.adaptation_loss_change.lt(0).all())
        self.assertFalse(pairs.harmed.any())
        self.assertTrue(summary.harm_fraction.eq(0).all())
        mask = scored.origin.eq(2018) & scored.family.eq("tcn_adapted") & scored.sex.eq("Male")
        mask &= scored.age.eq(CONFIG["ages"][0]) & scored.horizon.eq(5)
        counterpart = scored.origin.eq(2018) & scored.family.eq("tcn_unadapted") & scored.sex.eq("Male")
        counterpart &= scored.age.eq(CONFIG["ages"][0]) & scored.horizon.eq(5)
        update_loss(scored, mask, scored.loc[counterpart, "absolute_log_error"].iloc[0])
        tied, _ = matched_tcn_harm(scored, CONFIG)
        zero = tied.loc[tied.origin.eq(2018) & tied.sex.eq("Male") & tied.age.eq(CONFIG["ages"][0]) & tied.horizon.eq(5)]
        self.assertEqual(zero.adaptation_loss_change.iloc[0], 0.)
        self.assertFalse(zero.harmed.iloc[0])
        scored.loc[mask, "ensemble_fingerprint"] = "a_different_source_model"
        with self.assertRaisesRegex(ValueError, "different source ensembles"):
            matched_tcn_harm(scored, CONFIG)

    def test_evaluation_origins_include_primary_once(self):
        origins = evaluation_origins(CONFIG)
        self.assertEqual(origins, [2014, 2015, 2016, 2017, 2018])
        self.assertEqual(origins.count(CONFIG["calendar"]["primary_origin"]), 1)
        repeated = deepcopy(CONFIG)
        repeated["calendar"]["reliability_origins"].append(2018)
        with self.assertRaises((ValueError, AssertionError)):
            evaluation_origins(repeated)

    def test_baseline_settings_ignore_uncompleted_inner_outcomes(self):
        for group in ["local", "nonneural"]:
            with self.subTest(group=group):
                scores, expected = baseline_scores(group)
                selected = select_baseline_settings(scores, CONFIG, [2014], group)
                changed = scores.copy()
                changed.loc[changed.origin.gt(2009), "absolute_log_error"] = np.nan
                other = select_baseline_settings(changed, CONFIG, [2014], group)
                pd.testing.assert_frame_equal(selected, other)
                self.assertEqual(len(selected), 2 * len(CONFIG["models"][group + "_order"]))
                for _, row in selected.iterrows():
                    self.assertEqual(row.setting_id, expected[(row.sex, row.family)])
                    self.assertEqual(row.last_inner_label_year, 2014)
                    self.assertEqual(row.inner_origins, "2003|2004|2005|2006|2007|2008|2009")

    def test_baseline_selection_rejects_incomplete_or_nonfinite_completed_evidence(self):
        for group in ["local", "nonneural"]:
            scores, _ = baseline_scores(group)
            with self.subTest(group=group, issue="entire_setting_missing"):
                setting = scores.setting_id.iloc[0]
                with self.assertRaises(ValueError):
                    select_baseline_settings(scores.loc[scores.setting_id.ne(setting)], CONFIG, [2014], group)
            with self.subTest(group=group, issue="duplicate_age_cell"):
                duplicate = pd.concat([scores, scores.iloc[:1]], ignore_index=True)
                with self.assertRaises(ValueError):
                    select_baseline_settings(duplicate, CONFIG, [2014], group)
            with self.subTest(group=group, issue="nonfinite_completed_error"):
                invalid = scores.copy()
                invalid.loc[0, "absolute_log_error"] = np.nan
                with self.assertRaises(ValueError):
                    select_baseline_settings(invalid, CONFIG, [2014], group)

    def test_forecast_ledger_requires_unique_complete_unverified_cells(self):
        frame = scored_fixture().drop(columns=["observed_rate", "absolute_log_error", "absolute_rate_error"])
        frame["last_inner_label_year"] = frame.origin
        families = frame.family.unique().tolist()
        validate_forecast_ledger(frame, CONFIG, families)
        self.assertEqual(len(frame.loc[frame.origin.eq(2018)]), len(families) * 110)
        for changed in [frame.drop(index=0), pd.concat([frame, frame.iloc[:1]], ignore_index=True)]:
            with self.assertRaises(ValueError):
                validate_forecast_ledger(changed, CONFIG, families)
        future = frame.copy()
        future.loc[0, "last_inner_label_year"] += 1
        with self.assertRaisesRegex(ValueError, "Future labels"):
            validate_forecast_ledger(future, CONFIG, families)
        frame["observed_rate"] = 100.
        with self.assertRaisesRegex(ValueError, "Verification values"):
            validate_forecast_ledger(frame, CONFIG, families)

    def test_committed_outputs_must_exist_and_remain_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="gbd_primary_test_") as directory:
            out = Path(directory)
            names = ["predictions.csv", "intervals.csv", "joint_draws.csv"]
            hashes = {}
            for name in names:
                path = out / name
                path.write_text("synthetic,ledger\n1,2\n")
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_committed_outputs(out, hashes)
            with self.assertRaises((ValueError, AssertionError)):
                verify_committed_outputs(out, {name: value for name, value in hashes.items() if name != "intervals.csv"})
            (out / "intervals.csv").write_text("changed\n")
            with self.assertRaises((ValueError, AssertionError)):
                verify_committed_outputs(out, hashes)

    def test_forecast_and_interval_commit_events_precede_scoring(self):
        names = ["settings_frozen", "forecast_fitting_started", "predictions_committed",
                 "intervals_committed", "evaluation_scoring_started"]
        events = [{"event": name, "time_utc": f"2026-01-01T00:00:0{index}Z"}
                  for index, name in enumerate(names)]
        verify_scoring_order(events)
        with self.assertRaises((ValueError, AssertionError)):
            verify_scoring_order([events[0], events[1], events[2], events[4], events[3]])
        with self.assertRaises((ValueError, AssertionError)):
            verify_scoring_order([event for event in events if event["event"] != "predictions_committed"])
        with self.assertRaises((ValueError, AssertionError)):
            verify_scoring_order(events + [events[-1]])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PrimaryTests))
    candidates = ["tests/test_primary.py", "src/gbd_park/evaluation.py", "scripts/run_primary.py",
                  "study_design/primary_implementation.md", "study_design/evaluation_implementation.md",
                  "study_design/primary_evaluation_implementation.md",
                  "src/gbd_park/intervals.py", "src/gbd_park/prequential.py", "src/gbd_park/scoring.py",
                  "src/gbd_park/local.py", "src/gbd_park/pooled.py", "src/gbd_park/adaptation.py",
                  "src/gbd_park/tcn.py", "src/gbd_park/tcn_forecasting.py"]
    files = [ROOT / name for name in candidates if (ROOT / name).exists()]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "device": "cpu",
              "fixture_type": "synthetic_no_final_outcome_scores_read",
              "tested_code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in files}}
    output = ROOT / "work/primary-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
