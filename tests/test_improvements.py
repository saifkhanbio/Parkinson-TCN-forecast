"""Synthetic checks for the bounded post-result exploratory procedures."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from gbd_park.improvements import (age_setting, fit_group_offsets, candidate_forecasts,
                                    select_age, apply_retention, retention_family, select_retention)
from gbd_park.intervals import build_residual_bank, apply_bank

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
AMENDMENT = json.loads((ROOT / "study_design/exploratory_v1_1.json").read_text())


def synthetic_original():
    origin = 2003
    current_meta = pd.DataFrame([{"sex": sex, "age": age} for sex in CONFIG["sexes"] for age in CONFIG["ages"]])
    target_meta = pd.DataFrame([{"sex": sex, "age": age, "label_end": origin-w}
                                for sex in CONFIG["sexes"] for age in CONFIG["ages"] for w in [0, 1]])
    levels = np.arange(len(current_meta)) * .05 + 3
    seeds = CONFIG["models"]["tcn"]["ensemble_seeds"]
    cal = [{"origin": origin, "sex": sex, "family": "tcn_adapted", "b0": .01*(i+1),
            "b1": -.004*(i+1), "status": "ok", "last_target_label_year": origin}
           for i, sex in enumerate(CONFIG["sexes"])]
    historical = [np.full((len(target_meta), 5), seed*.0001) for seed in seeds]
    current = [np.full((len(current_meta), 5), seed*.0002) for seed in seeds]
    target_y = np.mean(historical, axis=0) + .03
    for correction in cal:
        mask = target_meta.sex.eq(correction["sex"]).to_numpy()
        target_y[mask] += correction["b0"] + correction["b1"] * np.arange(1, 6) / 5
    payloads = [{"origin": origin, "seed": seed, "base": {"channels": 16}, "target_meta": target_meta.copy(),
                 "current_meta": current_meta.copy(), "target_y": target_y.copy(), "levels": levels.copy(),
                 "target_changes": hist, "current_changes": cur,
                 "audit": {"status": "ok", "fingerprint_before": str(seed), "fingerprint_after": str(seed)}}
                for seed, hist, cur in zip(seeds, historical, current)]
    rows = []
    current_mean = np.mean(current, axis=0)
    for i, row in current_meta.iterrows():
        correction = next(c for c in cal if c["sex"] == row.sex)
        changes = current_mean[i] + correction["b0"] + correction["b1"] * np.arange(1, 6) / 5
        for h in range(1, 6):
            log_prediction = levels[i] + changes[h-1]
            rows.append({"target": CONFIG["primary_target"], "outcome": "prevalence", "origin": origin,
                         "sex": row.sex, "age": row.age, "horizon": h, "forecast_year": origin+h,
                         "family": "tcn_adapted", "setting_id": "original", "parameter_count": 1784,
                         "log_prediction": log_prediction, "prediction": np.exp(log_prediction),
                         "status": "ok", "fallback_reason": ""})
    return payloads, cal, pd.DataFrame(rows)


def age_evidence():
    return pd.DataFrame([{"origin": origin, "sex": sex, "age": age, "horizon": 5,
                          "setting_id": age_setting(penalty), "absolute_log_error": [.2, .1, .08, .09][index]}
                         for origin in range(2003, 2014) for sex in CONFIG["sexes"] for age in CONFIG["ages"]
                         for index, penalty in enumerate(AMENDMENT["age_candidates"])])


def interval_inputs():
    rows = []
    for origin in range(2003, 2008):
        for s, sex in enumerate(CONFIG["sexes"]):
            for a, age in enumerate(CONFIG["ages"]):
                for h in range(1, 6):
                    error = (origin-2003)**2*.01 + s*.02 + a*.001 + h*.002
                    rows.append({"target": CONFIG["primary_target"], "outcome": "prevalence", "origin": origin,
                                 "sex": sex, "age": age, "horizon": h, "forecast_year": origin+h,
                                 "family": "tcn_v1_unchanged", "log_prediction": 3., "prediction": np.exp(3.),
                                 "observed_rate": np.exp(3.+error)})
    scored = pd.DataFrame(rows)
    points = scored[scored.origin.eq(2003)].drop(columns="observed_rate").copy()
    points["origin"], points["forecast_year"] = 2012, 2012+points.horizon
    return points, build_residual_bank(scored, CONFIG, 2012, "tcn_v1_unchanged")


class ImprovementsTests(unittest.TestCase):
    def test_analytic_group_weighting_and_shrinkage_bound(self):
        meta = pd.DataFrame({"age": CONFIG["ages"]*2})
        offsets, objective = fit_group_offsets(np.ones((22, 5)), meta, 10, CONFIG)
        for group, row in offsets.items():
            fraction = len(CONFIG["age_groups"][group]) / 11
            self.assertAlmostEqual(row["offset"], fraction / 20)
            self.assertAlmostEqual(row["absolute_offset_bound"], fraction / 20)
        self.assertLess(objective, 1)

    def test_larger_penalty_shrinks_offsets_and_zero_is_exact(self):
        meta = pd.DataFrame({"age": CONFIG["ages"]})
        residual = np.full((11, 5), .05)
        small, _ = fit_group_offsets(residual, meta, 1, CONFIG)
        large, _ = fit_group_offsets(residual, meta, 100, CONFIG)
        zero, _ = fit_group_offsets(residual, meta, None, CONFIG)
        for group in CONFIG["age_groups"]:
            self.assertLess(abs(large[group]["offset"]), abs(small[group]["offset"]))
            self.assertEqual(zero[group]["offset"], 0)

    def test_frozen_ensemble_original_correction_and_zero_reproduction(self):
        payloads, calibrations, baseline = synthetic_original()
        fingerprint = joblib.hash((payloads, calibrations, baseline))
        rows, audit = candidate_forecasts(payloads, calibrations, baseline, CONFIG, AMENDMENT)
        self.assertEqual(joblib.hash((payloads, calibrations, baseline)), fingerprint)
        self.assertEqual(len(rows), 440)
        zero = pd.DataFrame(rows).query("setting_id == 'age_penalty=none'")
        keys = ["sex", "age", "horizon"]
        np.testing.assert_array_equal(zero.set_index(keys).sort_index().log_prediction,
                                      baseline.set_index(keys).sort_index().log_prediction)
        self.assertTrue(all(not row["original_correction_refitted"] for row in audit))
        for row in audit:
            original = next(c for c in calibrations if c["sex"] == row["sex"])
            self.assertEqual(row["original_b0"], original["b0"])
            self.assertEqual(row["original_b1"], original["b1"])

    def test_future_training_labels_and_missing_seed_are_rejected(self):
        payloads, calibrations, baseline = synthetic_original()
        with self.assertRaises(ValueError):
            candidate_forecasts(payloads[:-1], calibrations, baseline, CONFIG, AMENDMENT)
        for payload in payloads:
            payload["target_meta"].loc[0, "label_end"] = 2004
        with self.assertRaisesRegex(ValueError, "future"):
            candidate_forecasts(payloads, calibrations, baseline, CONFIG, AMENDMENT)

    def test_source_bounds_do_not_enter_correction(self):
        payloads, calibrations, baseline = synthetic_original()
        original, _ = candidate_forecasts(payloads, calibrations, baseline, CONFIG, AMENDMENT)
        baseline["rate_lower"], baseline["rate_upper"] = -np.inf, np.nan
        altered, _ = candidate_forecasts(payloads, calibrations, baseline, CONFIG, AMENDMENT)
        np.testing.assert_array_equal(pd.DataFrame(original).prediction, pd.DataFrame(altered).prediction)

    def test_age_selection_completed_windows_and_bad_values(self):
        scores = age_evidence()
        expected = select_age(scores, CONFIG, AMENDMENT, 2008, "Male")
        self.assertEqual(expected["penalty"], 10)
        self.assertEqual(expected["inner_origins"], [2003])
        scores.loc[scores.origin.gt(2003), "absolute_log_error"] = np.nan
        self.assertEqual(expected, select_age(scores, CONFIG, AMENDMENT, 2008, "Male"))
        scores.loc[scores.origin.eq(2003), "absolute_log_error"] = np.nan
        with self.assertRaises(ValueError):
            select_age(scores, CONFIG, AMENDMENT, 2008, "Male")

    def test_age_cold_start_ties_and_incomplete_grid(self):
        scores = age_evidence()
        self.assertEqual(select_age(scores, CONFIG, AMENDMENT, 2007, "Female")["penalty"], 100)
        scores["absolute_log_error"] = 0
        self.assertIsNone(select_age(scores, CONFIG, AMENDMENT, 2008, "Female")["penalty"])
        scores = scores.drop(scores[scores.origin.eq(2003) & scores.sex.eq("Female")].index[0])
        with self.assertRaises(ValueError):
            select_age(scores, CONFIG, AMENDMENT, 2008, "Female")

    def test_zero_retention_reproduces_frozen_interval_rule(self):
        points, bank = interval_inputs()
        expected, _ = apply_bank(points, bank, CONFIG)
        actual, _ = apply_retention(points, bank, CONFIG, {"Male": 0, "Female": 0}, "unchanged")
        np.testing.assert_array_equal(actual[["lower", "median", "upper"]], expected[["lower", "median", "upper"]])

    def test_full_and_sex_specific_retention_preserve_paired_blocks(self):
        points, bank = interval_inputs()
        before = joblib.hash(bank)
        _, draws = apply_retention(points, bank, CONFIG, {"Male": 1, "Female": .5}, "retained")
        self.assertEqual(joblib.hash(bank), before)
        self.assertEqual(draws.groupby("residual_origin").size().tolist(), [110]*5)
        matrix = draws.log_draw.to_numpy().reshape(5, 110)
        fraction = bank["coords"].sex.map({"Male": 1, "Female": .5}).to_numpy()
        expected = 3 + bank["centered_residuals"] + fraction[None, :]*bank["center"][None, :]
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.cov(matrix, rowvar=False), np.cov(bank["raw_residuals"], rowvar=False), atol=1e-14)

    def test_interval_selection_sparse_completed_chronology(self):
        family = "tcn_age_group_correction"
        rows = [{"origin": o, "sex": sex, "age": age, "horizon": 5, "scale": "log_rate",
                 "family": retention_family(family, r), "wis_50_80": ([.3, .1, .2] if o == 2012 else [.6, .4, 0])[i]}
                for o in [2012, 2013] for sex in CONFIG["sexes"] for age in CONFIG["ages"]
                for i, r in enumerate([0, .5, 1])]
        wis = pd.DataFrame(rows)
        self.assertEqual(select_retention(wis, CONFIG, AMENDMENT, 2016, "Male", family)["retention"], 0)
        self.assertEqual(select_retention(wis, CONFIG, AMENDMENT, 2017, "Male", family)["retention"], .5)
        self.assertEqual(select_retention(wis, CONFIG, AMENDMENT, 2018, "Male", family)["retention"], 1)
        wis.loc[wis.origin.eq(2013), "wis_50_80"] = np.nan
        self.assertEqual(select_retention(wis, CONFIG, AMENDMENT, 2017, "Male", family)["retention"], .5)
        with self.assertRaises(ValueError):
            select_retention(wis, CONFIG, AMENDMENT, 2018, "Male", family)

    def test_invalid_group_penalty_and_retention_rejected(self):
        with self.assertRaises(ValueError):
            fit_group_offsets(np.ones((11, 5)), pd.DataFrame({"age": CONFIG["ages"]}), -1, CONFIG)
        points, bank = interval_inputs()
        with self.assertRaises(ValueError):
            apply_retention(points, bank, CONFIG, {"Male": 2, "Female": 0}, "bad")


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ImprovementsTests))
    names = ["src/gbd_park/improvements.py", "scripts/run_improvements.py", "tests/test_improvements.py",
             "study_design/exploratory_v1_1.md", "study_design/exploratory_v1_1.json",
             "src/gbd_park/intervals.py", "src/gbd_park/scoring.py", "src/gbd_park/adaptation.py"]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "errors": len(result.errors), "failures": len(result.failures), "synthetic_data_only": True,
              "tested_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    output = ROOT / "work/improvements-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
