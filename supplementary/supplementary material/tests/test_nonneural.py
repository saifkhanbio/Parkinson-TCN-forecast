"""Leakage, weighting, adaptation and numerical checks for non-neural controls."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gbd_park.adaptation import fit_adaptation, correction
from gbd_park.pooled import (base_grid, settings_grid, build_examples, country_pool,
                            balanced_weights, fit_base, forecast_origin, predict_changes)
from gbd_park.scoring import select_settings

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def synthetic_panel():
    return pd.DataFrame([
        {"location_name": country["name"], "sex": sex, "outcome": "prevalence", "age": age,
         "year": year, "rate": np.exp(2 + c*.05 + a*.1 + s*.1 + .01*(year-1990) + .025*np.sin(year+a+c))}
        for c, country in enumerate(CONFIG["countries"]) for s, sex in enumerate(CONFIG["sexes"])
        for a, age in enumerate(CONFIG["ages"]) for year in range(1990, 2024)])


class NonneuralTests(unittest.TestCase):
    def test_completed_labels_and_donor_exclusion(self):
        x, y, meta = build_examples(synthetic_panel(), CONFIG, 2003, country_pool(CONFIG, "Saudi Arabia", "donor"))
        self.assertEqual(x.shape, (264, 21))
        self.assertEqual(y.shape, (264, 5))
        self.assertEqual(meta.label_end.max(), 2003)
        self.assertEqual(meta.input_end.max(), 1998)
        self.assertNotIn("Saudi Arabia", set(meta.country))
        self.assertEqual(set(meta.sex), {"Male", "Female"})
        np.testing.assert_array_equal(x[:, 7], 0)
        np.testing.assert_array_equal(x[:, -11:].sum(axis=1), 1)

    def test_weights_balance_unequal_history_lengths(self):
        meta = pd.DataFrame([{"country": c, "sex": s, "age": a} for c in ["A", "B"]
                             for s in ["Male", "Female"] for a in ["young", "old"]
                             for _ in range(1 if (c, s, a) == ("A", "Male", "young") else 4)])
        meta["weight"] = balanced_weights(meta)
        np.testing.assert_allclose(meta.groupby("country").weight.sum(), len(meta)/2)
        np.testing.assert_allclose(meta.groupby(["country", "sex"]).weight.sum(), len(meta)/4)
        np.testing.assert_allclose(meta.groupby(["country", "sex", "age"]).weight.sum(), len(meta)/8)

    def test_future_changes_do_not_change_fitted_forecasts(self):
        panel = synthetic_panel()
        altered = panel.copy()
        altered.loc[altered.year.gt(2003), "rate"] *= 500
        bases = [base_grid(CONFIG)[0], next(x for x in base_grid(CONFIG) if x["algorithm"] == "boosting")]
        before = forecast_origin(panel, CONFIG, 2003, selected_bases=bases)
        after = forecast_origin(altered, CONFIG, 2003, selected_bases=bases)
        np.testing.assert_array_equal([r["prediction"] for r in before[0]], [r["prediction"] for r in after[0]])
        self.assertTrue(all(f["status"] == "ok" for f in before[1]))
        self.assertTrue(all(f["base_fingerprint_before"] == f["base_fingerprint_after"] for f in before[1]))

    def test_target_changes_cannot_change_donor_fit(self):
        panel = synthetic_panel()
        altered = panel.copy()
        altered.loc[altered.location_name.eq("Saudi Arabia"), "rate"] *= 100
        countries = country_pool(CONFIG, "Saudi Arabia", "donor")
        x, y, meta = build_examples(panel, CONFIG, 2003, countries)
        xx, yy, mm = build_examples(altered, CONFIG, 2003, countries)
        for base in [base_grid(CONFIG)[0], base_grid(CONFIG)[5]]:
            model = fit_base(x, y, meta, base, CONFIG)
            other = fit_base(xx, yy, mm, base, CONFIG)
            self.assertEqual(joblib.hash(model), joblib.hash(other))
        self.assertIn("Saudi Arabia", country_pool(CONFIG, "Saudi Arabia", "pooled"))

    def test_scaler_uses_only_fit_rows_and_predictions_do_not_mutate(self):
        x, y, meta = build_examples(synthetic_panel(), CONFIG, 2003, country_pool(CONFIG, "Saudi Arabia", "donor"))
        model = fit_base(x, y, meta, base_grid(CONFIG)[0], CONFIG)
        np.testing.assert_allclose(model["scaler"].mean_, np.average(x, axis=0, weights=balanced_weights(meta)), atol=1e-14)
        fingerprint = joblib.hash(model)
        predict_changes(model, x * 100)
        self.assertEqual(joblib.hash(model), fingerprint)

    def test_analytic_adaptation_solution(self):
        cal = fit_adaptation(np.ones((3, 5)), np.zeros((3, 5)), 10)
        np.testing.assert_allclose([cal["b0"], cal["b1"]], [.05, .03], atol=1e-8)
        zero = fit_adaptation(np.zeros((3, 5)), np.zeros((3, 5)), 1)
        np.testing.assert_array_equal(correction(zero), 0)

    def test_adaptation_agrees_with_independent_slack_optimizer(self):
        error = np.random.default_rng(123).normal(.04, .03, (6, 5))
        penalty = 0.1
        cal = fit_adaptation(error, np.zeros_like(error), penalty)
        e = error.ravel()
        design = np.column_stack([np.ones(len(e)), np.tile(np.arange(1, 6)/5, 6)])
        x0 = np.r_[0., 0., np.abs(e)+.01]
        objective = lambda x: x[2:].mean() + penalty*np.dot(x[:2], x[:2])
        constraints = {"type": "ineq", "fun": lambda x: np.r_[x[2:] - e + design@x[:2], x[2:] + e - design@x[:2]]}
        result = minimize(objective, x0, method="SLSQP", constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 1000})
        self.assertTrue(result.success)
        self.assertAlmostEqual(cal["objective"], result.fun, places=8)
        np.testing.assert_allclose([cal["b0"], cal["b1"]], result.x[:2], atol=2e-5)

    def test_missing_donor_cell_is_rejected(self):
        panel = synthetic_panel()
        panel = panel.drop(panel[(panel.location_name == "Jordan") & (panel.year == 1999)].index)
        with self.assertRaises(ValueError):
            build_examples(panel, CONFIG, 2003, country_pool(CONFIG, "Saudi Arabia", "donor"))

    def test_source_failure_preserves_forecast_cells(self):
        with patch("gbd_park.pooled.fit_base", side_effect=ValueError("synthetic fit failure")):
            forecasts, fits, calibrations, _ = forecast_origin(synthetic_panel(), CONFIG, 2003, selected_bases=[base_grid(CONFIG)[0]])
        self.assertEqual(len(forecasts), 6*2*11*5)  # pooled, unadapted, four penalties
        self.assertTrue(all(r["status"] == "fallback" for r in forecasts))
        self.assertTrue(all(f["reason"] for f in fits))
        self.assertEqual(len(calibrations), 8)
        self.assertTrue(all(c["status"] == "fallback" for c in calibrations))

    def test_target_female_history_does_not_change_male_adaptation(self):
        panel = synthetic_panel()
        altered = panel.copy()
        altered.loc[altered.location_name.eq("Saudi Arabia") & altered.sex.eq("Female"), "rate"] *= 3
        before = forecast_origin(panel, CONFIG, 2003, selected_bases=[base_grid(CONFIG)[0]])[0]
        after = forecast_origin(altered, CONFIG, 2003, selected_bases=[base_grid(CONFIG)[0]])[0]
        select = lambda rows: [r["prediction"] for r in rows if r["family"] == "donor_ridge_adapted" and r["sex"] == "Male"]
        np.testing.assert_array_equal(select(before), select(after))

    def test_fixed_grid_and_time_restricted_adaptation_tuning(self):
        grid = settings_grid(CONFIG)
        self.assertEqual(len(grid), 78)
        family = "donor_ridge_adapted"
        specs = [s for s in grid if s["family"] == family]
        rows = [dict(sex="Male", family=family, origin=o, horizon=5, age=a, setting_id=s["setting_id"],
                     absolute_log_error=i if o <= 2004 else 100-i, parameter_count=112, grid_order=s["grid_order"])
                for o in range(2003, 2014) for i, s in enumerate(specs) for a in CONFIG["ages"]]
        scored = pd.DataFrame(rows)
        chosen = select_settings(scored, CONFIG, 2009, "Male", family)
        self.assertEqual(chosen[0], specs[0]["setting_id"])
        self.assertEqual(chosen[2], [2003, 2004])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NonneuralTests))
    files = [ROOT/f for f in ["src/gbd_park/pooled.py", "src/gbd_park/adaptation.py", "src/gbd_park/scoring.py",
                              "tests/test_nonneural.py", "scripts/run_nonneural.py", "study_design/nonneural_implementation.md"]]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "tested_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    out = ROOT / "work/nonneural-validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tests.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
