"""Scientific and leakage checks for stage-1 baseline forecasting."""

import os

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "1"

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from gbd_park.local import age_smooth_forecast, arima_forecast, forecast_setting, select_history, settings_grid
from gbd_park.scoring import score_forecasts, select_family, select_settings

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def synthetic_panel():
    return pd.DataFrame([
        {"location_name": country, "sex": sex, "outcome": "prevalence", "age": age,
         "year": year, "rate": np.exp(3 + 0.12 * a + 0.02 * (year - 1990) + 0.04 * np.sin(year + a))}
        for country in ["Saudi Arabia", "Jordan"] for sex in CONFIG["sexes"]
        for a, age in enumerate(CONFIG["ages"]) for year in range(1990, 2024)
    ])


class BaselineTests(unittest.TestCase):
    def test_log_linear_extrapolation(self):
        panel = synthetic_panel()
        panel["rate"] = np.exp(2 + 0.03 * (panel.year - 1990))
        spec = next(s for s in settings_grid(CONFIG) if s["family"] == "log_trend")
        rows, _, _ = forecast_setting(panel, CONFIG, 2003, "Male", spec)
        expected = [np.exp(2 + 0.03 * (r["forecast_year"] - 1990)) for r in rows]
        np.testing.assert_allclose([r["prediction"] for r in rows], expected, rtol=1e-12)

    def test_all_fitted_families_ignore_future_and_other_country(self):
        panel = synthetic_panel()
        changed = panel.copy()
        changed.loc[changed.year.gt(2003) | changed.location_name.ne("Saudi Arabia"), "rate"] *= 1000
        for family in CONFIG["models"]["local_order"]:
            cfg = copy.deepcopy(CONFIG)
            if family == "arima":
                # Exercise the actual complete ARIMA order search on one age cell.
                cfg["ages"] = ["45-49"]
            spec = next(s for s in settings_grid(cfg) if s["family"] == family)
            before, fits, _ = forecast_setting(panel, cfg, 2003, "Male", spec)
            after, _, _ = forecast_setting(changed, cfg, 2003, "Male", spec)
            np.testing.assert_array_equal([r["prediction"] for r in before], [r["prediction"] for r in after])
            self.assertTrue(all(f["status"] == "ok" for f in fits), family)

    def test_95plus_slope_is_not_smoothed(self):
        years = np.arange(1990, 2004)
        slopes = np.r_[np.linspace(0.1, 0.2, 10), -0.5]
        values = np.arange(11)[None, :] + (years - years[-1])[:, None] / 10 * slopes[None, :]
        prediction, _ = age_smooth_forecast(years, values, [1, 5], 100)
        np.testing.assert_allclose(prediction[-1], 10 - 0.5 * np.array([1, 5]) / 10, atol=1e-10)

    def test_missing_history_is_rejected(self):
        panel = synthetic_panel()
        bad = panel.drop(panel[(panel.location_name == "Saudi Arabia") & (panel.sex == "Male") & (panel.year == 1999)].index)
        with self.assertRaises(ValueError):
            select_history(bad, CONFIG, 2003, "Male")

    def test_age_smooth_numerical_failure_emits_persistence(self):
        panel = synthetic_panel()
        spec = next(s for s in settings_grid(CONFIG) if s["family"] == "age_smooth_trend")
        with patch("gbd_park.local.age_smooth_forecast", side_effect=np.linalg.LinAlgError("test failure")):
            rows, fits, _ = forecast_setting(panel, CONFIG, 2003, "Male", spec)
        self.assertEqual(len(rows), 55)
        self.assertTrue(all(f["status"] == "fallback" for f in fits))
        for row in rows:
            actual = panel.loc[(panel.location_name == "Saudi Arabia") & (panel.sex == "Male")
                               & (panel.year == 2003) & (panel.age == row["age"]), "rate"].iloc[0]
            self.assertAlmostEqual(row["prediction"], actual)

    def test_arima_failures_remain_in_audit(self):
        values = np.linspace(2, 3, 14)
        with patch("gbd_park.local.ARIMA", side_effect=ValueError("test failure")):
            prediction, meta, audit = arima_forecast(values, 5, CONFIG["models"]["arima"])
        np.testing.assert_array_equal(prediction, np.repeat(3, 5))
        self.assertEqual(meta["status"], "fallback")
        self.assertEqual(len(audit), 16)
        self.assertTrue(all(a["status"] == "failed" for a in audit))

    def test_known_metric_values_and_complete_grid(self):
        predictions = pd.DataFrame([
            dict(target="Saudi Arabia", sex="Male", age=a, outcome="prevalence", origin=2000,
                 horizon=1, forecast_year=2001, family="persistence", setting_id="persistence", prediction=p)
            for a, p in [("a", 20.0), ("b", 10.0)]])
        truth = predictions[["target", "sex", "age", "outcome", "forecast_year"]].copy()
        truth["observed_rate"] = [10., 20.]
        result = score_forecasts(predictions, truth, ["a", "b"], [1], 2018)
        np.testing.assert_allclose(result.absolute_log_error, np.log(2))
        np.testing.assert_allclose(result.absolute_rate_error, 10)
        with self.assertRaises(ValueError):
            score_forecasts(predictions.iloc[:1], truth, ["a", "b"], [1], 2018)
        with self.assertRaises(ValueError):
            score_forecasts(predictions, truth.iloc[:1], ["a", "b"], [1], 2018)
        bad = predictions.copy()
        bad["forecast_year"] = 2002
        with self.assertRaises(ValueError):
            score_forecasts(bad, truth, ["a", "b"], [1], 2018)
        with self.assertRaises(ValueError):
            score_forecasts(predictions, truth, ["a", "b"], [1], 2000)

    def test_tuning_cannot_use_uncompleted_labels(self):
        data = []
        for origin in range(2003, 2014):
            for i, setting in enumerate(["a", "b"]):
                for age in CONFIG["ages"]:
                    data.append(dict(origin=origin, sex="Male", family="log_trend", setting_id=setting,
                                     age=age, horizon=5, absolute_log_error=(1 if i == 0 else 2) if origin <= 2004 else (100 if i == 0 else 0),
                                     parameter_count=2, grid_order=i))
        scored = pd.DataFrame(data)
        before = select_settings(scored, CONFIG, 2009, "Male", "log_trend")
        scored.loc[scored.origin.gt(2004), "absolute_log_error"] *= 100
        self.assertEqual(before, select_settings(scored, CONFIG, 2009, "Male", "log_trend"))
        self.assertEqual(before[0], "a")
        self.assertEqual(before[2], [2003, 2004])

    def test_family_selection_respects_completion_date(self):
        rows = []
        for origin in range(2009, 2014):
            for i, family in enumerate(CONFIG["models"]["local_order"]):
                for age in CONFIG["ages"]:
                    rows.append(dict(origin=origin, sex="Male", age=age, horizon=5, family=family,
                                     absolute_log_error=i if origin == 2009 else 20-i, parameter_count=i))
        scores = pd.DataFrame(rows)
        result = select_family(scores, CONFIG, 2014, "Male")
        self.assertEqual(result["selected_family"], "persistence")
        self.assertEqual(result["selection_origins"], "2009")
        self.assertEqual(result["last_selection_target_year"], 2014)

    def test_locked_finite_grid(self):
        grid = settings_grid(CONFIG)
        self.assertEqual(len(grid), 18)
        self.assertEqual(len({s["setting_id"] for s in grid}), 18)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BaselineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    code = sorted((ROOT / "src/gbd_park").glob("*.py")) + [Path(__file__).resolve(), ROOT / "scripts/run_local_baselines.py"]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "tested_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code}}
    out = ROOT / "work/local-baselines-validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
