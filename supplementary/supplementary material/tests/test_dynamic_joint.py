"""Synthetic checks for coherent robust dynamic count/composition/population forecasts."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from gbd_park.dynamic_joint import SETTINGS, fit_dynamic_joint, helmert_basis, simulate_fitted

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
ARRAYS = ["point_rates", "point_counts", "point_populations", "rate_draws", "count_draws", "population_draws", "point_latent", "draw_latent"]


def fixture(constant_curvature=False):
    records = []
    for year in range(1990, 2024):
        t = year-1990
        for si, sex in enumerate(CONFIG["sexes"]):
            for ai, age in enumerate(CONFIG["ages"]):
                wave = 0 if constant_curvature else .03*np.sin(t/4+ai/5+si/3)
                population = (1.8e6-.2e6*si)*np.exp(-.18*ai+(.012+.001*ai)*t+wave)
                rate_wave = 0 if constant_curvature else .025*np.sin(t/3+.15*ai+si/2)
                rate = (12+5*ai**1.3)*np.exp((.009+.001*ai)*t+rate_wave)
                records.append({"location_name": "Saudi Arabia", "outcome": "prevalence", "year": year,
                                "sex": sex, "age": age, "rate": rate, "count": rate*population/100000,
                                "implied_population": population})
    return pd.DataFrame(records)


class DynamicJointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = fixture()
        cls.result = fit_dynamic_joint(cls.panel, CONFIG, 2018, "Saudi Arabia", "prevalence", seed=71)
        cls.fitted = cls.result["fitted_state"]

    def test_identifiable_basis_and_age_slope_smoothing(self):
        basis = helmert_basis(11)
        np.testing.assert_allclose(basis.T @ basis, np.eye(10), atol=3e-16)
        np.testing.assert_allclose(basis.T @ np.ones(11), 0, atol=5e-16)
        smooth = self.fitted["age_smoothing"]
        np.testing.assert_allclose(smooth @ np.ones(11), 1, atol=5e-16)
        age_slope = np.array([(-1)**i for i in range(11)], dtype=float)
        self.assertLess(np.sum(np.diff(smooth @ age_slope)**2), np.sum(np.diff(age_slope)**2))
        np.testing.assert_allclose(self.fitted["slope_smoothing"][1:11, 1:11], basis.T @ smooth @ basis, atol=1e-16)
        self.assertEqual(self.fitted["transformed_history"].shape, (29, 44))

    def test_history_cutoff_future_and_other_case_invariance(self):
        altered = self.panel.copy()
        altered.loc[altered.year.gt(2018), ["rate", "count", "implied_population"]] = np.nan
        other = altered.copy()
        other["location_name"] = "Bahrain"
        other[["rate", "count", "implied_population"]] = -1.
        another_outcome = altered.copy()
        another_outcome["outcome"] = "incidence"
        another_outcome[["rate", "count", "implied_population"]] = np.nan
        test = fit_dynamic_joint(pd.concat([other, altered, another_outcome], ignore_index=True), CONFIG,
                                 2018, "Saudi Arabia", "prevalence", seed=71)
        for name in ARRAYS:
            np.testing.assert_array_equal(test[name], self.result[name])
        self.assertEqual(test["diagnostics"]["last_input_year"], 2018)
        self.assertEqual(test["diagnostics"]["curvature_years"], list(range(1992, 2019)))
        self.assertEqual(test["diagnostics"]["filter_update_years"], list(range(1991, 2019)))

    def test_input_grid_positivity_and_population_consistency(self):
        index = self.panel.index[self.panel.year.eq(2010)][0]
        for kind in ["missing", "duplicate", "rate_zero", "count_nan", "population_mismatch", "age"]:
            with self.subTest(kind=kind):
                panel = self.panel.copy()
                if kind == "missing":
                    panel = panel.drop(index)
                elif kind == "duplicate":
                    panel = pd.concat([panel, panel.loc[[index]]], ignore_index=True)
                elif kind == "rate_zero":
                    panel.loc[index, "rate"] = 0
                elif kind == "count_nan":
                    panel.loc[index, "count"] = np.nan
                elif kind == "population_mismatch":
                    panel.loc[index, "implied_population"] *= 1.05
                else:
                    panel.loc[index, "age"] = "unknown"
                with self.assertRaises(ValueError):
                    fit_dynamic_joint(panel, CONFIG, 2018, "Saudi Arabia", "prevalence")
        with self.assertRaises(ValueError):
            fit_dynamic_joint(self.panel, CONFIG, 2018, "Jordan", "prevalence")

    def test_complete_positive_arrays_and_rate_count_population_identity(self):
        self.assertEqual(self.result["coords"].to_records(index=False).tolist(),
                         [(sex, age) for sex in CONFIG["sexes"] for age in CONFIG["ages"]])
        for name in ARRAYS[:6]:
            expected = (5, 22) if name.startswith("point_") else (1024, 5, 22)
            self.assertEqual(self.result[name].shape, expected)
            self.assertTrue(np.isfinite(self.result[name]).all())
            self.assertTrue((self.result[name] > 0).all())
        np.testing.assert_array_equal(self.result["point_rates"], self.result["point_counts"]/self.result["point_populations"]*100000)
        np.testing.assert_array_equal(self.result["rate_draws"], self.result["count_draws"]/self.result["population_draws"]*100000)

    def test_exact_positive_totals_and_joint_age_share_accounting(self):
        for sex_index in range(2):
            counts = self.result["count_draws"][..., sex_index*11:(sex_index+1)*11]
            total = np.exp(self.result["draw_latent"][..., sex_index*22])
            np.testing.assert_allclose(counts.sum(axis=-1), total, rtol=3e-15)
            shares = counts/total[..., None]
            np.testing.assert_allclose(shares.sum(axis=-1), 1, rtol=3e-15)
            share80 = shares[..., 7:].sum(axis=-1)
            self.assertTrue(((share80 > 0) & (share80 < 1)).all())
            point_total = np.exp(self.result["point_latent"][:, sex_index*22])
            np.testing.assert_allclose(self.result["point_counts"][:, sex_index*11:(sex_index+1)*11].sum(axis=1), point_total, rtol=3e-15)
        np.testing.assert_allclose(self.result["count_draws"].sum(axis=-1),
                                   self.result["count_draws"][..., :11].sum(axis=-1)+self.result["count_draws"][..., 11:].sum(axis=-1),
                                   rtol=22*np.finfo(float).eps, atol=0)

    def test_antithetic_latent_paths_and_point_not_median_definition(self):
        draws = self.result["draw_latent"]
        expected = np.broadcast_to(self.result["point_latent"], draws[:512].shape)
        np.testing.assert_allclose((draws[:512]+draws[512:])/2, expected, rtol=2e-15, atol=2e-14)
        self.assertIn("not_predictive_median", self.result["diagnostics"]["point_definition"])
        self.assertGreater(np.max(np.abs(np.median(self.result["count_draws"], axis=0)-self.result["point_counts"])), 1e-9)
        self.assertFalse(self.result["diagnostics"]["draws_are_independent_temporal_blocks"])

    def test_seed_determinism_and_source_immutability(self):
        before = self.panel.copy(deep=True)
        config = copy.deepcopy(CONFIG)
        replay = fit_dynamic_joint(self.panel, config, 2018, "Saudi Arabia", "prevalence", seed=71)
        for name in ARRAYS:
            np.testing.assert_array_equal(replay[name], self.result[name])
        other_seed = simulate_fitted(self.fitted, seed=72)
        for name in ARRAYS[:3]:
            np.testing.assert_array_equal(other_seed[name], self.result[name])
        self.assertFalse(np.array_equal(other_seed["rate_draws"], self.result["rate_draws"]))
        pd.testing.assert_frame_equal(before, self.panel)
        self.assertEqual(config, CONFIG)

    def test_covariance_is_regularized_and_state_uncertainty_retained(self):
        fit = self.fitted
        self.assertGreater(np.linalg.eigvalsh(fit["sigma"]).min(), 0)
        self.assertGreater(np.linalg.eigvalsh(fit["final_covariance"]).min(), 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(fit["shrunk_correlation"]).min(), .75-1e-12)
        np.testing.assert_allclose(np.diag(fit["sigma"]), fit["mad_scales"]**2, rtol=2e-16)
        np.testing.assert_array_equal(fit["process_covariance"][:44, :44], .2*fit["sigma"])
        np.testing.assert_array_equal(fit["process_covariance"][44:, 44:], .1*fit["sigma"])
        # Multiplication by the stored1/12 and direct division can differ by one ulp.
        np.testing.assert_allclose(fit["observation_covariance"], fit["sigma"]/12, rtol=3e-16, atol=0)
        self.assertTrue(((fit["filter_weights"] > 0) & (fit["filter_weights"] <= 1)).all())
        self.assertFalse(SETTINGS["extra_global_variance_mixture"])
        self.assertFalse(SETTINGS["empirical_residual_bank_added"])

    def test_covariance_inputs_reconstruct_fixed_shrinkage(self):
        fit = self.fitted
        curvature = np.diff(fit["transformed_history"], n=2, axis=0)
        center = np.median(curvature, axis=0)
        scales = np.maximum(1.482602218505602*np.median(abs(curvature-center), axis=0), 1e-4)
        z = (curvature-center)/scales
        rms = np.linalg.norm(z, axis=1)/np.sqrt(44)
        factors = np.minimum(1, 2.5/np.maximum(rms, np.finfo(float).tiny))
        clipped = z*factors[:, None]
        empirical = np.cov(clipped, rowvar=False, ddof=1)
        variance = np.diag(empirical)
        keep = np.flatnonzero(variance > 1e-12)
        correlation = np.zeros((44, 44))
        correlation[np.ix_(keep, keep)] = empirical[np.ix_(keep, keep)]/np.sqrt(np.outer(variance[keep], variance[keep]))
        np.fill_diagonal(correlation, 1)
        expected = np.outer(scales, scales)*(.75*np.eye(44)+.25*correlation)
        np.testing.assert_allclose(fit["sigma"], expected, atol=1e-18, rtol=1e-12)
        np.testing.assert_allclose(fit["radial_weights"], factors, atol=1e-14)

    def test_first_observation_conditioned_once_and_slopes_dynamic(self):
        fit = self.fitted
        np.testing.assert_array_equal(fit["initial_mean"][:44], fit["transformed_history"][0])
        np.testing.assert_array_equal(fit["initial_mean"][44:], np.zeros(44))
        np.testing.assert_array_equal(fit["filtered_means"][0], fit["initial_mean"])
        np.testing.assert_array_equal(fit["initial_covariance"][:44, :44], fit["observation_covariance"])
        np.testing.assert_array_equal(fit["initial_covariance"][44:, 44:], 25*fit["sigma"])
        self.assertEqual(len(fit["filter_weights"]), len(fit["years"])-1)
        self.assertGreater(np.max(np.std(fit["filtered_means"][:, 44:], axis=0)), 1e-4)
        first = fit["final_mean"][:44]+fit["final_mean"][44:]
        np.testing.assert_allclose(self.result["point_latent"][0], first, atol=1e-14)

    def test_joint_outlier_weight_and_constant_curvature_floor(self):
        panel = self.panel.copy()
        take = panel.year.eq(2010)
        panel.loc[take, ["rate", "count"]] *= 5
        outlier = fit_dynamic_joint(panel, CONFIG, 2018, "Saudi Arabia", "prevalence")
        fit = outlier["fitted_state"]
        self.assertLess(fit["radial_weights"].min(), .1)
        self.assertLess(fit["filter_weights"].min(), .1)
        self.assertLessEqual(np.sqrt(np.mean(fit["clipped_standardized_curvature"]**2, axis=1)).max(), 2.5+1e-12)
        constant = self.panel.copy()
        constant["count"] = 100
        constant["rate"] = 10
        constant["implied_population"] = 1e6
        result = fit_dynamic_joint(constant, CONFIG, 2018, "Saudi Arabia", "prevalence")
        np.testing.assert_array_equal(result["fitted_state"]["mad_scales"], np.repeat(1e-4, 44))
        np.testing.assert_array_equal(result["fitted_state"]["shrunk_correlation"], np.eye(44))
        self.assertTrue(np.isfinite(result["rate_draws"]).all())

    def test_saved_state_replay_json_diagnostics_and_fixed_draw_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"fit.joblib"
            joblib.dump(self.fitted, path)
            saved = joblib.load(path)
            replay = simulate_fitted(saved, seed=71)
            for name in ARRAYS:
                np.testing.assert_array_equal(replay[name], self.result[name])
        json.dumps(self.result["diagnostics"], allow_nan=False)
        for count in [512, 2048, 1024., True]:
            with self.assertRaises(ValueError):
                simulate_fitted(self.fitted, draws=count)
        changed = copy.deepcopy(self.fitted)
        changed["settings"]["slope_damping"] = .9
        with self.assertRaises(ValueError):
            simulate_fitted(changed)


if __name__ == "__main__":
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DynamicJointTests))
    files = ["src/gbd_park/dynamic_joint.py", "tests/test_dynamic_joint.py"]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "errors": len(result.errors),
              "failures": len(result.failures), "elapsed_seconds": time.perf_counter()-started,
              "fixture_type": "synthetic_no_evaluation_results_inspected", "device": "cpu",
              "tested_code_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    out = ROOT/"work/dynamic-joint-validation"
    out.mkdir(parents=True, exist_ok=True)
    (out/"unit_tests.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
