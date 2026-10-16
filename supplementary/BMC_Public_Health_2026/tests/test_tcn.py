"""CPU checks for the causal TCN, source exclusions and frozen adaptation."""

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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from gbd_park.adaptation import correction, fit_adaptation
from gbd_park.pooled import balanced_weights, build_examples, country_pool, target_inputs
from gbd_park.tcn import (CompactTCN, base_grid, configure_torch, fit_tcn,
                          load_checkpoint, predict_changes, save_checkpoint,
                          state_fingerprint)
from gbd_park.tcn_forecasting import (base_id, fit_seed, make_forecasts,
                                      select_tcn_settings)

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def synthetic_panel():
    """Complete synthetic country/sex/age histories; no evaluation data are read."""
    return pd.DataFrame([
        {"location_name": country["name"], "sex": sex, "outcome": "prevalence",
         "age": age, "year": year,
         "rate": np.exp(2 + c * .05 + a * .1 + s * .1 + .01 * (year - 1990)
                        + .025 * np.sin(year + a + c))}
        for c, country in enumerate(CONFIG["countries"])
        for s, sex in enumerate(CONFIG["sexes"])
        for a, age in enumerate(CONFIG["ages"])
        for year in range(1990, 2024)
    ])


class TCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()
        cls.base = {**base_grid(CONFIG)[0], "epochs": 2}
        cls.countries = country_pool(CONFIG, CONFIG["primary_target"], "donor")
        cls.x, cls.y, cls.meta = build_examples(cls.panel, CONFIG, 2003, cls.countries)
        cls.fitted = fit_tcn(cls.x, cls.y, cls.meta, cls.base, CONFIG, seed=11, device="cpu")
        cls.payloads = [fit_seed(cls.panel, CONFIG, 2003, cls.base, seed, device="cpu")
                        for seed in CONFIG["models"]["tcn"]["ensemble_seeds"]]

    def test_causal_encoder_excludes_future_positions(self):
        for channels in CONFIG["models"]["tcn"]["channels"]:
            configure_torch(11, "cpu")
            model = CompactTCN(channels, CONFIG).eval()
            sequence = torch.randn(3, 1, 8)
            original = model.temporal_features(sequence).detach()
            for split in range(1, 8):
                altered = sequence.clone()
                altered[:, :, split:] += 100
                changed = model.temporal_features(altered).detach()
                torch.testing.assert_close(original[:, :, :split], changed[:, :, :split],
                                           rtol=0, atol=0)

    def test_parameter_cap_and_five_horizon_output(self):
        for channels, expected in [(16, 1782), (32, 6566)]:
            model = CompactTCN(channels, CONFIG).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertLess(expected, CONFIG["models"]["tcn"]["maximum_parameters"])
            self.assertEqual(tuple(model(torch.zeros(2, 21)).shape), (2, 5))
        with self.assertRaisesRegex(ValueError, "parameter cap"):
            CompactTCN(64, CONFIG)

    def test_same_seed_is_exact_and_different_seed_changes_fit(self):
        repeated = fit_tcn(self.x, self.y, self.meta, self.base, CONFIG, seed=11, device="cpu")
        different = fit_tcn(self.x, self.y, self.meta, self.base, CONFIG, seed=23, device="cpu")
        self.assertEqual(state_fingerprint(self.fitted), state_fingerprint(repeated))
        self.assertNotEqual(state_fingerprint(self.fitted), state_fingerprint(different))
        np.testing.assert_array_equal(predict_changes(self.fitted, self.x),
                                      predict_changes(repeated, self.x))
        self.assertFalse(np.array_equal(predict_changes(self.fitted, self.x),
                                        predict_changes(different, self.x)))

    def test_future_rows_cannot_change_training_or_model(self):
        changed = self.panel.copy()
        changed.loc[changed.year.gt(2003), "rate"] *= 500
        x, y, meta = build_examples(changed, CONFIG, 2003, self.countries)
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self.y)
        pd.testing.assert_frame_equal(meta, self.meta)
        fitted = fit_tcn(x, y, meta, self.base, CONFIG, seed=11, device="cpu")
        self.assertEqual(state_fingerprint(self.fitted), state_fingerprint(fitted))

    def test_all_target_rows_are_excluded_from_donor_fit(self):
        changed = self.panel.copy()
        changed.loc[changed.location_name.eq(CONFIG["primary_target"]), "rate"] *= 100
        x, y, meta = build_examples(changed, CONFIG, 2003, self.countries)
        self.assertNotIn(CONFIG["primary_target"], set(meta.country))
        self.assertEqual(set(meta.sex), set(CONFIG["sexes"]))
        np.testing.assert_array_equal(x, self.x)
        np.testing.assert_array_equal(y, self.y)
        fitted = fit_tcn(x, y, meta, self.base, CONFIG, seed=11, device="cpu")
        self.assertEqual(state_fingerprint(self.fitted), state_fingerprint(fitted))

    def test_training_requires_completed_full_horizon(self):
        self.assertEqual(self.x.shape, (264, 21))
        self.assertEqual(self.y.shape, (264, 5))
        self.assertEqual(int(self.meta.input_end.max()), 1998)
        self.assertEqual(int(self.meta.label_end.max()), 2003)
        np.testing.assert_array_equal(self.meta.label_end - self.meta.window_origin, 5)
        self.assertTrue(self.meta.label_end.le(2003).all())
        with self.assertRaisesRegex(ValueError, "No complete training windows"):
            build_examples(self.panel, CONFIG, 2001, self.countries)

    def test_scaler_uses_weighted_training_rows_only(self):
        # Keep one window in one stratum to make its hierarchical weight differ.
        retain = ~((self.meta.country == self.countries[0]) & (self.meta.sex == "Male")
                   & (self.meta.age == CONFIG["ages"][0]) & (self.meta.window_origin == 1998))
        x, y, meta = self.x[retain], self.y[retain], self.meta.loc[retain].reset_index(drop=True)
        fitted = fit_tcn(x, y, meta, self.base, CONFIG, seed=11, device="cpu")
        weights = balanced_weights(meta)
        mean = np.average(x, axis=0, weights=weights)
        variance = np.average((x - mean) ** 2, axis=0, weights=weights)
        np.testing.assert_allclose(fitted["scaler"].mean_, mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(fitted["scaler"].var_, variance, rtol=1e-12, atol=1e-12)
        before = state_fingerprint(fitted)
        predict_changes(fitted, x * 100 + 100)
        self.assertEqual(state_fingerprint(fitted), before)

    def test_prediction_and_target_adaptation_preserve_frozen_network(self):
        before = state_fingerprint(self.fitted)
        x, y, meta = build_examples(self.panel, CONFIG, 2003, [CONFIG["primary_target"]])
        unadapted = predict_changes(self.fitted, x)
        for sex in CONFIG["sexes"]:
            mask = meta.sex.eq(sex).to_numpy()
            fitted_correction = fit_adaptation(y[mask], unadapted[mask], penalty=1)
            self.assertTrue(np.isfinite(correction(fitted_correction)).all())
        self.assertEqual(state_fingerprint(self.fitted), before)
        self.assertTrue(all(not p.requires_grad for p in self.fitted["model"].parameters()))
        self.assertFalse(self.fitted["model"].training)

    def test_checkpoint_round_trip_preserves_forecasts(self):
        x, _, _ = target_inputs(self.panel, CONFIG, 2003, CONFIG["primary_target"])
        expected = predict_changes(self.fitted, x)
        with tempfile.TemporaryDirectory(prefix="gbd_tcn_test_") as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(self.fitted, path)
            restored = load_checkpoint(path)
        self.assertEqual(state_fingerprint(self.fitted), state_fingerprint(restored))
        np.testing.assert_array_equal(expected, predict_changes(restored, x))

    def test_cuda_request_cannot_silently_fall_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises((RuntimeError, ValueError)):
                configure_torch(11, "cuda")
            with self.assertRaises((RuntimeError, ValueError)):
                fit_tcn(self.x, self.y, self.meta, self.base, CONFIG, seed=11, device="cuda")
            with self.assertRaises((RuntimeError, ValueError)):
                fit_seed(self.panel, CONFIG, 2003, self.base, seed=11, device="cuda")

    def test_full_pipeline_future_rows_cannot_change_forecasts(self):
        changed = self.panel.copy()
        changed.loc[changed.year.gt(2003), "rate"] *= 500
        altered = fit_seed(changed, CONFIG, 2003, self.base, seed=11, device="cpu")
        expected = self.payloads[0]
        for name in ["current_changes", "target_changes", "target_y", "levels"]:
            np.testing.assert_array_equal(altered[name], expected[name])
        self.assertEqual(altered["audit"]["fingerprint_before"],
                         expected["audit"]["fingerprint_before"])
        before, before_cal = make_forecasts([expected], CONFIG, [1], include_intercept=True)
        after, after_cal = make_forecasts([altered], CONFIG, [1], include_intercept=True)
        pd.testing.assert_frame_equal(pd.DataFrame(before), pd.DataFrame(after))
        pd.testing.assert_frame_equal(pd.DataFrame(before_cal), pd.DataFrame(after_cal))

    def test_target_female_history_cannot_change_male_adaptation(self):
        changed = self.panel.copy()
        changed.loc[changed.location_name.eq(CONFIG["primary_target"])
                    & changed.sex.eq("Female"), "rate"] *= 3
        altered = fit_seed(changed, CONFIG, 2003, self.base, seed=11, device="cpu")
        expected = self.payloads[0]
        self.assertEqual(altered["audit"]["fingerprint_before"],
                         expected["audit"]["fingerprint_before"])
        before, before_cal = make_forecasts([expected], CONFIG, [1], include_intercept=True)
        after, after_cal = make_forecasts([altered], CONFIG, [1], include_intercept=True)
        for left, right in [(before, after), (before_cal, after_cal)]:
            left, right = pd.DataFrame(left), pd.DataFrame(right)
            pd.testing.assert_frame_equal(left.loc[left.sex.eq("Male")], right.loc[right.sex.eq("Male")])
        female = expected["current_meta"].sex.eq("Female").to_numpy()
        np.testing.assert_allclose(altered["levels"][female] - expected["levels"][female], np.log(3))

    def test_ensemble_log_mean_precedes_one_correction_per_sex(self):
        # Deliberately asymmetric residuals make mean-of-corrections incorrect.
        payloads = deepcopy(self.payloads)
        for index, (payload, residual) in enumerate(zip(payloads, [-2, -1, .1, .2, .3])):
            payload["current_changes"][:] = index * .1
            female = payload["target_meta"].sex.eq("Female").to_numpy()
            errors = np.full_like(payload["target_y"], residual)
            errors[female] += .03
            payload["target_changes"] = payload["target_y"] - errors
        records, calibrations = make_forecasts(payloads, CONFIG, [1], include_intercept=True)
        frame, cal_frame = pd.DataFrame(records), pd.DataFrame(calibrations)
        first = payloads[0]
        current = np.mean([p["current_changes"] for p in payloads], axis=0)
        historical = np.mean([p["target_changes"] for p in payloads], axis=0)
        self.assertEqual(len(frame), 3 * 2 * 11 * 5)
        self.assertEqual(len(cal_frame), 4)
        self.assertEqual(set(frame.seed_or_ensemble), {"11|23|37|53|71"})
        for sex in CONFIG["sexes"]:
            selected = first["current_meta"].sex.eq(sex).to_numpy()
            eligible = first["target_meta"].sex.eq(sex).to_numpy()
            reference = fit_adaptation(first["target_y"][eligible], historical[eligible], 1)
            chosen = cal_frame.loc[cal_frame.sex.eq(sex) & cal_frame.family.eq("tcn_adapted")].iloc[0]
            np.testing.assert_allclose([chosen.b0, chosen.b1], [reference["b0"], reference["b1"]], atol=1e-12)
            wrong = np.mean([correction(fit_adaptation(p["target_y"][eligible],
                                                      p["target_changes"][eligible], 1))
                             for p in payloads], axis=0)
            self.assertGreater(np.max(np.abs(wrong - correction(reference))), .01)
            expected_unadapted = first["levels"][selected, None] + current[selected]
            for family, expected in [("tcn_unadapted", expected_unadapted),
                                     ("tcn_adapted", expected_unadapted + correction(reference))]:
                actual = frame.loc[frame.sex.eq(sex) & frame.family.eq(family), "log_prediction"]
                np.testing.assert_allclose(actual.to_numpy().reshape(11, 5), expected, atol=1e-12)
            intercept = cal_frame.loc[cal_frame.sex.eq(sex) & cal_frame.family.eq("tcn_intercept")].iloc[0]
            self.assertEqual(intercept.b1, 0)
        self.assertTrue(all(p["audit"]["fingerprint_before"] == p["audit"]["fingerprint_after"]
                            for p in self.payloads))

    def test_failed_seed_remains_in_complete_ensemble_fallback(self):
        with patch("gbd_park.tcn_forecasting.fit_tcn", side_effect=RuntimeError("synthetic failure")):
            failed = fit_seed(self.panel, CONFIG, 2003, self.base, seed=23, device="cpu")
        self.assertEqual(failed["audit"]["status"], "fallback")
        payloads = [failed if p["seed"] == 23 else p for p in self.payloads]
        records, calibrations = make_forecasts(payloads, CONFIG, [1], include_intercept=True)
        self.assertEqual(len(records), 3 * 2 * 11 * 5)
        self.assertTrue(all(r["status"] == "fallback" for r in records + calibrations))
        self.assertTrue(all(r["seed_or_ensemble"] == "11|23|37|53|71" for r in records + calibrations))
        self.assertTrue(all("23: RuntimeError: synthetic failure" in r["fallback_reason"] for r in records))
        levels = {(row.sex, row.age): failed["levels"][i]
                  for i, row in failed["current_meta"].iterrows()}
        np.testing.assert_array_equal([r["log_prediction"] for r in records],
                                      [levels[(r["sex"], r["age"])] for r in records])

    @staticmethod
    def selection_fixture():
        rows = []
        for origin in range(2003, 2014):
            for index, base in enumerate(base_grid(CONFIG)):
                for sex in CONFIG["sexes"]:
                    best = (0.1 if sex == "Male" else 1) if index == 1 else .01
                    for penalty in CONFIG["adaptation"]["penalties"]:
                        loss = .5 if index == 1 else (.1 if sex == "Male" else 2)
                        if index > 1:
                            loss = 3
                        loss += 0 if penalty == best else 1
                        if origin > 2004:
                            loss = 0 if index == 7 else 100
                        rows.extend({"family": "tcn_adapted", "base_id": base_id(base), "sex": sex,
                                     "origin": origin, "horizon": 5, "age": age,
                                     "adaptation_penalty": penalty, "absolute_log_error": loss}
                                    for age in CONFIG["ages"])
        return pd.DataFrame(rows)

    def test_selection_uses_completed_blocks_and_shared_equal_sex_base(self):
        scores = self.selection_fixture()
        chosen = select_tcn_settings(scores, CONFIG, 2009)
        self.assertEqual(chosen["base"], base_grid(CONFIG)[1])
        self.assertEqual(chosen["penalties"], {"Male": .1, "Female": 1})
        self.assertEqual(chosen["inner_origins"], [2003, 2004])
        self.assertEqual(chosen["last_inner_label_year"], 2009)
        self.assertEqual(chosen["loss"], .5)
        scores.loc[scores.origin.gt(2004), "absolute_log_error"] = np.nan
        self.assertEqual(select_tcn_settings(scores, CONFIG, 2009), chosen)

    def test_selection_rejects_incomplete_grid_and_resolves_ties(self):
        scores = self.selection_fixture()
        scores["absolute_log_error"] = 1.
        chosen = select_tcn_settings(scores, CONFIG, 2009)
        self.assertEqual(chosen["base"], base_grid(CONFIG)[0])
        self.assertEqual(chosen["penalties"], {"Male": .01, "Female": .01})
        with self.assertRaisesRegex(ValueError, "Incomplete chronological"):
            select_tcn_settings(scores.drop(index=0), CONFIG, 2009)
        cold = select_tcn_settings(pd.DataFrame(), CONFIG, 2007)
        self.assertEqual(cold["status"], "cold_start_defaults")
        self.assertEqual(cold["inner_origins"], [])
        self.assertEqual(cold["base"], base_grid(CONFIG)[0])
        self.assertEqual(cold["penalties"], {"Male": 1, "Female": 1})


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(TCNTests))
    files = [ROOT / f for f in ["src/gbd_park/tcn.py", "src/gbd_park/tcn_forecasting.py", "tests/test_tcn.py",
                               "src/gbd_park/pooled.py", "src/gbd_park/adaptation.py",
                               "src/gbd_park/scoring.py", "study_design/tcn_implementation.md"]]
    runner = ROOT / "scripts/run_tcn.py"
    if runner.exists():
        files.append(runner)
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "device": "cpu", "test_only_epochs": 2,
              "tested_code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in files}}
    output = ROOT / "work/tcn-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
