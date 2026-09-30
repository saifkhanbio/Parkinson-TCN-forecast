"""Synthetic checks for matched-device mortality history sensitivity."""

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import pandas as pd
import run_mortality_history_sensitivity as runner
from gbd_park.secondary import job_fingerprint, restore_outcome
from gbd_park.tcn_forecasting import make_forecasts
from test_supporting import CONFIG, synthetic_panel


class MortalityHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()

    def test_fixed_twenty_four_arms_and_one_hundred_sixty_eight_jobs(self):
        tasks = runner.history_tasks(CONFIG)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len({task["id"] for task in tasks}), 24)
        self.assertEqual({task["outcome"] for task in tasks}, {"deaths", "ylls"})
        self.assertEqual({task["history_start"] for task in tasks}, {1980, 1990})
        self.assertNotIn("Jordan", {task["target"] for task in tasks})
        jobs = runner.job_specs(CONFIG, "cuda:0")
        self.assertEqual(len(jobs), 168)
        neural = [job for job in jobs if job["kind"] == "tcn"]
        local = [job for job in jobs if job["kind"] == "local"]
        self.assertEqual(len(neural), 120)
        self.assertEqual(len(local), 48)
        self.assertEqual(len({job_fingerprint(job) for job in jobs}), 168)
        self.assertTrue(all(job["origin"] == 2018 for job in jobs))
        self.assertTrue(all(job["base"] == dict(channels=16, weight_decay=.001, epochs=50) for job in neural))
        self.assertEqual({job["device"] for job in neural}, {"cuda:0"})
        self.assertEqual({job["device"] for job in local}, {"cpu"})
        for task in tasks:
            self.assertEqual([job["seed"] for job in neural if job["trial_id"] == task["id"]], [11, 23, 37, 53, 71])

    def test_actual_outcome_history_restriction_and_no_config_mutation(self):
        original = deepcopy(CONFIG)
        for start in [1980, 1990]:
            work, cfg = runner.history_context(self.panel, CONFIG, "Qatar", "deaths", start)
            selected = self.panel[self.panel.outcome.eq("deaths") & self.panel.year.between(start, 2018)]
            np.testing.assert_array_equal(work.rate, selected.rate)
            self.assertTrue(work.outcome.eq("prevalence").all())
            self.assertTrue(work.source_outcome.eq("deaths").all())
            self.assertEqual((work.year.min(), work.year.max()), (start, 2018))
            self.assertEqual(cfg["calendar"]["history_start"], start)
            self.assertEqual(cfg["primary_target"], "Qatar")
            self.assertEqual(work.groupby("location_name").year.nunique().unique().tolist(), [2018-start+1])
        self.assertEqual(CONFIG, original)

    def test_history_context_rejects_incomplete_nonpositive_and_wrong_outcomes(self):
        for outcome, start in [("ylds", 1980), ("prevalence", 1990), ("deaths", 1985)]:
            with self.subTest(outcome=outcome, start=start), self.assertRaises(ValueError):
                runner.history_context(self.panel, CONFIG, "Saudi Arabia", outcome, start)
        selected = self.panel.outcome.eq("deaths") & self.panel.location_name.eq("Jordan") & self.panel.year.eq(1980)
        missing = self.panel.drop(self.panel[selected].index[0])
        with self.assertRaises(ValueError):
            runner.history_context(missing, CONFIG, "Saudi Arabia", "deaths", 1980)
        # An unavailable earlier row does not invalidate the shorter arm.
        runner.history_context(missing, CONFIG, "Saudi Arabia", "deaths", 1990)
        bad = self.panel.copy()
        row = bad.index[bad.outcome.eq("ylls") & bad.year.eq(2000)][0]
        bad.loc[row, "rate"] = 0.
        with self.assertRaises(ValueError):
            runner.history_context(bad, CONFIG, "Saudi Arabia", "ylls", 1990)
        duplicate = pd.concat([self.panel, self.panel.loc[[row]]], ignore_index=True)
        with self.assertRaises(ValueError):
            runner.history_context(duplicate, CONFIG, "Saudi Arabia", "ylls", 1990)

    def test_two_actual_seed_fits_per_arm_and_irrelevant_data_invariance(self):
        jobs = runner.job_specs(CONFIG, "cpu")
        windows = {}
        for start in [1980, 1990]:
            chosen = [job for job in jobs if job["target"] == "Saudi Arabia" and job["outcome"] == "deaths"
                      and job["history_start"] == start and job["kind"] == "tcn" and job["seed"] in [11, 23]]
            with self.subTest(start=start), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                payloads = []
                for original in chosen:
                    job = {**original, "base": {**original["base"], "epochs": 2}}
                    with patch.object(runner, "source_panel", return_value=self.panel):
                        runner.fit_history_job(CONFIG, job, out)
                    result = runner.load_history_result(out, job)
                    payloads.append(result)
                    audit = result["audit"]
                    self.assertEqual(audit["status"], "ok")
                    self.assertEqual(audit["history_start"], start)
                    self.assertEqual(audit["target_and_donor_history_years"], 2018-start+1)
                    self.assertEqual(audit["maximum_label_year"], 2018)
                    self.assertNotIn("Saudi Arabia", audit["countries"])
                    self.assertEqual(audit["fingerprint_before"], audit["fingerprint_after"])
                    self.assertTrue((runner.result_path(out, job).parent / "checkpoint.joblib").is_file())
                windows[start] = payloads[0]["audit"]["training_windows"]
                self.assertNotEqual(payloads[0]["audit"]["fingerprint_before"], payloads[1]["audit"]["fingerprint_before"])
                _, cfg = runner.history_context(self.panel, CONFIG, "Saudi Arabia", "deaths", start)
                rows, corrections = make_forecasts(payloads, cfg, [1])
                forecasts = restore_outcome(pd.DataFrame(rows), "deaths")
                self.assertEqual(len(forecasts), 220)
                self.assertEqual(set(forecasts.family), {"tcn_adapted", "tcn_unadapted"})
                self.assertTrue(forecasts.outcome.eq("deaths").all())
                self.assertTrue(all(row["last_target_label_year"] <= 2018 for row in corrections))
                altered = self.panel.copy()
                irrelevant = altered.outcome.ne("deaths") | ~altered.year.between(start, 2018)
                altered.loc[irrelevant, "rate"] = np.nan
                changed_out = out / "perturbed"
                original = chosen[0]
                job = {**original, "base": {**original["base"], "epochs": 2}}
                with patch.object(runner, "source_panel", return_value=altered):
                    runner.fit_history_job(CONFIG, job, changed_out)
                changed = runner.load_history_result(changed_out, job)
                self.assertEqual(changed["audit"]["fingerprint_before"], payloads[0]["audit"]["fingerprint_before"])
                np.testing.assert_array_equal(changed["current_changes"], payloads[0]["current_changes"])
                np.testing.assert_array_equal(changed["target_y"], payloads[0]["target_y"])
        self.assertGreater(windows[1980], windows[1990])

    def test_actual_ets_both_arms_cache_and_artifact_guards(self):
        for start in [1980, 1990]:
            job = next(job for job in runner.job_specs(CONFIG, "cpu") if job["target"] == "Oman"
                       and job["outcome"] == "ylls" and job["history_start"] == start
                       and job["kind"] == "local" and job["sex"] == "Male")
            with self.subTest(start=start), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                with patch.object(runner, "source_panel", return_value=self.panel):
                    path = runner.fit_history_job(CONFIG, job, out)
                result = runner.load_history_result(out, job)
                frame = pd.DataFrame(result["forecasts"])
                self.assertEqual(len(frame), 55)
                self.assertTrue(frame.outcome.eq("ylls").all())
                self.assertTrue(frame.family.eq("damped_ets").all())
                self.assertTrue(frame.forecast_year.between(2019, 2023).all())
                with patch.object(runner, "forecast_setting", side_effect=AssertionError("Unexpected refit")):
                    self.assertEqual(runner.fit_history_job(CONFIG, job, out), path)
                changed = deepcopy(CONFIG)
                changed["version"] = "changed"
                with self.assertRaises(ValueError):
                    runner.fit_history_job(changed, job, out)
                Path(path).write_bytes(b"corrupt synthetic result")
                with self.assertRaises(ValueError):
                    runner.load_history_result(out, job)

    def test_forecast_commit_required_and_full_arm_grid_checked_before_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(ValueError):
                runner.score_all(CONFIG, out)
            frame = pd.DataFrame([dict(trial_id=runner.history_tasks(CONFIG)[0]["id"])])
            path = out / "predictions.csv"
            frame.to_csv(path, index=False)
            runner.commit_phase(out, "issued_commit", [path])
            with self.assertRaisesRegex(ValueError, "all 24 complete"):
                runner.score_all(CONFIG, out)

    def test_all_arm_scoring_and_paired_history_comparison_have_known_errors(self):
        rows = []
        indexed = self.panel.set_index(["location_name", "outcome", "year", "sex", "age"]).rate
        for task in runner.history_tasks(CONFIG):
            error = .03 if task["history_start"] == 1980 else .1
            for family in ["tcn_adapted", "tcn_unadapted", "damped_ets"]:
                for sex in CONFIG["sexes"]:
                    for age in CONFIG["ages"]:
                        for horizon in range(1, 6):
                            observed = indexed.loc[(task["target"], task["outcome"], 2018+horizon, sex, age)]
                            prediction = observed*np.exp(error)
                            rows.append(dict(trial_id=task["id"], target=task["target"], outcome=task["outcome"],
                                             history_start=task["history_start"], origin=2018, family=family,
                                             sex=sex, age=age, horizon=horizon, forecast_year=2018+horizon,
                                             prediction=prediction, log_prediction=np.log(prediction),
                                             setting_id=family+"__synthetic", status="ok", fallback_reason=""))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            points = out / "predictions.csv"
            pd.DataFrame(rows).to_csv(points, index=False)
            runner.commit_phase(out, "issued_commit", [points])
            with patch.object(runner, "source_panel", return_value=self.panel):
                runner.score_all(CONFIG, out)
            summary = pd.read_csv(out / "summary.csv")
            compared = pd.read_csv(out / "history_comparisons.csv")
            self.assertEqual(len(summary), 1440)
            self.assertEqual(len(compared), 720)
            np.testing.assert_allclose(summary.loc[summary.history_start.eq(1980), "mean_absolute_log_error"], .03, atol=1e-14)
            np.testing.assert_allclose(summary.loc[summary.history_start.eq(1990), "mean_absolute_log_error"], .1, atol=1e-14)
            np.testing.assert_allclose(compared.relative_improvement_percent, 70., atol=1e-11)
            self.assertEqual(set(summary.loc[summary.age_group.eq("80+"), "age_cells"]), {4})
            self.assertEqual(set(summary.loc[summary.age_group.eq("45+"), "age_cells"]), {11})
            validation = json.loads((out / "validation_report.json").read_text())
            self.assertTrue(validation["all_arms_committed_before_scoring"])
            self.assertFalse(validation["intervals_fitted"])
            self.assertFalse(validation["hyperparameters_selected"])
            self.assertFalse(validation["primary_result_replaced"])
            self.assertFalse(validation["cross_device_accuracy_claim"])
            self.assertTrue(runner.phase_complete(out, "scoring_complete"))


if __name__ == "__main__":
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MortalityHistoryTests))
    required = runner.REQUIRED + ["tests/test_supporting.py"]
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), device="cpu",
                  elapsed_seconds=time.perf_counter()-start,
                  fixture_type="synthetic_no_production_forecast_scores_read",
                  actual_tiny_fits="two TCN seeds per history arm plus perturbation fits; actual damped ETS both arms",
                  tested_code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in required})
    path = ROOT / "work/supporting-validation/history_tests.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
