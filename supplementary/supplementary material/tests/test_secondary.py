"""Synthetic adapter, chronological selection, and resumable-job checks."""

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
sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import numpy as np
import pandas as pd
from gbd_park.secondary import (secondary_tasks, context_config, working_context, restore_outcome,
                               internal_outcome, actual_truth, score_actual, baseline_choices,
                               family_mappings, actual_population_weights, actual_evaluation, job_fingerprint)
from gbd_park.local import settings_grid
from gbd_park.pooled import build_examples, country_pool
from run_secondary import (job_spec, candidate_jobs, fit_job, load_result, result_path,
                           phase_complete, commit_phase, score_trial, selected_jobs)
from test_primary import baseline_scores, scored_fixture, population_fixture
CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def panel_fixture():
    rows = []
    for ci, country in enumerate(CONFIG["countries"]):
        for outcome, multiplier in [("prevalence", 1.), ("incidence", 7.)]:
            for year in range(1990, 2024):
                for sex in CONFIG["sexes"]:
                    for ai, age in enumerate(CONFIG["ages"]):
                        rate = multiplier*(10+ci+ai)*np.exp(.003*(year-1990))
                        population = 1000*(ai+1)
                        rows.append({"location_name": country["name"], "outcome": outcome, "year": year,
                                     "sex": sex, "age": age, "rate": rate, "count": rate*population/100000,
                                     "implied_population": population})
    return pd.DataFrame(rows)


class SecondaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = panel_fixture()

    def test_eleven_tasks_preserve_primary(self):
        tasks = secondary_tasks(CONFIG)
        self.assertEqual(len(tasks), 11)
        self.assertEqual(tasks[0]["id"], "SAU_incidence")
        self.assertNotIn("SAU_prevalence", [t["id"] for t in tasks])
        self.assertNotIn("Jordan", [t["target"] for t in tasks])
        self.assertEqual(len({t["id"] for t in tasks}), 11)

    def test_actual_outcome_cutoff_and_no_mutation(self):
        original = self.panel.copy(deep=True)
        cfg_original = deepcopy(CONFIG)
        work, cfg = working_context(self.panel, CONFIG, "Oman", "incidence", origin=2003)
        self.assertTrue(work.outcome.eq("prevalence").all())
        self.assertTrue(work.source_outcome.eq("incidence").all())
        self.assertEqual(work.year.max(), 2003)
        self.assertEqual(cfg["primary_target"], "Oman")
        np.testing.assert_array_equal(work.rate, self.panel.loc[self.panel.outcome.eq("incidence") & self.panel.year.le(2003), "rate"])
        pd.testing.assert_frame_equal(self.panel, original)
        self.assertEqual(CONFIG, cfg_original)

    def test_donor_subset_keeps_both_sexes_excludes_target_source(self):
        donors = ["Saudi Arabia", "Jordan", "Bahrain"]
        work, cfg = working_context(self.panel, CONFIG, "Qatar", "incidence", donors, 2003)
        countries = country_pool(cfg, "Qatar", "donor")
        self.assertEqual(set(countries), set(donors))
        x, y, meta = build_examples(work, cfg, 2003, countries)
        self.assertEqual(set(meta.sex), set(CONFIG["sexes"]))
        self.assertNotIn("Qatar", set(meta.country))
        self.assertTrue(meta.label_end.le(2003).all())
        self.assertEqual(x.shape[1], 21)
        self.assertEqual(y.shape[1], 5)
        with self.assertRaises(ValueError):
            context_config(CONFIG, "Qatar", ["Qatar"])
        with self.assertRaises(ValueError):
            context_config(CONFIG, "Qatar", ["Jordan", "Jordan"])

    def test_future_and_other_outcome_perturbation_invariance(self):
        a, cfg = working_context(self.panel, CONFIG, "Oman", "incidence", origin=2003)
        perturbed = self.panel.copy()
        perturbed.loc[perturbed.outcome.eq("prevalence") | perturbed.year.gt(2003), "rate"] = np.nan
        b, _ = working_context(perturbed, CONFIG, "Oman", "incidence", origin=2003)
        pd.testing.assert_frame_equal(a, b)
        countries = country_pool(cfg, "Oman", "donor")
        for left, right in zip(build_examples(a, cfg, 2003, countries)[:2], build_examples(b, cfg, 2003, countries)[:2]):
            np.testing.assert_array_equal(left, right)

    def test_restore_truth_and_actual_score(self):
        cfg = context_config(CONFIG, "Oman")
        truth = actual_truth(self.panel, "Oman", "incidence", 2008)
        rows = []
        for sex in cfg["sexes"]:
            for age in cfg["ages"]:
                for horizon in range(1, 6):
                    value = truth.loc[truth.sex.eq(sex) & truth.age.eq(age) & truth.forecast_year.eq(2003+horizon), "observed_rate"].iloc[0]
                    rows.append({"target": "Oman", "outcome": "prevalence", "origin": 2003, "sex": sex,
                                 "age": age, "horizon": horizon, "forecast_year": 2003+horizon,
                                 "family": "persistence", "setting_id": "synthetic", "prediction": value,
                                 "log_prediction": np.log(value)})
        predictions = restore_outcome(pd.DataFrame(rows), "incidence")
        score = score_actual(predictions, self.panel, cfg, "Oman", "incidence", 2008)
        self.assertTrue(score.absolute_log_error.eq(0).all())
        self.assertTrue(score.outcome.eq("incidence").all())
        with self.assertRaises(ValueError):
            score_actual(predictions, self.panel, cfg, "Oman", "prevalence", 2008)
        with self.assertRaises(ValueError):
            internal_outcome(predictions, "prevalence")

    def test_selection_uses_only_completed_blocks_and_rejects_nan(self):
        scores, expected = baseline_scores("local")
        choices = baseline_choices(scores, CONFIG, [2014], "local")
        for row in choices.itertuples():
            self.assertEqual(row.setting_id, expected[(row.sex, row.family)])
            self.assertEqual(row.last_inner_label_year, 2014)
        future = scores.copy()
        future.loc[future.origin.gt(2009), "absolute_log_error"] = np.nan
        pd.testing.assert_frame_equal(choices, baseline_choices(future, CONFIG, [2014], "local"))
        invalid = scores.copy()
        invalid.loc[invalid.index[0], "absolute_log_error"] = np.nan
        with self.assertRaises(ValueError):
            baseline_choices(invalid, CONFIG, [2014], "local")
        with self.assertRaises(ValueError):
            baseline_choices(scores.iloc[1:], CONFIG, [2014], "local")

    def test_population_and_evaluation_restore_actual_estimand(self):
        cfg = context_config(CONFIG, "Oman")
        panel = population_fixture().replace({"location_name": {"Saudi Arabia": "Oman"}})
        weights = actual_population_weights(panel, cfg, cfg["calendar"]["reliability_origins"], "incidence")
        self.assertTrue(weights.outcome.eq("incidence").all())
        scores = scored_fixture().replace({"target": {"Saudi Arabia": "Oman"}, "outcome": {"prevalence": "incidence"}})
        tables, verdict = actual_evaluation(scores, weights, cfg, "incidence")
        self.assertEqual(verdict["target"], "Oman")
        self.assertEqual(verdict["outcome"], "incidence")
        self.assertFalse(verdict["saudi_prevalence_primary_result_replaced"])
        self.assertEqual(len(tables["endpoint_contrasts"]), 4)
        for table in tables.values():
            if "outcome" in table:
                self.assertTrue(table.outcome.eq("incidence").all())

    def test_candidate_job_grid_and_identity(self):
        task = secondary_tasks(CONFIG)[0]
        jobs = candidate_jobs(task, CONFIG, "cpu")
        self.assertEqual(len(jobs), 121)
        self.assertEqual(sum(j["kind"] == "tcn" for j in jobs), 88)
        self.assertEqual(len({job_fingerprint(j) for j in jobs}), len(jobs))
        job = jobs[-1]
        for change in [{"outcome": "prevalence"}, {"target": "Oman"}, {"device": "cuda:0"}, {"seed": 23}]:
            self.assertNotEqual(job_fingerprint(job), job_fingerprint({**job, **change}))

    def test_family_mapping_complete_eligible_grid(self):
        rows = []
        for group in ["local", "nonneural"]:
            for fi, family in enumerate(CONFIG["models"][group+"_order"]):
                for origin in range(2009, 2014):
                    for sex in CONFIG["sexes"]:
                        for age in CONFIG["ages"]:
                            rows.append({"family": family, "origin": origin, "sex": sex, "age": age,
                                         "horizon": 5, "absolute_log_error": .1+fi*.01,
                                         "parameter_count": 1})
        scores = pd.DataFrame(rows)
        scores.loc[scores.origin.gt(2009), "absolute_log_error"] = np.nan
        mapped = family_mappings(scores, CONFIG, [2014])
        self.assertEqual(set(mapped.source_family), {"persistence", "pooled_ridge"})
        self.assertTrue(mapped.last_selection_target_year.eq(2014).all())
        with self.assertRaises(ValueError):
            family_mappings(scores.iloc[1:], CONFIG, [2014])
        invalid = scores.copy()
        invalid.loc[0, "age"] = "not_an_age"
        with self.assertRaises(ValueError):
            family_mappings(invalid, CONFIG, [2014])

    def test_persistence_job_cache_and_corruption_guard(self):
        task = {"id": "OMN_incidence", "target": "Oman", "outcome": "incidence"}
        spec = next(s for s in settings_grid(CONFIG) if s["family"] == "persistence")
        job = job_spec(task, "local", 2003, sex="Male", specs=[spec])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch("run_secondary.source_panel", return_value=self.panel):
                path = fit_job(CONFIG, job, out)
            result = load_result(out, job)
            self.assertEqual(len(result["forecasts"]), 55)
            self.assertTrue(all(r["outcome"] == "incidence" and r["target"] == "Oman" for r in result["forecasts"]))
            with patch("run_secondary.forecast_setting", side_effect=AssertionError("No refit allowed")):
                self.assertEqual(fit_job(CONFIG, job, out), path)
            changed = deepcopy(CONFIG)
            changed["version"] = "changed"
            with self.assertRaises(ValueError):
                fit_job(changed, job, out)
            Path(path).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                fit_job(CONFIG, job, out)

    def test_phase_and_scoring_require_committed_ledgers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root / "predictions.csv"
            table.write_text("synthetic\n")
            self.assertFalse(phase_complete(root, "issued"))
            commit_phase(root, "issued", [table])
            self.assertTrue(phase_complete(root, "issued"))
            with self.assertRaises(FileExistsError):
                commit_phase(root, "issued", [table])
            table.write_text("changed\n")
            with self.assertRaises(ValueError):
                phase_complete(root, "issued")
            task = secondary_tasks(CONFIG)[0]
            directory = root / "trials" / task["id"]
            directory.mkdir(parents=True)
            with self.assertRaises(ValueError):
                score_trial(task, CONFIG, root)
            with self.assertRaises(ValueError):
                selected_jobs(task, CONFIG, root, "cpu")


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SecondaryTests))
    files = ["src/gbd_park/secondary.py", "scripts/run_secondary.py", "tests/test_secondary.py",
             "study_design/secondary_implementation.md"]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "device": "cpu",
              "fixture_type": "synthetic_no_final_outcome_scores_read",
              "tested_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
    out = ROOT / "work/secondary-validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tests.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
