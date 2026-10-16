"""Synthetic integrity and real tiny-fit checks for supporting outcomes."""

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import io
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
import run_supporting as runner
from gbd_park import supporting
from gbd_park.local import settings_grid as local_grid
from gbd_park.pooled import build_examples, country_pool, base_grid as pooled_grid, fit_base, predict_changes
from gbd_park.tcn import base_grid as tcn_grid
from gbd_park.tcn_forecasting import make_forecasts
from gbd_park.secondary import job_fingerprint, score_actual, actual_population_weights, baseline_choices
from test_primary import baseline_scores, population_fixture, scored_fixture

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
OUTCOMES = ("deaths", "ylds", "ylls", "dalys")


def synthetic_panel():
    """Full old/future/other-outcome distractors; only generated values are used."""
    rows = []
    for ci, country in enumerate(CONFIG["countries"]):
        for oi, outcome in enumerate(("prevalence", "incidence") + OUTCOMES):
            for year in range(1980, 2024):
                for si, sex in enumerate(CONFIG["sexes"]):
                    for ai, age in enumerate(CONFIG["ages"]):
                        rate = np.exp(2 + .2*oi + .05*ci + .1*ai + .07*si
                                      + (.004+.001*oi)*(year-1990) + .02*np.sin(year+ai+ci))
                        population = (1000+100*ci)*(ai+1)*(1+.006*(year-1990))
                        rows.append(dict(location_name=country["name"], outcome=outcome,
                                         year=year, sex=sex, age=age, rate=rate,
                                         count=rate*population/100000, implied_population=population))
    return pd.DataFrame(rows)


def irrelevant_perturbation(panel, outcome, origin):
    altered = panel.copy(deep=True)
    irrelevant = panel.outcome.ne(outcome) | panel.year.lt(1990) | panel.year.gt(origin)
    altered.loc[irrelevant, ["rate", "count", "implied_population"]] = np.nan
    return altered


class SupportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()

    def test_exact_twenty_four_tasks_exclude_primary_outcomes_and_jordan(self):
        tasks = supporting.supporting_tasks(CONFIG)
        expected = {(country["name"], outcome) for country in CONFIG["countries"]
                    if country["gcc"] for outcome in OUTCOMES}
        self.assertEqual(len(tasks), 24)
        self.assertEqual({(task["target"], task["outcome"]) for task in tasks}, expected)
        self.assertEqual(len({task["id"] for task in tasks}), 24)
        self.assertNotIn("Jordan", {task["target"] for task in tasks})
        self.assertFalse({"SAU_prevalence", "SAU_incidence"} & {task["id"] for task in tasks})

    def test_all_actual_outcomes_and_history_bounds_are_selected_before_alias(self):
        untouched = self.panel.copy(deep=True)
        config_before = deepcopy(CONFIG)
        for outcome in OUTCOMES:
            with self.subTest(outcome=outcome):
                work, cfg = supporting.working_context(self.panel, CONFIG, "Oman", outcome, origin=2003)
                expected = self.panel[self.panel.outcome.eq(outcome) & self.panel.year.between(1990, 2003)]
                self.assertTrue(work.outcome.eq("prevalence").all())
                self.assertTrue(work.source_outcome.eq(outcome).all())
                self.assertEqual((work.year.min(), work.year.max()), (1990, 2003))
                self.assertEqual(cfg["primary_target"], "Oman")
                np.testing.assert_array_equal(work.rate, expected.rate)
                changed, _ = supporting.working_context(
                    irrelevant_perturbation(self.panel, outcome, 2003), CONFIG, "Oman", outcome, origin=2003)
                pd.testing.assert_frame_equal(work, changed)
        pd.testing.assert_frame_equal(self.panel, untouched)
        self.assertEqual(CONFIG, config_before)

    def test_invalid_actual_outcome_and_empty_panel_fail(self):
        for outcome in ["prevalence", "incidence", "Deaths", "missing"]:
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                supporting.working_context(self.panel, CONFIG, "Oman", outcome, origin=2003)
        with self.assertRaises(ValueError):
            supporting.working_context(self.panel[self.panel.outcome.ne("deaths")], CONFIG,
                                       "Oman", "deaths", origin=2003)

    def test_donor_subset_excludes_target_and_training_labels_are_mature(self):
        donors = ["Saudi Arabia", "Jordan", "Bahrain"]
        work, cfg = supporting.working_context(self.panel, CONFIG, "Qatar", "ylls", donors, 2003)
        countries = country_pool(cfg, "Qatar", "donor")
        self.assertEqual(set(countries), set(donors))
        x, y, meta = build_examples(work, cfg, 2003, countries)
        self.assertEqual(set(meta.sex), set(CONFIG["sexes"]))
        self.assertEqual(set(meta.age), set(CONFIG["ages"]))
        self.assertNotIn("Qatar", set(meta.country))
        self.assertTrue(meta.label_end.le(2003).all())
        self.assertTrue(meta.input_end.le(1998).all())
        self.assertEqual(x.shape[1], 21)
        self.assertEqual(y.shape[1], 5)

    def test_actual_ridge_fit_is_invariant_to_all_irrelevant_rows(self):
        before, cfg = supporting.working_context(self.panel, CONFIG, "Oman", "ylls", origin=2003)
        after, _ = supporting.working_context(irrelevant_perturbation(self.panel, "ylls", 2003),
                                              CONFIG, "Oman", "ylls", origin=2003)
        countries = country_pool(cfg, "Oman", "donor")
        x, y, meta = build_examples(before, cfg, 2003, countries)
        xx, yy, mm = build_examples(after, cfg, 2003, countries)
        base = pooled_grid(cfg)[0]
        fitted = fit_base(x, y, meta, base, cfg)
        perturbed = fit_base(xx, yy, mm, base, cfg)
        np.testing.assert_array_equal(predict_changes(fitted, x), predict_changes(perturbed, xx))
        self.assertTrue(np.isfinite(predict_changes(fitted, x)).all())

    def test_four_actual_persistence_jobs_restore_estimand_and_score_correct_truth(self):
        spec = next(s for s in local_grid(CONFIG) if s["family"] == "persistence")
        for outcome in OUTCOMES:
            task = dict(id="OMN_"+outcome, target="Oman", outcome=outcome)
            job = runner.job_spec(task, "local", 2003, sex="Male", specs=[spec])
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                with patch.object(runner, "source_panel", return_value=self.panel):
                    runner.fit_job(CONFIG, job, out)
                result = runner.load_result(out, job)
                frame = pd.DataFrame(result["forecasts"])
                self.assertEqual(len(frame), 55)
                self.assertTrue(frame.target.eq("Oman").all())
                self.assertTrue(frame.outcome.eq(outcome).all())
                source = self.panel[self.panel.location_name.eq("Oman") & self.panel.outcome.eq(outcome)
                                    & self.panel.sex.eq("Male") & self.panel.year.eq(2003)].set_index("age").rate
                np.testing.assert_allclose(frame.prediction, frame.age.map(source), rtol=1e-14)
                cfg = supporting.context_config(CONFIG, "Oman")
                scored = score_actual(frame, self.panel, cfg, "Oman", outcome, 2008)
                self.assertTrue(scored.outcome.eq(outcome).all())
                expected = self.panel[self.panel.location_name.eq("Oman") & self.panel.outcome.eq(outcome)
                                      & self.panel.sex.eq("Male")].set_index(["age", "year"]).rate
                observed = np.array([expected.loc[(row.age, row.forecast_year)] for row in scored.itertuples()])
                np.testing.assert_allclose(scored.observed_rate, observed)
                np.testing.assert_allclose(scored.absolute_log_error, np.abs(np.log(scored.prediction/observed)), atol=1e-14)
                with self.assertRaises(ValueError):
                    score_actual(frame, self.panel, cfg, "Oman", "prevalence", 2008)

    def test_nonneural_runner_checkpoint_and_adaptation_use_actual_outcome(self):
        task = dict(id="BHR_dalys", target="Bahrain", outcome="dalys")
        job = runner.job_spec(task, "nonneural", 2003, selected_bases=[pooled_grid(CONFIG)[0]])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(runner, "source_panel", return_value=self.panel):
                runner.fit_job(CONFIG, job, out)
            result = runner.load_result(out, job)
            forecasts = pd.DataFrame(result["forecasts"])
            self.assertEqual(len(forecasts), 660)
            self.assertTrue(forecasts.outcome.eq("dalys").all())
            self.assertTrue(forecasts.target.eq("Bahrain").all())
            self.assertTrue(forecasts.status.eq("ok").all())
            self.assertEqual(len(list((runner.result_path(out, job).parent / "checkpoints").glob("*.joblib"))), 2)
            for fitted in result["fits"]:
                self.assertGreaterEqual(fitted["minimum_input_year"], 1990)
                self.assertLessEqual(fitted["maximum_label_year"], 2003)
                self.assertEqual(fitted["base_fingerprint_before"], fitted["base_fingerprint_after"])
                self.assertEqual("Bahrain" in fitted["countries"], fitted["pool"] == "pooled")
            self.assertTrue(all(row["last_target_label_year"] <= 2003 for row in result["calibrations"]))

    def test_actual_tiny_tcn_fit_checkpoint_future_invariance_and_frozen_adaptation(self):
        task = dict(id="SAU_ylds", target="Saudi Arabia", outcome="ylds")
        base = {**tcn_grid(CONFIG)[0], "epochs": 2}
        job = runner.job_spec(task, "tcn", 2003, base=base, seed=11, device="cpu")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            payloads = []
            for directory, panel in [(first, self.panel), (second, irrelevant_perturbation(self.panel, "ylds", 2003))]:
                out = Path(directory)
                with patch.object(runner, "source_panel", return_value=panel):
                    runner.fit_job(CONFIG, job, out)
                payloads.append(runner.load_result(out, job))
                checkpoint = runner.result_path(out, job).parent / "checkpoint.joblib"
                self.assertTrue(checkpoint.is_file())
                audit = payloads[-1]["audit"]
                self.assertEqual(audit["status"], "ok")
                self.assertEqual(audit["actual_outcome"], "ylds")
                self.assertLessEqual(audit["maximum_label_year"], 2003)
                self.assertNotIn("Saudi Arabia", audit["countries"])
                self.assertEqual(audit["fingerprint_before"], audit["fingerprint_after"])
                with patch.object(runner, "fit_seed", side_effect=AssertionError("Unexpected refit")):
                    runner.fit_job(CONFIG, job, out)
            self.assertEqual(payloads[0]["audit"]["fingerprint_before"], payloads[1]["audit"]["fingerprint_before"])
            for key in ["current_changes", "target_changes", "target_y", "levels"]:
                np.testing.assert_array_equal(payloads[0][key], payloads[1][key])
            rows, adaptations = make_forecasts([payloads[0]], CONFIG, [1], include_intercept=True)
            public = supporting.restore_outcome(pd.DataFrame(rows), "ylds")
            self.assertTrue(public.outcome.eq("ylds").all())
            self.assertEqual(len(public), 330)
            self.assertTrue(all(row["last_target_label_year"] <= 2003 for row in adaptations))
            self.assertEqual(payloads[0]["audit"]["fingerprint_before"], payloads[0]["audit"]["fingerprint_after"])

    def test_supporting_evaluation_role_cannot_replace_primary(self):
        cfg = supporting.context_config(CONFIG, "Oman")
        panel = population_fixture().replace({"location_name": {"Saudi Arabia": "Oman"}})
        for outcome in OUTCOMES:
            weights = actual_population_weights(panel, cfg, cfg["calendar"]["reliability_origins"], outcome)
            scored = scored_fixture().replace({"target": {"Saudi Arabia": "Oman"}, "outcome": {"prevalence": outcome}})
            tables, verdict = supporting.actual_evaluation(scored, weights, cfg, outcome)
            self.assertEqual(verdict["target"], "Oman")
            self.assertEqual(verdict["outcome"], outcome)
            self.assertIn("supporting", verdict["endpoint_role"])
            self.assertFalse(verdict["saudi_prevalence_primary_result_replaced"])
            self.assertNotIn("joint_success", verdict)
            self.assertIn("all_four_endpoint_comparisons_favor_tcn", verdict)
            self.assertFalse(verdict["multiplicity_adjusted_confirmatory_claim"])
            for frame in tables.values():
                if "outcome" in frame:
                    self.assertTrue(frame.outcome.eq(outcome).all())

    def test_selection_uses_only_completed_five_year_blocks(self):
        scores, expected = baseline_scores("local")
        choices = baseline_choices(scores, CONFIG, [2014], "local")
        for row in choices.itertuples():
            self.assertEqual(row.setting_id, expected[(row.sex, row.family)])
            self.assertEqual(row.last_inner_label_year, 2014)
        changed = scores.copy()
        changed.loc[changed.origin.gt(2009), "absolute_log_error"] = np.nan
        pd.testing.assert_frame_equal(choices, baseline_choices(changed, CONFIG, [2014], "local"))

    def test_job_grid_identity_and_cache_corruption_guards(self):
        task = supporting.supporting_tasks(CONFIG)[0]
        jobs = runner.candidate_jobs(task, CONFIG, "cpu")
        self.assertEqual(len(jobs), 121)
        self.assertEqual(sum(job["kind"] == "tcn" for job in jobs), 88)
        self.assertEqual(len({job_fingerprint(job) for job in jobs}), 121)
        neural = jobs[-1]
        for changed in [dict(outcome="ylds"), dict(target="Oman"), dict(device="cuda:0"), dict(seed=23)]:
            if any(neural.get(key) != value for key, value in changed.items()):
                self.assertNotEqual(job_fingerprint(neural), job_fingerprint({**neural, **changed}))
        spec = next(s for s in local_grid(CONFIG) if s["family"] == "persistence")
        job = runner.job_spec(task, "local", 2003, sex="Male", specs=[spec])
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with patch.object(runner, "source_panel", return_value=self.panel):
                path = runner.fit_job(CONFIG, job, out)
            with patch.object(runner, "forecast_setting", side_effect=AssertionError("Unexpected refit")):
                self.assertEqual(runner.fit_job(CONFIG, job, out), path)
            modified_config = deepcopy(CONFIG)
            modified_config["version"] = "changed"
            with self.assertRaises(ValueError):
                runner.fit_job(modified_config, job, out)
            Path(path).write_bytes(b"corrupted synthetic checkpoint")
            with self.assertRaises(ValueError):
                runner.fit_job(CONFIG, job, out)

    def test_global_commit_gate_requires_all_cases_and_checks_nested_ledgers(self):
        tasks = supporting.supporting_tasks(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(ValueError):
                runner.score_trial(tasks[0], CONFIG, out)
            markers, ledgers = [], []
            for task in tasks:
                directory = out / "trials" / task["id"]
                directory.mkdir(parents=True)
                ledger = directory / "predictions.csv"
                ledger.write_text("synthetic issued ledger\n")
                runner.commit_phase(directory, "issued_commit", [ledger])
                markers.append(directory / "issued_commit.json")
                ledgers.append(ledger)
            # Local issuance alone never authorizes final scoring.
            with self.assertRaises(ValueError):
                runner.verify_global_commit(out, CONFIG)
            runner.commit_phase(out, "global_issued_commit", markers)
            record = runner.verify_global_commit(out, CONFIG)
            self.assertEqual(len(record["artifact_sha256"]), 24)
            ledgers[-1].write_text("changed after issuance\n")
            with self.assertRaises(ValueError):
                runner.verify_global_commit(out, CONFIG)

    def test_global_commit_rejects_complete_but_incomplete_twenty_three_case_set(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            markers = []
            for task in supporting.supporting_tasks(CONFIG)[:-1]:
                directory = out / "trials" / task["id"]
                directory.mkdir(parents=True)
                ledger = directory / "predictions.csv"
                ledger.write_text("synthetic\n")
                runner.commit_phase(directory, "issued_commit", [ledger])
                markers.append(directory / "issued_commit.json")
            runner.commit_phase(out, "global_issued_commit", markers)
            with self.assertRaisesRegex(ValueError, "exactly all 24"):
                runner.verify_global_commit(out, CONFIG)

    def test_partial_issuance_cleanup_preserves_committed_choices_and_completed_outputs(self):
        task = supporting.supporting_tasks(CONFIG)[0]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            directory = out / "trials" / task["id"]
            directory.mkdir(parents=True)
            partial_names = ["tcn_adaptation_audit.jsonl", "seed_references.jsonl", "local_fit_audit.jsonl",
                             "nonneural_fit_audit.jsonl", "nonneural_adaptation_audit.jsonl"]
            for name in partial_names:
                (directory/name).write_text("interrupted write\n")
            with self.assertRaises(ValueError):
                runner.assemble_issued(task, CONFIG, out, "cpu")
            self.assertTrue(all((directory/name).exists() for name in partial_names))
            choices = directory / "tcn_choices.json"
            choices.write_text("[]\n")
            runner.commit_phase(directory, "choices_frozen", [choices])
            def checked_assembly(*args):
                self.assertFalse(any((directory/name).exists() for name in partial_names))
                self.assertTrue(runner.phase_complete(directory, "choices_frozen"))
            with patch.object(runner, "frozen_assemble_issued", side_effect=checked_assembly) as assembly:
                runner.assemble_issued(task, CONFIG, out, "cpu")
                assembly.assert_called_once()
            # A completed phase exits before touching any of its audited files.
            issued = directory / partial_names[0]
            issued.write_text("completed audited output\n")
            runner.commit_phase(directory, "issued_commit", [issued])
            with patch.object(runner, "frozen_assemble_issued", side_effect=AssertionError("Unexpected reassembly")):
                runner.assemble_issued(task, CONFIG, out, "cpu")
            self.assertEqual(issued.read_text(), "completed audited output\n")

    def test_parallel_case_phase_returns_after_all_results_and_propagates_failure(self):
        tasks = supporting.supporting_tasks(CONFIG)
        completed = []
        def worker(task, config, out, device):
            self.assertEqual(device, "cpu")
            completed.append(task["id"])
        def threads(max_workers, mp_context):
            self.assertEqual(max_workers, 2)
            self.assertEqual(mp_context.get_start_method(), "spawn")
            return ThreadPoolExecutor(max_workers=max_workers)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            with patch.object(runner, "ProcessPoolExecutor", side_effect=threads), \
                    patch.object(runner, "prepare_choices", side_effect=worker):
                runner.run_trial_phase(tasks, CONFIG, Path(temp), "cpu", 2, "choices")
            self.assertEqual(set(completed), {task["id"] for task in tasks})
            self.assertEqual(len(completed), 24)
            with patch.object(runner, "ProcessPoolExecutor", side_effect=threads), \
                    patch.object(runner, "score_trial", side_effect=RuntimeError("synthetic worker failed")):
                with self.assertRaisesRegex(RuntimeError, "synthetic worker failed"):
                    runner.run_trial_phase(tasks, CONFIG, Path(temp), "cpu", 2, "scoring")
            with self.assertRaises(ValueError):
                runner.run_trial_phase(tasks, CONFIG, Path(temp), "cpu", 2, "invalid")

    def test_resume_identity_checks_code_and_test_report_before_any_fit(self):
        class StopBeforeFit(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config_path = root / "study_design/locked_v1/design.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps(CONFIG))
            source = root / "data/processed/design_v1/regional_outcomes.csv"
            source.parent.mkdir(parents=True)
            source.write_text("synthetic never read\n")
            prior = root / "results/primary_v1/run_manifest.json"
            prior.parent.mkdir(parents=True)
            prior.write_text(json.dumps({"versions": {}}))
            code = root / "tested_code.py"
            code.write_text("# synthetic version 1\n")
            test_report = root / "work/supporting-validation/tests.json"
            test_report.parent.mkdir(parents=True)
            def write_passed_tests():
                test_report.write_text(json.dumps(dict(passed=True,
                    tested_code_sha256={"tested_code.py": hashlib.sha256(code.read_bytes()).hexdigest()})))
            write_passed_tests()
            args = ["run_supporting.py", "--output", "results/synthetic", "--workers", "1", "--neural-workers", "1"]
            with patch.object(runner, "ROOT", root), patch.object(runner, "REQUIRED", ["tested_code.py"]), \
                    patch.object(runner, "check_lock"), patch.object(runner, "verify_prior", return_value=({}, {})), \
                    patch.object(runner, "run_queue", side_effect=StopBeforeFit):
                with patch.object(sys, "argv", args), self.assertRaises(StopBeforeFit):
                    runner.main()
                manifest = json.loads((root / "results/synthetic/run_manifest.json").read_text())
                self.assertEqual(manifest["identity"]["code_sha256"]["tested_code.py"], hashlib.sha256(code.read_bytes()).hexdigest())
                with patch.object(sys, "argv", args+["--resume"]), self.assertRaises(StopBeforeFit):
                    runner.main()
                code.write_text("# synthetic version 2\n")
                with patch.object(sys, "argv", args+["--resume"]), self.assertRaisesRegex(ValueError, "Missing or changed artifact"):
                    runner.main()
                write_passed_tests()
                with patch.object(sys, "argv", args+["--resume"]), self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
                    runner.main()


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SupportingTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = ["src/gbd_park/supporting.py", "scripts/run_supporting.py", "tests/test_supporting.py",
             "study_design/supporting_outcomes_implementation.md"]
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), device="cpu",
                  elapsed_seconds=time.perf_counter()-started,
                  fixture_type="synthetic_generated_values_no_production_outcome_scores_read",
                  actual_tiny_fits=["persistence_all_four_outcomes", "ridge", "pooled_and_donor_ridge_runner", "tcn_two_epochs"],
                  tested_code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                      for name in files})
    output = ROOT / "work/supporting-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.json").write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
