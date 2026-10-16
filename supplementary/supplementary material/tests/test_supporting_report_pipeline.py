"""Read-only primary-schema reporting and isolated follow-up supervision checks."""

import os
for variable in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[variable] = "1"

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import report_supporting as reporter
import finalize_supporting as pipeline

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
PRIMARY_FILES = ["results/primary_v1/"+name for name in ["point_scores.csv", "interval_scores.csv", "wis_scores.csv"]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def isolated_pipeline():
    with tempfile.TemporaryDirectory(prefix="supporting-pipeline-smoke-") as temp:
        root = Path(temp)
        work = root / "work/supporting-validation"
        work.mkdir(parents=True)
        protected = root / "frozen.py"
        protected.write_text("# synthetic protected input\n")
        state = dict(status="starting", steps={}, frozen_sha256={"frozen.py": digest(protected)})
        with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "WORK", work), \
                patch.object(pipeline, "STATE", work / "pipeline_status.json"), \
                patch.object(pipeline, "FROZEN", ["frozen.py"]):
            yield root, work, state


class ReportSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hashes = {name: digest(ROOT/name) for name in PRIMARY_FILES}
        frames = [reporter.read(ROOT/name) for name in PRIMARY_FILES]
        cls.points, cls.intervals, cls.wis = [frame.loc[frame.family.isin(reporter.ROLES)].copy() for frame in frames]
        for frame in [cls.points, cls.intervals, cls.wis]:
            frame["procedure"] = "direct"
        cls.point_summary, cls.interval_summary = reporter.summaries(cls.points, cls.intervals, cls.wis)

    @classmethod
    def tearDownClass(cls):
        if cls.source_hashes != {name: digest(ROOT/name) for name in PRIMARY_FILES}:
            raise AssertionError("A primary source ledger changed during reporting tests")

    def test_real_primary_schema_age_counts_errors_coverage_and_tail_accounting(self):
        self.assertEqual(len(self.point_summary), 300)
        self.assertEqual(len(self.interval_summary), 1800)
        for band, count, ages in [("45+", 11, CONFIG["ages"]), ("80+", 4, CONFIG["ages"][-4:])]:
            points = self.point_summary[self.point_summary.age_band.eq(band)]
            bounds = self.interval_summary[self.interval_summary.age_band.eq(band)]
            self.assertEqual(set(points.age_cells), {count})
            self.assertEqual(set(bounds.age_cells), {count})
            selected = (self.points.family.eq("tcn_adapted") & self.points.sex.eq("Male")
                        & self.points.origin.eq(2018) & self.points.horizon.eq(5) & self.points.age.isin(ages))
            expected = self.points.loc[selected, "absolute_log_error"].mean()
            row = points[points.family.eq("tcn_adapted") & points.sex.eq("Male")
                         & points.origin.eq(2018) & points.horizon.eq(5)].iloc[0]
            self.assertAlmostEqual(row.mean_ale, expected, places=14)
            interval = self.intervals[self.intervals.family.eq("tcn_adapted") & self.intervals.sex.eq("Male")
                                      & self.intervals.origin.eq(2018) & self.intervals.horizon.eq(5)
                                      & self.intervals.age.isin(ages) & self.intervals.scale.eq("rate")
                                      & self.intervals.level.eq(.8)]
            reported = bounds[bounds.family.eq("tcn_adapted") & bounds.sex.eq("Male") & bounds.origin.eq(2018)
                              & bounds.horizon.eq(5) & bounds.scale.eq("rate") & bounds.level.eq(.8)].iloc[0]
            self.assertAlmostEqual(reported.coverage, interval.covered.mean())
            self.assertAlmostEqual(reported.mean_width, interval.width.mean())
            wis = self.wis[self.wis.family.eq("tcn_adapted") & self.wis.sex.eq("Male") & self.wis.origin.eq(2018)
                           & self.wis.horizon.eq(5) & self.wis.age.isin(ages) & self.wis.scale.eq("rate")]
            self.assertAlmostEqual(reported.mean_wis_50_80, wis.wis_50_80.mean())
        np.testing.assert_allclose(self.interval_summary.coverage+self.interval_summary.below_lower
                                   +self.interval_summary.above_upper, 1.)

    def test_comparison_sign_favors_smaller_derived_error_and_zero_denominator_is_nan(self):
        direct = self.point_summary.assign(outcome="dalys")
        derived = direct.assign(procedure="component_sum", mean_ale=direct.mean_ale*.8)
        compared = reporter.comparison(pd.concat([direct, derived], ignore_index=True), "dalys", "component_sum")
        self.assertEqual(len(compared), len(direct))
        self.assertTrue(compared.derived_minus_direct_ale.lt(0).all())
        np.testing.assert_allclose(compared.derived_relative_improvement_percent, 20.)
        worse = derived.assign(mean_ale=direct.mean_ale*1.5)
        compared = reporter.comparison(pd.concat([direct, worse], ignore_index=True), "dalys", "component_sum")
        np.testing.assert_allclose(compared.derived_relative_improvement_percent, -50.)
        zero = pd.concat([direct.assign(mean_ale=0.), derived.assign(mean_ale=0.)], ignore_index=True)
        self.assertTrue(reporter.comparison(zero, "dalys", "component_sum").derived_relative_improvement_percent.isna().all())

    def test_report_figures_render_from_primary_schema_only_to_temporary_directory(self):
        point_frames = []
        for country in CONFIG["countries"]:
            if country["gcc"]:
                for outcome in ["deaths", "ylds", "ylls", "dalys"]:
                    point_frames.append(self.point_summary.assign(target=country["name"], outcome=outcome))
        points = pd.concat(point_frames, ignore_index=True)
        component = reporter.comparison(pd.concat([points, points[points.outcome.eq("dalys")].assign(
            procedure="component_sum", mean_ale=lambda frame: frame.mean_ale*.8)], ignore_index=True), "dalys", "component_sum")
        ratio = reporter.comparison(pd.concat([points, points[points.outcome.eq("ylds")].assign(
            procedure="prevalence_training_ratio", mean_ale=lambda frame: frame.mean_ale*1.2)], ignore_index=True),
            "ylds", "prevalence_training_ratio")
        with tempfile.TemporaryDirectory(prefix="supporting-figure-smoke-") as temp:
            out = Path(temp)
            reporter.figures(out, points, component, ratio)
            self.assertEqual(len(list(out.iterdir())), 4)
            for path in out.glob("*.png"):
                values = mpimg.imread(path)
                self.assertGreater(values.shape[0], 500)
                self.assertGreater(values.shape[1], 500)
                self.assertGreater(values.std(), .05)
            for path in out.glob("*.svg"):
                self.assertTrue(ET.parse(path).getroot().tag.endswith("svg"))
                self.assertGreater(path.stat().st_size, 10000)

    def test_report_verification_rejects_partial_or_duplicate_audit_scope(self):
        with tempfile.TemporaryDirectory(prefix="supporting-report-integrity-") as temp:
            root = Path(temp)
            config = root / "study_design/locked_v1/design.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps(CONFIG))
            run = root / "results/synthetic"
            run.mkdir(parents=True)
            output = run / "output.csv"
            output.write_text("synthetic audit-integrity fixture\n")
            code = root / "synthetic.py"
            code.write_text("# synthetic\n")
            manifest = run / "run_manifest.json"
            manifest.write_text(json.dumps(dict(status="complete", output_sha256={"output.csv": digest(output)},
                                                code_sha256={"synthetic.py": digest(code)})))
            work = root / "work/supporting-validation"
            work.mkdir(parents=True)
            countries = [country["iso3"] for country in CONFIG["countries"] if country["gcc"]]
            scopes = [
                ("audit_rates", "cases", "case", [f"{country}_{outcome}" for country in countries for outcome in ["deaths", "ylds", "ylls", "dalys"]]),
                ("audit_components", "countries", "country", countries),
                ("audit_history", "arms", "case", [f"{country}_{outcome}_history{start}" for country in countries
                                                      for outcome in ["deaths", "ylls"] for start in [1980, 1990]]),
            ]
            with patch.object(reporter, "ROOT", root):
                for name, field, key, identifiers in scopes:
                    audit_code = work / (name+".py")
                    audit_code.write_text("# synthetic scope fixture\n")
                    path = audit_code.with_suffix(".json")
                    complete = dict(passed=True, run_manifest_sha256=digest(manifest), audit_code_sha256=digest(audit_code))
                    complete[field] = [{key: identifier, "passed": True} for identifier in identifiers]
                    path.write_text(json.dumps(complete))
                    reporter.verify_run(run, name)
                    for changed_rows in [complete[field][:-1], complete[field][:-1]+[complete[field][0]]]:
                        partial = {**complete, field: changed_rows}
                        path.write_text(json.dumps(partial))
                        with self.assertRaises(AssertionError):
                            reporter.verify_run(run, name)
                # The CLI refuses partial first-stage evidence before creating its output directory.
                with patch.object(sys, "argv", ["report_supporting.py", "--core", "results/synthetic", "--output", "reports/test"]):
                    with self.assertRaises(AssertionError):
                        reporter.main()
                self.assertFalse((root / "reports/test").exists())


class SupervisorSmokeTests(unittest.TestCase):
    def test_changed_hash_fails_before_subprocess_or_state_mutation(self):
        with isolated_pipeline() as (root, work, state):
            (root / "frozen.py").write_text("changed\n")
            with patch.object(pipeline.subprocess, "run") as child:
                with self.assertRaisesRegex(ValueError, "Changed pipeline input"):
                    pipeline.step(state, "audit", ["synthetic.py"], "result.json")
                child.assert_not_called()
            self.assertFalse((work / "pipeline_status.json").exists())
            self.assertEqual(state["steps"], {})

    def test_nonzero_child_marks_supervisor_failed_and_stops_later_stages(self):
        with isolated_pipeline() as (root, work, state), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with patch.object(sys, "argv", ["finalize_supporting.py"]), patch.object(pipeline, "await_run"), \
                    patch.object(pipeline.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as child:
                with self.assertRaisesRegex(RuntimeError, "audit_history exited 7"):
                    pipeline.main()
            saved = json.loads((work / "pipeline_status.json").read_text())
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["current_stage"], "audit_history")
            self.assertEqual(set(saved["steps"]), {"audit_history"})
            self.assertNotEqual(saved["steps"]["audit_history"]["status"], "complete")
            self.assertEqual(child.call_count, 1)
            self.assertEqual(child.call_args.kwargs["cwd"], root)

    def test_successful_step_reuses_only_intact_witness_and_rejects_unrecorded_output(self):
        with isolated_pipeline() as (root, work, state), redirect_stdout(io.StringIO()):
            def complete(command, **kwargs):
                (root / "result.json").write_text(json.dumps(dict(passed=True)))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(pipeline.subprocess, "run", side_effect=complete) as child:
                pipeline.step(state, "audit", ["synthetic.py"], "result.json")
                self.assertEqual(state["steps"]["audit"]["status"], "complete")
                pipeline.step(state, "audit", ["synthetic.py"], "result.json")
                self.assertEqual(child.call_count, 1)
            (root / "result.json").write_text(json.dumps(dict(passed=False)))
            with patch.object(pipeline.subprocess, "run") as child:
                with self.assertRaisesRegex(ValueError, "Changed pipeline input"):
                    pipeline.step(state, "audit", ["synthetic.py"], "result.json")
                child.assert_not_called()
            with self.assertRaisesRegex(FileExistsError, "Unrecorded output"):
                pipeline.step(state, "different_audit", ["synthetic.py"], "result.json")

    def test_zero_child_exit_is_not_success_without_passing_evidence(self):
        with isolated_pipeline() as (root, work, state), redirect_stdout(io.StringIO()):
            def failed_evidence(command, **kwargs):
                (root / "result.json").write_text(json.dumps(dict(passed=False)))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(pipeline.subprocess, "run", side_effect=failed_evidence):
                with self.assertRaises(AssertionError):
                    pipeline.step(state, "audit", ["synthetic.py"], "result.json")
            self.assertNotEqual(state["steps"]["audit"]["status"], "complete")

    def test_existing_derived_manifest_uses_explicit_resume_and_validates_completion(self):
        with isolated_pipeline() as (root, work, state), redirect_stdout(io.StringIO()):
            witness = root / "run_manifest.json"
            witness.write_text(json.dumps(dict(status="running")))
            def complete(command, **kwargs):
                self.assertEqual(command[-1], "--resume")
                witness.write_text(json.dumps(dict(status="complete")))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(pipeline.subprocess, "run", side_effect=complete):
                pipeline.step(state, "derive", ["synthetic_derive.py", "--workers", "4"], "run_manifest.json")
            self.assertEqual(state["steps"]["derive"]["status"], "complete")

    def test_training_wait_accepts_completed_hash_and_fails_on_errors_or_expired_deadline(self):
        with isolated_pipeline() as (root, work, state), redirect_stdout(io.StringIO()):
            folder = root / "results/synthetic"
            folder.mkdir(parents=True)
            manifest = folder / "run_manifest.json"
            manifest.write_text(json.dumps(dict(status="complete")))
            with patch.object(pipeline.time, "sleep", side_effect=AssertionError("Unexpected polling")):
                pipeline.await_run(state, "synthetic", "work/train.log", time.monotonic()+10)
            self.assertEqual(state["training_manifests_sha256"]["results/synthetic/run_manifest.json"], digest(manifest))
            manifest.write_text(json.dumps(dict(status="failed")))
            with self.assertRaisesRegex(RuntimeError, "status failed"):
                pipeline.await_run(state, "synthetic", "work/train.log", time.monotonic()+10)
            manifest.write_text(json.dumps(dict(status="running")))
            (root / "work/train.log").write_text("Traceback (most recent call last):\nsynthetic failure\n")
            with self.assertRaisesRegex(RuntimeError, "Training error"):
                pipeline.await_run(state, "synthetic", "work/train.log", time.monotonic()+10)
            with self.assertRaises(TimeoutError):
                pipeline.await_run(state, "synthetic", "work/train.log", time.monotonic()-1)


if __name__ == "__main__":
    start = time.perf_counter()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ReportSmokeTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(SupervisorSmokeTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = ["scripts/report_supporting.py", "scripts/finalize_supporting.py", "tests/test_supporting_report_pipeline.py"]
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun, failures=len(result.failures),
                  errors=len(result.errors), elapsed_seconds=time.perf_counter()-start,
                  scope="Read-only immutable primary score schemas; synthetic temp-only report and supervisor artifacts",
                  production_pipeline_status_touched=False, production_outputs_written=False,
                  tested_code_sha256={name: digest(ROOT/name) for name in files},
                  readonly_primary_source_sha256={name: digest(ROOT/name) for name in PRIMARY_FILES})
    destination = ROOT / "work/supporting-validation/report_pipeline_tests.json"
    destination.write_text(json.dumps(report, indent=2)+"\n")
    sys.exit(0 if result.wasSuccessful() else 1)
