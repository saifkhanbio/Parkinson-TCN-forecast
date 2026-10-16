"""Isolated CPU-recovery finalization checks; no audit or model process runs."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("cpu_recovery_finalizer", ROOT / "scripts/finalize_study_cpu_recovery.py")
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)
FLOW = RECOVERY.flow


class RecoverySupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "work/global-asr-cpu-recovery-validation"
        self.work.mkdir(parents=True)
        self.patches = [patch.object(RECOVERY, "ROOT", self.root), patch.object(RECOVERY, "WORK", self.work),
                        patch.object(FLOW, "ROOT", self.root), patch.object(FLOW, "WORK", self.work),
                        patch.object(FLOW, "STATE", self.work / "pipeline_status.json")]
        for item in self.patches:
            item.start()
        for name in ["scripts/finalize_study_cpu_recovery.py", "scripts/report_study_synthesis_recovery.py",
                     "work/learning-curves-validation/audit_global_asr.py"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic frozen code")
        self.protected = {}
        for index in range(46):
            path = self.root / f"preserved/original_{index:02}.json"
            self.write(path, {"original": index})
            self.protected[str(path.relative_to(self.root))] = FLOW.digest(path)
        self.write(self.work / "preserved_sha256.json", {"sha256": self.protected})
        self.manifest = self.root / "results/global_asr_cpu_recovery_v1/run_manifest.json"
        self.write(self.manifest, {"status": "complete"})
        self.audit_path = self.work / "independent_audit.json"
        self.audit = {"passed": True, "source_checkpoint_replays": 216, "forecast_rows": 2640,
                      "cases": [{} for _ in range(12)], "run_manifest_sha256": FLOW.digest(self.manifest),
                      "audit_code_sha256": FLOW.digest(self.root / "work/learning-curves-validation/audit_global_asr.py")}
        self.write(self.audit_path, self.audit)
        self.report_path = self.root / "reports/global_asr_cpu_recovery_v1/validation.json"
        self.report = {"passed": True, "run_manifest_sha256": FLOW.digest(self.manifest),
                       "audit_sha256": FLOW.digest(self.audit_path), "fallback_cells": 0}
        self.write(self.report_path, self.report)
        self.state = {"status": "waiting", "steps": {},
                      "frozen_sha256": {"scripts/finalize_study_cpu_recovery.py": FLOW.digest(
                          self.root / "scripts/finalize_study_cpu_recovery.py")},
                      "training_manifest_sha256": {str(self.manifest.relative_to(self.root)): FLOW.digest(self.manifest)}}

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_strict_existing_audit_acceptance_and_no_second_execution(self):
        before = self.audit_path.read_bytes()
        with patch.object(FLOW.subprocess, "run") as process:
            RECOVERY.accept_independent_audit(self.state)
            RECOVERY.accept_independent_audit(self.state)
        process.assert_not_called()
        self.assertEqual(self.audit_path.read_bytes(), before)
        self.assertEqual(self.state["steps"]["independent_audit"]["status"], "complete")
        self.assertTrue(FLOW.STATE.is_file())
        self.assertFalse((self.root / "work/completion-validation/pipeline_status.json").exists())

    def test_each_missing_execution_success_condition_fails(self):
        variants = [{"passed": False}, {"source_checkpoint_replays": 106}, {"source_checkpoint_replays": 215},
                    {"forecast_rows": 2639}, {"cases": [{}]*11}, {"run_manifest_sha256": "wrong"},
                    {"audit_code_sha256": "wrong"}]
        for changes in variants:
            self.write(self.audit_path, dict(self.audit, **changes))
            self.write(self.report_path, dict(self.report, audit_sha256=FLOW.digest(self.audit_path)))
            with self.assertRaisesRegex(ValueError, "complete independent checkpoint replay"):
                RECOVERY.accept_independent_audit(self.state)
        self.assertFalse(self.state["steps"])

    def test_report_hash_or_fallback_failure_is_not_accepted(self):
        for changes in [{"audit_sha256": "wrong"}, {"fallback_cells": 1}, {"fallback_cells": 720}]:
            self.write(self.report_path, dict(self.report, **changes))
            with self.assertRaises(ValueError):
                RECOVERY.accept_independent_audit(self.state)
        self.assertFalse(self.state["steps"])

    def test_changed_recorded_witness_fails_without_overwriting(self):
        RECOVERY.accept_independent_audit(self.state)
        altered = dict(self.audit, new_date="changed")
        self.write(self.audit_path, altered)
        self.write(self.report_path, dict(self.report, audit_sha256=FLOW.digest(self.audit_path)))
        before = FLOW.STATE.read_bytes()
        with self.assertRaisesRegex(ValueError, "Changed pipeline input"):
            RECOVERY.accept_independent_audit(self.state)
        self.assertEqual(FLOW.STATE.read_bytes(), before)

    def test_preservation_all_46_and_changed_original_rejected(self):
        self.assertEqual(len(self.protected), 46)
        RECOVERY.preserve_prior()
        original = self.root / "preserved/original_45.json"
        original.write_text("modified")
        with self.assertRaisesRegex(ValueError, "Changed pipeline input"):
            RECOVERY.preserve_prior()

    def synthetic_final_report(self):
        directory = self.root / RECOVERY.REPORT
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.md").write_text("new version only")
        self.write(directory / "validation.json", {"passed": True,
            "reporter_sha256": FLOW.digest(self.root / "scripts/report_study_synthesis_recovery.py"),
            "source_sha256": self.protected, "artifact_sha256": {"report.md": FLOW.digest(directory / "report.md")}})
        return directory

    def test_new_reporter_identity_and_all_report_artifacts_verified(self):
        RECOVERY.accept_independent_audit(self.state)
        directory = self.synthetic_final_report()
        artifacts = RECOVERY.verify_completion(self.state)
        self.assertIn("reports/study_synthesis_v1_1/report.md", artifacts)
        self.assertTrue(all(name.startswith("reports/study_synthesis_v1_1/") for name in artifacts))
        reporter = self.root / "scripts/report_study_synthesis_recovery.py"
        reporter.write_text("changed reporter")
        with self.assertRaisesRegex(ValueError, "reporter identity"):
            RECOVERY.verify_completion(self.state)

    def test_main_handoff_writes_only_new_paths_and_resume_repeats_nothing(self):
        old_report = self.root / "reports/study_synthesis_v1/report.md"
        old_report.parent.mkdir(parents=True)
        old_report.write_text("preserve original synthesis")
        old_audit = self.root / "work/learning-curves-validation/global_asr_independent_audit.json"
        self.write(old_audit, {"passed": True, "original_gpu_fallback_audit": True})
        sentinels = {old_report: old_report.read_bytes(), old_audit: old_audit.read_bytes()}
        for filename in ["tests.json", "reporter_tests.json"]:
            self.write(self.work / filename, {"passed": True})
        frozen = ["scripts/finalize_study_cpu_recovery.py", "scripts/report_study_synthesis_recovery.py",
                  "work/learning-curves-validation/audit_global_asr.py"]

        def fake_wait(state, name, log, deadline):
            self.assertEqual(name, "global_asr_cpu_recovery_v1")
            self.assertEqual(log, "work/global-asr-cpu-recovery-validation/run.log")
            state["training_manifest_sha256"] = self.state["training_manifest_sha256"]
            FLOW.save(state)

        def fake_step(state, name, arguments, witness):
            self.assertEqual(arguments, ["scripts/report_study_synthesis_recovery.py"])
            self.assertEqual(witness, "reports/study_synthesis_v1_1/validation.json")
            directory = self.synthetic_final_report()
            state["steps"][name] = {"status": "complete", "witness_sha256": {
                witness: FLOW.digest(directory / "validation.json")}}
            FLOW.save(state)

        with patch.object(RECOVERY, "FROZEN", frozen), patch.object(sys, "argv", ["finalizer"]), \
                patch.object(FLOW, "await_run", side_effect=fake_wait) as waiter, \
                patch.object(FLOW, "step", side_effect=fake_step) as stepper, \
                patch.object(FLOW.subprocess, "run") as process:
            RECOVERY.main()
            self.assertEqual(FLOW.read(FLOW.STATE)["status"], "complete")
            with patch.object(sys, "argv", ["finalizer", "--resume"]):
                RECOVERY.main()
        waiter.assert_called_once()
        stepper.assert_called_once()
        process.assert_not_called()
        for path, original in sentinels.items():
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    started = time.perf_counter()
    results = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoverySupervisorTests))
    evidence = {"passed": results.wasSuccessful(), "tests_run": results.testsRun,
                "elapsed_seconds": time.perf_counter()-started,
                "tested_code_sha256": {name: FLOW.digest(ROOT / name) for name in [
                    "scripts/finalize_study_cpu_recovery.py", "scripts/finalize_study.py",
                    "work/global-asr-cpu-recovery-validation/test_supervisor.py"]},
                "failures": [test.id() for test, _ in results.failures],
                "errors": [test.id() for test, _ in results.errors],
                "real_processes_started": False, "model_fitting_performed": False}
    (ROOT / "work/global-asr-cpu-recovery-validation/supervisor_tests.json").write_text(json.dumps(evidence, indent=2)+"\n")
    raise SystemExit(0 if results.wasSuccessful() else 1)
