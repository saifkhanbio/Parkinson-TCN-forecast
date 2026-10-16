"""Isolated supervisor state tests: no production processes or models are run."""

import importlib.util
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("finalize_study_tested", ROOT / "scripts/finalize_study.py")
SUPERVISOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


class FinalizeStudyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.work = self.root / "work/completion-validation"
        self.work.mkdir(parents=True)
        self.state_path = self.work / "pipeline_status.json"
        self.patches = [patch.object(SUPERVISOR, "ROOT", self.root),
                        patch.object(SUPERVISOR, "WORK", self.work),
                        patch.object(SUPERVISOR, "STATE", self.state_path)]
        for mock in self.patches:
            mock.start()
        (self.root / "frozen.txt").write_text("original frozen source")
        self.state = {"status": "starting", "steps": {},
                      "frozen_sha256": {"frozen.txt": SUPERVISOR.digest(self.root / "frozen.txt")}}
        self.manifest = self.root / "results/synthetic_v1/run_manifest.json"
        self.report = self.root / "reports/synthetic_v1/validation.json"
        self.log = self.root / "work/production.log"

    def tearDown(self):
        for mock in reversed(self.patches):
            mock.stop()
        self.directory.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def ready(self, field="run_manifest_sha256"):
        self.write(self.manifest, {"status": "complete"})
        self.write(self.report, {"passed": True, field: SUPERVISOR.digest(self.manifest)})

    def await_once(self):
        SUPERVISOR.await_run(self.state, "synthetic_v1", "work/production.log", 10)

    def test_complete_manifest_without_report_is_not_ready(self):
        self.write(self.manifest, {"status": "complete"})
        with patch.object(SUPERVISOR.time, "monotonic", side_effect=[0, 1]), \
                patch.object(SUPERVISOR.time, "sleep", side_effect=lambda seconds: self.ready()) as sleeper:
            self.await_once()
        sleeper.assert_called_once_with(30)
        self.assertEqual(self.state["training_manifest_sha256"], {
            "results/synthetic_v1/run_manifest.json": SUPERVISOR.digest(self.manifest)})

    def test_running_manifest_with_matching_report_is_not_ready(self):
        self.write(self.manifest, {"status": "running"})
        self.write(self.report, {"passed": True, "run_manifest_sha256": SUPERVISOR.digest(self.manifest)})
        with patch.object(SUPERVISOR.time, "monotonic", side_effect=[0, 1]), \
                patch.object(SUPERVISOR.time, "sleep", side_effect=lambda seconds: self.ready()) as sleeper:
            self.await_once()
        sleeper.assert_called_once()

    def test_both_producer_manifest_field_names_are_supported(self):
        for field in ["run_manifest_sha256", "source_run_manifest_sha256"]:
            self.ready(field)
            with patch.object(SUPERVISOR.time, "monotonic", return_value=0), \
                    patch.object(SUPERVISOR.time, "sleep") as sleeper:
                self.await_once()
            sleeper.assert_not_called()

    def test_mismatched_or_failed_report_fails_closed(self):
        for passed, fingerprint, exception in [(True, "wrong", ValueError),
                                               (False, "wrong", RuntimeError)]:
            self.ready()
            self.write(self.report, {"passed": passed, "run_manifest_sha256": fingerprint})
            with patch.object(SUPERVISOR.time, "monotonic", return_value=0), self.assertRaises(exception):
                self.await_once()
        self.assertNotIn("training_manifest_sha256", self.state)

    def test_traceback_takes_priority_over_completed_markers(self):
        self.ready()
        self.log.write_text("completed logs\nTraceback (most recent call last):\nRuntimeError: failure")
        with patch.object(SUPERVISOR.time, "monotonic", return_value=0), self.assertRaisesRegex(RuntimeError, "Production error"):
            self.await_once()
        self.assertNotIn("training_manifest_sha256", self.state)

    def test_partial_json_is_tolerated_until_producer_finishes(self):
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{"status":')
        with patch.object(SUPERVISOR.time, "monotonic", side_effect=[0, 1]), \
                patch.object(SUPERVISOR.time, "sleep", side_effect=lambda seconds: self.ready()) as sleeper:
            self.await_once()
        sleeper.assert_called_once()

    def test_timeout_preserves_production_and_does_not_run_commands(self):
        self.write(self.manifest, {"status": "running"})
        before = self.manifest.read_bytes()
        with patch.object(SUPERVISOR.time, "monotonic", return_value=11), \
                patch.object(SUPERVISOR.subprocess, "run") as process, self.assertRaises(TimeoutError):
            self.await_once()
        process.assert_not_called()
        self.assertEqual(self.manifest.read_bytes(), before)

    def test_changed_frozen_input_blocks_step(self):
        (self.root / "frozen.txt").write_text("altered")
        with patch.object(SUPERVISOR.subprocess, "run") as process, self.assertRaisesRegex(ValueError, "Changed pipeline input"):
            SUPERVISOR.step(self.state, "audit", ["audit.py"], "witness.json")
        process.assert_not_called()

    def test_changed_training_manifest_blocks_step(self):
        self.ready()
        self.state["training_manifest_sha256"] = {
            "results/synthetic_v1/run_manifest.json": SUPERVISOR.digest(self.manifest)}
        self.write(self.manifest, {"status": "complete", "changed": True})
        with patch.object(SUPERVISOR.subprocess, "run") as process, self.assertRaisesRegex(ValueError, "Changed pipeline input"):
            SUPERVISOR.step(self.state, "audit", ["audit.py"], "witness.json")
        process.assert_not_called()

    def test_unrecorded_witness_is_never_overwritten(self):
        witness = self.root / "witness.json"
        self.write(witness, {"passed": True, "original": True})
        before = witness.read_bytes()
        with patch.object(SUPERVISOR.subprocess, "run") as process, self.assertRaisesRegex(FileExistsError, "Unrecorded"):
            SUPERVISOR.step(self.state, "audit", ["audit.py"], "witness.json")
        process.assert_not_called()
        self.assertEqual(witness.read_bytes(), before)

    def test_successful_step_is_recorded_and_not_repeated(self):
        def fake_run(command, **kwargs):
            self.assertEqual(command, [SUPERVISOR.PYTHON, "-u", "audit.py", "--verify"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertFalse(kwargs["check"])
            self.write(self.root / "witness.json", {"passed": True})
            return SimpleNamespace(returncode=0)
        with patch.object(SUPERVISOR.subprocess, "run", side_effect=fake_run) as process:
            SUPERVISOR.step(self.state, "audit", ["audit.py", "--verify"], "witness.json")
            SUPERVISOR.step(self.state, "audit", ["audit.py", "--verify"], "witness.json")
        process.assert_called_once()
        self.assertEqual(self.state["steps"]["audit"]["status"], "complete")
        self.assertEqual(self.state["steps"]["audit"]["witness_sha256"], {
            "witness.json": SUPERVISOR.digest(self.root / "witness.json")})
        self.assertFalse(self.state_path.with_suffix(".tmp").exists())

    def test_changed_completed_witness_is_not_silently_repeated(self):
        self.write(self.root / "witness.json", {"passed": True})
        self.state["steps"]["audit"] = {"status": "complete", "witness_sha256": {
            "witness.json": SUPERVISOR.digest(self.root / "witness.json")}}
        self.write(self.root / "witness.json", {"passed": True, "changed": True})
        with patch.object(SUPERVISOR.subprocess, "run") as process, self.assertRaises(ValueError):
            SUPERVISOR.step(self.state, "audit", ["audit.py"], "witness.json")
        process.assert_not_called()

    def test_nonzero_command_is_never_marked_complete(self):
        with patch.object(SUPERVISOR.subprocess, "run", return_value=SimpleNamespace(returncode=7)), \
                self.assertRaisesRegex(RuntimeError, "exited 7"):
            SUPERVISOR.step(self.state, "audit", ["audit.py"], "witness.json")
        self.assertNotEqual(self.state["steps"]["audit"]["status"], "complete")

    def test_zero_exit_requires_explicit_passed_evidence(self):
        def fake_run(*args, **kwargs):
            self.write(self.root / "witness.json", {"passed": False})
            return SimpleNamespace(returncode=0)
        with patch.object(SUPERVISOR.subprocess, "run", side_effect=fake_run), self.assertRaisesRegex(ValueError, "successful evidence"):
            SUPERVISOR.step(self.state, "audit", ["audit.py"], "witness.json")
        self.assertNotEqual(self.state["steps"]["audit"]["status"], "complete")

    def test_final_verification_protects_sources_report_and_preserved_files(self):
        report = self.root / "reports/study_synthesis_v1"
        report.mkdir(parents=True)
        (report / "report.md").write_text("Complete synthetic report")
        script = self.root / "scripts/report_study_synthesis.py"
        script.parent.mkdir(parents=True)
        script.write_text("# synthetic reporter identity")
        self.write(self.work / "preserved_manifest_hashes.json", {"sha256": self.state["frozen_sha256"]})
        self.write(report / "validation.json", {"passed": True,
            "reporter_sha256": SUPERVISOR.digest(script), "source_sha256": self.state["frozen_sha256"],
            "artifact_sha256": {"report.md": SUPERVISOR.digest(report / "report.md")}})
        for name in ["audit_global_asr", "audit_projections", "integrated_report"]:
            witness = report / "validation.json" if name == "integrated_report" else self.root / (name+".json")
            if name != "integrated_report":
                self.write(witness, {"passed": True})
            self.state["steps"][name] = {"status": "complete", "witness_sha256": {
                str(witness.relative_to(self.root)): SUPERVISOR.digest(witness)}}
        artifacts = SUPERVISOR.verify_completion(self.state)
        self.assertIn("reports/study_synthesis_v1/report.md", artifacts)
        (report / "report.md").write_text("Modified report")
        with self.assertRaises(ValueError):
            SUPERVISOR.verify_completion(self.state)


if __name__ == "__main__":
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FinalizeStudyTests))
    evidence = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
                "elapsed_seconds": time.perf_counter()-started,
                "tested_script_sha256": SUPERVISOR.digest(ROOT / "scripts/finalize_study.py"),
                "test_code_sha256": SUPERVISOR.digest(Path(__file__)),
                "failures": [test.id() for test, _ in result.failures],
                "errors": [test.id() for test, _ in result.errors],
                "production_processes_started": False, "model_fitting_performed": False}
    (ROOT / "work/completion-validation/supervisor_tests.json").write_text(json.dumps(evidence, indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
