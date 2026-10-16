"""Hardware-recovery acceptance and immutable-witness tests; no retuning."""
import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
import pandas as pd
import run_global_asr_cpu_recovery as recovery
from test_global_asr import CONFIG, synthetic, full_predictions


def audits_fixture():
    rows = []
    for job in recovery.original.jobs(CONFIG, "cpu"):
        if job["kind"] != "tcn":
            continue
        countries = ([f"Synthetic global {index}" for index in range(202)] if job["scope"] == "global"
                     else [country["name"] for country in CONFIG["countries"] if country["name"] != job["target"]])
        rows.append({"target": job["target"], "outcome": job["outcome"], "scope": job["scope"], "seed": job["seed"],
            "device": "cpu", "status": "ok", "base": recovery.asr.fixed_base(CONFIG), "parameter_count": 1732,
            "fingerprint_before": "same_frozen_source", "fingerprint_after": "same_frozen_source",
            "countries": countries, "maximum_label_year": 2018})
    return rows


class CPURecoveryTests(unittest.TestCase):
    def test_exact_design_job_identity_except_neural_device(self):
        before = recovery.original.jobs(CONFIG, "cuda:0")
        after = recovery.original.jobs(CONFIG, "cpu")
        self.assertEqual(len(after), 156)
        for old, new in zip(before, after):
            expected = dict(old)
            if expected["kind"] == "tcn": expected["device"] = "cpu"
            self.assertEqual(new, expected)
        self.assertEqual(sum(job["kind"] == "tcn" for job in after), 120)

    def test_complete_source_gate(self):
        result = recovery.source_gate(audits_fixture(), CONFIG)
        self.assertEqual(result["successful_source_fits"], 120)
        self.assertEqual(result["complete_five_seed_ensembles"], 24)
        self.assertEqual(result["all_scopes_same_device"], "cpu")

    def test_failed_missing_duplicate_or_wrong_device_seed_rejected(self):
        for modification in ["failed", "missing", "duplicate", "wrong_device", "wrong_scope", "wrong_settings", "mutated_state", "target_leak"]:
            rows = audits_fixture()
            if modification == "failed": rows[0]["status"] = "fallback"
            elif modification == "missing": rows.pop()
            elif modification == "duplicate": rows[-1] = deepcopy(rows[0])
            elif modification == "wrong_device": rows[0]["device"] = "cuda:0"
            elif modification == "wrong_scope": rows[0]["countries"] = rows[0]["countries"][:6]
            elif modification == "wrong_settings": rows[0]["base"]["epochs"] = 100
            elif modification == "mutated_state": rows[0]["fingerprint_after"] = "changed"
            elif modification == "target_leak": rows[0]["countries"][0] = rows[0]["target"]
            with self.subTest(modification=modification), self.assertRaises(ValueError):
                recovery.source_gate(rows, CONFIG)

    def test_original_gpu_whole_ensemble_failures_cannot_pass_recovery_gate(self):
        # A previously successful seed cannot salvage any of the failed ensembles.
        rows = audits_fixture()
        for index, row in enumerate(rows):
            if index % 5 == 0: row["status"] = "fallback"
        with self.assertRaises(ValueError): recovery.source_gate(rows, CONFIG)

    def test_point_success_gate_all2640_and720_neural(self):
        audits = audits_fixture()
        lookup = {(row["target"], row["outcome"], row["scope"], row["seed"]): row for row in audits}
        def load(_, job):
            return {"audit": lookup[(job["target"], job["outcome"], job["scope"], job["seed"]) ]}
        points = full_predictions(synthetic()[3])
        with tempfile.TemporaryDirectory() as temp, patch.object(recovery.original, "load_job", side_effect=load):
            folder = Path(temp)
            points.to_csv(folder / "predictions.csv", index=False)
            gate = recovery.execution_gate(folder, CONFIG, True)
            self.assertEqual(gate["forecast_rows"], 2640)
            self.assertEqual(gate["successful_neural_forecast_cells"], 720)
            for family in ["tcn_adapted", "persistence"]:
                changed = points.copy()
                changed.loc[changed.family.eq(family), "status"] = "fallback"
                changed.to_csv(folder / "predictions.csv", index=False)
                with self.subTest(family=family), self.assertRaises(ValueError):
                    recovery.execution_gate(folder, CONFIG, True)

    def test_read_only_witness_verification_preserves_every_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); out = folder / "run"; reports = folder / "reports"
            out.mkdir(); reports.mkdir(); audit_path = folder / "independent_audit.json"
            recovery.write(out / "run_manifest.json", {"status": "complete"})
            (reports / "report.md").write_text("Synthetic successful recovery report\n")
            manifest_hash = recovery.asr.digest(out / "run_manifest.json")
            recovery.write(audit_path, {"passed": True, "forecast_rows": 2640, "source_checkpoint_replays": 216,
                "run_manifest_sha256": manifest_hash, "audit_code_sha256": recovery.asr.digest(ROOT / recovery.AUDITOR)})
            recovery.write(reports / "validation.json", {"passed": True, "run_manifest_sha256": manifest_hash,
                "audit_sha256": recovery.asr.digest(audit_path), "fallback_cells": 0,
                "output_sha256": {"report.md": recovery.asr.digest(reports / "report.md")}})
            originals = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
            with patch.object(recovery, "AUDIT_PATH", audit_path), patch.object(recovery, "REPORTS", reports), \
                    patch.object(recovery, "execution_gate", return_value={"passed": True}):
                recovery.verify_existing(out, CONFIG)
                for path, content in originals.items(): self.assertEqual(path.read_bytes(), content)
                (reports / "report.md").write_text("tampered")
                with self.assertRaises(ValueError): recovery.verify_existing(out, CONFIG)

    def test_original_artifacts_paths_are_distinct_and_identity_preservation_passes(self):
        self.assertNotEqual(recovery.OUTPUT, recovery.ORIGINAL)
        self.assertEqual(recovery.AUDIT_PATH, ROOT / "work/global-asr-cpu-recovery-validation/independent_audit.json")
        old_audit = ROOT / "work/global-asr-validation/audit.json"
        before = recovery.asr.digest(old_audit)
        result = recovery.preservation()
        self.assertGreaterEqual(result["preserved_identities"], 46)
        self.assertEqual(before, recovery.asr.digest(old_audit))


if __name__ == "__main__":
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CPURecoveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = list(dict.fromkeys(recovery.NEW_FILES + recovery.original.REQUIRED + recovery.original.DEPENDENCIES + [recovery.AUDITOR]))
    recovery.write(recovery.TESTS, {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "elapsed_seconds": time.perf_counter()-start,
        "tested_code_sha256": {name: recovery.asr.digest(ROOT / name) for name in files},
        "failures": [case.id() for case, _ in result.failures], "errors": [case.id() for case, _ in result.errors]})
    raise SystemExit(0 if result.wasSuccessful() else 1)
