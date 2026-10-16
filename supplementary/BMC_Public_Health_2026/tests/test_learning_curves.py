"""Information boundaries, actual fits and complete-ledger learning-curve checks."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"

import copy
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
import joblib
import numpy as np
import pandas as pd
from gbd_park.learning_curves import (context, training_examples, target_arrays, fixed_base, local_specs,
    fit_local, fit_source, nonneural_forecasts, check_ledger, summarize, tasks, families)
from gbd_park.pooled import balanced_weights
from gbd_park.tcn import load_checkpoint, state_fingerprint
from gbd_park.tcn_forecasting import make_forecasts
import run_learning_curves as runner


def config():
    cfg = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
    cfg["models"]["tcn"]["ensemble_seeds"] = [11]
    cfg["cold_start_defaults"]["tcn_epochs"] = 1
    cfg["cold_start_defaults"]["boosting_trees"] = 2
    cfg["models"]["arima"].update(p=[0], d=[0], q=[0])
    return cfg


def panel(cfg):
    rows = []
    for ci, country in enumerate(cfg["countries"]):
        for si, sex in enumerate(cfg["sexes"]):
            for ai, age in enumerate(cfg["ages"]):
                for year in range(1990, 2024):
                    for oi, outcome in enumerate(["prevalence", "incidence"]):
                        t = year-1990
                        lograte = 2+ai*.15+si*.08+ci*.03-oi*.6+t*(.004+ci*.0003+oi*.0007)+.00003*t*t
                        rows.append({"location_name": country["name"], "sex": sex, "age": age,
                                     "year": year, "outcome": outcome, "rate": np.exp(lograte)})
    return pd.DataFrame(rows)


class LearningCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.panel = panel(cls.cfg)

    def test_fixed_design_defaults_and_job_counts(self):
        cfg = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())
        jobs = runner.job_specs(cfg)
        self.assertEqual(len(jobs), 38)
        self.assertEqual(sum(j["kind"] == "tcn" for j in jobs), 10)
        self.assertEqual(len(tasks(cfg)), 6)
        self.assertEqual(len(families(cfg)), 14)
        self.assertEqual(fixed_base(cfg, "tcn"), {"channels": 16, "weight_decay": .001, "epochs": 50})
        self.assertEqual([s["family"] for s in local_specs(cfg)], cfg["models"]["local_order"])
        self.assertTrue(all("history_years" not in j for j in jobs if j.get("mode") == "donor"))

    def test_complete_context_and_window_counts(self):
        for years, windows in [(15, 3), (20, 8), (29, 17)]:
            work = context(self.panel, self.cfg, "incidence", years)
            self.assertEqual(set(work.source_outcome), {"incidence"})
            self.assertEqual(work.year.max(), 2018)
            self.assertEqual(work.loc[work.location_name.eq("Saudi Arabia"), "year"].min(), 2019-years)
            x, y, meta = training_examples(work, self.cfg, years, "target")
            self.assertEqual(x.shape, (22*windows, 21))
            self.assertEqual(y.shape, (22*windows, 5))
            self.assertEqual(set(meta.groupby(["sex", "age"]).size()), {windows})
            donor = training_examples(work, self.cfg, years, "donor")[2]
            self.assertEqual(len(donor), 6*22*17)
            self.assertFalse(donor.country.eq("Saudi Arabia").any())
            self.assertEqual(donor.input_start.min(), 1990)
            self.assertLessEqual(donor.label_end.max(), 2018)

    def test_invalid_grids_are_errors_not_fallbacks(self):
        target = self.panel.index[(self.panel.location_name.eq("Saudi Arabia")) &
                                  self.panel.year.eq(2018) & self.panel.outcome.eq("prevalence")][0]
        for bad in [self.panel.drop(index=target), pd.concat([self.panel, self.panel.loc[[target]]])]:
            with self.assertRaises(ValueError):
                context(bad, self.cfg, "prevalence", 15)
        bad = self.panel.copy()
        bad.loc[target, "rate"] = -1
        with self.assertRaises(ValueError):
            fit_source(bad, self.cfg, "prevalence", "ridge")

    def test_outcome_and_future_rows_excluded(self):
        before = context(self.panel, self.cfg, "incidence", 15)
        changed = self.panel.copy()
        changed.loc[changed.year.gt(2018) | changed.outcome.ne("incidence"), "rate"] *= 50
        pd.testing.assert_frame_equal(before, context(changed, self.cfg, "incidence", 15))
        changed.loc[changed.location_name.eq("Saudi Arabia") & changed.year.lt(2004), "rate"] *= 8
        pd.testing.assert_frame_equal(before, context(changed, self.cfg, "incidence", 15))

    def test_balanced_weights_and_fit_only_scalers(self):
        for years in [15, 20, 29]:
            work = context(self.panel, self.cfg, "prevalence", years)
            x, y, meta = training_examples(work, self.cfg, years, "pooled")
            weights = balanced_weights(meta)
            country = meta.assign(weight=weights).groupby("country").weight.sum()
            strata = meta.assign(weight=weights).groupby(["country", "sex", "age"]).weight.sum()
            np.testing.assert_allclose(country, country.iloc[0], rtol=1e-12)
            np.testing.assert_allclose(strata, strata.iloc[0], rtol=1e-12)
            fit = fit_source(self.panel, self.cfg, "prevalence", "ridge", "pooled", years)
            self.assertEqual(fit["audit"]["status"], "ok")
            mean = np.average(x, axis=0, weights=weights)
            np.testing.assert_allclose(fit["audit"]["feature_mean"], mean, atol=1e-12)
            np.testing.assert_allclose(fit["audit"]["feature_variance"], np.average((x-mean)**2, axis=0, weights=weights), atol=1e-12)

    def test_restricted_pooled_and_local_fit_invariance(self):
        changed = self.panel.copy()
        changed.loc[(changed.location_name.eq("Saudi Arabia") & changed.year.lt(2004)) |
                    changed.year.gt(2018), "rate"] *= 9
        for kind in ["ridge", "boosting"]:
            first = fit_source(self.panel, self.cfg, "incidence", kind, "pooled", 15)
            second = fit_source(changed, self.cfg, "incidence", kind, "pooled", 15)
            self.assertEqual(first["audit"]["fingerprint_before"], second["audit"]["fingerprint_before"])
            np.testing.assert_array_equal(first["payloads"][15]["current_changes"], second["payloads"][15]["current_changes"])
        first = fit_local(self.panel, self.cfg, "prevalence", 15, "Male")
        second = fit_local(changed, self.cfg, "prevalence", 15, "Male")
        pd.testing.assert_frame_equal(pd.DataFrame(first["forecasts"]), pd.DataFrame(second["forecasts"]))
        self.assertEqual({r["family"] for r in first["forecasts"]}, set(self.cfg["models"]["local_order"]))
        self.assertTrue(all(r["history_start"] >= 2004 for r in first["forecasts"]))

    def test_actual_tcn_shared_checkpoint_and_budget_invariance(self):
        changed = self.panel.copy()
        changed.loc[(changed.location_name.eq("Saudi Arabia") & changed.year.lt(2004)) |
                    changed.year.gt(2018), "rate"] *= 4
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.joblib"
            first = fit_source(self.panel, self.cfg, "prevalence", "tcn", seed=11, checkpoint_path=checkpoint)
            second = fit_source(changed, self.cfg, "prevalence", "tcn", seed=11)
            self.assertEqual(first["audit"]["status"], "ok")
            self.assertEqual(first["audit"]["fingerprint_before"], state_fingerprint(load_checkpoint(checkpoint)))
            self.assertEqual(first["audit"]["fingerprint_before"], second["audit"]["fingerprint_before"])
            payload = first["payloads"][15]
            for years in [15, 20, 29]:
                np.testing.assert_array_equal(payload["current_changes"], first["payloads"][years]["current_changes"])
            np.testing.assert_array_equal(payload["target_y"], second["payloads"][15]["target_y"])
            np.testing.assert_array_equal(payload["target_changes"], second["payloads"][15]["target_changes"])
            before, c1 = make_forecasts([payload], self.cfg, [1], include_intercept=True)
            after, c2 = make_forecasts([second["payloads"][15]], self.cfg, [1], include_intercept=True)
            pd.testing.assert_frame_equal(pd.DataFrame(before), pd.DataFrame(after))
            self.assertEqual(c1, c2)
            self.assertEqual(len(before), 330)
            self.assertEqual({r["family"] for r in before}, {"tcn_unadapted", "tcn_intercept", "tcn_adapted"})

    def test_nonneural_adaptation_matches_own_budget(self):
        result = fit_source(self.panel, self.cfg, "incidence", "ridge")
        previous = None
        for years in [15, 20, 29]:
            rows, corrections = nonneural_forecasts(result["payloads"][years], self.cfg, "donor", "ridge")
            frame = pd.DataFrame(rows)
            current = frame.loc[frame.family.eq("donor_ridge_unadapted"), "prediction"].to_numpy()
            if previous is not None:
                np.testing.assert_array_equal(current, previous)
            previous = current
            self.assertEqual(len(corrections), 2)
            self.assertTrue(all(c["target_windows"] == (years-12)*11 for c in corrections))
            self.assertTrue(all(c["last_target_label_year"] == 2018 for c in corrections))
            self.assertTrue(all(c["first_target_input_year"] == 2019-years for c in corrections))

    def test_fallback_retained_for_all_budgets(self):
        with patch("gbd_park.learning_curves.fit_base", side_effect=ValueError("synthetic fit failure")):
            result = fit_source(self.panel, self.cfg, "prevalence", "ridge")
        for years, payload in result["payloads"].items():
            rows, corrections = nonneural_forecasts(payload, self.cfg, "donor", "ridge")
            self.assertEqual(len(rows), 220)
            self.assertTrue(all(r["status"] == "fallback" for r in rows))
            self.assertTrue(all("synthetic fit failure" in r["fallback_reason"] for r in rows))
            self.assertTrue(all(c["status"] == "fallback" for c in corrections))

    def test_full_issuance_scoring_and_tamper_protection(self):
        # Actual small donor/pooled fits; local ledger uses known persistence values
        # to keep this integration test focused on issuance and score arithmetic.
        def known_local(panel, cfg, outcome, years, sex):
            work = context(panel, cfg, outcome, years)
            current, levels, meta, _, _, _ = target_arrays(work, cfg, years)
            rows = []
            for family in cfg["models"]["local_order"]:
                for a, row in meta.loc[meta.sex.eq(sex)].iterrows():
                    for h in range(1, 6):
                        rows.append({"target": cfg["primary_target"], "sex": sex, "age": row.age,
                            "origin": 2018, "forecast_year": 2018+h, "horizon": h, "outcome": "prevalence",
                            "family": family, "setting_id": family+"__test", "parameter_count": 0,
                            "prediction": float(np.exp(levels[a])), "log_prediction": float(levels[a]),
                            "status": "ok", "fallback_reason": ""})
            return {"forecasts": rows, "fits": [], "arima": []}

        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with patch.object(runner, "source_panel", return_value=self.panel), patch.object(runner, "fit_local", side_effect=known_local):
                with self.assertRaises(ValueError):
                    runner.score_all(self.cfg, out)
                jobs = runner.job_specs(self.cfg)
                for job in jobs:
                    runner.fit_job(self.cfg, job, out)
                # Verified cache may be reused, while changed configuration must fail.
                runner.fit_job(self.cfg, jobs[0], out)
                changed = copy.deepcopy(self.cfg)
                changed["cold_start_defaults"]["ridge_penalty"] = 10
                with self.assertRaises(ValueError):
                    runner.load_result(out, jobs[0], changed)
                runner.assemble_issued(self.cfg, out)
                self.assertFalse((out / "point_scores.csv").exists())
                points = pd.read_csv(out / "predictions.csv")
                self.assertEqual(check_ledger(points, self.cfg)["prediction_rows"], 9240)
                with self.assertRaises(ValueError):
                    check_ledger(points.iloc[:-1], self.cfg)
                broken = points.copy()
                broken.loc[0, "history_years"] = 99
                with self.assertRaises(ValueError):
                    check_ledger(broken, self.cfg)
                runner.score_all(self.cfg, out)
                scored = pd.read_csv(out / "point_scores.csv")
                self.assertEqual(len(scored), 9240)
                np.testing.assert_allclose(scored.absolute_log_error, abs(np.log(scored.prediction/scored.observed_rate)), atol=1e-14)
                self.assertEqual(len(pd.read_csv(out / "summary.csv")), 1680)
                validation = json.loads((out / "validation_report.json").read_text())
                self.assertTrue(validation["all_six_arms_committed_before_scoring"])
                write = runner.write_json
                write(out / "run_manifest.json", {"status": "complete", "code_sha256": {},
                    "identity": {"source_sha256": runner.sha(ROOT / "data/processed/design_v1/regional_outcomes.csv")},
                    "output_sha256": runner.hash_files(out, [p for p in out.rglob("*") if p.is_file()])})
                audit = runner.audit_run(self.cfg, out)
                self.assertTrue(audit["passed"])
                self.assertEqual(len(audit["source_checkpoint_replays"]), 18)
                with tempfile.TemporaryDirectory() as report_parent:
                    destination = Path(report_parent) / "report"
                    runner.report(self.cfg, out, destination, audit)
                    self.assertTrue((destination / "report.md").is_file())
                    self.assertTrue((destination / "learning_curves.svg").is_file())
                # A forged correction record must fail numerical reconstruction,
                # even if an attacker updates every containing artifact hash.
                ap = out / "adaptation_audit.json"
                corrupt = json.loads(ap.read_text())
                corrupt[0]["b0"] += 1
                write(ap, corrupt)
                ip = out / "issued_commit.json"
                issue = json.loads(ip.read_text())
                issue["artifact_sha256"]["adaptation_audit.json"] = runner.sha(ap)
                write(ip, issue)
                mp = out / "run_manifest.json"
                manifest = json.loads(mp.read_text())
                manifest["output_sha256"]["adaptation_audit.json"] = runner.sha(ap)
                manifest["output_sha256"]["issued_commit.json"] = runner.sha(ip)
                write(mp, manifest)
                with self.assertRaisesRegex(ValueError, "Reconstructed adaptation_audit"):
                    runner.audit_run(self.cfg, out)
                # The issuance commitment covers nested source payloads as well as CSV.
                path = runner.result_path(out, jobs[0])
                path.write_bytes(path.read_bytes()+b"changed")
                with self.assertRaises(ValueError):
                    runner.score_all(self.cfg, out)
                with self.assertRaises(ValueError):
                    runner.load_result(out, jobs[0], self.cfg)


if __name__ == "__main__":
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LearningCurveTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    directory = ROOT / "work/learning-curves-validation"
    directory.mkdir(parents=True, exist_ok=True)
    runner.write_json(directory / "tests.json", {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "completed_utc": runner.now(),
        "elapsed_seconds": time.perf_counter()-start,
        "tested_code_sha256": {name: runner.sha(ROOT / name) for name in runner.REQUIRED},
        "tests": [str(test) for test, _ in result.failures+result.errors],
        "note": "Small deterministic synthetic fits; production settings remain the locked defaults."})
    raise SystemExit(0 if result.wasSuccessful() else 1)
