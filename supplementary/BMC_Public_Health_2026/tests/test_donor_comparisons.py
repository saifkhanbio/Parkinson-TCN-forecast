"""Synthetic CPU tests for the GPU donor experiment's orchestration and cache."""

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
for directory in [ROOT / "src", ROOT / "scripts", ROOT / "tests"]:
    sys.path.insert(0, str(directory))
import joblib
import numpy as np
import pandas as pd
from gbd_park.tcn import base_grid
from gbd_park.tcn_forecasting import select_tcn_settings
from run_donor_comparisons import (arms, cached_fit_job, compose_forecasts, fit_job_description,
                                   load_cached_job, score_arms, source_context,
                                   validate_reported_grid, write_frame, write_json)
from test_donors import CONFIG, DonorTests, panel_fixture


def composition_fixture():
    origin, arm, base = 2003, "similar_three", {**base_grid(CONFIG)[0], "epochs": 2}
    current = pd.DataFrame([{"country": "Saudi Arabia", "sex": sex, "age": age}
                            for sex in CONFIG["sexes"] for age in CONFIG["ages"]])
    historical = pd.concat([current.assign(window_origin=o, label_end=o + 5) for o in [1997, 1998]], ignore_index=True)
    plans, payloads = {}, {}
    for sex_index, sex in enumerate(CONFIG["sexes"]):
        countries = ["Bahrain", "Jordan", "Kuwait"] if sex == "Male" else ["Kuwait", "Oman", "Qatar"]
        plans[(origin, arm, sex)] = {"countries": countries, "strategy": "three_similar", "seed": None,
                                   "donor_list_sha256": f"pool{sex_index}", "device": "cpu"}
        for index, seed in enumerate(CONFIG["models"]["tcn"]["ensemble_seeds"]):
            job = fit_job_description(origin, countries, base, seed, "cpu")
            payloads[job["job_id"]] = {"origin": origin, "base": base, "seed": seed,
                "current_changes": np.full((22, 5), .1 + .6 * sex_index + index * .02),
                "target_changes": np.full((44, 5), index * .01), "target_y": np.zeros((44, 5)),
                "levels": np.full(22, np.log(100.)), "current_meta": current, "target_meta": historical,
                "audit": {"status": "ok", "reason": "", "fingerprint_before": job["job_id"]}, "cache_job": job}
    return origin, arm, base, plans, payloads


class DonorComparisonTests(unittest.TestCase):
    def test_all_prespecified_arms_are_kept_and_cache_keys_share_only_equal_fits(self):
        definitions = arms(CONFIG)
        self.assertEqual(len(definitions), 8)
        self.assertEqual([a["seed"] for a in definitions if a["strategy"] == "three_random"], [101, 211, 307, 401, 503])
        base = base_grid(CONFIG)[0]
        first = fit_job_description(2003, ["Bahrain", "Jordan", "Kuwait"], base, 11, "cuda:0")
        repeated = fit_job_description(2003, ["Kuwait", "Bahrain", "Jordan"], base, 11, "cuda:0")
        self.assertEqual(first, repeated)
        for change in [dict(origin=2004), dict(countries=["Bahrain", "Jordan", "Qatar"]),
                       dict(seed=23), dict(device="cpu"), dict(base=base_grid(CONFIG)[1])]:
            args = dict(origin=2003, countries=["Bahrain", "Jordan", "Kuwait"], base=base, seed=11, device="cuda:0")
            args.update(change)
            self.assertNotEqual(first["job_id"], fit_job_description(**args)["job_id"])

    def test_source_context_restricts_pool_outcome_year_and_retains_both_sexes(self):
        panel = panel_fixture()
        config_before = deepcopy(CONFIG)
        job = fit_job_description(2003, ["Bahrain", "Jordan", "Kuwait"], base_grid(CONFIG)[0], 11, "cpu")
        working, copied = source_context(panel, CONFIG, job)
        self.assertEqual(set(working.location_name), {"Saudi Arabia", "Bahrain", "Jordan", "Kuwait"})
        self.assertEqual(set(working.sex), {"Male", "Female"})
        self.assertEqual(set(working.outcome), {"prevalence"})
        self.assertEqual(working.year.max(), 2003)
        self.assertEqual(CONFIG, config_before)
        self.assertEqual([c["name"] for c in copied["countries"]], ["Saudi Arabia", "Bahrain", "Kuwait", "Jordan"])
        bad = {**job, "countries": ["Saudi Arabia", "Bahrain"]}
        with self.assertRaises(ValueError):
            source_context(panel, CONFIG, bad)

    def test_target_sexes_use_their_respective_pools_then_keep_one_shared_base(self):
        origin, arm, base, plans, payloads = composition_fixture()
        records, corrections = compose_forecasts(CONFIG, plans, payloads, origin, arm, base,
                                                  CONFIG["models"]["tcn"]["ensemble_seeds"],
                                                  {"Male": 1, "Female": .1}, True)
        frame = pd.DataFrame(records)
        self.assertEqual(len(frame), 330)
        self.assertEqual(len(corrections), 4)
        self.assertEqual(set(frame.seed_or_ensemble), {"11|23|37|53|71"})
        for sex, expected in [("Male", .14), ("Female", .74)]:
            selected = frame.loc[frame.sex.eq(sex) & frame.family.eq("tcn_unadapted")]
            np.testing.assert_allclose(selected.log_prediction, np.log(100) + expected)
            self.assertEqual(set(selected.donor_countries), {"|".join(plans[(origin, arm, sex)]["countries"])})
        validate_reported_grid(frame, CONFIG, [{"arm": arm}], [origin])
        invalid = frame.copy()
        invalid.loc[invalid.sex.eq("Female"), "base_json"] = "different_architecture"
        with self.assertRaisesRegex(ValueError, "shared architecture"):
            validate_reported_grid(invalid, CONFIG, [{"arm": arm}], [origin])

    def test_failed_seed_preserves_that_target_sex_without_dropping_other_pool(self):
        origin, arm, base, plans, payloads = composition_fixture()
        job = fit_job_description(origin, plans[(origin, arm, "Male")]["countries"], base, 23, "cpu")
        payloads[job["job_id"]]["audit"].update(status="fallback", reason="synthetic source failure")
        records, _ = compose_forecasts(CONFIG, plans, payloads, origin, arm, base,
                                       CONFIG["models"]["tcn"]["ensemble_seeds"], {"Male": 1, "Female": 1}, True)
        frame = pd.DataFrame(records)
        self.assertTrue(frame.loc[frame.sex.eq("Male"), "status"].eq("fallback").all())
        self.assertTrue(frame.loc[frame.sex.eq("Female"), "status"].eq("ok").all())
        np.testing.assert_allclose(frame.loc[frame.sex.eq("Male"), "prediction"], 100.)
        validate_reported_grid(frame, CONFIG, [{"arm": arm}], [origin])
        with self.assertRaises(ValueError):
            validate_reported_grid(frame.drop(index=0), CONFIG, [{"arm": arm}], [origin])

    def test_scoring_keeps_arms_distinct_and_refuses_future_verification(self):
        origin, arm, base, plans, payloads = composition_fixture()
        records, _ = compose_forecasts(CONFIG, plans, payloads, origin, arm, base,
                                       CONFIG["models"]["tcn"]["ensemble_seeds"], {"Male": 1, "Female": 1}, True)
        first = pd.DataFrame(records)
        second = first.assign(arm="independent_arm")
        truth = first[["target", "outcome", "sex", "age", "forecast_year"]].drop_duplicates()
        truth["observed_rate"] = 100.
        scored = score_arms(pd.concat([first, second], ignore_index=True), truth, CONFIG, 2008)
        self.assertEqual(len(scored), 660)
        self.assertEqual(set(scored.arm), {arm, "independent_arm"})
        with self.assertRaises(ValueError):
            score_arms(first, truth, CONFIG, 2007)

    def test_shared_setting_selection_uses_completed_historical_blocks_only(self):
        rows = []
        for origin in range(2003, 2014):
            for index, base in enumerate(base_grid(CONFIG)):
                ident = "__".join(f"{key}={base[key]}" for key in ["channels", "weight_decay", "epochs"])
                for sex in CONFIG["sexes"]:
                    for penalty in CONFIG["adaptation"]["penalties"]:
                        best_penalty = .1 if sex == "Male" else 1
                        loss = (.2 if index == 1 else 1) + (0 if penalty == best_penalty else .5)
                        if origin > 2004:
                            loss = np.nan
                        rows.extend({"origin": origin, "family": "tcn_adapted", "sex": sex, "horizon": 5,
                                     "age": age, "base_id": ident, "adaptation_penalty": penalty,
                                     "absolute_log_error": loss} for age in CONFIG["ages"])
        choice = select_tcn_settings(pd.DataFrame(rows), CONFIG, 2009)
        self.assertEqual(choice["base"], base_grid(CONFIG)[1])
        self.assertEqual(choice["penalties"], {"Male": .1, "Female": 1})
        self.assertEqual(choice["inner_origins"], [2003, 2004])

    def test_real_cpu_job_cache_replays_exact_payload_and_detects_corruption(self):
        base = {**base_grid(CONFIG)[0], "epochs": 2}
        job = fit_job_description(2003, ["Bahrain", "Jordan", "Kuwait"], base, 11, "cpu")
        with tempfile.TemporaryDirectory(prefix="donor_cache_test_") as directory:
            out = Path(directory)
            (out / "cache").mkdir()
            (out / "checkpoints").mkdir()
            with patch("run_donor_comparisons.pd.read_csv", return_value=panel_fixture()):
                fitted = cached_fit_job(CONFIG, job, out)
            with patch("run_donor_comparisons.fit_seed", side_effect=AssertionError("Cache should prevent refitting")):
                repeated = cached_fit_job(CONFIG, job, out)
            self.assertEqual(joblib.hash(fitted), joblib.hash(repeated))
            self.assertEqual(fitted["audit"]["status"], "ok")
            self.assertNotIn("Saudi Arabia", set(fitted["training_meta"].country))
            self.assertEqual(set(fitted["training_meta"].sex), {"Male", "Female"})
            self.assertLessEqual(fitted["training_meta"].label_end.max(), 2003)
            path = out / "cache" / f"{job['job_id']}.joblib"
            path.write_bytes(path.read_bytes() + b"corrupted")
            with self.assertRaisesRegex(ValueError, "cached payload changed"):
                load_cached_job(out, job)

    def test_resumed_forecast_and_metadata_writes_require_exact_values(self):
        with tempfile.TemporaryDirectory(prefix="donor_commit_test_") as directory:
            frame_path, json_path = Path(directory) / "ledger.csv", Path(directory) / "choices.json"
            original = pd.DataFrame({"prediction": [1., 2.]})
            write_frame(frame_path, original)
            write_frame(frame_path, original.copy())
            with self.assertRaisesRegex(ValueError, "changed an existing committed table"):
                write_frame(frame_path, original.assign(prediction=[1., 3.]))
            write_json(json_path, {"settings": [1, 2]})
            write_json(json_path, {"settings": [1, 2]})
            with self.assertRaisesRegex(ValueError, "changed committed metadata"):
                write_json(json_path, {"settings": [1, 3]})


if __name__ == "__main__":
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(DonorTests),
                               unittest.defaultTestLoader.loadTestsFromTestCase(DonorComparisonTests)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    names = ["src/gbd_park/donors.py", "scripts/run_donor_comparisons.py", "tests/test_donors.py",
             "tests/test_donor_comparisons.py", "study_design/donor_implementation.md",
             "study_design/donor_comparisons_implementation.md", "src/gbd_park/tcn.py",
             "src/gbd_park/tcn_forecasting.py", "src/gbd_park/pooled.py", "src/gbd_park/adaptation.py",
             "src/gbd_park/scoring.py", "src/gbd_park/intervals.py", "src/gbd_park/evaluation.py"]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "device": "cpu", "fixtures": "synthetic; one two-epoch CPU cache fit",
              "tested_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    out = ROOT / "work/donor-comparison-validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
