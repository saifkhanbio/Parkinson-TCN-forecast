"""Meaningful bounded-ASR geography, leakage, fit, ensemble and ledger checks."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import pandas as pd

from gbd_park import global_asr as asr
from gbd_park.pooled import build_examples, balanced_weights, target_inputs
from gbd_park.tcn import load_checkpoint, predict_changes, state_fingerprint
import run_global_asr as runner

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def synthetic():
    rows, un = [], []
    countries = CONFIG["countries"] + [{"name": "Synthetic extra territory", "gbd_id": 999, "iso3": "SYN", "un_id": 9999}]
    for index, country in enumerate(countries):
        un.append({"LocID": country["un_id"], "ISO3_code": country["iso3"], "Location": country["name"],
                   "LocTypeID": 4, "LocTypeName": "Country/Area", "ParentID": 99999})
        for year in range(1990, 2024):
            row = {"location_id": country["gbd_id"], "location_name": country["name"], "year": year}
            for outcome in asr.OUTCOMES:
                for sex in ["male", "female"]:
                    scale = 1 if outcome == "prevalence" else .13
                    row[f"{outcome}_rate_age_std_{sex}"] = scale * np.exp(3.5 + index * .08 + .003 * (year-1990)
                        + .01 * np.sin((year-1990)/4 + index) + (.12 if sex == "male" else 0))
            rows.append(row)
    wide = pd.DataFrame(rows)
    registry = asr.location_registry(wide, pd.DataFrame(un))
    return wide, pd.DataFrame(un), registry, asr.asr_panel(wide)


def full_predictions(panel):
    rows = []
    views = [("local", family) for family in asr.LOCAL_FAMILIES]
    views += [(scope, family) for scope in asr.SCOPES for family in asr.LEARNED_FAMILIES]
    for task in asr.tasks(CONFIG):
        for scope, family in views:
            for sex in CONFIG["sexes"]:
                base = panel.loc[panel.location_name.eq(task["target"]) & panel.outcome.eq(task["outcome"])
                                 & panel.sex.eq(sex) & panel.year.eq(2018), "rate"].item()
                for horizon in range(1, 6):
                    rows.append({"target": task["target"], "outcome": task["outcome"], "donor_scope": scope,
                        "family": family, "sex": sex, "horizon": horizon, "origin": 2018, "forecast_year": 2018+horizon,
                        "age": asr.AGE, "unit": "ASR_per_100000", "prediction": base, "log_prediction": np.log(base),
                        "status": "ok"})
    return pd.DataFrame(rows)


class GlobalASRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wide, cls.un, cls.registry, cls.panel = synthetic()

    def test_full_source_registry_and_only_asr_fields(self):
        changed = self.wide.copy()
        changed["prevalence_count_male"] = -99999
        changed["prevalence_rate_male"] = np.nan
        pd.testing.assert_frame_equal(asr.asr_panel(changed), self.panel)
        self.assertEqual(self.panel.shape[0], 8 * 34 * 4)
        self.assertTrue(self.registry.un_location_type.eq("Country/Area").all())
        self.assertEqual(self.registry.iso3.nunique(), 8)

    def test_reject_aggregates_unknown_aliases_duplicates_and_bad_rates(self):
        un = self.un.copy(); un.loc[0, "LocTypeName"] = "Geographic region"
        with self.assertRaises(ValueError): asr.location_registry(self.wide, un)
        un = self.un.copy(); un.loc[0, "Location"] = "Unknown label"
        with self.assertRaises(ValueError): asr.location_registry(self.wide, un)
        un = self.un.copy(); un.loc[0, "ISO3_code"] = un.loc[1, "ISO3_code"]
        with self.assertRaises(ValueError): asr.location_registry(self.wide, un)
        for bad in [self.wide.iloc[1:], pd.concat([self.wide, self.wide.iloc[:1]], ignore_index=True)]:
            with self.assertRaises(ValueError): asr.asr_panel(bad)
        bad = self.wide.copy(); bad.loc[0, "incidence_rate_age_std_female"] = 0
        with self.assertRaises(ValueError): asr.asr_panel(bad)

    def test_alias_mapping_and_irrelevant_aggregate_name_ambiguity(self):
        wide = self.wide.copy(); un = self.un.copy()
        wide.loc[wide.location_name.eq("Synthetic extra territory"), "location_name"] = "Taiwan"
        un.loc[un.Location.eq("Synthetic extra territory"), "Location"] = "China, Taiwan Province of China"
        # UN legitimately repeats some aggregate names under distinct hierarchies.
        extra = pd.DataFrame([dict(LocID=1, ISO3_code=None, Location="Unused aggregate", LocTypeID=2,
                                   LocTypeName="Geographic region", ParentID=0),
                              dict(LocID=2, ISO3_code=None, Location="Unused aggregate", LocTypeID=12,
                                   LocTypeName="SDG region", ParentID=0)])
        registry = asr.location_registry(wide, pd.concat([un, extra], ignore_index=True))
        matched = registry.loc[registry.location_name.eq("Taiwan")].iloc[0]
        self.assertEqual(matched.match_method, "explicit_alias")
        self.assertEqual(matched.un_location_name, "China, Taiwan Province of China")

    def test_fixed_jobs_scope_counts_and_no_age_smoother(self):
        jobs = runner.jobs(CONFIG, "cuda:0")
        self.assertEqual(len(asr.tasks(CONFIG)), 12)
        self.assertEqual(len(jobs), 156)
        self.assertEqual(sum(job["kind"] == "tcn" for job in jobs), 120)
        self.assertEqual(len({str(runner.job_path(Path("/tmp"), job)) for job in jobs}), 156)
        self.assertNotIn("age_smooth_trend", asr.LOCAL_FAMILIES)
        self.assertEqual(asr.fixed_base(CONFIG), dict(channels=16, weight_decay=.001, epochs=50))
        self.assertEqual(len(asr.LOCAL_FAMILIES) + 2*len(asr.LEARNED_FAMILIES), 22)

    def test_context_removes_future_other_outcomes_and_excludes_target_from_source(self):
        original = deepcopy(CONFIG)
        for scope in asr.SCOPES:
            work, cfg = asr.context(self.panel, CONFIG, self.registry, "Saudi Arabia", "incidence", scope)
            changed = self.panel.copy()
            changed.loc[changed.outcome.ne("incidence") | changed.year.gt(2018), "rate"] = np.nan
            perturbed, _ = asr.context(changed, CONFIG, self.registry, "Saudi Arabia", "incidence", scope)
            pd.testing.assert_frame_equal(work, perturbed)
            self.assertEqual(work.year.max(), 2018)
            self.assertTrue(work.source_outcome.eq("incidence").all())
            donors = [country["name"] for country in cfg["countries"] if country["name"] != cfg["primary_target"]]
            x, y, meta = build_examples(work, cfg, 2018, donors)
            self.assertFalse(meta.country.eq("Saudi Arabia").any())
            self.assertEqual(len(meta), (7 if scope == "global" else 6) * 2 * 17)
            self.assertEqual(x.shape[1], 11)
            self.assertEqual(int(meta.label_end.max()), 2018)
            weights = balanced_weights(meta)
            self.assertTrue(np.isfinite(weights).all())
            masses = meta.assign(weight=weights).groupby(["country", "sex"]).weight.sum()
            np.testing.assert_allclose(masses, masses.iloc[0])
        self.assertEqual(CONFIG, original)

    def test_incomplete_history_and_invalid_context_fail_closed(self):
        missing = self.panel.drop(self.panel.loc[self.panel.year.eq(2000) & self.panel.outcome.eq("prevalence")].index[0])
        with self.assertRaises(ValueError): asr.context(missing, CONFIG, self.registry, "Saudi Arabia", "prevalence", "global")
        with self.assertRaises(ValueError): asr.context(self.panel, CONFIG, self.registry, "Saudi Arabia", "deaths", "global")
        with self.assertRaises(ValueError): asr.context(self.panel, CONFIG, self.registry, "Saudi Arabia", "prevalence", "global", 2017)

    def test_actual_tcn_preprocessing_checkpoint_and_future_invariance(self):
        work, cfg = asr.context(self.panel, CONFIG, self.registry, "Saudi Arabia", "prevalence", "regional")
        base = dict(channels=16, weight_decay=.001, epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.joblib"
            fitted = asr.fit_seed(work, cfg, 11, "cpu", path, base)
            self.assertEqual(fitted["audit"]["status"], "ok")
            self.assertEqual(fitted["audit"]["parameter_count"], 1732)
            self.assertEqual(fitted["audit"]["fingerprint_before"], fitted["audit"]["fingerprint_after"])
            self.assertEqual(fitted["audit"]["target_windows"], 34)
            loaded = load_checkpoint(path)
            cx, _, _ = target_inputs(work, cfg, 2018, "Saudi Arabia")
            np.testing.assert_array_equal(predict_changes(loaded, cx), fitted["current_changes"])
            changed = self.panel.copy()
            changed.loc[changed.year.gt(2018) | changed.outcome.ne("prevalence"), "rate"] = 1e12
            same_work, _ = asr.context(changed, CONFIG, self.registry, "Saudi Arabia", "prevalence", "regional")
            same = asr.fit_seed(same_work, cfg, 11, "cpu", base=base)
            self.assertEqual(fitted["audit"]["fingerprint_before"], same["audit"]["fingerprint_before"])
            # Arbitrary target changes leave source weights and scaler identical.
            changed = work.copy(); changed.loc[changed.location_name.eq("Saudi Arabia"), "rate"] *= 10
            other = asr.fit_seed(changed, cfg, 11, "cpu", base=base)
            self.assertEqual(fitted["audit"]["fingerprint_before"], other["audit"]["fingerprint_before"])

    def test_actual_seed_ensemble_frozen_adaptation_missing_seed_and_fallback(self):
        work, cfg = asr.context(self.panel, CONFIG, self.registry, "Saudi Arabia", "incidence", "regional")
        cfg["models"]["tcn"]["ensemble_seeds"] = [11, 23]
        payloads = [asr.fit_seed(work, cfg, seed, "cpu", base=dict(channels=16, weight_decay=.001, epochs=2)) for seed in [11, 23]]
        rows, corrections = asr.ensemble_forecasts(payloads, cfg, "incidence", "regional")
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows.parameter_count.unique().tolist(), [1732, 1734, 1733])
        self.assertEqual(rows.fitted_seed_parameters.unique().tolist(), [3464])
        self.assertTrue(all(row["last_target_label_year"] == 2018 for row in corrections))
        self.assertTrue(all(row["target_windows"] == 17 for row in corrections))
        with self.assertRaises(ValueError): asr.ensemble_forecasts(payloads[:1], cfg, "incidence", "regional")
        payloads[0]["audit"].update(status="fallback", reason="synthetic failed seed")
        fallback, _ = asr.ensemble_forecasts(payloads, cfg, "incidence", "regional")
        self.assertTrue(fallback.status.eq("fallback").all())
        for sex in cfg["sexes"]:
            level = payloads[0]["levels"][cfg["sexes"].index(sex)]
            np.testing.assert_allclose(fallback.loc[fallback.sex.eq(sex), "log_prediction"], level)

    def test_actual_non_neural_fit_source_metadata_and_models(self):
        work, cfg = asr.context(self.panel, CONFIG, self.registry, "Qatar", "incidence", "global")
        with tempfile.TemporaryDirectory() as directory:
            value = asr.nonneural_forecasts(work, cfg, "incidence", "global", Path(directory))
            self.assertEqual(len(value["predictions"]), 60)
            self.assertEqual(len(list(Path(directory).glob("*.joblib"))), 4)
            self.assertTrue(value["predictions"].outcome.eq("incidence").all())
            self.assertTrue(value["predictions"].donor_strategy.str.startswith("global_").all())
            for fit in value["fits"]:
                self.assertEqual("Qatar" in fit["countries"], fit["pool"] == "pooled")
                self.assertEqual(fit["maximum_label_year"], 2018)
                self.assertEqual(fit["base_fingerprint_before"], fit["base_fingerprint_after"])
            self.assertEqual(len(value["adaptation"]), 4)

    def test_actual_local_families_and_correct_outcome(self):
        work, cfg = asr.context(self.panel, CONFIG, self.registry, "Bahrain", "incidence", "regional")
        # Bound the synthetic ARIMA grid; the production configuration is unchanged.
        cfg["models"]["arima"]["p"] = [0, 1]
        cfg["models"]["arima"]["q"] = [0]
        result = asr.local_forecasts(work, cfg, "incidence")
        self.assertEqual(len(result["predictions"]), 40)
        self.assertEqual(set(result["predictions"].family), set(asr.LOCAL_FAMILIES))
        self.assertTrue(result["predictions"].outcome.eq("incidence").all())
        self.assertTrue(result["predictions"].history_end.eq(2018).all())
        self.assertTrue(result["arima"])

    def test_full_ledger_scores_analytic_errors_and_missing_cells(self):
        predictions = full_predictions(self.panel)
        asr.check_forecasts(predictions, CONFIG)
        self.assertEqual(len(predictions), 2640)
        truth = self.panel.copy()
        for sex in CONFIG["sexes"]:
            for outcome in asr.OUTCOMES:
                for target in [country["name"] for country in CONFIG["countries"] if country["gcc"]]:
                    base = self.panel.loc[self.panel.location_name.eq(target) & self.panel.outcome.eq(outcome)
                                          & self.panel.sex.eq(sex) & self.panel.year.eq(2018), "rate"].item()
                    truth.loc[truth.location_name.eq(target) & truth.outcome.eq(outcome) & truth.sex.eq(sex)
                              & truth.year.gt(2018), "rate"] = base * np.exp(.1)
        scored = asr.score_forecasts(predictions, truth, CONFIG)
        np.testing.assert_allclose(scored.absolute_log_error, .1, atol=1e-12)
        np.testing.assert_allclose(scored.absolute_rate_error, scored.prediction * (np.exp(.1) - 1), atol=1e-12)
        with self.assertRaises(ValueError): asr.score_forecasts(predictions.iloc[1:], truth, CONFIG)
        with self.assertRaises(ValueError): asr.score_forecasts(predictions.loc[predictions.target.eq("Saudi Arabia")], truth, CONFIG)

    def test_global_commit_required_and_tamper_rejected_before_scoring(self):
        points = full_predictions(self.panel)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            points.to_csv(out / "predictions.csv", index=False)
            self.panel.to_csv(out / "source_panel.csv.gz", index=False)
            with self.assertRaises(FileNotFoundError): runner.score_all(out, CONFIG)
            commit = {"final_period_scored": False, "cases": [task["id"] for task in asr.tasks(CONFIG)],
                      "output_sha256": runner.hashes(out, [out / "predictions.csv", out / "source_panel.csv.gz"])}
            runner.write_json(out / "issued_commit.json", commit)
            runner.score_all(out, CONFIG)
            self.assertEqual(len(pd.read_csv(out / "point_scores.csv")), 2640)
            self.assertEqual(len(pd.read_csv(out / "donor_scope_comparisons.csv")), 1080)
            self.assertEqual(len(pd.read_csv(out / "adaptation_comparisons.csv")), 960)
            (out / "predictions.csv").write_text((out / "predictions.csv").read_text() + "\n")
            with self.assertRaises(ValueError): runner.score_all(out, CONFIG)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GlobalASRTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "elapsed_seconds": time.perf_counter()-started,
              "tested_code_sha256": {name: asr.digest(ROOT / name) for name in runner.REQUIRED + runner.DEPENDENCIES},
              "failures": [name.id() for name, _ in result.failures], "errors": [name.id() for name, _ in result.errors]}
    runner.write_json(ROOT / runner.TESTS, report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
