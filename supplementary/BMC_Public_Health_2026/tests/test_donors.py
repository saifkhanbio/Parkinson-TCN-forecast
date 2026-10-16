"""Analytic and temporal checks for prespecified donor-country selection."""

import os
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from gbd_park.donors import select_donors, TRAJECTORY_NAMES

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def panel_fixture():
    return pd.DataFrame([
        {"location_name": country["name"], "sex": sex, "age": age,
         "year": year, "outcome": outcome,
         "rate": np.exp(2 + c * .03 + s * .1 + a * (.1 + c * .003)
                        + c * .002 * (year - 1996) + (c + 1) * .0001 * (year - 1996) ** 2)}
        for c, country in enumerate(CONFIG["countries"])
        for s, sex in enumerate(CONFIG["sexes"])
        for a, age in enumerate(CONFIG["ages"])
        for year in range(1990, 2024) for outcome in ["prevalence", "incidence"]])


class DonorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = panel_fixture()

    def choose(self, panel=None, **kwargs):
        return select_donors(self.panel if panel is None else panel, CONFIG, 2003,
                             "Saudi Arabia", "Male", "three_similar", **kwargs)

    def test_closed_form_similarity_features(self):
        chosen = self.choose()
        for row in chosen["feature_audit"]["rows"]:
            c = next(i for i, country in enumerate(CONFIG["countries"]) if country["name"] == row["country"])
            quadratic = (c + 1) * .0001
            expected = [2 + c * .03 + 5 * (.1 + c * .003) + c * .002 * 7 + quadratic * 49,
                        c * .002 + quadratic * 7,
                        2 * quadratic * np.std(np.arange(1, 8), ddof=1),
                        c * .002 * 3 + quadratic * 33]
            np.testing.assert_allclose([row["trajectory"][name] for name in TRAJECTORY_NAMES], expected, atol=2e-15)
            np.testing.assert_allclose(list(row["centered_log_age_profile"].values()),
                                       (np.arange(11) - 5) * (.1 + c * .003), atol=2e-15)
        self.assertEqual(chosen["minimum_feature_year"], 1996)
        self.assertEqual(chosen["maximum_feature_year"], 2003)

    def test_future_other_sex_and_other_outcome_values_do_not_change_selection(self):
        original = self.choose()
        changed = self.panel.copy()
        unused = changed.year.gt(2003) | changed.year.lt(1996) | changed.sex.eq("Female") | changed.outcome.eq("incidence")
        changed.loc[unused, "rate"] = np.nan
        changed = pd.concat([changed, changed.loc[unused].iloc[:1]], ignore_index=True)
        self.assertEqual(self.choose(changed), original)

    def test_fixed_pools_exclude_target_and_supply_both_sexes(self):
        for target in [c["name"] for c in CONFIG["countries"] if c["gcc"]]:
            for strategy, size in [("all_six_other_regional_countries", 6), ("other_five_gcc", 5)]:
                results = [select_donors(pd.DataFrame(), CONFIG, 2003, target, sex, strategy)
                           for sex in CONFIG["sexes"]]
                self.assertEqual(results[0]["countries"], results[1]["countries"])
                self.assertEqual(len(results[0]["countries"]), size)
                self.assertNotIn(target, results[0]["countries"])
                self.assertEqual(results[0]["contributing_sexes"], ["Male", "Female"])
                self.assertIsNone(results[0]["maximum_feature_year"])
                if strategy == "other_five_gcc":
                    self.assertNotIn("Jordan", results[0]["countries"])
        with self.assertRaises(ValueError):
            select_donors(self.panel, CONFIG, 2003, "Jordan", "Male", "other_five_gcc")

    def test_identical_features_tie_by_gbd_id_without_zero_division(self):
        panel = self.panel.copy()
        panel["rate"] = np.exp(2.)
        chosen = self.choose(panel)
        expected = sorted([c for c in CONFIG["countries"] if c["name"] != "Saudi Arabia"], key=lambda c: c["gbd_id"])
        self.assertEqual(chosen["countries"], [c["name"] for c in expected[:3]])
        self.assertTrue(all(row["distance"] == 0 for row in chosen["distance_table"]))
        for block in ["trajectory_scaling", "profile_scaling"]:
            self.assertEqual(chosen["feature_audit"][block]["retained_components"], 0)
            self.assertEqual(chosen["feature_audit"][block]["weight"], .5)

    def test_scaling_is_donor_only_and_equal_weighted_between_blocks(self):
        chosen = self.choose()
        audit = chosen["feature_audit"]
        names = [row["country"] for row in audit["rows"]]
        target = names.index("Saudi Arabia")
        donors = [i for i, name in enumerate(names) if name != "Saudi Arabia"]
        trajectory = np.array([[row["trajectory"][name] for name in TRAJECTORY_NAMES] for row in audit["rows"]])
        profile = np.array([list(row["centered_log_age_profile"].values()) for row in audit["rows"]])
        expected = []
        for values, block in [(trajectory, "trajectory_scaling"), (profile, "profile_scaling")]:
            scaling = audit[block]
            np.testing.assert_allclose(scaling["median"], np.median(values[donors], axis=0))
            iqr = np.diff(np.quantile(values[donors], [.25, .75], axis=0, method="linear"), axis=0)[0]
            np.testing.assert_allclose(scaling["iqr"], iqr)
            keep = np.array(scaling["retained"])
            expected.append(np.mean(((values[donors][:, keep] - values[target, keep]) / iqr[keep]) ** 2, axis=1))
        table = pd.DataFrame(chosen["distance_table"]).set_index("country").reindex([names[i] for i in donors])
        np.testing.assert_allclose(table.distance, .5 * expected[0] + .5 * expected[1])
        # Outlying target values cannot enter the six-donor normalization fit.
        changed = self.panel.copy()
        changed.loc[changed.location_name.eq("Saudi Arabia"), "rate"] *= np.exp(3)
        altered = self.choose(changed)["feature_audit"]
        self.assertEqual(audit["trajectory_scaling"], altered["trajectory_scaling"])
        self.assertEqual(audit["profile_scaling"], altered["profile_scaling"])

    def test_random_pools_are_fixed_seed_deterministic_and_order_independent(self):
        pool = sorted([c for c in CONFIG["countries"] if c["name"] != "Saudi Arabia"], key=lambda c: c["gbd_id"])
        for seed in CONFIG["donors"]["random_seeds"]:
            chosen = select_donors(pd.DataFrame(), CONFIG, 2003, "Saudi Arabia", "Male", "three_random", seed=seed)
            indices = sorted(np.random.default_rng(seed).choice(6, size=3, replace=False))
            self.assertEqual(chosen["countries"], [pool[i]["name"] for i in indices])
            reordered = deepcopy(CONFIG)
            reordered["countries"].reverse()
            repeated = select_donors(pd.DataFrame(), reordered, 2003, "Saudi Arabia", "Male", "three_random", seed=seed)
            self.assertEqual(chosen, repeated)
            self.assertEqual(len(set(chosen["countries"])), 3)
        for seed in [None, 11, True, 101.0]:
            with self.assertRaises(ValueError):
                select_donors(self.panel, CONFIG, 2003, "Saudi Arabia", "Male", "three_random", seed=seed)

    def test_similarity_is_invariant_to_row_and_configuration_order(self):
        original = self.choose()
        changed = deepcopy(CONFIG)
        changed["countries"].reverse()
        actual = select_donors(self.panel.sample(frac=1, random_state=11), changed, 2003,
                               "Saudi Arabia", "Male", "three_similar")
        self.assertEqual(actual, original)

    def test_similarity_rejects_missing_duplicate_or_nonpositive_source_cells(self):
        eligible = self.panel.index[self.panel.year.eq(2000) & self.panel.sex.eq("Male")
                                    & self.panel.outcome.eq("prevalence")][0]
        with self.assertRaises(ValueError):
            self.choose(self.panel.drop(index=eligible))
        with self.assertRaises(ValueError):
            self.choose(pd.concat([self.panel, self.panel.loc[[eligible]]], ignore_index=True))
        invalid = self.panel.copy()
        invalid.loc[eligible, "rate"] = 0
        with self.assertRaises(ValueError):
            self.choose(invalid)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DonorTests))
    files = [ROOT / name for name in ["src/gbd_park/donors.py", "tests/test_donors.py", "study_design/donor_implementation.md"]]
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors),
              "tested_code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    output = ROOT / "work/donor-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
