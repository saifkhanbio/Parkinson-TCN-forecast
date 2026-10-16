"""Synthetic checks for source matching and interpretation-critical arithmetic."""
import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("release_sensitivity", Path(__file__).resolve().parents[1] / "scripts/run_release_sensitivity.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def fixture():
    old, new = [], []
    for location in MOD.COUNTRIES:
        for measure in MOD.MEASURES:
            for year, point in [(1990, 100.), (2019, 200.)]:
                for metric in MOD.METRICS:
                    old.append(dict(location=location, measure=measure, sex="Both",
                                    year_or_period=str(year), metric=metric, value=point))
                    new.append(dict(location=location, measure=measure, year=year, metric=metric,
                                    new_value=point * (1.5 if year == 1990 else 1.25),
                                    new_count_lower=0. if metric == "Number" else np.nan,
                                    new_count_upper=1000. if metric == "Number" else np.nan))
    return pd.DataFrame(old), pd.DataFrame(new)


class ReleaseComparisonTests(unittest.TestCase):
    def test_same_year_contrast_and_endpoint_growth_are_distinct(self):
        old, new = fixture()
        points = MOD.compare_points(old, new)
        row = points[points.year.eq(2019)].iloc[0]
        self.assertEqual(row.absolute_difference, 50)
        self.assertEqual(row.relative_difference_percent, 25)
        self.assertAlmostEqual(row.log_ratio, math.log(1.25))
        published = pd.DataFrame([dict(location=l, measure=m, metric="Percent change in ASR",
                                       value=99.9, original_cell_text="99.9(90,110)")
                                  for l in MOD.COUNTRIES for m in MOD.MEASURES])
        endpoints = MOD.endpoint_changes(points, published)
        e = endpoints[endpoints.metric.eq("Age-standardized rate")].iloc[0]
        self.assertEqual(e.old_point_endpoint_change_percent, 100)
        self.assertAlmostEqual(e.new_point_endpoint_change_percent, 100 * (250 / 150 - 1))
        self.assertEqual(e.published_old_asr_change_percent, 99.9)

    def test_duplicate_keys_fail(self):
        old, new = fixture()
        with self.assertRaises((ValueError, pd.errors.MergeError)):
            MOD.compare_points(old, pd.concat([new, new.iloc[:1]]))

    def test_missing_country_cell_fails(self):
        old, new = fixture()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            MOD.compare_points(old, new.iloc[1:])

    def test_crude_rate_cannot_replace_asr(self):
        old, new = fixture()
        new.loc[new.metric.eq("Age-standardized rate"), "metric"] = "Rate"
        with self.assertRaisesRegex(ValueError, "metric"):
            MOD.compare_points(old, new)

    def test_sex_mismatch_fails(self):
        old, new = fixture()
        old.loc[0, "sex"] = "Male"
        with self.assertRaisesRegex(ValueError, "both-sex"):
            MOD.compare_points(old, new)

    def test_nonpositive_point_fails(self):
        old, new = fixture()
        new.loc[0, "new_value"] = 0
        with self.assertRaisesRegex(ValueError, "Nonpositive"):
            MOD.compare_points(old, new)

    def test_count_bounds_not_accepted_for_asr(self):
        old, new = fixture()
        new.loc[new.metric.eq("Age-standardized rate"), "new_count_lower"] = 0
        with self.assertRaisesRegex(ValueError, "bounds assigned to ASR"):
            MOD.compare_points(old, new)

    def test_rounding_envelope_is_metric_specific(self):
        points = MOD.compare_points(*fixture())
        self.assertTrue(np.allclose(points.loc[points.metric.eq("Number"), "conditional_rounding_difference_halfwidth"], 1))
        self.assertTrue(np.allclose(points.loc[points.metric.eq("Age-standardized rate"), "conditional_rounding_difference_halfwidth"], .055))


if __name__ == "__main__":
    unittest.main()
