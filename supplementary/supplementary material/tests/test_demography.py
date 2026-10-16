"""Checks for operational population forecasting and joint burden accounting."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from gbd_park.demography import (population_forecast, population_residuals, hierarchy,
                                 count_and_share_views, rate_ratio_views, joint_count_draws, shapley_change)

CONFIG = json.loads((ROOT / "study_design/locked_v1/design.json").read_text())


def fixture():
    rows = []
    for year in range(1990, 2024):
        for sex in CONFIG["sexes"]:
            for index, age in enumerate(CONFIG["ages"]):
                population = 10000*np.exp(.01*(year-1990))*(index+1)*(1 if sex == "Male" else 2)
                rate = 10*(index+1)
                rows.append(dict(location_name="Saudi Arabia", outcome="prevalence", year=year, sex=sex,
                                 age=age, rate=rate, count=rate*population/100000))
    return pd.DataFrame(rows)


class DemographyTests(unittest.TestCase):
    def test_population_future_invariance_and_exact_exponential_trend(self):
        panel = fixture()
        predicted = population_forecast(panel, CONFIG, 2018, "Saudi Arabia", "prevalence")
        changed = panel.copy()
        changed.loc[changed.year.gt(2018), ["rate", "count"]] = np.nan
        pd.testing.assert_frame_equal(predicted, population_forecast(changed, CONFIG, 2018, "Saudi Arabia", "prevalence"))
        actual = panel.loc[panel.year.eq(2023) & panel.sex.eq("Male")].set_index("age")
        final = predicted.loc[predicted.horizon.eq(5) & predicted.sex.eq("Male")].set_index("age")
        np.testing.assert_allclose(final.population, actual.loc[final.index, "count"]/actual.loc[final.index, "rate"]*100000, rtol=1e-12)

    def test_population_residual_chronology_and_pairing(self):
        panel = fixture()
        errors = population_residuals(panel, CONFIG, 2014, "Saudi Arabia", "prevalence", "log_trend_last8", list(range(2003, 2010)))
        self.assertEqual(len(errors), 7*110)
        np.testing.assert_allclose(errors.centered_population_log_error, 0, atol=1e-13)
        with self.assertRaises(ValueError):
            population_residuals(panel, CONFIG, 2014, "Saudi Arabia", "prevalence", "persistence", [2010])

    def test_hierarchy_is_complete_and_counts_and_shares_are_coherent(self):
        matrix, nodes, bottom = hierarchy(CONFIG)
        self.assertEqual(matrix.shape, (31, 22))
        np.testing.assert_array_equal(matrix[:22], np.eye(22))
        np.testing.assert_array_equal(matrix[-1], np.ones(22))
        frame = pd.DataFrame([dict(sex=s, age=a, count=1., case="synthetic") for s,a in bottom])
        counts, shares = count_and_share_views(frame, CONFIG, identifiers=["case"])
        self.assertEqual(float(counts.loc[counts.level.eq("grand_total"), "count"].iloc[0]), 22)
        np.testing.assert_allclose(shares.loc[shares.threshold.eq(65), "share"], 7/11)
        np.testing.assert_allclose(shares.loc[shares.threshold.eq(80), "share"], 4/11)
        with self.assertRaises(ValueError):
            count_and_share_views(frame.iloc[:-1], CONFIG, identifiers=["case"])

    def test_sex_ratio_is_rate_ratio_not_ratio_of_counts(self):
        frame = pd.DataFrame([dict(sex="Male", age="45-49", prediction=30), dict(sex="Female", age="45-49", prediction=10)])
        result = rate_ratio_views(frame, CONFIG, identifiers=["age"])
        self.assertEqual(result.male_female_rate_ratio.iloc[0], 3)
        with self.assertRaises(ValueError):
            rate_ratio_views(frame.iloc[:1], CONFIG, identifiers=["age"])

    def test_joint_population_rate_errors_keep_same_origin(self):
        panel = fixture()
        pop = population_forecast(panel, CONFIG, 2014, "Saudi Arabia", "prevalence")
        errors = population_residuals(panel, CONFIG, 2014, "Saudi Arabia", "prevalence", "persistence", [2003,2004])
        pop["population_method"] = "persistence"
        parts=[]
        for origin in [2003,2004]:
            x=pop.drop(columns=["population_method","population","log_population"]).copy()
            x["residual_origin"] = origin
            x["log_draw"] = np.log(100.)
            parts.append(x)
        counts=joint_count_draws(pd.concat(parts, ignore_index=True),pop,errors)
        np.testing.assert_allclose(counts["count"],np.exp(counts.log_population)/1000,rtol=1e-12)
        with self.assertRaises(ValueError):
            joint_count_draws(pd.concat(parts,ignore_index=True),pop,errors.loc[errors.residual_origin.eq(2003)])

    def test_shapley_adds_to_change_and_reverses_sign(self):
        n0=np.array([100.,200.]); n1=np.array([180.,250.])
        r0=np.array([20.,40.]); r1=np.array([25.,50.])
        parts,total=shapley_change(n0,r0,n1,r1)
        backward,negative=shapley_change(n1,r1,n0,r0)
        self.assertAlmostEqual(sum(parts.values()),total)
        self.assertAlmostEqual(total,-negative)
        for key in parts:
            self.assertAlmostEqual(parts[key],-backward[key])
        only,change=shapley_change(n0,r0,n0*2,r0)
        self.assertAlmostEqual(only['population_size'],change)
        self.assertAlmostEqual(only['age_composition'],0)
        self.assertAlmostEqual(only['rate_component'],0)

    def test_nonzero_joint_errors_are_paired_by_origin_not_row_order(self):
        panel=fixture()
        pop=population_forecast(panel,CONFIG,2014,'Saudi Arabia','prevalence')
        errors=population_residuals(panel,CONFIG,2014,'Saudi Arabia','prevalence','log_trend_last8',[2003,2004])
        errors['centered_population_log_error']=np.where(errors.residual_origin.eq(2003),-.2,.2)
        frames=[]
        for origin,delta in [(2003,-.1),(2004,.1)]:
            frame=pop.drop(columns=['population_method','population','log_population']).copy()
            frame['residual_origin']=origin
            frame['log_draw']=np.log(100)+delta
            frames.append(frame)
        draws=pd.concat(frames,ignore_index=True).sample(frac=1,random_state=4)
        actual=joint_count_draws(draws,pop,errors.sample(frac=1,random_state=5))
        combined=np.where(actual.residual_origin.eq(2003),-.3,.3)
        np.testing.assert_allclose(actual['count'],np.exp(actual.log_population+combined)/1000,rtol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
