"""Distribution transformation and scoring checks on an analytic joint panel."""
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tests'))
import numpy as np
import pandas as pd
from test_demography import CONFIG
from run_demography import burden_statistics,ratio_statistics,distribution_summary,score_distributions,STAT


class RunnerTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(target='Saudi Arabia',outcome='prevalence',origin=2018,family='synthetic',
             horizon=5,forecast_year=2023,population_method='persistence',sex=sex,age=age,
             count=(index+1)*(2 if sex=='Male' else 1),prediction=30 if sex=='Male' else 10)
             for sex in CONFIG['sexes'] for index,age in enumerate(CONFIG['ages'])])

    def test_joint_transform_quantiles_and_wis(self):
        frame=self.frame()
        points=burden_statistics(frame,CONFIG)
        self.assertEqual(len(points),35)
        self.assertAlmostEqual(points.loc[points.node.eq('Both__45+'),'value'].iloc[0],198)
        draws=[]
        for origin,multiplier in enumerate([.8,.9,1.,1.1,1.2],2003):
            copy=frame.copy();copy['count']*=multiplier;copy['residual_origin']=origin
            draws.append(burden_statistics(copy,CONFIG,True))
        draws=pd.concat(draws,ignore_index=True)
        intervals=distribution_summary(draws)
        self.assertEqual(len(intervals),105)
        subset=intervals.loc[intervals.node.eq('Both__45+')&intervals.level.eq(.8)].iloc[0]
        self.assertAlmostEqual(subset.lower,198*.84)
        self.assertAlmostEqual(subset.upper,198*1.16)
        truth=points[STAT+['value']].rename(columns={'value':'observed'})
        scores,wis=score_distributions(intervals,truth)
        self.assertTrue(scores.covered.all())
        expected=(.25*(198*.2)+.1*(198*.32))/2.5
        self.assertAlmostEqual(wis.loc[wis.node.eq('Both__45+'),'wis_50_80'].iloc[0],expected)
        self.assertTrue((intervals.loc[intervals.measure.eq('age_share'),'upper']<=100).all())

    def test_sex_ratio_transformation_and_duplicate_block_rejection(self):
        frame=self.frame()
        ratios=ratio_statistics(frame,CONFIG)
        np.testing.assert_array_equal(ratios.value,3.)
        frame['rate_draw']=frame.prediction
        frame['residual_origin']=2003
        ratios=ratio_statistics(frame,CONFIG,True)
        with self.assertRaises(ValueError):
            distribution_summary(pd.concat([ratios,ratios],ignore_index=True))


if __name__=='__main__':
    unittest.main(verbosity=2)
