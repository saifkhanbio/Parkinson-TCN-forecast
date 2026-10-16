"""Scientific invariants for the committed Saudi demographic extension outputs."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/saudi_raw_integration_v1'
DATA=ROOT/'data/processed/saudi_raw_integration_v1'


class SaudiRawIntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells=pd.read_csv(REPORT/'conditional_projection_cells_2024_2028.csv')
        cls.national=pd.read_csv(DATA/'national_population_2022_2024.csv')

    def test_all_scenario_rates_equal_saved_forecasts(self):
        keys=['sex','age','year','family']
        for outcome in ['prevalence','incidence']:
            saved=pd.read_csv(ROOT/f'results/projections_v1/trials/SAU_{outcome}/predictions.csv')
            saved=saved.rename(columns={'forecast_year':'year','prediction':'saved_prediction'})
            actual=self.cells.loc[self.cells.outcome.eq(outcome)]
            checked=actual.merge(saved[keys+['saved_prediction']],on=keys,validate='many_to_one')
            self.assertEqual(len(checked),len(actual))
            np.testing.assert_allclose(checked.prediction,checked.saved_prediction,rtol=1e-14)

    def test_national_2024_scenario_preserves_source_at_anchor(self):
        p=self.cells.loc[self.cells.scenario.eq('national_2024_un_growth')&self.cells.year.eq(2024)]
        nat=self.national.loc[self.national.year.eq(2024)&self.national.age_start.ge(45)]
        expected=nat.groupby(['sex','age']).population.sum().sort_index()
        for _,g in p.groupby(['outcome','family','within_80plus_allocation']):
            got=g.groupby(['sex','broad_age']).population.sum().sort_index()
            self.assertEqual(list(got.index),list(expected.index))
            np.testing.assert_allclose(got.values,expected.values,rtol=1e-13)

    def test_allocation_only_changes_unobserved_within_80plus_mix(self):
        p=self.cells.loc[self.cells.scenario.str.startswith('national')]
        keys=['outcome','year','family','scenario','sex','age','broad_age']
        a=p.loc[p.within_80plus_allocation.eq('gbd_aligned_within_80plus')]
        b=p.loc[p.within_80plus_allocation.eq('un_within_80plus')]
        joined=a.merge(b,on=keys,validate='one_to_one',suffixes=('_a','_b'))
        young=joined.loc[joined.broad_age.ne('80+')]
        np.testing.assert_allclose(young.population_a,young.population_b,rtol=1e-13)
        np.testing.assert_allclose(young.count_a,young.count_b,rtol=1e-13)
        by=keys[:-2]
        sums=joined.groupby(by)[['population_a','population_b']].sum()
        np.testing.assert_allclose(sums.population_a,sums.population_b,rtol=1e-13)

    def test_age_linkage_cannot_treat_65plus_as_65_to_69(self):
        linked=pd.read_csv(DATA/'population_2022_corrected_linkage.csv')
        flagged=linked.loc[linked.source_flag.eq('GCC 65-69 value equals 65+ total')]
        self.assertEqual(len(flagged),4)
        self.assertTrue(flagged.age.eq('65-69').all())
        for row in flagged.itertuples():
            same=linked.loc[linked.sex.eq(row.sex)&linked.nationality.eq(row.nationality)&linked.age_start.ge(65)]
            self.assertEqual(same.population.sum(),row.population_persons_original)
            self.assertLess(row.population,row.population_persons_original)

    def test_count_sensitivity_stays_within_allocation_limits(self):
        frame=pd.read_csv(REPORT/'national_denominator_count_sensitivity_summary.csv')
        self.assertEqual(len(frame),24)
        self.assertTrue((frame.national_count_45plus>=frame.allocation_total_min-1e-8).all())
        self.assertTrue((frame.national_count_45plus<=frame.allocation_total_max+1e-8).all())
        self.assertTrue((frame.national_80plus_share>=frame.allocation_80plus_share_min-1e-8).all())
        self.assertTrue((frame.national_80plus_share<=frame.allocation_80plus_share_max+1e-8).all())

    def test_insurance_metrics_and_ambiguous_ages_remain_separate(self):
        summary=pd.read_csv(REPORT/'chi_quarter_composition.csv')
        self.assertEqual(len(summary),14)
        q=summary.loc[summary.file.str.contains('Q2-2024',regex=False)].set_index('metric')
        self.assertEqual(q.loc['insured_persons','total'],12381657)
        self.assertEqual(q.loc['subscribers','total'],12424219)
        ambiguous=summary.loc[summary.file.str.contains('Q4-2025',regex=False)]
        self.assertTrue(ambiguous.ambiguous_age_label.all())
        self.assertTrue(ambiguous.over_65_percent.isna().all())
        compare=pd.read_csv(REPORT/'chi_2024_population_composition_comparison.csv')
        for _,g in compare.groupby('metric'):
            self.assertAlmostEqual(g.chi_composition_percent.sum(),100)
            self.assertAlmostEqual(g.national_composition_percent.sum(),100)


if __name__=='__main__':unittest.main()
