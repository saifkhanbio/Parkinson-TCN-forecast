"""Checks for native-count forecasting and reconciliation."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
import pandas as pd
from test_demography import CONFIG, fixture
from gbd_park.count_coherence import node_history, independent_counts, reconciliation_weights, reconcile_counts
from gbd_park.demography import hierarchy


class CountTests(unittest.TestCase):
    def test_history_and_future_invariance(self):
        panel = fixture()
        first, _ = independent_counts(panel, CONFIG, 2018, 'Saudi Arabia', 'prevalence')
        panel.loc[panel.year.gt(2018), 'count'] = np.nan
        second, _ = independent_counts(panel, CONFIG, 2018, 'Saudi Arabia', 'prevalence')
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 155)
        history = node_history(panel, CONFIG, 'Saudi Arabia', 'prevalence', 2018)
        np.testing.assert_allclose(history.iloc[:, -1], history.iloc[:, :22].sum(axis=1))

    def test_nonnegative_coherent_and_weighted_objective(self):
        matrix, _, _ = hierarchy(CONFIG)
        values = np.arange(1., 32.)
        variance = np.arange(1., 32.)**2
        result = reconcile_counts(values, variance, CONFIG)
        for key in ['bottom_up', 'nonnegative_diagonal_wls']:
            np.testing.assert_allclose(result[key], matrix @ result[key][:22])
            self.assertTrue((result[key] >= 0).all())
        loss = lambda x: np.sum((x-values)**2/variance)
        self.assertLessEqual(loss(result['nonnegative_diagonal_wls']), loss(result['bottom_up'])+1e-10)
        coherent = matrix @ np.arange(1.,23.)
        np.testing.assert_allclose(reconcile_counts(coherent, variance, CONFIG)['nonnegative_diagonal_wls'], coherent, rtol=1e-12)

    def test_weights_center_floor_and_exclude_future(self):
        history = node_history(fixture(), CONFIG, 'Saudi Arabia', 'prevalence', 2014)
        rows = [dict(origin=o, node=n, horizon=h, count_residual=100. if o<=2009 else 1e12)
                for o in range(2003,2014) for n in history.columns for h in range(1,6)]
        errors = pd.DataFrame(rows)
        weight, audit = reconciliation_weights(errors, history, CONFIG, 2014, 5)
        np.testing.assert_allclose(weight, 1e-8*history.mean().to_numpy()**2)
        self.assertEqual(audit['residual_blocks'],7)
        self.assertEqual(audit['last_residual_label_year'],2014)
        other, _ = reconciliation_weights(errors.loc[errors.origin.le(2009)], history, CONFIG, 2014,5)
        np.testing.assert_array_equal(weight,other)
        fallback, audit = reconciliation_weights(errors.loc[errors.origin.le(2005)],history,CONFIG,2014,5)
        np.testing.assert_array_equal(fallback,np.ones(31))
        self.assertEqual(audit['weight_status'],'equal_weight_fallback')
        with self.assertRaises(ValueError):
            reconciliation_weights(errors,history,CONFIG,2013,5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
