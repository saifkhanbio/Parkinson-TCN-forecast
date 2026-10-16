"""Independent Gaussian-conditioning, chronology and predictive coherence checks."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gbd_park.bayesian_age_time import age_kernel, fit, mixture_quantiles, joint_log_draws, training_matrix


class BayesianAgeTimeTests(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads((ROOT/'study_design/bounded_extension_2026-10-01.json').read_text())['model']
        self.one = dict(self.spec, age_lengthscales=[1.], observation_sd=[.005],
                        level_innovation_sd=[.005], slope_innovation_sd=[.001], damping=[.95])
        self.y = np.arange(10)[:, None]*np.array([[.015, -.01, .003]])+np.array([[2., 3., 4.]])

    def test_independent_full_matrix_kalman_equivalence(self):
        result = fit(self.y, self.one)
        k = age_kernel(3, 1.)
        identity = np.eye(3); phi = .95
        transition = np.block([[identity, phi*identity], [np.zeros((3,3)), phi*identity]])
        process = np.block([[.005**2*k, np.zeros((3,3))], [np.zeros((3,3)), .001**2*k]])
        observation = np.column_stack([identity, np.zeros((3,3))]); noise=.005**2*identity
        state=np.r_[self.y[0],np.zeros(3)]
        cov=np.block([[noise,np.zeros((3,3))],[np.zeros((3,3)),.02**2*k]])
        loglike=0.
        for y in self.y[1:]:
            state=transition@state; cov=transition@cov@transition.T+process
            innovation=y-observation@state; variance=observation@cov@observation.T+noise
            loglike-=.5*(3*np.log(2*np.pi)+np.linalg.slogdet(variance)[1]+innovation@np.linalg.solve(variance,innovation))
            gain=np.linalg.solve(variance,observation@cov).T
            state=state+gain@innovation; cov=cov-gain@observation@cov
        self.assertAlmostEqual(loglike,result['log_likelihood'][0],places=10)
        for h in range(5):
            state=transition@state; cov=transition@cov@transition.T+process
            np.testing.assert_allclose(result['means'][0,h],state[:3],atol=1e-12)
            np.testing.assert_allclose(result['variances'][0,h],np.diag(cov[:3,:3]+noise),atol=1e-12)

    def test_quantiles_equal_analytic_normal_for_single_component(self):
        r=fit(self.y,self.one); q=[.025,.1,.5,.9,.975]
        np.testing.assert_allclose(mixture_quantiles(r,q),r['means'][0,...,None]+np.sqrt(r['variances'][0,...,None])*norm.ppf(q),atol=1e-12)

    def test_joint_simulation_matches_analytic_predictive_moments(self):
        r=fit(self.y,self.one); draws=joint_log_draws(r,60000,152)
        np.testing.assert_allclose(draws.mean(axis=0),r['means'][0],atol=.0005)
        np.testing.assert_allclose(draws.var(axis=0),r['variances'][0],rtol=.035)
        self.assertGreater(np.corrcoef(draws[:,0,0],draws[:,4,0])[0,1],.2)
        self.assertGreater(np.corrcoef(draws[:,4,0],draws[:,4,1])[0,1],.15)

    def test_future_poison_cannot_affect_training(self):
        rows=[dict(location_name='Saudi Arabia',outcome='prevalence',sex='Male',year=y,age=a,rate=10.+y-1990)
              for y in range(1990,2001) for a in ['45-49','50-54']]
        panel=pd.DataFrame(rows)
        expected=training_matrix(panel,'Saudi Arabia','prevalence','Male',1999,['45-49','50-54'])
        panel.loc[panel.year.gt(1999),'rate']=np.nan
        actual=training_matrix(panel,'Saudi Arabia','prevalence','Male',1999,['45-49','50-54'])
        np.testing.assert_array_equal(expected,actual)

    def test_scale_equivariance_and_complete_discrete_prior(self):
        a=fit(self.y,self.spec); b=fit(self.y+4.,self.spec)
        self.assertEqual(len(a['grid']),162)
        np.testing.assert_allclose(a['weights'],b['weights'],atol=1e-10)
        np.testing.assert_allclose(mixture_quantiles(a,[.1,.5,.9])+4.,mixture_quantiles(b,[.1,.5,.9]),atol=1e-10)


if __name__=='__main__':
    unittest.main()
