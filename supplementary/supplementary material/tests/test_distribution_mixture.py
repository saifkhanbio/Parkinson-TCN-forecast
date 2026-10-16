"""Distribution pooling, dependence, chronology and scoring invariants."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
from gbd_park.distribution_mixture import inverse_ecdf, pool_trajectories, validate_residual_origins
import run_distribution_mixture as runner
from report_distribution_mixture import decision_gates

CONFIG = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
SPEC = json.loads((ROOT / 'study_design/distribution_mixture_v1.json').read_text())


def arrays():
    result = {}
    for i, role in enumerate(SPEC['roles']):
        point = np.exp(np.linspace(3, 5, 110).reshape(5, 22)) * (i + 1)
        rate = point[None] * np.exp(np.linspace(-.1, .1, 7)[:, None, None])
        part = {'point_rates': point, 'rate_draws': rate}
        for method in SPEC['population_methods']:
            pop = np.broadcast_to(np.linspace(10000, 90000, 22)[None], (5, 22)).copy()
            draw = pop[None] * np.exp(np.linspace(-.03, .03, 7)[:, None, None])
            part['point_populations__' + method], part['population_draws__' + method] = pop, draw
            part['point_counts__' + method], part['count_draws__' + method] = point * pop / 100000, rate * draw / 100000
        result[role] = part
    return result


class MixtureTests(unittest.TestCase):
    def test_inverse_cdf_hand_calculation_and_replication(self):
        x = np.array([2., 4., 8., 16.])
        q = [0, .1, .25, .5, .75, .8, 1]
        np.testing.assert_array_equal(inverse_ecdf(x, q), [2, 2, 2, 4, 8, 16, 16])
        np.testing.assert_array_equal(inverse_ecdf(x, q), inverse_ecdf(np.tile(x, 3), q))
        np.testing.assert_array_equal(inverse_ecdf(x, q), inverse_ecdf(x[::-1], q))
        with self.assertRaises(ValueError):
            inverse_ecdf([np.nan], [.5])
        with self.assertRaises(ValueError):
            inverse_ecdf(x, [1.1])

    def test_identical_components_do_not_change_quantiles(self):
        original = arrays()[SPEC['roles'][0]]
        parts = {role: copy.deepcopy(original) for role in SPEC['roles']}
        pooled, _ = pool_trajectories(parts, SPEC['roles'], SPEC['population_methods'], list(range(2003, 2010)), 2014)
        for field in ['rate_draws', 'count_draws__log_trend_last8']:
            np.testing.assert_array_equal(inverse_ecdf(pooled[field], [.1, .5, .9]), inverse_ecdf(original[field], [.1, .5, .9]))

    def test_pool_keeps_entire_role_trajectory_and_mass(self):
        parts = arrays()
        before = copy.deepcopy(parts)
        pooled, atoms = pool_trajectories(parts, SPEC['roles'], SPEC['population_methods'], list(range(2003, 2010)), 2014)
        self.assertEqual(len(atoms), 21)
        for i, role in enumerate(SPEC['roles']):
            np.testing.assert_array_equal(pooled['rate_draws'][7*i:7*(i+1)], parts[role]['rate_draws'])
            self.assertAlmostEqual(sum(a['mass'] for a in atoms if a['role'] == role), 1/3)
            for key in parts[role]:
                np.testing.assert_array_equal(parts[role][key], before[role][key])
        np.testing.assert_allclose(pooled['point_rates'], np.mean([p['point_rates'] for p in parts.values()], axis=0))
        # A mixture interval is not an interval around coordinatewise mean draws.
        mean_draws = np.mean([p['rate_draws'] for p in parts.values()], axis=0)
        self.assertGreater(float(inverse_ecdf(pooled['rate_draws'], [.9])[0, 0, 0]), float(inverse_ecdf(mean_draws, [.9])[0, 0, 0]))

    def test_missing_future_and_unpaired_population_blocks_rejected(self):
        for origins in [list(range(2003, 2011)), [2003]*7, list(range(2003, 2009))]:
            with self.assertRaises(ValueError):
                validate_residual_origins(origins, 2014)
        parts = arrays()
        role = SPEC['roles'][1]
        parts[role]['population_draws__log_trend_last8'] = parts[role]['population_draws__log_trend_last8'][::-1]
        parts[role]['count_draws__log_trend_last8'] = parts[role]['rate_draws'] * parts[role]['population_draws__log_trend_last8'] / 100000
        with self.assertRaises(ValueError):
            pool_trajectories(parts, SPEC['roles'], SPEC['population_methods'], list(range(2003, 2010)), 2014)

    def test_nonlinear_burden_uses_each_complete_draw(self):
        parts = arrays()
        role = SPEC['roles'][2]
        # Change only the male oldest-age cells of one role; retain count identity.
        parts[role]['rate_draws'][:, :, 7:11] *= 2
        for method in SPEC['population_methods']:
            parts[role]['count_draws__' + method] = parts[role]['rate_draws'] * parts[role]['population_draws__' + method] / 100000
        pooled, _ = pool_trajectories(parts, SPEC['roles'], SPEC['population_methods'], list(range(2003, 2010)), 2014)
        values, meta = runner.derived_values(pooled['rate_draws'], pooled['count_draws__log_trend_last8'], CONFIG)
        total = meta.index[meta.node == 'Both__45+'][0]
        share = meta.index[meta.node == 'Male__80+_within_45+'][0]
        np.testing.assert_allclose(values[:, :, total], pooled['count_draws__log_trend_last8'].sum(axis=-1))
        expected = 100 * pooled['count_draws__log_trend_last8'][:, :, 7:11].sum(-1) / pooled['count_draws__log_trend_last8'][:, :, :11].sum(-1)
        np.testing.assert_allclose(values[:, :, share], expected)
        ratios, _ = runner.derived_values(pooled['rate_draws'], None, CONFIG, True)
        np.testing.assert_allclose(ratios, pooled['rate_draws'][:, :, :11] / pooled['rate_draws'][:, :, 11:])
        np.testing.assert_array_equal(inverse_ecdf(ratios, [.5]), inverse_ecdf(pooled['rate_draws'][:, :, :11]/pooled['rate_draws'][:, :, 11:], [.5]))

    def test_quantile_ledgers_and_score_arithmetic(self):
        parts = arrays()
        pooled, _ = pool_trajectories(parts, SPEC['roles'], SPEC['population_methods'], list(range(2003, 2010)), 2014)
        ctx = dict(target='Saudi Arabia', outcome='prevalence', origin=2014, family=SPEC['candidate'])
        _, intervals, _, burden = runner.ledgers(pooled, CONFIG, SPEC, ctx, 7, 'cdf')
        r = intervals[intervals.scale.eq('rate') & intervals.level.eq(.8)]
        np.testing.assert_array_equal(r.lower.to_numpy(), inverse_ecdf(pooled['rate_draws'], [.1]).ravel())
        self.assertTrue(intervals.n_blocks.eq(7).all())
        self.assertTrue(intervals.n_draws.eq(21).all())
        self.assertEqual(len(burden), 1215)
        truth = intervals[runner.RATE_KEYS].drop_duplicates().copy()
        truth['observed_value'] = np.where(truth.scale.eq('rate'), 100., np.log(100.))
        cells, wis = runner.score_bounds(intervals, truth, runner.RATE_KEYS, 'observed_value')
        alpha = 1-cells.level
        expected = cells.upper-cells.lower + 2/alpha*(np.maximum(cells.lower-cells.observed_value, 0)+np.maximum(cells.observed_value-cells.upper, 0))
        np.testing.assert_allclose(cells.interval_score, expected)
        idx = wis.iloc[0]
        select = np.ones(len(cells), dtype=bool)
        for key in runner.RATE_KEYS:
            select &= cells[key].eq(idx[key])
        subset = cells[select].set_index('level')
        hand = (.5*abs(idx.observed_value-idx['median']) + .25*subset.loc[.5,'interval_score'] + .1*subset.loc[.8,'interval_score']) / 2.5
        self.assertAlmostEqual(idx.wis_50_80, hand)


class IssuanceTests(unittest.TestCase):
    def test_complete_issuance_without_truth_and_future_truth_perturbation(self):
        spec = dict(SPEC, evaluation_origins=[2014])
        case = dict(id='SAU_prevalence', target='Saudi Arabia', outcome='prevalence', source='source', demography='demography')
        parts = arrays()
        points, draws, old_intervals, old_burden, maps = [], [], [], [], []
        for role, a in parts.items():
            context = dict(target=case['target'], outcome=case['outcome'], origin=2014, family=role)
            p, r = runner.rate_ledgers(a, CONFIG, context, 7, 'historical_joint_blocks')
            _, b = runner.burden_ledgers(a, spec['population_methods'], CONFIG, context, 7, 'historical_joint_blocks')
            points.append(p); old_intervals.append(r); old_burden.append(b)
            d = pd.MultiIndex.from_product([range(2003, 2010), range(1, 6), CONFIG['sexes'], CONFIG['ages']], names=['residual_origin','horizon','sex','age']).to_frame(index=False)
            for k,v in context.items():d[k]=v
            d['forecast_year']=2014+d.horizon
            d['rate_draw']=a['rate_draws'].ravel();d['log_draw']=np.log(d.rate_draw)
            draws.append(d)
            for sex in CONFIG['sexes']:
                maps.append(dict(role=role,sex=sex,fit_origin=2014,source_family=role,last_selection_target_year=2014))
        p, d = pd.concat(points,ignore_index=True), pd.concat(draws,ignore_index=True)
        populations, errors = [], []
        for method in spec['population_methods']:
            pop=p[p.family.eq(spec['roles'][0])].drop(columns=['family','prediction','log_prediction']).copy()
            pop['population_method']=method
            pop['population']=parts[spec['roles'][0]]['point_populations__'+method].ravel();pop['log_population']=np.log(pop.population)
            populations.append(pop)
            err=d[d.family.eq(spec['roles'][0])].drop(columns=['family','rate_draw','log_draw']).copy()
            err['fit_origin']=2014;err['origin']=err.residual_origin;err['forecast_year']=err.origin+err.horizon
            err['population_method']=method;err['centered_population_log_error']=np.repeat(np.linspace(-.03,.03,7),110)
            errors.append(err)
        pop, err = pd.concat(populations,ignore_index=True),pd.concat(errors,ignore_index=True)
        first,_=runner.build_components(p,d,pop,err,CONFIG,spec,2014)
        # Future observations are not an issuance input; even extraneous columns are ignored.
        changed_d=d.assign(future_observed_rate=1e100);changed_err=err.assign(actual_population=1e100)
        second,_=runner.build_components(p,changed_d,pop,changed_err,CONFIG,spec,2014)
        for role in spec['roles']:
            for key in first[role]:np.testing.assert_array_equal(first[role][key],second[role][key])
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);(temp/'source').mkdir();(temp/'demography').mkdir();(temp/'output').mkdir()
            for name,frame in [('predictions.csv',p),('joint_draws.csv',d),('intervals.csv',pd.concat(old_intervals)),('champion_family_mappings.csv',pd.DataFrame(maps))]:frame.to_csv(temp/'source'/name,index=False)
            for name,frame in [('population_forecasts.csv',pop),('population_residuals.csv',err),('intervals.csv',pd.concat(old_burden))]:frame.to_csv(temp/'demography'/name,index=False)
            original_read=runner.read
            def guarded_read(path):
                self.assertNotIn('regional_outcomes',str(path))
                return original_read(path)
            with patch.object(runner,'ROOT',temp),patch.object(runner,'read',side_effect=guarded_read):
                result=runner.issue_case(case,CONFIG,spec,temp/'output')
            self.assertTrue(result['passed'])
            commit=json.loads((temp/'output/SAU_prevalence/issued_commit.json').read_text())
            self.assertFalse(commit['evaluation_scored'])
            self.assertEqual(len(json.loads((temp/'output/SAU_prevalence/draw_index.json').read_text())),4)


class DecisionTests(unittest.TestCase):
    def test_joint_rule_rejects_female_harm_even_when_male_improves(self):
        rates=[]
        families=['tcn_adapted__cdf','local_champion__cdf','nonneural_champion__cdf',SPEC['candidate']]
        for sex in ['Male','Female']:
            for scope in ['45+','80+']:
                for scale in ['rate','log_rate']:
                    for family in families:
                        mix=family==SPEC['candidate']
                        rates.append(dict(horizon=5,sex=sex,age_scope=scope,scale=scale,family=family,
                                          wis=.9 if mix else 1.,coverage=.8 if mix else .6,mean_width=1.,point_ale=.1))
        burden=[]
        for family in ['tcn_adapted__cdf',SPEC['candidate']]:
            mix=family==SPEC['candidate']
            for node in SPEC['assessment']['guarded_count_nodes']+SPEC['assessment']['guarded_share_nodes']:
                burden.append(dict(horizon=5,family=family,population_method='log_trend_last8',node=node,
                                   measure='age_share' if 'within' in node else 'count',wis=.9 if mix else 1.,coverage=.8 if mix else .6,mean_width=1.))
            for age in CONFIG['ages']:
                burden.append(dict(horizon=5,family=family,population_method='not_applicable',node='Male_Female__'+age,
                                   measure='sex_rate_ratio',wis=.9 if mix else 1.,coverage=.8 if mix else .6,mean_width=1.))
        rates,burden=pd.DataFrame(rates),pd.DataFrame(burden)
        good=decision_gates(rates,burden,SPEC)
        self.assertEqual(len(good),38)
        self.assertTrue(good.passed.all())
        rates.loc[rates.family.eq(SPEC['candidate'])&rates.sex.eq('Female')&rates.scale.eq('rate'),'wis']=1.2
        bad=decision_gates(rates,burden,SPEC)
        self.assertFalse(bad.passed.all())
        self.assertTrue((bad[~bad.passed].endpoint.str.startswith('Female')).all())
        # A nominal-coverage gain alone cannot excuse worse interval score.
        self.assertTrue(bad[bad.gate.eq('coverage_distance')].passed.all())


if __name__ == '__main__':
    unittest.main()
