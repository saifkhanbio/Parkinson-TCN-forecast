"""Exploratory demographic accounting for the JDR reporting extension."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/JDR_GBD_PARK/analysis'


def shapley(p0, p1, r0, r1):
    """Average six replacement orders of size, composition and rate factors."""
    n = [p0.sum(), p1.sum()]
    s = [p0 / n[0], p1 / n[1]]
    r = [r0, r1]
    def burden(state):
        return n[state[0]] * np.dot(s[state[1]], r[state[2]]) / 1e5
    contributions = np.zeros(3)
    for order in itertools.permutations(range(3)):
        state = [0, 0, 0]
        for component in order:
            before = burden(state)
            state[component] = 1
            contributions[component] += (burden(state) - before) / 6
    c0, c1 = burden([0, 0, 0]), burden([1, 1, 1])
    np.testing.assert_allclose(contributions.sum(), c1-c0, rtol=1e-12, atol=1e-9)
    # Independent closed-form Shapley weights for the population-size term.
    closed = (n[1]-n[0]) * (np.dot(s[0], r[0])/3 + np.dot(s[1], r[0])/6
                           + np.dot(s[0], r[1])/6 + np.dot(s[1], r[1])/3) / 1e5
    np.testing.assert_allclose(contributions[0], closed, rtol=1e-12, atol=1e-9)
    return c0, c1, contributions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = ['reports/forecast_percentage_change_v1/scenario_baseline_2023_age_cells.csv',
              'reports/saudi_raw_integration_v1/conditional_projection_cells_2024_2028.csv',
              'forecast_percentage_change.csv', 'data/processed/design_v1/regional_outcomes.csv']
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    base, future, percentages, regional = [pd.read_csv(ROOT/p) for p in inputs]
    future = future.loc[future.family.eq('tcn_adapted')]
    rows = []
    for keys, f in future.groupby(['outcome', 'year', 'scenario', 'within_80plus_allocation']):
        outcome, year, scenario, allocation = keys
        b = base.loc[base.outcome.eq(outcome) & base.scenario.eq(scenario)
                     & base.within_80plus_allocation.eq(allocation)]
        merged = f.merge(b, on=['sex', 'age'], validate='one_to_one', suffixes=('_future', '_base'))
        assert len(merged) == 22
        for sex in ['Male','Female','Both']:
            m = merged if sex == 'Both' else merged.loc[merged.sex.eq(sex)]
            p0, p1 = m.population_base.to_numpy(), m.population_future.to_numpy()
            r0, r1 = m.rate.to_numpy(), m.prediction.to_numpy()
            c0, c1, components = shapley(p0,p1,r0,r1)
            check = percentages.loc[percentages.outcome.eq(outcome) & percentages.sex.eq(sex)
                    & percentages.forecast_year.eq(year) & percentages.population_scenario.eq(scenario)
                    & percentages.within_80plus_allocation.eq(allocation)].iloc[0]
            np.testing.assert_allclose([c0,c1], [check.baseline_count_2023,check.forecast_count], rtol=1e-11)
            row = dict(outcome=outcome,year=year,sex=sex,scenario=scenario,allocation=allocation,
                       baseline_count=c0,forecast_count=c1,total_change=c1-c0,
                       total_change_percent=100*(c1/c0-1),population_2023=p0.sum(),population_future=p1.sum(),
                       role='exploratory_post_primary_reporting_analysis',
                       composition_definition='age-sex composition' if sex=='Both' else 'age composition')
            for key, value in zip(['population_size','composition','rates'],components):
                row[key+'_count_contribution'] = value
                row[key+'_percentage_point_contribution'] = 100*value/c0
                row[key+'_share_of_signed_change_percent'] = 100*value/(c1-c0) if c1 != c0 else np.nan
            rows.append(row)
    d = pd.DataFrame(rows)
    assert len(d)==180
    d.to_csv(OUT/'forecast_growth_decomposition.csv',index=False)
    # Non-causal boundary checks: zero change, size-only, rate-only, composition-only.
    p=np.array([100.,200.]);r=np.array([10.,30.])
    np.testing.assert_allclose(shapley(p,p,r,r)[2],0,atol=1e-12)
    np.testing.assert_allclose(shapley(p,2*p,r,r)[2][1:],0,atol=1e-12)
    np.testing.assert_allclose(shapley(p,p,r,2*r)[2][:2],0,atol=1e-12)
    c=shapley(p,p[::-1],r,r)[2]
    np.testing.assert_allclose(c[[0,2]],0,atol=1e-12)
    sa=regional.loc[regional.location_name.eq('Saudi Arabia') & regional.year.eq(2023)].copy()
    sa['broad_age']=np.select([sa.age_start.lt(65),sa.age_start.lt(80)],['45–64','65–79'],default='80+')
    summary=sa.groupby(['outcome','sex','broad_age'],as_index=False)['count'].sum()
    totals=summary.groupby(['outcome','sex'])['count'].transform('sum')
    summary['share_within_45plus_percent']=100*summary['count']/totals
    summary.to_csv(OUT/'saudi_2023_burden_composition.csv',index=False)
    assert hashes == {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    validation=dict(status='passed',new_models_fitted=0,decomposition_rows=len(d),
        exact_additivity=True,closed_form_crosscheck=True,synthetic_boundary_checks=4,
        original_forecasts_reproduced=True,source_sha256=hashes,
        interpretation='Order-averaged arithmetic contributions; not causal attribution or uncertainty intervals.')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    selected=d.loc[d.year.eq(2028)&d.scenario.eq('gbd_2023_aligned_un_growth')]
    print(selected[['outcome','sex','total_change_percent','population_size_percentage_point_contribution',
                    'composition_percentage_point_contribution','rates_percentage_point_contribution']].round(5).to_string(index=False))


if __name__=='__main__': main()
