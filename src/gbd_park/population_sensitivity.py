"""Deterministic population scenarios applied to frozen age-specific forecasts."""

import numpy as np
import pandas as pd

from .demography import hierarchy


CELL = ['target', 'outcome', 'origin', 'horizon', 'forecast_year', 'sex', 'age']
CONTEXT = ['target', 'outcome', 'origin', 'horizon', 'forecast_year', 'family', 'scenario']
NODE = ['measure', 'node', 'sex', 'age_group', 'unit', 'hierarchy_level']
SCENARIOS = ['log_trend_last8', 'persistence', 'gbd_realized_oracle',
             'un_2024_unaligned', 'un_2024_origin_aligned']
ROLES = ['tcn_adapted', 'local_champion', 'nonneural_champion']


def positive(frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Missing, nonpositive or nonfinite values: ' + ', '.join(columns))


def populations(panel, un, operational, config, target, outcome):
    """Historical comparisons; the last three scenarios explicitly use future data."""
    origins = config['calendar']['reliability_origins']
    horizons = config['calendar']['horizons']
    keys = ['sex', 'age', 'year']
    truth = panel.loc[panel.location_name.eq(target) & panel.outcome.eq(outcome),
                      keys + ['rate', 'count']].copy()
    demo = un.loc[un.location_name.eq(target), keys + ['population_persons']].copy()
    if truth.duplicated(keys).any() or demo.duplicated(keys).any():
        raise ValueError('Duplicate source population cells')
    positive(truth, ['rate', 'count'])
    positive(demo, ['population_persons'])
    truth['gbd_population'] = truth['count'] / truth.rate * 100000
    grid = pd.MultiIndex.from_product([origins, horizons, config['sexes'], config['ages']],
                                      names=['origin', 'horizon', 'sex', 'age']).to_frame(index=False)
    grid['forecast_year'] = grid.origin + grid.horizon
    grid['target'], grid['outcome'] = target, outcome
    for time, prefix in [('origin', 'origin'), ('forecast_year', 'future')]:
        grid = grid.merge(truth[keys + ['gbd_population']].rename(
            columns={'year': time, 'gbd_population': prefix + '_gbd'}),
            on=['sex', 'age', time], how='left', validate='many_to_one')
        grid = grid.merge(demo.rename(columns={'year': time, 'population_persons': prefix + '_un'}),
                          on=['sex', 'age', time], how='left', validate='many_to_one')
    positive(grid, ['origin_gbd', 'future_gbd', 'origin_un', 'future_un'])
    frames = []
    for scenario in SCENARIOS:
        frame = grid.copy()
        if scenario in SCENARIOS[:2]:
            saved = operational.loc[operational.population_method.eq(scenario), CELL + ['population']]
            if len(saved) != len(grid) or saved.duplicated(CELL).any():
                raise ValueError('Incomplete or duplicate operational populations')
            frame = frame.merge(saved, on=CELL, how='left', validate='one_to_one')
            role = 'historical_operational'
        elif scenario == 'gbd_realized_oracle':
            frame['population'] = frame.future_gbd
            role = 'realized_population_oracle'
        elif scenario == 'un_2024_unaligned':
            frame['population'] = frame.future_un
            role = 'later_vintage_UN_source_scenario'
        else:
            frame['population'] = frame.origin_gbd * frame.future_un / frame.origin_un
            role = 'later_vintage_UN_growth_scenario'
        positive(frame, ['population'])
        frame['scenario'], frame['population_information_role'] = scenario, role
        frame['uses_future_population_information'] = scenario not in SCENARIOS[:2]
        frame['alignment_year'] = frame.origin if scenario == 'un_2024_origin_aligned' else np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate(cells, config, identifiers=CONTEXT, value='count', allow_signed=False):
    """Vectorized count nodes and shares; each row group must have all 22 cells."""
    matrix, nodes, bottom = hierarchy(config)
    keys = list(identifiers)
    if cells.duplicated(keys + ['sex', 'age']).any():
        raise ValueError('Duplicate age-sex cells')
    if set(zip(cells.sex, cells.age)) != set(bottom):
        raise ValueError('Unexpected or absent age-sex cells')
    pivot = cells.pivot(index=keys, columns=['sex', 'age'], values=value)
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(bottom))
    numbers = pivot.to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (not allow_signed and (numbers <= 0).any()):
        raise ValueError('Incomplete or invalid age-sex cells')
    arrays = numbers @ matrix.T
    frames = []
    for index, row in nodes.iterrows():
        frame = pivot.index.to_frame(index=False)
        frame['value'] = arrays[:, index]
        frame['measure'], frame['unit'] = 'count', 'modeled_number'
        for column in ['node', 'sex', 'age_group']:
            frame[column] = row[column]
        frame['hierarchy_level'] = row['level']
        frames.append(frame)
    if not allow_signed:
        for sex in config['sexes']:
            mask = np.array([s == sex for s, a in bottom])
            denominator = numbers[:, mask].sum(axis=1)
            for threshold in [65, 80]:
                high = np.array([s == sex and int(a.split('-')[0].rstrip('+')) >= threshold
                                 for s, a in bottom])
                frame = pivot.index.to_frame(index=False)
                frame['value'] = 100 * numbers[:, high].sum(axis=1) / denominator
                frame['measure'], frame['unit'], frame['sex'] = 'age_share', 'percent', sex
                frame['age_group'] = f'{threshold}+_within_45+'
                frame['node'] = sex + '__' + frame.age_group
                frame['hierarchy_level'] = 'sex_total'
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)[keys + NODE + ['value']]


def conditional_intervals(draw_statistics):
    """Quantiles of intact frozen rate blocks with deterministic population inputs."""
    keys = CONTEXT + NODE
    if draw_statistics.duplicated(keys + ['residual_origin']).any():
        raise ValueError('Duplicate residual-block statistics')
    block_sets = draw_statistics.groupby(keys).residual_origin.agg(frozenset).reset_index()
    for _, part in block_sets.groupby(['target', 'outcome', 'origin', 'family', 'scenario']):
        if len(set(part.residual_origin)) != 1:
            raise ValueError('Inconsistent joint residual-block identities')
    positive(draw_statistics, ['value'])
    group = draw_statistics.groupby(keys, sort=True).value
    quantiles = group.quantile([.025, .1, .25, .5, .75, .9, .975], interpolation='linear').unstack()
    n = group.size()
    if n.min() < 5:
        raise ValueError('Insufficient joint historical blocks')
    frames = []
    for level, lo, hi in [(.5, .25, .75), (.8, .1, .9), (.95, .025, .975)]:
        frame = pd.DataFrame({'lower': quantiles[lo], 'median': quantiles[.5],
                              'upper': quantiles[hi], 'n_blocks': n}).reset_index()
        frame['level'], frame['uncertainty_scope'] = level, 'rate_conditional_fixed_population'
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def error_accounting(frame):
    """Exact arithmetic decomposition of point-count errors, without causal claims."""
    positive(frame, ['prediction', 'observed_rate', 'population', 'observed_population'])
    result = frame.copy()
    rhat, rate = result.prediction, result.observed_rate
    nhat, pop = result.population, result.observed_population
    result['predicted_count'] = rhat * nhat / 100000
    result['observed_count'] = rate * pop / 100000
    result['log_rate_error'] = np.log(rhat / rate)
    result['log_population_error'] = np.log(nhat / pop)
    result['log_count_error'] = np.log(result.predicted_count / result.observed_count)
    result['rate_effect'] = (rhat - rate) * (nhat + pop) / 200000
    result['population_effect'] = (nhat - pop) * (rhat + rate) / 200000
    np.testing.assert_allclose(result.log_count_error, result.log_rate_error + result.log_population_error,
                               rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(result.rate_effect + result.population_effect,
                               result.predicted_count - result.observed_count, rtol=1e-10, atol=1e-9)
    return result


def projection_populations(panel, un, config):
    """Locked population-only handoff, aligned at 2023, without disease forecasts."""
    countries = [c['name'] for c in config['countries'] if c['gcc']]
    origin = config['calendar']['projection_origin']
    keys = ['location_name', 'sex', 'age']
    baseline = panel.loc[panel.location_name.isin(countries) & panel.year.eq(origin)
                         & panel.outcome.isin(['prevalence', 'incidence']), keys + ['outcome', 'rate', 'count']].copy()
    if baseline.duplicated(keys + ['outcome']).any() or un.duplicated(keys + ['year']).any():
        raise ValueError('Duplicate population projection keys')
    expected_base = {(c, s, a, o) for c in countries for s in config['sexes'] for a in config['ages']
                     for o in ['prevalence', 'incidence']}
    if set(map(tuple, baseline[keys + ['outcome']].to_numpy())) != expected_base:
        raise ValueError('Incomplete age-sex population baseline')
    positive(baseline, ['rate', 'count'])
    baseline['gbd_baseline'] = baseline['count'] / baseline.rate * 100000
    baseline = baseline.merge(un.loc[un.year.eq(origin), keys + ['population_persons']].rename(
        columns={'population_persons': 'un_baseline'}), on=keys, how='left', validate='many_to_one')
    future = un.loc[un.year.isin(config['calendar']['projection_years']) & un.location_name.isin(countries),
                    keys + ['year', 'population_persons']].rename(columns={'population_persons': 'un_future'})
    expected_future = {(c, s, a, y) for c in countries for s in config['sexes'] for a in config['ages']
                       for y in config['calendar']['projection_years']}
    if set(map(tuple, future[keys + ['year']].to_numpy())) != expected_future:
        raise ValueError('Incomplete age-sex population projection grid')
    joined = baseline.merge(future, on=keys, how='left', validate='many_to_many')
    expected = len(countries) * 2 * len(config['sexes']) * len(config['ages']) * len(config['calendar']['projection_years'])
    if len(joined) != expected:
        raise ValueError('Incomplete population projection grid')
    positive(joined, ['gbd_baseline', 'un_baseline', 'un_future'])
    frames = []
    for scenario in ['un_medium_unaligned', 'gbd_2023_aligned_un_growth']:
        frame = joined.copy()
        frame['scenario'] = scenario
        frame['population'] = frame.un_future if scenario == 'un_medium_unaligned' else frame.gbd_baseline * frame.un_future / frame.un_baseline
        frame['origin'], frame['unit'] = origin, 'persons'
        frame['role'] = 'population_only_handoff_not_disease_forecast'
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
