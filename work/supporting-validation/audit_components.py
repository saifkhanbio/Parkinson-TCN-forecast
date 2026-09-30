"""Independent paired-component, historical-ratio and operational-burden replay.

This audit imports no new production module and performs no model fitting.
"""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_rates import (ROOT, CONFIG, PANEL, KEY, COORD, read, sha, same, truth_for,
                         score_points, independent_quantiles, interval_scores, wis_scores)

ROLES = ['tcn_adapted', 'local_champion', 'nonneural_champion']
RATE_KEY = ['target', 'outcome', 'procedure'] + KEY
BURDEN = ['target', 'outcome', 'origin', 'horizon', 'forecast_year', 'family', 'procedure', 'population_method']
NODE = ['measure', 'node', 'sex', 'age_group', 'unit', 'hierarchy_level']
STAT = BURDEN + NODE
BOTTOM = list(itertools.product(CONFIG['sexes'], CONFIG['ages']))
MATRIX, NODES = [], []
for index, (sex, age) in enumerate(BOTTOM):
    MATRIX.append(np.eye(22)[index])
    NODES.append(dict(node=sex+'__'+age, sex=sex, age_group=age, hierarchy_level='age_sex'))
for sex in CONFIG['sexes']:
    for label, low, high in [('45-64', 45, 64), ('65-79', 65, 79), ('80+', 80, 200)]:
        MATRIX.append([float(s == sex and low <= int(a[:2]) <= high) for s, a in BOTTOM])
        NODES.append(dict(node=sex+'__'+label, sex=sex, age_group=label, hierarchy_level='broad_age_sex'))
for sex in CONFIG['sexes']:
    MATRIX.append([float(s == sex) for s, _ in BOTTOM])
    NODES.append(dict(node=sex+'__45+', sex=sex, age_group='45+', hierarchy_level='sex_total'))
MATRIX.append(np.ones(22))
NODES.append(dict(node='Both__45+', sex='Both', age_group='45+', hierarchy_level='grand_total'))
MATRIX = np.asarray(MATRIX)


def native_ratios(target):
    source = PANEL.loc[PANEL.location_name.eq(target) & PANEL.outcome.isin(['prevalence', 'ylds'])]
    rows = []
    for origin in range(2003, 2019):
        sums = source.loc[source.year.between(origin-7, origin)].groupby(['sex', 'age', 'outcome']).rate.sum().unstack()
        assert len(sums) == 22
        ratio = (sums.ylds/sums.prevalence).rename('training_ratio').reset_index()
        ratio['origin'], ratio['ratio_history_start'], ratio['ratio_history_end'] = origin, origin-7, origin
        rows.append(ratio)
    return pd.concat(rows, ignore_index=True)


def ratio_points(prevalence, ratios):
    result = prevalence.merge(ratios, on=['origin', 'sex', 'age'], validate='many_to_one')
    assert len(result) == len(prevalence)
    result['prevalence_prediction'] = result.prediction
    result['prediction'] *= result.training_ratio
    result['log_prediction'] = np.log(result.prediction)
    result['outcome'], result['procedure'] = 'ylds', 'prevalence_training_ratio'
    return result


def component_sum(left, right, draw=False):
    keys = ['target']+KEY+(['residual_origin'] if draw else [])
    a = left.set_index(keys).sort_index()
    b = right.set_index(keys).sort_index()
    assert a.index.equals(b.index)
    result = a.index.to_frame(index=False)
    result['outcome'], result['procedure'] = 'dalys', 'component_sum'
    result['prediction'] = a.prediction.to_numpy()+b.prediction.to_numpy()
    result['log_prediction'] = np.log(result.prediction)
    for suffix, table in [('yld', a), ('yll', b)]:
        result['source_family_'+suffix] = table.source_family.to_numpy()
    if draw:
        result['rate_draw'] = a.rate_draw.to_numpy()+b.rate_draw.to_numpy()
        result['log_draw'] = np.log(result.rate_draw)
    return result


def aggregate(cells, draw=False):
    keys = BURDEN+(['residual_origin'] if draw else [])
    assert not cells.duplicated(keys+['sex', 'age']).any()
    pivot = cells.pivot(index=keys, columns=['sex', 'age'], values='count').reindex(columns=pd.MultiIndex.from_tuples(BOTTOM))
    values = pivot.to_numpy()
    assert np.isfinite(values).all() and (values > 0).all()
    summed = values @ MATRIX.T
    result = []
    for index, node in enumerate(NODES):
        part = pivot.index.to_frame(index=False)
        for key, value in node.items():
            part[key] = value
        part['measure'] = 'count'
        part['unit'] = np.where(part.outcome.eq('deaths'), 'modeled_events', 'modeled_burden_years')
        part['value'] = summed[:, index]
        result.append(part)
    for sex in CONFIG['sexes']:
        denom = values[:, [s == sex for s, _ in BOTTOM]].sum(axis=1)
        for threshold in [65, 80]:
            numer = values[:, [s == sex and int(a[:2]) >= threshold for s, a in BOTTOM]].sum(axis=1)
            part = pivot.index.to_frame(index=False)
            part['measure'], part['unit'], part['hierarchy_level'] = 'age_share', 'percent', 'sex_total'
            part['sex'], part['age_group'] = sex, f'{threshold}+_within_45+'
            part['node'] = sex+'__'+part.age_group
            part['value'] = 100*numer/denom
            result.append(part)
    return pd.concat(result, ignore_index=True)


def burden_quantiles(statistics):
    frames = []
    for origin, part in statistics.groupby('origin'):
        matrix = part.pivot(index=STAT, columns='residual_origin', values='value')
        assert set(matrix.columns) == set(range(2003, int(origin)-4))
        assert np.isfinite(matrix.to_numpy()).all()
        q = np.quantile(matrix.to_numpy(), [.025, .1, .25, .5, .75, .9, .975], axis=1, method='linear')
        for level, lo, hi in [(.5, 2, 4), (.8, 1, 5), (.95, 0, 6)]:
            frame = matrix.index.to_frame(index=False)
            frame['level'], frame['lower'], frame['median'], frame['upper'] = level, q[lo], q[3], q[hi]
            frame['n_blocks'] = len(matrix.columns)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_population(iso, target):
    """Independently reconstruct common operational population forecasts/errors."""
    directory = ROOT / 'results/demography_v1' / (iso+'_prevalence')
    saved = read(directory / 'population_forecasts.csv')
    saved_errors = read(directory / 'population_residuals.csv')
    source = PANEL.loc[PANEL.location_name.eq(target) & PANEL.outcome.eq('prevalence')].copy()
    source['population'] = source['count']/source.rate*100000
    indexed = source.set_index(['sex', 'age', 'year']).population
    rows = []
    for origin in range(2003, 2019):
        for sex, age in BOTTOM:
            history = np.log([indexed.loc[(sex, age, year)] for year in range(origin-7, origin+1)])
            x = np.arange(-7, 1)
            slope = np.sum((x-x.mean())*(history-history.mean())) / np.sum((x-x.mean())**2)
            intercept = history.mean()-slope*x.mean()
            for method in ['log_trend_last8', 'persistence']:
                for h in range(1, 6):
                    pred = intercept+slope*h if method == 'log_trend_last8' else history[-1]
                    rows.append(dict(origin=origin, sex=sex, age=age, horizon=h, forecast_year=origin+h,
                                     population_method=method, log_population=pred, population=np.exp(pred),
                                     raw_population_log_error=np.log(indexed.loc[(sex, age, origin+h)])-pred))
    full = pd.DataFrame(rows)
    current = full.loc[full.origin.ge(2014)].copy()
    popkeys = ['origin', 'sex', 'age', 'horizon', 'forecast_year', 'population_method']
    same(saved, current, popkeys, ['population', 'log_population'], atol=2e-7, rtol=2e-11)
    errors = []
    for fit_origin in range(2014, 2019):
        part = full.loc[full.origin.le(fit_origin-5)].copy()
        part['fit_origin'], part['residual_origin'] = fit_origin, part.origin
        part['centered_population_log_error'] = part.raw_population_log_error-part.groupby(
            ['sex', 'age', 'horizon', 'population_method']).raw_population_log_error.transform('mean')
        errors.append(part)
    errors = pd.concat(errors, ignore_index=True)
    same(saved_errors, errors, ['fit_origin', 'residual_origin', 'sex', 'age', 'horizon', 'population_method'],
         ['raw_population_log_error', 'centered_population_log_error'], atol=3e-13, rtol=3e-10)
    return saved, saved_errors


def audit_country(task):
    out, core, country = task
    started = time.perf_counter()
    iso, target = country['iso3'], country['name']
    directory = out / iso
    direct_p, direct_d = {}, {}
    for outcome in ['deaths', 'ylds', 'ylls', 'dalys']:
        source = core / 'trials' / (iso+'_'+outcome)
        direct_p[outcome] = read(source / 'predictions.csv')
        direct_d[outcome] = read(source / 'joint_draws.csv')
    saved_points = read(directory / 'rate_predictions.csv')
    saved_draws = read(directory / 'rate_draws.csv.gz')
    saved_intervals = read(directory / 'rate_intervals.csv')
    sum_points = component_sum(direct_p['ylds'], direct_p['ylls'])
    sum_draws = component_sum(direct_d['ylds'], direct_d['ylls'], True)
    same(saved_points.loc[saved_points.procedure.eq('component_sum')], sum_points, RATE_KEY, ['prediction', 'log_prediction'])
    same(saved_draws.loc[saved_draws.procedure.eq('component_sum')], sum_draws, RATE_KEY+['residual_origin'], ['rate_draw', 'log_draw'])
    for observed, expected in [(saved_points, sum_points), (saved_draws, sum_draws)]:
        keys = RATE_KEY+(['residual_origin'] if 'residual_origin' in expected else [])
        a = observed.loc[observed.procedure.eq('component_sum')].set_index(keys).sort_index()
        b = expected.set_index(keys).sort_index()
        for col in ['source_family_yld', 'source_family_yll']:
            assert a[col].equals(b[col])
    previous = ROOT / ('results/primary_v1' if iso == 'SAU' else 'results/secondary_v1/trials/'+iso+'_prevalence')
    history_dir = ROOT / 'results/intervals_v1' if iso == 'SAU' else previous
    prev_points = read(previous / 'predictions.csv')
    prev_history = read(history_dir / 'prequential_predictions.csv')
    mapping = read(previous / 'champion_family_mappings.csv')
    ratios = native_ratios(target)
    ratio_history = ratio_points(prev_history, ratios)
    same(read(directory / 'ratio_prequential_predictions.csv'), ratio_history, RATE_KEY,
         ['prediction', 'log_prediction', 'training_ratio', 'ratio_history_start', 'ratio_history_end'])
    ratio_current = ratio_points(prev_points.loc[prev_points.family.isin(ROLES)], ratios)
    same(saved_points.loc[saved_points.procedure.eq('prevalence_training_ratio')], ratio_current, RATE_KEY,
         ['prediction', 'log_prediction', 'training_ratio', 'ratio_history_start', 'ratio_history_end'])
    history_scores = score_points(ratio_history, truth_for(target, 'ylds'))
    ratio_draws = []
    coords = pd.MultiIndex.from_product([CONFIG['sexes'], CONFIG['ages'], range(1, 6)], names=COORD)
    for origin in range(2014, 2019):
        origins = list(range(2003, origin-4))
        for role in ROLES:
            by_sex = ({sex: role for sex in CONFIG['sexes']} if role == 'tcn_adapted' else
                      mapping.loc[mapping.fit_origin.eq(origin) & mapping.role.eq(role)].set_index('sex').source_family.to_dict())
            index = pd.MultiIndex.from_product([origins, CONFIG['sexes'], CONFIG['ages'], range(1, 6)], names=['origin']+COORD)
            source = pd.concat([history_scores.loc[history_scores.sex.eq(sex) & history_scores.family.eq(family)
                      & history_scores.origin.isin(origins)] for sex, family in by_sex.items()]).set_index(['origin']+COORD).reindex(index)
            residuals = source.log_residual.to_numpy().reshape(len(origins), 110)
            centered = residuals-residuals.mean(axis=0)
            bank = joblib.load(directory / 'ratio_banks' / f'origin{origin}__{role}.joblib')
            assert bank['prevalence_source_family_by_sex'] == by_sex and bank['origins'] == origins
            assert bank['outcome'] == 'ylds' and bank['target'] == target
            np.testing.assert_allclose(bank['raw_residuals'], residuals, atol=3e-12, rtol=3e-11)
            np.testing.assert_allclose(bank['centered_residuals'], centered, atol=3e-12, rtol=3e-11)
            point = ratio_current.loc[ratio_current.origin.eq(origin) & ratio_current.family.eq(role)].set_index(COORD).reindex(coords).reset_index()
            for ri, residual_origin in enumerate(origins):
                frame = point[RATE_KEY+['prediction', 'log_prediction']].copy()
                frame['residual_origin'] = residual_origin
                frame['log_draw'] = point.log_prediction.to_numpy()+centered[ri]
                frame['rate_draw'] = np.exp(frame.log_draw)
                ratio_draws.append(frame)
    ratio_draws = pd.concat(ratio_draws, ignore_index=True)
    same(saved_draws.loc[saved_draws.procedure.eq('prevalence_training_ratio')], ratio_draws, RATE_KEY+['residual_origin'], ['rate_draw', 'log_draw'])
    for procedure, outcome, points, draws in [('component_sum', 'dalys', sum_points, sum_draws),
                                            ('prevalence_training_ratio', 'ylds', ratio_current, ratio_draws)]:
        truth = truth_for(target, outcome)
        intervals = independent_quantiles(draws)
        same(saved_intervals.loc[saved_intervals.procedure.eq(procedure)], intervals, KEY+['scale', 'level'], ['lower', 'median', 'upper', 'n_blocks'])
        scored = score_points(points, truth)
        actual_scores = read(directory / 'rate_point_scores.csv')
        same(actual_scores.loc[actual_scores.procedure.eq(procedure)], scored, KEY, ['observed_rate', 'absolute_log_error', 'absolute_rate_error'])
        cells = interval_scores(intervals, truth)
        cells['below_lower'], cells['above_upper'] = cells.observed_value.lt(cells.lower), cells.observed_value.gt(cells.upper)
        actual_intervals = read(directory / 'rate_interval_scores.csv.gz')
        same(actual_intervals.loc[actual_intervals.procedure.eq(procedure)], cells, KEY+['scale', 'level'],
             ['observed_value', 'covered', 'width', 'interval_score', 'below_lower', 'above_upper'])
        wis = wis_scores(cells)
        actual_wis = read(directory / 'rate_wis_scores.csv')
        same(actual_wis.loc[actual_wis.procedure.eq(procedure)], wis, KEY+['scale'], ['wis_50_80', 'wis_50_80_95'])
    # Reconstruct every operational burden from direct/derived joint rate blocks.
    point_parts, draw_parts = [], []
    for outcome in direct_p:
        point_parts.append(direct_p[outcome].loc[direct_p[outcome].family.isin(ROLES)].assign(procedure='direct'))
        draw_parts.append(direct_d[outcome].loc[direct_d[outcome].family.isin(ROLES)].assign(procedure='direct'))
    point_parts += [sum_points.loc[sum_points.family.isin(ROLES)], ratio_current]
    draw_parts += [sum_draws.loc[sum_draws.family.isin(ROLES)], ratio_draws]
    rate_points, rate_draws = pd.concat(point_parts, ignore_index=True), pd.concat(draw_parts, ignore_index=True)
    population, errors = check_population(iso, target)
    popkeys = ['target', 'origin', 'sex', 'age', 'horizon', 'forecast_year']
    popcols = popkeys+['population_method', 'population', 'log_population']
    cells = rate_points.merge(population[popcols], on=popkeys, validate='many_to_many')
    cells['count'] = cells.prediction*cells.population/100000
    expected_burden = aggregate(cells)
    same(read(directory / 'burden_predictions.csv'), expected_burden, STAT, ['value'])
    draws = rate_draws.merge(population[popcols], on=popkeys, validate='many_to_many')
    errors = errors.drop(columns='origin').rename(columns={'fit_origin': 'origin'})
    error_keys = ['target', 'origin', 'sex', 'age', 'horizon', 'residual_origin', 'population_method']
    draws = draws.merge(errors[error_keys+['centered_population_log_error']], on=error_keys, validate='many_to_one')
    draws['population_draw'] = np.exp(draws.log_population+draws.centered_population_log_error)
    draws['count'] = draws.rate_draw*draws.population_draw/100000
    statistics = aggregate(draws, True)
    bounds = burden_quantiles(statistics)
    same(read(directory / 'burden_intervals.csv'), bounds, STAT+['level'], ['lower', 'median', 'upper', 'n_blocks'])
    for frame, draw in [(cells, False), (draws, True)]:
        selected = frame.loc[(frame.procedure.eq('direct') & frame.outcome.isin(['ylds', 'ylls'])) | frame.procedure.eq('component_sum')]
        keys = [col for col in BURDEN if col not in ['outcome', 'procedure']]+['sex', 'age']+(['residual_origin'] if draw else [])
        pivot = selected.pivot(index=keys, columns='outcome', values='count')
        np.testing.assert_allclose(pivot.dalys, pivot.ylds+pivot.ylls, atol=1e-9, rtol=3e-12)
    native = PANEL.loc[PANEL.location_name.eq(target)].rename(columns={'year': 'forecast_year'})
    true_cells = cells.drop(columns='count').merge(native[['outcome', 'sex', 'age', 'forecast_year', 'count']],
                    on=['outcome', 'sex', 'age', 'forecast_year'], validate='many_to_one')
    truth = aggregate(true_cells).rename(columns={'value': 'observed'})
    scored = expected_burden.merge(truth[STAT+['observed']], on=STAT, validate='one_to_one')
    scored['signed_error'] = scored.value-scored.observed
    scored['absolute_error'] = abs(scored.signed_error)
    scored['absolute_log_error'] = abs(np.log(scored.value/scored.observed))
    same(read(directory / 'burden_point_scores.csv'), scored, STAT, ['observed', 'signed_error', 'absolute_error', 'absolute_log_error'])
    scored_bounds = bounds.merge(truth[STAT+['observed']], on=STAT, validate='many_to_one')
    scored_bounds['width'] = scored_bounds.upper-scored_bounds.lower
    scored_bounds['covered'] = scored_bounds.observed.ge(scored_bounds.lower) & scored_bounds.observed.le(scored_bounds.upper)
    scored_bounds['below_lower'], scored_bounds['above_upper'] = scored_bounds.observed.lt(scored_bounds.lower), scored_bounds.observed.gt(scored_bounds.upper)
    scored_bounds['interval_score'] = scored_bounds.width + 2/(1-scored_bounds.level) * (
        np.maximum(scored_bounds.lower-scored_bounds.observed, 0)+np.maximum(scored_bounds.observed-scored_bounds.upper, 0))
    same(read(directory / 'burden_interval_scores.csv.gz'), scored_bounds, STAT+['level'],
         ['observed', 'width', 'covered', 'below_lower', 'above_upper', 'interval_score'])
    wis = scored_bounds.loc[scored_bounds.level.eq(.5), STAT+['observed', 'median']].copy().reset_index(drop=True)
    for levels, label in [([.5, .8], 'wis_50_80'), ([.5, .8, .95], 'wis_50_80_95')]:
        values = .5*abs(wis.observed-wis['median'])
        for level in levels:
            part = wis[STAT].merge(scored_bounds.loc[scored_bounds.level.eq(level), STAT+['interval_score']], on=STAT, validate='one_to_one')
            values += (1-level)/2*part.interval_score
        wis[label] = values/(len(levels)+.5)
    same(read(directory / 'burden_wis_scores.csv'), wis, STAT, ['wis_50_80', 'wis_50_80_95'])
    issued = json.loads((directory / 'issued_commit.json').read_text())
    scoring = json.loads((directory / 'scoring_complete.json').read_text())
    global_commit = json.loads((out / 'global_issued_commit.json').read_text())
    assert datetime.fromisoformat(issued['committed_utc']) <= datetime.fromisoformat(global_commit['committed_utc']) <= datetime.fromisoformat(scoring['committed_utc'])
    return dict(country=iso, passed=True, derived_rate_points=len(saved_points), derived_rate_draws=len(saved_draws),
                derived_rate_intervals=len(saved_intervals), ratio_banks_replayed=15, ratio_history_points=len(ratio_history),
                burden_points=len(expected_burden), burden_intervals=len(bounds), burden_joint_statistics=len(statistics),
                units_and_component_champion_provenance_verified=True, historical_ratio_chronology_verified=True,
                common_population_forecasts_and_residuals_rebuilt=True, elapsed_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/disability_components_v1')
    parser.add_argument('--core', default='results/supporting_v1')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    out, core = ROOT / args.run, ROOT / args.core
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, digest in manifest['output_sha256'].items():
        assert sha(out / name) == digest, name
    for name, digest in manifest['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in manifest['prior_manifests_sha256'].items():
        assert sha(ROOT / name) == digest, name
    tasks = [(out, core, country) for country in CONFIG['countries'] if country['gcc']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        countries = list(pool.map(audit_country, tasks))
    source = PANEL.loc[PANEL.year.ge(1990)].copy()
    keys = ['location_name', 'sex', 'age', 'year']
    check = read(out / 'source_accounting_checks.csv').set_index(keys).sort_index()
    for field in ['rate', 'count']:
        values = source.pivot(index=keys, columns='outcome', values=field).sort_index()
        expected = (values.dalys-values.ylds-values.ylls)/values.dalys
        assert check.index.equals(expected.index)
        np.testing.assert_allclose(check[field+'_relative_identity_error'], expected, atol=1e-15, rtol=1e-8)
    source['population'] = source['count']/source.rate*100000
    population = source.pivot(index=keys, columns='outcome', values='population').sort_index()
    diff = abs(population.div(population.prevalence, axis=0)-1).max(axis=1)
    np.testing.assert_allclose(check.maximum_relative_population_difference, diff, atol=1e-15, rtol=1e-8)
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), audit_code_sha256=sha(__file__),
                  audit_helper_sha256=sha(HERE / 'audit_rates.py'), run_manifest_sha256=sha(out / 'run_manifest.json'),
                  all_source_output_hashes_verified=True, source_accounting_cells=len(check), countries=countries)
    (HERE / 'audit_components.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, countries=len(countries), rate_intervals=sum(row['derived_rate_intervals'] for row in countries),
                          burden_intervals=sum(row['burden_intervals'] for row in countries))))


if __name__ == '__main__':
    main()
