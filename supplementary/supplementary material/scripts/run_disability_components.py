"""Derived disability rates and coherent operational burden after core fitting."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from gbd_park.disability_components import (KEYS, ROLES, check_grid, sum_daly, intervals_from_draws,
                                           issue_ratio_forecasts, source_accounting)
from gbd_park.population_sensitivity import aggregate, NODE
from gbd_park.intervals import score_intervals, interval_score, weighted_interval_score
from gbd_park.secondary import score_actual, actual_truth
from run_local_baselines import check_lock, sha, now
from run_secondary import write_json, verify_hashes, hash_files, phase_complete, commit_phase

BURDEN = ['target', 'outcome', 'origin', 'horizon', 'forecast_year', 'family', 'procedure', 'population_method']
STAT = BURDEN + NODE
REQUIRED = ['src/gbd_park/disability_components.py', 'scripts/run_disability_components.py',
            'tests/test_disability_components.py', 'study_design/supporting_outcomes_implementation.md']


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def source_paths(country):
    iso = country['iso3']
    if iso == 'SAU':
        return {'points': ROOT / 'results/primary_v1/predictions.csv',
                'history': ROOT / 'results/intervals_v1/prequential_predictions.csv',
                'mapping': ROOT / 'results/primary_v1/champion_family_mappings.csv'}
    directory = ROOT / 'results/secondary_v1/trials' / (iso + '_prevalence')
    return {'points': directory / 'predictions.csv', 'history': directory / 'prequential_predictions.csv',
            'mapping': directory / 'champion_family_mappings.csv'}


def burden_intervals(statistics):
    if statistics.duplicated(STAT + ['residual_origin']).any():
        raise ValueError('Duplicate burden-block coordinates')
    group = statistics.groupby(STAT).value
    q = group.quantile([.025, .1, .25, .5, .75, .9, .975], interpolation='linear').unstack()
    n = group.size()
    assert n.min() >= 7 and n.max() <= 11
    result = []
    for level, low, high in [(.5, .25, .75), (.8, .1, .9), (.95, .025, .975)]:
        part = pd.DataFrame({'lower': q[low], 'median': q[.5], 'upper': q[high], 'n_blocks': n}).reset_index()
        part['level'] = level
        part['uncertainty_scope'] = 'joint_historical_rate_population_errors'
        result.append(part)
    return pd.concat(result, ignore_index=True)


def burden_units(frame):
    frame = frame.copy()
    frame.loc[frame.measure.eq('count') & frame.outcome.eq('deaths'), 'unit'] = 'modeled_events'
    frame.loc[frame.measure.eq('count') & frame.outcome.isin(['ylds', 'ylls', 'dalys']), 'unit'] = 'modeled_burden_years'
    return frame


def convert_burden(points, draws, population, population_errors, config):
    """Use one frozen population process for every component and role."""
    contexts = KEYS + ['outcome', 'procedure']
    if set(map(tuple, points[contexts].to_numpy())) != set(map(tuple, draws[contexts].to_numpy())):
        raise ValueError('Point and draw burden contexts do not match')
    for frame, is_draw in [(points, False), (draws, True)]:
        for _, part in frame.groupby(['outcome', 'procedure']):
            check_grid(part, config, draw=is_draw)
    keys = ['target', 'origin', 'sex', 'age', 'horizon', 'forecast_year']
    pop = population[keys + ['population_method', 'population', 'log_population']]
    if set(pop.population_method) != {'log_trend_last8', 'persistence'}:
        raise ValueError('Both locked population procedures are required')
    if pop.duplicated(keys + ['population_method']).any():
        raise ValueError('Duplicate common population forecasts')
    cells = points.merge(pop, on=keys, how='left', validate='many_to_many')
    if len(cells) != 2 * len(points) or not np.isfinite(cells.population).all():
        raise ValueError('Missing common population inputs')
    cells['count'] = cells.prediction * cells.population / 100000
    count_draws = draws.merge(pop, on=keys, how='left', validate='many_to_many')
    errors = population_errors.drop(columns='origin').rename(columns={'fit_origin': 'origin'})
    error_keys = ['target', 'origin', 'sex', 'age', 'horizon', 'residual_origin', 'population_method']
    count_draws = count_draws.merge(errors[error_keys + ['centered_population_log_error']], on=error_keys,
                                    how='left', validate='many_to_one')
    fields = ['rate_draw', 'log_population', 'centered_population_log_error']
    if len(count_draws) != 2 * len(draws) or not np.isfinite(count_draws[fields]).all().all():
        raise ValueError('Population and rate residual blocks do not match')
    count_draws['population_draw'] = np.exp(count_draws.log_population + count_draws.centered_population_log_error)
    count_draws['count'] = count_draws.rate_draw * count_draws.population_draw / 100000
    predicted = burden_units(aggregate(cells, config, identifiers=BURDEN))
    statistics = burden_units(aggregate(count_draws, config, identifiers=BURDEN + ['residual_origin']))
    # Exact draw-by-draw count additivity under the common population assumption.
    checks = []
    for frame, value, is_draw in [(cells, 'count', False), (count_draws, 'count', True)]:
        join_keys = [c for c in BURDEN if c not in ['outcome', 'procedure']] + ['sex', 'age']
        if is_draw:
            join_keys += ['residual_origin']
        part = frame.loc[(frame.procedure.eq('direct') & frame.outcome.isin(['ylds', 'ylls']))
                         | frame.procedure.eq('component_sum')].copy()
        pivot = part.pivot(index=join_keys, columns='outcome', values=value)
        np.testing.assert_allclose(pivot.dalys, pivot.ylds + pivot.ylls, rtol=1e-12, atol=1e-10)
        checks.append({'draw': is_draw, 'coordinates': len(pivot),
                       'maximum_absolute_count_identity_error': float(abs(pivot.dalys - pivot.ylds - pivot.ylls).max())})
    return predicted, burden_intervals(statistics), checks


def issue_country(country, config, core, destination):
    started = time.perf_counter()
    directory = Path(destination) / country['iso3']
    directory.mkdir(exist_ok=True)
    if phase_complete(directory, 'issued_commit'):
        return {'country': country['iso3'], 'cached': True}
    panel = read(ROOT / 'data/processed/design_v1/regional_outcomes.csv')
    direct_points, direct_draws = {}, {}
    for outcome in ['deaths', 'ylds', 'ylls', 'dalys']:
        source = Path(core) / 'trials' / (country['iso3'] + '_' + outcome)
        direct_points[outcome] = read(source / 'predictions.csv')
        direct_draws[outcome] = read(source / 'joint_draws.csv')
    summed_points = sum_daly(direct_points['ylds'], direct_points['ylls'], config)
    summed_draws = sum_daly(direct_draws['ylds'], direct_draws['ylls'], config, draw=True)
    summed_intervals = intervals_from_draws(summed_points, summed_draws, config)
    prevalence = source_paths(country)
    ratio_points, ratio_intervals, ratio_draws, banks, history = issue_ratio_forecasts(
        read(prevalence['history']), read(prevalence['points']), read(prevalence['mapping']), panel, config, country['name'])
    history.to_csv(directory / 'ratio_prequential_predictions.csv', index=False)
    bank_folder = directory / 'ratio_banks'
    bank_folder.mkdir(exist_ok=True)
    for (origin, role), bank in banks.items():
        joblib.dump(bank, bank_folder / f'origin{origin}__{role}.joblib', compress=3)
    issued_points = pd.concat([summed_points, ratio_points], ignore_index=True)
    issued_intervals = pd.concat([summed_intervals, ratio_intervals], ignore_index=True)
    issued_draws = pd.concat([summed_draws, ratio_draws], ignore_index=True)
    issued_points.to_csv(directory / 'rate_predictions.csv', index=False)
    issued_intervals.to_csv(directory / 'rate_intervals.csv', index=False)
    issued_draws.to_csv(directory / 'rate_draws.csv.gz', index=False, compression='gzip')
    assert len(issued_points) == 10450 and len(issued_intervals) == 62700 and len(issued_draws) == 94050
    points, draws = [], []
    for outcome in direct_points:
        p = direct_points[outcome].loc[direct_points[outcome].family.isin(ROLES)].copy()
        d = direct_draws[outcome].loc[direct_draws[outcome].family.isin(ROLES)].copy()
        p['procedure'], d['procedure'] = 'direct', 'direct'
        points.append(p)
        draws.append(d)
    points.append(issued_points.loc[issued_points.family.isin(ROLES)])
    draws.append(issued_draws.loc[issued_draws.family.isin(ROLES)])
    burden_points, burden_bounds, checks = convert_burden(pd.concat(points, ignore_index=True), pd.concat(draws, ignore_index=True),
        read(ROOT / 'results/demography_v1' / (country['iso3'] + '_prevalence') / 'population_forecasts.csv'),
        read(ROOT / 'results/demography_v1' / (country['iso3'] + '_prevalence') / 'population_residuals.csv'), config)
    assert len(burden_points) == 31500 and len(burden_bounds) == 94500
    burden_points.to_csv(directory / 'burden_predictions.csv', index=False)
    burden_bounds.to_csv(directory / 'burden_intervals.csv', index=False)
    write_json(directory / 'count_identity_checks.json', checks)
    commit_phase(directory, 'issued_commit', [p for p in directory.rglob('*') if p.is_file()],
                 final_period_scored=False, elapsed_seconds=time.perf_counter()-started,
                 all_derived_methods_fixed_before_core_fitting=True)
    return {'country': country['iso3'], 'cached': False, 'elapsed_seconds': time.perf_counter()-started}


def score_burden(intervals, truth):
    frame = intervals.merge(truth, on=STAT, how='left', validate='many_to_one')
    assert np.isfinite(frame.observed).all()
    frame['width'] = frame.upper - frame.lower
    frame['covered'] = frame.observed.ge(frame.lower) & frame.observed.le(frame.upper)
    frame['below_lower'], frame['above_upper'] = frame.observed.lt(frame.lower), frame.observed.gt(frame.upper)
    frame['interval_score'] = interval_score(frame.observed, frame.lower, frame.upper, 1-frame.level)
    low = frame.pivot(index=STAT, columns='level', values='lower')[[.5, .8, .95]]
    high = frame.pivot(index=STAT, columns='level', values='upper')[[.5, .8, .95]]
    base = frame.drop_duplicates(STAT).set_index(STAT).reindex(low.index)
    wis = base[['observed', 'median', 'n_blocks']].copy()
    wis['wis_50_80'] = weighted_interval_score(base.observed, base['median'], low.iloc[:, :2], high.iloc[:, :2], [.5, .8])
    wis['wis_50_80_95'] = weighted_interval_score(base.observed, base['median'], low, high, [.5, .8, .95])
    return frame, wis.reset_index()


def score_country(country, config, destination):
    directory = Path(destination) / country['iso3']
    if phase_complete(directory, 'scoring_complete'):
        return
    assert phase_complete(Path(destination), 'global_issued_commit')
    global_commit = json.loads((Path(destination) / 'global_issued_commit.json').read_text())
    expected_markers = {c['iso3'] + '/issued_commit.json' for c in config['countries'] if c['gcc']}
    if set(global_commit['artifact_sha256']) != expected_markers:
        raise ValueError('Derived scoring requires every configured GCC issuance marker')
    for item in expected_markers:
        if not phase_complete((Path(destination) / item).parent, 'issued_commit'):
            raise ValueError('Derived scoring requires intact issued artifacts for every country')
    panel = read(ROOT / 'data/processed/design_v1/regional_outcomes.csv')
    predictions = read(directory / 'rate_predictions.csv')
    intervals = read(directory / 'rate_intervals.csv')
    scores, cell_scores, wis = [], [], []
    for outcome, procedure in [('dalys', 'component_sum'), ('ylds', 'prevalence_training_ratio')]:
        points = predictions.loc[predictions.procedure.eq(procedure)]
        scored = score_actual(points, panel, config, country['name'], outcome, 2023)
        bound_scores, w = score_intervals(intervals.loc[intervals.procedure.eq(procedure)],
                                          actual_truth(panel, country['name'], outcome, 2023), config, 2023)
        bound_scores['below_lower'] = bound_scores.observed_value.lt(bound_scores.lower)
        bound_scores['above_upper'] = bound_scores.observed_value.gt(bound_scores.upper)
        w['procedure'] = procedure
        scores.append(scored)
        cell_scores.append(bound_scores)
        wis.append(w)
    pd.concat(scores, ignore_index=True).to_csv(directory / 'rate_point_scores.csv', index=False)
    pd.concat(cell_scores, ignore_index=True).to_csv(directory / 'rate_interval_scores.csv.gz', index=False, compression='gzip')
    pd.concat(wis, ignore_index=True).to_csv(directory / 'rate_wis_scores.csv', index=False)
    burden_points = read(directory / 'burden_predictions.csv')
    contexts = burden_points[BURDEN].drop_duplicates()
    native = panel.loc[panel.location_name.eq(country['name']) & panel.outcome.isin(['deaths', 'ylds', 'ylls', 'dalys']),
                        ['outcome', 'year', 'sex', 'age', 'count']].rename(columns={'year': 'forecast_year'})
    native_cells = contexts.merge(native, on=['outcome', 'forecast_year'], how='left', validate='many_to_many')
    truth = burden_units(aggregate(native_cells, config, identifiers=BURDEN)).rename(columns={'value': 'observed'})
    scored = burden_points.merge(truth, on=STAT, how='left', validate='one_to_one')
    assert np.isfinite(scored.observed).all()
    scored['signed_error'] = scored.value - scored.observed
    scored['absolute_error'] = abs(scored.signed_error)
    scored['absolute_log_error'] = abs(np.log(scored.value / scored.observed))
    scored.to_csv(directory / 'burden_point_scores.csv', index=False)
    cells, wis = score_burden(read(directory / 'burden_intervals.csv'), truth)
    cells.to_csv(directory / 'burden_interval_scores.csv.gz', index=False, compression='gzip')
    wis.to_csv(directory / 'burden_wis_scores.csv', index=False)
    write_json(directory / 'validation_report.json', {'passed': True, 'country': country['iso3'],
        'derived_rate_points': len(predictions), 'derived_rate_intervals': len(intervals),
        'burden_point_statistics': len(scored), 'burden_interval_rows': len(cells),
        'common_prevalence_implied_population': True, 'all_countries_issued_before_derived_scoring': True})
    commit_phase(directory, 'scoring_complete', [p for p in directory.rglob('*') if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', default='results/supporting_v1')
    parser.add_argument('--output', default='results/disability_components_v1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('Use one through six case workers')
    check_lock()
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    countries = [c for c in config['countries'] if c['gcc']]
    tests_path = ROOT / 'work/supporting-validation/component_tests.json'
    tests = json.loads(tests_path.read_text())
    assert tests['passed'] and set(REQUIRED).issubset(tests['tested_code_sha256'])
    verify_hashes(ROOT, tests['tested_code_sha256'])
    core = ROOT / args.core
    manifests = {}
    for name in [args.core, 'results/primary_v1', 'results/secondary_v1', 'results/intervals_v1', 'results/demography_v1']:
        directory = ROOT / name
        manifest = json.loads((directory / 'run_manifest.json').read_text())
        assert manifest['status'] == 'complete'
        verify_hashes(directory, manifest['output_sha256'])
        manifests[name + '/run_manifest.json'] = sha(directory / 'run_manifest.json')
    code = {name: sha(ROOT / name) for name in REQUIRED + [
        'src/gbd_park/population_sensitivity.py', 'src/gbd_park/demography.py', 'src/gbd_park/intervals.py',
        'src/gbd_park/prequential.py', 'src/gbd_park/scoring.py', 'src/gbd_park/secondary.py',
        'scripts/run_secondary.py', 'scripts/run_local_baselines.py']}
    source = {'data/processed/design_v1/regional_outcomes.csv': sha(ROOT / 'data/processed/design_v1/regional_outcomes.csv'),
              'study_design/locked_v1/design.json': sha(ROOT / 'study_design/locked_v1/design.json')}
    identity = dict(code_sha256=code, source_sha256=source, prior_manifests_sha256=manifests,
                    test_report_sha256=sha(tests_path), workers=args.workers)
    out = ROOT / args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError('Existing derived analysis requires explicit --resume')
        manifest = json.loads((out / 'run_manifest.json').read_text())
        if manifest['identity'] != identity:
            raise ValueError('Derived-analysis resume identity changed')
        if manifest['status'] == 'complete':
            verify_hashes(out, manifest['output_sha256'])
            print('Completed derived analysis verified; no recomputation', flush=True)
            return
    else:
        if args.resume:
            raise FileNotFoundError('Cannot resume missing derived analysis')
        out.mkdir(parents=True)
        manifest = dict(status='running', created_utc=now(), identity=identity, code_sha256=code,
                        role='prespecified_supporting_disability_accounting', prior_manifests_sha256=manifests)
        write_json(out / 'run_manifest.json', manifest)
    started = time.perf_counter()
    checks = source_accounting(read(ROOT / 'data/processed/design_v1/regional_outcomes.csv'), config)
    assert len(checks) == 5236
    checks.to_csv(out / 'source_accounting_checks.csv', index=False)
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for future in as_completed([pool.submit(issue_country, c, config, core, out) for c in countries]):
            print('Committed supporting derivations: ' + future.result()['country'], flush=True)
    if not phase_complete(out, 'global_issued_commit'):
        commit_phase(out, 'global_issued_commit', [out / c['iso3'] / 'issued_commit.json' for c in countries])
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for future in as_completed([pool.submit(score_country, c, config, out) for c in countries]):
            future.result()
    verify_hashes(ROOT, code)
    verify_hashes(ROOT, source)
    verify_hashes(ROOT, manifests)
    check_lock()
    write_json(out / 'validation_report.json', dict(passed=True, countries=6, elapsed_seconds=time.perf_counter()-started,
        cases=[json.loads((out / c['iso3'] / 'validation_report.json').read_text()) for c in countries]))
    manifest.update(status='complete', completed_utc=now(), output_sha256=hash_files(out,
        [p for p in out.rglob('*') if p.is_file() and p != out / 'run_manifest.json']))
    write_json(out / 'run_manifest.json', manifest)
    print('Derived disability and operational burden complete', flush=True)


if __name__ == '__main__':
    main()
