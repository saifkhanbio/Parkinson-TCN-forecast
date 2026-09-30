"""Issue one locked equal-weight trajectory mixture and matched controls."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
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
import numpy as np
import pandas as pd
from gbd_park.distribution_mixture import inverse_ecdf, pool_trajectories, validate_residual_origins
from gbd_park.demography import joint_count_draws
from run_calibration import cases, labels, match_numeric, rate_summaries, underlying_family
from run_demography import STAT, burden_statistics, ratio_statistics
from run_reliability_v1_3 import RATE_KEYS, tensor, derived_values, rate_ledgers, burden_ledgers, score_bounds
from run_local_baselines import sha, now

SPEC = ROOT / 'study_design/distribution_mixture_v1.json'
LOCK = ROOT / 'study_design/distribution_mixture_v1.lock.json'
PREFLIGHT = ROOT / 'work/distribution-mixture-validation/preflight.json'


def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def write_json(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def build_components(points, draws, populations, errors, config, spec, origin):
    components = {}
    residual_origins = list(range(2003, origin - 4))
    validate_residual_origins(residual_origins, origin)
    for role in spec['roles']:
        p = points[points.origin.eq(origin) & points.family.eq(role)]
        d = draws[draws.origin.eq(origin) & draws.family.eq(role)]
        if sorted(d.residual_origin.unique()) != residual_origins:
            raise ValueError('Missing, duplicate or future residual origins')
        arrays = {'point_rates': tensor(p, config, 'prediction'),
                  'rate_draws': tensor(d, config, 'rate_draw', True)}
        if not np.allclose(np.log(arrays['rate_draws']), tensor_log(d, config), atol=1e-12, rtol=1e-12):
            raise ValueError('Saved log/rate draws disagree')
        for method in spec['population_methods']:
            pop = populations[populations.origin.eq(origin) & populations.population_method.eq(method)]
            err = errors[errors.fit_origin.eq(origin) & errors.population_method.eq(method)]
            if not err.forecast_year.le(origin).all() or sorted(err.residual_origin.unique()) != residual_origins:
                raise ValueError('Unavailable or incomplete population error blocks')
            joined = joint_count_draws(d, pop, err)
            arrays['point_populations__' + method] = tensor(pop, config, 'population')
            arrays['point_counts__' + method] = arrays['point_rates'] * arrays['point_populations__' + method] / 100000
            arrays['population_draws__' + method] = tensor(joined, config, 'population_draw', True)
            arrays['count_draws__' + method] = tensor(joined, config, 'count', True)
        components[role] = arrays
    return components, residual_origins


def tensor_log(draws, config):
    coords = pd.MultiIndex.from_product([sorted(draws.residual_origin.unique()), config['calendar']['horizons'],
                                        config['sexes'], config['ages']], names=['residual_origin', 'horizon', 'sex', 'age'])
    return draws.set_index(coords.names).reindex(coords).log_draw.to_numpy().reshape(-1, 5, 22)


def ledgers(arrays, config, spec, context, n_blocks, quantile):
    kind = 'finite_role_origin_mixture' if context['family'] == spec['candidate'] else 'historical_joint_blocks'
    p, r = rate_ledgers(arrays, config, context, n_blocks, kind)
    bp, b = burden_ledgers(arrays, spec['population_methods'], config, context, n_blocks, kind)
    if quantile == 'cdf':
        for scale in ['rate', 'log_rate']:
            values = arrays['rate_draws'] if scale == 'rate' else np.log(arrays['rate_draws'])
            for level in spec['levels']:
                bounds = inverse_ecdf(values, [(1-level)/2, .5, (1+level)/2])
                mask = r.scale.eq(scale) & r.level.eq(level)
                r.loc[mask, ['lower', 'median', 'upper']] = bounds.reshape(3, -1).T
        for method in spec['population_methods'] + ['not_applicable']:
            ratio = method == 'not_applicable'
            values, _ = derived_values(arrays['rate_draws'], None if ratio else arrays['count_draws__' + method], config, ratio)
            for level in spec['levels']:
                bounds = inverse_ecdf(values, [(1-level)/2, .5, (1+level)/2])
                mask = b.population_method.eq(method) & b.level.eq(level)
                b.loc[mask, ['lower', 'median', 'upper']] = bounds.reshape(3, -1).T
    elif quantile != 'original':
        raise ValueError('Unknown quantile rule')
    for frame in [p, r, bp, b]:
        frame['quantile_rule'] = 'inverse_ecdf' if quantile == 'cdf' else 'linear'
    return p, r, bp, b


def issue_case(case, config, spec, output):
    directory = Path(output) / case['id']
    directory.mkdir()
    (directory / 'draws').mkdir()
    source, demographic = ROOT / case['source'], ROOT / case['demography']
    points, draws = read(source / 'predictions.csv'), read(source / 'joint_draws.csv')
    points = points[points.family.isin(spec['roles']) & points.origin.isin(spec['evaluation_origins'])].copy()
    draws = draws[draws.family.isin(spec['roles']) & draws.origin.isin(spec['evaluation_origins'])].copy()
    for frame in [points, draws]:
        if set(frame.target) != {case['target']} or set(frame.outcome) != {case['outcome']}:
            raise ValueError('Wrong country/outcome in input')
    mappings = read(source / 'champion_family_mappings.csv')
    populations, errors = read(demographic / 'population_forecasts.csv'), read(demographic / 'population_residuals.csv')
    points.to_csv(directory / 'component_point_audit.csv', index=False)
    frames = [[], [], [], []]
    index, atom_tables, role_mapping = [], [], []
    for origin in spec['evaluation_origins']:
        components, origins = build_components(points, draws, populations, errors, config, spec, origin)
        mixture, atoms = pool_trajectories(components, spec['roles'], spec['population_methods'], origins, origin)
        for role in spec['roles']:
            role_mapping.append({'origin': origin, 'role': role,
                                 'source_family_by_sex': {s: underlying_family(mappings, role, s, origin) for s in config['sexes']}})
        atom_frame = pd.DataFrame(atoms)
        atom_frame['origin'] = origin
        atom_tables.append(atom_frame)
        for role, arrays in list(components.items()) + [('mixture', mixture)]:
            family = spec['candidate'] if role == 'mixture' else role
            path = f'draws/origin{origin}__{role}.npz'
            np.savez_compressed(directory / path, **arrays)
            index.append({'origin': origin, 'role': role, 'path': path, 'residual_origins': origins,
                          'n_blocks': len(origins), 'n_atoms': len(arrays['rate_draws'])})
            variants = ['cdf'] if role == 'mixture' else spec['controls']
            for variant in variants:
                context = dict(target=case['target'], outcome=case['outcome'], origin=origin,
                               family=family if role == 'mixture' else role + '__' + variant)
                for storage, ledger in zip(frames, ledgers(arrays, config, spec, context, len(origins), variant)):
                    storage.append(ledger)
    rate_points, rate_intervals, burden_points, burden_intervals = [labels(pd.concat(f, ignore_index=True)) for f in frames]
    # Replay the archived originals exactly; these comparisons contain no verification outcomes.
    original_rate = rate_intervals[rate_intervals.variant.eq('original')].copy()
    original_rate['family'] = original_rate.role
    old_rate = read(source / 'intervals.csv')
    old_rate = old_rate[old_rate.family.isin(spec['roles']) & old_rate.origin.isin(spec['evaluation_origins'])]
    rate_difference = match_numeric(original_rate, old_rate, RATE_KEYS + ['level'], ['lower', 'median', 'upper', 'point_prediction'])
    original_points = rate_points[rate_points.variant.eq('original')].copy()
    original_points['family'] = original_points.role
    match_numeric(original_points, points, RATE_KEYS[:-1], ['prediction', 'log_prediction'])
    original_burden = burden_intervals[burden_intervals.variant.eq('original')].copy()
    original_burden['family'] = original_burden.role
    old_burden = read(demographic / 'intervals.csv')
    old_burden = old_burden[old_burden.family.isin(spec['roles']) & old_burden.origin.isin(spec['evaluation_origins'])]
    burden_difference = match_numeric(original_burden, old_burden, STAT + ['level'], ['lower', 'median', 'upper'])
    for name, frame in [('rate_predictions.csv', rate_points), ('rate_intervals.csv', rate_intervals),
                        ('burden_predictions.csv', burden_points), ('burden_intervals.csv', burden_intervals)]:
        frame.to_csv(directory / name, index=False)
    pd.concat(atom_tables, ignore_index=True).to_csv(directory / 'mixture_atoms.csv', index=False)
    write_json(directory / 'component_mappings.json', role_mapping)
    write_json(directory / 'draw_index.json', index)
    validation = dict(case=case['id'], passed=True, procedures=7, rate_original_max_difference=rate_difference,
                      burden_original_max_difference=burden_difference, models_fitted=0)
    write_json(directory / 'issuance_validation.json', validation)
    write_json(directory / 'issued_commit.json', {'committed_utc': now(), 'evaluation_scored': False,
               'artifact_sha256': {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}})
    return validation


def score_case(case, config, spec, output):
    out, directory = Path(output), Path(output) / case['id']
    global_commit = json.loads((out / 'global_issued_commit.json').read_text())
    assert len(global_commit['case_commit_sha256']) == 12
    assert all(sha(out / p) == digest for p, digest in global_commit['case_commit_sha256'].items())
    commit = json.loads((directory / 'issued_commit.json').read_text())
    assert all(sha(directory / p) == digest for p, digest in commit['artifact_sha256'].items())
    write_json(directory / 'scoring_event.json', {'started_utc': now(), 'global_commit_sha256': sha(out / 'global_issued_commit.json')})
    panel = read(ROOT / 'data/processed/design_v1/regional_outcomes.csv')
    actual = panel[panel.location_name.eq(case['target']) & panel.outcome.eq(case['outcome'])].rename(
        columns={'location_name': 'target', 'year': 'forecast_year', 'rate': 'observed_rate'})
    points, intervals = read(directory / 'rate_predictions.csv'), read(directory / 'rate_intervals.csv')
    keys = ['target', 'outcome', 'sex', 'age', 'forecast_year']
    truth = intervals[RATE_KEYS].drop_duplicates().merge(actual[keys + ['observed_rate']], on=keys, validate='many_to_one')
    truth['observed_value'] = np.where(truth.scale.eq('rate'), truth.observed_rate, np.log(truth.observed_rate))
    cells, wis = score_bounds(intervals, truth, RATE_KEYS, 'observed_value')
    cells, wis = labels(cells), labels(wis)
    cells.to_csv(directory / 'rate_interval_scores.csv.gz', index=False, compression='gzip')
    wis.to_csv(directory / 'rate_wis_scores.csv', index=False)
    summary, wsummary = rate_summaries(cells, wis, config)
    summary.to_csv(directory / 'rate_summary.csv', index=False)
    wsummary.to_csv(directory / 'rate_wis_summary.csv', index=False)
    scored = points.merge(actual[keys + ['observed_rate', 'count']], on=keys, validate='many_to_one')
    scored['absolute_log_error'] = abs(scored.log_prediction - np.log(scored.observed_rate))
    scored['signed_log_error'] = scored.log_prediction - np.log(scored.observed_rate)
    scored.to_csv(directory / 'rate_point_scores.csv', index=False)
    verified = scored.copy()
    verified['prediction'] = verified.observed_rate
    truth_parts = [ratio_statistics(verified, config)]
    for method in spec['population_methods']:
        verified['population_method'] = method
        truth_parts.append(burden_statistics(verified, config))
    burden_truth = pd.concat(truth_parts, ignore_index=True)[STAT + ['value']].rename(columns={'value': 'observed'})
    bcells, bwis = score_bounds(read(directory / 'burden_intervals.csv'), burden_truth, STAT, 'observed')
    labels(bcells).to_csv(directory / 'burden_interval_scores.csv.gz', index=False, compression='gzip')
    labels(bwis).to_csv(directory / 'burden_wis_scores.csv', index=False)
    bp = read(directory / 'burden_predictions.csv').merge(burden_truth, on=STAT, validate='one_to_one')
    bp['absolute_error'] = abs(bp.value - bp.observed)
    bp['absolute_log_error'] = abs(np.log(bp.value / bp.observed))
    bp.to_csv(directory / 'burden_point_scores.csv', index=False)
    assert all(sha(directory / p) == digest for p, digest in commit['artifact_sha256'].items())
    result = dict(case=case['id'], passed=True, rate_interval_rows=len(cells), burden_interval_rows=len(bcells), scored_utc=now())
    write_json(directory / 'validation_report.json', result)
    return result


def verify_preflight(preflight):
    for group in ['protected_sha256', 'code_sha256', 'input_sha256', 'specification_sha256']:
        for path, digest in preflight[group].items():
            assert sha(ROOT / path) == digest, (group, path)
    assert preflight['tests_passed']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/distribution_mixture_v1')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    spec, preflight = json.loads(SPEC.read_text()), json.loads(PREFLIGHT.read_text())
    assert 1 <= args.workers <= spec['max_workers']
    verify_preflight(preflight)
    out = (ROOT / args.output).resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError('Output must remain inside repository')
    out.mkdir(parents=True, exist_ok=False)
    tasks = cases(config, spec)
    write_json(out / 'cases.json', tasks)
    write_json(out / 'preflight.json', preflight)
    start = time.perf_counter()
    write_json(out / 'run_started.json', dict(started_utc=now(), workers=args.workers, numerical_threads_per_worker=1,
                                              candidate=spec['candidate'], models_fitted=0, status='running'))
    context = multiprocessing.get_context('spawn')
    issued = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(issue_case, t, config, spec, out): t for t in tasks}
        for future in as_completed(futures):
            issued.append(future.result())
            print('Issued and committed: ' + futures[future]['id'], flush=True)
    write_json(out / 'global_issued_commit.json', {'committed_utc': now(),
               'case_commit_sha256': {t['id'] + '/issued_commit.json': sha(out / t['id'] / 'issued_commit.json') for t in tasks}})
    scored = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(score_case, t, config, spec, out): t for t in tasks}
        for future in as_completed(futures):
            scored.append(future.result())
            print('Scored: ' + futures[future]['id'], flush=True)
    verify_preflight(preflight)
    write_json(out / 'validation_report.json', {'passed': True, 'issuance': issued, 'scoring': scored,
               'all_cases_committed_before_scoring': True, 'protected_files_unchanged': True})
    write_json(out / 'run_manifest.json', {'status': 'complete', 'completed_utc': now(), 'models_fitted': 0,
               'elapsed_seconds': time.perf_counter() - start, 'workers': args.workers,
               'preflight_sha256': sha(PREFLIGHT), 'specification_lock_sha256': sha(LOCK),
               'output_sha256': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print('All 12 mixture cases completed.', flush=True)


if __name__ == '__main__':
    if not __debug__:
        raise SystemExit('Run without -O.')
    main()
