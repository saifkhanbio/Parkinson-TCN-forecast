"""Apply fixed population-source diagnostics to saved GCC disease forecasts."""
import os
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from gbd_park.population_sensitivity import (CELL, CONTEXT, NODE, SCENARIOS, ROLES, populations,
    aggregate, conditional_intervals, error_accounting, projection_populations)
from gbd_park.intervals import interval_score, weighted_interval_score
from run_local_baselines import check_lock


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def score(intervals, truth):
    keys = CONTEXT + NODE
    frame = intervals.merge(truth, on=keys, how='left', validate='many_to_one')
    assert np.isfinite(frame.observed).all()
    frame['width'] = frame.upper - frame.lower
    frame['covered'] = frame.observed.ge(frame.lower) & frame.observed.le(frame.upper)
    frame['below_lower'] = frame.observed.lt(frame.lower)
    frame['above_upper'] = frame.observed.gt(frame.upper)
    frame['interval_score'] = interval_score(frame.observed, frame.lower, frame.upper, 1 - frame.level)
    lo = frame.pivot(index=keys, columns='level', values='lower')[[.5, .8, .95]]
    hi = frame.pivot(index=keys, columns='level', values='upper')[[.5, .8, .95]]
    base = frame.drop_duplicates(keys).set_index(keys).reindex(lo.index)
    wis = base[['observed', 'median', 'n_blocks', 'uncertainty_scope']].copy()
    wis['wis_50_80'] = weighted_interval_score(base.observed, base['median'], lo.iloc[:, :2], hi.iloc[:, :2], [.5, .8])
    wis['wis_50_80_95'] = weighted_interval_score(base.observed, base['median'], lo, hi, [.5, .8, .95])
    return frame, wis.reset_index()


def case_job(case, config, destination):
    started = time.perf_counter()
    out = Path(destination) / case['id']
    out.mkdir()
    source, old = ROOT / case['source'], ROOT / 'results/demography_v1' / case['id']
    panel = read(ROOT / 'data/processed/design_v1/regional_outcomes.csv')
    un = read(ROOT / 'data/processed/design_v1/un_population_1990_2028.csv')
    operational = read(old / 'population_forecasts.csv')
    pop = populations(panel, un, operational, config, case['target'], case['outcome'])
    points = read(source / 'predictions.csv', usecols=CELL + ['family', 'prediction'])
    points = points.loc[points.family.isin(ROLES)].copy()
    draws = read(source / 'joint_draws.csv', usecols=CELL + ['family', 'residual_origin', 'rate_draw'])
    draws = draws.loc[draws.family.isin(ROLES)].copy()
    assert len(points) == 5 * 5 * 22 * 3
    assert not points.duplicated(CELL + ['family']).any()
    assert set(points.target) == {case['target']} and set(points.outcome) == {case['outcome']}
    for (origin, family), part in draws.groupby(['origin', 'family']):
        expected_origins = set(range(2003, int(origin) - 4))
        assert set(part.residual_origin) == expected_origins
        assert part.groupby('residual_origin').size().eq(110).all()
        assert not part.duplicated(CELL + ['residual_origin']).any()
        assert part.groupby(['residual_origin', 'horizon']).size().eq(22).all()
    assert draws.groupby(['origin', 'family']).ngroups == 15
    pop_fields = CELL + ['scenario', 'population']
    cells = points.merge(pop[pop_fields], on=CELL, how='left', validate='many_to_many')
    assert len(cells) == len(points) * 5
    cells['count'] = cells.prediction * cells.population / 100000
    predicted = aggregate(cells, config)
    transformed = draws.merge(pop[pop_fields], on=CELL, how='left', validate='many_to_many')
    transformed['count'] = transformed.rate_draw * transformed.population / 100000
    assert len(transformed) == len(draws) * 5
    statistics = aggregate(transformed, config, identifiers=CONTEXT + ['residual_origin'])
    intervals = conditional_intervals(statistics)
    assert len(predicted) == 3 * 5 * 5 * 5 * 35
    assert len(intervals) == len(predicted) * 3
    pop.to_csv(out / 'population_inputs.csv', index=False)
    predicted.to_csv(out / 'predictions.csv', index=False)
    intervals.to_csv(out / 'intervals.csv', index=False)
    commits = {p.name: sha(p) for p in sorted(out.iterdir())}
    write_json(out / 'transformation_commit.json', dict(created_utc=now(), sha256=commits,
        role='retrospective_sensitivity_not_new_holdout', future_population_used_in_diagnostics=True))
    # Only the scoring stage joins rate/count verification to transformed outputs.
    scoring_started = now()
    truth_cells = panel.loc[panel.location_name.eq(case['target']) & panel.outcome.eq(case['outcome']),
                            ['year', 'sex', 'age', 'rate', 'count']].rename(columns={
        'year': 'forecast_year', 'rate': 'observed_rate', 'count': 'native_count'})
    verified = cells.merge(truth_cells, on=['forecast_year', 'sex', 'age'], how='left', validate='many_to_one')
    assert np.isfinite(verified[['observed_rate', 'native_count']]).all().all()
    verified['observed_population'] = verified.native_count / verified.observed_rate * 100000
    verified['count'] = verified.native_count
    truth = aggregate(verified, config).rename(columns={'value': 'observed'})
    scored = predicted.merge(truth, on=CONTEXT + NODE, how='left', validate='one_to_one')
    scored['signed_error'] = scored.value - scored.observed
    scored['absolute_error'] = abs(scored.signed_error)
    scored['percent_error'] = 100 * scored.signed_error / scored.observed
    scored.to_csv(out / 'point_scores.csv', index=False)
    scored_intervals, wis = score(intervals, truth)
    scored_intervals.to_csv(out / 'interval_scores.csv.gz', index=False, compression='gzip')
    wis.to_csv(out / 'wis_scores.csv', index=False)
    accounting = error_accounting(verified)
    np.testing.assert_allclose(accounting.observed_count, accounting.native_count, rtol=1e-13, atol=1e-10)
    accounting.drop(columns='count').to_csv(out / 'cell_error_accounting.csv.gz', index=False, compression='gzip')
    effects = []
    for component in ['rate_effect', 'population_effect']:
        view = aggregate(accounting, config, value=component, allow_signed=True)
        view = view.rename(columns={'value': component})
        effects.append(view)
    effect = effects[0].merge(effects[1], on=CONTEXT + NODE, validate='one_to_one')
    effect = effect.merge(scored.loc[scored.measure.eq('count'), CONTEXT + NODE + ['value', 'observed', 'signed_error']],
                          on=CONTEXT + NODE, validate='one_to_one')
    np.testing.assert_allclose(effect.rate_effect + effect.population_effect, effect.signed_error, rtol=1e-10, atol=1e-8)
    effect.to_csv(out / 'count_error_accounting.csv', index=False)
    # Preserve original joint rate/population uncertainty as an explicitly separate reference.
    reference = read(old / 'interval_scores.csv.gz')
    reference = reference.loc[reference.family.isin(ROLES) & reference.measure.isin(['count', 'age_share'])].copy()
    reference = reference.rename(columns={'population_method': 'scenario'})
    reference['uncertainty_scope'] = 'joint_rate_population_original'
    reference['below_lower'] = reference.observed.lt(reference.lower)
    reference['above_upper'] = reference.observed.gt(reference.upper)
    reference.to_csv(out / 'original_joint_reference.csv.gz', index=False, compression='gzip')
    reference_wis = read(old / 'wis_scores.csv')
    reference_wis = reference_wis.loc[reference_wis.family.isin(ROLES) & reference_wis.measure.isin(['count', 'age_share'])].copy()
    reference_wis = reference_wis.rename(columns={'population_method': 'scenario'})
    reference_wis['uncertainty_scope'] = 'joint_rate_population_original'
    reference_wis.to_csv(out / 'original_joint_wis_reference.csv', index=False)
    assert len(reference) == 3 * 5 * 5 * 2 * 35 * 3
    assert all(sha(out / name) == digest for name, digest in commits.items())
    validation = dict(passed=True, case=case['id'], original_rates_reused=True, fitted_models=0,
        population_inputs=len(pop), point_statistics=len(predicted), conditional_interval_rows=len(intervals),
        original_joint_interval_rows=len(reference), whole_rate_blocks=int(len(statistics)),
        minimum_blocks=int(intervals.n_blocks.min()), maximum_blocks=int(intervals.n_blocks.max()),
        retrospective_diagnostics_use_future_population=True, scoring_started_utc=scoring_started,
        elapsed_seconds=time.perf_counter() - started)
    write_json(out / 'validation_report.json', validation)
    return validation


def source_tables(config, destination):
    panel = read(ROOT / 'data/processed/design_v1/regional_outcomes.csv')
    un = read(ROOT / 'data/processed/design_v1/un_population_1990_2028.csv')
    countries = [c['name'] for c in config['countries'] if c['gcc']]
    keys = ['location_name', 'sex', 'age', 'year']
    gbd = panel.loc[panel.location_name.isin(countries) & panel.outcome.isin(['prevalence', 'incidence'])
                    & panel.year.between(1990, 2023), keys + ['outcome', 'age_start', 'rate', 'count']].copy()
    gbd['gbd_population'] = gbd['count'] / gbd.rate * 100000
    compared = gbd.merge(un[keys + ['population_persons']], on=keys, how='left', validate='many_to_one')
    assert len(compared) == 6 * 2 * 2 * 11 * 34
    assert compared.population_persons.notna().all()
    compared['un_minus_gbd_percent'] = 100 * (compared.population_persons / compared.gbd_population - 1)
    compared.to_csv(destination / 'population_source_cells_1990_2023.csv', index=False)
    rows = []
    for (target, outcome, sex, year), part in compared.groupby(['location_name', 'outcome', 'sex', 'year']):
        for label, lower, upper in [('45-64', 45, 64), ('65-79', 65, 79), ('80+', 80, 110), ('45+', 45, 110)]:
            subset = part.loc[part.age_start.between(lower, upper)]
            n_gbd, n_un = subset.gbd_population.sum(), subset.population_persons.sum()
            rows.append(dict(target=target, outcome=outcome, sex=sex, year=year, age_group=label,
                             gbd_population=n_gbd, un_population=n_un, un_minus_gbd_percent=100*(n_un/n_gbd-1)))
    broad = pd.DataFrame(rows)
    broad.to_csv(destination / 'population_source_broad_1990_2023.csv', index=False)
    gastat = read(ROOT / 'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv')
    gastat = gastat.loc[gastat.year.eq(2023) & gastat.nationality.eq('Total') & gastat.sex.isin(config['sexes'])
                        & gastat.age_start.ge(45)].copy()
    assert len(gastat) == 16 and not gastat.duplicated(['sex', 'age_group']).any()
    local_rows = []
    for sex, part in gastat.groupby('sex'):
        for label, lower, upper in [('45-64', 45, 64), ('65-79', 65, 79), ('80+', 80, 110), ('45+', 45, 110)]:
            subset = part.loc[part.age_start.between(lower, upper)]
            local_rows.append(dict(sex=sex, age_group=label, gastat_population=subset.population_persons.sum(),
                                   gastat_source_cells='|'.join(subset.source_cell)))
    national = broad.loc[broad.target.eq('Saudi Arabia') & broad.year.eq(2023)].merge(
        pd.DataFrame(local_rows), on=['sex', 'age_group'], how='left', validate='many_to_one')
    assert len(national) == 16 and national.gastat_population.notna().all()
    national['un_minus_gastat_percent'] = 100 * (national.un_population/national.gastat_population - 1)
    national['gbd_minus_gastat_percent'] = 100 * (national.gbd_population/national.gastat_population - 1)
    national['role'] = 'source_comparison_reference_date_comparability_unverified'
    national.to_csv(destination / 'saudi_gastat_comparison_2023.csv', index=False)
    projections = projection_populations(panel, un, config)
    projections.to_csv(destination / 'population_scenarios_2024_2028.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--output', default='results/population_sensitivity_v1')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('Use one to twelve workers')
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError('Refusing existing population sensitivity directory')
    check_lock()
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    tests_path = ROOT / 'work/population-sensitivity-validation/tests.json'
    tests = json.loads(tests_path.read_text())
    assert tests['passed']
    assert all(sha(ROOT / name) == digest for name, digest in tests['tested_code_sha256'].items())
    code = ['src/gbd_park/population_sensitivity.py', 'scripts/run_population_sensitivity.py',
            'tests/test_population_sensitivity.py', 'study_design/population_sensitivity_implementation.md',
            'src/gbd_park/demography.py', 'src/gbd_park/intervals.py', 'scripts/run_local_baselines.py']
    cases = []
    for country in config['countries']:
        if country['gcc']:
            for outcome in ['prevalence', 'incidence']:
                identifier = country['iso3'] + '_' + outcome
                source = 'results/primary_v1' if identifier == 'SAU_prevalence' else 'results/secondary_v1/trials/' + identifier
                cases.append(dict(id=identifier, target=country['name'], outcome=outcome, source=source))
    source_hashes = {}
    for name in ['primary_v1', 'secondary_v1', 'demography_v1']:
        folder = ROOT / 'results' / name
        manifest_path = folder / 'run_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'complete'
        for filename, digest in manifest['output_sha256'].items():
            assert sha(folder / filename) == digest, str(folder / filename)
        source_hashes[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
    consumed = ['data/processed/design_v1/regional_outcomes.csv', 'data/processed/design_v1/un_population_1990_2028.csv',
                'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv',
                'study_design/locked_v1/design.json', 'study_design/locked_v1/lock_manifest.json',
                'work/population-sensitivity-validation/tests.json']
    for case in cases:
        consumed.extend([case['source'] + '/' + name for name in ['predictions.csv', 'joint_draws.csv']])
        consumed.extend(['results/demography_v1/' + case['id'] + '/' + name for name in
                         ['population_forecasts.csv', 'interval_scores.csv.gz', 'wis_scores.csv']])
    source_hashes.update({name: sha(ROOT / name) for name in consumed})
    code_hashes = {name: sha(ROOT / name) for name in code}
    out.mkdir(parents=True)
    manifest = dict(status='running', created_utc=now(), source_sha256=source_hashes, code_sha256=code_hashes,
                    workers=args.workers, cases=cases, disease_models_fitted=0,
                    role='retrospective_population_source_sensitivity_not_forecast_selection',
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__))
    write_json(out / 'run_manifest.json', manifest)
    started = time.perf_counter()
    source_tables(config, out)
    validations = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = [executor.submit(case_job, case, config, out) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            validations.append(result)
            print('Completed population sensitivity: ' + result['case'], flush=True)
    assert len(validations) == 12 and all(v['passed'] for v in validations)
    assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    assert all(sha(ROOT / name) == digest for name, digest in code_hashes.items())
    check_lock()
    write_json(out / 'validation_report.json', dict(passed=True, cases=validations, source_hashes_preserved=True,
        no_disease_models_fitted=True, elapsed_seconds=time.perf_counter() - started))
    manifest.update(status='complete', completed_utc=now(), output_sha256={
        str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run_manifest.json'})
    write_json(out / 'run_manifest.json', manifest)


if __name__ == '__main__':
    main()
