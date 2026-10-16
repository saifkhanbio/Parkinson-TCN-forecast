"""Independent replay of frozen population scenarios; no production imports."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/population_sensitivity_v1'
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
PANEL = pd.read_csv(ROOT / 'data/processed/design_v1/regional_outcomes.csv', float_precision='round_trip')
UN = pd.read_csv(ROOT / 'data/processed/design_v1/un_population_1990_2028.csv', float_precision='round_trip')
ROLES = ['tcn_adapted', 'local_champion', 'nonneural_champion']
SCENARIOS = ['log_trend_last8', 'persistence', 'gbd_realized_oracle', 'un_2024_unaligned', 'un_2024_origin_aligned']
CELL = ['origin', 'horizon', 'forecast_year', 'sex', 'age']
CONTEXT = ['origin', 'horizon', 'forecast_year', 'family', 'scenario']
KEY = CONTEXT + ['node']
BOTTOM = [(s, a) for s in ['Male', 'Female'] for a in CONFIG['ages']]
NODES = []
MATRIX = []
for i, (sex, age) in enumerate(BOTTOM):
    NODES.append(f'{sex}__{age}')
    MATRIX.append(np.eye(22)[i])
for sex in ['Male', 'Female']:
    for group, low, high in [('45-64', 45, 64), ('65-79', 65, 79), ('80+', 80, 200)]:
        NODES.append(f'{sex}__{group}')
        MATRIX.append([float(s == sex and low <= int(a[:2]) <= high) for s, a in BOTTOM])
for sex in ['Male', 'Female']:
    NODES.append(f'{sex}__45+')
    MATRIX.append([float(s == sex) for s, a in BOTTOM])
NODES.append('Both__45+')
MATRIX.append(np.ones(22))
MATRIX = np.asarray(MATRIX)


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def same(actual, expected, keys, columns, rtol=2e-11, atol=2e-9):
    assert not actual.duplicated(keys).any(), ('actual duplicate', keys)
    assert not expected.duplicated(keys).any(), ('expected duplicate', keys)
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    assert a.index.equals(b.index), ('key mismatch', len(a), len(b), keys)
    for col in columns:
        np.testing.assert_allclose(a[col], b[col], rtol=rtol, atol=atol, err_msg=col)


def aggregate(frame, draw=False, shares=True):
    keys = CONTEXT + (['residual_origin'] if draw else [])
    assert not frame.duplicated(keys + ['sex', 'age']).any()
    pivot = frame.pivot(index=keys, columns=['sex', 'age'], values='count').reindex(
        columns=pd.MultiIndex.from_tuples(BOTTOM))
    values = pivot.to_numpy()
    assert np.isfinite(values).all()
    stats = pd.DataFrame(values @ MATRIX.T, index=pivot.index, columns=NODES)
    for sex in ['Male', 'Female'] if shares else []:
        use = np.array([s == sex for s, a in BOTTOM])
        for threshold in [65, 80]:
            older = np.array([s == sex and int(a[:2]) >= threshold for s, a in BOTTOM])
            stats[f'{sex}__{threshold}+_within_45+'] = values[:, older].sum(axis=1) / values[:, use].sum(axis=1) * 100
    return stats.rename_axis(columns='node').stack().rename('value').reset_index()


def source_population(case, target, outcome):
    panel = PANEL.loc[PANEL.location_name.eq(target) & PANEL.outcome.eq(outcome)].copy()
    panel['n'] = panel['count'] / panel.rate * 100000
    truth = panel.set_index(['year', 'sex', 'age'])
    un = UN.loc[UN.location_name.eq(target)].set_index(['year', 'sex', 'age'])
    saved = read(ROOT / 'results/demography_v1' / case / 'population_forecasts.csv')
    records = []
    for o in range(2014, 2019):
        for sex, age in BOTTOM:
            hist = np.log([truth.loc[(y, sex, age), 'n'] for y in range(o - 7, o + 1)])
            x = np.arange(-7, 1)
            slope = np.sum((x - x.mean()) * (hist - hist.mean())) / np.sum((x - x.mean()) ** 2)
            intercept = hist.mean() - slope * x.mean()
            for h in range(1, 6):
                ng0 = truth.loc[(o, sex, age), 'n']
                ng1 = truth.loc[(o + h, sex, age), 'n']
                nu0 = un.loc[(o, sex, age), 'population_persons']
                nu1 = un.loc[(o + h, sex, age), 'population_persons']
                expected = [np.exp(intercept + slope * h), np.exp(hist[-1]), ng1, nu1, ng0 * nu1 / nu0]
                for scenario, pop in zip(SCENARIOS, expected):
                    records.append(dict(origin=o, horizon=h, forecast_year=o+h, sex=sex, age=age,
                                        scenario=scenario, population=pop, origin_gbd=ng0, future_gbd=ng1,
                                        origin_un=nu0, future_un=nu1))
    result = pd.DataFrame(records)
    operational = saved.rename(columns={'population_method': 'scenario'})
    same(operational, result.loc[result.scenario.isin(SCENARIOS[:2])], CELL + ['scenario'], ['population'])
    return result, panel


def independent_intervals(stats):
    rows = []
    for origin, group in stats.groupby('origin', sort=True):
        assert set(group.residual_origin) == set(range(2003, int(origin) - 4))
        pivot = group.pivot(index=KEY, columns='residual_origin', values='value')
        assert np.isfinite(pivot.to_numpy()).all()
        q = np.quantile(pivot.to_numpy(), [.025, .1, .25, .5, .75, .9, .975], method='linear', axis=1)
        for level, low, high in [(.5, 2, 4), (.8, 1, 5), (.95, 0, 6)]:
            frame = pivot.index.to_frame(index=False)
            frame['level'], frame['lower'], frame['median'], frame['upper'], frame['n_blocks'] = level, q[low], q[3], q[high], len(pivot.columns)
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def case_audit(country, outcome):
    started = time.perf_counter()
    case = country['iso3'] + '_' + outcome
    directory = OUT / case
    source = ROOT / ('results/primary_v1' if case == 'SAU_prevalence' else 'results/secondary_v1/trials/' + case)
    pops, panel = source_population(case, country['name'], outcome)
    actual_pop = read(directory / 'population_inputs.csv')
    assert len(actual_pop) == 2750
    assert set(actual_pop.age) == set(CONFIG['ages'])
    same(actual_pop, pops, CELL + ['scenario'], ['population', 'origin_gbd', 'future_gbd', 'origin_un', 'future_un'])
    assert actual_pop.loc[actual_pop.scenario.eq('un_2024_origin_aligned'), 'alignment_year'].eq(
        actual_pop.loc[actual_pop.scenario.eq('un_2024_origin_aligned'), 'origin']).all()
    assert actual_pop.uses_future_population_information.eq(~actual_pop.scenario.isin(SCENARIOS[:2])).all()
    points = read(source / 'predictions.csv')
    points = points.loc[points.family.isin(ROLES)]
    draws = read(source / 'joint_draws.csv')
    draws = draws.loc[draws.family.isin(ROLES)]
    assert len(points) == 1650
    for (origin, family), part in draws.groupby(['origin', 'family']):
        assert set(part.residual_origin) == set(range(2003, origin - 4))
        assert part.groupby('residual_origin').size().eq(110).all()
        assert not part.duplicated(['residual_origin', 'sex', 'age', 'horizon']).any()
    point_cells = points.merge(pops, on=CELL, validate='many_to_many')
    point_cells['count'] = point_cells.prediction * point_cells.population / 100000
    expected_points = aggregate(point_cells)
    actual_points = read(directory / 'predictions.csv')
    same(actual_points, expected_points, KEY, ['value'])
    assert set(actual_points.target) == {country['name']} and set(actual_points.outcome) == {outcome}
    assert actual_points.sex.eq(actual_points.node.str.split('__').str[0]).all()
    assert actual_points.age_group.eq(actual_points.node.str.split('__').str[1]).all()
    assert actual_points.measure.eq(np.where(actual_points.node.str.contains('within'), 'age_share', 'count')).all()
    assert actual_points.unit.eq(np.where(actual_points.node.str.contains('within'), 'percent', 'modeled_number')).all()
    draw_cells = draws.merge(pops, on=CELL, validate='many_to_many')
    draw_cells['count'] = draw_cells.rate_draw * draw_cells.population / 100000
    expected_intervals = independent_intervals(aggregate(draw_cells, True))
    actual_intervals = read(directory / 'intervals.csv')
    same(actual_intervals, expected_intervals, KEY + ['level'], ['lower', 'median', 'upper', 'n_blocks'])
    assert set(actual_intervals.uncertainty_scope) == {'rate_conditional_fixed_population'}
    truth = panel.rename(columns={'year': 'forecast_year', 'rate': 'observed_rate', 'count': 'native_count'})
    point_cells = point_cells.merge(truth[['sex', 'age', 'forecast_year', 'observed_rate', 'native_count']],
                                    on=['sex', 'age', 'forecast_year'], validate='many_to_one')
    point_cells['observed_population'] = point_cells.native_count / point_cells.observed_rate * 100000
    true_cells = point_cells.copy()
    true_cells['count'] = true_cells.native_count
    true_nodes = aggregate(true_cells).rename(columns={'value': 'observed'})
    point_scores = expected_points.merge(true_nodes, on=KEY, validate='one_to_one')
    point_scores['signed_error'] = point_scores.value - point_scores.observed
    point_scores['absolute_error'] = abs(point_scores.signed_error)
    point_scores['absolute_log_error'] = abs(np.log(point_scores.value / point_scores.observed))
    point_scores['percent_error'] = 100 * point_scores.signed_error / point_scores.observed
    same(read(directory / 'point_scores.csv'), point_scores, KEY,
         ['value', 'observed', 'signed_error', 'absolute_error', 'percent_error'])
    scores = expected_intervals.merge(true_nodes, on=KEY, validate='many_to_one')
    scores['width'] = scores.upper - scores.lower
    scores['covered'] = scores.observed.ge(scores.lower) & scores.observed.le(scores.upper)
    scores['below_lower'] = scores.observed.lt(scores.lower)
    scores['above_upper'] = scores.observed.gt(scores.upper)
    alpha = 1 - scores.level
    scores['interval_score'] = scores.width + 2 / alpha * np.maximum(scores.lower-scores.observed, 0) + 2 / alpha * np.maximum(scores.observed-scores.upper, 0)
    saved_scores = read(directory / 'interval_scores.csv.gz')
    same(saved_scores, scores, KEY + ['level'], ['observed', 'lower', 'median', 'upper', 'width', 'covered', 'interval_score', 'below_lower', 'above_upper'])
    wbase = scores.loc[scores.level.eq(.5), KEY + ['observed', 'median']].copy().reset_index(drop=True)
    for levels, name in [([.5, .8], 'wis_50_80'), ([.5, .8, .95], 'wis_50_80_95')]:
        wis = .5 * abs(wbase.observed-wbase['median'])
        for level in levels:
            partial = scores.loc[scores.level.eq(level), KEY + ['interval_score']]
            partial = wbase[KEY].merge(partial, on=KEY, validate='one_to_one').interval_score
            wis += (1-level)/2 * partial
        wbase[name] = wis / (len(levels)+.5)
    same(read(directory / 'wis_scores.csv'), wbase, KEY, ['wis_50_80', 'wis_50_80_95'])
    error = point_cells.copy()
    error['predicted_count'] = error['count']
    error['observed_count'] = error.native_count
    error['log_rate_error'] = np.log(error.prediction/error.observed_rate)
    error['log_population_error'] = np.log(error.population/error.observed_population)
    error['log_count_error'] = np.log(error.predicted_count/error.native_count)
    error['rate_effect'] = (error.prediction-error.observed_rate)*(error.population+error.observed_population)/200000
    error['population_effect'] = (error.population-error.observed_population)*(error.prediction+error.observed_rate)/200000
    same(read(directory / 'cell_error_accounting.csv.gz'), error, CONTEXT + ['sex', 'age'],
         ['prediction', 'observed_rate', 'population', 'observed_population', 'predicted_count', 'observed_count',
          'log_rate_error', 'log_population_error', 'log_count_error', 'rate_effect', 'population_effect'])
    np.testing.assert_allclose(error.rate_effect+error.population_effect, error.predicted_count-error.observed_count, atol=2e-9, rtol=2e-11)
    np.testing.assert_allclose(error.log_rate_error+error.log_population_error, error.log_count_error, atol=1e-13)
    accounting = []
    for variable in ['rate_effect', 'population_effect', 'predicted_count', 'observed_count']:
        cells = error.copy()
        cells['count'] = cells[variable]
        view = aggregate(cells, shares=False)
        view = view.loc[view.node.isin(NODES)].rename(columns={'value': variable})
        accounting.append(view)
    expected_accounting = accounting[0]
    for view in accounting[1:]:
        expected_accounting = expected_accounting.merge(view, on=KEY, validate='one_to_one')
    expected_accounting = expected_accounting.rename(columns={'predicted_count': 'value', 'observed_count': 'observed'})
    same(read(directory / 'count_error_accounting.csv'), expected_accounting, KEY,
         ['rate_effect', 'population_effect', 'value', 'observed'])
    reference = read(directory / 'original_joint_reference.csv.gz')
    old = read(ROOT / 'results/demography_v1' / case / 'interval_scores.csv.gz')
    old = old.loc[old.family.isin(ROLES) & old.measure.isin(['count', 'age_share'])]
    old = old.rename(columns={'population_method': 'scenario'})
    reference_keys = KEY + ['level']
    same(reference, old, reference_keys, ['observed', 'lower', 'median', 'upper', 'width', 'covered', 'interval_score', 'n_blocks'])
    assert set(reference.uncertainty_scope) == {'joint_rate_population_original'}
    old_wis = read(ROOT / 'results/demography_v1' / case / 'wis_scores.csv')
    old_wis = old_wis.loc[old_wis.family.isin(ROLES) & old_wis.measure.isin(['count', 'age_share'])].rename(columns={'population_method': 'scenario'})
    same(read(directory / 'original_joint_wis_reference.csv'), old_wis, KEY, ['observed', 'median', 'n_blocks', 'wis_50_80', 'wis_50_80_95'])
    commit = json.loads((directory / 'transformation_commit.json').read_text())
    validation = json.loads((directory / 'validation_report.json').read_text())
    assert all(sha(directory / name) == digest for name, digest in commit['sha256'].items())
    assert datetime.fromisoformat(commit['created_utc']) <= datetime.fromisoformat(validation['scoring_started_utc'])
    return dict(case=case, passed=True, population_rows=len(pops), point_statistics=len(actual_points),
                conditional_interval_rows=len(actual_intervals), original_joint_reference_rows=len(reference),
                complete_joint_blocks_verified=True, frozen_rates_replayed=True,
                exact_count_and_log_error_identities_verified=True, elapsed_seconds=time.perf_counter()-started)


def source_audit():
    countries = [c['name'] for c in CONFIG['countries'] if c['gcc']]
    panel = PANEL.loc[PANEL.location_name.isin(countries) & PANEL.outcome.isin(['prevalence', 'incidence'])].copy()
    panel['gbd_population'] = panel['count']/panel.rate*100000
    keys = ['location_name', 'sex', 'age', 'year']
    expected = panel.merge(UN[keys+['population_persons']], on=keys, validate='many_to_one')
    expected['un_minus_gbd_percent'] = 100*(expected.population_persons/expected.gbd_population-1)
    same(read(OUT/'population_source_cells_1990_2023.csv'), expected, keys+['outcome'],
         ['rate', 'count', 'gbd_population', 'population_persons', 'un_minus_gbd_percent'])
    broad = []
    for (target, outcome, sex, year), part in expected.groupby(['location_name', 'outcome', 'sex', 'year']):
        for label, lo, hi in [('45-64',45,64), ('65-79',65,79), ('80+',80,200), ('45+',45,200)]:
            subset = part.loc[part.age_start.between(lo, hi)]
            g = subset.gbd_population.sum()
            u = subset.population_persons.sum()
            broad.append(dict(target=target, outcome=outcome, sex=sex, year=year, age_group=label,
                              gbd_population=g, un_population=u, un_minus_gbd_percent=100*(u/g-1)))
    broad = pd.DataFrame(broad)
    bkeys = ['target', 'outcome', 'sex', 'year', 'age_group']
    same(read(OUT/'population_source_broad_1990_2023.csv'), broad, bkeys,
         ['gbd_population', 'un_population', 'un_minus_gbd_percent'])
    wide = read(ROOT/'supporting_data/2026-09-26/processed/un_wpp2024_gcc_age_sex_1990_2050.csv')
    wide = wide.loc[wide.year.le(2028) & wide.sex.isin(['Male','Female']) & wide.age_start.ge(45)].copy()
    np.testing.assert_allclose(wide.population_persons, wide.population_thousands_original*1000, rtol=1e-13, atol=1e-8)
    wide['age_start'] = wide.age_start.clip(upper=95)
    wide['age'] = wide.age_start.map(lambda a: '95+' if a == 95 else f'{a}-{a+4}')
    other = wide.groupby(['iso3','year','sex','age'], as_index=False).population_persons.sum()
    saved = UN.loc[UN.location_name.isin(countries)]
    same(saved, other, ['iso3','year','sex','age'], ['population_persons'], rtol=0, atol=1e-8)
    gastat = read(ROOT/'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv')
    gastat = gastat.loc[gastat.year.eq(2023) & gastat.nationality.eq('Total') & gastat.sex.isin(['Male','Female']) & gastat.age_start.ge(45)]
    assert len(gastat)==16
    import zipfile
    import posixpath
    from xml.etree import ElementTree
    ns = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(ROOT/'supporting_data/2026-09-26/raw/GASTAT_Population_Estimates_EN.xlsx') as book:
        workbook = ElementTree.fromstring(book.read('xl/workbook.xml'))
        relationships = ElementTree.fromstring(book.read('xl/_rels/workbook.xml.rels'))
        targets = {e.attrib['Id']: e.attrib['Target'] for e in relationships}
        cells = {}
        for sheet in workbook.find('s:sheets',ns):
            if sheet.attrib['name'] not in set(gastat.source_sheet):
                continue
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = targets[rid]
            path = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
            worksheet = ElementTree.fromstring(book.read(path))
            cells[sheet.attrib['name']] = {c.attrib['r']: c.find('s:v',ns).text for c in worksheet.findall('.//s:c',ns) if c.find('s:v',ns) is not None}
        for row in gastat.itertuples():
            assert float(cells[row.source_sheet][row.source_cell]) == row.population_persons
    nat = []
    for sex, part in gastat.groupby('sex'):
        for label, lo, hi in [('45-64',45,64), ('65-79',65,79), ('80+',80,200), ('45+',45,200)]:
            nat.append(dict(sex=sex, age_group=label, gastat_population=part.loc[part.age_start.between(lo,hi)].population_persons.sum()))
    nat = broad.loc[broad.target.eq('Saudi Arabia') & broad.year.eq(2023)].merge(pd.DataFrame(nat), on=['sex','age_group'], validate='many_to_one')
    nat['un_minus_gastat_percent'] = 100*(nat.un_population/nat.gastat_population-1)
    nat['gbd_minus_gastat_percent'] = 100*(nat.gbd_population/nat.gastat_population-1)
    same(read(OUT/'saudi_gastat_comparison_2023.csv'), nat, bkeys,
         ['gbd_population','un_population','gastat_population','un_minus_gastat_percent','gbd_minus_gastat_percent'])
    base = panel.loc[panel.year.eq(2023), ['location_name','sex','age','outcome','gbd_population']].rename(columns={'gbd_population':'gbd_baseline'})
    pkeys = ['location_name','sex','age']
    base = base.merge(UN.loc[UN.year.eq(2023),pkeys+['population_persons']].rename(columns={'population_persons':'un_baseline'}), on=pkeys, validate='many_to_one')
    future = UN.loc[UN.location_name.isin(countries) & UN.year.between(2024,2028), pkeys+['year','population_persons']].rename(columns={'population_persons':'un_future'})
    future = base.merge(future, on=pkeys, validate='many_to_many')
    handoff = []
    for scenario in ['un_medium_unaligned','gbd_2023_aligned_un_growth']:
        frame=future.copy()
        frame['scenario']=scenario
        frame['population']=frame.un_future if scenario=='un_medium_unaligned' else frame.gbd_baseline*frame.un_future/frame.un_baseline
        handoff.append(frame)
    handoff=pd.concat(handoff,ignore_index=True)
    saved_handoff=read(OUT/'population_scenarios_2024_2028.csv')
    same(saved_handoff,handoff,pkeys+['outcome','year','scenario'],['gbd_baseline','un_baseline','un_future','population'])
    assert len(saved_handoff)==2640 and saved_handoff.origin.eq(2023).all()
    assert set(saved_handoff.unit)=={'persons'} and set(saved_handoff.role)=={'population_only_handoff_not_disease_forecast'}
    return dict(passed=True, source_cell_rows=len(expected), source_broad_rows=len(broad),
                independent_UN_rows=len(saved), UN95plus_sum_verified=True, original_thousands_to_persons_verified=True,
                raw_GASTAT_cells_verified=len(gastat), GASTAT_comparison_rows=len(nat), population_handoff_rows=len(handoff))


def main():
    started = time.perf_counter()
    audit_inputs = ['study_design/locked_v1/design.json', 'data/processed/design_v1/regional_outcomes.csv',
                    'data/processed/design_v1/un_population_1990_2028.csv',
                    'supporting_data/2026-09-26/processed/un_wpp2024_gcc_age_sex_1990_2050.csv',
                    'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv',
                    'supporting_data/2026-09-26/raw/GASTAT_Population_Estimates_EN.xlsx']
    audit_hashes = {name: sha(ROOT/name) for name in audit_inputs}
    manifest = json.loads((OUT/'run_manifest.json').read_text())
    assert manifest['status']=='complete'
    for name,digest in manifest['source_sha256'].items():
        assert sha(ROOT/name)==digest, name
    for name,digest in manifest['code_sha256'].items():
        assert sha(ROOT/name)==digest, name
    for name,digest in manifest['output_sha256'].items():
        assert sha(OUT/name)==digest, name
    sources = source_audit()
    cases = [(country, outcome) for country in CONFIG['countries'] if country['gcc']
             for outcome in ['prevalence', 'incidence']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda pair: case_audit(*pair), cases))
    assert all(sha(ROOT/name)==digest for name,digest in audit_hashes.items())
    report = dict(passed=True, cases=results, source_audit=sources, audit_script_sha256=sha(__file__),
                  audited_code_sha256={str(Path(__file__).relative_to(ROOT)): sha(__file__)},
                  audit_input_sha256=audit_hashes,
                  run_manifest_sha256=sha(OUT/'run_manifest.json'),
                  elapsed_seconds=time.perf_counter()-started,
                  numerical_production_helpers_imported=False)
    (HERE / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
