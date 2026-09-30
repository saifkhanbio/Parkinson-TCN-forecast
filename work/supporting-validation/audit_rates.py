"""Independent numerical/chronological audit of supporting rate forecasts.

No models are fitted. Numerical scores, selections, residuals, and quantiles
are reconstructed without importing the new production implementation.
Trusted frozen TCN checkpoints are replayed with the original inference API.
"""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
PANEL = pd.read_csv(ROOT / 'data/processed/design_v1/regional_outcomes.csv', float_precision='round_trip')
KEY = ['origin', 'family', 'sex', 'age', 'horizon', 'forecast_year']
COORD = ['sex', 'age', 'horizon']
ROLES = ['local_champion', 'nonneural_champion']
FAMILIES = CONFIG['models']['local_order'] + CONFIG['models']['nonneural_order'] + ['tcn_adapted', 'tcn_unadapted', 'tcn_intercept'] + ROLES


def read(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def same(actual, expected, keys, cols, rtol=3e-11, atol=3e-10):
    assert not actual.duplicated(keys).any(), ('actual duplicates', keys)
    assert not expected.duplicated(keys).any(), ('expected duplicates', keys)
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    assert left.index.equals(right.index), ('different keys', keys, len(left), len(right))
    for col in cols:
        np.testing.assert_allclose(left[col], right[col], rtol=rtol, atol=atol, err_msg=col)


def truth_for(target, outcome):
    truth = PANEL.loc[PANEL.location_name.eq(target) & PANEL.outcome.eq(outcome)].copy()
    return truth.rename(columns={'year': 'forecast_year', 'rate': 'observed_rate'})


def score_points(points, truth):
    assert 'observed_rate' not in points
    result = points.merge(truth[['sex', 'age', 'forecast_year', 'observed_rate']],
                          on=['sex', 'age', 'forecast_year'], validate='many_to_one')
    assert len(result) == len(points)
    result['absolute_log_error'] = abs(np.log(result.prediction) - np.log(result.observed_rate))
    result['absolute_rate_error'] = abs(result.prediction - result.observed_rate)
    result['log_residual'] = np.log(result.observed_rate) - result.log_prediction
    return result


def independent_quantiles(draws):
    parts = []
    for origin, part in draws.groupby('origin', sort=True):
        assert set(part.residual_origin) == set(range(2003, int(origin)-4))
        for scale, column in [('rate', 'rate_draw'), ('log_rate', 'log_draw')]:
            matrix = part.pivot(index=KEY, columns='residual_origin', values=column)
            assert np.isfinite(matrix.to_numpy()).all()
            quantiles = np.quantile(matrix.to_numpy(), [.025, .1, .25, .5, .75, .9, .975], axis=1, method='linear')
            for level, low, high in [(.5, 2, 4), (.8, 1, 5), (.95, 0, 6)]:
                frame = matrix.index.to_frame(index=False)
                frame['scale'], frame['level'] = scale, level
                frame['lower'], frame['median'], frame['upper'] = quantiles[low], quantiles[3], quantiles[high]
                frame['n_blocks'] = len(matrix.columns)
                parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def interval_scores(intervals, truth):
    result = intervals.merge(truth[['sex', 'age', 'forecast_year', 'observed_rate']],
                             on=['sex', 'age', 'forecast_year'], validate='many_to_one')
    result['observed_value'] = np.where(result.scale.eq('rate'), result.observed_rate, np.log(result.observed_rate))
    result['covered'] = result.observed_value.ge(result.lower) & result.observed_value.le(result.upper)
    result['width'] = result.upper - result.lower
    result['interval_score'] = result.width + 2/(1-result.level) * (
        np.maximum(result.lower-result.observed_value, 0) + np.maximum(result.observed_value-result.upper, 0))
    return result


def wis_scores(scored):
    keys = KEY + ['scale']
    result = scored.loc[scored.level.eq(.5), keys+['median', 'observed_value']].copy().reset_index(drop=True)
    result['median_absolute_error'] = abs(result.observed_value-result['median'])
    for levels, name in [([.5, .8], 'wis_50_80'), ([.5, .8, .95], 'wis_50_80_95')]:
        wis = .5 * result.median_absolute_error
        for level in levels:
            values = result[keys].merge(scored.loc[scored.level.eq(level), keys+['interval_score']], on=keys, validate='one_to_one')
            wis += (1-level)/2 * values.interval_score
        result[name] = wis/(len(levels)+.5)
    return result


def audit_selections(directory, history, points, mapping):
    candidates = {group: read(directory / ('candidate_'+group+'_scores.csv')) for group in ['local', 'nonneural', 'tcn']}
    for group, table in candidates.items():
        assert table.forecast_year.le(2018).all()
        assert table.outcome.eq(points.outcome.iloc[0]).all()
        for name in ['last_inner_label_year', 'history_end']:
            if name in table:
                assert table[name].dropna().le(table.loc[table[name].notna(), 'origin']).all()
    settings = pd.concat([read(directory / 'historical_setting_decisions.csv'), read(directory / 'settings_decisions.csv')], ignore_index=True)
    for decision in settings.itertuples():
        eligible = list(range(2003, int(decision.origin)-4))
        if not eligible:
            assert decision.selection_status == 'cold_start_defaults'
            continue
        group = 'local' if decision.family in CONFIG['models']['local_order'] else 'nonneural'
        table = candidates[group]
        table = table.loc[table.sex.eq(decision.sex) & table.family.eq(decision.family) & table.origin.isin(eligible) & table.horizon.eq(5)]
        summary = table.groupby('setting_id', as_index=False).agg(loss=('absolute_log_error', 'mean'),
                    parameter_count=('parameter_count', 'mean'), grid_order=('grid_order', 'first'))
        best = summary.sort_values(['loss', 'parameter_count', 'grid_order']).iloc[0]
        assert best.setting_id == decision.setting_id
        np.testing.assert_allclose(best.loss, decision.inner_loss, atol=2e-13, rtol=2e-11)
        assert int(decision.last_inner_label_year) == max(eligible)+5 <= decision.origin
    for selected in mapping.itertuples():
        families = CONFIG['models']['local_order' if selected.role == 'local_champion' else 'nonneural_order']
        eligible = [year for year in range(2009, 2014) if year+5 <= selected.fit_origin]
        data = history.loc[history.sex.eq(selected.sex) & history.family.isin(families) & history.origin.isin(eligible) & history.horizon.eq(5)]
        summary = data.groupby('family', as_index=False).agg(loss=('absolute_log_error', 'mean'), parameter_count=('parameter_count', 'mean'))
        summary['order'] = summary.family.map({family: i for i, family in enumerate(families)})
        best = summary.sort_values(['loss', 'parameter_count', 'order']).iloc[0]
        assert best.family == selected.source_family
        np.testing.assert_allclose(best.loss, selected.selection_loss, atol=2e-13, rtol=2e-11)
        assert selected.last_selection_target_year == max(eligible)+5 <= selected.fit_origin
        alias = points.loc[points.origin.eq(selected.fit_origin) & points.sex.eq(selected.sex) & points.family.eq(selected.role)]
        source = points.loc[points.origin.eq(selected.fit_origin) & points.sex.eq(selected.sex) & points.family.eq(selected.source_family)]
        same(alias, source, ['sex', 'age', 'horizon'], ['prediction', 'log_prediction'])
        assert alias.source_family.eq(selected.source_family).all()
    choices = json.loads((directory / 'tcn_choices.json').read_text())
    grid = [dict(channels=c, weight_decay=w, epochs=e) for c, w, e in itertools.product(
        CONFIG['models']['tcn']['channels'], CONFIG['models']['tcn']['weight_decay'], CONFIG['models']['tcn']['epochs'])]
    for choice in choices:
        eligible = list(range(2003, choice['fit_origin']-4))
        assert choice['inner_origins'] == eligible
        if not eligible:
            assert choice['status'] == 'cold_start_defaults'
            assert choice['last_inner_label_year'] is None
            continue
        assert choice['last_inner_label_year'] == max(eligible)+5 <= choice['fit_origin']
        table = candidates['tcn']
        table = table.loc[table.origin.isin(eligible) & table.horizon.eq(5) & table.family.eq('tcn_adapted')]
        options = []
        for order, base in enumerate(grid):
            ident = '__'.join(f'{key}={base[key]}' for key in ['channels', 'weight_decay', 'epochs'])
            penalties, losses = {}, []
            for sex in CONFIG['sexes']:
                allowed = table.loc[table.base_id.eq(ident) & table.sex.eq(sex)]
                candidates_sex = []
                for penalty_order, penalty in enumerate(CONFIG['adaptation']['penalties']):
                    selected = allowed.loc[allowed.adaptation_penalty.eq(penalty)]
                    assert len(selected) == len(eligible)*11
                    candidates_sex.append((selected.absolute_log_error.mean(), penalty_order, penalty))
                loss, _, penalty = min(candidates_sex)
                penalties[sex] = penalty
                losses.append(loss)
            count = 6*base['channels']**2+11*base['channels']+70
            options.append((float(np.mean(losses)), count, order, base, penalties))
        best = min(options, key=lambda row: row[:3])
        assert choice['base'] == best[3] and choice['penalties'] == best[4]
        np.testing.assert_allclose(choice['loss'], best[0], atol=2e-13, rtol=2e-11)
    return len(settings), len(mapping), len(choices)


def replay_checkpoint(directory, target, outcome):
    sys.path.insert(0, str(ROOT / 'src'))
    from gbd_park.tcn import load_checkpoint, predict_changes
    references = [json.loads(line) for line in (directory / 'seed_references.jsonl').read_text().splitlines() if line]
    replayed, fallbacks = 0, 0
    for item in references:
        if item['origin'] not in [2003, 2018] or item['seed'] != 11:
            continue
        payload = joblib.load(ROOT / item['payload_path'])
        assert payload['actual_target'] == target and payload['actual_outcome'] == outcome
        origin = item['origin']
        train = payload['training_meta']
        assert train.input_start.ge(1990).all() and train.label_end.le(origin).all()
        assert not train.country.eq(target).any()
        assert set(train.country) == {c['name'] for c in CONFIG['countries']} - {target}
        assert payload['target_meta'].country.eq(target).all() and payload['target_meta'].label_end.le(origin).all()
        raw = PANEL.loc[PANEL.location_name.eq(target) & PANEL.outcome.eq(outcome) & PANEL.year.between(1990, origin)]
        indexed = raw.set_index(['sex', 'age', 'year']).rate
        x, levels = [], []
        for sex in CONFIG['sexes']:
            for a, age in enumerate(CONFIG['ages']):
                history = np.log([indexed.loc[(sex, age, y)] for y in range(origin-7, origin+1)])
                age_vector = np.eye(len(CONFIG['ages']))[a]
                x.append(np.r_[history-history[-1], history[-1], float(sex == 'Male'), age_vector])
                levels.append(history[-1])
        np.testing.assert_allclose(payload['levels'], levels, atol=1e-13, rtol=1e-13)
        expected_y = []
        for row in payload['target_meta'].itertuples():
            before = np.log(indexed.loc[(row.sex, row.age, row.window_origin)])
            expected_y.append([np.log(indexed.loc[(row.sex, row.age, row.window_origin+h)])-before for h in range(1, 6)])
        np.testing.assert_allclose(payload['target_y'], expected_y, atol=1e-13, rtol=1e-13)
        if payload['audit']['status'] == 'ok':
            fitted = load_checkpoint(ROOT / item['checkpoint_path'])
            replay = predict_changes(fitted, np.asarray(x))
            np.testing.assert_allclose(replay, payload['current_changes'], atol=2e-7, rtol=2e-6)
            replayed += 1
        else:
            assert payload['audit']['reason']
            np.testing.assert_array_equal(payload['current_changes'], np.zeros((22, 5)))
            fallbacks += 1
    assert replayed+fallbacks == 2, ('Missing representative TCN replay/fallback record', target, outcome)
    return replayed


def replay_nonneural(out, case, target, outcome):
    """Replay one pooled and one donor ridge checkpoint against raw inputs."""
    sys.path.insert(0, str(ROOT / 'src'))
    from gbd_park.pooled import predict_changes
    matches = []
    for path in (out / 'jobs' / case).glob('*/complete.json'):
        job = json.loads(path.read_text())['job']
        if job['kind'] == 'nonneural' and job['origin'] == 2018:
            matches.append(path.parent)
    assert len(matches) == 1
    job_directory = matches[0]
    payload = joblib.load(job_directory / 'result.joblib')
    assert payload['actual_target'] == target and payload['actual_outcome'] == outcome
    history = PANEL.loc[PANEL.outcome.eq(outcome) & PANEL.year.between(1990, 2018)].set_index(
        ['location_name', 'sex', 'age', 'year']).rate
    x, levels = [], []
    for sex in CONFIG['sexes']:
        for a, age in enumerate(CONFIG['ages']):
            logs = np.log([history.loc[(target, sex, age, year)] for year in range(2011, 2019)])
            x.append(np.r_[logs-logs[-1], logs[-1], float(sex == 'Male'), np.eye(11)[a]])
            levels.append(logs[-1])
    x, levels = np.asarray(x), np.asarray(levels)
    forecasts = pd.DataFrame(payload['forecasts'])
    training = pd.DataFrame(payload['training'])
    assert training.input_start.ge(1990).all() and training.label_end.le(2018).all()
    replayed = 0
    for pool, family in [('pooled', 'pooled_ridge'), ('donor', 'donor_ridge_unadapted')]:
        fits = [fit for fit in payload['fits'] if fit['pool'] == pool and fit['base']['algorithm'] == 'ridge']
        fit = fits[0]
        assert fit['status'] == 'ok', ('Unexpected representative ridge fallback', case, pool)
        assert fit['minimum_input_year'] == 1990 and fit['maximum_label_year'] == 2018
        countries = [country['name'] for country in CONFIG['countries'] if pool == 'pooled' or country['name'] != target]
        assert fit['countries'] == countries
        fitted = joblib.load(job_directory / 'checkpoints' / (fit['model_id']+'.joblib'))
        train_x = []
        for country in countries:
            for sex in CONFIG['sexes']:
                for a, age in enumerate(CONFIG['ages']):
                    for year in range(1997, 2014):
                        logs = np.log([history.loc[(country, sex, age, y)] for y in range(year-7, year+1)])
                        train_x.append(np.r_[logs-logs[-1], logs[-1], float(sex == 'Male'), np.eye(11)[a]])
        train_x = np.asarray(train_x)
        np.testing.assert_allclose(fitted['scaler'].mean_, train_x.mean(axis=0), atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(fitted['scaler'].var_, train_x.var(axis=0), atol=2e-12, rtol=2e-11)
        changes = predict_changes(fitted, x)
        expected = []
        for index, (sex, age) in enumerate(itertools.product(CONFIG['sexes'], CONFIG['ages'])):
            for h in range(1, 6):
                expected.append(dict(sex=sex, age=age, horizon=h, log_prediction=levels[index]+changes[index, h-1]))
        actual = forecasts.loc[forecasts.family.eq(family) & forecasts.model_id.eq(fit['model_id'])]
        same(actual, pd.DataFrame(expected), COORD, ['log_prediction'], atol=2e-12, rtol=2e-11)
        replayed += 1
    return replayed


def audit_case(task):
    out, country, outcome = task
    started = time.perf_counter()
    case = country['iso3']+'_'+outcome
    directory = out / 'trials' / case
    points = read(directory / 'predictions.csv')
    assert len(points) == 8800 and set(points.family) == set(FAMILIES)
    assert points.target.eq(country['name']).all() and points.outcome.eq(outcome).all()
    expected = set(itertools.product(range(2014, 2019), FAMILIES, CONFIG['sexes'], CONFIG['ages'], range(1, 6)))
    assert set(map(tuple, points[['origin', 'family', 'sex', 'age', 'horizon']].to_numpy())) == expected
    assert points.forecast_year.eq(points.origin+points.horizon).all()
    assert np.isfinite(points.prediction).all() and points.prediction.gt(0).all()
    np.testing.assert_allclose(np.log(points.prediction), points.log_prediction, atol=1e-12, rtol=1e-12)
    truth = truth_for(country['name'], outcome)
    scored = score_points(points, truth)
    same(read(directory / 'point_scores.csv'), scored, KEY, ['observed_rate', 'absolute_log_error', 'absolute_rate_error'])
    historical = read(directory / 'prequential_predictions.csv')
    assert len(historical) == 16940 and historical.outcome.eq(outcome).all()
    assert historical.forecast_year.le(2018).all()
    for col in ['history_start', 'history_end', 'last_inner_label_year']:
        if col == 'history_start':
            assert historical[col].dropna().ge(1990).all()
        else:
            assert historical[col].dropna().le(historical.loc[historical[col].notna(), 'origin']).all()
    history = score_points(historical, truth)
    same(read(directory / 'prequential_scores.csv'), history, KEY, ['observed_rate', 'absolute_log_error', 'log_residual'])
    mapping = read(directory / 'champion_family_mappings.csv')
    decisions = audit_selections(directory, history, points, mapping)
    draws = read(directory / 'joint_draws.csv')
    assert len(draws) == 79200 and draws.outcome.eq(outcome).all()
    rebuilt = []
    coords = pd.MultiIndex.from_product([CONFIG['sexes'], CONFIG['ages'], range(1, 6)], names=COORD)
    for (origin, family), part in points.groupby(['origin', 'family'], sort=True):
        bank = joblib.load(directory / 'banks' / f'origin{origin}__{family}.joblib')
        if family in ROLES:
            families_by_sex = mapping.loc[mapping.fit_origin.eq(origin) & mapping.role.eq(family)].set_index('sex').source_family.to_dict()
        else:
            families_by_sex = {sex: family for sex in CONFIG['sexes']}
        origins = list(range(2003, int(origin)-4))
        expected_index = pd.MultiIndex.from_product([origins, CONFIG['sexes'], CONFIG['ages'], range(1, 6)], names=['origin']+COORD)
        pieces = [history.loc[history.sex.eq(sex) & history.family.eq(source) & history.origin.isin(origins)] for sex, source in families_by_sex.items()]
        source = pd.concat(pieces).set_index(['origin']+COORD).reindex(expected_index)
        residuals = source.log_residual.to_numpy().reshape(len(origins), 110)
        assert np.isfinite(residuals).all()
        centered = residuals-residuals.mean(axis=0)
        assert bank['origins'] == origins and bank['n_blocks'] == len(origins)
        assert bank['outcome'] == outcome and bank['target'] == country['name']
        assert bank['source_family_by_sex'] == families_by_sex
        np.testing.assert_allclose(bank['raw_residuals'], residuals, atol=2e-12, rtol=2e-11)
        np.testing.assert_allclose(bank['centered_residuals'], centered, atol=2e-12, rtol=2e-11)
        ordered = part.set_index(COORD).reindex(coords).reset_index()
        for index, residual_origin in enumerate(origins):
            block = ordered[KEY].copy()
            block['residual_origin'] = residual_origin
            block['log_draw'] = ordered.log_prediction.to_numpy()+centered[index]
            block['rate_draw'] = np.exp(block.log_draw)
            rebuilt.append(block)
    expected_draws = pd.concat(rebuilt, ignore_index=True)
    same(draws, expected_draws, KEY+['residual_origin'], ['log_draw', 'rate_draw'])
    intervals = independent_quantiles(expected_draws)
    same(read(directory / 'intervals.csv'), intervals, KEY+['scale', 'level'], ['lower', 'median', 'upper', 'n_blocks'])
    scored_intervals = interval_scores(intervals, truth)
    same(read(directory / 'interval_scores.csv'), scored_intervals, KEY+['scale', 'level'], ['observed_value', 'covered', 'width', 'interval_score'])
    wis = wis_scores(scored_intervals)
    same(read(directory / 'wis_scores.csv'), wis, KEY+['scale'], ['observed_value', 'wis_50_80', 'wis_50_80_95'])
    replays = replay_checkpoint(directory, country['name'], outcome)
    nonneural_replays = replay_nonneural(out, case, country['name'], outcome)
    verdict = json.loads((directory / 'endpoint_verdict.json').read_text())
    assert verdict['target'] == country['name'] and verdict['outcome'] == outcome
    assert verdict['saudi_prevalence_primary_result_replaced'] is False
    expected_role = 'support' if outcome in ['deaths', 'ylds', 'ylls', 'dalys'] else 'secondary'
    assert expected_role in verdict['endpoint_role']
    return dict(case=case, passed=True, prediction_rows=len(points), interval_rows=len(intervals),
                joint_draw_rows=len(draws), residual_banks=80, independently_reconstructed_settings=decisions[0],
                independently_reconstructed_champions=decisions[1], independently_reconstructed_tcn_choices=decisions[2],
                actual_outcome_checkpoint_replays=replays, nonneural_checkpoint_replays=nonneural_replays,
                elapsed_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/supporting_v1')
    parser.add_argument('--case')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='audit_rates.json')
    args = parser.parse_args()
    out = ROOT / args.run
    tasks = [(out, country, outcome) for country in CONFIG['countries'] if country['gcc']
             for outcome in ['deaths', 'ylds', 'ylls', 'dalys']]
    if args.case:
        tasks = [task for task in tasks if task[1]['iso3']+'_'+task[2] == args.case]
        assert len(tasks) == 1
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, digest in manifest['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for run, digest in manifest['prior_manifests_sha256'].items():
        assert sha(ROOT / 'results' / run / 'run_manifest.json') == digest, run
    for name, digest in manifest['output_sha256'].items():
        assert sha(out / name) == digest, name
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(audit_case, tasks))
    event = json.loads((out / 'events.json').read_text())
    issued = next(row for row in event if row['event'] == 'all_trials_issued_before_final_scoring')
    scoring = next(row for row in event if row['event'] == 'evaluation_scoring_started')
    assert datetime.fromisoformat(issued['time_utc']) <= datetime.fromisoformat(scoring['time_utc'])
    for case in issued['issued_commit_sha256']:
        path = out / 'trials' / case / 'issued_commit.json'
        assert sha(path) == issued['issued_commit_sha256'][case]
        assert datetime.fromisoformat(json.loads(path.read_text())['committed_utc']) <= datetime.fromisoformat(scoring['time_utc'])
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), run=args.run,
                  audit_code_sha256=sha(__file__), source_sha256=sha(ROOT / 'data/processed/design_v1/regional_outcomes.csv'),
                  run_manifest_sha256=sha(out / 'run_manifest.json'), cases=cases,
                  all_output_hashes_verified=True, all_issued_before_final_scoring=True)
    (HERE / args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, cases=len(cases), checkpoint_replays=sum(c['actual_outcome_checkpoint_replays'] for c in cases))))


if __name__ == '__main__':
    main()
