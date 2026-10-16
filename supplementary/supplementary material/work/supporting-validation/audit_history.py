"""Read-only audit of the fixed 1980 versus 1990 mortality-history sensitivity."""
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'

from datetime import datetime, timezone
import argparse
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
from audit_rates import ROOT, CONFIG, PANEL, read, sha, same, truth_for, score_points
sys.path.insert(0, str(ROOT / 'src'))
from gbd_park.tcn import load_checkpoint, predict_changes

OUT = ROOT / 'results/mortality_history_v1'
KEY = ['target', 'outcome', 'history_start', 'family', 'sex', 'age', 'horizon']


def audit_arm(task, points, seeds, adaptation):
    country, outcome, start = task
    ident = f"{country['iso3']}_{outcome}_history{start}"
    data = points.loc[points.trial_id.eq(ident)].copy()
    assert len(data) == 330 and data.origin.eq(2018).all() and data.history_start.eq(start).all()
    assert data.target.eq(country['name']).all() and data.outcome.eq(outcome).all()
    assert data.forecast_year.eq(2018+data.horizon).all()
    assert set(map(tuple, data[['family', 'sex', 'age', 'horizon']].to_numpy())) == set(itertools.product(
        ['tcn_adapted', 'tcn_unadapted', 'damped_ets'], CONFIG['sexes'], CONFIG['ages'], range(1, 6)))
    refs = [record for record in seeds if record['trial_id'] == ident]
    assert [record['seed'] for record in refs] == CONFIG['models']['tcn']['ensemble_seeds']
    payloads = [joblib.load(OUT / record['payload_path']) for record in refs]
    for payload in payloads:
        assert payload['actual_target'] == country['name'] and payload['actual_outcome'] == outcome
        assert payload['history_start'] == start and payload['origin'] == 2018
        assert payload['base'] == {'channels': 16, 'weight_decay': .001, 'epochs': 50}
        audit = payload['audit']
        assert audit['device'] == data.tcn_device.iloc[0]
        assert audit['maximum_label_year'] == 2018 and audit['last_target_label_year'] == 2018
        assert audit['target_in_base_fit'] is False and country['name'] not in audit['countries']
        assert audit['fingerprint_before'] == audit['fingerprint_after']
        assert audit['parameter_count'] < 10000
        expected_windows = 2018-start-11
        assert audit['target_windows'] == 22*expected_windows
        assert audit['training_windows'] == 6*22*expected_windows
        for name in ['target_meta', 'training_meta']:
            meta = payload[name]
            assert meta.input_start.min() == start and meta.label_end.max() == 2018
            assert meta.input_end.max() == 2013
            assert set(meta.window_origin) == set(range(start+7, 2014))
        assert payload['target_meta'].country.eq(country['name']).all()
    first = payloads[0]
    for payload in payloads[1:]:
        pd.testing.assert_frame_equal(payload['current_meta'], first['current_meta'])
        pd.testing.assert_frame_equal(payload['target_meta'], first['target_meta'])
        np.testing.assert_array_equal(payload['target_y'], first['target_y'])
        np.testing.assert_array_equal(payload['levels'], first['levels'])
    failed = [payload for payload in payloads if payload['audit']['status'] != 'ok']
    assert all(payload['audit']['reason'] for payload in failed)
    source = PANEL.loc[PANEL.location_name.eq(country['name']) & PANEL.outcome.eq(outcome)
                       & PANEL.year.between(start, 2018)].set_index(['sex', 'age', 'year']).rate
    x, levels = [], []
    for sex in CONFIG['sexes']:
        for ai, age in enumerate(CONFIG['ages']):
            log_values = np.log([source.loc[(sex, age, year)] for year in range(2011, 2019)])
            x.append(np.r_[log_values-log_values[-1], log_values[-1], float(sex == 'Male'), np.eye(11)[ai]])
            levels.append(log_values[-1])
    np.testing.assert_allclose(first['levels'], levels, atol=1e-13, rtol=1e-13)
    expected_y = []
    for row in first['target_meta'].itertuples():
        level = np.log(source.loc[(row.sex, row.age, row.window_origin)])
        expected_y.append([np.log(source.loc[(row.sex, row.age, row.window_origin+h)])-level for h in range(1, 6)])
    np.testing.assert_allclose(first['target_y'], expected_y, atol=1e-13, rtol=1e-13)
    checkpoint_replays = 0
    if first['audit']['status'] == 'ok':
        fitted = load_checkpoint(OUT / refs[0]['checkpoint_path'])
        np.testing.assert_allclose(predict_changes(fitted, np.asarray(x)), first['current_changes'], atol=2e-7, rtol=2e-6)
        checkpoint_replays = 1
    else:
        np.testing.assert_array_equal(first['current_changes'], np.zeros((22, 5)))
    changes = np.mean([payload['current_changes'] for payload in payloads], axis=0)
    corrections = [record for record in adaptation if record['trial_id'] == ident]
    assert len(corrections) == 2
    by_sex = {record['sex']: record for record in corrections}
    expected = []
    for index, row in first['current_meta'].iterrows():
        correction = by_sex[row.sex]
        assert correction['penalty'] == 1 and correction['last_target_label_year'] == 2018
        assert correction['target_windows'] == 11*(2018-start-11)
        for h in range(1, 6):
            unadapted = levels[index] if failed else levels[index]+changes[index, h-1]
            adapted = (levels[index] if correction['status'] != 'ok' else
                       unadapted+correction['b0']+correction['b1']*h/5)
            for family, value in [('tcn_unadapted', unadapted),
                                  ('tcn_adapted', adapted)]:
                expected.append(dict(family=family, sex=row.sex, age=row.age, horizon=h,
                                     log_prediction=value, prediction=np.exp(value)))
    same(data.loc[data.family.str.startswith('tcn')], pd.DataFrame(expected),
         ['family', 'sex', 'age', 'horizon'], ['log_prediction', 'prediction'])
    return dict(case=ident, passed=True, neural_seed_payloads=5, checkpoint_replays=checkpoint_replays,
                failed_seed_count=len(failed),
                target_training_windows=22*(2018-start-11), donor_training_windows=132*(2018-start-11),
                actual_outcome_matched=True, history_start=start)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='results/mortality_history_v1')
    args = parser.parse_args()
    OUT = ROOT / args.run
    started = time.perf_counter()
    manifest = json.loads((OUT / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    for name, digest in manifest['output_sha256'].items():
        assert sha(OUT / name) == digest, name
    for name, digest in manifest['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for run, digest in manifest['identity']['prior_manifest_sha256'].items():
        assert sha(ROOT / 'results' / run / 'run_manifest.json') == digest, run
    points = read(OUT / 'predictions.csv')
    assert len(points) == 7920 and not points.duplicated(KEY).any()
    assert set(points.tcn_device) == {manifest['identity']['device']}
    assert set(points.history_role) == {'both_donor_training_and_target_adaptation'}
    seeds = json.loads((OUT / 'seed_audit.json').read_text())
    adaptation = json.loads((OUT / 'adaptation_audit.json').read_text())
    tasks = [(country, outcome, start) for country in CONFIG['countries'] if country['gcc']
             for outcome in ['deaths', 'ylls'] for start in [1980, 1990]]
    arms = [audit_arm(task, points, seeds, adaptation) for task in tasks]
    expected_scores = []
    for country, outcome, start in tasks:
        part = points.loc[points.target.eq(country['name']) & points.outcome.eq(outcome) & points.history_start.eq(start)]
        expected_scores.append(score_points(part, truth_for(country['name'], outcome)))
    scored = pd.concat(expected_scores, ignore_index=True)
    same(read(OUT / 'point_scores.csv'), scored, KEY, ['observed_rate', 'absolute_log_error', 'absolute_rate_error'])
    summary = []
    keys = ['target', 'outcome', 'history_start', 'sex', 'family', 'horizon']
    for name, part in [('45+', scored), ('80+', scored.loc[scored.age.isin(CONFIG['ages'][-4:])])]:
        frame = part.groupby(keys, as_index=False).agg(mean_absolute_log_error=('absolute_log_error', 'mean'),
                   mean_absolute_rate_error=('absolute_rate_error', 'mean'), age_cells=('age', 'size'))
        frame['age_group'] = name
        summary.append(frame)
    summary = pd.concat(summary, ignore_index=True)
    same(read(OUT / 'summary.csv'), summary, keys+['age_group'], ['mean_absolute_log_error', 'mean_absolute_rate_error', 'age_cells'])
    compare_keys = ['target', 'outcome', 'sex', 'family', 'horizon', 'age_group']
    paired = summary.loc[summary.history_start.eq(1980)].merge(summary.loc[summary.history_start.eq(1990)],
             on=compare_keys, validate='one_to_one', suffixes=('_1980', '_1990'))
    paired['absolute_log_error_change_1980_minus_1990'] = paired.mean_absolute_log_error_1980-paired.mean_absolute_log_error_1990
    paired['relative_improvement_percent'] = -100*paired.absolute_log_error_change_1980_minus_1990/paired.mean_absolute_log_error_1990
    same(read(OUT / 'history_comparisons.csv'), paired, compare_keys,
         ['absolute_log_error_change_1980_minus_1990', 'relative_improvement_percent'])
    issued = json.loads((OUT / 'issued_commit.json').read_text())
    scoring = json.loads((OUT / 'scoring_complete.json').read_text())
    assert datetime.fromisoformat(issued['committed_utc']) <= datetime.fromisoformat(scoring['committed_utc'])
    assert issued['all_24_history_arms_issued_before_scoring'] is True
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  audit_code_sha256=sha(__file__), helper_sha256=sha(HERE / 'audit_rates.py'),
                  run_manifest_sha256=sha(OUT / 'run_manifest.json'), arms=arms,
                  point_scores=len(scored), summary_rows=len(summary), history_contrasts=len(paired),
                  all_source_output_hashes_verified=True, elapsed_seconds=time.perf_counter()-started)
    (HERE / 'audit_history.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, arms=len(arms), points=len(points),
                          checkpoint_replays=sum(arm['checkpoint_replays'] for arm in arms))))


if __name__ == '__main__':
    main()
