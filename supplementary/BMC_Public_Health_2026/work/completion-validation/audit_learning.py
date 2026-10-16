"""Independent learning-budget feature, checkpoint, adaptation and score replay."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from gbd_park.tcn import load_checkpoint, predict_changes as neural_predict, state_fingerprint
from gbd_park.pooled import predict_changes as nonneural_predict
from gbd_park.adaptation import fit_adaptation, regularized_location


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def verify(root, hashes):
    for name, digest in hashes.items():
        assert sha(root / name) == digest, name


def arrays(source, config, names, target_years):
    """Construct raw examples independently, without new implementation helpers."""
    xs, ys, rows = [], [], []
    for country in names:
        start = 2019-target_years if country == 'Saudi Arabia' else 1990
        for sex in config['sexes']:
            for ai, age in enumerate(config['ages']):
                for origin in range(start+7, 2014):
                    values = np.log([source.loc[(country, sex, age, year)] for year in range(origin-7, origin+6)])
                    xs.append(np.r_[values[:8]-values[7], values[7], float(sex == 'Male'), np.eye(11)[ai]])
                    ys.append(values[8:]-values[7])
                    rows.append(dict(country=country, sex=sex, age=age, window_origin=origin,
                                     input_start=origin-7, input_end=origin, label_start=origin+1, label_end=origin+5))
    meta = pd.DataFrame(rows)
    # Each country's 22 strata have equal mass, irrespective of window count.
    counts = meta.groupby(['country', 'sex', 'age']).age.transform('size').to_numpy()
    weights = len(meta) / (len(names)*22*counts)
    return np.asarray(xs), np.asarray(ys), meta, weights


def current_arrays(source, config):
    xs, levels = [], []
    for sex in config['sexes']:
        for ai, age in enumerate(config['ages']):
            values = np.log([source.loc[('Saudi Arabia', sex, age, year)] for year in range(2011, 2019)])
            xs.append(np.r_[values-values[-1], values[-1], float(sex == 'Male'), np.eye(11)[ai]])
            levels.append(values[-1])
    return np.asarray(xs), np.asarray(levels)


def main():
    out = ROOT / 'results/learning_curves_v1'
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete'
    verify(out, manifest['output_sha256'])
    verify(ROOT, manifest['code_sha256'])
    config = json.loads((ROOT / 'study_design/locked_v1/design.json').read_text())
    source_path = ROOT / 'data/processed/design_v1/regional_outcomes.csv'
    assert sha(source_path) == manifest['identity']['source_sha256']
    # Match the preserved training parser; score ledgers use round-trip parsing.
    panel = pd.read_csv(source_path)
    source = {o: panel.loc[panel.outcome.eq(o)].set_index(['location_name', 'sex', 'age', 'year']).rate
              for o in ['prevalence', 'incidence']}
    points = read(out / 'predictions.csv')
    keys = ['outcome', 'history_years', 'family', 'sex', 'age', 'horizon']
    families = config['models']['local_order'] + config['models']['nonneural_order'] + ['tcn_adapted', 'tcn_unadapted', 'tcn_intercept']
    expected = set(itertools.product(source, [15, 20, 29], families, config['sexes'], config['ages'], range(1, 6)))
    assert len(points) == 9240 and set(map(tuple, points[keys].to_numpy())) == expected
    assert points.origin.eq(2018).all() and points.target.eq('Saudi Arabia').all()
    assert points.forecast_year.eq(2018+points.horizon).all()
    assert 'observed_rate' not in points
    payloads, neural, reconstructed = [], {}, []
    replayed_checkpoints = 0
    replayed_neural = 0
    markers = sorted((out / 'jobs').glob('*/complete.json'))
    assert len(markers) == 38
    for marker in markers:
        record = json.loads(marker.read_text())
        verify(marker.parent, record['artifact_sha256'])
        result = joblib.load(marker.parent / 'result.joblib')
        job = record['job']
        assert result['job'] == job
        outcome = job['outcome']
        if job['kind'] == 'local':
            frame = pd.DataFrame(result['forecasts']).assign(outcome=outcome, history_years=job['history_years'])
            reconstructed.append(frame[keys+['log_prediction']])
            continue
        audit = result['audit']
        years = job.get('history_years', 15)
        names = [c['name'] for c in config['countries'] if job['mode'] == 'pooled' or c['name'] != 'Saudi Arabia']
        x, y, meta, weights = arrays(source[outcome], config, names, years)
        pd.testing.assert_frame_equal(result['training_meta'], meta.assign(sample_weight=weights))
        assert audit['source_countries'] == names and audit['maximum_label_year'] == 2018
        assert audit['target_in_base_fit'] == (job['mode'] == 'pooled')
        predictor = neural_predict if job['kind'] == 'tcn' else nonneural_predict
        fitted = None
        if audit['status'] == 'ok':
            fitted = load_checkpoint(marker.parent / 'checkpoint.joblib') if job['kind'] == 'tcn' else joblib.load(marker.parent / 'checkpoint.joblib')
            assert audit['fingerprint_before'] == audit['fingerprint_after']
            if job['kind'] == 'tcn':
                assert state_fingerprint(fitted) == audit['fingerprint_before']
                replayed_neural += 1
            # Artifact hashes bind the serialized non-neural model; feature
            # statistics and all current/historical predictions are replayed below.
            replayed_checkpoints += 1
            mean = np.average(x, axis=0, weights=weights)
            var = np.average((x-mean)**2, axis=0, weights=weights)
            np.testing.assert_allclose(fitted['scaler'].mean_, mean, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(fitted['scaler'].var_, var, atol=1e-12, rtol=1e-12)
        else:
            assert audit['status'] == 'fallback' and audit['reason']
        current, levels = current_arrays(source[outcome], config)
        for budget, payload in result['payloads'].items():
            tx, ty, tm, _ = arrays(source[outcome], config, ['Saudi Arabia'], budget)
            pd.testing.assert_frame_equal(payload['target_meta'], tm)
            np.testing.assert_allclose(payload['target_y'], ty, atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(payload['levels'], levels, atol=1e-13, rtol=1e-13)
            assert len(tm) == 22*(budget-12) and tm.input_start.min() == 2019-budget
            if payload['audit']['status'] == 'ok':
                np.testing.assert_allclose(predictor(fitted, current), payload['current_changes'], atol=1e-12, rtol=1e-12)
                if job['mode'] == 'donor':
                    np.testing.assert_allclose(predictor(fitted, tx), payload['target_changes'], atol=1e-12, rtol=1e-12)
            if job['kind'] == 'tcn':
                neural.setdefault((outcome, budget), []).append(payload)
            else:
                payloads.append((outcome, budget, job['mode']+'_'+job['kind'], [payload]))
    assert len(neural) == 6
    for (outcome, budget), entries in neural.items():
        assert sorted(p['seed'] for p in entries) == config['models']['tcn']['ensemble_seeds']
        payloads.append((outcome, budget, 'tcn', entries))
    for outcome, budget, base, entries in payloads:
        first = entries[0]
        failed = any(p['audit']['status'] != 'ok' for p in entries)
        changes = np.mean([p['current_changes'] for p in entries], axis=0)
        history = np.mean([p['target_changes'] for p in entries], axis=0)
        modes = ['none'] if base.startswith('pooled') else ['none', 'two_parameter'] + (['intercept'] if base == 'tcn' else [])
        for sex in config['sexes']:
            take = first['current_meta'].sex.eq(sex).to_numpy()
            eligible = first['target_meta'].sex.eq(sex).to_numpy()
            for mode in modes:
                family = base if base.startswith('pooled') else base + {'none':'_unadapted', 'two_parameter':'_adapted', 'intercept':'_intercept'}[mode]
                adjusted = np.zeros_like(changes[take]) if failed else changes[take].copy()
                if not failed and mode == 'two_parameter':
                    c = fit_adaptation(first['target_y'][eligible], history[eligible], 1)
                    adjusted += c['b0']+c['b1']*np.arange(1, 6)/5
                elif not failed and mode == 'intercept':
                    adjusted += regularized_location((first['target_y'][eligible]-history[eligible]).ravel(), 1)
                logs = first['levels'][take, None]+adjusted
                for ai, age in enumerate(config['ages']):
                    for hi in range(5):
                        reconstructed.append(pd.DataFrame([dict(outcome=outcome, history_years=budget, family=family,
                            sex=sex, age=age, horizon=hi+1, log_prediction=logs[ai, hi])]))
    rebuilt = pd.concat(reconstructed, ignore_index=True).sort_values(keys).reset_index(drop=True)
    saved = points.sort_values(keys).reset_index(drop=True)
    assert rebuilt[keys].equals(saved[keys])
    np.testing.assert_allclose(rebuilt.log_prediction, saved.log_prediction, atol=1e-12, rtol=1e-12)
    truth = panel.loc[panel.location_name.eq('Saudi Arabia'), ['outcome','sex','age','year','rate']].rename(columns={'year':'forecast_year', 'rate':'observed_rate'})
    scored = points.merge(truth, on=['outcome','sex','age','forecast_year'], validate='many_to_one')
    scored['absolute_log_error'] = abs(np.log(scored.prediction)-np.log(scored.observed_rate))
    scored['absolute_rate_error'] = abs(scored.prediction-scored.observed_rate)
    saved_scores = read(out / 'point_scores.csv').sort_values(keys)
    for column in ['observed_rate', 'absolute_log_error', 'absolute_rate_error']:
        np.testing.assert_allclose(scored.sort_values(keys)[column], saved_scores[column], atol=1e-11, rtol=1e-11)
    summaries = []
    sk = ['target', 'outcome', 'history_years', 'sex', 'family', 'horizon', 'age_group']
    for band, ages in [('45+', config['ages']), ('80+', config['ages'][-4:])]:
        part = scored.loc[scored.age.isin(ages)].assign(age_group=band)
        summaries.append(part.groupby(sk, as_index=False).agg(mean_absolute_log_error=('absolute_log_error','mean'),
            mean_absolute_rate_error=('absolute_rate_error','mean'), age_cells=('age','size')))
    rebuilt = pd.concat(summaries).sort_values(sk)
    saved = read(out / 'summary.csv').sort_values(sk)
    for column in ['mean_absolute_log_error', 'mean_absolute_rate_error', 'age_cells']:
        np.testing.assert_allclose(rebuilt[column], saved[column], atol=1e-11, rtol=1e-11)
    issued = json.loads((out / 'issued_commit.json').read_text())
    scoring = json.loads((out / 'scoring_complete.json').read_text())
    assert issued['all_six_arms_issued_before_scoring']
    assert datetime.fromisoformat(issued['committed_utc']) <= datetime.fromisoformat(scoring['committed_utc'])
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        audit_code_sha256=sha(__file__), run_manifest_sha256=sha(out / 'run_manifest.json'),
        prediction_rows=len(points), checkpoint_jobs=replayed_checkpoints, neural_seed_checkpoints=replayed_neural,
        independent_feature_and_weight_reconstruction=True, independent_adaptation_and_score_reconstruction=True,
        local_forecasts_verified_against_hashed_fit_payloads=True, no_refitting=True)
    (Path(__file__).parent / 'audit_learning.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
