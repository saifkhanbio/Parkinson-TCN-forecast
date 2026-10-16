"""Preserve and verify the new regional ASR export without changing forecasts."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from audit_source_compatibility import ROOT, WIDE, MEASURES, read, rel, sha

INPUT = ROOT / 'More data/IHME-GBD_2023_DATA-5dd493d4-1'
CSV = INPUT / 'IHME-GBD_2023_DATA-5dd493d4-1.csv'
PACKAGE = ROOT / 'supporting_data/2026-09-30_asr_verification'
RESULTS = ROOT / 'results/source_asr_verification_v1'
REGIONAL = {140, 144, 145, 150, 151, 152, 156}
SEXES = {1: 'male', 2: 'female', 3: 'combined'}
KEY = ['location_id', 'measure_id', 'sex_id', 'year']


def verify_record():
    record = json.loads((RESULTS / 'validation.json').read_text())
    for group in ['input_sha256', 'protected_sha256', 'output_sha256']:
        for path, digest in record[group].items():
            assert sha(ROOT / path) == digest, path
    assert sha(Path(__file__)) == record['script_sha256']
    print('New ASR source, audit outputs and protected study files retain their recorded identities.')


def main():
    if PACKAGE.exists() or RESULTS.exists():
        raise SystemExit('Refusing to replace an existing ASR verification package.')
    previous = json.loads((ROOT / 'results/source_hierarchy_verification_v1/validation.json').read_text())
    protected = dict(previous['protected_sha256'])
    protected.update(previous['input_sha256'])
    protected.update(previous['output_sha256'])
    for directory in ['source_inventory_recheck_v1']:
        v = json.loads((ROOT / 'data/processed' / directory / 'validation.json').read_text())
        protected.update(v['output_sha256'])
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    inputs = [CSV, INPUT / 'citation.txt', WIDE,
              ROOT / 'results/release_sensitivity_v1/matched_points.csv',
              ROOT / 'data/processed/source_inventory_recheck_v1/saudi_asr_existing_54.csv']
    source_hashes = {rel(p): sha(p) for p in inputs}
    n = read(CSV)
    checks = []

    def check(name, condition, detail):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
        assert condition, (name, detail)

    check('native_schema', len(n.columns) == 18 and set(KEY + ['val', 'lower', 'upper', 'age_id', 'metric_id', 'cause_id']).issubset(n.columns), 'Native ID/name schema and marginal bounds')
    check('unique_keys', not n.duplicated(KEY).any(), 'One row per location/outcome/sex/year for this single age/metric/cause')
    check('correct_estimand', set(n.age_id) == {27} and set(n.age_name) == {'Age-standardized'} and set(n.metric_id) == {3} and set(n.metric_name) == {'Rate'} and set(n.cause_id) == {544} and set(n.population_group_id) == {1}, 'Age-standardized Rate, Parkinson disease, All Population')
    expected = set()
    for measure in MEASURES:
        years = range(1980 if measure in [1, 4] else 1990, 2024)
        expected.update(itertools.product(REGIONAL, [measure], SEXES, years))
    check('complete_regional_export', len(n) == 4704 and set(n[KEY].itertuples(index=False, name=None)) == expected, '7 locations x 3 sexes; 44 fatal years, 34 years for other measures')
    check('finite_ordered_bounds', np.isfinite(n[['val', 'lower', 'upper']]).all().all() and (n.lower > 0).all() and (n.lower <= n.val).all() and (n.val <= n.upper).all(), 'All 4704 point/lower/upper triplets')
    n['source_row'] = np.arange(2, len(n) + 2)
    n['source_file'] = rel(CSV)
    n['measure'] = n.measure_id.map(MEASURES)
    n['rate_unit'] = 'per 100000 population'
    n['source_release_label'] = 'GBD 2023'
    n['uncertainty_type'] = 'source_marginal_uncertainty_not_forecast_prediction_interval'
    w = read(WIDE)
    w = w[w.location_id.isin(REGIONAL)]
    pieces = []
    for mid, measure in MEASURES.items():
        for sid, suffix in SEXES.items():
            column = f'{measure}_rate_age_std_{suffix}'
            frame = w[['location_id', 'year', column]].rename(columns={column: 'workbook_value'}).copy()
            frame['measure_id'], frame['sex_id'] = mid, sid
            frame['workbook_source_column'] = column
            pieces.append(frame)
    points = pd.concat(pieces, ignore_index=True)
    check('workbook_comparison_unique', not points.duplicated(KEY).any() and len(points) == 4284, '7 countries x 34 years x 6 measures x 3 sexes')
    comparison = n.merge(points, on=KEY, how='left', validate='one_to_one', indicator=True)
    overlap = comparison[comparison._merge == 'both'].drop(columns='_merge').copy()
    additional = comparison[comparison._merge == 'left_only'].drop(columns='_merge').copy()
    overlap['native_minus_workbook'] = overlap.val - overlap.workbook_value
    overlap['matches_workbook_rounding'] = overlap.native_minus_workbook.abs() <= .005000001
    check('all_overlapping_points_match', len(overlap) == 4284 and overlap.matches_workbook_rounding.all(), 'Tolerance 0.005000001 per 100000 for workbook display precision of 0.01')
    check('additional_rows_are_earlier_fatal_history', len(additional) == 420 and additional.year.between(1980, 1989).all() and additional.measure_id.isin([1, 4]).all(), 'Earlier years are absent from workbook, not mismatched')
    checkpoint = overlap[(overlap.location_id == 152) & overlap.year.isin([1990, 2019, 2023])].copy()
    check('all_saudi_checkpoints_match', len(checkpoint) == 54 and checkpoint.matches_workbook_rounding.all(), 'All requested Saudi points and source bounds now available')
    old_checkpoint = read(inputs[-1]).rename(columns={'value': 'previously_extracted_value'})
    cp = checkpoint.merge(old_checkpoint[['year', 'measure', 'sex', 'previously_extracted_value']], left_on=['year', 'measure', 'sex_name'], right_on=['year', 'measure', 'sex'], validate='one_to_one')
    check('prior_local_extraction_reproduced', len(cp) == 54 and np.allclose(cp.workbook_value, cp.previously_extracted_value, rtol=0, atol=1e-12), 'Previously extracted workbook checkpoints agree')
    components = n.pivot(index=['location_id', 'sex_id', 'year'], columns='measure_id', values='val').dropna(subset=[2, 3, 4])
    residual = components[2] - components[3] - components[4]
    check('daly_point_identity', len(components) == 714 and np.allclose(components[2], components[3] + components[4], rtol=1e-7, atol=1e-8), 'Point DALY ASR approximately equals YLD plus YLL ASR; marginal bounds are not added')
    old = read(ROOT / 'results/release_sensitivity_v1/matched_points.csv')
    old_asr = old[old.metric == 'Age-standardized rate'].copy()
    old_asr['measure'] = old_asr.measure.str.lower()
    contrast = old_asr.merge(n[n.sex_id == 3][['location_name', 'measure', 'year', 'val', 'lower', 'upper', 'source_row']], left_on=['location', 'measure', 'year'], right_on=['location_name', 'measure', 'year'], validate='one_to_one')
    check('published_release_asr_points_linked', len(contrast) == 36 and (abs(contrast.val - contrast.new_value) <= .005000001).all(), '36 GCC published old-release ASR points linked to native new-release points')
    contrast['native_minus_published_old'] = contrast.val - contrast.old_value
    contrast['native_relative_difference_pct'] = 100 * (contrast.val / contrast.old_value - 1)
    contrast['interpretation'] = 'same_year_release_source_contrast; common_weights_and_definitions_not_verified; not_forecast_validation'
    saudi2019 = contrast[(contrast.location == 'Saudi Arabia') & (contrast.measure == 'prevalence') & (contrast.year == 2019)].iloc[0]
    citation_identical = (INPUT / 'citation.txt').read_bytes() == (ROOT / 'More data/IHME-GBD_2023_DATA-e22a15d9-1/citation.txt').read_bytes()
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    assert all(sha(ROOT / p) == digest for p, digest in source_hashes.items())
    (PACKAGE / 'raw').mkdir(parents=True)
    (PACKAGE / 'processed').mkdir()
    RESULTS.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    manifest = []
    for original in [CSV, INPUT / 'citation.txt']:
        target = PACKAGE / 'raw' / original.name
        shutil.copyfile(original, target)
        assert sha(original) == sha(target)
        manifest.append({'original_path': rel(original), 'preserved_path': rel(target), 'bytes': target.stat().st_size,
                         'sha256': sha(target), 'source_url': 'https://vizhub.healthdata.org/gbd-results/',
                         'provided_by': 'user after authenticated portal download instructions', 'inspected_utc': now,
                         'source_release_label': 'GBD 2023', 'exact_internal_revision': 'not supplied',
                         'citation_year_verbatim': 2024, 'download_timestamp_independently_verified': False})
    with (PACKAGE / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    n.sort_values(KEY).to_csv(PACKAGE / 'processed/regional_asr_native_1980_2023.csv', index=False, mode='x')
    checkpoint.sort_values(KEY).to_csv(PACKAGE / 'processed/saudi_asr_checkpoint_54.csv', index=False, mode='x')
    overlap.sort_values(KEY).to_csv(RESULTS / 'workbook_comparison_4284.csv', index=False, mode='x')
    additional.sort_values(KEY).to_csv(RESULTS / 'additional_fatal_history_420.csv', index=False, mode='x')
    contrast.to_csv(RESULTS / 'published_release_asr_comparison_36.csv', index=False, mode='x')
    n.groupby(['location_id', 'location_name', 'measure_id', 'measure_name', 'sex_id', 'sex_name']).year.agg(['min', 'max', 'count']).reset_index().to_csv(RESULTS / 'coverage.csv', index=False, mode='x')
    pd.DataFrame(checks).to_csv(RESULTS / 'checks.csv', index=False, mode='x')
    citations = []
    for path in sorted(PACKAGE.rglob('*')):
        if path.is_file():
            citations.append({'file': rel(path), 'citation': 'Global Burden of Disease Collaborative Network. Global Burden of Disease Study 2023 Results. IHME, 2024 (year as supplied in export citation).',
                              'source_url': 'https://vizhub.healthdata.org/gbd-results/', 'sha256': sha(path),
                              'transformation': 'Exact user-supplied original copy' if path.parent.name == 'raw' else 'Source extraction or acquisition record; no model fitting'})
    pd.DataFrame(citations).to_csv(PACKAGE / 'file_citations.csv', index=False, mode='x')
    summary = {'created_utc': now, 'native_rows': len(n), 'matched_workbook_values': len(overlap),
               'matched_saudi_checkpoints': len(checkpoint), 'additional_fatal_history_rows': len(additional),
               'maximum_absolute_rounding_difference': float(overlap.native_minus_workbook.abs().max()),
               'daly_identity_maximum_absolute_difference': float(abs(residual).max()),
               'daly_identity_maximum_relative_difference': float((abs(residual) / components[2]).max()),
               'saudi_2019_both_sex_prevalence_asr': float(saudi2019.val),
               'saudi_2019_both_sex_prevalence_asr_lower': float(saudi2019.lower),
               'saudi_2019_both_sex_prevalence_asr_upper': float(saudi2019.upper),
               'published_older_2019_prevalence_asr': float(saudi2019.old_value),
               'same_year_native_versus_published_difference_pct': float(saudi2019.native_relative_difference_pct),
               'both_native_export_citations_identical': citation_identical,
               'checks_passed': len(checks), 'checks_failed': 0, 'models_fitted': 0,
               'regional_workbook_asr_point_values_independently_confirmed': True,
               'regional_asr_source_bounds_available': True,
               'remaining_196_workbook_locations_independently_confirmed': False,
               'cross_release_standard_weights_harmonized': False, 'joint_source_draws_available': False,
               'population_source_differences_resolved': False, 'forecast_intervals_changed': False,
               'raw_and_locked_files_unchanged': True}
    with (RESULTS / 'summary.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    outputs = {rel(p): sha(p) for p in sorted(list(PACKAGE.rglob('*')) + list(RESULTS.glob('*'))) if p.is_file()}
    with (RESULTS / 'validation.json').open('x') as stream:
        json.dump({'input_sha256': source_hashes, 'protected_sha256': protected, 'output_sha256': outputs,
                   'script_sha256': sha(Path(__file__)), 'scope': 'regional ASR export verification; no source covariance or forecast calibration guarantee'}, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    if not __debug__:
        raise SystemExit('Run without -O; assertions are required.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    if parser.parse_args().verify_only:
        verify_record()
    else:
        main()
