"""Inventory existing inputs and extract checkpoints without new downloads."""
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_source_compatibility import (
    ROOT, NATIVE, WORKBOOK, WIDE, MEASURES, read, rel, rows_frame, sha, workbook,
)

OUT = ROOT / 'data/processed/source_inventory_recheck_v1'
YEARS = [1990, 2019, 2023]


def main():
    if not __debug__:
        raise SystemExit('Run without -O; validation assertions are required.')
    if OUT.exists():
        raise SystemExit('Refusing to overwrite an existing checkpoint extraction.')
    inputs = [NATIVE, WORKBOOK, WIDE, NATIVE.parent / 'citation.txt']
    hashes = {rel(p): sha(p) for p in inputs}
    previous = json.loads((ROOT / 'results/source_compatibility_v1/validation.json').read_text())
    assert all(previous['source_sha256'][name] == value for name, value in hashes.items())
    native, wide = read(NATIVE), read(WIDE)
    regional = native[native.year.between(1990, 2023)]
    columns = ['location_id', 'sex_id', 'age_id', 'measure_id', 'metric_id', 'year']
    expected = set(itertools.product(
        [152, 140, 145, 150, 151, 156, 144], [1, 2],
        list(range(14, 21)) + [30, 31, 32, 235], range(1, 7), [1, 3], range(1990, 2024)))
    assert len(regional) == len(expected) == 62832
    assert set(regional[columns].itertuples(index=False, name=None)) == expected
    assert len(native) == 68992 and len(native) - len(regional) == 6160
    frames = {name: rows_frame(rows) for name, rows in workbook(WORKBOOK)
              if name in ['Saudi_Arabia', 'Raw_Data']}
    saudi = frames['Saudi_Arabia']
    saudi['source_row'] = np.arange(2, len(saudi) + 2)
    selected = saudi[saudi.year.astype(int).isin(YEARS)]
    assert len(selected) == 18
    long = frames['Raw_Data']
    records = []
    for _, row in selected.iterrows():
        year, measure = int(row.year), row.measure
        reference = long[(long.location_id.astype(int) == 152) &
                         (long.year.astype(int) == year) & (long.measure == measure)]
        assert len(reference) == 1
        reference = reference.iloc[0]
        w = wide[(wide.location_id == 152) & (wide.year == year)]
        assert len(w) == 1
        for suffix, sex in [('male', 'Male'), ('female', 'Female'), ('combined', 'Both')]:
            for prefix, age, metric in [('rate_age_std', 'Age-standardized', 'Rate'),
                                        ('count', 'All ages', 'Number'),
                                        ('rate', 'All ages', 'Rate')]:
                point_column = prefix + '_' + suffix
                point = float(row[point_column])
                assert point == float(reference[point_column])
                assert np.isclose(point, w.iloc[0][measure + '_' + point_column], rtol=1e-12)
                bound_columns = [prefix + '_' + bound + '_' + suffix for bound in ['lower', 'upper']]
                has_bounds = all(c in saudi.columns for c in bound_columns)
                lower, upper = (float(row[c]) for c in bound_columns) if has_bounds else (None, None)
                if has_bounds:
                    assert lower <= point <= upper
                    assert all(float(row[c]) == float(reference[c]) for c in bound_columns)
                records.append({
                    'location_id': 152, 'location_name': row.location_name,
                    'cause_id': int(row.cause_id), 'cause_name': row.cause_name,
                    'year': year, 'measure': measure, 'sex': sex, 'age': age,
                    'metric': metric, 'value': point, 'lower': lower, 'upper': upper,
                    'bounds_available': has_bounds, 'source_file': rel(WORKBOOK),
                    'source_sheet': 'Saudi_Arabia', 'source_row': int(row.source_row),
                    'source_point_column': point_column,
                    'source_lower_column': bound_columns[0] if has_bounds else '',
                    'source_upper_column': bound_columns[1] if has_bounds else '',
                    'source_status': 'existing_prepared_workbook_not_independent_native_reexport',
                })
    checkpoints = pd.DataFrame(records)
    assert len(checkpoints) == 162 and checkpoints.bounds_available.sum() == 108
    assert not checkpoints.duplicated(['year', 'measure', 'sex', 'age', 'metric']).any()
    assert set(checkpoints.measure) == set(MEASURES.values())
    asr_columns = [f'{m}_rate_age_std_{s}' for m in ['prevalence', 'incidence'] for s in ['male', 'female']]
    assert len(wide) == 6902 and wide[asr_columns].notna().all().all()
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items())
    OUT.mkdir(parents=True)
    for name, data in [('saudi_checkpoints_existing_162.csv', checkpoints),
                       ('saudi_asr_existing_54.csv', checkpoints[checkpoints.age == 'Age-standardized']),
                       ('saudi_all_ages_existing_108.csv', checkpoints[checkpoints.age == 'All ages'])]:
        data.to_csv(OUT / name, index=False, mode='x')
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input_sha256': hashes, 'extraction_script_sha256': sha(Path(__file__)),
        'identical_to_previous_audit_inputs': True,
        'native_rows': len(native), 'regional_1990_2023_complete_rows': len(regional),
        'additional_fatal_1980_1989_rows': len(native) - len(regional),
        'existing_global_prevalence_incidence_male_female_asr_values': len(wide) * len(asr_columns),
        'checkpoint_points_existing': len(checkpoints), 'checkpoint_bounds_existing': 108,
        'checkpoint_asr_bounds_absent': 54,
        'source_values_preserved': True, 'original_export_provenance_newly_verified': False,
        'models_fitted': 0, 'downloads_made': 0,
        'output_sha256': {rel(p): sha(p) for p in sorted(OUT.glob('*.csv'))},
    }
    with (OUT / 'validation.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
