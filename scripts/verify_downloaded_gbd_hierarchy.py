"""Preserve user-provided hierarchy files and verify the existing donor roster."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd

from audit_source_compatibility import ROOT, NATIVE, WIDE, read, rel, rows_frame, sha, workbook

PACKAGE = ROOT / 'supporting_data/2026-09-30_hierarchy_verification'
RESULTS = ROOT / 'results/source_hierarchy_verification_v1'
FILES = ['IHME_GBD_2023_HIERARCHIES_Y2025M10D23.XLSX',
         'IHME_GBD_2023_HIERARCHIES_INFO_SHEET_Y2025M10D23.pdf']
SOURCE_URL = 'https://ghdx.healthdata.org/record/gbd-2023-cause-rei-and-location-hierarchies'
GCC = {140, 145, 150, 151, 152, 156}


def main():
    if not __debug__:
        raise SystemExit('Run without -O; assertions are required.')
    if PACKAGE.exists() or RESULTS.exists():
        raise SystemExit('Refusing to overwrite existing hierarchy verification.')
    previous = json.loads((ROOT / 'results/source_compatibility_v1/validation.json').read_text())
    protected = dict(previous['source_sha256'])
    protected.update(previous['protected_sha256'])
    protected.update(previous['output_sha256'])
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    inputs = [ROOT / 'More data' / name for name in FILES]
    hashes = {rel(p): sha(p) for p in inputs}
    frames = {name: rows_frame(rows) for name, rows in workbook(inputs[0])}
    assert set(frames) == {'All Location Hierarchies', 'Cause Hierarchy', 'REI Hierarchy'}
    locations, causes, rei = [frames[n] for n in ['All Location Hierarchies', 'Cause Hierarchy', 'REI Hierarchy']]
    for column in ['Location Set Version ID', 'Location ID', 'Parent ID', 'Level', 'Sort Order']:
        locations[column] = locations[column].astype(int)
    for column in ['Cause ID', 'Parent ID', 'Level', 'Sort Order']:
        causes[column] = causes[column].astype(int)
    assert (len(locations), len(causes), len(rei)) == (1510, 381, 227)
    assert not locations.duplicated(['Location Set Version ID', 'Location ID']).any()
    assert not causes['Cause ID'].duplicated().any()
    main_set = locations[locations['Location Set Version ID'] == 1511]
    national = main_set[main_set.Level == 3]
    assert len(national) == 204
    wb_region = locations[(locations['Location Set Version ID'] == 1300) & (locations.Level == 2)]
    wb_income = locations[(locations['Location Set Version ID'] == 1305) & (locations.Level == 2)]
    registry = read(WIDE)[['location_id', 'location_name']].drop_duplicates()
    supplied_ids = set(registry.location_id)
    national_ids = set(national['Location ID'])
    assert len(supplied_ids) == 203
    assert supplied_ids == set(wb_region['Location ID']) == set(wb_income['Location ID'])
    assert national_ids - supplied_ids == {320, 374, 413}
    assert supplied_ids - national_ids == {354, 361}
    native = read(NATIVE)
    regional_ids = set(native.location_id)
    assert regional_ids == GCC | {144}
    assert regional_ids <= national_ids
    gcc_set = locations[(locations['Location Set Version ID'] == 1277) & (locations.Level == 1)]
    assert set(gcc_set['Location ID']) == GCC
    assert set(native.cause_id) == {544}
    parkinson = causes[causes['Cause ID'] == 544]
    assert len(parkinson) == 1
    assert parkinson.iloc[0]['Cause Name'] == "Parkinson's disease"
    assert parkinson.iloc[0]['Parent ID'] == 542 and parkinson.iloc[0]['Level'] == 3
    assert parkinson.iloc[0]['Parent Name'] == 'Neurological disorders'
    by_version = {v: g.set_index('Location ID') for v, g in locations.groupby('Location Set Version ID')}
    source_names = registry.set_index('location_id').location_name.to_dict()
    rows = []
    for location_id in sorted(national_ids | supplied_ids):
        record = {'location_id': location_id, 'workbook_location_name': source_names.get(location_id, ''),
                  'present_in_workbook': location_id in supplied_ids}
        for prefix, version in [('gbd', 1511), ('wb_region', 1300), ('wb_income', 1305)]:
            table = by_version[version]
            present = location_id in table.index
            record[prefix + '_present'] = present
            record[prefix + '_set_version_id'] = version
            for column, suffix in [('Location Name', 'source_name'), ('Parent ID', 'parent_id'), ('Level', 'level')]:
                record[prefix + '_' + suffix] = table.loc[location_id, column] if present else None
            parent = int(table.loc[location_id, 'Parent ID']) if present else None
            record[prefix + '_parent_name'] = table.loc[parent, 'Location Name'] if parent in table.index else None
        record['status'] = ('national_and_world_bank' if location_id in national_ids & supplied_ids
                            else 'world_bank_only_in_supplied_panel' if location_id in supplied_ids
                            else 'national_not_in_supplied_world_bank_panel')
        rows.append(record)
    crosswalk = pd.DataFrame(rows)
    assert crosswalk.status.value_counts().to_dict() == {
        'national_and_world_bank': 201, 'world_bank_only_in_supplied_panel': 2,
        'national_not_in_supplied_world_bank_panel': 3}
    old_cross = read(ROOT / 'results/source_compatibility_v1/published_national_location_crosswalk.csv')
    old_matches = set(old_cross.loc[old_cross.status == 'matched_national_reference', 'location_id'].astype(int))
    assert old_matches == national_ids & supplied_ids
    name_differences = registry.merge(wb_region, left_on='location_id', right_on='Location ID', validate='one_to_one')
    name_differences = name_differences[name_differences.location_name != name_differences['Location Name']]
    assert set(name_differences.location_id) == {155, 205}
    information = subprocess.run(['pdftotext', '-layout', str(inputs[1]), '-'], check=True,
                                 text=True, capture_output=True).stdout
    assert 'GBD 2023' in information and 'October 23, 2025' in information
    assert 'GBD 2021' in information  # Preserve the official wording inconsistency.
    assert all(sha(ROOT / p) == digest for p, digest in protected.items())
    assert all(sha(ROOT / p) == digest for p, digest in hashes.items())
    (PACKAGE / 'raw').mkdir(parents=True)
    (PACKAGE / 'processed').mkdir()
    RESULTS.mkdir(parents=True)
    manifest = []
    now = datetime.now(timezone.utc).isoformat()
    for original in inputs:
        target = PACKAGE / 'raw' / original.name
        shutil.copyfile(original, target)
        assert sha(original) == sha(target)
        manifest.append({'original_path': rel(original), 'preserved_path': rel(target),
                         'bytes': original.stat().st_size, 'sha256': sha(target),
                         'acquired_by': 'user', 'download_timestamp_verified': False,
                         'inspected_utc': now, 'official_release_date': '2025-10-23',
                         'source_record_url': SOURCE_URL, 'source_doi': '10.6069/KMAH-ET96'})
    with (PACKAGE / 'manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    outputs = [('location_hierarchies.csv', locations), ('cause_hierarchy.csv', causes),
               ('rei_hierarchy.csv', rei), ('gbd_national_locations_204.csv', national),
               ('wb_region_locations_203.csv', wb_region), ('wb_income_locations_203.csv', wb_income)]
    for name, frame in outputs:
        frame.to_csv(PACKAGE / 'processed' / name, index=False, mode='x')
    with (PACKAGE / 'processed/information_sheet.txt').open('x') as stream:
        stream.write(information)
    crosswalk.to_csv(RESULTS / 'location_crosswalk.csv', index=False, mode='x')
    name_differences[['location_id', 'location_name', 'Location Name']].to_csv(
        RESULTS / 'source_name_encoding_differences.csv', index=False, mode='x')
    citations = []
    for p in sorted(PACKAGE.rglob('*')):
        if not p.is_file():
            continue
        citations.append({'file': rel(p), 'citation': 'Global Burden of Disease Collaborative Network. GBD 2023 Cause, REI, and Location Hierarchies. IHME, 2025.',
                          'doi': '10.6069/KMAH-ET96', 'source_url': SOURCE_URL,
                          'transformation': 'Exact original copy' if p.parent.name == 'raw' else 'Derived source tables, PDF text or acquisition record',
                          'sha256': sha(p)})
    pd.DataFrame(citations).to_csv(PACKAGE / 'file_citations.csv', index=False, mode='x')
    summary = {'inspected_utc': now, 'location_set_versions': int(locations['Location Set Version ID'].nunique()),
               'location_rows': len(locations), 'cause_rows': len(causes), 'rei_rows': len(rei),
               'gbd_national_locations': 204, 'supplied_locations': 203, 'matched_national_locations': 201,
               'exact_match_world_bank_region_set_1300': True, 'exact_match_world_bank_income_set_1305': True,
               'national_locations_absent': national[national['Location ID'].isin({320, 374, 413})][['Location ID', 'Location Name']].to_dict('records'),
               'supplied_locations_outside_main_gbd_set': [354, 361],
               'regional_seven_locations_verified': True, 'gcc_six_locations_verified': True,
               'pd_cause_544_parent_542_level_3_verified': True,
               'hierarchy_defines_population_boundary_overlap': False,
               'information_sheet_contains_2021_wording_inconsistency': True,
               'original_disease_value_provenance_verified': False, 'asr_bounds_added': False,
               'new_asr_checkpoint_export_found_at_inspection': False,
               'models_fitted': 0, 'raw_and_locked_files_unchanged': True,
               'input_sha256': hashes, 'script_sha256': sha(Path(__file__)),
               'protected_sha256': protected,
               'output_sha256': {rel(p): sha(p) for p in sorted(list(PACKAGE.rglob('*')) + list(RESULTS.glob('*'))) if p.is_file()}}
    with (RESULTS / 'validation.json').open('x') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith('sha256')}, indent=2))


if __name__ == '__main__':
    main()
