"""Replay UN source cells and verify the additive source-audit report package."""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree, html

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'supporting_data/2026-09-30_source_audit'
RESULTS = ROOT / 'results/source_compatibility_v1'
REPORT = ROOT / 'reports/source_compatibility_v1'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
Q = {'Lower 95': .025, 'Lower 80': .1, 'Median': .5,
     'Upper 80': .9, 'Upper 95': .975}
CSV_NAME = 'un_gcc_age_sex_marginal_quantiles_2024_2028.csv'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def replay_cells(rows):
    """Use lxml and recorded source coordinates independently of extraction."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_file'], row['source_sheet']].append(row)
    checked = 0
    for (filename, sheet_name), records in groups.items():
        wanted = defaultdict(list)
        for record in records:
            wanted[int(re.search(r'\d+$', record['source_cell']).group())].append(record)
        with zipfile.ZipFile(ROOT / filename) as archive:
            strings = [''.join(e.itertext()) for e in
                       etree.fromstring(archive.read('xl/sharedStrings.xml'))]
            relationships = {e.get('Id'): e.get('Target') for e in
                             etree.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
            sheets = etree.fromstring(archive.read('xl/workbook.xml')).find('s:sheets', NS)
            selected = [s for s in sheets if s.get('name') == sheet_name]
            assert len(selected) == 1
            rid = selected[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            target = relationships[rid]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target

            def value(cell):
                assert cell.find('s:f', NS) is None, 'Unexpected source formula'
                v = cell.find('s:v', NS)
                if v is None:
                    return ''.join(cell.itertext())
                return strings[int(v.text)] if cell.get('t') == 's' else v.text

            header = None
            with archive.open(target) as stream:
                for _, row in etree.iterparse(stream, events=('end',), tag='{%s}row' % NS['s']):
                    index = int(row.get('r'))
                    if index == 17 or index in wanted:
                        cells = {re.sub(r'\d', '', c.get('r')): value(c) for c in row}
                        if index == 17:
                            header = cells
                        else:
                            assert header is not None
                            for record in wanted[index]:
                                col = re.sub(r'\d', '', record['source_cell'])
                                assert cells['F'] == record['iso3']
                                assert cells['C'] == record['location_name']
                                assert int(float(cells['K'])) == int(record['year'])
                                assert header[col] == record['source_age_group']
                                assert float(record['quantile']) == Q[sheet_name]
                                sex = 'Female' if 'female.xlsx' in filename else 'Male'
                                assert record['sex'] == sex
                                assert math.isclose(float(cells[col]), float(record['source_population_thousands']), rel_tol=1e-12)
                                assert math.isclose(float(cells[col]) * 1000, float(record['population_persons']), rel_tol=1e-12)
                                checked += 1
                    row.clear()
                    while row.getprevious() is not None:
                        del row.getparent()[0]
    assert checked == len(rows) == 3600
    return checked


def verify_country_tables():
    expected = {r['published_name'] for r in csv_rows(RESULTS / 'published_national_location_crosswalk.csv')
                if r['published_name']}
    assert len(expected) == 204
    document = html.parse(str(PACKAGE / 'raw/gbd2023_demographics.html'))
    # Two subsequent tables inconsistently indent a regional aggregate.
    # Identify aggregates from the first table's explicit three-column labels.
    aggregates = {r[0].text_content().strip()
                  for r in document.find('.//table').findall('.//tr')
                  if len(r) and r[0].tag == 'td' and r[0].get('colspan') == '3'}
    verified = []
    for index, table in enumerate(document.findall('.//table'), 1):
        names = []
        for row in table.findall('.//tr'):
            cells = row.findall('td')
            if len(cells) > 2 and not cells[0].text_content().strip() and cells[1].get('colspan') == '2':
                for sup in cells[1].findall('.//sup'):
                    sup.getparent().remove(sup)
                label = cells[1].text_content().strip()
                if label not in aggregates:
                    names.append(label)
        if len(names) == 204:
            assert set(names) == expected, ('National table discrepancy', index)
            verified.append(index)
    assert len(verified) >= 2, 'Need a second published national table'
    return verified


def validate(record=False):
    if not __debug__:
        raise SystemExit('Run without -O: assertions are required.')
    subprocess.run([sys.executable, str(ROOT / 'scripts/audit_source_compatibility.py'), '--verify-only'], cwd=ROOT, check=True)
    rows = csv_rows(PACKAGE / 'processed' / CSV_NAME)
    assert sha(PACKAGE / 'processed' / CSV_NAME) == sha(RESULTS / CSV_NAME)
    assert len({tuple(r[k] for k in ['iso3', 'sex', 'year', 'source_age_group', 'quantile']) for r in rows}) == 3600
    replayed = replay_cells(rows)
    tables = verify_country_tables()
    citations = csv_rows(PACKAGE / 'file_citations.csv')
    assert len(citations) == 12
    for row in citations:
        assert sha(ROOT / row['file']) == row['sha256']
    assert len(csv_rows(RESULTS / 'discrepancy_ledger.csv')) == 11
    assert len(csv_rows(RESULTS / 'uncertainty_inventory.csv')) == 10
    documents = [REPORT / 'report.md', REPORT / 'requested_exports.md',
                 ROOT / 'reports/global_asr_source_review_notice.md', PACKAGE / 'README.md']
    output = REPORT / 'validation.json'
    local_links = 0
    local_targets = []
    for document in documents:
        for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
            if re.match(r'https?://', target) or target.startswith('#'):
                continue
            path = (document.parent / target.split('#')[0]).resolve()
            assert path.is_file() or (record and path == output), (document, target)
            local_targets.append(path)
            local_links += 1
    artifacts = documents + [PACKAGE / 'file_citations.csv', PACKAGE / 'processed' / CSV_NAME,
                              RESULTS / 'discrepancy_ledger.csv', RESULTS / 'uncertainty_inventory.csv',
                              RESULTS / 'validation.json', Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in artifacts}
    if record:
        payload = {'created_utc': datetime.now(timezone.utc).isoformat(),
                   'package_verified': True, 'source_authentication_complete': False,
                   'core_checks_passed': 33, 'core_checks_failed': 1,
                   'un_cells_independently_replayed': replayed,
                   'national_table_indices_crosschecked': tables,
                   'local_markdown_links_checked': local_links,
                   'artifact_sha256': hashes, 'models_fitted': 0}
        with output.open('x') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
    else:
        assert json.loads(output.read_text())['artifact_sha256'] == hashes
    assert all(path.is_file() for path in local_targets)
    print(json.dumps({'un_source_cells_verified': replayed, 'national_tables_verified': tables,
                      'local_links_verified': local_links, 'source_authentication_complete': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='Create a new validation record; never overwrite.')
    validate(parser.parse_args().record)
