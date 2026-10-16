"""Retrieve public documentation only; preserve successful responses and access failures."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 12 * 1024 * 1024
SOURCES = [
    ('un_download_catalog', 'un_wpp2024_downloads.json',
     'https://population.un.org/wpp/assets/downloads.json'),
    ('un_methods', 'un_wpp2024_methodology.pdf',
     'https://population.un.org/wpp/assets/Files/WPP2024_Methodology-Report_Final.pdf'),
    ('gbd2023_demographics', 'gbd2023_demographics.html',
     'https://pmc.ncbi.nlm.nih.gov/articles/PMC12535839/'),
    ('gbd2023_nonfatal', 'gbd2023_nonfatal.html',
     'https://pmc.ncbi.nlm.nih.gov/articles/PMC12535840/'),
    ('gbd2023_catalog', 'gbd2023_catalog.html',
     'https://ghdx.healthdata.org/gbd-2023'),
]
FOLLOWUP_SOURCES = [
    ('gbd2023_hierarchy_record', 'gbd2023_hierarchy_record.html',
     'https://ghdx.healthdata.org/record/gbd-2023-cause-rei-and-location-hierarchies'),
    ('gbd2023_demographic_record', 'gbd2023_demographic_record.html',
     'https://ghdx.healthdata.org/record/ihme-data/gbd-2023-demographics-1950-2023'),
    ('gbd2023_nonfatal_record', 'gbd2023_nonfatal_record.html',
     'https://ghdx.healthdata.org/record/ihme-data/gbd-2023-yld-daly-hale-risk-1990-2023'),
    ('gbd2023_demographic_appendix', 'gbd2023_demographic_appendix.pdf',
     'https://pmc.ncbi.nlm.nih.gov/articles/instance/12535839/bin/mmc1.pdf'),
    ('gbd2023_nonfatal_appendix', 'gbd2023_nonfatal_appendix.pdf',
     'https://pmc.ncbi.nlm.nih.gov/articles/instance/12535840/bin/mmc1.pdf'),
    ('un_probabilistic_totals', 'un_ppp2024_total_population.xlsx',
     'https://population.un.org/wpp/assets/Excel%20Files/2_Indicators%20(Probabilistic)/EXCEL_FILES/2_Population/UN_PPP2024_Output_PopTot.xlsx'),
]
AGE_INTERVAL_SOURCES = [
    ('un_probabilistic_age_female', 'un_ppp2024_age_female.xlsx',
     'https://population.un.org/wpp/assets/Excel%20Files/2_Indicators%20(Probabilistic)/EXCEL_FILES/2_Population/UN_PPP2024_Output_PopulationByAge_Female.xlsx'),
    ('un_probabilistic_age_male', 'un_ppp2024_age_male.xlsx',
     'https://population.un.org/wpp/assets/Excel%20Files/2_Indicators%20(Probabilistic)/EXCEL_FILES/2_Population/UN_PPP2024_Output_PopulationByAge_Male.xlsx'),
]


def retrieve(spec):
    key, filename, url = spec
    record = {'id': key, 'url': url, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'role': 'public source documentation; no disease outcomes requested'}
    try:
        with requests.get(url, timeout=(15, 40), stream=True) as response:
            record.update(http_status=response.status_code, final_url=response.url,
                          content_type=response.headers.get('Content-Type'),
                          declared_content_length=response.headers.get('Content-Length'))
            if response.status_code != 200:
                record['status'] = 'http_access_failed'
                return record
            if int(response.headers.get('Content-Length', '0') or '0') > MAX_BYTES:
                record['status'] = 'declared_size_exceeds_limit'
                return record
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > MAX_BYTES:
                    record['status'] = 'size_limit_exceeded'
                    return record
                chunks.append(chunk)
        content = b''.join(chunks)
        record['bytes'] = len(content)
        if filename.endswith('.html') and (
            b'Checking your browser' in content or b'captcha' in content[:3000].lower()
        ):
            record['status'] = 'browser_challenge_not_source_content'
            return record
        if filename.endswith('.json'):
            json.loads(content)
        if filename.endswith('.pdf') and not content.startswith(b'%PDF'):
            record['status'] = 'unexpected_content'
            return record
        if filename.endswith('.xlsx') and not content.startswith(b'PK'):
            record['status'] = 'unexpected_content'
            return record
        path = ROOT / 'raw' / filename
        with path.open('xb') as stream:
            stream.write(content)
        record.update(status='downloaded', path='raw/' + filename,
                      sha256=hashlib.sha256(content).hexdigest())
    except (requests.RequestException, ValueError) as error:
        record.update(status='retrieval_failed', error=type(error).__name__ + ': ' + str(error))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['initial', 'followup', 'age_intervals', 'age_intervals_download'], default='initial')
    args = parser.parse_args()
    if args.stage == 'age_intervals_download':
        # Earlier response headers established that each workbook is about 21.5 MB.
        MAX_BYTES = 32 * 1024 * 1024
    manifest = ROOT / ('manifest.json' if args.stage == 'initial' else 'manifest_' + args.stage + '.json')
    if manifest.exists():
        raise SystemExit('Refusing to replace an existing metadata manifest')
    (ROOT / 'raw').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        selected = {'initial': SOURCES, 'followup': FOLLOWUP_SOURCES, 'age_intervals': AGE_INTERVAL_SOURCES,
                    'age_intervals_download': AGE_INTERVAL_SOURCES}[args.stage]
        records = list(executor.map(retrieve, selected))
    with manifest.open('x') as stream:
        json.dump(records, stream, indent=2)
        stream.write('\n')
    print(json.dumps([{k: r.get(k) for k in ['id', 'status', 'http_status', 'bytes']}
                      for r in records], indent=2))
