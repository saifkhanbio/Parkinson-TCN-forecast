"""Preserve public provider citation/attribution policies for the JDR audit."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pip._vendor import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/JDR_GBD_PARK/references/provider_citation_audit'
SOURCES = {
    'ihme_terms': 'https://www.healthdata.org/about/terms-and-conditions',
    'ihme_agreement': 'https://www.healthdata.org/data-tools-practices/data-practices/ihme-free-charge-non-commercial-user-agreement',
    'gastat_terms': 'https://stats.gov.sa/en/use-policy',
    'gccstat_terms': 'https://dp.marsa.gccstat.org/terms-use',
    'chi_terms': 'https://www.chi.gov.sa/en/open-data/Pages/requirements.aspx',
    'moh_open_data': 'https://www.moh.gov.sa/en/Ministry/OpenData/Pages/default.aspx',
    'moh_policy': 'https://www.moh.gov.sa/en/Ministry/OpenData/Pages/OpenDataUsagePolicy.aspx',
    'ihme_faq': 'https://www.healthdata.org/gbd/faq',
    'wpp_terms': 'https://population.un.org/wpp/TermsOfUse/',
}


def acquire(item):
    key, url = item
    record = {'id': key, 'url': url, 'reviewed_utc': datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=35, headers={'User-Agent': 'Mozilla/5.0 citation verification'})
        path = OUT / (key + '.html')
        path.write_bytes(response.content)
        record.update(status=response.status_code, final_url=response.url,
                      path=str(path.relative_to(ROOT)), bytes=len(response.content),
                      sha256=hashlib.sha256(response.content).hexdigest())
    except requests.RequestException as exc:
        record['error'] = str(exc)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(acquire, SOURCES.items()))
    (OUT / 'retrieval_manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    for row in records:
        print(row['id'], row.get('status', row.get('error')), row.get('bytes', ''))
    if any('error' in row for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
