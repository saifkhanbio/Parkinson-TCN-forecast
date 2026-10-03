"""Acquire public citation metadata and reference documents without changing data.

Run from any directory. Network access is required. Failed requests remain in the
manifest and are never represented as downloaded references. NCBI requests use
the installed life-sciences-literature skill adapters.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

BASE = Path(__file__).resolve().parent
SKILLS = Path('/home/saif/.codex/plugins/cache/openai-curated-remote/life-sciences-literature/0.1.5/skills')
ARTICLES = {
    'gbd_nonfatal': '10.1016/S0140-6736(25)01637-X',
    'gbd_fatal': '10.1016/S0140-6736(25)01917-8',
    'gbd_demography': '10.1016/S0140-6736(25)01330-3',
    'safiri': '10.1186/s12889-023-15018-x',
    'menasa': '10.3389/ijph.2025.1608016',
    'saudi_genetics': '10.1371/journal.pone.0135950',
}
DATACITE = {
    'hierarchy': '10.6069/KMAH-ET96',
    'nhrsp_ncd': '10.60840/nhrsp-a355',
    'nhrsp_hiss': '10.60840/nhrsp-f207',
    'gbd_demographic_catalog': '10.6069/6HTR-RB51',
    'gbd_nonfatal_catalog': '10.6069/PHWS-2783',
    'sdi_catalog': '10.6069/BCX2-VV69',
}
PAGES = {
    'chi_statistics': 'https://www.chi.gov.sa/en/open-data/Pages/Indicators-and-statistics.aspx',
    'chi_terms': 'https://www.chi.gov.sa/en/open-data/Pages/requirements.aspx',
    'moh_yearbooks': 'https://www.moh.gov.sa/ministry/statistics/book/pages/default.aspx',
    'nhrsp_ncd': 'https://nhrsp.shc.gov.sa/Researches/DetailsExternal/c724a03c-cda9-ec11-8213-0050568978b7',
    'nhrsp_hiss': 'https://nhrsp.shc.gov.sa/Researches/DetailsExternal/5ff104ed-0f0e-ec11-81bf-0050568978b7',
    'who_mortality': 'https://www.who.int/data/data-collection-tools/who-mortality-database',
    'wpp': 'https://population.un.org/wpp/',
    'gccstat': 'https://dp.marsa.gccstat.org/dataset/population',
    'gastat_population_search': 'https://www.stats.gov.sa/en/search?category=all&delta=20&q=Population+estimates+by+gender%2C+nationality%2C+and+age+group',
}


def get_file(job):
    key, url, relative, expected = job
    row = {'id': key, 'url': url, 'file': relative,
           'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    path = BASE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=(15, 55), headers={'User-Agent': 'Research citation audit/1.0'})
        row.update(http_status=response.status_code, final_url=response.url,
                   content_type=response.headers.get('Content-Type', ''))
        response.raise_for_status()
        body = response.content
        if expected == 'json':
            response.json()
        if expected == 'pdf' and not body.startswith(b'%PDF'):
            raise ValueError('Response is not a PDF')
        if expected == 'xml' and b'<?xml' not in body[:100] and b'<article' not in body[:500]:
            raise ValueError('Response is not article XML')
        path.write_bytes(body)
        row.update(status='downloaded', bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    except (requests.RequestException, ValueError) as exc:
        row.update(status='failed', error=str(exc)[:400])
    return row


def skill(name, payload, output):
    script = SKILLS / name / 'scripts' / ('ncbi_entrez.py' if name == 'ncbi-entrez-skill' else 'ncbi_pmc.py')
    result = subprocess.run([sys.executable, str(script)], input=json.dumps(payload),
                            text=True, capture_output=True, timeout=150)
    path = BASE / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.stdout, encoding='utf-8')
    try:
        data = json.loads(result.stdout)
    except ValueError:
        data = {'ok': False, 'error': result.stderr[:300]}
    print(json.dumps({'skill': name, 'file': output, 'ok': data.get('ok'),
                      'error': data.get('error')}), flush=True)
    return data


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else 'metadata'
    if stage == 'metadata':
        jobs = [(key, 'https://api.datacite.org/dois/' + doi,
                 'metadata/' + key + '_datacite.json', 'json') for key, doi in DATACITE.items()]
        jobs += [(key, 'https://api.crossref.org/works/' + quote(doi, safe=''),
                  'metadata/' + key + '_crossref.json', 'json') for key, doi in ARTICLES.items()]
        jobs += [(key, url, 'documents/' + key + '.html', 'html') for key, url in PAGES.items()]
    elif stage == 'pubmed':
        raw = BASE / 'metadata/pubmed_search.json'
        skill('ncbi-entrez-skill', {'endpoint': 'esearch', 'params': {
            'db': 'pubmed', 'term': ' OR '.join('"' + doi + '"[AID]' for doi in ARTICLES.values()),
            'retmode': 'json', 'retmax': 30}, 'max_items': 30, 'save_raw': True,
            'raw_output_path': str(raw)}, 'metadata/pubmed_search_skill.json')
        if raw.exists():
            ids = json.loads(raw.read_text())['esearchresult']['idlist']
            time.sleep(0.5)
            if ids:
                skill('ncbi-entrez-skill', {'endpoint': 'efetch', 'params': {
                    'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml'}, 'max_items': 30,
                    'save_raw': True, 'raw_output_path': str(BASE / 'metadata/pubmed_articles.xml')},
                    'metadata/pubmed_fetch_skill.json')
        return
    elif stage == 'pmc':
        import xml.etree.ElementTree as ET
        for article in ET.parse(BASE / 'metadata/pubmed_articles.xml').findall('.//PubmedArticle'):
            ids = {e.get('IdType'): e.text for e in article.findall('./PubmedData/ArticleIdList/ArticleId')}
            key = next((k for k, v in ARTICLES.items() if v.lower() == ids.get('doi', '').lower()), None)
            if key and ids.get('pmc'):
                skill('ncbi-pmc-skill', {'params': {'id': ids['pmc']}, 'max_items': 3,
                    'save_raw': True, 'raw_output_path': str(BASE / ('metadata/' + key + '_pmc_raw.json'))},
                    'metadata/' + key + '_pmc_skill.json')
        return
    elif stage == 'extra':
        filename = sys.argv[2] if len(sys.argv) > 2 else 'additional_downloads.json'
        jobs = json.loads((BASE / filename).read_text())
        stage = Path(filename).stem
    else:
        raise SystemExit('Use metadata, pubmed, pmc, or extra')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(get_file, jobs))
    (BASE / ('download_manifest_' + stage + '.json')).write_text(json.dumps(results, indent=2), encoding='utf-8')
    for row in results:
        print(json.dumps({k: row.get(k) for k in ['id', 'status', 'http_status', 'bytes', 'error']}), flush=True)


if __name__ == '__main__':
    main()
