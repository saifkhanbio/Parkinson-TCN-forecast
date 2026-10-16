"""Acquire public Saudi evidence and official access documentation, with hashes.

Only anonymous GET requests are used. No applications, contacts, or logins.
Run with the agpu environment; each response is preserved once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin

import requests

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ("moh_request_form", "moh_data_request_form.pdf", "https://www.moh.gov.sa/Ministry/Statistics/Documents/Data-Request-Form.pdf", "access_documentation"),
    ("moh_sharing", "moh_data_sharing.html", "https://hdp.moh.gov.sa/en/data-sharing", "access_documentation"),
    ("chi_request", "chi_access_data.html", "https://www.chi.gov.sa/en/open-data/Pages/access-data.aspx", "access_documentation"),
    ("nhrsp", "shc_nhrsp_access.html", "https://shc.gov.sa/ar/EServices/Pages/nhrsp.aspx", "access_documentation"),
    ("gastat_request", "gastat_microdata_request.html", "https://www.stats.gov.sa/en/request-for-scientific-use-files", "access_documentation"),
    ("gastat_guide", "gastat_microdata_guide.pdf", "https://www.stats.gov.sa/documents/20117/2435245/Microdata%2BRequest_userguide_EN.pdf/745f3620-d60e-f8e3-405c-040200645c56", "access_documentation"),
    ("saudi_genetics_2015", "saudi_genetics_2015.html", "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0135950", "published_selected_clinical_sample"),
    ("who_mortality_catalog", "who_mortality_catalog.html", "https://www.who.int/data/data-collection-tools/who-mortality-database", "reported_mortality_catalog"),
    ("iqvia_pd_abstract", "iqvia_saudi_pd_2025_abstract.html", "https://www.mdsabstracts.org/abstract/effect-of-covid-19-pandemic-on-utilization-of-parkinsons-medications-in-saudi-arabia-a-repeated-cross-sectional-study/", "published_medication_sales_abstract"),
    ("menasa_protocol", "menasa_protocol.html", "https://www.ssph-journal.org/journals/international-journal-of-public-health/articles/10.3389/ijph.2025.1608016/full", "registry_protocol_not_patient_data"),
    ("kfmc_registry_listing", "moh_research_2016.pdf", "https://www.moh.gov.sa/Ministry/MediaCenter/Publications/Pages/MOH-2016-VIEW.pdf", "historical_registry_project_listing"),
    ("chi_pd_guidance", "chi_parkinson_guidance.pdf", "https://www.chi.gov.sa/Style%20Library/IDF_Branding/Indication/155%20-%20Parkinson%20Disease-Indication%20Update.pdf", "clinical_guidance_not_surveillance"),
]


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.links = []; self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current = [dict(attrs).get("href", ""), ""]

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.links.append(self.current); self.current = None


def retrieve(spec):
    ident, filename, url, role = spec
    dest = ROOT / "raw" / filename
    record = dict(id=ident, url=url, role=role, checked_utc=datetime.now(timezone.utc).isoformat())
    if dest.exists():
        content = dest.read_bytes()
        return dict(record, status="existing_preserved", path="raw/"+filename,
                    bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    try:
        with requests.get(url, timeout=(15, 45), stream=True) as response:
            record.update(http_status=response.status_code, final_url=response.url,
                          content_type=response.headers.get("Content-Type", ""))
            response.raise_for_status()
            pieces, size = [], 0
            for chunk in response.iter_content(65536):
                pieces.append(chunk); size += len(chunk)
                if size > 50 * 1024 * 1024:
                    raise ValueError("Public response exceeds 50 MiB bounded download")
            content = b"".join(pieces)
        if filename.endswith(".pdf") and not content.startswith(b"%PDF"):
            raise ValueError("Response is not a PDF")
        if filename.endswith(".zip") and not content.startswith(b"PK"):
            raise ValueError("Response is not a ZIP archive")
        if b"Checking your browser" in content or b"Just a moment..." in content[:5000]:
            raise ValueError("Browser challenge, not source content")
        with dest.open("xb") as stream:
            stream.write(content)
        record.update(status="downloaded", path="raw/"+filename, bytes=len(content),
                      sha256=hashlib.sha256(content).hexdigest())
    except (requests.RequestException, ValueError) as error:
        record.update(status="failed", error=str(error)[:500])
    return record


def discover():
    selected, inventory = [], []
    for ident, filename, base, role in SOURCES:
        path = ROOT / "raw" / filename
        if not path.exists() or not filename.endswith(".html"):
            continue
        parser = Links(); parser.feed(path.read_text(errors="replace"))
        for href, label in parser.links:
            url = urljoin(base, href)
            inventory.append(dict(source_id=ident, text=" ".join(label.split()), url=url))
            low = url.lower()
            if ident == "who_mortality_catalog" and low.split("?")[0].endswith(".zip"):
                leaf = low.split("?")[0].split("/")[-1]
                if any(x in leaf for x in ["avail", "country", "documentation", "notes"]):
                    selected.append(("who_"+leaf[:-4], "who_"+leaf, url, "mortality_availability_and_documentation"))
            if ident == "saudi_genetics_2015" and "article/file" in low and "0135950.s" in low:
                # Download published supporting tables; figures are not needed.
                suffix = low.split("0135950.s")[-1].split("&")[0]
                if suffix.isdigit() and int(suffix) >= 4:
                    selected.append(("saudi_genetics_s"+suffix, "saudi_genetics_s"+suffix+".bin", url, "published_genetic_supporting_table"))
    (ROOT / "public_link_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+"\n")
    return list({row[2]: row for row in selected}.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["initial", "linked", "mortality", "population_methods"], default="initial")
    args = parser.parse_args()
    (ROOT / "raw").mkdir(parents=True, exist_ok=True)
    manifest = ROOT / ("manifest_"+args.stage+".json")
    if manifest.exists():
        raise SystemExit("Refusing to overwrite acquisition manifest")
    if args.stage == 'initial':
        specs = SOURCES
    elif args.stage == 'linked':
        specs = discover()
    elif args.stage == 'population_methods':
        specs = [
            ('gastat_population_methods','gastat_population_methods.pdf','https://www.stats.gov.sa/documents/d/guest/methodology-and-quality-report-for-population-projections-and-estimates-statistics-en','population_definition_documentation'),
            ('gastat_population_methods_page','gastat_population_methods.html','https://www.stats.gov.sa/en/w/methodology-and-quality-report-for-population-projections-and-estimates-statistics','population_definition_documentation'),
        ]
    else:
        # Saudi availability: 2009/2012 and 2021–2024. Obtain only these blocks.
        inventory = json.loads((ROOT/'public_link_inventory.json').read_text())
        specs = []
        for item in inventory:
            url = item['url']; leaf = url.split('?')[0].split('/')[-1]
            if item['source_id']=='who_mortality_catalog' and leaf in ['morticd10_part3.zip','morticd10_part6.zip','mort_pop.zip']:
                specs.append(('who_'+leaf[:-4], 'who_'+leaf, url, 'reported_mortality_or_reference_population'))
        specs.append(('nhrsp_userguide','nhrsp_userguide.pdf','https://shc.gov.sa/ar/EServices/Documents/nhrsp/UserGuide.pdf','access_documentation'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(retrieve, specs))
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False)+"\n")
    for r in records:
        print(r["id"], r["status"], r.get("bytes", ""), r.get("error", ""), flush=True)
