"""Acquire bibliographic metadata for the JDR reference-style conversion."""
import concurrent.futures
import json
from pathlib import Path
import re
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/JDR_GBD_PARK/references'
EXTRA={
 'schrag':'10.1136/jnnp.69.3.308',
 'sturkenboom':'10.1016/S1474-4422(14)70055-9',
 'hely':'10.1002/mds.21956',
 'gather':'10.1371/journal.pmed.1002056',
 'tripod':'10.1136/bmj-2023-078378',
 'shorrocks':'10.1007/s10888-011-9214-z',
 'scikit':'10.5555/1953048.2078195',
}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 refs=json.loads((ROOT/'manuscript/ije_draft_v2_raw_integration/references.json').read_text())
 dois={}
 for key,r in refs.items():
  m=re.search(r'https://doi.org/(\S+)',r['text'])
  if m and not key=='hierarchy':dois[key]=m.group(1).rstrip('.')
 dois.update(EXTRA)
 def fetch(item):
  key,doi=item;dest=OUT/f'{key}_crossref.json'
  if dest.exists():return key,'cached'
  try:
   r=requests.get('https://api.crossref.org/works/'+quote(doi,safe=''),timeout=40,
       headers={'User-Agent':'ParkinsonForecastResearch/1.0 (bibliographic verification)'})
   if r.ok:
    message=r.json()['message'];dest.write_text(json.dumps(message,ensure_ascii=False,indent=2)+'\n')
    return key,'downloaded'
   return key,'http_'+str(r.status_code)
  except requests.RequestException as e:return key,type(e).__name__
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  statuses=dict(pool.map(fetch,dois.items()))
 (OUT/'metadata_acquisition.json').write_text(json.dumps({'status':statuses,'requested_doi':dois},indent=2)+'\n')
 print(json.dumps(statuses,indent=2))

if __name__=='__main__':main()
