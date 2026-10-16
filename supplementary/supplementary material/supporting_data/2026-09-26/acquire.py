"""Public supplementary data acquisition. Use preflight before download.

Raw files are preserved verbatim (apart from HTTP transfer decompression).
No authentication, cookies or personal information are saved.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json, hashlib, sys
import pip._vendor.requests as requests

ROOT=Path(__file__).resolve().parent
CATALOG=ROOT/'catalog.json'
MANIFEST=ROOT/'manifest.json'

def run(item, download=False):
 record=dict(item)
 record['checked_utc']=datetime.now(timezone.utc).isoformat()
 dest=ROOT/'raw'/item['filename']
 try:
  if download and dest.exists():
   record.update(status='already_downloaded',bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
   return record
  with requests.get(item['url'],stream=True,timeout=(20,60),headers={'User-Agent':'GBD-Parkinson-research-data-download/1.0'}) as r:
   record.update(http_status=r.status_code,final_url=r.url,content_type=r.headers.get('Content-Type'),content_length=r.headers.get('Content-Length'))
   r.raise_for_status()
   if not download:
    record['status']='preflight_ok'
    return record
   dest.parent.mkdir(parents=True,exist_ok=True)
   data=bytearray()
   for chunk in r.iter_content(1024*1024):
    data.extend(chunk)
    if len(data)>150_000_000: raise ValueError('File exceeds 150 MB safety limit; not saved')
   head=bytes(data[:200]).lower()
   if b'<html' in head or b'<!doctype html' in head: raise ValueError('HTML response instead of requested data')
   ext=dest.suffix.lower()
   if ext in ('.xlsx','.docx','.zip') and not data.startswith(b'PK'): raise ValueError('Expected ZIP-based document signature')
   if ext=='.pdf' and not data.startswith(b'%PDF'): raise ValueError('Expected PDF signature')
   if ext=='.doc' and not data.startswith(bytes.fromhex('d0cf11e0a1b11ae1')): raise ValueError('Expected binary Word document signature')
   dest.write_bytes(data)
   record.update(status='downloaded',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),path=str(dest.relative_to(ROOT)))
 except Exception as e:
  record.update(status='failed',error=str(e))
 return record

if __name__=='__main__':
 sys.stdout.reconfigure(encoding='utf-8')
 mode=sys.argv[1]
 items=json.loads(CATALOG.read_text(encoding='utf-8'))
 if len(sys.argv)>2: items=[i for i in items if i['id'] in sys.argv[2:]]
 with ThreadPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(lambda x:run(x,mode=='download'),items))
 target=MANIFEST if mode=='download' else ROOT/'preflight.json'
 old=json.loads(target.read_text(encoding='utf-8')) if target.exists() else []
 merged={r['id']:r for r in old}; merged.update({r['id']:r for r in results})
 target.write_text(json.dumps(list(merged.values()),indent=2,ensure_ascii=False),encoding='utf-8')
 for r in results: print(json.dumps({k:v for k,v in r.items() if k not in ('url','final_url','source_page','role')},ensure_ascii=False))
