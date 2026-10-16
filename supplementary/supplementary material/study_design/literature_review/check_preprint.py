"""Check the identified medRxiv DOI through the installed literature skill."""
import sys, json, importlib.util
from pathlib import Path
import pip._vendor.requests
sys.modules['requests'] = pip._vendor.requests
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
skill = Path(r'C:\Users\saifk\.codex\plugins\cache\openai-curated-remote\life-sciences-literature\0.1.5\skills\biorxiv-skill\scripts\rest_request.py')
spec = importlib.util.spec_from_file_location('preprints', skill)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
out=[]
for path in ['/details/medrxiv/10.1101/2025.10.22.25338533/na/json','/pubs/medrxiv/10.1101/2025.10.22.25338533/na/json']:
    r=api.execute({'base_url':'https://api.biorxiv.org','path':path,'method':'GET','max_items':30,'max_depth':8,'timeout_sec':45})
    print(json.dumps(r, ensure_ascii=False),flush=True)
    out.append({'path':path,'ok':r.get('ok'),'summary':r.get('summary'),'records':r.get('records'),'sources':r.get('sources'),'checked_sources':r.get('checked_sources'),'error':r.get('error')})
Path(__file__).with_name('preprint_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
