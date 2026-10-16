"""Find public file links from official pages."""
import concurrent.futures, json,re,sys
from html import unescape
from urllib.parse import urljoin
import pip._vendor.requests as r
sys.stdout.reconfigure(encoding='utf-8')
pages={
 'gastat':'https://www.stats.gov.sa/en/search?delta=60&q=Population_Estimates&sort=&start=1',
 'un_js':'https://population.un.org/wpp/main-JDJAGSLT.js',
 'gastat_age':'https://www.stats.gov.sa/en/w/population-estimate-by-age-group',
}
def get(item):
 key,url=item
 try:
  resp=r.get(url,timeout=(20,45));text=resp.text
  if key=='un_js':
   hits=list(dict.fromkeys(re.findall(r'[^\s\"\x27<>]{0,100}(?:\.json|\.csv|\.xlsx|assets/|Download)[^\"\x27<>]{0,100}',text)))
   return dict(id=key,status=resp.status_code,hits=hits[:100])
  links=[(urljoin(url,unescape(h)), re.sub('<[^>]+>','',t).strip()) for h,t in re.findall(r'<a[^>]+href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',text,re.S|re.I)]
  return dict(id=key,status=resp.status_code,links=[(h,t) for h,t in links if re.search(r'\.xlsx|/documents/|estimate|tableau',h+' '+t,re.I)],iframes=re.findall(r'<(?:iframe|tableau-viz)[^>]*>',text,re.I))
 except Exception as e:return dict(id=key,error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for result in pool.map(get,pages.items()): print(json.dumps(result,ensure_ascii=False))
