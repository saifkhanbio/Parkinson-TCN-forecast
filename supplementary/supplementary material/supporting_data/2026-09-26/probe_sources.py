"""Read public source pages / response headers without downloading data files."""
import concurrent.futures, json, re, sys
from html import unescape
from urllib.parse import urljoin
import pip._vendor.requests as requests

sys.stdout.reconfigure(encoding='utf-8')
URLS = {
 'gcc_population_csv': 'https://sdmx.marsa.gccstat.org/FusionRegistry/ws/public/sdmxapi/rest/data/GCCSTAT.PSS,DF_PSS_DEM_POP,1.0/all/all/?labels=name&format=csv-:-comma-true',
 'gastat_search': 'https://www.stats.gov.sa/en/search?delta=60&q=population&sort=&start=1',
 'un_wpp': 'https://population.un.org/wpp/',
 'un_swagger': 'https://population.un.org/dataportalapi/swagger/DataPortalOpenAPISpecificationv1.0/swagger.json',
}

def probe(item):
 key, url = item
 try:
  if key == 'gcc_population_csv':
   r = requests.get(url, stream=True, timeout=(20,45))
   result = dict(source=key, url=r.url, status=r.status_code, headers=dict(r.headers))
   r.close()
  else:
   r = requests.get(url, timeout=(20,45))
   result = dict(source=key, url=r.url, status=r.status_code)
   if key == 'un_swagger':
    j=r.json(); result['paths']=list(j.get('paths',{})); result['security']=j.get('components',{}).get('securitySchemes',{})
   else:
    links = [(unescape(h), re.sub('<[^>]+>', '', t).strip()) for h,t in re.findall(r'<a[^>]+href=[\"\x27]([^\"\x27]+)[\"\x27][^>]*>(.*?)</a>',r.text,re.S|re.I)]
    result['links']=[(urljoin(r.url,h),t) for h,t in links if re.search('popul|estimate|xlsx|download|excel|csv|\.js',h+' '+t,re.I)]
    result['scripts']=re.findall(r'<script[^>]+src=[\"\x27]([^\"\x27]+)',r.text,re.I)
    result['text_start']=re.sub('<[^>]+>', ' ',r.text)[:1200]
  return result
 except Exception as e:
  return dict(source=key,error=str(e))

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for result in pool.map(probe,URLS.items()): print(json.dumps(result,ensure_ascii=False))
