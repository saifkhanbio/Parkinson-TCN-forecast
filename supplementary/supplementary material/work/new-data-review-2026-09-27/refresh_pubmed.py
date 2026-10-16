"""Targeted novelty search through the installed NCBI skill; no raw payloads saved."""
from pathlib import Path
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parent
SKILL = Path('/home/saif/.codex/plugins/cache/openai-curated-remote/life-sciences-literature/0.1.5/skills/ncbi-entrez-skill/scripts/ncbi_entrez.py')
spec = importlib.util.spec_from_file_location('ncbi_entrez', SKILL)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
pd = '("Parkinson Disease"[Mesh] OR parkinson*[tiab])'
end_date = ' AND ("1800/01/01"[Date - Publication] : "2026/09/27"[Date - Publication])'
queries = {
    'regional_forecasting': pd + ' AND (Saudi[tiab] OR Gulf[tiab] OR Bahrain[tiab] OR Kuwait[tiab] OR Oman[tiab] OR Qatar[tiab] OR "United Arab Emirates"[tiab] OR Jordan[tiab]) AND (forecast*[tiab] OR project*[tiab] OR "transfer learning"[tiab])',
    'population_transfer': pd + ' AND ("transfer learning"[tiab] OR "domain adaptation"[tiab] OR "negative transfer"[tiab] OR transportab*[tiab]) AND (forecast*[tiab] OR prevalence[tiab] OR incidence[tiab] OR demographic*[tiab] OR burden[tiab])',
    'age_sex_decomposition': pd + ' AND ("global burden"[tiab] OR GBD[tiab]) AND (decomposition[tiab] OR demograph*[tiab]) AND (forecast*[tiab] OR project*[tiab])',
    'population_denominators': pd + ' AND ("World Population Prospects"[tiab] OR "population estimates"[tiab] OR denominator*[tiab]) AND (forecast*[tiab] OR prevalence[tiab] OR incidence[tiab])',
    'regional_migration': pd + ' AND (migrant*[tiab] OR migration[tiab] OR nationality[tiab]) AND (Saudi[tiab] OR Gulf[tiab] OR Qatar[tiab] OR Emirates[tiab] OR Bahrain[tiab] OR Kuwait[tiab] OR Oman[tiab])',
}
logs=[]
all_ids=set()
for name, query in queries.items():
    term=query+end_date
    r=api.execute({'endpoint':'esearch','params':{'db':'pubmed','term':term,'retmode':'json','retmax':50,'sort':'pub date'},'record_path':'esearchresult','max_items':50,'max_depth':6,'timeout_sec':30})
    summary=r.get('summary',{})
    ids=summary.get('idlist',[])
    pages=[]
    if r.get('ok'):
        for offset in range(50,int(summary.get('count',0)),10):
            time.sleep(0.4)
            page=api.execute({'endpoint':'esearch','params':{'db':'pubmed','term':term,'retmode':'json','retmax':10,'retstart':offset,'sort':'pub date'},'record_path':'esearchresult','max_items':10,'max_depth':6,'timeout_sec':30})
            if not page.get('ok'):
                raise RuntimeError('Pagination failed: '+str(page.get('error')))
            ids.extend(page.get('summary',{}).get('idlist',[]))
            pages.append({'summary':page.get('summary',{}),'sources':page.get('sources',[]),'checked_sources':page.get('checked_sources',[])})
    all_ids.update(ids)
    record={'name':name,'query':term,'ok':r.get('ok'),'summary':summary,'all_retrieved_ids':ids,'additional_pages':pages,'sources':r.get('sources',[]),'checked_sources':r.get('checked_sources',[]),'error':r.get('error')}
    logs.append(record)
    (ROOT/'pubmed_search_summary.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    print(json.dumps({'name':name,'ok':r.get('ok'),'summary':summary,'error':r.get('error')}),flush=True)
    if not r.get('ok'):
        raise RuntimeError('PubMed request failed; see saved error before retrying with appropriate network permission.')
    time.sleep(0.4)
record_path=ROOT/'pubmed_records_summary.json'
records=json.loads(record_path.read_text()) if record_path.exists() else []
done={r.get('pmid') for r in records}
ids=sorted(all_ids-done)
for start in range(0,len(ids),5):
    batch=ids[start:start+5]
    r=api.execute({'endpoint':'efetch','params':{'db':'pubmed','id':','.join(batch),'retmode':'xml'},'response_format':'xml','max_items':12,'max_depth':15,'timeout_sec':30})
    root=r.get('summary',{}).get('PubmedArticleSet',{})
    articles=root.get('PubmedArticle',[])
    if isinstance(articles,dict): articles=[articles]
    for item in articles:
        citation=item.get('MedlineCitation',{})
        article=citation.get('Article',{})
        record={'pmid':str(citation.get('PMID','')),'title':article.get('ArticleTitle'),'abstract':article.get('Abstract'),'journal':article.get('Journal'),'identifiers':item.get('PubmedData',{}).get('ArticleIdList'),'sources':[s for s in r.get('sources',[]) if s.get('supports_claim') and s.get('kind')=='evidence'],'checked_sources':r.get('checked_sources',[])}
        records.append(record)
        print(json.dumps({'pmid':record['pmid'],'title':record['title']}),flush=True)
    if not r.get('ok'):
        records.append({'failed_batch':batch,'error':r.get('error'),'checked_sources':r.get('checked_sources',[])})
    (ROOT/'pubmed_records_summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    time.sleep(0.4)
