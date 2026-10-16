"""Reproducible PubMed/MEDLINE search using the installed literature skill.
Stores processed search metadata and compact bibliographic records, not raw API payloads.
"""
import sys, json, time, importlib.util
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import pip._vendor.requests
sys.modules['requests'] = pip._vendor.requests
ROOT = Path(__file__).resolve().parent
SKILL = Path(r'C:\Users\saifk\.codex\plugins\cache\openai-curated-remote\life-sciences-literature\0.1.5\skills\ncbi-entrez-skill\scripts\ncbi_entrez.py')
spec = importlib.util.spec_from_file_location('literature_entrez', SKILL)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
PD = '("Parkinson Disease"[Mesh] OR parkinson*[tiab])'
SA = '("Saudi Arabia"[Mesh] OR Saudi[tiab] OR Riyadh[tiab] OR Jeddah[tiab] OR Thugbah[tiab] OR Dammam[tiab])'
GULF = '(Bahrain[tiab] OR Kuwait[tiab] OR Oman[tiab] OR Qatar[tiab] OR "United Arab Emirates"[tiab] OR Gulf[tiab] OR "Middle East"[tiab] OR MENASA[tiab] OR MENA[tiab])'
DATE = ' AND ("1800/01/01"[Date - Publication] : "2026/09/26"[Date - Publication])'
QUERIES = {
 'P1_Saudi_broad': f'{PD} AND {SA}',
 'P2_Saudi_MEDLINE': f'{PD} AND {SA} AND medline[sb]',
 'P3_Saudi_epidemiology_data': f'{PD} AND {SA} AND (epidemiolog*[tiab] OR prevalen*[tiab] OR inciden*[tiab] OR mortal*[tiab] OR burden[tiab] OR registr*[tiab] OR cohort[tiab] OR database*[tiab] OR dataset*[tiab])',
 'P4_Saudi_anyfield_prediction': f'{PD} AND Saudi[All Fields] AND (forecast*[tiab] OR predict*[tiab] OR "transfer learning"[tiab] OR "deep learning"[tiab] OR "machine learning"[tiab])',
 'P5_GBD_forecasting_global': f'{PD} AND ("global burden"[tiab] OR GBD[tiab]) AND (forecast*[tiab] OR project*[tiab] OR predict*[tiab] OR "machine learning"[tiab] OR "deep learning"[tiab] OR "transfer learning"[tiab])',
 'P6_Gulf_regional_epidemiology': f'{PD} AND {GULF} AND (epidemiolog*[tiab] OR prevalen*[tiab] OR inciden*[tiab] OR mortal*[tiab] OR burden[tiab] OR registr*[tiab] OR database*[tiab] OR cohort[tiab])',
 'P7_transfer_population': f'{PD} AND ("transfer learning"[tiab] OR "domain adaptation"[tiab] OR "multi-task"[tiab] OR multitask[tiab]) AND (burden[tiab] OR epidemiolog*[tiab] OR forecast*[tiab] OR population[tiab] OR GBD[tiab])',
 'P8_GCC_specific_burden': f'{PD} AND (Saudi[tiab] OR Bahrain[tiab] OR Kuwait[tiab] OR Oman[tiab] OR Qatar[tiab] OR "United Arab Emirates"[tiab] OR "Gulf Cooperation"[tiab]) AND (GBD[tiab] OR "global burden"[tiab] OR forecast*[tiab] OR project*[tiab])',
}

def call(payload):
    for attempt in range(3):
        out = api.execute(payload)
        if out.get('ok'): return out
        time.sleep(attempt+1)
    return out

def search():
    logs=[]
    for name,term in QUERIES.items():
        term += DATE
        records=[];sources=[];checked=[];count=None
        for start in range(0,10000,100):
            r=call({'endpoint':'esearch','params':{'db':'pubmed','term':term,'retmode':'json','retmax':100,'retstart':start,'sort':'pub date'},'record_path':'esearchresult','max_items':110,'max_depth':6,'timeout_sec':45})
            if not r.get('ok'):
                print(name, r, flush=True); break
            summary=r.get('summary',{})
            count=int(summary['count']);ids=summary.get('idlist',[])
            records.extend(ids);sources.extend(r.get('sources',[]));checked.extend(r.get('checked_sources',[]))
            if start+len(ids)>=count or not ids: break
            time.sleep(.4)
        logs.append({'id':name,'query':term,'count':count,'pmids':records,'sources':sources,'checked_sources':checked})
        (ROOT/'pubmed_search_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
        print(name, 'count=',count,'retrieved=',len(records),flush=True)
        time.sleep(.4)
    ids=sorted({p for q in logs for p in q['pmids']},reverse=True)
    (ROOT/'pubmed_ids.json').write_text(json.dumps(ids),encoding='utf-8')
    print('UNIQUE',len(ids),flush=True)

def fetch():
    ids=json.loads((ROOT/'pubmed_ids.json').read_text())
    output=ROOT/'pubmed_records.json'
    existing=json.loads(output.read_text()) if output.exists() else []
    done={x['pmid'] for x in existing}
    todo=[p for p in ids if p not in done]
    for i in range(0,len(todo),10):
        batch=todo[i:i+10]
        r=call({'endpoint':'efetch','params':{'db':'pubmed','id':','.join(batch),'retmode':'xml'},'response_format':'xml','max_items':40,'max_depth':15,'timeout_sec':45})
        if not r.get('ok'): print('FETCH_ERROR',batch,r,flush=True);continue
        articles=r.get('summary',{}).get('PubmedArticleSet',{}).get('PubmedArticle',[])
        if isinstance(articles,dict):articles=[articles]
        for a in articles:
            citation=a.get('MedlineCitation',{}); article=citation.get('Article',{})
            pmid=str(citation.get('PMID',''))
            if pmid:
                existing.append({'pmid':pmid,'title':article.get('ArticleTitle'),'journal':article.get('Journal'),'abstract':article.get('Abstract'),'publication_types':article.get('PublicationTypeList'),'mesh':citation.get('MeshHeadingList'),'identifiers':a.get('PubmedData',{}).get('ArticleIdList'),'authors':article.get('AuthorList'),'sources':[s for s in r.get('sources',[]) if s.get('supports_claim') and s.get('kind')=='evidence'],'checked_sources':r.get('checked_sources',[])})
                print(pmid,article.get('ArticleTitle'),flush=True)
        books=r.get('summary',{}).get('PubmedArticleSet',{}).get('PubmedBookArticle',[])
        if isinstance(books,dict): books=[books]
        for a in books:
            doc=a.get('BookDocument',{}); book=doc.get('Book',{})
            pmid=str(doc.get('PMID',''))
            if pmid:
                existing.append({'pmid':pmid,'title':book.get('BookTitle'),'journal':book.get('Publisher'),'abstract':doc.get('Abstract'),'publication_types':doc.get('PublicationType'),'identifiers':doc.get('ArticleIdList'),'authors':book.get('AuthorList'),'sources':r.get('sources',[]),'checked_sources':r.get('checked_sources',[])})
                print(pmid,book.get('BookTitle'),flush=True)
        output.write_text(json.dumps(existing,indent=2),encoding='utf-8')
        time.sleep(.4)

if __name__=='__main__':
    {'search':search,'fetch':fetch}[sys.argv[1]]()
