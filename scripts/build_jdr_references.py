"""Create transparent APA-style records from provider and Crossref metadata."""
import html
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/JDR_GBD_PARK'


def clean(text):
 return re.sub(r'\s+',' ',html.unescape(html.unescape(re.sub('<[^>]+>','',str(text))))).strip()


def initials(given):
 return ' '.join(p[0].upper()+'.' for p in re.split(r'[\s.]+',given.strip()) if p)


def author(a):
 return a.get('literal') or a.get('name') or (a.get('family','')+', '+initials(a.get('given',''))).strip(', ')


def main():
 used=set(re.findall(r'@([a-z_]+)',(OUT/'authoring/body.txt').read_text()))
 records={}
 for path in (OUT/'references').glob('*_crossref.json'):
  key=path.stem.replace('_crossref','')
  if key not in used:continue
  d=json.loads(path.read_text())
  year=d.get('published-print',d.get('published',{})).get('date-parts',[[None]])[0][0]
  records[key]={'authors':d.get('author',[]),'year':str(year),'title':clean(d['title'][0]),
      'journal':clean(d.get('container-title',[''])[0]),'volume':d.get('volume',''),
      'issue':d.get('issue','') if d.get('issue')!='none' else '',
      'pages':d.get('page') or ('Article '+d['article-number'] if d.get('article-number') else ''),
      'url':'https://doi.org/'+d['DOI'],'kind':'article','provenance':'Crossref DOI metadata; bibliographic corrections recorded below'}
 # Cite consortium bylines as in the source articles, rather than the expanded contributor list.
 for key,name in [('gbd_nonfatal','GBD 2023 Disease and Injury and Risk Factor Collaborators'),
                  ('gbd_fatal','GBD 2023 Causes of Death Collaborators'),
                  ('gbd_pd','GBD 2016 Parkinson’s Disease Collaborators')]:
  records[key]['authors']=[{'literal':name}]
 if 'gbd_demography' in used:
  d=json.loads((ROOT/'study_design/dataset_citations_2026-10-01/metadata/gbd_demography_crossref.json').read_text())
  d=d.get('message',d)
  records['gbd_demography']={'authors':[{'literal':'GBD 2023 Demographics Collaborators'}],
    'year':'2025','title':clean(d['title'][0]),'journal':clean(d['container-title'][0]),
    'volume':d.get('volume',''),'issue':d.get('issue',''),'pages':d.get('page',''),
    'url':'https://doi.org/'+d['DOI'],'kind':'article',
    'provenance':'Preserved Crossref record, verified source-methods article; not a separately acquired demographic dataset'}
 records['schrag']['authors']=[{'family':'Schrag','given':'A'},{'family':'Jahanshahi','given':'M'},{'family':'Quinn','given':'N'}]
 records['schrag']['provenance']='Crossref plus full author list verified in PubMed PMID 10945804'
 records['pooling']['pages']='1747–1782'
 records['shorrocks']['year']='2013'
 records['reconciliation']['year']='2019'
 records['arima']['title']='Automatic time series forecasting: The forecast package for R'
 records['transfer']['title']='A survey on transfer learning'
 records['reconciliation']['title']='Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization'
 records['gather']['title']='Guidelines for accurate and transparent health estimates reporting: The GATHER statement'
 records['norway']={'authors':[{'family':s,'given':g} for s,g in [('Brakedal','Brage'),('Toker','Lilah'),('Haugarvoll','Kristoffer'),('Tzoulis','Charalampos')]],
   'year':'2022','title':'A nationwide study of the incidence, prevalence and mortality of Parkinson’s disease in the Norwegian population',
   'journal':'npj Parkinson’s Disease','volume':'8','issue':'','pages':'Article 19','url':'https://doi.org/10.1038/s41531-022-00280-4',
   'kind':'article','provenance':'Full author list and article metadata verified on the primary Nature publisher page'}
 records['tcn']={'authors':[{'family':'Bai','given':'Shaojie'},{'family':'Kolter','given':'J Zico'},{'family':'Koltun','given':'Vladlen'}],
   'year':'2018','title':'An empirical evaluation of generic convolutional and recurrent networks for sequence modeling',
   'journal':'arXiv','volume':'','issue':'','pages':'Preprint arXiv:1803.01271','url':'https://doi.org/10.48550/arXiv.1803.01271',
   'kind':'preprint','provenance':'Primary arXiv record, DataCite DOI, preprint status explicit'}
 data=json.loads((ROOT/'study_design/dataset_citations_2026-10-01/references.json').read_text())
 ids={'gbd_results':'D1','hierarchy':'hierarchy','wpp':'D2','wpp_methods':'wpp_methods','gastat':'D4',
      'gccstat':'D3','moh_yearbook':'D5','chi':'chi_data','gastat_methods':'gastat_methods',
      'gastat_series':'gastat_series','moh_population':'moh_population'}
 for key,source_id in ids.items():
  if key not in used:continue
  d=next(x for x in data if x['id']==source_id)
  records[key]={'authors':d['authors'],'year':str(d.get('year') or 'n.d.'),'title':d['title'],
    'publisher':d.get('publisher',''),'url':('https://doi.org/'+d['doi']) if d.get('doi') else d['url'],
    'kind':d.get('kind','dataset'),'accessed':d.get('accessed','2026-10-01'),
    'provenance':'Preserved dataset citation package; reference year is not substituted for unverified publication date'}
 records['hierarchy']['authors']=[{'literal':'Global Burden of Disease Collaborative Network'}]
 records['hierarchy']['provenance']='Suggested citation in the provider Data Release Information Sheet, 23 October 2025; this explicit citation takes precedence over the DataCite creator field'
 records['gbd_results']['year']='2025'
 records['gbd_results']['accessed']='2026-10-03'
 records['gbd_results']['provenance']='Current IHME GBD FAQ retrieved directly on 3 October 2026 explicitly supplies the GBD 2023 citation year 2025; current provider guidance supersedes the retained export citation template. Original export citation is preserved unchanged.'
 records['wpp']['title']='World Population Prospects 2024, Online Edition'
 records['wpp']['provenance']='Suggested citation and CC BY 3.0 IGO notice verified in the retained UN population workbook'
 records['wpp_methods']['version']='UN DESA/POP/2024/DC/NO. 10, July 2024; Advance unedited version'
 records['wpp_methods']['provenance']='Exact report identifier and edition statement from provider suggested citation, page ii'
 records['gccstat']['publisher']='GCC-Statistical Centre, Sultanate of Oman'
 records['software']={'authors':[{'literal':'saifkhanbio'}],'year':'2026','title':'Parkinson-TCN-forecast',
    'publisher':'GitHub','version':'Source snapshot b333073e0305f6480e40a15167288c76a7a4e6fc',
    'url':'https://github.com/saifkhanbio/Parkinson-TCN-forecast/tree/b333073e0305f6480e40a15167288c76a7a4e6fc',
    'kind':'software','provenance':'Prior source audit matched all 135 published blobs; new reporting code is supplied in Supplementary Software S1'}
 records['dasgupta']={'authors':[{'family':'Das Gupta','given':'Prithwis'}],'year':'1993',
    'title':'Standardization and decomposition of rates: A user’s manual','publisher':'U.S. Bureau of the Census',
    'version':'Current Population Reports, P23-186','url':'https://www.census.gov/library/publications/1993/demo/p23-186.html',
    'kind':'report','provenance':'Official US Census author, report and publication date record'}
 assert set(records)==used,(set(records)-used,used-set(records))
 for r in records.values():
  r['authors']=[a for a in r['authors'] if a.get('literal') or a.get('family') or a.get('name')]
  assert r['authors']
  r['sort_author']=r['authors'][0].get('literal') or r['authors'][0].get('family') or r['authors'][0].get('name')
 groups={}
 for k,r in records.items():groups.setdefault((tuple(author(a) for a in r['authors']),r['year']),[]).append(k)
 for members in groups.values():
  for i,key in enumerate(sorted(members,key=lambda k:re.sub(r'[^\w\s]',' ',records[k]['title'].lower()).split())):
   records[key]['display_year']=records[key]['year']+(('-' if records[key]['year']=='n.d.' else '')+chr(97+i) if len(members)>1 else '')
 for key,r in records.items():
  names=[author(a) for a in r['authors']]
  if len(names)>20:authors=', '.join(names[:19])+', … '+names[-1]
  elif len(names)>1:authors=', '.join(names[:-1])+', & '+names[-1]
  else:authors=names[0]
  short=[a.get('literal') or a.get('family') or a.get('name') for a in r['authors']]
  r['cite']=(' & '.join(short) if len(short)<=2 else short[0]+' et al.')+', '+r['display_year']
  r['reference_prefix']=authors+('' if authors.endswith('.') else '.')+f" ({r['display_year']}). "
  if r['kind']=='article':
   r['reference_title']=r['title']+(' ' if r['title'].endswith(('.', '?', '!')) else '. ')
   r['reference_italic']=r['journal']+(', '+r['volume'] if r['volume'] else '')
   r['reference_suffix']=(f"({r['issue']})" if r['issue'] else '')+(', '+r['pages'].replace('-','–') if r['pages'] else '')+'. '+r['url']
  else:
   r['reference_title']='';r['reference_italic']=r['title']
   label={'dataset':'Data set','software':'Computer software','preprint':'Preprint'}.get(r['kind'])
   r['reference_suffix']=(' ('+r['version']+')' if r.get('version') else '')+(' ['+label+']' if label else '')+'. '
   publisher=r.get('publisher') or r.get('journal','')
   if publisher and publisher!=short[0]:r['reference_suffix']+=publisher+'. '
   if r['year']=='n.d.' or key=='gbd_results':
    from datetime import datetime
    accessed=datetime.strptime(r.get('accessed','2026-10-01'),'%Y-%m-%d')
    r['reference_suffix']+=f"Retrieved {accessed.strftime('%B')} {accessed.day}, {accessed.year}, from "
   r['reference_suffix']+=r['url']
  r['apa_text']=r['reference_prefix']+r['reference_title']+r['reference_italic']+r['reference_suffix']
 (OUT/'references/reference_register.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
 text='\n\n'.join(r['apa_text'] for r in sorted(records.values(),key=lambda r:(r['sort_author'].lower(),r['display_year'],r['title'])))
 (OUT/'references/references_APA7.txt').write_text(text+'\n')
 print('References prepared:',len(records))

if __name__=='__main__':main()
