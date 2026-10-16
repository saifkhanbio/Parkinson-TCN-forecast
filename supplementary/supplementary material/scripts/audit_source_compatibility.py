#!/usr/bin/env python3
"""Read-only source audit. New artifacts only; never fits or corrects a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET
import zipfile

from lxml import html
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/source_compatibility_v1'
METADATA = ROOT / 'supporting_data/2026-09-30_source_audit'
RAW = ROOT / 'supporting_data/2026-09-26/raw'
NATIVE = ROOT / 'More data/IHME-GBD_2023_DATA-e22a15d9-1/IHME-GBD_2023_DATA-e22a15d9-1.csv'
WORKBOOK = ROOT / 'age_standard/parkinsons_gbd2023_dataset.xlsx'
WIDE = ROOT / 'age_standard/parkinsons_ml_ready_wide.csv'
PANEL = ROOT / 'data/processed/design_v1/regional_outcomes.csv'
S = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
MEASURES = {1: 'deaths', 2: 'dalys', 3: 'ylds', 4: 'ylls', 5: 'prevalence', 6: 'incidence'}
ALIASES = {
    'Bolivia': 'Bolivia (Plurinational State of)', 'Brunei': 'Brunei Darussalam',
    'Congo (Brazzaville)': 'Congo', 'DR Congo': 'Democratic Republic of the Congo',
    'Federated States of Micronesia': 'Micronesia (Federated States of)',
    'Iran': 'Iran (Islamic Republic of)', 'Laos': "Lao People's Democratic Republic",
    'Moldova': 'Republic of Moldova', 'North Korea': "Democratic People's Republic of Korea",
    'Russia': 'Russian Federation', 'South Korea': 'Republic of Korea',
    'Syria': 'Syrian Arab Republic', 'São Tomé and Príncipe': 'Sao Tome and Principe',
    'Tanzania': 'United Republic of Tanzania', 'The Bahamas': 'Bahamas', 'The Gambia': 'Gambia',
    'UK': 'United Kingdom', 'USA': 'United States of America',
    'Venezuela': 'Venezuela (Bolivarian Republic of)', 'Virgin Islands': 'United States Virgin Islands',
}
GCC_ISO = {'SAU','BHR','KWT','OMN','QAT','ARE'}
QUANTILES = {'Lower 95':.025,'Lower 80':.1,'Median':.5,'Upper 80':.9,'Upper 95':.975}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def workbook(path):
    """Parse XML cells without requiring a new spreadsheet dependency."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            shared = [''.join(t.text or '' for t in e.findall('.//s:t', S))
                      for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        relationships = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        targets = {e.attrib['Id']: e.attrib['Target'] for e in relationships}
        sheets = ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', S)
        for sheet in sheets:
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = targets[rid]
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            with z.open(target) as stream:
                for _, row in ET.iterparse(stream, events=['end']):
                    if row.tag != '{' + S['s'] + '}row':
                        continue
                    values = {}
                    for cell in row:
                        address = cell.attrib['r']
                        value = cell.find('s:v', S)
                        if cell.find('s:f', S) is not None:
                            raise ValueError('Unexpected formula in input workbook: ' + address)
                        if cell.attrib.get('t') == 's':
                            value = shared[int(value.text)]
                        elif value is not None:
                            value = value.text
                        else:
                            value = ''.join(t.text or '' for t in cell.findall('.//s:t', S))
                        values[re.sub(r'\d', '', address)] = value
                    rows.append(values)
                    row.clear()
            yield sheet.attrib['name'], rows


def rows_frame(rows):
    names = rows[0]
    return pd.DataFrame([{names[k]: v for k, v in row.items() if k in names} for row in rows[1:]])


def un_age_quantiles(path, sex):
    """Extract source marginal quantiles without adding ages, sexes or quantiles."""
    result=[]
    with zipfile.ZipFile(path) as z:
        shared=[''.join(t.text or '' for t in e.findall('.//s:t',S))
                for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        targets={e.get('Id'):e.get('Target') for e in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets=ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets',S)
        names=[s.get('name') for s in sheets]
        if set(names)!=set(QUANTILES)|{'NOTES'}:
            raise ValueError('Unexpected UN age-quantile schema')
        def value(cell):
            v=cell.find('s:v',S)
            if v is None:return ''.join(t.text or '' for t in cell.findall('.//s:t',S))
            return shared[int(v.text)] if cell.get('t')=='s' else v.text
        for sheet in sheets:
            if sheet.get('name') not in QUANTILES:continue
            target=targets[sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
            target=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            header=None
            with z.open(target) as stream:
                for _,row in ET.iterparse(stream,events=['end']):
                    if row.tag!='{'+S['s']+'}row':continue
                    index=int(row.get('r'))
                    if index==17:
                        header={re.sub(r'\d','',c.get('r')):value(c) for c in row}
                        if header.get('F')!='ISO3 Alpha-code' or header.get('K')!='Year' or header.get('AF')!='100+':
                            raise ValueError('UN age headers changed')
                    if index>17:
                        cells={re.sub(r'\d','',c.get('r')):c for c in row}
                        iso=value(cells['F']) if 'F' in cells else ''
                        if iso in GCC_ISO and 'K' in cells and 2024<=int(float(value(cells['K'])))<=2028:
                            for col,age in header.items():
                                if not re.fullmatch(r'\d+-\d+|100\+',age):continue
                                if int(age.split('-')[0].rstrip('+'))<45:continue
                                result.append({'iso3':iso,'location_name':value(cells['C']),'sex':sex,
                                               'year':int(float(value(cells['K']))),'source_age_group':age,
                                               'quantile':QUANTILES[sheet.get('name')],
                                               'population_persons':float(value(cells[col]))*1000,
                                               'source_population_thousands':float(value(cells[col])),
                                               'source_file':rel(path),'source_sheet':sheet.get('name'),
                                               'source_cell':cells[col].get('r'),
                                               'scope':'marginal_age_sex_year_quantile_not_joint_draw'})
                    row.clear()
    return pd.DataFrame(result),names


def protected():
    snapshot = json.loads((ROOT / 'work/global-asr-cpu-recovery-validation/preserved_sha256.json').read_text())['sha256']
    lock_path = ROOT / 'study_design/locked_v1/lock_manifest.json'
    lock = json.loads(lock_path.read_text())
    for name in ['source_sha256', 'design_sha256', 'output_sha256']:
        snapshot.update(lock[name])
    for path in [ROOT / 'AGENTS.md', lock_path,
                 ROOT / 'reports/study_synthesis_v1_1/report.md',
                 ROOT / 'results/global_asr_cpu_recovery_v1/run_manifest.json',
                 ROOT / 'reports/global_asr_cpu_recovery_v1/report.md']:
        snapshot.setdefault(rel(path), sha(path))
    for name, expected in snapshot.items():
        if sha(ROOT / name) != expected:
            raise ValueError('Previously protected identity mismatch: ' + name)
    return snapshot


def audit(out):
    if out.exists():
        raise SystemExit('Refusing to overwrite existing audit; use --verify-only or an empty --output')
    preserved = protected()
    inputs = {NATIVE, NATIVE.parent / 'citation.txt', WORKBOOK, WIDE, PANEL, Path(__file__),
              ROOT / 'supporting_data/2026-09-26/manifest.json',
              ROOT / 'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv',
              ROOT / 'data/processed/design_v1/un_population_1990_2028.csv',
              ROOT / 'results/release_sensitivity_v1/matched_points.csv',
              ROOT / 'results/projections_v1/trials/SAU_prevalence/joint_draws.csv.gz'}
    inputs.update(METADATA.glob('manifest*.json'))
    inputs.update((METADATA / 'raw').glob('*'))
    inputs.update(RAW.glob('*'))
    source_hashes = {rel(p): sha(p) for p in sorted(inputs)}
    out.mkdir(parents=True)
    checks = []
    def check(name, ok, details):
        checks.append({'check': name, 'status': 'pass' if bool(ok) else 'fail', 'details': str(details)})
    def save(name, frame):
        frame.to_csv(out / name, index=False)

    manifest = json.loads((ROOT / 'supporting_data/2026-09-26/manifest.json').read_text())
    file_checks = []
    for item in manifest:
        path = RAW / item['filename']
        file_checks.append({'file': rel(path), 'bytes': path.stat().st_size,
                            'hash_matches_original': sha(path) == item['sha256'],
                            'size_matches_original': path.stat().st_size == item['bytes']})
    save('preserved_raw_files.csv', pd.DataFrame(file_checks))
    check('18_preserved_supporting_downloads', all(x['hash_matches_original'] and x['size_matches_original'] for x in file_checks), len(file_checks))

    n, panel, wide = read(NATIVE), read(PANEL), read(WIDE)
    identifiers = []
    for name in ['population_group', 'measure', 'location', 'sex', 'age', 'cause', 'metric']:
        pairs = n[[name + '_id', name + '_name']].drop_duplicates()
        check(name + '_identifier_bijection', not pairs.iloc[:, 0].duplicated().any() and not pairs.iloc[:, 1].duplicated().any(), len(pairs))
        identifiers.extend({'dimension': name, 'id': int(a), 'name': b, 'authority': 'observed in supplied export; official codebook not acquired'} for a, b in pairs.itertuples(index=False, name=None))
    save('observed_native_identifiers.csv', pd.DataFrame(identifiers))
    key = ['population_group_id', 'measure_id', 'location_id', 'sex_id', 'age_id', 'cause_id', 'metric_id', 'year']
    check('native_unique_keys', not n.duplicated(key).any(), len(n))
    check('native_positive_finite_ordered_bounds', np.isfinite(n[['val','lower','upper']]).all().all() and n.lower.gt(0).all() and n.lower.le(n.val).all() and n.val.le(n.upper).all(), 'point and source bounds, not predictive intervals')
    coverage = n.groupby(['measure_id','measure_name','metric_id','metric_name']).year.agg(['min','max','nunique','size']).reset_index()
    save('native_coverage.csv', coverage)
    check('native_scope_and_complete_grid', len(n)==68992 and len(n.location_id.unique())==7 and len(n.age_id.unique())==11 and all(len(g)==7*2*11*(44 if m in [1,4] else 34) for (m,_),g in n.groupby(['measure_id','metric_id'])), '6 outcomes; 2 sexes; 11 ages; 1980–2023 deaths/YLLs and 1990–2023 other outcomes')
    n['outcome'] = n.measure_id.map(MEASURES)
    n['sex'], n['age'] = n.sex_name, n.age_name.str.replace(' years','',regex=False)
    keys = ['location_id','location_name','sex','age','outcome','year']
    joined = panel.copy()
    for metric_id, label in [(1,'count'),(3,'rate')]:
        raw = n[n.metric_id.eq(metric_id)][keys+['val','lower','upper']].rename(columns={'val': label+'_raw','lower':label+'_lower_raw','upper':label+'_upper_raw'})
        j = joined.merge(raw,on=keys,how='outer',validate='one_to_one',indicator=True)
        values = [label,label+'_lower',label+'_upper']
        check('processed_' + label + '_raw_replay', j._merge.eq('both').all() and all(np.allclose(j[c],j[c+'_raw'],rtol=1e-13,atol=1e-12) for c in values), f'{len(j)} rows; all points and bounds')
    counts = n[n.metric_id.eq(1)].set_index(keys)
    rates = n[n.metric_id.eq(3)].set_index(keys)
    den = counts[['val','lower','upper']]/rates[['val','lower','upper']]*100000
    den = den.rename(columns={k:'population_from_'+k for k in ['val','lower','upper']}).reset_index()
    den['max_bound_ratio_difference'] = (den[['population_from_lower','population_from_upper']].div(den.population_from_val,axis=0)-1).abs().max(axis=1)
    spread = den.groupby(['location_name','sex','age','year']).population_from_val.agg(['min','median','max']).reset_index()
    spread['relative_spread'] = (spread['max']-spread['min'])/spread['median']
    save('native_denominator_consistency.csv', spread)
    check('denominators_agree_across_outcomes', spread.relative_spread.max()<1e-9, 'max relative spread '+str(spread.relative_spread.max()))
    check('number_rate_bounds_share_denominator', den.max_bound_ratio_difference.max()<1e-9, 'max relative difference '+str(den.max_bound_ratio_difference.max())+'; does not identify joint draw dependence')
    common = panel[panel.year.ge(1990)].pivot(index=['location_name','sex','age','year'],columns='outcome',values=['count','rate'])
    for metric in ['count','rate']:
        d = common[metric].dalys-common[metric].ylds-common[metric].ylls
        check('native_daly_identity_'+metric, np.allclose(d,0,atol=1e-8,rtol=0), 'max absolute discrepancy '+str(abs(d).max()))

    sheets = dict(workbook(WORKBOOK))
    dump(out/'workbook_metadata.json', {'README':[list(x.values()) for x in sheets['README']], 'Data_Dictionary':[list(x.values()) for x in sheets['Data_Dictionary']], 'acquisition_date':'not established by workbook creation timestamp'})
    raw_wb, wide_wb = rows_frame(sheets['Raw_Data']), rows_frame(sheets['ML_Ready_Wide'])
    for d in [raw_wb,wide_wb]:
        d['location_id'],d['year'] = d.location_id.astype(int), d.year.astype(int)
    wbkey=['location_id','location_name','year']
    j=wide.merge(wide_wb,on=wbkey,how='outer',suffixes=('_csv','_wb'),validate='one_to_one',indicator=True)
    numeric=[c for c in wide if c not in wbkey]
    check('csv_replays_workbook_wide', j._merge.eq('both').all() and all(np.allclose(j[c+'_csv'].astype(float),j[c+'_wb'].astype(float),rtol=1e-13,atol=1e-10) for c in numeric), f'{len(j)} rows, {len(numeric)*len(j)} numeric cells')
    all_raw=[]
    for outcome, part in raw_wb.groupby('measure'):
        renamed=part[wbkey+[c for c in raw_wb if c.startswith(('count_','rate_')) and '_upper_' not in c and '_lower_' not in c]].rename(columns={c:outcome+'_'+c for c in raw_wb if c not in wbkey})
        j=wide.merge(renamed,on=wbkey,how='outer',suffixes=('_csv','_raw'),validate='one_to_one',indicator=True)
        cs=[c for c in renamed if c not in wbkey]
        check('csv_replays_workbook_raw_'+outcome,j._merge.eq('both').all() and all(np.allclose(j[c+'_csv'],j[c+'_raw'].astype(float),rtol=1e-13,atol=1e-10) for c in cs), f'{len(j)} rows, {len(cs)*len(j)} numeric cells')
        for sex in ['male','female','combined']:
            cols=['count_'+sex,'rate_'+sex,'rate_age_std_'+sex]
            a=part[wbkey+cols].copy()
            for c in cols:a[c]=a[c].astype(float)
            a=a.rename(columns={cols[0]:'count',cols[1]:'crude_rate',cols[2]:'asr'})
            a['outcome'],a['sex']=outcome,sex
            all_raw.append(a)
    wb=pd.concat(all_raw,ignore_index=True)
    check('workbook_structural_scope', len(raw_wb)==41412 and len(wide)==6902 and wide.location_id.nunique()==203 and raw_wb.cause_id.eq('544').all(), 'prepared all-age and full-age ASR fields; full-age component rates not supplied')
    totals=panel[panel.year.ge(1990)].groupby(['location_name','year','outcome','sex'])['count'].sum().rename('native_45plus').reset_index()
    totals.sex=totals.sex.str.lower()
    scope=totals.merge(wb[['location_name','year','outcome','sex','count']],on=['location_name','year','outcome','sex'],validate='one_to_one').rename(columns={'count':'workbook_all_ages'})
    scope['excess_over_all_age_display']=scope.native_45plus-scope.workbook_all_ages
    scope['violates_under_nearest_integer_rounding']=scope.excess_over_all_age_display.gt(.5000001)
    save('native_45plus_vs_workbook_all_ages.csv',scope)
    check('age_subset_not_above_all_age_rounding_envelope', not scope.violates_under_nearest_integer_rounding.any(), f'{len(scope)} comparisons; +/-0.5 is an assumed nearest-integer envelope, not confirmed export rounding')
    add=[]
    for sex in ['male','female','combined']:
        for quantity,tolerance in [('count',1.5000001),('rate',.015000001),('rate_age_std',.015000001)]:
            delta=wide['dalys_'+quantity+'_'+sex]-wide['ylds_'+quantity+'_'+sex]-wide['ylls_'+quantity+'_'+sex]
            add.append({'identity':'DALYs=YLDs+YLLs','sex':sex,'quantity':quantity,'max_abs_difference':abs(delta).max(),'outside_assumed_rounding_envelope':int((abs(delta)>tolerance).sum()),'envelope':tolerance})
    save('workbook_component_identities.csv',pd.DataFrame(add))
    check('workbook_daly_rounding_identities',not any(x['outside_assumed_rounding_envelope'] for x in add),'nearest integer/0.01 grid sensitivity, not verified rounding rules')
    rounding=[]
    for sex in ['male','female','combined']:
        lower,upper=[],[]
        for outcome in MEASURES.values():
            count=wide[outcome+'_count_'+sex]
            rate=wide[outcome+'_rate_'+sex]
            lower.append(np.maximum(count-.5,0)/(rate+.005)*100000)
            upper.append(np.where(rate>.005,(count+.5)/(rate-.005)*100000,np.inf))
        part=wide[wbkey].copy()
        part['sex']=sex
        part['intersection_lower_population']=np.max(lower,axis=0)
        part['intersection_upper_population']=np.min(upper,axis=0)
        part['intersection_nonempty']=part.intersection_lower_population.le(part.intersection_upper_population+1e-6)
        rounding.append(part)
    rounding=pd.concat(rounding,ignore_index=True)
    save('workbook_population_rounding_intersections.csv',rounding)
    check('workbook_common_crude_denominator_under_rounding',rounding.intersection_nonempty.all(),f'{len(rounding)} location-year-sex comparisons; conditional on assumed +/-0.5 counts and +/-0.005 rates')
    sex_difference=max(float(abs(wide[m+'_count_male']+wide[m+'_count_female']-wide[m+'_count_combined']).max()) for m in MEASURES.values())
    check('workbook_both_sex_count_additivity_under_rounding',sex_difference<=1.5000001,'max absolute count difference '+str(sex_difference))

    article=html.parse(str(METADATA/'raw/gbd2023_demographics.html'))
    table=article.find('.//table')
    national=[]
    for position,row in enumerate(table.findall('.//tr'),1):
        cells=list(row)
        if len(cells)>2 and cells[0].tag=='td' and not cells[0].text_content().strip() and cells[1].get('colspan')=='2':
            for sup in cells[1].findall('.//sup'):sup.getparent().remove(sup)
            label=cells[1].text_content().strip()
            national.append({'published_name':label,'matched_name':ALIASES.get(label,label),'table_row':position,'alias_used':label in ALIASES})
    national=pd.DataFrame(national)
    check('primary_publication_204_unique_national_locations',len(national)==204 and national.matched_name.nunique()==204,'GBD 2023 Demographics, first table; national row indentation parsed explicitly')
    registry=wide[['location_id','location_name']].drop_duplicates()
    cross=national.merge(registry,left_on='matched_name',right_on='location_name',how='outer',validate='one_to_one',indicator=True)
    cross['status']=cross._merge.map({'both':'matched_national_reference','left_only':'national_reference_absent_from_workbook','right_only':'workbook_location_outside_national_reference'}).astype(str)
    cross['evidence_url']='https://pmc.ncbi.nlm.nih.gov/articles/PMC12535839/'
    save('published_national_location_crosswalk.csv',cross.drop(columns='_merge'))
    missing=cross[cross.status.eq('national_reference_absent_from_workbook')].matched_name.tolist()
    additional=cross[cross.status.eq('workbook_location_outside_national_reference')].location_name.tolist()
    check('workbook_claim_203_of_204_national_reference',len(missing)==1 and not additional, json.dumps({'missing':missing,'additional':additional}))

    un=read(ROOT/'data/processed/design_v1/un_population_1990_2028.csv')
    gastat=read(ROOT/'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv')
    nat=panel[panel.location_name.eq('Saudi Arabia') & panel.year.eq(2023) & panel.outcome.eq('prevalence')][['sex','age','age_start','implied_population']].copy()
    nat['age_group']=np.where(nat.age_start.ge(80),'80+',nat.age)
    nat=nat.groupby(['sex','age_group']).implied_population.sum().reset_index()
    u=un[un.location_name.eq('Saudi Arabia') & un.year.eq(2023)].copy()
    u['age_group']=np.where(u.age.str.extract(r'(\d+)',expand=False).astype(int).ge(80),'80+',u.age)
    u=u.groupby(['sex','age_group']).population_persons.sum().reset_index().rename(columns={'population_persons':'un_population'})
    g=gastat[gastat.year.eq(2023)&gastat.nationality.eq('Total')&gastat.sex.isin(['Male','Female'])&gastat.age_start.ge(45)][['sex','age_group','population_persons']].rename(columns={'population_persons':'gastat_population'})
    pop=nat.merge(u,on=['sex','age_group'],validate='one_to_one').merge(g,on=['sex','age_group'],validate='one_to_one')
    check('saudi_population_comparison_complete',len(pop)==16,'8 common bands per sex; 80+ retained intact; source dates/coverage not assumed identical')
    for source in ['un','gastat']:pop['gbd_vs_'+source+'_pct']=(pop.implied_population/pop[source+'_population']-1)*100
    save('saudi_2023_population_comparison.csv',pop)

    catalog=json.loads((METADATA/'raw/un_wpp2024_downloads.json').read_text())
    products=[]
    def walk(obj,context=()):
        if isinstance(obj,dict):
            context=context+tuple(str(obj[k]) for k in ['display','name'] if k in obj)
            if 'Path' in obj:
                products.append({'title':obj.get('Title',''),'format':obj.get('type',''),'path':obj['Path'],'context':' / '.join(context),'url':urljoin('https://population.un.org/wpp/',quote(obj['Path'],safe='/()'))})
            for value in obj.values():walk(value,context)
        elif isinstance(obj,list):
            for value in obj:walk(value,context)
    walk(catalog)
    products=pd.DataFrame(products)
    probabilistic=products[products.context.str.contains('Probabilistic Projections',regex=False)].drop_duplicates('path')
    probabilistic=probabilistic.copy()
    probabilistic['product_role']=np.where(probabilistic.path.str.contains('2024',regex=False),'current_2024_summary_product','older_archive_product')
    current_product_count=int(probabilistic.product_role.eq('current_2024_summary_product').sum())
    save('un_probabilistic_product_inventory.csv',probabilistic)
    prob_sheets=dict(workbook(METADATA/'raw/un_ppp2024_total_population.xlsx'))
    dump(out/'un_uncertainty_schema.json',{'inspected_file':'un_ppp2024_total_population.xlsx','sheets':list(prob_sheets),'has_joint_draw_identifiers':False,'interpretation':'inspected total-population workbook contains marginal quantile sheets; age/sex products listed separately; no individual trajectories found in examined catalog','catalog_current_probabilistic_products':current_product_count,'catalog_older_archive_products':len(probabilistic)-current_product_count})
    check('un_probability_workbook_is_marginal_quantiles',set(prob_sheets)=={'Lower 95','Lower 80','Median','Upper 80','Upper 95','NOTES'},'Not a joint trajectory sample; central paths cannot be added to form joint quantiles')
    age_quantiles=[]
    for sex in ['Female','Male']:
        path=METADATA/'raw'/('un_ppp2024_age_'+sex.lower()+'.xlsx')
        q,names=un_age_quantiles(path,sex)
        age_quantiles.append(q)
    age_quantiles=pd.concat(age_quantiles,ignore_index=True)
    qkey=['iso3','sex','year','source_age_group']
    pivot=age_quantiles.pivot(index=qkey,columns='quantile',values='population_persons').sort_index(axis=1)
    check('un_age_marginals_complete_ordered_gcc_2024_2028', len(age_quantiles)==3600 and len(pivot)==720 and not pivot.isna().any().any() and (np.diff(pivot.to_numpy(),axis=1)>=0).all(), '6 GCC x 2 sexes x 5 years x 12 retained source ages x 5 quantiles; 95–99 and 100+ kept separate')
    save('un_gcc_age_sex_marginal_quantiles_2024_2028.csv',age_quantiles)
    dump(out/'un_age_uncertainty_schema.json',{'files':['un_ppp2024_age_female.xlsx','un_ppp2024_age_male.xlsx'],'sheets':names,'source_age_groups':sorted(age_quantiles.source_age_group.unique(),key=lambda x:int(x.split('-')[0].rstrip('+'))),'source_unit':'thousands','prepared_unit':'persons','has_joint_draw_identifiers':False,'open_age_95plus_directly_available':False,'warning':'Do not add 95–99 and 100+ quantiles and label their sum a 95+ quantile; joint draws or an appropriate source aggregate are required.'})
    # The saved forecasts are empirical residual-block draws, not IHME source draws.
    draw_path=ROOT/'results/projections_v1/trials/SAU_prevalence/joint_draws.csv.gz'
    draws=read(draw_path)
    dump(out/'forecast_draw_schema.json',{'file':rel(draw_path),'columns':list(draws),'rows':len(draws),'residual_origin_values':sorted(int(x) for x in draws.residual_origin.unique()),'gbd_estimation_draws':False})

    save('checks.csv',pd.DataFrame(checks))
    summary={'native_rows':len(n),'processed_rows':len(panel),'workbook_rows':len(wide),'raw_workbook_rows':len(raw_wb),'national_locations_matched':int(cross.status.eq('matched_national_reference').sum()),'national_locations_missing':missing,'additional_workbook_locations':additional,'source_authentication_complete':False,'asr_reconstruction_possible':False,'full_age_component_rates_available':False,'numeric_standard_weights_verified':False,'joint_gbd_draws_available_locally':False,'joint_un_draws_verified':False,'gbd_marginal_bounds_available':True,'un_current_probabilistic_products_listed':current_product_count,'un_older_archives_listed':len(probabilistic)-current_product_count,'saudi_80plus_populations':pop[pop.age_group.eq('80+')].to_dict('records'),'checks_passed':sum(x['status']=='pass' for x in checks),'checks_failed':sum(x['status']=='fail' for x in checks),'models_fitted':0,'decision':'source issues remain; no model expansion or uncertainty guarantee'}
    dump(out/'summary.json',summary)
    for name,expected in {**preserved,**source_hashes}.items():
        if sha(ROOT/name)!=expected:raise ValueError('Input changed during audit: '+name)
    outputs={rel(p):sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    dump(out/'validation.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,'audit_completed':True,'scientific_source_compatibility_passed':False,'source_sha256':source_hashes,'protected_sha256':preserved,'output_sha256':outputs,'raw_and_locked_files_unchanged':True,'models_fitted':0})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to(ROOT):raise SystemExit('Audit output must remain inside repository')
    if args.verify_only:
        report=json.loads((out/'validation.json').read_text())
        for group in ['source_sha256','protected_sha256','output_sha256']:
            for name,expected in report[group].items():
                if sha(ROOT/name)!=expected:raise ValueError('Identity mismatch: '+name)
        print('Audit source, protected, and output identities verified; this does not authenticate extraction.')
    else:
        audit(out)
