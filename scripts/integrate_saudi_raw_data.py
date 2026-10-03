"""Integrate Saudi raw demographic and insurance data as exploratory evidence.

No disease model is fitted. Frozen rate forecasts and original inputs are read-only.
Requires numpy, pandas, matplotlib and python-docx in the study environment.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/gbd_raw_integration_mpl')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/saudi_raw_integration_v1'
DATA = ROOT / 'data/processed/saudi_raw_integration_v1'
NS = {'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
O = {'t':'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
     'o':'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
     'p':'urn:oasis:names:tc:opendocument:xmlns:text:1.0'}
INPUTS = {}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def track(path):
    p=ROOT/path if not Path(path).is_absolute() else Path(path)
    INPUTS[p.relative_to(ROOT).as_posix()]=sha(p)
    return p


def read(path, **kwargs):
    return pd.read_csv(track(path), **kwargs)


def save(frame, name, data=False):
    frame.to_csv((DATA if data else OUT)/name,index=False,encoding='utf-8-sig')


def xlsx_sheets(path):
    """Read cached values only; preserve cells and sheet names for provenance."""
    with ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',NS)]
        rels={e.get('Id'):e.get('Target') for e in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        result={}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet',NS):
            target=rels[sheet.get('{'+NS['r']+'}id')]
            member=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            rows=[]
            for row in ET.fromstring(z.read(member)).findall('s:sheetData/s:row',NS):
                cells={}
                for c in row.findall('s:c',NS):
                    v=c.findtext('s:v','',NS)
                    if c.get('t')=='s' and v:v=strings[int(v)]
                    if c.get('t')=='inlineStr':v=''.join(c.find('s:is',NS).itertext())
                    if v!='':cells[re.sub(r'\d','',c.get('r'))]=v
                if cells:rows.append((int(row.get('r')),cells))
            result[sheet.get('name')]=rows
        return result


def ods_sheets(path):
    with ZipFile(path) as z: root=ET.fromstring(z.read('content.xml'))
    out={}
    for sheet in root.findall('.//t:table',O):
        rows=[];rn=0
        for row in sheet.findall('t:table-row',O):
            repeat=int(row.get('{'+O['t']+'}number-rows-repeated','1'))
            cells=[]
            for c in row:
                n=int(c.get('{'+O['t']+'}number-columns-repeated','1'))
                value=c.get('{'+O['o']+'}value') or ''.join(c.itertext()).strip()
                if n>50 and not value:break
                assert n<100
                cells.extend([value]*n)
            if any(cells):
                assert repeat<10
                for _ in range(repeat):
                    rn+=1;rows.append((rn,{chr(65+i):v for i,v in enumerate(cells) if v!=''}))
            else:rn+=repeat
        out[sheet.get('{'+O['t']+'}name')]=rows
    return out


def national_population():
    raw=read('data/raw/moh/Population by Nationality, Gender, and Age groups, 2022G.csv')
    raw['age']=raw['Age Group'].str.strip().str.lstrip("'").replace({'05-9':'5-9'})
    raw['age_start']=raw.age.str.extract(r'^(\d+)').astype(int)
    raw['sex']=raw.Variable2.str.strip().map({'MALE':'Male','FEMALE':'Female'})
    raw['nationality']=raw.Variable1.str.strip().map({'Saudi':'Citizens','NonSaudi':'Non-citizens'})
    raw['population']=pd.to_numeric(raw.Variable_Value)
    raw['year']=2022
    assert len(raw)==68 and raw.population.sum()==32175224
    assert not raw[['sex','nationality']].isna().any().any()
    table=xlsx_sheets(track('supporting_data/2026-09-26/raw/saudi_moh_yearbook_2023.xlsx'))
    matches=[(name,rows) for name,rows in table.items() if any('Table 1-3' in v for _,r in rows for v in r.values())]
    assert len(matches)==1
    name,rows=matches[0]
    by_age={r.get('A','').replace('05-9','5-9'):(rn,r) for rn,r in rows}
    cols={('Citizens','Male'):'B',('Citizens','Female'):'C',('Non-citizens','Male'):'E',('Non-citizens','Female'):'F'}
    trace=[]
    for r in raw.itertuples():
        rn,cells=by_age[r.age];col=cols[(r.nationality,r.sex)]
        assert int(cells[col])==r.population
        trace.append(dict(age=r.age,sex=r.sex,nationality=r.nationality,population=r.population,sheet=name,cell=col+str(rn),matches=True))
    save(pd.DataFrame(trace),'moh_2022_source_cells.csv')
    raw['reference_id']='moh_population;D5'
    raw['reference_date']='2022 reference year; exact date not established from export'
    fields=['year','sex','nationality','age','age_start','population','reference_id','reference_date']
    nat=raw[fields].copy()
    gastat=read('supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv')
    g=gastat.loc[gastat.sex.isin(['Male','Female']) & gastat.nationality.isin(['Citizens','Non-citizens']) & gastat.age_start.notna()].copy()
    g=g.rename(columns={'age_group':'age','population_persons':'population'})
    g['reference_id']='D4';g['reference_date']='1 July; GASTAT population methods'
    nat=pd.concat([nat,g[fields]],ignore_index=True)
    nat.age_start=nat.age_start.astype(int)
    assert len(nat)==204 and not nat.duplicated(['year','sex','nationality','age']).any()
    save(nat,'national_population_2022_2024.csv',data=True)
    # This new GASTAT export duplicates the older GCC-Stat series exactly.
    broad=read('data/raw/gastat/Population estimates by gender, nationality, and age group 2010 - 2024_data.csv')
    broad['age']=broad['Age Groups'].replace({'4/1/00':'0-4','5/9/01':'5-9','10/14/01':'10-14'})
    broad['year']=broad.Year.str.replace(',','',regex=False).astype(int)
    broad['nationality']=broad.Nationality.map({'Saudi':'Citizens','Non-Saudi':'Non-citizens'})
    broad['sex']=broad.Gender
    broad['population']=broad['Population estimates'].str.replace(',','',regex=False).astype(int)
    gcc=read('supporting_data/2026-09-26/processed/gccstat_saudi_population_age_sex_nationality_2010_2024.csv')
    gcc=gcc.rename(columns={'AGE':'age','TIME_PERIOD':'year'})
    old=broad.merge(gcc[['year','sex','nationality','age','population_persons_original']],
                    on=['year','sex','nationality','age'],how='left',validate='one_to_one')
    old['difference']=old.population-old.population_persons_original
    assert len(broad)==len(old)==840 and old.difference.eq(0).all()
    assert broad['Age Groups'].isin(['4/1/00','5/9/01','10/14/01']).sum()==180
    save(old,'gastat_840_cell_duplicate_check.csv')
    # Preserve the previously independently verified label issue in a new usable linkage table.
    age_issue=read('reports/raw_download_review_2026-10-01/moh_existing_population_comparison.csv')
    save(age_issue,'moh_gccstat_age_key_audit.csv')
    # Export a corrected linkage for 2022, leaving the historical GCC file intact.
    linkage=raw.merge(gcc.loc[gcc.year.eq(2022),['year','sex','nationality','age','population_persons_original']],
                      on=['year','sex','nationality','age'],how='left',validate='one_to_one')
    linkage['difference']=linkage.population-linkage.population_persons_original
    linkage['source_flag']=np.where(linkage.population_persons_original.isna(),'GCC age key absent',
        np.where(linkage.difference.eq(0),'exact age-cell match','GCC 65-69 value equals 65+ total'))
    bad=linkage.loc[linkage.difference.ne(0)&linkage.population_persons_original.notna()]
    assert len(bad)==4 and bad.age.eq('65-69').all()
    for r in bad.itertuples():
        total=raw.loc[raw.age_start.ge(65)&raw.sex.eq(r.sex)&raw.nationality.eq(r.nationality),'population'].sum()
        assert total==r.population_persons_original
    save(linkage[fields+['population_persons_original','difference','source_flag']], 'population_2022_corrected_linkage.csv',data=True)
    return nat


def population_and_counts(nat,gbd,un):
    national=nat.groupby(['year','sex','age','age_start'],as_index=False).population.sum()
    sa=gbd.loc[gbd.location_name.eq('Saudi Arabia') & gbd.year.isin([2022,2023])].copy()
    sa['broad_age']=np.where(sa.age_start.ge(80),'80+',sa.age)
    base=sa.loc[sa.outcome.eq('prevalence')].copy()
    pop=base.groupby(['year','sex','broad_age'],as_index=False).implied_population.sum().rename(columns={'broad_age':'age','implied_population':'gbd_population'})
    u=un.loc[un.location_name.eq('Saudi Arabia') & un.year.isin([2022,2023])].copy()
    u['age']=np.where(u.age_start.ge(80),'80+',u.age)
    u=u.groupby(['year','sex','age'],as_index=False).population_persons.sum().rename(columns={'population_persons':'un_population'})
    pop=pop.merge(national[national.age_start.ge(45)],on=['year','sex','age'],validate='one_to_one')
    pop=pop.merge(u,on=['year','sex','age'],validate='one_to_one').rename(columns={'population':'national_population'})
    assert len(pop)==32
    for provider in ['gbd','un']:
        pop[provider+'_percent_difference_from_national']=100*(pop[provider+'_population']/pop.national_population-1)
    save(pop,'population_comparison_2022_2023.csv')
    grouped=sa.groupby(['year','sex','outcome','broad_age'],as_index=False).agg(gbd_population=('implied_population','sum'),native_count=('count','sum'),rate_min=('rate','min'),rate_max=('rate','max'))
    grouped=grouped.rename(columns={'broad_age':'age'}).merge(pop[['year','sex','age','national_population']],on=['year','sex','age'],validate='many_to_one')
    grouped['gbd_weighted_rate']=1e5*grouped.native_count/grouped.gbd_population
    grouped['national_count_reference']=grouped.gbd_weighted_rate*grouped.national_population/1e5
    grouped['allocation_count_min']=grouped.rate_min*grouped.national_population/1e5
    grouped['allocation_count_max']=grouped.rate_max*grouped.national_population/1e5
    assert len(grouped)==192
    assert (grouped.national_count_reference>=grouped.allocation_count_min-1e-7).all()
    assert (grouped.national_count_reference<=grouped.allocation_count_max+1e-7).all()
    save(grouped,'national_denominator_count_sensitivity_cells.csv')
    totals=[]
    for (year,sex,outcome),part in grouped.groupby(['year','sex','outcome']):
        oldest=part.loc[part.age.eq('80+')].iloc[0]
        native=part.native_count.sum();alternative=part.national_count_reference.sum()
        younger=part.loc[part.age.ne('80+'),'national_count_reference'].sum()
        totals.append(dict(year=year,sex=sex,outcome=outcome,native_count_45plus=native,national_count_45plus=alternative,
            count_change_percent=100*(alternative/native-1),native_80plus_share=100*oldest.native_count/native,
            national_80plus_share=100*oldest.national_count_reference/alternative,
            allocation_total_min=younger+oldest.allocation_count_min,allocation_total_max=younger+oldest.allocation_count_max,
            allocation_80plus_share_min=100*oldest.allocation_count_min/(younger+oldest.allocation_count_min),
            allocation_80plus_share_max=100*oldest.allocation_count_max/(younger+oldest.allocation_count_max)))
    totals=pd.DataFrame(totals);save(totals,'national_denominator_count_sensitivity_summary.csv')
    return pop,totals


def projections(nat,gbd,un):
    un=un.loc[un.location_name.eq('Saudi Arabia')].copy()
    un['broad_age']=np.where(un.age_start.ge(80),'80+',un.age)
    ug=un.groupby(['year','sex','broad_age']).population_persons.sum()
    base=gbd.loc[gbd.location_name.eq('Saudi Arabia') & gbd.year.eq(2023) & gbd.outcome.eq('prevalence'),['sex','age','implied_population']]
    ub=un.loc[un.year.eq(2023),['sex','age','population_persons']].rename(columns={'population_persons':'un_2023'})
    n=nat.groupby(['year','sex','age']).population.sum()
    output=[]
    roles=['tcn_adapted','local_champion','nonneural_champion']
    for outcome in ['prevalence','incidence']:
        raw=read(f'results/projections_v1/trials/SAU_{outcome}/predictions.csv')
        r=raw.loc[raw.family.isin(roles),['sex','age','forecast_year','family','prediction']].copy()
        assert len(r)==330 and not r.duplicated(['sex','age','forecast_year','family']).any()
        r=r.rename(columns={'forecast_year':'year'}).merge(un[['year','sex','age','population_persons','broad_age']],on=['year','sex','age'],validate='many_to_one')
        r=r.merge(base,on=['sex','age'],validate='many_to_one').merge(ub,on=['sex','age'],validate='many_to_one')
        r['gbd_aligned_population']=r.implied_population*r.population_persons/r.un_2023
        for scenario in ['gbd_2023_aligned_un_growth','un_medium_unaligned','national_2022_un_growth','national_2024_un_growth']:
            allocations=['native_age_detail'] if not scenario.startswith('national') else ['gbd_aligned_within_80plus','un_within_80plus']
            for allocation in allocations:
                part=r.copy()
                if scenario=='gbd_2023_aligned_un_growth':part['population']=part.gbd_aligned_population
                elif scenario=='un_medium_unaligned':part['population']=part.population_persons
                else:
                    anchor=int(scenario.split('_')[1])
                    part['broad_population']=[n.loc[(anchor,sex,age)]*ug.loc[(year,sex,age)]/ug.loc[(anchor,sex,age)]
                       for year,sex,age in zip(part.year,part.sex,part.broad_age)]
                    weights=part.gbd_aligned_population if allocation.startswith('gbd') else part.population_persons
                    part['raw_weight']=weights
                    denominator=part.groupby(['year','sex','family','broad_age']).raw_weight.transform('sum')
                    part['population']=part.broad_population*part.raw_weight/denominator
                    sums=part.groupby(['year','sex','family','broad_age']).population.sum()
                    intended=part.groupby(['year','sex','family','broad_age']).broad_population.first()
                    np.testing.assert_allclose(sums,intended,rtol=1e-13)
                part['count']=part.prediction*part.population/1e5
                part['outcome']=outcome;part['scenario']=scenario;part['within_80plus_allocation']=allocation
                output.append(part[['outcome','year','sex','age','broad_age','family','prediction','scenario','within_80plus_allocation','population','count']])
    cells=pd.concat(output,ignore_index=True);save(cells,'conditional_projection_cells_2024_2028.csv')
    keys=['outcome','year','family','scenario','within_80plus_allocation']
    summaries=[]
    for key,g in cells.groupby(keys):
        for sex in ['Male','Female','Both']:
            sub=g if sex=='Both' else g.loc[g.sex.eq(sex)]
            total=sub['count'].sum();oldest=sub.loc[sub.broad_age.eq('80+'),'count'].sum()
            summaries.append(dict(zip(keys,key),sex=sex,count_45plus=total,count_80plus=oldest,share_80plus=100*oldest/total))
    summary=pd.DataFrame(summaries);save(summary,'conditional_projection_summary_2024_2028.csv')
    prior=read('reports/study_synthesis_v1_2/saudi_2028_scenario_totals.csv')
    checked=summary.loc[summary.year.eq(2028)&summary.family.eq('tcn_adapted')&summary.sex.eq('Both')&summary.within_80plus_allocation.eq('native_age_detail')]
    checked=checked.merge(prior[['outcome','scenario','value']],on=['outcome','scenario'],validate='one_to_one')
    assert len(checked)==4
    np.testing.assert_allclose(checked.count_45plus,checked.value,rtol=1e-12)
    save(checked,'original_projection_reproduction.csv')
    return summary


def insurance(nat):
    audit=read('reports/raw_download_review_2026-10-01/chi_internal_accounting.csv')
    records=[];excluded=[]
    for filename,g in audit.groupby('file'):
        p=track(filename);sheets=ods_sheets(p) if p.suffix=='.ods' else xlsx_sheets(p)
        for _,record in g.iterrows():
            indicator=int(record.indicator)
            sheet=next(s for s in sheets if s.replace('_','').replace(' ','')=='مؤشر'+str(indicator))
            rows=sheets[sheet];header=rows[0][1]
            if p.suffix=='.ods':
                columns={'count':'A','sex':'B','age':'C','beneficiary_type':'D','nationality':'G'}
            else:
                columns={}
                for col,label in header.items():
                    if label=='الجنسية':columns['nationality']=col
                    if label=='الفئة العمرية':columns['age']=col
                    if label=='الجنس':columns['sex']=col
                    if label.startswith('عدد'):columns['count']=col
                    if label in ['نوع المشترك','نوع المستفيد','نوع المنشأة']:columns['beneficiary_type']=col
            assert set(columns)=={'nationality','age','sex','count','beneficiary_type'}
            values=[]
            for rn,row in rows[1:]:
                sex=row.get(columns['sex'],'').strip()
                if not sex:
                    excluded.append(dict(file=filename,sheet=sheet,row=rn,reason='Total/blank row; not an individual category'));continue
                sex={'ذكر':'Male','انثى':'Female','أنثى':'Female','Male':'Male','Female':'Female'}[sex]
                nationality={'سعودي':'Citizens','غير سعودي':'Non-citizens','Saudi':'Citizens','Non-Saudi':'Non-citizens'}[row[columns['nationality']].strip()]
                val=float(row[columns['count']].replace(',',''))
                assert val>=0 and int(val)==val
                values.append(dict(file=filename,sheet=sheet,source_row=rn,source_cell=columns['count']+str(rn),
                    period=record.period_label,indicator=indicator,metric='subscribers' if indicator==3 else 'insured_persons',
                    sex=sex,nationality=nationality,age_original=row[columns['age']].strip(),
                    beneficiary_type_original=row.get(columns['beneficiary_type'],''),count=int(val)))
            assert sum(v['count'] for v in values)==record.headline_total
            records.extend(values)
    df=pd.DataFrame(records);save(df,'chi_category_cells.csv',data=True)
    save(pd.DataFrame(excluded),'chi_excluded_total_rows.csv')
    groups=['file','period','indicator','metric']
    rows=[]
    for key,g in df.groupby(groups):
        total=g['count'].sum()
        rows.append(dict(zip(groups,key),total=total,
            male_percent=100*g.loc[g.sex.eq('Male'),'count'].sum()/total,
            noncitizen_percent=100*g.loc[g.nationality.eq('Non-citizens'),'count'].sum()/total,
            oldest_original_labels=' | '.join(sorted(a for a in g.age_original.unique() if '61' in a or '65' in a)),
            over_65_percent=(100*g.loc[g.age_original.eq('age > 65'),'count'].sum()/total if 'age > 65' in set(g.age_original) else np.nan),
            ambiguous_age_label=any('>=' in a for a in g.age_original.unique())))
    summary=pd.DataFrame(rows);save(summary,'chi_quarter_composition.csv')
    ages=df.groupby(groups+['sex','age_original'],as_index=False)['count'].sum()
    save(ages,'chi_original_age_composition.csv')
    # Only compare sex/nationality definitions that do not require age splitting.
    q=df.loc[df.file.str.contains('Q2-2024',regex=False)]
    national=nat.loc[nat.year.eq(2024)].groupby(['sex','nationality']).population.sum()
    comparisons=[]
    for metric,part in q.groupby('metric'):
        counted=part.groupby(['sex','nationality'])['count'].sum();total=counted.sum()
        for (sex,nationality),value in counted.items():
            npop=national.loc[(sex,nationality)]
            comparisons.append(dict(period='Q2 2024',metric=metric,sex=sex,nationality=nationality,
                chi_count=value,national_population=npop,chi_composition_percent=100*value/total,
                national_composition_percent=100*npop/national.sum(),
                composition_difference_pp=100*(value/total-npop/national.sum()),
                snapshot_count_per_100_residents=100*value/npop,
                interpretation='Descriptive snapshot ratio; not coverage probability or person-time'))
    comparison=pd.DataFrame(comparisons);save(comparison,'chi_2024_population_composition_comparison.csv')
    # Retain the separate beneficiary class table; do not equate its total to a quarter.
    classes=read('data/raw/chi/CHI beneficiary class to the Age group 2024.csv')
    valcol=next(c for c in classes if c.startswith('CHI beneficiaries'))
    classes['count']=classes[valcol].str.replace(',','',regex=False).str.strip().astype(int)
    detail=classes.loc[classes['Class Name'].isin(['A','B','C','VIP'])].copy()
    assert len(detail)==28 and detail['count'].sum()==13158204
    detail['age_label_ambiguous']=detail['Age Group'].str.contains('>=',regex=False)
    detail['percent_of_class']=100*detail['count']/detail.groupby('Class Name')['count'].transform('sum')
    save(detail,'chi_beneficiary_class_by_original_age.csv',data=True)
    return summary,comparison


def source_eligibility():
    inv=read('reports/raw_download_review_2026-10-01/download_inventory.csv')
    rows=[]
    for r in inv.itertuples():
        p=track(r.file);assert sha(p)==r.sha256
        name=p.name.lower()
        if p.suffix=='.csv' and p.parent.name in ['moh','gastat']:
            role='Population source audit and conditional burden sensitivity'
            reason='Demographic counts only; no Parkinson outcome observations'
        elif p.parent.name=='chi' and ('open' in name or p.suffix=='.csv'):
            role='Insurance composition and future validation eligibility'
            reason='Preserve snapshot, indicator and age definitions; no Parkinson numerator'
        elif 'nphies' in name:
            role='Outcome-field eligibility audit'
            reason='Visible top-10 summaries and embedded aggregate service records lack required age-sex Parkinson outcomes'
        elif p.parent.name=='nhrsp' and p.suffix=='.xlsx':
            role='Survey-instrument eligibility assessment'
            reason='Codebook/questionnaire, no respondent observations or identified Parkinson-specific item'
        else:
            role='Definitions, attribution or access documentation'
            reason='Not observations for training; no access approval inferred'
        rows.append(dict(file=r.file,sha256=r.sha256,integration_role=role,eligible_disease_training_input=False,reason=reason))
    save(pd.DataFrame(rows),'data_use_and_eligibility.csv')


def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    track('study_design/raw_data_integration_2026-10-03.json')
    source_eligibility()
    nat=national_population()
    gbd=read('data/processed/design_v1/regional_outcomes.csv')
    un=read('data/processed/design_v1/un_population_1990_2028.csv')
    pop,counts=population_and_counts(nat,gbd,un)
    projection=projections(nat,gbd,un)
    chi,comparison=insurance(nat)
    assert all(sha(ROOT/p)==h for p,h in INPUTS.items())
    validation=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=INPUTS,
        raw_files_integrated=19,national_population_cells=len(nat),moh_source_cells_verified=68,
        population_comparison_cells=len(pop),historical_count_summaries=len(counts),
        chi_totals_reconciled=len(chi),projection_reference_rows_reproduced=4,
        no_disease_model_refit=True,no_new_interval_calibration=True,original_inputs_unchanged=True,
        analysis_role='Exploratory demographic and insurance evidence; not new independent forecast validation')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps({k:v for k,v in validation.items() if k!='source_sha256'}))


if __name__=='__main__':main()
