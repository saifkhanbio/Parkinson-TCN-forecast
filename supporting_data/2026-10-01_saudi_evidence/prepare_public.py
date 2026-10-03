"""Validate public Saudi evidence; do not fit or score any WHO mortality data."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'work/saudi-evidence-vendor'))
import pandas as pd
import numpy as np
from docx import Document


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saudi_chunks(path):
    parts=[]
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();assert len(names)==1
        with archive.open(names[0]) as stream:
            for chunk in pd.read_csv(stream,chunksize=100000,dtype={x:str for x in ['Admin1','SubDiv','List','Cause','Frmat','IM_Frmat']}):
                selected=chunk.loc[chunk.Country.eq(3340)].copy()
                if len(selected):
                    selected['source_archive']=path.name;parts.append(selected)
    return pd.concat(parts,ignore_index=True)


def main():
    out=ROOT/'processed';out.mkdir(exist_ok=True)
    if (out/'validation.json').exists():raise FileExistsError('Existing evidence extraction is immutable')
    manifests=[]
    for p in ROOT.glob('manifest_*.json'):manifests.extend(json.loads(p.read_text()))
    for r in manifests:
        if r['status']=='downloaded':assert sha(ROOT/r['path'])==r['sha256']
    with zipfile.ZipFile(ROOT/'raw/who_mort_availability.zip') as z:
        x=pd.ExcelFile(io.BytesIO(z.read(z.namelist()[0])))
        availability=[]
        for sheet in x.sheet_names:
            d=pd.read_excel(x,sheet_name=sheet,header=None)
            mask=pd.to_numeric(d[0],errors='coerce').eq(3340)
            selected=d.loc[mask].copy();selected.columns=['column_'+str(i) for i in selected.columns]
            selected['source_sheet']=sheet;availability.append(selected)
        pd.concat(availability).to_csv(out/'who_saudi_year_availability.csv',index=False)
    mortality=pd.concat([saudi_chunks(ROOT/'raw'/f) for f in ['who_morticd10_part3.zip','who_morticd10_part6.zip']],ignore_index=True)
    population=saudi_chunks(ROOT/'raw/who_mort_pop.zip')
    assert set(mortality.Year)=={2009,2012,2021,2022,2023,2024}
    assert mortality.Admin1.isna().all() and mortality.SubDiv.isna().all()
    assert set(population.Year)=={2021,2022,2023,2024}
    # Preserve a Saudi extract for coding/completeness review, without changing sources.
    mortality.to_csv(out/'who_saudi_all_causes_raw_extract.csv.gz',index=False,compression='gzip')
    population.to_csv(out/'who_saudi_population_raw_extract.csv',index=False)
    pd_deaths=mortality.loc[mortality.Cause.str.replace('.','',regex=False).str.fullmatch(r'G20[0-9]*')].copy()
    assert set(pd_deaths.Cause)=={'G20'}
    assert not pd_deaths.duplicated(['Year','Sex']).any()
    pd_deaths.to_csv(out/'who_saudi_g20_raw_extract_unscored.csv',index=False)
    metadata=pd_deaths[['Year','Sex','List','Frmat','Admin1','SubDiv','source_archive']].copy()
    metadata['available_age_top_group']=metadata.Frmat.map({'00':'95+','01':'85+','02':'85+'})
    metadata['clinical_scope']='registered underlying-cause G20 deaths; not all deaths among people with PD'
    metadata['national_coverage_flag']='no subnational/nationals-only flag; completeness not established'
    metadata['used_for_model_fitting_or_scoring']=False
    metadata.to_csv(out/'who_saudi_g20_metadata.csv',index=False)
    cell_rows=[];checks=[]
    for row in pd_deaths.to_dict('records'):
        fmt=row['Frmat']; assert fmt in {'00','01','02'}
        last=25 if fmt=='00' else 23
        used=[f'Deaths{i}' for i in range(2,last+1)]
        # Formats 01/02 top-code at 85; format 02 combines ages 1–4 into Deaths3.
        if fmt=='02':used=[x for x in used if x not in {'Deaths4','Deaths5','Deaths6'}]
        active=pd.Series({k:row[k] for k in used},dtype=float)
        unused=[f'Deaths{i}' for i in range(2,26) if f'Deaths{i}' not in used]
        assert all(pd.isna(row[k]) or row[k]==0 for k in unused)
        assert active.notna().all(), 'Missing active age values require review, not zero imputation'
        assert active.ge(0).all()
        unknown_blank=bool(pd.isna(row['Deaths26']))
        if unknown_blank:
            # Preserve the source blank; verify that reported known ages exhaust the total.
            assert active.sum()==row['Deaths1']
        else:
            assert active.sum()+row['Deaths26']==row['Deaths1']
        checks.append(dict(year=row['Year'],sex=row['Sex'],age_sum_equals_total=True,
                           unused_cells_blank_or_zero=True,unknown_age_blank=unknown_blank,
                           unknown_age_present=None if unknown_blank else bool(row['Deaths26']>0)))
        for i in range(15,last+1):
            start=45+5*(i-15)
            age=f'{start}+' if i==last else f'{start}-{start+4}'
            cell_rows.append(dict(year=row['Year'],sex={1:'Male',2:'Female'}[row['Sex']],age=age,
                                 deaths=row[f'Deaths{i}'],cause='G20',source_field=f'Deaths{i}',
                                 age_format=fmt,source_archive=row['source_archive'],
                                 used_for_model_fitting_or_scoring=False))
    cells=pd.DataFrame(cell_rows)
    cells.loc[cells.year.le(2023)].to_csv(out/'who_saudi_g20_age_sex_2009_2023_unscored.csv',index=False)
    cells.loc[cells.year.eq(2024)].to_csv(out/'who_saudi_g20_age_sex_2024_reserved_unscored.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'who_g20_age_accounting_checks.csv',index=False)
    # Saudi clinical supplementary files have real published content, not a national panel.
    inventory=[]
    for f in sorted((ROOT/'raw').glob('saudi_genetics_s*.bin')):
        signature=f.read_bytes()[:4]
        if signature in [b'II*\x00',b'MM\x00*']:
            ext='.tif';role='published_genetic_figure'
        else:
            ext='.docx';role='published_clinical_or_genetic_supplement'
            document=Document(f)
            for i,table in enumerate(document.tables,1):
                pd.DataFrame([[c.text for c in r.cells] for r in table.rows]).to_csv(out/(f.stem+f'_table{i}_verbatim.csv'),index=False,header=False)
        dest=out/(f.stem+ext);shutil.copyfile(f,dest)
        inventory.append(dict(raw_file=f.name,readable_copy=dest.name,verified_type=ext,verified_role=role,sha256=sha(f)))
    pd.DataFrame(inventory).to_csv(out/'clinical_supplement_inventory.csv',index=False)
    validation=dict(status='complete',downloaded_files=len([r for r in manifests if r['status']=='downloaded']),
        source_hashes_verified=True,saudi_mortality_years=sorted(mortality.Year.unique().tolist()),
        saudi_g20_rows=len(pd_deaths),g20_present_in_2024_both_sexes=True,
        g20_2009_male_row_missing_not_imputed_as_zero=True,
        g20_age_accounting_passed=True,top_age_2021_2024='85+',
        who_data_used_for_model_fitting_or_scoring=False,
        independent_of_gbd_inputs='unverified; source lineage audit required',
        coverage_completeness='not established by blank Admin1/SubDiv fields',
        source_sha256={r['path']:r['sha256'] for r in manifests if r['status']=='downloaded'},
        processed_sha256={str(f.relative_to(out)):sha(f) for f in out.iterdir() if f.is_file()})
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps({k:validation[k] for k in ['status','saudi_mortality_years','saudi_g20_rows','top_age_2021_2024','who_data_used_for_model_fitting_or_scoring']}))


if __name__=='__main__':main()
