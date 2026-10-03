"""Resolve population alignment dimensions without replacing frozen forecasts."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/population_denominator_audit_v2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():raise FileExistsError('Existing audit is immutable')
    OUT.mkdir()
    paths={
        'gbd':'data/processed/design_v1/regional_outcomes.csv',
        'gastat':'supporting_data/2026-09-26/processed/gastat_saudi_population_age_sex_nationality_2023_2024.csv',
        'gccstat':'supporting_data/2026-09-26/processed/gccstat_saudi_population_age_sex_nationality_2010_2024.csv',
        'un':'data/processed/design_v1/un_population_1990_2028.csv',
        'source_uncertainty':'results/source_compatibility_v1/uncertainty_inventory.csv',
        'un_marginal_quantiles':'results/source_compatibility_v1/un_gcc_age_sex_marginal_quantiles_2024_2028.csv',
    }
    hashes={v:sha(ROOT/v) for v in paths.values()}
    gbd=pd.read_csv(ROOT/paths['gbd']);gastat=pd.read_csv(ROOT/paths['gastat'])
    gcc=pd.read_csv(ROOT/paths['gccstat']);un=pd.read_csv(ROOT/paths['un'])
    gbd['recomputed_population']=gbd['count']/gbd.rate*100000
    np.testing.assert_allclose(gbd.recomputed_population,gbd.implied_population,rtol=1e-10,atol=1e-8)
    consistency=gbd.groupby(['location_name','sex','age','year'],as_index=False).agg(
        smallest=('recomputed_population','min'),largest=('recomputed_population','max'),outcomes=('outcome','nunique'))
    consistency['relative_range']=(consistency.largest-consistency.smallest)/consistency.smallest
    common=consistency.loc[consistency.year.ge(1990)]
    assert common.outcomes.eq(6).all()
    assert common.relative_range.max()<1e-6
    consistency.to_csv(OUT/'gbd_implied_denominator_consistency.csv',index=False)
    # Check totals before restricting to the disease ages. Keep nationality universes explicit.
    checks=[]
    for (year,sex,age),part in gastat.groupby(['year','sex','age_group']):
        v=part.set_index('nationality').population_persons
        difference=v['Total']-v['Citizens']-v['Non-citizens']
        checks.append(dict(check='nationality_additivity',year=year,sex=sex,age_group=age,difference=difference))
    for (year,nationality,age),part in gastat.groupby(['year','nationality','age_group']):
        v=part.set_index('sex').population_persons
        difference=v['Both']-v['Male']-v['Female']
        checks.append(dict(check='sex_additivity',year=year,nationality=nationality,age_group=age,difference=difference))
    for (year,nationality,sex),part in gastat.groupby(['year','nationality','sex']):
        v=part.set_index('age_group').population_persons
        difference=v['All Ages']-v.drop('All Ages').sum()
        checks.append(dict(check='age_additivity',year=year,nationality=nationality,sex=sex,difference=difference))
    checks=pd.DataFrame(checks);assert checks.difference.abs().max()<=1
    checks.to_csv(OUT/'gastat_internal_accounting.csv',index=False)
    saudi=gbd.loc[gbd.location_name.eq('Saudi Arabia')&gbd.outcome.eq('prevalence')&gbd.year.eq(2023)].copy()
    saudi['age_group']=np.where(saudi.age_start.ge(80),'80+',saudi.age)
    saudi=saudi.groupby(['sex','age_group'],as_index=False).recomputed_population.sum().rename(columns={'recomputed_population':'gbd_implied_persons'})
    national=gastat.loc[gastat.year.eq(2023)&gastat.sex.isin(['Male','Female'])].pivot(index=['sex','age_group'],columns='nationality',values='population_persons').reset_index()
    national=national.rename(columns={'Total':'gastat_total_persons','Citizens':'gastat_citizen_persons','Non-citizens':'gastat_noncitizen_persons'})
    u=un.loc[un.location_name.eq('Saudi Arabia')&un.year.eq(2023)].copy()
    u['age_group']=np.where(u.age_start.ge(80),'80+',u.age)
    u=u.groupby(['sex','age_group'],as_index=False).population_persons.sum().rename(columns={'population_persons':'un_persons'})
    c=gcc.loc[gcc.TIME_PERIOD.eq(2023)&gcc.nationality.eq('Total')&gcc.sex.isin(['Male','Female'])].rename(columns={'AGE':'age_group','population_persons_original':'gccstat_persons'})
    table=saudi.merge(national,on=['sex','age_group'],validate='one_to_one').merge(u,on=['sex','age_group'],validate='one_to_one').merge(c[['sex','age_group','gccstat_persons']],on=['sex','age_group'],validate='one_to_one')
    assert len(table)==16
    table['gbd_vs_gastat_pct']=100*(table.gbd_implied_persons/table.gastat_total_persons-1)
    table['gbd_vs_un_pct']=100*(table.gbd_implied_persons/table.un_persons-1)
    table['gccstat_minus_gastat']=table.gccstat_persons-table.gastat_total_persons
    table.to_csv(OUT/'saudi_2023_harmonized_population_comparison.csv',index=False)
    table.loc[table.age_group.eq('80+')].to_csv(OUT/'saudi_80plus_population_comparison.csv',index=False)
    pd.DataFrame([
        dict(source='GBD-implied',universe='national outcome location; no nationality breakdown in export',reference_time='annual year; exact demographic reference date not independently verified from supplied export',unit='persons inferred as count/rate*100000',age_top='95+',status='algebra verified across six outcomes; direct population export still desirable'),
        dict(source='GASTAT',universe='Total = citizens + non-citizens; both sexes or sex-specific',reference_time='1 July per official methodology',unit='persons in extracted workbook',age_top='80+',status='source accounting passed; cannot split 80+ without additional data'),
        dict(source='UN WPP2024',universe='de facto population',reference_time='1 July',unit='raw thousands multiplied by 1000 once',age_top='95+ after summing 95–99 and 100+',status='age/unit alignment established; population estimate differs from GBD'),
        dict(source='GCC-Stat',universe='Total = national and non-national categories where supplied',reference_time='annual; export lacks Saudi cell reference-date metadata',unit='persons',age_top='80+',status='compare with GASTAT; do not count agreement as independent demographic evidence'),
    ]).to_csv(OUT/'source_definition_crosswalk.csv',index=False)
    pd.DataFrame([
        dict(component='historical disease forecast error',available='completed prequential residual blocks',included='original matched intervals',remaining='few overlapping origins; systematic bias may persist'),
        dict(component='population forecast error',available='paired historical population residuals',included='matched joint burden intervals',remaining='conditional on inferred denominator and historical error process'),
        dict(component='model state and innovation uncertainty',available='Bayesian conditional posterior',included='native age-time intervals',remaining='finite prior grid and Gaussian structural assumptions'),
        dict(component='GBD source-estimation uncertainty',available='marginal lower/upper bounds only',included='no',remaining='joint draws needed across outcomes/ages/sex/years'),
        dict(component='UN demographic projection uncertainty',available='marginal age-sex quantiles',included='no in joint disease/population posterior',remaining='joint trajectories and dependence required; do not add marginal bounds'),
        dict(component='population-provider definition and estimation differences',available='matched source scenarios',included='sensitivity analysis only',remaining='scenario spread is not a probability interval'),
    ]).to_csv(OUT/'uncertainty_accounting.csv',index=False)
    audit=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
        common_denominator_cells=len(common),maximum_relative_outcome_denominator_range=float(common.relative_range.max()),
        gastat_internal_maximum_rounding_difference_persons=float(checks.difference.abs().max()),
        gccstat_gastat_all_16_cells_exact=bool(table.gccstat_minus_gastat.eq(0).all()),
        source_equivalence_established=False,direct_gbd_population_export_verified=False,
        all_missing_uncertainty_components_resolved=False,frozen_forecasts_or_denominators_replaced=False,
        audit_code_sha256=sha(Path(__file__)))
    for path,digest in hashes.items():assert sha(ROOT/path)==digest
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','common_denominator_cells','maximum_relative_outcome_denominator_range','gccstat_gastat_all_16_cells_exact','source_equivalence_established']}))


if __name__=='__main__':main()
