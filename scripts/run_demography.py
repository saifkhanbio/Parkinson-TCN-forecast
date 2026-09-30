"""Joint age shares, sex-rate ratios and operational burden-count evaluation."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from gbd_park.demography import population_forecast,population_residuals,count_and_share_views,rate_ratio_views,joint_count_draws,shapley_change
from gbd_park.intervals import interval_score,weighted_interval_score
from run_local_baselines import sha,now,check_lock

CONTEXT=['target','outcome','origin','family','horizon','forecast_year','population_method']
STAT=CONTEXT+['measure','node','sex','age_group','unit']
METHODS=['log_trend_last8','persistence']


def burden_statistics(cells,config,draw=False):
    ids=CONTEXT+(['residual_origin'] if draw else [])
    counts,shares=count_and_share_views(cells,config,identifiers=ids)
    counts=counts.rename(columns={'count':'value','level':'hierarchy_level'})
    counts['measure'],counts['unit']='count','modeled_number'
    shares['value']=shares.share*100
    shares['measure'],shares['unit']='age_share','percent'
    shares['age_group']=shares.threshold.astype(str)+'+_within_45+'
    shares['node']=shares.sex+'__'+shares.age_group
    shares['hierarchy_level']='sex_total'
    return pd.concat([counts[STAT+['value','hierarchy_level']+(['residual_origin'] if draw else [])],
                      shares[STAT+['value','hierarchy_level']+(['residual_origin'] if draw else [])]],ignore_index=True)


def ratio_statistics(frame,config,draw=False):
    ids=[c for c in CONTEXT if c!='population_method']+['age']+(['residual_origin'] if draw else [])
    ratios=rate_ratio_views(frame,config,value='rate_draw' if draw else 'prediction',identifiers=ids)
    ratios=ratios.rename(columns={'male_female_rate_ratio':'value','age':'age_group'})
    ratios['measure'],ratios['unit'],ratios['sex']='sex_rate_ratio','rate_ratio','Male/Female'
    ratios['node']='Male_Female__'+ratios.age_group
    ratios['hierarchy_level']='age_sex_ratio'
    ratios['population_method']='not_applicable'
    return ratios[STAT+['value','hierarchy_level']+(['residual_origin'] if draw else [])]


def distribution_summary(draws):
    if draws.duplicated(STAT+['residual_origin']).any():
        raise ValueError('Duplicate transformed joint-block coordinates')
    if not np.isfinite(draws.value).all() or draws.value.le(0).any():
        raise ValueError('Invalid transformed draw')
    group=draws.groupby(STAT,sort=True,dropna=False).value
    quantiles=group.quantile([.025,.1,.25,.5,.75,.9,.975],interpolation='linear').unstack()
    blocks=group.count()
    if blocks.min()<5:
        raise ValueError('Insufficient whole residual blocks')
    frames=[]
    for level,lo,hi in [(.5,.25,.75),(.8,.1,.9),(.95,.025,.975)]:
        part=pd.DataFrame({'lower':quantiles[lo],'median':quantiles[.5],'upper':quantiles[hi],'n_blocks':blocks}).reset_index()
        part['level']=level
        frames.append(part)
    return pd.concat(frames,ignore_index=True)


def score_distributions(intervals,truth):
    frame=intervals.merge(truth,on=STAT,validate='many_to_one',how='left')
    assert np.isfinite(frame.observed).all()
    frame['width']=frame.upper-frame.lower
    frame['covered']=(frame.observed>=frame.lower)&(frame.observed<=frame.upper)
    frame['interval_score']=interval_score(frame.observed,frame.lower,frame.upper,1-frame.level)
    levels=[.5,.8,.95]
    ordered=frame.sort_values(STAT+['level'])
    lower=ordered.pivot(index=STAT,columns='level',values='lower')[levels]
    upper=ordered.pivot(index=STAT,columns='level',values='upper')[levels]
    base=ordered.drop_duplicates(STAT).set_index(STAT).reindex(lower.index)
    wis=base[['observed','median','n_blocks']].copy()
    wis['wis_50_80']=weighted_interval_score(base.observed,base['median'],lower.iloc[:,:2],upper.iloc[:,:2],levels[:2])
    wis['wis_50_80_95']=weighted_interval_score(base.observed,base['median'],lower,upper,levels)
    return frame,wis.reset_index()


def case_job(case,config,out):
    started=time.perf_counter()
    directory=Path(out)/case['id'];directory.mkdir()
    source=ROOT/case['source']
    points=pd.read_csv(source/'predictions.csv',float_precision='round_trip')
    draws=pd.read_csv(source/'joint_draws.csv',float_precision='round_trip')
    assert set(points.target)=={case['target']} and set(points.outcome)=={case['outcome']}
    panel=pd.read_csv(ROOT/'data/processed/design_v1/regional_outcomes.csv')
    predictions=[ratio_statistics(points,config)]
    statistics=[ratio_statistics(draws,config,True)]
    populations,population_errors=[],[]
    for method in METHODS:
        for origin in range(2014,2019):
            current=points.loc[points.origin.eq(origin)].copy()
            rate_draws=draws.loc[draws.origin.eq(origin)].copy()
            residual_origins=sorted(rate_draws.residual_origin.unique().tolist())
            assert residual_origins==list(range(2003,origin-4))
            for _,part in rate_draws.groupby('family'):
                assert len(part)==len(residual_origins)*110
                assert set(part.residual_origin)==set(residual_origins)
            pop=population_forecast(panel,config,origin,case['target'],case['outcome'],method)
            errors=population_residuals(panel,config,origin,case['target'],case['outcome'],method,residual_origins)
            populations.append(pop);population_errors.append(errors)
            keys=['target','outcome','origin','sex','age','horizon','forecast_year']
            cells=current.merge(pop[keys+['population_method','population']],on=keys,validate='many_to_one')
            cells['count']=cells.prediction*cells.population/100000
            predictions.append(burden_statistics(cells,config))
            joint=joint_count_draws(rate_draws,pop,errors)
            statistics.append(burden_statistics(joint,config,True))
    predicted=pd.concat(predictions,ignore_index=True).sort_values(STAT)
    transformed=pd.concat(statistics,ignore_index=True).sort_values(STAT+['residual_origin'])
    intervals=distribution_summary(transformed)
    populations=pd.concat(populations,ignore_index=True)
    errors=pd.concat(population_errors,ignore_index=True)
    predicted.to_csv(directory/'predictions.csv',index=False)
    intervals.to_csv(directory/'intervals.csv',index=False)
    transformed.to_csv(directory/'joint_statistics.csv.gz',index=False,compression='gzip')
    populations.to_csv(directory/'population_forecasts.csv',index=False)
    errors.to_csv(directory/'population_residuals.csv',index=False)
    commits={p.name:sha(p) for p in sorted(directory.iterdir())}
    (directory/'pre_score_commit.json').write_text(json.dumps(dict(time_utc=now(),sha256=commits),indent=2)+'\n')
    events=[dict(event='forecasts_committed',time_utc=now(),sha256=commits),dict(event='scoring_started',time_utc=now())]
    (directory/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    # Verification values enter transformed scoring only after all issued distributions are committed.
    actual=panel.loc[panel.location_name.eq(case['target'])&panel.outcome.eq(case['outcome'])].rename(
        columns={'location_name':'target','year':'forecast_year','rate':'observed_rate','count':'observed_count'})
    verified=points.merge(actual[['target','outcome','sex','age','forecast_year','observed_rate','observed_count']],
        on=['target','outcome','sex','age','forecast_year'],validate='many_to_one',how='left')
    assert np.isfinite(verified[['observed_rate','observed_count']]).all().all()
    verified['prediction']=verified.observed_rate
    truths=[ratio_statistics(verified,config)]
    verified['count']=verified.observed_count
    for method in METHODS:
        verified['population_method']=method
        truths.append(burden_statistics(verified,config))
    truth=pd.concat(truths,ignore_index=True)[STAT+['value']].rename(columns={'value':'observed'})
    scored=predicted.merge(truth,on=STAT,validate='one_to_one')
    assert len(scored)==len(predicted)
    scored['absolute_error']=abs(scored.value-scored.observed)
    scored['signed_error']=scored.value-scored.observed
    scored['absolute_log_error']=abs(np.log(scored.value)-np.log(scored.observed))
    scored.to_csv(directory/'point_scores.csv',index=False)
    cells,wis=score_distributions(intervals,truth)
    cells.to_csv(directory/'interval_scores.csv.gz',index=False,compression='gzip')
    wis.to_csv(directory/'wis_scores.csv',index=False)
    groups=['target','outcome','origin','family','horizon','population_method','measure','sex','hierarchy_level']
    scored.groupby(groups,as_index=False).agg(mean_absolute_error=('absolute_error','mean'),mean_absolute_log_error=('absolute_log_error','mean')).to_csv(directory/'summary.csv',index=False)
    scored.loc[scored.measure.eq('age_share')].groupby(groups+['age_group'],as_index=False).agg(
        mean_absolute_percentage_point_error=('absolute_error','mean')).to_csv(directory/'age_share_summary.csv',index=False)
    # Descriptive population-size/composition/rate accounting for 2018 to 2023.
    start=actual.loc[actual.forecast_year.eq(2018)].set_index(['sex','age'])
    finish=actual.loc[actual.forecast_year.eq(2023)].set_index(['sex','age'])
    decompositions=[]
    for sex in config['sexes']:
        n0=(start.observed_count/start.observed_rate*100000).loc[sex].reindex(config['ages']).to_numpy()
        r0=start.observed_rate.loc[sex].reindex(config['ages']).to_numpy()
        n1=(finish.observed_count/finish.observed_rate*100000).loc[sex].reindex(config['ages']).to_numpy()
        r1=finish.observed_rate.loc[sex].reindex(config['ages']).to_numpy()
        components,change=shapley_change(n0,r0,n1,r1)
        decompositions.append(dict(target=case['target'],outcome=case['outcome'],sex=sex,family='observed_accounting',population_method='implied_gbd',total_change=change,**components))
        for family,part in points.loc[points.origin.eq(2018)&points.horizon.eq(5)&points.sex.eq(sex)].groupby('family'):
            r1=part.set_index('age').prediction.reindex(config['ages']).to_numpy()
            for method in METHODS:
                n1=populations.loc[populations.origin.eq(2018)&populations.horizon.eq(5)&populations.sex.eq(sex)&populations.population_method.eq(method)].set_index('age').population.reindex(config['ages']).to_numpy()
                components,change=shapley_change(n0,r0,n1,r1)
                decompositions.append(dict(target=case['target'],outcome=case['outcome'],sex=sex,family=family,population_method=method,total_change=change,**components))
    pd.DataFrame(decompositions).to_csv(directory/'accounting_decomposition_2018_2023.csv',index=False)
    assert all(sha(directory/name)==digest for name,digest in commits.items())
    validation=dict(passed=True,case=case['id'],point_statistics=len(predicted),joint_statistics=len(transformed),interval_rows=len(intervals),
                    paired_rate_population_blocks=True,source_bounds_used_as_draws=False,minimum_blocks=int(intervals.n_blocks.min()),
                    forecasts_committed_before_final_scoring=True,elapsed_seconds=time.perf_counter()-started)
    (directory/'validation_report.json').write_text(json.dumps(validation,indent=2)+'\n')
    events.append(dict(event='scoring_complete',time_utc=now()))
    (directory/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    return validation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',default='results/demography_v1')
    args=parser.parse_args()
    assert 1<=args.workers<=12
    check_lock()
    config=json.loads((ROOT/'study_design/locked_v1/design.json').read_text())
    tests=json.loads((ROOT/'work/demography-validation/tests.json').read_text())
    assert tests['passed'] and 'scripts/run_demography.py' in tests['tested_code_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in tests['tested_code_sha256'].items())
    sources={}
    for run in ['primary_v1','secondary_v1']:
        folder=ROOT/'results'/run
        manifest=json.loads((folder/'run_manifest.json').read_text())
        assert manifest['status']=='complete'
        assert all(sha(folder/name)==digest for name,digest in manifest['output_sha256'].items())
        sources[run]=sha(folder/'run_manifest.json')
    cases=[]
    for country in config['countries']:
        if not country['gcc']:continue
        for outcome in ['prevalence','incidence']:
            ident=country['iso3']+'_'+outcome
            cases.append(dict(id=ident,target=country['name'],outcome=outcome,source='results/primary_v1' if ident=='SAU_prevalence' else 'results/secondary_v1/trials/'+ident))
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',created_utc=now(),source_manifest_sha256=sources,code_sha256=tests['tested_code_sha256'],workers=args.workers)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validations=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(case_job,case,config,out) for case in cases]
        for future in as_completed(futures):
            result=future.result();validations.append(result)
            print(f"Burden statistics complete: {result['case']}",flush=True)
    assert len(validations)==12 and all(r['passed'] for r in validations)
    assert all(sha(ROOT/name)==digest for name,digest in tests['tested_code_sha256'].items())
    assert all(sha(ROOT/'results'/run/'run_manifest.json')==digest for run,digest in sources.items())
    check_lock()
    (out/'validation_report.json').write_text(json.dumps(dict(passed=True,cases=validations),indent=2)+'\n')
    manifest.update(status='complete',completed_utc=now(),output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_manifest.json'})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
