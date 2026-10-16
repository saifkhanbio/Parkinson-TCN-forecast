"""Score committed forecasts, matched intervals and burden diagnostics once."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from gbd_park.intervals import build_residual_bank, apply_bank, interval_score, weighted_interval_score
from gbd_park.demography import population_residuals

RUN=ROOT/'results/bayesian_age_time_v1'
OUT=ROOT/'reports/bayesian_age_time_v1'
KEYS=['target','outcome','origin','horizon','forecast_year','sex','age']
TRUTH_KEYS=['target','outcome','forecast_year','sex','age']
ROLES=['tcn_adapted','local_champion','nonneural_champion']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def score_intervals(frame,truth):
    d=frame.merge(truth[TRUTH_KEYS+['observed_rate']],on=TRUTH_KEYS,validate='many_to_one')
    assert len(d)==len(frame)
    d['observed']=np.where(d.scale.eq('rate'),d.observed_rate,np.log(d.observed_rate))
    return add_interval_scores(d,KEYS+['family','procedure','scale'])


def add_interval_scores(d,keys):
    assert np.isfinite(d[['observed','lower','median','upper']]).all().all()
    assert (d.lower<=d['median']).all() and (d['median']<=d.upper).all()
    d['covered']=(d.observed>=d.lower)&(d.observed<=d.upper)
    d['width']=d.upper-d.lower
    d['below_lower']=d.observed<d.lower;d['above_upper']=d.observed>d.upper
    d['interval_score']=interval_score(d.observed,d.lower,d.upper,1-d.level)
    lo=d.pivot(index=keys,columns='level',values='lower')[[.5,.8]]
    hi=d.pivot(index=keys,columns='level',values='upper')[[.5,.8]]
    b=d.drop_duplicates(keys).set_index(keys).reindex(lo.index)
    w=b[['observed','median']].copy()
    w['wis_50_80']=weighted_interval_score(b.observed,b['median'],lo,hi,[.5,.8])
    return d,w.reset_index()


def summarize_intervals(cells,wis):
    frames=[]
    for label,ages in [('45+',list(range(45,100,5))),('80+',list(range(80,100,5)))]:
        a=cells.loc[cells.age.map(lambda x:int(x.split('-')[0].rstrip('+'))).isin(ages)]
        b=wis.loc[wis.age.map(lambda x:int(x.split('-')[0].rstrip('+'))).isin(ages)]
        keys=['target','outcome','family','procedure','sex','horizon','scale']
        scores=a.groupby(keys+['level'],as_index=False).agg(coverage=('covered','mean'),mean_width=('width','mean'),cells=('covered','size'))
        scores=scores.merge(b.groupby(keys,as_index=False).agg(wis_50_80=('wis_50_80','mean')),on=keys,validate='many_to_one')
        scores['age_scope']=label;frames.append(scores)
    return pd.concat(frames,ignore_index=True)


def metric_values(counts):
    # Last axis is ordered age; preceding dimensions retain sex, draw and horizon.
    total=counts.sum(axis=-1);old=counts[...,7:].sum(axis=-1)
    return {'count_45plus':total,'count_80plus':old,'share_80plus':100*old/total}


def main():
    if OUT.exists():raise FileExistsError('Refusing to overwrite report')
    manifest=json.loads((RUN/'run_manifest.json').read_text())
    assert manifest['status']=='predictions_committed_unscored'
    for name,digest in manifest['output_sha256'].items():assert sha(RUN/name)==digest
    for name,digest in manifest['input_sha256'].items():assert sha(ROOT/name)==digest
    protocol=json.loads((ROOT/'study_design/bounded_extension_2026-10-01.json').read_text())
    config=json.loads((ROOT/'study_design/locked_v1/design.json').read_text())
    OUT.mkdir()
    panel=pd.read_csv(ROOT/'data/processed/design_v1/regional_outcomes.csv')
    truth=panel.rename(columns={'location_name':'target','year':'forecast_year','rate':'observed_rate'})
    pred=pd.read_csv(RUN/'predictions.csv')
    past=pred.merge(truth[TRUTH_KEYS+['observed_rate']],on=TRUTH_KEYS,validate='many_to_one')
    assert len(past)==len(pred)
    current=pred.loc[pred.origin.isin(protocol['evaluation_origins'])].copy()
    native=pd.read_csv(RUN/'native_intervals.csv')
    all_points=[current];all_intervals=[native];burden_rows=[];burden_points=[];banks=[];sources={}
    iso={r['name']:r['iso3'] for r in config['countries']}
    for target in [protocol['target']]+protocol['replications']:
        for outcome in protocol['outcomes']:
            case=iso[target]+'_'+outcome
            old=ROOT/('results/primary_v1' if case=='SAU_prevalence' else 'results/secondary_v1/trials/'+case)
            demo=ROOT/'results/demography_v1'/case
            for file in [old/'predictions.csv',old/'intervals.csv',demo/'population_forecasts.csv',demo/'interval_scores.csv.gz']:
                sources[str(file.relative_to(ROOT))]=sha(file)
            comp=pd.read_csv(old/'predictions.csv');comp=comp.loc[comp.family.isin(ROLES)]
            all_points.append(comp)
            ci=pd.read_csv(old/'intervals.csv');ci=ci.loc[ci.family.isin(ROLES)].copy()
            ci['procedure']='original_matched_centered_residual_blocks';all_intervals.append(ci)
            populations=pd.read_csv(demo/'population_forecasts.csv')
            populations=populations.loc[populations.population_method.eq('log_trend_last8')]
            case_past=past.loc[past.target.eq(target)&past.outcome.eq(outcome)]
            for origin in protocol['evaluation_origins']:
                point=current.loc[current.target.eq(target)&current.outcome.eq(outcome)&current.origin.eq(origin)]
                bank=build_residual_bank(case_past,config,origin,'bayesian_age_time')
                intervals,draws=apply_bank(point,bank,config)
                intervals['procedure']='original_matched_centered_residual_blocks';all_intervals.append(intervals)
                banks.append(dict(target=target,outcome=outcome,origin=origin,n_blocks=bank['n_blocks'],last_residual_outcome_year=max(bank['origins'])+5))
                coordinates=pd.MultiIndex.from_product([config['sexes'],config['ages'],config['calendar']['horizons']],names=['sex','age','horizon'])
                def array(frame,value):
                    return frame.set_index(['sex','age','horizon']).reindex(coordinates)[value].to_numpy().reshape(2,11,5).transpose(0,2,1)
                pop=array(populations.loc[populations.origin.eq(origin)],'population')
                p=array(point,'prediction')
                observed=point.merge(truth[TRUTH_KEYS+['count']],on=TRUTH_KEYS,validate='many_to_one')
                obs=metric_values(array(observed,'count'))
                points=metric_values(p*pop/100000)
                log_resid=bank['centered_residuals'].reshape(bank['n_blocks'],2,11,5).transpose(0,1,3,2)
                pop_res=population_residuals(panel,config,origin,target,outcome,'log_trend_last8',bank['origins'])
                pop_error=np.stack([array(pop_res.loc[pop_res.residual_origin.eq(b)],'centered_population_log_error') for b in bank['origins']])
                matched_counts=np.exp(np.log(p)[None,...]+log_resid)*pop[None,...]*np.exp(pop_error)/100000
                native_draws=[]
                for sex in config['sexes']:
                    name='__'.join(map(str,[target,outcome,sex,origin])).replace(' ','_')+'.npz'
                    with np.load(RUN/'posterior_paths'/name) as data:
                        assert data['age'].tolist()==config['ages'] and data['horizon'].tolist()==config['calendar']['horizons']
                        native_draws.append(np.exp(data['log_draws']))
                native_counts=np.stack(native_draws,axis=1)*pop[None,...]/100000
                for procedure,counts,scope in [('native_model_posterior',native_counts,'rate_posterior_conditional_on_fixed_operational_population'),
                                                ('original_matched_centered_residual_blocks',matched_counts,'paired_historical_rate_population_errors_not_source_uncertainty')]:
                    values=metric_values(counts)
                    for measure,values_draw in values.items():
                        q=np.quantile(values_draw,[.025,.1,.25,.5,.75,.9,.975],axis=0)
                        for si,sex in enumerate(config['sexes']):
                            for hi,h in enumerate(config['calendar']['horizons']):
                                context=dict(target=target,outcome=outcome,origin=origin,horizon=h,forecast_year=origin+h,sex=sex,
                                             family='bayesian_age_time',procedure=procedure,measure=measure,uncertainty_scope=scope)
                                for level,lo,up in [(.5,2,4),(.8,1,5),(.95,0,6)]:
                                    burden_rows.append(dict(context,level=level,lower=q[lo,si,hi],median=q[3,si,hi],upper=q[up,si,hi],observed=obs[measure][si,hi]))
                                if procedure=='native_model_posterior':
                                    burden_points.append(dict(context,prediction=points[measure][si,hi],observed=obs[measure][si,hi]))
            # Existing joint burden intervals are reused as a scope-matched reference.
            old_b=pd.read_csv(demo/'interval_scores.csv.gz')
            old_b=old_b.loc[old_b.family.isin(ROLES)&old_b.population_method.eq('log_trend_last8')&old_b.sex.isin(config['sexes'])].copy()
            def identify(row):
                if row['measure']=='count' and row['age_group']=='45+':return 'count_45plus'
                if row['measure']=='count' and row['age_group']=='80+':return 'count_80plus'
                if row['measure']=='age_share' and row['age_group']=='80+_within_45+':return 'share_80plus'
                return None
            old_b['selected_measure']=old_b.apply(identify,axis=1);old_b=old_b.loc[old_b.selected_measure.notna()].copy()
            old_b['measure']=old_b.pop('selected_measure');old_b['procedure']='original_matched_centered_residual_blocks'
            old_b['uncertainty_scope']='paired_historical_rate_population_errors_not_source_uncertainty'
            burden_rows.extend(old_b[['target','outcome','origin','horizon','forecast_year','sex','family','procedure','measure','uncertainty_scope','level','lower','median','upper','observed']].to_dict('records'))
    point=pd.concat(all_points,ignore_index=True)[KEYS+['family','prediction','log_prediction']]
    point=point.merge(truth[TRUTH_KEYS+['observed_rate']],on=TRUTH_KEYS,validate='many_to_one')
    assert len(point)==6*2*2*5*11*5*4
    point['absolute_log_error']=abs(np.log(point.prediction)-np.log(point.observed_rate))
    point['absolute_rate_error']=abs(point.prediction-point.observed_rate)
    point.to_csv(OUT/'point_scores.csv',index=False)
    keys=['target','outcome','sex','family','origin','horizon']
    by_origin=point.groupby(keys,as_index=False).agg(mean_absolute_log_error=('absolute_log_error','mean'),rate_mae=('absolute_rate_error','mean'))
    by_origin.to_csv(OUT/'point_by_origin.csv',index=False)
    by_origin.groupby([x for x in keys if x!='origin'],as_index=False).mean(numeric_only=True).drop(columns='origin').to_csv(OUT/'rolling_point_summary.csv',index=False)
    endpoint=by_origin.loc[by_origin.origin.eq(2018)&by_origin.horizon.eq(5)]
    endpoint.to_csv(OUT/'endpoint_point_summary.csv',index=False)
    interval_frame=pd.concat(all_intervals,ignore_index=True)
    unscored=interval_frame[KEYS+['family','procedure','scale','level','lower','median','upper']]
    unscored.to_csv(OUT/'interval_predictions.csv.gz',index=False,compression='gzip')
    scored,wis=score_intervals(unscored,truth)
    scored.to_csv(OUT/'interval_scores.csv.gz',index=False,compression='gzip');wis.to_csv(OUT/'wis_scores.csv',index=False)
    summarize_intervals(scored,wis).to_csv(OUT/'interval_summary.csv',index=False)
    burden=pd.DataFrame(burden_rows)
    bkeys=['target','outcome','origin','horizon','forecast_year','sex','family','procedure','measure','uncertainty_scope']
    burden,bwis=add_interval_scores(burden,bkeys)
    burden.to_csv(OUT/'burden_interval_scores.csv',index=False);bwis.to_csv(OUT/'burden_wis.csv',index=False)
    group=['target','outcome','sex','family','procedure','horizon','measure','uncertainty_scope']
    summary=burden.groupby(group+['level'],as_index=False).agg(coverage=('covered','mean'),mean_width=('width','mean'),origins=('covered','size'))
    summary=summary.merge(bwis.groupby(group,as_index=False).agg(wis_50_80=('wis_50_80','mean')),on=group,validate='many_to_one')
    summary.to_csv(OUT/'burden_interval_summary.csv',index=False)
    pd.DataFrame(burden_points).to_csv(OUT/'burden_point_scores.csv',index=False)
    pd.DataFrame(banks).to_csv(OUT/'residual_bank_audit.csv',index=False)
    assert len(banks)==60 and min(x['n_blocks'] for x in banks)==7 and max(x['n_blocks'] for x in banks)==11
    assert len(burden)==6*2*5*2*5*3*3*5
    for name,digest in sources.items():assert sha(ROOT/name)==digest
    for name,digest in manifest['output_sha256'].items():assert sha(RUN/name)==digest
    audit=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),scientific_status='exploratory',
               prediction_manifest_sha256=sha(RUN/'run_manifest.json'),source_sha256=sources,
               report_code_sha256=sha(Path(__file__)),point_rows=len(point),interval_rows=len(scored),
               burden_interval_rows=len(burden),prior_run_files_unchanged=True,
               interval_procedures_separated=True,source_estimation_uncertainty_included=False,
               report_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.glob('*') if p.is_file()})
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['status','point_rows','interval_rows','burden_interval_rows']}))


if __name__=='__main__':main()
