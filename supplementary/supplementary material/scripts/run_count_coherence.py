"""Prespecified native-count hierarchy experiment; no primary rate-model changes."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from gbd_park.count_coherence import node_history, independent_counts, reconciliation_weights, reconcile_counts
from gbd_park.demography import hierarchy
from run_local_baselines import sha, now, check_lock


def fit_job(task, config, origin):
    panel = pd.read_csv(ROOT/'data/processed/design_v1/regional_outcomes.csv')
    forecasts,audit = independent_counts(panel,config,origin,task['target'],task['outcome'])
    return forecasts,[dict(target=task['target'],outcome=task['outcome'],**row) for row in audit]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',default='results/count_coherence_v1')
    args=parser.parse_args()
    if not 1<=args.workers<=12:
        raise ValueError('Use 1..12 workers')
    check_lock()
    config=json.loads((ROOT/'study_design/locked_v1/design.json').read_text())
    tests_path=ROOT/'work/demography-validation/tests.json'
    tests=json.loads(tests_path.read_text())
    required=['src/gbd_park/demography.py','src/gbd_park/count_coherence.py','scripts/run_count_coherence.py',
              'tests/test_demography.py','tests/test_count_coherence.py','study_design/demography_implementation.md']
    assert tests['passed'] and set(required).issubset(tests['tested_code_sha256'])
    assert all(sha(ROOT/name)==digest for name,digest in tests['tested_code_sha256'].items())
    out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    manifest=dict(status='running',created_utc=now(),code_sha256=tests['tested_code_sha256'],workers=args.workers,
                  source_sha256=sha(ROOT/'data/processed/design_v1/regional_outcomes.csv'),
                  config_sha256=sha(ROOT/'study_design/locked_v1/design.json'),final_period_scored=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tasks=[dict(target=c['name'],outcome=o) for c in config['countries'] if c['gcc'] for o in ['prevalence','incidence']]
    predictions,audits=[],[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fit_job,task,config,origin) for task in tasks for origin in range(2003,2019)]
        for index,future in enumerate(as_completed(futures),1):
            p,a=future.result();predictions.append(p);audits.extend(a)
            if index%12==0:
                print(f'Native-count ETS batches: {index}/{len(futures)}',flush=True)
    independent=pd.concat(predictions,ignore_index=True).sort_values(['target','outcome','origin','node','horizon'])
    assert len(independent)==12*16*31*5
    independent.to_csv(out/'independent_predictions.csv',index=False)
    (out/'fit_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    events=[dict(event='independent_forecasts_committed',time_utc=now(),sha256=sha(out/'independent_predictions.csv'))]
    panel=pd.read_csv(ROOT/'data/processed/design_v1/regional_outcomes.csv')
    matrix,nodes,_=hierarchy(config)
    results,weights=[],[]
    for task in tasks:
        current=independent.loc[independent.target.eq(task['target']) & independent.outcome.eq(task['outcome'])]
        history=node_history(panel,config,task['target'],task['outcome'],2018)
        truth=history.rename_axis('forecast_year').reset_index().melt(id_vars='forecast_year',var_name='node',value_name='observed_count')
        historical=current.loc[current.origin.le(2013)].merge(truth,on=['node','forecast_year'],validate='many_to_one')
        assert len(historical)==11*31*5
        historical['count_residual']=historical.observed_count-historical.count_prediction
        for origin in range(2014,2019):
            train=history.loc[history.index<=origin]
            for horizon in range(1,6):
                base=current.loc[current.origin.eq(origin)&current.horizon.eq(horizon)].set_index('node').reindex(nodes.node)
                variance,audit=reconciliation_weights(historical,train,config,origin,horizon)
                for node,value in zip(nodes.node,variance):
                    weights.append(dict(**task,origin=origin,horizon=horizon,node=node,variance=value,**audit))
                reconciled=reconcile_counts(base.count_prediction.to_numpy(),variance,config)
                for family,values in reconciled.items():
                    frame=base.reset_index().copy()
                    frame['family'],frame['count_prediction']=family,values
                    frame['weight_status']=audit['weight_status']
                    frame['residual_blocks']=audit['residual_blocks']
                    frame['coherence_discrepancy']=values-matrix@values[:22]
                    results.append(frame)
    points=pd.concat(results,ignore_index=True)
    assert len(points)==12*5*31*5*3
    assert np.isfinite(points.count_prediction).all() and points.count_prediction.ge(0).all()
    points.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(weights).to_csv(out/'weights.csv',index=False)
    commits={n:sha(out/n) for n in ['independent_predictions.csv','predictions.csv','weights.csv']}
    events.append(dict(event='reconciled_forecasts_committed',time_utc=now(),sha256=commits))
    events.append(dict(event='scoring_started',time_utc=now()))
    (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    frames=[]
    for task in tasks:
        full=node_history(panel,config,task['target'],task['outcome'],2023)
        truth=full.rename_axis('forecast_year').reset_index().melt(id_vars='forecast_year',var_name='node',value_name='observed_count')
        truth['target'],truth['outcome']=task['target'],task['outcome']
        frames.append(truth)
    scored=points.merge(pd.concat(frames),on=['target','outcome','forecast_year','node'],validate='many_to_one')
    assert len(scored)==len(points)
    scored['absolute_count_error']=abs(scored.count_prediction-scored.observed_count)
    scored['signed_count_error']=scored.count_prediction-scored.observed_count
    scored.to_csv(out/'point_scores.csv',index=False)
    summary=scored.groupby(['target','outcome','origin','horizon','family','level'],as_index=False).agg(
        count_mae=('absolute_count_error','mean'),max_coherence_discrepancy=('coherence_discrepancy',lambda x:float(abs(x).max())),
        nodes=('node','nunique'))
    summary.to_csv(out/'summary.csv',index=False)
    for family in ['bottom_up','nonnegative_diagonal_wls']:
        assert abs(scored.loc[scored.family.eq(family),'coherence_discrepancy']).max()<1e-8
    assert all(sha(out/n)==d for n,d in commits.items())
    assert all(sha(ROOT/n)==d for n,d in tests['tested_code_sha256'].items())
    check_lock()
    validation=dict(passed=True,tests=tests['tests_run'],native_node_fits=12*16*31,evaluation_predictions=len(points),
                    fit_fallbacks=sum(a['status']!='ok' for a in audits),weight_fallbacks=int(pd.DataFrame(weights).weight_status.ne('ok').sum()),
                    forecasts_committed_before_final_scoring=True,elapsed_seconds=time.perf_counter()-started)
    (out/'validation_report.json').write_text(json.dumps(validation,indent=2)+'\n')
    events.append(dict(event='scoring_complete',time_utc=now()))
    (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    manifest.update(status='complete',completed_utc=now(),final_period_scored=True,
                    output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='run_manifest.json'})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':
    main()
