"""Fit the frozen exploratory age-time comparator; commit before scoring."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import scipy
from gbd_park.bayesian_age_time import fit, mixture_quantiles, joint_log_draws, training_matrix

PROTOCOL=ROOT/'study_design/bounded_extension_2026-10-01.json'
LOCK=ROOT/'study_design/bounded_extension_2026-10-01_lock.json'
PANEL=ROOT/'data/processed/design_v1/regional_outcomes.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def fit_job(job, protocol, output):
    target,outcome,sex,origin=job
    before=time.perf_counter()
    panel=pd.read_csv(PANEL)
    history=training_matrix(panel,target,outcome,sex,origin,protocol['ages'])
    model=fit(history,protocol['model'],protocol['horizons'])
    probabilities=protocol['model']['quantiles']
    quantiles=mixture_quantiles(model,probabilities)
    prefix=dict(target=target,outcome=outcome,sex=sex,origin=origin,
                family='bayesian_age_time',setting_id='fixed_discrete_hyperprior_v1',
                history_start=1990,history_end=origin,status='ok',fallback_reason='')
    points,intervals=[],[]
    for hi,h in enumerate(protocol['horizons']):
        for ai,age in enumerate(protocol['ages']):
            q=dict(zip(probabilities,quantiles[hi,ai]))
            context=dict(prefix,age=age,horizon=h,forecast_year=origin+h)
            points.append(dict(context,log_prediction=q[.5],prediction=float(np.exp(q[.5]))))
            if origin in protocol['evaluation_origins']:
                for level in protocol['evaluation']['interval_levels']:
                    alpha=(1-level)/2
                    lower=quantiles[hi,ai,np.argmin(np.abs(np.array(probabilities)-alpha))]
                    upper=quantiles[hi,ai,np.argmin(np.abs(np.array(probabilities)-(1-alpha)))]
                    for scale in ['rate','log_rate']:
                        fn=np.exp if scale=='rate' else lambda v:v
                        intervals.append(dict(context,level=level,scale=scale,
                            median=float(fn(q[.5])),lower=float(fn(lower)),upper=float(fn(upper)),
                            procedure='native_model_posterior',uncertainty_scope='conditional_GBD_point_estimate_prediction'))
    if origin in protocol['evaluation_origins']:
        ident='__'.join(map(str,job)).replace(' ','_')
        seed=protocol['model']['seed']+int(hashlib.sha256(ident.encode()).hexdigest()[:8],16)
        draws=joint_log_draws(model,protocol['model']['joint_posterior_draws_per_origin_sex'],seed)
        np.savez_compressed(Path(output)/'posterior_paths'/f'{ident}.npz',log_draws=draws,
                            age=np.array(protocol['ages']),horizon=np.array(protocol['horizons']))
    weights=pd.DataFrame(model['grid'],columns=['age_lengthscale','observation_sd','level_sd','slope_sd','damping'])
    weights['posterior_weight']=model['weights'];weights['log_likelihood']=model['log_likelihood']
    for k in ['target','outcome','sex','origin']:weights[k]=prefix[k]
    audit=dict(prefix,training_rows=history.size,training_sha256=hashlib.sha256(history.tobytes()).hexdigest(),
               component_count=len(model['grid']),effective_components=float(1/np.sum(model['weights']**2)),
               maximum_weight=float(model['weights'].max()),elapsed_seconds=time.perf_counter()-before,
               future_outcomes_used=False)
    return points,intervals,weights,audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--output',default='results/bayesian_age_time_v1');args=parser.parse_args()
    output=ROOT/args.output
    if output.exists():raise FileExistsError('Existing run is immutable; choose a new run name for a documented correction')
    if not 1<=args.workers<=20:raise ValueError('Use 1–20 workers')
    protocol=json.loads(PROTOCOL.read_text());lock=json.loads(LOCK.read_text())
    for name,digest in lock['sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen input changed: '+name)
    validation=subprocess.run([sys.executable,str(ROOT/'tests/test_bayesian_age_time.py')],capture_output=True,text=True)
    if validation.returncode:raise RuntimeError(validation.stdout+validation.stderr)
    output.mkdir();(output/'posterior_paths').mkdir()
    (output/'numerical_tests.txt').write_text(validation.stdout+validation.stderr)
    source_files=[PROTOCOL,LOCK,PANEL,Path(__file__).resolve(),ROOT/'src/gbd_park/bayesian_age_time.py',ROOT/'tests/test_bayesian_age_time.py']
    countries=[protocol['target']]+protocol['replications']
    jobs=[(t,o,s,y) for t in countries for o in protocol['outcomes'] for s in protocol['sexes'] for y in protocol['fit_origins']]
    manifest=dict(status='running',started_utc=now(),workers=args.workers,jobs=len(jobs),
                  scientific_status='exploratory_after_original_outcomes_inspected',
                  original_primary_results_modified=False,
                  input_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
                  versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__))
    write_json(output/'run_manifest.json',manifest)
    completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fit_job,j,protocol,output) for j in jobs]
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed)%24==0:print(f'Completed {len(completed)}/{len(jobs)} fits',flush=True)
    keys=['target','outcome','origin','sex','age','horizon']
    predictions=pd.DataFrame([r for x in completed for r in x[0]]).sort_values(keys)
    intervals=pd.DataFrame([r for x in completed for r in x[1]]).sort_values(keys+['scale','level'])
    predictions.to_csv(output/'predictions.csv',index=False)
    intervals.to_csv(output/'native_intervals.csv',index=False)
    pd.concat([x[2] for x in completed],ignore_index=True).sort_values(['target','outcome','origin','sex','age_lengthscale','observation_sd','level_sd','slope_sd','damping']).to_csv(output/'component_weights.csv',index=False)
    pd.DataFrame([x[3] for x in completed]).sort_values(['target','outcome','origin','sex']).to_csv(output/'fit_audit.csv',index=False)
    assert len(predictions)==len(jobs)*len(protocol['ages'])*len(protocol['horizons'])
    assert not predictions.duplicated(keys).any()
    assert len(intervals)==6*2*2*5*11*5*3*2
    for name,digest in manifest['input_sha256'].items():assert sha(ROOT/name)==digest
    manifest.update(status='predictions_committed_unscored',finished_utc=now(),prediction_rows=len(predictions),
                    interval_rows=len(intervals),all_fits_successful=True,
                    prediction_sha256=sha(output/'predictions.csv'),native_intervals_sha256=sha(output/'native_intervals.csv'),
                    posterior_paths=120,output_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='run_manifest.json'})
    write_json(output/'run_manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','jobs','prediction_rows','interval_rows','finished_utc']}),flush=True)


if __name__=='__main__':main()
