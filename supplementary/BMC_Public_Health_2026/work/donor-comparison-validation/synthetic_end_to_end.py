"""Temporary orchestration QA; every fit and verification value is synthetic."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
from concurrent.futures import Future
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
for directory in [ROOT/'src',ROOT/'scripts',ROOT/'tests']:
    sys.path.insert(0,str(directory))
import numpy as np
from gbd_park.pooled import build_examples,target_inputs
import run_donor_comparisons as runner
from test_donors import panel_fixture
from test_primary import scored_fixture

panel=panel_fixture()
reference=scored_fixture()
truth = panel.loc[(panel.location_name == 'Saudi Arabia') &
                  (panel.outcome == 'prevalence'), ['sex', 'age', 'year', 'rate']]
reference = reference.drop(columns=['observed_rate']).merge(
    truth.rename(columns={'year': 'forecast_year', 'rate': 'observed_rate'}),
    on=['sex', 'age', 'forecast_year'], validate='many_to_one')
reference['prediction'] = reference.observed_rate * np.exp(reference.absolute_log_error)
reference['log_prediction'] = np.log(reference.prediction)
reference['absolute_rate_error'] = abs(reference.prediction - reference.observed_rate)
original_arms=runner.arms


class ImmediatePool:
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def submit(self,function,*args,**kwargs):
        future=Future()
        try:future.set_result(function(*args,**kwargs))
        except BaseException as error:future.set_exception(error)
        return future


def fake_fit(config,job,out):
    working,cfg=runner.source_context(panel,config,job)
    target_x,target_y,target_meta=build_examples(working,cfg,job['origin'],[config['primary_target']])
    current_x,levels,current_meta=target_inputs(working,cfg,job['origin'],config['primary_target'])
    change=np.arange(1,6)*(.002+job['base']['channels']*.00001)+job['seed']*.00001
    audit=dict(origin=job['origin'],base=job['base'],seed=job['seed'],device=job['device'],status='ok',reason='',
               countries=job['countries'],maximum_input_year=job['origin']-5,maximum_label_year=job['origin'],
               last_target_label_year=job['origin'],fingerprint_before=job['job_id'],fingerprint_after=job['job_id'])
    return dict(origin=job['origin'],base=job['base'],seed=job['seed'],current_meta=current_meta,target_meta=target_meta,
                levels=levels,target_y=target_y,current_changes=np.tile(change,(len(current_x),1)),
                target_changes=np.tile(change,(len(target_x),1)),audit=audit,cache_job=job)


def synthetic_read(path,*args,**kwargs):
    path=str(path)
    if path.endswith('regional_outcomes.csv'):return panel.copy()
    if path.endswith('primary_v1/point_scores.csv'):return reference.copy()
    raise AssertionError('Unexpected source read: '+path)


with tempfile.TemporaryDirectory(prefix='gbd_donor_synthetic_e2e_') as directory:
    output=Path(directory)/'run'
    with patch.object(runner,'ProcessPoolExecutor',ImmediatePool),patch.object(runner,'cached_fit_job',fake_fit), \
         patch.object(runner,'_reference_manifest',return_value='synthetic_reference_only'), \
         patch.object(runner.pd,'read_csv',side_effect=synthetic_read), \
         patch.object(runner.torch.cuda,'is_available',return_value=True), \
         patch.object(runner,'arms',side_effect=lambda config:[original_arms(config)[0],original_arms(config)[2]]), \
         patch.object(sys,'argv',['run_donor_comparisons.py','--output',str(output),'--workers','1']):
        runner.main()
    print('SYNTHETIC_END_TO_END_PASSED: two arms, mocked fits, real selection/calibration/scoring')
