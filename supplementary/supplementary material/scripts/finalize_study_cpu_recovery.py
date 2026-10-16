"""Audit the CPU execution recovery and publish a separately versioned synthesis."""
import argparse
import fcntl
from pathlib import Path
import sys
import time
import traceback

import finalize_study as flow

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work/global-asr-cpu-recovery-validation'
flow.WORK = WORK
flow.STATE = WORK / 'pipeline_status.json'
REPORT = 'reports/study_synthesis_v1_1'
FROZEN = [
    'scripts/finalize_study_cpu_recovery.py', 'scripts/finalize_study.py',
    'scripts/run_global_asr_cpu_recovery.py', 'scripts/report_study_synthesis_recovery.py',
    'scripts/report_study_synthesis.py',
    'work/learning-curves-validation/audit_global_asr.py',
    'work/global-asr-cpu-recovery-validation/preserved_sha256.json',
    'work/global-asr-cpu-recovery-validation/tests.json',
    'work/global-asr-cpu-recovery-validation/reporter_tests.json',
    'work/completion-validation/global_gpu_failure_review.json',
    'reports/study_synthesis_v1_review_notice.md',
]


def preserve_prior():
    flow.verify(flow.read(WORK / 'preserved_sha256.json')['sha256'])


def accept_independent_audit(state):
    """The recovery runner invokes the frozen independent auditor once before reporting."""
    flow.verify(state['frozen_sha256'])
    flow.verify(state['training_manifest_sha256'])
    path = WORK / 'independent_audit.json'
    audit = flow.read(path)
    manifest_path = ROOT / 'results/global_asr_cpu_recovery_v1/run_manifest.json'
    report = flow.read(ROOT / 'reports/global_asr_cpu_recovery_v1/validation.json')
    if (audit.get('passed') is not True or audit['source_checkpoint_replays'] != 216
            or audit['forecast_rows'] != 2640 or len(audit['cases']) != 12
            or audit['run_manifest_sha256'] != flow.digest(manifest_path)
            or audit['audit_code_sha256'] != flow.digest(ROOT / 'work/learning-curves-validation/audit_global_asr.py')
            or report['audit_sha256'] != flow.digest(path) or report['fallback_cells'] != 0):
        raise ValueError('CPU recovery requires complete independent checkpoint replay and no fallbacks')
    record = dict(status='complete', completed_utc=flow.now(),
                  performed_by='recovery runner invoking frozen independent auditor',
                  witness_sha256={str(path.relative_to(ROOT)): flow.digest(path)})
    old = state['steps'].get('independent_audit')
    if old:
        flow.verify(old['witness_sha256'])
    else:
        state['steps']['independent_audit'] = record
        flow.save(state)


def verify_completion(state):
    flow.verify(state['frozen_sha256'])
    flow.verify(state.get('training_manifest_sha256', {}))
    for record in state['steps'].values():
        if record['status'] != 'complete':
            raise ValueError('Incomplete CPU recovery finalization step')
        flow.verify(record['witness_sha256'])
    report_path = ROOT / REPORT / 'validation.json'
    report = flow.read(report_path)
    if report.get('passed') is not True:
        raise ValueError('Revised synthesis has not passed validation')
    if report['reporter_sha256'] != flow.digest(ROOT / 'scripts/report_study_synthesis_recovery.py'):
        raise ValueError('Revised synthesis reporter identity changed')
    flow.verify(report['source_sha256'])
    artifacts = {REPORT+'/'+k: v for k, v in report['artifact_sha256'].items()}
    artifacts[REPORT+'/validation.json'] = flow.digest(report_path)
    flow.verify(artifacts)
    preserve_prior()
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(flow.PYTHON).resolve():
        raise RuntimeError('Use the agpu Python interpreter')
    with (WORK / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if flow.STATE.exists():
            if not args.resume:
                raise FileExistsError('Existing recovery pipeline requires --resume')
            state = flow.read(flow.STATE)
            flow.verify(state['frozen_sha256'])
            if state['status'] == 'complete':
                verify_completion(state)
                print('Completed CPU recovery pipeline verified; nothing rerun', flush=True)
                return
        else:
            if args.resume:
                raise FileNotFoundError('No CPU recovery pipeline exists to resume')
            for name in ['tests.json', 'reporter_tests.json']:
                if flow.read(WORK / name).get('passed') is not True:
                    raise ValueError('Recovery test gate failed: '+name)
            preserve_prior()
            state = dict(created_utc=flow.now(), status='starting', steps={},
                         frozen_sha256={p: flow.digest(ROOT / p) for p in FROZEN})
            flow.save(state)
        try:
            flow.await_run(state, 'global_asr_cpu_recovery_v1',
                           'work/global-asr-cpu-recovery-validation/run.log', time.monotonic()+4*3600)
            preserve_prior()
            accept_independent_audit(state)
            preserve_prior()
            flow.step(state, 'revised_synthesis', ['scripts/report_study_synthesis_recovery.py'], REPORT+'/validation.json')
            state['report_artifacts_sha256'] = verify_completion(state)
            state.update(status='complete', current_stage='complete', completed_utc=flow.now(), report=REPORT+'/report.md')
            flow.save(state)
            print(flow.now(), 'CPU recovery, independent audit and revised synthesis complete', flush=True)
        except BaseException as error:
            state.update(status='failed', error=str(error), failed_utc=flow.now())
            flow.save(state)
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
