"""Finish the running ASR/projection queues with independent audits and reporting."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'work/completion-validation'
STATE = WORK / 'pipeline_status.json'
PYTHON = '/home/saif/agpu_env/bin/python'
FROZEN = [
    'scripts/finalize_study.py', 'scripts/report_study_synthesis.py',
    'work/completion-validation/preserved_manifest_hashes.json',
    'work/completion-validation/audit_learning.py', 'work/completion-validation/audit_learning.json',
    'work/learning-curves-validation/audit_global_asr.py',
    'work/global-asr-validation/audit_projections.py',
    'reports/study_synthesis_v1_draft/results.md', 'reports/study_synthesis_v1_draft/validation.json',
    'study_design/completion_sequence_2026-09-30.md',
    *['work/'+name+'-validation/tests.json' for name in ['learning-curves', 'global-asr', 'projections']],
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(mapping):
    for path, expected in mapping.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Changed pipeline input: '+path)


def save(state):
    state['updated_utc'] = now()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2)+'\n')
    temporary.replace(STATE)


def await_run(state, name, log, deadline):
    """Read readiness markers, never training scores, and leave training untouched."""
    state.update(status='waiting', current_stage='wait_'+name)
    save(state)
    print(now(), 'Waiting for completed run and report:', name, flush=True)
    manifest_path = ROOT / 'results' / name / 'run_manifest.json'
    report_path = ROOT / 'reports' / name / 'validation.json'
    while time.monotonic() < deadline:
        log_path = ROOT / log
        if log_path.exists() and 'Traceback (most recent call last):' in log_path.read_text():
            raise RuntimeError('Production error in '+log)
        try:
            manifest = read(manifest_path)
            if manifest['status'] not in ['running', 'complete']:
                raise RuntimeError(name+' has status '+manifest['status'])
            if manifest['status'] == 'complete' and report_path.exists():
                report = read(report_path)
                if not report['passed']:
                    raise RuntimeError('Production report failed: '+name)
                if report.get('run_manifest_sha256', report.get('source_run_manifest_sha256')) != digest(manifest_path):
                    raise ValueError('Report does not match completed run: '+name)
                state.setdefault('training_manifest_sha256', {})[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
                save(state)
                return
        except (FileNotFoundError, json.JSONDecodeError):
            # Producers write their final JSON markers last; tolerate an in-progress write.
            pass
        time.sleep(30)
    raise TimeoutError('Deadline reached for '+name+'; production jobs remain untouched')


def step(state, name, arguments, witness):
    verify(state['frozen_sha256'])
    verify(state.get('training_manifest_sha256', {}))
    previous = state['steps'].get(name)
    if previous and previous['status'] == 'complete':
        verify(previous['witness_sha256'])
        return
    witness_path = ROOT / witness
    if witness_path.exists():
        raise FileExistsError('Unrecorded audit/report requires review: '+witness)
    command = [PYTHON, '-u', *arguments]
    log = WORK / ('pipeline_'+name+'.log')
    state.update(status='running', current_stage=name)
    state['steps'][name] = dict(status='running', command=command, started_utc=now(),
                                log=str(log.relative_to(ROOT)))
    save(state)
    print(now(), 'Starting:', name, flush=True)
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{name} exited {result.returncode}; see {log.relative_to(ROOT)}')
    evidence = read(witness_path)
    if evidence.get('passed') is not True:
        raise ValueError('Missing successful evidence for '+name)
    state['steps'][name].update(status='complete', completed_utc=now(),
                               witness_sha256={witness: digest(witness_path)})
    save(state)
    print(now(), 'Completed:', name, flush=True)


def verify_completion(state):
    verify(state['frozen_sha256'])
    verify(state.get('training_manifest_sha256', {}))
    for record in state['steps'].values():
        if record['status'] != 'complete':
            raise ValueError('Incomplete finalization step')
        verify(record['witness_sha256'])
    report_path = ROOT / 'reports/study_synthesis_v1/validation.json'
    report = read(report_path)
    if report.get('passed') is not True or report['reporter_sha256'] != digest(ROOT / 'scripts/report_study_synthesis.py'):
        raise ValueError('Invalid final report evidence')
    verify(report['source_sha256'])
    artifacts = {'reports/study_synthesis_v1/'+k: v for k, v in report['artifact_sha256'].items()}
    artifacts['reports/study_synthesis_v1/validation.json'] = digest(report_path)
    verify(artifacts)
    verify(read(ROOT / 'work/completion-validation/preserved_manifest_hashes.json')['sha256'])
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-wait-hours', type=float, default=4)
    args = parser.parse_args()
    if not 0 < args.max_wait_hours <= 24:
        raise ValueError('Wait must be positive and at most 24 hours')
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise RuntimeError('Use the agpu Python interpreter')
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists():
            if not args.resume:
                raise FileExistsError('Existing pipeline requires --resume')
            state = read(STATE)
            verify(state['frozen_sha256'])
            if state['status'] == 'complete':
                verify_completion(state)
                print('Completed pipeline verified; no work repeated', flush=True)
                return
        else:
            if args.resume:
                raise FileNotFoundError('No pipeline exists to resume')
            state = dict(created_utc=now(), status='starting', steps={},
                         frozen_sha256={p: digest(ROOT / p) for p in FROZEN})
            save(state)
        try:
            deadline = time.monotonic()+args.max_wait_hours*3600
            await_run(state, 'global_asr_v1', 'work/global-asr-validation/run.log', deadline)
            step(state, 'audit_global_asr', ['work/learning-curves-validation/audit_global_asr.py', '--run', 'results/global_asr_v1'],
                 'work/learning-curves-validation/global_asr_independent_audit.json')
            await_run(state, 'projections_v1', 'work/projections-validation/run.log', deadline)
            step(state, 'audit_projections', ['work/global-asr-validation/audit_projections.py', '--run', 'results/projections_v1', '--workers', '3'],
                 'work/global-asr-validation/projections_independent_audit.json')
            step(state, 'integrated_report', ['scripts/report_study_synthesis.py'], 'reports/study_synthesis_v1/validation.json')
            state['report_artifacts_sha256'] = verify_completion(state)
            state.update(status='complete', current_stage='complete', completed_utc=now(), report='reports/study_synthesis_v1/report.md')
            save(state)
            print(now(), 'All remaining analyses, independent audits and reporting complete', flush=True)
        except BaseException as error:
            state.update(status='failed', error=str(error), failed_utc=now())
            save(state)
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
