"""Wait for active training, then audit, derive burdens and report; never tune models."""
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
WORK = ROOT / 'work/supporting-validation'
STATE = WORK / 'pipeline_status.json'
PYTHON = '/home/saif/agpu_env/bin/python'
FROZEN = [
    'scripts/finalize_supporting.py', 'scripts/report_supporting.py',
    'scripts/run_disability_components.py', 'src/gbd_park/disability_components.py',
    'tests/test_disability_components.py', 'work/supporting-validation/component_tests.json',
    'study_design/supporting_outcomes_implementation.md',
    *['work/supporting-validation/' + name for name in
      ['audit_rates.py', 'audit_history.py', 'audit_components.py']],
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(state):
    state['updated_utc'] = now()
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(STATE)


def verify_files(mapping):
    for name, expected in mapping.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Changed pipeline input: ' + name)


def await_run(state, name, log, deadline):
    state.update(status='waiting', current_stage='wait_' + name)
    save(state)
    print(now(), 'Waiting for completed manifest:', name, flush=True)
    path = ROOT / 'results' / name / 'run_manifest.json'
    while time.monotonic() < deadline:
        if path.exists():
            record = read(path)
            if record['status'] == 'complete':
                state.setdefault('training_manifests_sha256', {})[str(path.relative_to(ROOT))] = digest(path)
                save(state)
                return
            if record['status'] != 'running':
                raise RuntimeError(name + ' has status ' + record['status'])
        log_path = ROOT / log
        if log_path.exists() and 'Traceback (most recent call last):' in log_path.read_text():
            raise RuntimeError('Training error in ' + log)
        time.sleep(30)
    raise TimeoutError('Training wait deadline reached for ' + name + '; existing processes left untouched')


def step(state, name, arguments, witness):
    verify_files(state['frozen_sha256'])
    verify_files(state.get('training_manifests_sha256', {}))
    old = state['steps'].get(name)
    if old and old['status'] == 'complete':
        verify_files(old['witness_sha256'])
        return
    witness_path = ROOT / witness
    # Reuse only a previously recorded successful step. Never overwrite an
    # unrecorded audit/report that might belong to a different invocation.
    if witness_path.exists() and name != 'derive':
        raise FileExistsError('Unrecorded output requires review: ' + witness)
    command = [PYTHON, '-u', *arguments]
    if name == 'derive' and witness_path.exists():
        command.append('--resume')
    log = WORK / ('pipeline_' + name + '.log')
    state.update(status='running', current_stage=name)
    state['steps'][name] = dict(status='running', started_utc=now(), command=command,
                                log=str(log.relative_to(ROOT)))
    save(state)
    print(now(), 'Starting:', name, flush=True)
    with log.open('a') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'{name} exited {result.returncode}; see {log.relative_to(ROOT)}')
    evidence = read(witness_path)
    if name == 'derive':
        assert evidence['status'] == 'complete'
    else:
        assert evidence['passed'] is True
    state['steps'][name].update(status='complete', completed_utc=now(),
                                witness_sha256={witness: digest(witness_path)})
    save(state)
    print(now(), 'Completed:', name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-wait-hours', type=float, default=12)
    args = parser.parse_args()
    assert 0 < args.max_wait_hours <= 24
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise RuntimeError('Use the agpu Python interpreter')
    with (WORK / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if STATE.exists():
            if not args.resume:
                raise FileExistsError('Existing pipeline status requires --resume')
            state = read(STATE)
            verify_files(state['frozen_sha256'])
            if state['status'] == 'complete':
                verify_files(state['report_artifacts_sha256'])
                print('Completed pipeline verified; nothing rerun', flush=True)
                return
        else:
            if args.resume:
                raise FileNotFoundError('No pipeline to resume')
            state = dict(created_utc=now(), status='starting', steps={},
                         frozen_sha256={name: digest(ROOT / name) for name in FROZEN})
            save(state)
        try:
            deadline = time.monotonic() + args.max_wait_hours * 3600
            await_run(state, 'mortality_history_v1', 'work/supporting-validation/history_run_gpu.log', deadline)
            step(state, 'audit_history', ['work/supporting-validation/audit_history.py'],
                 'work/supporting-validation/audit_history.json')
            await_run(state, 'supporting_v1', 'work/supporting-validation/core_run_foreground.log', deadline)
            step(state, 'audit_rates', ['work/supporting-validation/audit_rates.py', '--workers', '4'],
                 'work/supporting-validation/audit_rates.json')
            step(state, 'derive', ['scripts/run_disability_components.py', '--workers', '4'],
                 'results/disability_components_v1/run_manifest.json')
            step(state, 'audit_components', ['work/supporting-validation/audit_components.py', '--workers', '3'],
                 'work/supporting-validation/audit_components.json')
            step(state, 'report', ['scripts/report_supporting.py'], 'reports/supporting_v1/validation.json')
            verify_files(state['frozen_sha256'])
            report = read(ROOT / 'reports/supporting_v1/validation.json')
            state['report_artifacts_sha256'] = {'reports/supporting_v1/' + k: v for k, v in report['artifact_sha256'].items()}
            state['report_artifacts_sha256']['reports/supporting_v1/validation.json'] = digest(ROOT / 'reports/supporting_v1/validation.json')
            verify_files(state['report_artifacts_sha256'])
            state.update(status='complete', current_stage='complete', completed_utc=now(),
                         report='reports/supporting_v1/report.md')
            save(state)
            print(now(), 'All supporting analyses, audits and reporting complete', flush=True)
        except BaseException as error:
            state.update(status='failed', error=str(error), failed_utc=now())
            save(state)
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
