#!/usr/bin/env python3
"""Bounded concurrent compatibility clients for an already-owned batch4 server.

Three serial phases each launch four clients concurrently: unchanged auto9
(36 checks), four original forced-choice cases in both modes(8 checks), and
additional required/named literal-tag reasoning cases(4 checks). This tests
cross-request isolation; process overlap is not proof of a particular GPU batch
shape or a throughput benchmark. Never starts, changes or stops the server.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import types
import uuid

HELPER_SHA = '48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586'
AUTO_NAMES = {'preflight_one_token_and_model'} | {
    f'auto_{variant}_{mode}' for variant in ('plain', 'reasoning', 'client_grammar', 'client_json_schema')
    for mode in ('nonstream', 'stream')}
FORCED = ('required', 'named', 'required_adversarial', 'named_adversarial')
EXPECTED = 48


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError('Frozen owned-process helper changed; review before running')
    module = types.ModuleType('reasoning_concurrency_owned_helper')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def validate_metadata(metadata, args):
    if metadata.get('engine', {}).get('commit') != args.expected_engine:
        raise ValueError('Metadata engine differs from expected source')
    if metadata.get('server', {}).get('commit') != args.expected_server:
        raise ValueError('Metadata Tabby differs from expected source')
    values = metadata['config']['values']
    if values['model'].get('max_batch_size') != 4:
        raise ValueError('This diagnostic requires recorded max_batch_size=4')
    if values['model'].get('model_name') != args.model:
        raise ValueError('Metadata served model differs from requested alias')
    if values['draft_model'].get('draft_mode') != 'mtp':
        raise ValueError('This diagnostic requires MTP serving')
    network = values['network']
    if network.get('host') != '127.0.0.1' or network.get('port') != 8899:
        raise ValueError('This scheduled diagnostic requires the owned loopback8899 server')
    if args.base_url.rstrip('/') != 'http://127.0.0.1:8899/v1':
        raise ValueError('Base URL differs from the owned loopback listener')


def start_identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    if fields[0] == 'Z':
        raise RuntimeError('Owned server has exited')
    return fields[19]


def commands(args, cases_dir):
    common = ['--base-url', args.base_url, '--model', args.model]
    metadata = ['--metadata', str(args.metadata)]
    py = sys.executable
    phase1 = []
    for index in range(1, 5):
        name = f'auto{index}'
        phase1.append({'name': name, 'kind': 'auto', 'expected': sorted(AUTO_NAMES),
                       'command': [py, str(args.recipe / 'bench/auto_compatibility.py'),
                                   *common, *metadata, '--timeout', str(args.timeout),
                                   '--label', args.label + '-' + name,
                                   '--api-key-env', args.api_key_env,
                                   '--output', str(cases_dir / (name + '.json'))]})
    phase2 = []
    for case in FORCED:
        name = 'forced-' + case
        phase2.append({'name': name, 'kind': 'tools',
                       'expected': [case + '/' + mode for mode in ('nonstream', 'stream')],
                       'command': [py, str(args.recipe / 'bench/tool_smoke.py'), *common,
                                   '--case', case, '--mode', 'both', '--repeat', '1',
                                   '--timeout', str(args.timeout), '--max-tokens', '1024',
                                   '--output', str(cases_dir / (name + '.json'))]})
    phase3 = []
    for choice in ('required', 'named'):
        for mode in ('nonstream', 'stream'):
            name = f'literal-{choice}-{mode}'
            phase3.append({'name': name, 'kind': 'literal',
                           'expected': [f'literal_reasoning_{choice}_{mode}'],
                           'command': [py, str(Path(__file__).with_name('reasoning_literal_smoke.py')),
                                       '--recipe', str(args.recipe), *common, *metadata,
                                       '--choice', choice, '--mode', mode,
                                       '--timeout', str(args.timeout), '--label', args.label + '-' + name,
                                       '--api-key-env', args.api_key_env,
                                       '--output', str(cases_dir / (name + '.json'))]})
    return [('auto4', phase1), ('forced4', phase2), ('literal4', phase3)]


def summarize_child(spec, report, exit_code):
    errors = []
    if spec['kind'] == 'tools':
        raw_rows = report.get('results', [])
        rows = [{'name': str(row.get('case')) + '/' + str(row.get('mode')),
                 'status': 'pass' if row.get('passed') is True else 'fail',
                 'errors': row.get('errors', []), 'repeat_index': row.get('repeat_index')}
                for row in raw_rows]
        if any(row['repeat_index'] != 0 for row in rows):
            errors.append('Tool report repeat index differs from0')
        done = report.get('completed_at_utc')
        summary = report.get('summary', {})
        counts = Counter(row['status'] for row in rows)
        if (summary.get('total') != len(rows) or summary.get('passed') != counts['pass']
                or summary.get('failed') != counts['fail']):
            errors.append('Tool summary inconsistent with recorded cases')
        if report.get('errors'):
            errors.append('Tool report contains setup errors')
    else:
        raw_rows = report.get('cases', [])
        rows = [{'name': row.get('name'), 'status': row.get('status'),
                 **({'error': row['error']} if 'error' in row else {})} for row in raw_rows]
        done = report.get('finished_utc')
        counts = Counter(row['status'] for row in rows)
        summary = report.get('summary', {})
        if any(summary.get(key, 0) != counts[key] for key in ('pass', 'fail')):
            errors.append('Diagnostic summary inconsistent with recorded cases')
        if report.get('setup_error'):
            errors.append('Diagnostic contains a setup error')
    names = [row['name'] for row in rows]
    if (len(names) != len(set(names)) or set(names) != set(spec['expected'])
            or len(names) != len(spec['expected'])):
        errors.append('Missing, duplicate or unexpected cases')
    if any(row['status'] not in ('pass', 'fail') for row in rows):
        errors.append('Invalid case status')
    if not done:
        errors.append('Report lacks completion marker')
    if exit_code != 0 and not counts['fail']:
        errors.append(f'Client exited {exit_code} without a recorded failed case')
    passed = exit_code == 0 and not errors and counts['pass'] == len(spec['expected'])
    return rows, {'expected_checks': len(spec['expected']), 'observed_checks': len(rows),
                  'counts': dict(counts), 'errors': errors, 'passed': passed}



def cleanup_clients(helper, entries, owner, timeout=35):
    """Stop all owned client sessions under one deadline, never the server.

    Verify and signal every surviving group before waiting. A failed attempt is
    recorded and is never repeated with another full deadline by outer cleanup.
    """
    selected = [(process, log, item) for process, log, item in entries
                if not item.get('cleanup_attempted')]
    if not selected:
        return
    started = time.monotonic()
    deadline = started + timeout
    errors = []
    for _, _, item in selected:
        item['cleanup_attempted'] = True
        item['cleanup'] = {'signals': []}
    blocked = set()
    def members(process, item):
        process.poll()
        try:
            return helper.group_members(process, owner)
        except Exception as exc:
            if process.pid not in blocked:
                errors.append(f'Client {process.pid} ownership check: {type(exc).__name__}: {exc}')
                blocked.add(process.pid)
            return []
    for sig, phase_end in ((signal.SIGTERM, deadline - min(5, timeout / 3)),
                           (signal.SIGKILL, deadline)):
        for process, _, item in selected:
            if process.pid in blocked or not members(process, item):
                continue
            try:
                os.killpg(process.pid, sig)
                item['cleanup']['signals'].append(sig.name)
            except ProcessLookupError:
                pass  # The verified group exited between inspection and signal.
            except OSError as exc:
                errors.append(f'Client {process.pid} signal: {type(exc).__name__}: {exc}')
        while True:
            active = [process.pid for process, _, item in selected
                      if process.pid not in blocked and members(process, item)]
            if not active or time.monotonic() >= phase_end:
                break
            time.sleep(min(.1, max(0, phase_end - time.monotonic())))
    for process, log, item in selected:
        remaining = members(process, item) if process.pid not in blocked else None
        item['owned_client_group_empty'] = remaining == [] and process.pid not in blocked
        code = process.poll()
        if code is None and item['owned_client_group_empty']:
            try:
                code = process.wait(timeout=max(.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                errors.append(f'Client {process.pid} was not reaped before cleanup deadline')
        item['cleanup'].update(exit_code=code, wall_seconds=time.monotonic() - started,
                               owned_group_empty=item['owned_client_group_empty'])
        if not item['owned_client_group_empty']:
            errors.append(f'Client {process.pid} group is not verified empty after cleanup')
        log.close()
    if errors:
        raise RuntimeError('; '.join(errors))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8899/v1')
    ap.add_argument('--model', required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--label', default='reasoning-batch4')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--server-pid', type=int, required=True)
    ap.add_argument('--expected-engine', required=True)
    ap.add_argument('--expected-server', required=True)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--phase-timeout', type=float, default=600)
    ap.add_argument('--api-key-env', default='TABBY_API_KEY')
    args = ap.parse_args(argv)
    if not 0 < args.timeout <= 300 or not 0 < args.phase_timeout <= 900:
        ap.error('Timeout bounds are request(0,300], phase(0,900]')
    if args.server_pid <= 1:
        ap.error('--server-pid must identify the already-owned server')
    for ref in (args.expected_engine, args.expected_server):
        if len(ref) != 40 or any(c not in '0123456789abcdef' for c in ref):
            ap.error('Expected source refs must be full lowercase40-character SHA1s')
    os.umask(0o077)
    sys.path.insert(0, str((args.recipe / 'bench').resolve()))
    from api_resilience import ReportWriter, read_metadata
    helper = load_helper(Path(__file__).with_name('spark_experiment_controller.py'))
    metadata, metadata_source = read_metadata(args.metadata)
    writer = ReportWriter(args.output)
    cases_dir = args.output.with_name(args.output.stem + '-cases')
    record = {'schema_version': 1, 'kind': 'reasoning_concurrent_compatibility', 'label': args.label,
              'started_utc': stamp(), 'scope': __doc__, 'base_url': args.base_url,
              'requested_model': args.model, 'provenance': metadata, 'metadata_source': metadata_source,
              'expected_checks': EXPECTED, 'cases': [], 'client_reports': [], 'phases': [],
              'passed': False, 'child_reports_directory': str(cases_dir), 'errors': []}
    def save():
        record['summary'] = dict(Counter(row['status'] for row in record['cases']))
        writer.write(record)
    save()  # Claim this exact report before any child or network activity.
    all_processes = []
    owner = uuid.uuid4().hex
    interruption_seen = False
    cleanup_active = False
    launch_active = False
    def interrupted(signum, frame):
        nonlocal interruption_seen
        if not interruption_seen:
            interruption_seen = True
            if not cleanup_active and not launch_active:
                raise KeyboardInterrupt(f'signal {signum}')
        # Repeated TERM/INT must not interrupt the bounded owned cleanup.
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous_handlers:
        signal.signal(sig, interrupted)
    try:
        cases_dir.mkdir(mode=0o700)
        validate_metadata(metadata, args)
        server_start = start_identity(args.server_pid)
        config_path = Path(metadata['config']['path'])
        files = [Path(__file__), Path(__file__).with_name('reasoning_literal_smoke.py'),
                 Path(__file__).with_name('spark_experiment_controller.py'), args.metadata, config_path]
        files += [args.recipe / 'bench' / name for name in
                  ('api_resilience.py', 'auto_compatibility.py', 'tool_smoke.py', 'api_client.py', 'bench_v1.py')]
        fingerprints = {str(path): sha(path) for path in files}
        if fingerprints[str(config_path)] != metadata['config']['sha256']:
            raise ValueError('Live configuration hash differs from deployment sidecar')
        record.update(server_pid=args.server_pid, server_start_identity=server_start,
                      source_hashes=fingerprints, client_owner_token=owner)
        def verify():
            if start_identity(args.server_pid) != server_start:
                raise RuntimeError('Owned server PID was replaced')
            helper.require_owned_listener(args.server_pid)
            for path, fingerprint in fingerprints.items():
                if sha(path) != fingerprint:
                    raise ValueError(f'Validation source or metadata changed: {path}')
        verify()
        environment = dict(os.environ, QWEN_EXPERIMENT_OWNER=owner)
        # tool_smoke reads API_KEY; the other clients read the explicitly named env.
        if os.environ.get(args.api_key_env):
            environment['API_KEY'] = os.environ[args.api_key_env]
        else:
            environment.pop('API_KEY', None)
        for phase_name, specs in commands(args, cases_dir):
            verify()
            phase = {'name': phase_name, 'started_utc': stamp(), 'clients': []}
            record['phases'].append(phase)
            running = []
            deadline = time.monotonic() + args.phase_timeout
            try:
                for spec in specs:
                    log_path = cases_dir / (spec['name'] + '.log')
                    log = log_path.open('xb')
                    # Defer a signal until Popen's returned session is registered.
                    # No signal mask is inherited by the subprocess.
                    launch_active = True
                    try:
                        process = subprocess.Popen(spec['command'], cwd=args.recipe,
                                                   env=environment, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        item = {'name': spec['name'], 'pid': process.pid, 'started_utc': stamp(),
                                'command': spec['command'], 'log': str(log_path),
                                'report': spec['command'][-1]}
                        running.append((process, log, spec, item))
                        all_processes.append((process, log, item))
                        phase['clients'].append(item)
                    except BaseException:
                        log.close()
                        raise
                    finally:
                        launch_active = False
                    if interruption_seen:
                        raise KeyboardInterrupt('signal received during client launch')
                save()
                while any(process.poll() is None for process, _, _, _ in running):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f'{phase_name} exceeded its bounded client deadline')
                    time.sleep(.1)
                for process, log, spec, item in running:
                    log.close()
                    item.update(exit_code=process.returncode, finished_utc=stamp())
                    report_path = Path(item['report'])
                    try:
                        child = json.loads(report_path.read_text())
                        rows, summary = summarize_child(spec, child, process.returncode)
                        item.update(summary, report_sha256=sha(report_path))
                        for row in rows:
                            record['cases'].append({**row, 'name': spec['name'] + '/' + str(row['name']),
                                                    'child_report': str(report_path)})
                        if summary['errors']:
                            record['errors'].append(spec['name'] + ': ' + '; '.join(summary['errors']))
                    except Exception as exc:
                        item.update(passed=False, error=f'{type(exc).__name__}: {exc}')
                        record['errors'].append(spec['name'] + ': ' + item['error'])
                    record['client_reports'].append(item)
                verify()
                phase['finished_utc'] = stamp()
                save()
            finally:
                cleanup_active = True
                try:
                    cleanup_clients(helper, [(process, log, item) for process, log, _, item in running], owner)
                finally:
                    cleanup_active = False
                if interruption_seen:
                    raise KeyboardInterrupt('signal received; owned client cleanup completed')
        verify()
    except BaseException as exc:
        record['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        cleanup_active = True
        try:
            cleanup_clients(helper, all_processes, owner)
        except Exception as exc:
            record['errors'].append(f'Owned client cleanup failed: {type(exc).__name__}: {exc}')
        for _, log, _ in all_processes:
            log.close()
        record['owned_client_groups_empty'] = all(item.get('owned_client_group_empty') is True
                                                  for _, _, item in all_processes)
        record['cleanup_failed'] = not record['owned_client_groups_empty']
        rows = record['cases']
        names = [row['name'] for row in rows]
        record['passed'] = (len(rows) == EXPECTED and len(set(names)) == EXPECTED
                            and all(row['status'] == 'pass' for row in rows)
                            and len(record['client_reports']) == 12
                            and all(item.get('passed') and item.get('owned_client_group_empty')
                                    for item in record['client_reports']) and not record['errors'])
        if record['errors']:
            record['setup_error'] = '; '.join(record['errors'])
        record['finished_utc'] = stamp()
        try:
            save()
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
    print(json.dumps({'output': str(args.output), 'summary': record['summary'],
                      'expected_checks': EXPECTED, 'passed': record['passed']}))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
