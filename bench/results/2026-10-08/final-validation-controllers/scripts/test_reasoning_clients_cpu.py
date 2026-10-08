"""No-network tests for extra reasoning clients, reports and child ownership."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import signal
import io
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
RECIPE = BASE / 'recipe'
sys.path.insert(0, str(RECIPE / 'bench'))
import api_resilience
import reasoning_literal_smoke as literal
import reasoning_concurrency_smoke as concurrent


def frozen_client(name):
    archive = BASE / 'tabbyapi-diagnostics/reasoning-clients-pre-unbudgeted'
    manifest = json.loads((archive / 'manifest.json').read_text())
    raw = (archive / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['files'][name]['sha256']:
        raise AssertionError('Frozen default client changed')
    module = ModuleType('frozen_' + name.replace('.', '_'))
    # Preserve the original live script location used to form child commands.
    module.__file__ = str(BASE / name)
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


def metadata(path, batch=4):
    cfg = path / 'config.yml'
    cfg.write_text('synthetic configuration\n')
    return {'engine': {'commit': 'a' * 40}, 'server': {'commit': 'b' * 40},
            'config': {'path': str(cfg), 'sha256': hashlib.sha256(cfg.read_bytes()).hexdigest(),
                       'values': {'model': {'max_batch_size': batch, 'model_name': 'model'},
                                  'draft_model': {'draft_mode': 'mtp'},
                                  'network': {'host': '127.0.0.1', 'port': 8899}}}}


def response_trace(request, text=literal.LITERAL):
    message = {'role': 'assistant', 'reasoning_content': 'brief reasoning', 'content': None,
               'tool_calls': [{'id': 'call-fixture', 'type': 'function',
                               'function': {'name': 'record_text', 'arguments': json.dumps({'text': text})}}]}
    usage = {'prompt_tokens': 12, 'completion_tokens': 8, 'total_tokens': 20}
    body = {'id': 'response-fixture', 'model': request['model'], 'usage': usage,
            'choices': [{'index': 0, 'message': message, 'finish_reason': 'tool_calls'}]}
    trace = {'status': 200, 'stream': request['stream'], 'request': request,
             'frames': [], 'done': bool(request['stream']), 'body': body, 'wall_seconds': .1}
    if request['stream']:
        delta = deepcopy(message)
        delta['tool_calls'][0]['index'] = 0
        body['choices'][0] = {'index': 0, 'delta': delta, 'finish_reason': 'tool_calls'}
        trace['frames'] = [body]
    return trace


class LiteralTests(unittest.TestCase):
    def test_four_new_requests_have_budget_thinking_and_literal_string_with_distinct_choices_modes(self):
        names = literal.expected_names()
        self.assertEqual(len(names), 4)
        self.assertEqual(len(set(names)), 4)
        for name in names:
            _, _, choice, mode = name.split('_')
            data = literal.payload('model', choice, mode == 'stream')
            self.assertEqual(data['reasoning_budget_tokens'], 24)
            self.assertTrue(data['enable_thinking'])
            self.assertEqual(data['max_tokens'], 256)
            self.assertIn(json.dumps(literal.LITERAL), data['messages'][0]['content'])
            self.assertFalse(data['parallel_tool_calls'])
            result = api_resilience.assembled(response_trace(data), 'model')
            self.assertIs(literal.validate(result, api_resilience.require), result)

    def test_default_payload_bytes_match_the_preserved_original_in_all_four_modes(self):
        previous = frozen_client('reasoning_literal_smoke.py')
        for choice in ('required', 'named'):
            for stream in (False, True):
                self.assertEqual(json.dumps(literal.payload('model', choice, stream)),
                                 json.dumps(previous.payload('model', choice, stream)))

    def test_unbudgeted_payload_removes_only_the_single_budget_field(self):
        for choice in ('required', 'named'):
            for stream in (False, True):
                original = literal.payload('model', choice, stream)
                observed = literal.payload('model', choice, stream, unbudgeted=True)
                self.assertNotIn('reasoning_budget_tokens', observed)
                original.pop('reasoning_budget_tokens')
                self.assertEqual(json.dumps(observed), json.dumps(original))

    def test_unbudgeted_cli_runs_four_original_strict_checks_without_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, calls = self.run_client(Path(tmp), unbudgeted=True)
            self.assertEqual(code, 0)
            self.assertTrue(report['unbudgeted'])
            self.assertEqual(report['summary'], {'pass': 4})
            self.assertEqual(len(calls), 4)
            for call, name in zip(calls, literal.expected_names()):
                _, _, choice, mode = name.split('_')
                original = frozen_client('reasoning_literal_smoke.py').payload('model', choice, mode == 'stream')
                original.pop('reasoning_budget_tokens')
                self.assertEqual(call, original)

    def test_wrong_literal_type_reasoning_finish_or_prose_fails(self):
        data = literal.payload('model', 'required', False)
        good = api_resilience.assembled(response_trace(data), 'model')
        for mutate in (
            lambda r: r.update(reasoning_content=''),
            lambda r: r.update(content='unfinished thought'),
            lambda r: r.update(finish_reason='length'),
            lambda r: r['tool_calls'][0]['function'].update(arguments='{"text": 3}'),
            lambda r: r['tool_calls'][0]['function'].update(arguments='{"text": "</think>"}'),
            lambda r: r['tool_calls'][0]['function'].update(name='other'),
        ):
            result = deepcopy(good)
            mutate(result)
            with self.assertRaises(api_resilience.CheckError):
                literal.validate(result, api_resilience.require)

    def run_client(self, directory, incorrect=False, unbudgeted=False):
        meta = directory / 'metadata.json'
        meta.write_text(json.dumps(metadata(directory)))
        output = directory / 'literal.json'
        calls = []
        def request(client, route, data):
            self.assertTrue(output.exists())
            self.assertEqual(route, '/chat/completions')
            calls.append(deepcopy(data))
            trace = response_trace(data, text='wrong' if incorrect else literal.LITERAL)
            client.traces.append(trace)
            return trace
        with patch.object(api_resilience.Client, 'request', request):
            code = literal.main(['--recipe', str(RECIPE), '--model', 'model',
                                 '--metadata', str(meta), '--output', str(output),
                                 *(['--unbudgeted'] if unbudgeted else [])])
        return code, json.loads(output.read_text()), calls

    def test_real_assembly_reports_all_four_actual_cases_and_preserves_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, calls = self.run_client(Path(tmp))
            self.assertEqual(code, 0)
            self.assertEqual(report['summary'], {'pass': 4})
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(report['requests']), 4)
            self.assertTrue(report['finished_utc'])
            self.assertEqual(report['cases'][0]['result']['usage']['total_tokens'], 20)

    def test_semantic_failure_is_not_hidden_and_remaining_cases_still_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, calls = self.run_client(Path(tmp), incorrect=True)
            self.assertEqual(code, 1)
            self.assertEqual(report['summary'], {'fail': 4})
            self.assertEqual(len(calls), 4)

    def test_existing_report_is_refused_before_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            meta = directory / 'metadata.json'
            meta.write_text('{}')
            output = directory / 'existing.json'
            output.write_text('preserve')
            with patch.object(api_resilience.Client, 'request', side_effect=AssertionError('no HTTP')) as request:
                with self.assertRaises(FileExistsError):
                    literal.main(['--recipe', str(RECIPE), '--model', 'model',
                                  '--metadata', str(meta), '--output', str(output)])
                request.assert_not_called()
            self.assertEqual(output.read_text(), 'preserve')


class CommandAndSummaryTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(base_url='http://127.0.0.1:8899/v1', model='model',
                               metadata=Path('/synthetic/deployment.json'), recipe=RECIPE,
                               timeout=180, label='label', api_key_env='TABBY_API_KEY')

    def test_three_phases_four_clients_each_and_exact48_expected_checks(self):
        phases = concurrent.commands(self.args(), Path('/synthetic/cases'))
        self.assertEqual([name for name, _ in phases], ['auto4', 'forced4', 'literal4'])
        self.assertEqual([len(specs) for _, specs in phases], [4, 4, 4])
        self.assertEqual(sum(len(spec['expected']) for _, specs in phases for spec in specs), 48)
        forced = [spec for _, specs in phases for spec in specs if spec['kind'] == 'tools']
        self.assertEqual({spec['command'][spec['command'].index('--case') + 1] for spec in forced},
                         {'required', 'named', 'required_adversarial', 'named_adversarial'})
        self.assertEqual(len({spec['command'][-1] for _, specs in phases for spec in specs}), 12)

    def test_default_commands_are_identical_to_the_preserved_original(self):
        previous = frozen_client('reasoning_concurrency_smoke.py')
        args = self.args()
        self.assertEqual(concurrent.commands(args, Path('/synthetic/cases')),
                         previous.commands(args, Path('/synthetic/cases')))
        args.include_unbudgeted = False
        self.assertEqual(concurrent.commands(args, Path('/synthetic/cases')),
                         previous.commands(args, Path('/synthetic/cases')))

    def test_optional_phase_appends_exactly_four_unbudgeted_clients_with_valid_parsers(self):
        args = self.args()
        original = concurrent.commands(args, Path('/synthetic/cases'))
        args.include_unbudgeted = True
        phases = concurrent.commands(args, Path('/synthetic/cases'))
        self.assertEqual(phases[:3], original)
        self.assertEqual(phases[3][0], 'literal-unbudgeted4')
        self.assertEqual([len(specs) for _, specs in phases], [4, 4, 4, 4])
        self.assertEqual(sum(len(spec['expected']) for _, specs in phases for spec in specs), 52)
        self.assertEqual(len({spec['name'] for _, specs in phases for spec in specs}), 16)
        self.assertEqual(len({spec['command'][-1] for _, specs in phases for spec in specs}), 16)
        class Parsed(Exception):
            pass
        real = argparse.ArgumentParser.parse_args
        observed = []
        def parse(self, *args, **kwargs):
            observed.append(real(self, *args, **kwargs))
            raise Parsed()
        for old, new in zip(original[2][1], phases[3][1]):
            self.assertEqual(new['expected'], old['expected'])
            self.assertEqual(new['kind'], 'literal')
            self.assertEqual(new['command'].count('--unbudgeted'), 1)
            with patch.object(argparse.ArgumentParser, 'parse_args', parse), patch.object(sys, 'argv', new['command'][1:]):
                with self.assertRaises(Parsed):
                    runpy.run_path(new['command'][1], run_name='__main__')
        self.assertEqual(len(observed), 4)
        self.assertTrue(all(args.unbudgeted for args in observed))

    def test_semantic_validators_and_owned_cleanup_are_source_identical(self):
        archive = BASE / 'tabbyapi-diagnostics/reasoning-clients-pre-unbudgeted'
        for file, names in (('reasoning_literal_smoke.py', ('validate', 'expected_names')),
                            ('reasoning_concurrency_smoke.py', ('summarize_child', 'cleanup_clients', 'validate_metadata', 'start_identity'))):
            def selected(path):
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(path.read_text()).body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}
            self.assertEqual(selected(BASE / file), selected(archive / file))

    def test_all_twelve_commands_parse_through_actual_client_parsers_without_running_requests(self):
        class Parsed(Exception):
            pass
        real = argparse.ArgumentParser.parse_args
        observed = []
        def parse(self, *args, **kwargs):
            observed.append(real(self, *args, **kwargs))
            raise Parsed()
        for _, specs in concurrent.commands(self.args(), Path('/synthetic/cases')):
            for spec in specs:
                with patch.object(argparse.ArgumentParser, 'parse_args', parse), patch.object(sys, 'argv', spec['command'][1:]):
                    with self.assertRaises(Parsed):
                        runpy.run_path(spec['command'][1], run_name='__main__')
        self.assertEqual(len(observed), 12)

    def test_missing_duplicate_unfinished_and_inconsistent_reports_cannot_pass(self):
        spec = {'kind': 'literal', 'expected': ['one']}
        good = {'cases': [{'name': 'one', 'status': 'pass'}], 'summary': {'pass': 1}, 'finished_utc': 'now'}
        self.assertTrue(concurrent.summarize_child(spec, good, 0)[1]['passed'])
        for mutate in (
            lambda r: r.pop('finished_utc'),
            lambda r: r['cases'].append(deepcopy(r['cases'][0])),
            lambda r: r['summary'].update({'pass': 5}),
            lambda r: r['cases'][0].update(name='unexpected'),
            lambda r: r.update(setup_error='error'),
        ):
            changed = deepcopy(good)
            mutate(changed)
            self.assertFalse(concurrent.summarize_child(spec, changed, 0)[1]['passed'])
        self.assertFalse(concurrent.summarize_child(spec, good, 1)[1]['passed'])

    def test_recorded_functional_failure_stays_a_failed_case(self):
        spec = {'kind': 'tools', 'expected': ['required/nonstream']}
        value = {'results': [{'case': 'required', 'mode': 'nonstream', 'repeat_index': 0,
                              'passed': False, 'errors': ['wrong args']}],
                 'summary': {'total': 1, 'passed': 0, 'failed': 1}, 'completed_at_utc': 'now'}
        rows, summary = concurrent.summarize_child(spec, value, 1)
        self.assertEqual(rows[0]['status'], 'fail')
        self.assertEqual(summary['counts'], {'fail': 1})
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['errors'], [])


class OrchestrationTests(unittest.TestCase):
    def run_driver(self, directory, *, batch=4, missing=False, bad_exit=False, final_crash=False,
                   existing_cases=False, timeout=False, terminate_signal=False, repeat_signal=False, signal_during_launch=False,
                   include_unbudgeted=False, missing_extra=False):
        meta = directory / 'deployment.json'
        meta.write_text(json.dumps(metadata(directory, batch)))
        output = directory / 'concurrent.json'
        if existing_cases:
            (directory / 'concurrent-cases').mkdir()
        spawned, killed, checks = [], [], []
        class Process:
            def __init__(self, command, **kwargs):
                self.pid = 8000 + len(spawned)
                self.returncode = None if timeout or terminate_signal or signal_during_launch else 0
                self.command = command
                spawned.append(self)
                assert kwargs['start_new_session'] is True
                assert kwargs['env']['QWEN_EXPERIMENT_OWNER']
                destination = Path(command[-1])
                if '--case' in command:
                    case = command[command.index('--case') + 1]
                    report = {'results': [{'case': case, 'mode': mode, 'repeat_index': 0, 'passed': True,
                                            'errors': []} for mode in ('nonstream', 'stream')],
                              'summary': {'passed': 2, 'failed': 0, 'total': 2}, 'completed_at_utc': 'done'}
                else:
                    if command[1].endswith('auto_compatibility.py'):
                        names = ['preflight_one_token_and_model'] + [f'auto_{v}_{m}'
                            for v in ('plain', 'reasoning', 'client_grammar', 'client_json_schema')
                            for m in ('nonstream', 'stream')]
                    else:
                        choice = command[command.index('--choice') + 1]
                        mode = command[command.index('--mode') + 1]
                        names = [f'literal_reasoning_{choice}_{mode}']
                    report = {'cases': [{'name': name, 'status': 'pass'} for name in names],
                              'summary': {'pass': len(names)}, 'finished_utc': 'done'}
                if not ((missing and len(spawned) == 1) or (missing_extra and len(spawned) == 13)):
                    destination.write_text(json.dumps(report))
                if bad_exit and len(spawned) == 1:
                    self.returncode = 1
                if signal_during_launch and len(spawned) == 1:
                    signal.raise_signal(signal.SIGTERM)
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                assert self.returncode is not None
                return self.returncode
        def killpg(pid, sig):
            process = next(p for p in spawned if p.pid == pid)
            killed.append(process.pid)
            process.returncode = -sig
            if repeat_signal and len(killed) == 1:
                signal.raise_signal(signal.SIGTERM)
        original_sleep = concurrent.time.sleep
        def sleep(seconds):
            if terminate_signal and not killed:
                signal.raise_signal(signal.SIGTERM)
            original_sleep(seconds)
        def identity(pid):
            checks.append(pid)
            if final_crash and len(spawned) == 12:
                raise ProcessLookupError('server exited')
            return 'stable-start'
        helper = SimpleNamespace(require_owned_listener=lambda pid: None,
                                 group_members=lambda process, owner: [process.pid] if process.poll() is None else [],
                                 stop_owned=lambda *args: self.fail('Sequential stop_owned must not be used'))
        with patch.object(concurrent, 'load_helper', return_value=helper), \
             patch.object(concurrent, 'start_identity', side_effect=identity), \
             patch.object(concurrent.subprocess, 'Popen', Process), \
             patch.object(concurrent.os, 'killpg', side_effect=killpg), \
             patch.object(concurrent.time, 'sleep', side_effect=sleep):
            code = concurrent.main(['--recipe', str(RECIPE), '--model', 'model', '--metadata', str(meta),
                                    '--expected-engine', 'a' * 40, '--expected-server', 'b' * 40,
                                    '--server-pid', '7000', '--output', str(output),
                                    '--phase-timeout', '.01' if timeout else '30',
                                    *(['--include-unbudgeted'] if include_unbudgeted else [])])
        return code, json.loads(output.read_text()), spawned, killed

    def test_complete_driver_has48_actual_rows12_client_reports_and_no_server_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, spawned, killed = self.run_driver(Path(tmp))
            self.assertEqual(code, 0)
            self.assertEqual(report['summary'], {'pass': 48})
            self.assertEqual(len(report['cases']), 48)
            self.assertEqual(len(report['client_reports']), 12)
            self.assertEqual(len(spawned), 12)
            self.assertEqual(killed, [])
            self.assertTrue(report['passed'])
            self.assertTrue(report['finished_utc'])

    def test_optional_driver_has52_actual_rows16_clients_and_no_server_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, spawned, killed = self.run_driver(Path(tmp), include_unbudgeted=True)
            self.assertEqual(code, 0)
            self.assertTrue(report['include_unbudgeted'])
            self.assertEqual(report['expected_checks'], 52)
            self.assertEqual(report['summary'], {'pass': 52})
            self.assertEqual(len(report['cases']), 52)
            self.assertEqual(len(report['client_reports']), 16)
            self.assertEqual(len(spawned), 16)
            self.assertEqual(killed, [])
            self.assertTrue(report['owned_client_groups_empty'])
            self.assertEqual([p['name'] for p in report['phases']], ['auto4', 'forced4', 'literal4', 'literal-unbudgeted4'])

    def test_missing_optional_case_cannot_pass_on_the_original48(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, spawned, killed = self.run_driver(Path(tmp), include_unbudgeted=True, missing_extra=True)
            self.assertEqual(code, 1)
            self.assertEqual(report['expected_checks'], 52)
            self.assertEqual(len(report['cases']), 51)
            self.assertFalse(report['passed'])
            self.assertTrue(report['errors'])
            self.assertEqual(len(spawned), 16)
            self.assertEqual(killed, [])

    def test_wrong_batch_or_existing_child_directory_rejects_before_launch(self):
        for options in ({'batch': 1}, {'existing_cases': True}):
            with tempfile.TemporaryDirectory() as tmp:
                code, report, spawned, killed = self.run_driver(Path(tmp), **options)
                self.assertEqual(code, 1)
                self.assertEqual(spawned, [])
                self.assertEqual(killed, [])
                self.assertTrue(report['setup_error'])

    def test_missing_report_or_bad_exit_does_not_claim_allpass(self):
        for options in ({'missing': True}, {'bad_exit': True}):
            with tempfile.TemporaryDirectory() as tmp:
                code, report, spawned, _ = self.run_driver(Path(tmp), **options)
                self.assertEqual(code, 1)
                self.assertFalse(report['passed'])
                self.assertTrue(report['setup_error'])
                self.assertEqual(len(spawned), 12)
                if options.get('missing'):
                    self.assertEqual(len(report['cases']), 39)

    def test_server_crash_after_last_client_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _, _ = self.run_driver(Path(tmp), final_crash=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(report['cases']), 48)
            self.assertFalse(report['passed'])
            self.assertIn('server exited', report['setup_error'])

    def test_timeout_cleans_only_four_owned_client_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, spawned, killed = self.run_driver(Path(tmp), timeout=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(spawned), 4)
            self.assertEqual(killed, [8000, 8001, 8002, 8003])
            self.assertNotIn(7000, killed)
            self.assertTrue(all(item['owned_client_group_empty']
                                for phase in report['phases'] for item in phase['clients']))


    def test_sigterm_after_launch_runs_finally_and_repeated_signal_does_not_interrupt_cleanup(self):
        old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        with tempfile.TemporaryDirectory() as tmp:
            code, report, spawned, killed = self.run_driver(Path(tmp), terminate_signal=True, repeat_signal=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(spawned), 4)
            self.assertEqual(killed, [8000, 8001, 8002, 8003])
            self.assertTrue(report['owned_client_groups_empty'])
            self.assertFalse(report['cleanup_failed'])
            self.assertTrue(report['finished_utc'])
            self.assertIn('KeyboardInterrupt', report['setup_error'])
        self.assertEqual({sig: signal.getsignal(sig) for sig in old}, old)


    def test_sigterm_during_popen_registration_cannot_orphan_the_just_created_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, spawned, killed = self.run_driver(Path(tmp), signal_during_launch=True)
            self.assertEqual(code, 1)
            self.assertEqual(len(spawned), 1)
            self.assertEqual(killed, [8000])
            self.assertTrue(report['owned_client_groups_empty'])
            self.assertFalse(report['cleanup_failed'])
            self.assertIn('KeyboardInterrupt', report['setup_error'])


class SharedCleanupTests(unittest.TestCase):
    def test_all_four_groups_receive_term_before_wait_and_share_one_deadline(self):
        now, signals = [0.0], []
        class Process:
            def __init__(self, pid):
                self.pid, self.returncode = pid, None
            def poll(self):
                return self.returncode
            def wait(self, timeout=None):
                assert self.returncode is not None
                return self.returncode
        processes = [Process(8100 + i) for i in range(4)]
        entries = [(p, io.BytesIO(), {}) for p in processes]
        helper = SimpleNamespace(group_members=lambda p, token: [p.pid] if p.poll() is None else [])
        def killpg(pid, sig):
            signals.append((pid, sig, now[0]))
            if sig == signal.SIGKILL:
                next(p for p in processes if p.pid == pid).returncode = -sig
        def sleep(seconds):
            now[0] += seconds
        with patch.object(concurrent.time, 'monotonic', side_effect=lambda: now[0]), \
             patch.object(concurrent.time, 'sleep', side_effect=sleep), \
             patch.object(concurrent.os, 'killpg', side_effect=killpg):
            concurrent.cleanup_clients(helper, entries, 'owned', timeout=35)
        self.assertEqual([row[:2] for row in signals],
                         [(p.pid, signal.SIGTERM) for p in processes] +
                         [(p.pid, signal.SIGKILL) for p in processes])
        self.assertEqual({row[2] for row in signals[:4]}, {0})
        self.assertLessEqual(now[0], 35)
        self.assertTrue(all(item['owned_client_group_empty'] for _, _, item in entries))
        with patch.object(concurrent.os, 'killpg', side_effect=AssertionError('no repeated full cleanup')):
            concurrent.cleanup_clients(helper, entries, 'owned')

    def test_unverified_group_is_never_signaled_and_never_claimed_empty(self):
        p = SimpleNamespace(pid=8100, poll=lambda: None)
        item = {}
        def members(*args):
            raise RuntimeError('owner mismatch')
        with patch.object(concurrent.os, 'killpg') as send:
            with self.assertRaisesRegex(RuntimeError, 'owner mismatch'):
                concurrent.cleanup_clients(SimpleNamespace(group_members=members), [(p, io.BytesIO(), item)], 'owned')
            send.assert_not_called()
        self.assertFalse(item['owned_client_group_empty'])
        self.assertTrue(item['cleanup_attempted'])



if __name__ == '__main__':
    unittest.main()
