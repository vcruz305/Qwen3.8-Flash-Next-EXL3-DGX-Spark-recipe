#!/usr/bin/env python3
"""Compose the real engine Job fixture and real Tabby incremental boundary guard.

CPU-only: engine test fixture owns scheduling/allocation doubles and executes
native Job/SeqTensor/MTP/mask methods. No server, CUDA or model is imported.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', type=Path, required=True)
    ap.add_argument('--tabby', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with args.output.open('x') as out:
        json.dump({'state': 'running', 'scope': __doc__}, out)
    sys.path.insert(0, str(args.tabby.resolve()))
    from backends.exllamav3.reasoning import ReasoningBoundaryGuard
    from endpoints.OAI.utils.stream_parser import Qwen3CoderStreamParser
    spec = importlib.util.spec_from_file_location('guard_native_fixture', args.engine / 'tests/test_token_budget_cpu.py')
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    import torch

    report = {'state': 'running', 'scope': __doc__, 'python': sys.version, 'platform': platform.platform(),
              'torch': torch.__version__, 'cases': [], 'sources': {}, 'harness_sha256': sha(__file__)}
    for label, root, paths in (
        ('engine', args.engine, ['tests/test_token_budget_cpu.py', 'exllamav3/generator/job.py',
                                'exllamav3/generator/generator.py', 'exllamav3/generator/async_generator.py']),
        ('tabby', args.tabby, ['backends/exllamav3/reasoning.py', 'endpoints/OAI/utils/stream_parser.py'])):
        report['sources'][label] = {
            'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'working_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest(),
            'files': {path: sha(root / path) for path in paths}}

    def job_for(parts, budget, **kwargs):
        job = native.make_job(input_ids=torch.tensor([[0, 0]]), **kwargs)
        for token, piece in parts.items():
            job.generator.tokenizer.pieces[token] = piece
        parser = Qwen3CoderStreamParser(reasoning_start='<think>', reasoning_end='</think>',
                                        tool_start='<tool_call>', tool_end='</tool_call>',
                                        start_in_reasoning=True, tool_calls_in_reasoning=True)
        guard = ReasoningBoundaryGuard(job, job.generator.tokenizer, parser)
        calls = []
        final_filter = native.Filter(allowed=8, packed=True)
        def ended(current):
            assert current is job
            assert not parser.in_tool and not parser.in_reasoning
            calls.append(current.rq_new_tokens + current.new_tokens)
            current.set_filters([final_filter])
        native.arm(job, budget, callback=ended, guard=guard)
        return job, parser, guard, calls, final_filter

    def run_case(name, callback):
        start = time.monotonic()
        try:
            detail = callback()
            row = {'name': name, 'passed': True, 'detail': detail}
        except Exception as exc:
            row = {'name': name, 'passed': False, 'error': repr(exc)}
        row['seconds'] = time.monotonic() - start
        report['cases'].append(row)

    def tool_case(wrapped, budget, mtp):
        parts = {1: 'thought ', 2: '<tool_' if wrapped else '<func',
                 3: 'call><function=echo><parameter=text>\n' if wrapped else 'tion=echo><parameter=text>\n',
                 9: '</think>', 5: 'literal', 6: '\n</parameter>',
                 7: '</function></tool_call>' if wrapped else '</function>', 8: 'answer'}
        job, parser, guard, calls, final_filter = job_for(parts, budget)
        expected = [1, 2, 3, 9, 5, 6, 7, 9, 8]
        if mtp:
            scores = torch.zeros((1, len(expected), native.VOCAB))
            for position, token in enumerate(expected):
                scores[0, position, token] = 5
            result = native.run_mtp(job, expected[:-1], scores=scores)
            got = native.emitted_ids(result['results'])
        else:
            rows = []
            for token in expected:
                native._prepare_masks(job)
                native.sample(job, token, rows)
            got = native.emitted_ids(rows)
        assert got == expected, (got, expected)
        assert calls == [8], calls
        assert final_filter.fed == [8]
        assert job.token_budget is None
        assert len(guard._states) == 1
        return {'ids': got, 'phase_callback_positions': calls,
                'literal_end_position': 4, 'real_end_position': 8, 'mtp': mtp,
                'budget': budget, 'forced': budget == 2, 'wrapped': wrapped}

    for wrapped in (False, True):
        for budget in (2, 100):
            for mtp in (False, True):
                run_case(f'tool_wrapped{wrapped}_budget{budget}_mtp{mtp}',
                         lambda w=wrapped, b=budget, m=mtp: tool_case(w, b, m))

    def rewind_partial_opener():
        parts = {1: 'thought ', 2: '<tool_', 3: 'call>BAD', 4: 'safe', 9: '</think>', 8: 'answer'}
        job, parser, guard, calls, final_filter = job_for(parts, 2, banned_strings=['<tool_call>BAD'])
        rows = []
        native.sample(job, 1, rows)
        native.sample(job, 2, rows)
        native.sample(job, 3, rows)
        assert job.checkpoint_rewound and job.new_tokens == 1
        assert not parser.in_tool and not parser._pending and parser.in_reasoning
        job.checkpoint_rewound = False  # Native Generator consumes this signal before the next iteration.
        native.sample(job, 4, rows)
        native.sample(job, 1, rows)
        native._prepare_masks(job)
        native.sample(job, 1, rows)
        got = native.emitted_ids(rows)
        assert got == [1, 4, 9, 8], got
        assert calls == [3] and final_filter.fed == [8]
        return {'ids': got, 'phase_callback_positions': calls}
    run_case('banned_rewind_partial_opener', rewind_partial_opener)

    def rewind_native_end():
        parts = {1: 'thought ', 9: '</think>', 10: 'x', 8: 'answer'}
        job, parser, guard, calls, final_filter = job_for(parts, 3, banned_strings=['</think>x'])
        rows = []
        native.sample(job, 1, rows)
        native.sample(job, 9, rows)
        assert job.token_budget is not None and calls == [] and not parser.in_reasoning
        native.sample(job, 10, rows)
        assert job.checkpoint_rewound and job.new_tokens == 1 and parser.in_reasoning
        assert job.token_budget is not None and not job.token_budget['end_seen']
        job.checkpoint_rewound = False
        native.sample(job, 1, rows)
        native.sample(job, 1, rows)
        native.sample(job, 1, rows)
        native._prepare_masks(job)
        native.sample(job, 1, rows)
        got = native.emitted_ids(rows)
        assert got == [1, 1, 1, 9, 8], got
        assert calls == [4] and final_filter.fed == [8]
        return {'ids': got, 'phase_callback_positions': calls}
    run_case('banned_rewind_natural_end', rewind_native_end)

    def requeue_tool():
        parts = {1: 'thought ', 2: '<tool_', 3: 'call><function=echo><parameter=text>',
                 9: '</think>', 6: '</parameter>', 7: '</function></tool_call>', 8: 'answer'}
        job, parser, guard, calls, final_filter = job_for(parts, 2)
        rows = []
        for token in (1, 2, 3, 9):
            native.sample(job, token, rows)
        assert parser.in_tool and parser.in_reasoning
        original = job
        job = job.prepare_for_requeue()
        assert job is original
        for token in (6, 7, 1):
            native.sample(job, token, rows)
        native._prepare_masks(job)
        native.sample(job, 1, rows)
        got = native.emitted_ids(rows)
        assert got == [1, 2, 3, 9, 6, 7, 9, 8], got
        assert calls == [7] and final_filter.fed == [8]
        return {'ids': got, 'phase_callback_positions': calls}
    run_case('requeue_inside_literal_tool_argument', requeue_tool)

    report['state'] = 'completed'
    report['summary'] = {'passed': sum(row['passed'] for row in report['cases']),
                         'failed': sum(not row['passed'] for row in report['cases']),
                         'total': len(report['cases'])}
    temporary = args.output.with_name(args.output.name + '.tmp')
    with temporary.open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    os.replace(temporary, args.output)
    print(json.dumps({'output': str(args.output), 'summary': report['summary']}))
    for row in report['cases']:
        if not row['passed']:
            print(json.dumps(row))
    return bool(report['summary']['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
