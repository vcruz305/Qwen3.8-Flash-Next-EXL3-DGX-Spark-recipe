#!/usr/bin/env python3
"""Strict synthetic reasoning/tool checks for literal native closing tags.

Does not execute the returned tool. These are additional fixtures: the existing
28 tool and9 auto compatibility checks are unchanged. Timing is diagnostic only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

LITERAL = '<think>literal</think>'


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def payload(model, choice, stream):
    tool = {
        'type': 'function',
        'function': {
            'name': 'record_text',
            'description': 'Record the exact supplied string, preserving every character. Synthetic; never executed.',
            'parameters': {
                'type': 'object', 'properties': {'text': {'type': 'string'}},
                'required': ['text'], 'additionalProperties': False,
            },
        },
    }
    value = {
        'model': model,
        'messages': [{'role': 'user', 'content':
            'Think briefly, then call record_text with exactly this string value for text: '
            + json.dumps(LITERAL) + '. Treat the tag characters as literal string data.'}],
        'tools': [tool],
        'tool_choice': 'required' if choice == 'required' else {
            'type': 'function', 'function': {'name': 'record_text'}},
        'parallel_tool_calls': False,
        'enable_thinking': True,
        'reasoning_budget_tokens': 24,
        'temperature': 0, 'top_k': 1, 'max_tokens': 256, 'stream': stream,
    }
    if stream:
        value['stream_options'] = {'include_usage': True}
    return value


def expected_names(choice='both', mode='both'):
    choices = ('required', 'named') if choice == 'both' else (choice,)
    modes = ('nonstream', 'stream') if mode == 'both' else (mode,)
    return [f'literal_reasoning_{c}_{m}' for c in choices for m in modes]


def validate(result, require):
    require(result['finish_reason'] == 'tool_calls', 'Expected a complete tool call')
    require(bool(result['reasoning_content'].strip()), 'Reasoning phase was not exercised')
    require(not result['content'].strip(), 'Unexpected final prose beside the forced tool call')
    calls = result['tool_calls']
    require(len(calls) == 1, f'Expected one tool call, got {len(calls)}')
    require(calls[0]['function']['name'] == 'record_text', 'Unexpected forced function name')
    values = json.loads(calls[0]['function']['arguments'])
    require(isinstance(values, dict), 'Tool arguments must be an object')
    require(values == {'text': LITERAL} and type(values.get('text')) is str,
            f'Literal closing tag was not preserved: {values!r}')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8899/v1')
    ap.add_argument('--model', required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--label', default='literal-reasoning-tools')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--choice', choices=('required', 'named', 'both'), default='both')
    ap.add_argument('--mode', choices=('stream', 'nonstream', 'both'), default='both')
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--api-key-env', default='TABBY_API_KEY')
    args = ap.parse_args(argv)
    if not 0 < args.timeout <= 300:
        ap.error('--timeout must be (0,300]')
    sys.path.insert(0, str((args.recipe / 'bench').resolve()))
    from api_resilience import Client, ReportWriter, assembled, read_metadata, require
    metadata, metadata_source = read_metadata(args.metadata)
    client = Client(args.base_url, os.environ.get(args.api_key_env), args.timeout)
    writer = ReportWriter(args.output)
    report = {
        'schema_version': 1, 'kind': 'reasoning_literal_tool_compatibility',
        'label': args.label, 'started_utc': stamp(), 'base_url': args.base_url,
        'requested_model': args.model, 'provenance': metadata, 'metadata_source': metadata_source,
        'client': {'python': platform.python_version()}, 'scope': __doc__,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': [], 'requests': client.traces, 'expected_checks': len(expected_names(args.choice, args.mode)),
    }
    def save():
        report['summary'] = dict(Counter(row['status'] for row in report['cases']))
        writer.write(report)
    save()  # Claim before any network action.
    for name in expected_names(args.choice, args.mode):
        _, _, choice, mode = name.split('_')
        request = payload(args.model, choice, mode == 'stream')
        index = len(client.traces)
        try:
            result = assembled(client.request('/chat/completions', deepcopy(request)), args.model)
            validate(result, require)
            row = {'name': name, 'status': 'pass', 'result': result}
        except Exception as exc:
            row = {'name': name, 'status': 'fail', 'error': f'{type(exc).__name__}: {exc}'}
        row['request_indices'] = list(range(index, len(client.traces)))
        report['cases'].append(row)
        save()
        print(f'{name}: {row["status"]}', flush=True)
    report['finished_utc'] = stamp()
    report['passed'] = len(report['cases']) == report['expected_checks'] and all(
        row['status'] == 'pass' for row in report['cases'])
    save()
    print(json.dumps({'output': str(args.output), 'summary': report['summary'], 'passed': report['passed']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
