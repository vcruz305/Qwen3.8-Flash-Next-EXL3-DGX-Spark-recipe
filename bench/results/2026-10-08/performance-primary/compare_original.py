#!/usr/bin/env python3
"""Compare retained baseline and primary control requests without making API calls."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(archive, original):
    rows = []
    labels = {'305': '01-control-305-ram', '405': '01-control-405-disk',
              '415': '01-control-415-disk', 'cyber387': '01-control-cyber387-disk'}
    for pack, label in labels.items():
        result_path = archive / 'reports' / label / 'result.json'
        result = json.loads(result_path.read_text())
        assert result['state'] == 'completed' and result['finished_at_utc']
        assert result['server_exit_before_cleanup'] is None
        assert result['server_cleanup'] == {'exit_code': 0, 'signals': ['SIGTERM']}
        attempt = result_path.parent / Path(result['attempt']).name
        for benchmark, suffix, metric in [('bench-1', '', 'server_decode_tok_s'),
                                          ('bench-2', '-prefill', 'server_prefill_tok_s')]:
            old_path = original / f'baseline-{pack}{suffix}.json'
            new_path = attempt / f'{benchmark}.json'
            old, new = [json.loads(p.read_text()) for p in [old_path, new_path]]
            assert old['completed_at_utc'] and new['completed_at_utc']
            assert not old['errors'] and not new['errors']
            assert old['settings'] == new['settings'], (pack, benchmark, 'settings')
            assert old['run_id'] == new['run_id'], (pack, benchmark, 'run_id')
            assert old['cases'].keys() == new['cases'].keys()
            for case in old['cases']:
                o, n = old['cases'][case], new['cases'][case]
                assert o['prompt_sha256'] == n['prompt_sha256']
                a, b = o['runs'], n['runs']
                assert len(a) == len(b) == old['settings']['repeat']
                paired = []
                for x, y in zip(a, b):
                    assert x['phase'] == y['phase'] == 'measured'
                    assert x['repeat_index'] == y['repeat_index']
                    assert x['request_sha256'] == y['request_sha256']
                    assert x['finish_reason'] == y['finish_reason'] == 'length'
                    for k in ['prompt_tokens', 'completion_tokens', 'cached_prompt_tokens']:
                        assert x[k] == y[k], (pack, benchmark, case, k)
                    assert x['cached_prompt_tokens'] == 0
                    assert x['completion_tokens'] == old['settings']['max_tokens']
                    paired.append({'repeat_index': x['repeat_index'], 'request_sha256': x['request_sha256'],
                                   'original_response_sha256': x['response_sha256'],
                                   'primary_response_sha256': y['response_sha256'],
                                   'prompt_tokens': x['prompt_tokens'], 'completion_tokens': x['completion_tokens'],
                                   'original_metric': x[metric], 'primary_metric': y[metric]})
                ov = statistics.median(x[metric] for x in a)
                nv = statistics.median(x[metric] for x in b)
                rows.append({'pack': pack, 'label': label, 'benchmark': benchmark, 'case': case,
                             'metric': metric, 'original_median': ov, 'primary_median': nv,
                             'change_percent': (nv / ov - 1) * 100,
                             'measured_requests_per_source': len(a),
                             'same_request_bytes': True, 'same_actual_token_counts': True,
                             'original_report': str(old_path), 'original_sha256': sha(old_path),
                             'primary_report': str(new_path), 'primary_sha256': sha(new_path),
                             'whole_primary_job_passed': result['passed'], 'requests': paired})
    return {'schema_version': 1, 'comparison_script_sha256': sha(Path(__file__)), 'rows': rows,
            'scope': 'Identical API request hashes and actual token counts per pack; medians exclude warmup. '
                     'Historical primary engine16ca/Tabbyf4, not the subsequent final source pair. '
                     'A valid throughput comparison does not turn a failed tool client into a passing whole job.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('.'))
    p.add_argument('--original', type=Path, default=Path('../original-api'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('output already exists')
    result = compare(a.archive, a.original)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps({'comparison_rows': len(result['rows']), 'output': str(a.output)}))


if __name__ == '__main__':
    main()
