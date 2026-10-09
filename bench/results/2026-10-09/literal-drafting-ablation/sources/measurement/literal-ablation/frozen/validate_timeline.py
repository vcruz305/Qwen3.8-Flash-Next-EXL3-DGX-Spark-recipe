"""Pure structural validation of the bounded diagnostic producer event trace.

This validates capture completeness, never model semantics or whether a phase
should have ended. The absence of a phase transition can be valid evidence.
"""
from collections import Counter


def validate_timeline(trace):
    events = trace.get('producer_events')
    if type(trace.get('producer_events_dropped')) is not int or trace['producer_events_dropped'] != 0:
        raise ValueError('Producer trace dropped events or has no explicit drop count')
    if not isinstance(events, list) or not 1 <= len(events) <= 2048:
        raise ValueError('Expected one through 2048 producer events')
    opens = {'sample_processed_before_budget': 'accepted', 'guard_before': 'guard',
             'force_before': 'force', 'phase_before': 'phase'}
    closes = {'sample_processed_after_budget': 'accepted',
              'guard_after': 'guard', 'guard_error': 'guard',
              'force_after': 'force', 'force_error': 'force',
              'phase_after': 'phase', 'phase_error': 'phase'}
    stack, kinds, last_time = [], Counter(), -1
    for index, event in enumerate(events):
        if not isinstance(event, dict) or type(event.get('index')) is not int or event['index'] != index:
            raise ValueError('Producer event indices must be contiguous integers')
        now = event.get('monotonic_ns')
        if type(now) is not int or now < 0 or now < last_time:
            raise ValueError('Producer event times must be ordered nonnegative integers')
        last_time = now
        kind = event.get('kind')
        state = event.get('state')
        if not isinstance(state, dict):
            raise ValueError('Missing event state')
        for key in ('new_tokens', 'rq_new_tokens', 'filter_count'):
            if type(state.get(key)) is not int:
                raise ValueError('Missing integer state: ' + key)
        if state['filter_count'] < 0:
            raise ValueError('Negative filter count')
        if type(state.get('checkpoint_rewound')) is not bool:
            raise ValueError('Missing rewind state')
        if type(state.get('filters_suspended')) is not bool:
            raise ValueError('Missing filter suspension state')
        if kind in opens:
            stack.append((opens[kind], event))
        elif kind in closes:
            if not stack or stack[-1][0] != closes[kind]:
                raise ValueError('Unbalanced or improperly nested producer events')
            _, before = stack.pop()
            if kind == 'sample_processed_after_budget':
                if (event.get('token_id'), event.get('eos')) != (before.get('token_id'), before.get('eos')):
                    raise ValueError('Accepted-token pair changed its observation')
            if kind == 'phase_after' and event.get('requested_reasoning') != before.get('requested_reasoning'):
                raise ValueError('Phase pair changed its requested observation')
        else:
            raise ValueError('Unknown producer event kind')
        if kind.startswith('sample_processed_'):
            if type(event.get('token_id')) is not int or event['token_id'] < 0 or type(event.get('eos')) is not bool:
                raise ValueError('Missing native accepted-token observation')
        if kind == 'guard_after' and type(event.get('allowed')) is not bool:
            raise ValueError('Missing guard decision')
        if kind in ('phase_before', 'phase_after') and type(event.get('requested_reasoning')) is not bool:
            raise ValueError('Missing requested phase')
        if kind == 'phase_after' and type(event.get('applied')) is not bool:
            raise ValueError('Missing phase result')
        if kind.endswith('_error') and not isinstance(event.get('exception_type'), str):
            raise ValueError('Missing diagnostic exception class')
        kinds[kind] += 1
    if stack:
        raise ValueError('Incomplete producer method observations')
    if not kinds['sample_processed_before_budget']:
        raise ValueError('No accepted-token observations')
    return {'capture_complete': True, 'event_count': len(events), 'event_counts': dict(kinds),
            'phase_transition_observed': bool(kinds['phase_after']),
            'forced_output_observed': bool(kinds['force_after'])}
