#!/usr/bin/env python3
import copy,hashlib,json
from pathlib import Path
from validate_timeline import validate_timeline
ROOT=Path(__file__).resolve().parent
report=json.loads((ROOT/'phase-hooks-cpu-report.json').read_text())
rows=[]
for scenario in report['scenarios']:
 trace={'producer_events':scenario['events'],'producer_events_dropped':0}
 result=validate_timeline(trace)
 rows.append({'scenario':scenario['scenario'],**result})
base={'producer_events':report['scenarios'][0]['events'],'producer_events_dropped':0}
changes=[lambda d:d.update(producer_events_dropped=1),lambda d:d.pop('producer_events_dropped'),
 lambda d:d.update(producer_events=[]),lambda d:d['producer_events'].pop(),
 lambda d:d['producer_events'][0].update(index=True),lambda d:d['producer_events'][0].update(kind='unknown'),
 lambda d:d['producer_events'][1].update(monotonic_ns=0),lambda d:d['producer_events'][0].pop('token_id'),
 lambda d:d['producer_events'][0].update(token_id=True),lambda d:d['producer_events'][0].update(eos=0),
 lambda d:d['producer_events'][0]['state'].update(filter_count=-1),
 lambda d:d['producer_events'][0]['state'].update(filters_suspended=None)]
for change in changes:
 bad=copy.deepcopy(base);change(bad)
 try:validate_timeline(bad)
 except ValueError:pass
 else:raise AssertionError('A corrupted timeline passed')
# A complete accepted token with no phase ending is legitimate evidence.
pair=copy.deepcopy(report['scenarios'][0]['events'][:2])
assert validate_timeline({'producer_events':pair,'producer_events_dropped':0})['phase_transition_observed'] is False
out={'passed':True,'actual_source_event_cases':rows,'negative_cases':len(changes),'no_transition_is_valid':True,
 'validator_sha256':hashlib.sha256((ROOT/'validate_timeline.py').read_bytes()).hexdigest(),
 'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
with (ROOT/'timeline-validator-cpu-report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'positive_cases':len(rows)+1,'negative_cases':len(changes),
 'validator_sha256':out['validator_sha256'],'report_sha256':hashlib.sha256((ROOT/'timeline-validator-cpu-report.json').read_bytes()).hexdigest()}))
