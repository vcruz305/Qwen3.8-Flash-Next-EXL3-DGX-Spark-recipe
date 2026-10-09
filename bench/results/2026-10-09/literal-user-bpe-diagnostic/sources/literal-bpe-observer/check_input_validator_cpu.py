#!/usr/bin/env python3
import copy,hashlib,json
from pathlib import Path
from validate_input_tokenization import validate_input_tokenization
ROOT=Path(__file__).resolve().parent
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
report=json.loads((ROOT/'input-hook-cpu-report.json').read_text())
prompt=(OLD/'literal-final-triage-5a/rendered-prompt.txt').read_text()
rows=[]
for case in report['cases']:
 trace={'rendered_prompt':prompt,'input_tokenization':case['proof'],'input_encoding_calls':case['encode_calls'],
        'raw_finish':{'native_metrics':{'prompt_tokens':case['generation_input_tokens']}}}
 result=validate_input_tokenization(trace);rows.append(result)
base=trace
bad_edits=[lambda d:d.update(rendered_prompt=prompt+' '),lambda d:d.pop('input_tokenization'),
 lambda d:d['input_tokenization'].update(changed_model_input=False),lambda d:d['input_tokenization'].update(user_span=[0,1]),
 lambda d:d['input_tokenization']['changed_token_ids'].__setitem__(0,1),lambda d:d['input_tokenization']['original_token_ids'].__setitem__(0,True),
 lambda d:d['input_tokenization']['replacements'].pop(),lambda d:d.update(input_encoding_calls=[]),
 lambda d:d['input_encoding_calls'][0].update(native_return_device='cuda'),
 lambda d:d['input_encoding_calls'][0].update(changed_prompt_tokens=348),
 lambda d:d['input_encoding_calls'][0].update(changed_ids_sha256='bad'),
 lambda d:d['raw_finish']['native_metrics'].update(prompt_tokens=348)]
for edit in bad_edits:
 bad=copy.deepcopy(base);edit(bad)
 try:validate_input_tokenization(bad)
 except ValueError:pass
 else:raise AssertionError('Invalid proof passed')
out={'passed':True,'scope':'Pure proof checks on actual CPU encode composition records; native usage in this fixture comes from the returned input tensor shape. Live API/native usage must be checked independently.','positive_cases':len(rows),'negative_cases':len(bad_edits),
 'validator_sha256':hashlib.sha256((ROOT/'validate_input_tokenization.py').read_bytes()).hexdigest(),
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows}
with (ROOT/'input-validator-cpu-report.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'positive_cases':len(rows),'negative_cases':len(bad_edits),'report_sha256':hashlib.sha256((ROOT/'input-validator-cpu-report.json').read_bytes()).hexdigest()}))
