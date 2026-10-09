#!/usr/bin/env python3
"""Independent completed f4 raw protocol/phase review; no inference or mutation."""
from pathlib import Path
import json,hashlib
B=Path('/home/vcruz/src/qwen-followup-20261009')
D=B/'mandatory-eos-generation-engine-review'
R=D/'raw'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan_path=B/'measurement/literal-feature-eos-generation/plan.json'
binding_path=B/'measurement/literal-feature-eos-generation/prepared-inputs.json'
plan=json.loads(plan_path.read_text());binding=json.loads(binding_path.read_text())
expected={x['name']:x for x in plan['requests']};inputs={x['name']:x for x in binding['requests']}
outer=json.loads((R/'result.json').read_text())
assert outer['state']=='completed' and outer['capture']['capture_valid']
generation_path=next(R.glob('*/attempt-*/generation.json'))
generation=json.loads(generation_path.read_text());client={x['name']:x for x in generation['results']}
assert len(client)==8 and set(client)==set(expected)
rows=[]
for p in sorted((R/'raw-generation').glob('request-*.json')):
 trace=json.loads(p.read_text());name=trace['case_name'];api=client[name];item=expected[name];prepared=inputs[name]
 events=trace['native_sample_events'];samples=[e for e in events if e['kind']=='native_sample_after'];befores=[e for e in events if e['kind']=='native_sample_before']
 assert len(samples)==len(befores)
 ids=[e['payload']['processed_token']['ids'][0] for e in samples]
 eos_positions=[i+1 for i,t in enumerate(ids)if t in (248044,248046)]
 assert eos_positions==[len(samples)] and ids[-1]==248046 and ids[-2]==248059
 assert sum(e['payload']['eos'] for e in samples)==1 and samples[-1]['payload']['eos']
 assert all(e['payload']['state']['stop_tokens']==[248044,248046]for e in samples)
 raw=trace['raw_finish'];native=raw['native_full_completion']
 assert native==raw['backend_full_response'] and native.rstrip().endswith('</tool_call>')
 assert raw['native_metrics']['eos_reason']=='stop_token' and raw['native_metrics']['eos_triggering_token_str']=='<|im_end|>'
 assert raw['native_metrics']['new_tokens']==len(samples)
 observed=api['trace'];result=api['result']
 assert observed['status']==200 and not observed.get('server_error') and not observed.get('client_aborted')
 assert not observed['stream'] or observed['done'] is True
 assert result['finish_reason']=='tool_calls' and len(result['tool_calls'])==1
 call=result['tool_calls'][0];assert call['function']['name']=='record_text'
 arguments=json.loads(call['function']['arguments']);assert set(arguments)=={'text'} and type(arguments['text'])is str
 exact=arguments['text']==item['expected']['value'];assert exact==api['semantic_passed']
 # Current fixtures have one outer parameter delimiter; markup inside the value is data.
 body=native.split('<parameter=text>',1)[1].split('</parameter>',1)[0]
 assert body.startswith('\n') and body.endswith('\n')
 assert body[1:-1]==arguments['text']
 assert result['usage']['prompt_tokens']==raw['native_metrics']['prompt_tokens']==prepared['prompt_tokens']
 assert observed['request']==item['request']
 producer=trace['producer_events'];phase=[e for e in producer if e['kind']=='phase_after' and e.get('applied') and e.get('requested_reasoning') is False]
 force=[e for e in producer if e['kind']=='force_before']
 thinking=trace['start_in_reasoning_mode']
 if thinking:
  assert item['request']['reasoning_budget_tokens']==24 and phase and len(force)==1
  transition=phase[0];assert transition['state']['new_tokens']==25 and transition['state']['filter_count']==1 and transition['state']['filters_suspended']is False
  assert ids[24]==248069 and force[0]['state']['new_tokens']==24
  assert befores[24]['monotonic_ns']<=transition['monotonic_ns']<=samples[24]['monotonic_ns']<befores[25]['monotonic_ns']
  assert transition['monotonic_ns']<samples[-1]['monotonic_ns']
 else:
  assert not producer and not phase and not force
 assert item['request'].get('stop')in(None,[])
 decode_count=sum(e['kind']=='native_decode_before'for e in events)
 rows.append({'name':name,'input_tokens':prepared['prompt_tokens'],'cached_tokens':result['usage']['prompt_tokens_details']['cached_tokens'],
  'initial_reasoning':thinking,'reasoning_budget':item['request'].get('reasoning_budget_tokens'),
  'phase_handoff_processed_position':25 if thinking else None,'forced_close_after_positions':24 if thinking else None,
  'only_implicit_eos_position':eos_positions[0],'eos_token_id':ids[-1],'preceding_call_close_id':ids[-2],
  'processed_samples':len(samples),'native_decode_calls':decode_count,'native_equals_backend':True,
  'native_parameter_equals_api_value':True,'complete_protocol':True,'exact_argument':exact,
  'actual_argument':arguments['text'],'expected_argument':item['expected']['value'],
  'sample_rewinds':sum(e['payload']['state']['checkpoint_rewound']for e in samples),
  'prefix_present_observations':sum(e['payload']['state']['prefix_token_present']for e in samples),
  'raw_sha256':sha(p)})
assert len(rows)==8 and sum(r['processed_samples']for r in rows)==481 and sum(r['native_decode_calls']for r in rows)==15
assert sum(r['exact_argument']for r in rows)==5 and sum(r['initial_reasoning']for r in rows)==6
report={'schema_version':1,'passed_review':True,'source_commits':{'engine':binding['engine_commit'],'tabby':binding['tabby_commit'],'recipe':binding['recipe_commit']},
 'finished_at_utc':outer['finished_at_utc'],'actual_chat_posts':8,'complete_protocol':8,'valid_raw_captures':8,'exact_arguments':5,'wrong_arguments':3,
 'processed_samples':481,'native_decode_calls':15,'native_equals_backend':8,'native_parameter_equals_api':8,
 'implicit_eos_only_after_complete_call':8,'initial_reasoning_handoff_before_implicit_eos':6,'initial_content_requests':2,
 'rows':rows,'source_hashes':{'script':sha(Path(__file__)),'plan':sha(plan_path),'binding':sha(binding_path),'outer_result':sha(R/'result.json'),'generation_report':sha(generation_path)},
 'conclusion':'Observed finite-budget mandatory-tool EOS behavior matches the source fix: six initially reasoning requests force close after24 processed tokens, install content filters at25 before the next sample, and later stop only after a complete call. Two thinking-off requests remain content-only. This is protocol completion, not model-value repair.',
 'limitations':['These eight requests specify no explicit caller stops; preservation of explicit stops and natural-only handoff is CPU/source coverage, not a new live result here.',
 'No earlier continuation/cache coverage8 ran before these requests. Cache histories differ from combined16, so output changes are not an isolated whole-model source A/B and no cache qualification is repeated.',
 'Five exact values and three wrong values remain separate from all eight complete protocols; both multiturn values are wrong and concurrent2 still omits endoftext.',
 'No sampled logits are captured; the trace demonstrates resulting stop/phase behavior, not the counterfactual unmasked distribution.',
 'Processed IDs carry rewind/healing/EOS caveats; this run recorded zero rewinds and no prefix-present samples.',
 'No GPU/model/API/lifecycle operation was performed by this independent completed-file review.']}
(D/'independent-review.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'review_sha256':sha(D/'independent-review.json'),'protocol':8,'semantic':5,'initial_reasoning':6,'samples':481,'decodes':15}))
