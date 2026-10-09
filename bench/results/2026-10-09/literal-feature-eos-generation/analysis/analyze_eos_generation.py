"""Read only completed f4 causal8; source inference and direct observations remain distinct."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'literal-feature-eos-generation-review'
SOURCE=ROOT/'tabby-literal-eos-agent'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
api=json.loads(next(DATA.rglob('generation.json')).read_text());rows={r['name']:r for r in api['results']}
result=[]
for path in sorted((DATA/'raw-generation').glob('request-*.json')):
 trace=json.loads(path.read_text());name=trace['case_name'];call=rows[name]['result'];raw=trace['raw_finish']['native_full_completion'];backend=trace['raw_finish']['backend_full_response'];parsed=json.loads(call['tool_calls'][0]['function']['arguments'])['text']
 assert raw==backend
 assert raw.count('<parameter=text>')==1
 body=raw.split('<parameter=text>',1)[1].split('</parameter>',1)[0]
 if body.startswith('\n'):body=body[1:]
 if body.endswith('\n'):body=body[:-1]
 assert body==parsed
 sampled=[e for e in trace['native_sample_events'] if e['kind']=='native_sample_after']
 assert all(not e['payload']['state']['checkpoint_rewound'] for e in sampled)
 ids=[i for e in sampled for i in e['payload']['processed_token']['ids']]
 eos=[{'position':i+1,'id':token}for i,token in enumerate(ids)if token in (248044,248046)]
 assert eos==[{'position':len(ids),'id':248046}]
 phases=[e for e in trace['producer_events']if e['kind']=='phase_after' and e['applied']]
 transitions=sorted(set(e['state']['new_tokens']for e in phases))
 forcing=[e for e in trace['producer_events']if e['kind']=='force_before']
 thought=trace['start_in_reasoning_mode']
 assert transitions==([25]if thought else[])
 assert len(forcing)==int(thought)
 assert call['finish_reason']=='tool_calls'
 assert call['usage']['prompt_tokens']==trace['input_plan']['prompt_tokens']
 result.append({'name':name,'raw_sha256':sha(path),'semantic_passed':rows[name]['semantic_passed'],'api_finish_reason':call['finish_reason'],'native_backend_equal':True,'physical_parameter_body_equals_api':True,'physical_parameter_body':body,'processed_token_count':len(ids),'implicit_eos_processed':eos,'phase_counter_positions':transitions,'budget_injection_counter_positions':[e['state']['new_tokens']for e in forcing],'decoder_call_count':sum(e['kind']=='native_decode_before'for e in trace['native_sample_events']),'input_plan':trace['input_plan'],'first24_processed_ids':ids[:24],'no_observed_rewind':True,'source_path_inference':{'mandatory_choice':trace['matched_request']['tool_choice'],'initial_reasoning':thought,'native_budget_limit':trace['matched_request']['reasoning_budget_tokens'],'policy_applies':thought,'direct_sampler_mask_observed':False}})
report={'scope':__doc__,'analyzer_sha256':sha(__file__),'top_result_sha256':sha(DATA/'result.json'),'source_commit':'f4aadf114b0044fa8cbe1b50241dc80ea7d61583','source_files':{name:sha(SOURCE/name)for name in ('backends/exllamav3/model.py','backends/exllamav3/reasoning.py','endpoints/OAI/utils/chat_completion.py')},'checks':{'requests':8,'structurally_complete_calls':8,'exact_values':sum(r['semantic_passed']for r in result),'native_backend_equal':8,'physical_body_api_equal':8,'premature_implicit_eos_before_handoff':0},'rows':result,'limitations':['Processed IDs include the final un-emitted stop token; no rewinds observed in this capture.','Raw observer does not serialize sampler masks. Activation is inferred from exact source, retained request, phase/budget inputs and prior actual-backend CPU proof.','Fresh generation8 server omits predecessor coverage8 cache history: matching request/token IDs do not isolate all numerical state.','Remaining wrong argument bodies were present in native output; no output rewriting is proposed.','The input feature remains experimental; this source qualification does not promote its broader semantic guarantees.']}
output=DATA/'independent-review.json'
with output.open('x')as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'sha256':sha(output),'checks':report['checks']}))
