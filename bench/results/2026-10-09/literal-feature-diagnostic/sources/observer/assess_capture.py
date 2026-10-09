"""Pure CPU integrity checks; generated semantics and raw differences are observations."""
from pathlib import Path
import hashlib,importlib.util,json
HERE=Path(__file__).resolve().parent
TABBY='a70ae1fa9e457e478c3d96bdc84012a3cb331796'
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
EXPECTED_SHA='74e477ee1917f64264205cd1cac9665058ec2476fe5b49b2289d3fe857bcc52c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def read(value):return json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else value
def ids(value):
 require(isinstance(value,dict) and not value.get('not_read') and not value.get('truncated'),'Native IDs unavailable/truncated')
 require(value.get('device')in('cpu','python'),'Non-CPU native IDs')
 values=value.get('ids');require(isinstance(values,list) and all(type(v)is int and v>=0 for v in values),'Invalid native IDs')
 require(value.get('count')==len(values),'Native ID count differs')
 return values
def validate_native(trace):
 events=trace.get('native_sample_events');require(isinstance(events,list) and 1<=len(events)<=2048,'Missing native sample observations')
 require(trace.get('native_sample_events_dropped')==0,'Native events were dropped')
 stack=[];samples=[];decoder_calls=0;last=-1
 def state(v):
  require(isinstance(v,dict),'Missing native state')
  require(type(v.get('new_tokens'))is int and type(v.get('rq_new_tokens'))is int,'Missing native counts')
  require(type(v.get('checkpoint_rewound'))is bool,'Missing native rewind state')
  held=v.get('held_text');require(isinstance(held,dict) and not held.get('truncated') and type(held.get('text'))is str,'Held text missing/truncated')
  require(held['length']==len(held['text']),'Held text length differs')
  ids(v['held_tokens'])
 for index,event in enumerate(events):
  require(event.get('index')==index,'Native event index gap');now=event.get('monotonic_ns')
  require(type(now)is int and now>=last and trace['started_monotonic_ns']<=now<=trace['finished_monotonic_ns'],'Native event timestamp outside record');last=now
  kind=event.get('kind');payload=event.get('payload')
  require(isinstance(payload,dict),'Missing native event payload')
  if kind=='native_sample_before':state(payload);stack.append('sample')
  elif kind=='native_decode_before':
   require(stack and stack[-1]=='sample','Decoder outside sampled job');ids(payload['input']);stack.append('decode');decoder_calls+=1
  elif kind=='native_decode_after':
   require(stack and stack.pop()=='decode','Unbalanced native decoder')
   texts=payload.get('result');texts=[texts] if isinstance(texts,dict) else texts
   require(isinstance(texts,list) and texts and all(isinstance(t,dict) and not t.get('truncated') and t.get('length')==len(t.get('text','')) for t in texts),'Decoder result missing/truncated')
  elif kind=='native_sample_after':
   require(stack and stack.pop()=='sample','Unbalanced native sample');state(payload['state'])
   values=ids(payload['processed_token']);require(len(values)==1,'Expected one processed native token')
   require(type(payload.get('eos'))is bool and type(payload.get('requeue'))is bool,'Missing native EOS/requeue')
   samples.append({'token_id':values[0],'new_tokens':payload['state']['new_tokens'],'eos':payload['eos'],'rewound':payload['state']['checkpoint_rewound'],'healing':payload['state']['prefix_token_present']})
  else:raise ValueError('Unexpected native observation exception/kind: '+str(kind))
 require(not stack and samples and samples[-1]['eos'],'Native sample trace incomplete')
 return {'processed_samples':len(samples),'decoder_calls':decoder_calls,'processed_token_ids':[s['token_id'] for s in samples],'rewound_samples':sum(s['rewound']for s in samples),'scope':'Processed IDs, not automatically final accepted output; rewind/healing/EOS state retained.'}
def assess_capture(raw_dir,generation_report,prepared_inputs):
 raw_dir=Path(raw_dir);report=read(generation_report);binding=read(prepared_inputs)
 expected_path=HERE/'observer/expected_requests.json';require(sha(expected_path)==EXPECTED_SHA,'Expected request manifest changed')
 expected={r['name']:r for r in read(expected_path)['requests']}
 require(report.get('completed_at_utc') and len(report.get('results',[]))==8,'Generation client incomplete')
 require(report.get('passed')is True and all(r.get('collection_complete')is True for r in report['results']),'Generation collection integrity failed')
 require(report.get('client_sha256')==sha(HERE/'generation_client.py'),'Generation client source differs')
 client={r['name']:r for r in report['results']};require(set(client)==set(expected),'Generation request names differ')
 require(report.get('expected_requests_sha256')==EXPECTED_SHA,'Client request binding differs')
 bound={r['name']:r for r in binding['requests']}
 require(all(n in bound for n in expected),'CPU preparation missing causal requests')
 manifest=read(raw_dir/'observer-manifest.json');completion=read(raw_dir/'observer-completion.json');source=manifest['source']
 require(source['tabby']['commit']==TABBY and source['engine']['commit']==ENGINE,'Observed source pins differ')
 for key,path in [('observer_sha256','observer/strings_observer.py'),('sitecustomize_sha256','observer/sitecustomize.py'),('native_sample_observer_sha256','observer/native_sample_observer.py')]:require(source[key]==sha(HERE/path),'Observed source bytes differ: '+key)
 require(manifest['expected_requests_sha256']==EXPECTED_SHA and set(manifest['expected_names'])==set(expected),'Observed scope differs')
 require(completion['completed_records']==8 and set(completion['names'])==set(expected) and len(completion['names'])==8 and completion['skipped_matching_requests']==0,'Recorder incomplete/duplicate/skipped')
 audit=completion['native_observer_audit'];require(audit['lookup_errors']==audit['observation_errors']==0 and audit['decoder_hook_installed']is True,'Native observation errors')
 paths=sorted(raw_dir.glob('request-*.json'));require(len(paths)==8,'Expected eight raw records')
 spec=importlib.util.spec_from_file_location('bounded_phase_validator',HERE/'validate_timeline.py');phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)
 seen=set();rows=[]
 for path in paths:
  trace=read(path);name=trace['case_name'];require(name in expected and name not in seen,'Unexpected/duplicate observed name');seen.add(name)
  e=expected[name];c=client[name];b=bound[name];wire=e['request']
  require(c['request_sha256']==trace['wire_request_sha256']==e['request_sha256']==b['request_sha256'],'Original request hash differs')
  require(c['trace']['request']==wire,'Original wire request differs')
  require(trace['request_id']==c['response_id'].removeprefix('chatcmpl-').removeprefix('cmpl-'),'Response/capture ID differs')
  require(trace['rendered_prompt_sha256']==e['prompt_sha256']==hashlib.sha256(trace['rendered_prompt'].encode()).hexdigest(),'Prompt text/hash differs')
  require(trace['start_in_reasoning_mode']is wire['enable_thinking'] and trace['streaming_mode']is wire['stream'],'Phase/mode scope differs')
  require(trace['matched_request']['top_p']==0.95 and trace['matched_request']['literal_user_control_tokens']is True,'Effective sampling/policy differs')
  required_plan={'prompt_tokens','original_prompt_tokens','input_ids_sha256','original_ids_sha256','replacement_count'}
  require(isinstance(trace.get('input_plan'),dict) and set(trace['input_plan'])==required_plan,'Immutable input plan missing fields')
  require(all(type(trace['input_plan'][k])is int for k in ('prompt_tokens','original_prompt_tokens','replacement_count')),'Input plan counts not integers')
  for k,v in trace['input_plan'].items():require(v==e[k]==b[k],'Immutable input plan differs: '+k)
  native=trace['raw_finish'];require(isinstance(native['native_full_completion'],str) and isinstance(native['backend_full_response'],str),'Raw finish strings missing')
  require(native['native_metrics']['prompt_tokens']==e['prompt_tokens'],'Native input usage differs')
  if c.get('result'):require(c['result']['usage']['prompt_tokens']==e['prompt_tokens'],'API input usage differs')
  require(trace['native_observer_audit']['lookup_errors']==trace['native_observer_audit']['observation_errors']==0,'Per-record observation error')
  native_summary=validate_native(trace)
  if trace['producer_events']:phase_summary=phase.validate_timeline(trace)
  else:
   require(not wire['enable_thinking'] and trace['producer_events_dropped']==0,'Missing reasoning trace')
   phase_summary={'capture_complete':True,'event_count':0,'scope':'No reasoning watcher for this thinking-off request.'}
  rows.append({'name':name,'raw_sha256':sha(path),'semantic_passed':c['semantic_passed'],'outcome_error':c.get('outcome_error'),'native_equals_backend':native['native_full_completion']==native['backend_full_response'],'input_tokens':e['prompt_tokens'],'native':native_summary,'phase':phase_summary})
 require(sum(r['native']['processed_samples']for r in rows)==audit['sample_calls'],'Native audit sample count differs')
 require(sum(r['native']['decoder_calls']for r in rows)==audit['decode_calls'],'Native decoder count differs')
 return {'capture_valid':True,'records':8,'semantic_passed':sum(r['semantic_passed']for r in rows),'semantic_failed':sum(not r['semantic_passed']for r in rows),'rows':rows,'native_audit':audit,'scope':'Integrity only; processed token IDs need rewind/healing/EOS interpretation. Model failures and native/backend differences remain observations.'}
