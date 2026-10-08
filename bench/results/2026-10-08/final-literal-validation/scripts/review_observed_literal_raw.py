"""Offline review of retained raw output; original failed assessor is unchanged."""
import asyncio,hashlib,importlib.util,json,sys
from pathlib import Path
BASE=Path('/home/vcruz/src/qwen-overnight-20261008')
DIRECTORY=BASE/'tabbyapi-diagnostics/literal-final-24f0-5a-p095/live-capture'
HELPER=BASE/'tabbyapi-diagnostics/strings-parser-replay/analyze_strings.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='022133f56829740ec5cd601792c0fb08dd8140299bbcad4337a2df7e4373f1b8'
spec=importlib.util.spec_from_file_location('reviewed_parser_replay',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
sys.path.insert(0,str(BASE/'recipe/bench'))
from api_resilience import assembled
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
async def main():
 source=helper.source_identity()
 hashes=read(DIRECTORY/'copied-file-hashes.json')
 assert all(sha(DIRECTORY/path)==value for path,value in hashes.items())
 cap=read(DIRECTORY/'literal-capture.json')
 assert cap['capture_valid'] is False and cap['capture_error']=='ValueError: Observed sampler/control changed: top_p'
 manifest=read(DIRECTORY/'raw-literal/observer-manifest.json')
 assert manifest['source']['engine']['commit']==helper.ENGINE_HEAD and manifest['source']['tabby']['commit']==helper.TABBY_HEAD
 assert manifest['source']['observer_sha256']=='674453a31893defd299939fe801cd58b48bdb20c7bbfaf0ec0c0fbb5f82418f6'
 assert len(list((DIRECTORY/'raw-literal').glob('request-*.json')))==8
 rows=[]
 for index in range(8):
  path=DIRECTORY/'raw-literal'/f'request-{index:02d}.json';trace=read(path)
  v=trace['variant'];name='literal_reasoning_'+v['choice']+('_stream' if v['stream'] else '_nonstream')
  report=read(DIRECTORY/('literal-unbudgeted.json' if v['unbudgeted'] else 'literal.json'))
  case=next(r for r in report['cases'] if r['name']==name)
  [request_index]=case['request_indices'];wire=report['requests'][request_index]
  request=wire['request']
  observed=assembled(wire,report['requested_model'])
  assert observed['id']==('chatcmpl-' if v['stream'] else 'cmpl-')+trace['request_id']
  for field in ('model','messages','tools','tool_choice'):
   assert request[field]==trace['matched_request'][field]
  assert trace['index']==index and trace['start_in_reasoning_mode'] is True
  assert trace['raised_exception_type'] is None and trace['collector_returned_error'] is False
  assert hashlib.sha256(trace['rendered_prompt'].encode()).hexdigest()==trace['rendered_prompt_sha256']=='32655bc461bfa9685942882754b89e75f6640a5605004d4a3d609ebfc6076f58'
  native=trace['raw_finish']['native_full_completion'];backend=trace['raw_finish']['backend_full_response']
  assert native==backend
  actual=helper.normalized(observed,observed['finish_reason'])
  finish=trace['raw_finish']['returned_metrics']
  a=await helper.partitions(native,request,True,finish)
  b=await helper.partitions(backend,request,True,finish)
  matched=all(r['result']==actual for r in a+b)
  # Independent literal span review: the sole closing parameter ends the first
  # generated record_text parameter. Nested opening tags are retained as data.
  opener='<parameter=text>';closer='</parameter>'
  assert native.count(closer)==1 and opener in native
  body=native.split(opener,1)[1].split(closer,1)[0]
  unframed=body[1:] if body.startswith('\n') else body
  unframed=unframed[:-1] if unframed.endswith('\n') else unframed
  value=actual['tool_calls'][0]['arguments'].get('text') if len(actual['tool_calls'])==1 else None
  physical_matches=unframed==value
  rows.append({'index':index,'variant':v,'request_id':trace['request_id'],'trace_sha256':sha(path),
    'original_case_status':case['status'],'original_error':case.get('error'),
    'observed_api':actual,'native_full_completion':native,'backend_full_response':backend,
    'raw_first_parameter_body':body,'raw_parameter_opening_count':native.count(opener),
    'raw_parameter_closing_count':native.count(closer),'raw_body_after_single_framing_lfs':unframed,
    'raw_body_equals_api_value':physical_matches,'native_equals_backend':True,
    'all16_replays_equal_api':matched,'native_replays':a,'backend_replays':b,
    'raw_value_matches_requested':unframed=='<think>literal</think>'})
 result={'scope':__doc__,'source':source,'review_source_sha256':sha(Path(__file__)),
   'original_capture_valid':False,'original_capture_error':cap['capture_error'],
   'raw_records':8,'replay_count':128,'all_native_equals_backend':True,
   'all_replays_equal_api':all(r['all16_replays_equal_api'] for r in rows),
   'all_raw_parameter_bodies_equal_api_values':all(r['raw_body_equals_api_value'] for r in rows),
   'exact_requested_values':sum(r['raw_value_matches_requested'] for r in rows),
   'wrong_values_already_in_native_parameter_body':sum(not r['raw_value_matches_requested'] and r['raw_body_equals_api_value'] for r in rows),
   'input_copy_hashes':hashes,'cases':rows,
   'limitation':'Original capture assessor failed on recorded scalar metadata and remains unchanged. These separate offline checks bind source, exact request/response IDs, copied hashes and raw text; ScalarFloat type reconciliation is separate. Does not establish the cause of model token selection or historical uncaptured failures.'}
 out=BASE/'tabbyapi-diagnostics/literal-final-24f0-5a-p095/raw-parser-review.json'
 with out.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps({'path':str(out),'sha256':sha(out),**{k:result[k] for k in ('raw_records','replay_count','all_replays_equal_api','all_raw_parameter_bodies_equal_api_values','exact_requested_values','wrong_values_already_in_native_parameter_body')}}))
asyncio.run(main())
