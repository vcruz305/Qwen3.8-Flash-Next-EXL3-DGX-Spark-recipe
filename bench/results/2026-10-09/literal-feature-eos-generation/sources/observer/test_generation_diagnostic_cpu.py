"""Replay retained synthetic HTTP results, plus capture-integrity negatives; no requests."""
import copy,hashlib,importlib.util,json,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import generation_client as client
import assess_capture as assessor

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
observer=load('causal_scope_observer',ROOT/'observer/strings_observer.py')
TRACES=json.loads((ROOT/'fixtures/previous-eight-traces.json').read_text())

class ClientReplay(unittest.TestCase):
 def test_original_eight_native_errors_remain_observations(self):
  api=client.load_api();rows=client.expected();bywire={json.dumps(r['request'],sort_keys=True):TRACES[r['name']]for r in rows};seen=[]
  class Replay:
   def __init__(self,*a,**k):pass
   def request(self,path,request):
    seen.append(copy.deepcopy(request));time.sleep(.03)
    return copy.deepcopy(bywire[json.dumps(request,sort_keys=True)])
  with tempfile.TemporaryDirectory() as tmp,patch.object(api,'Client',Replay),patch.object(client,'load_api',return_value=api):
   args=SimpleNamespace(metadata=ROOT/'fixtures/deployment.json',output=Path(tmp)/'report.json',base_url='http://unused',api_key_env='UNSET_CAUSAL_TEST',timeout=1)
   self.assertEqual(client.run(args),0)
   report=json.loads(args.output.read_text());self.assertTrue(report['passed']);self.assertEqual(report['summary']['collection_complete'],8);self.assertEqual(report['summary']['semantic_passed'],5);self.assertEqual(report['summary']['semantic_failed'],3)
   self.assertEqual(sorted(json.dumps(x,sort_keys=True)for x in seen),sorted(bywire))
 def test_usage_mismatch_fails_collection(self):
  api=client.load_api();row=next(r for r in client.expected()if r['name']=='concurrent_0');trace=copy.deepcopy(TRACES[row['name']]);trace['body']['usage']['prompt_tokens']+=1
  with patch.object(api,'Client',return_value=SimpleNamespace(request=lambda *a:trace)):
   out=client.execute(row,api,SimpleNamespace(base_url='http://unused',api_key_env='UNSET_CAUSAL_TEST',timeout=1))
  self.assertFalse(out['collection_complete']);self.assertFalse(out['semantic_passed'])
 def test_duplicate_arguments_are_not_semantic_success(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):client.unique_json('{"text":"wrong","text":"right"}')
 def test_incomplete_sse_without_server_error_fails_collection(self):
  api=client.load_api();row=next(r for r in client.expected()if r['name']=='multi_turn_user_stream_optin');trace=copy.deepcopy(TRACES[row['name']]);trace.pop('server_error')
  with patch.object(api,'Client',return_value=SimpleNamespace(request=lambda *a:trace)):
   out=client.execute(row,api,SimpleNamespace(base_url='http://unused',api_key_env='UNSET_CAUSAL_TEST',timeout=1))
  self.assertFalse(out['collection_complete'])

class CaptureIntegrity(unittest.TestCase):
 def fixture(self,directory):
  expected=client.expected();rows=[];raw=Path(directory)
  source={'tabby':{'commit':client.TABBY},'engine':{'commit':client.ENGINE},**{k:client.sha(ROOT/p)for k,p in [('observer_sha256','observer/strings_observer.py'),('sitecustomize_sha256','observer/sitecustomize.py'),('native_sample_observer_sha256','observer/native_sample_observer.py')]}}
  audit={'lookup_errors':0,'observation_errors':0,'sample_calls':8,'decode_calls':0,'decoder_hook_installed':True}
  (raw/'observer-manifest.json').write_text(json.dumps({'source':source,'expected_requests_sha256':client.EXPECTED_SHA,'expected_names':[r['name']for r in expected]}))
  (raw/'observer-completion.json').write_text(json.dumps({'completed_records':8,'names':[r['name']for r in expected],'native_observer_audit':audit,'skipped_matching_requests':0}))
  for i,e in enumerate(expected):
   phase_state={'new_tokens':1,'rq_new_tokens':1,'filter_count':0,'checkpoint_rewound':False,'filters_suspended':False}
   native_state={'new_tokens':1,'rq_new_tokens':1,'checkpoint_rewound':False,'prefix_token_present':False,'held_text':{'text':'','length':0,'truncated':False},'held_tokens':{'ids':[],'count':0,'device':'cpu','truncated':False}}
   events=[{'index':0,'kind':'native_sample_before','monotonic_ns':2,'payload':native_state},{'index':1,'kind':'native_sample_after','monotonic_ns':3,'payload':{'state':native_state,'processed_token':{'ids':[248046],'count':1,'device':'cpu','truncated':False},'eos':True,'requeue':False}}]
   phase=[{'index':0,'kind':'sample_processed_before_budget','monotonic_ns':2,'state':phase_state,'token_id':248046,'eos':True},{'index':1,'kind':'sample_processed_after_budget','monotonic_ns':3,'state':phase_state,'token_id':248046,'eos':True}]if e['request']['enable_thinking']else[]
   prompt='synthetic-'+str(i)
   # The actual expected prompt hash is checked independently; use its real prepared prompt here.
   prompt=json.loads((ROOT/'fixtures/expected-prompts.json').read_text())[e['name']]
   record={'case_name':e['name'],'wire_request_sha256':e['request_sha256'],'request_id':str(i),'rendered_prompt':prompt,'rendered_prompt_sha256':e['prompt_sha256'],'start_in_reasoning_mode':e['request']['enable_thinking'],'streaming_mode':e['request']['stream'],'matched_request':{'top_p':.95,'literal_user_control_tokens':True},'input_plan':{k:e[k]for k in ('prompt_tokens','original_prompt_tokens','input_ids_sha256','original_ids_sha256','replacement_count')},'raw_finish':{'native_full_completion':'raw','backend_full_response':'raw','native_metrics':{'prompt_tokens':e['prompt_tokens']}},'native_observer_audit':audit,'native_sample_events':events,'native_sample_events_dropped':0,'producer_events':phase,'producer_events_dropped':0,'started_monotonic_ns':1,'finished_monotonic_ns':4}
   (raw/f'request-{i:02d}.json').write_text(json.dumps(record))
   rows.append({'name':e['name'],'request_sha256':e['request_sha256'],'response_id':'chatcmpl-'+str(i),'trace':{'request':e['request']},'semantic_passed':False,'collection_complete':True})
  return {'completed_at_utc':'CPU','passed':True,'client_sha256':client.sha(ROOT/'generation_client.py'),'expected_requests_sha256':client.EXPECTED_SHA,'results':rows},{'requests':expected}
 def test_integrity_does_not_turn_semantic_failure_into_capture_failure(self):
  with tempfile.TemporaryDirectory()as tmp:
   report,prepared=self.fixture(tmp);out=assessor.assess_capture(tmp,report,prepared)
   self.assertTrue(out['capture_valid']);self.assertEqual(out['semantic_failed'],8)
 def test_count_metadata_and_missing_native_data_fail(self):
  for key,value in [('input_plan',{}),('input_plan',{'prompt_tokens':999}),('native_sample_events_dropped',1),('native_sample_events',[]),('native_observer_audit',{'lookup_errors':1,'observation_errors':0})]:
   with self.subTest(key=key),tempfile.TemporaryDirectory()as tmp:
    report,prepared=self.fixture(tmp);p=Path(tmp)/'request-00.json';d=json.loads(p.read_text());d[key]=value;p.write_text(json.dumps(d))
    with self.assertRaises((ValueError,KeyError)):assessor.assess_capture(tmp,report,prepared)
 def test_completed_api_usage_mismatch_fails_capture(self):
  with tempfile.TemporaryDirectory()as tmp:
   report,prepared=self.fixture(tmp);report['results'][0]['result']={'usage':{'prompt_tokens':1}}
   with self.assertRaisesRegex(ValueError,'API input usage'):assessor.assess_capture(tmp,report,prepared)

if __name__=='__main__':unittest.main()
