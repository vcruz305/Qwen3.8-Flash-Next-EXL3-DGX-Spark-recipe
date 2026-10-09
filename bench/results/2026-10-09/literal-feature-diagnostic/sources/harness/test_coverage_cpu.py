import copy,json,unittest
from pathlib import Path
import coverage_client as client
HERE=Path(__file__).resolve().parent
class Coverage(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((HERE/'plan.json').read_text());cls.parent=json.loads((HERE/'parent-plan.json').read_text());cls.api=client.load_api();cls.trace=json.loads((HERE/'fixtures/previous-api-traces.json').read_text())['requests'][0]
 def test_only_declared_payload_deltas(self):
  original={r['name']:r for r in self.parent['requests']}
  for row in self.plan['requests']:
   name=row['name'].removeprefix('coverage_');p=copy.deepcopy(original[name]['request'])
   if row['group']=='continuation2':p['tool_choice']='auto'
   elif row['group']=='cache6':p['messages'][0]['content']=p['messages'][0]['content'].replace('alpha beta gamma delta 0123456789 '*125,'alpha beta gamma delta 0123456789 '*135,1)
   self.assertEqual(row['request'],p);self.assertEqual(row['request_sha256'],client.digest(p));self.assertEqual(row['original_request_sha256'],original[name]['request_sha256'])
 def test_coverage_first_and_original_semantic_policy_retained(self):
  self.assertEqual([r['coverage_gate'] for r in self.plan['requests']],[True]*8+[False]*8)
  for row in self.plan['requests'][:8]:
   old=next(r for r in self.parent['requests'] if r['name']==row['name'].removeprefix('coverage_'))
   self.assertEqual(row['expected'],old['expected']);self.assertEqual(row['semantic_gate'],old['semantic_gate'])
 def test_exact_prepared_400_without_substring_assumptions(self):
  item=self.plan['requests'][0];prepared={'error_detail':'Exact source-bound rejection without field spelling'}
  trace={'status':400,'frames':[],'done':False,'stream':False,'body':{'detail':prepared['error_detail']}}
  self.assertTrue(client.assess(item,prepared,trace,self.api,self.plan['model'])['passed'])
  trace['body']['detail']='Unrelated error'
  self.assertFalse(client.assess(item,prepared,trace,self.api,self.plan['model'])['passed'])
 def test_actual_preparation_feature_guard_and_bound(self):
  plan,binding,rows=client.validate_binding(HERE/'plan.json',HERE/'prepared-inputs.json','b210985b6c747542753008bd70d40c4195e8af5b03d365415b65c41f96f626ea')
  for item in plan['requests'][:2]:
   prepared=rows[item['name']];self.assertEqual(prepared['validation_stage'],'actual apply_chat_template');self.assertEqual(prepared['error_detail'],'literal_user_control_tokens: Literal user control tokens do not support continued messages')
   trace={'status':400,'frames':[],'done':False,'stream':item['request']['stream'],'body':{'detail':prepared['error_detail']}}
   self.assertTrue(client.assess(item,prepared,trace,self.api,plan['model'])['passed'])
  self.assertEqual([rows[r['name']]['cache_max_tokens'] for r in plan['requests'][2:8]],[0,2323,4118,287,2324,4115])
 def test_rejection_must_precede_sse(self):
  item=self.plan['requests'][0];prepared={'error_detail':'same'}
  for extra in [{'done':True},{'frames':[{}]},{'content_type':'text/event-stream'}]:
   t={'status':400,'frames':[],'done':False,'stream':False,'body':{'detail':'same'},**extra}
   self.assertFalse(client.assess(item,prepared,t,self.api,self.plan['model'])['passed'])
 def test_actual_saved_trace_cached_prefix_is_insufficient(self):
  item=copy.deepcopy(self.plan['requests'][3]);item['expected']={'kind':'exact_tool','value':'<think>literal</think>'};prepared={'prompt_tokens':352,'input_ids':[1]*352,'cache_max_tokens':300}
  for cached,passed in [(0,False),(256,False),(257,True),(300,True),(301,False)]:
   t=copy.deepcopy(self.trace);t['body']['usage']['prompt_tokens_details']['cached_tokens']=cached
   self.assertEqual(client.assess(item,prepared,t,self.api,self.plan['model'])['passed'],passed)
 def test_cold_first_cache_allows_zero_only_with_bound(self):
  item=copy.deepcopy(self.plan['requests'][2]);item['expected']={'kind':'exact_tool','value':'<think>literal</think>'};prepared={'prompt_tokens':352,'input_ids':[1]*352,'cache_max_tokens':0};t=copy.deepcopy(self.trace);t['body']['usage']['prompt_tokens_details']['cached_tokens']=0
  self.assertTrue(client.assess(item,prepared,t,self.api,self.plan['model'])['passed'])
if __name__=='__main__':unittest.main()
