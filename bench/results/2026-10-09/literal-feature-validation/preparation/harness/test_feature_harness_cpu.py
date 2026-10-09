"""Offline protocol and actual harmless-process lifecycle checks; no GPU/API."""
import copy,json,os,signal,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import client,controller
HERE=Path(__file__).resolve().parent
class Protocol(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.api=client.load_api();cls.plan=json.loads((HERE/'plan.json').read_text())
  cls.traces=json.loads((HERE/'fixtures/previous-api-traces.json').read_text())['requests']
 def positive(self,stream=False):
  item=copy.deepcopy(next(x for x in self.plan['requests'] if x['group']=='known8' and x['request']['stream']==stream and x['policy']=='optin'))
  trace=copy.deepcopy(self.traces[int(stream)]);prepared={'prompt_tokens':352,'input_ids':[1]*352}
  return item,prepared,trace
 def test_actual_saved_nonstream_and_stream_responses(self):
  for mode in (False,True):
   item,p,t=self.positive(mode);r=client.assess(item,p,t,self.api,self.plan['model'])
   self.assertTrue(r['passed'],r)
 def test_native_wrong_literal_remains_observation(self):
  item,p,t=self.positive();item['semantic_gate']=False;item['expected']['value']='different'
  r=client.assess(item,p,t,self.api,self.plan['model'])
  self.assertTrue(r['protocol_passed']);self.assertFalse(r['semantic_passed']);self.assertTrue(r['passed'])
 def test_optin_wrong_literal_fails(self):
  item,p,t=self.positive();item['expected']['value']='different'
  self.assertFalse(client.assess(item,p,t,self.api,self.plan['model'])['passed'])
 def test_actual_prompt_count_must_match_bound_ids(self):
  item,p,t=self.positive();p['prompt_tokens']=348
  self.assertFalse(client.assess(item,p,t,self.api,self.plan['model'])['protocol_passed'])
 def test_missing_done_is_protocol_failure_even_native(self):
  item,p,t=self.positive(True);item['semantic_gate']=False;t['done']=False
  self.assertFalse(client.assess(item,p,t,self.api,self.plan['model'])['passed'])
 def test_wrong_response_model_is_protocol_failure(self):
  item,p,t=self.positive();t['body']['model']='another'
  self.assertFalse(client.assess(item,p,t,self.api,self.plan['model'])['protocol_passed'])
 def test_cache_beyond_lcp_rejected(self):
  item,p,t=self.positive();item['group']='cache6';p['cache_max_tokens']=0;t['body']['usage']['prompt_tokens_details']['cached_tokens']=256
  self.assertFalse(client.assess(item,p,t,self.api,self.plan['model'])['passed'])
 def test_zero_cache_does_not_qualify_reuse(self):
  item,p,t=self.positive();item['group']='cache6';item['require_observed_reuse']=True;p['cache_max_tokens']=300;t['body']['usage']['prompt_tokens_details']['cached_tokens']=0
  self.assertFalse(client.assess(item,p,t,self.api,self.plan['model'])['passed'])
 def test_positive_cache_within_bound(self):
  item,p,t=self.positive();item['group']='cache6';item['require_observed_reuse']=True;p['cache_max_tokens']=300;t['body']['usage']['prompt_tokens_details']['cached_tokens']=256
  self.assertTrue(client.assess(item,p,t,self.api,self.plan['model'])['passed'])
 def test_400_requires_exact_feature_diagnostic(self):
  item=copy.deepcopy(next(x for x in self.plan['requests'] if x['name']=='list_content_nonstream'));prepared={'error_detail':'literal_user_control_tokens: require strings'}
  base={'status':400,'frames':[],'done':False,'stream':False}
  for body,good in [({'detail':prepared['error_detail']},True),({'error':{'message':prepared['error_detail']}},True),({'detail':'Bad model'},False),({'detail':'literal_user_control_tokens: another problem'},False),({},False)]:
   with self.subTest(body=body):
    self.assertEqual(client.assess(item,prepared,dict(base,body=body),self.api,self.plan['model'])['passed'],good)
 def test_422_requires_exact_field_and_type(self):
  item=copy.deepcopy(next(x for x in self.plan['requests'] if x['name']=='strict_bool_stream'));prepared={'error_types':['bool_type']}
  correct={'loc':['body','literal_user_control_tokens'],'type':'bool_type'}
  for details,good in [([correct],True),([dict(correct,loc=['body','messages'])],False),([dict(correct,type='string_type')],False),([correct,correct],False),([],False),('error',False)]:
   t={'status':422,'frames':[],'done':False,'stream':True,'body':{'detail':details}}
   with self.subTest(details=details):self.assertEqual(client.assess(item,prepared,t,self.api,self.plan['model'])['passed'],good)
 def test_validation_must_happen_before_sse(self):
  item=next(x for x in self.plan['requests'] if x['name']=='strict_bool_stream');t={'status':422,'frames':[{}],'done':False,'stream':True,'body':{'detail':[{'loc':['body','literal_user_control_tokens'],'type':'bool_type'}]}}
  self.assertFalse(client.assess(item,{'error_types':['bool_type']},t,self.api,self.plan['model'])['passed'])
 def test_exact_unicode_whitespace_and_duplicate_keys(self):
  expected={'kind':'exact_tool','value':'é 中文\n    <think>x</think>\n','reasoning_empty':True}
  r={'finish_reason':'tool_calls','content':'','reasoning_content':'','tool_calls':[{'function':{'name':'record_text','arguments':json.dumps({'text':expected['value']})}}]}
  client.semantic(r,expected)
  for bad in [expected['value'].strip(),expected['value'].replace('é','e\u0301')]:
   r['tool_calls'][0]['function']['arguments']=json.dumps({'text':bad})
   with self.assertRaises(ValueError):client.semantic(r,expected)
  r['tool_calls'][0]['function']['arguments']='{"text":"x","text":"'+expected['value'].replace('\n','\\n')+'"}'
  with self.assertRaises(ValueError):client.semantic(r,expected)
class Lifecycle(unittest.TestCase):
 def helper(self):return controller.load(HERE/'frozen/run_matrix.py',controller.HELPER_SHA,'cpu_matrix_guard')
 def guard(self):return controller.selected_ast(HERE/'frozen/strings_only_controller.py',controller.GUARD_SHA,{'SignalGuard'}, {})['SignalGuard']()
 def test_diagnostic_environment_removed(self):
  e=controller.clean_env({'PATH':'/bin','PYTHONPATH':'/hook','PYTHONHOME':'/bad','QWEN_OBSERVER_CONFIG':'x','STRINGS_OBSERVER_CONFIG':'y','EXL3_REF':'good'})
  self.assertEqual(e,{'PATH':'/bin','EXL3_REF':'good'});controller.no_hooks(e)
  with self.assertRaises(ValueError):controller.no_hooks({'PYTHONPATH':''})
 def test_pending_launch_signal_registered_and_term_only_cleanup(self):
  h=self.helper();g=self.guard();real=subprocess.Popen
  def interrupted_popen(*a,**k):
   p=real(*a,**k);os.kill(os.getpid(),signal.SIGTERM);return p
  h.subprocess=types.SimpleNamespace(**vars(subprocess));h.subprocess.Popen=interrupted_popen
  controller.attach_guard(h,g,['never-server'])
  previous=signal.getsignal(signal.SIGTERM);signal.signal(signal.SIGTERM,g.interrupt)
  proc=None
  try:
   proc=h.subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,QWEN_EXPERIMENT_OWNER='fixture-owned'),start_new_session=True)
   raw=Path(f'/proc/{proc.pid}/status').read_text();masked=int(next(x.split(':')[1].strip() for x in raw.splitlines() if x.startswith('SigBlk:')),16)
   self.assertFalse(masked & (1<<(signal.SIGTERM-1)));self.assertFalse(masked & (1<<(signal.SIGINT-1)))
   with self.assertRaises(KeyboardInterrupt):proc.poll()
   result=h.stop_owned(proc,'fixture-owned')
   self.assertEqual(result['signals'],['SIGTERM']);self.assertEqual(result['exit_code'],-signal.SIGTERM)
  finally:
   signal.signal(signal.SIGTERM,previous)
   if proc and proc.returncode is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 def test_cleanup_rejects_foreign_token(self):
  h=self.helper();p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,QWEN_EXPERIMENT_OWNER='right'),start_new_session=True)
  try:
   with self.assertRaises(RuntimeError):h.group_members(p,'wrong')
   self.assertIsNone(p.poll())
  finally:h.stop_owned(p,'right')
 def test_exact_server_finalbody_first_signal_still_cleans(self):
  import ast
  h=self.helper();g=self.guard();controller.attach_guard(h,g,['never-server'])
  tree=ast.parse((HERE/'frozen/run_matrix.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_job');outer=next(n for n in fn.body if isinstance(n,ast.Try) and n.finalbody)
  stopped=[]
  class Server:
   def poll(self):g.interrupt(signal.SIGTERM,None);return None
  ns={'server':Server(),'record':{'state':'measuring','passed':True},'server_log':None,'stop_owned':lambda *_:stopped.append(True) or {'signals':['SIGTERM'],'exit_code':0},'token':'owned','stamp':lambda:'now','save':lambda:None}
  exec(compile(ast.Module(body=outer.finalbody,type_ignores=[]),'exact-helper-finalbody','exec'),ns)
  self.assertEqual(stopped,[True]);self.assertEqual(g.pending,signal.SIGTERM)
 def test_outer_registry_cleans_after_escape_from_inherited_finally(self):
  h=self.helper();g=self.guard();registry=controller.attach_guard(h,g,['never-server'])
  p=h.subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,QWEN_EXPERIMENT_OWNER='outer-owned'),start_new_session=True)
  try:
   g.interrupt(signal.SIGTERM,None)
   with self.assertRaises(KeyboardInterrupt):p.poll()
   out=controller.finalize_owned(h,g,registry)
   self.assertEqual(len(out),1);self.assertEqual(out[0]['signals'],['SIGTERM']);self.assertTrue(out[0]['owned_group_empty'])
  finally:
   if p.returncode is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
 def test_first_and_repeated_signals_during_outer_cleanup_are_deferred(self):
  h=self.helper();g=self.guard();registry=controller.attach_guard(h,g,['never-server'])
  p=h.subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,QWEN_EXPERIMENT_OWNER='outer-owned'),start_new_session=True)
  original=h.stop_owned
  def signal_then_stop(*a):g.interrupt(signal.SIGTERM,None);g.interrupt(signal.SIGINT,None);return original(*a)
  h.stop_owned=signal_then_stop
  try:
   out=controller.finalize_owned(h,g,registry)
   self.assertEqual(out[0]['signals'],['SIGTERM']);self.assertEqual(g.pending,signal.SIGTERM);self.assertTrue(out[0]['owned_group_empty'])
  finally:
   if p.returncode is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
 def test_outer_cleanup_continues_after_one_ownership_refusal(self):
  h=self.helper();g=self.guard();registry=controller.attach_guard(h,g,['never-server'])
  a=h.subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,QWEN_EXPERIMENT_OWNER='a'),start_new_session=True)
  b=h.subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],env=dict(os.environ,QWEN_EXPERIMENT_OWNER='b'),start_new_session=True)
  registry[-1]=(b,'wrong')
  try:
   out=controller.finalize_owned(h,g,registry)
   self.assertFalse(out[0]['owned_group_empty']);self.assertTrue(out[1]['owned_group_empty']);self.assertIsNone(b.poll())
  finally:
   h.stop_owned(b,'b')
   if a.returncode is None:h.stop_owned(a,'a')
 def test_frozen_helper_drift_refused(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'bad.py';p.write_text('raise RuntimeError("must not execute")')
   with self.assertRaises(ValueError):controller.load(p,controller.HELPER_SHA,'bad')
if __name__=='__main__':unittest.main()
