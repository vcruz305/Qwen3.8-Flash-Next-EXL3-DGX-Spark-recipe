import copy,json,os,signal,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import controller
HERE=Path(__file__).resolve().parent
class CustomizationAdmission(unittest.TestCase):
 def admitted(self):return {'usercustomize_loaded':False,'sitecustomize':{'file':controller.SYSTEM_SITE_PATH,'resolved':controller.SYSTEM_SITE_RESOLVED,'sha256':controller.SYSTEM_SITE_SHA}}
 def test_exact_stock_or_absent_customization_only(self):
  controller.validate_customization(self.admitted());controller.validate_customization({'usercustomize_loaded':False,'sitecustomize':None})
  for field,value in [('file','/tmp/sitecustomize.py'),('resolved','/tmp/sitecustomize.py'),('sha256','0'*64)]:
   bad=self.admitted();bad['sitecustomize'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):controller.validate_customization(bad)
  bad=self.admitted();bad['usercustomize_loaded']=True
  with self.assertRaises(ValueError):controller.validate_customization(bad)
 def test_preflight_failure_is_terminal_and_preserves_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   args=types.SimpleNamespace(output=Path(d));p=Path(d)/'result.json';p.write_text(json.dumps({'state':'preflight','setup_check_exit_code':0,'input_hashes':{'fixed':'hash'}}))
   with patch.object(controller,'run',side_effect=ValueError('unadmitted module')):self.assertEqual(controller.run_recorded(args),1)
   value=json.loads(p.read_text());self.assertEqual(value['state'],'preflight_failed');self.assertFalse(value['passed']);self.assertFalse(value['server_launched']);self.assertEqual(value['actual_chat_posts'],0);self.assertTrue(value['finished_at_utc']);self.assertEqual(value['input_hashes'],{'fixed':'hash'});self.assertEqual(value['setup_check_exit_code'],0)
 def test_pre_state_failure_is_terminal(self):
  with tempfile.TemporaryDirectory() as d:
   args=types.SimpleNamespace(output=Path(d))
   with patch.object(controller,'run',side_effect=ValueError('source mismatch')):self.assertEqual(controller.run_recorded(args),1)
   value=json.loads((Path(d)/'result.json').read_text());self.assertEqual(value['state'],'preflight_failed');self.assertTrue(value['finished_at_utc'])
 def test_success_does_not_relabel_existing_result(self):
  with tempfile.TemporaryDirectory() as d:
   args=types.SimpleNamespace(output=Path(d))
   with patch.object(controller,'run',return_value=0):self.assertEqual(controller.run_recorded(args),0)
   self.assertFalse((Path(d)/'result.json').exists())
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

class ObserverAdmission(unittest.TestCase):
 def test_only_server_receives_exact_observer_environment(self):
  helper=controller.load(HERE/'frozen/run_matrix.py',controller.HELPER_SHA,'cpu_observer_matrix')
  guard=controller.selected_ast(HERE/'frozen/strings_only_controller.py',controller.GUARD_SHA,{'SignalGuard'}, {})['SignalGuard']()
  command=['/bin/sleep','30'];observation={'PYTHONPATH':'/synthetic-observer-only','TABBY_STRINGS_OBSERVER_CONFIG':'/synthetic-config'}
  registry=controller.attach_guard(helper,guard,command,observation)
  server=helper.subprocess.Popen(command,env={'PATH':'/bin','QWEN_EXPERIMENT_OWNER':'server-hook-test'},start_new_session=True)
  client=helper.subprocess.Popen(['/bin/sleep','31'],env={'PATH':'/bin','QWEN_EXPERIMENT_OWNER':'client-clean-test'},start_new_session=True)
  try:
   def env(p):return dict(x.split(b'=',1) for x in Path(f'/proc/{p.pid}/environ').read_bytes().split(b'\0') if b'=' in x)
   self.assertEqual(env(server)[b'PYTHONPATH'],observation['PYTHONPATH'].encode());self.assertEqual(env(server)[b'TABBY_STRINGS_OBSERVER_CONFIG'],observation['TABBY_STRINGS_OBSERVER_CONFIG'].encode())
   self.assertNotIn(b'PYTHONPATH',env(client));self.assertNotIn(b'TABBY_STRINGS_OBSERVER_CONFIG',env(client))
  finally:
   outcomes=controller.finalize_owned(helper,guard,registry);self.assertEqual(len(outcomes),2);self.assertTrue(all(x['owned_group_empty'] for x in outcomes))
 def test_undeclared_hook_rejected_before_child_launch(self):
  helper=controller.load(HERE/'frozen/run_matrix.py',controller.HELPER_SHA,'cpu_observer_reject')
  guard=controller.selected_ast(HERE/'frozen/strings_only_controller.py',controller.GUARD_SHA,{'SignalGuard'}, {})['SignalGuard']()
  registry=controller.attach_guard(helper,guard,['/bin/sleep','30'],{'PYTHONPATH':'/approved'})
  with self.assertRaises(ValueError):helper.subprocess.Popen(['/bin/sleep','30'],env={'PYTHONPATH':'/unexpected','QWEN_EXPERIMENT_OWNER':'unlaunched'},start_new_session=True)
  self.assertEqual(registry,[])
if __name__=='__main__':unittest.main()
