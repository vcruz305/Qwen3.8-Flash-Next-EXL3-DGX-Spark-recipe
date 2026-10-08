#!/usr/bin/env python3
"""Actual-5a formatting and CPU-only hook checks for the bounded literal observer."""
import ast,asyncio,hashlib,importlib.util,json,subprocess,sys,tempfile,types
from pathlib import Path
BASE=Path('/home/vcruz/src/qwen-overnight-20261008')
TABBY=BASE/'tabbyapi-natural-agent';sys.path.insert(0,str(TABBY))
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from tests.test_qwen_nullable_guidance import container,render
ROOT=Path(__file__).resolve().parent
OLD=BASE/'tabbyapi-diagnostics/strings-final-9c-3adc/observer/strings_observer.py'
NEW=ROOT/'observer/strings_observer.py'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
observer=load('literal_only_observer',NEW)
predecessor=load('literal_observer_p1',BASE/'tabbyapi-diagnostics/literal-final-24f0-5a/observer/strings_observer.py')
client=load('unchanged_literal_client',BASE/'reasoning_literal_smoke.py')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def function(path,name):
 t=ast.parse(path.read_text());return ast.dump(next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name),include_attributes=False)
assert sha(OLD)=='77d606e2f2e894f7503a6f41c99a5a94696339b5c15c7b8415b68f8416b945d3'
assert sha(BASE/'reasoning_literal_smoke.py')=='8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054'
assert subprocess.check_output(['git','-C',str(TABBY),'rev-parse','HEAD'],text=True).strip()==observer.TABBY_HEAD
assert not subprocess.check_output(['git','-C',str(TABBY),'status','--porcelain','--untracked-files=no'],text=True).strip()
unchanged=['install_hooks','make_run_wrapper','verified_source','atomic_new','plain','scalar','sha']
for name in unchanged:assert function(OLD,name)==function(NEW,name),name
assert sha(NEW.with_name('sitecustomize.py'))=='3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f'
cfg=json.loads((BASE/'tabbyapi-agent/.tokenizer-cpu/tokenizer_config.json').read_text())
PROMPT=(BASE/'literal-final-triage-5a/rendered-prompt.txt').read_text()
async def main():
 from common.sampling import overrides_from_dict,overrides_container
 deployment_path=BASE/'tabbyapi-diagnostics/literal-final-24f0-5a/live-attempt1/deployment.json'
 deployment=json.loads(deployment_path.read_text())
 actual_sampling=deployment['config']['values']['sampling']
 assert actual_sampling['top_p']=={'force':False,'override':0.95}
 overrides_from_dict(actual_sampling)
 assert overrides_container.effective()==actual_sampling
 rows=[];calls=[];sentinel=object();finish_return={'gen_tokens':17,'finish_reason':'tool_calls'}
 raw='<think>reasoning</think><tool_call><function=record_text><parameter=text><think>literal</think></parameter></function></tool_call>'
 class Native:
  def handle_finish_chunk(self,result,request_id,full_text,label=None):
   calls.append((result,request_id,full_text,label));return finish_return
 native=Native()
 async def original_collector(prompt,request_id,params,streaming_mode,start_in_reasoning_mode):
  result={'full_completion':raw,'new_tokens':17,'eos':True}
  got=native.handle_finish_chunk(result,request_id,raw,'native-label')
  assert got is finish_return and calls[-1][0] is result
  return sentinel
 module=types.SimpleNamespace(_chat_stream_collector=original_collector)
 with tempfile.TemporaryDirectory() as tmp:
  rec=observer.Recorder(Path(tmp)/'records','literal-cpu',8,{'cpu_only':True})
  observer.install_hooks(module,Native,rec)
  for i,v in enumerate(observer.EXPECTED_VARIANTS):
   wire=client.payload('Qwen3.8-Flash-Next-EXL3',v['choice'],v['stream'],unbudgeted=v['unbudgeted'])
   params=ChatCompletionRequest(**wire)
   prompt=(await render(params,container(raw_template=cfg['chat_template'])))[0]
   assert prompt==PROMPT and hashlib.sha256(prompt.encode()).hexdigest()==observer.PROMPT_SHA256
   assert observer.matching_request(params,v['stream'])
   assert params.top_p==0.95 and params.temperature==0 and params.top_k==1
   assert not predecessor.matching_request(params,v['stream'])
   got=await module._chat_stream_collector(prompt,'cpu-literal-'+str(i),params,v['stream'],True)
   assert got is sentinel
   record=json.loads((rec.directory/f'request-{i:02d}.json').read_text())
   assert record['variant']==v and record['request_id']=='cpu-literal-'+str(i)
   assert record['raw_finish']['native_full_completion']==raw
   assert record['raw_finish']['backend_full_response']==raw
   assert record['matched_request']['tool_choice']==wire['tool_choice']
   assert record['matched_request']['reasoning_budget_tokens']==wire.get('reasoning_budget_tokens')
   assert record['start_in_reasoning_mode'] is True and record['raw_finish']['returned_metrics']['gen_tokens']==17
   rows.append({'variant':v,'real_formatting_matches':True,'published_json_matches':True,'native_and_backend_raw_preserved':True,'return_identity_preserved':True})
  assert rec.count==8 and not rec.active
  assert await module._chat_stream_collector(PROMPT,'cpu-extra',params,True,True) is sentinel
  assert rec.count==8 and rec.skipped_matching_requests==1 and len(list(rec.directory.glob('request-*.json')))==8
  rejected=[]
  for key,value in [('tool_choice','auto'),('max_tokens',255),('parallel_tool_calls',True),('reasoning_budget_tokens',25),('temperature',0.1),('top_k',2),('top_p',1.0),('n',2)]:
   bad=params.model_copy(deep=True);setattr(bad,key,value)
   assert not observer.matching_request(bad,True),key;rejected.append(key)
  for key,value in [('enable_thinking',False),('tool_choice','auto'),('unexpected_scope_key',True)]:
   bad=params.model_copy(deep=True);bad.template_vars[key]=value
   assert not observer.matching_request(bad,True),key;rejected.append('template_vars.'+key)
  bad=params.model_copy(deep=True);bad.messages[0].content+=' extra'
  assert not observer.matching_request(bad,True);rejected.append('message')
  bad=params.model_copy(deep=True);bad.tools[0].function.name='different_tool'
  assert not observer.matching_request(bad,True);rejected.append('tool_schema')
  assert not observer.matching_request(params,False);rejected.append('stream')
  fresh=observer.Recorder(Path(tmp)/'rejections','literal-rejections',8)
  assert fresh.begin('no-phase',PROMPT,params,True,False) is None
  assert fresh.begin('wrong-prompt',PROMPT+' ',params,True,True) is None
  assert fresh.count==0 and not fresh.active
  for cap in (0,9,True):
   try:observer.Recorder(Path(tmp)/('bad-'+str(cap)),'bad-cap',cap)
   except ValueError:pass
   else:raise AssertionError(cap)
 result={'actual_sampling_defaults':actual_sampling,'deployment_sha256':sha(deployment_path),'old_literal_matcher_rejects_all_eight':True,'passed':True,'observer_sha256':sha(NEW),'sitecustomize_sha256':sha(NEW.with_name('sitecustomize.py')),'test_source_sha256':sha(Path(__file__)),'old_observer_sha256':sha(OLD),'unchanged_function_asts':unchanged,'actual_formatted_and_hook_cases':rows,'negative_request_fields':rejected,'cap_and_initial_phase_prompt_guards_passed':True,'scope':'Actual clean5a Pydantic/formatting with original eight client payloads, then fake native finish/collector on CPU. No API, CUDA, model or production source changes. Existing hook ASTs and sitecustomize bytes match the previously reviewed observer.'}
 with (ROOT/'observer-cpu-report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'passed':True,'actual_formatted_hook_cases':len(rows),'negative_request_guards':len(rejected),'report_sha256':sha(ROOT/'observer-cpu-report.json'),'observer_sha256':sha(NEW)},indent=2))
asyncio.run(main())
