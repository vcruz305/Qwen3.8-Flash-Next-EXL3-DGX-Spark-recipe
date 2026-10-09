#!/usr/bin/env python3
"""One source-bound feature qualification. Parent schedules GPU; this owns only its children."""
from __future__ import annotations
import argparse,ast,datetime,fcntl,hashlib,json,os,re,signal,stat,subprocess,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
TABBY='a70ae1fa9e457e478c3d96bdc84012a3cb331796'
RECIPE='254b2b03027f25094845dc31f5f87739f3584d2e'
PLAN_SHA='ca62602653e1e872481a568c4005218a5cc5388063b76f012312a7ae47827645'
HELPER_SHA='7e83208d473a124c6a90f8f07de0f338dea7d0510c6cd03f83d63f8cb31e103a'
GUARD_SHA='cfc3619894c4ee8f07d46c2ee7ed0fd085f3fdc044c81c8d9a01d009304c4116'
OWNERSHIP_SHA='2a0a0e3c36f482581847dc9c1e84ebc03dc72014006ed45c8b23cd348d0bed2e'
MODEL_SHA='6dc2117bb602a3f57e798613c005c0c1ec8ad519378168181b00465e40b90840'
LOCK='/home/cruzspark/redsnow-gpu.lock'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(x,message):
 if not x:raise ValueError(message)
def command(argv,**kwargs):return subprocess.check_output(argv,text=True,timeout=30,**kwargs).strip()
def load(p,expected,name):
 require(sha(p)==expected,'Frozen helper changed: '+str(p))
 mod=types.ModuleType(name);mod.__file__=str(p);sys.modules[name]=mod
 exec(compile(p.read_bytes(),str(p),'exec'),mod.__dict__);return mod
def selected_ast(p,expected,names,namespace):
 require(sha(p)==expected,'Frozen lifecycle source changed')
 tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
 require({n.name for n in nodes}==set(names),'Lifecycle definitions missing')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
 return namespace
def clean_env(env):
 return {k:v for k,v in env.items() if k not in {'PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONINSPECT'} and 'OBSERVER' not in k and not k.startswith('QWEN_')}
def no_hooks(env):
 require(not any(k in env for k in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONINSPECT')),'Python hook environment present')
 require(not any('OBSERVER' in k for k in env),'Diagnostic observer environment present')
def check_sources(identity,args):
 for name,pin in [('recipe',RECIPE),('engine',ENGINE),('server',TABBY)]:
  require(identity[name]['commit']==pin and not identity[name]['tracked_changes'],'Expected clean exact '+name+' source')
 require(identity['packages_sha256']==args.packages_sha256,'Installed package set differs from admitted fingerprint')
def verify_deployment(deployment,current,binding,model_path):
 config=deployment['config']['values'];model=config['model'];contract=binding['runtime_contract']
 for key in ('ngram_ram','max_seq_len','cache_size','max_batch_size','chunk_size','tool_format','vision'):
  require(model.get(key)==contract[key],'Actual config differs from prepared '+key)
 require(model.get('cache_mode')=='8,8' and model.get('reasoning') is True,'Cache/reasoning mode differs')
 require(config['draft_model']=={'draft_cache_mode':'8,8','draft_mode':'mtp','draft_num_tokens':5,'dynamic_draft':True},'Actual MTP settings differ')
 require(deployment['environment'].get('PROFILE')=='concurrent' and deployment['environment'].get('EXL3_DRAFT_ROW_BUDGET')=='8' and deployment['environment'].get('EXL3_DRAFT_CONFIDENCE')=='0.6','Actual measured profile differs')
 require(not model.get('template_vars_default') and not model.get('template_vars_force'),'Template overrides differ from input binding')
 sampling=config.get('sampling',{})
 require(all(not v.get('force') for v in config.get('sampling',{}).values()) and sampling==binding['sampling_defaults'],'Actual sampling defaults differ')
 tokenizer_config=Path(model_path)/'tokenizer_config.json'
 require(sha(tokenizer_config)==binding['template_config_sha256'] and sha(Path(model_path)/'tokenizer.json')==binding['tokenizer_sha256'],'Tokenizer assets differ')
 template=json.loads(tokenizer_config.read_text())['chat_template']
 require(current.get('parameters',{}).get('prompt_template_content')==template,'Loaded template differs from CPU preparation')
 for key in ('max_seq_len','cache_size','max_batch_size','chunk_size'):
  require(current.get('parameters',{}).get(key)==contract[key],'Loaded geometry differs: '+key)

def attach_guard(helper,guard,server_command):
 """Deferral until the helper assigns each child; no inherited signal masks."""
 original=helper.subprocess.Popen;helper.subprocess=types.SimpleNamespace(**vars(helper.subprocess))
 launches=[]
 def popen(argv,*a,**kw):
  if argv==server_command:
   require(not launches,'Only one server launch allowed');launches.append(argv)
  no_hooks(kw.get('env',{}));guard.launching=True
  try:
   process=original(argv,*a,**kw)
   wait,poll=process.wait,process.poll
   def safe_wait(*x,**y):guard.launching=False;guard.deliver();return wait(*x,**y)
   def safe_poll(*x,**y):guard.launching=False;guard.deliver();return poll(*x,**y)
   process.wait=safe_wait;process.poll=safe_poll
   return process
  except BaseException:guard.launching=False;raise
 helper.subprocess.Popen=popen
 atomic=helper.atomic
 def save(path,value):
  atomic(path,value)
  if value.get('server_pid') and not value.get('finished_at_utc'):
   guard.launching=False;guard.deliver()
 helper.atomic=save;helper.stop_owned=guard.cleanup(helper.stop_owned)
 return launches
def verify_tool_report(p):
 d=json.loads(p.read_text());rows=d.get('results',[])
 require(d.get('completed_at_utc') and d.get('summary')=={'passed':28,'failed':0,'total':28} and len(rows)==28 and not d.get('errors'),'Default-off tools28 incomplete/failed')
 require(all(r.get('passed') is True and not r.get('errors') for r in rows),'Tool case failed')
 posts=[]
 for row in rows:
  require(isinstance(row.get('response'),dict) and isinstance(row.get('request'),dict),'Missing tool request/response')
  posts.append(row['request'])
  for key in ('followup','repeated_turn'):
   if key in row:
    require(isinstance(row[key].get('request'),dict) and isinstance(row[key].get('response'),dict),'Incomplete extra tool turn')
    posts.append(row[key]['request'])
 require(len(posts)==32 and all('literal_user_control_tokens' not in p for p in posts),'Default-off tool POST count/policy changed')
 return {'checks':28,'chat_posts':32,'passed':True,'sha256':sha(p)}
def run(args):
 helper=load(HERE/'frozen/run_matrix.py',HELPER_SHA,'feature_matrix')
 guardns=selected_ast(HERE/'frozen/strings_only_controller.py',GUARD_SHA,{'SignalGuard'}, {})
 guard=guardns['SignalGuard']()
 ownership=selected_ast(HERE/'frozen/run_literal_user_bpe.py',OWNERSHIP_SHA,{'owner_gate','verify_model'},{'Path':Path,'command':command,'require':require,'subprocess':subprocess,'hashlib':hashlib,'stamp':stamp})
 client=load(HERE/'client.py',args.client_sha256,'feature_client')
 plan,binding,_=client.validate_binding(args.plan,args.inputs,args.inputs_sha256)
 require(sha(args.model_inputs)==MODEL_SHA,'Audited model snapshot changed')
 model=json.loads(args.model_inputs.read_text())['models']['305']
 job={'label':'literal-feature-a70-concurrent','model_path':'/home/cruzspark/models/flashnext-exl3-3.05bpw','bench':[],
      'env':{'PROFILE':'concurrent','NGRAM_RAM':'false','MAX_BATCH_SIZE':'4','CACHE_SIZE':'262144','MAX_SEQ_LEN':'262144','CHUNK_SIZE':'2048','DRAFT_MODE':'mtp','DRAFT_NUM_TOKENS':'5','DYNAMIC_DRAFT':'true','EXL3_DRAFT_CONFIDENCE':'0.6','EXL3_DRAFT_ROW_BUDGET':'8','EXL3_MTP_HEAD_N':'65536','EXL3_REF':ENGINE,'EXL3_GDN_PROJ_FP32':'1','EXL3_GDN_CONV_TOKEN_MAJOR':'0','EXL3_GDN_CONV_BF16_PRODUCT':'1','EXL3_ATTN_DECODE_LEGACY_SPLITS':'1','EXL3_GEMM_LEGACY_TILES':'1','EXL3_MOE_COOP_MIXEDK':'0','EXL3_MOE_MIXEDK_NOSYNC':'1','EXL3_MOE_COOP_KSPLIT':'1','EXL3_INT8_GEMV':'0','GPU_LOCK_FILE':LOCK},
      'tools':{'case':'all','mode':'both','repeat':1,'max_tokens':512,'timeout':120}}
 job=helper.normalize(job);identity=helper.source_identity(args.recipe,args.runtime);check_sources(identity,args)
 # Source exactness binds all tracked candidate files; preparation independently binds actual renderer/native IDs.
 require(binding.get('recipe_commit')==RECIPE,'Input preparation used another recipe')
 original_env=helper.resolved_env
 def resolved(*a,**k):
  env=clean_env(original_env(*a,**k));env['PYTHONNOUSERSITE']='1';return env
 helper.resolved_env=resolved
 env=helper.resolved_env(args.recipe,args.runtime,job);no_hooks(env)
 frozen={str(p):sha(p) for p in (Path(__file__),HERE/'client.py',args.plan,args.inputs,args.model_inputs,HERE/'frozen/run_matrix.py',HERE/'frozen/strings_only_controller.py',HERE/'frozen/run_literal_user_bpe.py',HERE/'frozen/api_resilience.py')}
 require(sha(HERE/'client.py')==args.client_sha256,'Client changed')
 lock=Path(LOCK);ls=lock.lstat();require(stat.S_ISREG(ls.st_mode),'GPU lock must already be a regular nonsymlink file')
 gate=ownership['owner_gate']('rexl3-manager.service')
 ownership['verify_model'](helper,job,model)
 state={'schema_version':1,'state':'preflight','passed':False,'started_at_utc':stamp(),'candidate_sources':identity,'input_hashes':frozen,'plan_sha256':PLAN_SHA,'input_binding_sha256':args.inputs_sha256,'expected_checks':89,'expected_chat_posts':93,'outer_holds_gpu_lock':False,'gpu_lock_path':LOCK,'gpu_lock_identity':{'device':ls.st_dev,'inode':ls.st_ino},'owner_gate_before':gate,'scope':'Actual default-off/opt-in production candidate, no diagnostic hooks. One isolated server; no enablement or source/package mutations.'}
 def save():helper.atomic(args.output/'result.json',state)
 save()
 # Hide CUDA during read-only setup/import preflight. This is not a CPU suite rerun.
 probe_env=dict(env,CUDA_VISIBLE_DEVICES='')
 with (args.output/'setup-check.log').open('x') as log:
  check=subprocess.run(['bash',str(args.recipe/'exllamav3-tabby/setup.sh'),'--check'],cwd=args.recipe,env=probe_env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 state['setup_check_exit_code']=check.returncode;save();require(check.returncode==0,'Read-only setup --check failed')
 code=("import json,sys,pathlib,torch,exllamav3; import common.model; "
       "from endpoints.OAI.types.chat_completion import ChatCompletionRequest; "
       "from common import literal_user_tokens as mod; "
       "assert not torch.cuda.is_initialized(); assert 'sitecustomize' not in sys.modules; "
       "assert ChatCompletionRequest(messages=[{'role':'user','content':'plain'}]).literal_user_control_tokens is False; "
       "print(json.dumps({'engine_file':exllamav3.__file__,'literal_module':mod.__file__,'markers':sorted(mod.LITERAL_USER_CONTROL_MARKERS),'cuda_initialized':torch.cuda.is_initialized(),'sitecustomize_loaded':'sitecustomize' in sys.modules}))")
 imported=json.loads(command([str(args.runtime/'venv/bin/python'),'-c',code],cwd=args.runtime/'tabbyAPI',env=probe_env))
 require(Path(imported['engine_file']).resolve()==(args.runtime/'exllamav3/exllamav3/__init__.py').resolve(),'Engine import path differs')
 require(Path(imported['literal_module']).resolve()==(args.runtime/'tabbyAPI/common/literal_user_tokens.py').resolve(),'Candidate import path differs')
 require(imported['markers']==sorted(plan['marker_list']) and not imported['cuda_initialized'] and not imported['sitecustomize_loaded'],'Candidate capability/hooks differ')
 state['imports']=imported;state['preflight_finished_at_utc']=stamp();save()
 prior_verify=helper.verify_inputs
 def verify(j,a,i,e):
  for path,h in frozen.items():require(sha(path)==h,'Frozen input changed: '+path)
  current=lock.lstat();require(stat.S_ISREG(current.st_mode) and (current.st_dev,current.st_ino)==(ls.st_dev,ls.st_ino),'GPU lock path changed')
  prior_verify(j,a,i,e);check_sources(i,args);ownership['verify_model'](helper,j,model)
  records=list((args.output/job['label']).glob('attempt-*/result.json')) if (args.output/job['label']).exists() else []
  pid=json.loads(records[0].read_text()).get('server_pid') if records else None
  ownership['owner_gate']('rexl3-manager.service',pid)
  if pid is not None:
   helper.require_owned_listener(pid)
   actual=dict(x.split(b'=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x)
   no_hooks({k.decode():v.decode() for k,v in actual.items()})
   deployment=Path(json.loads(records[0].read_text())['attempt'])/'deployment.json'
   if deployment.exists():verify_deployment(json.loads(deployment.read_text()),helper.get_json('/model'),binding,job['model_path'])
 helper.verify_inputs=verify
 original_commands=helper.commands
 def commands(j,attempt,recipe,python,model_name):
  yield 'feature', [python,str(HERE/'client.py'),'--plan',str(args.plan),'--inputs',str(args.inputs),'--inputs-sha256',args.inputs_sha256,'--metadata',str(attempt/'deployment.json'),'--timeout','120']
  yield from original_commands(j,attempt,recipe,python,model_name)
 helper.commands=commands
 attach_guard(helper,guard,['bash',str(args.recipe/'exllamav3-tabby/serve.sh')])
 previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
 for s in previous:signal.signal(s,guard.interrupt)
 try:
  state['state']='running';save()
  result=helper.run_job(job,args,identity,env);state['lifecycle_result_sha256']=sha(args.output/job['label']/'result.json')
  attempt=Path(result['attempt']);feature=json.loads((attempt/'feature.json').read_text())
  require(feature.get('completed_at_utc') and feature.get('expected_checks')==61 and len(feature.get('results',[]))==61,'Feature client incomplete')
  tools=verify_tool_report(attempt/'tools.json')
  require(result['server_exit_before_cleanup'] is None and result['server_cleanup']['signals']==['SIGTERM'] and result['server_cleanup']['exit_code']==0,'Unexpected server exit/cleanup')
  state.update(state='completed',passed=bool(result['passed'] and feature['passed'] and tools['passed']),feature_summary=feature['summary'],tools_summary=tools,actual_checks=61+tools['checks'],actual_chat_posts=feature['summary']['actual_chat_requests']+tools['chat_posts'],feature_sha256=sha(attempt/'feature.json'),owner_gate_after=ownership['owner_gate']('rexl3-manager.service'),server_cleanup=result['server_cleanup'])
 except BaseException as exc:
  state.update(state='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',passed=False,error=type(exc).__name__+': '+str(exc))
 finally:
  for s,h in previous.items():signal.signal(s,h)
  state['finished_at_utc']=stamp();save()
 return 0 if state['passed'] else 1
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('recipe','runtime','plan','inputs','model-inputs','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('inputs-sha256','client-sha256','packages-sha256'):p.add_argument('--'+name,required=True)
 p.add_argument('--ready-timeout',type=float,default=300);p.add_argument('--client-timeout',type=float,default=2400)
 a=p.parse_args();a.resume=False;a.sample_interval=5.;a.max_samples=600
 for name in ('recipe','runtime','plan','inputs','model_inputs'):setattr(a,name,getattr(a,name).expanduser().resolve(strict=True))
 require(a.output.is_absolute() and not a.output.exists(),'Use a fresh absolute output path')
 require(0<a.ready_timeout<=600 and 900<=a.client_timeout<=3600,'Timeouts outside bounded range')
 require(all(re.fullmatch('[0-9a-f]{64}',getattr(a,n)) for n in ('inputs_sha256','client_sha256','packages_sha256')),'Expected SHA256 values')
 a.output.mkdir(parents=True)
 # This only serializes port8899 experiment controllers; the server alone holds the shared GPU lock.
 with Path(f'/tmp/qwen-experiment-8899-{os.getuid()}.lock').open('a') as port:
  fcntl.flock(port,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return run(a)
if __name__=='__main__':raise SystemExit(main())
