#!/usr/bin/env python3
"""Separate unchanged default-tool28 EOS regression. Parent schedules GPU; this owns only its children."""
from __future__ import annotations
import argparse,ast,datetime,fcntl,hashlib,json,os,re,signal,stat,subprocess,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
TABBY='fd8cbeb1fbb3cec4d2141558d5cfd63a500d50c4'
RECIPE='254b2b03027f25094845dc31f5f87739f3584d2e'
HELPER_SHA='7e83208d473a124c6a90f8f07de0f338dea7d0510c6cd03f83d63f8cb31e103a'
GUARD_SHA='cfc3619894c4ee8f07d46c2ee7ed0fd085f3fdc044c81c8d9a01d009304c4116'
OWNERSHIP_SHA='2a0a0e3c36f482581847dc9c1e84ebc03dc72014006ed45c8b23cd348d0bed2e'
MODEL_SHA='6dc2117bb602a3f57e798613c005c0c1ec8ad519378168181b00465e40b90840'
LOCK='/home/cruzspark/redsnow-gpu.lock'
SYSTEM_SITE_PATH='/usr/lib/python3.12/sitecustomize.py'
SYSTEM_SITE_RESOLVED='/etc/python3.12/sitecustomize.py'
SYSTEM_SITE_SHA='43d81125d92376b1a69d53a71126a041cc9a18d8080e92dea0a2ae23be138b1e'
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
def validate_customization(imported):
 require(imported.get('usercustomize_loaded') is False,'Unexpected usercustomize module')
 site=imported.get('sitecustomize')
 require(site is None or site=={'file':SYSTEM_SITE_PATH,'resolved':SYSTEM_SITE_RESOLVED,'sha256':SYSTEM_SITE_SHA},'Unadmitted sitecustomize module')
def verify_system_site():
 require(str(Path(SYSTEM_SITE_PATH).resolve(strict=True))==SYSTEM_SITE_RESOLVED and sha(SYSTEM_SITE_PATH)==SYSTEM_SITE_SHA,'Admitted Ubuntu customization changed')
def run_recorded(args):
 """Preserve a terminal record when read-only preflight fails before run_job exists."""
 try:return run(args)
 except BaseException as exc:
  path=args.output/'result.json'
  state=json.loads(path.read_text()) if path.exists() else {'schema_version':1,'started_at_utc':stamp(),'state':'preflight'}
  state.update(state='preflight_failed' if state.get('state')=='preflight' else 'failed',passed=False,error=type(exc).__name__+': '+str(exc),finished_at_utc=stamp())
  if state['state']=='preflight_failed':state.update(server_launched=False,actual_chat_posts=0)
  temporary=path.with_suffix('.failed.tmp');temporary.write_text(json.dumps(state,indent=2,sort_keys=True)+'\n');os.replace(temporary,path)
  return 1
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

def attach_guard(helper,guard,server_command,observer_env=None):
 """Deferral until the helper assigns each child; no inherited signal masks."""
 original=helper.subprocess.Popen;helper.subprocess=types.SimpleNamespace(**vars(helper.subprocess))
 launches=[];registry=[]
 # Record asynchronously; deliver at poll/wait/save checkpoints, never in a cleanup-entry gap.
 def record_signal(signum,frame):
  if guard.pending is None:guard.pending=signum
 guard.interrupt=record_signal
 def popen(argv,*a,**kw):
  if argv==server_command:
   require(not launches,'Only one server launch allowed');launches.append(argv)
  no_hooks(kw.get('env',{}))
  if argv==server_command and observer_env is not None:kw['env']={**kw.get('env',{}),**observer_env}
  guard.launching=True
  try:
   token=kw.get('env',{}).get('QWEN_EXPERIMENT_OWNER')
   require(bool(token),'Owned child lacks ownership token')
   process=original(argv,*a,**kw)
   registry.append((process,token))
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
 return registry
def finalize_owned(helper,guard,registry):
 outcomes=[];guard.cleaning+=1
 try:
  for process,token in reversed(registry):
   try:
    before=process.poll()
    result=helper.stop_owned(process,token)
    require(not helper.group_members(process,token),'Owned group remains after outer cleanup')
    outcomes.append({'pid':process.pid,'exit_before_outer_cleanup':before,**result,'owned_group_empty':True})
   except Exception as exc:
    outcomes.append({'pid':process.pid,'owned_group_empty':False,'cleanup_error':type(exc).__name__+': '+str(exc)})
 finally:guard.cleaning-=1
 return outcomes
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
 require(sha(args.contract)==args.contract_sha256,'Runtime contract changed')
 binding=json.loads(args.contract.read_text())
 require(sha(args.model_inputs)==MODEL_SHA,'Audited model snapshot changed')
 model=json.loads(args.model_inputs.read_text())['models']['305']
 job={'label':'eos-default-tool28','model_path':'/home/cruzspark/models/flashnext-exl3-3.05bpw','bench':[],
      'env':{'PROFILE':'concurrent','NGRAM_RAM':'false','MAX_BATCH_SIZE':'4','CACHE_SIZE':'262144','MAX_SEQ_LEN':'262144','CHUNK_SIZE':'2048','DRAFT_MODE':'mtp','DRAFT_NUM_TOKENS':'5','DYNAMIC_DRAFT':'true','EXL3_DRAFT_CONFIDENCE':'0.6','EXL3_DRAFT_ROW_BUDGET':'8','EXL3_MTP_HEAD_N':'65536','EXL3_REF':ENGINE,'EXL3_GDN_PROJ_FP32':'1','EXL3_GDN_CONV_TOKEN_MAJOR':'0','EXL3_GDN_CONV_BF16_PRODUCT':'1','EXL3_ATTN_DECODE_LEGACY_SPLITS':'1','EXL3_GEMM_LEGACY_TILES':'1','EXL3_MOE_COOP_MIXEDK':'0','EXL3_MOE_MIXEDK_NOSYNC':'1','EXL3_MOE_COOP_KSPLIT':'1','EXL3_INT8_GEMV':'0','GPU_LOCK_FILE':LOCK},
      'tools':{}}
 job=helper.normalize(job);identity=helper.source_identity(args.recipe,args.runtime);check_sources(identity,args)
 # Exact candidate source and prior recipe/model/runtime contract bind this default-tool regression.
 require(binding.get('recipe_commit')==RECIPE,'Input preparation used another recipe')
 original_env=helper.resolved_env
 def resolved(*a,**k):
  env=clean_env(original_env(*a,**k));env['PYTHONNOUSERSITE']='1';return env
 helper.resolved_env=resolved
 env=helper.resolved_env(args.recipe,args.runtime,job);no_hooks(env)
 frozen={str(p):sha(p) for p in (Path(__file__),args.contract,args.model_inputs,HERE/'frozen/run_matrix.py',HERE/'frozen/strings_only_controller.py',HERE/'frozen/run_literal_user_bpe.py',HERE/'frozen/api_resilience.py')}
 lock=Path(LOCK);ls=lock.lstat();require(stat.S_ISREG(ls.st_mode),'GPU lock must already be a regular nonsymlink file')
 gate=ownership['owner_gate']('rexl3-manager.service')
 ownership['verify_model'](helper,job,model)
 state={'schema_version':1,'state':'preflight','passed':False,'started_at_utc':stamp(),'candidate_sources':identity,'input_hashes':frozen,'runtime_contract_sha256':args.contract_sha256,'expected_checks':28,'expected_chat_posts':32,'outer_holds_gpu_lock':False,'gpu_lock_path':LOCK,'gpu_lock_identity':{'device':ls.st_dev,'inode':ls.st_ino},'owner_gate_before':gate,'scope':'Unchanged default tool28 (32 POSTs) against EOS-only candidate. No observer, input feature, semantic retry, enablement or source/package mutation.'}
 def save():helper.atomic(args.output/'result.json',state)
 save()
 # Hide CUDA during read-only setup/import preflight. This is not a CPU suite rerun.
 probe_env=dict(env,CUDA_VISIBLE_DEVICES='')
 with (args.output/'setup-check.log').open('x') as log:
  check=subprocess.run(['bash',str(args.recipe/'exllamav3-tabby/setup.sh'),'--check'],cwd=args.recipe,env=probe_env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
 state['setup_check_exit_code']=check.returncode;save();require(check.returncode==0,'Read-only setup --check failed')
 code=("import json,sys,pathlib,hashlib,torch,exllamav3; import common.model; "
       "from endpoints.OAI.types.chat_completion import ChatCompletionRequest; "
       "assert not torch.cuda.is_initialized(), 'CUDA initialized in CPU preflight'; "
       "assert 'literal_user_control_tokens' not in ChatCompletionRequest.model_fields, 'Unexpected experimental input feature'; "
       "site=sys.modules.get('sitecustomize'); path=pathlib.Path(site.__file__) if site is not None else None; "
       "print(json.dumps({'engine_file':exllamav3.__file__,'server_file':common.model.__file__,'cuda_initialized':torch.cuda.is_initialized(),'usercustomize_loaded':'usercustomize' in sys.modules,'sitecustomize':None if path is None else {'file':str(path),'resolved':str(path.resolve(strict=True)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}))")

 imported=json.loads(command([str(args.runtime/'venv/bin/python'),'-c',code],cwd=args.runtime/'tabbyAPI',env=probe_env))
 require(Path(imported['engine_file']).resolve()==(args.runtime/'exllamav3/exllamav3/__init__.py').resolve(),'Engine import path differs')
 require(Path(imported['server_file']).resolve()==(args.runtime/'tabbyAPI/common/model.py').resolve(),'Candidate import path differs')
 require(not imported['cuda_initialized'],'CUDA initialized in CPU preflight')
 validate_customization(imported);verify_system_site()
 frozen[SYSTEM_SITE_PATH]=SYSTEM_SITE_SHA
 state['imports']=imported;state['preflight_finished_at_utc']=stamp();save()
 prior_verify=helper.verify_inputs
 def verify(j,a,i,e):
  verify_system_site()
  for path,h in frozen.items():require(sha(path)==h,'Frozen input changed: '+path)
  current=lock.lstat();require(stat.S_ISREG(current.st_mode) and (current.st_dev,current.st_ino)==(ls.st_dev,ls.st_ino),'GPU lock path changed')
  prior_verify(j,a,i,e);check_sources(i,args);ownership['verify_model'](helper,j,model)
  records=list((args.output/job['label']).glob('attempt-*/result.json')) if (args.output/job['label']).exists() else []
  pid=json.loads(records[0].read_text()).get('server_pid') if records else None
  ownership['owner_gate']('rexl3-manager.service',pid)
  if pid is not None:
   helper.require_owned_listener(pid)
   actual=dict(x.split(b'=',1) for x in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x)
   observed={k.decode():v.decode() for k,v in actual.items()}
   no_hooks(observed)
   deployment=Path(json.loads(records[0].read_text())['attempt'])/'deployment.json'
   if deployment.exists():verify_deployment(json.loads(deployment.read_text()),helper.get_json('/model'),binding,job['model_path'])
 helper.verify_inputs=verify
 registry=attach_guard(helper,guard,['bash',str(args.recipe/'exllamav3-tabby/serve.sh')])
 previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
 for s in previous:signal.signal(s,guard.interrupt)
 try:
  state['state']='running';save()
  result=helper.run_job(job,args,identity,env);state['lifecycle_result_sha256']=sha(args.output/job['label']/'result.json')
  require(result['state']=='completed' and result.get('inputs_verified_after_measurements_at_utc') and not result.get('error'),'Owned lifecycle/source verification incomplete')
  require([r['name'] for r in result['clients']]==['tools'] and all(r.get('completed_report') and not r.get('timed_out') for r in result['clients']),'Expected one completed unchanged tool client')
  attempt=Path(result['attempt']);report=json.loads((attempt/'tools.json').read_text())
  require(result['server_exit_before_cleanup'] is None and result['server_cleanup']['signals']==['SIGTERM'] and result['server_cleanup']['exit_code']==0,'Unexpected server exit/cleanup')
  recorded=report.get('results',[])
  posts=sum(isinstance(r.get('request'),dict)+sum(isinstance(r.get(k,{}).get('request'),dict) for k in ('followup','repeated_turn')) for r in recorded)
  state.update(tool_summary=report.get('summary'),tool_sha256=sha(attempt/'tools.json'),client_exit_code=result['clients'][0]['exit_code'],actual_checks=len(recorded),actual_chat_posts=posts);save()
  proof=verify_tool_report(attempt/'tools.json')
  require(result['clients'][0]['exit_code']==0,'Unexpected client exit status')
  state.update(state='completed',passed=True,tool_proof=proof,actual_checks=28,actual_chat_posts=32,owner_gate_after=ownership['owner_gate']('rexl3-manager.service'),server_cleanup=result['server_cleanup'])

 except BaseException as exc:
  state.update(state='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',passed=False,error=type(exc).__name__+': '+str(exc))
 finally:
  try:
   state['outer_owned_cleanup']=finalize_owned(helper,guard,registry)
   require(all(r['owned_group_empty'] for r in state['outer_owned_cleanup']),'Outer owned cleanup incomplete')
  except BaseException as exc:
   state.update(state='cleanup_failed',passed=False,outer_cleanup_error=type(exc).__name__+': '+str(exc))
  finally:
   if guard.pending is not None:state.update(state='interrupted',passed=False,signal=guard.pending)
   for s,h in previous.items():signal.signal(s,h)
   state['finished_at_utc']=stamp();save()
 return 0 if state['passed'] else 1
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('recipe','runtime','contract','model-inputs','output'):p.add_argument('--'+name,type=Path,required=True)
 for name in ('contract-sha256','packages-sha256'):p.add_argument('--'+name,required=True)
 p.add_argument('--ready-timeout',type=float,default=300);p.add_argument('--client-timeout',type=float,default=2400)
 a=p.parse_args();a.resume=False;a.sample_interval=5.;a.max_samples=600
 for name in ('recipe','runtime','contract','model_inputs'):setattr(a,name,getattr(a,name).expanduser().resolve(strict=True))
 require(a.output.is_absolute() and not a.output.exists(),'Use a fresh absolute output path')
 require(0<a.ready_timeout<=600 and 900<=a.client_timeout<=3600,'Timeouts outside bounded range')
 require(all(re.fullmatch('[0-9a-f]{64}',getattr(a,n)) for n in ('contract_sha256','packages_sha256')),'Expected SHA256 values')
 a.output.mkdir(parents=True)
 # This only serializes port8899 experiment controllers; the server alone holds the shared GPU lock.
 with Path(f'/tmp/qwen-experiment-8899-{os.getuid()}.lock').open('a') as port:
  fcntl.flock(port,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return run_recorded(a)
if __name__=='__main__':raise SystemExit(main())
