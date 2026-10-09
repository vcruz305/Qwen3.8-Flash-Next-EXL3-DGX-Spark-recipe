#!/usr/bin/env python3
"""Scheduled, owned three-cell flat4.05 attribution. No installation or ref changes.

The owner must stop REXL3 first. Hold its existing cooperative flock throughout.
All cells run the unchanged public launcher/client. B uses the final venv and
engine with a separately staged old Tabby checkout. Never edit either venv.
"""
from __future__ import annotations
import argparse, contextlib, fcntl, hashlib, json, os, signal, subprocess, time, types, uuid
from pathlib import Path

RECIPE="a8c72bdf811646f413fdc03a4e49811a2753d0cf"
OLD_ENGINE="94ba01d50a13fa9ff672473f2d0eef8b51a71e99"
NEW_ENGINE="24f0dece34f09c8d1e2359d6b3b3f7befef7331b"
OLD_TABBY="816c32195887aaecea1c64528f2921566766259b"
NEW_TABBY="f650bb5389e0a273549e47d4d26a765760c013e1"
HELPER="48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586"
PARENT="78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589"
ALIAS="Qwen3.8-Flash-Next-EXL3"
MODEL="/home/cruzspark/models/flashnext-exl3-4.05bpw"
BASE="http://127.0.0.1:8899/v1"
LOCK=Path("/home/cruzspark/redsnow-gpu.lock")
TUNING={
 "PROFILE":"single","NGRAM_RAM":"false","MAX_SEQ_LEN":"262144","CACHE_SIZE":"262144",
 "MAX_BATCH_SIZE":"1","CHUNK_SIZE":"2048","DRAFT_MODE":"mtp","DRAFT_NUM_TOKENS":"5",
 "DYNAMIC_DRAFT":"true","SYSMEM_RECURRENT_CACHE":"4096","VISION":"false","REASONING":"true",
 "TOOL_FORMAT":"qwen3_5","SERVED_NAME":ALIAS,"BIGCORES":"5-9,15-19",
 "PROMPT_TEMPLATE":"","CUDA_HOME":"/usr/local/cuda-13.0","TORCH_CUDA_ARCH_LIST":"12.1",
 "EXL3_INT8_GEMV":"0","EXL3_GR_INT8":"1","EXL3_MOE_COOP_WIDE":"1",
 "EXL3_MTP_HEAD_N":"65536","EXL3_DRAFT_CONFIDENCE":"0.6","EXL3_DRAFT_ROW_BUDGET":"0",
 "EXL3_GDN_PROJ_FP32":"1","EXL3_GDN_CONV_TOKEN_MAJOR":"0","EXL3_GDN_CONV_BF16_PRODUCT":"1",
 "EXL3_ATTN_DECODE_LEGACY_SPLITS":"1","EXL3_MOE_COOP_MIXEDK":"0",
 "EXL3_MOE_MIXEDK_NOSYNC":"1","EXL3_GEMM_LEGACY_TILES":"1","EXL3_MOE_COOP_KSPLIT":"1",
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok: raise ValueError(message)
def load(p,digest,name):
 raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==digest,"Frozen helper changed: "+str(p))
 m=types.ModuleType(name);m.__file__=str(p);exec(compile(raw,str(p),"exec"),m.__dict__);return m
def modules():
 b=Path(__file__).resolve().parent/"frozen"
 return load(b/"spark_experiment_controller.py",HELPER,"flat_helper"),load(b/"api_f4_gemm_controller.py",PARENT,"flat_owner")
def cells(args):
 return [
  {"label":"A-old-engine-old-server","runtime":args.old_runtime,"tabby":args.old_runtime/"tabbyAPI",
   "engine_sha":OLD_ENGINE,"tabby_sha":OLD_TABBY,"minimum":"1.5.1"},
  {"label":"B-new-engine-old-server","runtime":args.new_runtime,"tabby":args.old_tabby,
   "engine_sha":NEW_ENGINE,"tabby_sha":OLD_TABBY,"minimum":"1.6.0.post1"},
  {"label":"C-new-engine-new-server","runtime":args.new_runtime,"tabby":args.new_runtime/"tabbyAPI",
   "engine_sha":NEW_ENGINE,"tabby_sha":NEW_TABBY,"minimum":"1.6.0.post1"},
 ]
def owner_gate(helper,allowed_pid=None):
 unit={}
 for name in ("rexl3-manager.service","qwen38-exl3.service"):
  text=helper.capture(["systemctl","--user","show",name,"-p","LoadState","-p","ActiveState","-p","MainPID"])
  v=dict(x.split("=",1) for x in text.splitlines())
  require(v.get("LoadState")=="loaded" and v.get("ActiveState")=="inactive" and v.get("MainPID")=="0",
          "Supervisor must remain inactive: "+name);unit[name]=v
 enabled=subprocess.run(["systemctl","--user","is-enabled","qwen38-exl3.service"],capture_output=True,text=True,timeout=10)
 require(enabled.stdout.strip()=="disabled","Preserve disabled Qwen unit")
 gpu=helper.capture(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"])
 pids=[]
 for line in gpu.splitlines():
  if not line.strip():continue
  first=line.split(",",1)[0].strip();require(first.isdigit(),"Unknown GPU ownership result");pids.append(int(first))
 require(set(pids)<=({allowed_pid} if allowed_pid else set()),"Another CUDA owner exists")
 return {"at_utc":helper.stamp(),"units":unit,"qwen_enabled":"disabled","gpu":gpu}
def source_identity(helper,args,cell):
 ident=helper.source_identity(args.recipe,cell["runtime"])
 ident["server"]=helper.git_identity(cell["tabby"])
 for name,pin in (("recipe",RECIPE),("engine",cell["engine_sha"]),("server",cell["tabby_sha"])):
  require(ident[name]["commit"]==pin and not ident[name]["tracked_changes"],"Exact clean source required: "+name)
 return ident
def environment(helper,args,cell,output,token):
 tuning=dict(TUNING,EXL3_REF=cell["engine_sha"],EXL3_MIN_VERSION=cell["minimum"])
 env=helper.resolved_env(args.recipe,cell["runtime"],{"model_path":MODEL,"env":tuning})
 env.update(TABBY_DIR=str(cell["tabby"]),TABBY_REF="main",STATE_DIR=str(output/"state"),
            QWEN_EXPERIMENT_OWNER=token,PYTHONPATH=str(cell["runtime"]/"exllamav3"),PYTHONUNBUFFERED="1")
 for key in ("API_KEY","TABBY_API_KEY","OPENAI_API_KEY"):env.pop(key,None)
 for key,value in tuning.items():require(env.get(key)==value,"Resolved tuning mismatch: "+key)
 return env
def check_model(helper,record):
 require(record["model_path"]==MODEL and helper.model_identity(Path(MODEL))==record["identity"],
         "Model configuration, loader order or file identity changed")
 # Read headers only; never weight payloads or full-weight hashes.
 import struct
 for row in record["headers_in_loader_order"]:
  p=Path(MODEL)/row["name"];a=p.stat()
  with p.open("rb") as f:
   n=struct.unpack("<Q",f.read(8))[0];require(n==row["header_bytes"],"Header length changed")
   raw=f.read(n)
  b=p.stat();require((a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino),"Weight changed during header read")
  require(hashlib.sha256(raw).hexdigest()==row["header_sha256"],"Weight header changed")
def import_probe(args,cell,env):
 # Import the same backend from the exact launch working tree. Its ordinary
 # version check is invoked explicitly; no constructor/model/GPU allocation.
 code="""import json,inspect,importlib.metadata as md,pathlib,torch,exllamav3
from exllamav3.version import __version__
from backends.exllamav3 import model
from common.optional_dependencies import check_package_version
check_package_version("exllamav3", "__MIN__")
out={"engine_file":str(pathlib.Path(exllamav3.__file__).resolve()),
"backend_file":str(pathlib.Path(model.__file__).resolve()),"version":__version__,
"generator_parameters":list(inspect.signature(exllamav3.Generator.__init__).parameters),
"torch_cuda_initialized":torch.cuda.is_initialized(),
"native_extension_file":str(pathlib.Path(exllamav3.ext.exllamav3_ext.__file__).resolve()),
"packages":dict(sorted((d.metadata["Name"],d.version) for d in md.distributions()))}
print("PREFLIGHT_JSON="+json.dumps(out,sort_keys=True))
""".replace("__MIN__","1.5.1" if cell["tabby_sha"]==OLD_TABBY else "1.5.4")
 r=subprocess.run([str(cell["runtime"]/"venv/bin/python"),"-c",code],cwd=cell["tabby"],env=env,
                  capture_output=True,text=True,timeout=120)
 records=[x.removeprefix("PREFLIGHT_JSON=") for x in r.stdout.splitlines() if x.startswith("PREFLIGHT_JSON=")]
 require(r.returncode==0 and len(records)==1,"Backend import preflight failed: "+r.stderr[-3000:])
 value=json.loads(records[0])
 require(Path(value["engine_file"])==(cell["runtime"]/"exllamav3/exllamav3/__init__.py").resolve(),"Wrong imported engine")
 require(Path(value["backend_file"])==(cell["tabby"]/"backends/exllamav3/model.py").resolve(),"Wrong imported server")
 require(value["version"]==cell["minimum"],"Unexpected real engine version")
 require(value["torch_cuda_initialized"] is False,"CPU import unexpectedly initialized CUDA")
 for name in ("draft_confidence","draft_model","num_draft_tokens","dynamic_draft_tokens"):
  require(name in value["generator_parameters"],"Missing original backend generator capability: "+name)
 return value,{"stdout":r.stdout,"stderr":r.stderr,"exit_code":r.returncode}
def preflight(helper,args,cell,output,token):
 output.mkdir()
 env=environment(helper,args,cell,output,token);env["CUDA_VISIBLE_DEVICES"]="";ident=source_identity(helper,args,cell)
 # This runs only runtime verification, not setup/install. The exact known
 # baseline wheel-tag defect is independently checked; all other failures stop.
 verify=subprocess.run(["bash","-c",'source "$1"; verify_runtime',"verify",
          str(args.recipe/"exllamav3-tabby/env.sh")],cwd=args.recipe,env=env,
          capture_output=True,text=True,timeout=180)
 (output/"runtime-check.log").write_text(verify.stdout+verify.stderr)
 require(verify.returncode==0,"Public runtime verification failed")
 imports,log=import_probe(args,cell,env);helper.atomic(output/"import-log.json",log)
 pip=subprocess.run([str(cell["runtime"]/"venv/bin/python"),"-m","pip","check"],
                     env=env,capture_output=True,text=True,timeout=90)
 helper.atomic(output/"pip-check.json",{"exit_code":pip.returncode,"stdout":pip.stdout,"stderr":pip.stderr})
 classification=None
 if pip.returncode:
  proof=json.loads((Path(__file__).resolve().parent/"old-vendor-current-proof.json").read_text())
  require(cell["engine_sha"]==OLD_ENGINE and cell["tabby_sha"]==OLD_TABBY
    and pip.returncode==1 and pip.stdout=="nvidia-cusparselt-cu13 0.8.1 is not supported on this platform\n"
    and pip.stderr=="","Unclassified dependency failure: "+pip.stdout+pip.stderr)
  verifier=Path(__file__).resolve().parent/"verify_historical_vendor.py"
  fresh=subprocess.run([str(cell["runtime"]/"venv/bin/python"),str(verifier)],env=env,
    capture_output=True,text=True,timeout=60)
  require(fresh.returncode==0,"Read-only baseline vendor proof failed")
  current=json.loads(fresh.stdout)
  require(proof["matches_independent_historical_proof"] and current==proof["current"],
    "Baseline package/library differs from independent historical proof")
  classification={"kind":"historical-baseline-only-known-vendor-wheel-tag",
    "current":current,"prior_proof_sha256":sha(Path(__file__).resolve().parent/"old-vendor-current-proof.json"),
    "note":"pip check remains exit1. This is not a passed full setup--check or deployment qualification. No metadata repair."}
 record={"at_utc":helper.stamp(),"sources":ident,"imports":imports,"runtime_check_exit_code":verify.returncode,
  "pip_check":{"exit_code":pip.returncode,"stdout":pip.stdout,"stderr":pip.stderr},
  "pip_check_passed":pip.returncode==0,"historical_classification":classification,
  "diagnostic_preflight_passed":True,"note":"No setup/install/ref/environment mutation. Imports only; model not instantiated."}
 helper.atomic(output/"preflight.json",record)
 require(source_identity(helper,args,cell)==ident,"Source/packages changed during preflight")
 return record
def validate_report(report,case,expected):
 require(report.get("completed_at_utc") and not report.get("errors"),"Incomplete/error benchmark")
 require(report["settings"]==expected["settings"] and report["run_id"]==expected["run_id"],"Benchmark settings changed")
 require(set(report["cases"])=={case},"Unexpected benchmark case")
 data=report["cases"][case];require(len(data["warmup"])==1 and len(data["runs"])==3,"Wrong sample counts")
 for kind in ("warmup","runs"):
  old=expected["cases"][case][kind];rows=data[kind]
  require(len(old)==len(rows),"Sample count mismatch")
  for a,b in zip(old,rows):
   for key in ("request_sha256","prompt_tokens","cached_prompt_tokens","completion_tokens","finish_reason"):
    require(a[key]==b[key],"Original fixture/actual usage changed: "+key)
   require(b["finish_reason"]=="length" and b["completion_tokens"]==400,"Not a complete400-token run")
   require(b["server_decode_tok_s"]>0,"Missing server timing")
   details=b["usage"]["completion_tokens_details"]
   require(all(type(details[k]) is int and details[k]>=0 for k in ("accepted_prediction_tokens","rejected_prediction_tokens")),
           "Missing draft work counters")
 return {"passed":True,"measured":3,"warmup":1,"actual_completion_tokens":1200}
def validate_deployment(d,cell,output,identity,env):
 for key in ("engine","server"):
  require(d[key]["commit"]==identity[key]["commit"] and not d[key]["tracked_changes"],"Deployment source mismatch")
 require(Path(d["model"]["resolved_path"]).resolve()==Path(MODEL).resolve(),"Wrong deployed pack")
 cfg=d["config"]["values"]
 wanted={"network":{"host":"127.0.0.1","port":8899,"disable_auth":True},
  "model":{"model_name":ALIAS,"model_dir":str(output/"state/models"),"cache_size":262144,"max_seq_len":262144,
           "max_batch_size":1,"chunk_size":2048,"cache_mode":"8,8","ngram_ram":False,
           "reasoning":True,"tool_format":"qwen3_5","vision":False},
  "draft_model":{"draft_mode":"mtp","draft_num_tokens":5,"dynamic_draft":True,"draft_cache_mode":"8,8"}}
 for section,values in wanted.items():
  for key,value in values.items():require(cfg.get(section,{}).get(key)==value,"Deployed setting mismatch: "+section+"."+key)
 for key in TUNING:
  if key.startswith("EXL3_") or key in ("PROFILE","NGRAM_RAM","CHUNK_SIZE","CACHE_SIZE","MAX_SEQ_LEN","MAX_BATCH_SIZE","DRAFT_MODE","DRAFT_NUM_TOKENS"):
   require(d["environment"].get(key)==env[key],"Deployed tuning mismatch: "+key)
def benchmark_command(args,cell,case,output):
 response=Path(MODEL).name if cell["tabby_sha"]==OLD_TABBY else ALIAS
 return [str(cell["runtime"]/"venv/bin/python"),str(args.recipe/"bench/bench_v1.py"),
  "--base-url",BASE,"--model",ALIAS,"--response-model",response,"--suite",case,"--max-tokens","400",
  "--warmup","1","--repeat","3","--cache-mode","cold","--context-tokens","0","--run-id","overnight-v1",
  "--timeout","180","--label",cell["label"],"--metadata",str(output/"deployment.json"),
  "--output",str(output/(case+".json"))]
@contextlib.contextmanager
def spawn_guard():
 # Defer Python interruption until the caller has registered the child. Do
 # not block OS signals: a Popen child would inherit that mask across exec.
 watched=(signal.SIGTERM,signal.SIGINT)
 before={sig:signal.getsignal(sig) for sig in watched};pending=[]
 def defer(signum,frame):
  if not pending:pending.append((signum,frame))
 for sig in watched:signal.signal(sig,defer)
 try:yield
 finally:
  for sig,handler in before.items():signal.signal(sig,handler)
  if pending:
   signum,frame=pending[0];handler=before[signum]
   if callable(handler):handler(signum,frame)
   elif handler!=signal.SIG_IGN:raise KeyboardInterrupt("deferred signal "+str(signum))
def run_cell(helper,parent,args,cell,output,pre,recorded,expected,verify_files):
 output.mkdir()
 token=uuid.uuid4().hex;env=environment(helper,args,cell,output,token)
 state={"label":cell["label"],"state":"preflight","passed":False,"started_at_utc":helper.stamp(),
        "ownership_token":token,"sources":pre["sources"],"clients":[]}
 server=client=serverlog=None
 sampler=helper.Sampler(output/"resources.json",2,900)
 def save():helper.atomic(output/"result.json",state)
 def verify():
  verify_files()
  require(source_identity(helper,args,cell)==pre["sources"],"Source/package drift during cell")
  require(helper.model_identity(Path(MODEL))==recorded["identity"],"Pack/config/order drift")
  require(environment(helper,args,cell,output,token)==env,"Environment/default drift")
  owner_gate(helper,server.pid if server else None)
  if "deployment_sha256" in state:
   require(sha(output/"deployment.json")==state["deployment_sha256"],"Deployment sidecar drift")
   require(sha(output/"state/config.yml")==deployment["config"]["sha256"],"Loaded config drift")
 save()
 try:
  verify();helper.require_free_port();sampler.tick("before_start")
  serverlog=(output/"server.log").open("x")
  cmd=["bash",str(args.recipe/"exllamav3-tabby/serve.sh")]
  # Register ownership before deferred Python interruption is re-raised.
  with spawn_guard():
   server=subprocess.Popen(cmd,cwd=args.recipe,env=env,stdout=serverlog,stderr=subprocess.STDOUT,start_new_session=True)
   state.update(server_pid=server.pid,server_command=cmd,state="loading");save()
  start=time.monotonic()
  while time.monotonic()-start<args.ready_timeout:
   require(server.poll() is None,"Server exited during load")
   sampler.tick("loading",server.pid)
   try:
    models=parent.api_json("/models")
    if models.get("data"):break
   except (OSError,ValueError):pass
   time.sleep(.5)
  else:raise TimeoutError("Readiness deadline")
  parent.require_listener(helper,server);current=parent.api_json("/model")
  require([x.get("id") for x in models["data"]]==[ALIAS],"Wrong advertised model")
  require(current.get("id") in (ALIAS,Path(MODEL).name),"Wrong loaded model")
  require(os.sched_getaffinity(server.pid)==set(range(5,10))|set(range(15,20)),"Server CPU affinity differs")
  deployment=json.loads((output/"state/deployment.json").read_text())
  validate_deployment(deployment,cell,output,pre["sources"],env)
  helper.atomic(output/"deployment.json",deployment)
  state.update(deployment_sha256=sha(output/"deployment.json"),models=models,current_model=current,
               load_wall_seconds=time.monotonic()-start,state="bench")
  verify();save()
  for case in ("code","devops"):
   verify();parent.require_listener(helper,server)
   cmd=benchmark_command(args,cell,case,output)
   row={"case":case,"command":cmd,"started_at_utc":helper.stamp()};state["clients"].append(row);save()
   with (output/(case+".log")).open("x") as log:
    with spawn_guard():
     client=subprocess.Popen(cmd,cwd=args.recipe,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     row["pid"]=client.pid;save()
    deadline=time.monotonic()+args.client_timeout
    while client.poll() is None and time.monotonic()<deadline:
     require(server.poll() is None,"Owned server exited during benchmark")
     sampler.tick(case,server.pid);time.sleep(.1)
    row["timed_out"]=client.poll() is None
    row["cleanup"]=parent.stop_owned(helper,client,token);row["exit_code"]=row["cleanup"]["exit_code"];client=None
   row["finished_at_utc"]=helper.stamp()
   require(not row["timed_out"] and row["exit_code"]==0,"Benchmark process failed")
   p=output/(case+".json");row["report_sha256"]=sha(p)
   row.update(validate_report(json.loads(p.read_text()),case,expected));save()
  verify();parent.require_listener(helper,server)
  state.update(state="completed",passed=True)
 except BaseException as exc:
  state.update(state="failed",passed=False,error=type(exc).__name__+": "+str(exc))
 finally:
  prior={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
  for s in prior:signal.signal(s,signal.SIG_IGN)
  for label,proc in (("client",client),("server",server)):
   if proc is None:continue
   try:
    cleaned=parent.cleanup_server(helper,proc,token) if label=="server" else parent.stop_owned(helper,proc,token)
    state[label+"_cleanup"]=cleaned
    if cleaned.get("unexpected_exit"):state.update(state="failed",passed=False)
   except BaseException as exc:
    state.update(state="failed",passed=False,cleanup_failed=True)
    state[label+"_cleanup_error"]=type(exc).__name__+": "+str(exc)
  if serverlog:serverlog.close()
  state["finished_at_utc"]=helper.stamp();save()
  for s,h in prior.items():signal.signal(s,h)
 return state
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 for name in ("recipe","old-runtime","new-runtime","old-tabby","model-inputs","output"):
  p.add_argument("--"+name,type=Path,required=True)
 p.add_argument("--ready-timeout",type=float,default=600);p.add_argument("--client-timeout",type=float,default=300)
 p.add_argument("--preflight-only",action="store_true")
 args=p.parse_args(argv)
 for key in ("recipe","old_runtime","new_runtime","old_tabby","model_inputs"):
  setattr(args,key,getattr(args,key).resolve(strict=True))
 require(args.old_runtime!=args.new_runtime,"Distinct preserved runtimes required")
 require(args.old_tabby not in (args.old_runtime/"tabbyAPI",args.new_runtime/"tabbyAPI"),"B needs its own isolated old source")
 require(0<args.ready_timeout<=1200 and 0<args.client_timeout<=600,"Invalid deadlines")
 args.output=args.output.resolve();require(not args.output.exists(),"Refusing existing output")
 os.umask(0o077);args.output.mkdir(parents=True)
 helper,parent=modules();parent.BASE=BASE
 root=Path(__file__).resolve().parent
 watched=[Path(__file__).resolve(),args.model_inputs,root/"original-fixture.json",
          root/"frozen/spark_experiment_controller.py",root/"frozen/api_f4_gemm_controller.py",
          root/"verify_historical_vendor.py",root/"old-vendor-current-proof.json"]
 initial={str(x):sha(x) for x in watched}
 def verify_files():
  require({str(x):sha(x) for x in watched}==initial,"Prepared input changed")
  if hasattr(args,"lock_inode"):
   require(LOCK.is_file() and not LOCK.is_symlink() and LOCK.stat().st_ino==args.lock_inode,"GPU lock changed")
 expected=json.loads((root/"original-fixture.json").read_text())
 recorded=json.loads(args.model_inputs.read_text())["models"]["405"]
 state={"state":"preflight","passed":False,"started_at_utc":helper.stamp(),"input_sha256":initial,
        "cells":[],"scope":"Diagnostic source-stack attribution; preserves all numerical compatibility settings."}
 def save():helper.atomic(args.output/"result.json",state)
 def interrupted(signum,frame):raise KeyboardInterrupt("signal "+str(signum))
 for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,interrupted)
 save()
 try:
  require(LOCK.is_file() and not LOCK.is_symlink(),"Existing regular cooperative lock required")
  with LOCK.open("r+") as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);inode=os.fstat(lock.fileno()).st_ino;args.lock_inode=inode
   state.update(lock_inode=inode,owner_before=owner_gate(helper));save()
   check_model(helper,recorded);pre=[]
   for cell in cells(args):
    verify_files();owner_gate(helper)
    pre.append(preflight(helper,args,cell,args.output/(cell["label"]+".preflight"),uuid.uuid4().hex))
   state["preflight_complete"]=True;save()
   if not args.preflight_only:
    for cell,proof in zip(cells(args),pre):
     verify_files();require(LOCK.stat().st_ino==inode,"Lock path replaced");owner_gate(helper)
     row=run_cell(helper,parent,args,cell,args.output/cell["label"],proof,recorded,expected,verify_files)
     state["cells"].append({"label":cell["label"],"passed":row["passed"],"result_sha256":sha(args.output/cell["label"]/"result.json")})
     save();owner_gate(helper)
     require(row["passed"] and not row.get("cleanup_failed"),"Failed cell stops sequence")
   state.update(state="completed",passed=True,owner_after=owner_gate(helper))
 except BaseException as exc:state.update(state="failed",passed=False,error=type(exc).__name__+": "+str(exc))
 finally:state["finished_at_utc"]=helper.stamp();save()
 print(json.dumps(state,indent=2))
 return 0 if state["passed"] else 2
if __name__=="__main__":raise SystemExit(main())
