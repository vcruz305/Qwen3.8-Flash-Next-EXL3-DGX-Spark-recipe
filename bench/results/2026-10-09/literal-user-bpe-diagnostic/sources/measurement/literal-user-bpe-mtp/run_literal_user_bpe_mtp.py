#!/usr/bin/env python3
"""One MTP literal8 confirmation with explicitly changed user-span encoding.

The shared GPU flock is held across the one diagnostic cell. This never stops a manager,
changes refs, installs packages, enables a service or repairs a failed fixture.
"""
from __future__ import annotations
import argparse, copy, datetime as dt, fcntl, hashlib, json, os
from pathlib import Path
import re, signal, subprocess, types

ENGINE="24f0dece34f09c8d1e2359d6b3b3f7befef7331b"
TABBY="f650bb5389e0a273549e47d4d26a765760c013e1"
RECIPE="a8c72bdf811646f413fdc03a4e49811a2753d0cf"
OBSERVER="72ad246d0f746c053431a992b0c03959904c2310fe37da21d83420ce4537e592"
INPUT_HOOK="a0207b478bf97b672f5bef024120ae4ecaf74146dc2074caeb5a967c912bcc0a"
ORIGINAL_CONTROLLER="2a0a0e3c36f482581847dc9c1e84ebc03dc72014006ed45c8b23cd348d0bed2e"
TOKENIZER="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
SITE="3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f"
FROZEN={
 "literal_capture_controller_p095.py":"d101f3ccd82eb94c75733e0c5e561a83ac5f392886a4d4dd1ee4f07d247e263d",
 "strings_only_controller.py":"cfc3619894c4ee8f07d46c2ee7ed0fd085f3fdc044c81c8d9a01d009304c4116",
 "final_api_controller.py":"ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d",
 "api_f4_gemm_controller.py":"78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589",
 "spark_experiment_controller.py":"48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586",
 "reasoning_literal_smoke.py":"8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054",
 "validate_timeline.py":"f90f3f8bf565550f599a060ab38a002018e49726dd5d818732a8ff82966c45cb",
 "validate_input_tokenization.py":"e55a40cabb5e6257f72b36e689b6b58a7189e2eb27a3410e9395c7095f443dfa",
}
GPU_LOCK=Path("/home/cruzspark/redsnow-gpu.lock")
def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, message):
    if not value: raise ValueError(message)
def atomic(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as file:
        json.dump(value,file,indent=2,sort_keys=True,allow_nan=False);file.write("\n")
        file.flush();os.fsync(file.fileno())
    temporary.replace(path)
def load(path, expected, name):
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,"Frozen source changed: "+str(path))
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),"exec"),module.__dict__);return module
def command(argv,**kwargs):
    return subprocess.check_output(argv,text=True,timeout=30,**kwargs).strip()
def git_identity(path):
    return {"path":str(path),"commit":command(["git","-C",str(path),"rev-parse","HEAD"]),
      "dirty":command(["git","-C",str(path),"status","--porcelain","--untracked-files=no"])}
def validate_jobs(jobs):
    require(type(jobs) is list and len(jobs)==1,"Exactly one MTP BPE confirmation is required")
    job=jobs[0]
    require(set(job)=={"label","engine","tabby","recipe","model_path","env"},"Unexpected job fields")
    require(job["engine"]==ENGINE and job["tabby"]==TABBY,"Exact qualified source pair required")
    require(job["label"]=="literal-user-bpe-mtp" and job["env"]["DRAFT_MODE"]=="mtp",
            "Only the MTP changed-input confirmation cell is authorized")
    require(job["env"]["PROFILE"]=="single" and str(job["env"]["NGRAM_RAM"]).lower()=="true"
            and int(job["env"]["MAX_BATCH_SIZE"])==1,"Preserve original single/RAM geometry")

def prepare_bpe_bundle(lib,observer_dir,bundle,output,job,runtime):
    sources={"strings_observer.py":OBSERVER,"sitecustomize.py":SITE,"user_bpe_diagnostic.py":INPUT_HOOK}
    for name,expected in sources.items():require(sha(observer_dir/name)==expected,"BPE observer source changed")
    tokenizer=(Path(job["model_path"])/"tokenizer.json").resolve(strict=True)
    require(sha(tokenizer)==TOKENIZER,"Audited tokenizer changed")
    bundle.mkdir(mode=0o700)
    for name in sources:(bundle/name).write_bytes((observer_dir/name).read_bytes())
    config={"engine_repo":str(runtime/"exllamav3"),"tabby_repo":str(runtime/"tabbyAPI"),
      "engine_commit":job["engine"],"tabby_commit":job["tabby"],"run_label":job["label"],
      "max_records":8,"output_dir":str(output/"raw-literal"),
      "input_tokenizer_json":str(tokenizer),"diagnostic_user_bpe":True}
    lib.atomic_new(bundle/"observer-config.json",config)
    return {"derived_observer_sha256":OBSERVER,"sitecustomize_sha256":SITE,"config":config,
      "files_sha256":{str(p):sha(p) for p in sorted(bundle.iterdir())},
      "input_tokenizer_sha256":TOKENIZER,"input_hook_sha256":INPUT_HOOK,
      "note":"Changed-input diagnostic: only the exact synthetic user span is encoded as ordinary BPE. "
             "Visible requests/templates remain unchanged; model input token IDs and prompt usage change."}

def validate_api_prompt_usage(output):
    rows=[]
    for name in ("literal","literal-unbudgeted"):
        report=json.loads((output/(name+".json")).read_text())
        require(len(report.get("requests",[]))==4,"Four requests per original client required")
        for request in report["requests"]:
            if request["stream"]:
                usages=[f["usage"] for f in request.get("frames",[]) if f.get("usage") is not None]
            else:usages=[request.get("body",{}).get("usage")]
            require(len(usages)==1 and isinstance(usages[0],dict),"Exactly one authoritative API usage required")
            usage=usages[0]
            require(type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"]==352,
                    "API prompt usage does not reflect the changed352-token input")
            cached=usage.get("prompt_tokens_details",{}).get("cached_tokens")
            require(type(cached) is int and 0<=cached<=352,"Invalid actual prompt-cache usage")
            completion=usage.get("completion_tokens")
            require(type(completion) is int and completion>0,"Missing actual completion usage")
            rows.append({"client":name,"request_index":request["index"],
                         "prompt_tokens":352,"cached_prompt_tokens":cached,"completion_tokens":completion,
                         "completion_tokens_details":usage.get("completion_tokens_details")})
    return rows

def validate_mtp_evidence(output,usage_rows):
    deployment=json.loads((output/"deployment.json").read_text())
    draft=deployment.get("config",{}).get("values",{}).get("draft_model",{})
    require(draft.get("draft_mode")=="mtp" and draft.get("draft_num_tokens")==5
            and draft.get("dynamic_draft") is True,"Actual deployment is not the requested dynamic MTP depth5")
    require(deployment.get("environment",{}).get("DRAFT_MODE")=="mtp","Deployed drafting environment changed")
    accepted=rejected=0
    for row in usage_rows:
        details=row.get("completion_tokens_details") or {}
        for key in ("accepted_prediction_tokens","rejected_prediction_tokens"):
            require(type(details.get(key)) is int and details[key]>=0,"Missing actual MTP work counters")
        accepted+=details["accepted_prediction_tokens"];rejected+=details["rejected_prediction_tokens"]
    require(accepted+rejected>0,"No actual MTP proposal work observed in the confirmation")
    return {"draft_mode":"mtp","draft_num_tokens":5,"dynamic_draft":True,
            "accepted_prediction_tokens":accepted,"rejected_prediction_tokens":rejected,
            "scope":"Observed proposal counts, not verification-window counts or a throughput claim."}

def owner_gate(manager_unit, allowed_pid=None):
    raw=command(["systemctl","--user","show",manager_unit,"-p","LoadState","-p","ActiveState","-p","MainPID"])
    fields=dict(line.split("=",1) for line in raw.splitlines())
    require(fields.get("LoadState")=="loaded" and fields.get("ActiveState")=="inactive"
            and fields.get("MainPID")=="0","Manager must be stopped by its owner before/through this window")
    qwen=command(["systemctl","--user","show","qwen38-exl3.service","-p","ActiveState","-p","MainPID"])
    qfields=dict(line.split("=",1) for line in qwen.splitlines())
    require(qfields.get("ActiveState")=="inactive" and qfields.get("MainPID")=="0","Qwen unit must remain inactive")
    enabled=subprocess.run(["systemctl","--user","is-enabled","qwen38-exl3.service"],
                           text=True,capture_output=True,timeout=10)
    require(enabled.stdout.strip()=="disabled","Preserve the observed disabled Qwen unit state")
    gpu=command(["nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory","--format=csv,noheader,nounits"])
    pids=[]
    for line in gpu.splitlines():
        if not line.strip():continue
        first=line.split(",",1)[0].strip()
        require(first.isdigit(),"Unrecognized GPU ownership result");pids.append(int(first))
    require(set(pids)<=({allowed_pid} if allowed_pid is not None else set()),"Another CUDA owner is present")
    return {"at_utc":stamp(),"manager":fields,"qwen":qfields,"qwen_enabled":"disabled","gpu":gpu}

def verify_model(helper,job,recorded):
    current=helper.model_identity(Path(job["model_path"]))
    require(current==recorded["identity"],"Pack metadata/loader order changed from Oct9 audit")
    import struct
    for entry in recorded["headers_in_loader_order"]:
        path=Path(job["model_path"])/entry["name"];before=path.stat()
        with path.open("rb") as handle:
            size=struct.unpack("<Q",handle.read(8))[0]
            require(size==entry["header_bytes"],"Safetensors header length changed")
            raw=handle.read(size)
        after=path.stat()
        require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),"Weight changed during header read")
        require(hashlib.sha256(raw).hexdigest()==entry["header_sha256"],"Safetensors header changed")
    return current

def check_preflight(value, recipe, runtime, recorded_sources):
    require(value.get("kind")=="fresh-read-only-runtime-preflight" and value.get("passed") is True
      and value.get("finished_at_utc") and value.get("runtime")==str(runtime)
      and value.get("recipe_commit")==RECIPE and value.get("sources")==recorded_sources
      and value.get("setup_check_exit_code")==0,"Actual completed runtime preflight required")
    imports=value.get("imports",{})
    require(Path(imports.get("engine_file","")).resolve()==(runtime/"exllamav3/exllamav3/__init__.py").resolve(),
            "Engine import path mismatch")
    require(imports.get("version")=="1.6.0.post1" and "can_end" in imports.get("budget_parameters",[]),
            "Qualified native phase capability is absent")
    for path,key in ((recipe,"recipe"),(runtime/"exllamav3","engine"),(runtime/"tabbyAPI","server")):
        require(git_identity(path)==recorded_sources[key],"Source changed after preflight")

def verify_control_inputs(args):
    for path,expected in args.input_hashes.items():
        require(sha(path)==expected,"Diagnostic input changed: "+path)
    if hasattr(args,"lock_inode"):
        require(args.gpu_lock.stat().st_ino==args.lock_inode,"GPU lock path changed")

def run_cell(args,job,job_path,output,inputs):
    verify_control_inputs(args)
    frozen=Path(__file__).resolve().parent/"frozen"
    cap=load(frozen/"literal_capture_controller_p095.py",FROZEN["literal_capture_controller_p095.py"],"literal_assessor")
    cap.ENGINE,cap.TABBY,cap.OBSERVER_SHA=ENGINE,TABBY,OBSERVER
    lib=load(frozen/"strings_only_controller.py",FROZEN["strings_only_controller.py"],"literal_attachment")
    wrapper=load(frozen/"final_api_controller.py",FROZEN["final_api_controller.py"],"literal_configuration")
    fixture=load(frozen/"reasoning_literal_smoke.py",FROZEN["reasoning_literal_smoke.py"],"literal_fixture")
    timeline=load(frozen/"validate_timeline.py",FROZEN["validate_timeline.py"],"literal_timeline_validator")
    encoding=load(frozen/"validate_input_tokenization.py",FROZEN["validate_input_tokenization.py"],"literal_input_validator")
    helper=load(frozen/"spark_experiment_controller.py",FROZEN["spark_experiment_controller.py"],"literal_helper")
    parent=wrapper.load_parent(frozen/"api_f4_gemm_controller.py")
    parent.RUNTIME=args.runtime
    # Retain the existing separate SDK environment used by the frozen lifecycle's metadata inventory.
    parent.ROOT=args.client_root
    setup=output.with_name(output.name+".preflight.json")
    configured=copy.deepcopy(job);configured.update(api=False,literal_client=str(frozen/"reasoning_literal_smoke.py"))
    wrapper.configure(parent,configured,job_path,output,setup)
    sources={"recipe":git_identity(parent.RECIPE),"engine":git_identity(args.runtime/"exllamav3"),
             "server":git_identity(args.runtime/"tabbyAPI")}
    for key,pin in (("recipe",RECIPE),("engine",ENGINE),("server",TABBY)):
        require(sources[key]["commit"]==pin and not sources[key]["dirty"],"Expected clean source: "+key)
    verify_model(helper,job,inputs)
    preflight={"kind":"fresh-read-only-runtime-preflight","started_at_utc":stamp(),"passed":False,
               "runtime":str(args.runtime),"recipe_commit":RECIPE,"sources":sources,
               "note":"Fresh setup --check/import checks only; no prior-day CPU test rerun is claimed."}
    env=helper.resolved_env(parent.RECIPE,args.runtime,{"env":parent.TUNING,"model_path":job["model_path"]})
    env["PYTHONPATH"]=str(args.runtime/"exllamav3")
    log=output.with_name(output.name+".setup-check.log")
    with log.open("x") as stream:
        check=subprocess.run(["bash",str(parent.RECIPE/"exllamav3-tabby/setup.sh"),"--check"],
           cwd=parent.RECIPE,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    preflight["setup_check_exit_code"]=check.returncode;preflight["setup_check_log_sha256"]=sha(log)
    atomic(setup,preflight);require(check.returncode==0,"Read-only setup --check failed")
    probe=("import inspect,json,exllamav3; from exllamav3.version import __version__; "
           "from exllamav3.generator import AsyncJob; "
           "print(json.dumps({'engine_file':exllamav3.__file__,'version':__version__,"
           "'budget_parameters':list(inspect.signature(AsyncJob.set_token_budget).parameters)}))")
    preflight["imports"]=json.loads(command([str(args.runtime/"venv/bin/python"),"-c",probe],env=env))
    preflight.update(passed=True,finished_at_utc=stamp())
    parent.check_setup=lambda value:check_preflight(value,parent.RECIPE,args.runtime,sources)
    try:
        parent.check_setup(preflight)
    except BaseException as exc:
        preflight.update(passed=False,error=type(exc).__name__+": "+str(exc))
        atomic(setup,preflight);raise
    atomic(setup,preflight)
    parent.EXTRA_FILES+=(str(Path(__file__).resolve()),str(args.model_inputs),str(args.jobs),str(frozen/"validate_timeline.py"),str(frozen/"validate_input_tokenization.py"))
    prior_load=parent.load_helpers
    def load_with_gates(path):
        h=prior_load(path);prior_verify=h.verify_inputs
        def verify(*a,**k):
            verify_control_inputs(args)
            prior_verify(*a,**k)
            record=output/"result.json";pid=None
            if record.exists():pid=json.loads(record.read_text()).get("server_pid")
            owner_gate(args.manager_unit,pid)
            for name,expected in FROZEN.items():require(sha(frozen/name)==expected,"Frozen diagnostic changed")
            verify_model(h,job,inputs)
        h.verify_inputs=verify;return h
    parent.load_helpers=load_with_gates
    bundle=output.with_name(output.name+".observer-inputs")
    provenance=prepare_bpe_bundle(lib,args.observer_dir,bundle,output,job,args.runtime)
    observer=load(bundle/"strings_observer.py",OBSERVER,"literal_observer")
    require(observer.ENGINE_HEAD==ENGINE and observer.TABBY_HEAD==TABBY,"Observer pins differ")
    guard=lib.SignalGuard();cap.attach(parent,lib,bundle,guard,provenance,None)
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in previous:signal.signal(sig,guard.interrupt)
    try:
        exit_code=parent.run(types.SimpleNamespace(bench=False,ready_timeout=args.ready_timeout,
                                                   client_timeout=args.client_timeout))
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    summary={"schema_version":1,"diagnostic_only":True,"passed":False,"capture_valid":False,
      "controller_sha256":sha(__file__),"derived_from_controller_sha256":ORIGINAL_CONTROLLER,
      "changed_model_input":True,"original_prompt_tokens":348,"expected_prompt_tokens":352,
      "frozen_assessor_sha256":FROZEN["literal_capture_controller_p095.py"],
      "assessor_global_bindings":{"ENGINE":ENGINE,"TABBY":TABBY,"OBSERVER_SHA":OBSERVER},
      "lifecycle_exit_code":exit_code,"job_sha256":sha(job_path),"provenance":provenance,
      "interruption_observed":guard.pending,"preflight_sha256":sha(setup)}
    try:
        summary.update(cap.assess(output,job,provenance,observer,fixture,parent.ALIAS,exit_code,False))
        summary["producer_timelines"]=[{ "trace_sha256":row["sha256"],
            **timeline.validate_timeline(json.loads(Path(row["path"]).read_text()))}
            for row in summary["traces"]]
        manifest=json.loads((output/"raw-literal/observer-manifest.json").read_text())
        require(manifest["source"].get("input_hook_sha256")==INPUT_HOOK
                and manifest["source"].get("changed_input_diagnostic") is True,"Input hook provenance mismatch")
        summary["input_encoding_proofs"]=[{"trace_sha256":row["sha256"],
            **encoding.validate_input_tokenization(json.loads(Path(row["path"]).read_text()))}
            for row in summary["traces"]]
        summary["api_prompt_usage"]=validate_api_prompt_usage(output)
        summary["mtp_evidence"]=validate_mtp_evidence(output,summary["api_prompt_usage"])
        summary["scope"]="Changed input tokenization on the original eight visible requests; not the unchanged baseline."
        summary["passed"]=summary["capture_valid"] and summary["all_clients_passed"]
    except Exception as exc:
        summary.update(capture_valid=False,passed=False,capture_error=type(exc).__name__+": "+str(exc))
    lib.atomic_new(output/"literal-capture.json",summary)
    return summary

def run_sequence(args,jobs,inputs,state,save):
    for job in jobs:
        verify_control_inputs(args)
        require(args.gpu_lock.stat().st_ino==state["lock_inode"],"GPU lock path changed")
        gate=owner_gate(args.manager_unit);cell={"label":job["label"],"owner_gate_before":gate}
        state["cells"].append(cell);state["state"]="running";save()
        job_path=args.output/(job["label"]+".job.json");atomic(job_path,job)
        try:
            summary=run_cell(args,job,job_path,args.output/job["label"],inputs)
            cell.update(capture_valid=summary["capture_valid"],passed=summary["passed"],
                 semantic_counts=summary.get("original_semantic_counts"),
                 assessment_sha256=sha(args.output/job["label"]/"literal-capture.json"))
        except BaseException as exc:
            cell["error"]=type(exc).__name__+": "+str(exc);save();raise
        cell["owner_gate_after"]=owner_gate(args.manager_unit);save()
        require(summary["capture_valid"] and not summary.get("interruption_observed"),
                "Invalid capture/interruption fails the diagnostic; semantic failures are retained separately")
    state.update(state="completed",passed=all(x["passed"] for x in state["cells"]),
                 capture_valid=all(x["capture_valid"] for x in state["cells"]))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("jobs","output","runtime","observer-dir","model-inputs","client-root"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--gpu-lock",type=Path,default=GPU_LOCK)
    p.add_argument("--manager-unit",default="rexl3-manager.service")
    p.add_argument("--ready-timeout",type=float,default=600)
    p.add_argument("--client-timeout",type=float,default=900)
    args=p.parse_args(argv)
    require(args.manager_unit=="rexl3-manager.service","Only the reviewed manager unit is supported")
    require(args.gpu_lock==GPU_LOCK,"Use the existing cooperative lock")
    require(0<args.ready_timeout<=1800 and 0<args.client_timeout<=1200,"Invalid bounded deadlines")
    for key in ("jobs","runtime","observer_dir","model_inputs","client_root"):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    args.input_hashes={str(path):sha(path) for path in
       (Path(__file__).resolve(),args.jobs,args.model_inputs,args.observer_dir/"strings_observer.py",
        args.observer_dir/"sitecustomize.py",args.observer_dir/"user_bpe_diagnostic.py")}
    require(not args.output.exists(),"Refusing existing output")
    jobs=json.loads(args.jobs.read_text());validate_jobs(jobs)
    inputs=json.loads(args.model_inputs.read_text())["models"]["305"]
    require(all(job["model_path"]==inputs["model_path"] for job in jobs),"Only audited original305 pack is supported")
    require(sha(args.observer_dir/"strings_observer.py")==OBSERVER and sha(args.observer_dir/"sitecustomize.py")==SITE
            and sha(args.observer_dir/"user_bpe_diagnostic.py")==INPUT_HOOK,
            "Observer source changed")
    os.umask(0o077);args.output.mkdir(parents=True,exist_ok=False)
    state={"schema_version":1,"started_at_utc":stamp(),"state":"preflight","passed":False,
       "controller_sha256":sha(__file__),"derived_from_controller_sha256":ORIGINAL_CONTROLLER,
       "changed_model_input":True,"jobs_sha256":sha(args.jobs),"model_inputs_sha256":sha(args.model_inputs),
       "gpu_lock":str(args.gpu_lock),"manager_unit":args.manager_unit,"cells":[],
       "input_hashes":args.input_hashes}
    def save():atomic(args.output/"result.json",state)
    def interrupted(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        raise KeyboardInterrupt("signal "+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    save()
    try:
        require(args.gpu_lock.is_file() and not args.gpu_lock.is_symlink(),"Existing regular GPU lock required")
        with args.gpu_lock.open("r+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state["lock_acquired_at_utc"]=stamp();state["lock_inode"]=os.fstat(lock.fileno()).st_ino
            args.lock_inode=state["lock_inode"]
            run_sequence(args,jobs,inputs,state,save)
    except BaseException as exc:
        state.update(state="failed",passed=False,error=type(exc).__name__+": "+str(exc))
    finally:
        state["finished_at_utc"]=stamp();save()
    print(json.dumps(state,indent=2),flush=True)
    return 0 if state["passed"] else 1 if state.get("capture_valid") else 2
if __name__=="__main__":raise SystemExit(main())
