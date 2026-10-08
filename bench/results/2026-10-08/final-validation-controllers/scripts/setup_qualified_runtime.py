#!/usr/bin/env python3
"""Run the ordinary recipe setup for an already selected clean final source pair."""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, re, socket, subprocess, traceback

def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",required=True);p.add_argument("--tabby",required=True)
    p.add_argument("--recipe",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--after-results",type=Path,action="append",default=[])
    p.add_argument("--after-jobs",type=Path,action="append",default=[])
    p.add_argument("--after-pid",type=int,action="append",default=[])
    args=p.parse_args()
    if len(args.after_results)!=len(args.after_jobs):
        p.error("Pair each --after-results with its --after-jobs file")
    args.recipe=args.recipe.resolve(strict=True)
    args.output=args.output.resolve()
    for v in (args.engine,args.tabby):
        if not re.fullmatch("[0-9a-f]{40}",v):p.error("Full exact source commits are required")
    root=Path("/home/cruzspark/qwen-overnight-20261008")
    runtime=Path("/home/cruzspark/qwen38-exl3-20261008")
    args.output.mkdir(mode=0o700)
    record={"state":"preflight","passed":False,"started_utc":stamp(),
            "controller_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "engine":args.engine,"tabby":args.tabby,"runtime":str(runtime),"commands":[]}
    def save(state=None,**kw):
        if state:record["state"]=state
        record.update(updated_utc=stamp(),**kw)
        q=args.output/"status.json.tmp";q.write_text(json.dumps(record,indent=2)+"\n")
        q.replace(args.output/"status.json")
    def capture(command):
        return subprocess.check_output(command,text=True).strip()
    env={k:v for k,v in os.environ.items()
         if not k.startswith("EXL3_") and k not in
         ("PYTHONPATH","PYTHONHOME","API_KEY","TABBY_API_KEY","OPENAI_API_KEY",
          "QWEN_REASONING_OBSERVER_DIR","QWEN_REASONING_OBSERVER_ENABLED")}
    env.update(RECIPE_HOME=str(runtime),STATE_DIR=str(runtime/"state"),
               EXL3_REF=args.engine,TABBY_REF=args.tabby,MAX_JOBS="12",
               CUDA_HOME="/usr/local/cuda",TORCH_CUDA_ARCH_LIST="12.1")
    env["PATH"]=str(runtime/"venv/bin")+":/usr/local/cuda/bin:"+os.environ["PATH"]
    def run(command,name,cwd=None,extra=None):
        save("running",active=name)
        row={"name":name,"command":command,"started_utc":stamp()}
        record["commands"].append(row);save()
        log=args.output/(name+".log")
        with log.open("x") as f:
            r=subprocess.run(command,cwd=cwd or args.recipe,env=dict(env,**(extra or {})),
                             stdout=f,stderr=subprocess.STDOUT,timeout=2400)
        row.update(exit_code=r.returncode,finished_utc=stamp(),log=str(log),
                   log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        save()
        if r.returncode:raise RuntimeError(name+" exited "+str(r.returncode))
    save()
    try:
        for pid in args.after_pid:
            if Path("/proc").joinpath(str(pid)).exists():
                raise ValueError("Prerequisite controller PID still exists: "+str(pid))
        record["prerequisites"]=[]
        for previous,jobs_path in zip(args.after_results,args.after_jobs):
            jobs=json.loads(jobs_path.read_text())
            labels={j["label"] for j in jobs}
            if not labels or len(labels)!=len(jobs):
                raise ValueError("Prerequisite jobs must have unique nonempty labels")
            rows=list(previous.glob("*/result.json"))
            if {q.parent.name for q in rows}!=labels:
                raise ValueError("Missing or unexpected prerequisite matrix labels")
            statuses=[json.loads(q.read_text()) for q in rows]
            if any(s.get("state")!="completed" or not s.get("finished_at_utc") or
                   s.get("cleanup_error") or s.get("cleanup_failed") for s in statuses):
                raise ValueError("A prerequisite matrix job is incomplete or cleanup is uncertain")
            record["prerequisites"].append({"results":str(previous),"jobs":str(jobs_path),
                "jobs_sha256":hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
                "result_sha256":{q.parent.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in rows}})
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1",8899))==0:
                raise RuntimeError("Port8899 is occupied; refusing setup")
        record["sources_before"]={}
        for name,path,expected in (("engine",runtime/"exllamav3",args.engine),
                                   ("tabby",runtime/"tabbyAPI",args.tabby),
                                   ("recipe",args.recipe,None)):
            head=capture(["git","-C",str(path),"rev-parse","HEAD"])
            dirty=capture(["git","-C",str(path),"status","--porcelain","--untracked-files=no"])
            if dirty or (expected is not None and head!=expected):
                raise ValueError("Expected exact clean source: "+name)
            record["sources_before"][name]={"commit":head,"tree":capture(
                ["git","-C",str(path),"rev-parse","HEAD^{tree}"])}
        save()
        run(["bash","exllamav3-tabby/setup.sh"],"setup")
        run(["bash","exllamav3-tabby/setup.sh","--check"],"setup-check")
        py=str(runtime/"venv/bin/python")
        run([py,"-m","pytest","--noconftest","-q","tests/test_token_budget_cpu.py"],
            "engine-budget-cpu",runtime/"exllamav3",{"PYTHONPATH":"."})
        run([py,"-m","pytest","-q","-ra"],"tabby-cpu",runtime/"tabbyAPI",{"PYTHONPATH":"."})
        run([py,"tests/check_qwen_tool_tokenizer.py",
             "/home/cruzspark/models/flashnext-exl3-3.05bpw/tokenizer.json"],
            "tool-tokenizer",runtime/"tabbyAPI",{"PYTHONPATH":"."})
        for name,path,expected in (("engine",runtime/"exllamav3",args.engine),
                                   ("tabby",runtime/"tabbyAPI",args.tabby),
                                   ("recipe",args.recipe,record["sources_before"]["recipe"]["commit"])):
            if capture(["git","-C",str(path),"rev-parse","HEAD"])!=expected or capture(
                    ["git","-C",str(path),"status","--porcelain","--untracked-files=no"]):
                raise ValueError("Source changed or became dirty during setup: "+name)
        marker=(runtime/"venv/.qwen38-recipe-runtime").read_text().strip()
        if marker!=args.engine:raise ValueError("Runtime marker does not match final engine")
        record["build_state"]=json.loads((runtime/"venv/.qwen38-recipe-runtime.json").read_text())
        record["recipe_commit"]=record["sources_before"]["recipe"]["commit"]
        save("completed",passed=True,active=None,finished_utc=stamp())
        return 0
    except BaseException as exc:
        save("failed",passed=False,error=type(exc).__name__+": "+str(exc),finished_utc=stamp())
        traceback.print_exc()
        return 1
if __name__=="__main__":
    raise SystemExit(main())
