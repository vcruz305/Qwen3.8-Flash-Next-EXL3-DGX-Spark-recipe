#!/usr/bin/env python3
"""Run explicit final API jobs serially, retaining failures and sparse resources."""
from pathlib import Path
import argparse, hashlib, json, os, signal, subprocess, time, types

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs",type=Path,required=True)
    p.add_argument("--setup",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--job-timeout",type=float,default=3600)
    args=p.parse_args()
    if not 0 < args.job_timeout <= 5400:p.error("Job timeout must be in (0,5400]")
    args.jobs=args.jobs.resolve(strict=True);args.setup=args.setup.resolve(strict=True)
    args.output=args.output.resolve();args.output.mkdir(mode=0o700)
    root=Path(__file__).resolve().parent
    helper_path=root/"spark_experiment_controller.py"
    if sha(helper_path)!="48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586":
        raise ValueError("Frozen helper changed")
    helper=types.ModuleType("final_batch_frozen_helpers");helper.__file__=str(helper_path)
    exec(compile(helper_path.read_bytes(),str(helper_path),"exec"),helper.__dict__)
    jobs=json.loads(args.jobs.read_text())
    if not isinstance(jobs,list) or not jobs:raise ValueError("A nonempty explicit job list is required")
    if any(j.get("concurrency_client") or j.get("concurrency_bench") for j in jobs) and args.job_timeout < 3600:
        p.error("Concurrent validation jobs require at least a 3600-second outer deadline")
    labels=[j["label"] for j in jobs]
    if len(labels)!=len(set(labels)):raise ValueError("Duplicate job labels")
    wrapper=root/"final_api_controller.py"
    evidence={str(q):sha(q) for q in (Path(__file__).resolve(),helper_path,wrapper,args.jobs,args.setup)}
    record={"state":"running","passed":False,"started_utc":helper.stamp(),
            "source_files_sha256":evidence,"jobs":[]}
    child=None
    launch_active=False
    cleanup_active=False
    interruption_seen=None
    termination_sent=False
    def save():
        helper.atomic(args.output/"status.json",record)
    def verify():
        if any(sha(Path(q))!=v for q,v in evidence.items()):
            raise ValueError("Batch inputs changed during validation")
    def interrupted(signum,frame):
        nonlocal interruption_seen
        if interruption_seen is None:
            interruption_seen=signum
            if not launch_active and not cleanup_active:
                raise KeyboardInterrupt("signal "+str(signum))
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in previous_handlers:signal.signal(sig,interrupted)
    save()
    try:
        for job in jobs:
            verify()
            label=job["label"]
            if "/" in label or label in ("",".",".."):raise ValueError("Invalid job label")
            jobfile=args.output/(label+"-job.json")
            with jobfile.open("x") as f:json.dump(job,f,indent=2);f.write("\n")
            output=args.output/label
            command=["python3",str(wrapper),"--job",str(jobfile),"--setup",str(args.setup),"--output",str(output)]
            if job.get("bench",False):command.append("--bench")
            row={"label":label,"command":command,"job_sha256":sha(jobfile),
                 "started_utc":helper.stamp(),"passed":False}
            record["jobs"].append(row);record["active"]=label;save()
            sampler=helper.Sampler(args.output/(label+"-resources.json"),10,540)
            with (args.output/(label+".log")).open("x") as log:
                launch_active=True
                try:
                    child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    termination_sent=False
                    row["controller_pid"]=child.pid;save()
                finally:
                    launch_active=False
                if interruption_seen is not None:
                    raise KeyboardInterrupt("signal received during child registration")
                started=time.monotonic()
                while child.poll() is None:
                    phase,pid="controller",None
                    resultpath=output/"result.json"
                    if resultpath.is_file():
                        current=json.loads(resultpath.read_text())
                        phase=current.get("active") or current.get("state","controller")
                        pid=current.get("server_pid")
                    sampler.tick(phase,pid)
                    if time.monotonic()-started>args.job_timeout:
                        row["timed_out"]=True
                        # Finally owns the single bounded cleanup wait. Its installed
                        # handler owns server/client cleanup; never signal descendants.
                        raise TimeoutError("Final API job exceeded its outer deadline")
                    time.sleep(.25)
                row["exit_code"]=child.returncode;child=None
            resultpath=output/"result.json"
            if not resultpath.is_file():raise RuntimeError("Child produced no lifecycle result")
            result=json.loads(resultpath.read_text())
            cleanup=result.get("server_cleanup",{})
            if result.get("cleanup_failed") or not cleanup.get("owned_group_empty"):
                raise RuntimeError("Child server cleanup is absent or uncertain")
            for client in result.get("clients",[]):
                if client.get("name") == "concurrency":
                    report_path=Path(client["report"])
                    if not report_path.is_file() or sha(report_path)!=client.get("report_sha256"):
                        raise RuntimeError("Concurrent child cleanup report is missing or changed")
                    nested=json.loads(report_path.read_text())
                    if nested.get("cleanup_failed") or nested.get("owned_client_groups_empty") is not True:
                        raise RuntimeError("Concurrent client session cleanup is uncertain")
            row.update(passed=result.get("passed") is True and row["exit_code"]==0,
                       state=result.get("state"),finished_utc=helper.stamp(),
                       result_sha256=sha(resultpath),
                       clients=[{"name":c["name"],"passed":c.get("passed"),
                                 "counts":c.get("counts")} for c in result.get("clients",[])])
            if sha(jobfile)!=row["job_sha256"]:raise ValueError("Per-job input changed")
            verify();save()
            print(json.dumps({"label":label,"passed":row["passed"],"clients":row["clients"]}),flush=True)
        record.update(state="completed",active=None,passed=all(j["passed"] for j in record["jobs"]))
    except BaseException as exc:
        record.update(state="failed",passed=False,error=type(exc).__name__+": "+str(exc))
    finally:
        cleanup_active=True
        try:
            if child is not None and child.poll() is None:
                if not termination_sent:
                    termination_sent=True;child.terminate()
                try:child.wait(timeout=210)
                except subprocess.TimeoutExpired:record["cleanup_error"]="Owned child did not finish its cleanup; inspect before another load"
            record["finished_utc"]=helper.stamp();save()
        finally:
            for sig,previous in previous_handlers.items():signal.signal(sig,previous)
    return 0 if record["passed"] else 1
if __name__=="__main__":
    raise SystemExit(main())
