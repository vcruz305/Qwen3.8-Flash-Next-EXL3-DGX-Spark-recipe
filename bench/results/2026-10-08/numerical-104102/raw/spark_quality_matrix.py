#!/usr/bin/env python3
"""Serial fixed-input numerical experiments with immutable per-job artifacts."""
from pathlib import Path
import argparse, datetime as dt, hashlib, json, os, socket, subprocess, traceback
ROOT=Path("/home/cruzspark/qwen-overnight-20261008")
p=argparse.ArgumentParser();p.add_argument("--jobs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
jobs=json.loads(args.jobs.read_text());assert isinstance(jobs,list) and jobs
args.output.mkdir(parents=True,exist_ok=False)
status={"state":"initializing","jobs":[],"jobs_sha256":hashlib.sha256(args.jobs.read_bytes()).hexdigest()}
def save(state,**extra):
    status.update(state=state,recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**extra)
    t=args.output/"status.tmp";t.write_text(json.dumps(status,indent=2)+"\n");t.replace(args.output/"status.json")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def commit(path):return subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip()
try:
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1",8899))==0:raise RuntimeError("Port8899 occupied")
    names=[j["name"] for j in jobs];assert len(names)==len(set(names))
    for j in jobs:
        assert j["name"] and "/" not in j["name"] and j["name"] not in (".","..")
        d=args.output/j["name"];d.mkdir()
        engine=Path(j["engine_root"]);py=Path(j["python"])
        assert commit(engine)==j["engine_sha"],"Unexpected source"
        tracked=subprocess.check_output(["git","-C",str(engine),"status","--porcelain"],text=True)
        assert not tracked.strip(),"Dirty source"
        marker=py.parent.parent/".qwen38-recipe-runtime"
        record={"name":j["name"],"config":j,"probe_sha256":sha(ROOT/"spark_quality_probe.py"),
                "assessor_sha256":sha(ROOT/"assess_paired_quality.py"),"gates_sha256":sha(ROOT/"results/quality-gates.json"),
                "compiled_marker":marker.read_text().strip() if marker.exists() else None,
                "purpose":"Numerical source diagnostic; source-only changes do not qualify normal serving"}
        (d/"input.json").write_text(json.dumps(record,indent=2)+"\n")
        status["jobs"].append(record);save("probing",active=j["name"])
        env={k:v for k,v in os.environ.items() if not k.startswith("EXL3_")}
        env.update(j.get("environment",{}),PYTHONPATH=str(engine))
        cmd=[str(py),str(ROOT/"spark_quality_probe.py"),"--model",j["model"],
             "--output",str(d/"probe.json"),"--save-logits",str(d/"logits.safetensors")]
        if j.get("ngram_ram"):cmd+=["--ngram-ram"]
        for opt in ("contexts","q_lens","batch_sizes","steps","prefill_chunk","cases"):
            if opt in j:cmd+=["--"+opt.replace("_","-"),str(j[opt])]
        if j.get("baseline_logits"):cmd+=["--compare-logits",j["baseline_logits"]]
        with (d/"probe.log").open("x") as log:
            run=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        record["probe_exit"]=run.returncode;save("checking",active=j["name"])
        assert run.returncode==0,"Probe process failed"
        report=json.loads((d/"probe.json").read_text())
        assert report.get("sanity_checks_passed") is True and report["runtime"]["git_commit"]==j["engine_sha"]
        assert commit(engine)==j["engine_sha"],"Source changed during probe"
        if j.get("baseline_json"):
            with (d/"assessment.log").open("x") as log:
                run=subprocess.run(["python3",str(ROOT/"assess_paired_quality.py"),"--gates",str(ROOT/"results/quality-gates.json"),
                      "--baseline",j["baseline_json"],"--candidate",str(d/"probe.json"),"--output",str(d/"assessment.json")],
                      cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            record["assessment_exit"]=run.returncode
            a=json.loads((d/"assessment.json").read_text())
            record.update(core_passed=a.get("core_passed"),investigation_count=a.get("investigation_count"))
        record.update(completed=True,report=str(d/"probe.json"))
        save("running",active=None)
    save("completed",active=None)
except Exception as exc:
    save("failed",error=f"{type(exc).__name__}: {exc}")
    traceback.print_exc();raise
