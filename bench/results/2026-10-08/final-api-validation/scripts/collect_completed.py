#!/usr/bin/env python3
"""Incrementally archive only completed final API jobs; never executes archived helpers."""
from pathlib import Path
import argparse,base64,datetime as dt,gzip,hashlib,json,subprocess

B=Path("/home/vcruz/src/qwen-overnight-20261008")
A=B/"recipe/bench/results/2026-10-08/final-api-validation"
EXPECTED_JOBS="50def7795a0a7401cbf17d585d5f0955e23ed99968de58bda3d92c836335b915"
p=argparse.ArgumentParser();p.add_argument("--finish",action="store_true");args=p.parse_args()
if A.exists():
    index=json.loads((A/"archive-status.json").read_text())
    assert index["kind"]=="completed-final-api-archive" and index["jobs_sha256"]==EXPECTED_JOBS
    if index["collection_complete"]:raise SystemExit("Refusing to modify a finalized archive")
else:
    index={"kind":"completed-final-api-archive","jobs_sha256":EXPECTED_JOBS,"collection_complete":False,"files":{},"completed_labels":[],"snapshots":[]}
known={label:index["files"]["reports/"+label+"/result.json"]["sha256"] for label in index["completed_labels"]}
remote=r'''
from pathlib import Path
import base64,datetime as dt,gzip,hashlib,json
ROOT=Path("/home/cruzspark/qwen-overnight-20261008")
RECIPE=Path("/home/cruzspark/qwen-spark-recipe")
D=ROOT/"results/final-api-24f0-5a"
KNOWN=__KNOWN__
FINISH=__FINISH__
FILES={
"final-api-jobs-selected.json":"50def7795a0a7401cbf17d585d5f0955e23ed99968de58bda3d92c836335b915",
"final_api_after_quality.py":"29a88c1576aa42f6383411f6df29fa55e97a6b1842ac2c6fba7b94fe33f79b2a",
"final_api_batch.py":"4820f2515b98b9f4698065bc03c393f82cc1c6f5aa9542d6d8599761478a1117",
"final_api_controller.py":"ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d",
"reasoning_literal_smoke.py":"8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054",
"reasoning_concurrency_smoke.py":"1e88c70eb82fa180b97d52b3ee71ad998153c24c99c2827e6da9646e1bbc102f",
"spark_experiment_controller.py":"48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586",
"api_f4_gemm_controller.py":"78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589"}
files={};omitted=[]
def read(p,limit=8_000_000):
    if p.is_symlink() or not p.is_file():raise RuntimeError("Not a regular evidence file: "+str(p))
    a=p.stat()
    if a.st_size>limit:raise RuntimeError("Unexpectedly large small-evidence file: "+str(p))
    raw=p.read_bytes();b=p.stat()
    if (a.st_size,a.st_mtime_ns)!=(b.st_size,b.st_mtime_ns):raise RuntimeError("Evidence changed during read: "+str(p))
    return raw,b
def add(p,rel,expected=None,limit=8_000_000):
    raw,stat=read(p,limit);digest=hashlib.sha256(raw).hexdigest()
    if expected is not None and digest!=expected:raise RuntimeError("Evidence hash mismatch: "+str(p))
    record={"archive_path":rel,"source_path":str(p),"bytes":len(raw),"mtime_ns":stat.st_mtime_ns,"sha256":digest,"data":base64.b64encode(raw).decode()}
    if rel in files and files[rel]["sha256"]!=digest:raise RuntimeError("Conflicting evidence destination")
    files[rel]=record
    return raw
batch_raw,_=read(D/"status.json",1_000_000);batch=json.loads(batch_raw)
jobs_raw=add(ROOT/"final-api-jobs-selected.json","inputs/final-api-jobs-selected.json",FILES["final-api-jobs-selected.json"])
jobs=json.loads(jobs_raw);labels=[j["label"] for j in jobs]
assert len(labels)==len(set(labels))==5
closed=[];closed_rows=[];new_labels=[]
for row in batch["jobs"]:
    label=row["label"]
    if label not in labels:raise RuntimeError("Unexpected batch job")
    if not row.get("finished_utc"):continue
    raw,_=read(D/label/"result.json")
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=row["result_sha256"]:raise RuntimeError("Final child result changed")
    result=json.loads(raw)
    if not result.get("finished_at_utc") or result.get("state") not in ("completed","failed"):raise RuntimeError("Batch marked an unfinished job complete")
    if result.get("cleanup_failed") or not result.get("server_cleanup",{}).get("owned_group_empty"):raise RuntimeError("Completed job cleanup is uncertain")
    if result["expected_engine"]!="24f0dece34f09c8d1e2359d6b3b3f7befef7331b" or result["expected_server"]!="5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb":raise RuntimeError("Unexpected final source cohort")
    deployment_raw = add(D/label/"deployment.json", "reports/"+label+"/deployment.json", result["deployment_sha256"])
    deployment = json.loads(deployment_raw)
    model_dir = Path(deployment["model"]["resolved_path"])
    expected_job = next(job for job in jobs if job["label"] == label)
    if str(model_dir) != expected_job["model_path"]:
        raise RuntimeError("Deployment model differs from the selected job")
    for name, metadata in deployment["model"].get("configuration_files", {}).items():
        if Path(name).name != name:
            raise RuntimeError("Invalid model configuration filename")
        add(model_dir/name, "source/models/"+label+"/"+name, metadata["sha256"], 500_000)
    closed.append(label);closed_rows.append(row)
    if label in KNOWN:
        if KNOWN[label]!=digest:raise RuntimeError("Previously archived result changed")
        continue
    new_labels.append(label)
    for q in sorted((D/label).rglob("*")):
        if q.is_symlink():
            omitted.append({"path":str(q),"reason":"Model-view symlink; identity retained in model/deployment metadata, target bytes not copied."})
        elif q.is_file():
            if q.suffix in (".json",".log",".yml",".yaml"):
                add(q,"reports/"+str(q.relative_to(D)))
            elif q.suffix==".lock":
                omitted.append({"path":str(q),"reason":"Empty/advisory runtime lock excluded."})
            else:raise RuntimeError("Unexpected completed artifact: "+str(q))
    for suffix in ("-job.json","-resources.json",".log"):
        q=D/(label+suffix)
        add(q,"reports/"+q.name,row["job_sha256"] if suffix=="-job.json" else None)
    for client in result["clients"]:
        q=Path(client["report"])
        if not q.is_relative_to(D/label):raise RuntimeError("Client report escaped its job directory")
        if hashlib.sha256(read(q)[0]).hexdigest()!=client["report_sha256"]:raise RuntimeError("Client report hash mismatch")
    sources=result["sources"]
    if sources["recipe"]["commit"]!="3337bc8d64e4befafa2a1aff342e07abbea242ac":raise RuntimeError("Unexpected recipe source")
    for rel,digest in sources.get("files_sha256",{}).items():
        if Path(rel).is_absolute() or ".." in Path(rel).parts:raise RuntimeError("Invalid recorded source path")
        add(RECIPE/rel,"source/recipe/"+rel,digest,500_000)
    for source,digest in result.get("extra_files_sha256",{}).items():
        q=Path(source)
        if q.is_relative_to(RECIPE):
            add(q,"source/recipe/"+str(q.relative_to(RECIPE)),digest,500_000)
        elif q.is_relative_to(ROOT):
            add(q,"source/work/"+str(q.relative_to(ROOT)),digest,500_000)
        else:
            omitted.append({"path":str(q),"recorded_sha256":digest,"reason":"External dependency identity retained in report; not copied."})
for name,digest in FILES.items():add(ROOT/name,"inputs/"+name,digest,500_000)
for name,rel in [
 ("results/setup-final-24f0-5a/status.json","inputs/setup-status.json"),
 ("results/final-api-handoff-24f0-5a/status.json","inputs/handoff-status.json")]:
    add(ROOT/name,rel)
for name in ("results/final-api-handoff-24f0-5a-launch.json","results/final-api-after-quality-launch.json","results/final-api-24f0-5a-launch.json"):
    q=ROOT/name
    if q.exists():
        raw=add(q,"inputs/"+q.name)
        if FINISH:
            log=Path(json.loads(raw)["log"])
            if not log.is_relative_to(ROOT/"logs"):raise RuntimeError("Unexpected handoff log path")
            add(log,"inputs/handoff-launcher.log")
if FINISH:
    if Path("/proc/430098").exists():raise RuntimeError("Final API matrix PID still exists")
    if batch.get("state")!="completed" or not batch.get("finished_utc") or set(closed)!=set(labels):raise RuntimeError("Final API batch is not fully completed")
    add(D/"status.json","batch/status.json")
if sum(x["bytes"] for x in files.values())>30_000_000:raise RuntimeError("Unexpected small-evidence volume")
payload={"collected_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
         "batch_status":base64.b64encode(batch_raw).decode(),"batch_status_sha256":hashlib.sha256(batch_raw).hexdigest(),
         "source_root":str(D),"closed_labels":closed,"closed_rows":closed_rows,"new_labels":new_labels,
         "final_matrix_pid_exists":Path("/proc/430098").exists(),"finalized":FINISH,
         "files":list(files.values()),"omitted":omitted}
print(base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())
'''
remote=remote.replace("__KNOWN__",repr(known)).replace("__FINISH__",repr(args.finish))
r=subprocess.run(["bash",str(B/"spark-ssh"),"python3","-"],input=remote,text=True,capture_output=True,check=True,timeout=45)
payload=json.loads(gzip.decompress(base64.b64decode(r.stdout.strip())))
A.mkdir(parents=True,exist_ok=True)
def immutable(rel,raw):
    p=A/rel;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        assert p.read_bytes()==raw,"Refusing changed archived file "+rel
    else:p.write_bytes(raw)
for f in payload["files"]:
    raw=base64.b64decode(f.pop("data"));assert len(raw)==f["bytes"] and hashlib.sha256(raw).hexdigest()==f["sha256"]
    immutable(f["archive_path"],raw)
    if f["archive_path"] not in index["files"]:index["files"][f["archive_path"]]=f
stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
snapshot="snapshots/"+stamp+"-batch.json"
batchraw=base64.b64decode(payload.pop("batch_status"));assert hashlib.sha256(batchraw).hexdigest()==payload["batch_status_sha256"]
immutable(snapshot,batchraw)
payload["batch_snapshot"]=snapshot
collection="snapshots/"+stamp+"-collection.json"
immutable(collection,(json.dumps(payload,indent=2,sort_keys=True)+"\n").encode())
index["snapshots"].append({"batch":snapshot,"collection":collection})
index.update(completed_labels=payload["closed_labels"],collection_complete=args.finish,
             updated_at_utc=payload["collected_at_utc"],matrix_pid_present_at_collection=payload["final_matrix_pid_exists"])
if args.finish:index["finished_at_utc"]=payload["collected_at_utc"]
(A/"archive-status.json").write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
for name in ["final-api-handoff-source-review.json"]:
    raw=(B/name).read_bytes();immutable("inputs/"+name,raw)
print(json.dumps({"archive":str(A),"collected_completed_jobs":len(payload["closed_labels"]),"new_jobs":payload["new_labels"],
                  "new_remote_files":len(payload["files"]),"new_remote_bytes":sum(f["bytes"] for f in payload["files"]),
                  "collection_complete":args.finish,"batch_snapshot":snapshot},indent=2))
