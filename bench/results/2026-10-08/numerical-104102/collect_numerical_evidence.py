#!/usr/bin/env python3
"""Copy a bounded numerical evidence snapshot from Spark; never run model code.

Remote side only reads selected existing files, stats and hashes them. Only the
local WSL destination is created. Large tensor bytes are hashed, never copied.
"""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/home/cruzspark/qwen-overnight-20261008"
REMOTE = r'''
import base64, hashlib, json, os, pathlib, time
from datetime import datetime, timezone
root=pathlib.Path("/home/cruzspark/qwen-overnight-20261008")
results=root/"results"
os.nice(10)
started=datetime.now(timezone.utc).isoformat()
selected=set()
tensors=set()
excluded=[]
completed_directories={}
limit=256*1024

def select(path):
    if path.is_file():
        selected.add(path)
def tensor(path):
    if path.is_file():
        tensors.add(path)
def matching_tensor(path):
    tensor(path.with_suffix(".safetensors"))

# Frozen single-report numerical attempts, including the failed saved baseline
# record. The latter is preserved as an error record, never a selected baseline.
for path in results.glob("quality-*.json"):
    select(path)
    if not any(part in path.name for part in ("assessment", "status", "gates", "save-error")):
        matching_tensor(path)

# Completed matrix directories only. In-flight statuses are copied with an
# explicit exclusion entry; their partial report/logit files are not collected.
for directory in sorted(results.glob("quality-*")):
    if not directory.is_dir():
        continue
    status=directory/"status.json"
    if not status.is_file():
        excluded.append({"path":str(directory),"reason":"No completion status; not collected"})
        continue
    raw=status.read_bytes()
    record=json.loads(raw)
    select(status)
    if record.get("state")!="completed" or record.get("active") is not None:
        excluded.append({"path":str(directory),"reason":"Matrix not completed at snapshot selection","state":record.get("state"),"active":record.get("active")})
        continue
    completed_directories[directory]=hashlib.sha256(raw).hexdigest()
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix==".json":
            select(path)
        elif path.is_file() and path.name in ("probe.log","assessment.log"):
            select(path)
        elif path.is_file() and path.suffix==".safetensors":
            tensor(path)

# Native BC old/new control, its private cache provenance, and raw pytest logs.
for path in results.glob("bc-row-control-405-b532-*.json"):
    select(path)
    matching_tensor(path)
for path in results.glob("bc-row-control-cache-*/manifest.json"):
    select(path)
select(results/"gpu-checks-status.json")
select(results/"validate-16ca/status.json")
for path in (results/"validate-16ca").glob("gpu-policy-*.log"):
    select(path)
logs=root/"logs"
for path in logs.glob("gpu-*.log"):
    select(path)
for path in logs.glob("quality-*.log"):
    select(path)
for path in logs.glob("bc-row-control-405-b532-*.log"):
    select(path)
for name in ("api3a-gpu-merge.log","setup-gemm-f4-gpu-merge.log"):
    select(logs/name)
for pattern in ("quality-*.json",):
    for path in root.glob(pattern):
        select(path)
for name in ("spark_quality_matrix.py","spark_quality_q8_matrix.py",
             "validate_16ca_controller.py","bc_row_control_405_controller.py"):
    select(root/name)

rows=[]
for path in sorted(selected):
    before=path.stat()
    if before.st_size>limit:
        excluded.append({"path":str(path),"reason":"Exceeds small-file limit","bytes":before.st_size})
        continue
    data=path.read_bytes()
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError("File changed while collecting: "+str(path))
    if path.suffix==".json":
        json.loads(data)
    rows.append({"path":str(path),"relative_path":str(path.relative_to(root)),
                 "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),
                 "mtime_ns":before.st_mtime_ns,"data_base64":base64.b64encode(data).decode("ascii")})

tensor_rows=[]
for path in sorted(tensors):
    before=path.stat()
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    after=path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError("Tensor changed while hashing: "+str(path))
    tensor_rows.append({"path":str(path),"relative_path":str(path.relative_to(root)),
                        "bytes":before.st_size,"sha256":digest.hexdigest(),
                        "mtime_ns":before.st_mtime_ns,"copied":False})

for directory,expected in completed_directories.items():
    if hashlib.sha256((directory/"status.json").read_bytes()).hexdigest()!=expected:
        raise RuntimeError("Completed matrix status changed during snapshot: "+str(directory))

print(json.dumps({"started_at_utc":started,"finished_at_utc":datetime.now(timezone.utc).isoformat(),
                  "remote_root":str(root),"read_only_remote":True,"small_file_limit_bytes":limit,
                  "selected_files":rows,"tensor_files":tensor_rows,"excluded":excluded,
                  "purpose":"Timestamped evidence preservation, not final deployment qualification"}))
'''

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    os.chmod(out,0o700)
    (out/"collecting.json").write_text(json.dumps({"started_at_utc":datetime.now(timezone.utc).isoformat(),"collector_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})+"\n")
    r=subprocess.run(["bash",str(ROOT/"spark-ssh"),"python3","-"],input=REMOTE,
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,timeout=600)
    if r.stderr.strip():
        (out/"ssh-stderr.txt").write_text(r.stderr)
    snapshot=json.loads(r.stdout)
    for row in snapshot["selected_files"]:
        relative=Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid relative remote artifact path")
        data=base64.b64decode(row.pop("data_base64"),validate=True)
        if len(data)!=row["bytes"] or hashlib.sha256(data).hexdigest()!=row["sha256"]:
            raise ValueError("Transferred bytes differ from remote hash")
        target=out/"raw"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as f:f.write(data)
    snapshot["collector_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot["local_finished_at_utc"]=datetime.now(timezone.utc).isoformat()
    (out/"collection.json").write_text(json.dumps(snapshot,indent=2)+"\n")
    (out/"collecting.json").unlink()
    print(json.dumps({"output":str(out),"small_files":len(snapshot["selected_files"]),
                      "copied_bytes":sum(x["bytes"] for x in snapshot["selected_files"]),
                      "tensors_hashed_not_copied":len(snapshot["tensor_files"]),
                      "tensor_bytes_hashed":sum(x["bytes"] for x in snapshot["tensor_files"]),
                      "exclusions":snapshot["excluded"]}))
if __name__=="__main__":
    main()
