#!/usr/bin/env python3
"""Collect small completed runtime-setup evidence; read-only on Spark."""
from pathlib import Path
import base64, datetime as dt, gzip, hashlib, json, subprocess

B=Path("/home/vcruz/src/qwen-overnight-20261008")
A=B/"recipe/bench/results/2026-10-08/final-runtime-setup"
if A.exists(): raise SystemExit("Refusing existing archive")
remote=r'''
from pathlib import Path
import base64,datetime as dt,gzip,hashlib,json
root=Path("/home/cruzspark/qwen-overnight-20261008")
runtime=Path("/home/cruzspark/qwen38-exl3-20261008")
files=[]
def add(p,rel):
    if p.is_symlink() or not p.is_file():raise RuntimeError("Not regular: "+str(p))
    before=p.stat()
    if before.st_size>1_000_000:raise RuntimeError("Unexpectedly large setup evidence")
    raw=p.read_bytes();after=p.stat()
    if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):raise RuntimeError("Changing evidence: "+str(p))
    files.append({"archive_path":rel,"source_path":str(p),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
                  "mtime_ns":after.st_mtime_ns,"data":base64.b64encode(raw).decode()})
for folder,rel in [("setup-final-24f0-5a","setup"),("qualified-transition-24f0-5a","transition")]:
    p=root/"results"/folder
    for q in sorted(p.rglob("*")):
        if q.is_file():
            if q.suffix not in (".json",".log"):raise RuntimeError("Unexpected evidence type")
            add(q,rel+"/"+str(q.relative_to(p)))
launch=root/"results/qualified-transition-24f0-5a-launch.json"
add(launch,"transition/launch.json")
launcher_log=Path(json.loads(launch.read_text())["log"])
if not launcher_log.is_relative_to(root/"logs"):raise RuntimeError("Unexpected launch log path")
add(launcher_log,"transition/launcher.log")
for name in ["qualified_transition.py","setup_qualified_runtime.py"]:
    add(root/name,"scripts/"+name)
for p,rel in [
    (runtime/"venv/.qwen38-recipe-runtime.json","runtime/build-state.json"),
    (runtime/"state/runtime-build-desired.json","runtime/desired-build-state.json"),
    (runtime/"state/setup-snapshot.json","runtime/setup-snapshot.json"),
    (runtime/"state/packages.json","runtime/packages.json")]:
    add(p,rel)
if sum(f["bytes"] for f in files)>2_000_000:raise RuntimeError("Unexpected collection size")
payload={"collected_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"scope":"Completed setup/transition small files only; no model, tensor, Git bundle or compiled extension bytes read.","files":files}
print(base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())
'''
result=subprocess.run(["bash",str(B/"spark-ssh"),"python3","-"],input=remote,text=True,capture_output=True,check=True,timeout=40)
payload=json.loads(gzip.decompress(base64.b64decode(result.stdout.strip())))
by={f["archive_path"]:f for f in payload["files"]}
decode=lambda name:json.loads(base64.b64decode(by[name]["data"]))
setup=decode("setup/status.json");transition=decode("transition/status.json")
engine="24f0dece34f09c8d1e2359d6b3b3f7befef7331b"
tabby="5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb"
recipe="3337bc8d64e4befafa2a1aff342e07abbea242ac"
assert setup["state"]=="completed" and setup["passed"] is True and setup["finished_utc"]
assert setup["engine"]==engine and setup["tabby"]==tabby and setup["recipe_commit"]==recipe
assert {c["name"] for c in setup["commands"]}=={"setup","setup-check","engine-budget-cpu","tabby-cpu","tool-tokenizer"}
assert all(c["exit_code"]==0 and c["finished_utc"] for c in setup["commands"])
assert transition["state"]=="completed" and transition["passed"] is True and transition["finished_utc"]
assert transition["quality_gate_review_required"] is True
assert by["scripts/qualified_transition.py"]["sha256"]==transition["script_sha256"]=="a0d818982a95a85d9944f11ecc4da5c2b2b318e38cb05175b392e0a4c23ce2db"
assert by["scripts/setup_qualified_runtime.py"]["sha256"]==setup["controller_sha256"]=="fdad4e0244a0a7a45440a74e44507808115785cdaae47b220c7dee2f3a002d5e"
assert decode("runtime/build-state.json")==setup["build_state"]==decode("runtime/desired-build-state.json")
for section,data in [("setup",setup),("transition",transition)]:
    for cmd in data["commands"]:
        assert cmd["exit_code"]==0 and cmd["finished_utc"]
        assert by[section+"/"+Path(cmd["log"]).name]["sha256"]==cmd["log_sha256"]
review=B/"qualified-transition-source-review.json"
review_data=json.loads(review.read_text())
assert review_data["status"]=="resolved" and review_data["source_sha256"]==transition["script_sha256"]
A.mkdir(parents=True,mode=0o700)
for f in payload["files"]:
    raw=base64.b64decode(f.pop("data"))
    assert len(raw)==f["bytes"] and hashlib.sha256(raw).hexdigest()==f["sha256"]
    p=A/f["archive_path"];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
review_raw=review.read_bytes()
(A/"scripts/qualified-transition-source-review.json").write_bytes(review_raw)
payload["local_source_review"]={"source_path":str(review),"archive_path":"scripts/qualified-transition-source-review.json",
    "bytes":len(review_raw),"sha256":hashlib.sha256(review_raw).hexdigest()}
(A/"source-provenance.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"archive":str(A),"remote_files":len(payload["files"]),"remote_bytes":sum(f["bytes"] for f in payload["files"]),
                  "setup_completed_utc":setup["finished_utc"],"quality_gate_review_required":True},indent=2))
