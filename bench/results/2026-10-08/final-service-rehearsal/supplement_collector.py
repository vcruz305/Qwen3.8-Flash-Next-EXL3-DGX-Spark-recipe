#!/usr/bin/env python3
"""Add sanitized installation/launch and merge provenance to completed R1 evidence."""
from pathlib import Path
import argparse,ast,base64,gzip,hashlib,importlib.util,json,subprocess
B=Path("/home/vcruz/src/qwen-overnight-20261008")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);a=p.parse_args()
    root=a.archive.resolve();record=json.loads((root/"collection.json").read_text())
    assert record["phase"]=="r1-rehearsal" and record["passed"] is True
    spec=importlib.util.spec_from_file_location("service_archive_filter",B/"collect_service_evidence.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    remote=module.FILTER_SOURCE+r"""
from pathlib import Path
import base64,gzip,hashlib,json,datetime as dt
B=Path("/home/cruzspark/qwen-overnight-20261008")
if Path("/proc/477438").exists():raise RuntimeError("Rehearsal controller still exists")
paths=[
 ("results/service-install-final/result.json","installation/result.json","json"),
 ("results/service-install-final/setup-check.log","installation/setup-check.lifecycle.log","log"),
 ("results/service-rehearsal-final-launch.json","launch/result.json","json"),
 ("logs/service-rehearsal-final.log","launch/controller.lifecycle.log","log")]
files=[]
for name,destination,kind in paths:
 p=B/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size>2_000_000:raise RuntimeError("Unexpected installation/launch artifact")
 st=p.stat();raw=p.read_bytes();after=p.stat()
 if (st.st_size,st.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError("Incomplete evidence changed during read")
 safe,transform=lifecycle_excerpt(raw) if kind=="log" else sanitize_json(raw)
 files.append({"source_path":str(p),"archive_path":destination,
  "source_sha256":hashlib.sha256(raw).hexdigest(),"source_bytes":len(raw),
  "archived_sha256":hashlib.sha256(safe).hexdigest(),"archived_bytes":len(safe),
  "transform":transform,"data":base64.b64encode(safe).decode()})
print(base64.b64encode(gzip.compress(json.dumps({"collected_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
 "controller_pid":477438,"controller_pid_absent":True,"files":files}).encode())).decode())
"""
    result=subprocess.run(["bash",str(B/"spark-ssh"),"python3","-"],input=remote,text=True,capture_output=True,check=True,timeout=30)
    payload=json.loads(gzip.decompress(base64.b64decode(result.stdout.strip())))
    for entry in payload["files"]:
        raw=base64.b64decode(entry.pop("data"));assert hashlib.sha256(raw).hexdigest()==entry["archived_sha256"]
        target=root/entry["archive_path"];target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists();target.write_bytes(raw)
    (root/"installation-collection.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    proof=B/"service-evidence-inputs/fork-promotion-result.json"
    assert sha(proof)=="5d30a2fd56cb71d56bc7aad6a432cebe5b77ac599e0b07efffdc82b3a00d17d6"
    value=json.loads(proof.read_text());assert value["passed"] and value["state"]=="completed"
    assert all(v["tree_identical"] and v["qualified_tree"]==v["merged_tree"] for v in value["items"])
    target=root/"promotion";target.mkdir(exist_ok=True)
    (target/proof.name).write_bytes(proof.read_bytes())
    for name in ("service-evidence-sanitizer-cpu.json","service-evidence-collector-sanitizer-provenance.json"):
        source=B/name
        if source.exists():(root/name).write_bytes(source.read_bytes())
    (root/"supplement_collector.py").write_bytes(Path(__file__).read_bytes())
    paths=sorted(p for p in root.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    manifest="".join(sha(p)+"  "+str(p.relative_to(root))+"\n" for p in paths)
    (root/"SHA256SUMS").write_text(manifest)
    for line in manifest.splitlines():
        expected,rel=line.split("  ",1);assert sha(root/rel)==expected
    print(json.dumps({"archive":str(root),"files":len(paths)+1,
        "bytes":sum(p.stat().st_size for p in paths)+len(manifest.encode()),
        "manifest_sha256":sha(root/"SHA256SUMS"),
        "logs_transferred_as_lifecycle_excerpts_only":True},indent=2))
if __name__=="__main__":main()
