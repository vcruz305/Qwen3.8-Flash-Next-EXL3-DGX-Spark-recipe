#!/usr/bin/env python3
"""Wait for frozen measurements, install exact qualified sources, then run existing setup and quality controllers."""
from pathlib import Path
import datetime as dt, hashlib, json, os, socket, subprocess, time, traceback
ROOT=Path("/home/cruzspark/qwen-overnight-20261008")
RUNTIME=Path("/home/cruzspark/qwen38-exl3-20261008")
RECIPE=Path("/home/cruzspark/qwen-spark-recipe")
ENGINE="24f0dece34f09c8d1e2359d6b3b3f7befef7331b"
TABBY="5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb"
R1="3337bc8d64e4befafa2a1aff342e07abbea242ac"
OUT=ROOT/"results/qualified-transition-24f0-5a"
SETUP=ROOT/"results/setup-final-24f0-5a"
QUALITY=ROOT/"results/quality-final-chunk4096"
FILES={
 "engine-24f0dece-from-16ca.bundle":"1c39a9af0579cf77d7bf351cc7b13b6c0e9595349ed59ae466cd4dc441c1c6c0",
 "tabby-5a4f3ef.bundle":"7b3bd00e66889b3c83a8f3490d184324699e5817272f3e02b031110f227e282f",
 "recipe-qualification-3337bc8.bundle":"664b066b335aa3ad0da91b6105e7537bcb1a50bee83a19c3bccddb0c9ef58ba9",
 "setup_qualified_runtime.py":"fdad4e0244a0a7a45440a74e44507808115785cdaae47b220c7dee2f3a002d5e",
 "spark_quality_q8_matrix.py":"2ebe476c6520da8a9f9f6558d03b7b9fe2dc6d2429b5918db65e3845b255cc71",
 "quality-chunk4096-final-jobs.json":"25753ea5a2d9e8f6d0c6662c300feef1879c2e241227bc391a047271028b4c9e",
 "experiment-jobs-16ca-ksplit1/primary17.json":"ba8f2d7e9dc24798f13cc7ca4c2b65c96fb7c7fa11290ffc2c85dabe568b1850",
 "experiment-jobs-16ca-ksplit1/optional-selected.json":"90234a6cdd68b2e2a4b260753d4ed31ab19a0aaa81cf3e00f7270b91c97b7c34"}
def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def capture(cmd): return subprocess.check_output(cmd,text=True).strip()
def repo(path,expected):
 head=capture(["git","-C",str(path),"rev-parse","HEAD"])
 dirty=capture(["git","-C",str(path),"status","--porcelain","--untracked-files=no"])
 if head!=expected or dirty: raise ValueError("Source identity/cleanliness changed: "+str(path))
 return {"commit":head,"tree":capture(["git","-C",str(path),"rev-parse","HEAD^{tree}"])}
def main():
 OUT.mkdir(mode=0o700)
 rec={"state":"waiting_measurements","passed":False,"started_utc":stamp(),"script_sha256":sha(Path(__file__)),"commands":[]}
 def save(state=None,**kw):
  if state: rec["state"]=state
  rec.update(updated_utc=stamp(),**kw)
  tmp=OUT/"status.json.tmp";tmp.write_text(json.dumps(rec,indent=2)+"\n");tmp.replace(OUT/"status.json")
 def run(cmd,name,timeout=7200):
  row={"name":name,"command":list(map(str,cmd)),"started_utc":stamp()};rec["commands"].append(row);save(active=name)
  log=OUT/(name+".log")
  with log.open("x") as f: proc=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  row.update(exit_code=proc.returncode,finished_utc=stamp(),log=str(log),log_sha256=sha(log));save()
  if proc.returncode: raise RuntimeError(name+" failed")
 save()
 try:
  for name,digest in FILES.items():
   if sha(ROOT/name)!=digest: raise ValueError("Staged input changed: "+name)
  deadline=time.monotonic()+7200
  while Path("/proc/368715").exists():
   if time.monotonic()>deadline: raise TimeoutError("Optional controller did not finish")
   time.sleep(2)
  if Path("/proc/321237").exists(): raise ValueError("Primary controller PID still exists")
  for name,digest in FILES.items():
   if sha(ROOT/name)!=digest: raise ValueError("Staged input changed during wait: "+name)
  for folder,jobs in [("performance-primary-16ca-ksplit1","primary17.json"),("performance-optional-16ca-ksplit1","optional-selected.json")]:
   labels={j["label"] for j in json.loads((ROOT/"experiment-jobs-16ca-ksplit1"/jobs).read_text())}
   rows={p.parent.name:json.loads(p.read_text()) for p in (ROOT/"results"/folder).glob("*/result.json")}
   if set(rows)!=labels or any(d.get("state")!="completed" or not d.get("finished_at_utc") or d.get("cleanup_error") or d.get("cleanup_failed") for d in rows.values()):
    raise ValueError("Incomplete measurements or uncertain cleanup: "+folder)
  with socket.socket() as s:
   if s.connect_ex(("127.0.0.1",8899))==0: raise ValueError("Port8899 is occupied")
  rec["sources_before"]={
   "engine":repo(RUNTIME/"exllamav3","16ca20d27c0e4cce15a9bbc131e6d047065395b5"),
   "tabby":repo(RUNTIME/"tabbyAPI","f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6"),
   "recipe":repo(ROOT/"recipe-updated","218bd438225243e795383f10ef7a886b06d5d839")}
  if RECIPE.exists(): raise ValueError("Canonical recipe path already exists")
  save("promoting_sources",measurements_finished_utc=stamp())
  run(["git","clone","--no-hardlinks","--no-checkout",ROOT/"recipe-updated",RECIPE],"recipe-clone")
  run(["git","-C",RECIPE,"bundle","verify",ROOT/"recipe-qualification-3337bc8.bundle"],"recipe-bundle-verify")
  run(["git","-C",RECIPE,"fetch",ROOT/"recipe-qualification-3337bc8.bundle","HEAD"],"recipe-fetch")
  run(["git","-C",RECIPE,"checkout","--detach",R1],"recipe-checkout")
  run(["git","-C",RECIPE,"remote","set-url","origin","https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe.git"],"recipe-origin")
  for name,path,bundle,ref in [("engine",RUNTIME/"exllamav3","engine-24f0dece-from-16ca.bundle",ENGINE),("tabby",RUNTIME/"tabbyAPI","tabby-5a4f3ef.bundle",TABBY)]:
   run(["git","-C",path,"bundle","verify",ROOT/bundle],name+"-bundle-verify")
   run(["git","-C",path,"fetch",ROOT/bundle,"HEAD"],name+"-fetch")
   run(["git","-C",path,"checkout","--detach",ref],name+"-checkout")
  rec["sources_selected"]={"engine":repo(RUNTIME/"exllamav3",ENGINE),"tabby":repo(RUNTIME/"tabbyAPI",TABBY),"recipe":repo(RECIPE,R1)}
  save("setup")
  cmd=["python3",ROOT/"setup_qualified_runtime.py","--engine",ENGINE,"--tabby",TABBY,"--recipe",RECIPE,"--output",SETUP,"--after-pid","321237","--after-pid","368715"]
  for folder,jobs in [("performance-primary-16ca-ksplit1","primary17.json"),("performance-optional-16ca-ksplit1","optional-selected.json")]:
   cmd+=["--after-results",ROOT/"results"/folder,"--after-jobs",ROOT/"experiment-jobs-16ca-ksplit1"/jobs]
  for name,digest in FILES.items():
   if sha(ROOT/name)!=digest: raise ValueError("Staged input changed before setup: "+name)
  run(cmd,"qualified-setup",timeout=9600)
  status=json.loads((SETUP/"status.json").read_text())
  if status.get("passed") is not True or status.get("state")!="completed": raise ValueError("Setup did not qualify")
  save("quality")
  for name,digest in FILES.items():
   if sha(ROOT/name)!=digest: raise ValueError("Staged input changed before quality: "+name)
  run(["python3",ROOT/"spark_quality_q8_matrix.py","--jobs",ROOT/"quality-chunk4096-final-jobs.json","--output",QUALITY],"quality-chunk4096",timeout=3600)
  save("completed",passed=True,active=None,finished_utc=stamp(),quality_gate_review_required=True)
 except BaseException as exc:
  save("failed",error=type(exc).__name__+": "+str(exc),finished_utc=stamp())
  traceback.print_exc()
  return 1
 return 0
if __name__=="__main__": raise SystemExit(main())
