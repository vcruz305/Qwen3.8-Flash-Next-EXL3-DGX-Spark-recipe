#!/usr/bin/env python3
"""Launch the frozen final API batch only after the paired serving-default numerical gate passes."""
from pathlib import Path
import datetime as dt, hashlib, json, os, sys, time, traceback
ROOT=Path("/home/cruzspark/qwen-overnight-20261008")
OUT=ROOT/"results/final-api-handoff-24f0-5a"
PAIR=ROOT/"results/quality-final-chunk2048-pair/status.json"
FILES={
 "quality-chunk2048-final-pair-jobs.json":"eefa1312ea922c63fd083553d8ed1d6cae210fd1f02d82bcc4a47a7c10d2829a",
 "final-api-jobs-selected.json":"50def7795a0a7401cbf17d585d5f0955e23ed99968de58bda3d92c836335b915",
 "final_api_batch.py":"4820f2515b98b9f4698065bc03c393f82cc1c6f5aa9542d6d8599761478a1117",
 "final_api_controller.py":"ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d",
 "reasoning_literal_smoke.py":"8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054",
 "reasoning_concurrency_smoke.py":"1e88c70eb82fa180b97d52b3ee71ad998153c24c99c2827e6da9646e1bbc102f",
 "spark_experiment_controller.py":"48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586",
 "api_f4_gemm_controller.py":"78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589"}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def main():
 OUT.mkdir(mode=0o700)
 record={"state":"waiting_pair","passed":False,"started_utc":now(),"script_sha256":sha(Path(__file__)),"after_pid":426966}
 def save(state=None,**kw):
  if state:record["state"]=state
  record.update(updated_utc=now(),**kw)
  p=OUT/"status.json.tmp";p.write_text(json.dumps(record,indent=2)+"\n");p.replace(OUT/"status.json")
 save()
 try:
  deadline=time.monotonic()+7200
  while Path("/proc/426966").exists():
   if time.monotonic()>deadline:raise TimeoutError("Paired quality process did not finish")
   time.sleep(2)
  for name,digest in FILES.items():
   if sha(ROOT/name)!=digest:raise ValueError("Frozen input changed: "+name)
  pair=json.loads(PAIR.read_text())
  if pair.get("state")!="completed" or pair.get("jobs_sha256")!=FILES["quality-chunk2048-final-pair-jobs.json"]:raise ValueError("Numerical pair did not complete with frozen jobs")
  rows=pair.get("jobs",[])
  names=["baseline-305-chunk2048-final-pair","candidate-305-chunk2048-final-pair"]
  if len(rows)!=2 or [x.get("name") for x in rows]!=names:raise ValueError("Unexpected quality jobs")
  if not all(x.get("completed") is True and x.get("probe_exit")==0 for x in rows):raise ValueError("A paired probe did not complete")
  candidate=rows[1]
  if candidate.get("assessment_exit")!=0 or candidate.get("core_passed") is not True or candidate.get("investigation_count")!=0:raise ValueError("Serving-default numerical gate needs investigation")
  setup=ROOT/"results/setup-final-24f0-5a/status.json"
  s=json.loads(setup.read_text())
  if s.get("passed") is not True or s.get("state")!="completed" or s.get("engine")!="24f0dece34f09c8d1e2359d6b3b3f7befef7331b" or s.get("tabby")!="5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb":raise ValueError("Setup qualification changed")
  destination=ROOT/"results/final-api-24f0-5a"
  if destination.exists():raise ValueError("Final API output already exists")
  command=[sys.executable,str(ROOT/"final_api_batch.py"),"--jobs",str(ROOT/"final-api-jobs-selected.json"),
   "--setup",str(setup),"--output",str(destination),"--job-timeout","3600"]
  save("handed_to_final_api",passed=True,finished_utc=now(),pair_status_sha256=sha(PAIR),setup_status_sha256=sha(setup),frozen_inputs=FILES,command=command,api_results=str(destination))
  os.execv(sys.executable,command)
 except BaseException as exc:
  save("failed",passed=False,error=type(exc).__name__+": "+str(exc),finished_utc=now());traceback.print_exc();return 1
if __name__=="__main__":raise SystemExit(main())
