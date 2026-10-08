#!/usr/bin/env python3
from pathlib import Path
import datetime as dt,hashlib,json,os,signal,socket,subprocess,time,urllib.request,traceback
ROOT=Path("/home/cruzspark/qwen-overnight-20261008")
RECIPE=ROOT/"recipe"
RUNTIME=Path("/home/cruzspark/qwen38-exl3")
STATE=ROOT/"state-baseline-305-prefill"
STATUS=ROOT/"results/baseline-305-prefill-status.json"
server=None;record={}
def save(state,**extra):
 record.update(state=state,recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**extra)
 t=STATUS.with_suffix(".json.tmp");t.write_text(json.dumps(record,indent=2)+"\n");t.replace(STATUS)
try:
 assert not STATUS.exists() and not (ROOT/"results/baseline-305-prefill.json").exists()
 with socket.socket() as s:assert s.connect_ex(("127.0.0.1",8899))!=0
 env={k:v for k,v in os.environ.items() if not k.startswith("EXL3_")}
 env.update(RECIPE_HOME=str(RUNTIME),STATE_DIR=str(STATE),PROFILE="single",NGRAM_RAM="true",
  MODEL_DIR="/home/cruzspark/models/flashnext-exl3-3.05bpw",HOST="127.0.0.1",PORT="8899",
  DISABLE_AUTH="true",EXL3_INT8_GEMV="0",EXL3_GR_INT8="1",EXL3_MOE_COOP_WIDE="1",
  EXL3_MTP_HEAD_N="65536",EXL3_DRAFT_CONFIDENCE="0.6",PYTHONUNBUFFERED="1")
 for repo,head in [("exllamav3","94ba01d50a13fa9ff672473f2d0eef8b51a71e99"),("tabbyAPI","816c32195887aaecea1c64528f2921566766259b")]:
  assert subprocess.check_output(["git","-C",str(RUNTIME/repo),"rev-parse","HEAD"],text=True).strip()==head
 with open(ROOT/"logs/baseline-305-prefill-server.log","x") as log:
  server=subprocess.Popen(["bash",str(RECIPE/"exllamav3-tabby/serve.sh")],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=RECIPE)
 save("loading",server_pid=server.pid)
 began=time.monotonic()
 for _ in range(600):
  if server.poll() is not None:raise RuntimeError("Original server exited before ready")
  try:
   with urllib.request.urlopen("http://127.0.0.1:8899/v1/models",timeout=2) as resp:
    models=json.load(resp)
   if models.get("data"):break
  except Exception:pass
  time.sleep(1)
 else:raise TimeoutError("Original readiness timeout")
 meta=json.loads((ROOT/"results/environment-baseline.json").read_text())
 meta.update(model="flashnext-exl3-3.05bpw",ngram_ram=True,profile="single",max_seq_len=262144,cache_size=262144,max_batch_size=1,
  launch_config=(STATE/"config.yml").read_text(),recorded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),server_pid=server.pid)
 meta["model_config_sha256"]=hashlib.sha256(Path(env["MODEL_DIR"],"config.json").read_bytes()).hexdigest()
 mp=ROOT/"results/baseline-305-prefill-environment.json";mp.write_text(json.dumps(meta,indent=2)+"\n")
 save("measuring",load_wall_seconds=time.monotonic()-began,models=models)
 with open(ROOT/"logs/baseline-305-prefill.log","x") as log:
  p=subprocess.run(["python3",str(RECIPE/"bench/bench_v1.py"),"--model","Qwen3.8-Flash-Next-EXL3","--response-model","flashnext-exl3-3.05bpw",
   "--suite","code","--context-tokens","16384","--max-tokens","256","--repeat","2","--warmup","1","--run-id","overnight-prefill",
   "--label","baseline-305-16k","--metadata",str(mp),"--output",str(ROOT/"results/baseline-305-prefill.json")],env=env,cwd=RECIPE,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 save("measured",client_exit=p.returncode)
 if p.returncode:raise RuntimeError("Baseline prefill failed")
except Exception as exc:
 save("failed",error=f"{type(exc).__name__}: {exc}");traceback.print_exc()
finally:
 if server is not None and server.poll() is None:
  os.killpg(server.pid,signal.SIGTERM)
  try:server.wait(timeout=30)
  except subprocess.TimeoutExpired:
   os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=5)
 record["server_cleanup_exit"]=None if server is None else server.poll()
 save("completed" if record.get("state")=="measured" else record.get("state","failed"))
