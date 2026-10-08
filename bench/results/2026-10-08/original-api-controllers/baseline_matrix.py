#!/usr/bin/env python3
"""Temporary sequential Spark baseline controller; owns only its model child."""
import datetime as dt
import json, os, signal, subprocess, time, urllib.request
from pathlib import Path

ROOT=Path.home()/"qwen-overnight-20261008"
RECIPE=ROOT/"recipe"
MODELS=[("405","flashnext-exl3-4.05bpw"),("415","flashnext-exl3-sage-4.15bpw"),("cyber387","CYBER-FROST-3.8-EXL3-SAGE-3.87bpw")]
STATUS={"started_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"models":{}}
def save():
    p=ROOT/"results/baseline-matrix-status.json"
    p.write_text(json.dumps(STATUS,indent=2)+"\n")
def say(s):
    print(dt.datetime.now(dt.timezone.utc).isoformat(),s,flush=True)
def stop_pid(pid, expected):
    try:
        cmd=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
    except FileNotFoundError:return
    if expected not in cmd:raise RuntimeError(f"refusing unexpected PID {pid}: {cmd}")
    os.kill(pid,signal.SIGTERM)
    for _ in range(80):
        if not Path(f"/proc/{pid}").exists():return
        time.sleep(.25)
    os.kill(pid,signal.SIGKILL)
def run_client(args, logname, timeout=1800):
    with (ROOT/"logs"/logname).open("w") as log:
        r=subprocess.run(["python3",str(RECIPE/"bench"/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    return r.returncode
save()
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    p=ROOT/"results/baseline-305-tools.json"
    if p.exists() and json.loads(p.read_text()).get("completed_at_utc"):break
    time.sleep(5)
else:raise RuntimeError("3.05 initial tests did not complete")
stop_pid(int((ROOT/"baseline.pid").read_text()),"qwen38-exl3/tabbyAPI/main.py")
time.sleep(2)
for key,name in MODELS:
    say(f"Loading baseline {name}")
    state=ROOT/f"state-baseline-{key}"
    env=dict(os.environ,PROFILE="single",NGRAM_RAM="false",MODEL_DIR=str(Path.home()/"models"/name),STATE_DIR=str(state),PYTHONUNBUFFERED="1")
    log=(ROOT/"logs"/f"baseline-{key}-start.log").open("w")
    process=subprocess.Popen(["bash",str(RECIPE/"exllamav3-tabby/serve.sh")],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/"active-matrix.pid").write_text(str(process.pid))
    record={"model":name,"ngram_ram":False,"profile":"single","max_seq_len":262144,"cache_size":262144,"max_batch_size":1,"pid":process.pid}
    STATUS["models"][key]=record;save()
    try:
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError(f"server exited {process.returncode}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8899/v1/models",timeout=2) as r:
                    if json.load(r).get("data"):break
            except Exception:pass
            time.sleep(2)
        else:raise RuntimeError("server readiness timeout")
        record["ready_at_utc"]=dt.datetime.now(dt.timezone.utc).isoformat();save()
        meta=json.loads((ROOT/"results/environment-baseline.json").read_text())
        meta.update(record)
        import hashlib
        model=Path.home()/"models"/name
        meta["model_config_sha256"]=hashlib.sha256((model/"config.json").read_bytes()).hexdigest()
        meta["launch_config"]=(state/"config.yml").read_text()
        meta["model_safetensors_bytes"]=sum(p.stat().st_size for p in model.glob("*.safetensors"))
        mp=ROOT/"results"/f"baseline-{key}-environment.json";mp.write_text(json.dumps(meta,indent=2)+"\n")
        common=["--model","Qwen3.8-Flash-Next-EXL3","--response-model",name]
        record["bench_exit"]=run_client(["bench_v1.py",*common,"--suite","all","--repeat","3","--warmup","1","--run-id","overnight-v1","--label",f"baseline-{key}-single-disk","--metadata",str(mp),"--output",str(ROOT/"results"/f"baseline-{key}.json")],f"baseline-{key}-bench.log")
        save()
        record["prefill_exit"]=run_client(["bench_v1.py",*common,"--suite","code","--context-tokens","16384","--max-tokens","256","--repeat","2","--warmup","1","--run-id","overnight-prefill","--label",f"baseline-{key}-16k","--metadata",str(mp),"--output",str(ROOT/"results"/f"baseline-{key}-prefill.json")],f"baseline-{key}-prefill.log")
        save()
        record["tools_exit"]=run_client(["tool_smoke.py",*common,"--case","auto,no_args,strings,typed,round_trip","--mode","both","--output",str(ROOT/"results"/f"baseline-{key}-tools.json")],f"baseline-{key}-tools.log")
        record["completed_at_utc"]=dt.datetime.now(dt.timezone.utc).isoformat();save()
        say(f"Finished {name}: benchmark={record['bench_exit']} tools={record['tools_exit']}")
    except Exception as exc:
        record["error"]=str(exc);save();say(f"FAILED {name}: {exc}")
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
        log.close()
        time.sleep(2)
STATUS["completed_at_utc"]=dt.datetime.now(dt.timezone.utc).isoformat();save()
say("Baseline matrix finished; GPU released")
