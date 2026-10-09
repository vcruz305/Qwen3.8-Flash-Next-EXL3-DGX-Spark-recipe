#!/usr/bin/env python3
"""Collect completed service evidence, sanitizing all logs on Spark before transfer.

Preparation only. This collector never starts/stops a unit, makes API requests,
reads credential stores, or executes the archived rehearsal/verifier.
"""
from pathlib import Path
import argparse
import base64
import datetime as dt
import gzip
import hashlib
import json
import re
import subprocess

BASE=Path("/home/vcruz/src/qwen-overnight-20261008")
REHEARSE_SHA="efaae01bc5a859948dd7a46d9e9584fd762ccfce143722946865a2aaf692224f"
PUBLISHED_SHA="aaafeb7cebc91d5bb2106b86bbaf28d4572607ddd67af56ec80c133eb587bfc0"
FILTER_SOURCE=r"""
import json,re
CREDENTIAL_KEYS={"apikey","adminkey","password","passwd","authorization","cookie","setcookie",
                 "accesstoken","refreshtoken","clientsecret","credential","credentials","secret"}
CREDENTIAL_TEXT=re.compile(r"(?i)(your\s+(?:api|admin)\s+key|\b(?:api[_ -]?key|admin[_ -]?key|password|passwd|client[_ -]?secret|access[_ -]?token|refresh[_ -]?token)[\"']?\s*(?:is\s*)?[:=]|\bauthorization\s*[:=]|\bbearer\s+\S+)")
ANSI=re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SAFE_LOG=re.compile(
    r"(?:Started server process \[\d+\]|Finished server process \[\d+\]|"
    r"Waiting for application startup\.|Application startup complete\.|"
    r"Waiting for application shutdown\.|Application shutdown complete\.|"
    r"Shutting down|"
    r"Uvicorn running on http://127\.0\.0\.1:8899 \(Press CTRL\+C to quit\)|"
    r"Main process exited, code=(?:killed|exited), status=\d+(?:/[A-Z0-9]+)?|"
    r"Scheduled restart job, restart counter is at \d+\.|"
    r"Consumed [0-9.]+s CPU time(?:, [0-9.]+[KMG] memory peak)?)"
)
def scrub_json(value,path="$"):
    changes=[]
    if isinstance(value,dict):
        out={}
        for key,item in value.items():
            normalized=re.sub(r"[^a-z0-9]","",str(key).lower())
            if normalized in CREDENTIAL_KEYS:
                out[key]="[REDACTED]";changes.append(path+"/"+str(key))
            else:
                out[key],other=scrub_json(item,path+"/"+str(key));changes.extend(other)
        return out,changes
    if isinstance(value,list):
        out=[]
        for i,item in enumerate(value):
            item,other=scrub_json(item,path+"/"+str(i));out.append(item);changes.extend(other)
        return out,changes
    if isinstance(value,str) and CREDENTIAL_TEXT.search(value):
        return "[REDACTED AUTHENTICATION MATERIAL]",[path]
    return value,changes
def sanitize_json(raw):
    value,paths=scrub_json(json.loads(raw))
    return ((json.dumps(value,indent=2,ensure_ascii=False)+"\n").encode() if paths else raw,
            {"kind":"credential-field-scrub","redaction_paths":paths,"redactions":len(paths)})
def sanitize_text(raw):
    lines=raw.decode("utf-8").splitlines(keepends=True);out=[];removed=[]
    for number,line in enumerate(lines,1):
        if CREDENTIAL_TEXT.search(ANSI.sub("",line)):
            out.append("[REDACTED AUTHENTICATION LINE]\n");removed.append(number)
        else:out.append(line)
    return "".join(out).encode(),{"kind":"credential-line-scrub","redacted_lines":removed,"redactions":len(removed)}
def lifecycle_excerpt(raw):
    # Keep only synthesized, narrowly recognized lifecycle messages. This also
    # removes wrapped/unlabelled authentication-key continuation lines.
    lines=raw.decode("utf-8",errors="replace").splitlines();out=[];matched=0
    credential_markers=0
    for number,line in enumerate(lines,1):
        clean=ANSI.sub("",line)
        if CREDENTIAL_TEXT.search(clean):
            credential_markers+=1;continue
        found=SAFE_LOG.search(clean)
        if found:
            out.append(f"[source line {number}] {found.group(0)}\n");matched+=1
    header=("# Sanitized lifecycle-only excerpt; filtering occurred on Spark before transfer.\n"
            "# All other original log lines were omitted, including authentication output.\n")
    return (header+"".join(out)).encode(),{
        "kind":"lifecycle-only-excerpt","source_lines":len(lines),"retained_lines":matched,
        "omitted_lines":len(lines)-matched,"credential_label_lines_omitted":credential_markers,
        "raw_log_transferred":False}
"""

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("r1-rehearsal","r3-published"),required=True)
    parser.add_argument("--source-dir",required=True)
    parser.add_argument("--expected-recipe",required=True)
    parser.add_argument("--expected-engine",required=True)
    parser.add_argument("--expected-tabby",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    for key in ("expected_recipe","expected_engine","expected_tabby"):
        if not re.fullmatch("[0-9a-f]{40}",getattr(args,key)):parser.error("Full exact expected source SHAs required")
    if not args.source_dir.startswith("/home/cruzspark/qwen-overnight-20261008/results/"):
        parser.error("Use the recorded service result directory under the Spark work root")
    args.output=args.output.resolve()
    if args.output.exists():parser.error("Use a new archive directory; existing evidence is never replaced")
    expected={"recipe":args.expected_recipe,"engine":args.expected_engine,"server":args.expected_tabby}
    settings={"phase":args.phase,"source_dir":args.source_dir,"expected":expected,
              "controller_sha":REHEARSE_SHA if args.phase=="r1-rehearsal" else PUBLISHED_SHA}
    remote=FILTER_SOURCE+r"""
from pathlib import Path
import base64,datetime as dt,gzip,hashlib,subprocess
S=__SETTINGS__
ROOT=Path("/home/cruzspark/qwen-overnight-20261008")
HOME=Path("/home/cruzspark")
STATE=HOME/".local/state/qwen38-exl3/runs"
D=Path(S["source_dir"]).resolve(strict=True)
if not D.is_relative_to(ROOT/"results"):raise RuntimeError("Result directory escaped work root")
def read(path,limit=2_000_000):
    if path.is_symlink() or not path.is_file():raise RuntimeError("Not a regular evidence file: "+str(path))
    before=path.stat()
    if before.st_size>limit:raise RuntimeError("Evidence exceeds small-file bound: "+str(path))
    raw=path.read_bytes();after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise RuntimeError("Evidence changed during collection: "+str(path))
    return raw,after
def sha(raw):return hashlib.sha256(raw).hexdigest()
result_raw,_=read(D/"result.json")
result=json.loads(result_raw)
if not result.get("finished_at_utc"):raise RuntimeError("Service evidence has not completed")
if result.get("controller_sha256")!=S["controller_sha"]:raise RuntimeError("Service controller differs from reviewed source")
sources=result.get("sources",{})
if not all(sources.get(key,{}).get("commit")==value for key,value in S["expected"].items()):
    raise RuntimeError("Service result does not identify the expected phase/source revisions")
files=[]
def publish(raw,relative,source,stat=None,kind="json"):
    original_sha=sha(raw)
    if kind=="log":safe,transform=lifecycle_excerpt(raw)
    elif kind=="json":safe,transform=sanitize_json(raw)
    elif kind=="text":safe,transform=sanitize_text(raw)
    elif kind=="source":safe,transform=raw,{"kind":"public-source-snapshot","redactions":0}
    else:raise RuntimeError("Unknown publication transform")
    files.append({"archive_path":relative,"source_path":source,"source_bytes":len(raw),
        "source_sha256":original_sha,"source_mtime_ns":None if stat is None else stat.st_mtime_ns,
        "source_mode":None if stat is None else oct(stat.st_mode&0o777),
        "archived_bytes":len(safe),"archived_sha256":sha(safe),"transform":transform,
        "data":base64.b64encode(safe).decode()})
def add(path,relative,kind="json",expected=None):
    raw,stat=read(path)
    if expected is not None and sha(raw)!=expected:raise RuntimeError("Recorded evidence hash mismatch: "+str(path))
    publish(raw,relative,str(path),stat,kind)
add(D/"result.json","reports/result.json")
phases=("01-initial","02-auto-restarted","04-returned-to-final") if S["phase"]=="r1-rehearsal" else ("published-main",)
phase_records=[]
for phase in phases:
    path=D/phase/"record.json"
    if not path.exists():
        if result.get("passed"):raise RuntimeError("Successful phase lacks record: "+phase)
        continue
    record=json.loads(read(path)[0]);phase_records.append(record)
    for key,value in S["expected"].items():
        if record["deployment"][key]["commit"]!=value:raise RuntimeError("Per-invocation source mismatch")
    invocation=record["unit"]["InvocationID"]
    if not re.fullmatch("[0-9a-f]{32}",invocation):raise RuntimeError("Invalid invocation ID")
    state=Path(record["state_dir"])
    if state!=STATE/invocation:raise RuntimeError("Unexpected per-start state path")
    add(path,"reports/"+phase+"/record.json")
    add(D/phase/"config.yml","reports/"+phase+"/config.yml","text",record["deployment"]["config"]["sha256"])
    if (D/phase/"journal.log").exists():add(D/phase/"journal.log","reports/"+phase+"/journal.lifecycle.log","log")
    for name in ("start.json","deployment.json","config.yml"):
        add(state/name,"invocations/"+invocation+"/"+name,"text" if name.endswith(".yml") else "json")
if S["phase"]=="r1-rehearsal":
    for name,kind,public_name in (
        ("03-original-api.json","json","03-original-api.json"),
        ("03-original-server.log","log","03-original-server.lifecycle.log"),
        ("old-state/config.yml","text","old-state/config.yml")):
        path=D/name
        if path.exists():add(path,"reports/"+public_name,kind)
    if result.get("passed"):
        required={"initial","automatic_restart","original_basic_api","returned_to_final"}
        if set(result["phases"])!=required:raise RuntimeError("Successful rehearsal lacks required phases")
        cleanup=result.get("old_cleanup",{})
        # The frozen stop_old returns exit_code/signals only; its source checks
        # that the token-verified group is empty before returning.
        if (type(cleanup.get("exit_code")) is not int or
            not isinstance(cleanup.get("signals"),list) or
            any(v not in ("SIGTERM","SIGKILL") for v in cleanup["signals"])):
            raise RuntimeError("Missing reviewed original cleanup result")
        initial=result["phases"]["initial"];restart=result["phases"]["automatic_restart"]
        if (initial["InvocationID"]==restart["InvocationID"] or initial["MainPID"]==restart["MainPID"] or
            int(restart["NRestarts"])!=int(initial["NRestarts"])+1):
            raise RuntimeError("Automatic restart did not match the declared gate")
else:
    if (D/"tools.json").exists():add(D/"tools.json","reports/tools.json")
    if (D/"tools.log").exists():add(D/"tools.log","reports/tools.lifecycle.log","log")
    if result.get("passed"):
        tool=json.loads(read(D/"tools.json")[0]);checks=tool.get("results",[])
        if (len(checks)!=28 or not all(v.get("passed") is True for v in checks) or
            tool.get("summary")!={"passed":28,"failed":0,"total":28} or
            not tool.get("completed_at_utc") or tool.get("errors")):
            raise RuntimeError("Published success differs from the raw28 tool checks")
        before=result["unit_before_tools"];after=result["unit_after_tools"]
        if any(before[k]!=after[k] for k in ("MainPID","InvocationID","NRestarts")):
            raise RuntimeError("Published service changed during the verified gate")
env=sources["service_environment"]
add(Path(env["path"]),"installed/service.env","text",env["sha256"])
for source,relative in (
    (HOME/".config/systemd/user/qwen38-exl3.service","installed/qwen38-exl3.service"),
    (HOME/".config/systemd/user/qwen38-exl3.service.d/10-invocation-state.conf","installed/10-invocation-state.conf"),
    (HOME/".local/libexec/qwen38-exl3-start.sh","installed/qwen38-exl3-start.sh")):
    add(source,relative,"text")
for name in ("rehearse.py","plan.json","qwen38-exl3.service.example","10-invocation-state.conf","qwen38-exl3-start.sh"):
    add(ROOT/"service-promotion"/name,"source/service-promotion/"+name,
        "json" if name.endswith(".json") else "source",
        "__REHEARSE_SHA__" if name=="rehearse.py" else None)
add(ROOT/"spark_experiment_controller.py","source/spark_experiment_controller.py","source",
    "48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586")
if S["phase"]=="r3-published":
    add(ROOT/"verify_published_service.py","source/verify_published_service.py","source",S["controller_sha"])
recipe=Path(sources["recipe"]["path"]);head=sources["recipe"]["commit"]
for relative in ("bench/api_client.py","bench/tool_smoke.py","exllamav3-tabby/env.sh",
                 "exllamav3-tabby/serve.sh","exllamav3-tabby/tabby-config.yml"):
    raw=subprocess.check_output(["git","-C",str(recipe),"show",head+":"+relative],timeout=10)
    if len(raw)>500_000:raise RuntimeError("Unexpectedly large source snapshot")
    publish(raw,"source/recipe/"+relative,"git:"+str(recipe)+"@"+head+":"+relative,kind="source")
if sum(v["archived_bytes"] for v in files)>12_000_000:raise RuntimeError("Unexpected service evidence volume")
payload={"collected_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"phase":S["phase"],
         "source_dir":str(D),"expected":S["expected"],"passed":result.get("passed"),
         "completed_at_utc":result["finished_at_utc"],"files":files,
         "publication_policy":"All logs filtered into conservative lifecycle-only excerpts on Spark before transfer; credential stores never read.",
         "result_counts":{"new_service_invocations":len(phase_records),
             "new_service_basic_api_requests":sum(len(v["api"]["requests"]) for v in phase_records),
             "original_runtime_basic_api_requests":len(json.loads(read(D/"03-original-api.json")[0])["api"]["requests"])
                 if S["phase"]=="r1-rehearsal" and (D/"03-original-api.json").exists() else 0,
             "tool_checks":len(json.loads(read(D/"tools.json")[0]).get("results",[]))
                 if S["phase"]=="r3-published" and (D/"tools.json").exists() else None}}
print(base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())
"""
    remote=remote.replace("__SETTINGS__",repr(settings)).replace("__REHEARSE_SHA__",REHEARSE_SHA)
    run=subprocess.run(["bash",str(BASE/"spark-ssh"),"python3","-"],input=remote,text=True,
                       capture_output=True,check=True,timeout=60)
    payload=json.loads(gzip.decompress(base64.b64decode(run.stdout.strip())))
    args.output.mkdir(parents=True,exist_ok=False)
    args.output.chmod(0o700)
    for entry in payload["files"]:
        raw=base64.b64decode(entry.pop("data"))
        assert len(raw)==entry["archived_bytes"] and hashlib.sha256(raw).hexdigest()==entry["archived_sha256"]
        target=args.output/entry["archive_path"];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    (args.output/"collection.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    (args.output/"collector.py").write_bytes(Path(__file__).read_bytes())
    note=("# Service evidence: "+args.phase+"\n\n"
          "This phase is complete; actual outcome: **"+("passed" if payload["passed"] else "failed")+"**.\n\n"
          "R1 rehearsal and R3 published-service verification are separate source-bound phases. "
          "The R1 result must not be relabelled as an R3 test. Later publication requires recorded "
          "runtime/client byte identity plus a normal restart and verification against the actual R3 revision.\n\n"
          "All logs are conservative lifecycle excerpts produced on Spark before transfer. Raw original "
          "rollback/authentication logs were not copied. collection.json records original and transformed "
          "hashes, byte counts, omitted-line counts and any structured redactions. Credential stores were "
          "never read. Source snapshots come from the recorded recipe commit, even if the checkout later moves.\n\n"
          "Review reports/result.json and each per-invocation record for the exact MainPID, invocation ID, "
          "restart count, environment hash, source revisions and actual API outcomes. These are availability, "
          "restart, rollback and tool-contract checks, with no new throughput claim.\n")
    (args.output/"README.md").write_text(note)
    files=sorted(p for p in args.output.rglob("*") if p.is_file())
    manifest="".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(p.relative_to(args.output))+"\n" for p in files)
    (args.output/"SHA256SUMS").write_text(manifest)
    print(json.dumps({"output":str(args.output),"phase":args.phase,"passed":payload["passed"],
        "files":len(files)+1,"bytes":sum(p.stat().st_size for p in files)+len(manifest.encode()),
        "sha256_manifest":hashlib.sha256(manifest.encode()).hexdigest(),
        "raw_logs_transferred":False,"source_refs":payload["expected"]},indent=2))

if __name__=="__main__":
    main()
