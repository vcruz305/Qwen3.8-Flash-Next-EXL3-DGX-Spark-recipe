#!/usr/bin/env python3
"""One-shot authorized diagnostic launch, guarded by final-batch cleanup evidence."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess

ROOT = Path("/home/cruzspark/qwen-overnight-20261008")
BATCH = ROOT / "results/final-api-24f0-5a"
OUTPUT = ROOT / "results/literal-final-305-24f0-5a-p095"
LABEL = "literal-final-305-24f0-5a-p095"
EXPECTED = {
    ROOT/"literal_capture_controller_p095.py": "d101f3ccd82eb94c75733e0c5e561a83ac5f392886a4d4dd1ee4f07d247e263d",
    ROOT/"literal_thinking_off_smoke.py": "789408e144d9046fdbd11cc2868886d5bb14cdc1e8bf365253105109725a7243",
    ROOT/"reasoning_literal_smoke.py": "8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054",
    ROOT/"strings_only_controller.py": "cfc3619894c4ee8f07d46c2ee7ed0fd085f3fdc044c81c8d9a01d009304c4116",
    ROOT/"final_api_controller.py": "ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d",
    ROOT/"api_f4_gemm_controller.py": "78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589",
    ROOT/"spark_experiment_controller.py": "48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586",
    ROOT/"literal-final-observer-p095/strings_observer.py": "674453a31893defd299939fe801cd58b48bdb20c7bbfaf0ec0c0fbb5f82418f6",
    ROOT/"literal-final-observer-p095/sitecustomize.py": "3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f",
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def exited(pid):
    if not isinstance(pid,int) or pid<=1:
        raise ValueError("Missing recorded process identity")
    path=Path(f"/proc/{pid}/stat")
    if not path.exists():
        return "absent"
    state=path.read_text().split(") ",1)[1].split()[0]
    if state=="Z":
        return "exited-zombie"
    raise RuntimeError("Predecessor process is still active: "+str(pid))

def preflight(require_sources=True):
    value=json.loads((BATCH/"status.json").read_text())
    wanted={"final-305-single","final-405-single","final-415-single","final-cyber387-single","final-305-concurrent"}
    assert value.get("state")=="completed" and value.get("finished_utc") and not value.get("active")
    assert not value.get("error") and not value.get("cleanup_error")
    assert len(value.get("jobs",[]))==5 and {r["label"] for r in value["jobs"]}==wanted
    pids={"430098":exited(430098)}
    for path,fingerprint in value["source_files_sha256"].items():
        assert sha(Path(path))==fingerprint,("Changed batch input",path)
    for row in value["jobs"]:
        assert row.get("state")=="completed" and row.get("finished_utc")
        pids[str(row["controller_pid"])]=exited(row["controller_pid"])
        result_path=BATCH/row["label"]/"result.json"
        assert sha(result_path)==row["result_sha256"]
        result=json.loads(result_path.read_text())
        assert result.get("state")=="completed" and result.get("finished_at_utc")
        assert not result.get("cleanup_failed") and not result.get("error")
        cleanup=result.get("server_cleanup",{})
        assert cleanup.get("owned_group_empty") is True and cleanup.get("unexpected_exit") is False
        pids[str(result["server_pid"])]=exited(result["server_pid"])
        for client in result.get("clients",[]):
            assert client.get("finished_at_utc") and client.get("cleanup",{}).get("owned_group_empty") is True
            assert not client.get("timed_out")
            pids[str(client["client_pid"])]=exited(client["client_pid"])
            if client["name"]=="concurrency":
                report=Path(client["report"]); assert sha(report)==client["report_sha256"]
                nested=json.loads(report.read_text())
                assert nested.get("owned_client_groups_empty") is True and not nested.get("cleanup_failed")
    for file in ("/proc/net/tcp","/proc/net/tcp6"):
        for line in Path(file).read_text().splitlines()[1:]:
            fields=line.split()
            assert not (fields[3]=="0A" and int(fields[1].split(":")[1],16)==8899), "Port8899 still listening"
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(("127.0.0.1",8899))
    # The first diagnostic is a preserved predecessor, including its failed
    # capture and semantic results. It must have released every owned process.
    previous=json.loads((ROOT/"results/literal-final-305-24f0-5a/result.json").read_text())
    assert previous.get("state")=="completed" and previous.get("finished_at_utc")
    assert not previous.get("cleanup_failed") and not previous.get("error")
    assert previous["server_cleanup"].get("owned_group_empty") is True
    assert previous["server_cleanup"].get("unexpected_exit") is False
    pids["470760"]=exited(470760)
    pids[str(previous["server_pid"])]=exited(previous["server_pid"])
    for client in previous["clients"]:
        assert client.get("finished_at_utc") and client["cleanup"].get("owned_group_empty") is True
        pids[str(client["client_pid"])]=exited(client["client_pid"])
    gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],
                                text=True,timeout=20).strip()
    assert not gpu, "GPU still has a compute owner"
    if require_sources:
        for path,fingerprint in EXPECTED.items():
            assert sha(path)==fingerprint,("Changed launch source",str(path))
    first=value["jobs"][0]["command"]
    setup=Path(first[first.index("--setup")+1])
    prior=json.loads((BATCH/"final-305-single-job.json").read_text())
    job={key:prior[key] for key in ("label","engine","tabby","recipe","model_path","env")}
    job["label"]=LABEL
    return {"checked_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
            "batch_status_sha256":sha(BATCH/"status.json"),
            "process_exit_evidence":pids,"gpu_compute_pids":[],"port8899_free":True,
            "setup":str(setup),"job":job}

def main():
    os.umask(0o077)
    evidence=preflight()
    assert not OUTPUT.exists() and not OUTPUT.is_symlink()
    assert not OUTPUT.with_name(OUTPUT.name+".observer-inputs").exists()
    job_path=ROOT/(LABEL+"-job.json")
    with job_path.open("x") as f:
        json.dump(evidence["job"],f,indent=2); f.write("\n")
    command=["python3",str(ROOT/"literal_capture_controller_p095.py"),"--job",str(job_path),
             "--setup",evidence["setup"],"--output",str(OUTPUT),
             "--observer-dir",str(ROOT/"literal-final-observer-p095")]
    record={"schema_version":1,"authorization":"Parent authorized one separately named corrected raw8 replay after diagnostic top_p mismatch; original payloads and production sources unchanged.",
            "preflight":evidence,"command":command,"sources_sha256":{str(k):v for k,v in EXPECTED.items()},
            "job_sha256":sha(job_path)}
    # Recheck the resource boundary immediately before opening our process.
    preflight()
    log_path=ROOT/"logs"/(LABEL+"-controller.log")
    with log_path.open("x") as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record["pid"]=process.pid
    record["launched_utc"]=dt.datetime.now(dt.timezone.utc).isoformat()
    with (ROOT/(LABEL+"-launch.json")).open("x") as f:
        json.dump(record,f,indent=2);f.write("\n")
    print(json.dumps({"pid":process.pid,"output":str(OUTPUT),"log":str(log_path),
                     "job_sha256":record["job_sha256"],"launched_utc":record["launched_utc"]}),flush=True)

if __name__=="__main__":
    main()
