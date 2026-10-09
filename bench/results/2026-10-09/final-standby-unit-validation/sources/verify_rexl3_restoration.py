#!/usr/bin/env python3
"""Verify the selected REXL3 handback without collecting private configuration."""
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import urllib.request

BASE = Path("/home/cruzspark/qwen-followup-20261009")
CONFIG = Path("/home/cruzspark/.config/rexl3-manager/config.json")
LOCK = Path("/home/cruzspark/redsnow-gpu.lock")
EXPECTED_CONFIG = "87c35a765fe1a7e537832a26cc2fb685f64330f497ff8b36905fd96bbee0b901"
EXPECTED_REQUEST = "bcb4e767c83a4aefb0dbe55cd2ae8c32"
EXPECTED_INVOCATION = "c873f7bda48e42f0a2b2e17dce3a694b"
EXPECTED_MODEL = "CyberFrost3.8"
EXPECTED_MANAGER_PID = 530616
EXPECTED_BACKEND_PID = 530624

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args):
    p = subprocess.run(args, capture_output=True, text=True, check=True, timeout=15)
    return p.stdout.strip()

def unit(name):
    properties = ["ActiveState", "SubState", "MainPID", "InvocationID", "NRestarts", "Result", "ExecMainStatus", "UnitFileState"]
    return dict(line.split("=", 1) for line in run(["systemctl", "--user", "show", name, *sum((["-p", k] for k in properties), [])]).splitlines())

def identity(pid):
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text()
    tail = stat[stat.rfind(")") + 2:].split()
    return {"pid": pid, "start_ticks": int(tail[19]), "exe": str((root / "exe").resolve(strict=True))}

def get(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        assert response.status == 200
        return response.read()

def status(headers):
    full = json.loads(get("http://127.0.0.1:8889/manager/status", headers))
    selected = {k: full.get(k) for k in ["request_id", "loaded_model", "requested_model", "state", "error", "pid", "phase", "active_requests"]}
    assert selected == {
        "request_id": EXPECTED_REQUEST, "loaded_model": EXPECTED_MODEL,
        "requested_model": EXPECTED_MODEL, "state": "ready", "error": None,
        "pid": EXPECTED_BACKEND_PID, "phase": "ready", "active_requests": 0,
    }, selected
    return selected

def lock_proof():
    st = LOCK.stat()
    assert st.st_ino == 2273157 and st.st_dev == 66306 and st.st_size == 0
    holders = []
    for pid in (EXPECTED_MANAGER_PID, EXPECTED_BACKEND_PID):
        for fd in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                fst = fd.stat()
            except FileNotFoundError:
                continue
            if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
                continue
            info = (Path("/proc") / str(pid) / "fdinfo" / fd.name).read_text()
            lock_lines = [line for line in info.splitlines() if line.startswith("lock:")]
            holders.append({"pid": pid, "fd": int(fd.name), "locks": lock_lines})
    assert any(h["pid"] == EXPECTED_MANAGER_PID and any("FLOCK" in x and "WRITE" in x for x in h["locks"]) for h in holders), holders
    blocked = False
    with LOCK.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            blocked = True
        else:
            fcntl.flock(f, fcntl.LOCK_UN)
    assert blocked, "shared GPU lock is not held"
    return {"path": str(LOCK), "inode": st.st_ino, "device": st.st_dev, "size": st.st_size, "holders": holders, "independent_exclusive_acquisition_blocked": blocked}

def main():
    started = utc()
    assert sha(CONFIG) == EXPECTED_CONFIG
    config = json.loads(CONFIG.read_text())
    key = Path(config["key_file"]).read_text().strip()
    headers = {"Authorization": "Bearer " + key}
    manager = unit("rexl3-manager.service")
    qwen = unit("qwen38-exl3.service")
    assert manager["ActiveState"] == "active" and manager["SubState"] == "running"
    assert manager["MainPID"] == str(EXPECTED_MANAGER_PID)
    assert manager["InvocationID"] == EXPECTED_INVOCATION
    assert manager["UnitFileState"] == "enabled" and manager["NRestarts"] == "0"
    assert qwen["ActiveState"] == "inactive" and qwen["MainPID"] == "0"
    assert qwen["UnitFileState"] == "disabled" and qwen["Result"] == "success"
    before = status(headers)
    process = identity(EXPECTED_BACKEND_PID)
    assert process["exe"] == "/home/cruzspark/rexl3-fn-eng-target/release/rexl3-serve"
    models = json.loads(get("http://127.0.0.1:8888/v1/models", headers))
    ids = [m["id"] for m in models["data"]]
    assert ids == [EXPECTED_MODEL], ids
    metrics_text = get("http://127.0.0.1:8888/metrics", headers).decode()
    names = ("rexl3_active_sequences", "rexl3_prefilling_sequences", "rexl3_queue_depth")
    metrics = {}
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0] in names:
            value = float(parts[1])
            assert math.isfinite(value) and value == 0
            metrics[parts[0]] = value
    assert set(metrics) == set(names), metrics
    lock = lock_proof()
    after = status(headers)
    assert identity(EXPECTED_BACKEND_PID) == process
    assert unit("rexl3-manager.service") == manager
    assert unit("qwen38-exl3.service") == qwen
    assert sha(CONFIG) == EXPECTED_CONFIG
    sources = {}
    for name, path, expected in (
        ("recipe", "/home/cruzspark/qwen-spark-recipe", "22d673463dba8e1f3a862056348769ab85a3c032"),
        ("engine", "/home/cruzspark/qwen38-exl3-20261008/exllamav3", "24f0dece34f09c8d1e2359d6b3b3f7befef7331b"),
        ("tabby", "/home/cruzspark/qwen38-exl3-20261008/tabbyAPI", "f391c96beea0bcd21cbae3c929f75bb98136ff32"),
    ):
        head = run(["git", "-C", path, "rev-parse", "HEAD"])
        dirty = run(["git", "-C", path, "status", "--porcelain"])
        assert head == expected and not dirty, (name, head, dirty)
        sources[name] = {"path": path, "head": head, "clean": True}
    receipt = {
        "schema": 1, "passed": True, "started_utc": started, "finished_utc": utc(),
        "purpose": "Verify the previously selected REXL3 CyberFrost workload has been restored and Qwen remains disabled standby after its completed live gate.",
        "source_sha256": sha(Path(__file__)), "source_path": str(Path(__file__).resolve()),
        "config_sha256_before_and_after": EXPECTED_CONFIG,
        "manager_unit": manager, "qwen_unit": qwen,
        "manager_status_before": before, "manager_status_after": after,
        "backend_process": process, "backend_model_ids": ids, "idle_metrics": metrics,
        "shared_gpu_lock": lock, "canonical_qwen_sources": sources,
        "scope": {"new_generation_requests": 0, "manager_or_backend_mutations": 0, "private_config_or_key_values_collected": False},
    }
    out = BASE / "rexl3-restoration-ready.json"
    with out.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(json.dumps({"path": str(out), "sha256": sha(out), "passed": True, "finished_utc": receipt["finished_utc"], "backend_model_ids": ids}))

if __name__ == "__main__":
    main()
