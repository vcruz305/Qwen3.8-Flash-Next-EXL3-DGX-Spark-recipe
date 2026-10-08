#!/usr/bin/env python3
"""One scheduled, owned Spark API compatibility run; preparation never starts it.

Requires completed setup-gemm-f4-status.json and exact b532/3a sources. Starts
only a fresh loopback server, runs unchanged packaged clients serially, and
always stops its own verified process groups. No setup, fetch, checkout,
installation, automatic retry, or existing-output replacement is performed.
Use --bench to append the original overnight-v1 all-suite benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import urllib.request
import uuid

ROOT = Path("/home/cruzspark/qwen-overnight-20261008")
RUNTIME = Path("/home/cruzspark/qwen38-exl3-20261008")
RECIPE = ROOT / "recipe-updated"
MODEL = Path("/home/cruzspark/models/flashnext-exl3-3.05bpw")
OUTPUT = ROOT / "results/api-f4-gemm"
SETUP = ROOT / "results/setup-gemm-f4-status.json"
ENGINE = "16ca20d27c0e4cce15a9bbc131e6d047065395b5"
TABBY = "f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6"
ALIAS = "Qwen3.8-Flash-Next-EXL3"
BASE = "http://127.0.0.1:8899/v1"
HELPER_SHA = "48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586"
EXPECTED = {"tools": 28, "sdk": 5, "resilience": 19, "auto": 9}
TOOL_CASES = set(("auto no_args strings typed required named required_adversarial "
                  "named_adversarial none parallel parallel_disabled reasoning_tool "
                  "round_trip truncated").split())
EXTRA_FILES = ("bench/sdk_smoke.py", "bench/requirements-sdk.txt",
               "bench/api_resilience.py", "bench/auto_compatibility.py",
               "exllamav3-tabby/drop-model-cache.sh")
TUNING = {
    "PROFILE": "single", "NGRAM_RAM": "true", "CACHE_SIZE": "262144",
    "MAX_SEQ_LEN": "262144", "MAX_BATCH_SIZE": "1", "CHUNK_SIZE": "2048",
    "DRAFT_MODE": "mtp", "DRAFT_NUM_TOKENS": "5", "DYNAMIC_DRAFT": "true",
    "SYSMEM_RECURRENT_CACHE": "4096", "VISION": "false", "REASONING": "true",
    "TOOL_FORMAT": "qwen3_5", "SERVED_NAME": ALIAS,
    "EXL3_REF": ENGINE, "EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "1",
    "EXL3_MOE_COOP_WIDE": "1", "EXL3_MTP_HEAD_N": "65536",
    "EXL3_DRAFT_CONFIDENCE": "0.6", "EXL3_GDN_PROJ_FP32": "1",
    "EXL3_GDN_CONV_TOKEN_MAJOR": "0", "EXL3_GDN_CONV_BF16_PRODUCT": "1",
    "EXL3_ATTN_DECODE_LEGACY_SPLITS": "1", "EXL3_MOE_COOP_MIXEDK": "0",
    "EXL3_MOE_MIXEDK_NOSYNC": "1",
    "EXL3_GEMM_LEGACY_TILES": "1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helpers(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HELPER_SHA:
        raise ValueError("Frozen experiment helper changed; review it before running")
    module = types.ModuleType("api3a_frozen_experiment_helpers")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def check_setup(value):
    if not isinstance(value, dict) or value.get("state") != "completed":
        raise ValueError("Root setup must already have state=completed; no waiting/retry")
    for key, expected in (("engine", ENGINE), ("tabby", TABBY), ("runtime", str(RUNTIME))):
        if key in value and value[key] != expected:
            raise ValueError(f"Setup status has unexpected {key}")


def commands(with_bench, output=OUTPUT, recipe=RECIPE, runtime=RUNTIME, root=ROOT):
    py = str(runtime / "venv/bin/python")
    common = ["--base-url", BASE, "--model", ALIAS]
    metadata = ["--metadata", str(output / "deployment.json")]
    label = ["--label", "api-f4-gemm-305"]
    yield "tools", [py, str(recipe / "bench/tool_smoke.py"), *common,
                    "--mode", "both", "--case", "all", "--repeat", "1",
                    "--timeout", "180"]
    yield "sdk", [str(root / "client-venv/bin/python"), str(recipe / "bench/sdk_smoke.py"),
                  *common, *metadata, *label]
    yield "resilience", [py, str(recipe / "bench/api_resilience.py"),
                         *common, *metadata, *label]
    yield "auto", [py, str(recipe / "bench/auto_compatibility.py"),
                   *common, *metadata, *label]
    if with_bench:
        yield "bench", [py, str(recipe / "bench/bench_v1.py"), *common, *metadata, *label,
                        "--suite", "all", "--warmup", "1", "--repeat", "3",
                        "--run-id", "overnight-v1"]


def summarize(name, report, exit_code):
    """Count actual completed outcomes; partial/skip/inconsistent reports fail."""
    if name == "bench":
        cases = report.get("cases", {})
        completed = bool(report.get("completed_at_utc"))
        counts = {key: {"warmup": len(value.get("warmup", [])),
                        "measured": len(value.get("runs", []))}
                  for key, value in cases.items()}
        okay = (set(cases) == {"code", "devops", "prose"} and
                all(row == {"warmup": 1, "measured": 3} for row in counts.values()) and
                report.get("run_id") == "overnight-v1" and not report.get("errors"))
        return {"completed_report": completed, "counts": counts,
                "passed": exit_code == 0 and completed and okay}
    if name == "tools":
        rows = report.get("results", [])
        statuses = [("pass" if row["passed"] else "fail")
                    if type(row.get("passed")) is bool else "invalid" for row in rows]
        keys = [(row.get("case"), row.get("mode"), row.get("repeat_index")) for row in rows]
        expected_keys = {(case, mode, 0) for case in TOOL_CASES
                         for mode in ("stream", "nonstream")}
        keys_ok = len(keys) == len(set(keys)) and set(keys) == expected_keys
        completed = bool(report.get("completed_at_utc"))
        summary = report.get("summary", {})
        summary_ok = (summary.get("total") == len(rows) and
                      summary.get("passed") == statuses.count("pass") and
                      summary.get("failed") == statuses.count("fail"))
        errors = report.get("errors")
    else:
        rows = report.get("cases", [])
        statuses = [row.get("status", "invalid") for row in rows]
        keys = [row.get("name") for row in rows]
        keys_ok = all(isinstance(key, str) and key for key in keys) and len(keys) == len(set(keys))
        completed = bool(report.get("finished_at_utc" if name == "sdk" else "finished_utc"))
        summary = report if name == "sdk" else report.get("summary", {})
        pairs = (("passed", "pass"), ("failed", "fail"), ("skipped", "skipped")) if name == "sdk" else (("pass", "pass"), ("fail", "fail"))
        summary_ok = all(summary.get(key, 0) == statuses.count(status) for key, status in pairs)
        errors = report.get("setup_error")
    counts = dict(Counter(statuses))
    okay = (exit_code == 0 and completed and len(rows) == EXPECTED[name] and
            counts.get("pass", 0) == EXPECTED[name] and keys_ok and summary_ok and not errors)
    return {"expected_checks": EXPECTED[name], "observed_checks": len(rows),
            "counts": counts, "completed_report": completed, "unique_expected_cases": keys_ok,
            "summary_consistent": summary_ok, "passed": bool(okay)}


def stop_owned(helper, process, token, timeout=90):
    """At most 90 s; signal only members verified by PGID, SID and owner token."""
    started = time.monotonic()
    deadline = started + timeout
    sent = []
    for sig, phase_end in ((signal.SIGTERM, deadline - min(10, timeout / 3)),
                            (signal.SIGKILL, deadline)):
        if not helper.group_members(process, token):
            break
        os.killpg(process.pid, sig)
        sent.append(sig.name)
        while helper.group_members(process, token) and time.monotonic() < phase_end:
            process.poll()
            time.sleep(.1)
    if helper.group_members(process, token):
        raise RuntimeError("Verified owned group survived cleanup deadline")
    code = process.wait(timeout=max(.01, deadline - time.monotonic()))
    return {"exit_code": code, "signals": sent,
            "wall_seconds": time.monotonic() - started, "owned_group_empty": True}


def cleanup_server(helper, process, token):
    before = process.poll()
    result = stop_owned(helper, process, token)
    result["exit_before_cleanup"] = before
    result["unexpected_exit"] = before is not None or not result["signals"]
    return result


def require_listener(helper, server):
    if server.poll() is not None:
        raise RuntimeError(f"Owned server exited: {server.returncode}")
    helper.require_owned_listener(server.pid)
    wanted = f"0100007F:{8899:04X}"
    listeners = [line.split() for line in Path("/proc/net/tcp").read_text().splitlines()[1:]]
    on_port = [row for row in listeners if row[3] == "0A" and row[1].split(":")[1] == wanted.split(":")[1]]
    if not on_port or any(row[1] != wanted for row in on_port):
        raise RuntimeError("Port 8899 is not exclusively bound to IPv4 loopback")


def api_json(endpoint):
    # This owned server deliberately uses loopback/no-auth; inherit no API key.
    with urllib.request.urlopen(BASE + endpoint, timeout=2) as response:
        return json.load(response)


def validate_deployment(deployment, identity, output):
    for key in ("engine", "server"):
        if deployment[key]["commit"] != identity[key]["commit"]:
            raise ValueError(f"Loaded {key} does not match preflight")
    if Path(deployment["model"]["resolved_path"]).resolve() != MODEL.resolve():
        raise ValueError("Loaded deployment model differs from the requested 3.05 pack")
    cfg = deployment["config"]["values"]
    wanted = {
        "network": {"host": "127.0.0.1", "port": 8899, "disable_auth": True},
        "model": {"model_name": ALIAS, "model_dir": str(output / "state/models"),
                  "cache_size": 262144, "max_seq_len": 262144, "max_batch_size": 1,
                  "chunk_size": 2048, "cache_mode": "8,8", "ngram_ram": True,
                  "reasoning": True, "tool_format": "qwen3_5", "vision": False},
        "draft_model": {"draft_mode": "mtp", "draft_num_tokens": 5,
                        "dynamic_draft": True, "draft_cache_mode": "8,8"},
    }
    for section, values in wanted.items():
        for key, value in values.items():
            if cfg.get(section, {}).get(key) != value:
                raise ValueError(f"Unexpected deployed {section}.{key}")
    recorded = deployment["environment"]
    for key, value in TUNING.items():
        if key.startswith("EXL3_") or key in ("PROFILE", "NGRAM_RAM", "CACHE_SIZE",
                "MAX_SEQ_LEN", "MAX_BATCH_SIZE", "CHUNK_SIZE", "DRAFT_MODE", "DRAFT_NUM_TOKENS"):
            if recorded.get(key) != value:
                raise ValueError(f"Unexpected deployed environment {key}")


def run(args):
    helper_path = Path(__file__).with_name("spark_experiment_controller.py")
    helper = load_helpers(helper_path)
    os.umask(0o077)
    # mkdir is the exclusive claim; files, directories and dangling symlinks all fail.
    OUTPUT.mkdir(mode=0o700)
    record = {"schema_version": 1, "state": "preflight", "passed": False,
              "diagnostic_only": True, "started_at_utc": helper.stamp(),
              "controller_sha256": sha(__file__), "helper_sha256": HELPER_SHA,
              "expected_engine": ENGINE, "expected_server": TABBY,
              "expected_checks": EXPECTED, "optional_benchmark": args.bench, "clients": []}
    def save():
        helper.atomic(OUTPUT / "result.json", record)
    save()
    server = server_log = client = None
    token = uuid.uuid4().hex
    record["ownership_token"] = token
    try:
        setup_bytes = SETUP.read_bytes()
        setup = json.loads(setup_bytes)
        check_setup(setup)
        record["setup_status_sha256"] = hashlib.sha256(setup_bytes).hexdigest()
        helper.atomic(OUTPUT / "setup-status.json", setup)
        job = {"label": "api-f4-gemm-305", "model_path": str(MODEL.resolve(strict=True)),
               "model_identity": helper.model_identity(MODEL), "env": dict(TUNING)}
        if not job["model_identity"]["weights"]:
            raise ValueError("Requested model has no weight files")
        identity = helper.source_identity(RECIPE, RUNTIME)
        for key, expected in (("engine", ENGINE), ("server", TABBY)):
            if identity[key]["commit"] != expected or identity[key]["tracked_changes"]:
                raise ValueError(f"Expected clean exact {key} revision {expected}")
        extra = {name: sha(RECIPE / name) for name in EXTRA_FILES}
        client_python = ROOT / "client-venv/bin/python"
        client_packages = json.loads(helper.capture([str(client_python), "-c",
            "import importlib.metadata,json,platform; print(json.dumps({'python':platform.python_version(),"
            "'packages':{n:importlib.metadata.version(n) for n in ('openai','httpx','pydantic')}}))"]))
        env = helper.resolved_env(RECIPE, RUNTIME, job)
        for key in ("API_KEY", "TABBY_API_KEY", "OPENAI_API_KEY"):
            env.pop(key, None)
        env.update(STATE_DIR=str(OUTPUT / "state"), TABBY_REF=TABBY,
                   QWEN_EXPERIMENT_OWNER=token, PYTHONPATH=str(RUNTIME / "exllamav3"),
                   PYTHONUNBUFFERED="1")
        for key, value in TUNING.items():
            if env.get(key) != value:
                raise ValueError(f"Resolved tuning changed: {key}")
        record.update(sources=identity, extra_files_sha256=extra, sdk_client=client_packages,
                      model_identity=job["model_identity"],
                      resolved_tuning={k: v for k, v in env.items() if helper.tuning_key(k)},
                      state_dir=env["STATE_DIR"], state="loading")
        verify_args = types.SimpleNamespace(recipe=RECIPE, runtime=RUNTIME)
        def verify():
            helper.verify_inputs(job, verify_args, identity, env)
            if {name: sha(RECIPE / name) for name in EXTRA_FILES} != extra:
                raise ValueError("Packaged client/launcher sources changed")
            if sha(helper_path) != HELPER_SHA or sha(__file__) != record["controller_sha256"]:
                raise ValueError("Controller/helper changed during the run")
            if sha(SETUP) != record["setup_status_sha256"]:
                raise ValueError("Completed setup status changed during the run")
            if "deployment_sha256" in record:
                if sha(OUTPUT / "deployment.json") != record["deployment_sha256"]:
                    raise ValueError("Verified deployment sidecar changed")
                if sha(OUTPUT / "state/config.yml") != deployment["config"]["sha256"]:
                    raise ValueError("Loaded configuration changed after verification")
        # TABBY_REF is compared by helper.verify_inputs; pin it for both resolution and launch.
        job["env"]["TABBY_REF"] = TABBY
        verify()
        helper.require_free_port()
        save()
        server_log = (OUTPUT / "server.log").open("x")
        started = time.monotonic()
        record["server_command"] = ["bash", str(RECIPE / "exllamav3-tabby/serve.sh")]
        server = subprocess.Popen(record["server_command"],
            cwd=RECIPE, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        record["server_pid"] = server.pid
        save()
        while time.monotonic() - started < args.ready_timeout:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited while loading: {server.returncode}")
            try:
                models = api_json("/models")
                if models.get("data"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Owned server readiness deadline exceeded")
        require_listener(helper, server)
        current = api_json("/model")
        if [row.get("id") for row in models["data"]] != [ALIAS]:
            raise ValueError("Advertised model alias differs from the requested one")
        if current.get("id") not in {ALIAS, MODEL.name}:
            raise ValueError("Loaded model endpoint identifies another pack")
        deployment = json.loads((OUTPUT / "state/deployment.json").read_text())
        validate_deployment(deployment, identity, OUTPUT)
        helper.atomic(OUTPUT / "deployment.json", deployment)
        record.update(load_wall_seconds=time.monotonic()-started, models=models,
                      current_model=current, deployment_sha256=sha(OUTPUT / "deployment.json"),
                      state="clients")
        verify()
        save()
        for name, command in commands(args.bench):
            require_listener(helper, server)
            verify()
            destination = OUTPUT / f"{name}.json"
            if destination.exists() or destination.is_symlink():
                raise ValueError("Client report path already exists")
            command += ["--output", str(destination)]
            row = {"name": name, "command": command, "started_at_utc": helper.stamp(),
                   "report": str(destination), "metadata": str(OUTPUT / "deployment.json"),
                   "metadata_sha256": record["deployment_sha256"]}
            record["clients"].append(row)
            record["active"] = name
            save()
            with (OUTPUT / f"{name}.log").open("x") as log:
                client = subprocess.Popen(command, cwd=RECIPE, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row["client_pid"] = client.pid
                try:
                    row["exit_code"] = client.wait(timeout=args.client_timeout)
                except subprocess.TimeoutExpired:
                    row["timed_out"] = True
                finally:
                    row["cleanup"] = stop_owned(helper, client, token)
                    row.setdefault("exit_code", row["cleanup"]["exit_code"])
                    client = None
            row["finished_at_utc"] = helper.stamp()
            if destination.is_file():
                row["report_sha256"] = sha(destination)
                try:
                    row.update(summarize(name, json.loads(destination.read_text()), row["exit_code"]))
                except (ValueError, TypeError, KeyError) as error:
                    row.update(passed=False, report_error=f"{type(error).__name__}: {error}")
            else:
                row.update(passed=False, report_error="No JSON report")
            if row.get("timed_out"):
                row["passed"] = False
            save()
            print(json.dumps({"client": name, "exit_code": row["exit_code"],
                              "passed": row["passed"], "counts": row.get("counts")}), flush=True)
        verify()
        require_listener(helper, server)
        record.update(state="completed", active=None,
                      passed=all(row.get("passed") for row in record["clients"]))
    except BaseException as error:
        record.update(state="failed", passed=False,
                      error=f"{type(error).__name__}: {error}")
        print(record["error"], file=sys.stderr, flush=True)
    finally:
        record["active"] = None
        for label, process in (("client", client), ("server", server)):
            if process is not None:
                try:
                    cleanup = cleanup_server(helper, process, token) if label == "server" else stop_owned(helper, process, token)
                    record[f"{label}_cleanup"] = cleanup
                    if label == "server" and cleanup["unexpected_exit"]:
                        record.update(state="failed", passed=False,
                                      server_exit_error="Server exited before the controller signaled its owned group")
                except BaseException as error:
                    record.update(state="failed", passed=False, cleanup_failed=True)
                    record[f"{label}_cleanup_error"] = f"{type(error).__name__}: {error}"
        if server_log is not None:
            server_log.close()
        record["finished_at_utc"] = helper.stamp()
        save()
    return 0 if record["passed"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--client-timeout", type=float, default=1200)
    args = parser.parse_args()
    if not 0 < args.ready_timeout <= 1800 or not 0 < args.client_timeout <= 3600:
        parser.error("Timeouts must be positive and bounded (ready<=1800, client<=3600)")
    return args


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(run(parse_args()))
