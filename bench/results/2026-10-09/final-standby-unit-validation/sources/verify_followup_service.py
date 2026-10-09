#!/usr/bin/env python3
"""Verify an already started, disabled Qwen unit at exact qualified source revisions.

This verifier does not start, stop, enable or edit any service. Root owns lifecycle.
It reuses the frozen October 8 service identity/readiness gates and current recipe
FD8 ownership check, then executes the unchanged five-case OpenAI SDK client.
"""
from pathlib import Path
import argparse, fcntl, hashlib, importlib.util, json, os, re, signal
import subprocess, sys, time, traceback

OLD = Path("/home/cruzspark/qwen-overnight-20261008")
RECIPE = Path("/home/cruzspark/qwen-spark-recipe")
REHEARSE = OLD / "service-promotion/rehearse.py"
REHEARSE_SHA = "efaae01bc5a859948dd7a46d9e9584fd762ccfce143722946865a2aaf692224f"
MATRIX_SHA = "3a127cdea1f0ed224ba8bcd773440e444acc8d1dd8274a58f98f02402a9fff73"
SDK_SHA = "f483990624c285f331b5752ab7691d89796e3b4d27d9479750a727631def2b6a"
API_SHA = "a834cd1ac95e1519a5a14856819c762a581d22ee1e61df5f4d57ca5f4c054adc"
LOCK = "/home/cruzspark/redsnow-gpu.lock"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path, expected, name):
    if sha(path) != expected:
        raise ValueError("Frozen source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine-ref", "tabby-ref", "recipe-ref", "service-env-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("engine_ref", "tabby_ref", "recipe_ref"):
        if not re.fullmatch("[0-9a-f]{40}", getattr(args, name)):
            parser.error("Exact full source revisions are required")
    if not re.fullmatch("[0-9a-f]{64}", args.service_env_sha256):
        parser.error("Exact service environment SHA256 is required")
    os.umask(0o077)
    args.output = args.output.resolve()
    args.output.mkdir(mode=0o700)
    module = load(REHEARSE, REHEARSE_SHA, "followup_service_frozen")
    sys.path.insert(0, str(RECIPE / "bench"))
    matrix = load(RECIPE / "bench/run_matrix.py", MATRIX_SHA, "followup_service_matrix")
    client = load(RECIPE / "bench/api_client.py", API_SHA, "followup_service_api")
    if sha(RECIPE / "bench/sdk_smoke.py") != SDK_SHA:
        raise ValueError("SDK source changed")
    guard = matrix.InterruptState()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, guard.receive)
    port_lock = Path("/tmp/qwen-experiment-8899-1000.lock").open("a")
    fcntl.flock(port_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.old_recipe = OLD / "recipe"
    args.ready_timeout = 600
    record = {
        "schema_version": 1, "passed": False, "state": "preflight",
        "started_at_utc": module.now(), "controller_sha256": sha(__file__),
        "frozen_sources": {"rehearsal": REHEARSE_SHA, "matrix": MATRIX_SHA,
                           "sdk": SDK_SHA, "api_client": API_SHA},
        "scope": "Actual disabled standby unit, 1,048,576-token pool; two raw transport checks and five unchanged SDK cases. No lifecycle mutation or general quality/throughput claim.",
    }
    def save():
        module.atomic(args.output / "result.json", record)
    process = None
    save()
    try:
        guard.checkpoint()
        helpers = module.load_helpers()
        _, settings, sources = module.preflight(args, helpers)
        if settings["__service_env_sha256"] != args.service_env_sha256:
            raise ValueError("Unexpected installed service environment")
        required = {"GPU_LOCK_FILE": LOCK, "PROFILE": "concurrent", "NGRAM_RAM": "false",
                    "MAX_BATCH_SIZE": "4", "CACHE_SIZE": "1048576", "MAX_SEQ_LEN": "262144",
                    "CHUNK_SIZE": "2048", "EXL3_DRAFT_ROW_BUDGET": "8"}
        if any(settings.get(k) != v for k, v in required.items()):
            raise ValueError("Selected standby-unit settings changed")
        if module.ctl("show", module.UNIT, "--property=UnitFileState", "--value") != "disabled":
            raise ValueError("Qwen standby unit enablement changed")
        record["sources"] = sources
        record["selected_settings"] = required
        record["state"] = "waiting_for_readiness"
        save()
        # Admit model requests only after the identified unit owns the shared lock.
        deadline = time.monotonic() + args.ready_timeout
        first_start = None
        while time.monotonic() < deadline:
            guard.checkpoint()
            pre_api = module.unit_status()
            if pre_api.get("ActiveState") in ("inactive", "failed"):
                raise ValueError("Unit stopped before the API admission gate")
            if int(pre_api.get("MainPID", "0")) > 0 and pre_api.get("InvocationID"):
                if first_start is None:
                    first_start = pre_api
                elif not module.same_start(first_start, pre_api):
                    raise ValueError("Unit restarted before the API admission gate")
            try:
                module.unit_owner(pre_api, settings, helpers)
                lock_before_api = matrix.verify_gpu_lock(int(pre_api["MainPID"]), settings)
            except (OSError, ValueError, RuntimeError):
                time.sleep(1)
                continue
            break
        else:
            raise TimeoutError("Unit GPU-lock/API admission gate timed out")
        record["unit_before_api"] = pre_api
        record["gpu_lock_before_api"] = lock_before_api
        save()
        guard.checkpoint()
        ready = module.ready_new("selected-unit", args, settings, helpers, client)
        guard.checkpoint()
        status = ready["unit"]
        if not module.same_start(pre_api, status):
            raise ValueError("Unit changed at the API admission boundary")
        pid = int(status["MainPID"])
        proc_env = dict(item.decode().split("=", 1) for item in
                        (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
                        if item and b"=" in item)
        if proc_env.get("GPU_LOCK_FILE") != LOCK:
            raise ValueError("MainPID lacks selected GPU lock environment")
        metadata = Path(ready["state_dir"]) / "deployment.json"
        deployment = json.loads(metadata.read_text())
        if deployment.get("environment", {}).get("GPU_LOCK_FILE") != LOCK:
            raise ValueError("Deployment snapshot lacks selected GPU lock")
        record["gpu_lock_before"] = matrix.verify_gpu_lock(pid, settings)
        record["unit_before_sdk"] = status
        record["deployment_metadata"] = {"path": str(metadata), "sha256": sha(metadata)}
        record["raw_transport_checks"] = len(ready["api"]["requests"])
        command = [str(OLD / "client-venv/bin/python"), str(RECIPE / "bench/sdk_smoke.py"),
                   "--base-url", module.BASE, "--model", module.ALIAS,
                   "--label", "followup-selected-standby-unit", "--metadata", str(metadata),
                   "--output", str(args.output / "sdk.json"), "--timeout", "120"]
        record["sdk_command"] = command
        record["state"] = "measuring"
        save()
        with (args.output / "sdk.log").open("x") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record["sdk_pid"] = process.pid
            save()
            guard.checkpoint()
            deadline = time.monotonic() + 700
            while process.poll() is None:
                guard.checkpoint()
                if time.monotonic() >= deadline:
                    raise TimeoutError("SDK verification exceeded its deadline")
                time.sleep(0.2)
        guard.checkpoint()
        report = json.loads((args.output / "sdk.json").read_text())
        record["sdk_exit_code"] = process.returncode
        record["sdk_report_sha256"] = sha(args.output / "sdk.json")
        record["sdk_summary"] = {key: report.get(key) for key in
                                 ("passed", "failed", "skipped", "finished_at_utc", "client")}
        if (process.returncode != 0 or report.get("setup_error") or
            len(report.get("cases", [])) != 5 or report.get("passed") != 5 or
            report.get("failed") != 0 or report.get("skipped") != 0 or
            not report.get("finished_at_utc")):
            raise ValueError("SDK verification did not pass all five cases")
        after = module.unit_status()
        if not module.same_start(status, after):
            raise ValueError("Unit changed during SDK verification")
        module.unit_owner(after, settings, helpers)
        record["gpu_lock_after"] = matrix.verify_gpu_lock(int(after["MainPID"]), settings)
        if sha(metadata) != record["deployment_metadata"]["sha256"]:
            raise ValueError("Deployment metadata changed during SDK verification")
        if module.ctl("show", module.UNIT, "--property=UnitFileState", "--value") != "disabled":
            raise ValueError("Qwen unit enablement changed")
        record["sources_after"] = {key: module.exact_repo(path, ref, helpers)
            for key, path, ref in (("recipe", RECIPE, args.recipe_ref),
                                  ("engine", module.NEW / "exllamav3", args.engine_ref),
                                  ("server", module.NEW / "tabbyAPI", args.tabby_ref))}
        record.update(passed=True, state="completed", unit_after_sdk=after,
                      finished_at_utc=module.now())
    except BaseException as exc:
        record.update(error=type(exc).__name__ + ": " + str(exc), state="failed",
                      finished_at_utc=module.now())
        traceback.print_exc()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            record["sdk_cleanup_exit_code"] = process.returncode
        record["interruption_signal"] = guard.signum
        if guard.signum is not None:
            record.update(passed=False, state="interrupted")
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        port_lock.close()
    return 0 if record["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
