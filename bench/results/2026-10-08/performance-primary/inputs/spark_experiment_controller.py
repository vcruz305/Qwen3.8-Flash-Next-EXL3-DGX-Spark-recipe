#!/usr/bin/env python3
"""Sequential, local-only Spark experiments. Preparation does not run setup or change refs.

Example job (JSON file contains an array):
{"label":"405-single", "model_path":"/home/cruzspark/models/flashnext-exl3-4.05bpw",
 "env":{"PROFILE":"single","NGRAM_RAM":"false"},
 "bench":[{"suite":"all","repeat":3,"warmup":1,"run_id":"overnight-v1"},
          {"suite":"code","context_tokens":16384,"max_tokens":256,
           "repeat":2,"warmup":1,"run_id":"overnight-prefill"}],
 "concurrency":{"streams":[1,2,4],"prompt_tokens":1024,"new_tokens":256,"rounds":3},
 "tool_cases":["auto","no_args","strings","typed","round_trip"]}

Invoke on Spark only when its owner schedules it:
python3 spark_experiment_controller.py --recipe /path/to/recipe-updated \
  --runtime /path/to/new-runtime --jobs /path/to/jobs.json --output /path/to/results

Each job gets a fresh server, state directory and process session. --resume skips
finished attempts only when the normalized job, actual source identities, recipe
scripts, resolved tuning defaults and installed package versions still match.
Interrupted attempts are preserved and retried in a new directory. To repeat a
finished measurement, choose a new output directory or label. Samples are sparse
observations, not guaranteed memory/power peaks. Results are diagnostic; this
controller never grants numerical-quality approval or selects a winning tuning.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:8899/v1"
TUNING = set(("PROFILE NGRAM_RAM BIGCORES OMP_NUM_THREADS MKL_NUM_THREADS CHUNK_SIZE "
              "CACHE_SIZE MAX_SEQ_LEN MAX_BATCH_SIZE DRAFT_MODE DRAFT_NUM_TOKENS "
              "DYNAMIC_DRAFT SYSMEM_RECURRENT_CACHE VISION REASONING TOOL_FORMAT "
              "SERVED_NAME CUDA_HOME TORCH_CUDA_ARCH_LIST").split())
CONTROLLED = set(("RECIPE_HOME VENV EXL3_SRC TABBY_DIR STATE_DIR MODEL_DIR MODEL_PARENT "
                  "MODEL_NAME HOST PORT DISABLE_AUTH DRY_RUN PYTHON_BIN TABBY_REF TABBY_REPO").split())
SETTINGS = {
    "bench": set("suite repeat warmup run_id context_tokens max_tokens cache_mode prompt thinking timeout".split()),
    "concurrency": set("streams prompt_tokens new_tokens rounds warmup run_id timeout".split()),
    "tools": set("case mode repeat max_tokens timeout".split()),
}
FILES = ["exllamav3-tabby/" + name for name in ("env.sh", "serve.sh", "tabby-config.yml",
         "tools/runtime_state.py", "tools/model_memory.py")] + [
         "bench/" + name for name in ("api_client.py", "bench_v1.py", "concurrency.py", "tool_smoke.py")]


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else
                          json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, timeout=30, **kwargs).strip()


def git_identity(path):
    prefix = ["git", "-C", str(path)]
    return {"path": str(path), "commit": capture(prefix + ["rev-parse", "HEAD"]),
            "tracked_changes": capture(prefix + ["status", "--porcelain", "--untracked-files=no"]),
            "tracked_diff_sha256": digest(capture(prefix + ["diff", "--no-ext-diff", "--binary", "HEAD"]))}


def source_identity(recipe, runtime):
    packages = capture([str(runtime / "venv/bin/python"), "-c",
        "import importlib.metadata,json; print(json.dumps(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())))"])
    return {"recipe": git_identity(recipe), "engine": git_identity(runtime / "exllamav3"),
            "server": git_identity(runtime / "tabbyAPI"), "runtime": str(runtime),
            "packages_sha256": digest(json.loads(packages)),
            "controller_sha256": digest(Path(__file__).read_bytes()),
            "files_sha256": {name: digest((recipe / name).read_bytes()) for name in FILES}}


def tuning_key(key):
    return key in TUNING or key.startswith("EXL3_")


def model_identity(model):
    # Literal loader enumeration: the final file wins duplicate tensor keys.
    order = glob.glob(os.path.join(model, "*.safetensors"))
    return {
        "config_sha256": {name: digest((model / name).read_bytes()) for name in
            ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "special_tokens_map.json", "tabby_config.yml") if (model / name).is_file()},
        "loader_glob_order": [Path(file).name for file in order],
        "weights": [{"name": file.name, "path": str(file.resolve()), "bytes": file.stat().st_size,
                     "mtime_ns": file.stat().st_mtime_ns} for file in sorted(map(Path, order))],
        "full_weight_hashes_computed": False}


def normalize(job):
    allowed = {"label", "model_path", "env", "bench", "concurrency", "tool_cases", "tools", "response_model"}
    if not isinstance(job, dict) or set(job) - allowed:
        raise ValueError("Each job must be an object containing only " + ", ".join(sorted(allowed)))
    job = json.loads(json.dumps(job, allow_nan=False))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", job.get("label", "")):
        raise ValueError("Job label must be a short, plain filename")
    model = Path(job["model_path"]).expanduser().resolve(strict=True)
    if not (model / "config.json").is_file():
        raise ValueError("Missing model config: " + str(model))
    job["model_path"] = str(model)
    job["model_identity"] = model_identity(model)
    if not job["model_identity"]["weights"]:
        raise ValueError("No model safetensors found")
    if "response_model" in job and (not isinstance(job["response_model"], str) or not job["response_model"]):
        raise ValueError("response_model must be an explicit nonempty model ID")
    env = job.setdefault("env", {})
    if not isinstance(env, dict) or any(not tuning_key(k) or k in CONTROLLED for k in env):
        raise ValueError("Job env accepts only recipe tuning variables; runtime/state/network are controlled")
    for key, value in env.items():
        if not isinstance(value, (str, int, float, bool)) or "\0" in str(value):
            raise ValueError("Environment values must be scalar strings/numbers/booleans")
        env[key] = str(value).lower() if isinstance(value, bool) else str(value)
    if env.get("PROFILE") not in ("single", "concurrent") or env.get("NGRAM_RAM") not in ("true", "false"):
        raise ValueError("Set PROFILE and explicit NGRAM_RAM=true/false for comparable measurements")
    benches = job.get("bench", [] if {"concurrency", "tools", "tool_cases"} & job.keys() else {})
    job["bench"] = [benches] if isinstance(benches, dict) else benches
    if not isinstance(job["bench"], list):
        raise ValueError("bench must be an object or array")
    for index, settings in enumerate(job["bench"]):
        check_settings(settings, "bench")
        settings.setdefault("run_id", f"experiment-v1-bench-{index + 1}")
    if "concurrency" in job:
        settings = job["concurrency"]
        check_settings(settings, "concurrency")
        settings.setdefault("streams", [1, 2, 4])
        settings.setdefault("prompt_tokens", 1024)
        settings.setdefault("new_tokens", 256)
        settings.setdefault("rounds", 3)
        settings.setdefault("run_id", "experiment-v1-concurrency")
    if "tool_cases" in job:
        if "tools" in job:
            raise ValueError("Use tool_cases or tools, not both")
        cases = job.pop("tool_cases")
        job["tools"] = {"case": ",".join(cases) if isinstance(cases, list) else cases}
    if "tools" in job:
        check_settings(job["tools"], "tools")
    if not job["bench"] and "concurrency" not in job and "tools" not in job:
        raise ValueError("Job contains no measurements")
    return job


def check_settings(settings, kind):
    if not isinstance(settings, dict) or set(settings) - SETTINGS[kind]:
        raise ValueError(f"Unsupported {kind} settings; allowed: {sorted(SETTINGS[kind])}")
    for key, value in settings.items():
        if key == "thinking":
            valid = isinstance(value, bool)
        elif key == "streams":
            valid = (isinstance(value, list) and bool(value) and
                     all(type(v) is int and v > 0 for v in value) and len(set(value)) == len(value))
        elif key in ("repeat", "max_tokens", "new_tokens", "rounds"):
            valid = type(value) is int and value > 0
        elif key in ("warmup", "context_tokens", "prompt_tokens"):
            valid = type(value) is int and value >= 0
        elif key == "timeout":
            valid = type(value) in (int, float) and value > 0
        else:
            valid = isinstance(value, str) and bool(value) and "\0" not in value
        if not valid:
            raise ValueError(f"Invalid {kind}.{key}: {value!r}")
    choices = {"suite": {"all", "code", "devops", "prose"}, "cache_mode": {"cold", "warm"},
               "mode": {"stream", "nonstream", "both"}}
    for key in choices.keys() & settings.keys():
        if settings[key] not in choices[key]:
            raise ValueError(f"Invalid {kind}.{key}")


def resolved_env(recipe, runtime, job):
    env = {k: v for k, v in os.environ.items() if not tuning_key(k) and k not in CONTROLLED}
    env.update(job["env"], RECIPE_HOME=str(runtime), VENV=str(runtime / "venv"),
               EXL3_SRC=str(runtime / "exllamav3"), TABBY_DIR=str(runtime / "tabbyAPI"),
               MODEL_DIR=job["model_path"], HOST="127.0.0.1", PORT="8899",
               DISABLE_AUTH="true", DRY_RUN="0", PYTHONUNBUFFERED="1")
    # Source current recipe defaults; no engine SHA, Tabby ref or version is embedded here.
    data = subprocess.check_output(["bash", "-c", 'set -a; source "$1"; env -0',
                                    "recipe-env", str(recipe / "exllamav3-tabby/env.sh")],
                                   env=env, timeout=30)
    return dict(item.decode().split("=", 1) for item in data.split(b"\0") if item)


def verify_inputs(job, args, identity, env):
    if source_identity(args.recipe, args.runtime) != identity:
        raise ValueError("Recipe/runtime identity changed during the matrix")
    if model_identity(Path(job["model_path"])) != job["model_identity"]:
        raise ValueError("Model files, configuration or loader order changed during the matrix")
    current = resolved_env(args.recipe, args.runtime, job)
    keys = {key for key in set(current) | set(env) if tuning_key(key) or key in {"TABBY_REF", "TABBY_REPO", "PYTHON_BIN"}}
    if any(current.get(key) != env.get(key) for key in keys):
        raise ValueError("Resolved recipe defaults changed during the matrix")


def require_free_port():
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 8899))


def require_owned_listener(pid):
    listeners = set()
    for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1].split(":")[1] == f"{8899:04X}" and fields[3] == "0A":
            listeners.add("socket:[" + fields[9] + "]")
    descriptors = set()
    for file in Path(f"/proc/{pid}/fd").iterdir():
        try:
            descriptors.add(os.readlink(file))
        except FileNotFoundError:
            pass
    if not listeners & descriptors:
        raise RuntimeError("Ready API socket is not owned by the launched server")


def get_json(endpoint):
    request = urllib.request.Request(BASE + endpoint)
    if os.environ.get("API_KEY"):
        request.add_header("Authorization", "Bearer " + os.environ["API_KEY"])
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)


def group_members(process, token):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z" or int(fields[2]) != process.pid or int(fields[3]) != process.pid:
                continue
            if f"QWEN_EXPERIMENT_OWNER={token}".encode() not in (entry / "environ").read_bytes().split(b"\0"):
                raise RuntimeError(f"Refusing to signal unverified process group {process.pid}")
            members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return members


def stop_owned(process, token):
    signals = []
    for sig, seconds in ((signal.SIGTERM, 30), (signal.SIGKILL, 5)):
        if not group_members(process, token):
            break
        os.killpg(process.pid, sig)
        signals.append(sig.name)
        deadline = time.monotonic() + seconds
        while group_members(process, token) and time.monotonic() < deadline:
            process.poll()
            time.sleep(.2)
    if group_members(process, token):
        raise RuntimeError(f"Owned group {process.pid} did not stop")
    return {"exit_code": process.wait(timeout=5), "signals": signals}


class Sampler:
    def __init__(self, output, interval, limit):
        self.output, self.interval, self.limit = output, interval, limit
        self.next, self.samples = 0, []

    def tick(self, phase, pid=None):
        if time.monotonic() < self.next or len(self.samples) >= self.limit:
            return
        self.next = time.monotonic() + self.interval
        row = {"at_utc": stamp(), "phase": phase}
        try:
            wanted = {"MemAvailable", "MemFree", "SwapFree", "SwapTotal"}
            row["system_kib"] = {k: int(v.split()[0]) for k, v in
                (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()) if k in wanted}
            if pid:
                row["server_rss_bytes"] = int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            sample = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
            row["gpu"] = {"exit_code": sample.returncode, "columns": ["used_MiB", "free_MiB", "util_percent", "power_W", "temp_C", "clock_MHz"],
                          "csv": sample.stdout[-4096:], "stderr": sample.stderr[-1024:]}
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            row["sample_error"] = type(exc).__name__
        self.samples.append(row)
        atomic(self.output, {"interval_seconds": self.interval, "sample_limit": self.limit,
                             "limit_reached": len(self.samples) >= self.limit, "samples": self.samples,
                             "note": "Sparse observations; not guaranteed resource peaks."})


def flag_args(settings):
    result = []
    for key, value in settings.items():
        if key == "thinking":
            result += ["--thinking"] if value else []
        else:
            result += ["--" + key.replace("_", "-"), str(value)]
    return result


def commands(job, attempt, recipe, python, model):
    common = ["--base-url", BASE, "--model", model]
    if job.get("response_model"):
        common += ["--response-model", job["response_model"]]
    metadata = ["--metadata", str(attempt / "deployment.json")]
    for index, settings in enumerate(job["bench"], 1):
        name = f"bench-{index}"
        yield name, [python, str(recipe / "bench/bench_v1.py"), *common, *metadata,
                     "--label", job["label"] + "-" + name, *flag_args(settings)]
    if "concurrency" in job:
        settings = dict(job["concurrency"])
        positional = [job["label"], ",".join(map(str, settings.pop("streams"))),
                      str(settings.pop("prompt_tokens")), str(settings.pop("new_tokens")), str(settings.pop("rounds"))]
        yield "concurrency", [python, str(recipe / "bench/concurrency.py"), *positional, *common, *metadata, *flag_args(settings)]
    if "tools" in job:
        yield "tools", [python, str(recipe / "bench/tool_smoke.py"), *common, *flag_args(job["tools"])]


def run_job(job, args, identity, env):
    job_dir = args.output / job["label"]
    job_dir.mkdir(exist_ok=True)
    config = {"job": job, "sources": identity, "resolved_tuning": {k: v for k, v in env.items() if tuning_key(k)},
              "ready_timeout": args.ready_timeout, "client_timeout": args.client_timeout,
              "sample_interval": args.sample_interval, "max_samples": args.max_samples}
    sha = digest(config)
    prior = job_dir / "result.json"
    if prior.exists():
        old = json.loads(prior.read_text())
        if not args.resume or old.get("config_sha256") != sha:
            raise ValueError(f"Refusing existing output or changed configuration: {job_dir}")
        if old.get("finished_at_utc"):
            print(f"Resume: preserved {job['label']} ({old['state']})", flush=True)
            return old
    elif any(job_dir.iterdir()):
        raise ValueError(f"Refusing unrecognized existing output: {job_dir}")
    require_free_port()
    attempt = job_dir / ("attempt-" + uuid.uuid4().hex[:12])
    attempt.mkdir()
    token = uuid.uuid4().hex
    env = dict(env, STATE_DIR=str(attempt / "state"), QWEN_EXPERIMENT_OWNER=token)
    record = {"schema_version": 1, "config_sha256": sha, "configuration": config,
              "started_at_utc": stamp(), "attempt": str(attempt), "state": "loading",
              "clients": [], "passed": False, "diagnostic_only": True}
    def save():
        atomic(attempt / "result.json", record)
        atomic(prior, record)
    save()
    sampler = Sampler(attempt / "resources.json", args.sample_interval, args.max_samples)
    server, server_log, interrupted = None, None, None
    try:
        verify_inputs(job, args, identity, env)
        record["inputs_verified_before_launch_at_utc"] = stamp()
        sampler.tick("before_start")
        server_log = (attempt / "server.log").open("w")
        start = time.monotonic()
        server = subprocess.Popen(["bash", str(args.recipe / "exllamav3-tabby/serve.sh")],
                                  cwd=args.recipe, env=env, stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        record["server_pid"] = server.pid
        save()
        deadline = start + args.ready_timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Server exited during load: {server.returncode}")
            sampler.tick("loading", server.pid)
            try:
                models = get_json("/models")
                if models.get("data"):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Model readiness deadline exceeded")
        require_owned_listener(server.pid)
        record["load_wall_seconds"] = time.monotonic() - start
        record["models"], record["current_model"] = models, get_json("/model")
        model = env.get("SERVED_NAME", "Qwen3.8-Flash-Next-EXL3")
        if [row.get("id") for row in models["data"]] != [model]:
            raise ValueError("Advertised model does not match the requested server alias")
        if record["current_model"].get("id") not in {model, Path(job["model_path"]).name}:
            raise ValueError("Loaded canonical model is not the requested pack")
        deployment = json.loads((attempt / "state/deployment.json").read_text())
        if Path(deployment["model"]["resolved_path"]).resolve() != Path(job["model_path"]):
            raise ValueError("Deployment model path differs from the requested pack")
        for name in ("engine", "server"):
            if deployment[name]["commit"] != identity[name]["commit"] or git_identity(
                    Path(identity[name]["path"])) != identity[name]:
                raise ValueError(f"{name} changed between preflight and load")
        atomic(attempt / "deployment.json", deployment)
        record["deployment_sha256"] = digest(deployment)
        verify_inputs(job, args, identity, env)
        record["inputs_verified_before_measurements_at_utc"] = stamp()
        record["state"] = "measuring"
        save()
        for name, command in commands(job, attempt, args.recipe, str(args.runtime / "venv/bin/python"), model):
            if server.poll() is not None:
                raise RuntimeError("Server exited between measurements")
            destination = attempt / (name + ".json")
            command += ["--output", str(destination)]
            client_record = {"name": name, "command": command, "started_at_utc": stamp(),
                             "output": str(destination), "metadata": str(attempt / "deployment.json")}
            record["clients"].append(client_record)
            save()
            with (attempt / (name + ".log")).open("w") as log:
                client = subprocess.Popen(command, cwd=args.recipe, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                began = time.monotonic()
                try:
                    while client.poll() is None:
                        if time.monotonic() - began > args.client_timeout:
                            client_record["timed_out"] = True
                            break
                        sampler.tick(name, server.pid)
                        time.sleep(.5)
                finally:
                    try:
                        client_record.update(stop_owned(client, token))
                    except Exception as exc:
                        record["cleanup_failed"] = True
                        client_record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        client_record["wall_seconds"] = time.monotonic() - began
            client_record["completed_report"] = destination.is_file() and bool(
                json.loads(destination.read_text()).get("completed_at_utc"))
            save()
        verify_inputs(job, args, identity, env)
        record["inputs_verified_after_measurements_at_utc"] = stamp()
        record["passed"] = bool(record["clients"]) and all(
            row["exit_code"] == 0 and row["completed_report"] and not row.get("timed_out")
            for row in record["clients"])
        record["state"] = "completed"
    except BaseException as exc:
        record["state"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupted = exc
    finally:
        if server is not None:
            record["server_exit_before_cleanup"] = server.poll()
            try:
                record["server_cleanup"] = stop_owned(server, token)
            except Exception as exc:
                record.update(state="cleanup_failed", passed=False, cleanup_error=f"{type(exc).__name__}: {exc}")
        if server_log:
            server_log.close()
        if record.get("cleanup_failed"):
            record.update(state="cleanup_failed", passed=False)
        if record.get("server_exit_before_cleanup") is not None:
            record["passed"] = False
        record["ended_at_utc"] = stamp()
        if record["state"] != "interrupted":
            record["finished_at_utc"] = record["ended_at_utc"]
        save()
    print(f"{job['label']}: {record['state']}, passed={record['passed']}", flush=True)
    if interrupted:
        raise interrupted
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("recipe", "runtime", "jobs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--client-timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=10)
    parser.add_argument("--max-samples", type=int, default=360)
    args = parser.parse_args()
    args.recipe, args.runtime = args.recipe.expanduser().resolve(strict=True), args.runtime.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    if min(args.ready_timeout, args.client_timeout, args.sample_interval, args.max_samples) <= 0:
        parser.error("Timeouts and sampling bounds must be positive")
    raw = json.loads(args.jobs.read_text())
    if not isinstance(raw, list) or not raw:
        parser.error("jobs must be a nonempty JSON array")
    jobs = [normalize(job) for job in raw]
    if len({job["label"] for job in jobs}) != len(jobs):
        parser.error("Job labels must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    def interrupt(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise KeyboardInterrupt(f"Received signal {signum}; stopping owned processes")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (args.output / ".controller.lock").open("a") as output_lock, Path(
            f"/tmp/qwen-experiment-8899-{os.getuid()}.lock").open("a") as port_lock:
        fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(port_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = source_identity(args.recipe, args.runtime)
        environments = [resolved_env(args.recipe, args.runtime, job) for job in jobs]
        records = []
        for job, env in zip(jobs, environments):
            record = run_job(job, args, identity, env)
            records.append(record)
            if record["state"] == "cleanup_failed":
                break  # Ownership/cleanup uncertainty must never overlap the next job.
        return 0 if len(records) == len(jobs) and all(record["passed"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
