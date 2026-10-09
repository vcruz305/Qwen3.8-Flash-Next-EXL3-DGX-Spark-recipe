"""Opt-in observation of eight frozen a70 synthetic held-out/concurrent requests.

Collector/finish callbacks and existing synchronous native phase hooks are observed.
No native token/KV access, new generation awaits, or response rewriting.
"""
from __future__ import annotations
import asyncio
from functools import wraps
import hashlib
import importlib
import inspect
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

TABBY_HEAD = "a70ae1fa9e457e478c3d96bdc84012a3cb331796"
ENGINE_HEAD = "24f0dece34f09c8d1e2359d6b3b3f7befef7331b"
NATIVE_SAMPLE_SHA256 = "177462637da1f61feb1374852e3feb375f26c0de0a58ce18beac24bd09d74323"
EXPECTED_REQUESTS_SHA256 = "74e477ee1917f64264205cd1cac9665058ec2476fe5b49b2289d3fe857bcc52c"
_EXPECTED_RAW = Path(__file__).with_name("expected_requests.json").read_bytes()
if hashlib.sha256(_EXPECTED_RAW).hexdigest() != EXPECTED_REQUESTS_SHA256:
    raise RuntimeError("Exact diagnostic request binding changed")
EXPECTED_REQUESTS = json.loads(_EXPECTED_RAW)["requests"]

METRIC_KEYS = ("stage", "eos", "eos_reason", "prompt_tokens", "cached_tokens",
               "new_tokens", "accepted_draft_tokens", "rejected_draft_tokens",
               "eos_triggering_token_str", "eos_triggering_string")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_new(path, value):
    """Publish a new private JSON file without replacing any existing inode."""
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".strings-observer-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    return value


def matching_request(params, streaming):
    """Match only the retained fields of the eight source-bound synthetic requests."""
    for expected in EXPECTED_REQUESTS:
        request = expected["request"]
        fields = {key: request.get(key) for key in
                  ("messages", "tools", "tool_choice", "parallel_tool_calls", "max_tokens",
                   "temperature", "top_k", "reasoning_budget_tokens", "literal_user_control_tokens")}
        fields.update(top_p=0.95, n=1, stream=bool(streaming))
        if request["stream"] is not bool(streaming):
            continue
        if any(plain(getattr(params, key, None)) != value for key, value in fields.items()):
            continue
        template_vars = plain(getattr(params, "template_vars", None))
        if not isinstance(template_vars, dict) or template_vars.get("enable_thinking") is not request["enable_thinking"]:
            continue
        framework = {"messages": request["messages"], "tools": request["tools"], "functions": None,
                     "add_generation_prompt": True, "tool_choice": request["tool_choice"],
                     "parallel_tool_calls": False}
        special_keys = {"bos_token", "eos_token", "pad_token", "unk_token"}
        if set(template_vars) - ({"enable_thinking"} | set(framework) | special_keys):
            continue
        if any(key in template_vars and template_vars[key] != value for key, value in framework.items()):
            continue
        if any(key in template_vars and not (template_vars[key] is None or isinstance(template_vars[key], (str, int)))
               for key in special_keys):
            continue
        if getattr(params, "functions", None) or getattr(params, "continue_final_message", False) or getattr(params, "response_prefix", None):
            continue
        return expected
    return None


def token_plan_proof(params, prompt):
    """Existing immutable Python tuples only; no native tensor access or encoding."""
    plan = getattr(params, "_literal_user_token_plan", None)
    if plan is None or plan.prompt != prompt:
        return None
    original = plan.original_ids
    changed = plan.changed_ids
    if not isinstance(original, tuple) or not isinstance(changed, tuple):
        return None
    if any(type(value) is not int for values in (original, changed) for value in values):
        return None
    digest = lambda values: sha(json.dumps(list(values), separators=(",", ":")).encode())
    return {"original_prompt_tokens": len(original), "prompt_tokens": len(changed),
            "original_ids_sha256": digest(original), "input_ids_sha256": digest(changed),
            "replacement_count": plan.replacement_count}


def scalar(value):
    """Record only ordinary CPU metadata; never coerce a native tensor."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


class Recorder:
    def __init__(self, directory, label, max_records=8, provenance=None):
        if type(max_records) is not int or not 1 <= max_records <= 8:
            raise ValueError("max_records must be from 1 through 8")
        if not isinstance(label, str) or not re.fullmatch("[a-z0-9][a-z0-9-]{0,63}", label):
            raise ValueError("A simple explicit run label is required")
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700)
        self.label = label
        self.max_records = max_records
        self.active = {}
        self.count = 0
        self.skipped_matching_requests = 0
        self.provenance = provenance or {}
        self.native_audit = None
        self.ended_names = []
        atomic_new(self.directory / "observer-manifest.json", {
            "schema_version": 1, "diagnostic_only": True, "run_label": label,
            "scope": "Eight unchanged a70 held-out/concurrent synthetic requests only. Raw native/backend output and bounded processed-token/phase events; no input hook, output changes or added generation awaits.",
            "max_records": max_records, "source": self.provenance,
            "expected_requests_sha256": EXPECTED_REQUESTS_SHA256,
            "expected_names": [row["name"] for row in EXPECTED_REQUESTS],
            "request_schema_note": "Exact retained request fields and rendered prompt; actual effective top_p0.95 is recorded as a finite builtin scalar.",
        })

    def begin(self, request_id, prompt, params, streaming, start_in_reasoning_mode):
        expected = matching_request(params, streaming)
        if expected is None:
            return None
        if start_in_reasoning_mode is not expected["request"]["enable_thinking"]:
            return None
        if not isinstance(prompt, str) or sha(prompt.encode()) != expected["prompt_sha256"]:
            return None
        proof = token_plan_proof(params, prompt)
        if proof is None or any(proof[key] != expected[key] for key in proof):
            return None
        if request_id in self.active:
            raise RuntimeError("Duplicate active diagnostic request ID")
        if self.count >= self.max_records:
            self.skipped_matching_requests += 1
            return None  # Bound observation without changing later requests.
        if not isinstance(prompt, str):
            raise TypeError("Expected an existing rendered string prompt")
        index = self.count
        self.count += 1
        record = {
            "schema_version": 1, "index": index, "run_label": self.label,
            "request_id": request_id, "streaming_mode": bool(streaming),
            "start_in_reasoning_mode": bool(start_in_reasoning_mode),
            "rendered_prompt": prompt, "rendered_prompt_sha256": sha(prompt.encode()),
            "case_name": expected["name"],
            "wire_request_sha256": expected["request_sha256"],
            "input_plan": proof,
            "matched_request": {"messages": plain(params.messages), "tools": plain(params.tools),
                                "model": scalar(getattr(params, "model", None)),
                                "tool_choice": plain(params.tool_choice),
                                "reasoning_budget_tokens": scalar(getattr(params, "reasoning_budget_tokens", None)),
                                "literal_user_control_tokens": scalar(getattr(params, "literal_user_control_tokens", None)),
                                "parallel_tool_calls": scalar(params.parallel_tool_calls),
                                "max_tokens": scalar(params.max_tokens),
                                "temperature": scalar(params.temperature),
                                "top_k": scalar(params.top_k),
                                "top_p": scalar(params.top_p),
                                "n": scalar(params.n), "stream": bool(streaming),
                                "template_vars": plain(params.template_vars)},
            "producer_events": [], "producer_events_dropped": 0,
            "native_sample_events": [], "native_sample_events_dropped": 0,
            "started_monotonic_ns": time.monotonic_ns(), "raw_finish": None,
            "collector_returned_error": False,
        }
        self.active[request_id] = record
        return record

    def end(self, record, exception_type=None):
        record["native_observer_audit"] = dict(self.native_audit or {})
        record["finished_monotonic_ns"] = time.monotonic_ns()
        record["raised_exception_type"] = exception_type
        record["skipped_matching_requests_so_far"] = self.skipped_matching_requests
        self.active.pop(record["request_id"], None)
        atomic_new(self.directory / f"request-{record['index']:02d}.json", record)
        self.ended_names.append(record["case_name"])
        if len(self.ended_names) == self.max_records and not self.active:
            atomic_new(self.directory / "observer-completion.json", {
                "completed_records": len(self.ended_names), "names": self.ended_names,
                "native_observer_audit": dict(self.native_audit or {}),
                "skipped_matching_requests": self.skipped_matching_requests,
            })


def install_hooks(collector_module, container_class, recorder):
    original_collector = collector_module._chat_stream_collector
    original_finish = container_class.handle_finish_chunk
    signature = inspect.signature(original_collector)

    @wraps(original_collector)
    async def collector(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        record = recorder.begin(values["request_id"], values["prompt"], values["params"],
                                values["streaming_mode"], values["start_in_reasoning_mode"])
        if record is None:
            return await original_collector(*args, **kwargs)
        raised = None
        try:
            result = await original_collector(*args, **kwargs)
            record["collector_returned_error"] = isinstance(result, BaseException)
            return result
        except BaseException as error:
            raised = type(error).__name__
            raise
        finally:
            try:
                recorder.end(record, raised)
            except Exception as error:
                # Missing evidence invalidates the diagnostic, not the original request.
                sys.stderr.write("Strings observer publication failed: " + type(error).__name__ + "\n")

    @wraps(original_finish)
    def finish(self, result, request_id, full_text, label=None):
        record = recorder.active.get(request_id)
        if record is not None:
            if record["raw_finish"] is not None:
                raise RuntimeError("Duplicate backend finish for observed request")
            native_text = result.get("full_completion")
            record["raw_finish"] = {
                "native_full_completion": native_text if isinstance(native_text, str) else None,
                "native_full_completion_present": isinstance(native_text, str),
                "backend_full_response": full_text if isinstance(full_text, str) else None,
                "native_metrics": {key: scalar(result.get(key)) for key in METRIC_KEYS},
                "backend_finish_monotonic_ns": time.monotonic_ns(),
            }
        # Preserve the native call arguments, return identity, exception and timing order.
        value = original_finish(self, result, request_id, full_text, label)
        if record is not None:
            record["raw_finish"]["returned_metrics"] = {
                key: scalar(value.get(key)) for key in
                ("gen_tokens", "finish_reason", "eos_reason", "stop_str",
                 "cached_tokens", "draft_accept", "draft_reject")
            }
        return value

    collector_module._chat_stream_collector = collector
    container_class.handle_finish_chunk = finish
    return {"collector": original_collector, "finish": original_finish}


def install_phase_hooks(collector_module, container_class, job_class, recorder):
    """Observe existing Python phase events for active matched requests only.

    No sequence/tensor reads, decoding calls, model work or awaits. Token IDs
    are Python ints already passed after native sample/stop/rewind processing.
    Rewound, healing and EOS observations are explicitly distinguishable; a
    processed sample is not automatically part of the final accepted output. A
    bounded trace is published with the existing record after collection.
    """
    maximum_events = 2048

    def current(native_job):
        container = getattr(getattr(collector_module, "model", None), "container", None)
        jobs = getattr(container, "active_job_ids", {})
        for request_id, record in recorder.active.items():
            if getattr(jobs.get(request_id), "job", None) is native_job:
                return record
        return None

    def snapshot(native_job):
        budget = getattr(native_job, "token_budget", None)
        guard = budget.get("can_end") if isinstance(budget, dict) else None
        parser = getattr(guard, "_parser", None)
        return {
            "new_tokens": scalar(getattr(native_job, "new_tokens", None)),
            "rq_new_tokens": scalar(getattr(native_job, "rq_new_tokens", None)),
            "checkpoint_rewound": scalar(getattr(native_job, "checkpoint_rewound", None)),
            "filters_suspended": scalar(getattr(native_job, "filters_suspended", None)),
            "held_text": scalar(getattr(native_job, "held_text", None)),
            "held_tokens_shape": list(getattr(getattr(native_job, "held_tokens", None), "shape", ())),
            "filter_count": len(getattr(native_job, "filters", None) or []),
            "budget": None if budget is None else {
                key: scalar(budget.get(key)) for key in
                ("deadline", "injecting", "end_seen", "end_position", "end_token_id")},
            "guard_parser": None if parser is None else {
                "in_reasoning": scalar(parser.in_reasoning),
                "in_tool": scalar(parser.in_tool),
                "pending_characters": len(parser._pending),
            },
        }

    def emit(record, kind, native_job, **data):
        if record is None:
            return
        events = record.setdefault("producer_events", [])
        if len(events) >= maximum_events:
            record["producer_events_dropped"] = record.get("producer_events_dropped", 0) + 1
            return
        events.append({"index": len(events), "kind": kind,
                       "monotonic_ns": time.monotonic_ns(),
                       "state": snapshot(native_job), **data})

    original_advance = job_class._advance_token_budget
    original_guard = job_class._token_budget_can_end
    original_force = job_class.constrain_output_now
    original_phase = container_class.set_generation_phase

    @wraps(original_advance)
    def advance(self, token, eos):
        record = current(self)
        emit(record, "sample_processed_before_budget", self,
             token_id=scalar(token), eos=scalar(eos))
        try:
            return original_advance(self, token, eos)
        finally:
            emit(record, "sample_processed_after_budget", self,
                 token_id=scalar(token), eos=scalar(eos))

    @wraps(original_guard)
    def guard(self, budget):
        record = current(self)
        emit(record, "guard_before", self)
        try:
            result = original_guard(self, budget)
        except BaseException as error:
            emit(record, "guard_error", self, exception_type=type(error).__name__)
            raise
        emit(record, "guard_after", self, allowed=scalar(result))
        return result

    @wraps(original_force)
    def force(self, output):
        record = current(self)
        emit(record, "force_before", self)
        try:
            result = original_force(self, output)
        except BaseException as error:
            emit(record, "force_error", self, exception_type=type(error).__name__)
            raise
        emit(record, "force_after", self)
        return result

    @wraps(original_phase)
    def phase(self, request_id, reasoning):
        active_container = getattr(getattr(collector_module, "model", None), "container", None)
        record = recorder.active.get(request_id) if self is active_container else None
        native_job = getattr(getattr(self, "active_job_ids", {}).get(request_id), "job", None)
        emit(record, "phase_before", native_job, requested_reasoning=scalar(reasoning))
        try:
            result = original_phase(self, request_id, reasoning)
        except BaseException as error:
            emit(record, "phase_error", native_job, exception_type=type(error).__name__)
            raise
        emit(record, "phase_after", native_job,
             requested_reasoning=scalar(reasoning), applied=scalar(result))
        return result

    job_class._advance_token_budget = advance
    job_class._token_budget_can_end = guard
    job_class.constrain_output_now = force
    container_class.set_generation_phase = phase
    return {"advance": original_advance, "guard": original_guard,
            "force": original_force, "phase": original_phase}


class NativeEventSink:
    def __init__(self, record):
        self.record = record

    def event(self, kind, payload):
        events = self.record["native_sample_events"]
        if len(events) >= 2048:
            self.record["native_sample_events_dropped"] += 1
            return
        events.append({"index": len(events), "kind": kind,
                       "monotonic_ns": time.monotonic_ns(), "payload": payload})


def install_native_observation(collector_module, job_class, tokenizer_class, recorder):
    import native_sample_observer
    expected_module = Path(__file__).with_name("native_sample_observer.py").resolve()
    if Path(native_sample_observer.__file__).resolve() != expected_module or sha(expected_module.read_bytes()) != NATIVE_SAMPLE_SHA256:
        raise RuntimeError("Native observer module/source mismatch")
    install_native_sample_hook = native_sample_observer.install_native_sample_hook
    def lookup(native_job):
        container = getattr(getattr(collector_module, "model", None), "container", None)
        jobs = getattr(container, "active_job_ids", {})
        for request_id, record in recorder.active.items():
            if getattr(jobs.get(request_id), "job", None) is native_job:
                return NativeEventSink(record)
        return None
    restore = install_native_sample_hook(job_class, lookup, tokenizer_class=tokenizer_class)
    recorder.native_audit = restore.audit
    return restore


def verified_source(directory, expected):
    directory = Path(directory).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if actual != expected or changes:
        raise RuntimeError("Observer source revision or clean-state mismatch")
    return {"path": str(directory), "commit": actual, "tracked_changes": changes}


def make_run_wrapper(original_run, target_main, activate):
    """Import the backend only after main applied config and allocator options."""
    @wraps(original_run)
    def run(coro, *args, **kwargs):
        code = getattr(coro, "cr_code", None)
        matching = code is not None and code.co_name == "entrypoint_async" and Path(code.co_filename).resolve() == target_main
        if not matching:
            return original_run(coro, *args, **kwargs)

        async def traced_startup():
            try:
                activate()
            except BaseException:
                coro.close()
                raise
            return await coro
        return original_run(traced_startup(), *args, **kwargs)
    return run


def arm_from_environment():
    config_path = Path(os.environ["TABBY_STRINGS_OBSERVER_CONFIG"]).resolve()
    config = json.loads(config_path.read_text())
    tabby = Path(config["tabby_repo"]).resolve()
    engine = Path(config["engine_repo"]).resolve()
    target_main = tabby / "main.py"
    if Path(sys.argv[0]).resolve() != target_main:
        return False
    if config.get("tabby_commit") != TABBY_HEAD or config.get("engine_commit") != ENGINE_HEAD:
        raise RuntimeError("Observer requires the exact frozen 24f0/a70 source pair")
    maximum = config.get("max_records", 8)
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("Invalid observer cap")
    activated = False
    original_run = asyncio.run

    def activate():
        nonlocal activated
        if activated:
            raise RuntimeError("Observer already installed")
        activated = True
        tabby_source = verified_source(tabby, TABBY_HEAD)
        engine_source = verified_source(engine, ENGINE_HEAD)
        collector_module = importlib.import_module("endpoints.OAI.utils.chat_completion")
        model_module = importlib.import_module("backends.exllamav3.model")
        engine_module = importlib.import_module("exllamav3.generator.job")
        tokenizer_module = importlib.import_module("exllamav3.tokenizer.tokenizer")
        if Path(collector_module.__file__).resolve() != tabby / "endpoints/OAI/utils/chat_completion.py":
            raise RuntimeError("Unexpected collector import path")
        if Path(model_module.__file__).resolve() != tabby / "backends/exllamav3/model.py":
            raise RuntimeError("Unexpected Tabby import path")
        if Path(engine_module.__file__).resolve() != engine / "exllamav3/generator/job.py":
            raise RuntimeError("Unexpected engine import path")
        if Path(tokenizer_module.__file__).resolve() != engine / "exllamav3/tokenizer/tokenizer.py":
            raise RuntimeError("Unexpected tokenizer import path")
        provenance = {
            "tabby": tabby_source, "engine": engine_source,
            "observer_sha256": sha(Path(__file__).read_bytes()),
            "expected_requests_sha256": EXPECTED_REQUESTS_SHA256,
            "native_sample_observer_sha256": sha(Path(__file__).with_name("native_sample_observer.py").read_bytes()),
            "sitecustomize_sha256": sha(Path(__file__).with_name("sitecustomize.py").read_bytes()),
            "config_sha256": sha(config_path.read_bytes()),
        }
        recorder = Recorder(config["output_dir"], config["run_label"], maximum, provenance)
        install_hooks(collector_module, model_module.ExllamaV3Container, recorder)
        install_phase_hooks(collector_module, model_module.ExllamaV3Container, engine_module.Job, recorder)
        install_native_observation(collector_module, engine_module.Job, tokenizer_module.Tokenizer, recorder)
        asyncio.run = original_run

    asyncio.run = make_run_wrapper(original_run, target_main, activate)
    return True
