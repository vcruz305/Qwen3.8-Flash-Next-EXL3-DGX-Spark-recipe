"""Opt-in observation of the frozen synthetic record_strings request.

Only collector entry/exit and the existing backend finish callback are wrapped.
No native token/KV access, new generation awaits, or response rewriting.
"""
from __future__ import annotations
import asyncio
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

TABBY_HEAD = "3adc9813f30c4b92d110e7e0235129031dae1c25"
ENGINE_HEAD = "9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2"
STRING_ARGS = {"number_text": "123", "bool_text": "true",
               "json_text": '{"nested": [1, false]}',
               "tag_text": "<think>literal</think>"}
USER_MESSAGE = "Call record_strings once with these exact string values: " + json.dumps(STRING_ARGS)
TOOLS = [{"type": "function", "function": {
    "name": "record_strings",
    "description": "Record each value exactly as a string; do not interpret its contents.",
    "parameters": {"type": "object",
                   "properties": {key: {"type": "string"} for key in STRING_ARGS},
                   "required": list(STRING_ARGS), "additionalProperties": False}}}]
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
    """Require the unchanged synthetic client request, in either stream mode."""
    expected = {
        "messages": [{"role": "user", "content": USER_MESSAGE}],
        "tools": TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "max_tokens": 1024,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "n": 1,
    }
    if any(plain(getattr(params, key, None)) != value for key, value in expected.items()):
        return False
    template_vars = plain(getattr(params, "template_vars", None))
    if not isinstance(template_vars, dict) or template_vars.get("enable_thinking") is not False:
        return False
    # apply_chat_template and format_messages_with_template mutate these fields
    # before creating the collector's deep request copy. Validate their actual
    # augmented state as well as the pre-render request form used by CPU clients.
    framework = {"messages": expected["messages"], "tools": TOOLS, "functions": None,
                 "add_generation_prompt": True, "tool_choice": "auto",
                 "parallel_tool_calls": True}
    special_keys = {"bos_token", "eos_token", "pad_token", "unk_token"}
    if set(template_vars) - ({"enable_thinking"} | set(framework) | special_keys):
        return False
    if any(key in template_vars and template_vars[key] != value for key, value in framework.items()):
        return False
    if any(key in template_vars and not (template_vars[key] is None or isinstance(template_vars[key], (str, int)))
           for key in special_keys):
        return False
    if getattr(params, "functions", None):
        return False
    if getattr(params, "stream", None) is not bool(streaming):
        return False
    return True


def scalar(value):
    """Record only ordinary CPU metadata; never coerce a native tensor."""
    return value if value is None or type(value) in (str, int, float, bool) else None


class Recorder:
    def __init__(self, directory, label, max_records=2, provenance=None):
        if type(max_records) is not int or not 1 <= max_records <= 2:
            raise ValueError("max_records must be 1 or 2")
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
        atomic_new(self.directory / "observer-manifest.json", {
            "schema_version": 1, "diagnostic_only": True, "run_label": label,
            "scope": "Exact retained fields of the original record_strings synthetic request only; raw native full_completion, backend full_response, and exact rendered prompt. No token/KV reads, response changes, or added generation awaits.",
            "max_records": max_records, "source": self.provenance,
            "request_schema_note": "Tabby does not retain the wire seed field. Greedy request temperature/top_k and all scope-defining message/schema fields are matched.",
            "expected_request": {"messages": [{"role": "user", "content": USER_MESSAGE}],
                                 "tools": TOOLS, "tool_choice": "auto",
                                 "template_vars": {"enable_thinking": False}},
        })

    def begin(self, request_id, prompt, params, streaming, start_in_reasoning_mode):
        if not matching_request(params, streaming):
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
            "matched_request": {"messages": plain(params.messages), "tools": plain(params.tools),
                                "model": scalar(getattr(params, "model", None)),
                                "tool_choice": params.tool_choice,
                                "template_vars": plain(params.template_vars)},
            "started_monotonic_ns": time.monotonic_ns(), "raw_finish": None,
            "collector_returned_error": False,
        }
        self.active[request_id] = record
        return record

    def end(self, record, exception_type=None):
        record["finished_monotonic_ns"] = time.monotonic_ns()
        record["raised_exception_type"] = exception_type
        record["skipped_matching_requests_so_far"] = self.skipped_matching_requests
        self.active.pop(record["request_id"], None)
        atomic_new(self.directory / f"request-{record['index']:02d}.json", record)


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
        raise RuntimeError("Observer requires the exact frozen 9c/3adc source pair")
    maximum = config.get("max_records", 2)
    if type(maximum) is not int or not 1 <= maximum <= 2:
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
        if Path(collector_module.__file__).resolve() != tabby / "endpoints/OAI/utils/chat_completion.py":
            raise RuntimeError("Unexpected collector import path")
        if Path(model_module.__file__).resolve() != tabby / "backends/exllamav3/model.py":
            raise RuntimeError("Unexpected Tabby import path")
        if Path(engine_module.__file__).resolve() != engine / "exllamav3/generator/job.py":
            raise RuntimeError("Unexpected engine import path")
        provenance = {
            "tabby": tabby_source, "engine": engine_source,
            "observer_sha256": sha(Path(__file__).read_bytes()),
            "sitecustomize_sha256": sha(Path(__file__).with_name("sitecustomize.py").read_bytes()),
            "config_sha256": sha(config_path.read_bytes()),
        }
        recorder = Recorder(config["output_dir"], config["run_label"], maximum, provenance)
        install_hooks(collector_module, model_module.ExllamaV3Container, recorder)
        asyncio.run = original_run

    asyncio.run = make_run_wrapper(original_run, target_main, activate)
    return True
