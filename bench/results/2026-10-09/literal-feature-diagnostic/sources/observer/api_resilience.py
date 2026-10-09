#!/usr/bin/env python3
"""Short, serial TabbyAPI compatibility/recovery checks; never execute a tool.

Run only between benchmarks. Uses stdlib HTTP, actual usage, and captured errors.
The raw prompt is verified as one token with BOS disabled before generation.
Example:
  python bench/api_resilience.py --model Qwen3.8-Flash-Next-EXL3 \
      --metadata state/deployment.json --output results/api-resilience.json
Groups: raw,aliases,forcing,nullable,invalid,disconnect. No automatic retries.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


class CheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


class ReportWriter:
    """Claim a new report atomically, then replace only this writer's snapshots."""

    def __init__(self, path):
        self.path = Path(path)
        self.identity = None

    def write(self, report):
        raw = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if self.identity is None:
                # Unlike exists()+replace(), this cannot race another new run,
                # and it also refuses a dangling symlink or directory target.
                os.link(temporary, self.path)
            else:
                current = self.path.lstat()
                if (current.st_dev, current.st_ino) != self.identity:
                    raise FileExistsError(
                        f"Report was replaced by another writer: {self.path}"
                    )
                os.replace(temporary, self.path)
            current = self.path.lstat()
            self.identity = (current.st_dev, current.st_ino)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def read_metadata(path):
    """Load the caller's deployment JSON and record exactly which file was used."""
    if path is None:
        return {}, None
    source = Path(path)
    raw = source.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("--metadata must contain a JSON object")
    return value, {
        "path": str(source.absolute()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def sse_events(response):
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif not line.startswith(":"):
            field, sep, value = line.partition(":")
            if field == "data":
                data.append(value[1:] if sep and value.startswith(" ") else value)
    if data:
        yield "\n".join(data)


def visible(frame):
    for choice in frame.get("choices", []):
        delta = choice.get("delta") or {}
        if (
            choice.get("text")
            or delta.get("content")
            or delta.get("reasoning_content")
            or delta.get("tool_calls")
        ):
            return True
    return False


class Client:
    def __init__(self, base, key=None, timeout=45, opener=None):
        self.base = base.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.opener = opener or urllib.request.urlopen
        self.traces = []

    def request(self, path, payload=None, abort_after_output=False, root=False):
        streaming = bool(payload and payload.get("stream"))
        headers = {
            "Content-Type": "application/json",
            "Connection": "close",
            "Accept": "text/event-stream" if streaming else "application/json",
            "User-Agent": "qwen-spark-api-resilience/1",
        }
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        base = self.base.removesuffix("/v1") if root else self.base
        body = (
            json.dumps(payload, ensure_ascii=False).encode()
            if payload is not None
            else None
        )
        request = urllib.request.Request(base + path, data=body, headers=headers)
        trace = {
            "index": len(self.traces),
            "path": path,
            "request": payload,
            "status": None,
            "stream": streaming,
            "frames": [],
            "done": False,
            "client_aborted": False,
        }
        self.traces.append(trace)
        start = time.perf_counter()
        try:
            with self.opener(request, timeout=self.timeout) as response:
                trace["status"] = response.status
                trace["content_type"] = response.headers.get("Content-Type", "")
                if streaming:
                    require(
                        "text/event-stream" in trace["content_type"],
                        "Expected SSE Content-Type",
                    )
                    for event in sse_events(response):
                        if event == "[DONE]":
                            trace["done"] = True
                            break
                        obj = json.loads(event)
                        require(isinstance(obj, dict), "SSE data must be a JSON object")
                        trace["frames"].append(obj)
                        if obj.get("error"):
                            trace["server_error"] = obj["error"]
                            break
                        if abort_after_output and visible(obj):
                            already_finished = any(
                                c.get("finish_reason") for c in obj.get("choices", [])
                            )
                            if not already_finished:
                                trace["client_aborted"] = True
                                trace["abort_after_sse_events"] = len(trace["frames"])
                                break
                else:
                    trace["body"] = json.load(response)
        except urllib.error.HTTPError as exc:
            trace["status"] = exc.code
            with exc:
                raw = exc.read(65536).decode("utf-8", "replace")
            try:
                trace["body"] = json.loads(raw)
            except ValueError:
                trace["body_text"] = raw
        except Exception as exc:  # noqa: BLE001 - retain unexpected protocol failures for recovery checks
            trace["transport_or_protocol_error"] = f"{type(exc).__name__}: {exc}"
        trace["wall_seconds"] = time.perf_counter() - start
        return trace


def json_body(trace):
    require(
        not trace.get("transport_or_protocol_error"),
        str(trace.get("transport_or_protocol_error")),
    )
    require(trace["status"] == 200, f"HTTP {trace['status']}: {trace.get('body')}")
    obj = trace.get("body")
    require(isinstance(obj, dict), "Expected JSON object")
    require(not obj.get("error"), f"Server error: {obj.get('error')}")
    return obj


def assembled(trace, expected_model, raw=False):
    require(
        not trace.get("transport_or_protocol_error"),
        str(trace.get("transport_or_protocol_error")),
    )
    require(trace["status"] == 200, f"HTTP {trace['status']}: {trace.get('body')}")
    require(not trace.get("server_error"), f"SSE error: {trace.get('server_error')}")
    frames = trace["frames"] if trace["stream"] else [trace.get("body")]
    require(bool(frames), "No completion response")
    if trace["stream"]:
        require(trace["done"], "SSE ended without [DONE]")
    text, reasoning, calls = [], [], {}
    role = finish = usage = response_id = None
    for obj in frames:
        require(isinstance(obj, dict), "Completion must be a JSON object")
        require(not obj.get("error"), f"Server error: {obj.get('error')}")
        require(
            obj.get("model") == expected_model,
            f"Standard model field {obj.get('model')!r}, expected {expected_model!r}",
        )
        require(isinstance(obj.get("id"), str) and obj["id"], "Response has no ID")
        require(
            response_id is None or response_id == obj["id"], "SSE response ID changed"
        )
        response_id = obj["id"]
        if obj.get("usage") is not None:
            usage = obj["usage"]
        choices = obj.get("choices")
        require(
            isinstance(choices, list) and len(choices) <= 1,
            "Expected one choice (or a usage-only chunk)",
        )
        for choice in choices:
            require(
                type(choice.get("index")) is int and choice["index"] == 0,
                "Expected choice index zero",
            )
            delta = (
                choice.get("delta", {})
                if trace["stream"]
                else choice.get("message", {})
            )
            require(isinstance(delta, dict), "Invalid assistant delta/message")
            if delta.get("role"):
                require(delta["role"] == "assistant", "Response role is not assistant")
                role = delta["role"]
            content = choice.get("text", "") if raw else delta.get("content") or ""
            thought = delta.get("reasoning_content") or ""
            incoming_calls = delta.get("tool_calls") or []
            if not raw and (content or thought or incoming_calls):
                require(
                    role == "assistant",
                    "Assistant role missing before generated output",
                )
            require(
                isinstance(content, str) and isinstance(thought, str),
                "Invalid content/reasoning type",
            )
            if finish is not None:
                require(
                    not (content or thought or incoming_calls),
                    "Output continued after finish_reason",
                )
            text.append(content)
            reasoning.append(thought)
            for i, incoming in enumerate(incoming_calls):
                index = incoming.get("index") if trace["stream"] else i
                require(
                    isinstance(index, int)
                    and not isinstance(index, bool)
                    and index >= 0,
                    "Invalid tool index",
                )
                call = calls.setdefault(
                    index,
                    {
                        "id": None,
                        "type": None,
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if incoming.get("id"):
                    require(
                        call["id"] in (None, incoming["id"]), "Tool call ID changed"
                    )
                    call["id"] = incoming["id"]
                if incoming.get("type"):
                    require(incoming["type"] == "function", "Unexpected tool type")
                    call["type"] = "function"
                function = incoming.get("function") or {}
                for field in ("name", "arguments"):
                    value = function.get(field, "")
                    require(
                        isinstance(value, str),
                        f"Tool function.{field} must be a string",
                    )
                    call["function"][field] += value
            if choice.get("finish_reason") is not None:
                require(
                    finish is None or finish == choice["finish_reason"],
                    "finish_reason changed",
                )
                finish = choice["finish_reason"]
    require(
        finish in ("stop", "length", "tool_calls"),
        f"Missing/unexpected finish_reason: {finish!r}",
    )
    if not raw:
        require(role == "assistant", "Assistant role missing from completion")
    require(isinstance(usage, dict), "Missing actual usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"Invalid usage.{field}: {value!r}",
        )
    require(
        usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0,
        "No prompt or completion tokens",
    )
    require(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        "Usage total does not add up",
    )
    require(
        usage["completion_tokens"] <= trace["request"]["max_tokens"],
        "Token budget exceeded",
    )
    require(
        sorted(calls) == list(range(len(calls))), "Tool call indices are not contiguous"
    )
    require(
        all(call["type"] == "function" for call in calls.values()),
        "Missing tool call type",
    )
    ids = [call["id"] for call in calls.values()]
    require(
        all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids),
        "Missing/duplicate tool call IDs",
    )
    return {
        "id": response_id,
        "model": expected_model,
        "role": role,
        "content": "".join(text),
        "reasoning_content": "".join(reasoning),
        "tool_calls": list(calls.values()),
        "finish_reason": finish,
        "usage": usage,
        "wall_seconds": trace["wall_seconds"],
    }


def function(name, properties=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Synthetic diagnostic; never executed.",
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(properties or {}),
                "additionalProperties": False,
            },
        },
    }


NULLABLE_VALUES = {
    "numeric_text": "123",
    "boolean_text": "true",
    "json_text": "[1,2]",
    "code": "    return 1\n",
    "actual_null": None,
}
GROUPS = ("raw", "aliases", "forcing", "nullable", "invalid", "disconnect")


class Diagnostic:
    report_kind = "tabby_api_resilience"

    def __init__(
        self,
        client,
        model,
        one_token_text="X",
        output=None,
        settle_seconds=2,
        *,
        metadata=None,
        metadata_source=None,
        label="api-resilience",
    ):
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a JSON object")
        self.client, self.model, self.prompt = client, model, one_token_text
        self.output, self.settle_seconds = output, settle_seconds
        self.writer = ReportWriter(output) if output is not None else None
        self.canonical = None
        self.report = {
            "schema_version": 1,
            "kind": self.report_kind,
            "label": label,
            "client": {"python": platform.python_version()},
            "provenance": metadata or {},
            "metadata_source": metadata_source,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "base_url": client.base,
            "requested_model": model,
            "cases": [],
            "requests": client.traces,
            "scope": "Serial functional checks; timings are diagnostic, not benchmark results.",
        }

    def save(self):
        self.report["summary"] = dict(
            Counter(case["status"] for case in self.report["cases"])
        )
        if self.writer is not None:
            self.writer.write(self.report)

    def case(self, name, callback):
        start = len(self.client.traces)
        try:
            result = callback()
            entry = {"name": name, "status": "pass", "result": result}
        except Exception as exc:  # noqa: BLE001 - a failed case must not skip the remaining checks
            entry = {
                "name": name,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
        entry["request_indices"] = list(range(start, len(self.client.traces)))
        self.report["cases"].append(entry)
        print(
            f"{name}: {entry['status']}"
            + (": " + entry["error"] if entry.get("error") else ""),
            file=sys.stderr,
            flush=True,
        )
        self.save()
        return entry["status"] == "pass"

    def preflight(self):
        current = json_body(self.client.request("/model"))
        self.canonical = current["id"]
        models = json_body(self.client.request("/models"))
        advertised = [item["id"] for item in models.get("data", [])]
        if self.model is None:
            require(len(advertised) == 1, f"Pass --model; advertised IDs: {advertised}")
            self.model = advertised[0]
        require(
            self.model in advertised or self.model == self.canonical,
            "Requested model is not advertised/current",
        )
        tokenized = json_body(
            self.client.request(
                "/token/encode",
                {
                    "text": self.prompt,
                    "add_bos_token": False,
                    "encode_special_tokens": False,
                },
            )
        )
        require(
            tokenized.get("length") == 1 and len(tokenized.get("tokens", [])) == 1,
            f"Raw prompt is not exactly one token: {tokenized}",
        )
        self.report.update(
            requested_model=self.model,
            canonical_model=self.canonical,
            advertised_models=advertised,
            one_token_prompt=self.prompt,
            verified_prompt_tokens=tokenized["tokens"],
        )
        return {"current_model": self.canonical, "one_token_id": tokenized["tokens"][0]}

    def raw_payload(self, stream=False, model=None, tokens=8):
        payload = {
            "model": model or self.model,
            "prompt": self.prompt,
            "add_bos_token": False,
            "stream": stream,
            "max_tokens": tokens,
            "min_tokens": tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "stop": [],
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def chat_payload(self, stream=False, model=None):
        payload = {
            "model": model or self.model,
            "messages": [{"role": "user", "content": "Reply with the word READY."}],
            "stream": stream,
            "max_tokens": 12,
            "enable_thinking": False,
            "temperature": 0,
            "top_k": 1,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def healthy(self):
        result = assembled(
            self.client.request("/chat/completions", self.chat_payload()), self.model
        )
        require(result["finish_reason"] == "stop", "Recovery did not end with stop")
        require(not result["tool_calls"], "Recovery leaked a tool call")
        require(
            result["content"].strip() == "READY",
            f"Recovery sentinel mismatch: {result['content']!r}",
        )
        return result

    def unchanged_model(self):
        current = json_body(self.client.request("/model"))["id"]
        require(
            current == self.canonical,
            f"Loaded model changed: {self.canonical!r} -> {current!r}",
        )
        return current

    def raw(self):
        def run(stream):
            result = assembled(
                self.client.request("/completions", self.raw_payload(stream)),
                self.model,
                raw=True,
            )
            require(
                result["usage"]["prompt_tokens"] == 1,
                "Raw request did not exercise the one-token prompt path",
            )
            require(
                result["usage"]["completion_tokens"] == 8
                and result["finish_reason"] == "length",
                "Fixed eight-token raw generation did not finish at its token budget",
            )
            return result

        for stream in (False, True):
            self.case(
                f"raw_one_token_{'stream' if stream else 'nonstream'}",
                lambda stream=stream: run(stream),
            )

    def aliases(self):
        self.case(
            "canonical_model_alias",
            lambda: assembled(
                self.client.request(
                    "/completions", self.raw_payload(model=self.canonical, tokens=4)
                ),
                self.canonical,
                raw=True,
            ),
        )
        unknown = "__tabby_diagnostic_nonexistent_model_7f91b3__"
        require(
            unknown not in self.report["advertised_models"],
            "Diagnostic unknown model unexpectedly exists",
        )

        def run(path, stream):
            raw = path == "/completions"
            payload = (
                self.raw_payload(stream, unknown)
                if raw
                else self.chat_payload(stream, unknown)
            )
            trace = self.client.request(path, payload)
            if trace["status"] in (400, 401, 403, 404, 422):
                require(
                    trace.get("body") or trace.get("body_text"),
                    "Unknown-model rejection has no diagnostic",
                )
                return {
                    "behavior": "rejected",
                    "status": trace["status"],
                    "error": trace.get("body", trace.get("body_text")),
                }
            result = assembled(trace, self.canonical, raw=raw)
            return {
                "behavior": "ignored_name_with_actual_model_reported",
                "completion": result,
            }

        for path in ("/completions", "/chat/completions"):
            for stream in (False, True):
                self.case(
                    f"unknown_{'raw' if path == '/completions' else 'chat'}_{'stream' if stream else 'nonstream'}",
                    lambda path=path, stream=stream: run(path, stream),
                )
        self.case("alias_group_model_unchanged", self.unchanged_model)

    def check_call(
        self, payload, expected_names, expected_args, require_reasoning=False
    ):
        result = assembled(
            self.client.request("/chat/completions", payload), self.model
        )
        require(
            result["finish_reason"] == "tool_calls",
            "Forced call did not complete with tool_calls",
        )
        require(len(result["tool_calls"]) == 1, "Expected one tool call")
        call = result["tool_calls"][0]
        require(
            call["function"]["name"] in expected_names,
            f"Wrong function: {call['function']['name']}",
        )
        args = json.loads(call["function"]["arguments"])
        require(
            isinstance(args, dict) and args == expected_args,
            f"Argument mismatch: actual={args!r}, expected={expected_args!r}",
        )
        for key, expected in expected_args.items():
            require(type(args[key]) is type(expected), f"Wrong JSON type for {key}")
        if require_reasoning:
            require(
                bool(result["reasoning_content"].strip()),
                "No reasoning output; reasoning phase was not exercised",
            )
        return result

    def forcing(self):
        for choice in ("required", {"type": "function", "function": {"name": "ping"}}):
            for stream in (False, True):
                named = isinstance(choice, dict)
                payload = self.chat_payload(stream)
                payload.update(
                    tools=[function("ping"), function("alternate_ping")],
                    tool_choice=choice,
                    parallel_tool_calls=False,
                    enable_thinking=True,
                    reasoning_budget_tokens=24,
                    max_tokens=128,
                )
                payload["messages"] = [
                    {
                        "role": "user",
                        "content": "Think briefly, then call alternate_ping exactly once. It takes no arguments.",
                    }
                ]
                allowed = {"ping"} if named else {"ping", "alternate_ping"}
                self.case(
                    f"reasoning_{'named' if named else 'required'}_{'stream' if stream else 'nonstream'}",
                    lambda payload=payload, allowed=allowed: self.check_call(
                        payload, allowed, {}, True
                    ),
                )

    def nullable(self):
        for stream in (False, True):
            payload = self.chat_payload(stream)
            properties = {key: {"type": ["string", "null"]} for key in NULLABLE_VALUES}
            payload.update(
                tools=[function("echo_nullable", properties)],
                tool_choice="required",
                parallel_tool_calls=False,
                max_tokens=224,
            )
            payload["messages"] = [
                {
                    "role": "user",
                    "content": "Call echo_nullable once with this exact JSON object. "
                    "Preserve each string literally. code begins with four spaces and ends with one newline. actual_null is JSON null. "
                    + json.dumps(NULLABLE_VALUES, ensure_ascii=False),
                }
            ]
            self.case(
                f"nullable_strings_{'stream' if stream else 'nonstream'}",
                lambda payload=payload: self.check_call(
                    payload, {"echo_nullable"}, NULLABLE_VALUES
                ),
            )

    def invalid(self):
        wrong_name = self.chat_payload()
        wrong_name.update(
            tools=[function("ping")],
            tool_choice={"type": "function", "function": {"name": "absent"}},
        )
        negative_budget = self.raw_payload()
        negative_budget["max_tokens"] = -1
        bad_grammar = self.raw_payload()
        bad_grammar["grammar_string"] = "start: nonexistent_diagnostic_rule"
        for name, path, payload, statuses in [
            ("invalid_named_tool", "/chat/completions", wrong_name, {400}),
            ("invalid_token_budget", "/completions", negative_budget, {400, 422}),
            ("invalid_output_grammar", "/completions", bad_grammar, {400, 422}),
        ]:

            def run(path=path, payload=payload, statuses=statuses):
                trace = self.client.request(path, payload)
                recovery = self.healthy()
                require(
                    trace["status"] in statuses,
                    f"Expected client error {statuses}, got HTTP {trace['status']}: {trace.get('body')}",
                )
                require(
                    trace.get("body") or trace.get("body_text"),
                    "Client error lacks a diagnostic",
                )
                return {
                    "error_status": trace["status"],
                    "error": trace.get("body", trace.get("body_text")),
                    "recovery": recovery,
                }

            self.case(name + "_then_healthy", run)

    def disconnect(self):
        def run():
            payload = self.chat_payload(stream=True)
            payload.update(max_tokens=128, min_tokens=64)
            payload["messages"] = [
                {
                    "role": "user",
                    "content": "Write the integers from 1 to 100 separated by spaces. Continue the sequence.",
                }
            ]
            trace = self.client.request(
                "/chat/completions", payload, abort_after_output=True
            )
            recovery = self.healthy()
            require(
                trace["client_aborted"],
                "No early disconnect after visible output before finish_reason",
            )
            require(
                not trace["done"],
                "Disconnect request completed before it could be interrupted",
            )
            if self.settle_seconds:
                time.sleep(self.settle_seconds)
            health = json_body(self.client.request("/health", root=True))
            require(
                health.get("status") == "healthy",
                f"Service is unhealthy after disconnect: {health}",
            )
            self.unchanged_model()
            return {
                "aborted_request_index": trace["index"],
                "events_before_close": trace["abort_after_sse_events"],
                "recovery": recovery,
                "health": health,
                "scope": "Observed client close and successful subsequent generation; server cancellation timing is available in server logs.",
            }

        self.case("early_disconnect_then_healthy", run)

    def run(self, groups):
        self.report["selected_groups"] = list(groups)
        # Fail before any HTTP request if a previous report owns this path.
        self.save()
        if self.case("preflight_one_token_and_model", self.preflight):
            for group in groups:
                try:
                    getattr(self, group)()
                except Exception as exc:  # noqa: BLE001 - retain evidence and continue independent groups
                    self.report["cases"].append(
                        {
                            "name": group,
                            "status": "fail",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    self.save()
        self.report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.save()
        return not any(case["status"] == "fail" for case in self.report["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8899/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument(
        "--api-key-env",
        default="TABBY_API_KEY",
        help="Environment variable containing the API key; the key is never recorded",
    )
    parser.add_argument("--one-token-text", default="X")
    parser.add_argument(
        "--cases",
        default=",".join(GROUPS),
        help="Comma-separated groups: " + ",".join(GROUPS),
    )
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2,
        help="Short post-recovery wait before final health check",
    )
    parser.add_argument(
        "--metadata", type=Path, help="JSON deployment/provenance snapshot"
    )
    parser.add_argument("--label", default="api-resilience")
    parser.add_argument(
        "--output",
        required=True,
        help="New JSON report path; existing files are refused",
    )
    args = parser.parse_args()
    groups = list(
        dict.fromkeys(part.strip() for part in args.cases.split(",") if part.strip())
    )
    if not groups or any(group not in GROUPS for group in groups):
        parser.error("Unknown/empty --cases selection")
    if not 0 < args.timeout <= 300 or not 0 <= args.settle_seconds <= 10:
        parser.error("--timeout must be (0,300]; --settle-seconds must be [0,10]")
    try:
        metadata, metadata_source = read_metadata(args.metadata)
        client = Client(args.base_url, os.environ.get(args.api_key_env), args.timeout)
        diagnostic = Diagnostic(
            client,
            args.model,
            args.one_token_text,
            args.output,
            args.settle_seconds,
            metadata=metadata,
            metadata_source=metadata_source,
            label=args.label,
        )
        okay = diagnostic.run(groups)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": args.output, "summary": diagnostic.report["summary"]}))
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
