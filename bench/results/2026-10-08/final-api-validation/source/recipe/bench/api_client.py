#!/usr/bin/env python3
"""Small, dependency-free OpenAI client shared by the recipe's validation tools.

SSE chunks are transport events, not tokens. Token counts and server timings come
only from response usage; missing usage never becomes the requested max_tokens.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any


class ApiError(RuntimeError):
    """An HTTP, protocol, or response integrity failure."""


def request(base: str, path: str, key: str | None, payload: dict | None = None):
    headers = {"Content-Type": "application/json", "User-Agent": "qwen-spark-recipe-bench/2"}
    if key:
        headers["Authorization"] = "Bearer " + key
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers)


def open_request(req, timeout: float):
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode("utf-8", "replace")
        raise ApiError(f"HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ApiError(f"Request failed: {exc}") from exc


def json_request(base, path, key=None, payload=None, timeout=60):
    with open_request(request(base, path, key, payload), timeout) as response:
        try:
            value = json.load(response)
        except (ValueError, UnicodeError) as exc:
            raise ApiError("Server returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise ApiError("Expected a JSON object")
    if value.get("error"):
        raise ApiError(f"Server error: {value['error']}")
    return value


def resolve_model(base: str, key: str | None, requested: str | None = None) -> str:
    models = json_request(base, "/models", key).get("data")
    if not isinstance(models, list):
        raise ApiError("/models did not return a data list")
    names = [m["id"] for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)]
    if requested:
        if requested not in names:
            raise ApiError(f"Requested model {requested!r} is not advertised; available: {names}")
        return requested
    if len(names) != 1:
        raise ApiError(f"Expected one model, got {names}; pass --model explicitly")
    return names[0]


def iter_sse(response):
    """Yield complete SSE data events, including multiline data and comment pings."""
    data = []
    for raw in response:
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeError as exc:
            raise ApiError("SSE stream contains invalid UTF-8") from exc
        if line == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith(":"):
            continue
        field, sep, value = line.partition(":")
        if field == "data":
            if sep and value.startswith(" "):
                value = value[1:]
            data.append(value)
    if data:
        yield "\n".join(data)


def _count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ApiError(f"usage.{field} must be a nonnegative integer; got {value!r}")
    return value


def _finite_rate(value):
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    return rate if math.isfinite(rate) and rate >= 0 else None


def _merge_tools(calls: dict, deltas: list):
    for delta in deltas:
        idx = delta.get("index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
            raise ApiError(f"Invalid tool-call stream index: {idx!r}")
        call = calls.setdefault(idx, {"id": "", "type": "function",
                                      "function": {"name": "", "arguments": ""}})
        if delta.get("id"):
            if call["id"] and call["id"] != delta["id"]:
                raise ApiError(f"Tool-call id changed at index {idx}")
            call["id"] = delta["id"]
        if delta.get("type"):
            call["type"] = delta["type"]
        function = delta.get("function") or {}
        for field in ("name", "arguments"):
            fragment = function.get(field)
            if fragment is not None:
                if not isinstance(fragment, str):
                    raise ApiError(f"Tool-call {field} fragment must be a string")
                call["function"][field] += fragment


def perform(base: str, key: str | None, payload: dict, path="/chat/completions",
            timeout=1800, require_usage=True, expected_response_model=None) -> dict:
    """Issue a completion and return validated transport metrics plus its message."""
    streaming = payload.get("stream", False)
    start = time.perf_counter()
    first_output = last_output = None
    usage = timings = None
    finish_reason = response_model = None
    text, reasoning, tool_deltas = [], [], {}
    chunks = 0
    done = not streaming
    if streaming:
        with open_request(request(base, path, key, payload), timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                raise ApiError(f"Expected SSE, received {content_type!r}")
            for event in iter_sse(response):
                if event == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(event)
                except ValueError as exc:
                    raise ApiError(f"Invalid SSE JSON: {event[:300]!r}") from exc
                if not isinstance(obj, dict):
                    raise ApiError("Expected an SSE JSON object")
                if obj.get("error"):
                    raise ApiError(f"Streaming error: {obj['error']}")
                chunks += 1
                if obj.get("model"):
                    if response_model and obj["model"] != response_model:
                        raise ApiError("Response model changed during streaming")
                    response_model = obj["model"]
                if obj.get("usage") is not None:
                    usage = obj["usage"]
                if obj.get("timings") is not None:
                    timings = obj["timings"]
                for choice in obj.get("choices") or []:
                    if choice.get("index", 0) != 0:
                        raise ApiError("Benchmark supports n=1 only")
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or choice.get("text") or ""
                    think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    calls = delta.get("tool_calls") or []
                    if content or think or calls:
                        now = time.perf_counter()
                        first_output = first_output if first_output is not None else now
                        last_output = now
                    text.append(content)
                    reasoning.append(think)
                    _merge_tools(tool_deltas, calls)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
        message = {"role": "assistant", "content": "".join(text) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_deltas:
            indices = sorted(tool_deltas)
            if indices != list(range(len(indices))):
                raise ApiError(f"Tool-call indices are not contiguous: {indices}")
            message["tool_calls"] = [tool_deltas[i] for i in indices]
    else:
        obj = json_request(base, path, key, payload, timeout)
        choices = obj.get("choices") or []
        if len(choices) != 1:
            raise ApiError("Expected exactly one completion choice")
        choice = choices[0]
        message = choice.get("message") or {"role": "assistant", "content": choice.get("text")}
        finish_reason = choice.get("finish_reason")
        usage, timings = obj.get("usage"), obj.get("timings")
        response_model = obj.get("model")
    end = time.perf_counter()
    if not done:
        raise ApiError("SSE stream ended without [DONE]")
    if finish_reason is None:
        raise ApiError("Completion has no finish_reason")
    expected_model = expected_response_model or payload.get("model")
    if response_model and response_model != expected_model:
        raise ApiError(f"Server ran {response_model!r}, expected {expected_model!r} "
                       f"for request model {payload.get('model')!r}")
    warnings = []
    if not isinstance(usage, dict):
        if require_usage:
            raise ApiError("Completion has no usage object; token counts cannot be verified")
        warnings.append("usage_missing")
        usage = {}
    prompt_tokens = _count(usage["prompt_tokens"], "prompt_tokens") if "prompt_tokens" in usage else None
    completion_tokens = (_count(usage["completion_tokens"], "completion_tokens")
                         if "completion_tokens" in usage else None)
    if require_usage and (prompt_tokens is None or completion_tokens is None):
        raise ApiError("Usage must contain both prompt_tokens and completion_tokens")
    if completion_tokens is not None and completion_tokens > payload.get("max_tokens", math.inf):
        raise ApiError("Server reported more completion tokens than requested")
    if usage.get("total_tokens") is not None and prompt_tokens is not None and completion_tokens is not None:
        total = _count(usage["total_tokens"], "total_tokens")
        if total != prompt_tokens + completion_tokens:
            raise ApiError("usage.total_tokens differs from prompt_tokens + completion_tokens")
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached = (_count(prompt_details["cached_tokens"], "prompt_tokens_details.cached_tokens")
              if "cached_tokens" in prompt_details else None)
    if cached is not None and prompt_tokens is not None and cached > prompt_tokens:
        raise ApiError("Cached prompt tokens exceed total prompt tokens")
    draft_details = usage.get("completion_tokens_details") or {}
    accepted = draft_details.get("accepted_prediction_tokens")
    rejected = draft_details.get("rejected_prediction_tokens")
    acceptance = None
    if accepted is not None and rejected is not None:
        accepted = _count(accepted, "completion_tokens_details.accepted_prediction_tokens")
        rejected = _count(rejected, "completion_tokens_details.rejected_prediction_tokens")
        if accepted + rejected:
            acceptance = accepted / (accepted + rejected)
    server_decode = _finite_rate(usage.get("completion_tokens_per_sec"))
    server_prefill = _finite_rate(usage.get("prompt_tokens_per_sec"))
    if isinstance(timings, dict):
        if server_decode is None:
            server_decode = _finite_rate(timings.get("predicted_per_second"))
        if server_prefill is None:
            server_prefill = _finite_rate(timings.get("prompt_per_second"))
    estimate = None
    if streaming and first_output is not None and last_output > first_output and completion_tokens:
        estimate = max(completion_tokens - 1, 0) / (last_output - first_output)
        warnings.append("client_decode_estimate_assumes_one_token_in_first_SSE_event")
    if server_decode is None:
        warnings.append("server_decode_timing_missing")
    if not completion_tokens:
        warnings.append("no_completion_tokens")
    result = {
        "model": response_model or payload.get("model"),
        "requested_model": payload.get("model"),
        "expected_response_model": expected_model,
        "stream": streaming,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": prompt_tokens - cached if prompt_tokens is not None and cached is not None else None,
        "completion_tokens": completion_tokens,
        "ttft_s": first_output - start if first_output is not None else None,
        "wall_s": end - start,
        "server_decode_tok_s": server_decode,
        "server_prefill_tok_s": server_prefill,
        "client_end_to_end_tok_s": completion_tokens / (end - start) if completion_tokens is not None else None,
        "client_decode_tok_s_estimate": estimate,
        "draft_acceptance": acceptance,
        "usage": usage,
        "timings": timings,
        "sse_events": chunks if streaming else None,
        "warnings": warnings,
        "request_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "response_sha256": hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest(),
        "message": message,
        "_start": start,
        "_first_output": first_output,
        "_last_output": last_output,
        "_end": end,
    }
    return result


def public_result(result: dict, include_message=False) -> dict:
    return {key: value for key, value in result.items()
            if not key.startswith("_") and (include_message or key != "message")}


def percentile(values, quantile):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summary(runs):
    fields = ("server_decode_tok_s", "server_prefill_tok_s", "ttft_s", "wall_s",
              "client_end_to_end_tok_s", "client_decode_tok_s_estimate", "draft_acceptance")
    result = {"requests": len(runs)}
    for field in fields:
        result[field] = {"p50": percentile([r.get(field) for r in runs], 0.5),
                         "p95": percentile([r.get(field) for r in runs], 0.95),
                         "samples": sum(r.get(field) is not None for r in runs)}
    result["completion_tokens"] = [r["completion_tokens"] for r in runs]
    result["prompt_tokens"] = [r["prompt_tokens"] for r in runs]
    result["cached_prompt_tokens"] = [r["cached_prompt_tokens"] for r in runs]
    return result
