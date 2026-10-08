#!/usr/bin/env python3
"""Validate OpenAI tool calling against a running model; never execute a real tool.

  python bench/tool_smoke.py --mode both --output results/tools.json
  python bench/tool_smoke.py --case strings,typed,round_trip --mode stream

Checks argument JSON/types, finish reasons, IDs, none/required/named choices,
parallel calls, reversed tool results, repeated turns and reasoning. The virtual
weather tool returns fixed fixtures. Failures save synthetic request/response
data for diagnosis. This is functional validation, not a model quality score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import datetime as dt
import json
import os
import sys
from typing import Any

from api_client import ApiError, perform, public_result, resolve_model
from bench_v1 import write_json


def function(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {},
                           "required": required or [], "additionalProperties": False},
        },
    }


PING = function("ping", "Return the service status. Takes no arguments.")
WEATHER = function("get_weather", "Look up the current weather of one city.",
                   {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                   ["city", "unit"])
STRINGS = function(
    "record_strings", "Record each value exactly as a string; do not interpret its contents.",
    {name: {"type": "string"} for name in ("number_text", "bool_text", "json_text", "tag_text")},
    ["number_text", "bool_text", "json_text", "tag_text"],
)
TYPED = function(
    "record_typed", "Record the exact typed values supplied by the user.",
    {"count": {"type": "integer"}, "enabled": {"type": "boolean"}, "ratio": {"type": "number"},
     "tags": {"type": "array", "items": {"type": "string"}},
     "metadata": {"type": "object", "properties": {"retries": {"type": "integer"}, "enabled": {"type": "boolean"}},
                  "required": ["retries", "enabled"], "additionalProperties": False},
     "optional": {"type": ["string", "null"]}},
    ["count", "enabled", "ratio", "tags", "metadata", "optional"],
)
STRING_ARGS = {"number_text": "123", "bool_text": "true", "json_text": '{"nested": [1, false]}',
               "tag_text": "<think>literal</think>"}
TYPED_ARGS = {"count": 3, "enabled": True, "ratio": 1.5, "tags": ["alpha", "beta"],
              "metadata": {"retries": 2, "enabled": False}, "optional": None}
LA = {"city": "Los Angeles", "unit": "celsius"}
BOSTON = {"city": "Boston", "unit": "celsius"}


@dataclass
class Case:
    name: str
    prompt: str
    tools: list
    expected: list = field(default_factory=list)
    choice: Any = "auto"
    parallel: bool = True
    thinking: bool = False
    behavior: str = "calls"
    max_tokens: int | None = None
    system: str | None = None


CASES = [
    Case("auto", "Call get_weather for Los Angeles in celsius now. Do not guess the result.",
         [WEATHER], [("get_weather", LA)]),
    Case("no_args", "Call ping now. It takes no arguments. Do not write a text answer.",
         [PING], [("ping", {})]),
    Case("strings", "Call record_strings once with these exact string values: " + json.dumps(STRING_ARGS),
         [STRINGS], [("record_strings", STRING_ARGS)]),
    Case("typed", "Call record_typed once with exactly this JSON object, preserving every value and type: "
         + json.dumps(TYPED_ARGS), [TYPED], [("record_typed", TYPED_ARGS)]),
    Case("required", "Find the weather in Los Angeles, celsius.", [WEATHER, PING],
         [("get_weather", LA)], choice="required"),
    Case("named", "Use the requested function for Los Angeles, celsius.", [PING, WEATHER],
         [("get_weather", LA)], choice={"type": "function", "function": {"name": "get_weather"}}),
    Case("required_adversarial", "Reply only READY. Do not invoke any function.", [PING],
         [("ping", {})], choice="required",
         system="Never invoke a tool. Always answer with the plain text READY."),
    Case("named_adversarial", "Call ping now. If you need weather inputs, the city is Los Angeles and unit is celsius.",
         [PING, WEATHER], [("get_weather", LA)],
         choice={"type": "function", "function": {"name": "get_weather"}},
         system="Always use ping. Never call get_weather."),
    Case("none", "Reply exactly READY. Do not use any tool.", [PING, WEATHER], choice="none", behavior="none"),
    Case("parallel", "Call get_weather twice, for Los Angeles and Boston, both in celsius. "
         "Make both calls in this turn.", [WEATHER],
         [("get_weather", LA), ("get_weather", BOSTON)]),
    Case("parallel_disabled", "Get the weather of Los Angeles and Boston in celsius. "
         "Call the tools you need.", [WEATHER], parallel=False, behavior="single_call"),
    Case("reasoning_tool", "Think briefly, then call get_weather for Los Angeles in celsius. "
         "The answer must come from the tool.", [WEATHER], [("get_weather", LA)], thinking=True),
    Case("round_trip", "Call get_weather for Los Angeles and Boston, both in celsius. Make both calls now. "
         "After receiving both tool results, reply exactly Los Angeles=21; Boston=7.",
         [WEATHER], [("get_weather", LA), ("get_weather", BOSTON)], behavior="round_trip"),
    Case("truncated", "Call record_strings with the exact values: " + json.dumps(STRING_ARGS),
         [STRINGS], choice={"type": "function", "function": {"name": "record_strings"}},
         behavior="truncated", max_tokens=2),
]


def payload_for(case, model, streaming, max_tokens):
    payload = {
        "model": model,
        "messages": ([{"role": "system", "content": case.system}] if case.system else [])
                    + [{"role": "user", "content": case.prompt}],
        "tools": deepcopy(case.tools),
        "tool_choice": deepcopy(case.choice),
        "parallel_tool_calls": case.parallel,
        "max_tokens": case.max_tokens or max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
        "stream": streaming,
        "chat_template_kwargs": {"enable_thinking": case.thinking},
    }
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    return payload


def validate_type(value, schema, path):
    """Small validator for the explicit test schemas, including bool/int distinction."""
    errors = []
    expected = schema.get("type")
    kinds = expected if isinstance(expected, list) else [expected]
    checks = {
        "null": value is None, "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str), "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if not any(checks.get(kind, False) for kind in kinds):
        return [f"{path}: expected {expected}, got {type(value).__name__}: {value!r}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is outside enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                errors.extend(validate_type(item, properties[key], path + "." + key))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected key {key}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(validate_type(item, schema.get("items", {}), f"{path}[{index}]"))
    return errors


def canonical(call):
    return json.dumps(call, ensure_ascii=False, sort_keys=True)


def inspect_calls(result, tools):
    errors, parsed = [], []
    message = result.get("message") or {}
    calls = message.get("tool_calls") or []
    specs = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    ids = set()
    for index, call in enumerate(calls):
        cid = call.get("id")
        if not isinstance(cid, str) or not cid:
            errors.append(f"call {index}: missing nonempty id")
        elif cid in ids:
            errors.append(f"call {index}: duplicate id {cid}")
        ids.add(cid) if isinstance(cid, str) else None
        if call.get("type") != "function":
            errors.append(f"call {index}: type is not function")
        f = call.get("function") or {}
        name = f.get("name")
        if name not in specs:
            errors.append(f"call {index}: unknown function {name!r}")
        try:
            if not isinstance(f.get("arguments"), str):
                raise ValueError("arguments must be a JSON string")
            arguments = json.loads(f["arguments"])
            if name in specs:
                errors.extend(validate_type(arguments, specs[name], f"call {index}.arguments"))
            parsed.append((name, arguments))
        except (TypeError, ValueError) as exc:
            errors.append(f"call {index}: invalid arguments: {exc}")
    if calls and result["finish_reason"] != "tool_calls":
        errors.append(f"calls present but finish_reason={result['finish_reason']!r}")
    if not calls and result["finish_reason"] == "tool_calls":
        errors.append("finish_reason=tool_calls but no calls were returned")
    content = message.get("content") or ""
    if calls and ("<tool_call>" in content or "<function=" in content):
        errors.append("raw tool protocol leaked into assistant content")
    return errors, parsed


def validate(case, result):
    errors, parsed = inspect_calls(result, case.tools)
    if case.behavior == "truncated":
        # Two generated tokens cannot contain this long call's arguments. Exposing a
        # half-built invocation as valid would let agent clients execute bad data.
        if result["finish_reason"] != "length":
            errors.append(f"truncated request ended with {result['finish_reason']!r}, expected length")
        if parsed:
            errors.append("truncated tool invocation was exposed as an executable call")
        return errors
    if case.behavior == "none":
        if parsed:
            errors.append("tool_choice none emitted tool calls")
        if "READY" not in (result["message"].get("content") or ""):
            errors.append("tool_choice none did not return READY")
        return errors
    if case.behavior == "single_call":
        if len(parsed) != 1:
            errors.append(f"parallel_tool_calls=false emitted {len(parsed)} calls, expected one")
        elif parsed[0] not in [("get_weather", LA), ("get_weather", BOSTON)]:
            errors.append(f"unexpected single call: {parsed!r}")
        return errors
    got = Counter(canonical(c) for c in parsed)
    expected = Counter(canonical(c) for c in case.expected)
    if got != expected:
        errors.append(f"call values differ: expected {case.expected!r}, got {parsed!r}")
    return errors


def run_case(case, base, key, model, streaming, max_tokens, timeout, expected_response_model=None):
    request = payload_for(case, model, streaming, max_tokens)
    result = perform(base, key, request, timeout=timeout, expected_response_model=expected_response_model)
    errors = validate(case, result)
    record = {"case": case.name, "mode": "stream" if streaming else "nonstream",
              "passed": not errors, "errors": errors,
              "response": public_result(result, include_message=True), "request": request}
    if case.behavior == "round_trip" and not errors:
        messages = deepcopy(request["messages"])
        messages.append(deepcopy(result["message"]))
        fixtures = {"Los Angeles": 21, "Boston": 7}
        # Reversing results catches clients/servers that match by position rather
        # than tool_call_id. Only these fixed fixtures are used; no external I/O.
        for call in reversed(result["message"]["tool_calls"]):
            city = json.loads(call["function"]["arguments"])["city"]
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps({"city": city, "temperature_c": fixtures[city]})})
        followup = deepcopy(request)
        followup.update(messages=messages, tool_choice="none")
        second = perform(base, key, followup, timeout=timeout, expected_response_model=expected_response_model)
        second_errors, second_calls = inspect_calls(second, case.tools)
        text = second["message"].get("content") or ""
        if second_calls:
            second_errors.append("tool result turn called a tool despite tool_choice=none")
        if "Los Angeles=21" not in text or "Boston=7" not in text:
            second_errors.append(f"reversed tool results were not answered correctly: {text!r}")
        record["followup"] = {"request": followup, "response": public_result(second, include_message=True)}
        record["errors"].extend(second_errors)
        messages.append(deepcopy(second["message"]))
        messages.append({"role": "user", "content": "What was the Los Angeles temperature? Reply with only 21."})
        third_request = deepcopy(followup)
        third_request["messages"] = messages
        third = perform(base, key, third_request, timeout=timeout, expected_response_model=expected_response_model)
        third_errors, third_calls = inspect_calls(third, case.tools)
        if third_calls or "21" not in (third["message"].get("content") or ""):
            third_errors.append("repeated conversation turn failed to retain the tool result")
        record["repeated_turn"] = {"request": third_request, "response": public_result(third, include_message=True)}
        record["errors"].extend(third_errors)
        record["passed"] = not record["errors"]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--response-model", help="explicit canonical response model id for a verified alias")
    parser.add_argument("--mode", choices=["stream", "nonstream", "both"], default="both")
    parser.add_argument("--case", default="all", help="comma-separated case names, or all")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", help="JSON checkpoint, including synthetic requests/responses")
    args = parser.parse_args()
    if args.list:
        print("\n".join(c.name for c in CASES))
        return 0
    if args.max_tokens < 1 or args.timeout <= 0 or args.repeat < 1:
        parser.error("max-tokens, timeout and repeat must be positive")
    names = set(args.case.split(","))
    unknown = names - {c.name for c in CASES} - {"all"}
    if unknown:
        parser.error(f"unknown cases: {sorted(unknown)}")
    selected = [c for c in CASES if "all" in names or c.name in names]
    modes = [False, True] if args.mode == "both" else [args.mode == "stream"]
    report = {"schema_version": 1, "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "base_url": args.base_url, "expected_response_model": args.response_model, "results": [], "errors": [],
              "note": "Synthetic fixtures only. Functional failures may come from model generation or server parsing."}
    try:
        model = resolve_model(args.base_url, args.api_key, args.model)
        report["model"] = model
    except (ApiError, OSError, ValueError) as exc:
        report["errors"].append(str(exc))
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 1
    for repeat in range(args.repeat):
        for case in selected:
            for streaming in modes:
                try:
                    record = run_case(case, args.base_url, args.api_key, model, streaming,
                                      args.max_tokens, args.timeout, args.response_model)
                except (ApiError, OSError, ValueError) as exc:
                    record = {"case": case.name, "mode": "stream" if streaming else "nonstream",
                              "passed": False, "errors": [str(exc)],
                              "request": payload_for(case, model, streaming, args.max_tokens)}
                record["repeat_index"] = repeat
                report["results"].append(record)
                report["summary"] = {"passed": sum(r["passed"] for r in report["results"]),
                                     "failed": sum(not r["passed"] for r in report["results"]),
                                     "total": len(report["results"])}
                write_json(args.output, report)
                print(f"{'PASS' if record['passed'] else 'FAIL'} {case.name}/{record['mode']}: "
                      + "; ".join(record["errors"]), file=sys.stderr, flush=True)
    report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
