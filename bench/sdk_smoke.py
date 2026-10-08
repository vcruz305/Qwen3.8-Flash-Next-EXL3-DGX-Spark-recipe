#!/usr/bin/env python3
"""Exercise the public OpenAI Python SDK against the deployed Tabby chat API.

Optional client dependency: openai. Keep it in a separate test virtualenv; the
server does not need the SDK. All tool results below are synthetic fixtures.
This is a functional integration check, not a throughput benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable

from api_client import ApiError, resolve_model


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up weather for the given city and unit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    },
    {"type": "function", "function": {"name": "ping", "description": "Return a ping."}},
]
NAMED = {"type": "function", "function": {"name": "get_weather"}}
TOOL_MESSAGES = [
    {"role": "system", "content": "Be concise. Use the requested tool when available."},
    {
        "role": "user",
        "content": "Call get_weather for Tokyo in celsius. Do not guess the weather.",
    },
]
TOOL_RESULT = {
    "city": "Tokyo",
    "unit": "celsius",
    "temperature": 21,
    "condition": "sunny",
    "source": "synthetic test fixture",
}


def sdk_completion(client, payload: dict, streaming: bool):
    """Use the SDK's public stream accumulator, including its final message."""
    events = Counter()
    if streaming:
        with client.chat.completions.stream(
            **payload, stream_options={"include_usage": True}
        ) as stream:
            for event in stream:
                events[event.type] += 1
            response = stream.get_final_completion()
    else:
        response = client.chat.completions.create(**payload)
    return response, dict(events)


def validate_completion(response, expected_model: str, finish_reason: str):
    from openai.types.chat import ChatCompletion

    # The SDK constructs partial snapshots permissively. Revalidate the final
    # object so missing assistant roles cannot silently survive into history.
    dumped = response.model_dump(mode="json", exclude_none=True)
    final = ChatCompletion.model_validate(dumped)
    if final.model != expected_model:
        raise ApiError(f"Response model {final.model!r} does not match {expected_model!r}")
    if len(final.choices) != 1:
        raise ApiError(f"Expected one completion choice, got {len(final.choices)}")
    choice = final.choices[0]
    if choice.finish_reason != finish_reason:
        raise ApiError(f"Expected finish_reason={finish_reason!r}, got {choice.finish_reason!r}")
    if choice.message.role != "assistant":
        raise ApiError(f"Expected assistant role, got {choice.message.role!r}")
    if not final.usage or final.usage.completion_tokens <= 0:
        raise ApiError("Expected actual positive completion token usage")
    return choice.message


def validate_weather_call(message):
    calls = message.tool_calls or []
    if len(calls) != 1:
        raise ApiError(f"Named single-call request returned {len(calls)} calls")
    call = calls[0]
    if call.type != "function" or call.function.name != "get_weather":
        raise ApiError("Named get_weather choice returned the wrong tool")
    if not call.id:
        raise ApiError("Tool call is missing an ID")
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, ValueError) as error:
        raise ApiError("Tool arguments are not complete JSON") from error
    if arguments != {"city": "Tokyo", "unit": "celsius"}:
        raise ApiError(f"Unexpected weather fixture arguments: {arguments!r}")
    return call


def validate_text(message, required_text: str | None = None):
    if message.tool_calls:
        raise ApiError("Text-only request returned tool calls")
    if not isinstance(message.content, str) or not message.content.strip():
        raise ApiError("Text-only request returned no assistant content")
    if required_text and required_text not in message.content:
        raise ApiError(f"Assistant did not use the synthetic result {required_text!r}")


def checkpoint(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    report["passed"] = sum(row["status"] == "pass" for row in report["cases"])
    report["failed"] = sum(row["status"] == "fail" for row in report["cases"])
    report["skipped"] = sum(row["status"] == "skipped" for row in report["cases"])
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def run_cases(model: str, max_tokens: int, record: Callable):
    common = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "extra_body": {"enable_thinking": False},
    }
    plain = {
        **common,
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Reply with the word READY."},
        ],
    }
    record(
        "plain_stream", plain, True, "stop", lambda message: validate_text(message, "READY")
    )

    for streaming in (False, True):
        mode = "stream" if streaming else "nonstream"
        payload = {
            **common,
            "messages": TOOL_MESSAGES,
            "tools": TOOLS,
            "tool_choice": NAMED,
            "parallel_tool_calls": False,
        }
        response, message = record(
            f"named_{mode}", payload, streaming, "tool_calls", validate_weather_call
        )
        if message is None:
            record(f"round_trip_{mode}", None, streaming, None, None)
            continue

        call = message.tool_calls[0]
        # Reuse the SDK-assembled assistant message itself. Do not repair or
        # inject its role; that would hide the exact interoperability bug tested.
        assistant = response.choices[0].message.model_dump(
            mode="json", exclude_none=True
        )
        history = [
            *TOOL_MESSAGES,
            assistant,
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(TOOL_RESULT),
            },
            {
                "role": "user",
                "content": "State Tokyo's temperature from that result in one sentence.",
            },
        ]
        followup = {
            **common,
            "messages": history,
            "tools": TOOLS,
            "tool_choice": "none",
        }
        record(
            f"round_trip_{mode}",
            followup,
            streaming,
            "stop",
            lambda reply: validate_text(reply, "21"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model")
    parser.add_argument("--response-model", help="explicit independently verified response alias")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--label", default="sdk-smoke")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/sdk-smoke.json"))
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be positive")
    try:
        import openai
    except ImportError:
        parser.error(
            "OpenAI SDK is an optional client dependency. Install it in a separate "
            "test virtualenv; see docs/tool-calling.md."
        )

    report = {
        "schema_version": 1,
        "kind": "openai_sdk_functional_smoke",
        "label": args.label,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "requested_model": args.model,
        "expected_response_model": args.response_model or args.model,
        "client": {"openai": openai.__version__, "python": platform.python_version()},
        "cases": [],
        "tool_results": "synthetic fixtures; no external tools are executed",
        "latencies": "wall seconds for diagnostics; not a throughput comparison",
    }
    if args.metadata:
        report["deployment"] = json.loads(args.metadata.read_text())
    checkpoint(args.output, report)

    try:
        model = resolve_model(args.base_url, args.api_key, args.model)
        expected_model = args.response_model or model
        report.update(requested_model=model, expected_response_model=expected_model)

        def record(name, payload, streaming, finish_reason, validator):
            row = {"name": name, "stream": streaming}
            report["cases"].append(row)
            if payload is None:
                row.update(status="skipped", reason="preceding named tool call failed")
                checkpoint(args.output, report)
                return None, None
            row["request"] = payload
            started = time.perf_counter()
            response = message = None
            try:
                response, events = sdk_completion(client, payload, streaming)
                row["response"] = response.model_dump(mode="json", exclude_none=True)
                row["sdk_events"] = events
                message = validate_completion(response, expected_model, finish_reason)
                validator(message)
                row["status"] = "pass"
            except Exception as error:
                row.update(status="fail", error_type=type(error).__name__, error=str(error))
                partial = getattr(error, "completion", None)
                if hasattr(partial, "model_dump"):
                    row["partial_completion"] = partial.model_dump(mode="json", exclude_none=True)
                message = None
            row["wall_s"] = time.perf_counter() - started
            checkpoint(args.output, report)
            print(f"{name}: {row['status']}", flush=True)
            return response, message

        with openai.OpenAI(
            base_url=args.base_url,
            api_key=args.api_key or "local-no-auth",
            timeout=args.timeout,
            max_retries=0,
        ) as client:
            run_cases(model, args.max_tokens, record)
    except Exception as error:
        report["setup_error"] = {"type": type(error).__name__, "message": str(error)}
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint(args.output, report)
    print(f"SDK smoke: {report['passed']} passed, {report['failed']} failed, "
          f"{report['skipped']} skipped; {args.output}", flush=True)
    if report.get("setup_error"):
        print(report["setup_error"]["message"], file=sys.stderr)
    return 1 if report.get("setup_error") or report["failed"] or report["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
