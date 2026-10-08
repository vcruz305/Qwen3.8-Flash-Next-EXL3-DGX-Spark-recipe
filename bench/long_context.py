#!/usr/bin/env python3
"""Check three exact retrieval values across a large, cold chat prompt.

This is a context/cache regression check, not a general model quality score.
Tokenizers is needed only in this test client; the HTTP harness is shared with
the other recipe benchmarks. The requested size is approximate: actual server
usage is authoritative and the saved report distinguishes both counts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from api_client import ApiError, perform, public_result, resolve_model
from bench_v1 import payload_for, write_json

FILLER = (
    "The station maintenance archive records routine inspections of valves, "
    "filters, pressure sensors, access doors, backup power and ventilation. "
    "The inspection team checks each item, records its condition and schedules "
    "ordinary maintenance. This paragraph contains no project verification value.\n"
)
POSITIONS = (0.12, 0.49, 0.88)
NAMES = ("ORCHID", "MAPLE", "CEDAR")


def build_prompt(tokenizer, target: int, run_id: str):
    values = {
        name: hashlib.sha256(f"{run_id}:{name}".encode()).hexdigest()[:16].upper()
        for name in NAMES
    }
    seed_ids = tokenizer.encode(FILLER * max(2, target // 20), add_special_tokens=False).ids
    if len(seed_ids) < target:
        raise ValueError("Filler construction did not cover the requested token size")
    parts = [f"Archive instance {run_id}. Read the complete archive and preserve its exact values.\n"]
    previous = 0
    locations = []
    for name, fraction in zip(NAMES, POSITIONS):
        end = int(target * fraction)
        parts.append(tokenizer.decode(seed_ids[previous:end], skip_special_tokens=False))
        marker = f"\nVERIFIED PROJECT RECORD: project {name}; verification value {values[name]}.\n"
        before = "".join(parts)
        position = len(tokenizer.encode(before, add_special_tokens=False).ids)
        locations.append({"project": name, "raw_text_token_offset": position, "value": values[name]})
        parts.append(marker)
        previous = end
    parts.append(tokenizer.decode(seed_ids[previous:target], skip_special_tokens=False))
    parts.append(
        "\nEnd of archive. Return one JSON object with exactly the keys ORCHID, MAPLE and CEDAR. "
        "For each key, copy its exact verification value from the VERIFIED PROJECT RECORD. "
        "Return the JSON only, without markdown or explanation.\n"
    )
    prompt = "".join(parts)
    count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    return prompt, values, locations, count


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_object(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ApiError("Malformed fenced response")
        stripped = "\n".join(lines[1:-1])
    try:
        result = json.loads(stripped, object_pairs_hook=reject_duplicate_keys)
    except ValueError as error:
        raise ApiError("Retrieval response is not one JSON object") from error
    if not isinstance(result, dict):
        raise ApiError("Retrieval response is not an object")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--response-model")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=240000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:16], help="Recorded prompt seed; repeat only on a fresh server for a cold A/B")
    parser.add_argument("--max-cached-tokens", type=int, default=0, help="Maximum accepted prompt-cache reuse; default requires cold prefill")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1024 <= args.context_tokens <= 260000 or not 1 <= args.max_tokens <= 1024 or args.max_cached_tokens < 0:
        parser.error("Use 1024..260000 prefix tokens and 1..1024 output tokens within the trained window")
    if args.output.exists():
        parser.error("Refusing to replace an existing result; use a distinct output path")
    report = {
        "format_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id, "context_tokens_requested": args.context_tokens,
        "max_tokens_requested": args.max_tokens, "max_cached_tokens": args.max_cached_tokens,
        "provenance": json.loads(args.metadata.read_text()) if args.metadata else {},
        "notes": [
            "Three exact synthetic retrieval values at early, middle and late positions.",
            "The local raw-text count excludes chat-template and generation-prefix tokens.",
            "API usage.prompt_tokens is the actual served prompt size.",
            "This checks context handling; it is not a broad reasoning/capability evaluation.",
        ],
        "passed": False, "errors": [],
    }
    try:
        from tokenizers import Tokenizer
        model = resolve_model(args.base_url, args.api_key, args.model)
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        prompt, expected, locations, local_count = build_prompt(tokenizer, args.context_tokens, args.run_id)
        if local_count + args.max_tokens + 512 > 262144:
            raise ValueError("Constructed prompt lacks safe room for the chat template and output")
        report.update(
            model=model, tokenizer_sha256=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            raw_text_tokens=local_count, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            expected=expected, records=locations,
        )
        write_json(args.output, report)
        result = perform(args.base_url, args.api_key, payload_for(model, prompt, args.max_tokens),
                         timeout=args.timeout, expected_response_model=args.response_model)
        report["response"] = public_result(result, include_message=True)
        actual = parse_object(result["message"].get("content") or "")
        report["retrieved"] = actual
        report["matches"] = {name: actual.get(name) == value for name, value in expected.items()}
        if actual != expected:
            raise ApiError("Retrieved project values or object keys do not exactly match")
        if result["finish_reason"] != "stop":
            raise ApiError(f"Retrieval did not finish cleanly: {result['finish_reason']}")
        if result["prompt_tokens"] > 262144 or result["prompt_tokens"] < args.context_tokens:
            raise ApiError("Actual prompt usage falls outside the tested context range")
        report["retrieval_passed"] = True
        cached = result.get("cached_prompt_tokens")
        if cached is None or cached > args.max_cached_tokens:
            raise ApiError(f"Prompt-cache reuse {cached!r} exceeds the allowed {args.max_cached_tokens}")
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
