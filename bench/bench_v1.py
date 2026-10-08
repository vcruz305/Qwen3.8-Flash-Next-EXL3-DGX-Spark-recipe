#!/usr/bin/env python3
"""Reproducible chat benchmarks over an OpenAI-compatible /v1 endpoint.

  python bench/bench_v1.py --suite all --repeat 3 --output results/api.json
  python bench/bench_v1.py --suite code --cache-mode warm --context-tokens 24000
  python bench/bench_v1.py --model NAME --metadata deployment.json --label baseline

Requires server usage. Server decode/prefill timings and client wall/TTFT are
reported separately. Each cold case changes an early prompt prefix; warm cases
reuse an identical request. Actual prompt/cache counts come from usage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import uuid

from api_client import ApiError, perform, public_result, resolve_model, summary

CODE_PROMPT = (
    "Write a Python function that parses an nginx access log line into a dict with fields ip, "
    "timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage "
    "example. Then explain each regex group in one bullet each."
)
PROMPTS = {
    "code": CODE_PROMPT,
    "devops": "Explain a safe Kubernetes rolling deployment with readiness probes, graceful shutdown, "
              "and rollback. Include complete example YAML and a practical verification checklist.",
    "prose": "Write a 600-word short story about a lighthouse keeper who receives a letter dated "
             "one hundred years in the future. Give the characters distinctive voices and a clear ending.",
}
FILLER = (
    "The maintenance log for reactor bay seven records a pressure excursion at "
    "oh four hundred hours, followed by a manual override and a return to nominal. "
)


def build_prompt(task: str, context_tokens: int, salt: str) -> str:
    """Approximate prefix size only; reported API usage supplies the real token count."""
    prefix = f"Benchmark record {salt}.\n"
    if context_tokens:
        prefix += FILLER * max(1, round(context_tokens * 4 / len(FILLER)))
    return prefix + "\nUse the following task, independent of the maintenance notes above:\n" + task


def payload_for(model, prompt, max_tokens, thinking=False):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
    }


def write_json(path, report):
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--response-model", help="explicit canonical model id expected in responses when the server resolves an alias")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"), help="prefer API_KEY to avoid shell history")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="unmeasured requests per prompt class")
    parser.add_argument("--suite", choices=[*PROMPTS, "all"], default="code")
    parser.add_argument("--prompt", help="custom task, overrides --suite")
    parser.add_argument("--context-tokens", type=int, default=0, help="approximate filler prefix size; usage reports actual tokens")
    parser.add_argument("--cache-mode", choices=["cold", "warm"], default="cold")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:16], help="recorded prompt salt; reuse across isolated A/B runs")
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--metadata", help="JSON provenance: engine/server commit, model revision, config, hardware")
    parser.add_argument("--output", help="JSON report, atomically checkpointed after every request")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.max_tokens < 1 or args.context_tokens < 0 or args.timeout <= 0:
        parser.error("repeat/max-tokens/timeout must be positive; warmup/context-tokens cannot be negative")
    base = args.base_url.rstrip("/")
    provenance = json.loads(Path(args.metadata).read_text()) if args.metadata else {}
    report = {
        "schema_version": 2,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": args.label,
        "expected_response_model": args.response_model,
        "run_id": args.run_id,
        "base_url": base,
        "client": {"python": platform.python_version(), "platform": platform.platform()},
        "provenance": provenance,
        "settings": {"max_tokens": args.max_tokens, "repeat": args.repeat, "warmup": args.warmup,
                     "context_tokens_approx": args.context_tokens, "cache_mode": args.cache_mode,
                     "thinking": args.thinking, "temperature": 0, "top_k": 1, "seed": 0},
        "metric_notes": [
            "server_decode_tok_s and server_prefill_tok_s are server-reported timings.",
            "client_end_to_end_tok_s includes prefill, queueing and HTTP overhead.",
            "TTFT is time to first visible content/reasoning/tool delta, not a role or ping event.",
            "SSE events may contain multiple speculative tokens; client_decode_tok_s_estimate is approximate.",
            "A unique cold prefix avoids reusable user prompt blocks; inspect actual cached_prompt_tokens.",
            "context_tokens_approx is only a prompt construction target; usage.prompt_tokens is authoritative.",
            "Early EOS is allowed and reported; no request is counted as max_tokens unless usage says so.",
        ],
        "cases": {},
        "errors": [],
    }
    try:
        model = resolve_model(base, args.api_key, args.model)
        report["model"] = model
        tasks = {"custom": args.prompt} if args.prompt else PROMPTS if args.suite == "all" else {args.suite: PROMPTS[args.suite]}
        for name, task in tasks.items():
            case = {"prompt_sha256": hashlib.sha256(task.encode()).hexdigest(), "warmup": [], "runs": []}
            report["cases"][name] = case
            count = args.warmup + args.repeat
            for index in range(count):
                warming = index < args.warmup
                # Warm requests reuse the exact prompt; cold requests change the prefix before filler.
                salt = f"{args.run_id}-{name}-" + ("shared" if args.cache_mode == "warm" else str(index))
                prompt = build_prompt(task, args.context_tokens, salt)
                raw = perform(base, args.api_key, payload_for(model, prompt, args.max_tokens, args.thinking), timeout=args.timeout, expected_response_model=args.response_model)
                result = public_result(raw)
                result["phase"] = "warmup" if warming else "measured"
                result["repeat_index"] = index - args.warmup if not warming else index
                case["warmup" if warming else "runs"].append(result)
                case["summary"] = summary(case["runs"])
                write_json(args.output, report)
                print(f"{name} {result['phase']} {result['repeat_index']}: "
                      f"prompt={result['prompt_tokens']} cache={result['cached_prompt_tokens']} "
                      f"gen={result['completion_tokens']} decode={result['server_decode_tok_s']} "
                      f"TTFT={result['ttft_s']} finish={result['finish_reason']}", file=sys.stderr, flush=True)
        report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    except (ApiError, OSError, ValueError) as exc:
        report["errors"].append(str(exc))
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return 1
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
