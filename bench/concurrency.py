#!/usr/bin/env python3
"""Measure actual aggregate throughput with synchronized concurrent requests.

  python bench/concurrency.py candidate 1,2,4 3000 400 3
  python bench/concurrency.py candidate 1,2,4 24000 400 --warmup 1 --output results/concurrency.json

Positional arguments remain compatible with the old client: tag, streams,
approximate prompt prefix tokens, max new tokens, optional rounds. The headline
is total emitted tokens / batch wall time, including prefill and queueing.
There is no invented constant-rate "steady" throughput. Every token count comes
from usage; each stream has a unique early prefix and every failure is recorded.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import os
from pathlib import Path
import threading
import sys
import uuid

from api_client import ApiError, perform, percentile, public_result, resolve_model, summary
from bench_v1 import CODE_PROMPT, build_prompt, payload_for, write_json


def batch(base, key, model, count, context, max_tokens, salt, timeout, expected_response_model=None):
    barrier = threading.Barrier(count)
    def worker(index):
        prompt = build_prompt(CODE_PROMPT, context, f"{salt}-{index}")
        payload = payload_for(model, prompt, max_tokens)
        barrier.wait(timeout=60)
        result = perform(base, key, payload, timeout=timeout, expected_response_model=expected_response_model)
        result["stream_index"] = index
        return result
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = {pool.submit(worker, i): i for i in range(count)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except (ApiError, OSError, ValueError, threading.BrokenBarrierError) as exc:
                errors.append({"stream_index": futures[future], "error": str(exc)})
    results.sort(key=lambda row: row["stream_index"])
    record = {"streams": [public_result(r) for r in results], "errors": errors,
              "complete": len(results) == count and not errors}
    if not record["complete"]:
        # A partial-success batch is not a valid throughput sample.
        return record
    total = sum(r["completion_tokens"] for r in results)
    wall = max(r["_end"] for r in results) - min(r["_start"] for r in results)
    first = [r["_first_output"] for r in results]
    last = [r["_last_output"] for r in results]
    overlap = None
    if all(x is not None for x in first + last):
        overlap = max(0.0, min(last) - max(first))
    record.update(
        completion_tokens=total,
        batch_wall_s=wall,
        end_to_end_tok_s=total / wall,
        all_streams_output_overlap_s=overlap,
        per_stream_summary=summary(record["streams"]),
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag")
    parser.add_argument("streams", help="comma-separated request concurrency, e.g. 1,2,4")
    parser.add_argument("prompt_tokens", type=int, help="approximate prefix size; API usage reports actual tokens")
    parser.add_argument("new_tokens", type=int)
    parser.add_argument("rounds", type=int, nargs="?", default=3)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1"))
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--response-model", help="explicit canonical response model id for a verified alias")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--run-id", default=uuid.uuid4().hex[:16])
    parser.add_argument("--metadata")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        counts = [int(value) for value in args.streams.split(",")]
    except ValueError:
        parser.error("streams must be comma-separated integers")
    if not counts or any(n < 1 for n in counts) or len(set(counts)) != len(counts):
        parser.error("streams must contain distinct positive integers")
    if args.prompt_tokens < 0 or args.new_tokens < 1 or args.rounds < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("invalid nonpositive size or count")
    destination = args.output or f"concurrency.{Path(args.tag).name}.json"
    report = {
        "schema_version": 2, "tag": args.tag, "run_id": args.run_id,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "expected_response_model": args.response_model,
        "provenance": json.loads(Path(args.metadata).read_text()) if args.metadata else {},
        "settings": {"concurrency": counts, "prompt_tokens_approx": args.prompt_tokens,
                     "max_tokens": args.new_tokens, "rounds": args.rounds, "warmup": args.warmup},
        "metric_notes": [
            "end_to_end_tok_s = all emitted completion tokens / earliest request start to final response end.",
            "Requests start through a barrier; queueing and prefill are included.",
            "Prompt sizes are approximate construction targets; each stream reports actual usage.",
            "all_streams_output_overlap_s is a time interval, not a token-rate estimate.",
            "No steady-state token rate is inferred from an assumed constant per-stream rate.",
            "Failed or partial batches have no throughput figure.",
        ],
        "results": {}, "errors": [],
    }
    try:
        model = resolve_model(args.base_url, args.api_key, args.model)
        report["model"] = model
        for count in counts:
            group = {"warmup": [], "rounds": []}
            report["results"][str(count)] = group
            for index in range(args.warmup + args.rounds):
                warming = index < args.warmup
                record = batch(args.base_url, args.api_key, model, count, args.prompt_tokens,
                               args.new_tokens, f"{args.run_id}-{count}-{index}", args.timeout, args.response_model)
                record["round_index"] = index if warming else index - args.warmup
                group["warmup" if warming else "rounds"].append(record)
                values = [row.get("end_to_end_tok_s") for row in group["rounds"] if row["complete"]]
                group["summary"] = {
                    "complete_rounds": len(values),
                    "failed_rounds": sum(not row["complete"] for row in group["rounds"]),
                    "end_to_end_tok_s_p50": percentile(values, 0.5),
                    "end_to_end_tok_s_p95": percentile(values, 0.95),
                }
                write_json(destination, report)
                print(f"N={count} {'warmup' if warming else 'round'} {record['round_index']}: "
                      f"complete={record['complete']} end_to_end_tok_s={record.get('end_to_end_tok_s')} "
                      f"errors={record['errors']}", file=sys.stderr, flush=True)
    except (ApiError, OSError, ValueError) as exc:
        report["errors"].append(str(exc))
    report["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    failed = report["errors"] or any(
        not row["complete"]
        for group in report["results"].values() for row in group["warmup"] + group["rounds"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
