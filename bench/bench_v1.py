#!/usr/bin/env python3
"""Single-stream decode tok/s over any OpenAI-compatible /v1 endpoint (TabbyAPI, vLLM, beta shim).

    python bench/bench_v1.py                                  # TabbyAPI default: 127.0.0.1:8899
    python bench/bench_v1.py --max-tokens 400 --repeat 3
    python bench/bench_v1.py --api-key $KEY                   # when auth is enabled

Greedy, thinking off, streamed. Decode excludes TTFT and counts tokens from the usage block.
The model id defaults to whatever /v1/models reports first.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request

CODE_PROMPT = (
    "Write a Python function that parses an nginx access log line into a dict with fields ip, "
    "timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage "
    "example. Then explain each regex group in one bullet each."
)


def _req(url: str, key: str | None, payload: dict | None = None) -> urllib.request.Request:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode() if payload is not None else None
    return urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")


def first_model(base: str, key: str | None) -> str:
    with urllib.request.urlopen(_req(base + "/models", key), timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def run_once(base: str, key: str | None, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    completion = 0
    with urllib.request.urlopen(_req(base + "/chat/completions", key, payload), timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            choice = (obj.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                ttft = time.perf_counter() - t0
            usage = obj.get("usage") or {}
            if usage.get("completion_tokens"):
                completion = int(usage["completion_tokens"])
    wall = time.perf_counter() - t0
    ttft = ttft or wall
    n = completion or max_tokens
    return {
        "ttft_s": round(ttft, 3),
        "wall_s": round(wall, 3),
        "completion_tokens": n,
        "decode_tok_s": round((n - 1) / max(wall - ttft, 1e-6), 2) if n > 1 else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1"))
    p.add_argument("--model", default=None, help="default: first id from /v1/models")
    p.add_argument("--api-key", default=os.environ.get("API_KEY"))
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--prompt", default=CODE_PROMPT)
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    model = args.model or first_model(base, args.api_key)
    runs = [run_once(base, args.api_key, model, args.prompt, args.max_tokens) for _ in range(args.repeat)]
    rates = [r["decode_tok_s"] for r in runs if r["decode_tok_s"]]
    print(json.dumps({"model": model, "runs": runs,
                      "decode_tok_s_median": statistics.median(rates) if rates else None}, indent=2))


if __name__ == "__main__":
    main()
