#!/usr/bin/env python3
"""Aggregate throughput under N concurrent streams, over any OpenAI-compatible /v1 endpoint.

    python bench/concurrency.py <tag> <streams, e.g. 1,2,4> <prompt tokens> <new tokens> [rounds]
    BASE_URL=http://127.0.0.1:8899/v1 API_KEY=... python bench/concurrency.py tabby 1,2,4 3000 512

Two aggregates are reported: 'window' (total tokens over the earliest first token to the latest
last token, which charges other streams' prefills to the first stream) and 'steady' (tokens over
the span from the LAST stream's first token to the FIRST stream's last token, where every stream
is decoding). Unique prefix per stream so prefix caching cannot fake a cold prefill. Uses
/v1/completions (raw prompt), greedy. The model id defaults to the first from /v1/models.
Results are also written to concurrency.<tag>.json.
"""
import json
import os
import sys
import threading
import time
import urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8899/v1").rstrip("/")
KEY = os.environ.get("API_KEY")
FILLER = ("The maintenance log for reactor bay seven records a pressure excursion at "
          "oh four hundred hours, followed by a manual override and a return to nominal. ")
_s = [int(time.time()) % 100000]
_lock = threading.Lock()


def _req(path, payload=None):
    h = {"Content-Type": "application/json"}
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
    return urllib.request.Request(BASE + path, json.dumps(payload).encode() if payload else None, h)


def prompt(ntok):
    with _lock:
        _s[0] += 1
        n = _s[0]
    return "Record %d. " % n + FILLER * max(1, int(ntok * 4 / len(FILLER))) + "\n\nSummarize in one sentence."


def one(model, p, maxgen, out):
    body = {"model": model, "prompt": p, "max_tokens": maxgen, "temperature": 0, "top_k": 1,
            "seed": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); tf = None; usage = None
    with urllib.request.urlopen(_req("/completions", body), timeout=3600) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            c = raw[6:].strip()
            if c == b"[DONE]":
                break
            d = json.loads(c)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and ch[0].get("text") and tf is None:
                tf = time.perf_counter()
    tl = time.perf_counter()
    out.append({"gen": usage["completion_tokens"], "t0": t0, "tf": tf or tl, "tl": tl})


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    tag, Ns, ctx, maxgen = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4])
    rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    model = os.environ.get("MODEL")
    if not model:
        with urllib.request.urlopen(_req("/models"), timeout=30) as r:
            model = json.load(r)["data"][0]["id"]
    res = {}
    for N in Ns:
        rows = []
        for r in range(rounds):
            out, th = [], []
            for _ in range(N):
                t = threading.Thread(target=one, args=(model, prompt(ctx), maxgen, out)); t.start(); th.append(t)
            for t in th:
                t.join()
            if len(out) != N:
                sys.exit(f"only {len(out)} of {N} streams completed; see the server log")
            tot = sum(o["gen"] for o in out)
            window = max(o["tl"] for o in out) - min(o["tf"] for o in out)
            steady_w = min(o["tl"] for o in out) - max(o["tf"] for o in out)
            # tokens emitted inside the steady window, assuming a constant rate per stream
            steady_tok = sum(o["gen"] * max(0.0, steady_w) / max(o["tl"] - o["tf"], 1e-6) for o in out)
            per = [round((o["gen"] - 1) / max(o["tl"] - o["tf"], 1e-6), 2) for o in out]
            rows.append({"window": round(tot / window, 2), "steady": round(steady_tok / steady_w, 2) if steady_w > 0 else None,
                         "steady_s": round(steady_w, 2), "per_stream": per})
            print(f"[{tag} ctx={ctx} gen={maxgen} N={N} r={r}] window={rows[-1]['window']} steady={rows[-1]['steady']} "
                  f"(steady window {rows[-1]['steady_s']}s) per_stream={per}", flush=True)
        res[N] = rows
        w = sorted(x["window"] for x in rows); s = sorted((x["steady"] or 0) for x in rows)
        print(f"SUMMARY {tag} ctx={ctx} gen={maxgen} N={N}: window_p50={w[len(w)//2]} steady_p50={s[len(s)//2]}", flush=True)
    json.dump({"model": model, "base_url": BASE, "results": res}, open(f"concurrency.{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
