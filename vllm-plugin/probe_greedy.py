#!/usr/bin/env python3
"""Greedy-decode probe for Qwen3.8-Flash-Next: 4 fixed prompts, temperature 0,
max_tokens 256. Prints each response and saves the full run (text, usage, and
the spec-decode acceptance rate scraped from /metrics) as JSON.

Adapted from /home/markus/work/text_probe.py: at temperature 0 speculative
decoding must reproduce the target model's own tokens exactly (the draft only
proposes; verification keeps or rejects), so spec-on text should match
spec-off text for the same plugin -- a divergence means verification is not
faithful, whatever the measured speedup.

usage: python3 probe_greedy.py <label> <out.json> [base_url] [max_tokens]
"""
import json
import sys
import urllib.request

PROMPTS = [
    "Write a detailed technical explanation of how neural networks work. " * 20,
    "Describe the historical events that led to the creation of the internet. " * 20,
    "Explain the principles of quantum computing and its potential applications. " * 20,
    "Write a Python function that parses an ISO-8601 timestamp without any imports, "
    "and explain each step.",
]


def metrics(url):
    txt = urllib.request.urlopen(f"{url}/metrics", timeout=20).read().decode()
    out = {}
    for line in txt.split("\n"):
        if not line or line.startswith("#"):
            continue
        head = line.split(None, 1)[0]
        base = head.split("{", 1)[0]
        if base in ("vllm:spec_decode_num_accepted_tokens_per_pos",
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            out["acc"] = out.get("acc", 0.0) + float(line.split()[-1])
        elif base in ("vllm:generation_tokens_total", "vllm:generation_tokens"):
            out["gen"] = out.get("gen", 0.0) + float(line.split()[-1])
    return out


def main():
    label, out_path = sys.argv[1], sys.argv[2]
    url = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8899"
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    mid = json.load(urllib.request.urlopen(f"{url}/v1/models", timeout=20))["data"][0]["id"]
    m0 = metrics(url)
    texts, usages = [], []
    for p in PROMPTS:
        body = json.dumps({"model": mid, "messages": [{"role": "user", "content": p}],
                            "max_tokens": max_tokens, "temperature": 0, "seed": 0}).encode()
        req = urllib.request.Request(f"{url}/v1/chat/completions", body,
                                      {"content-type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        msg = r["choices"][0]["message"]
        texts.append((msg.get("content") or "") + (("\n[reasoning]" + msg["reasoning_content"])
                                                     if msg.get("reasoning_content") else ""))
        usages.append(r.get("usage", {}))
    m1 = metrics(url)
    acc = m1.get("acc", 0) - m0.get("acc", 0)
    gen = m1.get("gen", 0) - m0.get("gen", 0)
    steps = gen - acc
    spec = {"accepted": acc, "generated": gen, "steps": steps,
            "mean_accept_len": round(gen / steps, 3) if steps > 0 else None}
    json.dump({"label": label, "model": mid, "texts": texts, "usage": usages, "spec": spec},
              open(out_path, "w"), indent=1)
    print(f"{label}: {len(texts)} texts, spec={spec}")
    for i, t in enumerate(texts):
        print(f"  [{i}] {len(t)} chars: {t[:110]!r}")


if __name__ == "__main__":
    main()
