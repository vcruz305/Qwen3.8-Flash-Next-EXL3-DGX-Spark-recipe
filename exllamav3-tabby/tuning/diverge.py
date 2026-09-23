#!/usr/bin/env python3
"""Divergence check: int8 mixer ON vs OFF, greedy, same prompt -> first token index where the
generated ids differ, per prompt class. Reads the token-level output by re-tokenizing the
chat.py -basic logs from i8bench (text -> ids is lossless for identical prefixes)."""
import os, re, sys
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Tokenizer
tok = Tokenizer.from_config(Config.from_directory(os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")))

def gen_text(path):
    raw = open(path, errors="replace").read().replace("\r", "\n")
    # answer runs from "Assistant: " to the -tps summary / cutoff notice
    body = re.split(r"\n !! Response exceeded|\nContext: \d+ new tokens", raw)[0]
    i = body.find("Assistant: ")
    return body[i + len("Assistant: "):].strip() if i >= 0 else body.strip()

print(f"{'prompt':7s} {'first diverging token':>22s} {'of':>5s}  {'identical text?':>15s}")
for pc in ("code", "devops", "prose"):
    a = gen_text(os.path.expanduser(f"~/i8_int8-off_dds_{pc}.log"))
    b = gen_text(os.path.expanduser(f"~/i8_int8-on_dds_{pc}.log"))
    ia = tok.encode(a, add_bos=False)[0].tolist(); ib = tok.encode(b, add_bos=False)[0].tolist()
    n = 0
    for x, y in zip(ia, ib):
        if x != y: break
        n += 1
    print(f"{pc:7s} {n:22d} {min(len(ia), len(ib)):5d}  {str(a == b):>15s}")
