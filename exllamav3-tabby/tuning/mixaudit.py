#!/usr/bin/env python3
"""Audit the hyperconnection / gated-residual mixer weights: dtype, shape, total bytes, and
what fraction of per-round decode traffic they represent. If they are fp16 in an otherwise
3-bit model they are the single largest non-quantized read in the decode loop."""
import sys, os, torch, collections
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
model.load(progressbar=False)

# Walk every module, bucket parameters by (module class, dtype)
buckets = collections.defaultdict(lambda: [0, 0])  # -> [bytes, count]
gr_detail = []

def walk(m, depth=0):
    cls = type(m).__name__
    for name, t in vars(m).items():
        if torch.is_tensor(t):
            nb = t.numel() * t.element_size()
            buckets[(cls, str(t.dtype))][0] += nb
            buckets[(cls, str(t.dtype))][1] += 1
            if "resid" in cls.lower() or "hyper" in cls.lower() or "gated" in cls.lower():
                gr_detail.append((getattr(m, "key", "?"), name, tuple(t.shape), str(t.dtype), nb))
    for sm in getattr(m, "modules", []) or []:
        walk(sm, depth + 1)
    inner = getattr(m, "inner", None)
    if inner is not None and inner is not m:
        walk(inner, depth + 1)

walk(model)
print("=== parameter bytes by (module class, dtype), top 15 ===")
tot = 0
for (cls, dt), (nb, n) in sorted(buckets.items(), key=lambda kv: -kv[1][0])[:15]:
    tot += nb
    print(f"  {cls:28s} {dt:16s} {nb/1e6:9.1f} MB  ({n} tensors)")
print(f"  ---- total counted: {sum(v[0] for v in buckets.values())/1e9:.2f} GB")

print("\n=== gated-residual / hyperconnection tensors (first 24) ===")
for k, name, shape, dt, nb in gr_detail[:24]:
    print(f"  {k:34s} .{name:16s} {str(shape):22s} {dt:12s} {nb/1e6:7.2f} MB")
if gr_detail:
    g = sum(nb for *_, nb in gr_detail)
    print(f"  ---- {len(gr_detail)} tensors, {g/1e9:.3f} GB total")
    print(f"  ---- per decode round (read once each): {g/1e9:.3f} GB "
          f"= {g/273e9*1e3:.1f} ms at 273 GB/s spec bandwidth")
