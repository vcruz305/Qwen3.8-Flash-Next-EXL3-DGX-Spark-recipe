#!/usr/bin/env python3
"""Does the hyperconnection mixer need fp16? Simulate int8 (and int6/int4) storage by
quantize->dequantize of fn_h / upx_h in place after load, keeping fp16 arithmetic. If greedy
output and draft acceptance survive int8, an int8 mixer kernel is justified (halves the 1.49
GB/round the fused path reads -> ~5 ms/round).

Per-row symmetric scales along the contracting dim, which is what a kernel would use.
"""
import sys, os, torch, time
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler

MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
BITS = int(os.environ.get("MIXBITS", "8"))
PROMPT = ("<|im_start|>user\nWrite a Python function that parses an nginx access log line into a "
          "dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type "
          "hints, and a usage example.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

def qdq(t, bits, axis):
    """Symmetric per-row quantize->dequantize along `axis`. Returns a NEW tensor:
    loaded weights are inference tensors and cannot be updated in place."""
    qmax = 2 ** (bits - 1) - 1
    f = t.float().clone()
    scale = f.abs().amax(dim=axis, keepdim=True).clamp_min(1e-8) / qmax
    f.div_(scale).round_().clamp_(-qmax - 1, qmax).mul_(scale)
    return f.half().contiguous()

config = Config.from_directory(MODEL)
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
# One cache set only: the GDN recurrent-state binding is per-MODEL, so a second
# Cache for the same model leaves the slots sized for whichever was made last.
# Each variant therefore runs in its OWN process, selected by MIXBITS
# (MIXBITS=0 -> fp16 baseline, no quantization).
NDT = int(os.environ.get("NDT", "5"))
cache = Cache(model, max_num_tokens=8192, max_history=NDT + 1)
dc = Cache(dm, max_num_tokens=8192)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)

# Collect every GatedResidual site on both models
sites = []
def walk(m):
    if type(m).__name__ == "GatedResidual" and getattr(m, "fn_h", None) is not None:
        sites.append(m)
    for sm in getattr(m, "modules", []) or []:
        walk(sm)
    inner = getattr(m, "inner", None)
    if inner is not None and inner is not m: walk(inner)
walk(model); walk(dm)
print(f"GatedResidual sites found: {len(sites)}  (bits={BITS})")

def run():
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc,
                    num_draft_tokens=NDT)
    ids = tok.encode(PROMPT, add_bos=False)
    job = Job(input_ids=ids, max_new_tokens=200, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job); txt = ""
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming": txt += r.get("text", "") or ""
    el = time.perf_counter() - t0
    acc, rej = job.accepted_draft_tokens, job.rejected_draft_tokens
    return txt, el, acc, rej

# Quantize every site (skipped entirely when MIXBITS=0):
#   fn_h  is (M, H*D)          contracting over H*D -> axis 1
#   upx_h is (H, D/4, rank, 4) contracting over rank -> axis 2
if BITS > 0:
    with torch.inference_mode(False):
        for m in sites:
            m.fn_h = qdq(m.fn_h, BITS, 1)
            m.upx_h = qdq(m.upx_h, BITS, 2)

txt, el, acc, rej = run()
label = "fp16 baseline" if BITS == 0 else f"int{BITS} mixer  "
import hashlib
print(f"\nRESULT bits={BITS} {label}: {200/el:5.1f} t/s  "
      f"accept {acc}/{acc+rej} = {acc/(acc+rej)*100:.1f}%  "
      f"sha1={hashlib.sha1(txt.encode()).hexdigest()[:12]}  chars={len(txt)}")
print("----- first 300 chars -----")
print(txt[:300])
