#!/usr/bin/env python3
"""Decode-only kernel breakdown per speculative round at the current best config."""
import sys, time, os, torch, collections
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
from torch.profiler import profile, ProfilerActivity

NDT = int(os.environ.get("NDT", "5"))
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_history=max(4, NDT))
dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=8192)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc, num_draft_tokens=NDT)
rounds = {"n": 0}
_tf = model.forward
def tf(*a, **k):
    rounds["n"] += 1; return _tf(*a, **k)
model.forward = tf
p = "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict. Include a docstring and type hints.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok.encode(p, add_bos=False)
def run(n, prof=None):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job); started = False
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and not started and prof is not None:
                prof.start(); rounds["n"] = 0; started = True   # skip prefill
    torch.cuda.synchronize()
run(12)
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    prof.stop()
    run(200, prof)
wall = time.perf_counter() - t0
R = rounds["n"]
ka = prof.key_averages()
def dev(e): return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
tot = sum(dev(e) for e in ka) / 1e3
print(f"ndt={NDT}: {R} target rounds (verify), profiled GPU time {tot:.0f} ms -> {tot/R:.2f} ms/round GPU; {200/wall:.1f} t/s incl. overhead")
print(f"{'kernel':<64} {'calls':>6} {'/rnd':>5} {'ms':>8} {'us/call':>8} {'ms/rnd':>7} {'%':>5}")
groups = collections.defaultdict(lambda: [0, 0.0])
for e in sorted(ka, key=lambda e: -dev(e))[:34]:
    ms = dev(e) / 1e3
    if ms < 1: continue
    print(f"{e.key[:64]:<64} {e.count:>6} {e.count/R:>5.1f} {ms:>8.1f} {ms*1000/e.count:>8.1f} {ms/R:>7.2f} {100*ms/tot:>5.1f}")
