#!/usr/bin/env python3
"""Instrument one speculative round: time draft_model.forward (x ndt) vs model.forward (target
verify, q_len = ndt+1) with CUDA sync, so we know the split. Also time a plain 1-token target step."""
import sys, time, os, torch
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.generator.generator as G

NDT = int(os.environ.get("NDT", "4"))
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_history=max(4, NDT))
dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=8192)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)

# monkeypatch forwards to accumulate synced time
acc = {"draft": 0.0, "draft_n": 0, "target": 0.0, "target_n": 0, "target_q": []}
_df, _tf = dm.forward, model.forward
def df(*a_, **k):
    torch.cuda.synchronize(); t = time.perf_counter(); r = _df(*a_, **k); torch.cuda.synchronize()
    acc["draft"] += time.perf_counter() - t; acc["draft_n"] += 1; return r
def tf(*a_, **k):
    x = a_[0] if a_ else k.get("input_ids")
    torch.cuda.synchronize(); t = time.perf_counter(); r = _tf(*a_, **k); torch.cuda.synchronize()
    acc["target"] += time.perf_counter() - t; acc["target_n"] += 1; acc["target_q"].append(int(x.shape[-1])); return r
dm.forward, model.forward = df, tf

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc, num_draft_tokens=NDT)
p = "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict. Include a docstring and type hints.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok.encode(p, add_bos=False)
def run(n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job); out = ""; iters = 0; t_first = time.perf_counter(); fin = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                out += r.get("text", "") or ""; iters += 1
                if r.get("eos"): fin = r
    torch.cuda.synchronize()
    if fin: print(f"  [job] new_tokens={fin.get('new_tokens')} time_generate={fin.get('time_generate'):.3f}s eos_reason={fin.get('eos_reason')}")
    return len(tok.encode(out, add_bos=False)[0]), iters, time.perf_counter() - t_first, job
run(12)
for k in acc: acc[k] = 0.0 if isinstance(acc[k], float) else (0 if isinstance(acc[k], int) else [])
ntok, iters, wall, job = run(200)
from collections import Counter
qs = Counter(acc["target_q"])
rounds = acc["target_n"]
print(f"ndt={NDT}: {ntok} tok, {iters} iters, wall {wall*1e3:.0f} ms -> {ntok/wall:.1f} t/s")
print(f"  target forwards: {rounds}  q_len histogram {dict(qs)}  total {acc['target']*1e3:.0f} ms  = {acc['target']/rounds*1e3:.2f} ms each")
print(f"  draft forwards : {acc['draft_n']} ({acc['draft_n']/rounds:.1f}/round)  total {acc['draft']*1e3:.0f} ms = {acc['draft']/max(1,acc['draft_n'])*1e3:.2f} ms each")
other = wall - acc["target"] - acc["draft"]
print(f"  other (host/sampling/gather/sync): {other*1e3:.0f} ms = {other/rounds*1e3:.2f} ms/round")
print(f"  tokens/round: {ntok/rounds:.2f}   accepted={job.accepted_draft_tokens} rejected={job.rejected_draft_tokens}")
