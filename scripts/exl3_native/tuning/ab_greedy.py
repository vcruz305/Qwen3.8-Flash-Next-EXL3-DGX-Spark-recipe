#!/usr/bin/env python3
"""Greedy A/B: same prompt, dump token ids + t/s. Run twice with EXL3_GR_RB=0 / 1 and diff."""
import sys, os, time, torch, json
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
NDT = int(os.environ.get("NDT", "5")); N = int(os.environ.get("NTOK", "300"))
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_history=max(4, NDT))
dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=8192)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc, num_draft_tokens=NDT)
p = "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok.encode(p, add_bos=False)
def run(n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    gen.enqueue(job); out = []; t0 = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                if t0 is None: t0 = time.perf_counter()
                if r.get("token_ids") is not None: out += r["token_ids"].view(-1).tolist()
    torch.cuda.synchronize(); return out, time.perf_counter() - t0, job
run(16)
out, dt, job = run(N)
tag = os.environ.get("EXL3_GR_RB", "1")
print(f"GR_RB={tag} ndt={NDT}: {len(out)} tok in {dt:.2f}s = {len(out)/dt:.1f} t/s  accepted={job.accepted_draft_tokens} rejected={job.rejected_draft_tokens}")
json.dump(out, open(os.path.expanduser(f"~/ab_rb{tag}.json"), "w"))
print("TEXT:", tok.decode(torch.tensor([out[:60]]))[0].replace("\n", " ")[:200])
