#!/usr/bin/env python3
"""Who calls ext.exl3_moe (the 1.9 ms prefill-tier MoE kernel) during MTP decode? Log input rows & module key."""
import sys, time, os, torch, collections, traceback
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.modules.block_sparse_mlp as B
NDT = 4
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_history=4)
dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=8192)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)
calls = collections.Counter(); shapes = collections.Counter()
_moe = B.ext.exl3_moe
def hook(*a, **k):
    st = traceback.extract_stack(limit=6)
    calls[tuple(f"{os.path.basename(f.filename)}:{f.lineno}" for f in st[-4:-1])] += 1
    return _moe(*a, **k)
B.ext.exl3_moe = hook
# also record forward row counts per module class path
_fwd = B.BlockSparseMLP.forward
def fwd(self, x, params, out_dtype=None):
    rows = x.numel() // x.shape[-1]
    key = ("mtp" if self.key.startswith("mtp") else "target", rows)
    shapes[key] += 1
    return _fwd(self, x, params, out_dtype)
B.BlockSparseMLP.forward = fwd
gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc, num_draft_tokens=NDT)
p = "<|im_start|>user\nWrite a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a usage example.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ids = tok.encode(p, add_bos=False)
job = Job(input_ids=ids, max_new_tokens=120, sampler=GreedySampler(), stop_conditions=[])
gen.enqueue(job)
while gen.num_remaining_jobs():
    for r in gen.iterate(): pass
rounds = int(round((job.accepted_draft_tokens + job.rejected_draft_tokens) / NDT))
print("rounds", rounds)
print("BlockSparseMLP.forward (model, rows) -> calls:", dict(shapes))
print("ext.exl3_moe call sites -> count:")
for k, v in calls.most_common(): print("  ", v, k)
