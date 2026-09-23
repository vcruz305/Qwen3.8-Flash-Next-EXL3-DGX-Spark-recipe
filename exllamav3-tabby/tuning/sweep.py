#!/usr/bin/env python3
"""In-process sweep: load the model once, then measure greedy decode t/s across (ndt, cpu-affinity,
sampler) configurations. Env-var knobs must be set before launch (read at import)."""
import sys, time, os, argparse, torch
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler, DefaultSampler

ap = argparse.ArgumentParser()
ap.add_argument("--ndts", default="3,4,5")
ap.add_argument("--affinity", default="all,big")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--prompts", default="code,prose")
ap.add_argument("--cs", type=int, default=8192)
ap.add_argument("--sampler", default="greedy")   # greedy | qwen_instruct
ap.add_argument("--tag", default="")
a = ap.parse_args()
NDTS = [int(x) for x in a.ndts.split(",")]
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
BIG = {c for c in range(20) if int(open(f"/sys/devices/system/cpu/cpu{c}/cpu_capacity").read()) > 900}
AFF = {"all": set(range(20)), "big": BIG, "one_big": {max(BIG)}}

config = Config.from_directory(MODEL)
model = Model.from_config(config)
mh = max(4, max(NDTS))
cache = Cache(model, max_num_tokens=a.cs, max_history=mh)
dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=a.cs)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)
PROMPTS = {
 "code": "Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each.",
 "prose": "Write a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm. Use varied sentence structure and specific sensory details.",
 "devops": "Explain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML.",
}
def make_sampler():
    if a.sampler == "greedy": return GreedySampler()
    # Qwen recommended instruct settings
    return DefaultSampler(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0)

def run(gen, prompt, n):
    p = f"<|im_start|>user\n{PROMPTS[prompt]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(p, add_bos=False)
    job = Job(input_ids=ids, max_new_tokens=n, sampler=make_sampler(), stop_conditions=[])
    gen.enqueue(job); out = ""; t_first = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                if t_first is None: t_first = time.time()
                out += r.get("text", "")
    torch.cuda.synchronize(); t1 = time.time()
    ntok = len(tok.encode(out, add_bos=False)[0])
    acc = ""
    if hasattr(job, "accepted_draft_tokens") and getattr(job, "total_draft_tokens", 0):
        acc = f" accept={job.accepted_draft_tokens}/{job.total_draft_tokens}"
    return ntok / (t1 - t_first), acc

print(f"big cores: {sorted(BIG)}  sampler={a.sampler} {a.tag}", flush=True)
for aff in a.affinity.split(","):
    os.sched_setaffinity(0, AFF[aff])
    for ndt in NDTS:
        gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm if ndt else None,
                        draft_cache=dc if ndt else None, num_draft_tokens=ndt or None)
        run(gen, "code", 24)  # warm
        for pr in a.prompts.split(","):
            tps, acc = run(gen, pr, a.tokens)
            print(f"aff={aff:7s} ndt={ndt} {pr:6s}: {tps:6.1f} t/s{acc}", flush=True)
        del gen
