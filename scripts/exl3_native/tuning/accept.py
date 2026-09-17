#!/usr/bin/env python3
"""Per-position MTP draft acceptance for a prompt: how far into the k-token draft does the target
keep agreeing? Greedy so runs are reproducible. Also the no-draft baseline for the same prompt."""
import sys, os, time, torch, collections
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import GreedySampler, DefaultSampler
PROMPTS = {
 "code": "Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each.",
 "prose": "Write a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm. Use varied sentence structure and specific sensory details.",
 "devops": "Explain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML.",
 "essay": "Write a 350-word essay arguing that remote work improves engineering productivity. Use a clear thesis, three supporting points, and a conclusion.",
}
NDTS = [int(x) for x in os.environ.get("NDTS", "1,2,3,5").split(",")]
NTOK = int(os.environ.get("NTOK", "400"))
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
config = Config.from_directory(MODEL); model = Model.from_config(config)
cache = Cache(model, max_num_tokens=8192, max_history=max(4, max(NDTS)))
dm = Model.from_config(config, component="mtp"); dc = Cache(dm, max_num_tokens=8192)
dm.load(progressbar=False); model.load(progressbar=False)
tok = Tokenizer.from_config(config)

def run(gen, name, n, sampler):
    p = f"<|im_start|>user\n{PROMPTS[name]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = tok.encode(p, add_bos=False)
    job = Job(input_ids=ids, max_new_tokens=n, sampler=sampler, stop_conditions=[])
    gen.enqueue(job); t0 = None; out = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming":
                if t0 is None: t0 = time.perf_counter()
                out += r.get("text", "") or ""
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    return job, len(tok.encode(out, add_bos=False)[0]) / dt, out

for name in os.environ.get("PROMPTS", "code,prose,essay").split(","):
    print(f"\n=== {name} ===")
    g0 = Generator(model=model, cache=cache, tokenizer=tok)
    run(g0, name, 16, GreedySampler())
    _, tps0, _ = run(g0, name, NTOK, GreedySampler()); del g0
    print(f"  no draft: {tps0:.1f} t/s")
    for k in NDTS:
        DDS = os.environ.get("DDS")  # e.g. "0.4" -> dynamic drafts with that confidence target
        kw = dict(dynamic_draft_tokens=True, draft_confidence=float(DDS)) if DDS else {}
        gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dc,
                        num_draft_tokens=k, record_draft_stats=True, **kw)
        run(gen, name, 16, GreedySampler())
        job, tps, out = run(gen, name, NTOK, GreedySampler())
        st = getattr(job, "draft_stats", None) or []
        # st: list of (position, window, accepted)
        acc_hist = collections.Counter(a for _, _, a in st)
        rounds = len(st); tot_acc = sum(a for _, _, a in st)
        per_pos = []
        for i in range(k):
            reached = sum(1 for _, w, a in st if w > i)
            ok = sum(1 for _, w, a in st if a > i)
            per_pos.append(f"{ok/max(reached,1):.2f}")
        print(f"  ndt={k}: {tps:5.1f} t/s  rounds={rounds} tok/round={(tot_acc+rounds)/max(rounds,1):.2f}  "
              f"accept={100*job.accepted_draft_tokens/max(job.accepted_draft_tokens+job.rejected_draft_tokens,1):.0f}%  "
              f"P(pos i accepted | reached)={per_pos}  accepted-count hist={dict(sorted(acc_hist.items()))}")
        del gen
    print("  sample:", out[:160].replace("\n", " "))
