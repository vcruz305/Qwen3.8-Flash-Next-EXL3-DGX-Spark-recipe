#!/usr/bin/env python3
"""Long-context fill test driven through Generator/Job (chat.py -prompt caps at 128 KiB argv).

Builds a prompt of ~K thousand tokens from repeated source text with a NEEDLE planted at a
random depth ("The secret access code for vault {n} is {code}."), asks for the code, and
reports: prompt tokens, prefill t/s, decode t/s at that depth, MTP acceptance, needle hit,
peak host memory. Cache is sized 1.1x the prompt (page-aligned) unless CS is given.

  K=128 python ctxfill.py            # ~128k tokens
  K=480 CS=524288 python ctxfill.py  # past max_position_embeddings (262144)
  CQ=8,8 -> quantized KV
"""
import sys, os, time, random, re, threading, subprocess
sys.path.insert(0, os.path.expanduser("~/exllamav3"))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler

K = int(os.environ.get("K", "32")); NDT = 5
NEW = int(os.environ.get("NEW", "32"))
TASK = os.environ.get("TASK", "needle")
MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3")
SRC = open(os.path.join(MODEL, "qbench_prompts.md")).read()

config = Config.from_directory(MODEL)
tok = Tokenizer.from_config(config)

# --- build prompt to a TOKEN target, needle at a random depth in the middle 80%
random.seed(K * 100 + int(os.environ.get("SEED", "0")))
code = f"{random.randint(1000, 9999)}-{random.choice('ABCDEFGH')}{random.randint(10, 99)}"
needle = f"\n\nThe secret access code for vault 7 is {code}. Remember it.\n\n"
target = K * 1000
chunks, n_tok = [], 0
i = 0
while n_tok < target:
    piece = f"\n\n===== DOCUMENT SECTION {i} =====\n" + SRC
    chunks.append(piece); n_tok += tok.encode(piece, add_bos=False).shape[-1]; i += 1
body = "".join(chunks)
ids_body = tok.encode(body, add_bos=False)[0]
ids_body = ids_body[:target]
body = tok.decode(ids_body.unsqueeze(0))[0]
pos = int(len(body) * random.uniform(0.1, 0.9))
body = body[:pos] + needle + body[pos:]
if TASK == "needle":
    question = "Question: what is the secret access code for vault 7? Answer with only the code."
else:
    # long generation at depth: a code task whose answer is unrelated to the documents, so the
    # decode t/s measures attention-over-context cost, not retrieval
    question = ("Ignore the documents above. Write a Python function that parses an nginx access log "
                "line into a dict with fields ip, timestamp, method, path, status, bytes. Include a "
                "docstring, type hints, and a short usage example. Then explain each regex group in one bullet each.")
prompt = ("<|im_start|>user\nBelow is a long collection of documents.\n" + body +
          "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
ids = tok.encode(prompt, add_bos=False)
n_prompt = ids.shape[-1]
CS = int(os.environ.get("CS", "0")) or ((n_prompt * 11 // 10 + 4096) // 4096 * 4096)
CQ = os.environ.get("CQ")
print(f"prompt tokens {n_prompt:,}  cache {CS:,}  needle depth {pos/len(body)*100:.0f}%  quant={CQ or 'fp16'}", flush=True)

# --- host memory watcher
peak = {"used": 0}
def watch():
    while not peak.get("stop"):
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f}
        peak["used"] = max(peak["used"], (m["MemTotal"] - m["MemAvailable"]) // 1024)
        time.sleep(1)
threading.Thread(target=watch, daemon=True).start()

# --- load, mirroring model_init's MTP wiring
model = Model.from_config(config)
dm = Model.from_config(config, component="mtp")
ckw = {}
if CQ:
    b = [int(x) for x in CQ.split(",")]; kb, vb = (b[0], b[0]) if len(b) == 1 else b
    ckw = dict(layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
cache = Cache(model, max_num_tokens=CS, max_history=NDT + 1, **ckw)
dcache = Cache(dm, max_num_tokens=CS, **ckw)
t0 = time.time()
dm.load(progressbar=False); model.load(progressbar=False)
print(f"loaded in {time.time()-t0:.0f}s, host used {peak['used']/1024:.1f} GiB", flush=True)

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache,
                num_draft_tokens=NDT, dynamic_draft_tokens=True, draft_confidence=0.6, max_chunk_size=4096)
job = Job(input_ids=ids, max_new_tokens=NEW, sampler=GreedySampler(), stop_conditions=[] if TASK != 'needle' else [tok.eos_token_id])
gen.enqueue(job)
out, t_start, t_first = "", time.time(), None
n_out = 0
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("error"): print("JOB ERROR:", repr(r["error"])); raise SystemExit(1)
        if r.get("stage") == "streaming":
            if t_first is None: t_first = time.time()
            out += r.get("text", "") or ""
t_end = time.time()
n_out = tok.encode(out, add_bos=False).shape[-1]
acc, rej = job.accepted_draft_tokens, job.rejected_draft_tokens
prefill_tps = n_prompt / (t_first - t_start)
decode_tps = (n_out - 1) / max(t_end - t_first, 1e-6)
hit = (code in out) if TASK == "needle" else None
peak["stop"] = True
print(f"RESULT K={K} prompt={n_prompt:,} cache={CS:,} quant={CQ or 'fp16'} "
      f"prefill={prefill_tps:,.0f} t/s ({t_first-t_start:.1f}s) decode={decode_tps:.1f} t/s "
      f"accept={acc}/{acc+rej} new={n_out} needle={'n/a' if hit is None else ('HIT' if hit else 'MISS')} peak_host={peak['used']/1024:.1f}GiB")
print(f"  expected {code!r}  got {out.strip()[:80]!r}")
