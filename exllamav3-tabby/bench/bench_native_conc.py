#!/usr/bin/env python3
"""Concurrency sweep on exllamav3's dynamic generator, mirroring the recipe's vLLM test:
N streams enqueued together, unique prompt prefix per stream, 128 new tokens each.
Aggregate is total new tokens over the window from the first stream's first token to the
last stream's last token. Per-stream is (new_tokens-1)/(its own decode window).
usage: bench_native_conc.py --model DIR --tag T --streams 1,2,4,8 --rounds 2 --modes 0,3
       --ctx 4096 --maxgen 128 --maxlen 65536 --prealloc-gib 100 --out OUT.json"""
import argparse, json, os, statistics, sys, threading, time
import torch

FILLER = ("The maintenance log for reactor bay seven records a pressure excursion at "
          "oh four hundred hours, followed by a manual override and a return to nominal. "
          "Subsequent inspection found no fault in the primary loop. ")

def build(ntok, salt):
    reps = max(1, int(ntok * 4 / len(FILLER)))
    return "Record %d. " % salt + FILLER * reps + "\n\nSummarize the above in one sentence."

def mem_avail_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return 0.0

def watchdog(floor_gib):
    while True:
        if mem_avail_gib() < floor_gib:
            print(f"WATCHDOG: MemAvailable under {floor_gib} GiB, aborting", flush=True)
            os._exit(3)
        time.sleep(1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--streams", default="1,2,4,8")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--modes", default="0,3")
ap.add_argument("--ctx", type=int, default=4096)
ap.add_argument("--maxgen", type=int, default=128)
ap.add_argument("--maxlen", type=int, default=65536)
ap.add_argument("--headroom-gib", type=float, default=6.0)
ap.add_argument("--floor-gib", type=float, default=2.0)
ap.add_argument("--prealloc-gib", type=float, default=0.0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler

pack_gib = sum(os.path.getsize(os.path.join(a.model, f)) for f in os.listdir(a.model)
               if f.endswith(".safetensors")) / 2**30
if mem_avail_gib() < pack_gib + a.headroom_gib:
    print("PRECHECK FAIL", flush=True); sys.exit(2)
threading.Thread(target=watchdog, args=(a.floor_gib,), daemon=True).start()
if a.prealloc_gib > 0:
    blk = torch.empty(int(a.prealloc_gib * 2**30), dtype=torch.uint8, device="cuda"); blk.fill_(0)
    del blk; torch.cuda.empty_cache()
print("cuda free/total: %.1f / %.1f GiB" % tuple(v / 2**30 for v in torch.cuda.mem_get_info()), flush=True)

streams = [int(s) for s in a.streams.split(",")]
modes = [int(m) for m in a.modes.split(",")]
nmax = max(streams)
hist = max(modes)
t0 = time.time()
config = Config.from_directory(a.model)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=a.maxlen, max_history=hist, max_batch_size=nmax)
model.load(progressbar=False)
tokenizer = Tokenizer.from_config(config)
draft_model = draft_cache = None
if hist > 0:
    draft_model = Model.from_config(config, component="mtp")
    draft_cache = Cache(draft_model, max_num_tokens=a.maxlen, max_history=hist, max_batch_size=nmax)
    draft_model.load(progressbar=False)
print(f"loaded in {time.time()-t0:.0f}s, MemAvailable {mem_avail_gib():.1f} GiB", flush=True)

def run_round(gen, n, salt0):
    jobs = []
    for i in range(n):
        ids = tokenizer.encode(build(a.ctx, salt0 + i), add_bos=False)
        jobs.append(Job(input_ids=ids, max_new_tokens=a.maxgen, min_new_tokens=a.maxgen,
                        sampler=ArgmaxSampler(), seed=0))
    t_enq = time.time()
    for j in jobs:
        gen.enqueue(j)
    while gen.num_remaining_jobs():
        gen.iterate()
    per = []
    for j in jobs:
        dec = j.time_last_token - j.time_first_token
        per.append({"gen": int(j.new_tokens), "ttft_s": round(j.time_first_token - t_enq, 3),
                    "decode_tok_s": round((j.new_tokens - 1) / dec, 2) if dec > 0 else 0.0,
                    "accepted": int(j.accepted_draft_tokens)})
    window = max(j.time_last_token for j in jobs) - min(j.time_first_token for j in jobs)
    total = sum(j.new_tokens for j in jobs)
    return {"streams": n, "aggregate_tok_s": round(total / window, 2), "window_s": round(window, 3),
            "wall_s": round(max(j.time_last_token for j in jobs) - t_enq, 3),
            "per_stream_median": round(statistics.median(p["decode_tok_s"] for p in per), 2),
            "per_stream": per}

results = {"tag": a.tag, "model": a.model, "engine": "exllamav3-native", "ctx": a.ctx,
           "maxgen": a.maxgen, "rounds": a.rounds, "modes": {}}
salt = 5000
for k in modes:
    kw = dict(max_batch_size=nmax, max_chunk_size=2048)
    if k > 0:
        kw.update(draft_model=draft_model, draft_cache=draft_cache, num_draft_tokens=k)
    gen = Generator(model, cache, tokenizer, **kw)
    run_round(gen, 1, 9)  # warmup
    rows = {}
    for n in streams:
        rs = []
        for r in range(a.rounds):
            salt += n
            res = run_round(gen, n, salt)
            rs.append(res)
            print(f"[{a.tag} k={k} streams={n} round={r}] aggregate={res['aggregate_tok_s']} "
                  f"per_stream_median={res['per_stream_median']} wall={res['wall_s']}s "
                  f"per={[p['decode_tok_s'] for p in res['per_stream']]}", flush=True)
        rows[str(n)] = {"aggregate_p50": statistics.median(x["aggregate_tok_s"] for x in rs),
                        "per_stream_p50": statistics.median(x["per_stream_median"] for x in rs),
                        "rounds": rs}
        print(f"SUMMARY {a.tag} k={k} streams={n}: aggregate={rows[str(n)]['aggregate_p50']} "
              f"per_stream={rows[str(n)]['per_stream_p50']}", flush=True)
    results["modes"][str(k)] = rows
    del gen
    torch.cuda.empty_cache()
    json.dump(results, open(a.out, "w"), indent=1)
print("DONE", flush=True)
