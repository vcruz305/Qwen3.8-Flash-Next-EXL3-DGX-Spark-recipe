#!/usr/bin/env python3
"""Native exllamav3 benchmark. One job at a time, same prompt builder and salts as
qbench.py so prompt_tokens line up with the vLLM runs. Decode excludes TTFT.
usage: bench_native.py --model DIR --tag T --ctxs 4096,32768 --samples 4 --maxgen 128
       --modes 0,2,3 --maxlen 40960 --out OUT.json
mode 0 = no draft; k>0 = MTP draft with num_draft_tokens=k."""
import argparse, json, os, sys, threading, time
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
        m = mem_avail_gib()
        if m < floor_gib:
            print(f"WATCHDOG: MemAvailable {m:.2f} GiB < {floor_gib} GiB, aborting", flush=True)
            os._exit(3)
        time.sleep(1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ctxs", default="4096,32768")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--maxgen", type=int, default=128)
ap.add_argument("--modes", default="0,2,3")
ap.add_argument("--maxlen", type=int, default=40960)
ap.add_argument("--headroom-gib", type=float, default=6.0)
ap.add_argument("--floor-gib", type=float, default=2.0)
ap.add_argument("--prealloc-gib", type=float, default=0.0,
                help="allocate and free this much on cuda first so the kernel reclaims page cache; "
                     "on GB10 unified memory cudaMemGetInfo reports MemFree, not MemAvailable")
ap.add_argument("--draft-stats", action="store_true",
                help="record per-round (position, window, accepted) and report per-position acceptance")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler

pack_gib = sum(os.path.getsize(os.path.join(a.model, f)) for f in os.listdir(a.model)
               if f.endswith(".safetensors")) / 2**30
need = pack_gib + a.headroom_gib
have = mem_avail_gib()
print(f"pack {pack_gib:.1f} GiB, need {need:.1f} GiB, MemAvailable {have:.1f} GiB", flush=True)
if have < need:
    print("PRECHECK FAIL: not enough memory to load safely", flush=True)
    sys.exit(2)
threading.Thread(target=watchdog, args=(a.floor_gib,), daemon=True).start()

def cuda_free_gib():
    f, t = torch.cuda.mem_get_info()
    return f / 2**30, t / 2**30
print("cuda free/total before: %.1f / %.1f GiB" % cuda_free_gib(), flush=True)
if a.prealloc_gib > 0:
    blk = torch.empty(int(a.prealloc_gib * 2**30), dtype=torch.uint8, device="cuda")
    blk.fill_(0)
    del blk
    torch.cuda.empty_cache()
    print("cuda free/total after prealloc %.0f GiB: %.1f / %.1f GiB" % ((a.prealloc_gib,) + cuda_free_gib()),
          flush=True)

modes = [int(x) for x in a.modes.split(",")]
# recurrent layers keep one past state per draft token; reserved at cache creation
hist = max(modes)
t0 = time.time()
config = Config.from_directory(a.model)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=a.maxlen, max_history=hist, max_batch_size=1)
model.load(progressbar=False)
tokenizer = Tokenizer.from_config(config)
print(f"model loaded in {time.time()-t0:.0f}s, MemAvailable {mem_avail_gib():.1f} GiB", flush=True)

draft_model = draft_cache = None
if any(k > 0 for k in modes):
    t1 = time.time()
    draft_model = Model.from_config(config, component="mtp")
    draft_cache = Cache(draft_model, max_num_tokens=a.maxlen, max_history=hist,
                        max_batch_size=1)
    draft_model.load(progressbar=False)
    print(f"mtp draft loaded in {time.time()-t1:.0f}s, caps {draft_model.caps}, "
          f"MemAvailable {mem_avail_gib():.1f} GiB", flush=True)

def run_job(gen, ids, maxgen):
    job = Job(input_ids=ids, max_new_tokens=maxgen, min_new_tokens=maxgen,
              sampler=ArgmaxSampler(), seed=0)
    gen.enqueue(job)
    text = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            text += r.get("text", "")
    n = job.new_tokens
    ttft = job.time_first_token - job.time_first_prefill
    dec_s = job.time_last_token - job.time_first_token
    per_pos = None
    if job.draft_stats:
        # unconditional per-position acceptance, the same definition vLLM reports:
        # share of verification rounds in which draft position i was accepted
        rounds = [st for st in job.draft_stats if st[1] > 0]
        kmax = max(st[1] for st in rounds)
        per_pos = [round(sum(1 for st in rounds if st[2] >= i) / len(rounds), 3)
                   for i in range(1, kmax + 1)]
        per_pos = {"per_position": per_pos, "rounds": len(rounds),
                   "mean_accepted": round(sum(st[2] for st in rounds) / len(rounds), 3)}
    return {"prompt_tokens": int(ids.shape[-1]), "gen": int(n), "draft": per_pos,
            "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(ids.shape[-1] / ttft, 1),
            "decode_tok_s": round((n - 1) / dec_s, 2) if n > 1 and dec_s > 0 else 0.0,
            "accepted_draft_tokens": int(job.accepted_draft_tokens),
            "text_head": text[:60].replace("\n", " ")}

results = {"tag": a.tag, "model": a.model, "engine": "exllamav3-native",
           "maxgen": a.maxgen, "modes": {}}
ctxs = [int(c) for c in a.ctxs.split(",")]
for k in modes:
    if k > 0:
        gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048,
                        draft_model=draft_model, draft_cache=draft_cache, num_draft_tokens=k,
                        record_draft_stats=a.draft_stats)
    else:
        gen = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048)
    # warmup: kernel autotune and allocator growth land here, not in a measured sample
    w = tokenizer.encode(build(1024, 7), add_bos=False)
    run_job(gen, w, 16)
    rows = {}
    for c in ctxs:
        rs = []
        for i in range(a.samples):
            ids = tokenizer.encode(build(c, 1000 + i), add_bos=False)
            r = run_job(gen, ids, a.maxgen)
            rs.append(r)
            print(f"[{a.tag} k={k} ctx={c} #{i}] ptok={r['prompt_tokens']} gen={r['gen']} "
                  f"draft={r['draft']} "
                  f"ttft={r['ttft_s']}s prefill={r['prefill_tok_s']} decode={r['decode_tok_s']} "
                  f"acc={r['accepted_draft_tokens']} | {r['text_head']}", flush=True)
        dec = sorted(x["decode_tok_s"] for x in rs)
        pre = sorted(x["prefill_tok_s"] for x in rs)
        rows[str(c)] = {"prompt_tokens": rs[0]["prompt_tokens"], "n": len(rs),
                        "decode_p50": dec[len(dec) // 2], "decode_max": dec[-1],
                        "prefill_p50": pre[len(pre) // 2],
                        "ttft_p50": sorted(x["ttft_s"] for x in rs)[len(rs) // 2],
                        "accept_share": round(sum(x["accepted_draft_tokens"] for x in rs)
                                              / max(1, sum(x["gen"] for x in rs)), 3),
                        "samples": rs}
        print(f"SUMMARY {a.tag} k={k} ctx={c}: ptok={rows[str(c)]['prompt_tokens']} "
              f"decode_p50={rows[str(c)]['decode_p50']} prefill_p50={rows[str(c)]['prefill_p50']} "
              f"ttft_p50={rows[str(c)]['ttft_p50']} accept_share={rows[str(c)]['accept_share']}",
              flush=True)
    results["modes"][str(k)] = rows
    del gen
    torch.cuda.empty_cache()
    json.dump(results, open(a.out, "w"), indent=1)
print("DONE", flush=True)
