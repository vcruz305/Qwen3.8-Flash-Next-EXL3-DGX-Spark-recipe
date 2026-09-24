# Qwen3.8-Flash-Next EXL3 on one NVIDIA DGX Spark

Runs [turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(revision `3.05bpw_h5_ng5`, about 80 GB) on a single NVIDIA DGX Spark (GB10,
128 GB unified memory, aarch64).

**The recipe is one path:** the latest [TabbyAPI](https://github.com/theroyallab/tabbyAPI)
as the OpenAI-compatible server, running on the
[vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) fork as the runtime,
built from source for sm_121. A vLLM route also exists; it is slower on this box
and lives, fully separate, in [`vllm-plugin/`](vllm-plugin/README.md).

> **Using an AI agent to set this up?** Point it at [AGENTS.md](AGENTS.md) first.
> It pins down which scripts to run and which to leave alone.

## Repository layout

| Path | What it is |
|---|---|
| [`exllamav3-tabby/`](exllamav3-tabby/) | **The recipe.** `setup.sh`, `serve.sh`, `chat.sh`, the TabbyAPI config, the page-cache helper |
| `exllamav3-tabby/beta/` | (Beta) a minimal OpenAI shim over the same runtime, kept for A/B work. Not the API to deploy |
| `exllamav3-tabby/tuning/`, `exllamav3-tabby/bench/` | The GB10 tuning research: A/B scripts, harnesses, logs, patches |
| `exllamav3-tabby/tools/` | `make_native_view.sh`, for a pack that was already rewritten for vLLM |
| `exllamav3-tabby/legacy/` | The stock exllamav3 1.5.0 build used for the stock-vs-vLLM baseline. Not the runtime |
| [`vllm-plugin/`](vllm-plugin/README.md) | **Separate, secondary route:** vLLM 0.29.0 + vllm-exl3. Own setup, own README |
| `bench/` | Engine-neutral `/v1` clients (`bench_v1.py`, `concurrency.py`) |
| `docs/`, `index.html` | The animated benchmark viewer |

## Quick start

On the Spark, from a clone of this repo:

```bash
# 1. Runtime + server (once; ~15 min, the CUDA extension build). Re-run to update TabbyAPI.
bash exllamav3-tabby/setup.sh

# 2. The pack (about 80 GB)
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 \
  --local-dir ~/models/Qwen3.8-Flash-Next-EXL3

# 3. Serve (OpenAI API on 127.0.0.1:8899)
bash exllamav3-tabby/serve.sh
```

Then:

```bash
curl -s http://127.0.0.1:8899/v1/models
curl -s http://127.0.0.1:8899/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next-EXL3",
  "messages": [{"role": "user", "content": "Write a haiku about unified memory."}],
  "max_tokens": 256}'
```

**What `setup.sh` installs, and what it refuses.** Everything goes under
`~/qwen38-exl3/` (`RECIPE_HOME`): one venv, the fork checked out at the commit
pinned in [`exllamav3-tabby/env.sh`](exllamav3-tabby/env.sh) and built for
sm_121, and TabbyAPI at the tip of `main`, deliberately unpinned. Re-running it
pulls the latest TabbyAPI and rebuilds the fork only when the pin moved. It
removes any stock `exllamav3` wheel it finds in the venv. Every launcher
(`serve.sh`, `chat.sh`, the beta shim) checks the *imported* module before it
loads anything, and exits with an error if it is not the fork: the version must be
≥ 1.5.1 (latest TabbyAPI requires it), the GB10 kernels (`gr_mix_int8`,
`exl3_moe_mixedk`) must be in the compiled extension, and the generator must
read `EXL3_DRAFT_CONFIDENCE`. `bash exllamav3-tabby/setup.sh --check` runs just
that check.

**Why TabbyAPI latest works with the fork unchanged.** The two GB10-specific
settings TabbyAPI has no config key for are environment variables the fork
reads, which `serve.sh` exports: the kernel knobs (`EXL3_INT8_GEMV=0
EXL3_MOE_COOP_WIDE=1 EXL3_GR_INT8=1 EXL3_MTP_HEAD_N=65536`) and the
dynamic-draft target (`EXL3_DRAFT_CONFIDENCE=0.6`, fork
[#11](https://github.com/vcruz305/exllamav3/pull/11)). Everything else, including MTP
depth 5, dynamic drafting, 8-bit KV, `ngram_ram`, and the `qwen3_5` tool format, is a
stock TabbyAPI key in [`tabby-config.yml`](exllamav3-tabby/tabby-config.yml). That
file is checked against TabbyAPI's own pydantic schema on `main` with no unknown keys.

### Context and concurrency

The KV cache in exllamav3 is **one pool shared by every concurrent request**,
not a per-request allowance. `max_seq_len` caps one request; `cache_size` is the
pool. The old instructions set `-cs 262144` with batch 4, which gives four users
262k *between them*, and the stock `chat.py` default is only 32k. Those are the
"context is tiny" reports. `serve.sh` has two profiles:

| `PROFILE=` | Per request | Shared pool | Concurrent jobs | n-gram table | Use |
|---|---:|---:|---:|---|---|
| `concurrent` (default) | 262,144 | 1,048,576 (4 full-length at once) | 4 | streamed from NVMe | agents, several clients |
| `single` | 262,144 | 262,144 | 1 | in RAM | one user, the measured single-stream setup |

At 8-bit KV this model costs about 13 KB per token of pool (12 of 48 layers
are full attention, plus the MTP layer, 2 KV heads at 256), so the concurrent
pool is ~13 GiB. Streaming the 30 GB n-gram table from NVMe (the engine
default) is what makes room for it. `serve.sh` prints the sizing and a memory
estimate against `MemAvailable` before it loads. Override any value: `CACHE_SIZE`,
`MAX_BATCH_SIZE`, `MAX_SEQ_LEN`, `NGRAM_RAM`, `DRAFT_NUM_TOKENS`, `PORT`, `HOST`,
`SERVED_NAME`, `MODEL_DIR`. `DRY_RUN=1` renders the config and prints the command
without starting.

Do not raise `MAX_SEQ_LEN` past 262,144. That is the trained window, and needle
retrieval fails beyond it at both KV precisions
([context](#context-the-ceiling-and-8-bit-kv)).

> **Measurement status.** The single-stream numbers below are `chat.py` on the
> fork, the same engine and knobs `serve.sh` uses. Throughput *through TabbyAPI*
> in either profile has not been re-measured since the move to TabbyAPI main;
> `bench/bench_v1.py` and `bench/concurrency.py` are the clients for it.

### Binding to the network

`serve.sh` binds `127.0.0.1` with auth off. With `HOST=0.0.0.0` it turns auth
on: TabbyAPI writes `api_tokens.yml` into `~/qwen38-exl3/tabbyAPI/` on first
start, and clients send `Authorization: Bearer <api_key>`.

### Console chat (single user, no server)

```bash
bash exllamav3-tabby/chat.sh
```

This is `examples/chat.py` with the tuned GB10 flags, the configuration behind
the single-stream table below.

### Beta: the minimal OpenAI shim

`exllamav3-tabby/beta/serve_openai.sh` is a small `/v1/chat/completions` server
over the same fork and knobs. **It is beta and not the deployment path.** It runs one
generation at a time, hand-rolls the prompt format, and exists for A/B comparisons
against TabbyAPI. It runs the same runtime check as the other launchers. See
[`exllamav3-tabby/beta/README.md`](exllamav3-tabby/beta/README.md).

## Contents

- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
  - [Context and concurrency](#context-and-concurrency)
- [Current numbers](#current-numbers-2026-09-17)
- [Which to use](#which-to-use)
- [Benchmarks: exllamav3](#benchmarks-exllamav3-native)
  - [Stock exllamav3 1.5.0 vs vLLM](#running-the-pack-through-exllamav3-directly)
  - [Can the plugin get there?](#can-the-plugin-get-there)
  - [The native engine, tuned for GB10](#the-native-engine-tuned-for-gb10)
- [History of the native-engine numbers](#history-of-the-native-engine-numbers)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Related repositories, credits, license](#related-repositories)
- vLLM route: [`vllm-plugin/README.md`](vllm-plugin/README.md)

## Current numbers (2026-09-17)

### exllamav3 fork (the recipe)

> **Hardware:** NVIDIA DGX Spark — GB10 SoC, 128 GB unified memory, aarch64, Grace CPU (10 big + 10 little cores).

**Fastest path.** [Configured for GB10](#the-native-engine-tuned-for-gb10).
One stream, greedy, 400 new tokens, cold load, through `examples/chat.py`
(`exllamav3-tabby/chat.sh`), on
[vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051).
The recipe now pins `94ba01d`, which contains `329e051` plus upstream's 1.5.1
bump and the `EXL3_DRAFT_CONFIDENCE` hook; not re-measured on it yet:

| Prompt class | Decode tok/s | Draft acceptance |
|---|---:|---:|
| Code (nginx log parser) | **79** | 73% |
| DevOps explainer + YAML | **62** | 59% |
| Prose (350-word story) | **53** | 46% |
| No draft, any prompt | 33 | |
| Code, **240k tokens of context** in the prompt† | **72** (fp16 KV: 65) | 71% |

Repeats reproduce to ±0.3 tok/s. †The 240k row is from the in-process `Generator`
harness (`ctxfill.py`), the only way to feed a 240k prompt; the same harness
reads 66.6 at 4k where `chat.py` reads 79, so compare it to its own fp16 column,
not to the rows above. The launchers run the full **262,144-token
context** per request with 8-bit KV; that is the model's trained window and the measured
ceiling — needle retrieval is exact at 240k and fails at 300k — and decode
loses nothing to depth at 8-bit KV (see [context](#context-the-ceiling-and-8-bit-kv)). The two levers that carry this over the stock
engine are stored precision and speculation: the pack's hyperconnection mixers
ship as **fp16 inside a 3-bit model** and are now stored int8 (+7 to +13%), and
the MTP draft runs at depth 5 with dynamic stopping on a 64K-column slice of the
head. The [native engine section](#the-native-engine-tuned-for-gb10) has what
each part is worth and the per-round profile; a
[dated history](#history-of-the-native-engine-numbers) is at the bottom.

#### Also tested: RTX PRO 6000 Blackwell (96 GB HBM3e)

> **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition — 96 GB HBM3e, discrete GPU, x86_64 host (EPYC).

Same engine ([vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051)),
different pack: [4.53 bpw mixed-K](https://huggingface.co/vcruz305/BLACKFROST-3.8-DERISKED-EXL3-4.53bpw_h8_ng8)
(head K8, per-expert mixed K3–K6). One stream, greedy, 400 new tokens, warm,
MTP ndt=5, int8 mixer, Q8 KV, per-expert mixed-K MoE kernel:

| Prompt class | Decode tok/s | Draft acceptance |
|---|---:|---:|
| Code (nginx log parser, /no_think) | **100** | 70% |
| Code (nginx log parser, thinking) | **86** | 53% |
| DevOps explainer + YAML | **76** | 40% |
| Prose (350-word story) | **66** | 31% |

The 4.53 bpw pack loads in ~73 GiB CUDA, leaving ~23 GiB free on the 96 GB card.
The per-expert mixed-K kernel + DDS fix + CPU sync skip (`329e051`) is what makes the mixed-K pack
competitive — the per-K-group dispatch (`785f206`) ran at ~60 tok/s, and per-expert Python dispatch at ~38 tok/s. With `/no_think` prompts (direct code output, no reasoning chain), the code prompt averages **100 tok/s** (median 101, peak 111, min 91) at 70% draft acceptance.

The vLLM route's numbers (50 to 53 tok/s at MTP k=3) are in
[`vllm-plugin/README.md`](vllm-plugin/README.md).

**Interactive benchmark:** [animated viewer](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/)
with seven scenes
([sweep](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=sweep) ·
[long prompt](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=long) ·
[cliff](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=cliff) ·
[scaling](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=scale) ·
[4.05 bpw](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=revision) ·
[native engine](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=native)),
PNG export, and downloadable data. It displays saved results and does not run
inference. Source and local-render notes are in [docs/README.md](docs/README.md).

## Which to use

**`exllamav3-tabby/serve.sh`** for an OpenAI-compatible API, which is what almost everyone
wants, including coding agents. **`exllamav3-tabby/chat.sh`** for a console session. Both
run the same fork runtime; against vLLM on this box that is 79 tok/s on code
against 52, a 47-second cold load against 9.5 minutes, and 45 to 60 GiB free against 17.

TabbyAPI brings `/v1/chat/completions` and `/v1/completions` with streaming,
the model's own chat template, Qwen XML tool calls (`tool_format: qwen3_5`),
reasoning splitting, and an admin API.

The **vLLM route** ([`vllm-plugin/`](vllm-plugin/README.md)) is for what needs
vLLM specifically: its structured output, tensor parallel across two Sparks,
tooling that assumes a vLLM endpoint, and packs exllamav3 cannot run. It has
its own setup, and it rewrites the pack in place, so keep a separate copy of the pack for it.

## Benchmarks: exllamav3 native

### Running the pack through exllamav3 directly

The same weights also run through exllamav3's own engine, with no vLLM in the
process. Measured here on 1.5.0 with the extension JIT-built for sm_121 on this
Spark, same prompt builder and salts as `qbench.py`, greedy, 128 new tokens,
decode excluding TTFT, median of four. Harnesses are in `exllamav3-tabby/bench/`; the `/v1` concurrency client is `bench/concurrency.py`.

**Single stream**

| Decode tok/s, 3.05 bpw | vLLM recipe (3k / 24k) | exllamav3 1.5.0 (3k / 24k) | gap |
|---|---:|---:|---:|
| No draft | 27.77 / 27.58 | **33.36 / 32.90** | +20% / +19% |
| MTP k=2 | 47.39 / 46.60 | **53.73 / 54.35** | +13% / +17% |
| MTP k=3 | 52.22 / 50.53 | **56.35 / 56.62** | +8% / +12% |

| Decode tok/s, 4.05 bpw | vLLM recipe (3k / 24k) | exllamav3 1.5.0 (3k / 24k) |
|---|---:|---:|
| No draft | not measured | 30.23 / 30.68 |
| MTP k=2 | not measured | 52.62 / 51.54 |
| MTP k=3 | 48.70 / 51.31 | 54.32 / 53.83 |

**Concurrency** (128 tokens per stream, unique prefix per stream, median of two
rounds at short prompts, aggregate tok/s. exllamav3's generator prefills every
queued job before it decodes, so its window aggregate is a steady-state number;
it is set against the steady aggregate from the vLLM
[Concurrency](vllm-plugin/README.md#concurrency) section)

| Streams | vLLM k=3 | exllamav3 k=3 | vLLM no draft | exllamav3 no draft |
|---:|---:|---:|---:|---:|
| 1, short prompts | 54.8 | **58.8** | 29.0 | **31.3** |
| 2 | **89.4** | 87.8 | 49.1 | **54.4** |
| 4 | **155.6** | 104.4 | 87.2 | **94.8** |
| 8 | **157.6** | 152.0 | **153.4** | 144.0 |
| 1, 3k prompts | 53.2 | **56.4** | 28.7 | **32.8** |
| 2 | 79.6 | **83.1** | 46.0 | **48.2** |
| 4 | **102.9** | 96.6 | 63.0 | **84.9** |
| 8 | 90.5 | **135.8** | 73.1 | **111.7** |

**Everything else**

| | vLLM recipe | exllamav3 1.5.0 |
|---|---:|---:|
| Cold prefill @ 24k, 3.05 / 4.05 | 1,142 / 1,137 tok/s | 1,129 / 1,112 tok/s |
| TTFT on a cold 3k prompt, 3.05 | about 2.8 s | about 5.0 s |
| Load time, 3.05 | 8 to 10 min (compile, graph capture) | 47 s |
| MemAvailable while running, 3.05 / 4.05 | 17.5 / 3.4 GiB | 62.7 / 47.8 GiB |
| Draft acceptance per position, k=3, 3.05 | 0.86 / 0.68 / 0.57 | 0.88 / 0.73 / 0.62 |

**What the numbers say.** At stock settings single-stream decode is 8 to 20%
faster on the same weights, and with the GB10 configuration in the next
subsection it is 79 tok/s on code against the recipe's 52, about 50%;
prefill is the same engine speed. Under concurrency the two are at
parity on short prompts (both reach 150 to 158 tok/s aggregate at eight
streams, vLLM ahead at four with MTP), and exllamav3 pulls ahead on 3k-token
prompts at eight streams (135.8 against 90.5), where vLLM pays more per batched
step for long contexts. Per-stream rates at eight streams are 17 to 20 tok/s
either way, so that is throughput for a queue, not eight interactive users.
Memory is the other large difference: exllamav3 keeps the 30 to 36 GiB
n-gram table on NVMe (`trellis_disk` mode) and reads it on demand, which costs
about 1.8 s on a cold short prompt and nothing at 24k, and leaves 45 to 60 GiB
free. The 4.05 revision is comfortable this way and marginal on vLLM. 4.05 is
still no faster than 3.05 (54.3 against 56.4 at k=3).

**What you give up.** The harness that produced these numbers is exllamav3's
Python generator driven in-process, and that is the honest scope of the
comparison:

- These are the in-process generator. The API is TabbyAPI
  ([Quick start](#quick-start)), which drives the same generator; TabbyAPI main already
  sizes both caches for MTP (`max_history` = draft depth, `max_batch_size` = concurrent
  jobs). A minimal beta shim, `exllamav3-tabby/beta/`, measured 79.5 wall tok/s on the
  greedy 400-token code job (74% acceptance), matching `chat.py`. A Nous Hermes tool loop at
  ~80k prompt survived it, while the vLLM overlay on this pack died on its second generate.
- No vLLM-style structured output. Tool calls are Qwen XML
  (`<function=…><parameter=…>`), parsed by TabbyAPI's `qwen3_5` tool format.
- No tensor parallel across two Sparks (exllamav3's TP path is one of the x86
  code paths the aarch64 patch stubs out).
- Prefix caching was not measured natively. exllamav3's generator has cache
  reuse, so vLLM's 114x cached-prefix result has no native counterpart yet.
- It cannot run everything the plugin can. DeepSeek V4.1 with Engram has no
  forward-correct graph in upstream exllamav3; for that pack the vLLM path is
  the only one.

**How to run it.** See [Quick start](#quick-start). If the pack was already
prepared for vLLM, `exllamav3-tabby/tools/make_native_view.sh` builds a symlink view of a
prepared pack with the native `config.json` and index, since `prepare_pack.sh`
rewrites those for vLLM; for a 4.05 pack that was prepared with the old
`rename_unsharded_ngram.py` step, pass `ngram_embedding.safetensors.native` as
the third argument so exllamav3 reads the original unsharded file. A pack
prepared with the current tools needs no third argument.
`exllamav3-tabby/bench/bench_native.py` and `bench_native_conc.py` are the two harnesses.

#### Can the plugin get there?

The obvious suspect was kernel age: the recipe pins exllamav3 1.4.7 and native
ran 1.5.0. Tested directly by running this exact vLLM build with the 1.5.0
extension underneath the plugin (1.5.0's `exl3_moe` grew five trailing
arguments for its deterministic-accumulation path; passing the values that
reproduce the 1.4.7 all-fused path lets the plugin call it, which is now
[vllm-exl3 PR #22](https://github.com/vcruz305/vllm-exl3/pull/22)):

| vLLM 0.29.0 + vllm-exl3 0.4.2, 3.05 bpw | exllamav3 1.4.7 | exllamav3 1.5.0 |
|---|---:|---:|
| Decode @ 3k / 24k, MTP k=3 | 52.22 / 50.53 | 52.05 / 49.57 |
| Decode @ 3k / 24k, no draft | 27.77 / 27.58 | 28.54 / 28.23 |
| Cold prefill @ 24k | 1,142 | 1,138 |
| Acceptance per position, k=3 | 0.86 / 0.68 / 0.57 | 0.87 / 0.69 / 0.54 |

**Kernel version is not the lever.** k=3 is unchanged and no-draft moves about
3%, at the edge of the day-to-day spread, which is what the profiler already
said: decode time is trellis dequantization in kernels both engines share. What
separates native from the recipe is above the kernels:

- **Per-step engine cost, hidden by MTP.** Without a draft a vLLM step is 36 ms
  against 30 ms native. At k=3 the step is 58 ms against 57 ms, so speculation
  already amortizes nearly all of it. The remaining no-draft gap is the vLLM
  scheduler, sampler, and output path, and none of it lives in the plugin.
- **Draft acceptance at positions 2 and 3.** vLLM accepts 0.69 and 0.54 where
  native accepts 0.73 and 0.62, which is 3.10 against 3.22 tokens per
  verification round, about 4%, and is most of the k=3 gap. Both drafts are the
  same MTP head on the same weights, so the difference is in how the proposer
  chains its k steps and feeds hidden state back, which in vLLM is the in-tree
  Qwen4Exp MTP proposer rather than the plugin. That is the one place a code
  change could recover measurable speed, and it needs a read of both proposers
  side by side before anyone touches it.
- **Concurrency.** Not a gap after all: the two-stream plateau reported
  earlier was the window metric, and steady-state vLLM matches native at four
  and eight streams on short prompts (see Concurrency). The 1.5.0 kernels
  under vLLM change nothing here either (76.9 / 67.3 / 49.5 window aggregate at
  2 / 4 / 8 streams on 3k prompts, the same as 1.4.7).
- **The n-gram table.** A disk-backed table option in the plugin would not
  change decode speed, but it is what would let the 4.05 revision boot at 262k
  on one Spark.

Everything else that could plausibly matter was measured empty earlier in this
README (`max-num-seqs`, CUDA graph mode, batched-token size, fp8 KV).


#### The native engine, tuned for GB10

The numbers above ran exllamav3 1.5.0 with its stock defaults. This is the same
engine at [vcruz305/exllamav3 `329e051`](https://github.com/vcruz305/exllamav3/commit/329e051)
(upstream master + the aarch64 guards, [#1](https://github.com/vcruz305/exllamav3/pull/1),
+ the GB10 decode changes, [#2](https://github.com/vcruz305/exllamav3/pull/2),
[#3](https://github.com/vcruz305/exllamav3/pull/3),
[#4](https://github.com/vcruz305/exllamav3/pull/4)),
configured for this box. `exllamav3-tabby/chat.sh` is that configuration, and
`serve.sh` passes the same settings to TabbyAPI (environment + `tabby-config.yml`):

```sh
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_GR_INT8=1 EXL3_MTP_HEAD_N=65536 EXL3_NGRAM_STREAM=0
taskset -c 5-9,15-19 python examples/chat.py -m $MODEL -mode qwen35 -mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs 262144
```

**Decode, single stream, greedy, 400 new tokens, cold load** (`bench.sh`:
page cache dropped before every load, `chat.py -tps -topk 1`, so this includes
console streaming, which is the number a user sees):

| Prompt | tok/s | draft acceptance | stock 1.5.0, k=3 (above) |
|---|---:|---:|---:|
| code (nginx log parser) | **79** | 73% | 56.4 |
| DevOps explainer + YAML | **62** | 59% | |
| prose (350-word story) | **53** | 46% | ~41 |
| no draft, any prompt | 33 | | 32.9 |

Greedy runs reproduce to ±0.3 tok/s. The default temperature-0.8 sampler moves
MTP acceptance 58–75% run to run and the code number with it; measure with
`-topk 1`. In-process harnesses (`sweep.py`, `split_time.py`) read above
`chat.py` — about 4 tok/s for kernel changes and about 10 for host-side ones,
because the per-token console write is itself a host sync — so only the
`chat.py` figure is quoted here.

**If you are serving rather than watching a console, the same configuration is
faster.** `chat.py -tps` writes every token to the terminal, and that write is
a host sync; an API server or agent loop does not do it. The identical stack
driven through `Generator`/`Job` with no per-token console write
(`silent_ceil.py`, greedy, 400 new tokens, cold load, `-cq 8,8`):

| Prompt | `chat.py` (above) | no console streaming |
|---|---:|---:|
| code | 79 | **85** |
| DevOps | 62 | **70** |
| prose | 53 | **55** |

Both columns are real and they answer different questions. The `chat.py` column
is what an interactive user sees, and it is what the rest of this README
quotes. The right-hand column is what a server on this box should reach, and it
is the measured ceiling for this pack on a single stream — the levers that were
tried to pass it are listed under [what is closed](#levers-that-are-closed-native-engine)
below.

**What the configuration does, and what each part is worth** (code prompt,
`chat.py`, each measured on top of the rest):

| | tok/s | Why |
|---|---:|---|
| `EXL3_INT8_GEMV=0` | +3 | the fused int8-activation GEMV for mul1 tensors is slower on GB10 than the fp16 kernel it replaces |
| `EXL3_MOE_COOP_WIDE=1` | +5 | the fused decode MoE kernel only picks its wide 128-column, 4-way k-split tile on datacenter Blackwell by default; GB10's 48 SMs want it too |
| `taskset -c 5-9,15-19` | +2 | GB10 pairs ten Cortex-X925 with ten A725; the launch thread lands on a little core often enough to show |
| `-ndt 5` (from 3) | +8 | the verify forward costs 37 ms at q=2, 52 at q=5, 59 at q=7, 86 at q=9; on code, acceptance stays high enough that 5 is the peak |
| `-dds -dc 0.6` | ±0 code, **+8 prose** | dynamic draft length; see below |
| `EXL3_GR_INT8=1` | **+5 code, +7 DevOps, +4 prose** | the hyperconnection mixer weights stored int8 instead of fp16; see below |
| `EXL3_MTP_HEAD_N=65536` | ±2 | draft argmax over a 64K-column slice of `lm_head` (105 MB) instead of the full 248K head (397 MB); 97% of drafts land in-slice, misses become rejections, verify still uses the full head. Worth 5 ms/round in-process; inside variance through `chat.py` |
| `-cq 8,8` | **+3 at 4k, +7 at 240k** | 8-bit KV. The 12 full-attention layers stream the whole KV every decode step, 5.9 GB per step at 240k in fp16; halving it is faster at every depth with acceptance unchanged, and 12 GB less KV at the full context. See below |

Env knobs that measured empty (±2): `EXL3_MOE_COOP_KSPLIT`, `EXL3_GEMV=2`,
`EXL3_INT8_GEMV=1`, `EXL3_INT8_GEMV_MAX_K`, `EXL3_GEMV_SMEM`, `EXL3_GR_RB=1`
(below), `-ngr` (the n-gram table in RAM is slower with MTP and costs 30 GB).

**The mixers are fp16 in a 3-bit model.** Walking every module's tensors and
bucketing bytes by dtype (`mixaudit.py`):

| Weights | Resident |
|---|---:|
| `LinearEXL3` int16 trellis (the 3-bit experts and 5-bit dense) | 47.5 GB |
| **`GatedResidual` fp16** (96 hyperconnection mixer sites + MTP) | **3.2 GB** |
| `LinearFP16` | 0.2 GB |

The quantizer had no rule for the mixers, so they stayed fp16. The fused decode
path reads 13.2 MB per site (`fn_h` 6.6 + `upx_h` 6.6), 96 sites per round =
1.27 GB, which is 4.6 ms at the 273 GB/s spec and measured 10.5 ms (the
`gr_dots` grid is `(M+1, R)` and every block re-reads its weight row per
stream; the 24 MB L2 hides that in a one-site microbench and not in the real
round). `EXL3_GR_INT8=1` quantizes `fn_h`/`upx_h` to int8 with per-row fp32
scales at load and frees the fp16 copies (1.6 GB back). The int8 kernels are
the fp16 kernels with only the weight load changed — same fmaf chains, same
reduction order, scale factored out of the k-loop — so quantization is the only
numerical difference. Before writing them, storage precision was simulated by
quantize→dequantize on the loaded weights (`mixq.py`): int8 and int6 kept
greedy acceptance at the fp16 level, int4 collapsed it to 20%. Kernel parity
against fp32 torch on the dequantized weights is 1–2e-4 (`gr_int8_parity.py`);
greedy trajectories diverge from fp16 at token 157 / 34 / 14 on code / DevOps /
prose (`diverge.py`) with acceptance unchanged (±0.3 pt).

**Why prose is slower than code, and why `-dds` is in the launcher.**
Per-position MTP acceptance, P(draft position *i* accepted | reached), greedy
(`accept.py`):

| | pos 0 | pos 1 | pos 2 | pos 3 | pos 4 | rounds accepting all 5 |
|---|---:|---:|---:|---:|---:|---:|
| code | 0.91 | 0.85 | 0.76 | 0.65 | 0.54 | 46 / 85 |
| prose (short story) | 0.68 | 0.38 | 0.20 | 0.09 | 0.02 | 4 / 168 |
| essay (argumentative) | 0.65 | 0.38 | 0.17 | 0.08 | 0.03 | 6 / 172 |

It is all natural-language text, not "creative writing": the essay collapses
identically. Position 0 is fed the true hidden state, so the 0.68-vs-0.91 gap
there is the entropy of English, not the 3-bit MTP head; positions 1+ compound
on a possibly-wrong guess. A fixed `-ndt 5` therefore has prose drafting 840
tokens to accept 240, paying the q=6 verify for each round: 41.7 tok/s, when
`-dds -dc 0.6` stops drafting once the running confidence product falls below
the target and gets 53. Prose's ceiling with this drafter is the drafter;
dequantizing the MTP head (1.0 GB at 3 bits, ~5.5 GB in BF16, +10–15 ms/round)
was ruled out by the position-0 numbers.

##### Context: the ceiling, and 8-bit KV

KV on this geometry is cheap: 12 of
the 48 layers are full attention with 2 KV heads at head_dim 256, so a token
costs 24 KB fp16 (12 KB at `-cq 8,8`), 6 GiB for the whole 262,144 window.
Every cache size tried loads and decodes — `-cs 524288` fp16 and `-cs 1048576`
at 8-bit both run at 71 tok/s on a short prompt — so memory is not what caps
context. The trained window is. Needle test (`ctxfill.py`: a random code planted
at a random depth in a prompt built to a token target, `Generator`-driven since
`chat.py -prompt` dies at 128 KiB of argv):

| Prompt tokens | fp16 KV | 8-bit KV | peak host memory |
|---:|---|---|---:|
| 32k / 128k / 240k | hit / hit / hit | hit | 102 / 106 / 110 GiB |
| 300k | miss | miss | 112 GiB |
| 480k, two needle positions | miss / miss | **hit** / miss | 118 GiB |

Every run inside 262,144 retrieves the code exactly; every run past it fails
the same way (the model emits `<|im_end|>` and starts a new turn about "the user
is asking me to reveal a secret") at both KV precisions, so it is positional,
not precision. The one 480k hit did not survive a second needle position. Host
memory would have been the next wall anyway — 118 of 121 GiB at 480k fp16, the
recurrent GDN history plus cache — so the two ceilings coincide. **The launcher
runs `-cs 262144`.**

Decode at depth, 400 new tokens of a code task placed after the documents
(in-process, same harness both columns, so the pair is clean; it sits under the
`chat.py` headline because the code prompt follows 240k tokens of unrelated
text):

| Context in prompt | fp16 KV | **8-bit KV** | prefill |
|---:|---:|---:|---:|
| 4k | 66.6 | **69.8** | ~900 tok/s |
| 128k | 66.6 | **69.5** | 1,150 |
| 240k | 65.1 | **72.0** | 1,140 |

fp16 loses 2% to depth at 240k; 8-bit loses nothing, and the margin grows with
context as the byte math says it should. Acceptance is unchanged (69% vs 71% at
240k). Prefill is flat at ~1,150 tok/s out to 480k (7 minutes for 480k). A
32-token needle answer at 240k read 104 tok/s at 8-bit; that number is not
quotable — the first speculative chunk lands in TTFT and a repetitive
`<|im_end|>` tail drafts perfectly — which is why the depth table is 400 tokens
of real generation.

**Where a decode round goes** (q=6, ~44 ms GPU, `kern_rounds.py`, int8 mixer
on). Decode on this pack is weight streaming: top-10 of 512 experts at 3 bits
is ~3.7 GB per round, plus ~0.6 GB of int8 mixer, ~1.3 GB of GDN projections
and ~0.5 GB of `lm_head`, about 6.5 GB, which is 24 ms at the 273 GB/s spec.

| Slice | ms/round | byte floor | note |
|---|---:|---:|---|
| fused MoE experts (`exl3_moe_coop_a/b`) | 16.5 | 13.6 | 82% of spec, ~at the ceiling |
| hyperconnection mixer (`gr_dots/gr_finalize_i8`) | ~8.5 | 2.3 | was 10.5 at fp16; the remaining gap is the per-stream re-read |
| GDN qkv/z projections (`exl3_mgemm`) | 7.9 | ~5 | |
| 110 small linears (`exl3_gemm` 32×128 tile) | 4.4 | 1.0 | 40 µs each, launch-latency bound |
| recurrent gated delta rule | 3.1 | — | |
| `lm_head` (5 draft slices + 1 verify) | 2.9 | 3.0 | at bandwidth |

An earlier version of this section concluded that "decode here is launch count
and small-kernel latency, not weight bandwidth", from the whole 48 GB pack
against a 215 GB/s copy figure. That was wrong: only the routed experts are
read, the round is ~6.5 GB, and it runs at roughly 60% of spec bandwidth.

The 110 small linears are **not** launch-bound either, which corrects a second
claim this section used to make ("only CUDA-graph capture of the decode round
would move it"). exllamav3's extension already captures the GDN and attention
linear chains into per-`(bsz, seqlen)` CUDA graphs (the ext `Graph` class,
`BC_LinearEXL3`, `exl3_gemm_gr`), so those 40 µs entries are GPU execution
time, not launch overhead — the profiler lists them because it times the
kernels *inside* the graph. Wrapping `model.forward` in a further
`torch.cuda.CUDAGraph` aborts at capture (`operation not permitted when stream
is capturing`, nested capture). There is no launch-overhead lever left in this
row.

##### Levers that are closed (native engine)

Everything in this list was measured on the `785f206` stack, at the launcher's
configuration, and none of it is shipped. It is here so nobody spends the same
days twice.

- **Deeper speculation (`-ndt` 6, 7, 8).** An older note in this repo said
  `-ndt >= 6` exhausted memory; 8-bit KV removed that, and all of them now run.
  They still do not pay, because `-dds` already truncates the draft window by
  confidence (silent harness, code prompt):

  | `-ndt` | tok/s | tokens/round | acceptance |
  |---:|---:|---:|---:|
  | **5** | **85.7** | 4.12 | 75.2% |
  | 6 | 82.5 | 3.81 | 70.7% |
  | 7 | 86.0 | 4.35 | 67.4% |
  | 8 | 77.9 | 4.55 | 66.4% |

  Tokens per round barely move while acceptance rots; `-ndt 7` is inside
  variance of `-ndt 5`. Depth is not a lever on this drafter.

- **Restructuring the mixer to stop re-reading streams.** The `gr_dots` grid is
  `(M+1, R)`, so each of 325 blocks re-reads a whole 40 KB stream: at R=6 that
  is ~80 MB of traffic against 3.3 MB of weights. Sweeping R and fitting
  `intercept + slope × R` per call localises it — the slope is L2 bandwidth
  (13 MB / 6.5 µs ≈ 1.9 TB/s). Two rewrites removed that traffic and both are
  numerically exact against an fp32 reference on the dequantized weights, and
  both are **slower** at this R:

  | variant | blocks | fit (µs) | R=6 |
  |---|---|---|---:|
  | shipped, one row per block | 325 × R | `8.2 + 6.17R` | **44.8 µs** |
  | contraction-tiled (k-chunks + partials) | 240 | `19.2 + 4.87R` | 46.8 µs |
  | row-blocked, 8 rows per block | 41 × R | `25.0 + 4.33R` | 47.1 µs |

  The slope falls exactly as the traffic model predicts, but the intercept
  tracks block count, and on a ~45 µs op a 48-SM part punishes starved
  parallelism harder than it rewards saved bandwidth. Both cross over only near
  **R ≈ 9**, and R is `ndt + 1 = 6`. Cutting the per-block stream re-read needs
  coarser blocks; filling the SMs needs ≥ ~300 blocks; at R=6 you cannot have
  both. Fit both terms and solve for the crossover before writing the kernel.

- **CUDA-graph capture of the decode round.** Already done inside the engine —
  see the paragraph above the negative-results list. Not a lever.

- **N-gram assist alongside MTP.** Mutually exclusive: `Generator` asserts
  `not ngram_match_min` when a draft model is set, so n-gram drafting replaces
  the MTP head rather than assisting it. On novel code, a 75%-acceptance MTP
  head is the better drafter.

- **A dequant-once MoE kernel — premise withdrawn, size unmeasured.** The fused
  expert kernel costs ~linearly in verify rows (0.127 ms at m=1 to 0.704 ms at
  m=8 for one layer), which looks like per-row re-dequantization of the same
  trellis and therefore like an easy 11 ms/round. It is not that: unique
  experts touched scale nearly 1:1 with rows (10, 20, 39, 56, 70 at
  m = 1, 2, 4, 6, 8), so the linearity is largely genuine distinct-weight
  traffic — with topk=10 of 512, each candidate token pulls its own expert set.
  That is also the real reason MTP verify is expensive on a fine-grained MoE,
  and why speculation tops out near 2× here however good acceptance gets.
  Caveat on that measurement: it used random hidden states, which route
  near-uniformly, and it read 28.3 ms ×48 layers at m=6 against ~16.5 ms in
  situ — so real routing overlaps more than random and the honest statement is
  that the *premise* for the rewrite was wrong, not that the remaining overlap
  is exactly zero. Sizing it properly needs an in-situ unique-expert census on
  real hidden states; that has not been published here. Do not size an
  expert-path bandwidth claim from a random-input microbench.

Tried and kept as negative results:

- *Row-batched fp16 GatedResidual kernels* (`EXL3_GR_RB=1`, default off):
  stream each site's weights once per call instead of once per row. Parity
  with the fp32 reference identical (max rel 4.3e-4), per-forward 49.7 → 48.7
  ms, and **−3 tok/s end to end**: the changed fp32 reduction order shifts the
  greedy trajectory and acceptance moved 4.17 → 3.92 tokens per round. The int8
  kernels above keep the original reduction order for exactly this reason.
- *Host-sync removal in the speculative loop* (batched verify, device-resident
  draft chain, pinned MTP block table; all default on in the fork). +12 tok/s in
  the in-process generator, inside ±2 through `chat.py`, whose per-token console
  write is itself a sync. Kept for server callers, not counted above.
- *A `kern_rounds.py` row that is not decode:* `exl3_moe_kernel<3,128>` shows
  1.2 calls/round at 1.9 ms; all 119 of those calls are the single 66-row
  prefill forward amortized over the rounds (`moepath.py`). Not a decode
  inefficiency.

`exllamav3-tabby/tuning/` holds `bench.sh`, the A/B matrix
scripts (`pubbench.sh`, `i8bench.sh`) with their logs under `logs/`, the
harnesses, and the three fork commits as `patches/`; its README lists what each
does.

## History of the native-engine numbers

Single stream, code prompt unless noted, `chat.py` cold greedy 400 tokens.
Each row is the state after that day's work; the current state is at the top
of this file.

| Date | Code | DevOps | Prose | What changed |
|---|---:|---:|---:|---|
| **2026-09-17** | **79** | **62** | **53** | 8-bit KV in the launcher (`-cq 8,8`: +3 at 4k, +7 at 240k, 12 GB less KV) at the full 262,144 context; ceiling measured — needle exact at 240k, fails at 300k/480k, memory wall at ~480k |
| 2026-09-17 | 79 | 62 | 53 | int8 GatedResidual mixer kernels (the mixers ship fp16 in a 3-bit pack; −1.6 GB, half the mixer bytes per round); pruned draft `lm_head`; bandwidth roofline corrected (round is ~6.5 GB, ~60% of spec, not launch-bound) |
| 2026-09-16 | 71–73 | 56 | 46.5 | `-dds -dc 0.6` dynamic draft length (+9.5 prose); per-position acceptance analysis; row-batched fp16 mixer kernels tried and rejected (−3, reduction order) |
| 2026-09-16 | 73 | | 37 | `EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1`, big-core affinity, `-ndt 5` |
| 2026-09-16 | 63 | | 43 | `-mtp -ndt 3` on exllamav3 master with the aarch64 guards |
| 2026-09-15 | 56.4 | | ~41 | stock exllamav3 1.5.0, `-mtp` k=3 (in-process harness, 128 tokens) |
| 2026-09-13 | 52 | | | vLLM 0.29.0 + vllm-exl3 0.4.2, MTP k=3 — the vLLM path's number, kept for scale |

## Hardware and model

NVIDIA DGX Spark: GB10 (sm_121), 128 GB unified memory (121.7 GiB visible), aarch64,
10 Cortex-X925 + 10 Cortex-A725 cores (the launchers pin to the X925s).

Qwen3.8-Flash-Next is a `Qwen4ExpForConditionalGeneration` model: 48 layers
(36 linear attention, 12 full attention, `full_attention_interval: 4`), 512
experts with top-10 routing, hidden size 2560, 2 KV heads at head_dim 256, one
MTP layer, a per-layer n-gram (PLE) embedding table (320,001,536 rows × 160,
K=5 packed), and a vision tower. `max_position_embeddings` is 262,144.

## Known limitations

- **262,144 tokens is the ceiling per request.** Needle retrieval fails at 300k and 480k at
  both KV precisions; see [context](#context-the-ceiling-and-8-bit-kv).
- **Throughput through TabbyAPI is not yet re-measured** on the current pin, in either
  profile. The per-stream numbers above are `chat.py`, and the concurrency table is
  stock 1.5.0 at k=3.
- **The concurrent profile streams the n-gram table from NVMe.** The measured single-stream
  numbers had it in RAM. The stock 1.5.0 runs above suggest streaming costs about 1.8 s on a
  cold short prompt and nothing at 24k, but that is not measured on the fork.
- No tensor parallel across two Sparks (exllamav3's TP path is one of the x86 code paths
  the aarch64 guards stub out). Use the vLLM route for TP.
- TabbyAPI publishes no aarch64 GPU extras. `setup.sh` installs TabbyAPI's base package
  into the fork's venv, and the recipe's runtime check refuses a stock exllamav3.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `... is not the vcruz305 fork runtime` / `lacks fork kernels` | a stock exllamav3 is being imported; re-run `bash exllamav3-tabby/setup.sh` (it removes stock wheels and rebuilds) |
| TabbyAPI: `exllamav3 ... required version 1.5.1` | the runtime predates the pin; re-run `setup.sh` |
| `no pack at ...` | set `MODEL_DIR=/path/to/pack` |
| `... was prepared for vLLM` | the pack went through `vllm-plugin/prepare_pack.sh`; build the native view the message prints |
| load refuses for memory while `MemAvailable` is high | GB10 page cache (`cudaMemGetInfo` reports MemFree); `serve.sh` drops the pack's pages first. Stop other servers, or run `exllamav3-tabby/drop-model-cache.sh` |
| `memory estimate ... needed ... available` warning | lower `CACHE_SIZE`, set `NGRAM_RAM=false`, or use `PROFILE=single` |
| replies cut at a few thousand tokens | the client sets `max_tokens`; TabbyAPI honours it |
| clients on another host get 401 | `HOST=0.0.0.0` enables auth; use the key from `~/qwen38-exl3/tabbyAPI/api_tokens.yml` |
| build fails | `~/qwen38-exl3/state/exllamav3-build.log`; needs CUDA 13.x `nvcc` (`CUDA_HOME`) |

The vLLM route's own failure table is in [`vllm-plugin/README.md`](vllm-plugin/README.md#troubleshooting).

## Related repositories

| Repo | Role |
|---|---|
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | the pack this recipe serves |
| [vllm-exl3](https://github.com/vcruz305/vllm-exl3) | the EXL3 plugin: source, releases, issues, and the pack-prep / vLLM-patch tools this recipe calls |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | my exllamav3 fork: master = upstream + aarch64 guards ([#1](https://github.com/vcruz305/exllamav3/pull/1)) + GB10 decode: int8 mixer, pruned draft head, MTP host-sync removal ([#2](https://github.com/vcruz305/exllamav3/pull/2), [#3](https://github.com/vcruz305/exllamav3/pull/3)) + **`exl3_moe_mixedk` per-expert K kernel** for mixed-K fused dispatch (`329e051`). Required for the 100 tok/s RTX 6000 numbers and for vllm-exl3 mixed-K fused mode. |
| [GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) | sibling recipe this one is modeled on |
| [DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe) | sibling recipe this one is modeled on |

## Credits and upstream work

**ExLlamaV3 by Turboderp ([@turboderp](https://github.com/turboderp-org/exllamav3)).**
The EXL3 trellis format, the MCG codebook, the quantization method, and the
`Qwen3.8-Flash-Next-exl3` pack itself are theirs. MIT, Copyright (c) 2025
Turboderp.

**[TabbyAPI](https://github.com/theroyallab/tabbyAPI)** by theroyallab is the server this
recipe runs on. **[vLLM](https://github.com/vllm-project/vllm)** is the engine behind the
secondary route.

**GLM-5.3-Flash-EXL3-2x-DGX-Sparks by Mia's AI Lab
([@MiaAI-Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)),
with [@plotarmordev](https://github.com/plotarmordev)).** No code from that
project is copied into this recipe. They are credited because this recipe
serves EXL3 packs through `vllm-exl3`, whose plugin derives part of its
`exl3.py` from their `overlay/exl3.py`; see `vllm-exl3`'s own
`THIRD_PARTY_NOTICES.md`. MIT, Copyright (c) 2026 Mia's AI Lab.

## License

MIT for the scripts and notes in this repo (see `LICENSE`). Weights are
**not** redistributed here; pull them from Hugging Face and respect turboderp's
pack license. TabbyAPI, vLLM, ExLlamaV3/EXL3, and `vllm-exl3` have their own licenses.

