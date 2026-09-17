# Qwen3.8-Flash-Next EXL3 on one NVIDIA DGX Spark

Serves [turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(revision `3.05bpw_h5_ng5`, about 80 GB) on a single NVIDIA DGX Spark (GB10,
128 GB unified memory, aarch64) through vLLM and the
[vllm-exl3](https://github.com/vcruz305/vllm-exl3) plugin.

Measured on `cruz-spark`, 2026-09-13, vLLM 0.29.0 with vllm-exl3 0.4.2, one
request at a time, full 262,144-token context configured:

| | |
|---|---|
| Decode, MTP k=3 | **50 to 53 tok/s** from 3k to 163k tokens of context |
| Decode, no draft | 27 to 28 tok/s across the same range |
| Fastest measured point | **53.1 tok/s at 163,506 tokens** |
| Cold prefill | about 1,100 tok/s, flat from 3k to 252k |
| Cached-prefix TTFT, 196k prompt | **1.56 s** (cold: 178.7 s) |
| Aggregate throughput, 4 streams, MTP k=3 | **156 tok/s** steady on short prompts, 103 on 3k-token prompts |
| exllamav3 directly, same pack, k=3 | 58.8 tok/s one stream, 152.0 aggregate at 8 streams (vLLM: 54.8 and 157.6) |
| exllamav3 directly, GB10-tuned (2026-09-16), k=5 | **73.0 tok/s** greedy on code (77 in-process), 37 to 41 on prose; see [tuning](#tuning-the-native-engine-on-gb10-2026-09-16-56--73-toks) |
| KV pool at the default config | 416,163 tokens, 1.6x concurrency at max context |
| 4.05 bpw at 262k, n-gram table on NVMe | boots at util 0.80, 954k-token KV pool, 41 to 48 tok/s at k=3 |

Two rules that fall out of the measurements: use MTP k=3, and turn it off past
163,840 tokens of prompt, where draft acceptance goes to exactly zero and
speculation becomes a 21% loss. The serve script defaults to the first and
warns about the second.

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

## Contents

- [Quick start](#quick-start)
- [Results, current build (2026-09-13)](#results-current-build-2026-09-13)
  - [Draft depth](#draft-depth)
  - [Context and the MTP acceptance cliff](#context-and-the-mtp-acceptance-cliff)
  - [Prefill and prefix caching](#prefill-and-prefix-caching)
  - [Sampling temperature](#sampling-temperature)
  - [Concurrency](#concurrency)
  - [Levers that measured empty](#levers-that-measured-empty)
  - [The 4.05 bpw revision](#the-405-bpw-revision)
  - [Running the pack through exllamav3 directly](#running-the-pack-through-exllamav3-directly)
    - [Tuning the native engine on GB10: 56 → 73 tok/s](#tuning-the-native-engine-on-gb10-2026-09-16-56--73-toks)
- [How things were measured](#how-things-were-measured)
- [Hardware, model, memory](#hardware-model-memory)
- [Historical results (2026-09-07 and 09-08, earlier build)](#historical-results-2026-09-07-and-09-08-earlier-build)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Related repositories, credits, license](#related-repositories)

## Quick start

The pack as published needs three one-time rewrites before vLLM's native loader
will serve it, and vLLM needs three runtime patches until this architecture is
upstreamed. Everything below is scripted and idempotent.

### Prerequisites

- One DGX Spark (GB10, aarch64), NVMe with room for the ~80 GB pack.
- **vLLM 0.29.0.** It carries the `Qwen4ExpForConditionalGeneration` model
  classes in-tree and installs from PyPI as a prebuilt aarch64 wheel, keeping
  `torch` 2.13.0+cu130:
  ```bash
  pip install vllm==0.29.0
  ```
  Confirm with `python -c "import vllm; print(vllm.__version__)"`. The current
  results in this README are on 0.29.0. The historical results were on a 0.28.1
  nightly (`0.28.1rc1.dev324+ga56654d6d`); if you need a nightly, the
  [GLM-5.3-Flash recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
  and the [DeepSeek-V4-Flash-Vision recipe](https://github.com/vcruz305/DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe)
  show how to build and verify one on this box.
- **`exllamav3` 1.4.7, built from source**, with the aarch64 patch shipped in
  the plugin repo (`tools/patch_exllamav3_aarch64.py`). The plugin imports the
  compiled `exllamav3_ext` module, so the pure-Python wheel is not enough.
- **`vllm-exl3` from `main`**, at `94c29ba` (2026-09-14) or newer. That is where
  unsharded n-gram tables and the disk-backed table mode landed (PRs #22 to
  #24); an older checkout still serves the 3.05 pack but needs the rename step
  below for 4.05:
  ```bash
  pip install git+https://github.com/vcruz305/vllm-exl3@main
  ```
  Anything older than commit 6b26e5c (2026-09-08) carries the fat-expert
  prefill bug that silently corrupts long prompts (fixed in PR #5) and the
  engine wedge on 33 to 144-token prefills (fixed in PR #7). See Known
  limitations.
- `torch` / CUDA 13.0 for aarch64.

### 1. Install the runtime

Install vLLM, `exllamav3` (from source, with the aarch64 patch), and the
`vllm-exl3` plugin per Prerequisites, then confirm:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import exllamav3_ext; print('exllamav3_ext OK')"
```

### 2. Download the pack

```bash
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 \
  --local-dir ~/models/Qwen3.8-Flash-Next-EXL3
```

About 80 GB: seven `model-0000N-of-00007.safetensors` shards,
`ngram_embedding.safetensors` (32.6 GB), `mtp_hyper_connection_mixer_patch.safetensors`,
`config.json`, `model.safetensors.index.json`, and the tokenizer files. The
serve script also accepts the pack under its revision name,
`~/models/Qwen3.8-Flash-Next-exl3-3.05bpw`.

### 3. Prepare the pack

Three one-time, idempotent rewrites (each keeps a backup) make the pack
loadable through vllm-exl3's native Qwen4Exp path:

```bash
PACK_DIR=~/models/Qwen3.8-Flash-Next-EXL3 \
  PLUGIN_REPO=/path/to/vllm-exl3 \
  bash scripts/prepare_pack.sh
```

This runs, in order:

1. `tools/exl3_pack_tools/qwen_pack_scan.py <pack>` writes `pack_scan.json`.
2. `tools/exl3_pack_tools/qwen_pack_config.py <pack>` rewrites `config.json`'s
   `quantization_config` for the plugin (backs up `config.json.native`).
3. `tools/exl3_pack_tools/regenerate_safetensors_index.py <pack>` adds
   `ngram_embedding.safetensors` and the MTP patch file to
   `model.safetensors.index.json`, which vLLM uses as the load list (backs up
   `.native`).

Packs from exllamav3 1.5.0 onward ship the n-gram table as one unsharded
tensor (the `4.05bpw_h6_ng6` revision does). The scan tool on `main` reads
that layout and the config it emits carries `ngram_embedding.sharded: false`;
`prepare_pack.sh` refuses an older plugin checkout rather than prepare a pack
that would OOM at boot. `scripts/rename_unsharded_ngram.py` stays in the repo
as the workaround for an old plugin build only.

Optional GPU verification gates ship in `tools/verify_native_pack/`
(`test_ngram_embedding.py`, `mul1_check.py`, `pad_check.py`,
`compile_check.py`); run them with `VERIFY=1 bash scripts/prepare_pack.sh`.

### 4. Patch vLLM

Three exact-anchor patches from the plugin repo add the quant-config plumbing
vLLM does not yet have for `Qwen4ExpForConditionalGeneration`. Each is
idempotent and keeps a `.orig` backup:

```bash
VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
python /path/to/vllm-exl3/tools/patch_vllm_qwen4_exp/patch_vllm_qwen4_ple.py "$VLLM_DIR"
python /path/to/vllm-exl3/tools/patch_vllm_qwen4_exp/patch_vllm_vision_split.py "$VLLM_DIR"
python /path/to/vllm-exl3/tools/patch_vllm_qwen4_exp/patch_vllm_mtp_lmhead.py "$VLLM_DIR"
```

| Script | What it fixes |
|---|---|
| `patch_vllm_qwen4_ple.py` | adds `quant_config` to `ParallelLMHead` and to the per-layer n-gram (PLE) embedding table |
| `patch_vllm_vision_split.py` | drops the pack's split vision attention q/k/v names; vision qkv is served from the pack's bf16 fused copy |
| `patch_vllm_mtp_lmhead.py` | adds `quant_config` to the MTP draft's `ParallelLMHead` |

`scripts/preflight.py` checks all of this in a couple of seconds and is worth
running before a serve attempt, since each model load takes about ten minutes.

### 5. Serve

Defaults are the measured-best configuration: MTP k=3, bf16 recurrent state,
and the full 262,144 context. `MODEL_DIR` is auto-detected from either
`~/models/Qwen3.8-Flash-Next-exl3-3.05bpw` or `~/models/Qwen3.8-Flash-Next-EXL3`.

```bash
bash scripts/serve_one_spark_qwen.sh
```

For workloads that routinely exceed 163,840 prompt tokens, turn the draft off.
Past the acceptance cliff it is a net loss of roughly 21%, see
[Context and the MTP acceptance cliff](#context-and-the-mtp-acceptance-cliff):

```bash
SPEC_CONFIG=none bash scripts/serve_one_spark_qwen.sh
```

Other depths, noting that vLLM's `num_speculative_tokens` counts *drafted*
tokens, so k=3 drafts three and verifies up to four per step:

```bash
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}' bash scripts/serve_one_spark_qwen.sh
```

To serve the 4.05 bpw revision at the full context, keep the n-gram table on
NVMe instead of the device. This is what makes 4.05 fit on one Spark (954k-token
KV pool at 262k, 18 GiB free); it costs 5 to 7% decode because the host-side
lookup has to stay outside CUDA graphs (PIECEWISE-only), see
[The 4.05 bpw revision](#the-405-bpw-revision):

```bash
MODEL_DIR=~/models/Qwen3.8-Flash-Next-exl3-4.05bpw NGRAM_TABLE=disk bash scripts/serve_one_spark_qwen.sh
```

Defaults are `MAX_MODEL_LEN=262144`, `GPU_MEM_UTIL=0.80`, `MAX_NUM_SEQS=4`,
`MAMBA_SSM_DTYPE=bfloat16`, port 8899, `--enable-prefix-caching`,
`--reasoning-parser qwen3`. Load takes about 9.5 minutes from cold NVMe and
roughly 2.5 minutes when the pack is still in page cache. See the script for
every override (`PORT`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `MAX_NUM_SEQS`,
`SPEC_CONFIG`, `MAMBA_SSM_DTYPE`, `SERVED_NAME`, `PROFILER_DIR`,
`NGRAM_TABLE=resident|disk`) and `VLLM_EXL3_NGRAM_KERNEL=ext|torch` (default
`ext`) for the n-gram embedding kernel.

### 6. Benchmark

```bash
# Decode tok/s, excluding TTFT
python scripts/bench_v1.py --base-url http://127.0.0.1:8899/v1 --model Qwen3.8-Flash-Next
```

`scripts/probe_greedy.py` is still in the tree but does not measure what it was
written to measure on this engine; see
[How things were measured](#how-things-were-measured). Use the
`spec_decode_num_accepted_tokens` counters from `/metrics` instead.

## Results, current build (2026-09-13)

vLLM 0.29.0, exllamav3 1.4.7, vllm-exl3 0.4.2, `MAX_MODEL_LEN=262144
GPU_MEM_UTIL=0.80 MAX_NUM_SEQS=4 --mamba-ssm-cache-dtype bfloat16`. Decode
excludes TTFT. Each short-context figure is the median of four samples, and
every sample uses a unique prompt prefix so prefix caching cannot fake a cold
prefill (that this works is verified below by vLLM's own hit counters).

### Draft depth

The earlier sweep (see Historical results) picked k=2 on the build that predated
the fat-expert and dense-routing fixes. On the fixed build the ranking flips:

| Config | Decode @ 4k | Decode @ 32k | KV pool | Per-position acceptance |
|---|---:|---:|---:|---|
| No draft | 27.77 | 27.58 | 512,439 | n/a |
| MTP k=2 | 47.39 | 46.60 | 411,319 | 0.829 / 0.671 |
| MTP k=3 | **50.09** | **49.73** | 385,422 | 0.863 / 0.675 / 0.571 |
| MTP k=4 | wedges | wedges | n/a | engine hangs after torch.compile, never allocates KV |

k=3 over k=2 is about 6%. The two were measured on separate boots at n=4 and
n=3 with overlapping sample ranges (48.1 to 52.8 against 45.5 to 49.9), so call
it likely rather than established.

`--mamba-ssm-cache-dtype bfloat16` for the 36 linear-attention layers raises the
KV pool from 385,422 to 416,163 tokens, about 8%. Decode measured 52.22 at 4k
against 50.09 without it, but the ranges overlap almost entirely at the ±5%
spread speculation introduces. Enable it for the memory; the speed gain is
unproven.

### Context and the MTP acceptance cliff

Only 12 of the 48 layers are full attention, with 2 KV heads at head_dim 256,
so KV costs 24,576 bytes per token and the full 262,144 ceiling needs about
6 GB. The engine boots at 262,144 with 1.6x concurrency headroom at the default
utilisation. Decode barely moves with context: MTP k=3 loses about 4% from 3k
to 163k tokens, and no-draft loses about 4% from 3k to 189k.

Then, at **163,840 prompt tokens (160 × 1024)**, draft acceptance collapses to
exactly 0.000 at every position. Pinned to within 273 tokens:

| Prompt tokens | Tokens per stream chunk | Decode tok/s | |
|---:|---:|---:|---|
| 163,506 | 2.86 | 53.14 | healthy |
| 163,818 | 2.86 | 51.76 | healthy, 22 tokens below the boundary |
| 164,091 | 1.00 | 21.99 | dead, 251 tokens above it |

The full curve, MTP k=3 with bf16 recurrent state. A chunk carrying 1.00 tokens
means no drafts were accepted:

| Prompt tokens | Tokens per chunk | Decode tok/s |
|---:|---:|---:|
| 3,060 | 3.03 | 52.22 |
| 24,354 | 2.98 | 50.53 |
| 97,362 | 2.88 | 48.13 |
| 148,569 | 2.91 | 50.72 |
| 156,018 | 2.91 | 51.98 |
| 163,428 | 2.75 | 48.71 |
| 163,818 | 2.86 | 51.76 |
| 178,287 | **1.00** | 21.95 |
| 208,005 | 1.00 | 21.94 |
| 252,582 | 1.00 | 21.30 |

The draft head keeps drafting past the cliff and the target rejects all of it,
so you pay the draft cost for nothing. The no-draft baseline at comparable
context is 26.59 (measured at 188,661), so above the cliff, **turning
speculation off is worth about 21%.** No-draft was not measured at 252k; the
comparison is established around 180k.

**The mechanism is not identified.** What is known:

- It keys on **prompt length at prefill**, not on decode position. At 163,818
  prompt tokens with 40 generated, about half the generated positions sit past
  163,840, and acceptance stayed at 2.86. Whatever breaks is built once during
  prefill, which points at the QSA block index rather than rotary embeddings.
- No hardcoded 163840 exists anywhere in the `qwen4_exp` or exl3 code path.
- Three candidates were checked and ruled out, recorded so nobody re-chases
  them: `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE = 163840` in `vllm/envs.py` is read
  only by the NVFP4 CUTLASS MoE helpers, which an EXL3 pack never calls, and
  raises rather than degrading silently; a draft config with a smaller
  `max_position_embeddings` is impossible because `config/speculative.py:1175`
  sets `draft_model_config = target_model_config` for MTP; and
  `get_max_prefill_buffer_size()` in the MLA indexer mentions 163840 only in a
  comment, returns `max_model_len * 40`, and is imported only by the DeepSeek
  models.

The next step is runtime instrumentation of the QSA indexer during prefill.
Until then: MTP k=3 below 163,840 prompt tokens, `SPEC_CONFIG=none` at or
above it. The serve script warns when `MAX_MODEL_LEN` crosses the boundary.

### Prefill and prefix caching

Cold prefill sits at 1,075 to 1,180 tok/s across an 80x range of prompt sizes.
That flatness is real, not a scheduling artifact. vLLM warns on every
speculative boot that MTP clamps `max_num_scheduled_tokens` to 2048 and suggests
raising `max_num_batched_tokens`; an 8x larger chunk buys 2%:

| max-num-batched-tokens | util | prefill @ 24k | prefill @ 97k | KV pool |
|---:|---:|---:|---:|---:|
| 2,048 (clamped by MTP) | 0.80 | 1,142.3 | 1,104.3 | 416,163 |
| 16,384 | 0.85 | 1,166.0 | 1,128.7 | 512,619 |

At 0.80 the larger chunk costs enough activation memory that the engine refuses
to start, naming 260,416 as the achievable length. 0.85 fixes that and gives the
largest KV pool measured, but the kernel log then carries
`NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY]` during startup, so it is
documented rather than recommended. The cold ceiling runs far below the box's
arithmetic capability, which points at the quantized-weight path; the profiler
figure in Historical results is for decode and was not re-measured for prefill,
so treat the cause as unconfirmed.

**The prefill lever that matters is prefix caching.** For the real workload, a
fixed document or system prompt with a new question each turn, the cached path
is what you get. Hit rates are vLLM's own counters:

| 196k-token prompt | TTFT | Cached |
|---|---:|---:|
| cold, novel prefix | 178.72 s | 0.0% (0 / 196,022) |
| same document, new question | 2.33 s | 98.9% (193,856 / 196,010) |
| identical repeat | 1.56 s | 99.3% (194,688 / 196,010) |

The 180-second TTFT is a first-turn cost, not a per-turn cost. Do not convert
those into a prefill tok/s figure: dividing the full prompt by the warm TTFT
gives a number in the tens of thousands for tokens the engine never computed.
What it actually computes is the uncached tail, at the same rate as everything
else (1,322 tokens in 1.56 s is 847 tok/s; 2,154 in 2.33 s is 924 tok/s). The
work is simply skipped. The 0.0% hit on a novel prefix is also the check that
validates every cold number in this file.

### Sampling temperature

Every figure above is greedy, which is where draft acceptance is highest. MTP
k=3 at 4k context, three samples per point, acceptance from `/metrics`:

| temperature | decode tok/s | acceptance |
|---:|---:|---:|
| 0.0 | 51.38 | 68.0% |
| 0.3 | 49.64 | 68.6% |
| 0.7 | 51.04 | 66.7% |
| 1.0 | 48.14 | 64.4% |

At a realistic 0.7 the speedup is intact. At 1.0 it costs about 6%.

### Concurrency

Two aggregates are reported, because they answer different questions. The
*window* aggregate is total tokens over the span from the first stream's first
token to the last stream's last token: what a queue of N users sees, and under
vLLM's chunked prefill it charges the other streams' prefills to the first
stream. The *steady* aggregate is total tokens over the span in which every
stream is decoding. Unique prefix per stream, `MAX_NUM_SEQS=8`, temperature 0.

**Short prompts (about 190 tokens), 128 tokens per stream, median of two rounds**

| streams | MTP k=3, steady | MTP k=3, window | no draft, steady | no draft, window |
|---:|---:|---:|---:|---:|
| 1 | 54.8 | 54.8 | 29.0 | 29.0 |
| 2 | 89.4 | 83.5 | 49.1 | 48.2 |
| 4 | **155.6** | 147.7 | 87.2 | 82.3 |
| 8 | **157.6** | 146.9 | **153.4** | 135.5 |

**3k-token prompts, 512 tokens per stream, one round**

| streams | MTP k=3, steady | MTP k=3, window | no draft, steady | no draft, window |
|---:|---:|---:|---:|---:|
| 1 | 53.2 | 53.2 | 28.7 | 28.7 |
| 2 | 79.6 | 77.0 | 46.0 | 44.8 |
| 4 | **102.9** | 64.3 | 63.0 | 41.5 |
| 8 | 90.5 | 47.0 | **73.1** | 40.9 |

**The engine scales.** An earlier version of this section reported a plateau at
two streams (78 tok/s). That came from 3k-token prompts, 128 tokens per stream
and the window aggregate alone: with that little decode per stream the window
is mostly the other streams' chunked prefills, so it measured the scheduler's
interleaving rather than decode. A torch profile of a 1-stream and a 4-stream
decode window (short prompts, 256 tokens) shows the kernels batching as they
should: the fused MoE takes 2.0x the GPU time for 4x the tokens and the dense
EXL3 layers are flat. Long contexts do cost more per batched step (3k prompts:
103 tok/s at four streams against 156 with short prompts), which is the
attention and PLE state work per token. MTP k=3 wins at every concurrency up to
four streams and is level with no draft at eight. **Throughput sweet spot: MTP
k=3 at four streams, about 150 tok/s aggregate on short prompts and about 100
on 3k-token prompts.**

### Levers that measured empty

Recorded so nobody re-tests them:

| Lever | Result |
|---|---|
| `--max-num-seqs 2` vs 4 | 52.84 vs 52.22 tok/s at 4k, inside noise; KV pool +2%. Keep 4 for burst tolerance. |
| CUDA graph mode | already `FULL_AND_PIECEWISE` on every boot; nothing to enable |
| `--max-num-batched-tokens 16384` | +2% prefill, see above |
| `--kv-cache-dtype fp8` | refused: `Qwen4Exp QSA requires a BF16 main KV cache` |
| MTP k=4 | wedges the engine after torch.compile |

### The 4.05 bpw revision

turboderp also publishes `4.05bpw_h6_ng6`: dense linears at K=6 and experts at
K=4 (against K=5 and K=3), a 6-bit n-gram table, 107.5 GB on disk. Measured on
the same build and harness with the table resident on the device (the default),
it needs `--gpu-memory-utilization 0.92` to boot at all, and a 131,072 context:

| | 3.05 bpw | 4.05 bpw |
|---|---:|---:|
| Model resident | 79 GiB | **100.81 GiB** |
| n-gram table, packed | 30.4 GiB | 36.36 GiB |
| Utilisation needed | 0.80 | **0.92** |
| Max context that boots | 262,144 | **131,072** |
| KV pool | 416,163 tokens (1.6x at 262k) | 195,509 tokens (1.49x at 131k) |
| MemAvailable while serving | 17 to 18 GiB | **3.4 GiB ready, 2.5 GiB under load** |
| Decode @ 4k, MTP k=3 | 52.22 | 48.70 |
| Decode @ 32k, MTP k=3 | 50.53 | 51.31 |
| Cold prefill @ 24k | 1,142 | 1,137 |
| Draft acceptance | 68% | **74%** |

**Speed is unchanged within noise.** Decode at 4k and 32k and cold prefill all
land inside the 3.05 pack's sample spread. That is consistent with the profiler
finding that decode is bound by trellis dequantization rather than bytes moved:
the per-weight dequant cost does not grow much with K. The one speed-adjacent
gain is acceptance, 68% to 74%, a higher-precision target agreeing with its
draft more often.

**The cost is entirely memory, and it is severe.** 22 GiB more resident,
utilisation forced to 0.92, and half the context. At 0.92 the box sits under
earlyoom's 3% trigger and survives only because swap is free; a 262k attempt at
0.93 without a draft drove MemAvailable to 1 GiB during load and was killed by
the watchdog before KV allocation. With the table resident, **the 4.05
revision does not reach 262k on one Spark**, and the configuration that does
boot is not one to serve from. Quality was not measured here; that needs the
sixcat harness from the historical comparison.

**With the n-gram table on NVMe it fits.** vllm-exl3 `main` (`94c29ba`) can
keep the table as the checkpoint's memory-mapped views and gather each
lookup's rows on the host (`NGRAM_TABLE=disk` in the serve script,
`VLLM_EXL3_NGRAM_TABLE=disk` underneath), so the 36 GiB never lands on the
device. Measured 2026-09-14, same harness, `MODEL_DIR=<4.05> NGRAM_TABLE=disk`:

| 4.05 bpw | table resident, 131k, util 0.92 | table on NVMe, 262k, util 0.80 |
|---|---:|---:|
| Boots at 262,144 | no | **yes** |
| KV pool | 195,509 tokens (1.49x at 131k) | **954,453 tokens (3.64x at 262k)** |
| MemAvailable while serving | 3.4 GiB | **18 to 20 GiB** |
| Decode @ 4k / 32k, MTP k=3 | 48.70 / 51.31 | 41.1 / 47.9 |
| Decode @ 4k, no draft | not measured | 24.4 |
| Cold prefill @ 24k | 1,137 | 1,128 |

The decode cost has two parts. The host gather is a synchronization point, so
the lookup must run outside CUDA graphs: disk mode uses PIECEWISE-only graphs
with the lookup as a splitting op, and PIECEWISE alone costs 5 to 7% on this
model (measured on the 3.05 pack: 48.25 / 48.01 against 52.22 / 50.53 at k=3).
On top of that, rows page in from NVMe the first time a prompt touches them,
which is why the short-prompt samples start slow (32.6 to 46.0 tok/s across
the four) and the 32k figure sits where PIECEWISE alone would put it. The
right way to read it: 4.05 at 262k on one Spark is now a real configuration,
about 10% slower than 3.05 at the same context, with 15 GiB more headroom than
3.05 resident.

This revision ships the n-gram table as one unsharded `ngram_embedding.trellis`
tensor (`I16 [320001536, 61]`, 39 GB) where 3.05 ships 128 `shard_N.trellis`
shards. Plugin builds before `94c29ba` misfiled it as a dense linear and the
boot OOMed in about 100 seconds; `scripts/rename_unsharded_ngram.py` is the
workaround for those builds only.

### Running the pack through exllamav3 directly

The same weights also run through exllamav3's own engine, with no vLLM in the
process. Measured here on 1.5.0 with the extension JIT-built for sm_121 on this
Spark, same prompt builder and salts as `qbench.py`, greedy, 128 new tokens,
decode excluding TTFT, median of four. Scripts are in `scripts/exl3_native/`; the vLLM concurrency client is `scripts/concurrency.py`.

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
it is set against the steady aggregate from the Concurrency section above)

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

**What the numbers say.** Single-stream decode is 8 to 20% faster on the same
weights and prefill is the same engine speed. Under concurrency the two are at
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

- No OpenAI-compatible API in what was measured. exllamav3 is normally served
  through [TabbyAPI](https://github.com/theroyallab/tabbyAPI); its overhead and
  its feature set on this model were not measured here, and the concurrency
  numbers above are for jobs enqueued directly on the generator.
- No reasoning parser, tool-call parser, structured output or the rest of
  vLLM's serving surface. Clients that expect `--reasoning-parser qwen3` and
  hermes tool calls get raw text.
- No tensor parallel across two Sparks (exllamav3's TP path is one of the x86
  code paths the aarch64 patch stubs out).
- Prefix caching was not measured natively. exllamav3's generator has cache
  reuse, so the 114x cached-prefix result above has no native counterpart yet.
- The MTP draft needs `max_history=k` on both caches and `max_batch_size` set
  to the number of concurrent jobs at cache creation; the harness does this.
- It cannot run everything the plugin can. DeepSeek V4.1 with Engram has no
  forward-correct graph in upstream exllamav3; for that pack the vLLM path is
  the only one.

**How to run it.** `scripts/exl3_native/setup_exllamav3_150.sh` builds a venv
with exllamav3 1.5.0 and JIT-compiles the extension (the release still carries
x86-only intrinsics in its CPU MoE offload and tensor-parallel paths, so it
applies the aarch64 patch from the plugin repo plus two stub symbols 1.5.0
added). `scripts/exl3_native/make_native_view.sh` builds a symlink view of a
prepared pack with the native `config.json` and index, since `prepare_pack.sh`
rewrites those for vLLM; for a 4.05 pack that was prepared with the old
`rename_unsharded_ngram.py` step, pass `ngram_embedding.safetensors.native` as
the third argument so exllamav3 reads the original unsharded file. A pack
prepared with the current tools needs no third argument. `bench_native.py` and `bench_native_conc.py` are
the two harnesses. One GB10 trap: `cudaMemGetInfo` reports MemFree rather than
MemAvailable, so page cache left by a previous model load makes exllamav3's
autosplit refuse a 100 GiB pack while `/proc/meminfo` shows 118 GiB available.
Both harnesses take `--prealloc-gib 100`, which allocates and frees that much on
CUDA first so the kernel reclaims the cache.

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

**Which to use.** For one user on one Spark, exllamav3 directly is the faster,
roomier and simpler engine, and it loads in under a minute. The vLLM path is
for what needs vLLM: the OpenAI API with its parsers and structured output,
tensor parallel across two Sparks, tooling that assumes a vLLM endpoint, and
packs exllamav3 cannot run.

#### Tuning the native engine on GB10 (2026-09-16): 56 → 73 tok/s

A second pass on the native path, this time against exllamav3 master
(`02aef45`, post-1.5.0) built from my fork with the aarch64 guards committed
([vcruz305/exllamav3#1](https://github.com/vcruz305/exllamav3/pull/1)) instead
of patched at install time. Measured with `examples/chat.py` through
`scripts/exl3_native/tuning/bench.sh`: page cache dropped before every load,
400 new tokens, the nginx-parser code prompt, `-tps`. Numbers are decode tok/s.

| Change (cumulative) | chat.py, greedy | chat.py, temp 0.8 | in-process `sweep.py` |
|---|---:|---:|---:|
| `-mtp -ndt 3` (starting point) | | 62.8 | 64.6 |
| `-ndt 4` | | ~61 | 69.4 |
| + pin to the ten Cortex-X925 cores, `taskset -c 5-9,15-19` | | | 71.0 |
| + `EXL3_INT8_GEMV=0` | | 64.5 | 71.9 |
| + `EXL3_MOE_COOP_WIDE=1` | | | 76.2 |
| + `-ndt 5` | **73.0 ± 0.3** (4 cold runs) | 60.3 to 73.3 | **77.0** |
| `-ndt 6` | 72.2 | 65.5 | |
| prose prompt, same config | | 37 to 41 | 37.5 |

Three levers did the work, and none is a flag you would guess from the docs:

- **`EXL3_INT8_GEMV=0`.** The fused int8-activation GEMV path for mul1 tensors
  is *slower* on GB10; disabling it is worth about 3 tok/s.
- **`EXL3_MOE_COOP_WIDE=1`.** The fused decode MoE kernel picks its tile
  geometry per architecture, and the default only chooses the wide 128-column,
  4-way k-split tile on datacenter Blackwell. GB10 has 48 SMs and wants it too:
  about 5 tok/s.
- **Big-core affinity.** GB10 pairs ten Cortex-X925 (capacity 997–1024) with
  ten A725 (718–731). Left to the scheduler, the launch thread lands on little
  cores often enough to cost 2 tok/s.

Deeper drafts stop paying at 5: the verify forward scales steeply with draft
length (q=2 → 37 ms, q=5 → 52, q=7 → 59, q=9 → 86, `split_time.py`).

**Sampling temperature is now the largest source of variance.** Greedy runs
reproduce to ±0.3 tok/s with an identical 315/425 acceptance every time; the
default temperature-0.8 sampler swings acceptance 58–75% run to run and with it
60–73 tok/s on the same prompt. A/B anything with `-topk 1`.

**Where a verify round goes** (q=6, 58 ms GPU, `kern_rounds.py`): fused MoE
coop kernels 29%; the 5-bit `lm_head` 17%; the GatedResidual hyper-connection
mixer (`gr_dots`/`gr_finalize`, 113 launches per round) 20%; attention and GDN
projections 14%; the recurrent GDN kernel 6%. Unified memory measured 215 GB/s
(`bw.py`), so the ~48 GB weight read per round is only ~5 ms of the 58: decode
here is launch count and small-kernel latency, not weight bandwidth.

**One kernel change, measured empty.** I wrote row-batched GatedResidual
kernels (`gr_dots_rb`/`gr_finalize_rb`, in the fork behind `EXL3_GR_RB=1`,
default off) that stream each site's ~13 MB of low-rank weights once per call
instead of once per row. Parity with the fp32 reference is identical to the
originals (max rel 4.3e-4) and the mixer drops 9.1 → 7.7 ms/round at R=6, but
end to end it is inside run-to-run noise (73–78 either way over four A/Bs), and
the different accumulation order shifts greedy trajectories enough to move MTP
acceptance. A 15% win on a 20% slice does not show under speculative decoding's
variance. `gr_parity.py` and `ab_greedy.py` reproduce both results.

**What is left is in the pack, not the runtime.** `lm_head` is 5-bit over the
248,320-token vocabulary, 397 MB, read six times per round (five draft steps
plus the verify): about 2.4 GB and 11 ms of the 58, at ~240 GB/s effective. It
does not get faster without fewer bytes, which means re-quantizing with
`head_bits 3` (and possibly `mtp_bits 2`) for an estimated 4 ms/round, ~7%.
Not done; it would be a `vcruz305` pack rather than turboderp's revision.

`scripts/exl3_native/tuning/run-qwen38-exl3.sh` is the launcher with all of the
above baked in; `bench.sh`, `sweep.py`, `split_time.py`, `kern_rounds.py`,
`bw.py` are the harnesses; `gr-row-batched.patch` is the kernel change as a
format-patch.

## How things were measured

**Decode** is `(completion_tokens - 1) / (wall - TTFT)`, counting tokens from
the usage block rather than stream chunks, because with speculation one chunk
carries several tokens. TTFT is time to the first chunk, which under
speculation holds up to k+1 tokens whose time lands in TTFT while they count in
`n`, so MTP decode figures run about 2% high. Uniform across configurations, so
comparisons hold.

**Cold prefill** means a unique prefix on every request. That this defeats the
cache is verified by the 0.0% hit rate above, not assumed.

**Aggregate throughput** is total completion tokens over the slowest stream's
decode window.

**Acceptance** is `spec_decode_num_accepted_tokens_total` over
`spec_decode_num_draft_tokens_total` from `/metrics`, read before and after
each batch of requests. Tokens-per-stream-chunk is a free proxy: 1.00 means
nothing was accepted.

**Greedy text-diff is not a valid fidelity test on this engine.** The engine
is not deterministic at temperature 0 with a fixed seed. Same server, same
request:

```
"Summarize the causes of the 1929 stock market crash in one paragraph."
  -> 4 distinct outputs from 5 identical requests
"What is 17 * 23? Show the steps."
  -> 1 distinct output from 5
```

Spec-on against spec-on, same boot and config, scores the same 2 of 6 identical
as spec-on against spec-off. The control is indistinguishable from the test, so
a text diff measures engine noise. Open-ended generations diverge, tightly
constrained ones do not, which is what near-ties in the logits resolved by
varying reduction order would look like. `scripts/probe_greedy.py` predates
this finding; the acceptance counters are the usable signal.

## Hardware, model, memory

```text
GPU:     NVIDIA GB10
Memory:  128 GB unified (aarch64), 121.7 GiB visible
Engine:  vLLM --quantization exl3, tensor-parallel size 1 only
```

Qwen3.8-Flash-Next is a `Qwen4ExpForConditionalGeneration` model: 48 layers
(36 linear attention, 12 full attention, `full_attention_interval: 4`), 512
experts with top-10 routing, hidden size 2560, 2 KV heads at head_dim 256, one
MTP layer, a per-layer n-gram (PLE) embedding table (320,001,536 rows × 160,
K=5 packed), and a vision tower. `max_position_embeddings` is 262,144, so that
is the context ceiling without RoPE overrides.

Memory at the default serve config (`GPU_MEM_UTIL=0.80`, 262,144 context, MTP
k=3, bf16 recurrent state):

- Model resident on device: about **79 GiB**, the n-gram table 30.4 GiB of it
  packed.
- KV pool: **416,163 tokens** (512,439 with no draft; the draft's own KV is the
  difference). One 262,144-token request needs about 7.2 GiB, so that is 1.6x
  concurrency at max context.
- MemAvailable while serving: about 17 to 18 GiB.
- **Do not go above 0.85.** 0.85 works and gives the largest KV pool measured,
  but the kernel log shows driver allocation failures during startup. Keep a
  memory watchdog running when experimenting there.
- Cold load about 9.5 minutes from NVMe; 2.5 minutes from page cache.

## Historical results (2026-09-07 and 09-08, earlier build)

These were measured on vLLM 0.28.1 nightly with vllm-exl3 at or before 0.3.1
and the pre-fix build (before PR #5 and PR #7). They are kept because the
quality comparison against the GGUF and the profiler breakdown are still
informative, and because the draft-depth ranking changed between builds. For
current speeds use the section above.

### Draft-depth sweep (2026-09-07, 32k context)

| Config | Decode tok/s (runs) | TTFT | Notes |
|---|---|---|---|
| No draft | 27.97 / 27.56 / 27.16 / 27.35 | 0.185 s | KV cache: 385,570 tokens |
| MTP k=1 | 36.37 / 33.80 / 35.26 / 35.28 | 0.199 s | mean acceptance 1.855 / 2; KV: 275,636 |
| MTP k=2 | 37.30 / 38.88 / 41.34 / 39.32 | 0.201 s | mean acceptance 2.535 / 3; KV: 235,412 |
| MTP k=3 | 37.91 / 34.44 / 36.73 / 38.18 / 35.08 / 37.35 / 35.22 | 0.21 s | mean acceptance 2.951 / 4; KV: 224,694 |

This sweep picked k=2. On the fixed build k=3 wins; see Draft depth above.

### Long context (2026-09-07, 262,144 configured, `MAX_NUM_SEQS=2`)

| Config | KV cache | TTFT on 122,902 tokens | Decode |
|---|---|---|---|
| No draft | 546,503 tokens (2.08x at 262k) | 107.0 s (about 1,150 tok/s) | 26.2 tok/s |
| MTP k=2 | 402,630 tokens (1.54x) | 110.6 s | about 43 tok/s, acceptance 2.49 to 2.65 |

At util 0.85 with 4 sequences and no draft: 759,773 tokens of KV (2.90x),
11.5 GiB MemAvailable, 27.4 tok/s decode on the same prompt. These runs
predate the fat-expert fix; the speeds stand, the output quality of those
long-prompt runs does not.

### Where the time goes (torch profiler, no draft, 32 decode steps, 32k context)

| Kernel family | Share |
|---|---:|
| EXL3 dense GEMV/GEMM (K5 attention, linear-attention, lm_head projections) | 47% |
| EXL3 fused MoE (`exl3_moe`, K3 experts) | 34% |
| bf16 GEMMs (hyper-connection mixers, router) | 5% |
| linear attention (gated delta rule) | 2% |
| norms | 1% |
| everything else (tiny elementwise launches) | 11% |

Decode is bound by trellis dequantization rather than raw memory bandwidth: the
EXL3 kernels take roughly 3 to 5x the time the weight bytes would need at
273 GB/s. That is why MTP helps, since fewer dequant passes are needed per
emitted token. This profile was taken on the pre-fix build; its explanation
for why k=3 did not beat k=2 there no longer applies on 0.4.2.

### EXL3 against the Q4_K_M GGUF (2026-09-08)

Setup: single DGX Spark. EXL3 side: this recipe's pack (3.05 bpw), vllm-exl3
main 6b26e5c, MTP k=2, 65,536 context, `--reasoning-parser qwen3
--enable-auto-tool-choice --tool-call-parser qwen3_xml`. GGUF side:
vcruz305/Qwen3.8-Flash-Next-GGUF Q4_K_M with the BF16 PLE table file-backed,
llama.cpp with the qwen4exp NextN/MTP draft head (upstream PR 27836),
`--spec-type draft-mtp --spec-draft-n-max 3`, same context. Both sides run their
own draft head, vendor sampling (thinking on, temperature 1.0, top_p 0.95,
top_k 20), one request at a time. Benchmark: sixcat-eval v0.5.1, 20 items per
category, 120 per side.

Quality (sixcat scores, percent; one item is five points). The two land within
one or two items of each other everywhere, which is a tie:

| Category | EXL3 | GGUF |
|---|---:|---:|
| Knowledge | 85.0 | 85.0 |
| Math | 100.0 | 100.0 |
| Truth | 80.0 | 85.0 |
| Instruct | 85.0 | 90.0 |
| Code | 90.0 | 85.0 |
| Tools | 80.0 | 90.0 |
| Overall | 86.7 | 89.2 |

The tools row needs a serving flag rather than a better model: this model emits
XML-style calls, so `--tool-call-parser qwen3_xml` is required; with the default
JSON parser the same run scores 15.0. Both columns score the first answer
recorded per item, except the EXL3 tools row, which is from the corrected-parser
pass. sixcat's printed overall is best-of-attempts (the EXL3 log carries 29
retries, 17 of them unparseable tools answers and 12 genuine failures, 6 of
which flipped on a second sample); the GGUF run was never offered a retry, so
the table compares first attempts on both sides.

Latency and throughput, twenty streamed requests per row with unique prefixes:

| Metric | EXL3 | GGUF |
|---|---:|---:|
| Greedy TTFT p50 / p95 | 0.300 / 0.323 s | 0.383 / 0.574 s |
| Greedy decode p50 | 47.6 tok/s | 32.9 tok/s |
| Greedy decode p05 / p95 | 43.4 / 50.6 | 28.2 / 40.1 |
| Thinking TTFT p50 / p95 | 0.361 / 0.388 s | 0.439 / 0.488 s |
| Thinking decode p50 | 38.4 tok/s | 24.0 tok/s |
| Prefill, 1,221-token prompt | 902 tok/s | 439 tok/s |
| Prefill, 9,483-token prompt | 1,122 tok/s | 634 tok/s |
| Whole 120-item suite | 37.8 tok/s | 26.5 tok/s |

Memory on one machine:

| Measure | EXL3 | GGUF |
|---|---|---|
| Weights | 79.96 GiB resident, n-gram table quantized and resident | 80.2 GiB backbone resident, 95.4 GiB BF16 PLE table paged from NVMe |
| KV cache | 11.04 GiB, 303,951 tokens at 64k | not separately reported |
| Device allocation | 93.2 GiB idle, 94.1 GiB under load | 82.5 GiB |
| System memory in use | 102.4 GiB idle, 103.2 GiB under load | 86.9 GiB idle, 91.7 under load, 95.7 after |
| On-disk serving set | 79.4 GiB (30.4 GiB the n-gram table) | 175.6 GiB (95.4 GiB the BF16 table) |

The GGUF's smaller resident figure is its 95.4 GiB embedding table living on
disk, which is why its process size climbs through a run and why its prefill is
about half the speed. An upstream llama.cpp change that reads PLE rows with
explicit preads (PR 28136) builds but aborts during decode with the MTP draft
head, so it is not usable yet. NVFP4 does not fit on one Spark for this model:
123.6 GiB of weights before any KV cache, against 121.7 GiB of unified memory.

## Known limitations

- **MTP acceptance cliff at 163,840 prompt tokens.** Speculation is a 21% loss
  above it. Mechanism unknown; see the Context section for what has been ruled
  out.
- **Batched decode slows with context.** Steady-state aggregate at four
  streams is 156 tok/s on short prompts and 103 on 3k-token prompts (MTP k=3).
  See Concurrency.
- **The 4.05 bpw revision needs `NGRAM_TABLE=disk` to reach 262k**, which costs 5 to 7% decode (PIECEWISE-only CUDA graphs) plus first-touch page-ins. Resident, it stops at 131k, util 0.92, 2 to 3 GiB free. See The 4.05 bpw revision.
- **Unsharded n-gram tables need vllm-exl3 `main` at `94c29ba` or newer.** Older builds misfile the table; `scripts/rename_unsharded_ngram.py` is their workaround.
- **Cold prefill ceiling of about 1,100 tok/s**, not improved by chunk size.
  Likely the quantized-weight path; unconfirmed without a prefill profile.
- Fat-expert prefill bug (fixed 2026-09-08, vllm-exl3 PR #5). Before the fix,
  any prompt that routed more than 256 tokens to one expert produced wrong
  hidden states while short prompts looked normal: mean NLL over a 6,000-token
  corpus was 4.21 through vLLM against 0.94 through exllamav3 on the same pack.
  With the fix the served model scores 0.941 to 0.944.
- Engine wedge on the vLLM V2 runner (fixed 2026-09-08, vllm-exl3 PR #7).
  exllamav3 dispatches dense calls by row count: up to 2 rows use
  non-cooperative GEMV, 3 to 144 rows use cooperative trellis GEMM, above 144
  it reconstructs the weight and runs hgemm. The cooperative GEMM wedged the
  engine deterministically on 33 to 144-token prompts and on MTP evaluation.
  The fix routes 17 to 144-row calls through the reconstruct path. A 4-worker
  stress that wedged the old build in 161 s ran clean for 45 minutes (1,085
  requests, 77 tok/s aggregate). Knobs: `VLLM_EXL3_RECONSTRUCT_MIN_ROWS`
  (default 17), `VLLM_EXL3_COOP_GEMM=1` restores the old dispatch for A/B runs.
  MTP k=4 wedging on the current build is likely the same class.
- Tensor-parallel size 1 only; the padded dense geometry and the n-gram table
  are not sharded for TP > 1.
- Vision attention q/k/v is served from the pack's bf16 fused copy, not the
  split EXL3 tensors.
- The plugin's native fused-MoE kernel is not used for this geometry; the
  `exllamav3` `exl3_moe` kernel is used instead.
- First boot needs the pack-preparation steps; a raw download will not load.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `EXL3 n-gram table: N of 128 shards never loaded` | `model.safetensors.index.json` was not regenerated; run `scripts/prepare_pack.sh` (step 3) |
| loading consumes all memory and dies with `CUDACachingAllocator ... memory mapping failed with OOM`, or is killed partway through | the n-gram table was allocated dense at ~95 GiB rather than 30.4 GiB packed because vLLM had no quant_config for it. Run `patch_vllm_qwen4_ple.py` (step 4). Confirm with `EXL3 n-gram embedding ready: ... 30.40 GiB packed` in the log. |
| `ValueError: There is no module or parameter named 'lm_head.mul1'` | run `patch_vllm_qwen4_ple.py`; with an MTP draft, `patch_vllm_mtp_lmhead.py` as well |
| `ValueError: There is no module or parameter named 'blocks.0.attn.k_proj'` | run `patch_vllm_vision_split.py` |
| `ModuleNotFoundError: No module named 'exllamav3_ext'` | build exllamav3 1.4.7 from source with the aarch64 patch; the pure-Python wheel is not enough |
| `ValueError: No available memory for the cache blocks` | `--gpu-memory-utilization` too low for weights plus KV; use 0.80 |
| `ValueError: To serve at least one request with the model's max seq len ... larger than the available KV cache memory` | activation memory (usually a large `--max-num-batched-tokens`) ate the KV pool; lower the chunk or raise util to 0.85 |
| boot OOMs within ~100 s of starting to load, `expandable_segments: memory mapping failed` with the device full, and the prep log said `ngram_embedding: None` | the pack's n-gram table is one unsharded tensor and got no quant spec, so it was allocated dense. Run `scripts/rename_unsharded_ngram.py <pack>` then `prepare_pack.sh` again; look for `n-gram embedding ready: 1 shards x ... packed` |
| `NotImplementedError: Qwen4Exp QSA requires a BF16 main KV cache` | `--kv-cache-dtype fp8` is not supported on this path; remove it |
| `max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings` | informational; raising the chunk was measured at +2%, ignore |
| decode collapses to ~22 tok/s on very long prompts with MTP on | the acceptance cliff at 163,840 prompt tokens; use `SPEC_CONFIG=none` for that workload |
| engine alive after `torch.compile` but never allocates KV | MTP k=4, or the cooperative-GEMM wedge class; use k=3 |
| `EXL3 linear load shape mismatch ... (4304,) != (4352,)` or `torch.compile: Attempted to call function marked as skipped` | `vllm-exl3` older than 0.4.0 |
| evaluation harness scores the reasoning text as the answer | serve with `--reasoning-parser qwen3` (the script does) |
| memory watchdog kills the process, or MemAvailable runs low | lower `GPU_MEM_UTIL` or `MAX_MODEL_LEN` |

These failures appear in a fixed order, each reached only after the previous is
resolved. The first is the most misleading: a missing patch step presents as an
out-of-memory failure that reads like insufficient hardware.

## Related repositories

| Repo | Role |
|---|---|
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | the pack this recipe serves |
| [vllm-exl3](https://github.com/vcruz305/vllm-exl3) | the EXL3 plugin: source, releases, issues, and the pack-prep / vLLM-patch tools this recipe calls |
| [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) | my exllamav3 fork: master = upstream + the aarch64 build guards and the opt-in row-batched GatedResidual kernels ([#1](https://github.com/vcruz305/exllamav3/pull/1)); what the native-engine tuning numbers were measured on |
| [GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe) | sibling recipe this one is modeled on |
| [DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe) | sibling recipe this one is modeled on |

## Credits and upstream work

**ExLlamaV3 by Turboderp ([@turboderp](https://github.com/turboderp-org/exllamav3)).**
The EXL3 trellis format, the MCG codebook, the quantization method, and the
`Qwen3.8-Flash-Next-exl3` pack itself are theirs. MIT, Copyright (c) 2025
Turboderp.

**[vLLM](https://github.com/vllm-project/vllm)** is the serving engine this
recipe runs on.

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
pack license. vLLM, ExLlamaV3/EXL3, and `vllm-exl3` have their own licenses.
