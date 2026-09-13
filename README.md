# Qwen3.8-Flash-Next EXL3 on one NVIDIA DGX Spark

## Interactive benchmark

[**Open the animated benchmark**](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/)

Explore the recorded MTP sweep and separate long-prompt probe in a landscape
viewer with a 19.5-second recording tour, PNG export, and downloadable HTML/data.
No model hardware is needed to view or render the presentation.

[**MTP sweep**](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=sweep) ·
[**Long prompt**](https://vcruz305.github.io/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/?scene=long) ·
[**Source and local-render instructions**](docs/README.md)

The full k sweep uses **32K configured context**. The separate long-prompt probe
uses **122,902 input tokens** with a **262,144-token configured limit**.
This viewer displays saved results; it does not run inference in the browser.

---

Serves [turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(revision `3.05bpw_h5_ng5`, about 80 GB) on a single NVIDIA DGX Spark (GB10,
128 GB unified memory, aarch64) through a vLLM nightly and the
[vllm-exl3](https://github.com/vcruz305/vllm-exl3) plugin. Measured on
`cruz-spark` on 2026-09-08, single in-flight request at 65,536-token context
with MTP k=2: **47.6 tok/s greedy decode (p50)** at **0.300 s TTFT (p50)**,
**38.4 tok/s at vendor thinking settings**, and **1,122 tok/s prefill** on a
9,483-token prompt. The whole model including the n-gram embedding table stays
resident in about 102 GiB, leaving 11.04 GiB of KV cache, which is 303,951
tokens at 64k context. k=2 is the recommended draft setting; k=1 and k=3 are
both slower (see the k-sweep below). Full numbers, and the comparison against
the Q4_K_M GGUF of the same model, are in
[Benchmark results](#benchmark-results-2026-09-08-one-dgx-spark).

This recipe covers install, pack preparation (the pack as published needs
three one-time rewrites before vLLM's native loader will serve it), the
three runtime patches vLLM needs until this architecture is upstreamed,
serving with and without the MTP draft, and the benchmark/probe tools used
to produce the numbers below.

## Draft-depth sweep (measured 2026-09-07, cruz-spark)

These four rows are the k-sweep that picked k=2, measured at 32k context on the
build that predates the fat-expert fix (PR #5) and the dense-routing fix (PR #7).
They are kept for the KV-cache and acceptance comparison across k. For current
speeds use [Benchmark results](#benchmark-results-2026-09-08-one-dgx-spark),
where the same k=2 configuration measures faster on the fixed build.

| Config | Decode tok/s (4 runs) | TTFT | Notes |
|---|---|---|---|
| No draft | 27.97 / 27.56 / 27.16 / 27.35 | 0.185 s @ 128 tok | KV cache: 385,570 tokens |
| MTP k=1 | 36.37 / 33.80 / 35.26 / 35.28 | 0.199 s | mean acceptance 1.855 / 2; KV cache: 275,636 tokens |
| MTP k=2 | 37.30 / 38.88 / 41.34 / 39.32 | 0.201 s | mean acceptance 2.535 / 3; KV cache: 235,412 tokens |
| MTP k=3 | 37.91 / 34.44 / 36.73 / 38.18 / 35.08 / 37.35 / 35.22 (7 runs) | 0.21 s | mean acceptance 2.951 / 4; KV cache: 224,694 tokens; k=2 remains the best setting |

Decode tok/s excludes TTFT (see `scripts/bench_v1.py`). Greedy output with
the MTP draft matched the no-draft text exactly on 1 of 4 fixed prompts (k=1,
k=2) or 2 of 4 (k=3) and diverged partway through -- with coherent text -- on the
others, for k=1
and k=2 alike. That is expected of greedy-consistent speculation (the target
either accepts or rejects each draft token; it never emits a token the
target itself would not have chosen), not evidence of a bit-exact match end
to end. A DeepSeek V4 Flash regression on this same plugin build passed.

## Operating envelope (measured 2026-09-13, vllm-exl3 0.4.2)

Re-swept on the current build (vLLM 0.29.0, exllamav3 1.4.7, vllm-exl3 0.4.2)
with `MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.80 MAX_NUM_SEQS=4`. Decode excludes
TTFT; each figure is the median of four samples, and every sample uses a unique
prompt prefix so prefix caching cannot fake a cold prefill.

**k=3 is now the best setting.** The 2026-09-07 sweep above picked k=2 on the
build that predated the fat-expert and dense-routing fixes. On the fixed build
the ranking flips:

| Config | Decode @ 4k | Decode @ 32k | KV pool | Per-position acceptance |
|---|---|---|---|---|
| No draft | 27.77 | 27.58 | 512,439 | n/a |
| MTP k=2 | 47.39 | 46.60 | 411,319 | 0.829 / 0.671 |
| MTP k=3 | **50.09** | **49.73** | 385,422 | 0.863 / 0.675 / 0.571 |
| MTP k=4 | wedges | wedges | n/a | engine hangs after torch.compile, never allocates KV |

`--mamba-ssm-cache-dtype bfloat16` for the 36 linear-attention layers raises the
KV pool from 385,422 to 416,163 tokens, about 8%. Decode measured 52.22 at 4k
against 50.09 without it, but the sample ranges overlap almost entirely at the
+/-5% spread that speculation introduces, so treat the memory gain as the reason
to enable it and the speed gain as unproven.

### The MTP acceptance cliff

Draft acceptance collapses to exactly 0.000 at every position at **163,840
tokens (160 x 1024)**, pinned to within 273 tokens. The draft head keeps
drafting and the target rejects all of it, so past the cliff you pay the full
draft cost for no benefit.

| Prompt tokens | Tokens per chunk | Decode tok/s | |
|---:|---:|---:|---|
| 163,506 | 2.86 | 53.14 | healthy |
| 163,818 | 2.86 | 51.76 | healthy, 22 tokens below the boundary |
| 164,091 | 1.00 | 21.99 | dead, 251 tokens above it |

| Context (real tokens) | Tokens per stream chunk | Decode tok/s |
|---:|---:|---:|
| 3,060 | 3.03 | 52.22 |
| 24,354 | 2.98 | 50.53 |
| 97,362 | 2.88 | 48.13 |
| 148,569 | 2.91 | 50.72 |
| 163,818 | 2.86 | 51.76 |
| 156,018 | 2.91 | **51.98** |
| 163,428 | 2.75 | 48.71 |
| 178,287 | **1.00** | 21.95 |
| 208,005 | 1.00 | 21.94 |
| 252,582 | 1.00 | 21.30 |

A chunk carrying 1.00 tokens means zero drafts were accepted. For comparison the
no-draft baseline holds up across the whole range: 27.77 at 3,060 tokens, 27.29
at 97,362, and 26.59 at 188,661. So above the cliff, turning speculation **off**
is worth about 21% (26.59 against 21.95, measured at comparable context). That
comparison is established around 180k; no-draft was not measured at 252k.

**The mechanism is not identified.** Three candidates were checked and all
three are ruled out, recorded here so nobody re-chases them:

- `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE = 163840` in `vllm/envs.py` is an exact
  numeric match, but it is read only by the NVFP4 CUTLASS MoE helpers in
  `_custom_ops.py`, which an EXL3 pack never calls, and exceeding it raises
  rather than degrading silently.
- A draft model config with a smaller `max_position_embeddings` is impossible
  here: `vllm/config/speculative.py:1175` sets
  `draft_model_config = target_model_config` for `method="mtp"`.
- `get_max_prefill_buffer_size()` in `v1/attention/backends/mla/indexer.py`
  mentions 163840 in a comment, but returns `max_model_len * 40` and is
  imported only by `deepseek_v2.py` and `deepseek_v32/attention.py`.

No hardcoded 163840 exists anywhere in the qwen4_exp or exl3 path, yet the
boundary lands on it to within 273 tokens. A power-of-two multiple that exact
is not coincidence, so the next step is runtime instrumentation of the QSA
indexer rather than more grepping. Until it is understood:

- Below 163,840 tokens: MTP k=3, which is where the 1.8x lives.
- At or above it: `SPEC_CONFIG=none`.

`scripts/serve_one_spark_qwen.sh` defaults to k=3 and prints a warning when
`MAX_MODEL_LEN` exceeds the cliff.

### Prefill

Cold prefill sits at 1,075 to 1,180 tok/s across an 80x range of prompt sizes,
from 3k to 252k tokens. That flatness is real and is not a scheduling artifact.

Raising the prefill chunk does almost nothing. vLLM warns that speculation
clamps `max_num_scheduled_tokens` to 2048 and suggests raising
`max_num_batched_tokens`, so it is worth testing, and the answer is that an 8x
larger chunk buys 2%:

| max-num-batched-tokens | util | prefill @ 24k | prefill @ 97k | KV pool |
|---:|---:|---:|---:|---:|
| 2,048 (clamped by MTP) | 0.80 | 1,142.3 | 1,104.3 | 416,163 |
| 16,384 | 0.85 | 1,166.0 | 1,128.7 | 512,619 |

Two things to note there. At 0.80 the larger chunk costs enough activation
memory to drop the KV pool below what one 262,144-token request needs, and the
engine refuses to start with a clear message naming 260,416 as the achievable
length. Raising utilisation to 0.85 fixes that and actually gives the largest
KV pool measured, but the kernel log then carries
`NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY]` during startup, so that
combination is not recommended as a default.

**The prefill lever that matters is prefix caching, and it is worth 114x of
TTFT.** Every benchmark above deliberately uses a unique prompt prefix so that
prefix caching cannot fake a cold prefill. For the real workload, a fixed
document or system prompt with a new question each turn, the cached path is
what you get. Hit rates are vLLM's own counters, not inferred from timing:

| 196k-token prompt | TTFT | Cached |
|---|---:|---:|
| cold, novel prefix | 178.72 s | 0.0% (0 / 196,022) |
| same document, new question | 2.33 s | 98.9% (193,856 / 196,010) |
| identical repeat | 1.56 s | 99.3% (194,688 / 196,010) |

So the 180-second TTFT is a first-turn cost, not a per-turn cost.
`--enable-prefix-caching` is on by default in the serve script and is the most
valuable flag in it for agent and document workloads.

Do not convert those into a prefill tok/s figure. Dividing the full prompt
length by the warm TTFT gives a number in the tens of thousands for tokens the
engine never computed. What it actually computes is the uncached tail, and that
runs at the same rate as everything else: 1,322 tokens in 1.56 s is 847 tok/s,
and 2,154 tokens in 2.33 s is 924 tok/s, both consistent with the ~1,100 tok/s
cold path. Nothing unusual is happening; the work is simply skipped.

The 0.0% hit rate on a novel prefix is also the check that validates every cold
number in this file: the unique-prefix methodology really does defeat the cache.

Cold prefill running this far below the box's arithmetic capability points at
the quantized-weight path rather than the GEMMs. The profiler section below
shows 81% of *decode* time inside EXL3 kernels; whether prefill shares that
profile has not been measured, so treat the cause as unconfirmed.

### Greedy probing does not work on this engine

`scripts/probe_greedy.py` cannot distinguish faithful speculation from
unfaithful speculation here, because the engine is not deterministic at
temperature 0 to begin with. Same server, same request, `temperature: 0`,
`seed: 0`:

```
"Summarize the causes of the 1929 stock market crash in one paragraph."
  -> 4 distinct outputs from 5 identical requests

"What is 17 * 23? Show the steps."
  -> 1 distinct output from 5
```

Running the probe spec-on against spec-on, same boot and same config, scores the
same 2 of 6 as spec-on against spec-off. The control is indistinguishable from
the test, so a text diff measures engine noise. Open-ended generations diverge;
tightly constrained ones do not, which is what you would expect if near-ties in
the logits are being resolved differently by varying reduction order. Use the
acceptance-rate metrics from `/metrics` instead, which are meaningful.

## Hardware

```text
GPU:     NVIDIA GB10
Memory:  128 GB unified (aarch64)
Engine:  vLLM --quantization exl3
```

One Spark. Tensor-parallel size 1 only -- see Known limitations.

## Model

Qwen3.8-Flash-Next is a `Qwen4ExpForConditionalGeneration` model: 48 layers
(36 linear attention, 12 full attention), 512 experts with top-10 routing,
hidden size 2560, one MTP (multi-token prediction) layer, a per-layer n-gram
(PLE) embedding table (320,001,536 rows x 160, K=5 packed), and a vision
tower.

## Prerequisites

- One DGX Spark (GB10, aarch64), NVMe with room for the ~80 GB pack.
- A vLLM aarch64 build containing the `Qwen4ExpForConditionalGeneration`
  model class. vLLM 0.29.0 carries the `Qwen4Exp` model classes in-tree
  and installs from PyPI as a prebuilt aarch64 wheel with
  `pip install vllm==0.29.0`, keeping `torch` 2.13.0+cu130. This removes
  the need to obtain a nightly build. Alternatively, for a nightly build,
  see the [GLM-5.3-Flash recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
  and the [DeepSeek-V4-Flash-Vision recipe](https://github.com/vcruz305/DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe)
  for how to build an aarch64 nightly and verify it on this box.
  The benchmark numbers in this README were measured on vLLM 0.28.1 nightly,
  verified with `0.28.1rc1.dev324+ga56654d6d` in the venv
  `/home/markus/venvs/vllm-vl`. Confirm your build with:
  ```bash
  pip show vllm
  python -c "import vllm; print(vllm.__version__)"
  ```
  vLLM 0.29.0 was checked on one DGX Spark at TP1 with this pack on 2026-09-10, in both
  serving configurations. The plugin binds the n-gram table at 30.40 GiB packed, corpus mean
  NLL against the exllamav3 reference of 0.9422 is 0.9398 with no draft head and 0.9426 with
  MTP k=2, greedy decode is 28.0 tok/s with no draft and 45.2 tok/s with MTP k=2 at mean
  acceptance 2.33, and prefill is about 970 tok/s on a 1,218-token prompt. The three vLLM
  patches in Quick start step 4 are required there exactly as they are on the nightly.
  Serving other models on 0.29.0 is untested.
- `exllamav3` **1.4.7, built from source**, with the aarch64 patch shipped in
  the plugin repo (`tools/patch_exllamav3_aarch64.py`) applied. The plugin
  imports the compiled `exllamav3_ext` module, so the pure-Python wheel alone
  is not enough.
- `torch`/CUDA 13.0 for aarch64 (matches the nightly vLLM build above).
- The `vllm-exl3` plugin, from `main`. Native-pack support merged there as
  unreleased `0.4.0`:
  ```bash
  pip install git+https://github.com/vcruz305/vllm-exl3@main
  ```
  The install must be from main at commit 6b26e5c (2026-09-08) or newer; main carries fixes for the fat-expert prefill bug (PR #5) that silently corrupts long prompts on older builds, and for the engine wedge on the vLLM nightly V2 runner (PR #7) that could hang on 33 to 144-token prefills and MTP evaluation runs.
  A tagged release will follow; until then, install from `main`.

## Quick start

### 1. Install the runtime

Install the vLLM nightly, `exllamav3` (from source, with the aarch64 patch),
and the `vllm-exl3` plugin per Prerequisites above, then confirm:

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
`config.json`, `model.safetensors.index.json`, and the tokenizer files.

### 3. Prepare the pack

The pack ships in turboderp's native format. Three one-time, idempotent
rewrites (each keeps a backup) make it loadable through vllm-exl3's native
Qwen4Exp path:

```bash
PACK_DIR=~/models/Qwen3.8-Flash-Next-EXL3 \
  PLUGIN_REPO=/path/to/vllm-exl3 \
  bash scripts/prepare_pack.sh
```

This runs, in order:

1. `tools/exl3_pack_tools/qwen_pack_scan.py <pack>` -- writes `pack_scan.json`.
2. `tools/exl3_pack_tools/qwen_pack_config.py <pack>` -- rewrites
   `config.json`'s `quantization_config` for the plugin (backs up
   `config.json.native`).
3. `tools/exl3_pack_tools/regenerate_safetensors_index.py <pack>` -- vLLM
   only loads files listed in `model.safetensors.index.json`, and the
   shipped index omits `ngram_embedding.safetensors` and the MTP patch file
   (backs up `.native`).

Optional GPU verification gates ship in `tools/verify_native_pack/`
(`test_ngram_embedding.py`, `mul1_check.py`, `pad_check.py`,
`compile_check.py`, each run as `python <script> <pack>`); run them with
`VERIFY=1 bash scripts/prepare_pack.sh`.

### 4. Patch vLLM

Until this architecture is upstreamed, three exact-anchor patches from the
plugin repo add the quant-config plumbing vLLM does not yet have for
`Qwen4ExpForConditionalGeneration`. Each is idempotent and keeps a `.orig`
backup:

```bash
VLLM_DIR="$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
python /path/to/vllm-exl3/tools/patch_vllm_qwen4_exp/patch_vllm_qwen4_ple.py "$VLLM_DIR"
python /path/to/vllm-exl3/tools/patch_vllm_qwen4_exp/patch_vllm_vision_split.py "$VLLM_DIR"
python /path/to/vllm-exl3/tools/patch_vllm_qwen4_exp/patch_vllm_mtp_lmhead.py "$VLLM_DIR"
```

| Script | What it fixes |
|---|---|
| `patch_vllm_qwen4_ple.py` | adds `quant_config` to `ParallelLMHead` and to the per-layer n-gram (PLE) embedding table |
| `patch_vllm_vision_split.py` | drops the pack's split vision attention q/k/v names -- vision qkv is served from the pack's bf16 fused copy |
| `patch_vllm_mtp_lmhead.py` | adds `quant_config` to the MTP draft's `ParallelLMHead` |

### 5. Serve

Defaults are the measured-best configuration: MTP k=3, bf16 recurrent state,
and the full 262,144 context. `MODEL_DIR` is auto-detected from either
`~/models/Qwen3.8-Flash-Next-exl3-3.05bpw` or `~/models/Qwen3.8-Flash-Next-EXL3`.

```bash
bash scripts/serve_one_spark_qwen.sh
```

For workloads that routinely exceed about 163k tokens, turn the draft off. Past
the acceptance cliff it is a net loss of roughly 21%, see
[Operating envelope](#operating-envelope-measured-2026-09-13-vllm-exl3-042):

```bash
SPEC_CONFIG=none bash scripts/serve_one_spark_qwen.sh
```

Other depths, noting that vLLM's `num_speculative_tokens` counts *drafted*
tokens, so k=3 drafts three and verifies up to four per step:

```bash
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}' bash scripts/serve_one_spark_qwen.sh
```

Defaults are `MAX_MODEL_LEN=262144`, `GPU_MEM_UTIL=0.80`, `MAX_NUM_SEQS=4`,
`MAMBA_SSM_DTYPE=bfloat16`, port 8899. Load takes about 9.5 minutes from cold
NVMe and roughly 2.5 minutes when the pack is still in page cache. See
`scripts/serve_one_spark_qwen.sh` for every override (`PORT`, `MAX_MODEL_LEN`,
`GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `SPEC_CONFIG`, `MAMBA_SSM_DTYPE`, `SERVED_NAME`,
`PROFILER_DIR`) and `VLLM_EXL3_NGRAM_KERNEL=ext|torch` (default `ext`) for the
n-gram embedding kernel.

### 6. Benchmark and probe

```bash
# Decode tok/s, excluding TTFT (copied unchanged from the GLM/DSV4 recipes)
python scripts/bench_v1.py --base-url http://127.0.0.1:8899/v1 --model Qwen3.8-Flash-Next

# Greedy-consistency probe: 4 fixed prompts, temperature 0, max_tokens 256
python scripts/probe_greedy.py no-draft no_draft.json http://127.0.0.1:8899
python scripts/probe_greedy.py mtp-k1   mtp_k1.json   http://127.0.0.1:8899
python scripts/probe_greedy.py mtp-k2   mtp_k2.json   http://127.0.0.1:8899
```

## Memory

- Model resident on device: **78.57 GiB** (the n-gram table alone is 30.4 GiB
  packed).
- Serving at `GPU_MEM_UTIL=0.80` and `MAX_MODEL_LEN=32768` leaves about
  **18.5 GiB MemAvailable**.
- **Do not go above 0.85** on this box; keep a memory watchdog running while
  experimenting with higher utilization or longer context.
- KV cache capacity: 385,570 tokens with no draft, 275,636 tokens with MTP
  k=1 (the draft's own KV consumes the difference).
- Load takes about 9.5 minutes from NVMe; the server is ready in 12-13
  minutes.

## Long context (262,144 tokens)

Measured 2026-09-07 with `MAX_MODEL_LEN=262144 GPU_MEM_UTIL=0.80 MAX_NUM_SEQS=2`, no draft:
These runs predate the fat-expert fix of 2026-09-08 (see Known limitations); the throughput numbers are unaffected by the fix, the quality of those long-prompt outputs was not.

| Item | Value |
|---|---|
| GPU KV cache | 546,503 tokens (vLLM: "Maximum concurrency for 262,144 tokens per request: 2.08x") |
| MemAvailable while serving | about 19 GiB |
| Weights load | 515 s (ready in about 11 minutes) |

The KV capacity is higher than at 32k with four sequences because fewer
sequences reserve less per-sequence state.

Long prompt (122,902 prompt tokens, one request, `max_tokens` 128):

| Config | KV cache | TTFT (prefill) | Decode |
|---|---|---|---|
| No draft | 546,503 tokens | 107.0 s (about 1,150 tok/s) | 26.2 tok/s |
| MTP k=2 | 402,630 tokens (1.54x at 262k) | 110.6 s | about 43 tok/s (128 tokens in 2.94 s), acceptance 2.49-2.65; probe coherent 4/4 |

Decode speed at 123k tokens of context is within a few percent of the short-prompt
numbers, so context length is not what limits decode on this model. Both runs
returned a coherent one-sentence answer about the prompt.

262,144 is the model's `max_position_embeddings`, so it is the context ceiling
without RoPE overrides. Raising utilization instead buys concurrency:

| Config | KV cache | MemAvailable while serving | Decode (short prompt) | 123k-token prompt |
|---|---|---|---|---|
| util 0.80, 2 sequences, no draft | 546,503 tokens (2.08x) | about 19 GiB | 26.2-27.5 tok/s | TTFT 107 s, decode 26.2 tok/s |
| util 0.85, 4 sequences, no draft | 759,773 tokens (2.90x) | 11.5 GiB | 27.1 tok/s | TTFT 107.5 s, decode 27.4 tok/s |

Keep a memory watchdog when running at 0.85; 11.5 GiB is the least headroom
measured in this recipe.

## Where the time goes (torch profiler, no draft, 32 decode steps)

Share of GPU kernel time, one request at 32k context:

| Kernel family | Share |
|---|---|
| EXL3 dense GEMV/GEMM (K5 attention, linear-attention, lm_head projections) | 47% |
| EXL3 fused MoE (`exl3_moe`, K3 experts) | 34% |
| bf16 GEMMs (hyper-connection mixers, router) | 5% |
| linear attention (gated delta rule) | 2% |
| norms | 1% |
| everything else (tiny elementwise launches) | 11% |

Decode is bound by trellis dequantization, not by memory bandwidth: the EXL3
kernels take roughly 3-5x the time the weight bytes would need at the GB10's
273 GB/s. That is why MTP helps (fewer dequant passes per emitted token) and
why k=3 does not beat k=2: vLLM's Qwen MTP replays the single draft layer and
the K5 lm_head once per extra draft token.

## Benchmark results (2026-09-08, one DGX Spark)

Setup: single DGX Spark (GB10, 121.7 GiB unified memory). EXL3 side: this recipe's pack (3.05 bpw), vllm-exl3 main 6b26e5c, MTP k=2, 65536 context, served with `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml`. GGUF side: vcruz305/Qwen3.8-Flash-Next-GGUF Q4_K_M with the BF16 PLE table file-backed, llama.cpp with the qwen4exp NextN/MTP draft head (upstream pull request 27836), `--spec-type draft-mtp --spec-draft-n-max 3`, same context. Both sides run their own speculative draft head, vendor sampling (thinking on, temperature 1.0, top_p 0.95, top_k 20), and one request at a time. Benchmark: sixcat-eval v0.5.1, vendor policy, 20 items per category, 120 items per side.

Quality (sixcat scores, percent). Each category holds 20 items, so one item is worth five points. The two runs land within one or two items of each other everywhere, which is a tie.

| Category | EXL3 | GGUF |
|---|---|---|
| Knowledge | 85.0 | 85.0 |
| Math | 100.0 | 100.0 |
| Truth | 80.0 | 85.0 |
| Instruct | 85.0 | 90.0 |
| Code | 90.0 | 85.0 |
| Tools | 80.0 | 90.0 |
| Overall | 86.7 | 89.2 |

The tools row needs a serving flag rather than a better model: this model emits XML-style calls, so `--tool-call-parser qwen3_xml` is required. With the default JSON parser the same run scores 15.0. Both columns score the first answer recorded for each of the 120 items, except the EXL3 tools row, which comes
from the corrected-parser pass because the original attempts measured the parser rather than the model. Note that
sixcat's printed overall is best-of-attempts: the EXL3 log carries 29 retried items and prints 90.0 on that basis,
where 17 of those retries are the unparseable tools answers and 12 are genuine failures, 6 of which flipped on a
second sample. The GGUF run was never offered a retry, so the table compares first attempts on both sides.

Latency and throughput. Twenty streamed requests per row, each with a unique prefix so prefix caching cannot flatter the numbers; prefill uses fresh random prompts.

| Metric | EXL3 | GGUF |
|---|---|---|
| Greedy TTFT p50 | 0.300 s | 0.383 s |
| Greedy TTFT p95 | 0.323 s | 0.574 s |
| Greedy decode p50 | 47.6 tok/s | 32.9 tok/s |
| Greedy decode p05 / p95 | 43.4 / 50.6 tok/s | 28.2 / 40.1 tok/s |
| Thinking TTFT p50 / p95 | 0.361 / 0.388 s | 0.439 / 0.488 s |
| Thinking decode p50 | 38.4 tok/s | 24.0 tok/s |
| Prefill, 1,221-token prompt | 902 tok/s | 439 tok/s |
| Prefill, 9,483-token prompt | 1,122 tok/s | 634 tok/s |
| Whole 120-item suite | 37.8 tok/s | 26.5 tok/s |

Suite throughput is each side's own 120-item pass, 105,254 completion tokens for EXL3 against 101,034 for the GGUF, timed over the 80 items the harness rates. That is 1.45x the greedy decode rate, 1.60x the thinking decode rate, 1.8 to 2.0x the prefill rate, and 1.44x the suite throughput.

Memory on one machine:

| Measure | EXL3 | GGUF |
|---|---|---|
| Weights | 79.96 GiB resident, n-gram table quantized and resident | 80.2 GiB backbone resident, 95.4 GiB BF16 PLE table paged from NVMe |
| KV cache | 11.04 GiB, 303,951 tokens at 64k context | not separately reported |
| Device allocation | 93.2 GiB idle, 94.1 GiB under load | 82.5 GiB |
| System memory in use | 102.4 GiB idle, 103.2 GiB under load | 86.9 GiB idle, 91.7 GiB under load, 95.7 GiB after the run |
| On-disk serving set | 79.4 GiB (30.4 GiB of it the n-gram table) | 175.6 GiB (95.4 GiB of it the BF16 embedding table) |

The GGUF's smaller resident figure is the 95.4 GiB embedding table living on disk rather than in memory, which is why its page cache and process size climb through a run and why its prefill is about half the speed. The EXL3 pack holds the whole model, embedding table included, in memory and still leaves room for 304k tokens of KV cache. Draft acceptance was 2.42 of 3 for EXL3 across the eval. An upstream llama.cpp change that reads PLE rows with explicit preads instead of demand paging (pull request 28136, reported to raise prefill from 300 to 750-800 tokens per second on this hardware) builds but aborts during decode when combined with the MTP draft head, so it is not usable yet.

NVFP4 does not fit on one Spark for this model: the weights alone come to 123.6 GiB before any KV cache, against 121.7 GiB of unified memory.

## Known limitations

- Fat-expert prefill bug (fixed 2026-09-08, vllm-exl3 PR #5). Before the fix, any prompt that routed more than 256 tokens to one expert produced wrong hidden states while short prompts looked normal: mean NLL over a 6000-token corpus was 4.21 through vLLM against 0.94 through exllamav3 on the same pack. With the fix the served model scores 0.941 to 0.944. The long-context decode speeds in this README were measured before the fix; the speeds stand, but any output quality claims for prompts beyond a few hundred tokens made before 2026-09-08 do not.
- Engine wedge on the vLLM nightly V2 runner (fixed 2026-09-08, vllm-exl3 PR #7). Root cause: exllamav3 dispatches dense calls by row count; up to 2 rows use non-cooperative GEMV, 3 to 144 rows use cooperative trellis GEMM (cudaLaunchCooperativeKernel with grid barriers), above 144 rows it reconstructs the weight and runs hgemm. On vLLM's nightly V2 runner the cooperative GEMM wedged the engine (EngineCore at 100% CPU, GPU idle power, no recovery) deterministically on 33 to 144-token prompts and on MTP k=2 evaluation after 30-60 minutes. The fix routes dense calls with 17 to 144 rows through the reconstruct path (exact); rows up to 16 keep the original dispatch. Validation: a 4-worker stress with random 15-220-word prompts that wedged the previous build in 161 seconds ran clean for 45 minutes (1085 requests, 77 tok/s aggregate), with the deterministic cases (72-token thinking, 128-token logprobs) running in about a second each. Decode is unchanged (44.6 tok/s greedy MTP k=2); TTFT on a 32-token prompt is about 0.25 s. Knobs: `VLLM_EXL3_RECONSTRUCT_MIN_ROWS` moves the threshold (default 17), `VLLM_EXL3_COOP_GEMM=1` restores the old dispatch for A/B runs.
- Tensor-parallel size 1 only -- the padded dense geometry and the n-gram
  table are not sharded for TP > 1.
- Vision attention q/k/v is served from the pack's bf16 fused copy, not the
  pack's split EXL3 tensors (see `patch_vllm_vision_split.py`).
- The plugin's native fused-MoE kernel is not used for this geometry; the
  `exllamav3` `exl3_moe` kernel is used instead.
- First boot needs the pack-preparation steps above; a raw download of the
  pack will not load as-is.
- The measured regime is per-request (single in-flight request) decode at
  32k context; there are no batching numbers yet.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `EXL3 n-gram table: N of 128 shards never loaded` | `model.safetensors.index.json` was not regenerated -- run `scripts/prepare_pack.sh` (step 3) |
| model loading consumes all system memory and dies with `CUDACachingAllocator ... memory mapping failed with OOM`, or the process is killed partway through loading | the per-layer n-gram embedding table was allocated dense at roughly 95 GiB rather than 30.4 GiB packed, because vLLM did not receive a quant_config for it. Run `patch_vllm_qwen4_ple.py` (Quick start step 4). Confirm the fix by looking for the line `EXL3 n-gram embedding ready: ... 30.40 GiB packed` in the server log during loading. |
| `ValueError: There is no module or parameter named 'lm_head.mul1'` | run `patch_vllm_qwen4_ple.py`, which also adds quant_config to the language model head. When serving with an MTP draft head, `patch_vllm_mtp_lmhead.py` is required as well. |
| `ValueError: There is no module or parameter named 'blocks.0.attn.k_proj'` | run `patch_vllm_vision_split.py`. The pack ships split vision attention q/k/v tensors alongside the fused bf16 copy, and the fused copy is the one served. |
| `ModuleNotFoundError: No module named 'exllamav3_ext'` | the compiled exllamav3 extension is missing from the active environment. Build exllamav3 1.4.7 from source with the aarch64 patch from the plugin repo. A pure-Python wheel is not sufficient, because the plugin imports the compiled module. |
| `ValueError: No available memory for the cache blocks` | `--gpu-memory-utilization` is set too low to hold the weights plus a KV cache. The pack is about 79 GiB, so 0.60 of 121.7 GiB leaves nothing for the cache. Use 0.80. |
| `EXL3 linear load shape mismatch ... (4304,) != (4352,)` | the `vllm-exl3` plugin is older than 0.4.0 |
| `torch.compile`: `Attempted to call function marked as skipped` | the `vllm-exl3` plugin is older than 0.4.0 |
| Evaluation harness scores the reasoning text as the answer | serve with `--reasoning-parser qwen3` so the `<think>` block is returned as `reasoning_content` (the serve script does this) |
| memory watchdog kills the process, or MemAvailable runs low | lower `GPU_MEM_UTIL` or `MAX_MODEL_LEN` |

These failures appear in a fixed order, each reached only after the previous is resolved. The first is the most misleading, because a missing patch step presents as an out-of-memory failure that reads like insufficient hardware. The `scripts/preflight.py` script checks all of these in a couple of seconds and is worth running before a serve attempt, since each model load takes about ten minutes.

## Related repositories

| Repo | Role |
|---|---|
| [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) | the pack this recipe serves |
| [vllm-exl3](https://github.com/vcruz305/vllm-exl3) | the EXL3 plugin: source, releases, issues, and the pack-prep / vLLM-patch tools this recipe calls |
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
`exl3.py` from their `overlay/exl3.py` -- see `vllm-exl3`'s own
`THIRD_PARTY_NOTICES.md`. MIT, Copyright (c) 2026 Mia's AI Lab.

## License

MIT for the scripts and notes in this repo (see `LICENSE`). Weights are
**not** redistributed here -- pull them from Hugging Face and respect
turboderp's pack license. vLLM, ExLlamaV3/EXL3, and `vllm-exl3` have their
own licenses.
