# Qwen3.8-Flash-Next EXL3 on one NVIDIA DGX Spark

Serves [turboderp's Qwen3.8-Flash-Next EXL3 pack](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3)
(revision `3.05bpw_h5_ng5`, about 80 GB) on a single NVIDIA DGX Spark (GB10,
128 GB unified memory, aarch64) through a vLLM nightly and the
[vllm-exl3](https://github.com/vcruz305/vllm-exl3) plugin. Measured on
2026-09-07 on `cruz-spark`: **27.2-28.0 tok/s decode with no draft**, and
**33.8-36.4 tok/s with a single MTP draft token** (mean acceptance 1.86 of a
possible 2 per step), both single in-flight request, excluding TTFT. These
are preliminary numbers; the MTP k=2 and k=3 rows below are pending.

This recipe covers install, pack preparation (the pack as published needs
three one-time rewrites before vLLM's native loader will serve it), the
three runtime patches vLLM needs until this architecture is upstreamed,
serving with and without the MTP draft, and the benchmark/probe tools used
to produce the numbers below.

## Headline (measured 2026-09-07, cruz-spark)

| Config | Decode tok/s (4 runs) | TTFT | Notes |
|---|---|---|---|
| No draft | 27.97 / 27.56 / 27.16 / 27.35 | 0.185 s @ 128 tok | KV cache: 385,570 tokens |
| MTP k=1 | 36.37 / 33.80 / 35.26 / 35.28 | 0.199 s | mean acceptance 1.855 / 2; KV cache: 275,636 tokens |
| MTP k=2 | pending | pending | |
| MTP k=3 | pending | pending | |

Decode tok/s excludes TTFT (see `scripts/bench_v1.py`). Greedy output with
the MTP draft matched the no-draft text exactly on 1 of 4 fixed prompts and
diverged partway through -- with coherent text -- on the other 3. That is
expected of greedy-consistent speculation (the target either accepts or
rejects each draft token; it never emits a token the target itself would not
have chosen), not evidence of a bit-exact match end to end. A DeepSeek V4
Flash regression on this same plugin build passed.

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
- A vLLM **nightly** aarch64 build containing the `Qwen4ExpForConditionalGeneration`
  model class. Verified with `0.28.1rc1.dev324+ga56654d6d` in the venv
  `/home/markus/venvs/vllm-vl`; confirm your own build with:
  ```bash
  pip show vllm
  python -c "import vllm; print(vllm.__version__)"
  ```
  See the [GLM-5.3-Flash recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
  and the [DeepSeek-V4-Flash-Vision recipe](https://github.com/vcruz305/DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe)
  for how a vLLM nightly aarch64 build is obtained and verified on this box;
  use the same route and confirm the version string above once installed.
- `exllamav3` **1.4.7, built from source**, with the aarch64 patch shipped in
  the plugin repo (`tools/patch_exllamav3_aarch64.py`) applied. The plugin
  imports the compiled `exllamav3_ext` module, so the pure-Python wheel alone
  is not enough.
- `torch`/CUDA 13.0 for aarch64 (matches the nightly vLLM build above).
- The `vllm-exl3` plugin from branch `feat/native-turboderp-packs`:
  ```bash
  pip install git+https://github.com/vcruz305/vllm-exl3@feat/native-turboderp-packs
  ```
  A tagged release will follow; until then, install from that branch.

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

No draft:

```bash
MODEL_DIR=~/models/Qwen3.8-Flash-Next-EXL3 bash scripts/serve_one_spark_qwen.sh
```

MTP k=1 (the measured draft configuration):

```bash
MODEL_DIR=~/models/Qwen3.8-Flash-Next-EXL3 \
  SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
  bash scripts/serve_one_spark_qwen.sh
```

Both boot at `MAX_MODEL_LEN=32768`, `GPU_MEM_UTIL=0.80`, `MAX_NUM_SEQS=4`,
port 8899. Load takes about 9.5 minutes from NVMe; the server is ready in
12-13 minutes. See `scripts/serve_one_spark_qwen.sh` for every override
(`PORT`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `SPEC_CONFIG`,
`SERVED_NAME`, `PROFILER_DIR`) and `VLLM_EXL3_NGRAM_KERNEL=ext|torch`
(default `ext`) for the n-gram embedding kernel.

### 6. Benchmark and probe

```bash
# Decode tok/s, excluding TTFT (copied unchanged from the GLM/DSV4 recipes)
python scripts/bench_v1.py --base-url http://127.0.0.1:8899/v1 --model Qwen3.8-Flash-Next

# Greedy-consistency probe: 4 fixed prompts, temperature 0, max_tokens 256
python scripts/probe_greedy.py no-draft no_draft.json http://127.0.0.1:8899
python scripts/probe_greedy.py mtp-k1   mtp_k1.json   http://127.0.0.1:8899
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

## Known limitations

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
| `There is no module or parameter named 'blocks.0.attn.k_proj'` | `patch_vllm_vision_split.py` was not applied |
| `no module or parameter named 'lm_head.mul1' in Qwen4ExpMTP` | `patch_vllm_mtp_lmhead.py` was not applied |
| `EXL3 linear load shape mismatch ... (4304,) != (4352,)` | the `vllm-exl3` plugin is older than 0.4.0 |
| `torch.compile`: `Attempted to call function marked as skipped` | the `vllm-exl3` plugin is older than 0.4.0 |
| memory watchdog kills the process, or MemAvailable runs low | lower `GPU_MEM_UTIL` or `MAX_MODEL_LEN` |

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
