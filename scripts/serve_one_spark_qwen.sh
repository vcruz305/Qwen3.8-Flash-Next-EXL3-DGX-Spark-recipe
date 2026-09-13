#!/usr/bin/env bash
# Serve turboderp's Qwen3.8-Flash-Next EXL3 pack (native quantization_config,
# after scripts/prepare_pack.sh) from a local vLLM nightly + exllamav3 +
# vllm-exl3 installation on one DGX Spark (GB10).
set -euo pipefail

# turboderp's pack unpacks under its own revision name; accept either.
MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  for cand in "$HOME/models/Qwen3.8-Flash-Next-exl3-3.05bpw" "$HOME/models/Qwen3.8-Flash-Next-EXL3"; do
    [[ -f "$cand/config.json" ]] && { MODEL_DIR="$cand"; break; }
  done
  MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-exl3-3.05bpw}"
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
# The full 262,144 ceiling costs about 6 GB of KV: only 12 of 48 layers are
# full attention (full_attention_interval 4) with 2 KV heads at head_dim 256,
# so 24,576 bytes/token. Measured pool at 0.80 util is 416,163 tokens, which
# is 1.5x concurrency at max context. There is no reason to cap at 32k.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# MTP k=3 measured fastest: 50.1 tok/s at 4k against 47.4 at k=2 and 27.8 with
# no draft. k=4 wedges the engine after torch.compile and never allocates KV.
#
# IMPORTANT: draft acceptance collapses to exactly 0.000 at every position
# at 163,840 tokens (160 x 1024). Past that the draft
# head still drafts and the target rejects all of it, so you pay the draft cost
# for nothing: 21.9 tok/s with MTP against 26.6 with none. Set SPEC_CONFIG=none
# for workloads that routinely run past ~163k. See README "Operating envelope".
SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":3\}}"
# bf16 recurrent state for the 36 linear-attention layers. Worth about 8% more
# KV pool (385,422 -> 416,163 tokens); any decode gain is inside sample noise.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"
MTP_SAFE_CONTEXT=163840
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
PROFILER_DIR="${PROFILER_DIR:-}"

# Measured-safe ceiling on this box: at 0.80/32k the model (78.57 GiB
# resident, n-gram table 30.4 GiB packed) leaves ~18.5 GiB MemAvailable.
# Going above 0.85 is untested here and risks the memory watchdog. See
# README "Memory".
if awk -v u="$GPU_MEM_UTIL" 'BEGIN{exit !(u > 0.85)}'; then
  echo "GPU_MEM_UTIL=$GPU_MEM_UTIL is above the measured-safe ceiling (0.85) for this box." >&2
  exit 2
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "missing $MODEL_DIR/config.json -- download the pack (README step 2)" >&2
  exit 1
fi
if ! grep -q '"quant_method"[[:space:]]*:[[:space:]]*"exl3"' "$MODEL_DIR/config.json" 2>/dev/null; then
  echo "warning: $MODEL_DIR/config.json does not look like a prepared pack." >&2
  echo "         run scripts/prepare_pack.sh first (README step 3)." >&2
fi
if [[ ! -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
  echo "missing $MODEL_DIR/model.safetensors.index.json -- run scripts/prepare_pack.sh (README step 3)" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"

ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --quantization exl3
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching
  --trust-remote-code
  --reasoning-parser qwen3
)

if [[ -n "$MAMBA_SSM_DTYPE" && "$MAMBA_SSM_DTYPE" != "none" ]]; then
  ARGS+=(--mamba-ssm-cache-dtype "$MAMBA_SSM_DTYPE")
fi
if [[ -n "$SPEC_CONFIG" && "$SPEC_CONFIG" != "none" ]]; then
  ARGS+=(--speculative-config "$SPEC_CONFIG")
  if (( MAX_MODEL_LEN > MTP_SAFE_CONTEXT )); then
    echo "note: MAX_MODEL_LEN=$MAX_MODEL_LEN is above the measured MTP acceptance cliff" >&2
    echo "      ($MTP_SAFE_CONTEXT tokens, 160Ki). Speculation is a large win below it and a" >&2
    echo "      ~21% loss above it. Use SPEC_CONFIG=none for long-context workloads." >&2
  fi
fi
if [[ -n "$PROFILER_DIR" ]]; then
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\"}")
fi

echo "VLLM_EXL3_NGRAM_KERNEL=$VLLM_EXL3_NGRAM_KERNEL GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS MAMBA_SSM_DTYPE=$MAMBA_SSM_DTYPE SPEC_CONFIG=${SPEC_CONFIG:-<none>}"
echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
