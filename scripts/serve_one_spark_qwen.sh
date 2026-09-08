#!/usr/bin/env bash
# Serve turboderp's Qwen3.8-Flash-Next EXL3 pack (native quantization_config,
# after scripts/prepare_pack.sh) from a local vLLM nightly + exllamav3 +
# vllm-exl3 installation on one DGX Spark (GB10).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
SPEC_CONFIG="${SPEC_CONFIG:-}"
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
# workaround for the vLLM nightly V2 runner wedge on 33 to 144-token prefills (plugin main e70a459 or newer); decode unchanged
export VLLM_EXL3_PREFILL_SYNC=256

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

if [[ -n "$SPEC_CONFIG" ]]; then
  ARGS+=(--speculative-config "$SPEC_CONFIG")
fi
if [[ -n "$PROFILER_DIR" ]]; then
  ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILER_DIR\"}")
fi

echo "VLLM_EXL3_NGRAM_KERNEL=$VLLM_EXL3_NGRAM_KERNEL GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS SPEC_CONFIG=${SPEC_CONFIG:-<none>}"
echo "vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"
