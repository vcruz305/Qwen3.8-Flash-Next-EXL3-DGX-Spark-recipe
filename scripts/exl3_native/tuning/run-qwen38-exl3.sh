#!/usr/bin/env bash
# Launch Qwen3.8-Flash-Next EXL3 (3.05bpw) via ExLlamaV3 on the DGX Spark (GB10, aarch64).
# Usage: ~/run-qwen38-exl3.sh [extra chat.py args...]
#   e.g. ~/run-qwen38-exl3.sh -prompt "hello"
#        ~/run-qwen38-exl3.sh                       # interactive chat
#
# Tuned 2026-09-16 (400-token generations, -tps):
#   baseline (stream ngram from NVMe)       ~32 t/s
#   -mtp (default 4 draft tokens)           60-61 t/s code / 37 t/s prose
#   -mtp -ndt 3                             63 t/s code / 43 t/s prose
#   + INT8_GEMV=0 COOP_WIDE=1 bigcores -ndt 5  73 t/s code / 37 t/s prose   <-- default here
#   -ngr (ngram table in unified RAM)       34 t/s alone, 31 t/s with -mtp -- SLOWER, and +30 GB. Don't.
set -euo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1
# Tuned 2026-09-16 for GB10 (48 SMs): int8-activation GEMV path is slower here; force the wide
# 128-col MoE coop tile (default heuristic assumes datacenter Blackwell). +~8 t/s together.
export EXL3_INT8_GEMV=${EXL3_INT8_GEMV:-0} EXL3_MOE_COOP_WIDE=${EXL3_MOE_COOP_WIDE:-1}
# Pin to the 10 Cortex-X925 big cores (5-9,15-19): +~2 t/s vs mixed big/little scheduling.
BIGCORES=5-9,15-19
MODEL="$HOME/models/Qwen3.8-Flash-Next-EXL3"
cd "$HOME/exllamav3"
# Unified memory: file page cache counts against GPU-allocatable RAM and the autosplit loader
# refuses to load if it looks tight. Drop the model's cached pages first (no root needed).
"$HOME/drop-model-cache.sh" >/dev/null 2>&1 || true
# -mtp -ndt 3: speculative decoding with the model's own MTP head, 3 draft tokens (best measured).
# -mode qwen35: reasoning-aware ChatML (Qwen3.x family). -cs: KV cache tokens.
exec taskset -c $BIGCORES python examples/chat.py -m "$MODEL" -mode qwen35 -mtp -ndt 5 -cs 32768 -tps "$@"
