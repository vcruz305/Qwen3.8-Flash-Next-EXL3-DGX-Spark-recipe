#!/usr/bin/env bash
TUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -uo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0 EXL3_MTP_HEAD_N=65536
cd "$HOME/exllamav3"
run() { # env assignments then label
  sleep 8; "$TUNING_DIR/../drop-model-cache.sh" >/dev/null 2>&1; sleep 6
  local log="$HOME/fill_$1.log"; shift
  env "$@" taskset -c 5-9,15-19 python "$HOME/ctxfill.py" > "$log" 2>&1
  grep -E "^RESULT|expected|JOB ERROR|Insufficient|Traceback|out of memory" "$log" | tail -3
}
echo "# $(date -Is) $(git rev-parse --short HEAD)"
run 32k        K=32
run 128k       K=128
run 240k       K=240
run 240k_q8    K=240 CQ=8,8
run 480k       K=480 CS=524288
run 480k_q8    K=480 CS=524288 CQ=8,8
echo "# done $(date -Is)"
