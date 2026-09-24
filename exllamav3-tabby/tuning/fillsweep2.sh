#!/usr/bin/env bash
TUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Disambiguate the 480k result: fp16 MISS vs q8 HIT on one sample each. Re-run both with a
# second needle position (SEED), plus 300k (just past the 262,144 trained window) both ways.
set -uo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0 EXL3_MTP_HEAD_N=65536
cd "$HOME/exllamav3"
run() {
  sleep 8; "$TUNING_DIR/../drop-model-cache.sh" >/dev/null 2>&1; sleep 6
  local log="$HOME/fill_$1.log"; shift
  env "$@" taskset -c 5-9,15-19 python "$HOME/ctxfill.py" > "$log" 2>&1
  grep -E "^RESULT|expected|JOB ERROR|Insufficient|Traceback|out of memory" "$log" | tail -2
}
echo "# $(date -Is)"
run 300k        K=300 CS=335872
run 300k_q8     K=300 CS=335872 CQ=8,8
run 480k_s2     K=480 CS=524288 SEED=2
run 480k_q8_s2  K=480 CS=524288 CQ=8,8 SEED=2
echo "# done $(date -Is)"
