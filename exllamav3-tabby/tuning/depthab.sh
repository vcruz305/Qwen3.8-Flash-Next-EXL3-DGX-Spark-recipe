#!/usr/bin/env bash
TUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Quotable decode-at-depth A/B: 240k-token prompt, 400 new tokens of code, fp16 vs 8,8 KV,
# plus the same at 128k and a 4k baseline so the depth curve is one table. Waits for any
# running fill sweep first.
set -uo pipefail
while pgrep -f "fillsweep2.sh" >/dev/null; do sleep 20; done
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0 EXL3_MTP_HEAD_N=65536
cd "$HOME/exllamav3"
run() {
  sleep 8; "$TUNING_DIR/../drop-model-cache.sh" >/dev/null 2>&1; sleep 6
  local log="$HOME/depth_$1.log"; shift
  env TASK=code NEW=400 "$@" taskset -c 5-9,15-19 python "$HOME/ctxfill.py" > "$log" 2>&1
  grep -E "^RESULT|JOB ERROR|Insufficient|Traceback|out of memory" "$log" | tail -1
}
echo "# $(date -Is) $(git rev-parse --short HEAD)  400 new tokens, code task, greedy, -ndt 5 -dds 0.6"
run 4k_fp16    K=4
run 4k_q8      K=4   CQ=8,8
run 128k_fp16  K=128
run 128k_q8    K=128 CQ=8,8
run 240k_fp16  K=240
run 240k_q8    K=240 CQ=8,8
echo "# done $(date -Is)"
