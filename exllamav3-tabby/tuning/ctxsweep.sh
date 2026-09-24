#!/usr/bin/env bash
TUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Context ceiling sweep on the launcher path: cold load at -cs N, generate 60 tokens from a
# short prompt (loads + allocs the cache; the long-prompt correctness check is separate).
# Reports load result, free memory after load, and t/s. Stops at the first OOM.
set -uo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0 EXL3_MTP_HEAD_N=65536
cd "$HOME/exllamav3"
MODEL="$HOME/models/Qwen3.8-Flash-Next-EXL3"
P="Write a Python function that parses an nginx access log line into a dict. Include a docstring."
echo "# $(date -Is) commit=$(git rev-parse --short HEAD)"
echo "# host free before: $(free -g | awk 'NR==2{print $7}') GiB avail"
run() { # <cs> [extra flags]
  local cs="$1"; shift
  sleep 8; "$TUNING_DIR/../drop-model-cache.sh" >/dev/null 2>&1; sleep 6
  local log="$HOME/ctx_${cs}$(echo "$*" | tr -d ' ' | tr -c 'a-zA-Z0-9_\n' '_').log"
  ( while true; do free -m | awk 'NR==2{print $7}'; sleep 2; done ) > "$log.mem" & local MP=$!
  taskset -c 5-9,15-19 python examples/chat.py -m "$MODEL" -mode qwen35 -cs "$cs" -tps -basic -lm -no_think -topk 1 \
     -maxr 60 -mtp -ndt 5 -dds -dc 0.6 -prompt "$P" "$@" > "$log" 2>&1
  kill $MP 2>/dev/null
  local res; res=$(tr '\r' '\n' < "$log" | grep -E "^Context:|Insufficient|Traceback|OutOfMemory|out of memory" | tail -1 | cut -c1-110)
  local minavail; minavail=$(sort -n "$log.mem" | head -1)
  printf '%-8s %-14s min_host_avail=%6s MB  %s\n' "$cs" "$*" "$minavail" "$res"
}
for cs in 32768 131072 262144; do run $cs; done
run 262144 -cq 8,8
run 524288
run 524288 -cq 8,8
run 1048576 -cq 8,8
echo "# done $(date -Is)"
