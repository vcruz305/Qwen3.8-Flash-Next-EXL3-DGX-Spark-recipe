#!/usr/bin/env bash
# Benchmark one exllamav3 chat.py configuration on the Spark. Usage: ~/bench.sh <tag> [chat.py flags...]
# Drops model page cache first, runs a ~300-token generation, prints the tps line + load time.
set -uo pipefail
TAG="$1"; shift
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=${EXL3_INT8_GEMV:-0} EXL3_MOE_COOP_WIDE=${EXL3_MOE_COOP_WIDE:-1}
cd "$HOME/exllamav3"
~/drop-model-cache.sh >/dev/null
LOG="$HOME/bench_${TAG}.log"
PROMPT="${PROMPT:-Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each.}"
/usr/bin/time -f "WALL %e s  MAXRSS %M KB" \
  taskset -c 5-9,15-19 python examples/chat.py -m "$HOME/models/Qwen3.8-Flash-Next-EXL3" -mode qwen35 -cs 32768 -tps -basic -lm -no_think \
    -maxr 400 -prompt "$PROMPT" "$@" > "$LOG" 2>&1
echo "=== $TAG  flags: $* ==="
grep -E "Load time|Total size|Context:|WALL|Error|error|Traceback|Insufficient|Warning" "$LOG" | tr '\r' '\n' | grep -v '^$' | tail -8
