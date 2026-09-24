#!/usr/bin/env bash
TUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# int8 mixer A/B on the launcher path: chat.py, cold, greedy, 400 tok, all 3 prompt classes.
# Everything else at the launcher's tuned config; the 4 host patches at their current defaults.
set -uo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0 EXL3_MTP_HEAD_N=65536
cd "$HOME/exllamav3"
MODEL="$HOME/models/Qwen3.8-Flash-Next-EXL3"
declare -A PROMPTS
PROMPTS[code]="Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each."
PROMPTS[devops]="Explain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML."
PROMPTS[prose]="Write a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm."
run() { # <tag> <pc> <GR_INT8> [flags]
  local tag="$1" pc="$2" i8="$3"; shift 3
  sleep 8; "$TUNING_DIR/../drop-model-cache.sh" >/dev/null 2>&1; sleep 6
  local log="$HOME/i8_${tag}_${pc}.log"
  env EXL3_GR_INT8=$i8 taskset -c 5-9,15-19 python examples/chat.py -m "$MODEL" -mode qwen35 -cs 32768 \
     -tps -basic -lm -no_think -topk 1 -maxr 400 -prompt "${PROMPTS[$pc]}" "$@" > "$log" 2>&1
  printf '%-14s %-7s ' "$tag" "$pc"; tr '\r' '\n' < "$log" | grep -E "^Context:|Insufficient|Traceback" | tail -1 | sed -E 's/.*Generate: //'
}
echo "# $(date -Is) commit=$(git rev-parse --short HEAD)"
for pc in code devops prose; do
  run int8-off_dds $pc 0 -mtp -ndt 5 -dds -dc 0.6
  run int8-on_dds  $pc 1 -mtp -ndt 5 -dds -dc 0.6
done
# second pass on code+prose for variance
run int8-off_dds2 code 0 -mtp -ndt 5 -dds -dc 0.6
run int8-on_dds2  code 1 -mtp -ndt 5 -dds -dc 0.6
run int8-off_dds2 prose 0 -mtp -ndt 5 -dds -dc 0.6
run int8-on_dds2  prose 1 -mtp -ndt 5 -dds -dc 0.6
echo "# done $(date -Is)"
