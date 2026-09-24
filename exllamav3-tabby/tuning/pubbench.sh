#!/usr/bin/env bash
TUNING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Publishable bench matrix: the launcher's exact chat.py path, cold load each run, greedy,
# 400 new tokens, three prompt classes x {stock tuned config, +patches}.
# Run on the box: bash ~/pubbench.sh > ~/pubbench.log 2>&1
set -uo pipefail
export PATH="$HOME/exllamav3/.venv/bin:/usr/local/cuda/bin:$PATH" TORCH_CUDA_ARCH_LIST=12.1
export EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0
cd "$HOME/exllamav3"
MODEL="$HOME/models/Qwen3.8-Flash-Next-EXL3"

declare -A PROMPTS
PROMPTS[code]="Write a Python function that parses an nginx access log line into a dict with fields ip, timestamp, method, path, status, bytes. Include a docstring, type hints, and a short usage example. Then explain each regex group in one bullet each."
PROMPTS[devops]="Explain, for a DevOps engineer, how Kubernetes horizontal pod autoscaling decides when to scale, including the formula it uses and two common pitfalls. Then give a complete example HPA YAML."
PROMPTS[prose]="Write a vivid 350-word short story about a lighthouse keeper on a remote island in Alaska who discovers something unexpected washed ashore after a storm."

run() {  # run <tag> <promptclass> [env KEY=VAL ...] -- [chat flags]
  local tag="$1" pc="$2"; shift 2
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift
  "$TUNING_DIR/../drop-model-cache.sh" >/dev/null 2>&1
  sleep 4
  local log="$HOME/pub_${tag}_${pc}.log"
  env "${envs[@]}" taskset -c 5-9,15-19 python examples/chat.py -m "$MODEL" -mode qwen35 -cs 32768 \
      -tps -basic -lm -no_think -topk 1 -maxr 400 -prompt "${PROMPTS[$pc]}" "$@" > "$log" 2>&1
  local tps
  tps=$(tr '\r' '\n' < "$log" | grep -oE '[0-9]+(\.[0-9]+)? tokens/second' | tail -1)
  local err
  err=$(grep -cE "Traceback|Error|Insufficient" "$log")
  printf '%-22s %-7s %-32s %s%s\n' "$tag" "$pc" "$*" "${tps:-NO_TPS}" "$([[ $err -gt 0 ]] && echo '  [ERRORS IN LOG]')"
}

echo "# $(date -Is)  host=$(hostname)  commit=$(git rev-parse --short HEAD)  dirty=$(git status --short | grep -cv '^??')"
echo "# config: EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1 EXL3_NGRAM_STREAM=0, big cores, greedy, 400 tok, cold load"
OFF=(EXL3_BATCH_VERIFY=0 EXL3_MTP_DEVICE_DRAFT=0 EXL3_EMBED_GPU=0 EXL3_MTP_HEAD_N=0)
ON=(EXL3_MTP_HEAD_N=65536)
for pc in code devops prose; do
  run "patches-off_ndt5"   $pc "${OFF[@]}" -- -mtp -ndt 5
  run "patches-off_dds0.6" $pc "${OFF[@]}" -- -mtp -ndt 5 -dds -dc 0.6
  run "patches-on_ndt5"    $pc "${ON[@]}"  -- -mtp -ndt 5
  run "patches-on_dds0.6"  $pc "${ON[@]}"  -- -mtp -ndt 5 -dds -dc 0.6
done
echo "# done $(date -Is)"
