#!/usr/bin/env bash
# BETA. Not the recommended API. Use exllamav3-tabby/serve.sh (latest TabbyAPI) instead.
#
# A minimal OpenAI /v1/chat/completions shim over the same fork runtime and GB10 knobs, kept for
# A/B work against TabbyAPI. Known limits: one generation at a time (a global lock, cache batch 1),
# hand-rolled Qwen prompt format instead of the pack's chat_template.jinja, tool definitions as
# prose, no /v1/completions. See exllamav3-tabby/beta/README.md.
#
# Measured 2026-09-20 on one GB10, vcruz305/exllamav3 785f206: greedy 400-token code job
# 79.5 wall tok/s, 83.8 engine, 74% draft acceptance (matches chat.py 79 / 73%). A Nous Hermes
# tool loop at ~80k prompt survived.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
echo "BETA: exllamav3-tabby/beta/serve_openai.sh is experimental; the supported API is exllamav3-tabby/serve.sh" >&2

CS="${CS:-262144}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_DIR")}"
# Same n-gram table placement as the measured numbers (in RAM).
export EXL3_NGRAM_STREAM="${EXL3_NGRAM_STREAM:-0}"

verify_runtime
verify_pack
drop_pack_cache
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export EXL3_ROOT="$EXL3_SRC" CS SERVED_NAME
cd "$EXL3_SRC"
exec $(pin_cmd) "$VENV/bin/python" "$HERE/serve_openai.py" \
  -m "$MODEL_DIR" -mtp -ndt 5 -dds -dc "$EXL3_DRAFT_CONFIDENCE" -cq 8,8 -cs "$CS" -topk 1 \
  --host "$HOST" --port "$PORT" "$@"
