#!/usr/bin/env bash
# Interactive console chat on the tuned GB10 configuration (exllamav3's examples/chat.py).
# This is the configuration behind the README's single-stream numbers, measured 2026-09-17,
# chat.py cold, greedy, 400 new tokens: code 79 / DevOps 62 / prose 53 tok/s (no draft: 33).
# For an OpenAI-compatible API, use serve.sh instead.
#
#   bash exllamav3-tabby/chat.sh                       interactive
#   bash exllamav3-tabby/chat.sh -prompt "hello"       extra args pass through to chat.py
#   CS=32768 bash exllamav3-tabby/chat.sh              smaller cache for a quick session
#   NGRAM_RAM=false bash exllamav3-tabby/chat.sh       stream the n-gram table from disk
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# 262,144 is the model's trained window and the measured ceiling: needle retrieval fails past it
# (300k, 480k) at both KV precisions. KV is ~24 KB/token fp16 on this geometry, ~12 KB at 8-bit.
CS="${CS:-262144}"
# The published numbers ran with the n-gram table in RAM (EXL3_NGRAM_STREAM=0). false streams it
# from disk instead: ~30 GB less resident; needed for packs whose table does not fit (4.05 bpw).
NGRAM_RAM="${NGRAM_RAM:-true}"
if [[ "$NGRAM_RAM" == "true" ]]; then export EXL3_NGRAM_STREAM=0; else export EXL3_NGRAM_STREAM=1; fi

verify_runtime
verify_pack
drop_pack_cache
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
cd "$EXL3_SRC"
# -mtp: the model's own MTP head as the drafter. -ndt 5 -dds -dc 0.6: up to 5 drafts, stop early
# when the running confidence falls below 0.6 (prose collapses past draft position 1; code does not).
# -cq 8,8: 8-bit KV. The full-attention layers stream the whole KV every step (5.9 GB/step at 240k
# fp16), so halving it is faster at every depth with acceptance unchanged.
exec $(pin_cmd) "$VENV/bin/python" examples/chat.py -m "$MODEL_DIR" -mode qwen35 \
  -mtp -ndt 5 -dds -dc "$EXL3_DRAFT_CONFIDENCE" -cq 8,8 -cs "$CS" -tps "$@"
