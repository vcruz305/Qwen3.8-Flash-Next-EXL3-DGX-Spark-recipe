#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next EXL3 as an OpenAI-compatible API: latest TabbyAPI on the
# vcruz305/exllamav3 fork runtime. This is the recommended way to run this recipe.
#
#   bash exllamav3-tabby/serve.sh                     # PROFILE=concurrent (default)
#   PROFILE=single bash exllamav3-tabby/serve.sh      # one user, fastest single stream
#   curl http://127.0.0.1:8899/v1/models
#
# Profiles (every value can still be overridden individually):
#   concurrent  MAX_BATCH_SIZE=4  CACHE_SIZE=1048576  NGRAM_RAM=false
#               4 concurrent jobs, each up to the full 262,144 context at the same time.
#               The n-gram table streams from disk, which frees ~30 GB for KV and batch.
#   single      MAX_BATCH_SIZE=1  CACHE_SIZE=262144   NGRAM_RAM=true
#               The configuration the README's single-stream numbers were measured on.
#
# Other knobs:
#   MODEL_DIR=~/models/Qwen3.8-Flash-Next-EXL3   HOST=127.0.0.1   PORT=8899
#   MAX_SEQ_LEN=262144      per-request context ceiling (the model's trained window)
#   CACHE_SIZE              ONE shared KV pool for all concurrent requests (8-bit: ~13 KB/token)
#   DRAFT_NUM_TOKENS=5      MTP draft ceiling (dynamic; EXL3_DRAFT_CONFIDENCE=0.6)
#   SERVED_NAME=Qwen3.8-Flash-Next-EXL3   model id in /v1/models and requests
#   VISION=false            load the vision tower
#   DISABLE_AUTH            default true on loopback, false otherwise (TabbyAPI api_tokens.yml)
#   DRY_RUN=1               render the config and print the command without starting
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
mkdir -p "$STATE_DIR"

PROFILE="${PROFILE:-concurrent}"
case "$PROFILE" in
  concurrent) : "${MAX_BATCH_SIZE:=4}" "${CACHE_SIZE:=1048576}" "${NGRAM_RAM:=false}" ;;
  single)     : "${MAX_BATCH_SIZE:=1}" "${CACHE_SIZE:=262144}"  "${NGRAM_RAM:=true}" ;;
  *) die "PROFILE must be 'concurrent' or 'single', got '$PROFILE'" ;;
esac
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-262144}"
DRAFT_NUM_TOKENS="${DRAFT_NUM_TOKENS:-5}"
VISION="${VISION:-false}"

# Sanity on the sizing, because this is where "the context is tiny" reports come from.
(( CACHE_SIZE % 256 == 0 )) || die "CACHE_SIZE must be a multiple of 256"
(( CACHE_SIZE >= MAX_SEQ_LEN )) || die "CACHE_SIZE ($CACHE_SIZE) < MAX_SEQ_LEN ($MAX_SEQ_LEN): one full-length request would not fit"
(( MAX_SEQ_LEN <= 262144 )) || echo "warning: MAX_SEQ_LEN > 262144 is past the trained window; retrieval fails there (README)" >&2
say "profile $PROFILE: ${MAX_SEQ_LEN} tokens per request, ${CACHE_SIZE}-token shared pool" \
    "($(( CACHE_SIZE / MAX_SEQ_LEN )) full-length at once), ${MAX_BATCH_SIZE} concurrent jobs, n-gram table in RAM: $NGRAM_RAM"

case "$HOST" in
  127.0.0.1|localhost|::1) DISABLE_AUTH="${DISABLE_AUTH:-true}" ;;
  *) DISABLE_AUTH="${DISABLE_AUTH:-false}"
     [[ "$DISABLE_AUTH" == "true" ]] && echo "warning: auth disabled on $HOST; anyone who can reach port $PORT can use the API" >&2 ;;
esac

# Serve from an isolated view directory holding one symlink. TabbyAPI's /v1/models lists every
# directory under model_dir for admin callers (everyone, with auth disabled), so pointing it at
# ~/models would advertise unrelated models and clients that take the first id pick the wrong one.
MODEL_DIR="$(cd "$MODEL_DIR" 2>/dev/null && pwd || echo "$MODEL_DIR")"
MODEL_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next-EXL3}"
MODEL_PARENT="$STATE_DIR/models"
mkdir -p "$MODEL_PARENT"
find "$MODEL_PARENT" -mindepth 1 -maxdepth 1 -type l -delete
ln -sfn "$MODEL_DIR" "$MODEL_PARENT/$MODEL_NAME"
export STATE_DIR HOST PORT DISABLE_AUTH MODEL_PARENT MODEL_NAME MAX_SEQ_LEN CACHE_SIZE MAX_BATCH_SIZE \
       NGRAM_RAM VISION DRAFT_NUM_TOKENS

CONFIG="$STATE_DIR/config.yml"
RENDER_PY="$VENV/bin/python"; [[ -x "$RENDER_PY" ]] || RENDER_PY="$PYTHON_BIN"
"$RENDER_PY" - "$RECIPE_EXL3_DIR/tabby-config.yml" "$CONFIG" <<'PY'
import os, string, sys
src, dst = sys.argv[1], sys.argv[2]
text = string.Template(open(src).read()).substitute(os.environ)
open(dst, "w").write(text)
PY
say "config: $CONFIG"

CMD=( $(pin_cmd) "$VENV/bin/python" "$TABBY_DIR/main.py" --config "$CONFIG" )
if [[ -n "${DRY_RUN:-}" ]]; then
  echo "EXL3_INT8_GEMV=$EXL3_INT8_GEMV EXL3_MOE_COOP_WIDE=$EXL3_MOE_COOP_WIDE EXL3_GR_INT8=$EXL3_GR_INT8 EXL3_MTP_HEAD_N=$EXL3_MTP_HEAD_N EXL3_DRAFT_CONFIDENCE=$EXL3_DRAFT_CONFIDENCE"
  echo "cd $TABBY_DIR && ${CMD[*]}"
  exit 0
fi

verify_runtime
[[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR. Run: bash exllamav3-tabby/setup.sh"
verify_pack
drop_pack_cache
check_memory "$CACHE_SIZE" "$NGRAM_RAM"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
say "TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD) on http://$HOST:$PORT/v1, model id: $MODEL_NAME"
# TabbyAPI resolves templates/, sampler_overrides/ and api_tokens.yml relative to its cwd.
cd "$TABBY_DIR"
exec "${CMD[@]}"
