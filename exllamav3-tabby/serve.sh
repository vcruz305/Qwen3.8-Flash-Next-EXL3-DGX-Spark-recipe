#!/usr/bin/env bash
# OpenAI API using the vcruz305 TabbyAPI + exllamav3 forks.
#
#   bash exllamav3-tabby/serve.sh                    # concurrent, streamed PLE
#   PROFILE=single bash exllamav3-tabby/serve.sh     # one job; PLE placement estimated
#   PROFILE=single NGRAM_RAM=true bash exllamav3-tabby/serve.sh  # explicit A/B placement
#   DRY_RUN=1 bash exllamav3-tabby/serve.sh           # render preview, never replace live config
#
# Concurrent: batch 4, shared pool 1,048,576, per-request ceiling 262,144.
# Single: batch 1, shared pool 262,144; NGRAM_RAM=auto selects RAM only when
# the pack-header estimate plus KV and headroom fit MemAvailable.
# The pool is a capacity setting, not a claim that every pack/workload fits.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PROFILE="${PROFILE:-concurrent}"
case "$PROFILE" in
  concurrent) : "${MAX_BATCH_SIZE:=4}" "${CACHE_SIZE:=1048576}" "${NGRAM_RAM:=false}" ;;
  single)     : "${MAX_BATCH_SIZE:=1}" "${CACHE_SIZE:=262144}" "${NGRAM_RAM:=auto}" ;;
  *) die "PROFILE must be 'concurrent' or 'single', got '$PROFILE'" ;;
esac
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-262144}"
DRAFT_MODE="${DRAFT_MODE:-mtp}"
DRAFT_NUM_TOKENS="${DRAFT_NUM_TOKENS:-5}"
DYNAMIC_DRAFT="${DYNAMIC_DRAFT:-true}"
CHUNK_SIZE="${CHUNK_SIZE:-2048}"
SYSMEM_RECURRENT_CACHE="${SYSMEM_RECURRENT_CACHE:-4096}"
VISION="${VISION:-false}"
REASONING="${REASONING:-true}"
TOOL_FORMAT="${TOOL_FORMAT:-qwen3_5}"
DRY_RUN="${DRY_RUN:-0}"

positive_int() {
  local name="$1" value="${!1}"
  [[ "$value" =~ ^[0-9]{1,10}$ ]] || die "$name must be a positive integer"
  (( 10#$value > 0 && 10#$value <= 2147483647 )) || die "$name is outside the supported positive integer range"
  printf -v "$name" '%d' "$((10#$value))"
}
boolean() {
  [[ "${!1}" == true || "${!1}" == false ]] || die "$1 must be true or false"
}
for name in PORT MAX_SEQ_LEN CACHE_SIZE MAX_BATCH_SIZE DRAFT_NUM_TOKENS CHUNK_SIZE; do positive_int "$name"; done
[[ "$SYSMEM_RECURRENT_CACHE" =~ ^[0-9]{1,7}$ ]] || die "SYSMEM_RECURRENT_CACHE must be a nonnegative integer in MiB"
(( PORT <= 65535 )) || die "PORT must be between 1 and 65535"
(( MAX_SEQ_LEN <= 262144 )) || die "MAX_SEQ_LEN must not exceed the trained 262144-token window"
(( CACHE_SIZE % 256 == 0 )) || die "CACHE_SIZE must be a multiple of 256"
(( CACHE_SIZE >= MAX_SEQ_LEN )) || die "CACHE_SIZE ($CACHE_SIZE) is smaller than MAX_SEQ_LEN ($MAX_SEQ_LEN)"
(( CHUNK_SIZE % 256 == 0 )) || die "CHUNK_SIZE must be a multiple of 256"
[[ "$DRAFT_MODE" == mtp || "$DRAFT_MODE" == disabled ]] || die "DRAFT_MODE must be mtp or disabled"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "DRY_RUN must be 0 or 1"
for name in VISION REASONING DYNAMIC_DRAFT; do boolean "$name"; done
[[ "$NGRAM_RAM" == auto || "$NGRAM_RAM" == true || "$NGRAM_RAM" == false ]] || die "NGRAM_RAM must be auto, true or false"

case "$HOST" in
  127.0.0.1|localhost|::1) DISABLE_AUTH="${DISABLE_AUTH:-true}" ;;
  *) DISABLE_AUTH="${DISABLE_AUTH:-false}" ;;
esac
boolean DISABLE_AUTH
if [[ "$DISABLE_AUTH" == true && "$HOST" != 127.0.0.1 && "$HOST" != localhost && "$HOST" != ::1 ]]; then
  echo "warning: authentication is disabled on $HOST:$PORT" >&2
fi

MODEL_DIR="$(realpath -m "$MODEL_DIR")"
MODEL_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next-EXL3}"
[[ -n "$MODEL_NAME" && "$MODEL_NAME" != . && "$MODEL_NAME" != .. && "$MODEL_NAME" != */* && "$MODEL_NAME" != *\\* && "$MODEL_NAME" != *$'\n'* ]] \
  || die "SERVED_NAME must be one nonempty directory name, without slashes or newlines"
MODEL_PARENT="$STATE_DIR/models"

# Auto is conservative and resolved before rendering. For a reproducible A/B
# benchmark, pass NGRAM_RAM=true/false explicitly and keep the same setting.
if [[ "$NGRAM_RAM" == auto ]]; then
  NGRAM_RAM="$("$PYTHON_BIN" "$RECIPE_EXL3_DIR/tools/model_memory.py" --model "$MODEL_DIR" \
    --cache-size "$CACHE_SIZE" --draft-mode "$DRAFT_MODE" --slack-gib "${MEMORY_SLACK_GIB:-10}" --choose-ngram)"
  boolean NGRAM_RAM
fi
say "profile $PROFILE: $MAX_SEQ_LEN tokens per request, $CACHE_SIZE shared tokens, $MAX_BATCH_SIZE jobs," \
    "ngram_ram=$NGRAM_RAM, chunk=$CHUNK_SIZE, draft=$DRAFT_MODE/$DRAFT_NUM_TOKENS"
if (( CACHE_SIZE < MAX_SEQ_LEN * MAX_BATCH_SIZE )); then
  say "the pool holds fewer than $MAX_BATCH_SIZE full-length requests; shorter requests can still overlap"
fi

mkdir -p "$STATE_DIR"
CONFIG="$STATE_DIR/config.yml"
if [[ "$DRY_RUN" == 1 ]]; then
  CONFIG="$STATE_DIR/config.preview.yml"
else
  # Separate STATE_DIR values are required for simultaneous servers. Holding this
  # lock across exec prevents one launcher from rewriting another's live view/config.
  command -v flock >/dev/null || die "flock is required (util-linux)"
  exec 9>"$STATE_DIR/serve.lock"
  flock -n 9 || die "another recipe server holds $STATE_DIR/serve.lock; use a separate STATE_DIR for another instance"
  verify_runtime
  [[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR. Run setup.sh."
  verify_pack
  mkdir -p "$MODEL_PARENT"
  find "$MODEL_PARENT" -mindepth 1 -maxdepth 1 -type l -delete
  ln -sfn "$MODEL_DIR" "$MODEL_PARENT/$MODEL_NAME"
fi

export STATE_DIR HOST PORT DISABLE_AUTH MODEL_PARENT MODEL_NAME MAX_SEQ_LEN CACHE_SIZE MAX_BATCH_SIZE \
       NGRAM_RAM VISION DRAFT_MODE DRAFT_NUM_TOKENS DYNAMIC_DRAFT CHUNK_SIZE SYSMEM_RECURRENT_CACHE \
       REASONING TOOL_FORMAT PROFILE BIGCORES
RENDER_PY="$VENV/bin/python"; [[ -x "$RENDER_PY" ]] || RENDER_PY="$PYTHON_BIN"
"$RENDER_PY" - "$RECIPE_EXL3_DIR/tabby-config.yml" "$CONFIG" <<'PY'
import json, os, pathlib, string, sys
src, dst = map(pathlib.Path, sys.argv[1:3])
values = dict(os.environ)
# JSON strings are valid YAML scalars, including spaces, quotes and colon characters.
for name in ("HOST", "MODEL_PARENT", "MODEL_NAME", "TOOL_FORMAT", "DRAFT_MODE"):
    values[name] = json.dumps(values[name])
text = string.Template(src.read_text()).substitute(values)
temporary = dst.with_suffix(dst.suffix + ".tmp")
temporary.write_text(text)
temporary.replace(dst)
PY
say "config: $CONFIG"

CMD=("$VENV/bin/python" "$TABBY_DIR/main.py" --config "$CONFIG")
if [[ -n "$BIGCORES" ]]; then
  command -v taskset >/dev/null || die "BIGCORES is set but taskset is unavailable"
  CMD=(taskset -c "$BIGCORES" "${CMD[@]}")
fi
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'EXL3_INT8_GEMV=%q EXL3_MOE_COOP_WIDE=%q EXL3_GR_INT8=%q EXL3_MTP_HEAD_N=%q EXL3_DRAFT_CONFIDENCE=%q\n' \
    "$EXL3_INT8_GEMV" "$EXL3_MOE_COOP_WIDE" "$EXL3_GR_INT8" "$EXL3_MTP_HEAD_N" "$EXL3_DRAFT_CONFIDENCE"
  printf 'cd %q && ' "$TABBY_DIR"; printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

drop_pack_cache
check_memory "$CACHE_SIZE" "$NGRAM_RAM"
snapshot_runtime "$STATE_DIR/deployment.json" --config "$CONFIG" --model "$MODEL_DIR"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
say "TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD) on http://$HOST:$PORT/v1, model id: $MODEL_NAME"
cd "$TABBY_DIR"
exec "${CMD[@]}"
