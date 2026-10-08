# Shared defaults for the exllamav3 + TabbyAPI path. Sourced by setup.sh, serve.sh, chat.sh.
# Every value can be overridden from the environment before calling those scripts.

# The runtime: vcruz305/exllamav3 (upstream + aarch64 guards + GB10 decode kernels + mixed-K MoE
# + EXL3_DRAFT_CONFIDENCE). Stock turboderp exllamav3 runs this model but misses the GB10 work.
EXL3_REPO="${EXL3_REPO:-https://github.com/vcruz305/exllamav3.git}"
EXL3_REF="${EXL3_REF:-b5785675d3ecc4c09754477880d112b44952d83d}"
EXL3_MIN_VERSION="${EXL3_MIN_VERSION:-1.6.0.post1}"

# The API server: latest vcruz305/tabbyAPI main, including the validated Qwen tool fixes.
# TabbyAPI follows the fork branch; setup records the exact deployed commit.
TABBY_REPO="${TABBY_REPO:-https://github.com/vcruz305/tabbyAPI.git}"
TABBY_REF="${TABBY_REF:-main}"

# Where things live.
RECIPE_HOME="${RECIPE_HOME:-$HOME/qwen38-exl3}"
VENV="${VENV:-$RECIPE_HOME/venv}"
EXL3_SRC="${EXL3_SRC:-$RECIPE_HOME/exllamav3}"
TABBY_DIR="${TABBY_DIR:-$RECIPE_HOME/tabbyAPI}"
STATE_DIR="${STATE_DIR:-$RECIPE_HOME/state}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"

# Toolchain.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# GB10 kernel knobs, measured in the README ("The native engine, tuned for GB10").
export EXL3_INT8_GEMV="${EXL3_INT8_GEMV:-0}"          # fp16 GEMV beats the int8-activation one on GB10 (+3)
export EXL3_MOE_COOP_WIDE="${EXL3_MOE_COOP_WIDE:-1}"  # wide 128-col MoE coop tile on 48 SMs (+5)
export EXL3_GR_INT8="${EXL3_GR_INT8:-1}"              # int8 hyperconnection mixers (+5 code, +7 DevOps)
export EXL3_MTP_HEAD_N="${EXL3_MTP_HEAD_N:-65536}"    # pruned draft lm_head slice
export EXL3_DRAFT_CONFIDENCE="${EXL3_DRAFT_CONFIDENCE:-0.6}"  # dynamic-draft target (+8 prose vs 0.4)
export TORCH_CUDA_ARCH_LIST CUDA_HOME

# GB10 has 10 Cortex-X925 (big) + 10 A725 (little); pin to the big ones (+2). Empty disables.
if [[ -z "${BIGCORES+x}" ]]; then
  if [[ "$(uname -m)" == "aarch64" ]] && [[ "$(nproc --all 2>/dev/null || echo 0)" == "20" ]]; then
    BIGCORES="5-9,15-19"
  else
    BIGCORES=""
  fi
fi

RECIPE_EXL3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="$VENV/.qwen38-recipe-runtime"
BUILD_STATE="$VENV/.qwen38-recipe-runtime.json"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*" >&2; }

# Refuse to run on anything but the fork. Agents most often go wrong by picking up a stock
# exllamav3 wheel or a different venv, so check the actual imported module, not just a path.
verify_runtime() {
  local py="$VENV/bin/python"
  [[ -x "$py" ]] || die "no venv at $VENV. Run: bash exllamav3-tabby/setup.sh"
  "$py" - "$EXL3_MIN_VERSION" "$EXL3_SRC" <<'PY' || die "the exllamav3 in $VENV is not the vcruz305 fork runtime. Run: bash exllamav3-tabby/setup.sh"
import sys
from pathlib import Path
from packaging.version import Version
import triton  # gated-delta-net kernels; fail here, not mid-load
import exllamav3
from exllamav3.version import __version__ as v
expected_path = (Path(sys.argv[2]) / "exllamav3").resolve()
if Path(exllamav3.__file__).resolve().parent != expected_path:
    print(f"exllamav3 imported from {exllamav3.__file__}; expected {expected_path}", file=sys.stderr)
    sys.exit(1)
from exllamav3 import ext
e = ext.exllamav3_ext
need = ["gr_mix_int8", "exl3_moe_mixedk"]
missing = [n for n in need if not hasattr(e, n)]
if missing:
    print(f"exllamav3 {v} at {exllamav3.__file__} lacks fork kernels: {missing}", file=sys.stderr); sys.exit(1)
if Version(v.split("+")[0]) < Version(sys.argv[1]):
    print(f"exllamav3 {v} < {sys.argv[1]} (latest TabbyAPI refuses it)", file=sys.stderr); sys.exit(1)
import inspect
from exllamav3 import Generator
parameter = inspect.signature(Generator.__init__).parameters.get("draft_confidence")
if parameter is None or parameter.default is not None:
    print("exllamav3 fork predates EXL3_DRAFT_CONFIDENCE (vcruz305/exllamav3#11)", file=sys.stderr); sys.exit(1)
print(f"exllamav3 {v} (fork) at {exllamav3.__path__[0]}", file=sys.stderr)
PY
  if [[ -f "$MARKER" ]]; then
    local built; built="$(cat "$MARKER")"
    local actual; actual="$(git -C "$EXL3_SRC" rev-parse HEAD)"
    [[ "$built" == "$actual" ]] || die "runtime marker is $built but source checkout is $actual. Re-run setup.sh."
    if [[ "$EXL3_REF" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
      [[ "$built" == "$EXL3_REF"* ]] || die "runtime is $built, recipe pins $EXL3_REF. Re-run setup.sh."
    fi
  fi
  if [[ -f "$BUILD_STATE" ]]; then
    "$py" "$RECIPE_EXL3_DIR/tools/runtime_state.py" fingerprint \
      --engine "$EXL3_SRC" --cuda-home "$CUDA_HOME" --arch "$TORCH_CUDA_ARCH_LIST" \
      --compare "$BUILD_STATE" || die "runtime build fingerprint changed; re-run setup.sh before serving"
  else
    echo "warning: no ABI fingerprint for this older install; re-run setup.sh to rebuild and record it" >&2
  fi
}

# A pack that went through vllm-plugin/prepare_pack.sh carries a rewritten config.json that
# exllamav3 cannot read. Catch it before a 60-second load fails.
verify_pack() {
  [[ -f "$MODEL_DIR/config.json" ]] || die "no pack at $MODEL_DIR. Download it (README, step 2) or set MODEL_DIR."
  if grep -q 'derived_from_headers' "$MODEL_DIR/config.json" 2>/dev/null; then
    die "$MODEL_DIR was prepared for vLLM. Build a native view: bash exllamav3-tabby/tools/make_native_view.sh $MODEL_DIR ${MODEL_DIR}-native ; then MODEL_DIR=${MODEL_DIR}-native"
  fi
  ls "$MODEL_DIR"/model-*.safetensors >/dev/null 2>&1 || ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 \
    || die "no safetensors in $MODEL_DIR"
}

# GB10: cudaMemGetInfo reports MemFree, not MemAvailable, so page cache left over from a download
# or an earlier load makes the autosplit loader refuse the pack. Drop this pack's pages first.
drop_pack_cache() {
  "$RECIPE_EXL3_DIR/drop-model-cache.sh" "$MODEL_DIR" >/dev/null 2>&1 || true
}

pin_cmd() {
  if [[ -n "$BIGCORES" ]] && command -v taskset >/dev/null; then echo "taskset -c $BIGCORES"; fi
}

# Conservative, header-based fit advisory. It distinguishes n-gram tensors even when
# they share a shard with ordinary weights. Actual peak use still needs validation.
check_memory() {
  local cache="$1" ngram_ram="$2"
  "$PYTHON_BIN" "$RECIPE_EXL3_DIR/tools/model_memory.py" --model "$MODEL_DIR" \
    --cache-size "$cache" --ngram-ram "$ngram_ram" --draft-mode "${DRAFT_MODE:-mtp}" \
    --slack-gib "${MEMORY_SLACK_GIB:-10}" || true
}

snapshot_runtime() {
  local output="$1"; shift
  "$VENV/bin/python" "$RECIPE_EXL3_DIR/tools/runtime_state.py" snapshot \
    --engine "$EXL3_SRC" --server "$TABBY_DIR" --build-state "$BUILD_STATE" \
    --output "$output" "$@"
}
