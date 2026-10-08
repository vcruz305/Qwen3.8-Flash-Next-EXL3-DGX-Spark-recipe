#!/usr/bin/env bash
# Install the vcruz305 exllamav3 runtime from source and the latest vcruz305 TabbyAPI.
# Re-run to update. Local edits are refused before any checkout or package mutation.
# The build fingerprint includes the engine SHA, Torch/CUDA ABI, nvcc, compiler,
# Python and architecture; changing any of them rebuilds the native extension.
#
#   bash exllamav3-tabby/setup.sh
#   bash exllamav3-tabby/setup.sh --check
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
source "$RECIPE_EXL3_DIR/tools/git_helpers.sh"

case "${1:-}" in
  --check)
    [[ "$#" == 1 ]] || die "usage: setup.sh [--check]"
    verify_runtime
    [[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR"
    say "TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD) at $TABBY_DIR"
    "$VENV/bin/python" -m pip check
    exit 0 ;;
  "") ;;
  *) die "usage: setup.sh [--check]" ;;
esac

command -v git >/dev/null || die "git not found"
command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN not found"

# These are read-only and run for BOTH repositories before creating a venv,
# fetching refs, migrating a remote or installing dependencies.
preflight_repo "$EXL3_SRC"
preflight_repo "$TABBY_DIR"
[[ "$(realpath -m "$EXL3_SRC")" != "$(realpath -m "$TABBY_DIR")" ]] || die "EXL3_SRC and TABBY_DIR must be different directories"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc not found at $CUDA_HOME/bin/nvcc (set CUDA_HOME; CUDA 13.x or later for GB10)"
NVCC_VERSION="$("$CUDA_HOME/bin/nvcc" --version)"
NVCC_MAJOR="$("$PYTHON_BIN" -c 'import re,sys; m=re.search(r"release (\d+)\.", sys.stdin.read()); print(m[1] if m else "")' <<< "$NVCC_VERSION")"
[[ "$NVCC_MAJOR" =~ ^[0-9]+$ ]] && (( NVCC_MAJOR >= 13 )) || die "CUDA 13.x or later is required for the default GB10 sm_121 build"
export PATH="$CUDA_HOME/bin:$PATH"

# Keep an existing checkout's origin and local branches intact. When the recipe
# changes upstream/fork URL, a dedicated 'recipe' remote supplies the new refs.
# A conflicting recipe remote is a real ambiguity and is never overwritten.
mkdir -p "$RECIPE_HOME" "$STATE_DIR"
EXL3_REMOTE="$(sync_repo "$EXL3_SRC" "$EXL3_REPO")"
TABBY_REMOTE="$(sync_repo "$TABBY_DIR" "$TABBY_REPO")"
WANT="$(resolve_ref "$EXL3_SRC" "$EXL3_REMOTE" "$EXL3_REF")"
TABBY_WANT="$(resolve_ref "$TABBY_DIR" "$TABBY_REMOTE" "$TABBY_REF")"
# No force/reset/clean: detached checkouts preserve the user's branches and commits.
git -C "$EXL3_SRC" checkout -q --detach "$WANT"
git -C "$TABBY_DIR" checkout -q --detach "$TABBY_WANT"

if [[ ! -x "$VENV/bin/python" ]]; then
  say "creating venv $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
# BuildExtension discovers Ninja through PATH, not through Python imports.
# A Ninja wheel installed only in the venv otherwise permits a silent fallback
# to sequential distutils builds even when MAX_JOBS is set.
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
hash -r
"$PY" -m pip install -q --upgrade pip setuptools wheel ninja packaging
command -v ninja >/dev/null || die "ninja is not on PATH after installing the build dependencies"
ninja --version >/dev/null || die "ninja is present but cannot run"

# Preserve an already working CUDA-enabled Torch. Explicit TORCH_SPEC is applied
# when FORCE_TORCH_INSTALL=1, or automatically if the environment lacks CUDA Torch.
if [[ "${FORCE_TORCH_INSTALL:-0}" == "1" ]] \
   || ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  TORCH_SPEC="${TORCH_SPEC:-torch==2.13.0}"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  say "installing $TORCH_SPEC from $TORCH_INDEX_URL"
  "$PY" -m pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "torch has no CUDA"; print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'
# Use Torch's own detection as the final gate; never silently choose distutils.
"$PY" -c 'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability()'
say "Ninja $(ninja --version) at $(command -v ninja), build workers: ${MAX_JOBS:-$(nproc)}"

# Base TabbyAPI only. GPU extras install incompatible stock runtime wheels on
# some platforms. Settle all server dependencies before recording the build ABI.
(cd "$TABBY_DIR" && "$PY" -m pip install -q .)
"$PY" -c 'import triton' 2>/dev/null || { say "installing triton"; "$PY" -m pip install -q triton; }
"$PY" -c 'import uvloop' 2>/dev/null || { say "installing uvloop for aarch64"; "$PY" -m pip install -q uvloop; }

# Editable installation dependencies can change too (e.g. llguidance), so make
# sure the runtime requirements are present even if its CUDA build is current.
"$PY" -m pip install -q -r "$EXL3_SRC/requirements.txt"
# chat.sh uses upstream's optional console dependencies; they are not in the
# API server package or the base runtime requirements.
"$PY" -m pip install -q blessed prompt_toolkit pyperclip
DESIRED_BUILD="$STATE_DIR/runtime-build-desired.json"
"$PY" "$RECIPE_EXL3_DIR/tools/runtime_state.py" fingerprint \
  --engine "$EXL3_SRC" --cuda-home "$CUDA_HOME" --arch "$TORCH_CUDA_ARCH_LIST" > "$DESIRED_BUILD"
REBUILD=false
[[ -f "$BUILD_STATE" ]] && cmp -s "$DESIRED_BUILD" "$BUILD_STATE" || REBUILD=true
[[ "$(cat "$MARKER" 2>/dev/null || true)" == "$WANT" ]] || REBUILD=true
"$PY" - "$EXL3_SRC" <<'PY' >/dev/null 2>&1 || REBUILD=true
import sys
from pathlib import Path
import exllamav3, exllamav3_ext
assert Path(exllamav3.__file__).resolve().parent == (Path(sys.argv[1]) / "exllamav3").resolve()
PY

if [[ "$REBUILD" == true ]]; then
  say "building exllamav3 ${WANT:0:12} for sm_${TORCH_CUDA_ARCH_LIST/./}; commit or toolchain changed"
  "$PY" -m pip uninstall -y -q exllamav3 >/dev/null 2>&1 || true
  # Ninja does not necessarily notice an nvcc executable changing at the same
  # path. Clear only the repository's known, untracked build output for an ABI rebuild.
  [[ ! -L "$EXL3_SRC/build" ]] || die "$EXL3_SRC/build is a symlink; choose a fresh source directory"
  rm -rf -- "$EXL3_SRC/build"
  if [[ -f "$STATE_DIR/exllamav3-build.log" ]]; then
    cp "$STATE_DIR/exllamav3-build.log" "$STATE_DIR/exllamav3-build.previous.log"
  fi
  (cd "$EXL3_SRC" && MAX_JOBS="${MAX_JOBS:-$(nproc)}" "$PY" -m pip install --no-build-isolation -v -e . > "$STATE_DIR/exllamav3-build.log" 2>&1) \
    || { tail -40 "$STATE_DIR/exllamav3-build.log" >&2; die "exllamav3 build failed; full log: $STATE_DIR/exllamav3-build.log"; }
  printf '%s\n' "$WANT" > "$MARKER.tmp"
  mv "$MARKER.tmp" "$MARKER"
  cp "$DESIRED_BUILD" "$BUILD_STATE.tmp"
  mv "$BUILD_STATE.tmp" "$BUILD_STATE"
else
  say "exllamav3 source and toolchain fingerprint are unchanged at ${WANT:0:12}"
fi

verify_runtime
"$PY" -m pip check
"$PY" -m pip list --format=json > "$STATE_DIR/packages.json"
snapshot_runtime "$STATE_DIR/setup-snapshot.json"
cat >&2 <<EOF

Setup complete.
  runtime: $EXL3_REPO ${WANT:0:12} ($EXL3_SRC)
  server:  $TABBY_REPO ${TABBY_WANT:0:12} ($TABBY_DIR)
  venv:    $VENV
  record:  $STATE_DIR/setup-snapshot.json

Next: download the pack (README step 2), then
  bash exllamav3-tabby/serve.sh
EOF
