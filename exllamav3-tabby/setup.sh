#!/usr/bin/env bash
# One-time setup for the primary path: vcruz305/exllamav3 (the runtime) built from source for
# sm_121, plus the latest TabbyAPI (the OpenAI-compatible server), in one venv.
#
#   bash exllamav3-tabby/setup.sh            # build or update everything
#   bash exllamav3-tabby/setup.sh --check    # only verify an existing install
#
# Idempotent: re-running updates TabbyAPI to latest main and rebuilds exllamav3 only when the
# pinned fork commit changed. The CUDA extension build takes ~15 minutes on a DGX Spark.
# Overrides: see env.sh (RECIPE_HOME, VENV, EXL3_REF, TABBY_REF, CUDA_HOME, MODEL_DIR, ...).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ "${1:-}" == "--check" ]]; then
  verify_runtime
  [[ -f "$TABBY_DIR/main.py" ]] || die "no TabbyAPI at $TABBY_DIR"
  say "TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD) at $TABBY_DIR"
  exit 0
fi

command -v git >/dev/null || die "git not found"
command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN not found"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || die "nvcc not found at $CUDA_HOME/bin/nvcc (set CUDA_HOME; CUDA 13.x for GB10)"
export PATH="$CUDA_HOME/bin:$PATH"

mkdir -p "$RECIPE_HOME" "$STATE_DIR"

# 1. venv
if [[ ! -x "$VENV/bin/python" ]]; then
  say "creating venv $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip setuptools wheel ninja packaging

# 2. torch with CUDA. Keep an existing CUDA-enabled torch; otherwise install the aarch64/cu130
#    wheel (TORCH_SPEC / TORCH_INDEX_URL override, e.g. for x86_64 or a different CUDA).
if ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  TORCH_SPEC="${TORCH_SPEC:-torch==2.13.0}"
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  say "installing $TORCH_SPEC from $TORCH_INDEX_URL"
  "$PY" -m pip install -q "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
fi
"$PY" -c 'import torch; assert torch.cuda.is_available(), "torch has no CUDA"; print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'
# The fork's gated-delta-net (linear attention) kernels are Triton. TabbyAPI's cu12/cu13 extras only
# list triton for x86_64, and the torch aarch64 wheel does not always bring it.
"$PY" -c 'import triton' 2>/dev/null || { say "installing triton"; "$PY" -m pip install -q triton; }

# 3. exllamav3 fork at the pinned commit. Uninstall any stock wheel first: a PyPI/TabbyAPI
#    exllamav3 left in the venv shadows the fork and is the #1 cause of "wrong runtime".
if [[ ! -d "$EXL3_SRC/.git" ]]; then
  say "cloning $EXL3_REPO"
  git clone "$EXL3_REPO" "$EXL3_SRC"
fi
git -C "$EXL3_SRC" fetch -q origin
BUILT="$(cat "$MARKER" 2>/dev/null || true)"
WANT="$(git -C "$EXL3_SRC" rev-parse "$EXL3_REF^{commit}")"
if [[ "$BUILT" != "$WANT" ]] || ! "$PY" -c 'import exllamav3_ext' 2>/dev/null; then
  say "building exllamav3 fork at ${WANT:0:12} for sm_${TORCH_CUDA_ARCH_LIST/./} (about 15 minutes)"
  git -C "$EXL3_SRC" checkout -q --detach "$WANT"
  "$PY" -m pip uninstall -y -q exllamav3 >/dev/null 2>&1 || true
  (cd "$EXL3_SRC" && MAX_JOBS="${MAX_JOBS:-$(nproc)}" "$PY" -m pip install --no-build-isolation -v -e . > "$STATE_DIR/exllamav3-build.log" 2>&1) \
    || { tail -40 "$STATE_DIR/exllamav3-build.log" >&2; die "exllamav3 build failed; full log: $STATE_DIR/exllamav3-build.log"; }
  echo "$WANT" > "$MARKER"
else
  say "exllamav3 fork already built at ${WANT:0:12}"
fi

# 4. TabbyAPI, latest main. Installed without its GPU extras: those pull x86_64/Windows
#    exllamav3 and torch wheels, and the runtime here is the fork built above.
if [[ ! -d "$TABBY_DIR/.git" ]]; then
  say "cloning $TABBY_REPO"
  git clone "$TABBY_REPO" "$TABBY_DIR"
fi
git -C "$TABBY_DIR" fetch -q origin
if [[ -n "$(git -C "$TABBY_DIR" status --porcelain --untracked-files=no)" ]]; then
  echo "warning: $TABBY_DIR has local changes; leaving its checkout as is" >&2
else
  git -C "$TABBY_DIR" checkout -q --detach "origin/$TABBY_REF" 2>/dev/null || git -C "$TABBY_DIR" checkout -q --detach "$TABBY_REF"
fi
say "TabbyAPI at $(git -C "$TABBY_DIR" log -1 --format='%h %cs %s' | cut -c1-90)"
(cd "$TABBY_DIR" && "$PY" -m pip install -q .)
# TabbyAPI's own pyproject can drag a stock exllamav3 back in on some platforms; make sure not.
if "$PY" -m pip show exllamav3 2>/dev/null | grep -q "^Location:.*site-packages$" \
   && ! "$PY" -m pip show exllamav3 2>/dev/null | grep -q "Editable project location"; then
  say "a stock exllamav3 wheel was pulled in; reinstalling the fork"
  "$PY" -m pip uninstall -y -q exllamav3
  (cd "$EXL3_SRC" && MAX_JOBS="${MAX_JOBS:-$(nproc)}" "$PY" -m pip install --no-build-isolation -e . >> "$STATE_DIR/exllamav3-build.log" 2>&1)
fi

# 5. Verify.
verify_runtime
cat >&2 <<EOF

Setup complete.
  runtime:  vcruz305/exllamav3 ${WANT:0:12}  ($EXL3_SRC)
  server:   TabbyAPI $(git -C "$TABBY_DIR" rev-parse --short HEAD)  ($TABBY_DIR)
  venv:     $VENV

Next: download the pack (README step 2), then
  bash exllamav3-tabby/serve.sh
EOF
