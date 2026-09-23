#!/usr/bin/env bash
# LEGACY / NOT THE RECIPE RUNTIME. Builds *stock* exllamav3 1.5.0, only to reproduce the
# stock-vs-vLLM baseline tables in the README. For the recipe, run exllamav3-tabby/setup.sh.
echo "legacy: stock exllamav3 1.5.0 baseline build, not the recipe runtime (use exllamav3-tabby/setup.sh)" >&2
# exllamav3 1.5.0 in its own venv on one DGX Spark (GB10, aarch64), extension JIT-built
# for sm_121. Run once; the build takes about 15 minutes and lands in
# ~/.cache/torch_extensions. Needs nvcc (CUDA 13.0) and the vllm-exl3 checkout for its
# aarch64 patch tool.
set -euo pipefail
VENV="${VENV:-$HOME/venvs/exl3-150}"
PLUGIN_SRC="${PLUGIN_SRC:-$HOME/work/vllm-exl3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q torch==2.13.0 exllamav3==1.5.0 ninja safetensors

EXT="$("$VENV/bin/python" -c 'import exllamav3, os; print(os.path.dirname(exllamav3.__file__))')/exllamav3_ext"
[[ -d "$EXT.pristine" ]] || cp -r "$EXT" "$EXT.pristine"

# 1.5.0 still carries x86-only intrinsics (AVX MoE offload, tensor-parallel CPU reduce).
"$VENV/bin/python" "$PLUGIN_SRC/tools/patch_exllamav3_aarch64.py" "$EXT"
# Two exported CPU-offload symbols that 1.5.0 added and the patch's stub predates.
if ! grep -q exl3_moe_cpu_has_avx512_bw "$EXT/cpu/moe_mul1.cpp"; then
  cat >> "$EXT/cpu/moe_mul1.cpp" <<'CPP'
int64_t exl3_moe_cpu_pool_stress(int, int, int, int) { return 0; }
bool exl3_moe_cpu_has_avx512_bw() { return false; }
CPP
fi

# First import compiles the extension. ninja and nvcc must both be on PATH.
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}"
"$VENV/bin/python" -c 'from exllamav3 import ext; print("extension built:", ext.exllamav3_ext.__file__)'
