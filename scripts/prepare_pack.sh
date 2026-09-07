#!/usr/bin/env bash
# Prepare a turboderp Qwen3.8-Flash-Next EXL3 pack for vllm-exl3's native
# Qwen4Exp loader. Runs, in order, the three tools that rewrite the pack in
# place (each keeps a backup): scan, rewrite quantization_config, and
# regenerate the safetensors index (vLLM only loads files listed there, and
# the shipped index omits ngram_embedding.safetensors and the MTP
# hyper-connection-mixer patch file).
set -euo pipefail

PACK_DIR="${PACK_DIR:?set PACK_DIR to the downloaded turboderp/Qwen3.8-Flash-Next-exl3 revision directory}"
PLUGIN_REPO="${PLUGIN_REPO:?set PLUGIN_REPO to a checkout of vcruz305/vllm-exl3 (feat/native-turboderp-packs)}"

TOOLS="$PLUGIN_REPO/tools/exl3_pack_tools"
for f in qwen_pack_scan.py qwen_pack_config.py regenerate_safetensors_index.py; do
  if [[ ! -f "$TOOLS/$f" ]]; then
    echo "missing $TOOLS/$f -- is PLUGIN_REPO checked out at feat/native-turboderp-packs?" >&2
    exit 1
  fi
done
if [[ ! -f "$PACK_DIR/config.json" ]]; then
  echo "missing $PACK_DIR/config.json -- is PACK_DIR the pack revision directory?" >&2
  exit 1
fi

echo "1/3 scanning pack ..."
python3 "$TOOLS/qwen_pack_scan.py" "$PACK_DIR"

echo "2/3 rewriting config.json quantization_config ..."
python3 "$TOOLS/qwen_pack_config.py" "$PACK_DIR"

echo "3/3 regenerating model.safetensors.index.json ..."
python3 "$TOOLS/regenerate_safetensors_index.py" "$PACK_DIR"

echo "pack prepared: $PACK_DIR"

if [[ "${VERIFY:-0}" == "1" ]]; then
  VERIFY_TOOLS="$PLUGIN_REPO/tools/verify_native_pack"
  for f in test_ngram_embedding.py mul1_check.py pad_check.py compile_check.py; do
    echo "verify: $f"
    python3 "$VERIFY_TOOLS/$f" "$PACK_DIR"
  done
fi
