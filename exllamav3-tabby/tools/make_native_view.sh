#!/usr/bin/env bash
# A pack directory prepared for vLLM (vllm-plugin/prepare_pack.sh) carries the plugin's
# rewritten config.json and index; exllamav3 wants the originals. This builds a symlink
# view of the pack with the native config and index, without copying weights.
# usage: make_native_view.sh <prepared pack dir> <view dir> [ngram file to use instead]
# The third argument is for the 4.05 revision: vllm-plugin/rename_unsharded_ngram.py left the
# original as ngram_embedding.safetensors.native, and exllamav3 1.5.0 reads that directly.
set -euo pipefail
SRC="${1:?prepared pack dir}"; DST="${2:?view dir}"; NGRAM="${3:-}"
[[ -f "$SRC/config.json.native" ]] || { echo "no config.json.native in $SRC (run vllm-plugin/prepare_pack.sh first)"; exit 1; }
rm -rf "$DST"; mkdir -p "$DST"
for f in "$SRC"/*; do
  b=$(basename "$f")
  case "$b" in
    config.json|config.json.*|model.safetensors.index.json|model.safetensors.index.json.*|*.native) continue;;
  esac
  ln -s "$f" "$DST/$b"
done
ln -s "$SRC/config.json.native" "$DST/config.json"
ln -s "$SRC/model.safetensors.index.json.native" "$DST/model.safetensors.index.json"
if [[ -n "$NGRAM" ]]; then
  rm -f "$DST/ngram_embedding.safetensors"
  ln -s "$SRC/$NGRAM" "$DST/ngram_embedding.safetensors"
fi
echo "$DST: $(ls "$DST"/*.safetensors | wc -l) safetensors, $(python3 -c "import json;print(json.load(open('$DST/config.json'))['quantization_config']['bits'])") bpw"
