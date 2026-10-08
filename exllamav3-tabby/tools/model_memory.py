#!/usr/bin/env python3
"""Conservative Qwen pack memory estimate from safetensors headers, without loading weights.

This is a startup advisory, not a guarantee of a maximum working context. CUDA
allocations, recurrent slots, kernel workspaces and the OS need measured headroom.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import sys

GIB = 1024 ** 3


def pack_bytes(directory):
    ngram = other = 0
    warnings = []
    files = list(Path(directory).glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors in {directory}; cannot estimate memory")
    # Deduplicate physical files, including native-view symlinks.
    seen = set()
    for file in files:
        target = file.resolve()
        info = target.stat()
        identity = (info.st_dev, info.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        try:
            with target.open("rb") as handle:
                raw = handle.read(8)
                if len(raw) != 8:
                    raise ValueError("missing safetensors header length")
                length = struct.unpack("<Q", raw)[0]
                if not 2 <= length <= min(info.st_size - 8, 100 * 1024 ** 2):
                    raise ValueError("invalid safetensors header size")
                header = json.loads(handle.read(length))
            file_ngram = file_other = 0
            for name, tensor in header.items():
                if name == "__metadata__":
                    continue
                begin, end = tensor["data_offsets"]
                if begin < 0 or end < begin or end > info.st_size - 8 - length:
                    raise ValueError("invalid tensor offsets")
                size = end - begin
                if ("ngram" in name.lower() or "ple_embedding" in name.lower()
                        or "ngram" in file.name.lower()):
                    file_ngram += size
                else:
                    file_other += size
            ngram += file_ngram
            other += file_other
        except (KeyError, TypeError, ValueError, OSError) as exc:
            # A file-size fallback stays conservative; explain why it was needed.
            if "ngram" in file.name.lower():
                ngram += info.st_size
            else:
                other += info.st_size
            warnings.append(f"{file.name}: header estimate unavailable ({exc}); used file size")
    return {"ngram_bytes": ngram, "other_weight_bytes": other,
            "weight_bytes": ngram + other, "unique_files": len(seen), "warnings": warnings}


def available_bytes():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "MemFree"}:
            values[key] = int(value.split()[0]) * 1024
    return values


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def cache_bytes(config, cache_size, draft_mode):
    """K8/V8 backing storage, including exact scale tensors and QSA side planes.

    Matches CacheLayer_quant and QSAPlanes. This describes GQA/QSA storage;
    it does not estimate other attention layouts such as MLA or sliding windows.
    """
    total_layers = positive_int(config.get("num_hidden_layers", 48), "num_hidden_layers")
    layer_types = config.get("layer_types")
    if layer_types:
        if not isinstance(layer_types, list) or len(layer_types) != total_layers:
            raise ValueError("layer_types must describe every hidden layer")
        full_layers = sum(x == "full_attention" for x in layer_types)
    else:
        # An ordinary GQA model without a hybrid interval uses attention in every layer.
        interval = positive_int(config.get("full_attention_interval", 1), "full_attention_interval")
        full_layers = total_layers // interval
    kv_heads = positive_int(config.get("num_key_value_heads", 2), "num_key_value_heads")
    head_dim = positive_int(config.get("head_dim", 256), "head_dim")
    token_dim = kv_heads * head_dim
    if token_dim % 32:
        raise ValueError("K8/V8 cache token dimension must be divisible by 32")
    draft_layers = (positive_int(config.get("mtp_num_hidden_layers", 1), "mtp_num_hidden_layers")
                    if draft_mode == "mtp" else 0)
    layers = full_layers + draft_layers
    # Tabby rounds requested capacities up to its 256-token page size.
    tokens = ((positive_int(cache_size, "cache_size") + 255) // 256) * 256
    data_bytes = tokens * layers * token_dim * 2  # one byte each for K and V
    scale_bytes = tokens * layers * (token_dim // 32) * 2 * 2  # two FP16 scales per group
    qsa = any(key in config for key in ("indexer_head_dim", "indexer_compress_ratio", "indexer_budget"))
    index_dim = ratio = 0
    raw_bytes = pooled_bytes = 0
    if qsa:
        index_dim = positive_int(config.get("indexer_head_dim"), "indexer_head_dim")
        ratio = positive_int(config.get("indexer_compress_ratio", 4), "indexer_compress_ratio")
        if 256 % ratio or config.get("indexer_kv_heads", 1) != 1:
            raise ValueError("QSA requires a page-dividing compression ratio and one indexer KV head")
        # QSAPlanes keeps raw and pooled indexer keys in FP16 even for quantized K/V.
        raw_bytes = tokens * layers * index_dim * 2
        pooled_bytes = (tokens // ratio) * layers * index_dim * 2
    return {
        "kv_data_bytes": data_bytes, "kv_scale_bytes": scale_bytes,
        "qsa_raw_key_bytes": raw_bytes, "qsa_pooled_key_bytes": pooled_bytes,
        "qsa_index_bytes": raw_bytes + pooled_bytes,
        "kv_estimate_bytes": data_bytes + scale_bytes + raw_bytes + pooled_bytes,
        "assumptions": {
            "cache_mode": "8,8", "cache_layout": "qsa" if qsa else "gqa",
            "requested_cache_tokens": cache_size, "allocated_cache_tokens": tokens,
            "full_attention_layers": full_layers, "mtp_layers": draft_layers,
            "kv_heads": kv_heads, "head_dim": head_dim,
            "kv_scale_group_size": 32, "kv_scale_dtype": "fp16",
            "qsa_layers": layers if qsa else 0, "qsa_plane_dtype": "fp16" if qsa else None,
            "qsa_index_head_dim": index_dim, "qsa_compress_ratio": ratio,
        },
    }


def estimate(args):
    pack = pack_bytes(args.model)
    config = json.loads((Path(args.model) / "config.json").read_text())
    config = config.get("text_config") or config
    cache = cache_bytes(config, args.cache_size, args.draft_mode)
    slack = int(args.slack_gib * GIB)
    memory = available_bytes()
    base = pack["other_weight_bytes"] + cache["kv_estimate_bytes"] + slack
    return {
        **pack, **cache, "slack_bytes": slack,
        "need_stream_bytes": base, "need_ram_bytes": base + pack["ngram_bytes"],
        "memory": memory,
        "ngram_ram_auto": base + pack["ngram_bytes"] <= memory["MemAvailable"],
        "assumptions": {**cache["assumptions"], "slack_gib": args.slack_gib},
        "slack_covers": [
            "Ordinary CPU token-embedding GPU mirror: default EXL3_EMBED_GPU=1, up to EXL3_EMBED_GPU_MAX_MB=4096 MiB. The CPU weight is already counted; its CUDA copy is additional.",
            "Recurrent checkpoints: SYSMEM_RECURRENT_CACHE defaults to 4096 MiB. Recurrent device slots and speculative history additionally grow with MAX_BATCH_SIZE and DRAFT_NUM_TOKENS.",
            "Load-time shard staging, pinned row buffers, pruned draft-head copies, CUDA graphs, workspaces, allocator overhead and remaining host-process memory.",
        ],
        "ngram_storage_note": "RAM mode holds one CPU PLE table and transfers gathered rows, without a full GPU mirror. Sharded RAM loads also stage one input shard. Disk mode has reclaimable file-cache pages.",
        "advisory_note": "The named cache components do not make the remaining slack a measured bound. Use an unloaded-host memory snapshot, then validate actual load and workload peaks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-size", type=int, required=True)
    parser.add_argument("--draft-mode", choices=["mtp", "disabled"], default="mtp")
    parser.add_argument("--ngram-ram", choices=["true", "false"])
    parser.add_argument("--slack-gib", type=float, default=10.0)
    parser.add_argument("--choose-ngram", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.cache_size < 1 or not math.isfinite(args.slack_gib) or args.slack_gib < 0:
        parser.error("invalid cache/slack size")
    try:
        result = estimate(args)
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        print(f"warning: memory estimate unavailable: {exc}", file=sys.stderr)
        if args.choose_ngram:
            print("false")
            return 0
        return 1
    in_ram = result["ngram_ram_auto"] if args.choose_ngram else args.ngram_ram == "true"
    need = result["need_ram_bytes" if in_ram else "need_stream_bytes"]
    available = result["memory"]["MemAvailable"]
    label = "warning" if need > available else "==>"
    print(f"{label} memory estimate: {need/GIB:.1f} GiB "
          f"(non-ngram weights {result['other_weight_bytes']/GIB:.1f}, "
          f"ngram {'RAM' if in_ram else 'stream'} {result['ngram_bytes']/GIB:.1f}, "
          f"K/V {result['kv_data_bytes']/GIB:.2f}, scales {result['kv_scale_bytes']/GIB:.2f}, "
          f"QSA {result['qsa_index_bytes']/GIB:.2f}, slack {args.slack_gib:g}); "
          f"MemAvailable {available/GIB:.1f} GiB. Estimate only.", file=sys.stderr)
    for warning in result["warnings"]:
        print("warning: " + warning, file=sys.stderr)
    if need > available:
        print("warning: lower CACHE_SIZE/MAX_BATCH_SIZE or stream the n-gram table; "
              "validate peak memory with the actual model and workload.", file=sys.stderr)
    if args.choose_ngram:
        print("true" if in_ram else "false")
    elif args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
