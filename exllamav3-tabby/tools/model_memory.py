#!/usr/bin/env python3
"""Conservative Qwen pack memory estimate from safetensors headers, without loading weights.

This is a startup advisory, not a guarantee of a maximum working context. CUDA
allocations, recurrent slots, kernel workspaces and the OS need measured headroom.
"""
from __future__ import annotations

import argparse
import json
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


def estimate(args):
    pack = pack_bytes(args.model)
    config = json.loads((Path(args.model) / "config.json").read_text())
    config = config.get("text_config") or config
    total_layers = config.get("num_hidden_layers", 48)
    interval = config.get("full_attention_interval", 4)
    layer_types = config.get("layer_types")
    full_layers = sum(x == "full_attention" for x in layer_types) if layer_types else total_layers // interval
    kv_heads = config.get("num_key_value_heads", 2)
    head_dim = config.get("head_dim", 256)
    draft_layers = 1 if args.draft_mode == "mtp" else 0
    # 8-bit K and V, with a conservative 10% allowance for quantization metadata.
    kv_bytes = int(args.cache_size * (full_layers + draft_layers) * kv_heads * head_dim * 2 * 1.10)
    slack = int(args.slack_gib * GIB)
    memory = available_bytes()
    base = pack["other_weight_bytes"] + kv_bytes + slack
    return {
        **pack, "kv_estimate_bytes": kv_bytes, "slack_bytes": slack,
        "need_stream_bytes": base, "need_ram_bytes": base + pack["ngram_bytes"],
        "memory": memory,
        "ngram_ram_auto": base + pack["ngram_bytes"] <= memory["MemAvailable"],
        "assumptions": {"cache_mode": "8,8", "kv_metadata_allowance": 0.10,
                        "full_attention_layers": full_layers, "mtp_layers": draft_layers,
                        "slack_gib": args.slack_gib},
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
    if args.cache_size < 1 or args.slack_gib < 0:
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
          f"KV {result['kv_estimate_bytes']/GIB:.1f}, slack {args.slack_gib:g}); "
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
