#!/usr/bin/env python3
"""Q8 KV adapter for the frozen format-2 target-logit probe.

Keep this file beside the exact frozen spark_quality_probe.py. All original CLI
arguments are forwarded unchanged. Only the cache factory and cache/probe metadata
are adapted; token construction, page permutation, recurrent commits, model
lifecycle, numerical metrics and saved tensor keys reuse the frozen harness.

Both sides of a comparison must use this adapter and actual K8/V8 cache storage.
Full-prefill references remain cache-free. MTP is still disabled; q_len=6 exercises
the target verification shape. Diagnostic timing is not API throughput.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE_SHA256 = "88d83c07acf46f8fbb5964ecc4920d199abe89106917e5f1b822dc75ff7c04f8"
CACHE_BITS = {"k": 8, "v": 8}
BASE_NAME = "spark_quality_probe.py"
ADAPTER_NAME = "spark_quality_q8_probe.py"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_frozen_base(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name(BASE_NAME)
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == BASE_SHA256,
            "The Q8 adapter requires the exact frozen format-2 probe SHA " + BASE_SHA256)
    spec = importlib.util.spec_from_file_location("_spark_quality_q8_frozen_base", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, source.decode("utf-8")


def _class_name(obj):
    cls = type(obj)
    return cls.__module__ + "." + cls.__name__


def _tensor_metadata(tensor, expected_dtype, expected_shape, name):
    import torch
    require(isinstance(tensor, torch.Tensor), name + ": cache tensor is not allocated")
    require(tensor.dtype == expected_dtype, name + ": wrong allocated tensor dtype")
    require(tuple(tensor.shape) == tuple(expected_shape), name + ": wrong allocated tensor shape")
    require(tensor.is_contiguous(), name + ": cache tensor is not contiguous")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
    }


def _make_q8_cache(model, *, max_num_tokens, max_batch_size, max_history):
    from exllamav3 import Cache
    from exllamav3.cache import CacheLayer_quant
    cache = Cache(
        model, max_num_tokens=max_num_tokens, max_batch_size=max_batch_size,
        max_history=max_history, layer_type=CacheLayer_quant, k_bits=8, v_bits=8,
    )
    require(bool(cache.layers), "Q8 probe requires at least one attention KV cache layer")
    require(cache.layer_type is CacheLayer_quant, "Cache ignored requested quantized layer type")
    for layer in cache.layers.values():
        require(isinstance(layer, CacheLayer_quant), "Attention cache is not a quantized layer")
        require(layer.k_bits == layer.v_bits == 8, "Attention cache is not K8/V8")
        require(layer.compand_a == 0.0, "Q8 probe expects the serving cache's default companding")
    return cache


def _audit_allocated_cache(cache):
    import torch
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.cache.qsa import QSAPlanes
    require(cache.initialized is True, "Model load did not initialize the Q8 cache")
    layers = []
    for key, layer in sorted(cache.layers.items()):
        require(isinstance(layer, CacheLayer_quant), "Allocated attention cache is not quantized")
        require(layer.k_bits == layer.v_bits == 8, "Allocated attention cache is not K8/V8")
        require(layer.compand_a == 0.0, "Allocated Q8 cache has unexpected companding")
        record = {
            "key": list(key), "class": _class_name(layer),
            "k_bits": layer.k_bits, "v_bits": layer.v_bits,
            "compand_a": layer.compand_a,
            "qk": _tensor_metadata(layer.qk, torch.int32, layer.qshape_k, "qk"),
            "qv": _tensor_metadata(layer.qv, torch.int32, layer.qshape_v, "qv"),
            "sk": _tensor_metadata(layer.sk, torch.float16, layer.qshape_s, "sk"),
            "sv": _tensor_metadata(layer.sv, torch.float16, layer.qshape_s, "sv"),
        }
        expected_bits = cache.max_num_tokens * layer.token_dim * 8
        require(layer.qk.numel() * 32 == expected_bits, "Packed K storage is not eight bits per value")
        require(layer.qv.numel() * 32 == expected_bits, "Packed V storage is not eight bits per value")
        indexer = getattr(layer.attention, "qsa_indexer", None)
        if indexer is not None:
            require(isinstance(layer, QSAPlanes), "QSA layer lacks its indexer cache planes")
            record["qsa_planes"] = {
                "raw_k": _tensor_metadata(layer.raw_k, torch.float16, layer.raw_k_shape, "QSA raw_k"),
                "pooled": _tensor_metadata(layer.pooled, torch.float16, layer.pooled_shape, "QSA pooled"),
            }
        layers.append(record)
    contract = {
        "schema_version": 1, "requested_layer_type": "CacheLayer_quant",
        "k_bits": 8, "v_bits": 8, "initialized": True,
        # Original94 overwrites cache.num_layers with the recurrent count.
        # Count the actual attention layer mapping instead.
        "layer_count": len(layers), "layers": layers,
        "max_num_tokens": cache.max_num_tokens,
        "max_batch_size": cache.num_slots, "max_history": cache.max_history,
        "recurrent_layer_classes": sorted({_class_name(layer) for layer in cache.recurrent_layers.values()}),
    }
    validate_cache_contract(contract)
    return contract


def _q8_contract(cache):
    # The base builds its report immediately after loading, before its try/finally.
    # A new audit failure must not leave that successfully loaded model allocated.
    try:
        return _audit_allocated_cache(cache)
    except Exception:
        cache.model.unload()
        raise


def validate_cache_contract(contract):
    require(isinstance(contract, dict), "Missing observed Q8 cache contract")
    require(contract.get("schema_version") == 1, "Unsupported Q8 cache contract version")
    require(contract.get("requested_layer_type") == "CacheLayer_quant", "Wrong requested cache layer class")
    require(contract.get("k_bits") == contract.get("v_bits") == 8, "Observed cache contract is not K8/V8")
    require(contract.get("initialized") is True, "Q8 cache allocation was not observed")
    layers = contract.get("layers")
    require(isinstance(layers, list) and bool(layers), "No observed Q8 attention layers")
    require(contract.get("layer_count") == len(layers), "Observed Q8 layer count is inconsistent")
    for layer in layers:
        require(isinstance(layer, dict), "Malformed observed Q8 layer")
        require(layer.get("k_bits") == layer.get("v_bits") == 8, "Observed attention layer is not K8/V8")
        require(layer.get("compand_a") == 0.0, "Observed Q8 companding differs")
        require(isinstance(layer.get("class"), str) and layer["class"].endswith(
            (".CacheLayer_quant", ".CacheLayer_qsa_quant")), "Wrong observed quantized layer class")
        for key, dtype in (("qk", "torch.int32"), ("qv", "torch.int32"),
                           ("sk", "torch.float16"), ("sv", "torch.float16")):
            tensor = layer.get(key)
            require(isinstance(tensor, dict) and tensor.get("dtype") == dtype,
                    "Missing/wrong observed " + key + " dtype")
            shape = tensor.get("shape")
            require(isinstance(shape, list) and len(shape) == 3
                    and all(type(n) is int and n > 0 for n in shape),
                    "Invalid observed " + key + " shape")
        if layer["class"].endswith(".CacheLayer_qsa_quant"):
            planes = layer.get("qsa_planes")
            require(isinstance(planes, dict), "Missing observed QSA indexer planes")
            for key in ("raw_k", "pooled"):
                tensor = planes.get(key)
                require(isinstance(tensor, dict) and tensor.get("dtype") == "torch.float16",
                        "QSA indexer planes must remain FP16")
    return contract


def _cache_tensor_metadata(report):
    return {
        "cache_type": "q8", "cache_k_bits": "8", "cache_v_bits": "8",
        "cache_contract": json.dumps(report["cache_contract"], sort_keys=True, separators=(",", ":")),
        "probe_base_sha256": report["probe_base_sha256"],
        "probe_adapter_sha256": report["probe_adapter_sha256"],
        "complete_prefill_cache_type": "none",
    }


def validate_comparison(path, adapter_sha256, expected_geometry=None):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata() or {}
    require(metadata.get("probe_format_version") == "2", "Q8 comparison requires probe format 2")
    require(metadata.get("cache_type") == "q8", "Comparison is not an explicitly labeled Q8 artifact")
    require(metadata.get("cache_k_bits") == metadata.get("cache_v_bits") == "8",
            "Comparison cache bits are not K8/V8")
    require(metadata.get("probe_base_sha256") == BASE_SHA256, "Comparison uses a different frozen probe")
    require(metadata.get("probe_adapter_sha256") == adapter_sha256, "Comparison uses a different Q8 adapter")
    require(metadata.get("complete_prefill_cache_type") == "none",
            "Q8 comparison lacks the cache-free full-prefill contract")
    try:
        observed = json.loads(metadata.get("cache_contract", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Comparison lacks valid observed Q8 cache metadata") from exc
    validate_cache_contract(observed)
    if expected_geometry is not None:
        for key, value in expected_geometry.items():
            require(observed.get(key) == value, "Comparison Q8 cache geometry differs: " + key)


def adapt_main_ast(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    require(len(functions) == 1, "Frozen probe main shape changed")
    main = copy.deepcopy(functions[0])
    cache_assignments = [
        node for node in ast.walk(main) if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "cache"
    ]
    expected = ast.parse(
        "cache = Cache(model, max_num_tokens=stride * batch_max, max_batch_size=batch_max, "
        "max_history=max(args.q_lens) - 1)"
    ).body[0]
    require(len(cache_assignments) == 1 and ast.dump(cache_assignments[0]) == ast.dump(expected),
            "Frozen probe Cache call shape changed")
    cache_assignments[0].value.func.id = "_make_q8_cache"

    reports = [node.value for node in main.body if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == "report"]
    require(len(reports) == 1 and isinstance(reports[0], ast.Dict), "Frozen report shape changed")
    report = reports[0]
    keys = [key.value if isinstance(key, ast.Constant) else None for key in report.keys]
    require(keys.count("cache_type") == 1, "Frozen report cache label shape changed")
    index = keys.index("cache_type")
    require(isinstance(report.values[index], ast.Constant) and report.values[index].value == "fp16",
            "Frozen cache label was not FP16")
    report.values[index] = ast.Constant("q8")
    additions = {
        "cache_bits": "dict(_Q8_CACHE_BITS)",
        "cache_contract": "_q8_contract(cache)",
        "complete_prefill_cache_type": "'none'",
        "probe_base_sha256": "_Q8_BASE_SHA256",
        "probe_adapter_sha256": "_Q8_ADAPTER_SHA256",
    }
    require(not set(additions).intersection(keys), "Frozen report already contains Q8 adapter fields")
    for key, expression in additions.items():
        report.keys.append(ast.Constant(key))
        report.values.append(ast.parse(expression, mode="eval").body)

    saves = [node for node in ast.walk(main) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "save_file"]
    require(len(saves) == 1, "Frozen tensor save shape changed")
    metadata = [kw for kw in saves[0].keywords if kw.arg == "metadata"]
    require(len(metadata) == 1 and isinstance(metadata[0].value, ast.Dict), "Frozen tensor metadata changed")
    metadata[0].value = ast.BinOp(
        left=metadata[0].value, op=ast.BitOr(),
        right=ast.parse("_cache_tensor_metadata(report)", mode="eval").body,
    )
    return ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))


def build_q8_main(base, source, adapter_sha256):
    namespace = dict(base.__dict__)
    namespace.update({
        "_make_q8_cache": _make_q8_cache, "_q8_contract": _q8_contract,
        "_cache_tensor_metadata": _cache_tensor_metadata,
        "_Q8_CACHE_BITS": CACHE_BITS, "_Q8_BASE_SHA256": BASE_SHA256,
        "_Q8_ADAPTER_SHA256": adapter_sha256,
    })
    exec(compile(adapt_main_ast(source), str(Path(__file__).resolve()) + ":frozen-main", "exec"), namespace)
    return namespace["main"], namespace


def run(args, base=None, source=None, adapter_sha256=None):
    if base is None or source is None:
        base, source = load_frozen_base()
    require(hashlib.sha256(source.encode("utf-8")).hexdigest() == BASE_SHA256,
            "Q8 run received a modified base source")
    adapter_sha256 = adapter_sha256 or file_sha256(__file__)
    for path in (args.output, args.save_logits):
        require(path is None or not os.path.lexists(path), "Refusing to overwrite existing Q8 output: " + str(path))
    # Do this before invoking base main: it imports/loads the model and sets CUDA.
    if args.compare_logits:
        largest = max(args.contexts) + args.steps + 1
        stride = (largest + 255) // 256 * 256
        batch_max = max(args.batch_sizes)
        validate_comparison(args.compare_logits, adapter_sha256, {
            "max_num_tokens": stride * batch_max,
            "max_batch_size": batch_max,
            "max_history": max(args.q_lens) - 1,
        })
    main, _ = build_q8_main(base, source, adapter_sha256)
    return main(args)


def main():
    base, source = load_frozen_base()
    return run(base.args_for_cli(), base, source)


if __name__ == "__main__":
    raise SystemExit(main())
