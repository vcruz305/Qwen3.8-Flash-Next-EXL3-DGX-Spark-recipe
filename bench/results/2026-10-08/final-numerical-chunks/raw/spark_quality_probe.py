#!/usr/bin/env python3
"""Teacher-forced Qwen target-logit probe. No draft model or sampler is loaded.

Compare a complete cache-free prefill with paged causal execution over exactly the
same tokens, including recurrent-state carry and physical-page permutations.
The q_len > 1 paths exercise the target verification shape without MTP proposals.

Example (stop the inference server first):
  PYTHONPATH=/path/to/exllamav3 python spark_quality_probe.py \
    --model /path/to/model --ngram-ram --output /path/to/run.json \
    --save-logits /path/to/run.safetensors

Repeat in a fresh process with one environment change. --compare-logits FILE
compares each path against an earlier run and verifies identical input IDs.
Timings include diagnostic transfers and are NOT serving throughput benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

CORPUS = {
    "code": (
        "User: Write a Python function that merges overlapping half-open intervals. "
        "Sort by the start, preserve empty input, and explain the complexity.\n"
        "Assistant: def merge_intervals(intervals):\n"
        "    merged = []\n"
        "    for start, end in sorted(intervals):\n"
        "        if not merged or start > merged[-1][1]:\n"
        "            merged.append([start, end])\n"
        "        else:\n"
        "            merged[-1][1] = max(merged[-1][1], end)\n"
        "    return merged\n"
        "Sorting costs O(n log n) time and the scan costs O(n). "
        "The output holds at most n intervals. Test empty input, nested intervals, "
        "touching boundaries, negative endpoints, and unsorted values.\n"
        "User: Add tests for repeated values and verify the original list is unchanged.\n"
        "Assistant: assert merge_intervals([(2, 5), (1, 3), (8, 9)]) == [[1, 5], [8, 9]]\n"
        "assert merge_intervals([]) == []\n"
    ),
    "tools": (
        "The assistant can call get_weather with a city string, and search_documents "
        "with a query string and an integer limit. A tool result is data to inspect.\n"
        "User: Compare the weather in Miami and Seattle, then summarize the result.\n"
        "Assistant: <tool_call>{\"name\":\"get_weather\",\"arguments\":"
        "{\"city\":\"Miami\"}}</tool_call>\n"
        "Tool: {\"city\":\"Miami\",\"temperature_c\":28,\"condition\":\"clear\"}\n"
        "Assistant: <tool_call>{\"name\":\"get_weather\",\"arguments\":"
        "{\"city\":\"Seattle\"}}</tool_call>\n"
        "Tool: {\"city\":\"Seattle\",\"temperature_c\":14,\"condition\":\"rain\"}\n"
        "Assistant: Miami is 14 degrees Celsius warmer. Seattle has rain.\n"
        "User: Find three documents about cache consistency.\n"
        "Assistant: <tool_call>{\"name\":\"search_documents\",\"arguments\":"
        "{\"query\":\"cache consistency\",\"limit\":3}}</tool_call>\n"
        "Tool: {\"results\":[{\"title\":\"Page tables\"},{\"title\":"
        "\"Version checks\"},{\"title\":\"Recurrent state\"}]}\n"
    ),
    "multilingual": (
        "用户：请解释如何在程序中检查输入，并给出一个简短的例子。\n"
        "助手：先确认输入的类型，再检查范围。对于列表中的每个数字，"
        "可以验证它是否为有限值，并在出错时返回清楚的信息。\n"
        "Usuario: Explica por qué debemos verificar los resultados después de "
        "cambiar una optimización.\n"
        "Asistente: Una implementación más rápida puede cambiar el orden de las "
        "operaciones. Comparamos los mismos datos, revisamos los errores y "
        "medimos tanto la exactitud como el tiempo.\n"
        "Utilisateur : Comment conserver le contexte entre plusieurs appels ?\n"
        "Assistant : On conserve un état associé à chaque requête. Les pages "
        "de mémoire et leur position doivent rester cohérentes.\n"
        "User: Give a concise English summary of these answers.\n"
        "Assistant: Validate inputs, compare identical test data, and preserve "
        "each request's state across calls.\n"
    ),
}

ENV_KEYS = (
    "EXL3_INT8_GEMV", "EXL3_GR_INT8", "EXL3_GR_RB", "EXL3_GR_MIX_TILED",
    "EXL3_MOE_COOP_MIXEDK", "EXL3_COOPMK_PLAN", "EXL3_MOE_COOP_WIDE",
    "EXL3_MOE_MIXEDK_NOSYNC", "EXL3_MIXEDK_LEGACY", "EXL3_MOE_FUSED_ROWS",
    "EXL3_GRAPHS", "EXL3_BC_ATTN", "EXL3_BC_GDN", "EXL3_GDN_SUB_CHUNK",
    "EXL3_GDN_PROJ_FP32", "EXL3_GDN_CONV_TOKEN_MAJOR",
    "EXL3_GDN_CONV_BF16_PRODUCT", "EXL3_ATTN_DECODE_LEGACY_SPLITS",
    "EXL3_QC_STAGING", "PYTORCH_CUDA_ALLOC_CONF",
)


def integer_list(value):
    try:
        result = sorted(set(int(x) for x in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not result or min(result) < 1:
        raise argparse.ArgumentTypeError("Values must be positive integers")
    return result


def args_for_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-logits", type=Path)
    p.add_argument("--compare-logits", type=Path)
    p.add_argument("--contexts", type=integer_list, default=[255, 1023],
                   help="Prefix lengths; defaults cross 256-token page boundaries")
    p.add_argument("--q-lens", type=integer_list, default=[1, 6])
    p.add_argument("--batch-sizes", type=integer_list, default=[1])
    p.add_argument("--steps", type=int, default=48)
    p.add_argument("--prefill-chunk", type=int, default=1024)
    p.add_argument("--cases", default="code,tools,multilingual")
    p.add_argument("--ngram-ram", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-kl", type=float,
                   help="Optional mean KL limit against complete prefill")
    p.add_argument("--max-nll-increase", type=float,
                   help="Optional NLL increase limit against complete prefill")
    p.add_argument("--min-top1", type=float,
                   help="Optional top-1 agreement limit against complete prefill")
    args = p.parse_args()
    args.cases = args.cases.split(",")
    if any(case not in CORPUS for case in args.cases):
        p.error(f"--cases choices: {','.join(CORPUS)}")
    if args.steps < max(args.q_lens) or args.prefill_chunk < 1:
        p.error("--steps must cover the largest q_len and --prefill-chunk must be positive")
    if max(args.contexts) + args.steps + 1 > 262144:
        p.error("This recipe supports at most 262144 tokens")
    for path in (args.output, args.save_logits):
        if path and path.exists():
            p.error(f"Refusing to overwrite existing output: {path}")
    return args


def ids_for_case(tokenizer, case, length, batch):
    rows = []
    for row in range(batch):
        # Reproducible text with section numbers and row-specific context.
        text = f"Conversation {row + 1}. Technical review and validation.\n"
        section = 0
        while True:
            text += f"\nSection {section + 1}.\n" + CORPUS[case]
            section += 1
            ids = tokenizer.encode(text)
            if ids.shape[-1] >= length:
                rows.append(ids[:, :length].long().cpu())
                break
    return torch.cat(rows, dim=0)


def digest_tensor(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def logit_metrics(reference, candidate, targets):
    # Both tensors are CPU [batch, positions, vocab]. Chunk the vocabulary-wide
    # intermediates over positions, so the metric itself has a bounded footprint.
    assert reference.shape == candidate.shape
    a = reference.reshape(-1, reference.shape[-1]).float()
    b = candidate.reshape(-1, candidate.shape[-1]).float()
    target = targets.reshape(-1).long()
    if target.numel() != a.shape[0]:
        raise ValueError("Target positions do not match logits")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("Nonfinite logits inside the unpadded vocabulary")
    kls, nll_a, nll_b, agree, conf, hits_a, hits_b = [], [], [], [], [], [], []
    max_abs = 0.0
    for start in range(0, a.shape[0], 16):
        stop = min(start + 16, a.shape[0])
        aa, bb, tt = a[start:stop], b[start:stop], target[start:stop]
        la, lb = F.log_softmax(aa, dim=-1), F.log_softmax(bb, dim=-1)
        kls.append((la.exp() * (la - lb)).sum(-1).clamp_min(0))
        nll_a.append(-la.gather(-1, tt[:, None]).squeeze(-1))
        nll_b.append(-lb.gather(-1, tt[:, None]).squeeze(-1))
        top_a, top_b = aa.argmax(-1), bb.argmax(-1)
        agree.append(top_a == top_b)
        conf.append(la.max(-1).values.exp())
        hits_a.append(top_a == tt)
        hits_b.append(top_b == tt)
        max_abs = max(max_abs, (aa - bb).abs().max().item())
    kl, na, nb = torch.cat(kls), torch.cat(nll_a), torch.cat(nll_b)
    agreement, confidence = torch.cat(agree), torch.cat(conf)
    high = confidence >= 0.8
    return {
        "positions": a.shape[0],
        "mean_kl_reference_to_candidate": kl.mean().item(),
        "p95_kl": torch.quantile(kl, 0.95).item(),
        "max_kl": kl.max().item(),
        "reference_nll": na.mean().item(),
        "candidate_nll": nb.mean().item(),
        "nll_increase": (nb - na).mean().item(),
        "reference_perplexity": math.exp(min(na.mean().item(), 700)),
        "candidate_perplexity": math.exp(min(nb.mean().item(), 700)),
        "top1_agreement": agreement.float().mean().item(),
        "reference_next_token_accuracy": torch.cat(hits_a).float().mean().item(),
        "candidate_next_token_accuracy": torch.cat(hits_b).float().mean().item(),
        "high_confidence_positions": high.sum().item(),
        "high_confidence_top1_agreement": agreement[high].float().mean().item() if high.any() else None,
        "max_absolute_logit_difference": max_abs,
    }


def active_paths(model):
    from exllamav3.modules.hyperconnections import GatedResidual
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    modules = list(model)
    gr = [m for m in modules if isinstance(m, GatedResidual)]
    moe = [m for m in modules if isinstance(m, BlockSparseMLP)]
    return {
        "gr_sites": len(gr),
        "gr_int8_sites": sum(getattr(m, "fn_q", None) is not None for m in gr),
        "gr_tiled_sites": sum(bool(getattr(m, "tiled", False)) for m in gr),
        "moe_layers": len(moe),
        "mixedk_unified_layers": sum(bool(getattr(m, "mixedk_unified", False)) for m in moe),
        "coopmk_layers": sum(getattr(m, "coopmk", None) is not None for m in moe),
        "coopmk_plans": sorted(set(int(m.coopmk.plan) for m in moe if getattr(m, "coopmk", None) is not None)),
    }


def package_info():
    import exllamav3
    root = Path(exllamav3.__file__).resolve().parent.parent
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         text=True, capture_output=True, check=False)
    return {"package": str(root), "version": getattr(exllamav3, "__version__", None),
            "git_commit": git.stdout.strip() if git.returncode == 0 else None}


def paged_logits(model, cache, ids, prefix, steps, q_len, chunk):
    batch = ids.shape[0]
    pages_per_seq = cache.max_num_tokens // cache.num_slots // 256
    # Deterministically permute physical page order to exercise paged addressing.
    block = torch.randperm(batch * pages_per_seq, generator=torch.Generator().manual_seed(231))
    block = block.reshape(batch, pages_per_seq).to(torch.int32).contiguous()
    states = [cache.get_new_state() for _ in range(batch)] if model.caps.get("recurrent_states") else None

    def params(position, q, history=False):
        return {
            "attn_mode": "flash_attn",
            "cache": cache,
            "block_table": block,
            "cache_seqlens": torch.full((batch,), position, dtype=torch.int32),
            "recurrent_states": states,
            "recurrent_history": history,
        }

    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for pos in range(0, prefix, chunk):
            end = min(pos + chunk, prefix)
            model.prefill(ids[:, pos:end], params(pos, end - pos))
        torch.cuda.synchronize()
        prefill_seconds = time.perf_counter() - start
        chunks = []
        start = time.perf_counter()
        for pos in range(prefix, prefix + steps, q_len):
            end = min(pos + q_len, prefix + steps)
            logits = model.forward(ids[:, pos:end], params(pos, end - pos, end - pos > 1))
            chunks.append(logits[..., :model.config.vocab_size].contiguous().half().cpu())
            del logits
            if states and end - pos > 1:
                # A verification-shaped forward records per-token history. Even
                # when every token is accepted, the generator calls rewind(0):
                # it commits the latest GDN convolution and PLE id/conv history
                # to the working slots before the next forward (position stays
                # unchanged). Omitting this silently reuses stale context.
                for state in states:
                    state.rewind(0)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if states:
            assert all(s.position == prefix + steps for s in states), "Recurrent position drift"
        return torch.cat(chunks, dim=1), {"prefill_seconds": prefill_seconds,
                                        "diagnostic_decode_seconds": seconds}
    finally:
        if states:
            for state in states:
                cache.release_state(state)


def threshold_errors(args, metrics):
    errors = []
    if args.max_kl is not None and metrics["mean_kl_reference_to_candidate"] > args.max_kl:
        errors.append("mean KL limit exceeded")
    if args.max_nll_increase is not None and metrics["nll_increase"] > args.max_nll_increase:
        errors.append("NLL increase limit exceeded")
    if args.min_top1 is not None and metrics["top1_agreement"] < args.min_top1:
        errors.append("top-1 agreement below limit")
    return errors


@torch.inference_mode()
def main(args):
    from exllamav3 import Config, Model, Tokenizer, Cache
    from safetensors.torch import save_file, load_file
    from safetensors import safe_open

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not run this probe on the WSL CPU checkout")
    torch.cuda.set_device(torch.device(args.device))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(args.device)
    largest = max(args.contexts) + args.steps + 1
    stride = (largest + 255) // 256 * 256
    batch_max = max(args.batch_sizes)
    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(stride)
    config.infer_params.ngram_stream_from_disk = not args.ngram_ram
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config, component="text")
    cache = Cache(model, max_num_tokens=stride * batch_max, max_batch_size=batch_max,
                  max_history=max(args.q_lens) - 1)
    model.load(device=device, progressbar=True, max_chunk_size=largest,
               max_output_size=args.steps, max_batch_size=batch_max)
    saved = {}
    comparison = None
    if args.compare_logits:
        with safe_open(str(args.compare_logits), framework="pt", device="cpu") as previous:
            metadata = previous.metadata() or {}
        if metadata.get("probe_format_version") != "2":
            raise ValueError("Comparison must be from probe format 2 (accepted-history commit semantics)")
        comparison = load_file(str(args.compare_logits), device="cpu")
    report = {
        "format_version": 2,
        "model": str(Path(args.model).resolve()),
        "runtime": package_info(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "environment": {k: os.environ.get(k) for k in ENV_KEYS},
        "active_paths": active_paths(model),
        "mtp_loaded": False,
        "cache_type": "fp16",
        "ngram_ram": args.ngram_ram,
        "timings_are_diagnostic_only": True,
        "input_note": "Fixed technical/code/tool/multilingual text; a numerical regression probe, not a broad quality benchmark.",
        "configured_thresholds": {"max_kl": args.max_kl, "max_nll_increase": args.max_nll_increase,
                                 "min_top1": args.min_top1},
        "cases": [],
    }
    try:
        for name in args.cases:
            for batch in args.batch_sizes:
                ids = ids_for_case(tokenizer, name, largest, batch)
                for prefix in args.contexts:
                    stem = f"{name}.b{batch}.p{prefix}"
                    input_ids = ids[:, :prefix + args.steps + 1].contiguous()
                    saved[f"{stem}.input_ids"] = input_ids.clone()
                    labels = input_ids[:, prefix + 1:prefix + args.steps + 1]
                    # Full context, no paged cache: only the suffix needs LM-head output.
                    ref = model.forward(input_ids[:, :-1],
                                        {"attn_mode": "flash_attn_nc", "last_tokens_only": args.steps})
                    ref = ref[..., :config.vocab_size].contiguous().half().cpu()
                    saved[f"{stem}.prefill"] = ref
                    if comparison is not None:
                        old_ids = comparison.get(f"{stem}.input_ids")
                        if old_ids is None or not torch.equal(old_ids, input_ids):
                            raise ValueError(f"Comparison input IDs differ or are missing: {stem}")
                    for q in args.q_lens:
                        key = f"{stem}.q{q}"
                        candidate, timing = paged_logits(model, cache, input_ids, prefix, args.steps,
                                                         q, args.prefill_chunk)
                        metrics = logit_metrics(ref, candidate, labels)
                        record = {"case": name, "batch": batch, "prefix": prefix, "q_len": q,
                                  "input_sha256": digest_tensor(input_ids),
                                  "metrics_vs_complete_prefill": metrics, "timing": timing,
                                  "threshold_errors": threshold_errors(args, metrics)}
                        if comparison is not None:
                            if key not in comparison:
                                raise ValueError(f"Comparison logits missing: {key}")
                            record["metrics_vs_previous_same_path"] = logit_metrics(comparison[key], candidate, labels)
                            record["prefill_vs_previous_prefill"] = logit_metrics(comparison[f"{stem}.prefill"], ref, labels)
                        saved[key] = candidate
                        report["cases"].append(record)
                        print(json.dumps({"probe": key, "metrics": metrics,
                                          "threshold_errors": record["threshold_errors"]}), flush=True)
        report["active_paths_after_inference"] = active_paths(model)
        report["sanity_checks_passed"] = True
        report["threshold_checks_configured"] = any(x is not None for x in report["configured_thresholds"].values())
        report["threshold_checks_passed"] = not any(r["threshold_errors"] for r in report["cases"])
    finally:
        model.unload()
    if args.save_logits:
        args.save_logits.parent.mkdir(parents=True, exist_ok=True)
        save_file(saved, str(args.save_logits), metadata={
            "probe_format_version": "2",
            "runtime": json.dumps(report["runtime"]), "model": report["model"],
            "purpose": "Teacher-forced target-logit regression comparison; MTP disabled"})
    # A completed report is published only after the requested logits artifact
    # was serialized successfully.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"]),
                      "threshold_checks_configured": report["threshold_checks_configured"],
                      "threshold_checks_passed": report["threshold_checks_passed"]}), flush=True)
    return 0 if report["threshold_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main(args_for_cli()))
