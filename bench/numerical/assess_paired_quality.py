#!/usr/bin/env python3
"""Apply a previously declared numerical smoke gate to format-2 probe JSON.

This is a CPU-only report audit. It checks model-view paths, exact input hashes,
format-2 successful-sanity evidence, and paired-metric consistency with the
provided baseline. The probe itself checked exact input tensors and finiteness
of unpadded logits. Weight-file provenance remains the deployment's independent
responsibility; this utility does not re-read model files or claim broad quality.

Exit 2 for a core failure or invalid evidence. Investigation-only drift exits 0
with status "investigate", so a numerical core pass is not deployment approval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

REQUIRED_GATES = (
    "paired_nll_max_pooled_increase_nats",
    "paired_nll_max_per_case_increase_nats",
    "high_confidence_additional_disagreements_max_per_48_positions",
    "high_confidence_additional_disagreements_max_pooled",
    "investigate_mean_paired_kl_above",
    "investigate_p95_paired_kl_above",
)
REQUIREMENTS = {"format2", "same_model_view", "exact_input_ids", "finite_unpadded_logits"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=2e-6)


def case_key(case):
    require(isinstance(case.get("case"), str) and case["case"], "Missing case name")
    for key in ("batch", "prefix", "q_len"):
        require(type(case.get(key)) is int and case[key] > 0, f"Invalid {key} in case")
    return (case["case"], case["batch"], case["prefix"], case["q_len"])


def key_label(key, prefill=False):
    case, batch, prefix, *q = key
    return f"{case}.b{batch}.p{prefix}." + ("prefill" if prefill else f"q{q[0]}")


def checked_metrics(metrics, label):
    require(isinstance(metrics, dict), f"{label}: missing metrics")
    n = metrics.get("positions")
    hc = metrics.get("high_confidence_positions")
    require(type(n) is int and n > 0, f"{label}: invalid position count")
    require(type(hc) is int and 0 <= hc <= n, f"{label}: invalid high-confidence count")
    keys = (
        "reference_nll", "candidate_nll", "nll_increase",
        "mean_kl_reference_to_candidate", "p95_kl", "max_kl",
        "top1_agreement", "reference_next_token_accuracy",
        "candidate_next_token_accuracy", "max_absolute_logit_difference",
    )
    for key in keys:
        value = metrics.get(key)
        require(type(value) in (int, float) and math.isfinite(value),
                f"{label}: invalid/nonfinite {key}")
    for key in ("top1_agreement", "reference_next_token_accuracy", "candidate_next_token_accuracy"):
        require(0 <= metrics[key] <= 1, f"{label}: invalid {key}")
    agreement = metrics.get("high_confidence_top1_agreement")
    if hc:
        require(type(agreement) in (int, float) and math.isfinite(agreement) and 0 <= agreement <= 1,
                f"{label}: invalid high-confidence agreement")
        count = hc * (1 - agreement)
        require(abs(count - round(count)) <= 1e-4, f"{label}: non-integral disagreement count")
    else:
        require(agreement is None, f"{label}: confidence agreement must be null for zero positions")
    require(close(metrics["candidate_nll"] - metrics["reference_nll"], metrics["nll_increase"]),
            f"{label}: inconsistent NLL delta")
    return metrics


def disagreements(metrics):
    n = metrics["high_confidence_positions"]
    return round(n * (1 - metrics["high_confidence_top1_agreement"])) if n else 0


def index_cases(report, label):
    require(report.get("format_version") == 2, f"{label}: probe format 2 required")
    require(report.get("sanity_checks_passed") is True, f"{label}: no successful finite-logit sanity evidence")
    require(report.get("mtp_loaded") is False, f"{label}: target-only probe required")
    require(isinstance(report.get("model"), str) and report["model"], f"{label}: missing model view")
    require(isinstance(report.get("cases"), list) and report["cases"], f"{label}: no cases")
    indexed = {}
    for case in report["cases"]:
        key = case_key(case)
        require(key not in indexed, f"{label}: duplicate {key_label(key)}")
        require(re.fullmatch("[0-9a-f]{64}", case.get("input_sha256", "")) is not None,
                f"{label}: invalid input SHA256 in {key_label(key)}")
        checked_metrics(case.get("metrics_vs_complete_prefill"), f"{label} {key_label(key)}")
        indexed[key] = case
    return indexed


def score_group(rows, gates):
    errors, investigations = [], []
    scored = []
    for row in rows:
        paired = row["paired_metrics"]
        n = paired["positions"]
        per_case = []
        if paired["nll_increase"] > gates["paired_nll_max_per_case_increase_nats"]:
            per_case.append("paired_nll_per_case")
        additional = row["candidate_high_confidence_disagreements"] - row["baseline_high_confidence_disagreements"]
        limit = gates["high_confidence_additional_disagreements_max_per_48_positions"] * n / 48
        if additional > limit:
            per_case.append("additional_high_confidence_disagreements_per_case")
        drift = []
        if paired["mean_kl_reference_to_candidate"] > gates["investigate_mean_paired_kl_above"]:
            drift.append("mean_paired_kl")
        if paired["p95_kl"] > gates["investigate_p95_paired_kl_above"]:
            drift.append("p95_paired_kl")
        result = {
            **row,
            "additional_high_confidence_disagreements": additional,
            "high_confidence_additional_limit": limit,
            "core_failures": per_case,
            "investigation_triggers": drift,
        }
        scored.append(result)
        errors.extend(f"{row['key']}: {item}" for item in per_case)
        investigations.extend(f"{row['key']}: {item}" for item in drift)
    total = sum(row["paired_metrics"]["positions"] for row in rows)
    pooled = sum(row["paired_metrics"]["nll_increase"] * row["paired_metrics"]["positions"] for row in rows) / total
    old_bad = sum(row["baseline_high_confidence_disagreements"] for row in rows)
    new_bad = sum(row["candidate_high_confidence_disagreements"] for row in rows)
    if pooled > gates["paired_nll_max_pooled_increase_nats"]:
        errors.append("pooled_paired_nll")
    if new_bad - old_bad > gates["high_confidence_additional_disagreements_max_pooled"]:
        errors.append("pooled_additional_high_confidence_disagreements")
    return {
        "core_passed": not errors,
        "cases": len(rows),
        "positions": total,
        "pooled_paired_nll_increase_nats": pooled,
        "baseline_high_confidence_disagreements": old_bad,
        "candidate_high_confidence_disagreements": new_bad,
        "additional_high_confidence_disagreements": new_bad - old_bad,
        "positive_per_case_excess_disagreements_diagnostic": sum(max(0, row["additional_high_confidence_disagreements"]) for row in scored),
        "core_failure_count": len(errors),
        "core_failures": errors,
        "investigation_count": len(investigations),
        "investigation_case_count": sum(bool(row["investigation_triggers"]) for row in scored),
        "investigation_triggers": investigations,
        "per_case": scored,
    }


def assess(baseline, candidate, gates):
    require(set(gates.get("requirements", [])) == REQUIREMENTS,
            "Unrecognized or incomplete declared requirements")
    for key in REQUIRED_GATES:
        value = gates.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                f"Missing/invalid declared gate: {key}")
    old = index_cases(baseline, "baseline")
    new = index_cases(candidate, "candidate")
    require(old.keys() == new.keys(), "Baseline and candidate case sets differ")
    require(baseline["model"] == candidate["model"], "Model-view paths differ")
    require(baseline.get("cache_type") == candidate.get("cache_type") == "fp16", "Cache types differ or are unsupported")
    paged, prefills = [], {}
    for key in sorted(old):
        b, c = old[key], new[key]
        label = key_label(key)
        require(b["input_sha256"] == c["input_sha256"], f"{label}: exact input hashes differ")
        bm = b["metrics_vs_complete_prefill"]
        cm = c["metrics_vs_complete_prefill"]
        pm = checked_metrics(c.get("metrics_vs_previous_same_path"), f"{label} paired")
        fm = checked_metrics(c.get("prefill_vs_previous_prefill"), f"{label} paired prefill")
        require(bm["positions"] == cm["positions"] == pm["positions"] == fm["positions"],
                f"{label}: position counts differ")
        require(close(pm["reference_nll"], bm["candidate_nll"]) and
                close(pm["candidate_nll"], cm["candidate_nll"]),
                f"{label}: paired metrics do not match supplied old/new paths")
        require(close(fm["reference_nll"], bm["reference_nll"]) and
                close(fm["candidate_nll"], cm["reference_nll"]),
                f"{label}: paired prefill does not match supplied old/new reference")
        paged.append({
            "key": label,
            "input_sha256": c["input_sha256"],
            "paired_metrics": pm,
            "baseline_metrics_vs_own_prefill": bm,
            "candidate_metrics_vs_own_prefill": cm,
            "baseline_high_confidence_disagreements": disagreements(bm),
            "candidate_high_confidence_disagreements": disagreements(cm),
            "baseline_high_confidence_positions": bm["high_confidence_positions"],
            "candidate_high_confidence_positions": cm["high_confidence_positions"],
        })
        stem = key[:3]
        if stem in prefills:
            require(prefills[stem]["paired_metrics"] == fm and
                    prefills[stem]["input_sha256"] == c["input_sha256"],
                    f"{label}: duplicated prefill evidence differs across q lengths")
        else:
            prefills[stem] = {
                "key": key_label(stem, prefill=True),
                "input_sha256": c["input_sha256"],
                "paired_metrics": fm,
                "baseline_high_confidence_disagreements": 0,
                "candidate_high_confidence_disagreements": disagreements(fm),
                "baseline_high_confidence_positions": fm["high_confidence_positions"],
                "candidate_high_confidence_positions": fm["high_confidence_positions"],
            }
    paged_result = score_group(paged, gates)
    prefill_result = score_group(list(prefills.values()), gates)
    passed = paged_result["core_passed"] and prefill_result["core_passed"]
    investigations = paged_result["investigation_count"] + prefill_result["investigation_count"]
    return {
        "format_version": 1,
        "status": "fail" if not passed else "investigate" if investigations else "pass",
        "core_passed": passed,
        "investigation_required": bool(investigations),
        "core_failure_count": paged_result["core_failure_count"] + prefill_result["core_failure_count"],
        "investigation_count": investigations,
        "model": baseline["model"],
        "baseline_runtime": baseline.get("runtime"),
        "candidate_runtime": candidate.get("runtime"),
        "baseline_environment": baseline.get("environment"),
        "candidate_environment": candidate.get("environment"),
        "declared_gates": gates,
        "paged": paged_result,
        "unique_prefill": prefill_result,
        "interpretation": [
            "Thresholds are applied unchanged from the supplied declaration.",
            "Paged high-confidence excess compares each run against its own complete-prefill reference; qualifying masks can differ and counts are reported.",
            "Pooled additional disagreements means candidate total minus baseline total; positive per-case excess is also reported as a diagnostic.",
            "Full-prefill NLL/KL is scored once per case/batch/prefix, separately from paged paths.",
            "Full-prefill high-confidence disagreement compares new prefill directly with old prefill, whose self-disagreement baseline is zero.",
            "JSON provenance checks rely on the successful format-2 probe for finite unpadded logits and exact tensor comparison; model-file provenance is separate.",
            "An investigation-only result requires review before choosing deployment settings. This short correlated corpus is not broad capability certification.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing assessment: {args.output}")
    paths = {"gates": args.gates, "baseline": args.baseline, "candidate": args.candidate}
    try:
        loaded = {key: json.loads(path.read_text()) for key, path in paths.items()}
        result = assess(loaded["baseline"], loaded["candidate"], loaded["gates"])
    except (ValueError, TypeError, KeyError, OSError) as error:
        result = {"format_version": 1, "status": "invalid", "core_passed": False,
                  "validation_error": str(error), "core_failure_count": 1,
                  "investigation_count": 0, "investigation_required": False}
    result["assessed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["artifacts"] = {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
        for key, path in paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"],
                      "core_failure_count": result["core_failure_count"],
                      "investigation_count": result["investigation_count"]}))
    return 0 if result["core_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
