#!/usr/bin/env python3
"""Summarize completed optional jobs using the frozen primary metric definitions."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

BASE_SHA256 = "d1292bba6e63d8cddbf25afe553c923aabe2a747d8a9f6d064e520376770e660"
COUNTS = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(path):
    if digest(path) != BASE_SHA256:
        raise ValueError("The frozen base summarizer hash does not match")
    spec = importlib.util.spec_from_file_location("frozen_performance_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indexed_requests(row, concurrent=False):
    result = {}
    values = ((rd["round_index"], stream["stream_index"], stream)
              for rd in row["rounds"] for stream in rd["streams"]) if concurrent else (
                  (value["repeat_index"], None, value) for value in row["requests"])
    for repeat, stream, value in values:
        key = (repeat, stream)
        if key in result:
            raise ValueError("Duplicate measured request identity")
        if not isinstance(value.get("request_sha256"), str) or len(value["request_sha256"]) != 64:
            raise ValueError("Missing measured request hash")
        result[key] = value
    return result


def request_match(control, variant, concurrent=False):
    old = indexed_requests(control, concurrent)
    new = indexed_requests(variant, concurrent)
    same_ids = set(old) == set(new) and bool(old)
    pairs = [(key, old[key], new[key]) for key in sorted(set(old) & set(new))]
    return {
        "request_count_control": len(old),
        "request_count_variant": len(new),
        "matched_repeat_and_stream_indices": same_ids,
        "matched_request_sha256": same_ids and all(a["request_sha256"] == b["request_sha256"] for _, a, b in pairs),
        "matched_actual_token_counts": same_ids and all(all(a.get(k) == b.get(k) for k in COUNTS) for _, a, b in pairs),
        "response_hash_matches": sum(a.get("response_sha256") == b.get("response_sha256") for _, a, b in pairs),
        "paired_requests": [
            {"repeat_or_round_index": key[0], "stream_index": key[1],
             "request_sha256_control": a["request_sha256"],
             "request_sha256_variant": b["request_sha256"],
             "counts_control": {k: a.get(k) for k in COUNTS},
             "counts_variant": {k: b.get(k) for k in COUNTS},
             "response_sha256_control": a.get("response_sha256"),
             "response_sha256_variant": b.get("response_sha256")}
            for key, a, b in pairs
        ],
    }


def augment(report):
    report["schema_version"] = 2
    report["matrix_kind"] = "optional"
    report["base_summary_script_sha256"] = BASE_SHA256
    report["summary_script_sha256"] = digest(Path(__file__))
    report["notes"] = [
        note.replace(
            "This historical primary matrix runs engine16ca/Tabbyf4. Finalf0 and its paired final Tabby source require separate live validation.",
            "This historical optional matrix runs engine16ca/Tabbyf4. Later engine/server revisions and any combined selected profile require separate live qualification.")
        for note in report["notes"]
    ]
    report["notes"] += [
        "The optional wrapper preserves the frozen primary metric and inclusion code. It adds exact measured request-hash/count comparisons without altering metric values or promoting failed jobs.",
        "A matched pair records identical source commits, one changed resolved setting, benchmark settings/run ID, and per-repeat or per-round/stream request hashes. Different response hashes and draft acceptance are retained as observations.",
        "These are sequential small-sample measurements from one machine. Medians and observed round ranges describe these runs; no repeated-matrix confidence interval or universal timing benefit is established.",
        "The static-drafting job has no preceding control in its own matrix block. It is displayed without an invented matched comparison.",
        "A RAM-placement comparison includes runtime scheduling and generated-output differences observed under that setting. Matching wire requests alone does not isolate every timing difference to memory-transfer speed.",
    ]
    bench = {(x["label"], x["benchmark"], x["case"]): x for x in report["bench_rows"]}
    concurrent = {(x["label"], x["concurrency"]): x for x in report["concurrency_rows"]}
    for row in report["concurrency_rows"]:
        rates = [rd["end_to_end_tok_s"] for rd in row["rounds"]]
        row["end_to_end_tok_s_rounds"] = rates
        row["end_to_end_tok_s_observed_range"] = [min(rates), max(rates)] if rates else None
    for pair in report["comparisons"]:
        for case in pair["cases"]:
            key = (case["benchmark"], case["case"])
            match = request_match(bench[(pair["control"], *key)], bench[(pair["variant"], *key)])
            case["request_match"] = match
            case["comparison_contract_met"] = (pair["single_knob_source_matched"] and case["matched_settings"]
                                               and match["matched_request_sha256"] and match["matched_actual_token_counts"])
        for case in pair["concurrency"]:
            key = case["concurrency"]
            match = request_match(concurrent[(pair["control"], key)], concurrent[(pair["variant"], key)], True)
            case["request_match"] = match
            case["comparison_contract_met"] = (pair["single_knob_source_matched"] and case["matched_settings"]
                                               and case["both_reports_valid"] and match["matched_request_sha256"]
                                               and match["matched_actual_token_counts"])
    return report


def markdown(base, report):
    rendered = base.markdown(report)
    rendered = rendered.replace("# Completed primary performance jobs", "# Completed optional performance jobs", 1)
    proof = ["", "## Matched request and timing scope", "",
             "| Control → variant | Case or concurrency | Requests per setting | Exact request hashes and counts | Same source, one setting and benchmark contract | Matching response hashes |",
             "|---|---|---:|---|---|---:|"]
    for pair in report["comparisons"]:
        for row in pair["cases"] + pair["concurrency"]:
            match = row["request_match"]
            case = str(row["concurrency"]) + " concurrent" if "concurrency" in row else row["benchmark"] + "/" + row["case"]
            matched = match["matched_request_sha256"] and match["matched_actual_token_counts"]
            proof.append(f"| {pair['control']} → {pair['variant']} | {case} | {match['request_count_control']}/{match['request_count_variant']} | {matched} | {row['comparison_contract_met']} | {match['response_hash_matches']} |")
    if report["concurrency_rows"]:
        proof += ["", "| Job | Concurrent requests | Measured batch rates, tok/s |",
                  "|---|---:|---|"]
        for row in report["concurrency_rows"]:
            values = ", ".join(f"{value:.4f}" for value in row["end_to_end_tok_s_rounds"])
            proof.append(f"| {row['label']} | {row['concurrency']} | {values} |")
    return rendered.replace("\n## Scope\n", "\n".join(proof) + "\n\n## Scope\n", 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--base-summarizer", type=Path, default=Path(__file__).with_name("summarize_completed_performance.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = load_base(args.base_summarizer)
    report = augment(base.summarize(args.reports, args.jobs))
    args.output.mkdir(mode=0o700)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown(base, report))
    shutil.copyfile(__file__, args.output / "summarize_optional_performance.py")
    shutil.copyfile(args.base_summarizer, args.output / "summarize_completed_performance.py")
    print(json.dumps({"completed": report["completed_job_count"], "expected": report["expected_job_count"],
                      "bench_rows": len(report["bench_rows"]), "concurrency_rows": len(report["concurrency_rows"]),
                      "comparisons": len(report["comparisons"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
