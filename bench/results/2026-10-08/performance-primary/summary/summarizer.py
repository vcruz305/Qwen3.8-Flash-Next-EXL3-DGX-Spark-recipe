#!/usr/bin/env python3
"""Summarize completed matrix jobs from a local copy of their unchanged reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ("server_decode_tok_s", "server_prefill_tok_s", "ttft_s", "wall_s",
           "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "draft_acceptance")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def med(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return statistics.median(values) if values else None


def summarize(reports, jobs_path):
    expected = json.loads(jobs_path.read_text())
    if not isinstance(expected, list) or len({j["label"] for j in expected}) != len(expected):
        raise ValueError("Expected a list of unique matrix job labels")
    observed = {p.parent.name: p for p in reports.glob("*/result.json")}
    unknown = set(observed) - {j["label"] for j in expected}
    if unknown:
        raise ValueError("Unexpected result labels: " + repr(sorted(unknown)))
    report = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "summary_script_sha256": digest(Path(__file__)),
              "source_directory": str(reports.resolve()), "jobs_file": str(jobs_path.resolve()),
              "jobs_sha256": digest(jobs_path), "completed_jobs": [], "excluded_jobs": [],
              "bench_rows": [], "concurrency_rows": [], "comparisons": [],
              "notes": [
                  "Only state=completed jobs with a final timestamp and verified cleanup enter the summary. A completed job may still fail a tool or concurrency gate.",
                  "Metrics are medians of measured requests, excluding warmup. Server throughput is reported by Tabby; client wall/TTFT are separately retained.",
                  "Concurrency throughput uses only complete measured batches: total output tokens divided by earliest request start to last request finish. Warmup and partial batches are excluded; per-request rates are not summed.",
                  "Each cold repeat has its own deterministic nonce. Different repeat response hashes do not imply nondeterminism, because request hashes differ.",
                  "Actual prompt, output, and cached counts are retained; nominal context size is not substituted for actual prompt tokens.",
                  "This historical primary matrix runs engine16ca/Tabbyf4. Finalf0 and its paired final Tabby source require separate live validation.",
                  "Single-knob comparisons use the nearest preceding control for the same matrix block/model, require exactly one changed resolved tuning variable, and compare identical benchmark settings.",
              ]}
    completed = {}
    for job in expected:
        label = job["label"]
        if label not in observed:
            report["excluded_jobs"].append({"label": label, "reason": "no final result file"})
            continue
        rp = observed[label]
        r = json.loads(rp.read_text())
        if (r.get("state") != "completed" or not r.get("finished_at_utc")
                or r.get("cleanup_failed") or r.get("cleanup_error")):
            report["excluded_jobs"].append({"label": label, "reason": "not completed or cleanup uncertain",
                                             "state": r.get("state")})
            continue
        actual_job = r["configuration"]["job"]
        if actual_job["label"] != label or actual_job["model_path"] != job["model_path"]:
            raise ValueError("Job label/model mismatch for " + label)
        attempt = rp.parent / Path(r["attempt"]).name
        sources = r["configuration"]["sources"]
        row = {"label": label, "model_path": actual_job["model_path"],
               "passed_all_clients": bool(r.get("passed")), "result_sha256": digest(rp),
               "source_commits": {k: sources[k]["commit"] for k in ("engine", "server", "recipe")},
               "source_controller_sha256": sources.get("controller_sha256"),
               "configuration_sha256": r.get("config_sha256"),
               "deployment_sha256": r.get("deployment_sha256"),
               "resolved_tuning": r["configuration"]["resolved_tuning"],
               "load_wall_seconds": r.get("load_wall_seconds"),
               "finished_at_utc": r["finished_at_utc"], "tools": None,
               "concurrency": None, "benchmarks": []}
        tp = attempt / "tools.json"
        if tp.exists():
            tool = json.loads(tp.read_text())
            if not tool.get("completed_at_utc"):
                raise ValueError("Completed job has incomplete tools report: " + label)
            row["tools"] = {"sha256": digest(tp), **tool["summary"]}
        cp = attempt / "concurrency.json"
        if cp.exists():
            concurrency = json.loads(cp.read_text())
            clients = [client for client in r["clients"] if client["name"] == "concurrency"]
            valid = (len(clients) == 1 and clients[0].get("exit_code") == 0
                     and clients[0].get("completed_report") is True
                     and bool(concurrency.get("completed_at_utc")) and not concurrency.get("errors"))
            row["concurrency"] = {"sha256": digest(cp), "valid": valid,
                                  "settings": concurrency["settings"], "run_id": concurrency["run_id"]}
            for width, group in concurrency["results"].items():
                width = int(width)
                rounds = group["rounds"]
                complete = [value for value in rounds if value.get("complete") is True
                            and not value.get("errors") and len(value.get("streams", [])) == width]
                if len({value["round_index"] for value in rounds}) != len(rounds):
                    raise ValueError("Duplicate measured concurrency rounds: " + label)
                for value in complete:
                    total = sum(stream["completion_tokens"] for stream in value["streams"])
                    if total != value["completion_tokens"] or value["batch_wall_s"] <= 0:
                        raise ValueError("Inconsistent concurrent token/wall totals: " + label)
                    if not math.isclose(total / value["batch_wall_s"], value["end_to_end_tok_s"], rel_tol=1e-12):
                        raise ValueError("Inconsistent aggregate concurrency rate: " + label)
                streams = [stream for value in complete for stream in value["streams"]]
                draft = [stream.get("usage", {}).get("completion_tokens_details", {}) for stream in streams]
                report["concurrency_rows"].append({
                    "label": label, "concurrency": width, "report_sha256": digest(cp),
                    "settings": concurrency["settings"], "run_id": concurrency["run_id"],
                    "complete_rounds": len(complete), "measured_rounds": len(rounds),
                    "failed_rounds": len(rounds) - len(complete),
                    "valid": valid and len(complete) == len(rounds) == concurrency["settings"]["rounds"],
                    "end_to_end_tok_s_median": med([value["end_to_end_tok_s"] for value in complete]),
                    "batch_wall_s_median": med([value["batch_wall_s"] for value in complete]),
                    "output_overlap_s_median": med([value["all_streams_output_overlap_s"] for value in complete]),
                    "actual_batch_completion_tokens": [value["completion_tokens"] for value in complete],
                    "actual_prompt_tokens": [stream["prompt_tokens"] for stream in streams],
                    "actual_completion_tokens": [stream["completion_tokens"] for stream in streams],
                    "actual_cached_prompt_tokens": [stream["cached_prompt_tokens"] for stream in streams],
                    "per_stream_medians": {key: med([stream.get(key) for stream in streams]) for key in METRICS},
                    "draft_accepted_tokens_median": med([d.get("accepted_prediction_tokens") for d in draft]),
                    "draft_rejected_tokens_median": med([d.get("rejected_prediction_tokens") for d in draft]),
                    "rounds": complete,
                })
        for client in r["clients"]:
            if not client["name"].startswith("bench-"):
                continue
            bp = attempt / (client["name"] + ".json")
            if (client.get("exit_code") != 0 or not client.get("completed_report")
                    or not bp.exists()):
                row["benchmarks"].append({"name": client["name"], "valid": False,
                                           "reason": "benchmark client did not complete successfully"})
                continue
            bench = json.loads(bp.read_text())
            if not bench.get("completed_at_utc") or bench.get("errors"):
                row["benchmarks"].append({"name": client["name"], "valid": False,
                                           "reason": "benchmark report incomplete or contains errors"})
                continue
            ref = {"name": client["name"], "valid": True, "sha256": digest(bp),
                   "settings": bench["settings"], "run_id": bench["run_id"]}
            row["benchmarks"].append(ref)
            for case, value in bench["cases"].items():
                runs = value["runs"]
                if not runs or any(run.get("phase") != "measured" for run in runs):
                    raise ValueError("Missing/invalid measured runs: " + label + "/" + case)
                medians = {key: med([run.get(key) for run in runs]) for key in METRICS}
                counts = []
                for run in runs:
                    usage = run.get("usage", {}).get("completion_tokens_details", {})
                    accepted = usage.get("accepted_prediction_tokens")
                    rejected = usage.get("rejected_prediction_tokens")
                    counts.append({"accepted": accepted, "rejected": rejected})
                entry = {"label": label, "benchmark": client["name"], "case": case,
                         "measured_requests": len(runs), "medians": medians,
                         "draft_accepted_tokens_median": med([c["accepted"] for c in counts]),
                         "draft_rejected_tokens_median": med([c["rejected"] for c in counts]),
                         "actual_prompt_tokens": [run["prompt_tokens"] for run in runs],
                         "actual_completion_tokens": [run["completion_tokens"] for run in runs],
                         "actual_cached_prompt_tokens": [run["cached_prompt_tokens"] for run in runs],
                         "settings": bench["settings"], "run_id": bench["run_id"],
                         "prompt_sha256": value["prompt_sha256"], "report_sha256": digest(bp),
                         "requests": [{"repeat_index": run["repeat_index"],
                                       "request_sha256": run["request_sha256"],
                                       "response_sha256": run["response_sha256"],
                                       **{key: run.get(key) for key in METRICS},
                                       "draft_accepted_tokens": count["accepted"],
                                       "draft_rejected_tokens": count["rejected"]}
                                      for run, count in zip(runs, counts)]}
                report["bench_rows"].append(entry)
        report["completed_jobs"].append(row)
        completed[label] = row

    controls = {}
    all_rows = {(x["label"], x["benchmark"], x["case"]): x for x in report["bench_rows"]}
    for job in expected:
        label = job["label"]
        key = (label.split("-", 1)[0], job["model_path"])
        if "-control-" in label:
            controls[key] = label
            continue
        reference_label = controls.get(key)
        if label not in completed or reference_label not in completed:
            continue
        current, control = completed[label], completed[reference_label]
        changes = {k: {"control": control["resolved_tuning"].get(k),
                       "variant": current["resolved_tuning"].get(k)}
                   for k in current["resolved_tuning"].keys() | control["resolved_tuning"].keys()
                   if current["resolved_tuning"].get(k) != control["resolved_tuning"].get(k)}
        valid = len(changes) == 1 and current["source_commits"] == control["source_commits"]
        comparison = {"control": reference_label, "variant": label, "changed_tuning": changes,
                      "single_knob_source_matched": valid,
                      "both_jobs_passed_all_clients": current["passed_all_clients"] and control["passed_all_clients"],
                      "cases": [], "concurrency": []}
        for entry in [x for x in report["bench_rows"] if x["label"] == label]:
            base = all_rows.get((reference_label, entry["benchmark"], entry["case"]))
            if not base:
                continue
            same_settings = entry["settings"] == base["settings"] and entry["run_id"] == base["run_id"]
            deltas = {}
            for metric in METRICS:
                new, old = entry["medians"][metric], base["medians"][metric]
                if new is not None and old is not None:
                    deltas[metric] = {"control": old, "variant": new,
                                      "absolute_delta": new - old,
                                      "percent_delta": 100 * (new / old - 1) if old else None}
            comparison["cases"].append({"benchmark": entry["benchmark"], "case": entry["case"],
                                        "matched_settings": same_settings, "deltas": deltas})
        for entry in [x for x in report["concurrency_rows"] if x["label"] == label]:
            base = next((x for x in report["concurrency_rows"]
                         if x["label"] == reference_label and x["concurrency"] == entry["concurrency"]), None)
            if base is None:
                continue
            old, new = base["end_to_end_tok_s_median"], entry["end_to_end_tok_s_median"]
            comparison["concurrency"].append({
                "concurrency": entry["concurrency"],
                "matched_settings": entry["settings"] == base["settings"] and entry["run_id"] == base["run_id"],
                "both_reports_valid": entry["valid"] and base["valid"],
                "end_to_end_tok_s": {"control": old, "variant": new,
                                     "percent_delta": 100 * (new / old - 1) if old and new else None},
                "per_stream_ttft_s": {"control": base["per_stream_medians"]["ttft_s"],
                                      "variant": entry["per_stream_medians"]["ttft_s"]},
            })
        report["comparisons"].append(comparison)
    report["completed_job_count"] = len(report["completed_jobs"])
    report["expected_job_count"] = len(expected)
    return report


def markdown(report):
    def number(value, decimals=2):
        return "—" if value is None else f"{value:.{decimals}f}"
    lines = ["# Completed primary performance jobs",
             "", f"Completed: **{report['completed_job_count']}/{report['expected_job_count']}** at {report['generated_at_utc']}.",
             "", "Historical engine16ca/Tabbyf4 measurements. A tool failure is retained and prevents treating that whole job as qualified.",
             "", "| Job | Benchmark/case | Prompt/output/cache tokens, median | Decode tok/s | Prefill tok/s | TTFT s | Draft acceptance | Accepted/rejected draft tokens, median |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["bench_rows"]:
        m = row["medians"]
        tokens = "/".join(number(m[k], 0) for k in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens"))
        acceptance = "—" if m["draft_acceptance"] is None else f"{100*m['draft_acceptance']:.2f}%"
        drafts = f"{number(row['draft_accepted_tokens_median'],0)}/{number(row['draft_rejected_tokens_median'],0)}"
        lines.append(f"| {row['label']} | {row['benchmark']}/{row['case']} | {tokens} | {number(m['server_decode_tok_s'])} | {number(m['server_prefill_tok_s'])} | {number(m['ttft_s'],3)} | {acceptance} | {drafts} |")
    lines += ["", "## Whole-job gates", "", "| Job | All clients pass | Tools pass/total |", "|---|---|---:|"]
    for job in report["completed_jobs"]:
        tools = job["tools"]
        count = f"{tools['passed']}/{tools['total']}" if tools else "—"
        lines.append(f"| {job['label']} | {job['passed_all_clients']} | {count} |")
    lines += ["", "## Paired settings", "", "| Control → variant | Only changed tuning | Case | Decode change | Prefill change | Both jobs pass |",
              "|---|---|---|---:|---:|---|"]
    for pair in report["comparisons"]:
        changes = "; ".join(f"{k}: {v['control']} → {v['variant']}" for k, v in sorted(pair["changed_tuning"].items()))
        for case in pair["cases"]:
            deltas = case["deltas"]
            decode = deltas.get("server_decode_tok_s", {}).get("percent_delta")
            prefill = deltas.get("server_prefill_tok_s", {}).get("percent_delta")
            lines.append(f"| {pair['control']} → {pair['variant']} | {changes} | {case['benchmark']}/{case['case']} | {number(decode)}% | {number(prefill)}% | {pair['both_jobs_passed_all_clients']} |")
    if report["concurrency_rows"]:
        lines += ["", "## Completed concurrency batches", "",
                  "| Job | Concurrent requests | Complete measured rounds | Prompt/output/cache per request, median | Aggregate end-to-end tok/s | TTFT per request s | Draft acceptance | Accepted/rejected draft tokens per request, median |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in report["concurrency_rows"]:
            m = row["per_stream_medians"]
            tokens = "/".join(number(m[k], 0) for k in ("prompt_tokens", "completion_tokens", "cached_prompt_tokens"))
            acceptance = "—" if m["draft_acceptance"] is None else f"{100*m['draft_acceptance']:.2f}%"
            drafts = f"{number(row['draft_accepted_tokens_median'],0)}/{number(row['draft_rejected_tokens_median'],0)}"
            lines.append(f"| {row['label']} | {row['concurrency']} | {row['complete_rounds']}/{row['measured_rounds']} | {tokens} | {number(row['end_to_end_tok_s_median'])} | {number(m['ttft_s'],3)} | {acceptance} | {drafts} |")
        lines += ["", "| Control → variant | Concurrent requests | Control aggregate tok/s | Variant aggregate tok/s | Change | Both jobs pass |",
                  "|---|---:|---:|---:|---:|---|"]
        for pair in report["comparisons"]:
            for row in pair["concurrency"]:
                rates = row["end_to_end_tok_s"]
                lines.append(f"| {pair['control']} → {pair['variant']} | {row['concurrency']} | {number(rates['control'])} | {number(rates['variant'])} | {number(rates['percent_delta'])}% | {pair['both_jobs_passed_all_clients']} |")
    lines += ["", "## Scope", ""] + ["- " + note for note in report["notes"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.reports, args.jobs)
    args.output.mkdir(mode=0o700)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.md").write_text(markdown(report))
    (args.output / "summarizer.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"completed": report["completed_job_count"], "expected": report["expected_job_count"],
                      "bench_rows": len(report["bench_rows"]), "concurrency_rows": len(report["concurrency_rows"]), "comparisons": len(report["comparisons"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
