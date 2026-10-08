#!/usr/bin/env python3
"""Summarize immutable, completed API reports; this utility makes no API calls."""
from collections import Counter
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import statistics

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text())

def stats(values):
    values = list(values)
    return {"median": statistics.median(values), "minimum": min(values),
            "maximum": max(values), "samples": len(values)}

def bench_rows(label, report, original, original_path):
    rows = []
    assert not report["errors"] and report.get("completed_at_utc")
    for name, case in report["cases"].items():
        runs = case["runs"]
        assert len(runs) == 3 and len(case["warmup"]) == 1
        assert [v["repeat_index"] for v in runs] == [0, 1, 2]
        row = {"label": label, "case": name, "measured_requests": len(runs),
               "warmup_requests": len(case["warmup"]),
               "decode_tok_s": stats(v["server_decode_tok_s"] for v in runs),
               "ttft_s": stats(v["ttft_s"] for v in runs),
               "end_to_end_tok_s": stats(v["client_end_to_end_tok_s"] for v in runs),
               "prompt_tokens": [v["prompt_tokens"] for v in runs],
               "cached_prompt_tokens": [v["cached_prompt_tokens"] for v in runs],
               "completion_tokens": [v["completion_tokens"] for v in runs],
               "finish_reasons": [v["finish_reason"] for v in runs],
               "report_settings": report["settings"],
               "original": None}
        if original is not None:
            old = original["cases"][name]["runs"]
            assert len(old) == len(runs) == 3
            request_fields = ("request_sha256", "repeat_index")
            token_fields = ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "finish_reason")
            matches = {
                "request_sha256_and_repeat_order": all(
                    all(a.get(k) == b.get(k) for k in request_fields) for a,b in zip(old,runs)),
                "actual_token_counts_and_finish": all(
                    all(a.get(k) == b.get(k) for k in token_fields) for a,b in zip(old,runs)),
                "client_settings": original["settings"] == report["settings"],
                "run_id": original["run_id"] == report["run_id"],
                "prompt_text_sha256": original["cases"][name]["prompt_sha256"] == case["prompt_sha256"],
                "response_sha256": all(a["response_sha256"] == b["response_sha256"] for a,b in zip(old,runs))}
            old_stats = stats(v["server_decode_tok_s"] for v in old)
            row["original"] = {
                "path": original_path.name, "sha256": digest(original_path),
                "decode_tok_s": old_stats,
                "prompt_tokens": [v["prompt_tokens"] for v in old],
                "completion_tokens": [v["completion_tokens"] for v in old],
                "matches": matches,
                "decode_median_change_percent": 100*(row["decode_tok_s"]["median"]/old_stats["median"]-1),
                "per_request": [
                    {"repeat_index": a["repeat_index"],
                     "original_request_sha256": a["request_sha256"],
                     "final_request_sha256": b["request_sha256"],
                     "original_response_sha256": a["response_sha256"],
                     "final_response_sha256": b["response_sha256"]}
                    for a,b in zip(old,runs)]}
        rows.append(row)
    return rows

def summarize(archive, baselines):
    index = read(archive/"archive-status.json")
    assert index["kind"] == "completed-final-api-archive"
    declared = read(archive/"inputs/final-api-jobs-selected.json")
    expected = {j["label"]: j for j in declared}
    result = {
        "schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collection_complete": index["collection_complete"],
        "completed_jobs": len(index["completed_labels"]), "declared_jobs": len(declared),
        "jobs": [], "benchmark_rows": [], "concurrency_benchmark_rows": [],
        "functional_check_totals": {}, "client_suite_totals": {},
        "notes": [
            "Only batch-recorded completed jobs enter this summary; completion does not mean every client passed.",
            "Functional cases are assertions, not a count of HTTP requests. Literal failures remain failures.",
            "Decode medians use three measured 400-token requests after one warmup per case.",
            "Original comparisons expose request, token-count, response and setting identity separately.",
            "Equal token counts alone do not prove identical rendered prompt IDs; Cyber has an explicit template correction.",
            "No statistical significance or general task-quality claim follows from this small deterministic workload.",
            "Resource minima/maxima are sparse ten-second observations, not guaranteed peaks or swap-I/O measurements.",
            "Full model tensor hashes were not computed; size, mtime, loader order and small configuration hashes are retained."]}
    functional = Counter()
    suites = {}
    for label in index["completed_labels"]:
        path = archive/"reports"/label
        meta = read(path/"result.json")
        assert meta.get("finished_at_utc") and meta["state"] in ("completed", "failed")
        assert meta["server_cleanup"]["owned_group_empty"] is True
        assert digest(path/"result.json") == index["files"][f"reports/{label}/result.json"]["sha256"]
        deployment = read(path/"deployment.json")
        assert digest(path/"deployment.json") == meta["deployment_sha256"]
        assert digest(path/"state/config.yml") == deployment["config"]["sha256"]
        selected = expected[label]
        for field, key in (("engine","engine"),("tabby","server")):
            assert selected[field] == meta["sources"][key]["commit"]
            assert deployment[key]["commit"] == selected[field]
        clients = []
        for client in meta["clients"]:
            report_path = path/Path(client["report"]).name
            assert digest(report_path) == client["report_sha256"]
            report = read(report_path)
            name = client["name"]
            row = {key: client.get(key) for key in (
                "name","passed","exit_code","counts","expected_checks","observed_checks",
                "completed_report","unique_expected_cases","summary_consistent")}
            row.update(report=str(report_path.relative_to(archive)), report_sha256=digest(report_path))
            if name == "tools":
                cases = report["results"]
                observed = Counter("pass" if v["passed"] else "fail" for v in cases)
                row["failed_cases"] = [{"case":v["case"],"mode":v["mode"],"errors":v["errors"]}
                                       for v in cases if not v["passed"]]
            elif "expected_checks" in client:
                cases = report["cases"]
                observed = Counter(v["status"] for v in cases)
                row["failed_cases"] = [v for v in cases if v["status"] != "pass"]
            else:
                observed = None
            if observed is not None:
                assert dict(observed) == client["counts"], (label,name,observed,client["counts"])
                assert sum(observed.values()) == client["observed_checks"]
                functional.update(observed)
                suite = "auto" if name.startswith("auto-") else name
                suites.setdefault(suite,Counter()).update(observed)
                row["independently_counted_cases"] = dict(observed)
            if isinstance(report.get("requests"),list):
                row["recorded_request_entries"] = len(report["requests"])
            if name == "bench":
                original_path = None
                if label.endswith("-single"):
                    pack = label.removeprefix("final-").removesuffix("-single")
                    original_path = baselines/f"baseline-{pack}.json"
                original = read(original_path) if original_path else None
                result["benchmark_rows"].extend(bench_rows(label,report,original,original_path))
            if name == "concurrency":
                children = report["client_reports"]
                assert len(children) == 16 and report["include_unbudgeted"] is True
                child_counts = Counter()
                proofs = []
                for child in children:
                    child_path = path/"concurrency-cases"/Path(child["report"]).name
                    assert digest(child_path) == child["report_sha256"]
                    assert child["cleanup_attempted"] and child["owned_client_group_empty"]
                    assert child["cleanup"]["owned_group_empty"]
                    assert child["cleanup"]["exit_code"] == child["exit_code"]
                    child_counts.update(child["counts"])
                    proofs.append({key:child[key] for key in
                        ("name","pid","report_sha256","exit_code","counts","cleanup",
                         "owned_client_group_empty")})
                assert dict(child_counts) == report["summary"]
                row["children"] = proofs
                row["children_owned_groups_empty"] = True
            if name == "concurrency-bench":
                assert not report["errors"] and report.get("completed_at_utc")
                for level, group in report["results"].items():
                    rounds = group["rounds"]
                    assert len(rounds)==3 and len(group["warmup"])==1
                    assert all(v["complete"] and not v["errors"] for v in rounds+group["warmup"])
                    result["concurrency_benchmark_rows"].append({
                        "label":label,"concurrency":int(level),"summary":group["summary"],
                        "measured_rounds":len(rounds),"measured_requests":sum(len(v["streams"]) for v in rounds),
                        "warmup_requests":sum(len(v["streams"]) for v in group["warmup"]),
                        "settings":report["settings"],"run_id":report["run_id"]})
            if name.startswith("long-"):
                row["long_context"] = {key: report.get(key) for key in
                    ("context_tokens_requested","actual_prompt_tokens","max_cached_tokens","run_id",
                     "matches","passed","retrieval_passed","errors","completed_at_utc")}
                row["response"] = report.get("response")
            clients.append(row)
        if meta["passed"]:
            assert all(v["passed"] for v in meta["clients"])
        resources = read(archive/"reports"/f"{label}-resources.json")
        mem = [v["system_kib"] for v in resources["samples"] if "system_kib" in v]
        available = [v["MemAvailable"]/1048576 for v in mem if "MemAvailable" in v]
        swap = [(v["SwapTotal"]-v["SwapFree"])/1048576 for v in mem if "SwapFree" in v and "SwapTotal" in v]
        result["jobs"].append({
            "label":label,"state":meta["state"],"passed_all_scheduled_clients":meta["passed"],
            "started_at_utc":meta["started_at_utc"],"finished_at_utc":meta["finished_at_utc"],
            "load_wall_seconds":meta["load_wall_seconds"],"sources":meta["sources"],
            "resolved_tuning":meta["resolved_tuning"],"model":deployment["model"],
            "prompt_template_override":deployment.get("prompt_template_override"),
            "server_cleanup":meta["server_cleanup"],"clients":clients,
            "resources":{"interval_seconds":resources["interval_seconds"],"sample_count":len(resources["samples"]),
                "min_sampled_memavailable_gib":min(available) if available else None,
                "initial_sampled_allocated_swap_gib":swap[0] if swap else None,
                "max_sampled_allocated_swap_gib":max(swap) if swap else None}})
    result["functional_check_totals"] = dict(functional)
    result["client_suite_totals"] = {key:dict(value) for key,value in suites.items()}
    result["jobs_passing_all_scheduled_clients"] = sum(v["passed_all_scheduled_clients"] for v in result["jobs"])
    result["measured_single_benchmark_requests"] = sum(v["measured_requests"] for v in result["benchmark_rows"])
    result["warmup_single_benchmark_requests"] = sum(v["warmup_requests"] for v in result["benchmark_rows"])
    result["measured_concurrent_benchmark_requests"] = sum(v["measured_requests"] for v in result["concurrency_benchmark_rows"])
    result["warmup_concurrent_benchmark_requests"] = sum(v["warmup_requests"] for v in result["concurrency_benchmark_rows"])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument("--baselines",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    baseline=a.baselines or a.archive.parent/"original-api"
    result=summarize(a.archive,baseline)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in (
        "collection_complete","completed_jobs","declared_jobs","jobs_passing_all_scheduled_clients",
        "functional_check_totals","client_suite_totals","measured_single_benchmark_requests",
        "measured_concurrent_benchmark_requests")},indent=2))
if __name__=="__main__":
    main()
