#!/usr/bin/env python3
"""Finalize only a completed optional16 archive; never edits the main report."""
from pathlib import Path
import datetime as dt
import hashlib
import json

BASE = Path("/home/vcruz/src/qwen-overnight-20261008")
A = BASE / "recipe/bench/results/2026-10-08/performance-optional"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def main():
    summary = json.loads((A / "summary/summary.json").read_text())
    records = {p.parent.name: json.loads(p.read_text()) for p in (A / "reports").glob("*/result.json")}
    jobs = json.loads((A / "inputs/optional16.json").read_text())
    assert set(records) == {j["label"] for j in jobs} and len(records) == 16
    assert summary["completed_job_count"] == summary["expected_job_count"] == 16
    assert not summary["excluded_jobs"]
    assert len(summary["bench_rows"]) == 36 and len(summary["concurrency_rows"]) == 18
    assert all(r["valid"] and r["complete_rounds"] == 3 for r in summary["concurrency_rows"])
    checks, logs, memory, failures = [], [], [], []
    measured, tools_total, tools_passed = 0, 0, 0
    for label, result in sorted(records.items()):
        attempt = A / "reports" / label / Path(result["attempt"]).name
        assert result["state"] == "completed" and result["finished_at_utc"]
        assert result["server_exit_before_cleanup"] is None
        assert result["server_cleanup"] == {"exit_code": 0, "signals": ["SIGTERM"]}
        assert not result.get("cleanup_error") and not result.get("cleanup_failed")
        assert (A / "reports" / label / "result.json").read_bytes() == (attempt / "result.json").read_bytes()
        deployment = json.loads((attempt / "deployment.json").read_text())
        assert canonical(deployment) == result["deployment_sha256"]
        assert canonical(result["configuration"]) == result["config_sha256"]
        assert all(result.get("inputs_verified_" + phase + "_at_utc") for phase in
                   ("before_launch", "before_measurements", "after_measurements"))
        sources = result["configuration"]["sources"]
        assert all(sources[k]["tracked_changes"] == "" for k in ("recipe", "engine", "server"))
        assert sources["engine"]["commit"] == "16ca20d27c0e4cce15a9bbc131e6d047065395b5"
        assert sources["server"]["commit"] == "f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6"
        assert sources["recipe"]["commit"] == "218bd438225243e795383f10ef7a886b06d5d839"
        assert sources["controller_sha256"] == sha(A / "inputs/spark_experiment_controller.py")
        for rel, expected in sources["files_sha256"].items():
            assert sha(A / "inputs/recipe" / rel) == expected
        for client in result["clients"]:
            assert client["completed_report"] and client["exit_code"] in (0, 1)
            report_path = attempt / Path(client["output"]).name
            report = json.loads(report_path.read_text())
            if client["name"].startswith("bench-"):
                assert client["exit_code"] == 0 and not report["errors"]
                measured += sum(len(case["runs"]) for case in report["cases"].values())
            elif client["name"] == "tools":
                assert report.get("completed_at_utc") and not report["errors"]
                assert len(report["results"]) == report["summary"]["total"] == 8
                assert sum(x["passed"] is True for x in report["results"]) == report["summary"]["passed"]
                assert (client["exit_code"] == 0) == (report["summary"]["failed"] == 0)
                tools_total += report["summary"]["total"]
                tools_passed += report["summary"]["passed"]
                failures.extend({"label": label, "case": item["case"], "mode": item["mode"],
                                 "errors": item["errors"], "report": str(report_path.relative_to(A))}
                                for item in report["results"] if not item["passed"])
            elif client["name"] == "concurrency":
                assert client["exit_code"] == 0
            else:
                raise AssertionError("Unexpected client " + client["name"])
        log_path = attempt / "server.log"
        lines = log_path.read_text().splitlines()
        selected = [{"line": i + 1, "text": line} for i, line in enumerate(lines)
                    if any(word in line.lower() for word in
                           ("warning", "error", "exception", "traceback",
                            "application startup complete", "finished server process"))]
        logs.append({"label": label, "server_log": str(log_path.relative_to(A)), "sha256": sha(log_path),
                     "line_count": len(lines), "selected_lines": selected[:30],
                     "selected_line_count": len(selected), "load_wall_seconds": result["load_wall_seconds"],
                     "server_cleanup": result["server_cleanup"]})
        resource_path = attempt / "resources.json"
        resource = json.loads(resource_path.read_text())
        samples = [x for x in resource["samples"] if isinstance(x.get("system_kib"), dict)]
        assert samples and not resource["limit_reached"]
        swaps = [(x["system_kib"]["SwapTotal"] - x["system_kib"]["SwapFree"]) / 1024**2 for x in samples]
        memory.append({"label": label, "resources": str(resource_path.relative_to(A)),
                       "sha256": sha(resource_path), "interval_seconds": resource["interval_seconds"],
                       "samples": len(samples), "initial_phase": samples[0]["phase"],
                       "min_mem_available_gib": min(x["system_kib"]["MemAvailable"] for x in samples) / 1024**2,
                       "initial_swap_used_gib": swaps[0], "max_swap_used_gib": max(swaps),
                       "max_increase_from_initial_swap_gib": max(swaps) - swaps[0]})
        checks.append({"label": label, "whole_job_passed": result["passed"],
                       "all_clients_complete": True, "identity_rechecked_after_measurements": True,
                       "canonical_config_and_deployment_hashes_match": True,
                       "duplicated_final_results_identical": True, "cleanup_exit_code": 0,
                       "controller_sent_sigterm": True})
    concurrent = sum(len(row["actual_completion_tokens"]) for row in summary["concurrency_rows"])
    assert measured == 102 and concurrent == 126 and tools_total == 104
    assert sum(bool(j.get("tools")) for j in jobs) == 13
    contracts = [case for pair in summary["comparisons"] for case in pair["cases"] + pair["concurrency"]]
    assert contracts and all(case["comparison_contract_met"] for case in contracts)
    info = {"schema_version": 1, "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "jobs": 16, "whole_jobs_passed": sum(x["whole_job_passed"] for x in checks),
            "whole_jobs_failed": sum(not x["whole_job_passed"] for x in checks),
            "single_request_measured_requests": measured, "complete_concurrency_measured_requests": concurrent,
            "tools_passed": tools_passed, "tools_total": tools_total,
            "jobs_with_tool_checks": sum(bool(j.get("tools")) for j in jobs),
            "jobs_without_tool_checks": [j["label"] for j in jobs if not j.get("tools")],
            "first_job_started_at_utc": min(x["started_at_utc"] for x in records.values()),
            "last_job_finished_at_utc": max(x["finished_at_utc"] for x in records.values()),
            "matched_case_and_concurrency_contracts": len(contracts), "checks": checks,
            "notes": ["Canonical JSON hashes in controller records are distinguished from raw SHA256SUMS.",
                      "No new tensor hashing or GPU/API calls were performed by evidence collection.",
                      "Failed functional jobs remain failures; complete timing rows are observations only."]}
    write = lambda name, obj: (A / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    write("integrity.json", info)
    write("server-log-summary.json", logs)
    write("memory-summary.json", {"scope": "Sparse 10-second samples, not true peaks or swap I/O. Initial swap is retained; later allocation can be inherited from earlier jobs.", "jobs": memory})
    write("tool-failures.json", failures)
    print(json.dumps({k: v for k, v in info.items() if k != "checks"}, indent=2))

if __name__ == "__main__":
    main()
