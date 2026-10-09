#!/usr/bin/env python3
"""Bounded external capture of the eight unchanged literal-tool fixtures.

Reuses the hash-frozen final owned lifecycle and strings observer attachment.
Capture integrity and model-value outcomes are separate. No execution is
authorized merely by preparing this script; root schedules the actual host run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import os
import re
import signal
import types

BASE_SHA = "cfc3619894c4ee8f07d46c2ee7ed0fd085f3fdc044c81c8d9a01d009304c4116"
WRAPPER_SHA = "ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d"
CLIENT_SHA = "8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054"
OBSERVER_SHA = "674453a31893defd299939fe801cd58b48bdb20c7bbfaf0ec0c0fbb5f82418f6"
SITE_SHA = "3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f"
WORKAROUND_SHA = "789408e144d9046fdbd11cc2868886d5bb14cdc1e8bf365253105109725a7243"
ENGINE = "24f0dece34f09c8d1e2359d6b3b3f7befef7331b"
TABBY = "5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb"
PROMPT_SHA = "32655bc461bfa9685942882754b89e75f6640a5605004d4a3d609ebfc6076f58"

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    return digest(Path(path).read_bytes())

def require(value, message):
    if not value:
        raise ValueError(message)

def load_exact(path, expected, name):
    raw = path.read_bytes()
    require(re.fullmatch("[0-9a-f]{64}", expected) and digest(raw) == expected,
            "Exact diagnostic source hash mismatch: " + str(path))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module

def prepare_bundle(lib, observer_dir, bundle, output, job, runtime):
    sources = {"strings_observer.py": OBSERVER_SHA, "sitecustomize.py": SITE_SHA}
    for name, expected in sources.items():
        require(sha(observer_dir / name) == expected, "Bounded observer source changed")
    bundle.mkdir(mode=0o700)
    for name in sources:
        (bundle / name).write_bytes((observer_dir / name).read_bytes())
    config = {
        "engine_repo": str(runtime / "exllamav3"), "tabby_repo": str(runtime / "tabbyAPI"),
        "engine_commit": job["engine"], "tabby_commit": job["tabby"],
        "run_label": job["label"], "max_records": 8, "output_dir": str(output / "raw-literal"),
    }
    lib.atomic_new(bundle / "observer-config.json", config)
    return {
        "derived_observer_sha256": OBSERVER_SHA, "sitecustomize_sha256": SITE_SHA,
        "config": config,
        "files_sha256": {str(p): sha(p) for p in sorted(bundle.iterdir())},
        "note": "This separately reviewed literal-only observer is copied byte-for-byte; no runtime source is modified.",
    }

def attach(parent, lib, bundle, guard, provenance, workaround):
    commands = parent.commands
    # This function supplies the reviewed server-only environment attachment and
    # launch/cleanup signal guard. Restore the already configured literal-only
    # commands immediately; its two-string commands are never executed.
    lib.attach_observation(parent, bundle, guard, provenance)
    parent.commands = commands
    parent.EXPECTED = {"literal": 4, "literal-unbudgeted": 4}
    parent.EXTRA_FILES += (str(Path(__file__).resolve()),)
    if workaround:
        parent.EXPECTED["literal-thinking-off"] = 4
        parent.EXTRA_FILES += (str(workaround),)
        def with_supplement(with_bench):
            require(not with_bench, "A raw diagnostic never measures throughput")
            yield from commands(False)
            yield "literal-thinking-off", [
                str(parent.RUNTIME / "venv/bin/python"), str(workaround),
                "--recipe", str(parent.RECIPE), "--base-url", parent.BASE,
                "--model", parent.ALIAS, "--metadata", str(parent.OUTPUT / "deployment.json"),
                "--label", "supplementary-literal-thinking-off", "--unbudgeted",
            ]
        parent.commands = with_supplement

def response_id(trace):
    require(trace.get("status") == 200 and not trace.get("client_aborted"),
            "The bounded response is not a complete HTTP200 result")
    if trace.get("stream"):
        frames = trace.get("frames", [])
        require(frames and trace.get("done") is True, "Incomplete SSE response")
        ids = {frame.get("id") for frame in frames}
        require(len(ids) == 1, "SSE response ID changed")
        identifier = next(iter(ids))
    else:
        identifier = trace.get("body", {}).get("id")
    require(isinstance(identifier, str) and identifier, "Missing API response ID")
    return identifier

def validate_client(report, client, fixture, alias, unbudgeted):
    require(report.get("finished_utc") and report.get("source_sha256") == CLIENT_SHA
            and report.get("unbudgeted") is unbudgeted and report.get("expected_checks") == 4,
            "Original literal client report is incomplete or its source/mode changed")
    rows, requests = report.get("cases", []), report.get("requests", [])
    require(len(rows) == 4 and len(requests) == 4, "Exactly four original literal requests are required")
    names = fixture.expected_names()
    require([row.get("name") for row in rows] == names, "Original literal case set/order changed")
    result = []
    for index, row in enumerate(rows):
        _, _, choice, mode = row["name"].split("_")
        streaming = mode == "stream"
        expected = fixture.payload(alias, choice, streaming, unbudgeted=unbudgeted)
        request = requests[index]
        require(row.get("request_indices") == [index] and request.get("index") == index
                and request.get("path") == "/chat/completions" and request.get("request") == expected
                and request.get("stream") is streaming, "Original request payload/index changed")
        require(row.get("status") in {"pass", "fail"}, "Invalid semantic status")
        result.append({"key": (unbudgeted, choice, streaming), "request": expected,
                       "id": response_id(request), "status": row["status"]})
    counts = dict(Counter(row["status"] for row in result))
    passed = counts.get("pass", 0) == 4
    require(report.get("summary") == counts and report.get("passed") is passed,
            "Original semantic summary is inconsistent")
    require(client.get("passed") is passed and client.get("exit_code") == (0 if passed else 1),
            "Client lifecycle result disagrees with semantic results")
    return result

def validate_workaround(report, client, fixture, alias):
    require(report.get("finished_utc") and report.get("source_sha256") == WORKAROUND_SHA
            and report.get("kind") == "supplementary_literal_thinking_off"
            and report.get("unbudgeted") is True and report.get("expected_checks") == 4,
            "Supplementary thinking-off report is incomplete")
    rows, requests = report.get("cases", []), report.get("requests", [])
    names = [name.replace("literal_reasoning_", "literal_copy_") for name in fixture.expected_names()]
    require(len(rows) == 4 and len(requests) == 4 and [r.get("name") for r in rows] == names,
            "Supplementary case set/order changed")
    ids = []
    for index, row in enumerate(rows):
        _, _, choice, mode = row["name"].split("_")
        expected = fixture.payload(alias, choice, mode == "stream", unbudgeted=True)
        expected["enable_thinking"] = False
        request = requests[index]
        require(row.get("request_indices") == [index] and request.get("index") == index
                and request.get("path") == "/chat/completions" and request.get("request") == expected,
                "Supplementary request changed more than enable_thinking")
        require(row.get("status") in {"pass", "fail"}, "Invalid supplementary outcome")
        ids.append(response_id(request))
    require(len(set(ids)) == 4, "Supplementary response IDs are not unique")
    counts = dict(Counter(row["status"] for row in rows))
    passed = counts.get("pass", 0) == 4
    require(report.get("summary") == counts and report.get("passed") is passed
            and client.get("passed") is passed and client.get("exit_code") == (0 if passed else 1),
            "Supplementary semantic/lifecycle summary differs")
    return counts


def validate_trace(trace, index, expected, observer, alias, label):
    require(trace.get("index") == index and trace.get("run_label") == label,
            "Trace index or run identity changed")
    key = expected["key"]
    identifier = trace.get("request_id")
    prefix = "chatcmpl-" if key[2] else "cmpl-"
    require(isinstance(identifier, str) and prefix + identifier == expected["id"],
            "Native collector ID does not match the actual API response")
    require(trace.get("streaming_mode") is key[2] and trace.get("start_in_reasoning_mode") is True,
            "The original reasoning/mode scope was not exercised")
    require(trace.get("raised_exception_type") is None
            and trace.get("collector_returned_error") is False
            and trace.get("skipped_matching_requests_so_far") == 0,
            "Original collector failed or capture exceeded its strict bound")
    prompt = trace.get("rendered_prompt")
    require(isinstance(prompt, str) and digest(prompt.encode()) == PROMPT_SHA
            and trace.get("rendered_prompt_sha256") == PROMPT_SHA,
            "The actual original rendered prompt changed")
    matched = trace.get("matched_request", {})
    wire = expected["request"]
    for field in ("model", "messages", "tools", "tool_choice"):
        require(matched.get(field) == wire[field], "Retained request field changed: " + field)
    require(matched.get("reasoning_budget_tokens") == wire.get("reasoning_budget_tokens"),
            "Retained reasoning-budget mode changed")
    require(trace.get("variant") == {"unbudgeted": key[0], "choice": key[1], "stream": key[2]},
            "Observed literal variant metadata changed")
    for field, value in {"parallel_tool_calls": False, "max_tokens": 256, "temperature": 0,
                         "top_k": 1, "top_p": 0.95, "n": 1, "stream": key[2]}.items():
        require(matched.get(field) == value, "Observed sampler/control changed: " + field)
    normalized = types.SimpleNamespace(**wire, template_vars=matched.get("template_vars"),
                                       n=1, functions=None, top_p=0.95)
    require(observer.matching_request(normalized, key[2]), "Retained observer request is out of scope")
    finish = trace.get("raw_finish")
    require(isinstance(finish, dict) and finish.get("native_full_completion_present") is True
            and isinstance(finish.get("native_full_completion"), str)
            and isinstance(finish.get("backend_full_response"), str),
            "Both complete native and backend strings are required")
    times = [trace.get("started_monotonic_ns"), finish.get("backend_finish_monotonic_ns"),
             trace.get("finished_monotonic_ns")]
    require(all(type(t) is int and t > 0 for t in times) and times == sorted(times),
            "Incomplete capture lifecycle times")
    return {
        "key": list(key), "request_id": identifier, "semantic_status": expected["status"],
        "native_sha256": digest(finish["native_full_completion"].encode()),
        "backend_sha256": digest(finish["backend_full_response"].encode()),
        "native_equals_backend": finish["native_full_completion"] == finish["backend_full_response"],
    }

def assess(output, job, provenance, observer, fixture, alias, lifecycle_exit, include_workaround):
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    require(result.get("state") == "completed" and result.get("finished_at_utc")
            and not result.get("cleanup_failed") and not result.get("error"),
            "Owned lifecycle did not finish cleanly")
    cleanup = result.get("server_cleanup", {})
    require(cleanup.get("owned_group_empty") is True and cleanup.get("unexpected_exit") is False,
            "Owned server cleanup is absent or unexpected")
    require(result.get("expected_engine") == ENGINE and result.get("expected_server") == TABBY,
            "Lifecycle source pins changed")
    clients = result.get("clients", [])
    wanted = ["literal", "literal-unbudgeted"] + (["literal-thinking-off"] if include_workaround else [])
    require([c.get("name") for c in clients] == wanted, "Wrong diagnostic client set/order")
    expected, report_hashes, supplementary_counts = [], {}, None
    for client in clients:
        name = client["name"]
        path = output / (name + ".json")
        require(client.get("cleanup", {}).get("owned_group_empty") is True
                and not client.get("timed_out") and client.get("report") == str(path)
                and client.get("report_sha256") == sha(path), "Client report/ownership/deadline invalid")
        report_hashes[name] = sha(path)
        report = json.loads(path.read_text())
        if name != "literal-thinking-off":
            expected += validate_client(report, client, fixture, alias, name == "literal-unbudgeted")
        else:
            supplementary_counts = validate_workaround(report, client, fixture, alias)
    require(len({row["key"] for row in expected}) == 8
            and len({row["id"] for row in expected}) == 8, "Original combinations/IDs are not unique")
    original_counts = dict(Counter(row["status"] for row in expected))
    original_passed = original_counts.get("pass", 0) == 8
    all_passed = all(c.get("passed") is True for c in clients)
    require(result.get("passed") is all_passed and lifecycle_exit == (0 if all_passed else 1),
            "Lifecycle exit/summary disagrees with separate semantic outcomes")
    raw = output / "raw-literal"
    require({p.name for p in raw.iterdir()} == {"observer-manifest.json"} |
            {f"request-{i:02d}.json" for i in range(8)}, "Exactly eight raw traces are required")
    manifest_path = raw / "observer-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("max_records") == 8 and manifest.get("run_label") == job["label"],
            "Observer manifest scope changed")
    source = manifest.get("source", {})
    for name, head in (("engine", ENGINE), ("tabby", TABBY)):
        require(source.get(name, {}).get("commit") == head
                and source[name].get("path") == provenance["config"][name + "_repo"]
                and source[name].get("tracked_changes") == "", "Observer source identity changed")
    config_raw = (json.dumps(provenance["config"], indent=2, ensure_ascii=False) + "\n").encode()
    require(source.get("observer_sha256") == OBSERVER_SHA
            and source.get("sitecustomize_sha256") == SITE_SHA
            and source.get("config_sha256") == digest(config_raw), "Observer source/config hashes changed")
    traces = []
    for index, row in enumerate(expected):
        path = raw / f"request-{index:02d}.json"
        trace = validate_trace(json.loads(path.read_text()), index, row, observer, alias, job["label"])
        traces.append({**trace, "path": str(path), "sha256": sha(path)})
    for path, expected_sha in provenance["files_sha256"].items():
        require(sha(path) == expected_sha and result.get("extra_files_sha256", {}).get(path) == expected_sha,
                "Observed source input changed during generation")
    require(sha(output / "deployment.json") == result.get("deployment_sha256"), "Deployment sidecar changed")
    return {
        "capture_valid": True, "original_semantic_passed": original_passed,
        "original_semantic_counts": original_counts, "all_clients_passed": all_passed,
        "supplementary_thinking_off_passed": clients[-1].get("passed") if include_workaround else None,
        "supplementary_thinking_off_counts": supplementary_counts,
        "traces": traces, "client_report_sha256": report_hashes,
        "lifecycle_sha256": sha(result_path), "observer_manifest_sha256": sha(manifest_path),
        "scope": "Original eight captures remain independent from the optional thinking-off outcomes.",
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--setup", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--observer-dir", type=Path, required=True)
    ap.add_argument("--include-thinking-off", action="store_true")
    ap.add_argument("--ready-timeout", type=float, default=600)
    ap.add_argument("--client-timeout", type=float, default=900)
    args = ap.parse_args(argv)
    require(0 < args.ready_timeout <= 1800 and 0 < args.client_timeout <= 1200, "Invalid bounded timeouts")
    root = Path(__file__).resolve().parent
    lib = load_exact(root / "strings_only_controller.py", BASE_SHA, "literal_frozen_attachment")
    wrapper = load_exact(root / "final_api_controller.py", WRAPPER_SHA, "literal_final_configuration")
    fixture = load_exact(root / "reasoning_literal_smoke.py", CLIENT_SHA, "literal_unchanged_fixture")
    args.job = args.job.resolve(strict=True); args.setup = args.setup.resolve(strict=True)
    args.observer_dir = args.observer_dir.resolve(strict=True); args.output = args.output.resolve()
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing existing output")
    job = json.loads(args.job.read_text())
    require(isinstance(job, dict) and not set(job) - {"label", "engine", "tabby", "model_path", "recipe", "env"},
            "Only exact source/model/tuning fields are accepted")
    require(job.get("engine") == ENGINE and job.get("tabby") == TABBY, "Only frozen24f0/5a may be observed")
    require(re.fullmatch("[a-z0-9][a-z0-9-]{0,63}", job.get("label", "")), "Invalid diagnostic run label")
    require(str(job["env"].get("MAX_BATCH_SIZE")) == "1"
            and str(job["env"].get("NGRAM_RAM")).lower() == "true", "This diagnostic requires one slot and RAM mode")
    parent = wrapper.load_parent(root / "api_f4_gemm_controller.py")
    configured = copy.deepcopy(job)
    configured.update(api=False, literal_client=str(root / "reasoning_literal_smoke.py"))
    wrapper.configure(parent, configured, args.job, args.output, args.setup)
    parent.check_setup(json.loads(args.setup.read_text()))
    workaround = root / "literal_thinking_off_smoke.py" if args.include_thinking_off else None
    if workaround:
        require(sha(workaround) == WORKAROUND_SHA, "Supplementary client source changed")
    os.umask(0o077)
    bundle = args.output.with_name(args.output.name + ".observer-inputs")
    provenance = prepare_bundle(lib, args.observer_dir, bundle, args.output, job, parent.RUNTIME)
    observer = load_exact(bundle / "strings_observer.py", OBSERVER_SHA, "literal_selected_observer")
    require(observer.ENGINE_HEAD == ENGINE and observer.TABBY_HEAD == TABBY, "Observer binding differs")
    guard = lib.SignalGuard()
    attach(parent, lib, bundle, guard, provenance, workaround)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, guard.interrupt)
    try:
        lifecycle_exit = parent.run(types.SimpleNamespace(
            bench=False, ready_timeout=args.ready_timeout, client_timeout=args.client_timeout))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = {
        "schema_version": 1, "diagnostic_only": True, "passed": False, "capture_valid": False,
        "controller_sha256": sha(__file__), "base_attachment_sha256": BASE_SHA,
        "final_wrapper_sha256": WRAPPER_SHA, "lifecycle_exit_code": lifecycle_exit,
        "job_sha256": sha(args.job), "interruption_observed": guard.pending,
        "provenance": provenance,
    }
    try:
        summary.update(assess(args.output, job, provenance, observer, fixture, parent.ALIAS,
                              lifecycle_exit, args.include_thinking_off))
        summary["passed"] = summary["capture_valid"] and summary["all_clients_passed"]
    except Exception as error:
        summary["capture_error"] = type(error).__name__ + ": " + str(error)
    lib.atomic_new(args.output / "literal-capture.json", summary)
    print(json.dumps({k: summary.get(k) for k in (
        "capture_valid", "original_semantic_counts", "supplementary_thinking_off_passed", "capture_error")}),
        flush=True)
    return 0 if summary["passed"] else 1 if summary["capture_valid"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
