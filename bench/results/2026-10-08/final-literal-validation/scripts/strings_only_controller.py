#!/usr/bin/env python3
"""Capture the two unchanged strings requests on one explicitly selected runtime.

Reuse the frozen owned API lifecycle and a hash-selected final configure adapter.
Only the server receives the bounded observer. Capture integrity and model
semantics are reported separately; no setup, source edits, retry or GPU action
occurs until this controller is explicitly executed on the model host.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import tempfile
import types

PARENT_SHA = "78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589"
OBSERVER_SHA = "77d606e2f2e894f7503a6f41c99a5a94696339b5c15c7b8415b68f8416b945d3"
SITE_SHA = "3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f"
CLIENT_SHA = "666d7e4540637ad6dcaa13012b0bd3c57733f469a47587a17f0debedbc52d312"
OLD_BINDINGS = {
    "ENGINE_HEAD": "9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2",
    "TABBY_HEAD": "3adc9813f30c4b92d110e7e0235129031dae1c25",
}
OBSERVER_ENV = "TABBY_STRINGS_OBSERVER_CONFIG"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_new(path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=".strings-controller-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_exact(path, expected, name):
    raw = path.read_bytes()
    require(re.fullmatch("[0-9a-f]{64}", expected) and digest(raw) == expected,
            f"Source hash mismatch: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def derive_observer(raw, engine, tabby):
    require(digest(raw) == OBSERVER_SHA, "Frozen observer source changed")
    replacements = []
    derived = raw
    for name, chosen in (("ENGINE_HEAD", engine), ("TABBY_HEAD", tabby)):
        require(re.fullmatch("[0-9a-f]{40}", chosen), "Full exact source commits are required")
        before = f'{name} = "{OLD_BINDINGS[name]}"'.encode()
        after = f'{name} = "{chosen}"'.encode()
        require(derived.count(before) == 1, f"Expected exactly one source assignment: {name}")
        derived = derived.replace(before, after, 1)
        replacements.append({"name": name, "original": OLD_BINDINGS[name], "selected": chosen})
    restored = derived
    for change in replacements:
        restored = restored.replace(
            f'{change["name"]} = "{change["selected"]}"'.encode(),
            f'{change["name"]} = "{change["original"]}"'.encode(), 1)
    require(restored == raw, "Observer derivation changed code beyond the two source constants")
    return derived, replacements


def prepare_bundle(observer_dir, bundle, output, job, runtime):
    original = observer_dir / "strings_observer.py"
    site = observer_dir / "sitecustomize.py"
    require(sha(site) == SITE_SHA, "Frozen site hook changed")
    raw = original.read_bytes()
    derived, replacements = derive_observer(raw, job["engine"], job["tabby"])
    # This separate input directory is exclusively claimed before the frozen
    # lifecycle claims output/. It is retained even after a preflight failure.
    bundle.mkdir(mode=0o700)
    (bundle / "strings_observer.original.py").write_bytes(raw)
    (bundle / "strings_observer.py").write_bytes(derived)
    (bundle / "sitecustomize.py").write_bytes(site.read_bytes())
    config = {
        "engine_repo": str(runtime / "exllamav3"), "tabby_repo": str(runtime / "tabbyAPI"),
        "engine_commit": job["engine"], "tabby_commit": job["tabby"],
        "run_label": job["label"], "max_records": 2, "output_dir": str(output / "raw-strings"),
    }
    atomic_new(bundle / "observer-config.json", config)
    provenance = {
        "schema_version": 1, "base_observer_sha256": OBSERVER_SHA,
        "derived_observer_sha256": digest(derived), "sitecustomize_sha256": SITE_SHA,
        "replacements": replacements, "config": config,
        "note": "Only the two expected source constants are rebound. All observer hook logic is byte-preserved.",
    }
    atomic_new(bundle / "derivation.json", provenance)
    provenance["files_sha256"] = {str(path): sha(path) for path in sorted(bundle.iterdir())}
    return provenance


class SignalGuard:
    """Defer launch-time signals until the frozen parent registers its process."""
    def __init__(self):
        self.launching = False
        self.cleaning = 0
        self.pending = None
        self.delivered = False

    def interrupt(self, signum, frame):
        if self.pending is None:
            self.pending = signum
            self.deliver()

    def deliver(self):
        if self.pending is not None and not self.delivered and not self.launching and not self.cleaning:
            self.delivered = True
            raise KeyboardInterrupt("signal " + str(self.pending))

    def cleanup(self, function):
        def wrapped(*args, **kwargs):
            self.cleaning += 1
            try:
                return function(*args, **kwargs)
            finally:
                self.cleaning -= 1
        return wrapped


def attach_observation(parent, bundle, guard, provenance):
    source_popen = parent.subprocess.Popen
    server_command = ["bash", str(parent.RECIPE / "exllamav3-tabby/serve.sh")]
    server_launches = []

    def popen(command, *args, **kwargs):
        env = dict(kwargs.get("env", {}))
        env.pop(OBSERVER_ENV, None)
        if command == server_command:
            require(not server_launches, "Only one observed server may be started")
            require(env.get("QWEN_EXPERIMENT_OWNER"), "Observed server lacks its ownership token")
            env[OBSERVER_ENV] = str(bundle / "observer-config.json")
            env["PYTHONPATH"] = str(bundle) + os.pathsep + env.get("PYTHONPATH", "")
            server_launches.append(command)
        else:
            require(str(bundle) not in env.get("PYTHONPATH", "").split(os.pathsep),
                    "Observer path leaked into a client")
        kwargs["env"] = env
        guard.launching = True
        try:
            process = source_popen(command, *args, **kwargs)
            original_wait = process.wait

            def wait(*wait_args, **wait_kwargs):
                guard.launching = False
                guard.deliver()  # Parent has assigned client before it calls wait.
                return original_wait(*wait_args, **wait_kwargs)
            process.wait = wait
            return process
        except BaseException:
            guard.launching = False
            raise

    # Replace only the parent's module reference, never global subprocess.Popen.
    parent.subprocess = types.SimpleNamespace(**vars(parent.subprocess))
    parent.subprocess.Popen = popen
    prior_load = parent.load_helpers

    def load_helpers(path):
        helper = prior_load(path)
        original_atomic = helper.atomic

        def atomic(path, value):
            original_atomic(path, value)
            if (path == parent.OUTPUT / "result.json" and value.get("server_pid")
                    and not value.get("finished_at_utc")):
                guard.launching = False
                guard.deliver()  # Parent has registered server before its save.
        helper.atomic = atomic
        return helper
    parent.load_helpers = load_helpers
    parent.stop_owned = guard.cleanup(parent.stop_owned)
    parent.cleanup_server = guard.cleanup(parent.cleanup_server)
    parent.EXTRA_FILES += (str(Path(__file__).resolve()), *provenance["files_sha256"])
    parent.EXPECTED = {"tools": 2}
    parent.TOOL_CASES = {"strings"}

    def commands(with_bench):
        require(not with_bench, "Strings observation never measures throughput")
        yield "tools", [
            str(parent.RUNTIME / "venv/bin/python"), str(parent.RECIPE / "bench/tool_smoke.py"),
            "--base-url", parent.BASE, "--model", parent.ALIAS, "--mode", "both",
            "--case", "strings", "--repeat", "1", "--max-tokens", "1024", "--timeout", "180",
        ]
    parent.commands = commands


def expected_request(observer, alias, streaming):
    value = {
        "model": alias, "messages": [{"role": "user", "content": observer.USER_MESSAGE}],
        "tools": observer.TOOLS, "tool_choice": "auto", "parallel_tool_calls": True,
        "max_tokens": 1024, "temperature": 0, "top_k": 1, "top_p": 1.0, "seed": 0,
        "stream": streaming, "chat_template_kwargs": {"enable_thinking": False},
    }
    if streaming:
        value["stream_options"] = {"include_usage": True}
    return value


def assess_capture(output, job, provenance, observer, alias, lifecycle_exit):
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    require(result.get("state") == "completed" and result.get("finished_at_utc"),
            "Owned lifecycle did not complete")
    require(not result.get("cleanup_failed") and not result.get("error"),
            "Owned lifecycle reported an error")
    cleanup = result.get("server_cleanup", {})
    require(cleanup.get("owned_group_empty") is True and cleanup.get("unexpected_exit") is False,
            "Server cleanup is absent or unexpected")
    require(result.get("expected_engine") == job["engine"] and result.get("expected_server") == job["tabby"],
            "Lifecycle source pins differ from the job")
    require(len(result.get("clients", [])) == 1, "Exactly one two-request client must run")
    client = result["clients"][0]
    require(client.get("name") == "tools" and client.get("cleanup", {}).get("owned_group_empty") is True
            and not client.get("timed_out"), "Client identity, deadline or cleanup failed")
    report_path = output / "tools.json"
    require(client.get("report") == str(report_path) and client.get("report_sha256") == sha(report_path),
            "Strict client report is missing or changed")
    report = json.loads(report_path.read_text())
    require(report.get("completed_at_utc") and not report.get("errors"), "Strict client report is incomplete")
    rows = report.get("results", [])
    require(len(rows) == 2 and {(r.get("case"), r.get("mode"), r.get("repeat_index")) for r in rows}
            == {("strings", "stream", 0), ("strings", "nonstream", 0)}, "Wrong client case set")
    require(all(type(row.get("passed")) is bool for row in rows), "Invalid semantic outcomes")
    semantic_passed = all(row["passed"] for row in rows)
    require(result.get("passed") is semantic_passed and client.get("passed") is semantic_passed,
            "Lifecycle/client success flags disagree with semantic outcomes")
    require(client.get("exit_code") == (0 if semantic_passed else 1)
            and lifecycle_exit == (0 if semantic_passed else 1), "Client/lifecycle exit disagrees with outcomes")
    require(report.get("summary") == {"total": 2, "passed": sum(r["passed"] for r in rows),
                                      "failed": sum(not r["passed"] for r in rows)},
            "Client summary disagrees with outcomes")
    raw_dir = output / "raw-strings"
    require({p.name for p in raw_dir.iterdir()} ==
            {"observer-manifest.json", "request-00.json", "request-01.json"}, "Expected exactly two trace files")
    manifest_path = raw_dir / "observer-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = manifest.get("source", {})
    require(manifest.get("max_records") == 2 and manifest.get("run_label") == job["label"],
            "Observer manifest scope mismatch")
    for name, commit in (("engine", job["engine"]), ("tabby", job["tabby"])):
        expected_path = provenance["config"][name + "_repo"]
        require(source.get(name, {}).get("commit") == commit
                and source[name].get("path") == expected_path
                and source[name].get("tracked_changes") == "", "Observer source identity mismatch")
    require(source.get("observer_sha256") == provenance["derived_observer_sha256"]
            and source.get("sitecustomize_sha256") == SITE_SHA
            and source.get("config_sha256") == digest(
                (json.dumps(provenance["config"], indent=2, ensure_ascii=False) + "\n").encode()),
            "Observer source/config hashes differ")
    traces, ids, modes = [], set(), set()
    for index in (0, 1):
        path = raw_dir / f"request-{index:02d}.json"
        trace = json.loads(path.read_text())
        require(trace.get("index") == index and trace.get("run_label") == job["label"], "Trace identity differs")
        identifier, streaming = trace.get("request_id"), trace.get("streaming_mode")
        require(isinstance(identifier, str) and identifier and identifier not in ids
                and type(streaming) is bool and streaming not in modes, "Duplicate/missing trace mode or request ID")
        ids.add(identifier); modes.add(streaming)
        require(trace.get("raised_exception_type") is None and trace.get("collector_returned_error") is False
                and trace.get("skipped_matching_requests_so_far") == 0, "Collector capture failed or exceeded its cap")
        prompt = trace.get("rendered_prompt")
        require(isinstance(prompt, str) and trace.get("rendered_prompt_sha256") == digest(prompt.encode()),
                "Rendered prompt hash differs")
        expected = expected_request(observer, alias, streaming)
        matched = trace.get("matched_request", {})
        fields = {"messages": expected["messages"], "tools": expected["tools"],
                  "model": alias, "tool_choice": "auto"}
        require(set(matched) == set(fields) | {"template_vars"}
                and all(matched[key] == value for key, value in fields.items()),
                "Trace matched request differs from the frozen original")
        # Apply the reviewed observer validator to the actual framework-augmented
        # template variables; do not mistake them for the raw wire dictionary.
        normalized = types.SimpleNamespace(**{key: value for key, value in expected.items()
                                             if key not in {"chat_template_kwargs", "stream_options"}},
                                           template_vars=matched["template_vars"], n=1, functions=None)
        require(observer.matching_request(normalized, streaming), "Observed template variables changed request scope")
        row = next(r for r in rows if (r["mode"] == "stream") == streaming)
        require(row.get("request") == expected, "Actual client payload differs from the frozen original")
        response = row.get("response", {})
        require(response.get("model") == alias and response.get("stream") is streaming,
                "Strict client did not retain a matching API response")
        finish = trace.get("raw_finish")
        require(isinstance(finish, dict) and finish.get("native_full_completion_present") is True
                and isinstance(finish.get("native_full_completion"), str)
                and isinstance(finish.get("backend_full_response"), str), "Both complete raw strings are required")
        times = [trace.get("started_monotonic_ns"), finish.get("backend_finish_monotonic_ns"),
                 trace.get("finished_monotonic_ns")]
        require(all(type(n) is int and n > 0 for n in times) and times == sorted(times),
                "Trace lifecycle timestamps are incomplete")
        traces.append({"path": str(path), "sha256": sha(path), "mode": row["mode"],
                       "request_id": identifier, "semantic_passed": row["passed"],
                       "native_sha256": digest(finish["native_full_completion"].encode()),
                       "backend_sha256": digest(finish["backend_full_response"].encode()),
                       "native_equals_backend": finish["native_full_completion"] == finish["backend_full_response"]})
    for path, fingerprint in provenance["files_sha256"].items():
        require(sha(path) == fingerprint and result.get("extra_files_sha256", {}).get(path) == fingerprint,
                "Recorded observer input changed")
    require(sha(output / "deployment.json") == result.get("deployment_sha256"),
            "Deployment sidecar changed")
    return {
        "capture_valid": True, "semantic_passed": semantic_passed,
        "semantic_counts": dict(Counter("pass" if row["passed"] else "fail" for row in rows)),
        "traces": traces, "lifecycle_sha256": sha(result_path), "client_report_sha256": sha(report_path),
        "observer_manifest_sha256": sha(manifest_path), "derivation": provenance,
        "matching_note": "The unchanged client omits response IDs from its summary. Traces match by the unique response mode and exact frozen payload; their native request IDs must be distinct.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observer-dir", type=Path, required=True)
    parser.add_argument("--final-wrapper-sha256", required=True)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--client-timeout", type=float, default=450)
    args = parser.parse_args(argv)
    require(0 < args.ready_timeout <= 1800 and 0 < args.client_timeout <= 900, "Invalid bounded timeout")
    root = Path(__file__).resolve().parent
    args.job = args.job.resolve(strict=True); args.setup = args.setup.resolve(strict=True)
    args.output = args.output.resolve(); args.observer_dir = args.observer_dir.resolve(strict=True)
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing existing output")
    job = json.loads(args.job.read_text())
    require(isinstance(job, dict) and not set(job) - {"label", "engine", "tabby", "model_path", "recipe", "env"},
            "Only source/model/tuning fields are accepted; exactly two strings requests are fixed")
    require(re.fullmatch("[a-z0-9][a-z0-9-]{0,63}", job.get("label", "")), "Observer label must be simple lowercase text")
    wrapper = load_exact(root / "final_api_controller.py", args.final_wrapper_sha256, "strings_final_configuration")
    parent = wrapper.load_parent(root / "api_f4_gemm_controller.py")
    require(wrapper.PARENT_SHA == PARENT_SHA, "Wrong owned lifecycle parent")
    wrapper.configure(parent, copy.deepcopy(job), args.job, args.output, args.setup)
    parent.check_setup(json.loads(args.setup.read_text()))
    require(sha(parent.RECIPE / "bench/tool_smoke.py") == CLIENT_SHA, "Original strict strings client changed")
    os.umask(0o077)
    bundle = args.output.with_name(args.output.name + ".observer-inputs")
    provenance = prepare_bundle(args.observer_dir, bundle, args.output, job, parent.RUNTIME)
    observer = load_exact(bundle / "strings_observer.py", provenance["derived_observer_sha256"], "selected_strings_observer")
    guard = SignalGuard()
    attach_observation(parent, bundle, guard, provenance)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    for sig in previous:
        signal.signal(sig, guard.interrupt)
    try:
        lifecycle_exit = parent.run(types.SimpleNamespace(bench=False, ready_timeout=args.ready_timeout,
                                                        client_timeout=args.client_timeout))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = {"schema_version": 1, "diagnostic_only": True, "passed": False, "capture_valid": False,
               "semantic_passed": None, "lifecycle_exit_code": lifecycle_exit,
               "controller_sha256": sha(__file__), "final_wrapper_sha256": args.final_wrapper_sha256,
               "interruption_observed": guard.pending, "job_sha256": sha(args.job)}
    try:
        summary.update(assess_capture(args.output, job, provenance, observer, parent.ALIAS, lifecycle_exit))
        summary["passed"] = summary["capture_valid"] and summary["semantic_passed"]
    except Exception as error:
        summary["capture_error"] = type(error).__name__ + ": " + str(error)
        summary["derivation"] = provenance
    atomic_new(args.output / "strings-capture.json", summary)
    print(json.dumps({key: summary.get(key) for key in
                      ("capture_valid", "semantic_passed", "passed", "capture_error")}), flush=True)
    return 0 if summary["passed"] else 1 if summary["capture_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
