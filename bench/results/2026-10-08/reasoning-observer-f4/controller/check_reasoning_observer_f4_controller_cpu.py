#!/usr/bin/env python3
"""CPU-only API controller checks. Never launches the server or sends HTTP."""
import copy
import ast
import hashlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("controller", HERE / "reasoning_observer_f4_controller.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def tools_report():
    rows = [dict(case=name, mode=mode, repeat_index=0, passed=True)
            for name in sorted(c.TOOL_CASES) for mode in ("stream", "nonstream")]
    return dict(results=rows, summary=dict(passed=28, failed=0, total=28),
                completed_at_utc="now", errors=[])


def other_report(name, n):
    rows = [dict(name=f"case{i}", status="pass") for i in range(n)]
    if name == "sdk":
        return dict(cases=rows, passed=n, failed=0, skipped=0, finished_at_utc="now")
    return dict(cases=rows, summary={"pass": n}, finished_utc="now")


class ControllerTests(unittest.TestCase):
    def test_setup_requires_completed_and_exact_optional_refs(self):
        c.check_setup({"state": "completed", "engine": c.ENGINE, "tabby": c.TABBY})
        for obj in ({}, {"state": "running"}, {"state": "failed"},
                    {"state": "completed", "engine": "wrong"},
                    {"state": "completed", "tabby": "wrong"}):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                c.check_setup(obj)

    def test_three_clients_are_serial_order_and_unchanged_packaged_auto(self):
        jobs = list(c.commands(False))
        self.assertEqual([n for n, _ in jobs], ["auto1", "auto2", "auto3"])
        for index, (name, args) in enumerate(jobs, 1):
            self.assertEqual(args[:2], [str(c.RUNTIME / "venv/bin/python"),
                                      str(c.RECIPE / "bench/auto_compatibility.py")])
            for flag, value in (("--base-url", c.BASE), ("--model", c.ALIAS),
                                ("--metadata", str(c.OUTPUT / "deployment.json")),
                                ("--label", f"reasoning-observer-f4-replay-{index}")):
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertEqual(len(args), 10)
        self.assertEqual(c.CLIENT_EXPECTED, {"auto1": 9, "auto2": 9, "auto3": 9})

    def test_optional_bench_is_rejected_for_bounded_observer_run(self):
        with self.assertRaises(ValueError):
            list(c.commands(True))

    def test_expected_all_pass_counts(self):
        self.assertTrue(c.summarize("tools", tools_report(), 0)["passed"])
        for name, n in (("sdk", 5), ("resilience", 19), ("auto", 9)):
            with self.subTest(name=name):
                self.assertTrue(c.summarize(name, other_report(name, n), 0)["passed"])

    def test_incomplete_duplicate_wrong_mode_and_nonzero_exit_fail(self):
        cases = []
        r = tools_report(); r.pop("completed_at_utc"); cases.append(r)
        r = tools_report(); r["results"].pop(); cases.append(r)
        r = tools_report(); r["results"][-1] = r["results"][0].copy(); cases.append(r)
        r = tools_report(); r["results"][0]["mode"] = "unknown"; cases.append(r)
        r = tools_report(); r["results"][0]["passed"] = 1; cases.append(r)
        for r in cases:
            with self.subTest(report=r):
                self.assertFalse(c.summarize("tools", r, 0)["passed"])
        self.assertFalse(c.summarize("tools", tools_report(), 1)["passed"])

    def test_sdk_skip_and_summary_lie_cannot_pass(self):
        r = other_report("sdk", 5)
        r["cases"][-1]["status"] = "skipped"
        r.update(passed=4, skipped=1)
        self.assertFalse(c.summarize("sdk", r, 0)["passed"])
        r = other_report("sdk", 5); r["failed"] = 1
        self.assertFalse(c.summarize("sdk", r, 0)["passed"])

    def test_resilience_fail_and_preflight_only_cannot_pass(self):
        r = other_report("resilience", 19)
        r["cases"][-1]["status"] = "fail"; r["summary"] = {"pass": 18, "fail": 1}
        out = c.summarize("resilience", r, 1)
        self.assertEqual(out["counts"], {"pass": 18, "fail": 1})
        self.assertFalse(out["passed"])
        self.assertFalse(c.summarize("resilience", other_report("resilience", 1), 0)["passed"])

    def test_bench_requires_three_complete_suites(self):
        r = {"cases": {key: {"warmup": [{}], "runs": [{}, {}, {}]}
                       for key in ("code", "devops", "prose")},
             "run_id": "overnight-v1", "completed_at_utc": "now", "errors": []}
        self.assertTrue(c.summarize("bench", r, 0)["passed"])
        r["cases"]["prose"]["runs"].pop()
        self.assertFalse(c.summarize("bench", r, 0)["passed"])

    def test_deployment_checks_model_config_bits_and_tuning(self):
        output = Path("/diagnostic/output")
        cfg = {
            "network": {"host": "127.0.0.1", "port": 8899, "disable_auth": True},
            "model": {"model_name": c.ALIAS, "model_dir": str(output/"state/models"),
                "cache_size": 262144, "max_seq_len": 262144, "max_batch_size": 1,
                "chunk_size": 2048, "cache_mode": "8,8", "ngram_ram": True,
                "reasoning": True, "tool_format": "qwen3_5", "vision": False},
            "draft_model": {"draft_mode": "mtp", "draft_num_tokens": 5,
                           "dynamic_draft": True, "draft_cache_mode": "8,8"}}
        identity = {"engine": {"commit": c.ENGINE}, "server": {"commit": c.TABBY}}
        d = {**copy.deepcopy(identity), "model": {"resolved_path": str(c.MODEL)},
             "config": {"values": cfg}, "environment": dict(c.TUNING)}
        c.validate_deployment(d, identity, output)
        for section, key, value in (("network", "host", "0.0.0.0"),
                                    ("model", "ngram_ram", False),
                                    ("model", "cache_mode", "fp16"),
                                    ("draft_model", "draft_num_tokens", 4)):
            bad = copy.deepcopy(d); bad["config"]["values"][section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_deployment(bad, identity, output)
        bad = copy.deepcopy(d); bad["environment"]["EXL3_MOE_COOP_MIXEDK"] = "1"
        with self.assertRaises(ValueError):
            c.validate_deployment(bad, identity, output)

    def test_frozen_helper_validates_before_import(self):
        helper = c.load_helpers(HERE / "spark_experiment_controller.py")
        self.assertTrue(callable(helper.model_identity))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "wrong.py"; p.write_text("raise AssertionError('must not execute')")
            with self.assertRaisesRegex(ValueError, "Frozen"):
                c.load_helpers(p)

    def test_existing_output_refused_before_any_subprocess_or_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("directory", "file", "dangling_symlink"):
                p = Path(tmp) / kind
                if kind == "directory": p.mkdir()
                elif kind == "file": p.write_text("preserve")
                else: p.symlink_to(Path(tmp)/"missing")
                with self.subTest(kind=kind), patch.object(c, "OUTPUT", p), \
                     patch.object(c.subprocess, "Popen", side_effect=AssertionError("no process")), \
                     patch.object(c, "api_json", side_effect=AssertionError("no HTTP")):
                    with self.assertRaises(FileExistsError):
                        c.run(types.SimpleNamespace(bench=False, ready_timeout=1, client_timeout=1))
                if kind == "file": self.assertEqual(p.read_text(), "preserve")

    def test_bad_setup_records_failure_without_any_subprocess_or_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"new"; status = Path(tmp)/"setup.json"
            status.write_text('{"state":"running"}')
            with patch.object(c, "OUTPUT", output), patch.object(c, "SETUP", status), \
                 patch.object(c.subprocess, "Popen", side_effect=AssertionError("no process")), \
                 patch.object(c, "api_json", side_effect=AssertionError("no HTTP")):
                code = c.run(types.SimpleNamespace(bench=False, ready_timeout=1, client_timeout=1))
            self.assertEqual(code, 1)
            report = json.loads((output/"result.json").read_text())
            self.assertEqual(report["state"], "failed")
            self.assertFalse(report["passed"])
            self.assertEqual(report["clients"], [])
            self.assertNotIn("server_pid", report)

    def test_unexpected_server_exit_is_not_successful_shutdown(self):
        for before, signals, unexpected in ((0, [], True), (None, [], True),
                                            (None, ["SIGTERM"], False)):
            process = types.SimpleNamespace(poll=lambda: before)
            with self.subTest(before=before, signals=signals), patch.object(c, "stop_owned", return_value={"exit_code": 0, "signals": signals}):
                result = c.cleanup_server(None, process, "owned")
                self.assertEqual(result["unexpected_exit"], unexpected)
                self.assertEqual(result["exit_before_cleanup"], before)

    def test_exact_ref_environment_resolves_identically_for_verification(self):
        helper = c.load_helpers(HERE / "spark_experiment_controller.py")
        recipe = HERE / "recipe"
        runtime = Path("/fixture/candidate-runtime")
        job = {"model_path": str(c.MODEL), "model_identity": {}, "env": dict(c.TUNING)}
        with patch.dict(os.environ, {"TABBY_REF": "wrong", "EXL3_GDN_PROJ_FP32": "0"}):
            env = helper.resolved_env(recipe, runtime, job)
            env["TABBY_REF"] = c.TABBY
            job["env"]["TABBY_REF"] = c.TABBY
            self.assertEqual(env["EXL3_GDN_PROJ_FP32"], "1")
            with patch.object(helper, "source_identity", return_value={}), patch.object(helper, "model_identity", return_value={}):
                helper.verify_inputs(job, types.SimpleNamespace(recipe=recipe, runtime=runtime), {}, env)

    def test_owned_cpu_process_cleanup_and_wrong_token_refusal(self):
        helper = c.load_helpers(HERE / "spark_experiment_controller.py")
        token = "api3a-cpu-cleanup-fixture"
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                env=dict(os.environ, QWEN_EXPERIMENT_OWNER=token),
                                start_new_session=True)
        try:
            with patch.object(c.os, "killpg", side_effect=AssertionError("wrong group must not signal")):
                with self.assertRaisesRegex(RuntimeError, "unverified"):
                    c.stop_owned(helper, proc, "different-token", timeout=2)
            self.assertIsNone(proc.poll())
            result = c.stop_owned(helper, proc, token, timeout=2)
            self.assertTrue(result["owned_group_empty"])
            self.assertEqual(result["signals"], ["SIGTERM"])
            self.assertLess(result["wall_seconds"], 2)
            self.assertEqual(c.stop_owned(helper, proc, token, timeout=2)["signals"], [])
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2)


    def test_critical_source_and_cleanup_gates_are_identical_to_frozen_f4_parent(self):
        parent = HERE / "api_f4_gemm_controller.py"
        self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(), c.PARENT_CONTROLLER_SHA)
        before = ast.parse(parent.read_text()); after = ast.parse(Path(c.__file__).read_text())
        def body(tree, name):
            return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
        for name in ("sha", "load_helpers", "check_setup", "summarize", "stop_owned", "cleanup_server", "require_listener", "api_json", "validate_deployment"):
            with self.subTest(name=name):self.assertEqual(body(before, name), body(after, name))
        def value(tree, name):
            return ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in n.targets)))
        for name in ("ENGINE", "TABBY", "ALIAS", "BASE", "HELPER_SHA"):
            self.assertEqual(value(before, name), value(after, name))
        old_tuning = next(n.value for n in before.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "TUNING" for x in n.targets))
        new_tuning = next(n.value for n in after.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "TUNING" for x in n.targets))
        self.assertEqual(ast.dump(old_tuning), ast.dump(new_tuning))

    def test_observer_flags_are_passed_to_server_only(self):
        tree = ast.parse(Path(c.__file__).read_text())
        calls = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):continue
            if not isinstance(node.value.func, ast.Attribute) or node.value.func.attr != "Popen":continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    calls[target.id] = next(k.value.id for k in node.value.keywords if k.arg == "env")
        self.assertEqual(calls, {"server": "server_env", "client": "env"})

    def observer_fixture(self, output):
        raw = output / "raw"; raw.mkdir()
        record = {"observer_config_sha256": "config-hash"}
        manifest = {"schema_version": 1, "diagnostic_only": True,
                    "prompt_sha256": c.OBSERVER_PROMPT_SHA, "max_records": 6,
                    "source": {
                        "tabby": {"path": str(c.RUNTIME/"tabbyAPI"), "commit": c.TABBY, "tracked_changes": ""},
                        "engine": {"path": str(c.RUNTIME/"exllamav3"), "commit": c.ENGINE, "tracked_changes": ""},
                        "observer_sha256": c.OBSERVER_FILES["reasoning_observer.py"],
                        "sitecustomize_sha256": c.OBSERVER_FILES["sitecustomize.py"],
                        "config_sha256": "config-hash"}}
        (raw/"observer-manifest.json").write_text(json.dumps(manifest))
        record["observer_manifest_sha256"] = c.validate_observer_manifest(output, record)
        for replay in range(3):
            cases=[]; requests=[]
            for mode in range(2):
                index=replay*2+mode; rid=f"fixture-{index}"; streamed=bool(mode)
                wire_id=("chatcmpl-" if streamed else "cmpl-")+rid
                requests.append({"stream": streamed, "frames": [{"id": wire_id}] if streamed else [],
                                 "body": None if streamed else {"id": wire_id}})
                cases.append({"name": "auto_reasoning_stream" if streamed else "auto_reasoning_nonstream",
                              "request_indices": [mode]})
                trace={"schema_version": 1, "index": index, "request_id": rid,
                       "streaming_mode": streamed, "prompt_sha256": c.OBSERVER_PROMPT_SHA,
                       "collector_returned_error": False, "raised_exception_type": None,
                       "raw_finish": {"text": "thought</think>answer"}, "events": [],
                       "finished_monotonic_ns": 123}
                (raw/f"request-{index:02d}.json").write_text(json.dumps(trace))
            (output/f"auto{replay+1}.json").write_text(json.dumps({"cases": cases, "requests": requests}))
        return record

    def test_exact_six_traces_must_match_the_original_api_response_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);record=self.observer_fixture(output)
            result=c.verify_observer_traces(output,record)
            self.assertEqual(result["completed"],6);self.assertTrue(result["matched_api_ids"])
            bad=output/"raw/request-05.json";value=json.loads(bad.read_text());value["request_id"]="wrong"
            bad.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,"match"):c.verify_observer_traces(output,record)

    def test_missing_extra_incomplete_or_wrong_provenance_trace_cannot_pass(self):
        for change in ("missing","extra","incomplete","manifest"):
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp);record=self.observer_fixture(output)
                if change=="missing":(output/"raw/request-05.json").unlink()
                elif change=="extra":(output/"raw/request-06.json").write_text('{}')
                elif change=="incomplete":
                    p=output/"raw/request-05.json";value=json.loads(p.read_text());value["raw_finish"]=None;p.write_text(json.dumps(value))
                else:
                    p=output/"raw/observer-manifest.json";value=json.loads(p.read_text());value["source"]["observer_sha256"]="wrong";p.write_text(json.dumps(value))
                with self.subTest(change=change), self.assertRaises(ValueError):c.verify_observer_traces(output,record)

    def test_observer_config_is_exact_and_uses_a_six_record_cap(self):
        config=c.observer_config()
        self.assertEqual(config["max_records"],6)
        self.assertEqual(config["output_dir"],str(c.OUTPUT/"raw"))
        self.assertEqual(config["tabby_commit"],c.TABBY)
        self.assertEqual(config["engine_commit"],c.ENGINE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
