"""CPU-only checks for the strings controller; no API/server/native import."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import time
import unittest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("strings_controller_tested", BASE / "strings_only_controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
OBSERVER_DIR = BASE / "tabbyapi-diagnostics/strings-final-9c-3adc/observer"
REPORT = BASE / "tabbyapi-diagnostics/pack-tool-errors-16ca-f4/405-tools.json"


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.runtime = self.root / "runtime"
        self.job = {"label": "peer-strings", "engine": "a" * 40, "tabby": "b" * 40}
        self.provenance = controller.prepare_bundle(OBSERVER_DIR, self.root / "inputs",
                                                    self.output, self.job, self.runtime)
        self.observer = controller.load_exact(self.root / "inputs/strings_observer.py",
                    self.provenance["derived_observer_sha256"], "derived_observer_fixture")
        self.record = None

    def fixture(self, passed=False):
        original = json.loads(REPORT.read_text())
        report = {key: value for key, value in original.items()
                  if key not in {"results", "summary"}}
        report["results"] = [copy.deepcopy(row) for row in original["results"] if row["case"] == "strings"]
        for row in report["results"]:
            row["passed"] = passed
            if passed:
                row["errors"] = []
                row["response"]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(self.observer.STRING_ARGS)
        report["summary"] = {"total": 2, "passed": 2 if passed else 0, "failed": 0 if passed else 2}
        report_path = self.output / "tools.json"
        report_path.write_text(json.dumps(report))
        (self.output / "deployment.json").write_text("{}")
        sources = {name: {"path": self.provenance["config"][name + "_repo"],
                          "commit": self.job["engine" if name == "engine" else "tabby"], "tracked_changes": ""}
                   for name in ("engine", "tabby")}
        sources.update(observer_sha256=self.provenance["derived_observer_sha256"],
                       sitecustomize_sha256=controller.SITE_SHA,
                       config_sha256=controller.sha(self.root / "inputs/observer-config.json"))
        recorder = self.observer.Recorder(self.output / "raw-strings", self.job["label"], 2, sources)
        for index, row in enumerate(report["results"]):
            stream = row["mode"] == "stream"
            wire = row["request"]
            params = SimpleNamespace(
                messages=wire["messages"], tools=wire["tools"], tool_choice=wire["tool_choice"],
                parallel_tool_calls=True, max_tokens=1024, temperature=0, top_k=1, top_p=1.0,
                n=1, template_vars={"enable_thinking": False}, functions=None, stream=stream,
                model=wire["model"],
            )
            params.template_vars.update({"messages": wire["messages"], "tools": wire["tools"],
                "functions": None, "add_generation_prompt": True, "tool_choice": "auto",
                "parallel_tool_calls": True, "bos_token": None, "eos_token": "<|im_end|>"})
            trace = recorder.begin("request-" + str(index), "fixture rendered prompt", params, stream, False)
            self.assertIsNotNone(trace)
            trace["raw_finish"] = {
                "native_full_completion": "raw native output", "native_full_completion_present": True,
                "backend_full_response": "raw backend output", "native_metrics": {"eos": True},
                "backend_finish_monotonic_ns": time.monotonic_ns(),
            }
            recorder.end(trace)
        self.record = {
            "state": "completed", "finished_at_utc": "fixture", "passed": passed,
            "expected_engine": self.job["engine"], "expected_server": self.job["tabby"],
            "server_cleanup": {"owned_group_empty": True, "unexpected_exit": False},
            "clients": [{"name": "tools", "passed": passed, "cleanup": {"owned_group_empty": True},
                         "report": str(report_path), "report_sha256": controller.sha(report_path),
                         "exit_code": 0 if passed else 1}],
            "extra_files_sha256": self.provenance["files_sha256"],
            "deployment_sha256": controller.sha(self.output / "deployment.json"),
        }
        self.save_record()
        return 0 if passed else 1

    def save_record(self):
        (self.output / "result.json").write_text(json.dumps(self.record))

    def assess(self, code=1):
        return controller.assess_capture(self.output, self.job, self.provenance, self.observer,
                                         "Qwen3.8-Flash-Next-EXL3", code)

    def alter(self, path, change):
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_valid_capture_preserves_original_semantic_failure_and_reports_different_raw_strings(self):
        self.fixture(False)
        before = {path.name: controller.sha(path) for path in self.output.glob("*.json")}
        result = self.assess()
        self.assertTrue(result["capture_valid"])
        self.assertFalse(result["semantic_passed"])
        self.assertEqual(result["semantic_counts"], {"fail": 2})
        self.assertTrue(all(not row["native_equals_backend"] for row in result["traces"]))
        self.assertEqual(before, {path.name: controller.sha(path) for path in self.output.glob("*.json")})

    def test_passing_semantics_require_consistent_exit(self):
        self.fixture(True)
        self.assertTrue(self.assess(0)["semantic_passed"])
        with self.assertRaisesRegex(ValueError, "exit disagrees"):
            self.assess(1)

    def test_missing_or_duplicated_trace_is_rejected(self):
        self.fixture()
        path = self.output / "raw-strings/request-01.json"
        saved = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.assess()
        path.write_bytes(saved)
        self.alter(path, lambda data: data.update(streaming_mode=False, request_id="request-0"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.assess()

    def test_bad_prompt_hash_or_collector_error_is_rejected(self):
        self.fixture()
        path = self.output / "raw-strings/request-00.json"
        saved = path.read_bytes()
        self.alter(path, lambda data: data.update(rendered_prompt="changed"))
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            self.assess()
        path.write_bytes(saved)
        self.alter(path, lambda data: data.update(collector_returned_error=True))
        with self.assertRaisesRegex(ValueError, "Collector capture"):
            self.assess()

    def test_unknown_source_and_missing_raw_finish_are_rejected(self):
        self.fixture()
        manifest = self.output / "raw-strings/observer-manifest.json"
        saved = manifest.read_bytes()
        self.alter(manifest, lambda data: data["source"]["engine"].update(commit="c"*40))
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.assess()
        manifest.write_bytes(saved)
        path = self.output / "raw-strings/request-01.json"
        self.alter(path, lambda data: data.update(raw_finish=None))
        with self.assertRaisesRegex(ValueError, "complete raw"):
            self.assess()

    def test_changed_client_payload_is_not_accepted_as_the_original_fixture(self):
        self.fixture()
        path = self.output / "tools.json"
        self.alter(path, lambda data: data["results"][0]["request"].update(max_tokens=1023))
        self.record["clients"][0]["report_sha256"] = controller.sha(path)
        self.save_record()
        with self.assertRaisesRegex(ValueError, "client payload"):
            self.assess()

    def test_conflicting_framework_variables_cannot_expand_capture_scope(self):
        self.fixture()
        path = self.output / "raw-strings/request-00.json"
        self.alter(path, lambda data: data["matched_request"]["template_vars"].update(unknown_client_setting=True))
        with self.assertRaisesRegex(ValueError, "template variables changed request scope"):
            self.assess()

    def test_unknown_cleanup_or_changed_input_never_passes(self):
        self.fixture()
        self.record["server_cleanup"]["owned_group_empty"] = False
        self.save_record()
        with self.assertRaisesRegex(ValueError, "cleanup"):
            self.assess()
        self.record["server_cleanup"]["owned_group_empty"] = True
        self.save_record()
        (self.root / "inputs/sitecustomize.py").write_text("# changed")
        with self.assertRaisesRegex(ValueError, "observer input"):
            self.assess()

    def test_only_source_constants_change_and_second_bundle_claim_is_refused(self):
        original = (OBSERVER_DIR / "strings_observer.py").read_bytes()
        derived, _ = controller.derive_observer(original, "a"*40, "b"*40)
        self.assertEqual(derived, original.replace(controller.OLD_BINDINGS["ENGINE_HEAD"].encode(), b"a"*40, 1)
                         .replace(controller.OLD_BINDINGS["TABBY_HEAD"].encode(), b"b"*40, 1))
        with self.assertRaises(ValueError):
            controller.derive_observer(original + b"\n", "a"*40, "b"*40)
        with self.assertRaises(FileExistsError):
            controller.prepare_bundle(OBSERVER_DIR, self.root / "inputs", self.output, self.job, self.runtime)


class EnvironmentAndSignalTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("/synthetic-only")
        self.output = self.root / "output"
        self.calls = []
        def launch(command, **kwargs):
            self.calls.append((command, kwargs))
            return SimpleNamespace(pid=3000+len(self.calls), wait=lambda timeout=None: 0)
        self.helper = SimpleNamespace(atomic=lambda *args: None)
        self.parent = SimpleNamespace(
            subprocess=SimpleNamespace(Popen=launch), RECIPE=self.root / "recipe",
            RUNTIME=self.root / "runtime", OUTPUT=self.output, BASE="http://127.0.0.1:8899/v1",
            ALIAS="Qwen3.8-Flash-Next-EXL3", load_helpers=lambda _: self.helper,
            stop_owned=lambda *a: {"done": True}, cleanup_server=lambda *a: {"done": True},
            EXTRA_FILES=(),
        )
        self.guard = controller.SignalGuard()
        self.bundle = self.root / "observer"
        controller.attach_observation(self.parent, self.bundle, self.guard, {"files_sha256": {}})

    def test_only_exact_server_receives_observer_and_fixed_command_contains_two_original_requests(self):
        environment = {"QWEN_EXPERIMENT_OWNER": "owner", "PYTHONPATH": "/engine"}
        server_command = ["bash", str(self.parent.RECIPE / "exllamav3-tabby/serve.sh")]
        self.parent.subprocess.Popen(server_command, env=environment)
        self.parent.load_helpers(None).atomic(self.output / "result.json", {"server_pid": 3001})
        name, command = list(self.parent.commands(False))[0]
        self.assertEqual(name, "tools")
        self.assertEqual(command[command.index("--case")+1], "strings")
        self.assertEqual(command[command.index("--mode")+1], "both")
        self.assertEqual(command[command.index("--repeat")+1], "1")
        self.assertEqual(command[command.index("--max-tokens")+1], "1024")
        client = self.parent.subprocess.Popen(command, env={**environment, controller.OBSERVER_ENV: "ambient"})
        client.wait()
        self.assertIn(controller.OBSERVER_ENV, self.calls[0][1]["env"])
        self.assertNotIn(controller.OBSERVER_ENV, self.calls[1][1]["env"])
        self.assertEqual(self.calls[1][1]["env"]["PYTHONPATH"], "/engine")
        self.assertEqual(environment, {"QWEN_EXPERIMENT_OWNER": "owner", "PYTHONPATH": "/engine"})
        with self.assertRaises(ValueError):
            self.parent.subprocess.Popen(server_command, env=environment)

    def test_signal_waits_for_server_registration_and_repeated_signal_does_not_break_cleanup(self):
        command = ["bash", str(self.parent.RECIPE / "exllamav3-tabby/serve.sh")]
        process = self.parent.subprocess.Popen(command, env={"QWEN_EXPERIMENT_OWNER": "owner"})
        self.guard.interrupt(signal.SIGTERM, None)
        self.assertFalse(self.guard.delivered)
        with self.assertRaises(KeyboardInterrupt):
            self.parent.load_helpers(None).atomic(self.output / "result.json", {"server_pid": process.pid})
        self.guard.interrupt(signal.SIGTERM, None)
        self.assertEqual(self.parent.stop_owned(None, process, "owner"), {"done": True})

    def test_client_launch_signal_is_delivered_only_after_assignment_at_wait(self):
        process = self.parent.subprocess.Popen(["python", "tool_smoke.py"], env={})
        self.guard.interrupt(signal.SIGTERM, None)
        self.assertFalse(self.guard.delivered)
        with self.assertRaises(KeyboardInterrupt):
            process.wait()
        self.assertTrue(self.guard.delivered)
        self.assertEqual(process.wait(), 0)


if __name__ == "__main__":
    unittest.main()
