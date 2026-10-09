"""Offline contracts; harmless local children only, no server/network/GPU/Spark access."""
import importlib.util
import json
import os
import select
import shutil
import sys
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment", HERE / "run_matrix.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pack = self.root / "pack-405"
        self.pack.mkdir()
        (self.pack / "config.json").write_text('{"model_type":"qwen"}')
        (self.pack / "model-00001.safetensors").write_bytes(b"synthetic-only")
        self.recipe = self.root / "recipe"
        (self.recipe / "exllamav3-tabby").mkdir(parents=True)
        self.runtime = self.root / "runtime"
        self.output = self.root / "output"
        self.output.mkdir()
        self.args = SimpleNamespace(recipe=self.recipe, runtime=self.runtime, output=self.output,
            resume=False, ready_timeout=30, client_timeout=30, sample_interval=10, max_samples=3)
        self.identity = {name: {"path": str(self.runtime / name), "commit": name + "-actual"}
                         for name in ("engine", "server")}
        self.env = {"RECIPE_HOME": str(self.runtime), "PROFILE": "single", "NGRAM_RAM": "false",
                    "EXL3_REF": "actual-recipe-default", "TABBY_REF": "main"}
        self.job = controller.normalize({"label": "test-405", "model_path": str(self.pack),
            "env": {"PROFILE": "single", "NGRAM_RAM": False},
            "bench": {"suite": "code", "repeat": 1, "warmup": 0}})
        self.launched = []
        self.server = SimpleNamespace(pid=10101, returncode=None, poll=lambda: None)
        self.client = SimpleNamespace(pid=20202, returncode=0, poll=lambda: 0)

    def mock_launch(self, command, **kwargs):
        self.launched.append((command, kwargs))
        if command[0] == "bash":
            state = Path(kwargs["env"]["STATE_DIR"])
            state.mkdir()
            metadata = {"model": {"resolved_path": str(self.pack)}, "environment": kwargs["env"],
                        **{k: {"commit": v["commit"]} for k, v in self.identity.items()}}
            (state / "deployment.json").write_text(json.dumps(metadata))
            return self.server
        destination = Path(command[command.index("--output") + 1])
        destination.write_text(json.dumps({"completed_at_utc": "synthetic", "usage": "fixture"}))
        return self.client

    def execute(self, **patches):
        defaults = {
            "require_free_port": Mock(), "require_owned_listener": Mock(), "verify_inputs": Mock(),
            "get_json": Mock(side_effect=lambda endpoint: {"data": [{"id": "Qwen3.8-Flash-Next-EXL3"}]}
                if endpoint == "/models" else {"id": self.pack.name}),
            "stop_owned": Mock(side_effect=lambda process, token:
                {"exit_code": -15 if process is self.server else process.returncode, "signals": ["SIGTERM"]}),
            "Sampler": Mock(), "git_identity": Mock(side_effect=lambda path:
                self.identity[Path(path).name]),
        }
        defaults.update(patches)
        with patch.multiple(controller, **defaults), patch.object(
                controller.subprocess, "Popen", side_effect=self.mock_launch):
            return controller.run_job(self.job, self.args, self.identity, self.env)

    def test_normalized_settings_and_exact_command_routing(self):
        job = controller.normalize({"label": "r1", "model_path": str(self.pack),
            "env": {"PROFILE": "concurrent", "NGRAM_RAM": False, "EXL3_DRAFT_ROW_BUDGET": 32},
            "bench": [{"suite": "all", "run_id": "same-ab"}, {"context_tokens": 16384}],
            "concurrency": {"streams": [1, 2, 4]}, "tool_cases": ["auto", "strings"],
            "response_model": "explicit-canonical"})
        self.assertEqual(job["env"]["EXL3_DRAFT_ROW_BUDGET"], "32")
        commands = list(controller.commands(job, self.root, self.recipe, "client-python", "alias"))
        self.assertEqual([name for name, _ in commands], ["bench-1", "bench-2", "concurrency", "tools"])
        for _, command in commands:
            self.assertEqual(command[command.index("--response-model") + 1], "explicit-canonical")
            self.assertNotIn("--api-key", command)
        self.assertIn("1,2,4", commands[2][1])
        self.assertIn("auto,strings", commands[3][1])
        self.assertNotIn("--metadata", commands[3][1])

    def test_rejects_reserved_environment_and_bad_measurement_settings(self):
        for env in ({"STATE_DIR": "/other"}, {"EXL3_SRC": "/other"}, {"API_KEY": "synthetic"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                controller.normalize({"label": "x", "model_path": str(self.pack),
                    "env": dict(PROFILE="single", NGRAM_RAM=False, **env)})
        for settings in ({"output": "/wrong"}, {"repeat": 0}, {"thinking": "false"},
                         {"suite": "typo"}, {"timeout": float("inf")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                controller.normalize({"label": "x", "model_path": str(self.pack),
                    "env": {"PROFILE": "single", "NGRAM_RAM": False}, "bench": settings})

    def test_model_configuration_change_affects_config_identity(self):
        before = self.job["model_identity"]
        (self.pack / "config.json").write_text('{"model_type":"other"}')
        newer = controller.normalize({"label": "test-405", "model_path": str(self.pack),
            "env": {"PROFILE": "single", "NGRAM_RAM": False}})
        self.assertNotEqual(controller.digest(before), controller.digest(newer["model_identity"]))

    def test_sources_current_defaults_and_ignores_ambient_pins(self):
        (self.recipe / "exllamav3-tabby/env.sh").write_text(
            'EXL3_REF="${EXL3_REF:-fresh-engine}"\nTABBY_REF="${TABBY_REF:-fresh-main}"\n')
        with patch.dict(controller.os.environ, {"EXL3_REF": "old", "TABBY_REF": "old", "CACHE_SIZE": "999"}):
            env = controller.resolved_env(self.recipe, self.runtime, self.job)
        self.assertEqual(env["EXL3_REF"], "fresh-engine")
        self.assertEqual(env["TABBY_REF"], "fresh-main")
        self.assertNotIn("CACHE_SIZE", env)
        self.assertEqual(env["HOST"], "127.0.0.1")
        self.assertEqual(env["PORT"], "8899")

    def test_gpu_lock_is_explicit_resolved_configuration_and_drift_is_rejected(self):
        requested = str(self.root / "gpu.lock")
        job = controller.normalize({"label": "shared-lock", "model_path": str(self.pack),
            "env": {"PROFILE": "single", "NGRAM_RAM": False, "GPU_LOCK_FILE": requested},
            "tool_cases": ["auto"]})
        (self.recipe / "exllamav3-tabby/env.sh").write_text(
            'GPU_LOCK_FILE="${GPU_LOCK_FILE:-}"\nacquire_gpu_lock() { :; }\n')
        with patch.dict(controller.os.environ, {"GPU_LOCK_FILE": "/ambient-owner.lock"}):
            env = controller.resolved_env(self.recipe, self.runtime, job)
            default_env = controller.resolved_env(self.recipe, self.runtime, self.job)
        self.assertEqual(env["GPU_LOCK_FILE"], requested)
        self.assertEqual(default_env["GPU_LOCK_FILE"], "")
        self.assertFalse(Path(requested).exists())
        with patch.object(controller, "source_identity", return_value=self.identity), \
             patch.object(controller, "resolved_env", return_value=env) as resolved:
            controller.verify_inputs(job, self.args, self.identity, env)
            resolved.return_value = dict(env, GPU_LOCK_FILE=requested + ".changed")
            with self.assertRaisesRegex(ValueError, "defaults changed"):
                controller.verify_inputs(job, self.args, self.identity, env)

    def test_requested_lock_rejects_older_recipe_before_launch_or_lock_creation(self):
        # Historical launchers can inherit this variable while never acquiring it.
        (self.recipe / "exllamav3-tabby/env.sh").write_text('GPU_LOCK_FILE="${GPU_LOCK_FILE:-}"\n')
        requested = self.root / "shared.lock"
        job = dict(self.job, env=dict(self.job["env"], GPU_LOCK_FILE=str(requested)))
        with self.assertRaises(subprocess.CalledProcessError):
            controller.resolved_env(self.recipe, self.runtime, job)
        self.assertFalse(requested.exists())
        self.assertEqual(controller.resolved_env(self.recipe, self.runtime, self.job)["GPU_LOCK_FILE"], "")

    def test_lock_ownership_failure_stops_before_clients_and_cleans_owned_server(self):
        self.env["GPU_LOCK_FILE"] = str(self.root / "requested.lock")
        cleanup = Mock(return_value={"exit_code": -15, "signals": ["SIGTERM"]})
        with patch.object(controller, "verify_gpu_lock", side_effect=ValueError("missing actual flock")):
            result = self.execute(stop_owned=cleanup)
        self.assertFalse(result["passed"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["clients"], [])
        cleanup.assert_called_once_with(self.server, unittest.mock.ANY)

    def test_lock_evidence_is_recorded_before_each_client_and_after_measurement(self):
        self.env["GPU_LOCK_FILE"] = str(self.root / "requested.lock")
        evidence = {"path": self.env["GPU_LOCK_FILE"], "descriptor": 8, "inode": 42}
        with patch.object(controller, "verify_gpu_lock", return_value=evidence) as verify:
            result = self.execute()
        self.assertTrue(result["passed"])
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(result["gpu_lock_before_measurements"], evidence)
        self.assertEqual(result["clients"][0]["gpu_lock_before"], evidence)
        self.assertEqual(result["gpu_lock_after_measurements"], evidence)

    def test_unset_lock_does_not_inspect_proc(self):
        with patch.object(controller.Path, "lstat", side_effect=AssertionError("unexpected filesystem read")):
            self.assertIsNone(controller.verify_gpu_lock(999999, {}))

    @unittest.skipUnless(shutil.which("bash") and shutil.which("flock") and Path("/proc/self/fdinfo").is_dir(),
                         "requires Linux procfs, bash and flock")
    def test_actual_inherited_flock_rejects_wrong_inode_replacement_and_unlocked_fd(self):
        requested = self.root / "actual shared.lock"
        other = self.root / "other.lock"
        other.touch()
        program = ('import fcntl,sys; print("READY",flush=True); '
                   'sys.stdin.readline(); fcntl.flock(8,fcntl.LOCK_UN); '
                   'print("UNLOCKED",flush=True); sys.stdin.readline()')
        child = subprocess.Popen(
            ["bash", "-euc", 'source "$1"; acquire_gpu_lock; shift; exec "$@"', "lock-test",
             str(HERE.parent / "exllamav3-tabby/env.sh"), sys.executable, "-u", "-c", program],
            env={**os.environ, "GPU_LOCK_FILE": str(requested), "STATE_DIR": str(self.root / "state")},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([child.stdout], [], [], 5)[0], "child did not become ready")
            self.assertEqual(child.stdout.readline().strip(), "READY")
            evidence = controller.verify_gpu_lock(child.pid, {"GPU_LOCK_FILE": str(requested)})
            self.assertEqual(evidence["inode"], requested.stat().st_ino)
            self.assertIn("FLOCK", evidence["fdinfo_lock"])
            with self.assertRaisesRegex(ValueError, "inode"):
                controller.verify_gpu_lock(child.pid, {"GPU_LOCK_FILE": str(other)})
            retained = self.root / "retained-inode"
            requested.rename(retained); requested.touch()
            with self.assertRaisesRegex(ValueError, "inode"):
                controller.verify_gpu_lock(child.pid, {"GPU_LOCK_FILE": str(requested)})
            requested.unlink(); retained.rename(requested)
            child.stdin.write("unlock\n"); child.stdin.flush()
            self.assertTrue(select.select([child.stdout], [], [], 5)[0], "child did not unlock")
            self.assertEqual(child.stdout.readline().strip(), "UNLOCKED")
            with self.assertRaisesRegex(ValueError, "exclusive cooperative flock"):
                controller.verify_gpu_lock(child.pid, {"GPU_LOCK_FILE": str(requested)})
            child.communicate("exit\n", timeout=5)
            self.assertEqual(child.returncode, 0)
        finally:
            if child.poll() is None:
                child.kill(); child.communicate(timeout=5)

    def test_success_captures_actual_exits_and_resume_preserves_artifacts(self):
        first = self.execute()
        self.assertTrue(first["passed"])
        self.assertEqual(first["clients"][0]["exit_code"], 0)
        self.assertEqual(first["server_cleanup"]["exit_code"], -15)
        self.assertIsNone(first["server_exit_before_cleanup"])
        self.assertTrue(Path(first["attempt"], "deployment.json").is_file())
        self.assertTrue(all(row[1]["start_new_session"] for row in self.launched))
        self.args.resume = True
        self.launched.clear()
        second = self.execute()
        self.assertEqual(first, second)
        self.assertFalse(self.launched)
        self.env["EXL3_GR_INT8"] = "0"
        with self.assertRaisesRegex(ValueError, "changed configuration"):
            self.execute()

    def test_refuses_completed_output_without_resume(self):
        self.execute()
        self.launched.clear()
        with self.assertRaisesRegex(ValueError, "Refusing existing"):
            self.execute()
        self.assertFalse(self.launched)

    def test_busy_port_never_starts_or_signals_a_process(self):
        cleanup = Mock()
        with self.assertRaises(OSError):
            self.execute(require_free_port=Mock(side_effect=OSError("occupied")), stop_owned=cleanup)
        self.assertFalse(self.launched)
        cleanup.assert_not_called()

    def test_wrong_model_or_unowned_listener_still_cleans_up_own_server(self):
        for override in (
            {"get_json": Mock(return_value={"data": [{"id": "unrelated"}], "id": "unrelated"})},
            {"require_owned_listener": Mock(side_effect=RuntimeError("unowned listener"))},
        ):
            with self.subTest(override=override):
                self.job["label"] += "x"
                cleanup = Mock(return_value={"exit_code": -15, "signals": ["SIGTERM"]})
                result = self.execute(stop_owned=cleanup, **override)
                self.assertFalse(result["passed"])
                self.assertEqual(result["state"], "failed")
                self.assertEqual(result["clients"], [])
                cleanup.assert_called_once()

    def test_failed_benchmark_exit_cannot_become_passed(self):
        self.client.returncode = 7
        self.client.poll = lambda: 7
        result = self.execute()
        self.assertFalse(result["passed"])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["clients"][0]["exit_code"], 7)

    def test_interrupt_preserves_attempt_and_cleans_owned_server(self):
        cleanup = Mock(return_value={"exit_code": -15, "signals": ["SIGTERM"]})
        with self.assertRaises(KeyboardInterrupt):
            self.execute(get_json=Mock(side_effect=KeyboardInterrupt("synthetic")), stop_owned=cleanup)
        old = json.loads((self.output / self.job["label"] / "result.json").read_text())
        self.assertEqual(old["state"], "interrupted")
        self.assertNotIn("finished_at_utc", old)
        cleanup.assert_called_once()
        self.args.resume = True
        new = self.execute()
        self.assertTrue(new["passed"])
        self.assertNotEqual(old["attempt"], new["attempt"])
        self.assertTrue(Path(old["attempt"], "result.json").is_file())

    def test_sampling_is_bounded_and_never_calls_real_nvidia_smi(self):
        sampler = controller.Sampler(self.root / "resources.json", interval=1, limit=2)
        with patch.object(controller.subprocess, "run", return_value=SimpleNamespace(
                returncode=0, stdout="1,2,3,4,5,6", stderr="")) as run:
            for _ in range(5):
                sampler.next = 0
                sampler.tick("fixture")
        self.assertEqual(run.call_count, 2)
        report = json.loads((self.root / "resources.json").read_text())
        self.assertTrue(report["limit_reached"])
        self.assertEqual(len(report["samples"]), 2)

    def test_unknown_owner_is_never_signaled(self):
        process = SimpleNamespace(pid=30303)
        with patch.object(controller, "group_members", side_effect=RuntimeError("owner mismatch")), \
             patch.object(controller.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "owner mismatch"):
                controller.stop_owned(process, "expected-owner")
        kill.assert_not_called()

    def test_tool_only_job_does_not_run_an_implicit_benchmark(self):
        job = controller.normalize({"label": "tools", "model_path": str(self.pack),
            "env": {"PROFILE": "single", "NGRAM_RAM": False}, "tool_cases": ["auto"]})
        self.assertEqual(job["bench"], [])
        self.assertEqual([name for name, command in controller.commands(
            job, self.root, self.recipe, "python", "alias")], ["tools"])

    def test_loader_enumeration_order_is_part_of_model_identity(self):
        patch_file = self.pack / "mtp-patch.safetensors"
        patch_file.write_bytes(b"synthetic-patch")
        names = [str(patch_file), str(self.pack / "model-00001.safetensors")]
        with patch.object(controller.glob, "glob", return_value=names):
            first = controller.model_identity(self.pack)
        with patch.object(controller.glob, "glob", return_value=list(reversed(names))):
            second = controller.model_identity(self.pack)
        self.assertEqual(first["weights"], second["weights"])
        self.assertNotEqual(controller.digest(first), controller.digest(second))
        self.assertEqual(first["loader_glob_order"], ["mtp-patch.safetensors", "model-00001.safetensors"])

    def test_recheck_rejects_source_model_and_default_drift(self):
        with patch.object(controller, "source_identity", return_value=self.identity) as source, \
             patch.object(controller, "resolved_env", return_value=self.env) as env:
            controller.verify_inputs(self.job, self.args, self.identity, self.env)
            source.return_value = dict(self.identity, changed=True)
            with self.assertRaisesRegex(ValueError, "Recipe/runtime"):
                controller.verify_inputs(self.job, self.args, self.identity, self.env)
            source.return_value = self.identity
            env.return_value = dict(self.env, EXL3_GR_INT8="0")
            with self.assertRaisesRegex(ValueError, "defaults changed"):
                controller.verify_inputs(self.job, self.args, self.identity, self.env)
            env.return_value = self.env
            (self.pack / "tokenizer.json").write_text('{"synthetic":"changed"}')
            with self.assertRaisesRegex(ValueError, "Model files"):
                controller.verify_inputs(self.job, self.args, self.identity, self.env)

    def test_drift_after_measured_client_invalidates_the_job(self):
        check = Mock(side_effect=[None, None, ValueError("Recipe/runtime identity changed")])
        result = self.execute(verify_inputs=check)
        self.assertFalse(result["passed"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["clients"][0]["exit_code"], 0)
        self.assertNotIn("inputs_verified_after_measurements_at_utc", result)

    def test_failed_client_cleanup_marks_job_unsafe_to_continue(self):
        def cleanup(process, token):
            if process is self.client:
                raise RuntimeError("unverified process group")
            return {"exit_code": -15, "signals": ["SIGTERM"]}
        result = self.execute(stop_owned=Mock(side_effect=cleanup))
        self.assertEqual(result["state"], "cleanup_failed")
        self.assertFalse(result["passed"])
        self.assertIn("unverified", result["clients"][0]["cleanup_error"])
        self.assertEqual(result["server_cleanup"]["exit_code"], -15)

    def test_real_group_membership_parser_checks_session_and_exact_owner(self):
        proc = self.root / "proc"
        proc.mkdir()
        for pid, group, session, owner in ((30303, 30303, 30303, "owned"),
                                           (30304, 30303, 30303, "owned"),
                                           (90909, 90909, 90909, "other")):
            entry = proc / str(pid)
            entry.mkdir()
            (entry / "stat").write_text(f"{pid} (name has spaces) S 1 {group} {session} 0")
            (entry / "environ").write_bytes(f"OTHER=abc\0QWEN_EXPERIMENT_OWNER={owner}\0".encode())
        real_path = Path
        with patch.object(controller, "Path", side_effect=lambda value:
                proc if str(value) == "/proc" else real_path(value)):
            self.assertEqual(sorted(controller.group_members(SimpleNamespace(pid=30303), "owned")), [30303, 30304])
            (proc / "30304/environ").write_bytes(b"QWEN_EXPERIMENT_OWNER=owned-wrong\0")
            with self.assertRaisesRegex(RuntimeError, "unverified process group"):
                controller.group_members(SimpleNamespace(pid=30303), "owned")

    def test_timeout_records_actual_terminated_client_exit_and_failure(self):
        self.args.client_timeout = .01
        self.client.returncode = -15
        self.client.poll = lambda: None
        clock = [0.0]
        def monotonic():
            clock[0] += .05
            return clock[0]
        with patch.object(controller.time, "monotonic", monotonic):
            result = self.execute()
        self.assertFalse(result["passed"])
        self.assertTrue(result["clients"][0]["timed_out"])
        self.assertEqual(result["clients"][0]["exit_code"], -15)

    def test_cleanup_signals_only_its_retained_group_and_records_exit(self):
        process = SimpleNamespace(pid=30303, poll=lambda: -15, wait=lambda timeout: -15)
        alive = [True]
        def terminate(pid, sig):
            self.assertEqual(pid, 30303)
            alive[0] = False
        with patch.object(controller, "group_members", side_effect=lambda p, t: [30303] if alive[0] else []), \
             patch.object(controller.os, "killpg", side_effect=terminate) as kill:
            result = controller.stop_owned(process, "retained-token")
        self.assertEqual(result, {"exit_code": -15, "signals": ["SIGTERM"]})
        kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
