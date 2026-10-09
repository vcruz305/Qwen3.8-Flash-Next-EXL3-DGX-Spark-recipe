"""Process-level contracts for optional cooperative GPU ownership."""
import fcntl
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "exllamav3-tabby"


@unittest.skipUnless(shutil.which("bash") and shutil.which("flock"), "requires bash and flock")
class CooperativeGPULockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {
            **os.environ,
            "RECIPE_HOME": str(self.root / "runtime"),
            "STATE_DIR": str(self.root / "state"),
            "VENV": str(self.root / "missing-venv"),
            "EXL3_SRC": str(self.root / "engine"),
            "TABBY_DIR": str(self.root / "tabby"),
            "MODEL_DIR": str(self.root / "pack"),
            "PYTHON_BIN": sys.executable,
            "CUDA_HOME": str(self.root / "missing-cuda"),
            "BIGCORES": "",
            "PROFILE": "concurrent",
            "NGRAM_RAM": "false",
            "GPU_LOCK_FILE": "",
            "DRY_RUN": "0",
        }

    def serve(self, **overrides):
        return subprocess.run(
            ["bash", str(RECIPE / "serve.sh")],
            cwd=ROOT, env={**self.env, **overrides},
            capture_output=True, text=True, timeout=10,
        )

    def test_contention_refuses_before_runtime_or_live_file_changes(self):
        lock = self.root / "shared GPU.lock"
        lock.write_text("existing supervisor metadata\n")
        python = self.root / "fake-venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        marker = self.root / "runtime-was-invoked"
        python.write_text('#!/bin/sh\ntouch "$RUNTIME_MARKER"\nexit 42\n')
        python.chmod(0o755)
        with lock.open("r+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in ("first-state", "second-state"):
                with self.subTest(state=name):
                    state = self.root / name
                    (state / "models").mkdir(parents=True)
                    config = state / "config.yml"
                    config.write_text("existing live config\n")
                    view = state / "models" / "existing-model"
                    view.symlink_to(self.root / "existing-pack")
                    result = self.serve(
                        STATE_DIR=str(state), GPU_LOCK_FILE=str(lock),
                        VENV=str(python.parent.parent), RUNTIME_MARKER=str(marker),
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("another GPU supervisor holds", result.stderr)
                    self.assertFalse(marker.exists(), result.stderr)
                    self.assertEqual(config.read_text(), "existing live config\n")
                    self.assertTrue(view.is_symlink())
                    self.assertFalse((state / "deployment.json").exists())
        self.assertEqual(lock.read_text(), "existing supervisor metadata\n")

    def test_dry_run_does_not_acquire_or_create_gpu_lock(self):
        lock = self.root / "preview.lock"
        result = self.serve(DRY_RUN="1", GPU_LOCK_FILE=str(lock))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(lock.exists())
        self.assertIn("preview does not acquire", result.stdout)
        lock.write_text("manager metadata\n")
        with lock.open("r+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.serve(DRY_RUN="1", GPU_LOCK_FILE=str(lock))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lock.read_text(), "manager metadata\n")
        self.assertTrue((Path(self.env["STATE_DIR"]) / "config.preview.yml").is_file())

    def test_unset_option_retains_runtime_preflight(self):
        result = self.serve()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no venv", result.stderr)
        self.assertNotIn("GPU supervisor", result.stderr)

    def test_invalid_lock_paths_refuse_without_opening_special_files(self):
        target = self.root / "target"
        target.write_text("preserve\n")
        link = self.root / "link"
        link.symlink_to(target)
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        cases = [
            ("relative.lock", "absolute path"),
            (str(self.root / "bad\nname"), "without newlines"),
            (str(link), "symlink"),
            (str(fifo), "regular file"),
            (str(self.root), "regular file"),
            (str(self.root / "missing" / "lock"), "parent directory"),
            (str(Path(self.env["STATE_DIR"]) / "serve.lock"), "must differ"),
        ]
        for path, message in cases:
            with self.subTest(path=path):
                result = self.serve(GPU_LOCK_FILE=path)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn("no venv", result.stderr)
        self.assertEqual(target.read_text(), "preserve\n")
        self.assertFalse((ROOT / "relative.lock").exists())

    def test_lock_survives_exec_and_is_released_on_server_exit(self):
        lock = self.root / "persistent.lock"
        lock.write_text("owner metadata survives\n")
        script = 'source "$1"; acquire_gpu_lock; shift; exec "$@"'
        child = subprocess.Popen(
            ["bash", "-euc", script, "gpu-lock-test", str(RECIPE / "env.sh"),
             sys.executable, "-u", "-c",
             'import sys; print("READY", flush=True); sys.stdin.read(1)'],
            env={**self.env, "GPU_LOCK_FILE": str(lock)},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready, _, _ = select.select([child.stdout], [], [], 5)
            self.assertTrue(ready, "executed process did not become ready")
            self.assertEqual(child.stdout.readline().strip(), "READY")
            blocked = subprocess.run(["flock", "-n", str(lock), "true"], timeout=5)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertEqual(lock.read_text(), "owner metadata survives\n")
            _, stderr = child.communicate("x", timeout=5)
            self.assertEqual(child.returncode, 0, stderr)
            released = subprocess.run(["flock", "-n", str(lock), "true"], timeout=5)
            self.assertEqual(released.returncode, 0)
            self.assertEqual(lock.read_text(), "owner metadata survives\n")
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
