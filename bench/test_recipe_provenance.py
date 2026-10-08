"""Exercise fork migration and ABI drift using temporary repositories and a fake Torch."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_recipe_setup import RECIPE, state


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def commit(path, content):
    (path / "tracked.txt").write_text(content)
    git(path, "add", "tracked.txt")
    git(path, "-c", "user.name=Recipe Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "fixture")
    return git(path, "rev-parse", "HEAD")


class ForkMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.upstream, self.fork, self.checkout = root / "upstream", root / "fork", root / "checkout"
        self.upstream.mkdir()
        subprocess.run(["git", "init", "-qb", "main", str(self.upstream)], check=True)
        self.original = commit(self.upstream, "original\n")
        subprocess.run(["git", "clone", "-q", str(self.upstream), str(self.fork)], check=True)
        self.candidate = commit(self.fork, "candidate\n")
        subprocess.run(["git", "clone", "-q", str(self.upstream), str(self.checkout)], check=True)

    def tearDown(self):
        self.temp.cleanup()

    def helper(self, action):
        script = ('set -euo pipefail\n'
                  'die() { echo "$*" >&2; exit 1; }\n'
                  'say() { echo "$*" >&2; }\n'
                  + "source " + shlex.quote(str(RECIPE / "tools/git_helpers.sh")) + "\n"
                  + action)
        return subprocess.run(["bash", "-c", script, "test", str(self.checkout), str(self.fork)],
                              text=True, capture_output=True, timeout=20)

    def test_fork_fetch_preserves_origin_and_local_branch(self):
        result = self.helper('preflight_repo "$1"\nremote="$(sync_repo "$1" "$2")"\n'
                             'wanted="$(resolve_ref "$1" "$remote" main)"\n'
                             'git -C "$1" checkout -q --detach "$wanted"\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(git(self.checkout, "remote", "get-url", "origin"), str(self.upstream))
        self.assertEqual(git(self.checkout, "remote", "get-url", "recipe"), str(self.fork))
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.candidate)
        self.assertEqual(git(self.checkout, "rev-parse", "refs/heads/main"), self.original)

    def test_conflicting_recipe_remote_is_not_replaced(self):
        git(self.checkout, "remote", "add", "recipe", str(self.upstream))
        result = self.helper('remote="$(sync_repo "$1" "$2")"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already points elsewhere", result.stderr)
        self.assertEqual(git(self.checkout, "remote", "get-url", "recipe"), str(self.upstream))
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.original)

    def test_fetch_failure_cannot_be_hidden_by_command_substitution(self):
        result = self.helper('remote="$(sync_repo "$1" "$2/missing")"\necho BAD_SUCCESS\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("BAD_SUCCESS", result.stdout)
        self.assertEqual(git(self.checkout, "rev-parse", "HEAD"), self.original)


class FingerprintTests(unittest.TestCase):
    def test_equivalent_runtime_symlink_does_not_change_abi_fingerprint(self):
        torch = SimpleNamespace(__version__="2.13.0+cu130",
                                version=SimpleNamespace(cuda="13.0"),
                                _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime-dated"
            python = runtime / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.symlink_to(Path(sys.executable).resolve())
            (runtime / "engine").mkdir()
            alias = root / "qwen38-exl3"
            alias.symlink_to(runtime, target_is_directory=True)
            def command(args):
                return "Cuda compilation tools, release 13.0, V13.0.0" if args[0].endswith("nvcc") else "g++ fixture 14.2"
            with patch.dict(sys.modules, {"torch": torch}), patch.object(state, "git", return_value="a" * 40), \
                 patch.object(state, "command", side_effect=command):
                with patch.object(sys, "executable", str(python)):
                    direct = state.build_fingerprint(runtime / "engine", "/usr/local/cuda", "12.1")
                with patch.object(sys, "executable", str(alias / "venv/bin/python")):
                    through_alias = state.build_fingerprint(alias / "engine", "/usr/local/cuda", "12.1")
            self.assertEqual(direct, through_alias)
            self.assertEqual(direct["python_executable"], str(python.resolve()))
            self.assertEqual(direct["engine_path"], str((runtime / "engine").resolve()))

    def test_torch_and_compile_flags_change_fingerprint_and_check_fails(self):
        torch = SimpleNamespace(__version__="2.13.0+cu130",
                                version=SimpleNamespace(cuda="13.0"),
                                _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True))
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "build.json"
            def command(args):
                return "Cuda compilation tools, release 13.0, V13.0.0" if args[0].endswith("nvcc") else "g++ fixture 14.2"
            with patch.dict(sys.modules, {"torch": torch}), patch.object(state, "git", return_value="a" * 40), \
                 patch.object(state, "command", side_effect=command), patch.dict(os.environ, {"CXXFLAGS": "-O2"}):
                first = state.build_fingerprint(tmp, "/usr/local/cuda", "12.1")
                self.assertEqual(first, state.build_fingerprint(tmp, "/usr/local/cuda", "12.1"))
                record.write_text(json.dumps(first))
                torch.__version__ = "2.14.0+cu130"
                second = state.build_fingerprint(tmp, "/usr/local/cuda", "12.1")
                self.assertNotEqual(first["fingerprint_sha256"], second["fingerprint_sha256"])
                argv = ["runtime_state.py", "fingerprint", "--engine", tmp,
                        "--cuda-home", "/usr/local/cuda", "--arch", "12.1", "--compare", str(record)]
                stderr = io.StringIO()
                with patch.object(sys, "argv", argv), redirect_stderr(stderr), self.assertRaises(SystemExit) as exc:
                    state.main()
                self.assertEqual(exc.exception.code, 1)
                self.assertIn("torch", stderr.getvalue())
                torch.__version__ = "2.13.0+cu130"
                with patch.dict(os.environ, {"CXXFLAGS": "-O3"}):
                    third = state.build_fingerprint(tmp, "/usr/local/cuda", "12.1")
                self.assertNotEqual(first["fingerprint_sha256"], third["fingerprint_sha256"])


if __name__ == "__main__":
    unittest.main()
