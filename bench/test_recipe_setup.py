"""CPU checks for non-destructive setup, launcher sizing and provenance."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "exllamav3-tabby"


def module(name):
    path = RECIPE / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


memory = module("model_memory")
state = module("runtime_state")


def small_pack(path):
    path.mkdir(parents=True, exist_ok=True)
    # Put n-gram and ordinary tensors in one generically named shard.
    header = {
        "model.layers.0.weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 8]},
        "model.ple_embedding.trellis": {"dtype": "I16", "shape": [2, 2], "data_offsets": [8, 16]},
    }
    raw = json.dumps(header).encode()
    (path / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(16))
    (path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 48, "full_attention_interval": 4,
        "num_key_value_heads": 2, "head_dim": 256,
    }))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.recipe_home = self.root / "runtime with spaces"
        self.env = dict(os.environ)
        self.env.update(
            RECIPE_HOME=str(self.recipe_home), STATE_DIR=str(self.recipe_home / "state"),
            MODEL_DIR=str(self.root / "pack"), VENV=str(self.recipe_home / "venv"),
            EXL3_SRC=str(self.recipe_home / "engine"), TABBY_DIR=str(self.recipe_home / "server"),
            PYTHON_BIN=sys.executable, CUDA_HOME=str(self.root / "missing-cuda"),
            DRY_RUN="1", BIGCORES="", PROFILE="concurrent",
        )

    def tearDown(self):
        self.temp.cleanup()

    def serve(self, **overrides):
        env = {**self.env, **overrides}
        return subprocess.run(["bash", str(RECIPE / "serve.sh")], cwd=ROOT, env=env,
                              text=True, capture_output=True, timeout=20)

    def test_context_cap_refuses_before_creating_state(self):
        result = self.serve(MAX_SEQ_LEN="262145")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("trained", result.stderr)
        self.assertFalse(Path(self.env["STATE_DIR"]).exists())

    def test_invalid_cache_and_boolean_are_rejected(self):
        for values in ({"CACHE_SIZE": "abc"}, {"CACHE_SIZE": "1000001"},
                       {"NGRAM_RAM": "yes"}, {"VISION": "maybe"}, {"DRY_RUN": "false"}):
            with self.subTest(values=values):
                self.assertNotEqual(self.serve(**values).returncode, 0)

    def test_live_config_and_model_links_survive_dry_run(self):
        directory = Path(self.env["STATE_DIR"])
        directory.mkdir(parents=True)
        live = directory / "config.yml"
        live.write_text("live config marker\n")
        models = directory / "models"
        models.mkdir()
        existing = models / "existing"
        existing.symlink_to(self.root / "other-pack")
        result = self.serve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(live.read_text(), "live config marker\n")
        self.assertTrue(existing.is_symlink())
        config = yaml.safe_load((directory / "config.preview.yml").read_text())
        self.assertEqual(config["model"]["max_batch_size"], 4)
        self.assertEqual(config["model"]["cache_size"], 1048576)
        self.assertFalse(config["model"]["ngram_ram"])

    def test_paths_and_names_are_yaml_strings(self):
        name = 'Qwen: "quoted" pack'
        result = self.serve(SERVED_NAME=name)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((Path(self.env["STATE_DIR"]) / "config.preview.yml").read_text())
        self.assertEqual(config["model"]["model_name"], name)
        self.assertEqual(config["model"]["model_dir"], str(Path(self.env["STATE_DIR"]) / "models"))
        self.assertNotEqual(self.serve(SERVED_NAME="../outside").returncode, 0)

    def test_draft_off_and_chunk_overrides_are_rendered(self):
        result = self.serve(DRAFT_MODE="disabled", CHUNK_SIZE="4096", MAX_SEQ_LEN="32768",
                            CACHE_SIZE="32768", MAX_BATCH_SIZE="1", SYSMEM_RECURRENT_CACHE="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((Path(self.env["STATE_DIR"]) / "config.preview.yml").read_text())
        self.assertEqual(config["draft_model"]["draft_mode"], "disabled")
        self.assertEqual(config["model"]["chunk_size"], 4096)
        self.assertEqual(config["memory"]["sysmem_recurrent_cache"], 0)

    def test_single_auto_uses_header_estimate(self):
        small_pack(Path(self.env["MODEL_DIR"]))
        result = self.serve(PROFILE="single", MAX_SEQ_LEN="256", CACHE_SIZE="256", MEMORY_SLACK_GIB="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((Path(self.env["STATE_DIR"]) / "config.preview.yml").read_text())
        self.assertTrue(config["model"]["ngram_ram"])
        self.assertEqual(config["model"]["max_batch_size"], 1)

    def test_network_binding_enables_auth_unless_explicit(self):
        result = self.serve(HOST="0.0.0.0")
        self.assertEqual(result.returncode, 0, result.stderr)
        config = yaml.safe_load((Path(self.env["STATE_DIR"]) / "config.preview.yml").read_text())
        self.assertFalse(config["network"]["disable_auth"])


class SetupPreflightTests(unittest.TestCase):
    def repo(self, path):
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        (path / "tracked.txt").write_text("committed\n")
        subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", str(path), "-c", "user.name=Recipe Test",
                        "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)

    def test_dirty_server_refused_before_venv_or_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine, server = root / "engine", root / "server"
            self.repo(engine)
            self.repo(server)
            (server / "tracked.txt").write_text("user work in progress\n")
            engine_head = subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True)
            env = {**os.environ, "EXL3_SRC": str(engine), "TABBY_DIR": str(server),
                   "VENV": str(root / "venv"), "STATE_DIR": str(root / "state"),
                   "RECIPE_HOME": str(root / "runtime"), "PYTHON_BIN": sys.executable,
                   "CUDA_HOME": str(root / "does-not-exist")}
            result = subprocess.run(["bash", str(RECIPE / "setup.sh")], env=env,
                                    text=True, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("local changes", result.stderr)
            self.assertFalse((root / "venv").exists())
            self.assertFalse((root / "state").exists())
            self.assertEqual((server / "tracked.txt").read_text(), "user work in progress\n")
            self.assertEqual(subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True),
                             engine_head)

    def test_untracked_work_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "engine"
            self.repo(engine)
            (engine / "my-experiment.py").write_text("# preserve me\n")
            env = {**os.environ, "EXL3_SRC": str(engine), "TABBY_DIR": str(root / "server"),
                   "VENV": str(root / "venv"), "RECIPE_HOME": str(root / "runtime"),
                   "STATE_DIR": str(root / "state"), "PYTHON_BIN": sys.executable,
                   "CUDA_HOME": str(root / "does-not-exist")}
            result = subprocess.run(["bash", str(RECIPE / "setup.sh")], env=env,
                                    text=True, capture_output=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("untracked", result.stderr)
            self.assertEqual((engine / "my-experiment.py").read_text(), "# preserve me\n")
            self.assertFalse((root / "venv").exists())


class MemoryAndProvenanceTests(unittest.TestCase):
    def test_ngram_classified_by_tensor_not_just_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            small_pack(path)
            (path / "duplicate.safetensors").symlink_to(path / "model-00001.safetensors")
            result = memory.pack_bytes(path)
            self.assertEqual(result["ngram_bytes"], 8)
            self.assertEqual(result["other_weight_bytes"], 8)
            self.assertEqual(result["unique_files"], 1)

    def test_bad_header_uses_conservative_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "ngram.safetensors").write_bytes(b"invalid")
            result = memory.pack_bytes(path)
            self.assertEqual(result["ngram_bytes"], 7)
            self.assertTrue(result["warnings"])

    def test_redaction_keeps_benchmark_token_counts(self):
        result = state.redact({
            "api_key": "secret-value", "nested": {"authorization": "Bearer secret"},
            "draft_num_tokens": 5, "completion_tokens": 400, "cache_size": 1048576,
        })
        self.assertEqual(result["api_key"], "<redacted>")
        self.assertEqual(result["nested"]["authorization"], "<redacted>")
        self.assertEqual(result["draft_num_tokens"], 5)
        self.assertEqual(result["completion_tokens"], 400)


if __name__ == "__main__":
    unittest.main()
