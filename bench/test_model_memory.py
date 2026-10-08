"""CPU checks for the recipe's K8/V8 and QSA memory advisory."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "exllamav3-tabby/tools/model_memory.py"
spec = importlib.util.spec_from_file_location("model_memory", HELPER)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)

QWEN = {
    "num_hidden_layers": 48,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "full_attention_interval": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "mtp_num_hidden_layers": 1,
    "indexer_head_dim": 128,
    "indexer_compress_ratio": 4,
    "indexer_kv_heads": 1,
}
QWEN["layer_types"] *= 12


def write_pack(path, config):
    header = {
        "model.layers.1.ple.ple_embedding.ngram_embedding.trellis": {
            "dtype": "I16", "shape": [2, 2], "data_offsets": [0, 8],
        },
        "model.embed_tokens.weight": {
            "dtype": "F16", "shape": [2, 2], "data_offsets": [8, 16],
        },
    }
    encoded = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(16))
    (path / "config.json").write_text(json.dumps(config))


class CacheGeometryTests(unittest.TestCase):
    def test_qwen_single_pool_counts_scales_and_qsa_planes_once(self):
        result = memory.cache_bytes(QWEN, 262144, "mtp")
        self.assertEqual(result["kv_data_bytes"], int(3.25 * memory.GIB))
        self.assertEqual(result["kv_scale_bytes"], int(.203125 * memory.GIB))
        self.assertEqual(result["qsa_raw_key_bytes"], int(.8125 * memory.GIB))
        self.assertEqual(result["qsa_pooled_key_bytes"], int(.203125 * memory.GIB))
        self.assertEqual(result["qsa_index_bytes"], int(1.015625 * memory.GIB))
        self.assertEqual(result["kv_estimate_bytes"], int(4.46875 * memory.GIB))
        self.assertEqual(result["assumptions"]["qsa_layers"], 13)

    def test_qwen_concurrent_pool(self):
        result = memory.cache_bytes(QWEN, 1048576, "mtp")
        self.assertEqual(result["kv_estimate_bytes"], int(17.875 * memory.GIB))
        self.assertEqual(result["qsa_index_bytes"], int(4.0625 * memory.GIB))

    def test_disabling_mtp_removes_its_kv_scales_and_qsa_planes(self):
        result = memory.cache_bytes(QWEN, 262144, "disabled")
        self.assertEqual(result["kv_estimate_bytes"], int(4.125 * memory.GIB))
        self.assertEqual(result["assumptions"]["mtp_layers"], 0)
        self.assertEqual(result["assumptions"]["qsa_layers"], 12)

    def test_ordinary_gqa_without_hybrid_or_qsa_configuration(self):
        ordinary = {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 128}
        result = memory.cache_bytes(ordinary, 1024, "disabled")
        self.assertEqual(result["kv_data_bytes"], 64 * 1024**2)
        self.assertEqual(result["kv_scale_bytes"], 4 * 1024**2)
        self.assertEqual(result["kv_estimate_bytes"], 68 * 1024**2)
        self.assertEqual(result["qsa_index_bytes"], 0)
        self.assertEqual(result["assumptions"]["full_attention_layers"], 32)
        self.assertEqual(result["assumptions"]["cache_layout"], "gqa")

    def test_interval_fallback_and_native_page_rounding(self):
        config = {k: v for k, v in QWEN.items() if k != "layer_types"}
        result = memory.cache_bytes(config, 257, "mtp")
        self.assertEqual(result["assumptions"]["full_attention_layers"], 12)
        self.assertEqual(result["assumptions"]["allocated_cache_tokens"], 512)
        self.assertEqual(result["kv_estimate_bytes"], 9371648)

    def test_unsupported_or_incomplete_geometry_fails_closed(self):
        for changes in (
            {"indexer_compress_ratio": 3}, {"indexer_kv_heads": 2},
            {"indexer_head_dim": None}, {"num_key_value_heads": True},
            {"head_dim": 15}, {"layer_types": ["full_attention"]},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                memory.cache_bytes(dict(QWEN, **changes), 262144, "mtp")


class AdvisoryCompatibilityTests(unittest.TestCase):
    def test_ram_choice_uses_corrected_total_and_existing_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_pack(path, {"text_config": QWEN})
            args = SimpleNamespace(model=path, cache_size=262144, draft_mode="mtp", slack_gib=10)
            required = 16 + int(14.46875 * memory.GIB)
            with patch.object(memory, "available_bytes", return_value={"MemAvailable": required - 1}):
                result = memory.estimate(args)
            self.assertEqual(result["need_ram_bytes"], required)
            self.assertEqual(result["need_stream_bytes"], required - 8)
            self.assertFalse(result["ngram_ram_auto"])
            self.assertEqual(result["memory"]["MemAvailable"], required - 1)
            self.assertEqual(result["slack_bytes"], 10 * memory.GIB)
            self.assertTrue(result["slack_covers"])
            with patch.object(memory, "available_bytes", return_value={"MemAvailable": required}):
                self.assertTrue(memory.estimate(args)["ngram_ram_auto"])

    def test_cli_json_reports_components_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_pack(path, QWEN)
            result = subprocess.run(
                [sys.executable, str(HELPER), "--model", str(path), "--cache-size", "262144",
                 "--draft-mode", "mtp", "--ngram-ram", "true", "--json"],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["kv_estimate_bytes"], int(4.46875 * memory.GIB))
            self.assertIn("QSA 1.02", result.stderr)
            self.assertIn("need_ram_bytes", data)
            self.assertIn("need_stream_bytes", data)
            self.assertIn("memory", data)

    def test_auto_selection_returns_false_for_unrecognized_qsa_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_pack(path, dict(QWEN, indexer_head_dim=None))
            result = subprocess.run(
                [sys.executable, str(HELPER), "--model", str(path), "--cache-size", "262144",
                 "--choose-ngram"],
                capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout.strip(), "false")
            self.assertIn("memory estimate unavailable", result.stderr)


if __name__ == "__main__":
    unittest.main()
