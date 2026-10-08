"""Scoring gates for long-context retrieval; these fixtures do not use a model."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
SOURCE = Path(os.environ.get(
    "RECIPE_LONG_CONTEXT_SOURCE", str(Path(__file__).with_name("long_context.py"))
))
if not SOURCE.is_file():
    raise unittest.SkipTest("long_context.py is not present in this checkout yet")
spec = importlib.util.spec_from_file_location("recipe_long_context_under_test", SOURCE)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


class CharacterTokenizer:
    """Scoring fixture only; actual Qwen tokenizer geometry is checked separately."""
    @classmethod
    def from_file(cls, path):
        return cls()

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[ord(char) for char in text])

    def decode(self, tokens, skip_special_tokens=False):
        return "".join(chr(token) for token in tokens)


class LongContextScoringTests(unittest.TestCase):
    def run_main(self, *, cached=0, finish="stop", wrong=False, duplicate=False,
                 prompt_tokens=None, explicit_seed=True):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text("{}")
            output = root / "result.json"
            argv = ["long_context.py", "--tokenizer", str(tokenizer),
                    "--context-tokens", "1024", "--output", str(output)]
            if explicit_seed:
                argv += ["--run-id", "scoring-fixture"]

            def perform(base, key, payload, **kwargs):
                prompt = payload["messages"][0]["content"]
                pairs = re.findall(
                    r"project (ORCHID|MAPLE|CEDAR); verification value ([0-9A-F]{16})\.",
                    prompt,
                )
                self.assertEqual(len(pairs), 3)
                values = dict(pairs)
                if wrong:
                    values["ORCHID"] = "WRONG"
                content = json.dumps(values)
                if duplicate:
                    content = '{"ORCHID":"WRONG",' + content[1:]
                return {
                    "model": "fixture-model", "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                    "prompt_tokens": len(prompt) + 8 if prompt_tokens is None else prompt_tokens,
                    "cached_prompt_tokens": cached, "completion_tokens": 40,
                }

            with patch.object(sys, "argv", argv), \
                 patch.dict(sys.modules, {"tokenizers": SimpleNamespace(Tokenizer=CharacterTokenizer)}), \
                 patch.object(lc, "resolve_model", return_value="fixture-model"), \
                 patch.object(lc, "perform", side_effect=perform), \
                 contextlib.redirect_stdout(io.StringIO()):
                exit_code = lc.main()
            return exit_code, json.loads(output.read_text())

    def test_correct_cold_response_passes(self):
        code, report = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertTrue(report["retrieval_passed"])

    def test_cached_and_unknown_cache_responses_do_not_pass_cold_gate(self):
        for cached in (1, 1500, None):
            with self.subTest(cached=cached):
                code, report = self.run_main(cached=cached)
                self.assertEqual(code, 1)
                self.assertFalse(report["passed"])
                self.assertTrue(report["retrieval_passed"])
                self.assertTrue(any("Prompt-cache" in error for error in report["errors"]))

    def test_duplicate_corrected_key_cannot_hide_a_wrong_retrieval(self):
        code, report = self.run_main(duplicate=True)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertFalse(report.get("retrieval_passed", False))

    def test_wrong_value_length_stop_and_invalid_actual_size_are_rejected(self):
        for settings in (
            {"wrong": True}, {"finish": "length"}, {"prompt_tokens": 1023},
            {"prompt_tokens": 262145},
        ):
            with self.subTest(settings=settings):
                code, report = self.run_main(**settings)
                self.assertEqual(code, 1)
                self.assertFalse(report["passed"])
                self.assertFalse(report.get("retrieval_passed", False))

    def test_default_seed_is_generated_per_invocation_and_recorded(self):
        with patch.object(lc.uuid, "uuid4", side_effect=[
            SimpleNamespace(hex="a" * 32), SimpleNamespace(hex="b" * 32)
        ]):
            _, first = self.run_main(explicit_seed=False)
            _, second = self.run_main(explicit_seed=False)
        self.assertEqual(first["run_id"], "a" * 16)
        self.assertEqual(second["run_id"], "b" * 16)
        self.assertNotEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertNotEqual(first["expected"], second["expected"])

    def test_needles_occur_once_and_offsets_are_ordered(self):
        prompt, values, locations, count = lc.build_prompt(
            CharacterTokenizer(), 10000, "geometry-fixture"
        )
        self.assertEqual(count, len(prompt))
        offsets = [row["raw_text_token_offset"] for row in locations]
        self.assertEqual(offsets, sorted(offsets))
        for row in locations:
            self.assertEqual(prompt.count(row["value"]), 1)
            self.assertIn(row["project"], values)
            self.assertTrue(prompt[row["raw_text_token_offset"]:].startswith("\nVERIFIED PROJECT RECORD:"))
        self.assertEqual(
            values["ORCHID"],
            hashlib.sha256(b"geometry-fixture:ORCHID").hexdigest()[:16].upper(),
        )


if __name__ == "__main__":
    unittest.main()
