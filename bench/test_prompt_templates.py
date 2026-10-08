"""CPU checks for explicit template selection, provenance and Cyber rendering.

The real Tabby renderer checks are optional: set TABBY_SOURCE to a checkout and
run with its base-dependency Python. No model, tokenizer, API or GPU is loaded.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "exllamav3-tabby"
ASSET = RECIPE / "templates/cyber-frost-3.87bpw-thinking.jinja"
PROVENANCE = ASSET.with_suffix(".provenance.json")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


state = load("template_runtime_state", RECIPE / "tools/runtime_state.py")
matrix = load("template_matrix", ROOT / "bench/run_matrix.py")


class TemplateProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'explicit "quoted": template.jinja'
        self.path.write_bytes(b"User: {{ messages[0].content }}\r\nAssistant:\r")
        self.pack = self.root / "pack"
        self.pack.mkdir()
        (self.pack / "config.json").write_text("{}")
        (self.pack / "synthetic.safetensors").write_bytes(b"fixture")

    def job(self):
        return matrix.normalize({"label": "template", "model_path": str(self.pack),
            "env": {"PROFILE": "single", "NGRAM_RAM": False,
                    "PROMPT_TEMPLATE": str(self.path)}})

    def test_disk_and_loaded_text_have_distinct_matching_contracts(self):
        raw = self.path.read_bytes()
        loaded = self.path.read_text(encoding="utf-8")
        expected = state.prompt_template_state(str(self.path))
        self.assertEqual(expected, matrix.prompt_template_identity(str(self.path)))
        self.assertEqual(expected["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(expected["content_sha256"], hashlib.sha256(loaded.encode()).hexdigest())
        self.assertNotEqual(expected["sha256"], expected["content_sha256"])
        alias = self.root / "alias.jinja"
        alias.symlink_to(self.path)
        aliased = state.prompt_template_state(str(alias))
        self.assertEqual(aliased["path"], str(alias))
        self.assertEqual(aliased["resolved_path"], str(self.path))
        self.assertEqual(aliased["sha256"], expected["sha256"])

    def test_invalid_paths_and_encoding_fail_before_a_server_can_start(self):
        bad_utf8 = self.root / "invalid.jinja"
        bad_utf8.write_bytes(b"\xff")
        directory = self.root / "directory.jinja"
        directory.mkdir()
        for path in ("relative.jinja", self.root / "missing.jinja",
                     self.path.with_suffix(".txt"), directory, bad_utf8):
            for function in (state.prompt_template_state, matrix.prompt_template_identity):
                with self.subTest(path=path, function=function.__name__), self.assertRaises((ValueError, OSError)):
                    function(str(path))

    def test_snapshot_records_override_and_rejects_different_config(self):
        config = self.root / "config.yml"
        args = SimpleNamespace(engine=self.root, server=self.root, config=config,
                               model=None, build_state=None)
        config.write_text(yaml.safe_dump({"model": {"prompt_template": str(self.path)}}))
        with patch.dict(os.environ, {"PROMPT_TEMPLATE": str(self.path)}), \
             patch.object(state, "source_state", return_value={"commit": "synthetic"}):
            recorded = state.snapshot(args)
            self.assertEqual(recorded["prompt_template_override"],
                             state.prompt_template_state(str(self.path)))
            self.assertEqual(recorded["environment"]["PROMPT_TEMPLATE"], str(self.path))
            config.write_text("model:\n  prompt_template: null\n")
            with self.assertRaisesRegex(ValueError, "Rendered config"):
                state.snapshot(args)

    def test_matrix_refuses_fallback_content_and_false_deployment_provenance(self):
        job = self.job()
        expected = job["prompt_template_identity"]
        deployment = {"prompt_template_override": expected,
                      "config": {"values": {"model": {"prompt_template": str(self.path)}}}}
        loaded = {"parameters": {"prompt_template_content": self.path.read_text()}}
        matrix.verify_prompt_template(job, deployment, loaded)
        for invalid in ({}, {"parameters": {"prompt_template_content": "pack fallback"}},
                        {"parameters": {"prompt_template_content": None}}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "Loaded template"):
                matrix.verify_prompt_template(job, deployment, invalid)
        altered = deepcopy(deployment)
        altered["config"]["values"]["model"]["prompt_template"] = None
        with self.assertRaisesRegex(ValueError, "Deployed config"):
            matrix.verify_prompt_template(job, altered, loaded)
        with self.assertRaisesRegex(ValueError, "Deployment template"):
            matrix.verify_prompt_template(job, {}, loaded)
        # Ordinary jobs retain their existing contract.
        matrix.verify_prompt_template({}, {}, {})

    def test_external_file_change_invalidates_measurement_and_resume_identity(self):
        job = self.job()
        first_digest = matrix.digest(job)
        args = SimpleNamespace(recipe=self.root, runtime=self.root)
        identity, env = {"fixture": True}, {}
        with patch.object(matrix, "source_identity", return_value=identity), \
             patch.object(matrix, "resolved_env", return_value=env):
            matrix.verify_inputs(job, args, identity, env)
            self.path.write_text("Changed: {{ messages[0].content }}")
            with self.assertRaisesRegex(ValueError, "prompt template changed"):
                matrix.verify_inputs(job, args, identity, env)
        self.assertNotEqual(first_digest, matrix.digest(self.job()))

    def test_asset_is_only_the_declared_prefix_removal(self):
        provenance = json.loads(PROVENANCE.read_text())
        raw = ASSET.read_bytes()
        original = provenance["removed_prefix"].encode() + raw
        self.assertEqual(len(raw), 8971)
        self.assertEqual(len(original), 9005)
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "666b82b29f5801f4f546e5724b45bf5f14be7d20b66149df44164626b072ce6d")
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         "ba1946683f7615254fb246f0c0a652fd3aa02066ef8328ed4b8af08219749395")


TABBY_SOURCE = os.environ.get("TABBY_SOURCE")


@unittest.skipUnless(TABBY_SOURCE, "set TABBY_SOURCE and use Tabby's base-dependency Python")
class RealTabbyTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(TABBY_SOURCE).resolve()))
        from common.templating import PromptTemplate, find_prompt_template
        cls.template_class = PromptTemplate
        cls.find_template = staticmethod(find_prompt_template)
        cls.override_text = ASSET.read_text(encoding="utf-8")
        provenance = json.loads(PROVENANCE.read_text())
        cls.original = PromptTemplate("original", provenance["removed_prefix"] + cls.override_text)
        cls.override = PromptTemplate("override", cls.override_text)

    def render(self, template, values):
        return asyncio.run(template.render(deepcopy(values)))

    def fixtures(self):
        tool = {"type": "function", "function": {"name": "record", "description": "Keep literal text.",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                           "required": ["text"], "additionalProperties": False}}}
        return [
            {"messages": [{"role": "user", "content": "Reply briefly."}]},
            {"messages": [{"role": "system", "content": "Client instruction: café <literal>."},
                          {"role": "user", "content": "Use the record function."}], "tools": [tool]},
            {"messages": [{"role": "user", "content": "Record the exact source."},
                          {"role": "assistant", "content": "", "reasoning_content": "Past reasoning.",
                           "tool_calls": [{"type": "function", "function": {"name": "record",
                                           "arguments": {"text": "    code\n<literal>\n"}}}]},
                          {"role": "tool", "content": "Stored café <literal>."},
                          {"role": "user", "content": "Continue."}], "tools": [tool]},
        ]

    def test_default_and_explicit_true_remain_byte_identical(self):
        for fixture in self.fixtures():
            for thinking in (None, True):
                for prompt in (False, True):
                    for effort in ("xhigh", "medium", "low"):
                        values = dict(fixture, add_generation_prompt=prompt, reasoning_effort=effort)
                        if thinking is not None:
                            values["enable_thinking"] = thinking
                        with self.subTest(thinking=thinking, prompt=prompt, effort=effort):
                            self.assertEqual(self.render(self.original, values), self.render(self.override, values))

    def test_false_only_changes_existing_reasoning_instruction_and_prefix_branches(self):
        note = ("Reasoning effort is set to xhigh. Please think carefully through the task, "
                "validate key assumptions, consider plausible alternatives, and prioritize "
                "correctness, consistency, and clarity in the final answer.\n\n")
        for fixture in self.fixtures():
            with self.subTest(fixture=fixture):
                values = dict(fixture, add_generation_prompt=True, enable_thinking=False)
                original = self.render(self.original, values)
                self.assertTrue(original.endswith("<|im_start|>assistant\n<think>\n"))
                expected = original.replace(note, "", 1) + "\n</think>\n\n"
                rendered = self.render(self.override, values)
                self.assertEqual(rendered, expected)
                self.assertIn("These instructions are absolute", rendered)
                if fixture.get("tools"):
                    self.assertIn('"name": "record"', rendered)
                if len(fixture["messages"]) > 2:
                    self.assertIn("    code\n<literal>\n", rendered)
                    self.assertIn("Stored café <literal>.", rendered)

    def test_absolute_file_lookup_wins_over_model_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            (pack / "tabby_template.jinja").write_text("FALLBACK")
            template = asyncio.run(self.find_template(str(ASSET), pack))
            self.assertEqual(template.raw_template, self.override_text)
            self.assertNotEqual(template.raw_template, "FALLBACK")

    def test_launcher_renders_quoted_path_and_invalid_syntax_preserves_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            state_dir = root / "state"
            external = root / 'quoted "file": override.jinja'
            external.write_text(self.override_text)
            # Leave TABBY_DIR unset so the launcher's derived default is tested.
            runtime.mkdir()
            (runtime / "tabbyAPI").symlink_to(Path(TABBY_SOURCE).resolve(), target_is_directory=True)
            env = {key: value for key, value in os.environ.items()
                   if key not in {"TABBY_DIR", "VENV", "EXL3_SRC"}}
            env.update(RECIPE_HOME=str(runtime), STATE_DIR=str(state_dir), MODEL_DIR=str(root / "model"),
                       PYTHON_BIN=sys.executable, DRY_RUN="1", PROFILE="single", NGRAM_RAM="false",
                       BIGCORES="", PROMPT_TEMPLATE=str(external))
            command = ["bash", str(RECIPE / "serve.sh")]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            preview = state_dir / "config.preview.yml"
            self.assertEqual(yaml.safe_load(preview.read_text())["model"]["prompt_template"], str(external))
            saved = preview.read_bytes()
            external.write_text("{% invalid-jinja %}")
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(preview.read_bytes(), saved)
            env["PROMPT_TEMPLATE"] = ""
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNone(yaml.safe_load(preview.read_text())["model"]["prompt_template"])


if __name__ == "__main__":
    unittest.main()
