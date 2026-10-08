"""Optional integration check against a real Qwen template and Tabby parser.

No weights, GPU, model inference or network calls are used. Set both
QWEN_TEST_TABBY_SOURCE and QWEN_TEST_TOKENIZER_CONFIG explicitly to run.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import unittest


class QwenTemplateRoundtripTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("QWEN_TEST_TABBY_SOURCE")
        config = os.environ.get("QWEN_TEST_TOKENIZER_CONFIG")
        if not source and not config:
            raise unittest.SkipTest(
                "Set QWEN_TEST_TABBY_SOURCE and QWEN_TEST_TOKENIZER_CONFIG "
                "to run the real-template/parser CPU integration check"
            )
        if not source or not config:
            raise ValueError("Both explicit source and tokenizer-config paths are required")
        source_path = Path(source).expanduser().resolve(strict=True)
        config_path = Path(config).expanduser().resolve(strict=True)
        if not (source_path / "endpoints/OAI/utils/toolcall_formats/qwen3_coder.py").is_file():
            raise ValueError("QWEN_TEST_TABBY_SOURCE is not a TabbyAPI source checkout")
        raw = json.loads(config_path.read_text())
        template_text = raw.get("chat_template")
        if not isinstance(template_text, str):
            raise ValueError("The tokenizer config must contain a string chat_template")

        sys.path.insert(0, str(source_path))
        try:
            templating = importlib.import_module("common.templating")
            parser = importlib.import_module(
                "endpoints.OAI.utils.toolcall_formats.qwen3_coder"
            )
        finally:
            sys.path.remove(str(source_path))
        for module in (templating, parser):
            if not Path(module.__file__).resolve().is_relative_to(source_path):
                raise RuntimeError(
                    "A module from another Tabby source is already imported; "
                    "run this integration check in a fresh Python process"
                )
        cls.template = templating.PromptTemplate.environment.from_string(template_text)
        cls.parse_calls = staticmethod(parser.parse_toolcalls)

    async def test_nullable_object_payload_preserves_exact_values(self):
        # The application controls this schema. Wrapping the nullable field uses
        # Qwen's existing JSON-object representation without a server wire change.
        tools = [{
            "type": "function",
            "function": {
                "name": "record_nullable_payload",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "properties": {"value": {"type": ["string", "null"]}},
                            "required": ["value"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["payload"],
                    "additionalProperties": False,
                },
            },
        }]
        cases = [
            ("null", None),
            ("empty_string", ""),
            ("literal_null", "null"),
            ("quoted_null", '"null"'),
            ("numeric_text", "123"),
            ("boolean_text", "true"),
            ("json_text", '{"a":[1,false]}'),
            ("code_final_lf", "  x = 1\n"),
            ("leading_and_final_lf", "\n  x\n\n"),
            ("final_cr", "line\r"),
            ("crlf", "line\r\n"),
            ("unicode", "東京 ☃"),
            ("literal_xml", "<think>x</think> </function> <tool_call></tool_call>"),
        ]
        core_bodies = {
            "null": '{"value": null}',
            "empty_string": '{"value": ""}',
            "literal_null": '{"value": "null"}',
        }
        for name, value in cases:
            with self.subTest(case=name):
                payload = {"value": value}
                messages = [
                    {"role": "user", "content": "Record the supplied payload."},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "function": {
                                "name": "record_nullable_payload",
                                "arguments": {"payload": payload},
                            }
                        }],
                    },
                ]
                originals = copy.deepcopy((messages, tools))
                rendered = await self.template.render_async(
                    messages=messages, tools=tools, enable_thinking=False,
                    add_generation_prompt=False,
                )
                # Exclude the template's tool-format example. The last wrapper
                # close belongs to this history call, even with literal XML in data.
                start = rendered.find("<function=record_nullable_payload>")
                end = rendered.rfind("</tool_call>")
                self.assertGreaterEqual(start, 0)
                self.assertGreater(end, start)
                call = rendered[start:end + len("</tool_call>")]
                if name in core_bodies:
                    self.assertIn(
                        "<parameter=payload>\n" + core_bodies[name] + "\n</parameter>",
                        call,
                    )
                calls = self.parse_calls(call, tools=tools, strict=True)
                self.assertEqual(len(calls), 1)
                parsed = json.loads(calls[0].function.arguments)
                self.assertEqual(parsed, {"payload": payload})
                self.assertIs(type(parsed["payload"]["value"]), type(value))
                self.assertEqual((messages, tools), originals)


if __name__ == "__main__":
    unittest.main()
