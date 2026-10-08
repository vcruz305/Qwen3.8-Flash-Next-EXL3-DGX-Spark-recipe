#!/usr/bin/env python3
"""Check the observer against real Tabby request normalization and frozen clients."""
import argparse
import ast
import asyncio
from types import SimpleNamespace
import tempfile
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tabby", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--recorded", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.tabby.resolve()))
    sys.path.insert(0, str((args.recipe / "bench").resolve()))
    from endpoints.OAI.types.chat_completion import ChatCompletionRequest
    import tool_smoke
    import strings_observer as observer
    case = next(c for c in tool_smoke.CASES if c.name == "strings")
    checks = []
    for stream in (False, True):
        wire = tool_smoke.payload_for(case, "Qwen3.8-Flash-Next-EXL3", stream, 1024)
        normalized = ChatCompletionRequest(**wire)
        assert observer.matching_request(normalized, stream), "Actual normalized client did not match"
        assert not hasattr(normalized, "seed"), "Seed retention changed; update the scope contract"
        checks.append({"kind": "actual_client", "stream": stream, "matched": True})
    # Run the real request formatting functions. Only the loaded model holder is
    # a CPU stand-in; Jinja rendering, Pydantic copies and template mutations are real.
    from common.templating import PromptTemplate, TemplateError
    from common.utils import unwrap
    from endpoints.OAI.utils.tool_choice import function_name, prepare_forced_tool_choice
    from endpoints.OAI.utils.qwen_tool_guidance import nullable_guidance_eligible, with_nullable_xml_guidance
    from test_strings_observer_cpu import classes
    source_path = args.tabby / "endpoints/OAI/utils/chat_completion.py"
    source_tree = ast.parse(source_path.read_text())
    selected = {"_sort_tool_messages", "format_messages_with_template", "resolve_template_vars",
                "normalize_message_roles", "apply_chat_template"}
    nodes = [node for node in source_tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in selected]
    assert {node.name for node in nodes} == selected
    template = PromptTemplate("observer-cpu", "{{ tools|tojson }}\n{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}assistant:{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}")
    container = SimpleNamespace(tool_format="qwen3_5", tokenizer=None, use_vision=False,
        template_vars_default={}, template_vars_force={}, prompt_template=template,
        get_special_tokens=lambda: {"bos_token": "", "eos_token": "<|im_end|>",
                                    "pad_token": None, "unk_token": ""},
        hf_model=SimpleNamespace(add_bos_token=lambda: False))
    namespace = {"model": SimpleNamespace(container=container), "json": json,
                 "unwrap": unwrap, "function_name": function_name,
                 "prepare_forced_tool_choice": prepare_forced_tool_choice,
                 "nullable_guidance_eligible": nullable_guidance_eligible,
                 "with_nullable_xml_guidance": with_nullable_xml_guidance,
                 "TemplateError": TemplateError}
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(source_path), "exec"), namespace)
    rendered_functions = {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest() for node in nodes}
    with tempfile.TemporaryDirectory() as temporary:
        recorder = observer.Recorder(Path(temporary) / "traces", "formatted-cpu", 2)
        module, cls, _, _, sentinel, _, _ = classes()
        observer.install_hooks(module, cls, recorder)
        for index, stream in enumerate((False, True)):
            normalized = ChatCompletionRequest(**tool_smoke.payload_for(case, "Qwen3.8-Flash-Next-EXL3", stream, 1024))
            prompt, embeddings = asyncio.run(namespace["apply_chat_template"](normalized))
            assert embeddings is None
            # The actual endpoint creates each collector request with this deep copy.
            copied = normalized.model_copy(deep=True)
            assert len(copied.template_vars) > 1 and copied.grammar_string
            assert observer.matching_request(copied, stream), copied.template_vars
            result = asyncio.run(module._chat_stream_collector(
                0, None, "formatted-" + str(index), prompt, copied, False, streaming_mode=stream))
            assert result is sentinel
            captured = json.loads((recorder.directory / f"request-{index:02d}.json").read_text())
            assert captured["rendered_prompt"] == prompt
            assert captured["matched_request"]["template_vars"] == copied.template_vars
            assert captured["raw_finish"]["native_full_completion_present"]
            checks.append({"kind": "actual_formatting_to_hooked_collector", "stream": stream,
                           "matched": True, "template_var_keys": sorted(copied.template_vars),
                           "rendered_prompt_sha256": observer.sha(prompt.encode())})
    recorded = []
    for path in args.recorded:
        value = json.loads(path.read_text())
        cases = [r for r in value["results"] if r["case"] == "strings"]
        assert len(cases) == 2 and {r["request"]["stream"] for r in cases} == {False, True}
        for result in cases:
            wire = result["request"]
            expected = tool_smoke.payload_for(case, wire["model"], wire["stream"], 1024)
            assert wire == expected, "Historical strings client request changed"
            assert observer.matching_request(ChatCompletionRequest(**wire), wire["stream"])
            checks.append({"kind": "historical_client", "file": path.name,
                           "stream": wire["stream"], "matched": True})
        recorded.append({"path": str(path.resolve()), "sha256": sha(path)})
    assert "torch" not in sys.modules, "Fixture check unexpectedly imported Torch"
    report = {"schema_version": 1, "passed": True, "checks": checks,
              "tabby_commit": subprocess.check_output(
                  ["git", "-C", str(args.tabby), "rev-parse", "HEAD"], text=True).strip(),
              "observer_sha256": sha(Path(observer.__file__)),
              "client_sha256": sha(Path(tool_smoke.__file__)),
              "recorded_reports": recorded,
              "actual_rendering_functions_sha256": rendered_functions,
              "rendering_scope": "Actual request-formatting source and Jinja renderer; model holder and collector body are CPU stand-ins. The observer entry/finish wrappers execute unchanged.",
              "note": "Actual Pydantic schema drops seed; the unchanged client still sends seed0 and uses greedy sampling."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "checks": len(checks), "output": str(args.output)}))


if __name__ == "__main__":
    main()
