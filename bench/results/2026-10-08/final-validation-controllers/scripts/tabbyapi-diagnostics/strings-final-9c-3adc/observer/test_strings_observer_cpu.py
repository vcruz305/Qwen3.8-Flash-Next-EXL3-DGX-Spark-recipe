"""CPU-only observation/lifecycle tests; no inference or native backend import."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import strings_observer as obs


def params(stream=False):
    return SimpleNamespace(messages=[{"role": "user", "content": obs.USER_MESSAGE}],
        tools=deepcopy(obs.TOOLS), tool_choice="auto", parallel_tool_calls=True,
        max_tokens=1024, temperature=0, top_k=1, top_p=1.0, seed=0, n=1,
        template_vars={"enable_thinking": False}, functions=None, stream=stream,
        model="Qwen3.8-Flash-Next-EXL3")


def classes(behavior="normal"):
    result = {"full_completion": "<tool_call>native</tool_call>", "new_tokens": 31,
              "cached_tokens": 0, "accepted_draft_tokens": 14, "eos": True}
    finish_value = {"gen_tokens": 31, "finish_reason": "stop"}
    failure = RuntimeError("original failure identity")
    calls = []
    sentinel = object()
    class Container:
        def handle_finish_chunk(self, native, request_id, full_text, label=None):
            calls.append((native, request_id, full_text, label))
            if behavior == "finish_raise":
                raise failure
            return finish_value
    container = Container()
    async def collector(task_idx, gen_queue, request_id, prompt, params,
                        start_in_reasoning_mode, mm_embeddings=None,
                        streaming_mode=True, disconnect_handler=None, label=None):
        if behavior == "raise":
            raise failure
        if behavior == "return_error":
            return failure
        if behavior != "noop":
            value = container.handle_finish_chunk(result, request_id, "backend accumulated text", label)
            assert value is finish_value
        return sentinel
    return SimpleNamespace(_chat_stream_collector=collector), Container, container, result, sentinel, failure, calls


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
    def tearDown(self):
        self.temporary.cleanup()
    def recorder(self, maximum=2):
        return obs.Recorder(self.root / "traces", "final405", maximum, {"test": True})
    def invoke(self, module, request_id="synthetic-a", p=None, stream=False):
        return asyncio.run(module._chat_stream_collector(
            0, None, request_id, "exact rendered prompt<think>", p or params(stream),
            False, streaming_mode=stream, label="original label"))

    def test_return_identity_and_both_existing_finish_strings_preserved(self):
        rec = self.recorder()
        module, C, container, native, sentinel, _, calls = classes()
        obs.install_hooks(module, C, rec)
        self.assertIs(self.invoke(module), sentinel)
        self.assertIs(calls[0][0], native)
        self.assertEqual(calls[0][1:], ("synthetic-a", "backend accumulated text", "original label"))
        report = json.loads((rec.directory / "request-00.json").read_text())
        self.assertEqual(report["rendered_prompt"], "exact rendered prompt<think>")
        self.assertEqual(report["rendered_prompt_sha256"], obs.sha(report["rendered_prompt"].encode()))
        self.assertEqual(report["raw_finish"]["native_full_completion"], native["full_completion"])
        self.assertEqual(report["raw_finish"]["backend_full_response"], "backend accumulated text")
        self.assertEqual(report["raw_finish"]["native_metrics"]["new_tokens"], 31)
        self.assertFalse(rec.active)
        self.assertEqual((rec.directory / "request-00.json").stat().st_mode & 0o777, 0o600)

    def test_exact_request_matching_in_both_modes(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assertTrue(obs.matching_request(params(stream), stream))
                self.assertFalse(obs.matching_request(params(stream), not stream))
        variants = {
            "messages": [{"role": "user", "content": "private unrelated prompt"}],
            "tools": obs.TOOLS + obs.TOOLS, "tool_choice": "required",
            "template_vars": {"enable_thinking": True}, "max_tokens": 1023,
            "parallel_tool_calls": False, "temperature": 0.1, "top_k": 2,
            "top_p": 0.9, "n": 2, "functions": [{"name": "private"}],
        }
        for key, value in variants.items():
            with self.subTest(field=key):
                p = params()
                setattr(p, key, value)
                self.assertFalse(obs.matching_request(p, False))
        p = params()
        p.tools[0]["function"]["parameters"]["properties"]["tag_text"] = {"type": "number"}
        self.assertFalse(obs.matching_request(p, False))

    def test_actual_framework_template_augmentation_matches_and_conflicts_do_not(self):
        p = params()
        p.template_vars.update(messages=deepcopy(p.messages), tools=deepcopy(p.tools),
                               functions=None, add_generation_prompt=True, tool_choice="auto",
                               parallel_tool_calls=True, bos_token="", eos_token="<|im_end|>",
                               pad_token=None, unk_token="")
        self.assertTrue(obs.matching_request(p, False))
        for key, value in (("messages", []), ("tools", []), ("functions", [{}]),
                           ("tool_choice", "required"), ("add_generation_prompt", False),
                           ("parallel_tool_calls", False), ("private_extra", "unexpected")):
            with self.subTest(field=key):
                changed = deepcopy(p)
                changed.template_vars[key] = value
                self.assertFalse(obs.matching_request(changed, False))

    def test_unmatched_request_and_over_cap_are_delegated_without_capture(self):
        rec = self.recorder()
        module, C, _, _, sentinel, _, _ = classes("noop")
        obs.install_hooks(module, C, rec)
        private = params()
        private.messages[0]["content"] = "private prompt"
        self.assertIs(self.invoke(module, p=private), sentinel)
        self.assertEqual(rec.count, 0)
        for index in range(4):
            self.assertIs(self.invoke(module, request_id=str(index)), sentinel)
        self.assertEqual(rec.count, 2)
        self.assertEqual(rec.skipped_matching_requests, 2)
        self.assertEqual(len(list(rec.directory.glob("request-*.json"))), 2)
        self.assertNotIn("private prompt", "".join(p.read_text() for p in rec.directory.glob("*.json")))

    def test_original_exception_and_returned_error_identities_are_preserved(self):
        for behavior in ("raise", "finish_raise", "return_error"):
            with self.subTest(behavior=behavior):
                rec = obs.Recorder(self.root / behavior, "finalcyber", 2)
                module, C, _, _, _, failure, _ = classes(behavior)
                obs.install_hooks(module, C, rec)
                if behavior == "return_error":
                    self.assertIs(self.invoke(module), failure)
                else:
                    try:
                        self.invoke(module)
                    except RuntimeError as error:
                        self.assertIs(error, failure)
                    else:
                        self.fail("Exception was swallowed")
                report = json.loads((rec.directory / "request-00.json").read_text())
                self.assertEqual(report["collector_returned_error"], behavior == "return_error")
                self.assertEqual(report["raised_exception_type"], None if behavior == "return_error" else "RuntimeError")
                self.assertNotIn("original failure identity", json.dumps(report))

    def test_publication_failure_preserves_original_return_or_exception(self):
        for behavior in ("normal", "raise"):
            with self.subTest(behavior=behavior):
                rec = obs.Recorder(self.root / behavior, "final405", 2)
                module, C, _, _, sentinel, failure, _ = classes(behavior)
                obs.install_hooks(module, C, rec)
                with patch.object(rec, "end", side_effect=OSError("diagnostic disk failure")), patch.object(obs.sys, "stderr") as stderr:
                    if behavior == "normal":
                        self.assertIs(self.invoke(module), sentinel)
                    else:
                        try:
                            self.invoke(module)
                        except RuntimeError as error:
                            self.assertIs(error, failure)
                        else:
                            self.fail("Original exception was swallowed")
                    stderr.write.assert_called_once_with("Strings observer publication failed: OSError\n")

    def test_no_native_tensor_coercion_or_reads(self):
        class Forbidden:
            def __str__(self): raise AssertionError("native conversion")
            def __int__(self): raise AssertionError("native conversion")
            def tolist(self): raise AssertionError("native read")
            def item(self): raise AssertionError("native read")
        rec = self.recorder()
        module, C, _, native, sentinel, _, _ = classes()
        native["new_tokens"] = Forbidden()
        native["full_completion"] = Forbidden()
        obs.install_hooks(module, C, rec)
        self.assertIs(self.invoke(module), sentinel)
        raw = json.loads((rec.directory / "request-00.json").read_text())["raw_finish"]
        self.assertIsNone(raw["native_full_completion"])
        self.assertFalse(raw["native_full_completion_present"])
        self.assertIsNone(raw["native_metrics"]["new_tokens"])

    def test_publication_refuses_existing_directory_file_or_symlink(self):
        rec = self.recorder()
        with self.assertRaises(FileExistsError):
            self.recorder()
        path = self.root / "new.json"
        obs.atomic_new(path, {"a": 1})
        with self.assertRaises(FileExistsError):
            obs.atomic_new(path, {"a": 2})
        self.assertEqual(json.loads(path.read_text()), {"a": 1})
        link = self.root / "dangling.json"
        link.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            obs.atomic_new(link, {"a": 3})
        self.assertFalse((self.root / "missing").exists())
        self.assertEqual(list(self.root.glob(".strings-observer-*")), [])

    def test_invalid_caps_and_labels_fail_before_claiming_directory(self):
        for value in (0, 3, True, "2"):
            with self.assertRaises(ValueError):
                obs.Recorder(self.root / "invalid", "final405", value)
        for label in ("", "../outside", "label with spaces"):
            with self.assertRaises(ValueError):
                obs.Recorder(self.root / "invalid", label)
        self.assertFalse((self.root / "invalid").exists())

    def test_activation_occurs_only_inside_exact_server_coroutine(self):
        target = self.root / "main.py"
        events = []
        namespace = {"events": events}
        exec(compile("async def entrypoint_async():\n events.append('body')\n return 7\n", str(target), "exec"), namespace)
        wrapper = obs.make_run_wrapper(asyncio.run, target, lambda: events.append("activate"))
        async def unrelated(): return 4
        self.assertEqual(wrapper(unrelated()), 4)
        self.assertEqual(events, [])
        self.assertEqual(wrapper(namespace["entrypoint_async"]()), 7)
        self.assertEqual(events, ["activate", "body"])

    def test_failed_activation_closes_original_coroutine(self):
        target = self.root / "main.py"
        namespace = {}
        exec(compile("async def entrypoint_async():\n return 7\n", str(target), "exec"), namespace)
        def fail(): raise RuntimeError("blocked")
        coro = namespace["entrypoint_async"]()
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            obs.make_run_wrapper(asyncio.run, target, fail)(coro)
        self.assertIsNone(coro.cr_frame)

    def test_nonserver_startup_is_inert_even_with_missing_config(self):
        env = dict(os.environ, PYTHONPATH=str(Path(obs.__file__).parent),
                   TABBY_STRINGS_OBSERVER_CONFIG=str(self.root / "missing.json"))
        r = subprocess.run([sys.executable, "-c",
            "import sys; assert 'strings_observer' not in sys.modules; print('inert')"],
            env=env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "inert")

    def test_real_site_startup_defers_native_import_until_configured(self):
        tabby = self.root / "tabby"
        engine = self.root / "engine"
        tabby.mkdir()
        engine.mkdir()
        configuration = self.root / "observer.json"
        configuration.write_text(json.dumps({
            "tabby_repo": str(tabby), "engine_repo": str(engine),
            "tabby_commit": obs.TABBY_HEAD, "engine_commit": obs.ENGINE_HEAD,
            "run_label": "final405", "max_records": 2, "output_dir": str(self.root / "startup")}))
        main = tabby / "main.py"
        main.write_text("""import asyncio,json,os,sys,types
import strings_observer as obs
assert 'backends.exllamav3.model' not in sys.modules
assert 'torch' not in sys.modules
assert hasattr(asyncio.run,'__wrapped__')
cfg=json.loads(open(os.environ['TABBY_STRINGS_OBSERVER_CONFIG']).read())
obs.verified_source=lambda p,s: {'path':str(p),'commit':s,'tracked_changes':''}
cc=types.ModuleType('endpoints.OAI.utils.chat_completion')
cc.__file__=cfg['tabby_repo']+'/endpoints/OAI/utils/chat_completion.py'
async def collector(request_id,prompt,params,streaming_mode=True,start_in_reasoning_mode=False): return None
cc._chat_stream_collector=collector
model=types.ModuleType('backends.exllamav3.model')
model.__file__=cfg['tabby_repo']+'/backends/exllamav3/model.py'
class C:
 def handle_finish_chunk(self,*a): return {}
model.ExllamaV3Container=C
job=types.ModuleType('exllamav3.generator.job')
job.__file__=cfg['engine_repo']+'/exllamav3/generator/job.py'
sys.modules[cc.__name__]=cc;sys.modules[model.__name__]=model;sys.modules[job.__name__]=job
os.environ['OBSERVER_CONFIG_READY']='1'
async def entrypoint_async():
 assert os.environ['OBSERVER_CONFIG_READY']=='1'
 assert hasattr(cc._chat_stream_collector,'__wrapped__')
 assert not hasattr(asyncio.run,'__wrapped__')
 assert 'torch' not in sys.modules
 return 19
assert asyncio.run(entrypoint_async())==19
print('configured and observed without torch')
""")
        env = dict(os.environ, PYTHONPATH=str(Path(obs.__file__).parent),
                   TABBY_STRINGS_OBSERVER_CONFIG=str(configuration))
        r = subprocess.run([sys.executable, str(main)], env=env, text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("configured and observed without torch", r.stdout)
        self.assertTrue((self.root / "startup/observer-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
