"""CPU-only checks for the bounded literal capture lifecycle adapter."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

c = module("literal_capture", ROOT / "literal_capture_controller_p095.py")
fixture = module("literal_original", ROOT / "reasoning_literal_smoke.py")
off = module("literal_off", ROOT / "literal_thinking_off_smoke.py")
observer = module("literal_observer", ROOT / "tabbyapi-diagnostics/literal-final-24f0-5a-p095/observer/strings_observer.py")
lib = c.load_exact(ROOT / "strings_only_controller.py", c.BASE_SHA, "frozen_attachment_test")
wrapper = c.load_exact(ROOT / "final_api_controller.py", c.WRAPPER_SHA, "frozen_wrapper_test")
SAVED = ROOT / "literal-final-triage-5a/original-final-305-single"
ALIAS = "Qwen3.8-Flash-Next-EXL3"
PROMPT = (ROOT / "literal-final-triage-5a/rendered-prompt.txt").read_text()

def original(unbudgeted=False):
    return json.loads((SAVED / ("literal-unbudgeted.json" if unbudgeted else "literal.json")).read_text())

def fake_trace(expected, index=0):
    # Synthetic record metadata tests the adapter only; not generation or parsing.
    wire = expected["request"]; key = expected["key"]
    identifier = expected["id"].split("-", 1)[1]
    return {
        "index": index, "run_label": "literal-cpu", "request_id": identifier,
        "streaming_mode": key[2], "start_in_reasoning_mode": True,
        "raised_exception_type": None, "collector_returned_error": False,
        "skipped_matching_requests_so_far": 0, "rendered_prompt": PROMPT,
        "rendered_prompt_sha256": c.PROMPT_SHA,
        "variant": {"unbudgeted": key[0], "choice": key[1], "stream": key[2]},
        "matched_request": {
            "model": ALIAS, "messages": wire["messages"], "tools": wire["tools"],
            "tool_choice": wire["tool_choice"], "reasoning_budget_tokens": wire.get("reasoning_budget_tokens"),
            "parallel_tool_calls": False, "max_tokens": 256, "temperature": 0,
            "top_k": 1, "top_p": 0.95, "n": 1, "stream": key[2],
            "template_vars": {"enable_thinking": True},
        },
        "started_monotonic_ns": 1, "finished_monotonic_ns": 3,
        "raw_finish": {"native_full_completion_present": True,
                       "native_full_completion": "synthetic raw fixture",
                       "backend_full_response": "synthetic raw fixture",
                       "backend_finish_monotonic_ns": 2},
    }

class AdapterTests(unittest.TestCase):
    def test_all_sources_are_exact(self):
        for path, expected in (
            (ROOT / "strings_only_controller.py", c.BASE_SHA),
            (ROOT / "final_api_controller.py", c.WRAPPER_SHA),
            (ROOT / "reasoning_literal_smoke.py", c.CLIENT_SHA),
            (ROOT / "literal_thinking_off_smoke.py", c.WORKAROUND_SHA),
            (ROOT / "tabbyapi-diagnostics/literal-final-24f0-5a-p095/observer/strings_observer.py", c.OBSERVER_SHA),
            (ROOT / "tabbyapi-diagnostics/literal-final-24f0-5a-p095/observer/sitecustomize.py", c.SITE_SHA),
        ):
            self.assertEqual(c.sha(path), expected)
    def test_original_actual_failed_reports_remain_failed_with_valid_capture_metadata(self):
        for unbudgeted in (False, True):
            rows = c.validate_client(original(unbudgeted), {"passed": False, "exit_code": 1},
                                     fixture, ALIAS, unbudgeted)
            self.assertEqual([r["status"] for r in rows], ["fail"] * 4)
            for index, row in enumerate(rows):
                validated = c.validate_trace(fake_trace(row, index), index, row, observer, ALIAS, "literal-cpu")
                self.assertEqual(validated["semantic_status"], "fail")
    def test_semantic_failure_cannot_be_relabelled_passed(self):
        report = original(); report["passed"] = True
        with self.assertRaises(ValueError):
            c.validate_client(report, {"passed": True, "exit_code": 0}, fixture, ALIAS, False)
    def test_original_payload_mutation_rejected(self):
        report = original(); report["requests"][0]["request"]["enable_thinking"] = False
        with self.assertRaises(ValueError):
            c.validate_client(report, {"passed": False, "exit_code": 1}, fixture, ALIAS, False)
    def test_incomplete_sse_rejected(self):
        report = original(); report["requests"][1]["done"] = False
        with self.assertRaises(ValueError):
            c.validate_client(report, {"passed": False, "exit_code": 1}, fixture, ALIAS, False)
    def test_trace_id_and_prefix_must_match_actual_api(self):
        row = c.validate_client(original(), {"passed": False, "exit_code": 1}, fixture, ALIAS, False)[0]
        trace = fake_trace(row); trace["request_id"] += "bad"
        with self.assertRaises(ValueError):
            c.validate_trace(trace, 0, row, observer, ALIAS, "literal-cpu")
    def test_trace_scope_and_raw_completeness_fail_closed(self):
        row = c.validate_client(original(), {"passed": False, "exit_code": 1}, fixture, ALIAS, False)[0]
        changes = [
            ("start_in_reasoning_mode", False), ("rendered_prompt", PROMPT + " "),
            ("rendered_prompt_sha256", "0" * 64), ("skipped_matching_requests_so_far", 1),
            ("variant", {"unbudgeted": True, "choice": "required", "stream": False}),
            ("raw_finish", {"native_full_completion_present": False}),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                trace = fake_trace(row); trace[field] = value
                with self.assertRaises(ValueError):
                    c.validate_trace(trace, 0, row, observer, ALIAS, "literal-cpu")
    def test_retained_native_controls_are_checked(self):
        row = c.validate_client(original(), {"passed": False, "exit_code": 1}, fixture, ALIAS, False)[0]
        for field, value in (("max_tokens", 1024), ("top_k", 2),
                             ("parallel_tool_calls", True), ("reasoning_budget_tokens", None)):
            with self.subTest(field=field):
                trace = fake_trace(row); trace["matched_request"][field] = value
                with self.assertRaises(ValueError):
                    c.validate_trace(trace, 0, row, observer, ALIAS, "literal-cpu")
    def test_supplement_changes_only_existing_enable_thinking_setting(self):
        for choice in ("required", "named"):
            for streaming in (False, True):
                original_wire = fixture.payload(ALIAS, choice, streaming, unbudgeted=True)
                expected = deepcopy(original_wire); expected["enable_thinking"] = False
                self.assertEqual(off.payload(ALIAS, choice, streaming, unbudgeted=True), expected)
                self.assertIn("Think briefly", expected["messages"][0]["content"])
    def test_supplement_validator_preserves_literal_and_requires_thinking_off(self):
        def require(condition, message):
            if not condition: raise ValueError(message)
        good = {"finish_reason": "tool_calls", "reasoning_content": "", "content": "",
                "tool_calls": [{"function": {"name": "record_text",
                  "arguments": json.dumps({"text": fixture.LITERAL})}}]}
        off.validate(good, require)
        for changed in (
            {**good, "reasoning_content": "unexpected reasoning"},
            {**good, "tool_calls": [{"function": {"name": "record_text",
                  "arguments": json.dumps({"text": "thinkaliteralthink"})}}]},
        ):
            with self.assertRaises(ValueError):
                off.validate(changed, require)
    def test_supplement_cannot_run_budgeted_variant(self):
        with self.assertRaises(SystemExit):
            off.main(["--recipe", str(ROOT / "recipe"), "--model", ALIAS,
                      "--metadata", str(SAVED / "deployment.json"), "--output", "/tmp/not-created-literal.json"])
    def test_controller_commands_and_source_boundaries(self):
        for include in (False, True):
            with tempfile.TemporaryDirectory() as d:
                path = Path(d); bundle = path / "bundle"; bundle.mkdir()
                parent = wrapper.load_parent(ROOT / "api_f4_gemm_controller.py")
                job = {
                    "label": "literal-cpu", "engine": c.ENGINE, "tabby": c.TABBY,
                    "model_path": str(ROOT / "tabbyapi-agent/.tokenizer-cpu"),
                    "recipe": str(ROOT / "recipe"),
                    "env": {"PROFILE":"262k", "NGRAM_RAM":"true", "CACHE_SIZE":"262144",
                            "MAX_SEQ_LEN":"262144", "MAX_BATCH_SIZE":"1", "CHUNK_SIZE":"2048",
                            "DRAFT_MODE":"mtp", "DRAFT_NUM_TOKENS":"5", "DYNAMIC_DRAFT":"true",
                            "EXL3_MOE_COOP_KSPLIT":"1", "EXL3_GEMM_LEGACY_TILES":"1"},
                    "api": False, "literal_client": str(ROOT / "reasoning_literal_smoke.py"),
                }
                wrapper.configure(parent, job, path / "job.json", path / "out", path / "setup.json")
                before = list(parent.commands(False))
                guard = lib.SignalGuard()
                c.attach(parent, lib, bundle, guard, {"files_sha256": {}},
                         ROOT / "literal_thinking_off_smoke.py" if include else None)
                actual = list(parent.commands(False))
                self.assertEqual(actual[:2], before)
                self.assertEqual([row[0] for row in actual],
                                 ["literal", "literal-unbudgeted"] + (["literal-thinking-off"] if include else []))
                self.assertEqual(parent.EXPECTED, {"literal":4, "literal-unbudgeted":4,
                                **({"literal-thinking-off":4} if include else {})})
                self.assertNotIn("--bench", str(actual))
                self.assertEqual(parent.ENGINE, c.ENGINE); self.assertEqual(parent.TABBY, c.TABBY)
                self.assertEqual(parent.TUNING["MAX_BATCH_SIZE"], "1")
    def test_full_capture_integrity_is_independent_of_eight_semantic_failures(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/"out"; out.mkdir()
            job={"engine":c.ENGINE,"tabby":c.TABBY,"label":"literal-cpu"}
            info=c.prepare_bundle(lib,ROOT/"tabbyapi-diagnostics/literal-final-24f0-5a-p095/observer",
                                  root/"bundle",out,job,root/"runtime")
            provenance={
                "observer_sha256":c.OBSERVER_SHA,"sitecustomize_sha256":c.SITE_SHA,
                "config_sha256":c.digest((json.dumps(info["config"],indent=2,ensure_ascii=False)+"\n").encode()),
                **{name:{"commit":head,"path":info["config"][name+"_repo"],"tracked_changes":""}
                   for name,head in (("engine",c.ENGINE),("tabby",c.TABBY))},
            }
            recorder=observer.Recorder(out/"raw-literal","literal-cpu",8,provenance)
            clients=[]; index=0
            for unbudgeted in (False,True):
                name="literal-unbudgeted" if unbudgeted else "literal"
                report=original(unbudgeted); path=out/(name+".json")
                path.write_text(json.dumps(report))
                client={"name":name,"passed":False,"exit_code":1,"report":str(path),
                        "report_sha256":c.sha(path),"cleanup":{"owned_group_empty":True}}
                clients.append(client)
                expected=c.validate_client(report,client,fixture,ALIAS,unbudgeted)
                for row in expected:
                    trace=fake_trace(row,index)
                    (out/"raw-literal"/f"request-{index:02d}.json").write_text(json.dumps(trace))
                    index+=1
            (out/"deployment.json").write_bytes((SAVED/"deployment.json").read_bytes())
            result={"state":"completed","finished_at_utc":"cpu-fixture","passed":False,
                    "expected_engine":c.ENGINE,"expected_server":c.TABBY,"clients":clients,
                    "server_cleanup":{"owned_group_empty":True,"unexpected_exit":False},
                    "extra_files_sha256":info["files_sha256"],
                    "deployment_sha256":c.sha(out/"deployment.json")}
            (out/"result.json").write_text(json.dumps(result))
            result=c.assess(out,job,info,observer,fixture,ALIAS,1,False)
            self.assertTrue(result["capture_valid"])
            self.assertFalse(result["original_semantic_passed"])
            self.assertFalse(result["all_clients_passed"])
            self.assertEqual(result["original_semantic_counts"],{"fail":8})
            self.assertEqual(len(result["traces"]),8)
            state=json.loads((out/"result.json").read_text())
            state["server_cleanup"]["unexpected_exit"]=True
            (out/"result.json").write_text(json.dumps(state))
            with self.assertRaises(ValueError):
                c.assess(out,job,info,observer,fixture,ALIAS,1,False)

    def test_exclusive_bundle_claim(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d); bundle=path/"bundle"
            observer_dir=ROOT/"tabbyapi-diagnostics/literal-final-24f0-5a-p095/observer"
            job={"engine":c.ENGINE,"tabby":c.TABBY,"label":"literal-cpu"}
            info=c.prepare_bundle(lib,observer_dir,bundle,path/"out",job,path/"runtime")
            self.assertEqual(info["config"]["max_records"],8)
            self.assertEqual(c.sha(bundle/"strings_observer.py"),c.OBSERVER_SHA)
            with self.assertRaises(FileExistsError):
                c.prepare_bundle(lib,observer_dir,bundle,path/"out",job,path/"runtime")

if __name__ == "__main__":
    unittest.main(verbosity=2)
