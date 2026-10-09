#!/usr/bin/env python3
"""CPU-only contract tests for the scheduled attribution controller."""
import copy, importlib.util, json, os, signal, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("attribution",ROOT/"run_attribution.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
EXPECTED=json.loads((ROOT/"original-fixture.json").read_text())
RAW=json.loads((ROOT/"testdata/baseline-305.json").read_text())
def args():
 return types.SimpleNamespace(recipe=Path("/recipe"),old_runtime=Path("/old"),new_runtime=Path("/new"),old_tabby=Path("/isolated816"))
def report(case="code"):
 r=copy.deepcopy(RAW);r["cases"]={case:r["cases"][case]};return r
class Contracts(unittest.TestCase):
 def test_supported_two_source_cells_only(self):
  a,c=m.cells(args())
  self.assertEqual([(r["engine_sha"],r["tabby_sha"]) for r in (a,c)],
   [(m.OLD_ENGINE,m.OLD_TABBY),(m.NEW_ENGINE,m.NEW_TABBY)])
  self.assertEqual(m.TUNING["NGRAM_RAM"],"true")
  self.assertTrue(m.MODEL.endswith("flashnext-exl3-3.05bpw"))
 def test_actual_frozen_client_payloads_match_all_original_hashes(self):
  benchdir=ROOT/"frozen"
  sys.path.insert(0,str(benchdir))
  try:
   import bench_v1,hashlib
   for case in ("code","prose"):
    for index in range(4):
     prompt=bench_v1.build_prompt(bench_v1.PROMPTS[case],0,f"overnight-v1-{case}-{index}")
     payload=bench_v1.payload_for(m.ALIAS,prompt,400,False)
     actual=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
     expected=EXPECTED["cases"][case]["warmup"][0] if index==0 else EXPECTED["cases"][case]["runs"][index-1]
     self.assertEqual(actual,expected["request_sha256"])
  finally:sys.path.remove(str(benchdir))
 def test_original_reports_validate(self):
  for c in ("code","prose"):self.assertEqual(m.validate_report(report(c),c,EXPECTED)["measured"],3)
 def test_server_output_trajectory_can_change_without_claiming_identical_work(self):
  r=report();r["cases"]["code"]["runs"][0]["response_sha256"]="different"
  r["cases"]["code"]["runs"][0]["usage"]["completion_tokens_details"]["accepted_prediction_tokens"]=1
  self.assertTrue(m.validate_report(r,"code",EXPECTED)["passed"])
 def test_incomplete_stream_fails(self):
  r=report();r["completed_at_utc"]=None
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_actual_short_output_fails(self):
  r=report();r["cases"]["code"]["runs"][0]["completion_tokens"]=399
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_cached_prompt_drift_fails(self):
  r=report();r["cases"]["code"]["runs"][0]["cached_prompt_tokens"]=1
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_changed_payload_fails(self):
  r=report();r["cases"]["code"]["runs"][1]["request_sha256"]="wrong"
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_extra_case_fails(self):
  with self.assertRaises(ValueError):m.validate_report(copy.deepcopy(RAW),"code",EXPECTED)
 def test_missing_repeat_fails(self):
  r=report();r["cases"]["code"]["runs"].pop()
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_missing_warmup_fails(self):
  r=report();r["cases"]["code"]["warmup"].clear()
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_no_draft_stats_fails(self):
  r=report();del r["cases"]["code"]["runs"][0]["usage"]["completion_tokens_details"]["accepted_prediction_tokens"]
  with self.assertRaises(KeyError):m.validate_report(r,"code",EXPECTED)
 def test_changed_sampling_fails(self):
  r=report();r["settings"]["temperature"]=1
  with self.assertRaises(ValueError):m.validate_report(r,"code",EXPECTED)
 def test_commands_keep_actual_original_fixture(self):
  a=args()
  for cell in m.cells(a):
   for case in ("code","prose"):
    cmd=m.benchmark_command(a,cell,case,Path("/out"))
    self.assertEqual(cmd[cmd.index("--run-id")+1],"overnight-v1")
    self.assertEqual(cmd[cmd.index("--max-tokens")+1],"400")
    self.assertEqual(cmd[cmd.index("--repeat")+1],"3")
    self.assertEqual(cmd[cmd.index("--warmup")+1],"1")
    self.assertNotIn("--thinking",cmd)
    self.assertEqual(cmd[0],str(cell["runtime"]/"venv/bin/python"))
 def test_old_alias_response_contract_is_explicit(self):
  cs=m.cells(args())
  vals=[]
  for c in cs:
   cmd=m.benchmark_command(args(),c,"code",Path("/out"));vals.append(cmd[cmd.index("--response-model")+1])
  self.assertEqual(vals,[Path(m.MODEL).name,m.ALIAS])
 def test_c_environment_uses_final_preserved_server(self):
  class H:
   def resolved_env(self,recipe,runtime,job):
    return dict(job["env"],TABBY_DIR=str(runtime/"tabbyAPI"),API_KEY="secret")
  a=args();b=m.cells(a)[1];v=m.environment(H(),a,b,Path("/output"),"owner")
  self.assertEqual(v["TABBY_DIR"],"/new/tabbyAPI")
  self.assertEqual(v["PYTHONPATH"],"/new/exllamav3")
  self.assertEqual(v["EXL3_MIN_VERSION"],"1.6.0.post1")
  self.assertNotIn("API_KEY",v)
 def test_no_unqualified_old_engine_new_server_combo(self):
  self.assertFalse(any(c["engine_sha"]==m.OLD_ENGINE and c["tabby_sha"]==m.NEW_TABBY for c in m.cells(args())))
 def test_frozen_lifecycle_helpers_hashes(self):
  helper,parent=m.modules();self.assertEqual(parent.HELPER_SHA,m.HELPER)
 def test_pending_signal_cannot_escape_process_registration(self):
  helper,parent=m.modules();proc=None;token="cpu-attribution-owner-unique"
  old=signal.getsignal(signal.SIGTERM)
  def interrupt(signum,frame):raise KeyboardInterrupt("scheduled CPU test")
  signal.signal(signal.SIGTERM,interrupt)
  try:
   with self.assertRaises(KeyboardInterrupt):
    with m.spawn_guard():
     proc=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],
      env=dict(os.environ,QWEN_EXPERIMENT_OWNER=token),start_new_session=True)
     os.kill(os.getpid(),signal.SIGTERM)
   self.assertIsNotNone(proc)
   status=dict(line.split(":",1) for line in Path(f"/proc/{proc.pid}/status").read_text().splitlines() if ":" in line)
   mask=int(status["SigBlk"].strip(),16)
   self.assertFalse(mask & ((1 << (signal.SIGTERM-1)) | (1 << (signal.SIGINT-1))))
   cleanup=parent.stop_owned(helper,proc,token,timeout=5)
   self.assertTrue(cleanup["owned_group_empty"]);self.assertEqual(cleanup["signals"],["SIGTERM"]);self.assertEqual(cleanup["exit_code"],-signal.SIGTERM)
  finally:
   signal.signal(signal.SIGTERM,old)
   if proc is not None and proc.poll() is None:parent.stop_owned(helper,proc,token,timeout=5)
class DerivativeProof(unittest.TestCase):
 def test_inherited_lifecycle_and_gates_unchanged(self):
  import ast,hashlib
  parent=ROOT/"parent-run_attribution.py"
  self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),"73d179bd97a8b5870421b7f6e6c3757a8e7c620c8dab0757276b6ce19a751a77")
  old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(parent.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/"run_attribution.py").read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
  for name in old.keys()-{"cells","validate_deployment","run_cell","main"}:
   self.assertEqual(old[name],new[name],name)
  # For the four scope-changing functions, the exact textual diff is archived.
if __name__=="__main__":unittest.main(verbosity=2)
