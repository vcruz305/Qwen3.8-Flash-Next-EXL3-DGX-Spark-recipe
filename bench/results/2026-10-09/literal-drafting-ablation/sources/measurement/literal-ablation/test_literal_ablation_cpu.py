#!/usr/bin/env python3
"""CPU-only diagnostic controller contracts; never contacts Spark/API/CUDA."""
import contextlib,copy,fcntl,importlib.util,io,json,struct,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("ablation",ROOT/"run_literal_ablation.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
JOBS=json.loads((ROOT/"jobs.json").read_text())

class Contracts(unittest.TestCase):
 def test_only_draft_mode_may_differ(self):
  m.validate_jobs(JOBS)
  wrong=copy.deepcopy(JOBS);wrong[1]["env"]["DRAFT_NUM_TOKENS"]=3
  with self.assertRaisesRegex(ValueError,"Only DRAFT_MODE"):m.validate_jobs(wrong)
 def test_old_server_and_reversed_modes_rejected(self):
  for wrong in (copy.deepcopy(JOBS),list(reversed(copy.deepcopy(JOBS)))):
   if wrong[0]["label"]=="literal-disabled":wrong[0]["tabby"]="5a"*20
   with self.assertRaises(ValueError):m.validate_jobs(wrong)
 def owner(self,gpu="",manager="inactive",enabled="disabled"):
  values=["LoadState=loaded\nActiveState="+manager+"\nMainPID="+("0" if manager=="inactive" else "27"),
   "ActiveState=inactive\nMainPID=0",gpu]
  return patch.object(m,"command",side_effect=values),patch.object(m.subprocess,"run",
   return_value=types.SimpleNamespace(stdout=enabled+"\n",returncode=1))
 def test_owned_gpu_allowed_and_unowned_rejected(self):
  for gpu,allowed,okay in (("",None,True),("55, python, 9123",55,True),("55, python, 9123",None,False),
                           ("55, python, 1\n66, python, 2",55,False)):
   a,b=self.owner(gpu)
   with a,b:
    if okay:self.assertEqual(m.owner_gate("rexl3-manager.service",allowed)["qwen_enabled"],"disabled")
    else:
     with self.assertRaisesRegex(ValueError,"CUDA owner"):m.owner_gate("rexl3-manager.service",allowed)
 def test_manager_restart_rejected(self):
  a,b=self.owner(manager="active")
  with a,b,self.assertRaisesRegex(ValueError,"Manager must"):m.owner_gate("rexl3-manager.service")
 def test_qwen_enablement_rejected(self):
  a,b=self.owner(enabled="enabled")
  with a,b,self.assertRaisesRegex(ValueError,"disabled Qwen"):m.owner_gate("rexl3-manager.service")
 def test_unrecognized_gpu_output_rejected(self):
  a,b=self.owner(gpu="N/A, unknown, N/A")
  with a,b,self.assertRaisesRegex(ValueError,"Unrecognized GPU"):m.owner_gate("rexl3-manager.service")
 def test_model_header_same_stat_identity_still_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);path=root/"one.safetensors"
   header=json.dumps({"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}).encode()
   path.write_bytes(struct.pack("<Q",len(header))+header+b"\0\0")
   identity={"same":"recorded"};helper=types.SimpleNamespace(model_identity=lambda p:identity)
   record={"identity":identity,"headers_in_loader_order":[{"name":path.name,"header_bytes":len(header),
       "header_sha256":m.hashlib.sha256(header).hexdigest()}]}
   m.verify_model(helper,{"model_path":str(root)},record)
   path.write_bytes(struct.pack("<Q",len(header))+header.replace(b"[1]",b"[2]")+b"\0\0")
   with self.assertRaisesRegex(ValueError,"header changed"):m.verify_model(helper,{"model_path":str(root)},record)
 def test_control_input_drift_rejected(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/"job.json";path.write_text("one");a=types.SimpleNamespace(input_hashes={str(path):m.sha(path)})
   m.verify_control_inputs(a);path.write_text("two")
   with self.assertRaisesRegex(ValueError,"input changed"):m.verify_control_inputs(a)
 def test_preflight_does_not_claim_historical_cpu_suite(self):
  root=Path("/tmp/test-runtime");recipe=Path("/tmp/test-recipe")
  sources={k:{"path":str(p),"commit":sha,"dirty":""} for k,p,sha in
   [("recipe",recipe,m.RECIPE),("engine",root/"exllamav3",m.ENGINE),("server",root/"tabbyAPI",m.TABBY)]}
  row={"kind":"fresh-read-only-runtime-preflight","passed":True,"finished_at_utc":"now",
       "runtime":str(root),"recipe_commit":m.RECIPE,"sources":sources,"setup_check_exit_code":0,
       "imports":{"engine_file":str(root/"exllamav3/exllamav3/__init__.py"),"version":"1.6.0.post1",
                  "budget_parameters":["self","max_tokens","can_end"]}}
  bypath={v["path"]:v for v in sources.values()}
  with patch.object(m,"git_identity",side_effect=lambda p:bypath[str(p)]):
   m.check_preflight(row,recipe,root,sources)
   wrong=copy.deepcopy(row);wrong["setup_check_exit_code"]=1
   with self.assertRaises(ValueError):m.check_preflight(wrong,recipe,root,sources)
   wrong=copy.deepcopy(row);wrong["imports"]["engine_file"]="/other/exllamav3/__init__.py"
   with self.assertRaises(ValueError):m.check_preflight(wrong,recipe,root,sources)
 def sequence(self,capture=True,interrupt=None):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);lock=root/"gpu.lock";lock.touch()
   args=types.SimpleNamespace(output=root,gpu_lock=lock,lock_inode=lock.stat().st_ino,
      manager_unit="rexl3-manager.service",input_hashes={})
   state={"cells":[],"lock_inode":args.lock_inode};calls=[]
   def cell(a,job,path,out,inputs):
    calls.append(job["label"]);out.mkdir()
    summary={"capture_valid":capture,"passed":False,"original_semantic_counts":{"fail":8},
             "interruption_observed":interrupt}
    (out/"literal-capture.json").write_text(json.dumps(summary));return summary
   with patch.object(m,"run_cell",side_effect=cell),patch.object(m,"owner_gate",return_value={"safe":True}):
    if capture and interrupt is None:
     m.run_sequence(args,JOBS,{},state,lambda:None)
     self.assertEqual(calls,["literal-disabled","literal-mtp"])
     self.assertEqual(state["state"],"completed");self.assertFalse(state["passed"]);self.assertTrue(state["capture_valid"])
    else:
     with self.assertRaises(ValueError):m.run_sequence(args,JOBS,{},state,lambda:None)
     self.assertEqual(calls,["literal-disabled"])
   return state
 def test_semantic_failures_retained_and_next_cell_runs(self):self.sequence()
 def test_invalid_capture_stops_next_cell(self):self.sequence(capture=False)
 def test_interruption_stops_next_cell(self):self.sequence(interrupt=15)
 def test_busy_cooperative_lock_never_enters_owner_or_server_gate(self):
  observer=ROOT.parents[1]/"literal-ablation-observer-timeline/observer"
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);lock=root/"gpu.lock";lock.touch()
   argv=["--jobs",str(ROOT/"jobs.json"),"--output",str(root/"result"),"--runtime",str(root),
         "--observer-dir",str(observer),"--model-inputs",str(ROOT/"model-inputs.json"),"--client-root",str(root)]
   with lock.open("r+") as held:
    fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with patch.object(m,"GPU_LOCK",lock),patch.object(m,"owner_gate") as owner,contextlib.redirect_stdout(io.StringIO()):
     self.assertEqual(m.main(argv),2);owner.assert_not_called()
   result=json.loads((root/"result/result.json").read_text())
   self.assertEqual(result["state"],"failed");self.assertIn("BlockingIOError",result["error"])
 def test_actual_lock_held_through_sequence_and_released_after(self):
  observer=ROOT.parents[1]/"literal-ablation-observer-timeline/observer"
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);lock=root/"gpu.lock";lock.touch()
   argv=["--jobs",str(ROOT/"jobs.json"),"--output",str(root/"result"),"--runtime",str(root),
         "--observer-dir",str(observer),"--model-inputs",str(ROOT/"model-inputs.json"),"--client-root",str(root)]
   def inside(args,jobs,inputs,state,save):
    with lock.open("r+") as another:
     with self.assertRaises(BlockingIOError):fcntl.flock(another,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state.update(state="completed",passed=True,capture_valid=True)
   with patch.object(m,"GPU_LOCK",lock),patch.object(m,"run_sequence",side_effect=inside),contextlib.redirect_stdout(io.StringIO()):
    self.assertEqual(m.main(argv),0)
   with lock.open("r+") as another:fcntl.flock(another,fcntl.LOCK_EX|fcntl.LOCK_NB)
if __name__=="__main__":unittest.main()
