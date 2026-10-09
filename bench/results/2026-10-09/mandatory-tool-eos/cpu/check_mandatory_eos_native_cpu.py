#!/usr/bin/env python3
"""Exact-source CPU Job/MTP and sampler proof; no model, CUDA extension, or API."""
from pathlib import Path
import argparse, ast, copy, dataclasses, enum, functools, hashlib, importlib.util, json, math, os, random, sys, unittest
from types import SimpleNamespace as NS
import torch
import torch.nn.functional as F
ENGINE=Path('/home/vcruz/src/qwen-overnight-20261008/exllamav3-agent')
FIXTURE=ENGINE/'tests/test_token_budget_cpu.py'
spec=importlib.util.spec_from_file_location('native_budget_fixture',FIXTURE)
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
CUSTOM=ENGINE/'exllamav3/generator/sampler/custom.py'
BASE=ENGINE/'exllamav3/generator/sampler/sampler.py'
def compile_nodes(path,predicate,ns):
 t=ast.parse(path.read_text());nodes=[copy.deepcopy(n) for n in t.body if predicate(n)]
 pre=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[pre,*nodes],type_ignores=[])),str(path),'exec'),ns)
ns={'__name__':__name__,'torch':torch,'math':math,'os':os,'random':random,'F':F,
    'dataclass':dataclasses.dataclass,'Enum':enum.Enum,'lru_cache':functools.lru_cache,
    'abstractmethod':lambda f:f,'override':lambda f:f,'Tokenizer':h.Tokenizer,
    'fused_sampler_enable':False,'buffered_arange':lambda n,device:torch.arange(n,device=device)}
compile_nodes(BASE,lambda n:isinstance(n,ast.ClassDef),ns)
compile_nodes(CUSTOM,lambda n:isinstance(n,(ast.ClassDef,ast.FunctionDef)),ns)
EOS=30
def sampler(bans=()):
 return ns['CustomSampler']([ns['SS_BanTokens'](list(bans)),ns['SS_Argmax']()])
def job(bans=(EOS,),**kwargs):
 j=h.make_job(sampler=sampler(bans),**kwargs);j.generator.tokenizer.actual_vocab_size=h.VOCAB;return j
def score(first,second=1,width=1):
 s=torch.zeros((1,width,h.VOCAB));s[...,first]=10;s[...,second]=5;return s
def take(j,s):
 rows=[];out=j.receive_sample(s,*j.receive_logits(s),rows);return out,rows
def arm(j,limit=None,output=None):
 content=sampler();calls=[]
 def done(native):
  calls.append(native.new_tokens);native.set_sampler(content)
 j.set_token_budget(limit,output,end_token_id=h.END,on_end=done)
 return content,calls
class NativeEosSamplerTests(unittest.TestCase):
 def test_actual_ban_and_argmax_preserve_other_bans(self):
  j=job(bans=(EOS,8));s=score(EOS,8);s[...,1]=4
  out,_=take(j,s)
  self.assertEqual(out[1].item(),1);self.assertFalse(out[0]);self.assertTrue(torch.isfinite(s).all())
  self.assertEqual(j.sampler.steps[0].mask.token_ids,[8,EOS])
 def test_proposed_implicit_eos_is_rejected_by_real_mtp_loop(self):
  j=job(stop_conditions=[EOS]);arm(j,None)
  state=h.run_mtp(j,[EOS,1,1],scores=score(EOS,1,width=4))
  self.assertEqual(h.emitted_ids(state['results']),[1])
  self.assertEqual(j.accepted_draft_tokens,0);self.assertEqual(j.rejected_draft_tokens,3)
  self.assertEqual(state['accepted_lengths'],[1]);self.assertEqual(state['state'].rewinds,[3])
  self.assertNotIn(j,state['completed_jobs'])
 def test_same_window_natural_and_forced_handoff_restore_sampler(self):
  for natural in (True,False):
   with self.subTest(natural=natural):
    j=job(stop_conditions=[EOS])
    content,calls=arm(j,None if natural else 1,None if natural else torch.tensor([[h.END]]))
    s=score(1,2,width=4)
    s[:,1,:]=score(h.END,EOS).view(h.VOCAB) if natural else score(EOS,2).view(h.VOCAB)
    s[:,2,:]=score(EOS,1).view(h.VOCAB)
    state=h.run_mtp(j,[1,h.END,EOS],scores=s)
    self.assertEqual(h.emitted_ids(state['results']),[1,h.END])
    self.assertEqual(calls,[2]);self.assertIs(j.sampler,content)
    self.assertTrue(state['results'][-1]['eos']);self.assertEqual(state['results'][-1]['eos_reason'],'stop_token')
    self.assertIsNone(j.token_budget)
 def test_explicit_stop_token_and_piece_string_remain_effective(self):
  for stop in (EOS,'[30]'):
   with self.subTest(stop=stop):
    j=job(bans=(),stop_conditions=[stop]);_,calls=arm(j,None)
    out,rows=take(j,score(EOS,1))
    self.assertTrue(out[0]);self.assertEqual(calls,[])
    self.assertEqual(rows[-1]['eos_reason'],'stop_token' if type(stop)is int else 'stop_string')
 def test_max_tokens_termination_still_wins(self):
  j=job(max_new_tokens=1);_,calls=arm(j,None)
  out,rows=take(j,score(EOS,1))
  self.assertTrue(out[0]);self.assertEqual(calls,[]);self.assertEqual(rows[-1]['eos_reason'],'max_new_tokens')
 def test_zero_budget_and_forced_tail_restore_only_after_drain(self):
  j=job(stop_conditions=[EOS]);content,calls=arm(j,0,torch.tensor([[h.END,10]]))
  out,_=take(j,score(EOS,1));self.assertEqual(out[1].item(),h.END);self.assertEqual(calls,[])
  self.assertIsNot(j.sampler,content)
  out,_=take(j,score(EOS,1));self.assertEqual(out[1].item(),10);self.assertEqual(calls,[2])
  self.assertIs(j.sampler,content)
  out,rows=take(j,score(EOS,1));self.assertTrue(out[0]);self.assertEqual(rows[-1]['eos_reason'],'stop_token')
 def test_interleaved_jobs_keep_separate_sampler_state(self):
  jobs=[job(bans=(EOS,) if i%2==0 else (),stop_conditions=[EOS]) for i in range(4)]
  for j in jobs:arm(j,None)
  observations=[]
  for j in reversed(jobs):
   out,_=take(j,score(EOS,1));observations.append((out[1].item(),out[0]))
  self.assertEqual(observations,[(EOS,True),(1,False),(EOS,True),(1,False)])
  self.assertEqual(len({id(j.sampler) for j in jobs}),4)
 def test_fused_plan_retains_ban_before_greedy_tail(self):
  ns['fused_sampler_enable']=True
  try:
   s=sampler((EOS,));self.assertEqual([type(x).__name__ for x in s.steps],['SS_BanTokens','SS_Fused'])
   self.assertEqual(s.steps[-1].mode,ns['SS_Fused'].MODE_GREEDY)
   self.assertFalse(s.supports_batch_verify);self.assertFalse(s.fused_only)
  finally:ns['fused_sampler_enable']=False

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
 assert not torch.cuda.is_initialized()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeEosSamplerTests))
 paths=[FIXTURE,CUSTOM,BASE,ENGINE/'exllamav3/generator/job.py',ENGINE/'exllamav3/generator/generator.py',Path(__file__)]
 report={'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'cuda_initialized':torch.cuda.is_initialized(),
 'engine_commit':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','source_files':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},
 'scope':'Production Job/SeqTensor/MTP acceptance and CustomSampler/TokenMask/SS_BanTokens/SS_Argmax execute from exact source on CPU. Model/logits/physical allocation are fixtures; fused stack is inspected but its CUDA kernel is not executed. This proves sequencing and sampler behavior, not model output or GPU throughput.',
 'limitations':['Tabby caller-explicit stop exclusion/wiring is independently tested by its own candidate suite.','Same-window handoff follows existing native budget callback; no engine source change.','No API/GPU/model operations.']}
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
 raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
