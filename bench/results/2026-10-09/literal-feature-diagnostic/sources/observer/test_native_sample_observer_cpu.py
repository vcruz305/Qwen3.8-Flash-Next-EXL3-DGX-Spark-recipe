#!/usr/bin/env python3
"""CPU provenance/parity checks for diagnostic native observation. No native extension."""
import ast,copy,hashlib,importlib.util,json,os,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/vcruz/src/qwen-overnight-20261008/exllamav3-agent')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
h=load('native_budget_cpu_proof',ENGINE/'tests/test_token_budget_cpu.py')
o=load('native_sample_observer_proof',HERE/'observer/native_sample_observer.py')
class Recorder:
 def __init__(self):self.events=[]
 def event(self,kind,value):self.events.append((kind,value))
def normalize(value):
 if isinstance(value,torch.Tensor):
  assert value.device.type=='cpu';return value.tolist()
 if isinstance(value,dict):return {k:normalize(v) for k,v in value.items() if k!='job' and not k.startswith('time_')}
 if isinstance(value,(list,tuple)):return [normalize(x) for x in value]
 if isinstance(value,BaseException):return [type(value).__name__,str(value)]
 return value
def run_case(kind,observe):
 kw={}
 if kind=='rewind':kw['banned_strings']=['ab']
 if kind=='stop':kw['stop_conditions']=[h.END]
 if kind=='healing':kw['token_healing']=True
 if kind=='length':kw['max_new_tokens']=2
 job=h.make_job(**kw);rec=Recorder();restore=None;results=[];returned=[];caught=None
 if kind in ('forced','healing'):h.arm(job,0 if kind=='healing' else 2)
 if kind=='callback_error':
  def error(j):raise ValueError('callback fixture')
  h.arm(job,0,callback=error)
 if kind=='decode_exception':
  job.generator.tokenizer.pieces[13]='�'
  job.generator.tokenizer.decode=lambda *a,**k: (_ for _ in ()).throw(ValueError('decode fixture'))
 if observe:restore=o.install_native_sample_hook(h.Job,lambda j:rec if j is job else None,tokenizer_class=h.Tokenizer)
 try:
  if kind=='mtp':
   run=h.run_mtp(job,[1,1]);results=run['results']
  else:
   tokens={'plain':[1,2,3],'forced':[1,1,1,2],'healing':[1,1,2],'rewind':[1,2,3],'unicode':[13,1],'stop':[1,h.END],'length':[1,2],'callback_error':[1],'decode_exception':[13]}[kind]
   for token in tokens:
    job.checkpoint_rewound=False
    r,results=h.sample(job,token,results);returned.append(normalize(r))
    if r[0]:break
 except BaseException as exc:caught=[type(exc).__name__,str(exc)]
 finally:
  if restore:restore()
 state={'results':normalize(results),'returned':returned,'exception':caught,'new_tokens':job.new_tokens,'full_completion':job.full_completion,'held_text':job.held_text,'held_tokens':job.held_tokens.torch().tolist(),'sequence':job.sequences[0].sequence_ids.torch().tolist(),'rewound':job.checkpoint_rewound,'budget_active':job.token_budget is not None}
 return state,rec.events,restore.audit if restore else None
class NativeObserver(unittest.TestCase):
 def test_actual_source_parity(self):
  for kind in ('plain','forced','healing','rewind','unicode','stop','length','callback_error','decode_exception','mtp'):
   with self.subTest(kind=kind):
    before,_,_=run_case(kind,False);after,events,audit=run_case(kind,True)
    self.assertEqual(before,after);self.assertTrue(events);self.assertEqual(audit['observation_errors'],0);self.assertEqual(audit['lookup_errors'],0)
    samples=[p for k,p in events if k=='native_sample_after']
    if kind!='decode_exception':self.assertTrue(samples)
    for sample in samples:self.assertEqual(sample['processed_token']['device'],'cpu')
 def test_existing_unicode_decode_is_observed(self):
  _,events,audit=run_case('unicode',True);self.assertGreater(audit['decode_calls'],0)
  before=[p for k,p in events if k=='native_decode_before'];self.assertEqual(before[0]['input']['ids'],[13])
 def test_processed_eos_is_not_promised_emitted(self):
  state,events,_=run_case('stop',True)
  last=[p for k,p in events if k=='native_sample_after'][-1]
  self.assertEqual(last['processed_token']['ids'],[h.END]);self.assertTrue(last['eos']);self.assertNotIn(h.END,[token for row in state['results'] for group in row.get('token_ids',[]) for token in group])
 def test_unrelated_job_inert(self):
  job=h.make_job();rec=Recorder();restore=o.install_native_sample_hook(h.Job,lambda j:None,tokenizer_class=h.Tokenizer)
  try:h.sample(job,1);job.generator.tokenizer.decode(torch.tensor([[1]]))
  finally:restore()
  self.assertEqual(rec.events,[]);self.assertEqual(restore.audit['sample_calls'],0);self.assertEqual(restore.audit['decode_calls'],0)
 def test_no_gpu_read_or_transfer(self):
  class Foreign:
   device=NS(type='cuda')
   def numel(self):raise AssertionError('must not inspect CUDA tensor')
   def cpu(self):raise AssertionError('must not copy CUDA tensor')
  self.assertTrue(o._cpu_ids(Foreign())['not_read'])
 def test_recorder_failure_preserves_native(self):
  job=h.make_job()
  class Bad:
   def event(self,*a):raise RuntimeError('diagnostic sink failure')
  restore=o.install_native_sample_hook(h.Job,lambda j:Bad(),tokenizer_class=h.Tokenizer)
  try:r,events=h.sample(job,1)
  finally:restore()
  self.assertEqual(r[1].item(),1);self.assertEqual(job.full_completion,'a');self.assertEqual(restore.audit['observation_errors'],2)
 def test_real_decode_source_pad_filter_is_visible(self):
  # Actual24 decode methods; a tiny fake HF backend only supplies ordinary pieces.
  source=ENGINE/'exllamav3/tokenizer/tokenizer.py';namespace={'torch':torch,'FIRST_MM_EMBEDDING_INDEX':1000000}
  cls=h._load_class(source,'Tokenizer',namespace,{'decode','decode_','decode_unspecial'})
  tokenizer=cls.__new__(cls);tokenizer.raw_vocab_size=32;tokenizer.pad_token_id=17;tokenizer.eos_token_id=0
  tokenizer.extended_id_to_piece={17:'PAD'};tokenizer.unspecial_id_to_piece={}
  tokenizer.tokenizer=NS(decode=lambda ids,**kw:''.join({1:'a',13:'�',17:'PAD'}[i] for i in ids))
  class Job:
   def receive_sample(self):return False,torch.tensor([[17]]),False
  original=Job.receive_sample
  def with_decode(job):
   job.decoded=tokenizer.decode(torch.tensor([[17,13]]),decode_special_tokens=True)
   return original(job)
  Job.receive_sample=with_decode
  job=Job();job.generator=NS(tokenizer=tokenizer);job.sequences=[];rec=Recorder()
  restore=o.install_native_sample_hook(Job,lambda j:rec,tokenizer_class=cls)
  try:job.receive_sample()
  finally:restore()
  self.assertEqual(job.decoded,['�'])
  self.assertEqual(next(v for k,v in rec.events if k=='native_decode_before')['input']['ids'],[17,13])
  self.assertEqual(next(v for k,v in rec.events if k=='native_decode_after')['result'][0]['text'],'�')
if __name__=='__main__':
 assert not torch.cuda.is_initialized()
 unittest.main(verbosity=2)
