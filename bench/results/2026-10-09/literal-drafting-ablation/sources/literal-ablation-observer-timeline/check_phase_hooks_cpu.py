#!/usr/bin/env python3
"""CPU parity of diagnostic hooks over actual engine24 token-budget source."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib, importlib.util, json, sys
ROOT=Path(__file__).resolve().parent
ENGINE=Path('/home/vcruz/src/qwen-overnight-20261008/exllamav3-agent')

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

h=load('engine_budget_real_source',ENGINE/'tests/test_token_budget_cpu.py')
o=load('observer_timeline_cpu',ROOT/'observer/strings_observer.py')
originals={name:getattr(h.Job,name) for name in ('_advance_token_budget','_token_budget_can_end','constrain_output_now')}


def scenario(kind, observed):
 for name,method in originals.items():setattr(h.Job,name,method)
 job=h.make_job(max_new_tokens=30,banned_strings=["</phase>x"] if kind=="rewind" else None)
 calls=[]
 class Container:
  def set_generation_phase(self,request_id,reasoning):
   calls.append((request_id,reasoning))
   if kind=='callback_error':raise RuntimeError('synthetic callback error')
   job.set_filters([h.Filter(allowed=3)])
   return True
 container=Container();container.active_job_ids={'synthetic':NS(job=job)}
 module=NS(model=NS(container=container))
 record={'producer_events':[],'producer_events_dropped':0}
 recorder=NS(active={'synthetic':record})
 guard_calls=[]
 def guard(native):
  guard_calls.append(native.rq_new_tokens+native.new_tokens)
  if kind=='guard_error':raise RuntimeError('synthetic guard error')
  return kind!='protected' or len(guard_calls)>1
 if observed:o.install_phase_hooks(module,Container,h.Job,recorder)
 limit=2 if kind=='forced' else None
 job.set_token_budget(limit, h.torch.tensor([[h.END]]) if limit is not None else None,
   end_token_id=h.END,on_end=lambda native:container.set_generation_phase('synthetic',False),can_end=guard)
 results=[]
 tokens=[h.END,6,1,2,h.END,3] if kind=='rewind' else [1,2,1,3] if kind=='forced' else [1,h.END,h.END,3] if kind=='protected' else [1,2,h.END,3]
 for token in tokens:
  if job.is_finished:break
  h.sample(job,token,results)
  # Generator clears this per-iteration flag before its next acceptance step.
  if job.checkpoint_rewound:job.checkpoint_rewound=False
 output={'ids':h.emitted_ids(results),'text':job.full_completion,'new_tokens':job.new_tokens,
   'callbacks':calls,'guard_positions':guard_calls,'budget_present':job.token_budget is not None,
   'finished':job.is_finished,'filters_suspended':job.filters_suspended,
   'error_types':[type(x['error']).__name__ for x in results if 'error' in x],
   'filter_feeds':[f.fed for f in job.filters]}
 return output,record

rows=[]
for kind in ('natural','forced','protected','callback_error','guard_error','rewind'):
 expected,_=scenario(kind,False)
 got,record=scenario(kind,True)
 assert got==expected,(kind,expected,got)
 events=record['producer_events'];assert events and record['producer_events_dropped']==0
 assert [x['index'] for x in events]==list(range(len(events)))
 assert all(type(x['token_id']) is int for x in events if x['kind'].startswith('sample_processed_'))
 if kind=='forced':
  assert sum(x['kind']=='force_before' for x in events)==1
  assert sum(x['kind']=='force_after' for x in events)==1
 if kind=='protected':assert [x['allowed'] for x in events if x['kind']=='guard_after']==[False,True]
 if kind=='callback_error':assert any(x['kind']=='phase_error' for x in events)
 if kind=='guard_error':assert any(x['kind']=='guard_error' for x in events)
 if kind=='rewind':assert any(x['state']['checkpoint_rewound'] is True for x in events)
 rows.append({'scenario':kind,'native_results_identical':True,'event_count':len(events),'output':got,'events':events})
for name,method in originals.items():setattr(h.Job,name,method)
# Unrelated jobs are not observed. No __float__ or tensor access is needed.
job=h.make_job(max_new_tokens=30)
class Container:
 def set_generation_phase(self,*args):return 'unchanged-return'
container=Container();container.active_job_ids={'different':NS(job=job)}
module=NS(model=NS(container=container));record={'producer_events':[],'producer_events_dropped':0}
recorder=NS(active={'synthetic':record})
o.install_phase_hooks(module,Container,h.Job,recorder)
job.set_token_budget(None,end_token_id=h.END)
h.sample(job,1,[])
assert record['producer_events']==[]
# Trace cap is a diagnostic failure indicator, never a generation error.
container.active_job_ids['synthetic']=NS(job=job)
for _ in range(1100):job._advance_token_budget(1,False)
assert len(record['producer_events'])==2048 and record['producer_events_dropped']==152
for name,method in originals.items():setattr(h.Job,name,method)
report={'passed':True,'scope':'CPU source-execution parity only. Actual frozen engine Job token acceptance/budget routines with existing fake model/cache harness; native outputs/state/exceptions match unobserved execution. No model inference, GPU or HTTP.',
 'observer_sha256':sha(ROOT/'observer/strings_observer.py'),'test_sha256':sha(Path(__file__)),
 'engine_job_sha256':sha(ENGINE/'exllamav3/generator/job.py'),'engine_harness_sha256':sha(ENGINE/'tests/test_token_budget_cpu.py'),
 'scenarios':rows,'unrelated_job_ignored':True,'bounded_cap_and_drop_counter':True}
with (ROOT/'phase-hooks-cpu-report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'passed':True,'real_source_parity_scenarios':len(rows),'scope_and_cap_checks':2,'report_sha256':sha(ROOT/'phase-hooks-cpu-report.json')}))
