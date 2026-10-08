#!/usr/bin/env python3
"""Compose actual Tabby backend setup with native Job, base Filter and MTP control.

No model/server/GPU. Test fixtures replace allocation, scheduling, logits and the
filter's content language. The real backend chooses/attaches the filter, the
real base Filter journals tokens, and the real Job/MTP code samples accepted IDs.
"""
import argparse
import ast
import asyncio
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--engine',type=Path,required=True)
    ap.add_argument('--tabby',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    with args.output.open('x') as stream:json.dump({'state':'running','scope':__doc__},stream)
    sys.path.insert(0,str(args.tabby.resolve()))
    # The Tabby CPU venv uses the same Python ABI as this already-installed CPU Torch.
    sys.path.append('/usr/local/lib/python3.10/dist-packages')
    from tests import test_reasoning_budget as tabby
    from backends.exllamav3.reasoning import prepare_native_reasoning_budget
    import torch
    spec=importlib.util.spec_from_file_location('natural_backend_native_fixture',args.engine/'tests/test_token_budget_cpu.py')
    native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
    filter_path=args.engine/'exllamav3/generator/filter/filter.py'
    filter_ns={'torch':torch}
    native._compile([n for n in ast.parse(filter_path.read_text()).body if isinstance(n,ast.Assign)],filter_path,filter_ns)
    BaseFilter=native._load_class(filter_path,'Filter',filter_ns)
    report={'scope':__doc__,'state':'running','cases':[],'python':sys.version,'platform':platform.platform(),
            'torch':torch.__version__,'harness_sha256':sha(__file__),'sources':{}}
    for label,root,files in (
        ('engine',args.engine,['exllamav3/generator/job.py','exllamav3/generator/generator.py',
                              'exllamav3/generator/filter/filter.py','tests/test_token_budget_cpu.py']),
        ('tabby',args.tabby,['backends/exllamav3/model.py','backends/exllamav3/reasoning.py',
                            'endpoints/OAI/utils/stream_parser.py','tests/test_reasoning_budget.py'])):
        report['sources'][label]={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=root)).hexdigest(),
            'files_sha256':{name:sha(root/name) for name in files}}

    async def run_case(wrapped,mtp,packed):
        parts={0:'',1:'thought ',2:'<tool_' if wrapped else '<func',
            3:'call><function=echo><parameter=text>\n' if wrapped else 'tion=echo><parameter=text>\n',
            9:'</think>',5:'literal',6:'\n</parameter>',
            7:'</function></tool_call>' if wrapped else '</function>',8:'answer'}
        proposals=[1,2,3,9,5,6,7,9,5]
        expected=[1,2,3,9,5,6,7,9,8]
        tokenizer=native.Tokenizer()
        for token,piece in parts.items():tokenizer.pieces[token]=piece
        tokenizer.bos_token_id=None;tokenizer.bos_token='';tokenizer.eos_token_id=31
        tokenizer.single_id=lambda text: 9 if text=='</think>' else None
        tokenizer.encode=lambda text,**kwargs: torch.tensor([[0,0]]) if text=='synthetic' else (_ for _ in ()).throw(AssertionError('Natural mode encoded forced output'))
        filters=[];jobs=[];phase_positions=[];states=[]
        class ContentFilter(BaseFilter):
            def __init__(self,trigger):
                super().__init__(tokenizer,trigger,None,True)
                self.accepted=[]
            def reset(self):self.accepted=[]
            def accept_token(self,token):self.accepted.append(token)
            def is_completed(self):return self.accepted==[8]
            def use_background_worker(self):return False
            def get_next_logit_mask(self):
                if packed:return torch.tensor([[1 << 8]],dtype=torch.int32)
                mask=torch.full((1,native.VOCAB),-torch.inf,dtype=torch.half);mask[...,8]=0;return mask
        class Grammar:
            def __init__(self):self.filters=[]
            def add_grammar_filter(self,*args,**kwargs):
                f=ContentFilter(kwargs.get('trigger_token_id'));self.filters.append(f);filters.append(f)
        class NativeAsync:
            supports_natural_token_budget=True
            def __init__(self,generator,**kwargs):
                self.creation=kwargs
                self.job=native.make_job(**kwargs)
                self.job.generator.tokenizer=tokenizer
                for f in self.job.filters:f.attach(self.job)
                self.cancelled=False;self.queue=asyncio.Queue();self.results=[];self.plan=None
                jobs.append(self)
            def set_token_budget(self,max_tokens,output=None,**kwargs):
                self.plan=(max_tokens,output)
                self.job.set_token_budget(max_tokens,output,**kwargs)
                assert self.job.token_budget['deadline'] is None and self.job.token_budget['output'] is None
            def set_filters(self,filters):self.job.set_filters(filters)
            def set_sampler(self,sampler):self.job.set_sampler(sampler)
            def set_banned_strings(self,strings):self.job.set_banned_strings(strings)
            async def cancel(self):self.cancelled=True;self.job.clear_token_budget()
            async def __aiter__(self):
                if mtp:
                    scores=torch.zeros((1,len(proposals),native.VOCAB))
                    for position,token in enumerate(proposals):scores[0,position,token]=5
                    self.results=native.run_mtp(self.job,proposals[:-1],scores=scores)['results']
                else:
                    for token in proposals:
                        native._prepare_masks(self.job)
                        native.sample(self.job,token,self.results)
                        if self.job.new_tokens==4:
                            states.append({'in_tool':plan.parser.in_tool,'in_reasoning':plan.parser.in_reasoning,
                                'job_filters':len(self.job.filters),'filter_journal':list(filters[0]._journal)})
                for row in self.results:yield row
        mc,params,disconnect,events,unused,other=tabby.BackendArmingTests().backend()
        ns=mc.generate_gen.__func__.__globals__
        sampler=native.Sampler()
        ns.update(AsyncJob=NativeAsync,ExLlamaV3Grammar=Grammar,torch=torch,time=time,
            status_display=SimpleNamespace(add_job=lambda *args:SimpleNamespace(generated=lambda *args:None),remove_job=lambda *args:None),
            ExllamaV3SamplerBuilder=SimpleNamespace(from_params=lambda *args:SimpleNamespace(
                build=lambda *args:sampler,settings=[])))
        mc.tokenizer=tokenizer
        async def recover(*args):pass
        mc._recover_from_generation_error=recover
        mc.hf_model=SimpleNamespace(add_bos_token=lambda:False,eos_tokens=lambda:[31])
        params.max_tokens=20
        real_phase=mc.set_generation_phase
        def phase(request_id,reasoning):
            phase_positions.append(jobs[0].job.new_tokens)
            return real_phase(request_id,reasoning)
        mc.set_generation_phase=phase
        plan=prepare_native_reasoning_budget(tokenizer,None,None,initial_reasoning=True,end_token='</think>',supported=True,parser=tabby.qwen_parser())
        _=[row async for row in mc.stream_generate('natural','synthetic',params,disconnect,
                                                  reasoning_phase=True,reasoning_budget=plan)]
        job=jobs[0]
        got=native.emitted_ids(job.results)
        assert got==expected,(got,expected)
        assert job.creation['filters']==[] and job.creation['max_new_tokens']==20
        assert job.plan==(None,None)
        assert phase_positions==[8],phase_positions
        assert filters[0].trigger_token is None
        assert filters[0]._journal==[(filter_ns['FJ_COMPLETE'],8)],filters[0]._journal
        assert job.job.token_budget is None and job.job.forced_ids is None
        assert mc.active_job_ids=={'other':other} and mc.job_phases=={}
        if not mtp:assert states==[{'in_tool':True,'in_reasoning':True,'job_filters':0,'filter_journal':[]}],states
        return {'accepted_ids':got,'phase_callback_positions':phase_positions,'filter_journal':filters[0]._journal,
            'literal_end_position':4,'real_end_position':8,'max_new_tokens':job.creation['max_new_tokens'],
            'forced_output':False,'protected_literal_state':states,'wrapped':wrapped,'mtp':mtp,'packed':packed}
    for wrapped in (False,True):
        for mtp in (False,True):
            for packed in (False,True):
                name=f'actual_backend_wrapped{wrapped}_mtp{mtp}_packed{packed}'
                started=time.monotonic()
                try:row={'name':name,'passed':True,'detail':asyncio.run(run_case(wrapped,mtp,packed))}
                except Exception as exc:row={'name':name,'passed':False,'error':repr(exc)}
                row['seconds']=time.monotonic()-started;report['cases'].append(row)
    report['state']='completed';report['summary']={'passed':sum(row['passed'] for row in report['cases']),
        'failed':sum(not row['passed'] for row in report['cases']),'total':len(report['cases'])}
    temporary=args.output.with_name(args.output.name+'.tmp')
    with temporary.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    os.replace(temporary,args.output)
    print(json.dumps({'output':str(args.output),'summary':report['summary']}))
    for row in report['cases']:
        if not row['passed']:print(json.dumps(row))
    return 1 if report['summary']['failed'] else 0

if __name__=='__main__':raise SystemExit(main())
