#!/usr/bin/env python3
"""CPU reproduction of default reasoning's raw END trigger inside literal tool data.

Executes the actual collector and backend methods, then actual native Job and
Filter.feed code through bounded CPU fixtures. No model, server, GPU or HTTP.
"""
import ast
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
TABBY=ROOT/'tabbyapi-agent'
ENGINE=ROOT/'exllamav3-budget-agent'
sys.path.insert(0,str(TABBY))
sys.path.append('/usr/local/lib/python3.10/dist-packages')
from tests.test_reasoning_budget import BackendArmingTests, collect, recording_backend, make_request, qwen_parser
from unittest.mock import Mock
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('repro_native_budget_fixture',ENGINE/'tests/test_token_budget_cpu.py')
native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
filter_path=ENGINE/'exllamav3/generator/filter/filter.py'
filter_ns={'torch':torch}
filter_nodes=[n for n in ast.parse(filter_path.read_text()).body if isinstance(n,ast.Assign)]
native._compile(filter_nodes,filter_path,filter_ns)
NativeFilter=native._load_class(filter_path,'Filter',filter_ns)

async def collector_and_backend():
    raw='<tool_call><function=echo><parameter=text>\n</think>\n</parameter></function></tool_call></think>answer'
    echo={'type':'function','function':{'name':'echo','parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}}}
    collector=[]
    for stream in (False,True):
        mc=recording_backend([raw])
        mc.prepare_reasoning_budget=Mock(side_effect=AssertionError('predecessor must not negotiate with budget omitted'))
        params=make_request(tools=[echo])
        assert params.reasoning_budget_tokens is None
        frames,result=await collect(mc,params,stream=stream)
        mc.prepare_reasoning_budget.assert_not_called()
        options=mc.options[0]
        assert options['reasoning_phase'] is True and 'reasoning_budget' not in options
        collector.append({'stream':stream,'reasoning_budget_tokens':params.reasoning_budget_tokens,'native_plan_forwarded':False})
    mc,params,disconnect,events,jobs,other=BackendArmingTests().backend()
    _=[row async for row in mc.stream_generate('no-budget','synthetic',params,disconnect,reasoning_phase=True)]
    assert jobs[0].budget is None
    filters=jobs[0].creation['filters']
    assert len(filters)==1 and filters[0].trigger_token==native.END
    return collector,filters[0].trigger_token

collector,trigger=asyncio.run(collector_and_backend())
class ContentFilter(NativeFilter):
    def __init__(self):
        super().__init__(None,trigger,None,False)
        self.accepted=[]
    def reset(self):self.accepted=[]
    def accept_token(self,token):self.accepted.append(token)
    def is_completed(self):return False
    def use_background_worker(self):return False
    def get_next_logit_mask(self):
        mask=torch.full((1,native.VOCAB),-torch.inf,dtype=torch.half)
        mask[...,8]=0
        return mask

f=ContentFilter()
job=native.make_job(input_ids=torch.tensor([[0,0]]),filters=[f])
f.attach(job)
parts={1:'<tool_call><function=echo><parameter=text>\n',native.END:'</think>',5:'literal',8:'CONTENT_ONLY'}
for token,piece in parts.items():job.generator.tokenizer.pieces[token]=piece
parser=qwen_parser()
rows=[]
for token in (1,native.END):
    native._prepare_masks(job)
    native.sample(job,token,rows)
    parser.feed(parts[token])
assert job.token_budget is None
assert parser.in_tool and parser.in_reasoning
assert f.is_active and f._journal[-1]==(filter_ns['FJ_TRIGGER'],native.END)
state={'parser_in_tool':parser.in_tool,'parser_in_reasoning':parser.in_reasoning,
       'filter_active':f.is_active,'journal':[list(x) for x in f._journal],
       'producer_budget':job.token_budget}
native._prepare_masks(job)
native.sample(job,5,rows)
accepted=native.emitted_ids(rows)
assert accepted[-1]==8,accepted
report={'scope':__doc__,'confirmed':True,'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
        'collector_cases':collector,'backend_trigger_token_id':trigger,'protected_literal_state':state,
        'next_proposed_token':5,'next_accepted_token':accepted[-1],'accepted_ids':accepted,'source':{}}
for label,path,files in (
    ('tabby',TABBY,['backends/exllamav3/model.py','backends/exllamav3/reasoning.py','endpoints/OAI/utils/chat_completion.py','tests/test_reasoning_budget.py']),
    ('engine',ENGINE,['exllamav3/generator/job.py','exllamav3/generator/filter/filter.py','tests/test_token_budget_cpu.py'])):
    report['source'][label]={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip(),
        'diff_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=path)).hexdigest(),
        'files_sha256':{name:sha(path/name) for name in files}}
report['harness_sha256']=sha(__file__)
output=ROOT/'tabbyapi-diagnostics/no-budget-trigger-repro-3adc-9c.json'
with output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report,indent=2))
