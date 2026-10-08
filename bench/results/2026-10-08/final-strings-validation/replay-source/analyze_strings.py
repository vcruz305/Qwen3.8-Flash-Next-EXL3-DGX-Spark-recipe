"""Replay only the saved two-request strings observer capture on the final CPU parser.

No API requests, tokenizer decoding, native generation, template changes or source
mutations. Raw strings are retained verbatim. The actual collector is replayed
with a deterministic text source; phase injection is neither simulated nor needed
for these already-generated traces. Fresh call IDs are deliberately excluded
from comparisons, because their random identity cannot be recreated.
"""
import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

TABBY_HEAD = '5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb'
ENGINE_HEAD = '24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
BASE = Path('/home/vcruz/src/qwen-overnight-20261008')
ROOT = BASE / 'tabbyapi-natural-agent'
sys.path.insert(0, str(ROOT))
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils import chat_completion as cc


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def dump(value):
    return value.model_dump(mode='json') if hasattr(value, 'model_dump') else value


def normalized(message, finish):
    calls=[]
    for item in message.get('tool_calls') or []:
        item=dump(item);function=dump(item['function'])
        calls.append({'name':function['name'], 'arguments':json.loads(function['arguments'])})
    return {'content':message.get('content') or '',
            'reasoning_content':message.get('reasoning_content') or '',
            'tool_calls':calls, 'finish_reason':finish}


async def replay(raw, request, initial_reasoning, finish, width, streaming):
    async def generated(*args,**kwargs):
        for offset in range(0,len(raw),width):
            yield {'text':raw[offset:offset+width]}
        yield {'text':'',**finish}
    backend=SimpleNamespace(harmony=False,muse_glimmer=False,tool_format='qwen3_5',
        reasoning=True,reasoning_start_token='<think>',reasoning_end_token='</think>',
        reasoning_budget_tokens=None,reasoning_budget_message=None,tool_calls_in_reasoning=True,
        stream_generate=generated,set_generation_phase=lambda *a:True,
        constrain_generation_output=lambda *a:False)
    queue=asyncio.Queue() if streaming else None
    params=ChatCompletionRequest.model_validate(deepcopy(request))
    with patch.object(cc,'model',SimpleNamespace(container=backend)):
        result=await cc._chat_stream_collector(0,queue,'cpu-replay','saved prompt',params,
                                             initial_reasoning,streaming_mode=streaming)
    if not streaming:
        if isinstance(result,BaseException):return {'error':type(result).__name__+': '+str(result)}
        return normalized(result,result.get('finish_reason'))
    content=[];reasoning=[];calls={};finished=None;errors=[]
    while not queue.empty():
        frame=queue.get_nowait()
        if isinstance(frame,BaseException):
            errors.append(type(frame).__name__+': '+str(frame));continue
        content.append(frame.get('delta_content') or '')
        reasoning.append(frame.get('delta_reasoning_content') or '')
        finished=frame.get('finish_reason') or finished
        for item in frame.get('delta_tool_calls') or []:
            item=dump(item);index=item.get('index')
            if index is None:index=len(calls)
            target=calls.setdefault(index,{'function':{'name':'','arguments':''}})
            function=dump(item['function'])
            target['function']['name']+=function.get('name') or ''
            target['function']['arguments']+=function.get('arguments') or ''
    if errors:return {'error':errors[-1],'all_errors':errors}
    return normalized({'content':''.join(content),'reasoning_content':''.join(reasoning),
                       'tool_calls':[calls[i] for i in sorted(calls)]},finished)


async def partitions(raw,request,initial_reasoning,finish):
    rows=[]
    for width in (1,7,31,max(1,len(raw))):
        for streaming in (False,True):
            rows.append({'chunk_characters':width,'collector_streaming':streaming,
                         'result':await replay(raw,request,initial_reasoning,finish,width,streaming)})
    return rows


def source_identity():
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    if head!=TABBY_HEAD or changes:raise ValueError('Exact clean final Tabby checkout required')
    paths=['endpoints/OAI/utils/'+name for name in ('chat_completion.py','stream_parser.py',
           'toolcall_stream.py','toolcall_formats/qwen3_coder.py')]
    return {'head':head,'clean':True,'source_sha256':{p:sha((ROOT/p).read_bytes()) for p in paths},
            'script_sha256':sha(Path(__file__).read_bytes()),'python':sys.version}


def classify(observed,expected_calls,native,backend,native_equals_backend):
    backend_matches=all(row['result']==observed for row in backend)
    native_matches=all(row['result']==observed for row in native)
    consistent=all(row['result']==backend[0]['result'] for row in backend)
    actual=observed.get('tool_calls')
    if not backend_matches:
        category='unresolved_collector_or_transport_discrepancy'
    elif not native_equals_backend or not native_matches:
        category='unresolved_native_backend_difference'
    elif actual==expected_calls:
        category='exact_requested_arguments_preserved'
    elif not actual:
        category='no_call_reproduced_by_parser_requires_raw_review'
    else:
        category='argument_mismatch_reproduced_by_parser_requires_raw_review'
    return {'category':category,'backend_replay_matches_api':backend_matches,
            'native_replay_matches_api':native_matches,'chunk_partition_consistent':consistent,
            'attribution_note':'Parser replay alone cannot exclude a deterministic parser defect. Attribution to model generation requires independent inspection of the retained raw parameter text.',
            'expected_calls':expected_calls,'observed_calls':actual}


async def analyze(directory):
    capture=read(directory/'strings-capture.json')
    if capture.get('capture_valid') is not True:
        raise ValueError('An independently verified complete strings capture is required')
    manifest=read(directory/'raw-strings/observer-manifest.json')
    for key,expected in [('engine',ENGINE_HEAD),('tabby',TABBY_HEAD)]:
        if manifest['source'][key]['commit']!=expected:raise ValueError('Wrong final '+key+' source')
    paths={'strings-capture.json':None,'tools.json':capture['client_report_sha256'],
           'result.json':capture['lifecycle_sha256'],
           'raw-strings/observer-manifest.json':capture['observer_manifest_sha256']}
    for path,expected in paths.items():
        if expected is not None and sha((directory/path).read_bytes())!=expected:
            raise ValueError('Saved capture file hash mismatch: '+path)
    report=read(directory/'tools.json');cases=[]
    if len(capture['traces'])!=2 or {r['mode'] for r in capture['traces']}!={'stream','nonstream'}:
        raise ValueError('Exactly two recorded response modes required')
    for item in capture['traces']:
        path=directory/'raw-strings'/Path(item['path']).name
        if sha(path.read_bytes())!=item['sha256']:raise ValueError('Trace hash mismatch')
        trace=read(path)
        row=next(r for r in report['results'] if r['mode']==item['mode'] and r['case']=='strings')
        request=row['request'];matched=trace['matched_request']
        for field in ('messages','tools','model','tool_choice'):
            if request[field]!=matched[field]:raise ValueError('Request/trace field mismatch: '+field)
        if trace['streaming_mode']!=request['stream'] or trace['request_id']!=item['request_id']:
            raise ValueError('Recorded response mode/identity differs')
        if sha(trace['rendered_prompt'].encode())!=trace['rendered_prompt_sha256']:
            raise ValueError('Rendered prompt hash differs')
        native=trace['raw_finish']['native_full_completion']
        backend=trace['raw_finish']['backend_full_response']
        if sha(native.encode())!=item['native_sha256'] or sha(backend.encode())!=item['backend_sha256']:
            raise ValueError('Raw generator string hash differs')
        finish=dict(trace['raw_finish']['returned_metrics'])
        if finish.get('finish_reason') not in ('stop','length'):raise ValueError('Unrecognized native finish')
        response=row['response'];observed=normalized(response['message'],response['finish_reason'])
        prompt=request['messages'][0]['content']
        prefix='Call record_strings once with these exact string values: '
        if not prompt.startswith(prefix):raise ValueError('Unexpected synthetic request')
        expected=[{'name':'record_strings','arguments':json.loads(prompt[len(prefix):])}]
        replay_native=await partitions(native,request,trace['start_in_reasoning_mode'],finish)
        replay_backend=await partitions(backend,request,trace['start_in_reasoning_mode'],finish)
        classification=classify(observed,expected,replay_native,replay_backend,native==backend)
        cases.append({'mode':row['mode'],'request_id':trace['request_id'],
            'original_case_passed':row['passed'],'original_case_errors':row.get('errors'),
            'rendered_prompt':trace['rendered_prompt'],'rendered_prompt_sha256':trace['rendered_prompt_sha256'],
            'start_in_reasoning_mode':trace['start_in_reasoning_mode'],
            'raw_native_full_completion':native,'raw_backend_full_response':backend,
            'native_equals_backend':native==backend,'observed_api':observed,
            'native_replays':replay_native,'backend_replays':replay_backend,**classification})
    return {'scope':__doc__,'source':source_identity(),'observed_engine_head':ENGINE_HEAD,
            'input_directory':str(directory),'files_sha256':{p:sha((directory/p).read_bytes()) for p in paths},
            'trace_sha256':{Path(i['path']).name:i['sha256'] for i in capture['traces']},
            'capture_semantic_passed':capture['semantic_passed'],'cases':cases,
            'replay_count':sum(len(r['native_replays'])+len(r['backend_replays']) for r in cases),
            'all_backend_replays_match_api':all(r['backend_replay_matches_api'] for r in cases),
            'normalization_note':'Only absent/null/empty textual API fields are normalized to empty text; tool argument JSON is decoded without string normalization. Fresh random tool-call IDs are omitted. This tests complete routed output and assembled arguments, not original SSE timing or IDs.'}


async def self_test():
    fixture=read(BASE/'tabbyapi-diagnostics/pack-tool-errors-16ca-f4/405-tools.json')
    request=next(r['request'] for r in fixture['results'] if r['case']=='strings')
    expected=json.loads(request['messages'][0]['content'].split(': ',1)[1])
    def xml(values):
        return '<tool_call><function=record_strings>'+''.join(
            '<parameter='+name+'>\n'+value+'\n</parameter>' for name,value in values.items())+'</function></tool_call>'
    rows=[]
    for kind,raw,initial in [('exact',xml(expected),False),
          ('reasoning_then_exact','prior reasoning</think>'+xml(expected),True),
          ('wrong_argument',xml({**expected,'tag_text':'wrong copied value'}),False),
          ('no_call','plain answer',False),('malformed','<tool_call><function=record_strings>',False)]:
        replayed=await partitions(raw,request,initial,{'finish_reason':'stop','eos_reason':'eot'})
        first=replayed[0]['result']
        if kind=='malformed':assert all('error' in r['result'] for r in replayed)
        else:
            assert all(r['result']==first for r in replayed)
            if kind in ('exact','reasoning_then_exact'):assert first['tool_calls']==[{'name':'record_strings','arguments':expected}]
            if kind=='wrong_argument':assert first['tool_calls'][0]['arguments']['tag_text']=='wrong copied value'
            if kind=='no_call':assert first['tool_calls']==[]
        rows.append({'fixture':kind,'replays':len(replayed),'passed':True})
    observed=normalized({'tool_calls':[{'function':{'name':'record_strings','arguments':json.dumps(expected)}}]},'tool_calls')
    good=[{'result':observed}];bad=[{'result':{**observed,'content':'changed'}}]
    assert classify(observed,observed['tool_calls'],good,good,True)['category']=='exact_requested_arguments_preserved'
    assert classify(observed,[],good,good,True)['category']=='argument_mismatch_reproduced_by_parser_requires_raw_review'
    assert classify(observed,[],good,bad,True)['category']=='unresolved_collector_or_transport_discrepancy'
    assert classify(observed,[],good,good,False)['category']=='unresolved_native_backend_difference'
    rows.append({'fixture':'classification_boundaries','checks':4,'passed':True})
    return {'scope':'CPU synthetic harness check only; no live observation or inference',
            'source':source_identity(),'cases':rows,'parser_replays':40,'passed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if bool(args.capture)==args.self_test:parser.error('Choose capture or self-test')
    if args.output.exists() or args.output.is_symlink():raise FileExistsError(args.output)
    source_identity()
    value=asyncio.run(self_test() if args.self_test else analyze(args.capture.resolve(strict=True)))
    with args.output.open('x') as file:json.dump(value,file,indent=2,ensure_ascii=False);file.write('\n')
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output.read_bytes()),
                      'replays':value.get('replay_count',value.get('parser_replays')),
                      'categories':[c.get('category',c.get('fixture')) for c in value['cases']]}))
if __name__=='__main__':main()
