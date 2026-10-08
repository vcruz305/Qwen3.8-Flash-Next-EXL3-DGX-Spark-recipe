"""Read saved synthetic API evidence and reconstruct exact-source Qwen prompts on CPU.
No model load, API requests, source edits, or coercion of observed responses.
"""
import ast
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path('/home/vcruz/src/qwen-overnight-20261008/tabbyapi-agent')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.templating import PromptTemplate
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils.qwen_tool_guidance import NULLABLE_XML_GUIDANCE, nullable_guidance_eligible, with_nullable_xml_guidance
from tests.test_qwen_nullable_guidance import container
from tokenizers import Tokenizer

REVISIONS = {'3a': '3a4d2d5732f0a4de23ffbf2c493d21e9534417aa', 'f4': 'f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6'}
MODULE_PATH = 'endpoints/OAI/utils/chat_completion.py'
REPORTS = {'3a': 'api-3a-compat-auto.json', 'f4': 'api-f4-gemm-auto.json'}
def sha(data): return hashlib.sha256(data).hexdigest()
def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args], text=True)
def func_hash(source, name):
    node=next(n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    return sha(ast.dump(node,include_attributes=False).encode())

def observed(entry):
    if not entry['stream']:
        message=entry['body']['choices'][0]['message']
        return {'reasoning':message.get('reasoning_content') or '', 'content':message.get('content') or '', 'finish':entry['body']['choices'][0]['finish_reason'], 'done':False, 'usage':entry['body']['usage']}
    reasoning=[];content=[];finish=None;usage=None;transitions=[]
    for index,frame in enumerate(entry['frames']):
        usage=frame.get('usage') or usage
        for choice in frame.get('choices',[]):
            delta=choice.get('delta',{})
            if 'reasoning_content' in delta: reasoning.append(delta['reasoning_content']);transitions.append([index,'reasoning',delta['reasoning_content']])
            if 'content' in delta: content.append(delta['content']);transitions.append([index,'content',delta['content']])
            finish=choice.get('finish_reason') or finish
    return {'reasoning':''.join(reasoning),'content':''.join(content),'finish':finish,'done':entry['done'],'usage':usage,'transitions':transitions}

async def main():
    assert git('rev-parse','HEAD').strip()==REVISIONS['f4']
    assert not git('status','--porcelain').strip()
    output=HERE/'prompt-and-sse-analysis.json'
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    config_path=ROOT/'.tokenizer-cpu/tokenizer_config.json'
    tokenizer_path=ROOT/'.tokenizer-cpu/tokenizer.json'
    config=json.loads(config_path.read_bytes())
    template=config['chat_template']
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    reports={label:json.loads((HERE/name).read_bytes()) for label,name in REPORTS.items()}
    sources={label:git('show',revision+':'+MODULE_PATH) for label,revision in REVISIONS.items()}
    assert sources['f4']==(ROOT/MODULE_PATH).read_text()
    formatter={}
    for label,source in sources.items():
        mod=ModuleType('endpoints.OAI.utils._diagnostic_'+label)
        mod.__package__='endpoints.OAI.utils';mod.__file__=str(ROOT/MODULE_PATH)
        exec(compile(source,'git:'+REVISIONS[label]+':'+MODULE_PATH,'exec'),mod.__dict__)
        formatter[label]=mod
    rows=[]
    for mode,index in [('nonstream',5),('stream',6)]:
        request=deepcopy(reports['f4']['requests'][index]['request'])
        assert request==reports['3a']['requests'][index]['request']
        reconstructed={}
        for label,mod in formatter.items():
            data=ChatCompletionRequest.model_validate(deepcopy(request))
            before=data.model_dump()
            mc=container(tool_format='qwen3_5',raw_template=template)
            mc.reasoning=True;mc.reasoning_start_token='<think>';mc.reasoning_end_token='</think>'
            mc.start_in_reasoning='auto';mc.harmony=False;mc.muse_glimmer=False
            mod.model=SimpleNamespace(container=mc)
            eligible=nullable_guidance_eligible(data,'qwen3_5')
            tools=deepcopy(data.model_dump()['tools'])
            assert with_nullable_xml_guidance(tools) is tools
            prompt,embeddings=await mod.apply_chat_template(data)
            token_ids=tokenizer.encode(prompt,add_special_tokens=False).ids
            assert len(token_ids)==reports[label]['requests'][index]['body']['usage']['prompt_tokens'] if not request['stream'] else len(token_ids)==326
            assert embeddings is None and NULLABLE_XML_GUIDANCE not in prompt
            assert data.model_dump()['tools']==before['tools'] and data.model_dump()['messages']==before['messages']
            reconstructed[label]={'prompt':prompt,'prompt_sha256':sha(prompt.encode()),'token_ids':token_ids,'token_count':len(token_ids),'start_in_reasoning':mod._resolve_start_in_reasoning(prompt,data),'guidance_request_eligible':eligible,'guidance_applied':False,'grammar_sha256':sha((data.grammar_string or '').encode())}
        assert reconstructed['3a']==reconstructed['f4']
        rows.append({'mode':mode,'request':request,'request_unchanged':True,'reconstructed':reconstructed,'observed':{label:observed(reports[label]['requests'][index]) for label in reports}})
    hashes={name:{label:func_hash(source,name) for label,source in sources.items()} for name in ('_chat_stream_collector','_reasoning_budget_injection','_resolve_reasoning_budget','_resolve_start_in_reasoning')}
    for values in hashes.values(): assert len(set(values.values()))==1
    parser_path='endpoints/OAI/utils/stream_parser.py'
    parser_hashes={label:sha(git('show',revision+':'+parser_path).encode()) for label,revision in REVISIONS.items()}
    assert len(set(parser_hashes.values()))==1
    result={'scope':'CPU reconstruction from exact requests, tokenizer/template and source revisions. Saved SSE has channel text, but omits the raw reason-closing tags; it cannot prove the backend token sequence or injection position. No inference.', 'source_commits':REVISIONS,'source_module_sha256':{label:sha(source.encode()) for label,source in sources.items()}, 'template_config_sha256':sha(config_path.read_bytes()), 'tokenizer_sha256':sha(tokenizer_path.read_bytes()),'report_sha256':{label:sha((HERE/name).read_bytes()) for label,name in REPORTS.items()},'unchanged_function_ast_sha256':hashes,'unchanged_stream_parser_sha256':parser_hashes,'cases':rows,'script_sha256':sha(Path(__file__).read_bytes())}
    with output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'output':str(output),'sha256':sha(output.read_bytes()),'identical_prompt_sha256':rows[0]['reconstructed']['f4']['prompt_sha256'],'token_count':326,'guidance_applied':False,'modes_checked':2,'routing_and_budget_functions_unchanged':True}))

if __name__=='__main__':asyncio.run(main())
