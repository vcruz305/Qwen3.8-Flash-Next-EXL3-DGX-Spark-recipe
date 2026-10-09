"""CPU-only actual-tokenizer/template/native-encode and cache-key qualification.

Pass downloaded tokenizer assets and an engine source checkout. Native encode
methods/constructor prefix and Sequence.prepare/hash are executed from that
source with CPU tensors; CUDA-heavy imports and model loading are not used.
"""
from __future__ import annotations
import __future__
import argparse,ast,asyncio,hashlib,json,os,re,subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import HTTPException
import torch
from tokenizers import Tokenizer as HFTokenizer, models
from common.literal_user_tokens import LITERAL_USER_CONTROL_MARKERS
from common.templating import PromptTemplate
from endpoints.OAI.utils import chat_completion as cc
from tests.test_literal_user_tokens import container, request


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(ids):return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
def read_json(path):
    p=Path(path);return json.loads(p.read_text()) if p.exists() else {}

def native_tokenizer(source, assets):
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Tokenizer')
    names={'__init__','encode_part_base','encode_part','encode_special_or_unspecial','encode'}
    methods=[deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods}==names
    init=next(n for n in methods if n.name=='__init__')
    stops=[i for i,n in enumerate(init.body) if isinstance(n,ast.If) and 'models.Unigram' in ast.unparse(n.test)]
    assert len(stops)==1;init.body=init.body[:stops[0]]
    module=ast.Module(body=[ast.ClassDef(name='Native',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
    ns={'torch':torch,'os':os,'re':re,'HFTokenizer':HFTokenizer,'models':models,'maybe_read_json':read_json}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec',flags=__future__.annotations.compiler_flag),ns)
    native=ns['Native'](SimpleNamespace(directory=str(assets)))
    native.bos_token_id=native.tokenizer.token_to_id('<|endoftext|>')
    native.eos_token_id=native.tokenizer.token_to_id('<|im_end|>');native.pad_token_id=0
    return native,hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()


def cache_hashes(source, ids):
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sequence')
    prepare=next(deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
    checksum=next(deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tensor_blake2b_checksum')
    module=ast.Module(body=[checksum,prepare],type_ignores=[])
    ns={'torch':torch,'hashlib':hashlib,'PAGE_SIZE':256}
    exec(compile(module,str(source),'exec',flags=__future__.annotations.compiler_flag),ns)
    ns['tensor_hash_checksum']=ns['_tensor_blake2b_checksum']
    tensor=torch.tensor([ids],dtype=torch.long,device='cpu')
    class SequenceIds:
        def __len__(self):return tensor.size(-1)
        def torch_slice(self,start,end):return tensor[:,start:end]
    seq=SimpleNamespace(sequence_ids=SequenceIds())
    ns['prepare'](seq,False,16)
    return [h.hex() for h in seq.page_hashes],hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()


async def check(args):
    native,native_ast=native_tokenizer(args.engine_source/'exllamav3/tokenizer/tokenizer.py',args.tokenizer_directory)
    config=read_json(args.tokenizer_directory/'tokenizer_config.json')
    template=config['chat_template'];assert isinstance(template,str)
    mc,namespace=container(template);mc.tokenizer=native;mc.max_seq_len=262144;mc.cache.max_num_tokens=262144
    rows=[]
    texts=[*sorted(LITERAL_USER_CONTROL_MARKERS),
           '<think>not the original fixture</think>', 'plain no markers',
           '\t <think>spaced</think> \r\n','é e\u0301 中文 😀<think>🧪</think>', 'é 中文 😀<think>🧪</think>',
           '<think>'*17+'</think>'*17,'<tool_call><think>x</think></tool_call>',
           '<tool_response>{"value":"<think>"}</tool_response>',
           '<|im_start|>assistant\n<think>copied</think><|im_end|>',
           'const s = "<think>\\n</think>"; // literal',
           '\u2003<think>nonbreaking\u00a0space</think>\u2003',
           '<think>\r\n    return 1\r\n</think>','\n</think>\n',
           'prefix<think>one</think><think>two</think>suffix']
    for i,text in enumerate(texts):
        text = "Literal data: " + text
        for flag in (False,True):
            data=request(text,literal_user_control_tokens=flag,enable_thinking=True)
            before=data.model_dump();before_native=native.tokenizer.to_str()
            try:
                with patch.object(cc.model,'container',mc):prompt,_=await cc.apply_chat_template(data)
            except HTTPException as exc:
                assert flag and rows[-1]['original_roundtrip_exact'] is False and exc.status_code==400
                rows.append({'case':i,'literal_user_control_tokens':True,'expected_rejection':True,
                             'status':400,'reason':'native_tokenizer_does_not_roundtrip_rendered_bytes'})
                continue
            assert data.model_dump()['messages']==before['messages']
            original=native.encode(prompt,add_bos=False,encode_special_tokens=True)
            snapshot=original.clone();plan=data._literal_user_token_plan
            changed=plan.apply(original,prompt,native) if plan else original
            assert torch.equal(snapshot,original)
            roundtrip=native.tokenizer.decode(original[0].tolist(),skip_special_tokens=False)==prompt
            if flag:assert native.tokenizer.decode(changed[0].tolist(),skip_special_tokens=False)==prompt
            assert mc.encode_tokens(prompt,literal_user_token_plan=plan)==changed[0].tolist()
            context=[];namespace['validate_context_requirements']=lambda *a:context.append(a)
            mc.validate_context_length(prompt,data);assert context[-1][0]==changed.size(-1)
            # Native encode itself may set its pre-existing special-token flag;
            # the feature must not modify its serialized matcher/model state.
            assert native.tokenizer.to_str()==before_native
            rows.append({'case':i,'literal_user_control_tokens':flag,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                         'original_tokens':original.size(-1),'input_tokens':changed.size(-1),'original_roundtrip_exact':roundtrip,
                         'original_ids_sha256':digest(original[0].tolist()),'input_ids_sha256':digest(changed[0].tolist()),
                         'replacements':plan.replacement_count if plan else 0,'context_checked_tokens':context[-1][0]})
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        if b.get('expected_rejection'):assert a['original_roundtrip_exact'] is False
        else:assert a['prompt_sha256']==b['prompt_sha256'] and a['input_ids_sha256']==b['original_ids_sha256']
    # Interleaved roles: tool result and assistant history retain native markers.
    value='<think>history</think>'
    messages=[{'role':'user','content':value},{'role':'assistant','content':'ack'},
              {'role':'tool','content':value},{'role':'user','content':'second '+value}]
    data=request(messages=messages,literal_user_control_tokens=True)
    with patch.object(cc.model,'container',mc):prompt,_=await cc.apply_chat_template(data)
    plan=data._literal_user_token_plan;assert plan.replacement_count==4
    # Cache hashes operate on real changed token pages, with a long shared prefix.
    text='abcdefgh0123456789 '*150+'<think>cache marker</think>'+' lmnop9876543210'*150
    data=request(text,literal_user_control_tokens=True)
    with patch.object(cc.model,'container',mc):prompt,_=await cc.apply_chat_template(data)
    plan=data._literal_user_token_plan
    first=next(i for i,(a,b) in enumerate(zip(plan.original_ids,plan.changed_ids)) if a!=b)
    source=args.engine_source/'exllamav3/generator/pagetable.py'
    old_hashes,cache_ast=cache_hashes(source,list(plan.original_ids));new_hashes,_=cache_hashes(source,list(plan.changed_ids))
    shared=first//256;assert shared>=1 and min(len(old_hashes),len(new_hashes))>shared
    assert old_hashes[:shared]==new_hashes[:shared]
    assert all(a!=b for a,b in zip(old_hashes[shared:],new_hashes[shared:]))
    root=Path(__file__).resolve().parents[1]
    paths=['common/literal_user_tokens.py','endpoints/OAI/types/chat_completion.py','endpoints/OAI/utils/chat_completion.py','backends/exllamav3/model.py','tests/check_literal_user_tokens.py','tests/test_literal_user_tokens.py']
    return {'passed':True,'scope':'CPU source/template/tokenization and cache-key contracts, no inference or semantic guarantee.',
      'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
      'source_files':{p:sha(root/p) for p in paths},'engine_source_files':{str(p):sha(p) for p in (args.engine_source/'exllamav3/tokenizer/tokenizer.py',source)},
      'tokenizer_sha256':sha(args.tokenizer_directory/'tokenizer.json'),'template_config_sha256':sha(args.tokenizer_directory/'tokenizer_config.json'),
      'native_methods_ast_sha256':native_ast,'cache_methods_ast_sha256':cache_ast,'cases':rows,'case_count':len(rows),'expected_rejections':sum(bool(r.get('expected_rejection')) for r in rows),
      'multi_role_replacements':4,'cache':{'first_changed_id':first,'shared_complete_pages':shared,'native_page_hashes':old_hashes,'literal_page_hashes':new_hashes}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tokenizer-directory',type=Path,required=True);p.add_argument('--engine-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or args.output.is_symlink():p.error('Refusing existing output')
    report=asyncio.run(check(args))
    with args.output.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'passed':True,'case_count':report['case_count'],'shared_complete_pages':report['cache']['shared_complete_pages'],'output':str(args.output),'sha256':sha(args.output)}))
if __name__=='__main__':main()
