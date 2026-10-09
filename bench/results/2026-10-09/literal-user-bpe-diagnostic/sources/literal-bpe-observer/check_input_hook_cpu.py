#!/usr/bin/env python3
"""Actual native encode and Tabby context bookkeeping, CPU only."""
import ast,copy,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tokenizers import Tokenizer as HFTokenizer,models
ROOT=Path(__file__).resolve().parent
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
ENGINE=OLD/'exllamav3-agent';TABBY=ROOT.parent/'tabby-literal-agent';ASSETS=OLD/'tabbyapi-agent/.tokenizer-cpu'
sys.path.insert(0,str(ROOT/'observer'))
import user_bpe_diagnostic as hook
import strings_observer as observer
PROMPT=(OLD/'literal-final-triage-5a/rendered-prompt.txt').read_text()

def sha(data):return hashlib.sha256(data).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def source_class(source,name,methods,namespace,init_prefix=False):
 cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)
 selected=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
 assert {n.name for n in selected}==methods
 if init_prefix:
  init=next(n for n in selected if n.name=='__init__')
  stop=next(i for i,n in enumerate(init.body) if isinstance(n,ast.If) and 'models.Unigram' in ast.unparse(n.test));init.body=init.body[:stop]
 module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='Subject',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])
 exec(compile(ast.fix_missing_locations(module),'<actual-source-'+name+'>','exec'),namespace)
 return namespace['Subject'],sha(ast.dump(module,include_attributes=False).encode())

def read_json(path):
 p=Path(path);return json.loads(p.read_text()) if p.exists() else {}
engine_source=subprocess.check_output(['git','-C',str(ENGINE),'show',observer.ENGINE_HEAD+':exllamav3/tokenizer/tokenizer.py'],text=True)
Native,native_ast=source_class(engine_source,'Tokenizer',{'__init__','encode_part_base','encode_part','encode_special_or_unspecial','encode'},
 {'torch':torch,'os':os,'re':re,'HFTokenizer':HFTokenizer,'models':models,'maybe_read_json':read_json},True)
native=Native(NS(directory=str(ASSETS)));native.bos_token_id=248044;native.eos_token_id=248044;native.pad_token_id=0
assert native.unspecial_piece_to_id=={} and native.missing_special_piece_to_id=={}
baseline=native.encode(PROMPT,encode_special_tokens=True)
assert baseline.shape==(1,348) and hook.digest_ids(baseline[0].tolist())==hook.BASELINE_IDS_SHA
recorder=NS(active={})
config={'diagnostic_user_bpe':True,'input_tokenizer_json':str(ASSETS/'tokenizer.json')}
handle=hook.install_user_bpe(Native,recorder,config,observer.PROMPT_SHA256,observer.USER_MESSAGE)
context_calls=[]
model_source=subprocess.check_output(['git','-C',str(TABBY),'show',observer.TABBY_HEAD+':backends/exllamav3/model.py'],text=True)
Container,container_ast=source_class(model_source,'ExllamaV3Container',{'encode_tokens','validate_context_length'},
 {'unwrap':lambda value,default:default if value is None else value,'validate_context_requirements':lambda *a:context_calls.append(a)})
mc=Container();mc.tokenizer=native;mc.hf_model=NS(add_bos_token=lambda:False);mc.max_seq_len=262144;mc.cache=NS(max_num_tokens=262144)
mc.generator=NS(generator=NS(recurrent_cache=None));mc.job_max_rq_tokens=lambda n:None
params=NS(add_bos_token=False,max_tokens=256)
# Context validation occurs before collector matching. It must already see352.
mc.validate_context_length(PROMPT,params)
assert context_calls[-1][0]==352 and not recorder.active
rows=[]
for i,variant in enumerate(observer.EXPECTED_VARIANTS):
 record={'rendered_prompt_sha256':observer.PROMPT_SHA256}
 recorder.active={'fixture-'+str(i):record}
 # Exact native operation used by generate_gen before context_len/cache usage.
 input_ids=native.encode(PROMPT,add_bos=False,encode_special_tokens=True,embeddings=[])
 assert input_ids.size(dim=-1)==352 and input_ids.dtype==baseline.dtype
 assert native.tokenizer.decode(input_ids[0].tolist(),skip_special_tokens=False)==PROMPT
 assert hook.digest_ids(baseline[0].tolist())==hook.BASELINE_IDS_SHA
 proof=record['input_tokenization'];assert len(proof['replacements'])==2
 assert proof['original_token_ids']==baseline[0].tolist()
 assert proof['changed_token_ids']==input_ids[0].tolist()
 assert len(record['input_encoding_calls'])==1
 rows.append({'variant':variant,'context_check_tokens':context_calls[-1][0],'generation_input_tokens':input_ids.size(-1),
              'decoded_bytes_equal':True,'proof':proof,'encode_calls':record['input_encoding_calls']})
recorder.active={}
# Normal input, forced reasoning close, and template markers are untouched.
untouched=[]
for text in ['hello','</think>','<|im_start|>assistant\n<think>\n',PROMPT+' ']:
 expected=handle['original_encode'](native,text,encode_special_tokens=True)
 got=native.encode(text,encode_special_tokens=True)
 assert torch.equal(expected,got);untouched.append(text)
rejected=[]
for key,value in [('add_bos',True),('add_eos',True),('encode_special_tokens',False),('return_offsets',True),('embeddings',[object()])]:
 kwargs={'encode_special_tokens':True,key:value}
 try:native.encode(PROMPT,**kwargs)
 except ValueError:rejected.append(key)
 else:raise AssertionError(key)
try:native.encode([PROMPT],encode_special_tokens=True)
except ValueError:rejected.append('batch')
else:raise AssertionError('batch')
record={'rendered_prompt_sha256':observer.PROMPT_SHA256}
recorder.active={'one':record,'two':dict(record)}
try:native.encode(PROMPT,encode_special_tokens=True)
except ValueError:rejected.append('concurrent')
else:raise AssertionError('concurrent')
recorder.active={}
old_encode_part=native.encode_part
native.encode_part=lambda *a: [1]+baseline[0].tolist()[1:]
try:native.encode(PROMPT,encode_special_tokens=True)
except ValueError:rejected.append('native_id_drift')
else:raise AssertionError('native drift')
native.encode_part=old_encode_part
try:hook.install_user_bpe(Native,recorder,{**config,'diagnostic_user_bpe':False},observer.PROMPT_SHA256,observer.USER_MESSAGE)
except ValueError:rejected.append('explicit_flag')
else:raise AssertionError('missing flag')
report={'passed':True,'scope':'CPU execution of actual24 native encode methods/constructor prefix and actualf650 encode_tokens/context validation. Fake container capacity fields/validator capture; no CUDA/model/HTTP. Native and generation bookkeeping both receive352 IDs. This is an explicit changed-input diagnostic, not production code or a model improvement result.',
 'engine':observer.ENGINE_HEAD,'tabby':observer.TABBY_HEAD,'native_source_sha256':sha(engine_source.encode()),'native_ast_sha256':native_ast,
 'container_source_sha256':sha(model_source.encode()),'container_ast_sha256':container_ast,'hook_sha256':sha((ROOT/'observer/user_bpe_diagnostic.py').read_bytes()),
 'test_sha256':sha(Path(__file__).read_bytes()),'original_tensor_unchanged':True,'actual_context_check_tokens':352,
 'cases':rows,'unchanged_nonmatching_inputs':untouched,'negative_cases':rejected}
with (ROOT/'input-hook-cpu-report.json').open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'passed':True,'original_cases':len(rows),'unchanged_inputs':len(untouched),'negative_cases':len(rejected),'report_sha256':sha((ROOT/'input-hook-cpu-report.json').read_bytes())}))
