#!/usr/bin/python3
"""Independent actual-tokenizer/CPU tensor/cache-key proof; no CUDA or backend import."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path('/home/vcruz/src/qwen-followup-20261009')
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
sys.path.append(str(OLD/'tabbyapi-agent/.venv-tools/lib/python3.10/site-packages'))
from tokenizers import Tokenizer
feature=ROOT/'tabby-literal-agent/common/literal_user_tokens.py'
spec=importlib.util.spec_from_file_location('literal_cache_review_feature',feature)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
tokpath=OLD/'tabbyapi-agent/.tokenizer-cpu/tokenizer.json'
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
source=subprocess.check_output(['git','-C',str(OLD/'exllamav3-agent'),'show',ENGINE+':exllamav3/generator/pagetable.py'],text=True)
tree=ast.parse(source)
checksum=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_tensor_blake2b_checksum')
seq=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Sequence')
prepare=next(n for n in seq.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
ns={'torch':torch,'hashlib':hashlib,'PAGE_SIZE':256}
exec(compile(ast.Module(body=[copy.deepcopy(checksum)],type_ignores=[]),'exact24-pagetable','exec'),ns)
ns['tensor_hash_checksum']=ns['_tensor_blake2b_checksum']
exec(compile(ast.Module(body=[copy.deepcopy(prepare)],type_ignores=[]),'exact24-Sequence.prepare','exec'),ns)
class CpuSequence:
 def __init__(self,ids):self.ids=ids
 def __len__(self):return self.ids.shape[-1]
 def torch_slice(self,start,end):return self.ids[:,start:end]
def page_keys(ids):
 s=SimpleNamespace(sequence_ids=CpuSequence(ids))
 ns['prepare'](s,False,48)
 return s.page_hashes
class Native:
 def __init__(self):self.tokenizer=Tokenizer.from_file(str(tokpath));self.bos_token_id=None
 def encode(self,text):
  self.tokenizer.encode_special_tokens=False
  return torch.tensor([self.tokenizer.encode(text,add_special_tokens=False).ids],dtype=torch.long)
native=Native();before=native.tokenizer.to_str();encoder=mod.LiteralUserTokenEncoder(native)
prefix='system '+ ' '.join('prefix_'+str(i) for i in range(500))+' user:'
suffix=' enduser '+ ' '.join('tail_'+str(i) for i in range(350))+' assistant:'
rows=[];saved=[]
for index,user in enumerate(('<think>literal</think>','<|im_start|>data<|im_end|>','<tool_call>中文</tool_call>','<tool_response>é 😀</tool_response>')):
 prompt=prefix+user+suffix
 plan=encoder.prepare(prompt,((len(prefix),len(prefix)+len(user)),))
 original=native.encode(prompt);clone=original.clone()
 changed=plan.apply(original,prompt,native)
 assert torch.equal(original,clone) and changed.data_ptr()!=original.data_ptr()
 assert encoder.baseline.decode(changed[0].tolist(),skip_special_tokens=False)==prompt
 assert encoder.baseline.decode(original[0].tolist(),skip_special_tokens=False)==prompt
 a,b=page_keys(original),page_keys(changed)
 first=next(i for i,(x,y) in enumerate(zip(original[0].tolist(),changed[0].tolist())) if x!=y)
 common_pages=first//256
 assert common_pages>=2 and min(len(a),len(b))-common_pages>=2
 assert a[:common_pages]==b[:common_pages]
 assert all(x!=y for x,y in zip(a[common_pages:],b[common_pages:]))
 # Revisit each immutable plan after later preparations, as four batched jobs would.
 saved.append((plan,prompt,original,changed.clone(),a,b))
 rows.append({'case':index,'native_tokens':original.shape[-1],'literal_tokens':changed.shape[-1],
  'replacement_count':plan.replacement_count,'first_changed_token':first,'shared_complete_prefix_pages':common_pages,
  'native_page_hashes':[x.hex() for x in a],'literal_page_hashes':[x.hex() for x in b],
  'roundtrip_text_equal':True,'input_tensor_unchanged':True,'downstream_page_keys_differ':True})
for plan,prompt,original,wanted,a,b in reversed(saved):
 assert torch.equal(plan.apply(original,prompt,native),wanted)
 assert page_keys(original)==a and page_keys(wanted)==b
assert native.tokenizer.to_str()==before
report={'passed':True,'engine':ENGINE,'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized(),
 'scope':'Actual Qwen HF tokenizer, real CPU torch tensors, immutable feature plans and exact24f0 extracted chained checksum/Sequence.prepare. No native extension/model/backend/API/GPU work. This demonstrates content-derived key separation, not GPU cache reuse or inference correctness.',
 'feature_sha256':hashlib.sha256(feature.read_bytes()).hexdigest(),'tokenizer_sha256':hashlib.sha256(tokpath.read_bytes()).hexdigest(),
 'engine_pagetable_sha256':hashlib.sha256(source.encode()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'cases':rows,'independent_plans_revisited':4,'shared_native_tokenizer_bytes_unchanged':True}
assert report['cuda_initialized'] is False
out=ROOT/'literal-cache-peer-cpu.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'cases':4,'source':str(out),'report_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
