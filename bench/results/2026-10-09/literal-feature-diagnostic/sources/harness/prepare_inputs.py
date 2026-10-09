#!/usr/bin/env python3
"""Freeze actual candidate CPU input IDs for the separate fresh-server 16-request diagnostic.

No model/backend/native-extension import, API request, CUDA initialization or
production hook. No standard tool suite is part of this separate diagnostic.
"""
from __future__ import annotations
import argparse,asyncio,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
from collections import Counter

TABBY='a70ae1fa9e457e478c3d96bdc84012a3cb331796'
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
RECIPE='254b2b03027f25094845dc31f5f87739f3584d2e'
PLAN='fd6da1e76e9f7e3ab635ca6e37b4c653c7eaf23f8afcee88f461088cb507f91a'
TOKENIZER='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
TEMPLATE='b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27'
GROUPS={'continuation2':2,'cache6':6,'heldout6':4,'concurrent4':4}
SOURCE_FILES=('common/literal_user_tokens.py','endpoints/OAI/types/chat_completion.py',
 'endpoints/OAI/utils/chat_completion.py','endpoints/OAI/utils/tool_choice.py',
 'endpoints/OAI/utils/qwen_tool_guidance.py','backends/exllamav3/model.py',
 'tests/check_literal_user_tokens.py','tests/test_literal_user_tokens.py',
 'common/templating.py','common/sampling.py','common/errors.py')

def require(value,message):
 if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def git_identity(path,pin):
 commit=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
 dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True)
 require(commit==pin and not dirty,'Exact clean source required: '+str(path))
 return {'path':str(path),'commit':commit,'tracked_changes':dirty}
def validate_plan(plan):
 require(plan['run_order']==['continuation2','cache6','unicode2','multi_turn2','concurrent4'],'Diagnostic order changed')
 rows=plan['requests'];require(len(rows)==16 and len({r['name'] for r in rows})==16,'Expected16 unique requests')
 require(dict(Counter(r['group'] for r in rows))==GROUPS,'Request groups changed')
 require(plan['client_requests']==16 and plan['coverage_requests']==8 and plan['generation_observation_requests']==8 and plan['total_chat_requests']==16,'Diagnostic count contract changed')
 require(plan['cache_geometry']['prefix_repeats']==135 and plan['cache_geometry']['chunk_size']==2048,'Diagnostic cache geometry changed')
 for row in rows:require(digest(row['request'])==row['request_sha256'],'Payload hash mismatch: '+row['name'])
 return rows
def lcp(a,b):
 for index,(x,y) in enumerate(zip(a,b)):
  if x!=y:return index
 return min(len(a),len(b))
def cache_bounds(current,prior,*,reject_strict_prefix):
 pairs=[]
 for row in prior:
  count=lcp(current,row['input_ids'])
  if reject_strict_prefix:
   require(not (count==len(row['input_ids'])<len(current)),
           'Earlier distinct prompt is a strict prefix of cache probe: '+row['name'])
  pairs.append({'name':row['name'],'tokens':count,'same_input':row['input_ids']==current})
 maximum=max((r['tokens'] for r in pairs),default=0)
 return {'cache_max_tokens':maximum,'cache_prior_names':[r['name'] for r in pairs if r['tokens']==maximum],
         'prior_lcp':pairs,'prior_prompt_count':len(prior)}

async def prepare(args):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES empty for CPU preparation')
 identities={'tabby':git_identity(args.tabby_source,TABBY),'engine':git_identity(args.engine_source,ENGINE),'recipe':git_identity(args.recipe_source,RECIPE)}
 require(sha(args.plan)==PLAN,'Prepared plan changed');require(sha(args.tokenizer_directory/'tokenizer.json')==TOKENIZER,'Tokenizer asset changed');require(sha(args.tokenizer_directory/'tokenizer_config.json')==TEMPLATE,'Template asset changed')
 plan=json.loads(args.plan.read_text());rows=validate_plan(plan)
 source_files={p:sha(args.tabby_source/p) for p in SOURCE_FILES}
 engine_files={p:sha(args.engine_source/p) for p in ('exllamav3/tokenizer/tokenizer.py','exllamav3/generator/pagetable.py')}
 recipe_config=args.recipe_source/'exllamav3-tabby/tabby-config.yml';config_sha=sha(recipe_config)
 sys.path.insert(0,str(args.tabby_source))
 import torch
 from pydantic import ValidationError
 from fastapi import HTTPException
 from ruamel.yaml import YAML
 from unittest.mock import patch
 import common.model
 from common.sampling import overrides_from_dict
 from common.errors import validate_context_requirements
 from endpoints.OAI.types.chat_completion import ChatCompletionRequest
 from endpoints.OAI.utils import chat_completion as cc
 from tests.check_literal_user_tokens import native_tokenizer
 from tests.test_literal_user_tokens import container
 require(not torch.cuda.is_initialized(),'CUDA already initialized')
 require(not any(k=='exllamav3' or k.startswith('exllamav3.') for k in sys.modules),'Native engine was imported')
 config=YAML(typ='safe').load(recipe_config.read_text())
 sampling=config['sampling'];overrides_from_dict(sampling)
 tc=json.loads((args.tokenizer_directory/'tokenizer_config.json').read_text())
 require(tc.get('add_bos_token') is False and tc.get('bos_token') is None,'This selected-model binding expects no automatic BOS')
 native,native_ast=native_tokenizer(args.engine_source/'exllamav3/tokenizer/tokenizer.py',args.tokenizer_directory)
 mc,namespace=container(tc['chat_template']);mc.tokenizer=native
 mc.tool_format='qwen3_5';mc.use_vision=False
 mc.template_vars_default=config.get('model',{}).get('template_vars_default') or {}
 mc.template_vars_force=config.get('model',{}).get('template_vars_force') or {}
 require(not mc.template_vars_default and not mc.template_vars_force,'Unexpected forced/default template variables')
 mc.max_seq_len=262144;mc.cache.max_num_tokens=262144
 context_seen=[]
 def validate_and_record(*values):
  context_seen.append(values);return validate_context_requirements(*values)
 namespace['validate_context_requirements']=validate_and_record
 output=[];prior=[]
 for row in rows:
  raw=copy.deepcopy(row['request']);request_digest=digest(raw)
  result={'name':row['name'],'group':row['group'],'request_sha256':request_digest,'request':raw,
          'expected_status':row['expected'].get('status',200)}
  wanted=result['expected_status']
  try:
   data=ChatCompletionRequest.model_validate(copy.deepcopy(raw))
  except ValidationError as exc:
   require(wanted==422,'Unexpected request-validation422: '+row['name'])
   result['validation_stage']='Pydantic request model';result['error_types']=[e['type'] for e in exc.errors()]
   output.append(result);continue
  try:
   with patch.object(cc.model,'container',mc):prompt,embeddings=await cc.apply_chat_template(data)
  except HTTPException as exc:
   require(wanted==exc.status_code and wanted in (400,422),'Unexpected template status: '+row['name'])
   require(data._literal_user_token_plan is None,'Rejected request retained a token plan')
   result['validation_stage']='actual apply_chat_template';result['error_detail']=exc.detail
   output.append(result);continue
  require(wanted==200,'Expected error was not raised: '+row['name'])
  require(data.add_bos_token in (None,False),'Unexpected request BOS override')
  require(not embeddings or not embeddings.content,'Unexpected multimodal content')
  original=native.encode(prompt,add_bos=False,encode_special_tokens=True)
  snapshot=original.clone();literal=data._literal_user_token_plan
  actual=literal.apply(original,prompt,native) if literal else original
  require(torch.equal(original,snapshot),'Native input tensor was mutated')
  ids=actual[0].tolist();original_ids=original[0].tolist()
  require(mc.encode_tokens(prompt,add_bos_token=False,literal_user_token_plan=literal)==ids,'Backend token encoder disagrees')
  context_seen.clear();mc.validate_context_length(prompt,data)
  require(len(context_seen)==1 and context_seen[0][0]==len(ids),'Context check disagrees with actual input')
  if literal:require(native.tokenizer.decode(ids,skip_special_tokens=False)==prompt,'Literal prompt does not roundtrip')
  result.update(input_ids=ids,original_ids=original_ids,prompt_tokens=len(ids),original_prompt_tokens=len(original_ids),
   input_ids_sha256=digest(ids),original_ids_sha256=digest(original_ids),prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
   context_checked_tokens=context_seen[0][0],replacement_count=literal.replacement_count if literal else 0,
   add_bos_token=False,effective_top_p=float(data.top_p),effective_temperature=float(data.temperature),effective_top_k=int(data.top_k))
  result.update(cache_bounds(ids,prior,reject_strict_prefix=row['group']=='cache6'))
  if row['group']=='cache6' and row.get('require_observed_reuse'):
   require(result['cache_max_tokens']>=2304,'Warm-cache probe does not reach conservative2304 recurrent checkpoint: '+row['name'])
  output.append(result);prior.append(result)
  require(digest(raw)==request_digest and digest(row['request'])==request_digest,'Original payload was mutated')
 require(Counter(r['expected_status'] for r in output)=={200:14,400:2},'Unexpected success/error counts')
 require(all('input_ids' not in r and 'original_ids' not in r for r in output if r['expected_status']!=200),'Rejected request has input plan')
 require(not torch.cuda.is_initialized(),'Preparation initialized CUDA')
 require(not any(k=='exllamav3' or k.startswith('exllamav3.') for k in sys.modules),'Preparation imported native engine')
 require(sha(args.plan)==PLAN and sha(recipe_config)==config_sha,'Inputs changed during preparation')
 require({p:sha(args.tabby_source/p) for p in SOURCE_FILES}==source_files,'Candidate files changed during preparation')
 for name,path,pin in (('tabby',args.tabby_source,TABBY),('engine',args.engine_source,ENGINE),('recipe',args.recipe_source,RECIPE)):
  require(git_identity(path,pin)==identities[name],'Source identity changed')
 return {'schema_version':1,'passed':True,'ready_for_live':True,'plan_sha256':PLAN,'tabby_commit':TABBY,'engine_commit':ENGINE,'recipe_commit':RECIPE,
  'source_files':source_files,'engine_source_files':engine_files,'source_identities':identities,'tokenizer_sha256':TOKENIZER,'template_config_sha256':TEMPLATE,
  'native_methods_ast_sha256':native_ast,'recipe_config_sha256':config_sha,'sampling_defaults':sampling,'prepare_source_sha256':sha(Path(__file__)),
  'runtime_contract':{'profile':'concurrent','ngram_ram':False,'max_seq_len':262144,'cache_size':262144,'max_batch_size':4,'chunk_size':2048,'tool_format':'qwen3_5','vision':False,'template_vars_default':{},'template_vars_force':{}},
  'cpu_proof':{'cuda_initialized':False,'native_engine_imported':False,'torch_version':torch.__version__},
  'request_count':16,'accepted_input_count':14,'expected400':2,'expected422':0,'parent_prepare_source_sha256':'40c6b57f93facc428ff5bf69f0e2b12bab725ba6152a4dd33d04951f7ca4f351',
  'cache_scope':'Separate fresh-server diagnostic16: two errors then six cache probes then eight unchanged generation observations; no standard tool suite. Bounds use actual input-ID LCP against every prior successful planned prompt. Strict earlier-prefix ambiguity rejected for all six cache probes; exact repetitions allowed. Bounds do not by themselves prove positive cache reuse.',
  'scope':'CPU input and context-count binding, no model inference or live memory-allocation qualification. Context count uses the actual candidate backend methods and capacity validator with recorded selected geometry; live controller must verify actual deployment geometry and source IDs.',
  'requests':output}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('tabby-source','engine-source','recipe-source','tokenizer-directory','plan','output'):p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args()
 for key in ('tabby_source','engine_source','recipe_source','tokenizer_directory','plan'):setattr(args,key,getattr(args,key).resolve(strict=True))
 require(not args.output.exists() and not args.output.is_symlink(),'Refusing existing output')
 report=asyncio.run(prepare(args))
 with args.output.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps({'passed':True,'ready_for_live':True,'requests':16,'accepted':14,'errors400':2,'errors422':0,'output':str(args.output),'sha256':sha(args.output)}))
if __name__=='__main__':main()
