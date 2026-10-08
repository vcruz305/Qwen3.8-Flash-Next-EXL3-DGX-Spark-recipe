#!/usr/bin/env python3
"""Source-faithful CPU proof of the retained observer's YAML scalar metadata omission."""
from pathlib import Path
import asyncio,hashlib,importlib.util,json,subprocess,sys
BASE=Path('/home/vcruz/src/qwen-overnight-20261008')
ROOT=Path(__file__).resolve().parent
TABBY=BASE/'tabbyapi-natural-agent'
sys.path.insert(0,str(TABBY))
from common.tabby_config import yaml
from common.config_models import TabbyConfigModel
from common.sampling import set_global_overrides,overrides_container
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from tests.test_qwen_nullable_guidance import container,render

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def typename(value):return type(value).__module__+'.'+type(value).__qualname__

async def main():
 observer_path=BASE/'tabbyapi-diagnostics/literal-final-24f0-5a-p095/observer/strings_observer.py'
 assert sha(observer_path)=='674453a31893defd299939fe801cd58b48bdb20c7bbfaf0ec0c0fbb5f82418f6'
 assert subprocess.check_output(['git','-C',str(TABBY),'rev-parse','HEAD'],text=True).strip()=='5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb'
 assert not subprocess.check_output(['git','-C',str(TABBY),'status','--porcelain','--untracked-files=no'],text=True).strip()
 observer=load('observer_p095_yaml_proof',observer_path)
 client_path=BASE/'reasoning_literal_smoke.py'
 assert sha(client_path)=='8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054'
 client=load('original_literal_client_yaml_proof',client_path)
 config_path=ROOT/'live-config.yml'
 assert sha(config_path)=='edb32190b2790220daaf715f1d4756ba6dd50b2ee00c6feb485921a25f30d13e'
 parsed=yaml.load(config_path.read_text())
 validated=TabbyConfigModel.model_validate(parsed)
 assert validated.sampling.override_preset is None
 inline=validated.sampling.inline_overrides()
 await set_global_overrides(validated.sampling.override_preset,inline)
 assert overrides_container.effective()['top_p']['override']==0.95
 tokenizer_cfg=json.loads((BASE/'tabbyapi-agent/.tokenizer-cpu/tokenizer_config.json').read_text())
 rows=[]
 for i,variant in enumerate(observer.EXPECTED_VARIANTS):
  wire=client.payload('Qwen3.8-Flash-Next-EXL3',variant['choice'],variant['stream'],unbudgeted=variant['unbudgeted'])
  assert 'top_p' not in wire
  params=ChatCompletionRequest(**wire)
  before_type=typename(params.top_p)
  prompt=(await render(params,container(raw_template=tokenizer_cfg['chat_template'])))[0]
  assert hashlib.sha256(prompt.encode()).hexdigest()==observer.PROMPT_SHA256
  assert params.top_p==0.95 and isinstance(params.top_p,float) and type(params.top_p) is not float
  assert typename(params.top_p)=='ruamel.yaml.scalarfloat.ScalarFloat'
  assert observer.matching_request(params,variant['stream']) is True
  assert observer.scalar(params.top_p) is None
  copied=params.model_copy(deep=True)
  assert typename(copied.top_p)==typename(params.top_p) and observer.scalar(copied.top_p) is None
  record_path=BASE/'tabbyapi-diagnostics/literal-final-24f0-5a-p095/live-capture/raw-literal'/f'request-{i:02d}.json'
  record=json.loads(record_path.read_text())
  assert record['matched_request']['top_p'] is None and record['variant']==variant
  rows.append({'variant':variant,'request_omits_top_p':True,'pre_render_type':before_type,'post_render_type':typename(params.top_p),'numeric_value':float(params.top_p),'matcher_passes':True,'record_scalar_is_none':True,'deep_copy_preserves_type':True,'retained_record_sha256':sha(record_path),'retained_record_top_p_is_none':True})
 report={'passed':True,'engine_sha':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby_sha':'5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb','config_sha256':sha(config_path),'observer_sha256':sha(observer_path),'client_sha256':sha(client_path),'proof_source_sha256':sha(Path(__file__)),'yaml_direct_type':typename(parsed['sampling']['top_p']['override']),'validated_inline_type':typename(inline['top_p']['override']),'cases':rows,'scope':'Actual clean5a YAML loader, config model, sampler override construction, request Pydantic parsing and rendering; original eight requests. The observer predicate admits the effective0.95 YAML ScalarFloat, but its exact-type scalar whitelist records null. This is a metadata recorder omission; no request, raw record, production source, or generation is changed. Retained live capture failure remains unchanged and any reconciled assessment must be separate.'}
 out=ROOT/'yaml-scalar-proof.json'
 with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'passed':True,'cases':len(rows),'actual_type':report['yaml_direct_type'],'report':str(out),'sha256':sha(out)}))

if __name__=='__main__':asyncio.run(main())
