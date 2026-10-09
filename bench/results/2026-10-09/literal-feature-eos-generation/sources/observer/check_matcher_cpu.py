"""Real f4a formatting/native CPU IDs through the bounded observation matcher."""
import ast,asyncio,hashlib,importlib.util,json,subprocess,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
NEW=ROOT.parent;OLD=NEW.parent/'qwen-overnight-20261008';TABBY=NEW/'tabby-literal-eos-agent'
sys.path.insert(0,str(TABBY))
import common.model
from common.sampling import overrides_from_dict
from ruamel.yaml import YAML
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from endpoints.OAI.utils import chat_completion as cc
from tests.check_literal_user_tokens import native_tokenizer
from tests.test_literal_user_tokens import container
spec=importlib.util.spec_from_file_location('bounded_a70_observer',ROOT/'observer/strings_observer.py');obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
async def main():
 assert subprocess.check_output(['git','-C',str(TABBY),'rev-parse','HEAD'],text=True).strip()==obs.TABBY_HEAD
 config=YAML(typ='safe').load((NEW/'recipe-attribution-source/exllamav3-tabby/tabby-config.yml').read_text());overrides_from_dict(config['sampling'])
 assets=OLD/'tabbyapi-agent/.tokenizer-cpu';native,_=native_tokenizer(OLD/'exllamav3-agent/exllamav3/tokenizer/tokenizer.py',assets)
 template=json.loads((assets/'tokenizer_config.json').read_text())['chat_template'];mc,_=container(template);mc.tokenizer=native;mc.tool_format='qwen3_5'
 raw='<tool_call><function=record_text><parameter=text>synthetic</parameter></function></tool_call>';sentinel=object();calls=[]
 class Native:
  def handle_finish_chunk(self,result,request_id,full_text,label=None):calls.append((result,request_id,full_text));return {'gen_tokens':7,'finish_reason':'tool_calls'}
 native_finish=Native()
 async def collector(prompt,request_id,params,streaming_mode,start_in_reasoning_mode):
  result={'full_completion':raw,'new_tokens':7,'eos':True};native_finish.handle_finish_chunk(result,request_id,raw);assert calls[-1][0]is result;return sentinel
 module=types.SimpleNamespace(_chat_stream_collector=collector)
 rows=[];negative=0
 with tempfile.TemporaryDirectory() as temporary:
  recorder=obs.Recorder(Path(temporary)/'records','a70-cpu',8);obs.install_hooks(module,Native,recorder)
  for i,expected in enumerate(obs.EXPECTED_REQUESTS):
   wire=expected['request'];params=ChatCompletionRequest.model_validate(wire)
   with patch.object(cc.model,'container',mc):prompt,_=await cc.apply_chat_template(params)
   assert obs.matching_request(params,wire['stream'])==expected
   proof=obs.token_plan_proof(params,prompt);assert all(proof[k]==expected[k] for k in proof)
   assert sha(ROOT/'observer/expected_requests.json')==obs.EXPECTED_REQUESTS_SHA256
   value=await module._chat_stream_collector(prompt,str(i),params,wire['stream'],wire['enable_thinking']);assert value is sentinel
   record=json.loads((recorder.directory/f'request-{i:02d}.json').read_text());assert record['case_name']==expected['name'] and record['input_plan']==proof
   assert record['raw_finish']['native_full_completion']==raw==record['raw_finish']['backend_full_response']
   for key,value in [('max_tokens',255),('top_p',1.0),('literal_user_control_tokens',False),('tool_choice','auto')]:
    bad=params.model_copy(deep=True);setattr(bad,key,value);assert obs.matching_request(bad,wire['stream'])is None;negative+=1
   assert recorder.begin('wrong-prompt',prompt+' ',params,wire['stream'],wire['enable_thinking'])is None;negative+=1
   bad=params.model_copy(deep=True);bad._literal_user_token_plan=None;assert recorder.begin('no-plan',prompt,bad,wire['stream'],wire['enable_thinking'])is None;negative+=1
   rows.append({'name':expected['name'],'prompt_tokens':proof['prompt_tokens'],'start_in_reasoning':wire['enable_thinking'],'return_identity':True,'native_backend_bytes_preserved':True})
  assert recorder.count==8 and not recorder.active
 report={'passed':True,'tabby':obs.TABBY_HEAD,'source_sha256':sha(__file__),'observer_sha256':sha(ROOT/'observer/strings_observer.py'),'expected_requests_sha256':obs.EXPECTED_REQUESTS_SHA256,'actual_formatted_cases':rows,'negative_guards':negative,'scope':'Actual source-bound f4a renderer/nativeCPU encode, immutable input plans, fake native completion/collector. No model, API or GPU work.'}
 with (ROOT/'matcher-cpu-report.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'passed':True,'actual_cases':len(rows),'negative_guards':negative}))
asyncio.run(main())
