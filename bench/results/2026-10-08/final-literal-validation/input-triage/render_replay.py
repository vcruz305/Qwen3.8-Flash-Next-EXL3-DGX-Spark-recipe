#!/usr/bin/env python3
"""CPU-only replay of the unchanged final literal fixture through exact5a formatting."""
import asyncio, hashlib, importlib.util, json, platform, subprocess, sys
from pathlib import Path
BASE=Path('/home/vcruz/src/qwen-overnight-20261008')
OUT=BASE/'literal-final-triage-5a'
TABBY=BASE/'tabbyapi-natural-agent'
sys.path.insert(0,str(TABBY))
from tests.test_qwen_nullable_guidance import container, render
from endpoints.OAI.types.chat_completion import ChatCompletionRequest
from tokenizers import Tokenizer
EXPECTED='5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb'
head=subprocess.check_output(['git','-C',str(TABBY),'rev-parse','HEAD'],text=True).strip()
assert head==EXPECTED
assert not subprocess.check_output(['git','-C',str(TABBY),'status','--porcelain','--untracked-files=no'],text=True).strip()
fixture=BASE/'reasoning_literal_smoke.py'
spec=importlib.util.spec_from_file_location('literal_fixture',fixture)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
cfg_path=BASE/'tabbyapi-agent/.tokenizer-cpu/tokenizer_config.json'
tok_path=cfg_path.with_name('tokenizer.json')
cfg=json.loads(cfg_path.read_text())
tok=Tokenizer.from_file(str(tok_path))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
async def main():
 rows=[]
 for unbudgeted in [False,True]:
  for choice in ['required','named']:
   for stream in [False,True]:
    wire=mod.payload('Qwen3.8-Flash-Next-EXL3',choice,stream,unbudgeted=unbudgeted)
    params=ChatCompletionRequest(**wire)
    before=params.messages[0].content
    prompt=(await render(params,container(raw_template=cfg['chat_template'])))[0]
    enc=tok.encode(prompt,add_special_tokens=False)
    dec=tok.decode(enc.ids,skip_special_tokens=False)
    assert before==wire['messages'][0]['content']==params.messages[0].content
    assert mod.LITERAL in prompt and prompt==dec and len(enc.ids)==348
    rows.append({'unbudgeted':unbudgeted,'choice':choice,'stream':stream,
      'request':wire,'message_after_pydantic':before,
      'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
      'input_tokens':len(enc.ids),'prompt_decode_exact':dec==prompt,
      'literal_preserved_in_prompt':mod.LITERAL in prompt})
 with (OUT/'rendered-prompt.txt').open('x') as f: f.write(prompt)
 with (OUT/'rendered-prompt-token-ids.json').open('x') as f: json.dump(enc.ids,f)
 result={'kind':'cpu_exact_literal_fixture_render_replay','passed':True,'tabby_head':head,
  'python':platform.python_version(),'platform':platform.platform(),'source':str(Path(__file__)),
  'source_sha256':sha(Path(__file__)),'fixture_source_sha256':sha(fixture),
  'tokenizer_config_sha256':sha(cfg_path),'tokenizer_json_sha256':sha(tok_path),
  'source_files':{str(p.relative_to(TABBY)):sha(p) for p in [
   TABBY/'common/templating.py',TABBY/'endpoints/OAI/utils/chat_completion.py',
   TABBY/'endpoints/OAI/types/chat_completion.py',TABBY/'backends/exllamav3/model.py']},
  'literal_native_tokens':list(zip(tok.encode(mod.LITERAL,add_special_tokens=False).ids,
   tok.encode(mod.LITERAL,add_special_tokens=False).tokens)),
  'cases':rows,'case_count':len(rows),
  'scope':'CPU Pydantic, actual Tabby formatting/template and actual HF tokenizer only; no model or HTTP requests. The container fixture substitutes only model metadata and uses the downloaded actual3.05 template. Does not attribute generated response errors.'}
 with (OUT/'render-replay.json').open('x') as f: json.dump(result,f,indent=2,ensure_ascii=False); f.write('\n')
 print(json.dumps({'passed':True,'cases':len(rows),'prompt_tokens':348,
  'prompt_sha256':sha(OUT/'rendered-prompt.txt'),'report_sha256':sha(OUT/'render-replay.json')}))
asyncio.run(main())
