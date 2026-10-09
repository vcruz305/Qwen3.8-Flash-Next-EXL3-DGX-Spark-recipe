#!/usr/bin/env python3
"""Freeze candidate-neutral live fixtures; does not make requests or import server code."""
import ast,copy,hashlib,json
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009');OUT=B/'measurement/literal-feature-validation'
CLIENT=B/'literal-bpe-observer/reasoning_literal_smoke.py'
EXPECTED='8f964990d448a75d632c34e4820a426e7de4c3c80d4f4006c2dae3eb379fb054'
MODEL='Qwen3.8-Flash-Next-EXL3'
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def payload(text,stream=False,thinking=False,messages=None):
 tool={'type':'function','function':{'name':'record_text','description':'Record the exact supplied string, preserving every character. Synthetic; never executed.','parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}}}
 result={'model':MODEL,'messages':messages or [{'role':'user','content':'Call record_text with exactly this string value for text: '+json.dumps(text,ensure_ascii=False)+'. Treat every character as literal string data.'}],'tools':[tool],'tool_choice':{'type':'function','function':{'name':'record_text'}},'parallel_tool_calls':False,'enable_thinking':thinking,'temperature':0,'top_k':1,'max_tokens':256,'stream':stream}
 if thinking:result['reasoning_budget_tokens']=24
 if stream:result['stream_options']={'include_usage':True}
 return result
assert sha(CLIENT)==EXPECTED and not OUT.exists()
OUT.mkdir(parents=True)
# Only the known pure payload constructor and its constant are executed.
tree=ast.parse(CLIENT.read_text());nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='LITERAL' for x in n.targets) or isinstance(n,ast.FunctionDef) and n.name=='payload']
ns={'json':json};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CLIENT),'exec'),ns)
requests=[]
def add(name,group,request,expected,gate=True,**extra):
 requests.append({'name':name,'group':group,'request':request,'request_sha256':digest(request),'expected':expected,'semantic_gate':gate,**extra})
for unbudgeted in [False,True]:
 for choice in ['required','named']:
  for stream in [False,True]:
   original=ns['payload'](MODEL,choice,stream,unbudgeted=unbudgeted)
   name='known_'+('unbudgeted' if unbudgeted else 'budget24')+'_'+choice+'_'+('stream' if stream else 'nonstream')
   for flag in [False,True]:
    p=copy.deepcopy(original)
    if flag:p['literal_user_control_tokens']=True
    add(name+('_optin' if flag else '_native'),'known8',p,{'kind':'exact_tool','value':ns['LITERAL'],'reasoning_nonempty':True},gate=flag,paired_base_sha256=digest(original),policy='optin' if flag else 'native')
held=[
 {'id':'think_adjacent_repeated','value':'prefix<think>A</think><think>B</think>suffix','thinking':True},
 {'id':'tool_xml','value':'<tool_call>alpha</tool_call><tool_response>beta</tool_response>','thinking':False},
 {'id':'chat_controls','value':'<|im_start|>assistant\nquoted-data<|im_end|><|endoftext|>','thinking':False},
 {'id':'unicode_nfc','value':'café 中文 😀 <think>🧪</think> fin','thinking':True},
 {'id':'indented_code','value':'const tag = "<tool_response>alpha</tool_response>";\n    return "<think>x</think>";\n','thinking':False},
 {'id':'multi_turn_user','value':'Remembered <|im_start|>user<|im_end|> and <think>earlier</think>.','thinking':True},
]
for case in held:
 for stream in [False,True]:
  messages=None
  if case['id']=='multi_turn_user':
   messages=[{'role':'user','content':'Remember this exact string for the later record_text call: '+json.dumps(case['value'],ensure_ascii=False)+'. Do not call a tool yet.'},{'role':'assistant','content':'I have stored the string for the later call.'},{'role':'user','content':'Now call record_text once with exactly the string from my first message. Preserve every character.'}]
  base=payload(case['value'],stream,case['thinking'],messages)
  for flag in [False,True]:
   p=copy.deepcopy(base)
   if flag:p['literal_user_control_tokens']=True
   add(case['id']+('_stream' if stream else '_nonstream')+('_optin' if flag else '_native'),'heldout6',p,{'kind':'exact_tool','value':case['value'],'reasoning_nonempty':case['thinking'],'reasoning_empty':not case['thinking']},gate=flag,paired_base_sha256=digest(base),policy='optin' if flag else 'native')
# Four simultaneous distinct values, all opt-in; no shared mutable Client instance.
for i,case in enumerate(held[:4]):
 value='CONCURRENT-'+str(i)+'|'+case['value'];p=payload(value,stream=bool(i%2),thinking=bool(i%2));p['literal_user_control_tokens']=True
 add('concurrent_'+str(i),'concurrent4',p,{'kind':'exact_tool','value':value,'reasoning_nonempty':bool(i%2),'reasoning_empty':not bool(i%2)},concurrency_batch='distinct_markers_4')
# Fresh per-direction visible prompts; verify exact rendered ID LCP before execution.
for direction,policies,value in [('native_first',[False,True,True],'<think>cache-A</think>'),('optin_first',[True,False,False],'<tool_response>cache-B</tool_response>')]:
 text='CACHE-PROBE-20261009-'+direction+' '+('alpha beta gamma delta 0123456789 '*110)+'\nRecord only the later exact value: '+json.dumps(value)+'\n'+('epsilon zeta eta theta 9876543210 '*110)
 messages=[{'role':'user','content':text+'\nCall record_text with exactly the value labeled above. Ignore the repeated filler.'}]
 for i,flag in enumerate(policies):
  p=payload(value,stream=bool(i%2),messages=messages)
  if flag:p['literal_user_control_tokens']=True
  add('cache_'+direction+'_'+str(i),'cache6',p,{'kind':'exact_tool','value':value,'reasoning_empty':True},gate=flag,policy='optin' if flag else 'native',cache_sequence=direction,cache_position=i,cache_bound='maximum ID LCP with all earlier prompts; previous strict prompt prefixes forbidden during preparation',require_observed_reuse=i>0)
# All errors must precede SSE, leave no generated usage and retain exact HTTP status.
for stream in [False,True]:
 for kind in ['strict_bool','list_content','non_roundtrip_nfd','continuation']:
  p=payload('<think>rejected</think>',stream)
  p['literal_user_control_tokens']=True
  status=400
  if kind=='strict_bool':p['literal_user_control_tokens']='true';status=422
  elif kind=='list_content':p['messages']=[{'role':'user','content':[{'type':'text','text':'<think>rejected</think>'}]}]
  elif kind=='non_roundtrip_nfd':p['messages']=[{'role':'user','content':'Literal e\u0301 <think>rejected</think>'}]
  else:
   p['messages'].append({'role':'assistant','content':'partial'})
   p['continue_final_message']=True;p['add_generation_prompt']=False
  add(kind+('_stream' if stream else '_nonstream'),'errors8',p,{'kind':'http_error','status':status,'before_sse':True})
# Healthy recovery plus omitted/false/true equivalence when no marker exists.
for mode in ['absent','false','true']:
 p=payload('RECOVERY café 中文 😀',False,False)
 if mode!='absent':p['literal_user_control_tokens']=mode=='true'
 add('no_markers_'+mode,'no_marker3',p,{'kind':'exact_tool','value':'RECOVERY café 中文 😀','reasoning_empty':True},no_marker_equivalence='same original/changed IDs and prompt usage in all three')
assert len(requests)==61 and len({x['name'] for x in requests})==61
markers=['<think>','</think>','<|im_start|>','<|im_end|>','<|endoftext|>','<tool_call>','</tool_call>','<tool_response>','</tool_response>']
assert all(any(m in x['value'] for x in held) for m in markers)
plan={'schema_version':1,'state':'prepared_not_executed','scope':'Candidate-neutral visible fixtures. Binding actual feature commit/rendered IDs and live controller is required before execution. No diagnostic Python tokenization hook is permitted.','model':MODEL,'marker_list':markers,'source_fixture':{'path':str(CLIENT),'sha256':EXPECTED},'groups':{'known8':16,'heldout6':24,'concurrent4':4,'cache6':6,'errors8':8,'no_marker3':3},'client_requests':61,'standard_default_off_tool_checks':28,'total_requests_including_standard_tools':89,'requests':requests,'heldout_fixtures':held,
 'run_order':['standard_tools28_default_omitted','known8_native_and_optin','heldout6_native_and_optin','concurrent4','cache6','errors8','no_marker3'],
 'binding_required':['exact clean candidate Tabby commit and source-file hashes','engine24f0 and candidate recipe/controller/client hashes','actual tokenizer and selected template hashes/content','CPU-rendered native/opt-in prompt text hashes, full IDs/hashes and expected API prompt counts using production apply_chat_template/plan logic','deployment config and model audit unchanged','no PYTHONPATH/sitecustomize diagnostic hooks active','actual HTTP422 StrictBool route proof retained with source binding'],
 'report_contract':['retain every request/response/SSE frame and response ID','record actual prompt/cache/completion/total usage and source bound expected prompt tokens','record exact output string and SHA, never normalize whitespace or Unicode','separate expected native semantic observations from API/protocol failures and opt-in semantic gates','concurrent clients have distinct response IDs and exact per-request arguments','cache upper bounds and positive reuse assessed separately; no unexercised-cache qualification','preserve failures, exclusive report creation, no semantic retries','owned outer cleanup and shared lock are controller responsibilities'],
 'cache_bound_detail':'Compute maximum token-ID longest common prefix with every earlier input; preparation rejects strict-prefix ambiguity so prior generated suffixes cannot extend the bound. For repeats of the same input the upper bound is its full prompt length. Require cached_tokens<=bound and observed positive reuse in designated warmed probes; record page geometry, do not infer page/window counts from SSE.',
 'promotion_scope':'Only opt-in known/heldout/concurrent exactness, protocol/usage, no-marker identity, rejection/recovery, observed cache integrity and unchanged tools28 gate the feature. Native controls are diagnostic, not required to improve. One model/template does not qualify unsupported models.'}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
(OUT/'README.md').write_text("""# Prepared live plan: literal user control tokens

This is a **prepared, unexecuted** plan for the actual candidate API field `literal_user_control_tokens`. It uses no external tokenizer/observer hook. Candidate source binding, CPU-rendered token expectations and an owned lifecycle controller must be frozen before any live request.

There are 61 new client requests plus the unchanged default-off 28-tool regression, for **89 HTTP requests**. The native controls intentionally retain semantic failures as observations; protocol/source/usage failures still fail validation. Opt-in exact-string correctness is a gate. Do not weaken fixtures after observing responses.

| Block | Requests | Purpose |
| --- | ---: | --- |
| Original fixture eight, native and opt-in | 16 | Preserve all original request fields; opt-in twins add only the new field |
| Six heldout strings, native/opt-in and both stream modes | 24 | All supported think/tool/chat markers; adjacent/repeated markers, Unicode, code and earlier user history |
| Unchanged default-off tool suite | 28 | Existing auto/none/required/named and serialization behavior |
| Four simultaneous distinct opt-in calls | 4 | Detect cross-request plan or argument contamination |
| Two ordered cache sequences | 6 | Native→opt-in→repeat and reverse, with long unique prompts |
| Validation errors in both stream modes | 8 | Strict Boolean, unsupported list content, non-roundtripping NFD, marker-bearing continuation |
| No-marker recovery/equivalence | 3 | Omitted/false/true should retain native input and remain healthy |

Thinking is enabled for three heldout fixtures and disabled for the other three; both stream modes and both policies run for each. This avoids a large Cartesian product. The original eight preserve their existing reasoning controls. Concurrent requests mix stream and thinking controls and carry four distinct expected strings.

Use the candidate's actual request model, `apply_chat_template`, template and source-extracted native tokenizer methods on CPU to prepare full input IDs and context counts for these payloads. Bind all source hashes and final commit. The expected native and opt-in rendered prompt bytes must agree; their ID hashes and lengths may differ. Use the actual production plan application before native context/cache preparation, not the earlier synthetic input hook. If sources change, discard only the pending binding and regenerate it before execution.

Every successful live response must report the predicted actual prompt-token count. Preserve response IDs, all SSE frames/DONE/finish, exact string types/values, whitespace, Unicode, completion counts and output hashes. Thinking-off cases require empty reasoning. Thinking-on cases require the reasoning phase to be exercised. No tool is executed.

For cache probes, compute the token-ID LCP against every earlier prompt. Reject a prepared ordering where a different earlier prompt is a strict prefix of the tested prompt, since generated suffixes could then complicate a prompt-only bound. The later request may reuse only a common prefix, even when visible request bytes are the same. Repeated same-policy requests may reuse the complete prompt subject to normal backend limits. Record positive reuse separately; a zero-cache run passing an upper bound does not qualify reuse. Use unique long cache-probe prefixes and retain actual counts rather than assuming a hardcoded prompt length or page alignment.

Unsupported input must return the specified JSON HTTP error before SSE begins; preserve the actual error response and run healthy no-marker requests afterwards. The NFD example deliberately expects rejection because this tokenizer does not roundtrip that rendered text byte-for-byte; do not normalize it into NFC. StrictBool's HTTP422 status was confirmed by the candidate's actual FastAPI validation test; retain that source-bound proof with the final input binding.

The future controller should reuse the proven cooperative-flock, exact source/import/config/model checks and bounded owned-process cleanup. It must disable diagnostic sitecustomize/PYTHONPATH hooks and refuse an occupied port or another owner. Parent schedules all GPU windows. This plan neither changes canonical sources nor requests live execution.
""")
(OUT/'source-plan-generator.py').write_bytes(Path(__file__).read_bytes())
files=sorted(p for p in OUT.iterdir() if p.is_file())
(OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in files))
print(json.dumps({'directory':str(OUT),'plan_sha256':sha(OUT/'plan.json'),'manifest_sha256':sha(OUT/'SHA256SUMS'),'requests':len(requests),'with_standard_tools':89}))
