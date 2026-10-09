#!/usr/bin/env python3
"""Offline known-prompt changed-input review. No network/model work."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from collections import Counter
NEW=Path('/home/vcruz/src/qwen-followup-20261009')
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
sys.path[:0]=[str(NEW),str(NEW/'literal-bpe-observer'),str(OLD/'recipe/bench')]
from analyze_literal_ablation import prefix,position
from validate_input_tokenization import validate_input_tokenization
from validate_timeline import validate_timeline
from api_resilience import assembled
from tokenizers import Tokenizer
spec=importlib.util.spec_from_file_location('original_literal_client',OLD/'reasoning_literal_smoke.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(value,message):
 if not value:raise AssertionError(message)
def main(args):
 cell=Path(args.cell);out=Path(args.output)
 cap=json.loads((cell/'literal-capture.json').read_text());assert cap['capture_valid'] is True
 assert cap['assessor_global_bindings']=={'ENGINE':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','TABBY':'f650bb5389e0a273549e47d4d26a765760c013e1','OBSERVER_SHA':'72ad246d0f746c053431a992b0c03959904c2310fe37da21d83420ce4537e592'}
 tokfile=OLD/'tabbyapi-agent/.tokenizer-cpu/tokenizer.json';tok=Tokenizer.from_file(str(tokfile))
 apis={};statuses={};wire={}
 for name,unbudgeted in [('literal.json',False),('literal-unbudgeted.json',True)]:
  d=json.loads((cell/name).read_text());assert len(d['cases'])==len(d['requests'])==4
  assert [c['name'] for c in d['cases']]==client.expected_names()
  counts=Counter()
  for c in d['cases']:
   assert len(c['request_indices'])==1;req=d['requests'][c['request_indices'][0]]
   choice='required' if '_required_' in c['name'] else 'named';stream=c['name'].endswith('_stream')
   assert req['request']==client.payload('Qwen3.8-Flash-Next-EXL3',choice,stream,unbudgeted=unbudgeted)
   api=assembled(req,'Qwen3.8-Flash-Next-EXL3')
   assert api['usage']['prompt_tokens']==352
   try:client.validate(api,require);status='pass'
   except (AssertionError,ValueError,KeyError):status='fail'
   assert status==c['status'];counts[status]+=1
   id_=api['id'].removeprefix('chatcmpl-').removeprefix('cmpl-');assert id_ not in apis
   apis[id_]=api;statuses[id_]=status;wire[id_]=hashlib.sha256(json.dumps(req['request'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
  assert dict(counts)==d['summary']
 baseline=json.loads((NEW/'literal-ablation-review/disabled-phase-review.json').read_text())
 base={json.dumps(r['variant'],sort_keys=True):r for r in baseline['cases']}
 rows=[]
 for entry in cap['traces']:
  path=cell/'raw-literal'/Path(entry['path']).name;assert sha(path)==entry['sha256']
  trace=json.loads(path.read_text());id_=trace['request_id'];assert id_==entry['request_id'] and id_ in apis
  api=apis[id_];proof=validate_input_tokenization(trace);timeline=validate_timeline(trace)
  assert tok.decode(trace['input_tokenization']['original_token_ids'],skip_special_tokens=False)==trace['rendered_prompt']
  assert tok.decode(trace['input_tokenization']['changed_token_ids'],skip_special_tokens=False)==trace['rendered_prompt']
  raw=trace['raw_finish']['native_full_completion'];assert raw==trace['raw_finish']['backend_full_response']
  events=trace['producer_events'];controls=[e for e in events if e['kind'] in ('force_before','phase_before')]
  cut=min((e['index'] for e in controls),default=len(events));pre=prefix(events,cut,tok)
  assert raw.startswith(pre['text'])
  before=[e for e in events if e['index']<cut and e['kind']=='sample_processed_before_budget']
  assert all(e['state']['filter_count']==0 for e in before)
  forces=[{'event':e['index'],'position':position(e),'state':e['state']} for e in events if e['kind']=='force_before']
  phases=[{'event':e['index'],'position':position(e),'applied':e['applied'],'requested_reasoning':e['requested_reasoning'],'filter_count':e['state']['filter_count']} for e in events if e['kind']=='phase_after']
  closes=[{'event':e['index'],'position':position(e),'token_id':e['token_id']} for e in events if e['kind']=='sample_processed_before_budget' and e['token_id']==248069 and not e['state']['checkpoint_rewound'] and not (e['state']['budget'] or {}).get('injecting')]
  values=[json.loads(c['function']['arguments']) for c in api['tool_calls']]
  b=base[json.dumps(trace['variant'],sort_keys=True)]
  rows.append({'trace':path.name,'trace_sha256':sha(path),'request_id':id_,'variant':trace['variant'],'wire_request_sha256':wire[id_],
   'semantic_status':statuses[id_],'baseline_disabled_semantic_status':b['semantic_status'],'finish_reason':api['finish_reason'],'api_tool_values':values,
   'input_proof':proof,'actual_api_usage':api['usage'],'timeline':timeline,'native_equals_backend':True,
   'pre_intervention_prefix':pre,'pre_intervention_prefix_is_exact_native_prefix':True,'pre_intervention_filter_count_zero':True,
   'old_wrong_spelling_before_intervention':'thinkaliteralthink' in pre['text'],
   'exact_requested_literal_before_intervention':client.LITERAL in pre['text'],
   'force_events':forces,'phase_results':phases,'natural_closing_tokens':closes,
   'native_before_first_close':raw.split('</think>',1)[0],
   'native_sha256':hashlib.sha256(raw.encode()).hexdigest(),
   'baseline_disabled_prefix':b['pre_intervention_prefix']['text']})
 assert len(rows)==8 and len({json.dumps(r['variant'],sort_keys=True) for r in rows})==8
 counts=dict(Counter(r['semantic_status'] for r in rows))
 assert counts==cap['original_semantic_counts']
 report={'schema_version':1,'kind':'offline_changed_input_literal_review','review_passed':True,'model_semantic_passed':counts.get('fail',0)==0,
  'scope':'One target-only known synthetic prompt experiment. Visible/wire bytes unchanged; model input intentionally changed348to352. No production policy or performance claim.',
  'source_sha256':sha(Path(__file__)),'prefix_helper_sha256':sha(NEW/'analyze_literal_ablation.py'),'proof_validator_sha256':sha(NEW/'literal-bpe-observer/validate_input_tokenization.py'),
  'client_sha256':sha(OLD/'reasoning_literal_smoke.py'),'tokenizer_sha256':sha(tokfile),'bindings':cap['assessor_global_bindings'],
  'baseline_disabled_review_sha256':sha(NEW/'literal-ablation-review/disabled-phase-review.json'),
  'source_files':{name:sha(cell/name) for name in ['literal-capture.json','literal.json','literal-unbudgeted.json','deployment.json','result.json']},
  'summary':{'cases':8,'semantic_statuses':counts,'input_proofs':8,'actual_api_prompt_tokens':352,'native_backend_equal':8,
   'old_wrong_spelling_before_intervention':sum(r['old_wrong_spelling_before_intervention'] for r in rows),'exact_literal_before_intervention':sum(r['exact_requested_literal_before_intervention'] for r in rows),
   'requests_with_forcing':sum(bool(r['force_events']) for r in rows),'requests_with_natural_end_token':sum(bool(r['natural_closing_tokens']) for r in rows)},'cases':rows}
 with out.open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps({'report':str(out),'sha256':sha(out),'summary':report['summary']}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',required=True);p.add_argument('--output',required=True);main(p.parse_args())
