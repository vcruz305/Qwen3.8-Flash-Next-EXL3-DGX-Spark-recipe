#!/usr/bin/env python3
"""Read-only offline attribution from immutable synthetic raw/phase captures."""
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path('/home/vcruz/src/qwen-followup-20261009')
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
sys.path.insert(0,str(ROOT/'literal-ablation-observer-timeline'))
sys.path.insert(0,str(OLD/'recipe/bench'))
from validate_timeline import validate_timeline
from api_resilience import assembled
from tokenizers import Tokenizer

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def position(event):return event['state']['new_tokens']+event['state']['rq_new_tokens']

def prefix(events,cut,tokenizer):
 ids=[];rewinds=[]
 for event in events:
  if event['index']>=cut:break
  if event['kind']!='sample_processed_before_budget':continue
  pos=position(event)
  if event['state']['checkpoint_rewound']:
   rewinds.append({'event':event['index'],'position':pos});ids=ids[:pos];continue
  if event['state']['new_tokens']<=0:continue
  if pos!=len(ids)+1:raise ValueError('Non-contiguous retained processed-token prefix')
  ids.append(event['token_id'])
 return {'ids':ids,'text':tokenizer.decode(ids,skip_special_tokens=False),'rewinds':rewinds}

def main(args):
 cell=Path(args.cell)
 capture=json.loads((cell/'literal-capture.json').read_text())
 assert capture['capture_valid'] is True
 bindings=capture['assessor_global_bindings']
 assert bindings=={'ENGINE':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','TABBY':'f650bb5389e0a273549e47d4d26a765760c013e1','OBSERVER_SHA':'5127cc9409aac7e80401f5c435d58e373733e3375c1c899ff22732e8201ebfde'}
 tok_path=OLD/'tabbyapi-agent/.tokenizer-cpu/tokenizer.json'
 tok=Tokenizer.from_file(str(tok_path))
 apis={};statuses={}
 for name in ['literal.json','literal-unbudgeted.json']:
  d=json.loads((cell/name).read_text())
  for case in d['cases']:
   assert len(case['request_indices'])==1
   index=case['request_indices'][0]
   trace=d['requests'][index]
   api=assembled(trace,'Qwen3.8-Flash-Next-EXL3')
   request_id=api['id'].removeprefix('chatcmpl-').removeprefix('cmpl-')
   assert request_id not in apis
   apis[request_id]=api;statuses[request_id]=case['status']
 rows=[]
 for bound in capture['traces']:
  path=cell/'raw-literal'/Path(bound['path']).name
  assert sha(path)==bound['sha256']
  trace=json.loads(path.read_text());assert trace['request_id']==bound['request_id']
  api=apis[trace['request_id']]
  checked=validate_timeline(trace)
  raw=trace['raw_finish']['native_full_completion']
  assert raw==trace['raw_finish']['backend_full_response']
  events=trace['producer_events']
  controls=[e for e in events if e['kind'] in ('force_before','phase_before')]
  cut=min(e['index'] for e in controls) if controls else len(events)
  pre=prefix(events,cut,tok)
  assert raw.startswith(pre['text']),path.name
  before=[e for e in events if e['index']<cut and e['kind']=='sample_processed_before_budget']
  assert all(e['state']['filter_count']==0 for e in before)
  guard_decisions=[{'index':e['index'],'position':position(e),'allowed':e['allowed'],'parser_after':e['state']['guard_parser']} for e in events if e['kind']=='guard_after']
  forces=[{'index':e['index'],'position':position(e),'state':e['state']} for e in events if e['kind']=='force_before']
  phases=[{'index':e['index'],'position':position(e),'requested_reasoning':e['requested_reasoning'],'applied':e['applied'],'filter_count':e['state']['filter_count']} for e in events if e['kind']=='phase_after']
  natural_closes=[{'index':e['index'],'position':position(e),'token_id':e['token_id']} for e in events if e['kind']=='sample_processed_before_budget' and e['token_id']==248069 and not e['state']['checkpoint_rewound'] and not (e['state']['budget'] or {}).get('injecting')]
  call_values=[]
  for c in api['tool_calls']:
   call_values.append(json.loads(c['function']['arguments']))
  wrong='thinkaliteralthink'
  rows.append({'trace':path.name,'trace_sha256':sha(path),'variant':trace['variant'],'request_id':trace['request_id'],
   'semantic_status':statuses[trace['request_id']],'finish_reason':api['finish_reason'],'api_tool_values':call_values,
   'native_metrics':trace['raw_finish']['native_metrics'],'timeline':checked,
   'native_equals_backend':True,'pre_intervention_prefix':pre,
   'pre_intervention_prefix_is_exact_native_prefix':True,'pre_intervention_filter_count_zero':True,
   'altered_spelling_present_before_first_intervention':wrong in pre['text'],
   'altered_spelling_complete_at_token':next((i for i in range(1,len(pre['ids'])+1) if wrong in tok.decode(pre['ids'][:i],skip_special_tokens=False)),None),
   'force_events':forces,'natural_closing_tokens':natural_closes,'phase_results':phases,'guard_decisions':guard_decisions,
   'first_native_close_character':raw.find('</think>'),'native_before_first_close':raw.split('</think>',1)[0],
   'native_sha256':hashlib.sha256(raw.encode()).hexdigest(),
   'scope':'Processed-token prefix checked against native output; rewind/healing states explicitly filtered. A natural close in a quotation does not establish model intent.'})
 assert len(rows)==8
 report={'schema_version':1,'passed':True,'kind':'offline_literal_ablation_phase_review','cell':cell.name,
  'scope':'Capture/source/timeline consistency and descriptive attribution only. Semantic failures remain failures. No inference or production changes.',
  'bindings':bindings,'source_sha256':sha(Path(__file__)),'tokenizer_sha256':sha(tok_path),
  'source_files':{name:sha(cell/name) for name in ['literal-capture.json','literal.json','literal-unbudgeted.json','deployment.json','result.json']},
  'summary':{'cases':8,'semantic_statuses':dict(Counter(r['semantic_status'] for r in rows)),
    'wrong_spelling_before_first_intervention':sum(r['altered_spelling_present_before_first_intervention'] for r in rows),
    'requests_with_forcing':sum(bool(r['force_events']) for r in rows),
    'requests_with_natural_end_token':sum(bool(r['natural_closing_tokens']) for r in rows),
    'requests_without_phase_event':sum(not r['phase_results'] for r in rows)},'cases':rows}
 with Path(args.output).open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
 print(json.dumps({'passed':True,'summary':report['summary'],'report':args.output,'sha256':sha(Path(args.output))}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cell',required=True);p.add_argument('--output',required=True);main(p.parse_args())
