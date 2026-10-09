#!/usr/bin/env python3
"""Bounded offline release review; never prints matched credential values."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, importlib.util, json, re, sys
NEW=Path('/home/vcruz/src/qwen-followup-20261009')
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
ARCHIVE=NEW/'results/literal-ablation-24f0-f650'
OUT=Path(__file__).parent/'report.json'
sys.path[:0]=[str(OLD/'recipe/bench'),str(NEW/'literal-ablation-observer-timeline')]
from api_resilience import assembled
from validate_timeline import validate_timeline
spec=importlib.util.spec_from_file_location('literal_client',OLD/'reasoning_literal_smoke.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inventory():
 return {str(p.relative_to(ARCHIVE)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ARCHIVE.rglob('*')) if p.is_file()}
def require(v,message):
 if not v:raise AssertionError(message)
before=inventory();assert len(before)==59 and sum(v['bytes'] for v in before.values())==2475817
patterns={
 'private_key_block':rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----',
 'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
 'huggingface_token':rb'\bhf_[A-Za-z0-9]{25,}',
 'openai_style_key':rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}',
 'aws_access_key':rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'google_api_key':rb'\bAIza[A-Za-z0-9_-]{30,}',
 'slack_token':rb'\bxox[baprs]-[A-Za-z0-9-]{20,}',
 'jwt':rb'\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}',
 'authorization_value':rb'(?im)\b(?:authorization|proxy-authorization)\s*[=:]\s*["\x27]?(?:bearer|basic)\s+[A-Za-z0-9+/_.=-]{8,}',
 'url_credentials':rb'(?i)https?://[^\s/:@]+:[^\s/@]+@',
 'credential_field':rb'(?i)\b(?:api[_-]?key|admin[_-]?key|authorization|password|passwd|credential|secret|access[_-]?token|refresh[_-]?token|bearer|private[_-]?key|set-cookie)\b',
}
hits=[]
for name in before:
 data=(ARCHIVE/name).read_bytes()
 for kind,pattern in patterns.items():
  for match in re.finditer(pattern,data):
   hits.append({'path':name,'line':data[:match.start()].count(b'\n')+1,'byte_offset':match.start(),'type':kind})
# No matched bytes, value hashes or snippets are retained.
outer=json.loads((ARCHIVE/'result.json').read_text())
assert outer['state']=='completed' and outer['capture_valid'] is True and outer['passed'] is False
assert len(outer['cells'])==2
rows=[];raw_ids=set();replayed=0;all_events=0
for label,expected_counts in [('literal-disabled',{'fail':8}),('literal-mtp',{'fail':7,'pass':1})]:
 cell=ARCHIVE/label;life=json.loads((cell/'result.json').read_text());cap=json.loads((cell/'literal-capture.json').read_text())
 assert life['state']=='completed' and life['passed'] is False
 cleanup=life['server_cleanup'];assert cleanup['owned_group_empty'] is True and cleanup['exit_code']==0 and cleanup['unexpected_exit'] is False and cleanup['exit_before_cleanup'] is None
 assert cap['capture_valid'] is True and cap['original_semantic_passed'] is False and cap['original_semantic_counts']==expected_counts
 bound=next(v for v in outer['cells'] if v['label']==label)
 assert bound['semantic_counts']==expected_counts and bound['assessment_sha256']==sha(cell/'literal-capture.json')
 apis={};statuses={};case_rows=[];counts=Counter()
 for report_name,unbudgeted in [('literal.json',False),('literal-unbudgeted.json',True)]:
  doc=json.loads((cell/report_name).read_text());assert len(doc['cases'])==4 and len(doc['requests'])==4 and doc['unbudgeted'] is unbudgeted
  assert [v['name'] for v in doc['cases']]==client.expected_names()
  part_counts=Counter()
  for c in doc['cases']:
   assert len(c['request_indices'])==1
   req=doc['requests'][c['request_indices'][0]]
   choice='required' if '_required_' in c['name'] else 'named';stream=c['name'].endswith('_stream')
   assert req['request']==client.payload('Qwen3.8-Flash-Next-EXL3',choice,stream,unbudgeted=unbudgeted)
   api=assembled(req,'Qwen3.8-Flash-Next-EXL3')
   try:client.validate(api,require);status='pass';error_type=None
   except (AssertionError,KeyError,ValueError) as exc:status='fail';error_type=type(exc).__name__
   assert c['status']==status
   assert api['usage']['prompt_tokens']==348
   id_=api['id'].removeprefix('chatcmpl-').removeprefix('cmpl-');assert id_ not in apis
   apis[id_]=api;statuses[id_]=status
   part_counts[status]+=1;counts[status]+=1;replayed+=1
   case_rows.append({'name':c['name'],'unbudgeted':unbudgeted,'status':status,'finish_reason':api['finish_reason'],'validation_error_type':error_type})
  assert dict(part_counts)==doc['summary'] and doc['passed']==(part_counts['fail']==0)
  life_client=next(x for x in life['clients'] if Path(x['report']).name==report_name)
  assert life_client['report_sha256']==sha(cell/report_name) and life_client['counts']==dict(part_counts) and life_client['cleanup']['owned_group_empty'] is True
 assert dict(counts)==expected_counts
 assert len(cap['traces'])==8 and len(cap['producer_timelines'])==8
 assert sorted(p.name for p in (cell/'raw-literal').glob('*.json'))==['observer-manifest.json']+[f'request-{i:02d}.json' for i in range(8)]
 variants=set()
 for tr in cap['traces']:
  p=cell/'raw-literal'/Path(tr['path']).name;assert sha(p)==tr['sha256']
  raw=json.loads(p.read_text());id_=raw['request_id'];assert id_ in apis and id_ not in raw_ids;raw_ids.add(id_)
  assert raw['raw_finish']['native_full_completion']==raw['raw_finish']['backend_full_response']
  validate_timeline(raw);assert raw['producer_events_dropped']==0
  all_events+=len(raw['producer_events']);variants.add(json.dumps(raw['variant'],sort_keys=True))
 assert len(variants)==8
 rows.append({'cell':label,'summary':dict(counts),'captures':8,'unique_variants':8,'native_backend_equal':8,'prompt_tokens':348,'server_cleanup_verified':True,'cases':case_rows})
# Confirm byte identity with the separately analyzed immutable subset.
prior=NEW/'literal-ablation-review';shared=[]
for name,entry in before.items():
 old=prior/name
 if old.is_file():assert sha(old)==entry['sha256'];shared.append(name)
assert len(shared)>=25
assert inventory()==before
report={'schema_version':1,'kind':'bounded_offline_literal_archive_release_review','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_archive':str(ARCHIVE),'source_files':before,'file_count':len(before),'total_bytes':sum(v['bytes'] for v in before.values()),
 'review_source_sha256':sha(Path(__file__)),'client_source_sha256':sha(OLD/'reasoning_literal_smoke.py'),
 'scope':'All59 files scanned as bytes including four pyc files; saved16 API responses assembled and validated offline with unchanged client. No API/GPU/lifecycle actions, raw files untouched.',
 'credential_scan':{'candidate_count':len(hits),'candidates':hits,'pattern_types':list(patterns),'limits':'Bounded recognizer/key-name scan; not proof of absence of arbitrary unlabelled credentials. Values and matching snippets are never emitted. Source/process IDs, hashes, ownership_token values and local paths are operational provenance, not authentication secrets.'},
 'semantic_replays':replayed,'raw_captures':len(raw_ids),'timeline_events_validated':all_events,'cells':rows,
 'prior_review_byte_equal_files':len(shared),'archive_unchanged':True,
 'review_passed':len(hits)==0,'model_semantic_passed':False}
with OUT.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'report':str(OUT),'sha256':sha(OUT),'review_source_sha256':sha(Path(__file__)),'file_count':len(before),'bytes':sum(v['bytes'] for v in before.values()),'credential_candidates':len(hits),'semantic_replays':replayed,'raw_captures':len(raw_ids),'timeline_events':all_events,'prior_byte_equal':len(shared),'cells':[{'cell':r['cell'],'summary':r['summary']} for r in rows]}))
