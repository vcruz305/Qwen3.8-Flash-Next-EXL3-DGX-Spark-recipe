#!/usr/bin/env python3
"""Offline exact-client replay and count summary for completed feature evidence."""
import argparse,collections,hashlib,importlib.util,json
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--harness',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists()
 assert sha(a.harness/'client.py')=='84e5ac0f85540592a52208010f2c3731de0e51e0f0958ab626c8694dcdb98b11'
 spec=importlib.util.spec_from_file_location('replay_feature_client',a.harness/'client.py');client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
 plan,binding,prepared=client.validate_binding(a.harness/'plan.json',a.harness/'prepared-inputs-final.json','ba9b7477f2b983fd6bf400acfd437f2b633d38c09d47d03bb5d23893154bfdbd')
 api=client.load_api()
 feature_paths=list(a.raw.rglob('feature.json'));assert len(feature_paths)==1
 fp=feature_paths[0];feature=json.loads(fp.read_text());assert feature['completed_at_utc']
 assert len(feature['results'])==61 and [r['name'] for r in feature['results']]==[r['name'] for r in plan['requests']]
 replay=[]
 for item,saved in zip(plan['requests'],feature['results']):
  row=client.assess(item,prepared[item['name']],saved['trace'],api,plan['model'])
  for key in ('protocol_passed','semantic_passed','passed','semantic_error','error','result','actual_usage','output_sha256','cache_evidence'):
   assert row.get(key)==saved.get(key),(item['name'],key)
  replay.append({'name':item['name'],'group':item['group'],'policy':item.get('policy'),'semantic_gate':item['semantic_gate'],'protocol_passed':row['protocol_passed'],'semantic_passed':row['semantic_passed'],'passed':row['passed'],'http_status':saved['trace'].get('status')})
 tools=json.loads((fp.parent/'tools.json').read_text());assert tools['completed_at_utc'] and len(tools['results'])==28
 posts=[]
 for row in tools['results']:
  posts.append(row['request'])
  for key in ('followup','repeated_turn'):
   if key in row:posts.append(row[key]['request'])
 assert len(posts)==32 and all('literal_user_control_tokens' not in p for p in posts)
 groups={}
 for name in dict.fromkeys(r['group'] for r in replay):
  rows=[r for r in replay if r['group']==name]
  groups[name]={'checks':len(rows),'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),'protocol_passed':sum(r['protocol_passed'] for r in rows),'gated_semantic_failures':sum(r['semantic_passed'] is False and r['semantic_gate'] for r in rows),'observational_semantic_failures':sum(r['semantic_passed'] is False and not r['semantic_gate'] for r in rows)}
 cache=[]
 for row in feature['results']:
  if row['group']=='cache6':cache.append({'name':row['name'],'passed':row['passed'],**row.get('cache_evidence',{}),'prompt_tokens':row.get('actual_usage',{}).get('prompt_tokens')})
 concurrent=[r for r in feature['results'] if r['group']=='concurrent4'];assert len(concurrent)==4
 actual_overlap=max(r['started_monotonic_ns'] for r in concurrent)<min(r['finished_monotonic_ns'] for r in concurrent)
 response_ids=[r['result']['id'] for r in feature['results'] if r.get('result')]
 report={'schema_version':1,'scope':'Offline replay of the unchanged frozen client assessor against already captured raw API traces; no requests/model/GPU actions. Native semantic controls remain observations. This is functional validation, not a throughput or broad model-quality result.','candidate_tabby':binding['tabby_commit'],'engine':binding['engine_commit'],'recipe':binding['recipe_commit'],'plan_sha256':binding['plan_sha256'],'input_binding_sha256':sha(a.harness/'prepared-inputs-final.json'),'client_sha256':sha(a.harness/'client.py'),'feature_report_sha256':sha(fp),'tools_report_sha256':sha(fp.parent/'tools.json'),'frozen_assessor_rows_replayed_exactly':len(replay),'feature_summary':feature['summary'],'feature_passed':feature['passed'],'groups':groups,'tools_summary':tools['summary'],'actual_chat_posts':len(replay)+len(posts),'total_checks':len(replay)+len(tools['results']),'accepted_response_ids':len(response_ids),'unique_response_ids':len(set(response_ids)),'http_status_counts':dict(collections.Counter(r['http_status'] for r in replay)),'cache':cache,'concurrent4':{'actual_request_interval_overlap':actual_overlap,'all_reported_overlap':all(r.get('concurrency_overlap_observed') for r in concurrent),'distinct_ids':len(set(r.get('result',{}).get('id') for r in concurrent))},'rows':replay}
 report['passed']=bool(feature['passed'] and tools['summary']=={'passed':28,'failed':0,'total':28} and all(r['passed'] for r in replay) and len(set(response_ids))==len(response_ids) and actual_overlap)
 a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({k:report[k] for k in ('passed','actual_chat_posts','total_checks','feature_summary','tools_summary','concurrent4')}))
if __name__=='__main__':main()
