#!/usr/bin/env python3
"""Source-bound 61-request feature client. No model lifecycle or tokenizer hooks."""
from __future__ import annotations
import argparse,copy,datetime,hashlib,importlib.util,json,os,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE=Path(__file__).resolve().parent
API_SHA='193ec973e9dd2d77317912a312b8865933f791aa4b943673539c6283671dbd90'
PLAN_SHA='ca62602653e1e872481a568c4005218a5cc5388063b76f012312a7ae47827645'
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
TABBY='a70ae1fa9e457e478c3d96bdc84012a3cb331796'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(value,message):
 if not value:raise ValueError(message)
def load_api():
 p=HERE/'frozen/api_resilience.py';require(sha(p)==API_SHA,'Frozen HTTP client changed')
 spec=importlib.util.spec_from_file_location('literal_feature_http',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate_binding(plan_path,inputs_path,inputs_sha):
 require(sha(plan_path)==PLAN_SHA,'Plan changed')
 require(sha(inputs_path)==inputs_sha,'Input binding changed')
 plan=json.loads(Path(plan_path).read_text());binding=json.loads(Path(inputs_path).read_text())
 require(binding.get('passed') is True and binding.get('ready_for_live') is True,'Input proof is incomplete')
 require(binding['plan_sha256']==PLAN_SHA and binding['tabby_commit']==TABBY and binding['engine_commit']==ENGINE,'Input source pins differ')
 requests=plan['requests'];rows=binding['requests']
 require(len(requests)==61 and len(rows)==61,'Exactly61 prepared requests required')
 by_name={r['name']:r for r in rows};require(len(by_name)==61,'Duplicate input names')
 for item in requests:
  row=by_name[item['name']]
  require(row['request_sha256']==item['request_sha256']==digest(item['request']),'Payload changed')
  expected=item['expected'];status=expected['status'] if expected['kind']=='http_error' else 200
  require(row['expected_status']==status,'Prepared HTTP status differs')
  if status==200:
   ids=row['input_ids'];native=row['original_ids']
   require(isinstance(ids,list) and ids and all(type(i)is int and i>=0 for i in ids),'Invalid prepared IDs')
   require(isinstance(native,list) and native and all(type(i)is int and i>=0 for i in native),'Invalid native IDs')
   require(row['prompt_tokens']==len(ids),'Prepared context count differs')
   if item['group']=='cache6':require(type(row.get('cache_max_tokens'))is int and 0<=row['cache_max_tokens']<=len(ids),'Missing cache LCP bound')
 return plan,binding,by_name
def unique_json(raw):
 def pairs(items):
  out={}
  for k,v in items:
   require(k not in out,'Duplicate argument key');out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs)
def semantic(result,expected):
 require(result['finish_reason']=='tool_calls','Expected complete tool_calls finish')
 require(not result['content'].strip(),'Unexpected final prose')
 if expected.get('reasoning_nonempty'):require(bool(result['reasoning_content'].strip()),'Reasoning phase not exercised')
 if expected.get('reasoning_empty'):require(not result['reasoning_content'],'Thinking-off request emitted reasoning')
 calls=result['tool_calls'];require(len(calls)==1 and calls[0]['function']['name']=='record_text','Unexpected function/count')
 values=unique_json(calls[0]['function']['arguments'])
 require(values=={'text':expected['value']} and type(values.get('text'))is str,'Exact argument differs')
 return {'argument_sha256':hashlib.sha256(values['text'].encode()).hexdigest()}
def assess(item,prepared,trace,api,model):
 row={'name':item['name'],'group':item['group'],'request_sha256':item['request_sha256'],'semantic_gate':item['semantic_gate'],'protocol_passed':False,'semantic_passed':None,'passed':False,'trace':trace}
 try:
  expected=item['expected']
  if expected['kind']=='http_error':
   require(not trace.get('transport_or_protocol_error'),'Transport failed instead of HTTP validation')
   require(trace['status']==expected['status'],'Unexpected error HTTP status')
   require(not trace.get('frames') and not trace.get('done') and 'text/event-stream' not in trace.get('content_type',''),'Validation began SSE')
   require(isinstance(trace.get('body'),dict),'Validation error is not JSON')
   body=trace['body']
   if expected['status']==400:
    detail=body.get('detail',body.get('error',{}).get('message') if isinstance(body.get('error'),dict) else None)
    require(isinstance(detail,str) and detail==prepared.get('error_detail') and 'literal_user_control_tokens' in detail,'HTTP400 is not the prepared feature rejection')
   else:
    detail=body.get('detail');require(isinstance(detail,list),'HTTP422 has no validation details')
    require(len(detail)==1 and detail[0].get('loc')==['body','literal_user_control_tokens'] and detail[0].get('type')=='bool_type','HTTP422 is unrelated to strict feature Boolean')
    require(prepared.get('error_types')==['bool_type'],'Prepared Boolean error differs')
   row.update(protocol_passed=True,semantic_passed=True,passed=True)
   return row
  result=api.assembled(trace,model);usage=result['usage'];row['result']=result
  require(usage['prompt_tokens']==prepared['prompt_tokens'],'Actual prompt count differs from native/expanded IDs')
  usages=[f['usage'] for f in trace['frames'] if f.get('usage') is not None] if trace['stream'] else [trace['body'].get('usage')]
  require(len(usages)==1,'Exactly one authoritative usage required')
  cached=usage.get('prompt_tokens_details',{}).get('cached_tokens')
  require(type(cached)is int and 0<=cached<=prepared['prompt_tokens'],'Invalid cache count')
  row['actual_usage']=usage;row['prepared_input_ids_sha256']=digest(prepared['input_ids'])
  row['output_sha256']=digest({k:result[k] for k in ('content','reasoning_content','tool_calls','finish_reason')})
  if item['group']=='cache6':
   maximum=prepared['cache_max_tokens'];require(cached<=maximum,'Cache reused beyond prepared token-ID LCP')
   if item.get('require_observed_reuse'):require(cached>0,'Warm cache reuse not exercised')
   row['cache_evidence']={'actual_cached_tokens':cached,'maximum_expected_lcp_tokens':maximum,'positive_reuse':cached>0}
  row['protocol_passed']=True
  try:
   row.update(semantic(result,expected));row['semantic_passed']=True
  except Exception as exc:
   row['semantic_passed']=False;row['semantic_error']=type(exc).__name__+': '+str(exc)
  row['passed']=row['protocol_passed'] and (row['semantic_passed'] or not item['semantic_gate'])
 except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
 return row
def execute(item,prepared,api,base,model,key,timeout,barrier=None,opener=None):
 client=api.Client(base,key,timeout,opener=opener)
 if barrier is not None:barrier.wait(timeout=15)
 start=time.monotonic_ns()
 trace=client.request('/chat/completions',copy.deepcopy(item['request']))
 finish=time.monotonic_ns()
 row=assess(item,prepared,trace,api,model);row['started_monotonic_ns']=start;row['finished_monotonic_ns']=finish;return row
def run(args):
 api=load_api();plan,binding,prepared=validate_binding(args.plan,args.inputs,args.inputs_sha256)
 metadata,metadata_source=api.read_metadata(args.metadata)
 require(metadata.get('engine',{}).get('commit')==ENGINE and metadata.get('server',{}).get('commit')==TABBY,'Deployment source pins differ')
 writer=api.ReportWriter(args.output)
 report={'schema_version':1,'kind':'literal_user_control_tokens_feature','started_at_utc':stamp(),'passed':False,'completed_at_utc':None,'plan_sha256':PLAN_SHA,'input_binding_sha256':args.inputs_sha256,'controller_deployment':metadata,'metadata_source':metadata_source,'client_sha256':sha(__file__),'http_client_sha256':API_SHA,'expected_checks':61,'expected_chat_requests':61,'scope':'Actual API flag, no external tokenization hook. Native semantics are observations; protocol, opt-in, rejection, recovery and cache checks gate.','results':[]}
 writer.write(report);key=os.environ.get(args.api_key_env);rows=plan['requests'];i=0
 while i<len(rows):
  item=rows[i]
  if item['group']=='concurrent4':
   group=rows[i:i+4];require(len(group)==4 and all(r['group']=='concurrent4' for r in group),'Malformed concurrency block')
   barrier=threading.Barrier(4)
   with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(execute,r,prepared[r['name']],api,args.base_url,plan['model'],key,args.timeout,barrier) for r in group]
    outcomes=[f.result() for f in futures]
   overlap=max(r['started_monotonic_ns'] for r in outcomes)<min(r['finished_monotonic_ns'] for r in outcomes)
   ids=[r.get('result',{}).get('id') for r in outcomes]
   good=overlap and all(ids) and len(set(ids))==4
   for row in outcomes:
    row['concurrency_overlap_observed']=overlap
    if not good:row.update(passed=False,concurrency_error='No four-request overlap or duplicate/missing response IDs')
   report['results'].extend(outcomes);i+=4
  else:
   report['results'].append(execute(item,prepared[item['name']],api,args.base_url,plan['model'],key,args.timeout));i+=1
  writer.write(report)
  print(json.dumps({'completed':len(report['results']),'latest':report['results'][-1]['name'],'passed':report['results'][-1]['passed']}),flush=True)
 require(sha(args.metadata)==metadata_source['sha256'],'Deployment metadata changed during clients')
 require(sha(args.plan)==PLAN_SHA and sha(args.inputs)==args.inputs_sha256,'Prepared inputs changed during clients')
 ids=[r['result']['id'] for r in report['results'] if r.get('result')]
 duplicate_ids=len(ids)!=len(set(ids))
 report['summary']={'total':len(report['results']),'passed':sum(r['passed'] for r in report['results']),'failed':sum(not r['passed'] for r in report['results']),'native_semantic_observations_failed':sum(r['semantic_passed']is False and not r['semantic_gate'] for r in report['results']),'semantic_gate_failures':sum(r['semantic_passed']is False and r['semantic_gate'] for r in report['results']),'protocol_failures':sum(not r['protocol_passed'] for r in report['results']),'actual_chat_requests':len(report['results']),'duplicate_response_ids':duplicate_ids}
 report['passed']=len(report['results'])==61 and all(r['passed'] for r in report['results']) and not duplicate_ids
 report['completed_at_utc']=stamp();writer.write(report);return 0 if report['passed'] else 1
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('plan','inputs','metadata','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--inputs-sha256',required=True);p.add_argument('--base-url',default='http://127.0.0.1:8899/v1');p.add_argument('--timeout',type=float,default=120);p.add_argument('--api-key-env',default='TABBY_API_KEY');a=p.parse_args()
 require(0<a.timeout<=180,'Timeout outside bounded range')
 return run(a)
if __name__=='__main__':raise SystemExit(main())
