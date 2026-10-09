#!/usr/bin/env python3
"""Eight unchanged causal probes. Model/API outcomes remain observations."""
import argparse,copy,datetime,hashlib,importlib.util,json,os,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPECTED_SHA='74e477ee1917f64264205cd1cac9665058ec2476fe5b49b2289d3fe857bcc52c'
# Use the unchanged reviewed stdlib HTTP implementation.
API_SHA='193ec973e9dd2d77317912a312b8865933f791aa4b943673539c6283671dbd90'
TABBY='f4aadf114b0044fa8cbe1b50241dc80ea7d61583'
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load_api():
 p=HERE/'api_resilience.py';require(sha(p)==API_SHA,'HTTP helper changed')
 spec=importlib.util.spec_from_file_location('causal_http',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def expected():
 p=HERE/'observer/expected_requests.json';require(sha(p)==EXPECTED_SHA,'Exact requests changed');rows=json.loads(p.read_text())['requests']
 require(len(rows)==8 and len({r['name'] for r in rows})==8,'Expected eight distinct requests')
 for r in rows:
  value=hashlib.sha256(json.dumps(r['request'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  require(value==r['request_sha256'],'Wire request hash mismatch')
 return rows
def unique_json(raw):
 def pairs(items):
  out={}
  for key,value in items:
   require(key not in out,'Duplicate argument key');out[key]=value
  return out
 return json.loads(raw,object_pairs_hook=pairs)
def execute(row,api,args,barrier=None):
 client=api.Client(args.base_url,os.environ.get(args.api_key_env),args.timeout)
 if barrier:barrier.wait(timeout=15)
 started=time.monotonic_ns();trace=client.request('/chat/completions',copy.deepcopy(row['request']));finished=time.monotonic_ns()
 observed={'name':row['name'],'request_sha256':row['request_sha256'],'trace':trace,'started_monotonic_ns':started,'finished_monotonic_ns':finished,'semantic_passed':False,'collection_complete':False}
 try:
  require(trace.get('status')==200,'Unexpected request rejection')
  require(not trace.get('transport_or_protocol_error'),'Transport failed')
  if trace['stream']:
   ids={f['id'] for f in trace['frames'] if 'id' in f};require(len(ids)==1,'Missing or changing stream ID')
   observed['response_id']=next(iter(ids));require(trace.get('done') or trace.get('server_error'),'SSE neither completed nor returned explicit error')
  else:observed['response_id']=trace['body']['id']
  usages=[f['usage']for f in trace['frames']if f.get('usage')is not None]if trace['stream']else[trace['body'].get('usage')]
  if usages:
   require(len(usages)==1 and isinstance(usages[0],dict) and usages[0].get('prompt_tokens')==row['prompt_tokens'],'Authoritative usage differs from prepared input')
  else:require(bool(trace.get('server_error')),'No usage or explicit SSE error')
  observed['collection_complete']=True
  try:value=api.assembled(trace,row['request']['model'])
  except Exception as error:
   observed['outcome_error']=type(error).__name__+': '+str(error)
   if not trace.get('server_error'):observed['collection_complete']=False
   return observed
  observed['result']=value
  try:
   if value['usage']['prompt_tokens']!=row['prompt_tokens']:
    observed['collection_complete']=False
    raise ValueError('Observed usage differs from prepared input')
   observed['input_usage_verified']=True
   calls=value['tool_calls'];require(value['finish_reason']=='tool_calls' and len(calls)==1,'Missing complete single tool call')
   require(calls[0]['function']['name']=='record_text','Wrong function')
   args_value=unique_json(calls[0]['function']['arguments']);observed['argument']=args_value
   require(args_value=={'text':row['expected_value']} and type(args_value.get('text'))is str,'Exact string differs')
   observed['semantic_passed']=True
  except Exception as error:observed['outcome_error']=type(error).__name__+': '+str(error)
 except Exception as error:observed['collection_error']=type(error).__name__+': '+str(error)
 return observed
def run(args):
 api=load_api();rows=expected();metadata,source=api.read_metadata(args.metadata)
 require(metadata.get('engine',{}).get('commit')==ENGINE and metadata.get('server',{}).get('commit')==TABBY,'Wrong runtime sources')
 writer=api.ReportWriter(args.output)
 report={'schema_version':1,'kind':'literal_feature_causal_requests','started_at_utc':stamp(),'completed_at_utc':None,'passed':False,'scope':'Eight unchanged original r2 requests; model errors, length and string mismatches are retained observations. Raw capture integrity is independently assessed after collection.','metadata_source':source,'expected_requests_sha256':EXPECTED_SHA,'client_sha256':sha(__file__),'results':[]}
 writer.write(report)
 for row in rows[:4]:report['results'].append(execute(row,api,args));writer.write(report)
 barrier=threading.Barrier(4)
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(execute,row,api,args,barrier) for row in rows[4:]]
  outcomes=[f.result() for f in futures]
 overlap=max(r['started_monotonic_ns'] for r in outcomes)<min(r['finished_monotonic_ns'] for r in outcomes)
 for r in outcomes:r['four_request_http_overlap']=overlap
 report['results'].extend(outcomes)
 ids=[r.get('response_id') for r in report['results']]
 report['summary']={'total':8,'collection_complete':sum(r['collection_complete'] for r in report['results']),'semantic_passed':sum(r['semantic_passed'] for r in report['results']),'semantic_failed':sum(not r['semantic_passed'] for r in report['results']),'four_request_http_overlap':overlap,'distinct_response_ids':len(set(ids))==8 and all(ids)}
 report['passed']=all(r['collection_complete'] for r in report['results']) and overlap and report['summary']['distinct_response_ids']
 require(sha(args.metadata)==source['sha256'],'Deployment metadata changed')
 report['completed_at_utc']=stamp();writer.write(report);return 0 if report['passed'] else 1
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ('metadata','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--base-url',default='http://127.0.0.1:8899/v1');p.add_argument('--timeout',type=float,default=120);p.add_argument('--api-key-env',default='TABBY_API_KEY');a=p.parse_args();require(0<a.timeout<=180,'Timeout outside bound');return run(a)
if __name__=='__main__':raise SystemExit(main())
