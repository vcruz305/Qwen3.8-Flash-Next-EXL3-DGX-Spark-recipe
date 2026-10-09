"""Frozen generation8 wire and actual candidate input-ID admission; no HTTP calls."""
from pathlib import Path
import hashlib,json
PLAN_SHA='a980ffc3070f578faacaed3ec20768a1781903173472d75b6861d416ef70b348'
ENGINE='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
TABBY='f4aadf114b0044fa8cbe1b50241dc80ea7d61583'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,separators=(',',':'),sort_keys=True).encode()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def validate_binding(plan_path,inputs_path,inputs_sha):
 require(sha(plan_path)==PLAN_SHA,'Plan changed')
 require(sha(inputs_path)==inputs_sha,'Input binding changed')
 plan=json.loads(Path(plan_path).read_text());binding=json.loads(Path(inputs_path).read_text())
 require(binding.get('passed') is True and binding.get('ready_for_live') is True,'Input proof is incomplete')
 require(binding['plan_sha256']==PLAN_SHA and binding['tabby_commit']==TABBY and binding['engine_commit']==ENGINE,'Input source pins differ')
 requests=plan['requests'];rows=binding['requests']
 require(len(requests)==8 and len(rows)==8,'Exactly8 prepared requests required')
 by_name={r['name']:r for r in rows};require(len(by_name)==8,'Duplicate input names')
 for item in requests:
  row=by_name[item['name']]
  require(row['request_sha256']==item['request_sha256']==digest(item['request']),'Payload changed')
  expected=item['expected'];status=expected['status'] if expected['kind']=='http_error' else 200
  require(row['expected_status']==status,'Prepared HTTP status differs')
  require(status==200,'Generation8 requires successful admission')
  if status==200:
   ids=row['input_ids'];native=row['original_ids']
   require(isinstance(ids,list) and ids and all(type(i)is int and i>=0 for i in ids),'Invalid prepared IDs')
   require(isinstance(native,list) and native and all(type(i)is int and i>=0 for i in native),'Invalid native IDs')
   require(row['prompt_tokens']==len(ids),'Prepared context count differs')

 return plan,binding,by_name
