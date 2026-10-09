#!/usr/bin/env python3
"""Read-only completed standby-unit evidence; sanitize all logs on Spark before transfer."""
import argparse,ast,base64,datetime,hashlib,json,re,subprocess
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009');OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
FILTER=OLD/'collect_service_evidence.py'
FILTER_FILE_SHA='cd530cac4639f082c7a6ad47b30528d46204d40b44d94e129708bb4a522e4251'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['source-dir','expected-recipe','expected-engine','expected-tabby','expected-controller','expected-result']:p.add_argument('--'+n,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert all(re.fullmatch('[0-9a-f]{40}',getattr(a,n))for n in ['expected_recipe','expected_engine','expected_tabby'])
 assert all(re.fullmatch('[0-9a-f]{64}',getattr(a,n))for n in ['expected_controller','expected_result'])
 assert a.source_dir.startswith('/home/cruzspark/qwen-followup-20261009/results/') and not a.output.exists()
 assert sha(FILTER)==FILTER_FILE_SHA,'Frozen sanitization source changed'
 tree=ast.parse(FILTER.read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FILTER_SOURCE'for t in n.targets));filter_source=ast.literal_eval(node.value)
 settings={'source':a.source_dir,'expected':{'recipe':a.expected_recipe,'engine':a.expected_engine,'server':a.expected_tabby},'controller':a.expected_controller,'result':a.expected_result}
 remote=filter_source+r'''
import base64,datetime,hashlib,stat
from pathlib import Path
S=SETTINGS
root=Path(S['source']);assert not root.is_symlink() and root.resolve().is_relative_to(Path('/home/cruzspark/qwen-followup-20261009/results'))
def raw(path):
 st=path.lstat();assert stat.S_ISREG(st.st_mode) and st.st_size<2000000
 data=path.read_bytes();end=path.lstat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(end.st_ino,end.st_size,end.st_mtime_ns)
 return data
result_raw=raw(root/'result.json');assert hashlib.sha256(result_raw).hexdigest()==S['result']
d=json.loads(result_raw);assert d['state']=='completed' and d['passed']is True and d['finished_at_utc'] and d['controller_sha256']==S['controller']
assert d['sdk_exit_code']==0 and d['raw_transport_checks']==2
assert d['sdk_summary']['passed']==5 and d['sdk_summary']['failed']==d['sdk_summary']['skipped']==0
for side in ['sources','sources_after']:
 for name,commit in S['expected'].items():assert d[side][name]['commit']==commit and not d[side][name]['tracked_changes']
assert d['unit_before_sdk']['MainPID']==d['unit_after_sdk']['MainPID'] and d['unit_before_sdk']['InvocationID']==d['unit_after_sdk']['InvocationID'] and d['unit_before_sdk']['NRestarts']==d['unit_after_sdk']['NRestarts']
assert d['unit_after_sdk']['ActiveState']=='active' and d['selected_settings']['CACHE_SIZE']=='1048576'
sdk=raw(root/'sdk.json');assert hashlib.sha256(sdk).hexdigest()==d['sdk_report_sha256'];sr=json.loads(sdk);assert len(sr['cases'])==5 and all(r['status']=='pass'for r in sr['cases'])
phase=json.loads(raw(root/'selected-unit/record.json'));assert len(phase['api']['requests'])==2
assert phase['unit']['MainPID']==d['unit_after_sdk']['MainPID'] and phase['unit']['InvocationID']==d['unit_after_sdk']['InvocationID']
files={}
for name in ['result.json','sdk.json','sdk.log','selected-unit/record.json','selected-unit/journal.log']:
 data=raw(root/name)
 if name.endswith('.json'):public,policy=sanitize_json(data)
 else:public,policy=lifecycle_excerpt(data)
 files[name]={'source_sha256':hashlib.sha256(data).hexdigest(),'source_bytes':len(data),'public_sha256':hashlib.sha256(public).hexdigest(),'public_bytes':len(public),'policy':policy,'base64':base64.b64encode(public).decode()}
print(json.dumps({'source':str(root),'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'excluded':['selected-unit/config.yml: deployment config already present in structured record; raw YAML not copied','All credential stores, private environment backups, model/extension/tensor files are outside the allowlist.'],'raw_logs_transferred':False}))
'''.replace('SETTINGS',repr(settings),1)
 result=subprocess.run(['bash',str(OLD/'spark-ssh'),'python3','-'],input=remote,text=True,capture_output=True,check=True,timeout=45)
 receipt=json.loads(result.stdout);a.output.mkdir(parents=True)
 for n,v in receipt['files'].items():
  rel=Path(n);assert not rel.is_absolute()and '..'not in rel.parts
  data=base64.b64decode(v.pop('base64'));assert hashlib.sha256(data).hexdigest()==v['public_sha256'];dest=a.output/'reports'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 receipt.update(collector_sha256=sha(Path(__file__)),filter_source_file_sha256=sha(FILTER),filter_literal_sha256=hashlib.sha256(filter_source.encode()).hexdigest(),expected=settings)
 (a.output/'collection-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(a.output),'receipt_sha256':sha(a.output/'collection-receipt.json'),'files':len(receipt['files']),'raw_logs_transferred':False}))
if __name__=='__main__':main()
