#!/usr/bin/env python3
"""Read-only safe installation, source and CPU receipts; no private env or backup access."""
from pathlib import Path
import ast,base64,hashlib,json,subprocess
B=Path('/home/vcruz/src/qwen-followup-20261009');OLD=Path('/home/vcruz/src/qwen-overnight-20261008');O=B/'final-standby-unit-collected/extras'
f=OLD/'collect_service_evidence.py';tree=ast.parse(f.read_text());filter_source=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='FILTER_SOURCE'for t in n.targets)))
names=['canonical-deployment-stage.json','service-gpu-lock-install.json','final-standby-unit-launch.json','final-standby-unit-release.json','results/canonical-recipe-cpu-22d/result.json','results/canonical-recipe-cpu-22d/recipe-tests.log','results/canonical-recipe-cpu-22d/setup-check.log']
remote=filter_source+r'''
from pathlib import Path
import base64,hashlib,json,stat
base=Path('/home/cruzspark/qwen-followup-20261009');out={}
for n in NAMES:
 p=base/n;st=p.lstat();assert stat.S_ISREG(st.st_mode)and st.st_size<2000000
 raw=p.read_bytes();after=p.lstat();assert(st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 if n.endswith('.json'):public,policy=sanitize_json(raw)
 else:
  lines=raw.decode('utf-8').splitlines();selected=[line for line in lines if re.fullmatch(r'Ran \d+ tests? in [0-9.]+s|OK(?: \(skipped=\d+\))?|FAILED \([a-z=0-9, ]+\)',line)]
  public=('# Validation-summary-only excerpt; all other raw log lines omitted on Spark.\n'+'\n'.join(selected)+'\n').encode()
  policy={'kind':'validation-summary-only','raw_log_transferred':False,'source_lines':len(lines),'retained_lines':len(selected)}
 out[n]={'source_sha256':hashlib.sha256(raw).hexdigest(),'source_bytes':len(raw),'public_sha256':hashlib.sha256(public).hexdigest(),'public_bytes':len(public),'policy':policy,'base64':base64.b64encode(public).decode()}
print(json.dumps(out))
'''.replace('NAMES',repr(names),1)
r=subprocess.run(['bash',str(OLD/'spark-ssh'),'python3','-'],input=remote,text=True,capture_output=True,check=True,timeout=45);data=json.loads(r.stdout);O.mkdir()
for n,v in data.items():
 p=O/n;p.parent.mkdir(parents=True,exist_ok=True);content=base64.b64decode(v.pop('base64'));assert hashlib.sha256(content).hexdigest()==v['public_sha256'];p.write_bytes(content)
(O/'collection-receipt.json').write_text(json.dumps({'files':data,'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'filter_source_file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'private_env_or_backup_opened':False},indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(O),'files':len(data),'redactions':sum(v['policy'].get('redactions',0)for v in data.values())}))
