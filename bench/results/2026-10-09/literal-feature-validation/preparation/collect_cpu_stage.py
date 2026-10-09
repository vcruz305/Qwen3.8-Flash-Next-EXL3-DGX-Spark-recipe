#!/usr/bin/env python3
"""Collect only completed CPU/source evidence; final live publication remains separate."""
import base64,datetime,hashlib,json,shlex,subprocess
from pathlib import Path
NEW=Path('/home/vcruz/src/qwen-followup-20261009')
OUT=NEW/'literal-feature-publication-staging'
assert not OUT.exists();OUT.mkdir()
code=r'''import base64,datetime,hashlib,json,stat
from pathlib import Path
b=Path('/home/cruzspark/qwen-followup-20261009');r=b/'results/literal-candidate-cpu-a70'
status=json.loads((r/'result.json').read_text())
files=[(r/name,'cpu-spark/'+name) for name in ('result.json','launch.json','run_cpu.py','controller.log','pytest.log','setup-check.log','packages.log')]
files.append((b/'literal-candidate-stage.json','source/literal-candidate-stage.json'))
out=[];total=0
for p,rel in files:
 assert not p.is_symlink();a=p.stat();assert stat.S_ISREG(a.st_mode) and a.st_size<4*1024*1024
 raw=p.read_bytes();z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 total+=len(raw);assert total<16*1024*1024
 out.append({'source':str(p),'destination':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'inode':a.st_ino,'mtime_ns':a.st_mtime_ns,'stable_before_after':True,'data':base64.b64encode(raw).decode()})
print(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':out,'exclusions':[],'remote_status':status}))
'''
r=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3 -c '+shlex.quote(code)],text=True,capture_output=True,check=True)
data=json.loads(r.stdout)
for row in data['files']:
 raw=base64.b64decode(row.pop('data'));assert hashlib.sha256(raw).hexdigest()==row['sha256']
 p=OUT/row['destination'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
def copy(source,dest):
 raw=source.read_bytes();assert not source.is_symlink() and len(raw)<4*1024*1024
 p=OUT/dest;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 data['files'].append({'source':str(source),'destination':dest,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
binding=json.loads((NEW/'literal-feature-a70-bindings.json').read_text())
for name,h in binding['evidence_files'].items():
 p=NEW/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 copy(p,'cpu-wsl/'+name)
for name in ['literal-feature-a70-bindings.json','literal-feature-final-peer-source-binding.json','literal-feature-final-peer-focused.log','literal-feature-final-peer-qwen.log','literal-feature-final-peer-qwen-launch.json']:
 copy(NEW/name,'cpu-wsl/'+name)
for rel,h in binding['source_files'].items():
 p=NEW/'tabby-literal-agent'/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 copy(p,'source/a70/'+rel)
git=NEW/'tabby-literal-agent'
assert subprocess.check_output(['git','-C',str(git),'rev-parse','HEAD'],text=True).strip()==binding['commit']
assert not subprocess.check_output(['git','-C',str(git),'status','--porcelain','--untracked-files=no'],text=True)
patch=subprocess.check_output(['git','-C',str(git),'show','--format=fuller','--binary',binding['commit']])
p=OUT/'source/a70.patch';p.write_bytes(patch)
data['generated_source_patch']={'command':'git show --format=fuller --binary '+binding['commit'],'sha256':hashlib.sha256(patch).hexdigest()}
copy(Path(__file__),'collect_cpu_stage.py')
data['scope']='Completed CPU/source proof staged outside final publication sibling. No live inference qualification is claimed; no model weights/library binaries/tensors/API/GPU operations.'
(OUT/'collection-receipt.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'staging':str(OUT),'files':len(data['files']),'bytes':sum(r['bytes'] for r in data['files']),'remote_status':data['remote_status']}))
