#!/usr/bin/env python3
"""Read-only small completed lock-test evidence collection."""
import base64,datetime,hashlib,json,subprocess
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009');OUT=B/'gpu-lock-live-collected'
CODE=r"""
from pathlib import Path
import os,stat,hashlib,base64,json,datetime
b=Path('/home/cruzspark/qwen-followup-20261009')
job=b/'results/gpu-lock-live-254b2b0/gpu-lock-live-305-concurrent/result.json'
d=json.loads(job.read_text());assert d['state']=='completed' and d['passed'] and d['finished_at_utc']
release=json.loads((b/'results/gpu-lock-live-release.json').read_text())
assert release['passed'] and release['source_result_sha256']==hashlib.sha256(job.read_bytes()).hexdigest()
files={};omitted=[];total=0
roots=['results/gpu-lock-live-254b2b0','results/gpu-lock-live-preflight']
paths=[]
for rel in roots:
 for parent,dirs,names in os.walk(b/rel,followlinks=False):
  for name in list(dirs):
   p=Path(parent)/name
   if p.is_symlink():
    omitted.append({'path':str(p.relative_to(b)),'reason':'symlink excluded without following','target':os.readlink(p)});dirs.remove(name)
  paths.extend(Path(parent)/name for name in names)
paths.extend(b/name for name in ['results/gpu-lock-live-release.json','gpu-lock-live-launch.json','logs/gpu-lock-live-254b2b0.log'])
for p in sorted(paths):
 st=p.lstat();rel=str(p.relative_to(b))
 if stat.S_ISLNK(st.st_mode):omitted.append({'path':rel,'reason':'symlink excluded without following','target':os.readlink(p)});continue
 assert stat.S_ISREG(st.st_mode) and st.st_size<10*1024*1024
 raw=p.read_bytes();after=p.lstat();assert (st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 total+=len(raw);assert total<10*1024*1024
 files[rel]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps({'source':str(b),'files':files,'file_count':len(files),'bytes':total,'omitted':omitted,'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
"""
assert not OUT.exists()
r=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3','-'],input=CODE,text=True,capture_output=True,check=True,timeout=30)
d=json.loads(r.stdout);OUT.mkdir()
for name,v in d['files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 raw=base64.b64decode(v.pop('base64'));assert hashlib.sha256(raw).hexdigest()==v['sha256'] and len(raw)==v['bytes']
 p=OUT/'raw'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
d['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
p=OUT/'remote-source-manifest.json';p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(OUT),'files':d['file_count'],'bytes':d['bytes'],'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
