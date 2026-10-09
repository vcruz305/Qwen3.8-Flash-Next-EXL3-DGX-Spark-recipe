#!/usr/bin/env python3
"""Bounded read-only collection of the completed two-cell run; no API or imports."""
import base64,datetime,hashlib,json,shlex,shutil,subprocess
from pathlib import Path
NEW=Path('/home/vcruz/src/qwen-followup-20261009')
OUT=NEW/'publication/2026-10-09/flat305-confirmation'
assert not OUT.exists();OUT.mkdir()
code=r'''import base64,datetime,hashlib,json,stat
from pathlib import Path
b=Path('/home/cruzspark/qwen-followup-20261009');r=b/'results/flat305-confirmation-f6e384ce'
d=json.loads((r/'result.json').read_text());assert d['state']=='completed' and d['passed'] and len(d['cells'])==2
files=[];excluded=[]
for p in r.rglob('*'):
 rel='reports/'+str(p.relative_to(r))
 if p.is_symlink():excluded.append({'source':str(p),'destination':rel,'reason':'model symlink; never followed'});continue
 if p.is_file():files.append((p,rel))
for name in ('before','after'):
 for p in sorted((b/('measurement/flat305-autotune-'+name)).iterdir()):files.append((p,'boundaries/autotune-'+name+'/'+p.name))
for name in ('flat305-confirmation-launch.json',):
 files.append((b/name,'boundaries/'+name))
for name in ('flat305-confirmation-f6e384ce.log',):
 files.append((b/'logs'/name,'boundaries/'+name))
files.append((b/'measurement/model-input-audit/current-models.json','model-input-audit/current-models.json'))
result=[];total=0
for p,rel in files:
 assert not p.is_symlink();a=p.stat();assert stat.S_ISREG(a.st_mode) and a.st_size<4*1024*1024
 raw=p.read_bytes();z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 total+=len(raw);assert total<20*1024*1024
 result.append({'source':str(p),'destination':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'inode':a.st_ino,'mtime_ns':a.st_mtime_ns,'stable_before_after':True,'data':base64.b64encode(raw).decode()})
print(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':result,'exclusions':excluded}))
'''
r=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
data=json.loads(r.stdout)
for row in data['files']:
 raw=base64.b64decode(row.pop('data'));assert hashlib.sha256(raw).hexdigest()==row['sha256']
 p=OUT/row['destination'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
# Preparation artifacts are copied without importing/executing them.
src=NEW/'measurement/flat305-confirmation'
for p in sorted(src.rglob('*')):
 if not p.is_file():continue
 rel='sources/flat305-confirmation/'+str(p.relative_to(src));raw=p.read_bytes()
 info={'source':str(p),'destination':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if '__pycache__' in p.parts or p.suffix=='.pyc':info['reason']='incidental generated Python cache';data['exclusions'].append(info);continue
 assert not p.is_symlink() and len(raw)<4*1024*1024
 q=OUT/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw);data['files'].append(info)
for source,dest in [
 (NEW/'measurement/flat305-confirmation-f6e384ce.tar.gz','sources/flat305-confirmation-f6e384ce.tar.gz'),
 (NEW/'engine-attribution/summarize_two_cell305.py','analysis/summarize_two_cell305.py'),
 (NEW/'engine-attribution/analyze_historical.py','analysis/analyze_historical.py'),
 (NEW/'engine-attribution/historical-attribution.json','analysis/historical-attribution.json'),
 (NEW/'engine-attribution/historical-attribution.md','analysis/historical-attribution.md'),
 (Path(__file__),'collect_completed_305.py')]:
 raw=source.read_bytes();p=OUT/dest;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 data['files'].append({'source':str(source),'destination':dest,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
data['scope']='Completed small-file evidence only; no model weights, tensor deserialization, library binaries, API, GPU or service actions. Stable before/after stats for every remote read.'
(OUT/'collection-receipt.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'out':str(OUT),'copied':len(data['files']),'bytes':sum(x['bytes'] for x in data['files']),'excluded':len(data['exclusions'])}))
