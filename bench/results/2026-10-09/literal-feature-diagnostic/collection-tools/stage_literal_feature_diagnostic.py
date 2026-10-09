#!/usr/bin/env python3
"""Freeze/copy only the reviewed combined diagnostic; never execute on Spark."""
import base64,datetime,hashlib,io,json,subprocess,tarfile
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009')
root=B/'measurement/literal-feature-diagnostic'
observer=B/'literal-feature-diagnostic-observer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'controller.py':'089af7736daab1a1eddd35c60451013ed8fb9562bd93ad977927fce5f82b06ba','coverage_client.py':'61efb1baf98d1d21468e06d3e5a702cf4b52deda6825d634ae75032399acd2df','plan.json':'fd6da1e76e9f7e3ab635ca6e37b4c653c7eaf23f8afcee88f461088cb507f91a','prepare_inputs.py':'81fd86e7de09865de218c226aaa89c861a77c833d8505c039bafff8daf45a018','prepared-inputs.json':'b210985b6c747542753008bd70d40c4195e8af5b03d365415b65c41f96f626ea'}
assert all(sha(root/p)==h for p,h in expected.items())
obs_sha='a496afc207c4f09cdd14ced7a7e713dc402cac4cd875f8ba0cb15fcc5236cfcd'
assert sha(observer/'manifest.json')==obs_sha
obs=json.loads((observer/'manifest.json').read_text())
for name,item in obs['files'].items():
 p=observer/name
 assert not p.is_symlink() and sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
files={}
for p in sorted(root.rglob('*')):
 if p.is_symlink():raise ValueError('Unexpected symlink')
 if p.is_file() and p.suffix!='.pyc' and p.name not in ('manifest.json','SHA256SUMS'):
  files[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size}
manifest={'schema_version':1,'state':'frozen_prepared_not_executed','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'main_sources':expected,'observer_manifest_sha256':obs_sha,'recipe_commit':'254b2b03027f25094845dc31f5f87739f3584d2e','engine_commit':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby_commit':'a70ae1fa9e457e478c3d96bdc84012a3cb331796','packages_sha256':'4f71ec141cedfa82781135669cb3492ad1547ace7dca61d370fc2acc9fa0210a','planned_chat_posts':16,'coverage_requests':8,'generation_observation_requests':8,'request_order':'Coverage8 first, unchanged generation8 second; last4 concurrent','diagnostic_observer':'Only the server receives the bounded observer; no input/output encoding changes.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
included=[root/p for p in files]+[root/'manifest.json']
(root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(included)))
included.append(root/'SHA256SUMS')
included += [observer/p for p in obs['files']]+[observer/'manifest.json']
if (observer/'SHA256SUMS').exists():included.append(observer/'SHA256SUMS')
allfiles={str(p.relative_to(B)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in included}
tarpath=B/'measurement/literal-feature-diagnostic-089af773.tar.gz';assert not tarpath.exists()
with tarfile.open(tarpath,'w:gz') as tar:
 for p in sorted(included):tar.add(p,arcname=str(p.relative_to(B)),recursive=False)
raw=tarpath.read_bytes();tarsha=hashlib.sha256(raw).hexdigest()
remote="""import base64,hashlib,io,json,tarfile
from pathlib import Path
raw=base64.b64decode(PAYLOAD);assert hashlib.sha256(raw).hexdigest()==TARSHA
base=Path('/home/cruzspark/qwen-followup-20261009');expected=EXPECTED
assert not (base/'measurement/literal-feature-diagnostic').exists()
assert not (base/'literal-feature-diagnostic-observer').exists()
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for m in tar.getmembers():
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.size<20*1024*1024
  files[m.name]=tar.extractfile(m).read()
assert set(files)==set(expected)
for name,data in files.items():assert hashlib.sha256(data).hexdigest()==expected[name]['sha256'] and len(data)==expected[name]['bytes']
for name,data in files.items():
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print(json.dumps({'staged':[str(base/'measurement/literal-feature-diagnostic'),str(base/'literal-feature-diagnostic-observer')],'files':len(files),'bytes':sum(map(len,files.values())),'tar_sha256':TARSHA,'executed':False}))
""".replace('PAYLOAD',repr(base64.b64encode(raw).decode())).replace('TARSHA',repr(tarsha)).replace('EXPECTED',repr(allfiles))
result=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3','-'],input=remote,text=True,capture_output=True,check=True,timeout=45)
receipt=json.loads(result.stdout);receipt.update(local_bundle=str(tarpath),manifest_sha256=sha(root/'manifest.json'),sha256sums_sha256=sha(root/'SHA256SUMS'),observer_manifest_sha256=obs_sha)
(B/'measurement/literal-feature-diagnostic-stage-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
