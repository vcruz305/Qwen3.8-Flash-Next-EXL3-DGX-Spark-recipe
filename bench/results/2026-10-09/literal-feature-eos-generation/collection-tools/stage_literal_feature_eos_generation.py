#!/usr/bin/env python3
"""Freeze/copy only the reviewed generation-only diagnostic; never execute on Spark."""
import base64,datetime,hashlib,io,json,subprocess,tarfile
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009')
root=B/'measurement/literal-feature-eos-generation'
observer=B/'literal-feature-eos-observer'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'controller.py': 'b72dbcd35ce43985a5ebf00ac6c9296e6155d5985da7e44b87a1b7ebbe2e8ec5', 'input_binding.py': '687ba31ff103415a49f3474d13a9aee6734a8405b9dc063dc384a0f05bb1209c', 'plan.json': 'a980ffc3070f578faacaed3ec20768a1781903173472d75b6861d416ef70b348', 'prepare_inputs.py': '3ddd44cef58edae20b6fb5ef2dcdf6a754598dac8bf7815ace69b091a1c1c932', 'prepared-inputs.json': '6a47b8afa81e387ab8d07aeca30d78afc461f646b6671bfebed05130797afae3'}
assert all(sha(root/p)==h for p,h in expected.items())
obs_sha='a45a4684524f2eb45614ae8f70808337c90c78bce7b0f7992fbde2c6bea7b865'
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
manifest={'schema_version':1,'state':'frozen_prepared_not_executed','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'main_sources':expected,'observer_manifest_sha256':obs_sha,'recipe_commit':'254b2b03027f25094845dc31f5f87739f3584d2e','engine_commit':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby_commit':'f4aadf114b0044fa8cbe1b50241dc80ea7d61583','packages_sha256':'4f71ec141cedfa82781135669cb3492ad1547ace7dca61d370fc2acc9fa0210a','planned_chat_posts':8,'coverage_requests':0,'generation_observation_requests':8,'request_order':'Unchanged generation8 only; last4 concurrent','diagnostic_observer':'Only the server receives the bounded observer; no input/output encoding changes.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
included=[root/p for p in files]+[root/'manifest.json']
(root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(included)))
included.append(root/'SHA256SUMS')
included += [observer/p for p in obs['files']]+[observer/'manifest.json']
if (observer/'SHA256SUMS').exists():included.append(observer/'SHA256SUMS')
allfiles={str(p.relative_to(B)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in included}
tarpath=B/'measurement/literal-feature-eos-generation-b72dbcd3.tar.gz';assert not tarpath.exists()
with tarfile.open(tarpath,'w:gz') as tar:
 for p in sorted(included):tar.add(p,arcname=str(p.relative_to(B)),recursive=False)
raw=tarpath.read_bytes();tarsha=hashlib.sha256(raw).hexdigest()
remote="""import base64,hashlib,io,json,tarfile
from pathlib import Path
raw=base64.b64decode(PAYLOAD);assert hashlib.sha256(raw).hexdigest()==TARSHA
base=Path('/home/cruzspark/qwen-followup-20261009');expected=EXPECTED
assert not (base/'measurement/literal-feature-eos-generation').exists()
assert not (base/'literal-feature-eos-observer').exists()
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for m in tar.getmembers():
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.size<20*1024*1024
  files[m.name]=tar.extractfile(m).read()
assert set(files)==set(expected)
for name,data in files.items():assert hashlib.sha256(data).hexdigest()==expected[name]['sha256'] and len(data)==expected[name]['bytes']
for name,data in files.items():
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print(json.dumps({'staged':[str(base/'measurement/literal-feature-eos-generation'),str(base/'literal-feature-eos-observer')],'files':len(files),'bytes':sum(map(len,files.values())),'tar_sha256':TARSHA,'executed':False}))
""".replace('PAYLOAD',repr(base64.b64encode(raw).decode())).replace('TARSHA',repr(tarsha)).replace('EXPECTED',repr(allfiles))
result=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3','-'],input=remote,text=True,capture_output=True,check=True,timeout=45)
receipt=json.loads(result.stdout);receipt.update(local_bundle=str(tarpath),manifest_sha256=sha(root/'manifest.json'),sha256sums_sha256=sha(root/'SHA256SUMS'),observer_manifest_sha256=obs_sha)
(B/'measurement/literal-feature-eos-generation-stage-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
