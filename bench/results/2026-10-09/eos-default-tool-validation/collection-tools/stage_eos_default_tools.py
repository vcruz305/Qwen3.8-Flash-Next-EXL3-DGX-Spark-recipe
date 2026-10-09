#!/usr/bin/env python3
"""Freeze and file-only stage reviewed EOS tool regression; never start a model."""
import base64,datetime,hashlib,json,subprocess,tarfile
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009');root=B/'measurement/eos-default-tool-validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'controller.py')=='659c6451f70f7a05e74ae1f9560cc40f9764982ccdd84b2503e132b28bc9e5b6'
assert sha(root/'runtime-contract.json')=='9825b80310e39b3656e314b39076109c1231857e74f20469a65d1d4c8a8ac4cb'
assert 'Ran 18 tests' in (root/'harness-cpu.log').read_text() and (root/'harness-cpu.log').read_text().rstrip().endswith('OK')
files={}
for p in sorted(root.rglob('*')):
 assert not p.is_symlink()
 if p.is_file() and p.suffix!='.pyc' and p.name not in ('manifest.json','SHA256SUMS'):
  files[str(p.relative_to(root))]={'sha256':sha(p),'bytes':p.stat().st_size}
manifest={'schema_version':1,'state':'frozen_prepared_not_executed','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'recipe_commit':'254b2b03027f25094845dc31f5f87739f3584d2e','engine_commit':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby_commit':'fd8cbeb1fbb3cec4d2141558d5cfd63a500d50c4','packages_sha256':'4f71ec141cedfa82781135669cb3492ad1547ace7dca61d370fc2acc9fa0210a','planned_chat_posts':32,'tool_checks':28,'diagnostic_observers':False,'experimental_literal_input_feature':False,'cpu_tests':18,'cpu_initial_fixture_failures_preserved':2}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
included=[root/n for n in files]+[root/'manifest.json']
(root/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in sorted(included)));included.append(root/'SHA256SUMS')
expected={str(p.relative_to(B)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in included}
tarpath=B/'measurement/eos-default-tool-validation-659c6451.tar.gz';assert not tarpath.exists()
with tarfile.open(tarpath,'w:gz') as tar:
 for p in sorted(included):tar.add(p,arcname=str(p.relative_to(B)),recursive=False)
raw=tarpath.read_bytes();tarsha=hashlib.sha256(raw).hexdigest()
remote="""import base64,hashlib,io,json,tarfile
from pathlib import Path
raw=base64.b64decode(PAYLOAD);assert hashlib.sha256(raw).hexdigest()==TARSHA
base=Path('/home/cruzspark/qwen-followup-20261009');expected=EXPECTED
assert not (base/'measurement/eos-default-tool-validation').exists()
files={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 for m in tar.getmembers():
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts and m.size<20*1024*1024
  files[m.name]=tar.extractfile(m).read()
assert set(files)==set(expected)
for name,data in files.items():assert hashlib.sha256(data).hexdigest()==expected[name]['sha256'] and len(data)==expected[name]['bytes']
for name,data in files.items():
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
print(json.dumps({'staged':str(base/'measurement/eos-default-tool-validation'),'files':len(files),'bytes':sum(map(len,files.values())),'tar_sha256':TARSHA,'executed':False}))
""".replace('PAYLOAD',repr(base64.b64encode(raw).decode())).replace('TARSHA',repr(tarsha)).replace('EXPECTED',repr(expected))
r=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3','-'],input=remote,text=True,capture_output=True,check=True,timeout=45)
receipt=json.loads(r.stdout);receipt.update(local_bundle=str(tarpath),manifest_sha256=sha(root/'manifest.json'),sha256sums_sha256=sha(root/'SHA256SUMS'))
(B/'measurement/eos-default-tool-validation-stage-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2))
