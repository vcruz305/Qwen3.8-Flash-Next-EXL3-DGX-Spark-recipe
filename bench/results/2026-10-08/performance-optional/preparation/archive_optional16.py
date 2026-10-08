#!/usr/bin/env python3
"""Read-only collection of bounded completed optional16 evidence; no GPU/API work."""
from pathlib import Path
import base64, datetime as dt, gzip, hashlib, json, subprocess
BASE = Path('/home/vcruz/src/qwen-overnight-20261008')
DEST = BASE / 'recipe/bench/results/2026-10-08/performance-optional'
REMOTE = '/home/cruzspark/qwen-overnight-20261008'
if DEST.exists():
    raise SystemExit(f'Refusing existing archive: {DEST}')
remote_code = r'''
from pathlib import Path
import base64, datetime as dt, gzip, hashlib, json
root=Path('/home/cruzspark/qwen-overnight-20261008')
reports=root/'results/performance-optional-16ca-ksplit1'
files=[]; omitted=[]
def add(p, rel):
    if p.is_symlink() or not p.is_file(): raise RuntimeError('Not a regular file: '+str(p))
    raw=p.read_bytes()
    if len(raw)>2_000_000: raise RuntimeError('Unexpectedly large evidence file: '+str(p))
    files.append({'archive_path':rel, 'source_path':str(p), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest(), 'mtime_ns':p.stat().st_mtime_ns, 'data':base64.b64encode(raw).decode()})
for p in sorted(reports.rglob('*')):
    if not p.is_file(): continue
    rel=str(p.relative_to(reports))
    if p.suffix=='.json' or p.name=='server.log' or p.name=='config.yml':
        add(p,'reports/'+rel)
    elif p.suffix=='.log':
        raw=p.read_bytes()
        if len(raw)>2_000_000: raise RuntimeError('Unexpected log size')
        lines=raw.decode('utf-8',errors='replace').splitlines()
        matches=[{'line':i+1,'text':s[:600]} for i,s in enumerate(lines) if any(t in s.lower() for t in ('traceback','exception','error:','warning:'))]
        omitted.append({'source_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'line_count':len(lines),'reason':'Duplicate client JSON/stdout not copied; structured report retained.','diagnostic_lines':matches[:20], 'diagnostic_lines_total':len(matches)})
    elif p.suffix!='.lock':
        raise RuntimeError('Unexpected report file: '+str(p))
for p, rel in [(root/'experiment-jobs-16ca-ksplit1/optional-selected.json','inputs/optional16.json'), (root/'spark_experiment_controller.py','inputs/spark_experiment_controller.py'), (root/'results/performance-optional-16ca-ksplit1-launch.json','inputs/launch.json')]:
    add(p,rel)
add(root/'experiment-jobs-16ca-ksplit1/optional-selected-manifest.json','inputs/optional-selected-manifest.json')
memory=root/'results/memory-optional-unloaded-20261008-corrected'
if not memory.is_dir(): raise RuntimeError('Missing corrected unloaded memory advisory')
for p in sorted(memory.rglob('*')):
    if p.is_file(): add(p,'inputs/memory-advisory/'+str(p.relative_to(memory)))
if sum(x['bytes'] for x in files)>8_000_000: raise RuntimeError('Unexpected archive size')
payload={'collected_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source_root':str(reports),'files':files,'omitted_client_logs':omitted}
print(base64.b64encode(gzip.compress(json.dumps(payload).encode())).decode())
'''
r=subprocess.run(['bash',str(BASE/'spark-ssh'),'python3','-'],input=remote_code,capture_output=True,text=True,check=True,timeout=40)
payload=json.loads(gzip.decompress(base64.b64decode(r.stdout.strip())))
expected = {
 'inputs/optional16.json':'90234a6cdd68b2e2a4b260753d4ed31ab19a0aaa81cf3e00f7270b91c97b7c34',
 'inputs/spark_experiment_controller.py':'48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586',
}
by_path={x['archive_path']:x for x in payload['files']}
for name, sha in expected.items():
    assert by_path[name]['sha256']==sha, (name, by_path[name]['sha256'])
records=[x for x in payload['files'] if x['archive_path'].count('/')==2 and x['archive_path'].endswith('/result.json')]
assert len(records)==16, len(records)
for item in records:
    d=json.loads(base64.b64decode(item['data']))
    assert d['state']=='completed' and d.get('finished_at_utc'), item['archive_path']
    assert not d.get('cleanup_failed') and not d.get('cleanup_error'), item['archive_path']
    assert d['configuration']['sources']['engine']['commit']=='16ca20d27c0e4cce15a9bbc131e6d047065395b5'
    assert d['configuration']['sources']['server']['commit']=='f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6'
DEST.mkdir(parents=True)
for item in payload['files']:
    raw=base64.b64decode(item.pop('data'))
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
    path=DEST/item['archive_path']; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
(DEST/'source-provenance.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
first=json.loads((DEST/records[0]['archive_path']).read_text())
sources=first['configuration']['sources']
source_snapshot=[]
for rel, expected_sha in sources['files_sha256'].items():
    raw=subprocess.run(['git','show',sources['recipe']['commit']+':'+rel],cwd=BASE/'recipe',capture_output=True,check=True).stdout
    assert hashlib.sha256(raw).hexdigest()==expected_sha, rel
    p=DEST/'inputs/recipe'/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
    source_snapshot.append({'archive_path':str(p.relative_to(DEST)),'git_commit':sources['recipe']['commit'],'git_path':rel,'sha256':expected_sha})
(DEST/'inputs/recipe-source.json').write_text(json.dumps(source_snapshot,indent=2,sort_keys=True)+'\n')
ple_source=BASE/'engine-source-recheck-f0/ple-ram-transient-review.json'
ple_bytes=ple_source.read_bytes()
ple_dest=DEST/'inputs/ple-ram-transient-review.json'
ple_dest.write_bytes(ple_bytes)
(DEST/'inputs/ple-ram-transient-provenance.json').write_text(json.dumps({
    'source_path':str(ple_source),'sha256':hashlib.sha256(ple_bytes).hexdigest(),
    'bytes':len(ple_bytes),'scope':'Read-only source placement audit, separate from observed memory samples. The 10 GiB helper slack is an allowance for runtime overhead, not a requirement for 10 GiB residual after loading.'
},indent=2)+'\n')
print(json.dumps({'archive':str(DEST),'copied_remote_files':len(payload['files']),'copied_remote_bytes':sum(x['bytes'] for x in payload['files']),'omitted_duplicate_client_logs':len(payload['omitted_client_logs']),'complete_jobs':len(records),'recipe_commit':sources['recipe']['commit']},indent=2))
