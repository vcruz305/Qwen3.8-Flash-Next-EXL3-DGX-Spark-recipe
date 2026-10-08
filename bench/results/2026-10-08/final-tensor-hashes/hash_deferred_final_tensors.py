#!/usr/bin/env python3
"""Bounded post-benchmark SHA256 reads of six already-produced numerical artifacts."""
from pathlib import Path
import datetime as dt,hashlib,json,os,stat,time
ROOT=Path('/home/cruzspark/qwen-overnight-20261008')
BATCH_PID=430098
DEST=ROOT/'results/tensor-hashes-final-24f0-post-bench'
INVENTORY_SHA='307bb1fef53d1a8c3f09ab40e6931adbfd70f294aff5deced4ecbb7e25fde4cc'
RAW_INVENTORY='{"recorded_at_utc": "2026-10-08T13:43:51.340847+00:00", "files": [{"path": "/home/cruzspark/qwen-overnight-20261008/results/quality-q8-16ca-ksplit1-flat-cyber/candidate-305-ksplit1/logits.safetensors", "bytes": 643789840, "mtime_ns": 1791456143180297033, "device": 66306, "inode": 47073384, "input_json": "/home/cruzspark/qwen-overnight-20261008/results/quality-q8-16ca-ksplit1-flat-cyber/candidate-305-ksplit1/input.json", "name": "candidate-305-ksplit1", "engine_sha": "16ca20d27c0e4cce15a9bbc131e6d047065395b5", "model": "/home/cruzspark/models/flashnext-exl3-3.05bpw"}, {"path": "/home/cruzspark/qwen-overnight-20261008/results/quality-q8-16ca-ksplit1-flat-cyber/candidate-405-ksplit1/logits.safetensors", "bytes": 643789840, "mtime_ns": 1791456287647719893, "device": 66306, "inode": 47073391, "input_json": "/home/cruzspark/qwen-overnight-20261008/results/quality-q8-16ca-ksplit1-flat-cyber/candidate-405-ksplit1/input.json", "name": "candidate-405-ksplit1", "engine_sha": "16ca20d27c0e4cce15a9bbc131e6d047065395b5", "model": "/home/cruzspark/models/flashnext-exl3-4.05bpw"}, {"path": "/home/cruzspark/qwen-overnight-20261008/results/quality-q8-16ca-ksplit1-flat-cyber/candidate-cyber387-ksplit1/logits.safetensors", "bytes": 643789848, "mtime_ns": 1791456406006239612, "device": 66306, "inode": 47073398, "input_json": "/home/cruzspark/qwen-overnight-20261008/results/quality-q8-16ca-ksplit1-flat-cyber/candidate-cyber387-ksplit1/input.json", "name": "candidate-cyber387-ksplit1", "engine_sha": "16ca20d27c0e4cce15a9bbc131e6d047065395b5", "model": "/home/cruzspark/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw"}, {"path": "/home/cruzspark/qwen-overnight-20261008/results/quality-final-chunk4096/candidate-305-chunk4096-final/logits.safetensors", "bytes": 643789840, "mtime_ns": 1791464546887672178, "device": 66306, "inode": 47075222, "input_json": "/home/cruzspark/qwen-overnight-20261008/results/quality-final-chunk4096/candidate-305-chunk4096-final/input.json", "name": "candidate-305-chunk4096-final", "engine_sha": "24f0dece34f09c8d1e2359d6b3b3f7befef7331b", "model": "/home/cruzspark/models/flashnext-exl3-3.05bpw"}, {"path": "/home/cruzspark/qwen-overnight-20261008/results/quality-final-chunk2048-pair/baseline-305-chunk2048-final-pair/logits.safetensors", "bytes": 3218905080, "mtime_ns": 1791465666948850915, "device": 66306, "inode": 47075337, "input_json": "/home/cruzspark/qwen-overnight-20261008/results/quality-final-chunk2048-pair/baseline-305-chunk2048-final-pair/input.json", "name": "baseline-305-chunk2048-final-pair", "engine_sha": "94ba01d50a13fa9ff672473f2d0eef8b51a71e99", "model": "/home/cruzspark/models/flashnext-exl3-3.05bpw"}, {"path": "/home/cruzspark/qwen-overnight-20261008/results/quality-final-chunk2048-pair/candidate-305-chunk2048-final-pair/logits.safetensors", "bytes": 3218905088, "mtime_ns": 1791466020866028943, "device": 66306, "inode": 47075388, "input_json": "/home/cruzspark/qwen-overnight-20261008/results/quality-final-chunk2048-pair/candidate-305-chunk2048-final-pair/input.json", "name": "candidate-305-chunk2048-final-pair", "engine_sha": "24f0dece34f09c8d1e2359d6b3b3f7befef7331b", "model": "/home/cruzspark/models/flashnext-exl3-3.05bpw"}], "total_bytes": 9012969536, "total_gib": 8.393981993198395}\n'
EXPECTED_LABELS={'final-305-single','final-405-single','final-415-single','final-cyber387-single','final-305-concurrent'}
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def absent(pid):
 assert type(pid) is int and pid>1
 assert not Path('/proc',str(pid)).exists(),f'Owned process still exists: {pid}'
def fingerprint(st):
 return {'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'device':st.st_dev,'inode':st.st_ino}
def checked_stat(path,item):
 st=path.lstat();assert stat.S_ISREG(st.st_mode) and not path.is_symlink(),str(path)
 assert fingerprint(st)=={k:item[k] for k in ('bytes','mtime_ns','device','inode')},str(path)
 return st
assert hashlib.sha256(RAW_INVENTORY.encode()).hexdigest()==INVENTORY_SHA
inventory=json.loads(RAW_INVENTORY)
assert len(inventory['files'])==6 and sum(i['bytes'] for i in inventory['files'])==9012969536
assert len({i['path'] for i in inventory['files']})==6
status_path=ROOT/'results/final-api-24f0-5a/status.json'
status=json.loads(status_path.read_text())
assert status['state']=='completed' and status.get('active') is None and status.get('finished_utc')
assert {j['label'] for j in status['jobs']}==EXPECTED_LABELS and len(status['jobs'])==5
assert all(j.get('finished_utc') and j.get('exit_code') is not None for j in status['jobs'])
absent(BATCH_PID)
clients_checked=[]
for j in status['jobs']:
 result_path=status_path.parent/j['label']/'result.json'
 assert sha(result_path)==j['result_sha256']
 result=json.loads(result_path.read_text())
 assert result.get('finished_at_utc') and result['server_cleanup']['owned_group_empty'] is True
 if result.get('server_pid'):absent(result['server_pid'])
 expected={'bench'}
 if j['label']=='final-305-concurrent':expected.update({'concurrency-bench','long-32768','long-240000'})
 found=[]
 for c in result['clients']:
  if c['name'] not in expected:continue
  assert c.get('finished_at_utc') and c.get('exit_code') is not None
  assert c['cleanup']['owned_group_empty'] is True
  absent(c['client_pid']);found.append(c['name'])
  clients_checked.append({'label':j['label'],'client':c['name'],'pid':c['client_pid'],'finished_at_utc':c['finished_at_utc'],'owned_group_empty':True})
 assert set(found)==expected and len(found)==len(expected)
for item in inventory['files']:
 p=Path(item['path']);assert p.is_relative_to(ROOT/'results') and p.name=='logits.safetensors'
 checked_stat(p,item)
DEST.mkdir(mode=0o700)
record={'schema_version':1,'state':'hashing','released_at_utc':now(),'inventory_sha256':INVENTORY_SHA,'batch_pid_absent':BATCH_PID,'batch_status_sha256':sha(status_path),'batch_finished_utc':status['finished_utc'],'completed_jobs':5,'verified_finished_measurement_processes':clients_checked,'method':'Sequential16MiB byte reads through hashlib.sha256; no safetensors/torch import or deserialization. Original size/mtime/device/inode checked before and after each read.','files':[]}
def save():
 temporary=DEST/'hashes.tmp';temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(DEST/'hashes.json')
save()
try:
 for item in inventory['files']:
  p=Path(item['path']);before=checked_stat(p,item);h=hashlib.sha256();count=0;started=time.monotonic()
  record['active']=str(p);save()
  with p.open('rb',buffering=0) as f:
   assert fingerprint(os.fstat(f.fileno()))==fingerprint(before)
   while True:
    chunk=f.read(16*1024*1024)
    if not chunk:break
    h.update(chunk);count+=len(chunk)
   assert fingerprint(os.fstat(f.fileno()))==fingerprint(before)
  after=checked_stat(p,item);assert count==item['bytes'] and fingerprint(after)==fingerprint(before)
  record['files'].append({**item,'sha256':h.hexdigest(),'bytes_read':count,'metadata_unchanged_before_after':True,'hash_seconds':time.monotonic()-started});save()
 record.update(state='completed',active=None,finished_at_utc=now(),total_bytes_read=sum(f['bytes_read'] for f in record['files']));save()
 print(json.dumps({'state':record['state'],'released_at_utc':record['released_at_utc'],'finished_at_utc':record['finished_at_utc'],'files':len(record['files']),'bytes':record['total_bytes_read'],'record':str(DEST/'hashes.json'),'record_sha256':sha(DEST/'hashes.json')}),flush=True)
except BaseException as exc:
 record.update(state='failed',error=type(exc).__name__+': '+str(exc),finished_at_utc=now());save();raise
