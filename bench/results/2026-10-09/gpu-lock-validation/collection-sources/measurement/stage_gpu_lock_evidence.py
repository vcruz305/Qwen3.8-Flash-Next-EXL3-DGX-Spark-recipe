#!/usr/bin/env python3
"""Local archive of already completed lock gate; no live operations."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009');O=B/'publication/2026-10-09/gpu-lock-validation';SRC=B/'gpu-lock-live-collected';REPO=B/'recipe-gpu-lock';PIN='254b2b03027f25094845dc31f5f87739f3584d2e'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt={}
def copy(p,q):
 assert p.is_file() and not p.is_symlink();h=sha(p);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);assert sha(q)==h==sha(p)
 receipt[str(q.relative_to(O))]={'source':str(p),'bytes':q.stat().st_size,'sha256':h}
assert not O.exists();O.mkdir(parents=True)
for p in SRC.rglob('*'):
 if p.is_file() and not p.is_symlink():copy(p,O/p.relative_to(SRC))
for dirname in ['gpu-lock-review','measurement/gpu-lock-matrix-review']:
 for p in (B/dirname).rglob('*'):
  if p.is_file() and not p.is_symlink():copy(p,O/'cpu-review'/dirname/p.relative_to(B/dirname))
for name in ['measurement/gpu-lock-live/jobs.json','measurement/collect_gpu_lock_evidence.py','measurement/review_small_archive.py','measurement/stage_gpu_lock_evidence.py','literal-archive-release-review/review_archive.py']:
 copy(B/name,O/'collection-sources'/name)
files=['AGENTS.md','exllamav3-tabby/env.sh','exllamav3-tabby/serve.sh','exllamav3-tabby/setup.sh','exllamav3-tabby/tabby-config.yml','exllamav3-tabby/tools/runtime_state.py','exllamav3-tabby/tools/model_memory.py','bench/run_matrix.py','bench/test_run_matrix.py','bench/test_recipe_gpu_lock.py','bench/tool_smoke.py','bench/api_client.py','bench/bench_v1.py','bench/concurrency.py','bench/matrix.md','docs/service.md']
gitfiles={}
for rel in files:
 raw=subprocess.check_output(['git','-C',str(REPO),'show',PIN+':'+rel]);dst=O/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw);gitfiles[rel]={'bytes':len(raw),'sha256':sha(dst)}
r=json.loads((O/'raw/results/gpu-lock-live-254b2b0/gpu-lock-live-305-concurrent/result.json').read_text());p=next((O/'raw').rglob('tools.json'));t=json.loads(p.read_text())
assert r['passed'] and r['state']=='completed' and t['summary']=={'passed':28,'failed':0,'total':28} and len(t['results'])==28
assert r['configuration']['sources']['recipe']['commit']==PIN
for key,h in r['configuration']['sources']['files_sha256'].items():assert gitfiles[key]['sha256']==h
assert gitfiles['bench/run_matrix.py']['sha256']==r['configuration']['sources']['controller_sha256']
release=json.loads((O/'raw/results/gpu-lock-live-release.json').read_text());launch=json.loads((O/'raw/gpu-lock-live-launch.json').read_text())
assert release['passed'] and release['shared_lock_reacquired'] and release['controller_gone']
assert release['lock_after']==launch['gpu_lock_before']
summary={'recipe':PIN,'engine':r['configuration']['sources']['engine']['commit'],'tabby':r['configuration']['sources']['server']['commit'],'passed':True,'tool_checks':t['summary'],'cpu_unittest_report':'Ran 148 tests in 1.894s; OK (skipped=5)','cpu_exit_code':0,'setup_check_exit_code':0,'lock_before':r['gpu_lock_before_measurements'],'lock_after_measurements':r['gpu_lock_after_measurements'],'server_cleanup':r['server_cleanup'],'release':release,'scope':'One owned GPU load and serialized tools28; no multi-process race stress or performance claim. Lock is cooperative.'}
(O/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
(O/'README.md').write_text("""# Cooperative GPU lock validation — 9 October 2026

Recipe candidate `254b2b03027f25094845dc31f5f87739f3584d2e` completed one live GPU load with `GPU_LOCK_FILE=/home/cruzspark/redsnow-gpu.lock`, then passed all **28 unchanged tool smoke checks**. The parent matrix controller did not hold the shared GPU lock; the server launcher acquired it and retained descriptor 8 across exec.

The controller checked the actual running server descriptor and Linux `fdinfo` before clients and after measurements. Both checks identify device 66306, inode 2273157 and an advisory write flock. The later release receipt verifies the controller had exited, no GPU owner remained, and the same file could be locked again. Its device/inode/empty contents remained unchanged. The owned server received SIGTERM and exited 0.

The recipe's optional setting defaults to unset. When requested, the launcher refuses a competing cooperative owner before runtime/model/config mutations; the public matrix requires launcher capability and validates actual descriptor ownership, rather than accepting only an environment variable. CPU checks exercise contention, exec lifetime, invalid paths, preview behavior, lock replacement and a selected historical launcher without the capability. The retained independent matrix review records 26 passing checks. The full ARM recipe test output reports **Ran 148 tests; OK (skipped=5)**, and `setup.sh --check` exited 0. We retain the runner's own count wording instead of subtracting skip events into an invented passed count.

The live job used the flat 3.05 model, disk PLE, max batch 4, a 262144-token shared cache, Q8 KV, chunk 2048 and dynamic depth-5 MTP with row budget 8, on engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` and Tabby `f650bb5389e0a273549e47d4d26a765760c013e1`. Its 28 client requests were sequential; the profile's four slots do not make this a concurrent throughput experiment. This gate did not replace the canonical published recipe or enable/disable services.

This is a **cooperative ownership mechanism**, not an access-control or GPU-isolation boundary: other software must honor the same persistent lock file. Do not remove/recreate the file while cooperating processes may hold it. The live gate is not a stress test of all possible process races.

## Evidence

- `raw/` contains exact small Spark result/client/resource/config/deployment files, preflight logs, launch record and release proof. Model-view symlinks are recorded as omissions and never followed.
- `source/` contains selected files read directly from Git at the exact tested recipe commit. The controller and recorded runtime/client file hashes match the live result.
- `cpu-review/` preserves earlier local checks and the independent integration review, each with its own source scope.
- `remote-source-manifest.json`, `release-review.json`, `collection-receipt.json` and `SHA256SUMS` bind collection and source identities.
- `summary.json` separates live clients, CPU preflight and release checks.

All copied remote data passed a bounded credential-pattern/key-name scan with zero candidates. The scan does not prove absence of arbitrary unlabeled secrets. CPU source fixtures may contain deliberate fake API-key strings; no credential store was opened. Collection copied only completed small files and performed no API, GPU, service, source or model mutation. Earlier preparation documents remain unchanged, including pending-run wording. Do not automatically execute archived host-specific scripts. Verify file integrity with `sha256sum -c SHA256SUMS`.
""")
(O/'collection-receipt.json').write_text(json.dumps({'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_copies':receipt,'git_source':{'repository':str(REPO),'commit':PIN,'files':gitfiles},'scope':'Read-only collection and local packaging only; no inference/lifecycle/source mutation.'},indent=2,sort_keys=True)+'\n')
for rel,x in receipt.items():assert sha(O/rel)==x['sha256']==sha(Path(x['source']))
paths=sorted(p for p in O.rglob('*') if p.is_file());assert not any(p.is_symlink() for p in O.rglob('*'))
(O/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(O))+'\n' for p in paths))
print(json.dumps({'archive':str(O),'files':len(paths)+1,'bytes':sum(p.stat().st_size for p in O.rglob('*') if p.is_file()),'manifest_sha256':sha(O/'SHA256SUMS'),'summary_sha256':sha(O/'summary.json')}))
