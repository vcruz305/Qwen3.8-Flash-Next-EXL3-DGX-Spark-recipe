#!/usr/bin/env python3
"""Archive completed EOS-only tool28 evidence without importing historical helpers."""
from pathlib import Path
import datetime,hashlib,json
B=Path('/home/vcruz/src/qwen-followup-20261009');C=B/'eos-default-tool28-collected';H=B/'measurement/eos-default-tool-validation';O=B/'publication/2026-10-09/eos-default-tool-validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert not O.exists();copies={}
def cp(src,dst):
 assert src.is_file() and not src.is_symlink() and src.stat().st_size<10*1024*1024
 dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();dst.write_bytes(src.read_bytes());copies[str(dst.relative_to(O))]={'source':str(src),'sha256':sha(dst),'bytes':dst.stat().st_size}
top=read(C/'raw/result.json');assert sha(C/'raw/result.json')=='607c06a8aeef7b4e77b7739ee39944e7bfdbd0604022bb97dab57befad392edd'
assert top['state']=='completed' and top['passed'] and top['finished_at_utc'] and top['actual_checks']==28 and top['actual_chat_posts']==32
report_path=next((C/'raw').rglob('tools.json'));report=read(report_path);assert report['summary']=={'passed':28,'failed':0,'total':28} and len(report['results'])==28 and all(r['passed']for r in report['results']) and not report['errors']
posts=[]
for r in report['results']:
 posts.append(r['request'])
 for name in ('followup','repeated_turn'):
  if name in r:posts.append(r[name]['request'])
assert len(posts)==32 and all('literal_user_control_tokens' not in p for p in posts)
release=read(C/'launch-release/eos-default-tool28-fd8-release.json');assert release['controller_result_sha256']==sha(C/'raw/result.json') and all(not r['exists']for r in release['processes'].values()) and release['lock']['reacquired'] and not release['gpu_compute_processes']
scan=read(C/'release-review.json');assert scan['review_passed'] and scan['candidate_count']==0
m=read(C/'remote-source-manifest.json')
for n,v in m['files'].items():assert sha(C/'raw'/n)==v['sha256'];cp(C/'raw'/n,O/'reports'/n)
assert sha(H/'manifest.json')=='319281e5a445bad3202ba5d82d9a7c9708a6ac66ed0c04a0410fc881b1d0587d'
for n,v in read(H/'manifest.json')['files'].items():assert sha(H/n)==v['sha256'];cp(H/n,O/'sources/harness'/n)
for n in ['manifest.json','SHA256SUMS']:cp(H/n,O/'sources/harness'/n)
for p in sorted((C/'launch-release').iterdir()):cp(p,O/'collection/launch-release'/p.name)
for n in ['remote-source-manifest.json','release-review.json']:cp(C/n,O/'collection'/n)
for n in ['collect_completed_small.py','review_small_archive.py']:cp(B/'literal-feature-diagnostic-publication-staging/collection-tools'/n,O/'collection-tools'/n)
cp(B/'literal-archive-release-review/review_archive.py',O/'collection-tools/pattern-source-review_archive.py');cp(B/'measurement/stage_eos_default_tools.py',O/'collection-tools/stage_eos_default_tools.py');cp(B/'measurement/eos-default-tool-validation-stage-receipt.json',O/'collection/stage-receipt.json');cp(Path(__file__),O/'collection-tools/stage_eos_tools_evidence.py')
summary={'schema_version':1,'scope':'Unchanged default-tool28 functional regression of independent EOS-only candidate. No experimental literal-input option or diagnostic observer.','sources':{'recipe':'254b2b03027f25094845dc31f5f87739f3584d2e','engine':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby':'fd8cbeb1fbb3cec4d2141558d5cfd63a500d50c4'},'started_at_utc':top['started_at_utc'],'finished_at_utc':top['finished_at_utc'],'passed':True,'tool_checks':28,'tool_checks_passed':28,'tool_checks_failed':0,'chat_posts':32,'literal_user_control_tokens_sent':False,'setup_check_exit_code':top['setup_check_exit_code'],'profile':'3.05 disk PLE, four slots, Q8 context/pool262144, chunk2048, dynamic MTP5/confidence0.6,rowbudget8','cleanup':{'server':top['server_cleanup'],'owned_groups':top['outer_owned_cleanup'],'root_release':release},'raw_result_sha256':sha(C/'raw/result.json'),'tool_report_sha256':sha(report_path),'raw_source_files':m['file_count'],'release_scan_candidates':0,'limitations':['Functional cases, not a general tool-answer quality or throughput measurement.','Final canonical actual-unit SDK/transport verification is a separate deployment gate.','Combined literal-input generation diagnostics and their semantic failures remain separate.','Credential recognizers cannot rule out arbitrary unlabeled secrets.']}
write(O/'summary.json',summary)
(O/'README.md').write_text('''# EOS-only default-tool regression

**All 28 unchanged tool checks passed across 32 chat POSTs.** This is the EOS-only Tabby fd8 candidate based on merged f650, without the experimental literal-user-input feature. Every actual request omits that option, and preflight verified that its API schema does not define the field. No raw observer or tokenization hook ran.

The fresh isolated test completed at 2026-10-09 08:21:51.480564 UTC with engine24f0 and recipe254. The existing public tool_smoke.py defaults were used without payload or semantic-fixture changes. Four extra POSTs belong to the two-mode followup/repeated-turn cases, so28 checks are32 requests. All completed responses, actual requests, errors, source/model/config/loaded geometry, sparse resources and lifecycle records are retained. The top result and unchanged tool report independently record28/28, zero failures and32 POSTs.

The profile remains3.05 flat-K, disk PLE, four request slots, Q8 context/pool262144, chunk2048, dynamic MTP depth5/confidence0.6 and draft-row budget8. Read-only setup/import checks passed before load, exact source/package/model identities were checked before and after clients, and the running server owned the persistent shared GPU flock through its verifiedFD8. Only that server held the shared lock; the outer controller held the port serialization lock. No source, package, model or service enablement changed during this run.

The server received SIGTERM and exited0. Both owned process groups were empty afterward. Root's08:23:21 release receipt independently confirms the controller/server/client PIDs absent, GPU and port free, and the same unchanged lock inode reacquired. REX manager remained enabled/inactive; Qwen remained disabled/inactive. Subsequent canonical promotion and the permanent-unit SDK5/raw2 checks are distinct evidence, not implied by this isolated result.

The [mandatory-EOS CPU/source archive](../mandatory-tool-eos/README.md) explains the narrow fix. The [combined experimental-feature generation replay](../literal-feature-eos-generation/README.md) retains5/8 exact-string semantics despite complete calls; this ordinary regression does not relabel those failures or establish universal exact copying. It is not a throughput or general model-quality benchmark.

reports/ contains12 exact small live files; sources/harness/ preserves the executable, declared runtime contract, frozen ownership helpers, test fixtures and CPU proof. The18 passing harness CPU cases include real harmless-process cleanup and unchanged client command/report checks. Two earlier synthetic model-fixture setup failures remain as preparation logs; the corrected checked subprocess passed before staging. The archived prior a70 tool report is only a CPU test fixture and is not counted as new fd8 live evidence.

collection/ binds remote hashes, file-only transfer, root launch/release and zero credential-pattern candidates. Model symlinks are excluded without following them; no weights, tensor outputs, extensions or credential stores are included. Bounded recognizers cannot prove absence of arbitrary unlabeled secrets. Nested pending preparation language is preserved, while this completed README governs results. Do not import archived standalone historical helpers merely to inspect them.
''')
p=O/'README.md';s=p.read_text()
for a,b in [('engine24f0','engine 24f0'),('recipe254','recipe 254'),('so28','so 28'),('are32','are 32'),('record28/28','record 28/28'),('and32','and 32'),('remains3.05','remains 3.05'),('pool262144','pool 262144'),('chunk2048','chunk 2048'),('depth5/confidence0.6','depth 5/confidence 0.6'),('budget8','budget 8'),('verifiedFD8','verified FD8'),('exited0','exited 0'),("Root's08:23:21","Root's 08:23:21"),('retains5/8','retains 5/8'),('contains12','contains 12'),('The18','The 18')]:s=s.replace(a,b)
p.write_text(s)
write(O/'collection-receipt.json',{'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':copies,'exclusions':m['omitted'],'scope':'Exact small completed artifacts and frozen sources. No GPU/API/lifecycle changes or large-model reads.'})
print(json.dumps({'archive':str(O),'summary_sha256':sha(O/'summary.json'),'state':'awaiting peer prose review before checksum freeze'},indent=2))
