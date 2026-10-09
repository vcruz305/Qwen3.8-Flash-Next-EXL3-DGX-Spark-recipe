#!/usr/bin/env python3
"""Stage already-sanitized completed evidence. Never imports archived scripts."""
from pathlib import Path
import datetime,hashlib,json
B=Path('/home/vcruz/src/qwen-followup-20261009'); C=B/'final-standby-unit-collected'; S=B/'final-standby-unit-publication-staging'; O=B/'publication/2026-10-09/final-standby-unit-validation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert not O.exists();copies={}
def cp(src,dst):
 assert src.is_file() and not src.is_symlink() and src.stat().st_size<10*1024*1024
 dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();dst.write_bytes(src.read_bytes());copies[str(dst.relative_to(O))]={'source':str(src),'sha256':sha(dst),'bytes':dst.stat().st_size}
top=load(C/'reports/result.json');release=load(C/'extras/final-standby-unit-release.json');sdk=load(C/'reports/sdk.json')
assert sha(C/'reports/result.json')=='a2759735e588270bfce1e5e8b733326b4ed232fbe97839b6317eac4cfa9c7a65'
assert top['state']=='completed' and top['passed'] and top['raw_transport_checks']==2 and top['sdk_summary']['passed']==5 and top['sdk_summary']['failed']==0 and top['sdk_summary']['skipped']==0
assert release['passed'] and release['verifier_result_sha256']==sha(C/'reports/result.json') and release['unit_after_stop']['UnitFileState']=='disabled' and release['unit_after_stop']['ActiveState']=='inactive' and release['gpu_lock']['reacquired']
for key,pin in [('recipe','22d673463dba8e1f3a862056348769ab85a3c032'),('engine','24f0dece34f09c8d1e2359d6b3b3f7befef7331b'),('server','f391c96beea0bcd21cbae3c929f75bb98136ff32')]:
 assert top['sources'][key]['commit']==pin==top['sources_after'][key]['commit'] and not top['sources_after'][key]['tracked_changes']
for a,b in [('unit_before_api','unit_after_sdk'),('unit_before_sdk','unit_after_sdk')]:
 for k in ['MainPID','InvocationID','NRestarts']:assert top[a][k]==top[b][k]
for n,v in load(C/'collection-receipt.json')['files'].items():assert sha(C/'reports'/n)==v['public_sha256']
for src in sorted(C.rglob('*')):
 if src.is_file():cp(src,O/('collection/remote-collection-receipt.json' if src==C/'collection-receipt.json' else str(src.relative_to(C))))
for src in sorted(S.rglob('*')):
 if src.is_file():cp(src,O/src.relative_to(S))
cp(B/'measurement/collect_followup_unit_extras.py',O/'collection-tools/collect_followup_unit_extras.py')
cp(Path(__file__),O/'collection-tools/stage_final_standby_evidence.py')
summary={'schema_version':1,'scope':'Final canonical standby service short functional and shared-lock qualification; subsequent workload restoration is separately recorded.','state':'completed','passed':True,'sources':{k:top['sources'][k]for k in ['recipe','engine','server']},'finished_at_utc':top['finished_at_utc'],'raw_transport_checks':2,'sdk_checks_passed':5,'sdk_checks_failed':0,'sdk_checks_skipped':0,'chat_posts':7,'client_versions':top['sdk_summary']['client'],'selected_settings':top['selected_settings'],'unit_main_pid':int(top['unit_before_api']['MainPID']),'unit_invocation':top['unit_before_api']['InvocationID'],'unit_nrestarts':0,'main_pid_and_invocation_stable':True,'gpu_lock_verified_before_requests_and_after_sdk':True,'deployment_metadata':top['deployment_metadata'],'unit_release':release,'canonical_cpu':{'unittest_report':'Ran 155 tests; OK (skipped=5)','unittest_exit_code':0,'setup_check_exit_code':0,'cuda_hidden':True,'passed_total_derived':False},'service_environment_change':load(C/'extras/service-gpu-lock-install.json'),'original_result_sha256':sha(C/'reports/result.json'),'sdk_report_sha256':sha(C/'reports/sdk.json'),'raw_logs_transferred':False,'private_environment_or_backup_copied':False,'limitations':['Only seven short requests in a configured 1,048,576-token pool; no full-capacity or long-context qualification here.','Two raw checks establish transport/positive usage; the five SDK cases have their own stricter semantics.','The isolated EOS tool28 regression used a 262,144-token pool and is separate evidence.','The experimental literal_user_control_tokens feature was not promoted.','Credential filtering is bounded and specific to these known structured records and lifecycle-only log extracts.']}
write(O/'summary.json',summary)
(O/'README.md').write_text('''# Final canonical standby-service verification

**The actual user service passed seven short chat requests: two raw transport checks and five unchanged OpenAI SDK cases.** The test completed at 2026-10-09 08:29:28.505749 UTC on canonical recipe `22d673463dba8e1f3a862056348769ab85a3c032`, engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` and merged Tabby `f391c96beea0bcd21cbae3c929f75bb98136ff32`. The merged Tabby tree equals the independently tested EOS-only fd8 candidate. The experimental literal-input feature was not promoted.

The [source-stage receipt](extras/canonical-deployment-stage.json), [merge proof](proofs/tabby-eos-merge.json) and [fresh upstream read](proofs/upstream-heads-before-eos-promotion.json) bind those sources. Current official engine and Tabby heads were unchanged and verified as ancestors. The preserved original runtime was not moved; its source identities are recorded separately. No new rollback rehearsal is claimed here.

The service used the measured 3.05 disk-PLE concurrent configuration: four request slots, a **1,048,576-token shared pool**, maximum sequence length 262,144, chunk size 2,048 and draft-row budget 8. This is a short functional gate on that configured service, **not a full-pool capacity test**. Earlier isolated follow-up tests used a 262,144-token pool. The two raw checks require valid nonempty responses and positive usage; the SDK report separately validates its five cases with OpenAI 3.26.0 on Python 3.12.3. Discovery GETs are not counted as chat requests.

Before any API request, the verifier checked the expected MainPID, invocation, source/configuration identity and a real exclusive advisory flock on FD8 for the persistent shared GPU-lock inode. It repeated source, environment, deployment, PID/invocation and lock checks after the SDK run. MainPID526983 and invocation55b62ef9cff044bdba68d25f61213408 remained stable with NRestarts0. The verifier itself did not start, stop, enable or restart the service; the root controller performed the explicitly owned lifecycle.

The [installer receipt](extras/service-gpu-lock-install.json) proves the sole service-environment change was appending GPU_LOCK_FILE for the existing shared lock. Prior bytes were preserved as a prefix and a private backup was verified, but **neither the private environment file nor its backup was opened or copied into this archive**. Only hashes, byte counts, mode and the one assignment are published. Both supervisors were inactive during that change; Qwen stayed disabled and REX manager stayed enabled.

The [parent release receipt](extras/final-standby-unit-release.json) records a normal stop of that exact MainPID/invocation at08:31:14, followed at08:31:17 by Qwen disabled/inactive, exit0, NRestarts0, verifier/SDK completion, a free GPU and port8899, and reacquisition of the same unchanged shared-lock inode. Subsequent REX workload restoration is an independent receipt, rather than inferred from the successful Qwen stop.

Canonical CPU validation immediately before the live gate reported **Ran155tests; OK(skipped=5)** and setup--check exit0 with CUDA hidden. The archived summary excerpts retain the exact counts and original full-log hashes; no passed-total arithmetic or new model-quality claim is added. The [EOS-only default-tool regression](../eos-default-tool-validation/README.md) separately passed28 checks/32POSTs, and the [mandatory-EOS source/CPU archive](../mandatory-tool-eos/README.md) explains that fix.

reports/ contains the structured result, all SDK responses and the selected invocation record. Logs were reduced **on Spark before transfer** to synthesized lifecycle-only excerpts. No matching lifecycle lines existed in these two logs, so their published files are explicit empty excerpts with source hashes retained; structured records provide the positive lifecycle evidence. The raw YAML configuration, credential stores, private backups, model weights, tensors and extension binaries were outside the collection allowlist. Structured JSON passed the bounded credential-field/text scrub without any redaction. This is a specific evidence workflow, not a general-purpose arbitrary-secret sanitizer.

The [collection receipt](collection/remote-collection-receipt.json), [extra receipt](extras/collection-receipt.json) and source-copy receipt preserve original/public hashes and transformations. The collector and verifier are archived as source for audit; their host-specific paths and exact historical bindings must be reviewed before reuse. Do not import archived historical scripts merely to inspect them. Assembling this evidence issued no inference, GPU, service or source mutation.
''')
p=O/'README.md';s=p.read_text()
for a,b in [('MainPID526983','MainPID 526983'),('invocation55','invocation 55'),('NRestarts0','NRestarts 0'),('at08:','at 08:'),('exit0','exit 0'),('port8899','port 8899'),('Ran155tests; OK(skipped=5)','Ran 155 tests; OK (skipped=5)'),('passed28','passed 28'),('checks/32POSTs','checks/32 POSTs')]:s=s.replace(a,b)
p.write_text(s)
write(O/'collection-receipt.json',{'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':copies,'scope':'Exact already-sanitized completed records and source artifacts. No private environments/backups, raw auth logs, model payloads, extensions or GPU/API/lifecycle actions.','publication_root':'outside Git worktree','restoration_receipt_pending':True})
print(json.dumps({'archive':str(O),'summary_sha256':sha(O/'summary.json'),'state':'staged; awaiting separate restoration receipt and prose review before freeze'}))
