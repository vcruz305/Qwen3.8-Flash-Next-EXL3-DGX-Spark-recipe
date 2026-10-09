#!/usr/bin/env python3
"""Archive completed small generation8 records; no historical controller imports."""
from pathlib import Path
import datetime,hashlib,json
B=Path('/home/vcruz/src/qwen-followup-20261009');C=B/'literal-feature-eos-generation-collected';H=B/'measurement/literal-feature-eos-generation';S=B/'literal-feature-eos-observer';O=B/'publication/2026-10-09/literal-feature-eos-generation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
assert not O.exists();copies={};excluded=[]
def cp(src,dst):
 assert src.is_file() and not src.is_symlink() and src.stat().st_size<10*1024*1024
 assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes());copies[str(dst.relative_to(O))]={'source':str(src),'sha256':sha(dst),'bytes':dst.stat().st_size}
def manifested(src,dst,expected):
 assert sha(src/'manifest.json')==expected
 m=read(src/'manifest.json')
 for n,v in m['files'].items():
  p=src/n;assert sha(p)==v['sha256'] and p.stat().st_size==v['bytes'];cp(p,dst/n)
 cp(src/'manifest.json',dst/'manifest.json')
 if (src/'SHA256SUMS').exists():cp(src/'SHA256SUMS',dst/'SHA256SUMS')
top=read(C/'raw/result.json');assert sha(C/'raw/result.json')=='9266c87d09fa9039a88979b8912749fe62f39a4cb931ae82f1e38c31e72091b9'
assert top['finished_at_utc'] and top['state']=='completed' and top['passed'] and top['capture']['capture_valid']
generation=read(next((C/'raw').rglob('generation.json')));assert generation['summary']['collection_complete']==8 and generation['summary']['semantic_passed']==5
assert all(r['result']['finish_reason']=='tool_calls' for r in generation['results'])
release=read(C/'launch-release/literal-feature-eos-generation-f4aadf-release.json');assert release['controller_result_sha256']==sha(C/'raw/result.json') and all(not x['exists']for x in release['processes'].values()) and release['lock']['reacquired'] and not release['gpu_compute_processes']
scan=read(C/'release-review.json');assert scan['review_passed'] and scan['candidate_count']==0
manifested(H,O/'sources/harness','039681eece4e6dd1049db8062930ecd942dfe7c4dfb4702ede1750dd702dc728');manifested(S,O/'sources/observer','a45a4684524f2eb45614ae8f70808337c90c78bce7b0f7992fbde2c6bea7b865')
m=read(C/'remote-source-manifest.json')
for n,v in m['files'].items():assert sha(C/'raw'/n)==v['sha256'];cp(C/'raw'/n,O/'reports'/n)
excluded.extend(m['omitted'])
for p in sorted((C/'launch-release').iterdir()):cp(p,O/'collection/launch-release'/p.name)
for n in ['remote-source-manifest.json','release-review.json']:cp(C/n,O/'collection'/n)
cp(C/'offline-capture-replay.json',O/'analysis/offline-capture-replay.json')
cp(H/'engine-peer-generation-review.json',O/'analysis/engine-peer-generation-review.json')
for n in ['collect_completed_small.py','review_small_archive.py']:cp(B/'literal-feature-diagnostic-publication-staging/collection-tools'/n,O/'collection-tools'/n)
cp(B/'literal-archive-release-review/review_archive.py',O/'collection-tools/pattern-source-review_archive.py')
cp(B/'measurement/stage_literal_feature_eos_generation.py',O/'collection-tools/stage_literal_feature_eos_generation.py');cp(B/'measurement/literal-feature-eos-generation-stage-receipt.json',O/'collection/stage-receipt.json');cp(Path(__file__),O/'collection-tools/stage_eos_generation_evidence.py')
rows=[{'name':r['name'],'prompt_tokens':r['result']['usage']['prompt_tokens'],'cached_tokens':r['result']['usage']['prompt_tokens_details']['cached_tokens'],'finish_reason':r['result']['finish_reason'],'exact_copy':r['semantic_passed'],'outcome_error':r.get('outcome_error')}for r in generation['results']]
summary={'schema_version':1,'scope':'Combined experimental input feature plus independent mandatory-EOS fix; complete captured calls do not qualify exact-copy semantics. No isolated numerical A/B due different preceding cache history.','sources':{'recipe':'254b2b03027f25094845dc31f5f87739f3584d2e','engine':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby':'f4aadf114b0044fa8cbe1b50241dc80ea7d61583'},'started_at_utc':top['started_at_utc'],'finished_at_utc':top['finished_at_utc'],'chat_posts':8,'capture_valid':True,'native_backend_equal':8,'complete_tool_calls':8,'exact_copy_passed':5,'exact_copy_failed':3,'experimental_feature_qualified':False,'rows':rows,'native_audit':top['capture']['native_audit'],'four_request_http_overlap':generation['summary']['four_request_http_overlap'],'distinct_response_ids':generation['summary']['distinct_response_ids'],'cleanup':{'server':top['server_cleanup'],'owned_groups':top['outer_owned_cleanup'],'root_release':release},'raw_result_sha256':sha(C/'raw/result.json'),'source_record_count':m['file_count'],'release_scan_candidates':0,'limitations':['Eight unchanged visible requests and full input IDs are preserved, but prior coverage8 cache history is absent.','The observer reads existing CPU data and is not an uninstrumented performance run.','Processed tokens require EOS/healing/rewind interpretation.','The independent default-off tool28 EOS-only regression is a separate archive.','A bounded credential recognizer cannot prove absence of arbitrary unlabeled secrets.']}
write(O/'summary.json',summary)
body='\n'.join('| '+r['name']+' | '+str(r['prompt_tokens'])+' | '+str(r['cached_tokens'])+' | '+('Pass'if r['exact_copy']else'Fail')+' |'for r in rows)
(O/'README.md').write_text('''# Generation replay with the mandatory-EOS fix

**All eight raw captures are valid and all eight API calls completed with `tool_calls`. Five of eight exact-string checks passed.** The two multi-turn calls completed with an incorrect argument, and concurrent_2 still omitted the requested trailing endoftext marker. These failures remain recorded; the experimental literal-input feature is not qualified by this run.

The fresh isolated run completed at2026-10-09 08:17:41.742352UTC using combined Tabbyf4aadf (a70 literal-input candidate plus the independent mandatory-EOS fix), engine24f0 and recipe254. It issued exactly the original Unicode2, multi-turn2 and concurrent4 requests. Actual candidate CPU preparation proves every visible payload, rendered prompt, full original/expanded ID list, context count, replacement count and effective sampling field matches the preceding a70 generation8. No continuation/cache probes or standard tool cases were repeated here.

Unlike the preceding [combined16 diagnostic](../literal-feature-diagnostic/README.md), this server did not first receive coverage8. That cache-history difference prevents treating output changes as an isolated numerical A/B. Source/CPU/native evidence for the EOS mechanism is in [mandatory-tool-eos](../mandatory-tool-eos/README.md); unchanged EOS-only default-tool regression is a separate gate. These artifacts do not claim every semantic failure was an EOS defect.

| Request | Actual prompt tokens | Cached tokens | Exact argument |
| --- | ---: | ---: | --- |
'''+body+'''

Both multi-turn modes now returned a complete call with the value `Remembered <tool_call>user`, which differs from the requested original string. Concurrent_2 again omitted the trailing endoftext marker. The Unicode pair passed, as in the preceding raw diagnostic; this does not erase or explain the earlier qualification's length failure. All four concurrent HTTP intervals overlap and response IDs are distinct.

The source-bound observer records native/backend text equality8/8,481 processed-sample observations,15 decoder calls and zero lookup/observation errors. The unchanged assessor exactly reproduces the saved capture result offline. Processed IDs retain EOS, healing and rewind state; they are not automatically emitted tokens. The observer reads existing CPU values and held text, without GPU tensor/KV reads or input/output rewriting. No uninstrumented speed claim is made.

The model/profile remains3.05 disk PLE, four slots, Q8 context/pool262144, chunk2048, dynamic MTP5/confidence0.6, rowbudget8. The server alone holds the shared flock. It received SIGTERM and exited0; both owned process groups were empty. Root's08:19:34 release receipt confirms all three retained PIDs absent, GPU/port free and the same lock inode reacquired. REX manager remained enabled/inactive and Qwen disabled/inactive. Canonical source and service policy were not changed by this diagnostic.

reports/ retains exact raw/SSE/observer/deployment/resource evidence. sources/ retains the frozen executable/input/observer bundles; nested preparation documents remain historical. analysis/ contains the exact frozen-assessor replay and input/lifecycle peer proof. collection/ binds remote hashes, file-only staging, root launch/release and the zero-candidate bounded scan. No model files, extensions, tensors, credentials or followed model symlinks are included. The bounded recognizer cannot exclude arbitrary unlabeled secrets. Do not import archived historical helpers merely to inspect them; some are standalone execute-on-import scripts.
''')
write(O/'collection-receipt.json',{'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':copies,'exclusions':excluded,'scope':'Exact small completed evidence and frozen sources; no GPU/API/lifecycle/source changes or large-model reads.'})
print(json.dumps({'archive':str(O),'state':'awaiting independent native conclusion before checksum freeze','summary_sha256':sha(O/'summary.json')},indent=2))
