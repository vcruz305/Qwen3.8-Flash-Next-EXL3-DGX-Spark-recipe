#!/usr/bin/env python3
"""Package completed diagnostic bytes only; never imports archived controllers."""
from pathlib import Path
import datetime,hashlib,json,shutil
B=Path('/home/vcruz/src/qwen-followup-20261009')
S=B/'literal-feature-diagnostic-publication-staging'
C=B/'literal-feature-diagnostic-collected'
O=B/'publication/2026-10-09/literal-feature-diagnostic'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
copies={};excluded=[]
def cp(src,dst):
 if src.is_symlink():excluded.append({'source':str(src),'target':str(src.readlink()),'reason':'symlink not followed'});return
 if src.suffix=='.pyc':excluded.append({'source':str(src),'sha256':sha(src),'reason':'incidental Python bytecode'});return
 assert src.is_file() and src.stat().st_size<10*1024*1024
 data=src.read_bytes();dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
 copies[str(dst.relative_to(O))]={'source':str(src),'sha256':sha(dst),'bytes':len(data)}
def tree(src,dst):
 for p in sorted(src.rglob('*')):
  if p.is_file() or p.is_symlink():cp(p,dst/p.relative_to(src))
assert not O.exists()
top=read(C/'raw/result.json');assert top['finished_at_utc'] and top['state']=='completed' and top['passed'] is True
assert sha(C/'raw/result.json')=='2a1fb1029cef6fc59d09c935405d9b28318f4ed2a1343bc20bd06b81873b2a9a'
manifest=read(C/'remote-source-manifest.json')
for name,row in manifest['files'].items():assert sha(C/'raw'/name)==row['sha256']
release=read(C/'launch-release/literal-feature-diagnostic-a70-release.json')
assert release['controller_result_sha256']==sha(C/'raw/result.json') and all(not row['exists'] for row in release['processes'].values()) and release['lock']['reacquired'] and not release['gpu_compute_processes']
scan=read(C/'release-review.json');assert scan['review_passed'] and scan['candidate_count']==0
coverage=read(next((C/'raw').rglob('coverage.json')));generation=read(next((C/'raw').rglob('generation.json')))
assert coverage['summary']['passed']==8 and len(coverage['results'])==8
assert generation['summary']['collection_complete']==8 and generation['summary']['semantic_passed']==6
cache=[{'name':r['name'],'policy':next(x['policy'] for x in read(S/'sources/harness/plan.json')['requests'] if x['name']==r['name']),'prompt_tokens':r['actual_usage']['prompt_tokens'],**r['cache_evidence'],'semantic_gate':r['semantic_gate'],'semantic_passed':r['semantic_passed']} for r in coverage['results'] if r['group']=='cache6']
assert [r['actual_cached_tokens'] for r in cache]==[0,2048,4096,0,2048,4096]
O.mkdir(parents=True)
tree(S/'sources',O/'sources');tree(S/'collection-tools',O/'collection-tools')
cp(S/'source-collection-receipt.json',O/'collection/source-collection-receipt.json')
tree(C/'raw',O/'reports');tree(C/'launch-release',O/'collection/launch-release')
for name in ['remote-source-manifest.json','release-review.json']:cp(C/name,O/'collection'/name)
cp(C/'offline-capture-replay.json',O/'analysis/offline-capture-replay.json')
cp(B/'literal-archive-release-review/review_archive.py',O/'collection-tools/pattern-source-review_archive.py')
cp(Path(__file__),O/'collection-tools/stage_feature_diagnostic_evidence.py')
summary={'schema_version':1,'scope':'Separate coverage and raw-generation diagnostic. Does not relabel the earlier failed93-POST qualification or qualify all exact-copy semantics.',
 'source_commits':{'engine':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby':'a70ae1fa9e457e478c3d96bdc84012a3cb331796','recipe':'254b2b03027f25094845dc31f5f87739f3584d2e'},
 'started_at_utc':top['started_at_utc'],'finished_at_utc':top['finished_at_utc'],'chat_posts':16,'diagnostic_gate_passed':True,'feature_semantics_qualified':False,
 'coverage':{'requests':8,'passed':8,'feature_specific_continuation400':2,'cache':cache,'native_wrong_value_observations':2},
 'generation':{'requests':8,'collection_complete':8,'exact_copy':6,'not_exact':2,'failed_names':[r['name'] for r in generation['results'] if not r['semantic_passed']],'four_request_http_overlap':generation['summary']['four_request_http_overlap'],'distinct_response_ids':generation['summary']['distinct_response_ids'],'capture_valid':top['capture']['capture_valid'],'native_equals_backend':sum(r['native_equals_backend']for r in top['capture']['rows']),'processed_samples':top['capture']['native_audit']['sample_calls'],'decoder_calls':top['capture']['native_audit']['decode_calls'],'observation_errors':top['capture']['native_audit']['observation_errors'],'lookup_errors':top['capture']['native_audit']['lookup_errors']},
 'cleanup':{'server':top['server_cleanup'],'owned_groups':top['outer_owned_cleanup'],'root_release':release},
 'original_qualification_reference':'../literal-feature-validation/README.md','raw_result_sha256':sha(C/'raw/result.json'),'release_review_scope':'All25 raw result files matched the remote receipt; zero candidates under bounded recognizers. This is not proof of arbitrary unlabeled-secret absence.',
 'limitations':['The two Unicode requests passed here; this does not erase or explain the prior length failure.','A raw observer was present, so this is not an uninstrumented throughput comparison.','Processed native IDs require rewind/healing/EOS interpretation; they are not automatically final accepted output.','No source/runtime/model mutation or service enablement was performed by this diagnostic.']}
write(O/'summary.json',summary)
rows='\n'.join('| '+r['name']+' | '+str(r['prompt_tokens'])+' | '+str(r['actual_cached_tokens'])+' | '+str(r['maximum_expected_lcp_tokens'])+' |' for r in cache)
(O/'README.md').write_text("""# Continuation, cache and raw-generation follow-up

**The eight coverage checks passed, and all eight raw generation captures are valid. Exact generation passed six of eight requests.** The original [failed feature qualification](../literal-feature-validation/README.md) remains unchanged: this separate diagnostic does not convert its failed gates into passes.

The run completed on 9 October 2026 at 07:59:11.839913 UTC, using the same isolated Tabby a70, engine24f0 and recipe254 sources. The model/profile remains3.05 flat-K, disk PLE, four request slots, Q8 cache/context262144, prefill chunk2048, dynamic MTP depth5/confidence0.6 and draft-row budget8. Exact source commits, deployment/config values, source and input hashes are retained in the reports.

## Coverage checks

The first two requests change only named tool choice to auto in the original continuation fixtures. Both returned JSON HTTP400 before SSE, with the exact actual-source-prepared message: `literal_user_control_tokens: Literal user control tokens do not support continued messages`. This reaches the intended feature-specific rejection without the earlier named-tool continuation incompatibility. The original two failed harness records are not edited or reassessed.

The six cache probes change only the pre-marker filler from125 to135 repetitions; their requested values, schemas, sampling and policy/stream sequence remain unchanged. CPU preparation records complete native/expanded ID arrays and exact prefix bounds. Native engine source proves that a cold start can checkpoint at2048 and an already reused256-token prefix can checkpoint at2304. The new cross-policy LCPs2323/2324 cover both.

| Request | Actual prompt tokens | Cached tokens | Maximum prior input-ID LCP |
| --- | ---: | ---: | ---: |
"""+rows+"""

Both directions started cold in this fresh run, then reused2048 tokens across the policy change and4096 tokens on the same-policy repeat. Every reuse stayed within its exact ID-prefix bound. All opt-in cache arguments were exact. Two native cache responses were wrong under their predeclared observation policy; that is separate from cache integrity. The original shorter probes reused only256 across policies and remain a separate limited result.

## Unchanged generation observations

After coverage, the server received the original Unicode pair, earlier-user-turn pair and four distinct concurrent requests, with unchanged payloads and full input IDs. Both Unicode requests passed in this run; the previous nonstreaming length failure is still preserved and is not explained or fixed by this observation.

The multiturn streaming request again stopped without completing the required call, and concurrent_2 again omitted the trailing endoftext spelling. Thus exact generation was6/8. The four concurrent request intervals overlap and their response IDs are distinct. No semantics were retried or normalized.

The bounded observer captured all eight requests, with native/backend output text equality8/8,470 processed-sample observations,15 native decoder calls and zero lookup/observation errors. Frozen capture assessment reproduces the stored result exactly in an offline replay. Raw processed IDs retain EOS, healing and rewind state; they must not be treated automatically as final accepted output. These records support a focused downstream investigation, rather than a broad claim that the input-token feature guarantees literal copying.

## Lifecycle and archive

The server received SIGTERM and exited0. The coverage client, generation client and server groups were empty after cleanup. Root's independent release receipt confirms all four retained PIDs, including the controller, were absent; GPU and port were free; the same cooperative lock inode was reacquired with unchanged contents. REX manager remained enabled/inactive and Qwen remained disabled/inactive. No service enablement or production replacement occurred.

- reports/: exact completed raw responses, SSE frames, observer data, deployment, resources and controller results.
- sources/harness/ and sources/observer/: exact frozen executables, plans, full IDs and CPU evidence.
- analysis/offline-capture-replay.json: independent frozen-assessor replay with no requests.
- collection/: remote file hashes, bounded raw release scan, launch/release and copy receipts.
- summary.json and SHA256SUMS: machine-readable scope/counts and archive hashes.

Nested preparation documents preserve their original pending language; this completed-result README governs the archive. The inherited observer module docstring predates its new native add-on: the actual reviewed scope includes existing CPU token/held-text observation, with no GPU tensor reads, KV access or input/output encoding changes. There is no uninstrumented speed claim.

Do not import archived historical controllers or the pattern-source review script just to inspect them; some historical helpers execute at import. The live controller uses reviewed source subsets and explicit hash-bound modules. The credential recognizer reported no candidates across25 raw files, but does not prove absence of arbitrary unlabeled secrets. Model symlinks are recorded as exclusions without following them; no weights, tensor artifacts, libraries or credential stores are included.
""")
p=O/'README.md';s=p.read_text()
for a,b in [('engine24f0','engine 24f0'),('recipe254','recipe 254'),('remains3.05','remains 3.05'),('context262144','context 262144'),('chunk2048','chunk 2048'),('depth5/confidence0.6','depth 5/confidence 0.6'),('budget8','budget 8'),('HTTP400','HTTP 400'),('from125 to135','from 125 to 135'),('at2048','at 2048'),('reused256-token','reused 256-token'),('at2304','at 2304'),('LCPs2323/2324','LCPs 2323/2324'),('reused2048','reused 2048'),('and4096','and 4096'),('only256','only 256'),('was6/8','was 6/8'),('equality8/8,470','equality 8/8, 470'),('observations,15','observations, 15'),('exited0','exited 0'),('across25','across 25')]:s=s.replace(a,b)
p.write_text(s)
write(O/'collection-receipt.json',{'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':copies,'exclusions':excluded+manifest['omitted'],'scope':'Exact small completed artifacts and frozen sources; no archived controller execution, source/runtime/service changes or large-model reads.'})
files=sorted(p for p in O.rglob('*') if p.is_file())
(O/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(O))+'\n' for p in files))
print(json.dumps({'path':str(O),'files':len(files)+1,'bytes':sum(p.stat().st_size for p in O.rglob('*')if p.is_file()),'sha256sums_sha256':sha(O/'SHA256SUMS'),'summary_sha256':sha(O/'summary.json')},indent=2))
