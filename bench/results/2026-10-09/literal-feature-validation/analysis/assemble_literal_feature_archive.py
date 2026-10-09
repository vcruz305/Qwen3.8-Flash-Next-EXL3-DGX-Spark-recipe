#!/usr/bin/env python3
"""Package completed source/CPU/r1/r2 evidence; no network or model operations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json
B=Path('/home/vcruz/src/qwen-followup-20261009')
D=B/'publication/2026-10-09/literal-feature-validation'
assert not D.exists(), 'Refuse replacing an evidence archive'
D.mkdir(parents=True)
copied=[];excluded=[]
def sha(raw): return hashlib.sha256(raw).hexdigest()
def copy(src,rel):
 assert src.is_file() and not src.is_symlink()
 before=src.stat();raw=src.read_bytes();after=src.stat()
 assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
 dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 copied.append({'source':str(src),'destination':str(rel),'bytes':len(raw),'sha256':sha(raw),'mtime_ns':before.st_mtime_ns,'inode':before.st_ino})
def tree(src,prefix):
 for f in sorted(src.rglob('*')):
  if not f.is_file():continue
  if '__pycache__' in f.parts or f.suffix=='.pyc':
   excluded.append({'source':str(f),'bytes':f.stat().st_size,'sha256':sha(f.read_bytes()),'reason':'Incidental Python bytecode; source is preserved.'});continue
  copy(f,Path(prefix)/f.relative_to(src))
tree(B/'literal-feature-publication-staging','preparation')
tree(B/'literal-feature-r1-collected','attempts/r1')
tree(B/'literal-feature-r2-collected','attempts/r2')
for name in ['analyze_literal_feature_completed.py','literal-feature-r2-analysis.json','collect_literal_feature_live.py','literal-feature-r2-engine-peer.json','literal-feature-r2-engine-peer.log','assemble_literal_feature_archive.py']:
 copy(B/name,Path('analysis')/name)
copy(B/'tabby-literal-a70.bundle','source/tabby-literal-a70.bundle')
a=json.loads((B/'literal-feature-r2-analysis.json').read_text())
r=json.loads((B/'literal-feature-r2-collected/live/result.json').read_text())
cpu=json.loads((B/'literal-feature-publication-staging/cpu-summary.json').read_text())
assert a['feature_summary']['passed']==56 and a['total_checks']==89 and a['actual_chat_posts']==93
assert r['state']=='completed' and not r['passed'] and len(r['outer_owned_cleanup'])==3
assert all(x['owned_group_empty'] for x in r['outer_owned_cleanup'])
failures=[
 {'name':'unicode_nfc_nonstream_optin','classification':'substantive generation failure','detail':'HTTP200 length finish at256 completion tokens; no complete record_text call.'},
 {'name':'multi_turn_user_stream_optin','classification':'substantive generation/protocol failure','detail':'HTTP200 SSE server error: model stopped without a complete call required by tool_choice.'},
 {'name':'concurrent_2','classification':'substantive exact-copy failure','detail':'Complete record_text call omitted trailing <|endoftext|>; original run has no native token observation and does not establish where omission occurred.'},
 {'name':'continuation_nonstream','classification':'original harness condition failure','detail':'Actual HTTP400 and detail equal prepared response, but frozen client also required a literal_user_control_tokens prefix. Named tool_choice hit an earlier continuation incompatibility; the feature-specific continuation guard was not reached.'},
 {'name':'continuation_stream','classification':'original harness condition failure','detail':'Same earlier HTTP400 and frozen-client prefix mismatch. Preserved as failed in original qualification; no relabeling.'}]
summary={'schema_version':1,'scope':'Completed original candidate qualification, including failed preflight attempt. No later diagnostic included; not a deployment or broad exact-copy success claim.',
 'sources':{'tabby':a['candidate_tabby'],'engine':a['engine'],'recipe':a['recipe']},'cpu':cpu,
 'r1':{'model_loads':0,'chat_posts':0,'original_record_state':'preflight','status_note':'Controller exited without terminal status after rejecting stock Ubuntu sitecustomize. Record unchanged; collection verifies absent PID and no attempt/client artifacts.'},
 'r2':{'started_at_utc':r['started_at_utc'],'finished_at_utc':r['finished_at_utc'],'completed':True,'passed':False,'logical_checks':89,'passed_checks':84,'failed_checks':5,'chat_posts':93,
 'feature_checks':a['feature_summary'],'tool_checks':a['tools_summary'],'tool_chat_posts':32,'groups':a['groups'],'failed_rows':failures,
 'known_fixture_exact':{'native':{'passed':0,'total':8},'optin':{'passed':8,'total':8}},'heldout_exact':{'native':{'passed':2,'total':12},'optin':{'passed':10,'total':12}},
 'concurrent4':{'exact':3,'total':4,**a['concurrent4']},'cache':a['cache'],'cache_scope':'Bounds and same-policy repeat reuse3840 observed; cross-policy probes reuse only256 and do not prove2048/2304 recurrent checkpoint reuse.',
 'no_marker_checks':3,'native_semantic_failures_are_observations':20,'available_usage_counts_match_prepared_ids':52,
 'server_cleanup':r['server_cleanup'],'outer_owned_cleanup':r['outer_owned_cleanup'],'owner_gate_after':r['owner_gate_after']},
 'frozen_method':{'plan_sha256':a['plan_sha256'],'input_binding_sha256':a['input_binding_sha256'],'client_sha256':a['client_sha256'],'offline_original_assessor_rows_replayed':61},
 'limitations':['Default-off input tokenization feature, not a guarantee of generated exact copying.','Inputs and full ID sequences archived; no native output-ID trace installed in r2.','No reasoning-boundary repair, stochastic reasoning, or Unicode decoder attribution established by these failures.','Runtime logs/timings are correctness evidence, not controlled throughput benchmarks.']}
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
readme="""# Literal user control-token candidate qualification — original r1/r2

**The completed r2 qualification failed: 84 of 89 logical checks passed, with 93 chat-completion POSTs.** Three failures concern generation or exact copying; two are original harness condition failures. All five failed gates and unchanged raw evidence are retained. This archive includes no later diagnostic results and establishes no deployment approval.

The default-off candidate is Tabby **a70ae1fa9e457e478c3d96bdc84012a3cb331796**, with ExLlamaV3 **24f0dece34f09c8d1e2359d6b3b3f7befef7331b** and recipe **254b2b03027f25094845dc31f5f87739f3584d2e**. The isolated Spark load used 3.05 flat-K, concurrent profile, disk n-grams, K8/V8, maximum batch4, 262144 shared cache tokens and prefill chunk2048. Recorded deployment supplies all settings.

The opt-in feature changes encoding of nine recognized control spellings in user text. Default/native requests retain their original encoding. Immutable private request plans feed changed IDs to both context-length validation and generation/cache input. This input behavior does not guarantee generated exact copying.

## CPU and source evidence

| Evidence | Result | Scope |
| --- | --- | --- |
| Spark full CPU |419 passed, 0 skipped, 5210 subtests|CUDA hidden, real installed engine import, setup exit0|
| Earlier WSL CPU |403 passed, 2 skipped, 5210 subtests|Environment without installed native engine; skipped coverage subsequently included on Spark|
| Focused feature |21 passed, 51 subtests|Strict flag, errors, private plan, default path, context/runtime integration|
| Actual Qwen utility |46 rows /23 pairs|One expected non-roundtrip HTTP400; seven complete shared prefix pages|
| Interleaved cache proof |4 plans passed|Real CPU tensors, no CUDA initialization|

[Preparation](preparation/) retains all seven candidate files, patch, bindings, commands and logs. The [source bundle](source/tabby-literal-a70.bundle) requires its recorded f650 predecessor; it is not a full repository. Peer review preceded removal of one trailing test-file space only; binding records AST equality and final focused rerun. Runtime/documentation/helper bytes were unchanged.

## Attempts and results

[r1](attempts/r1/live/result.json) exited preflight before model load or chat requests. Its original record remains in state preflight without an invented terminal timestamp. An overstrict absent-sitecustomize requirement rejected Ubuntu's apport exception handler. The installed module was identified by path, resolved path, hash and package conffile metadata. [r2](preparation/harness-r2/R2-NOTE.md) admits only that exact module (or absence), rejects user customization, rechecks bytes and records terminal preflight failures. Model, candidate, requests and input IDs were unchanged.

[r2](attempts/r2/live/result.json) ran07:34:07.683831–07:37:52.092261 UTC on2026-10-09. Frozen61 feature POSTs ran first, then existing28 tool checks involving32 POSTs: **89 checks and93 POSTs**.

| Group | Logical gates | Exact-copy observations / scope |
| --- | --- | --- |
| Known native/opt-in pairs |16/16 passed|Native0/8 exact, opt-in8/8; native values observational|
| Six held-out fixtures, both modes/policies |22/24 passed|Native2/12 exact, opt-in10/12|
| Four simultaneous opt-in requests |3/4 passed|Four overlapping intervals and distinct response IDs|
| Six cache probes |6/6 passed|Bounds respected, opt-in3/3 exact; native values observational|
| Eight invalid-input checks |6/8 passed|Six actual400 and two422; two continuation harness failures|
| No-marker absent/false/true |3/3 passed|Native behavior controls|
| Existing tools |28/28 passed|32 POSTs including round-trip follow-ups|

The known requested value is exactly **&lt;think&gt;literal&lt;/think&gt;**, with no leading/trailing newline. Structural newlines in parameter markup are not requested value bytes.

The unchanged failed rows are:

- unicode_nfc_nonstream_optin: HTTP200, length finish at256 completion tokens, no complete tool call.
- multi_turn_user_stream_optin: SSE error that the model stopped without a complete call required by tool_choice.
- concurrent_2: complete tool call omitted trailing &lt;|endoftext|&gt;. No native output-ID observer ran, so this alone does not locate omission within generation/decoding/parsing.
- continuation_nonstream and continuation_stream: actual400 and detail equal prepared response, but frozen client additionally required a literal_user_control_tokens detail prefix. Named tool selection hit an earlier continuation incompatibility. They remain failures of this original harness; its intended feature-specific continuation guard was not exercised.

There are52 assembled responses with usage and52 distinct IDs; all available prompt counts match prepared IDs. One of53 HTTP200 feature responses is the SSE error. [Offline replay](analysis/literal-feature-r2-analysis.json) reproduces all61 frozen-assessor rows from saved traces, making no requests.

## Cache interpretation

Cached-token counts were **256,256,3840** in both sequence directions. Full-ID common-prefix bounds were274,2173,3968 and287,2174,3965. This proves bounded reuse and substantial same-policy repetition reuse, but not cross-policy reuse of a full2048-token recurrent checkpoint.

With an already cached256-token start and2048 chunk/checkpoint interval, the next normal checkpoint is2304. Cross-policy prefixes2173/2174 end earlier. This is a coverage limitation, not evidence of corruption. Separately lengthened probes must retain distinct results.

## Ownership and preservation

Server SIGTERM produced exit0. Clients exited1(feature)/0(tools); all three owned groups were verified empty. Final checks reported no GPU process, manager inactive and Qwen inactive/disabled. Shared GPU lock and inherited server lock were verified. No service enablement, production replacement or diagnostic observer was part of r2.

- [Summary](summary.json): counts, failures, sources, bounds and cleanup.
- [r1 collection](attempts/r1/) and [r2 collection](attempts/r2/): unchanged raw results/logs/config, launch records and receipts.
- [Original harness](preparation/harness/) and [r2 harness](preparation/harness-r2/): plan, full native/changed IDs, clients/controllers/tests and earlier preparation snapshots.
- [Receipt](collection-receipt.json), [release review](release-review.json), [checksums](SHA256SUMS).

Nested preparation documents retain historical pending language; this completed-result README governs this archive. Nested receipts retain their original relative scope. No weights, runtime libraries, tensors or credential stores are included. Release recognizers do not prove absence of arbitrary unlabelled secrets.
"""
(D/'README.md').write_text(readme)
receipt={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Local stable copies of completed bounded Spark collections/source/CPU evidence. Original nested receipts unchanged. No network/GPU/API operations.','files':copied,'excluded':excluded,'copied_files':len(copied),'copied_bytes':sum(x['bytes'] for x in copied)}
(D/'collection-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'archive':str(D),'copied_files':len(copied),'copied_bytes':receipt['copied_bytes'],'excluded':len(excluded)}))
