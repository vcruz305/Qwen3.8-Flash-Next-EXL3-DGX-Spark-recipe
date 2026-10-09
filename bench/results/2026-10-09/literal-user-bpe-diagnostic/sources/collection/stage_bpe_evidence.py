#!/usr/bin/env python3
"""Local-only immutable-source evidence packaging; no archived imports."""
from pathlib import Path
import datetime,hashlib,json,shutil
B=Path('/home/vcruz/src/qwen-followup-20261009')
OUT=B/'publication/2026-10-09/literal-user-bpe-diagnostic'
COPIED={};OMITTED=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def cp(src,dst):
 if src.is_symlink():
  OMITTED.append({'source':str(src),'destination':str(dst.relative_to(OUT)),'reason':'symlink not followed','target':str(src.readlink())});return
 if src.suffix=='.pyc':
  OMITTED.append({'source':str(src),'destination':str(dst.relative_to(OUT)),'reason':'incidental Python bytecode','bytes':src.stat().st_size,'sha256':sha(src)});return
 assert src.is_file() and src.stat().st_size<10*1024*1024
 h=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(src)==h==sha(dst)
 COPIED[str(dst.relative_to(OUT))]={'source':str(src),'bytes':dst.stat().st_size,'sha256':h}
def tree(src,dst):
 for p in sorted(src.rglob('*')):
  if p.is_symlink() or p.is_file():cp(p,dst/p.relative_to(src))
def main():
 assert not OUT.exists();OUT.mkdir(parents=True)
 rows=[]
 for label,name in [('target-only-aborted-r1','literal-user-bpe-review-r1'),('target-only-r2','literal-user-bpe-review-r2'),('mtp-confirmation','literal-user-bpe-review-mtp')]:
  src=B/name;top=json.loads((src/'raw/result.json').read_text());cell=top['cells'][0];actual=cell['label']
  run=json.loads((src/'raw'/actual/'result.json').read_text());cap=json.loads((src/'raw'/actual/'literal-capture.json').read_text())
  assert top['finished_at_utc'] and run['finished_at_utc']
  clean=run['server_cleanup'];assert clean['signals']==['SIGTERM'] and clean['exit_code']==0 and clean['owned_group_empty'] and not clean['unexpected_exit']
  review=json.loads((src/'release-review.json').read_text());assert review['review_passed'] and review['candidate_count']==0
  manifest=json.loads((src/'remote-source-manifest.json').read_text())
  for rel,info in manifest['files'].items():
   p=src/'raw'/rel;assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes']
  tree(src/'raw',OUT/'reports'/label)
  for f in ['remote-source-manifest.json','release-review.json']:cp(src/f,OUT/'collection'/label/f)
  row={'label':label,'state':top['state'],'capture_valid':cap['capture_valid'],'client_processes':len(run['clients']),'finished_at_utc':top['finished_at_utc'],'top_result_sha256':sha(src/'raw/result.json'),'assessment_sha256':sha(src/'raw'/actual/'literal-capture.json'),'cleanup':clean}
  if label=='target-only-aborted-r1':
   assert not run['clients'] and not cap['capture_valid'];row.update(semantic_requests=0,semantic_passes=0,semantic_failures=0,excluded_from_semantic_denominator=True,error=top['error'])
  else:
   assert top['passed'] and cap['capture_valid'] and cap['original_semantic_counts']=={'pass':8} and len(cap['traces'])==8
   assert len(cap['input_encoding_proofs'])==8 and len(cap['api_prompt_usage'])==8
   row.update(semantic_requests=8,semantic_passes=8,semantic_failures=0,actual_prompt_tokens=352,changed_model_input=True,mtp_evidence=cap.get('mtp_evidence'))
  rows.append(row)
 cp(B/'literal-user-bpe-review-r2/phase-review.json',OUT/'analysis/target-only-phase-review.json')
 cp(B/'analyze_literal_user_bpe.py',OUT/'analysis/analyze_literal_user_bpe.py')
 cp(B/'analyze_literal_ablation.py',OUT/'analysis/analyze_literal_ablation.py')
 tree(B/'literal-user-bpe-mtp-independent-review',OUT/'analysis/mtp-independent-review')
 # Independent shared raw files must match the full remote copy.
 base=B/'literal-user-bpe-review-mtp/raw';peer=B/'literal-user-bpe-mtp-independent-review/raw'
 for p in peer.rglob('*'):
  if p.is_file() and not p.is_symlink():assert sha(p)==sha(base/p.relative_to(peer))
 for name in ['literal-bpe-observer','measurement/literal-user-bpe','measurement/literal-user-bpe-mtp']:
  tree(B/name,OUT/'sources'/name)
 for f in ['review_small_archive_v1.py','review_small_archive.py','collect_completed_small.py','stage_bpe_evidence.py']:
  cp(B/'measurement'/f,OUT/'sources/collection'/f)
 cp(B/'literal-archive-release-review/review_archive.py',OUT/'sources/collection/pattern-source-review_archive.py')
 summary={'schema_version':1,'scope':'Changed-input diagnostic, one deterministic synthetic prompt in eight request variants per completed cell. Original visible request bytes unchanged; model token IDs intentionally differ. No production deployment or throughput claim.',
  'engine':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','tabby':'f650bb5389e0a273549e47d4d26a765760c013e1','recipe':'a8c72bdf811646f413fdc03a4e49811a2753d0cf',
  'cells':rows,'completed_semantic_requests':16,'completed_semantic_passes':16,'aborted_client_requests':0,
  'original_prompt_tokens':348,'changed_prompt_tokens':352,'original_disabled_reference':{'passed':0,'total':8,'archive':'../literal-drafting-ablation'},'original_mtp_reference':{'passed':1,'total':8,'archive':'../literal-drafting-ablation'},
  'native_output_sha256':'ecbd48e3c7474ea7532f97ebaca00283cc18bdd9f70d71043e490656ad7c2d68',
  'boundary_limit':'Both completed cells naturally emit END at processed position12 after an unfinished quote and attach grammar, with no forced output. Exact final argument copying does not establish a reasoning-boundary repair.',
  'release_review':'Zero candidates under bounded recognizers; this does not prove absence of arbitrary unlabeled secrets.'}
 write(OUT/'summary.json',summary)
 readme="""# Changed-input user-span BPE diagnostic — 9 October 2026

This archive records a bounded experiment on the 3.05-bpw model. The visible request bytes are unchanged, but two native added-token IDs inside the known synthetic user text are expanded into ordinary BPE tokens. The actual model input therefore changes from 348 to 352 tokens. Template control IDs remain unchanged.

| Attempt | Drafting | Client requests | Exact literal checks | Capture |
| --- | --- | ---: | ---: | --- |
| First target-only attempt | Disabled | 0 | Not run | Invalid; operational refusal |
| Target-only retry | Disabled | 8 | 8/8 passed | Valid |
| MTP confirmation | Dynamic, maximum depth 5 | 8 | 8/8 passed | Valid |

The eight requests are variants of **one deterministic synthetic prompt**, combining required/named tool choice, streaming/non-streaming, and explicit-budget/unbudgeted behavior. They are not eight diverse examples. The original 348-token experiment is preserved separately in [literal-drafting-ablation](../literal-drafting-ablation/README.md): target-only passed 0/8 and MTP passed 1/8. These results support further evaluation of input-token semantics; they do not establish a general tokenizer policy or broad model-quality improvement.

## What is verified

Each completed request has a native input proof, actual API usage of 352 prompt tokens, unchanged visible wire payload, exact final tool argument, native/backend text equality, and a bounded producer timeline with no dropped records. The altered ID sequence expands only the two known user-span markers; the decoded complete prompt is unchanged. Context length is assessed after this expansion, before generation.

The MTP confirmation records 268 accepted and 157 rejected prediction tokens across eight requests. This demonstrates actual proposal work; it does not identify verification-window counts or establish a throughput benefit. All 16 completed requests have identical native text SHA256 `ecbd48e3c7474ea7532f97ebaca00283cc18bdd9f70d71043e490656ad7c2d68`.

Both completed cells naturally emit the closing reasoning marker at processed position 12 after an unfinished quote, then attach the content grammar. There is no forced output in these traces. Correct final literal copying therefore **does not demonstrate a reasoning-boundary repair** or establish what continuation the model intended. The independent phase analyses retain this distinction.

## Sources and lifecycle

The recorded sources are engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`, Tabby `f650bb5389e0a273549e47d4d26a765760c013e1`, and published recipe `a8c72bdf811646f413fdc03a4e49811a2753d0cf`. The external diagnostic hook does not alter those checkouts or the model files. The target-only controller is `2a0a0e3c…`; the narrowly derived MTP controller is `7c9ca1ac…`. Full hashes, jobs, actual deployments, configs, source checks, observer records, client reports, CPU review artifacts and owned cleanup evidence are included.

The first attempt stopped before clients because the manager-state ownership gate detected a conflict. Its failure and zero-client result remain intact. Its own server, and both completed servers, received SIGTERM, exited 0, and left their owned process groups empty. Controllers held the cooperative GPU lock and checked owner state; they did not enable or disable either service.

The API source checks are live source/import checks, not a claim that the complete CPU suite was rerun on Spark for every cell. The retained local CPU reports describe their own narrower source-execution, request matching, tokenization and lifecycle scopes.

## Archive layout and reproduction boundary

- `reports/`: exact small result files for aborted r1, target-only r2, and MTP confirmation.
- `analysis/`: unchanged independent phase reviews, their source, and the MTP review's matching raw subset.
- `sources/measurement/`: frozen controllers, jobs, model-input audit, exact parent helpers and CPU review evidence.
- `sources/literal-bpe-observer/`: the source-bound input hook, observer, validators, payloads and CPU evidence.
- `collection/`: original remote manifests and bounded release-review receipts.
- `collection-receipt.json`: byte-for-byte source mapping and exclusions.
- `summary.json` and `SHA256SUMS`: machine-readable scope and integrity.

This is evidence for a host-specific diagnostic, not an automatic installer. Do not import arbitrary historical review scripts: some older analysis helpers execute their analysis on import. The collection tool extracts only literal pattern data from its hash-verified pattern source using the Python AST, without executing that source. Controller invocations require the documented Spark paths, published recipe, client environment, exact model metadata, ports and cooperative lock; copied scripts retain those explicit paths and hashes. Read each frozen controller README before any separately authorized run.

Publication omits incidental Python bytecode and model-view symlinks without following them. Original local/remote result files are retained. All regular small text/data copies are exact. The bounded scan found zero credential-pattern/key-name candidates; it cannot guarantee absence of arbitrary unlabeled secrets. No tensors, weight payloads, shared libraries or credential stores were copied or hashed. The scanner's v1 source remains archived for the first receipt; v2 records excluded symlinks explicitly. One initial local MTP analysis assertion assumed zero formatting newlines; the retained direct raw-parameter check documents its correction to one framing newline on each side, without changing raw output or model results.
"""
 (OUT/'README.md').write_text(readme)
 write(OUT/'collection-receipt.json',{'schema_version':1,'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_copies':COPIED,'exclusions':OMITTED,'notes':['Only completed small evidence collected; first failed attempt retained separately with zero semantic denominator.','Source archives, previous-date evidence and deployed source untouched.','Original scan v1 refused a symlink in r2 before writing a receipt; v2 records the excluded link without following it.']})
 for rel,meta in COPIED.items():assert sha(OUT/rel)==meta['sha256']==sha(Path(meta['source']))
 paths=[p for p in sorted(OUT.rglob('*')) if p.is_file()];assert not any(p.is_symlink() for p in OUT.rglob('*'))
 (OUT/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(OUT))+'\n' for p in paths))
 print(json.dumps({'archive':str(OUT),'files':len(paths)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'manifest_sha256':sha(OUT/'SHA256SUMS'),'summary_sha256':sha(OUT/'summary.json'),'receipt_sha256':sha(OUT/'collection-receipt.json'),'exclusions':len(OMITTED)}))
if __name__=='__main__':main()
