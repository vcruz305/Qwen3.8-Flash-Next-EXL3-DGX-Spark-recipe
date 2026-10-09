#!/usr/bin/env python3
"""Copy completed small Oct9 evidence locally; no imports of archived controllers."""
from pathlib import Path
import collections,datetime,hashlib,json,shutil
BASE=Path('/home/vcruz/src/qwen-followup-20261009')
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
OUT=BASE/'publication/2026-10-09'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def copy_tree(src,dst,receipt):
 for p in sorted(src.rglob('*')):
  if p.is_symlink():raise ValueError('Symlink refused: '+str(p))
  if not p.is_file():continue
  if p.stat().st_size>10*1024*1024:raise ValueError('Oversize artifact: '+str(p))
  rel=p.relative_to(src);row={'source':str(p),'destination':str((dst/rel).relative_to(OUT)),
      'bytes':p.stat().st_size,'sha256':sha(p)}
  if '__pycache__' in p.parts or p.suffix=='.pyc':
   receipt['excluded'].append({**row,'reason':'Incidental compiled Python cache; original remains untouched.'});continue
  target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
  assert sha(target)==row['sha256'];receipt['copied'].append(row)
def copy_one(src,dst,receipt):
 assert src.is_file() and not src.is_symlink() and src.stat().st_size<10*1024*1024
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 row={'source':str(src),'destination':str(dst.relative_to(OUT)),'bytes':src.stat().st_size,'sha256':sha(src)}
 assert sha(dst)==row['sha256'];receipt['copied'].append(row)
def verify_sums(directory):
 for line in (directory/'SHA256SUMS').read_text().splitlines():
  expected,name=line.split('  ',1);assert sha(directory/name)==expected

def main():
 if OUT.exists():raise ValueError('Publication root already exists; refuse overwrite')
 OUT.mkdir(parents=True)
 receipt={'schema_version':1,'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'scope':'Local copy of completed evidence only; no source/runtime/GPU/API/Spark changes.',
  'collector_source_sha256':sha(Path(__file__)),'copied':[],'excluded':[],
  'preparation_note':'One earlier local copy pass stopped on a missing optional timed_out key; corrected to the frozen assessor schema. Original live evidence was never changed or rerun.'}
 audit=OUT/'model-input-audit';literal=OUT/'literal-drafting-ablation'
 copy_tree(BASE/'measurement/model-input-audit',audit/'raw',receipt);verify_sums(audit/'raw')
 comparison=json.loads((audit/'raw/comparison.json').read_text());assert comparison['all_saved_identities_equal'] is True
 for key,row in comparison['models'].items():
  historical=Path(row['historical_result']);assert sha(historical)==row['historical_result_sha256']
  copy_one(historical,audit/'historical-reference'/f'{key}-result.json',receipt)
 write(audit/'summary.json',{'schema_version':1,'scope':'Read-only model input metadata/header audit',
  'captured_at_utc':comparison['captured_at_utc'],'all_four_saved_identities_equal':True,
  'models':{key:{'model_path':row['model_path'],'saved_identity_equal':row['exact_saved_identity_equal'],
       'loader_order_equal':row['loader_order_equal'],'historical_header_hashes_available':False,
       'historical_result_sha256':row['historical_result_sha256']} for key,row in comparison['models'].items()},
  'full_weight_payloads_read_or_hashed':False,'limitation':'Current config hashes and weight metadata/header records do not prove equality of all weight payload bytes; historical snapshots did not contain header hashes.'})
 raw=BASE/'results/literal-ablation-24f0-f650'
 review_path=BASE/'literal-archive-release-review/report.json'
 assert sha(review_path)=='ba4e148bc1a748010f9f733da5c30323860c2b67bf0408917c22672b7650bfe3'
 release=json.loads(review_path.read_text());assert release['review_passed'] and release['archive_unchanged']
 source_files={str(p.relative_to(raw)):p for p in raw.rglob('*') if p.is_file()}
 assert set(source_files)==set(release['source_files']) and len(source_files)==59
 for name,p in source_files.items():assert sha(p)==release['source_files'][name]['sha256']
 copy_tree(raw,literal/'reports',receipt)
 copy_tree(BASE/'literal-ablation-review',literal/'peer-analysis',receipt)
 copy_tree(BASE/'measurement/literal-ablation',literal/'sources/measurement/literal-ablation',receipt)
 verify_sums(literal/'sources/measurement/literal-ablation')
 copy_tree(BASE/'literal-ablation-observer-timeline',literal/'sources/literal-ablation-observer-timeline',receipt)
 manifest=json.loads((literal/'sources/literal-ablation-observer-timeline/manifest.json').read_text())
 for name,value in manifest['files'].items():assert sha(literal/'sources/literal-ablation-observer-timeline'/name)==value['sha256']
 copy_one(BASE/'analyze_literal_ablation.py',literal/'peer-analysis/analyze_literal_ablation.py',receipt)
 copy_tree(BASE/'literal-archive-release-review',literal/'release-review',receipt)
 for name in ('api_resilience.py','api_client.py'):
  copy_one(OLD/'recipe/bench'/name,literal/'sources/analysis-dependencies'/name,receipt)
 outer=json.loads((literal/'reports/result.json').read_text())
 assert outer['state']=='completed' and outer['capture_valid'] is True and outer['passed'] is False
 cells=[];total=collections.Counter();shared_payloads=None
 for item in outer['cells']:
  label=item['label'];folder=literal/'reports'/label
  record=json.loads((folder/'result.json').read_text());capture=json.loads((folder/'literal-capture.json').read_text())
  assert sha(folder/'literal-capture.json')==item['assessment_sha256']
  assert record['state']=='completed' and record['finished_at_utc'] and capture['capture_valid']
  assert record['server_cleanup']['owned_group_empty'] and not record['server_cleanup']['unexpected_exit']
  assert record['server_cleanup']['exit_code']==0
  requests=[];counts=collections.Counter();payloads=[]
  assert [r['name'] for r in record['clients']]==['literal','literal-unbudgeted']
  for client in record['clients']:
   path=folder/(client['name']+'.json');assert sha(path)==client['report_sha256']
   assert client['cleanup']['owned_group_empty'] and not client.get('timed_out')
   result=json.loads(path.read_text());assert result['finished_utc'] and len(result['cases'])==len(result['requests'])==4
   for case,request in zip(result['cases'],result['requests']):
    counts[case['status']]+=1;payloads.append(request['request'])
    response=request['body'] if not request['stream'] else [f for f in request['frames'] if f.get('usage')][-1]
    usage=response['usage'];assert usage['prompt_tokens']==348
    requests.append({'client':client['name'],'case':case['name'],'status':case['status'],
     'request_sha256':hashlib.sha256(json.dumps(request['request'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
     'prompt_tokens':usage['prompt_tokens'],'cached_prompt_tokens':usage['prompt_tokens_details']['cached_tokens'],
     'completion_tokens':usage['completion_tokens'],'finish_reason':response['choices'][0]['finish_reason']})
  if shared_payloads is None:shared_payloads=payloads
  else:assert payloads==shared_payloads
  assert dict(counts)==item['semantic_counts']==capture['original_semantic_counts'];total.update(counts)
  cells.append({'label':label,'capture_valid':True,'semantic_counts':dict(counts),'all_semantics_passed':False,
     'expected_engine':record['expected_engine'],'expected_server':record['expected_server'],
     'server_cleanup':record['server_cleanup'],'requests':requests,
     'assessment_sha256':sha(folder/'literal-capture.json'),'deployment_sha256':sha(folder/'deployment.json')})
 assert dict(total)=={'fail':15,'pass':1}
 phase=json.loads((literal/'peer-analysis/phase-summary.json').read_text())
 assert phase['capture_valid'] and not phase['semantic_passed']
 write(literal/'summary.json',{'schema_version':1,'kind':'completed_original_literal_drafting_ablation',
  'started_at_utc':outer['started_at_utc'],'finished_at_utc':outer['finished_at_utc'],
  'capture_valid':True,'semantic_counts':dict(total),'live_requests':16,'fresh_servers':2,
  'shared_visible_payloads_equal':True,'changed_model_input':False,'prompt_tokens_each':348,
  'cells':cells,'offline_semantic_replays':release['semantic_replays'],
  'native_backend_pairs_equal':16,'validated_timeline_events':release['timeline_events_validated'],
  'release_review_sha256':sha(review_path),'phase_summary_sha256':sha(literal/'peer-analysis/phase-summary.json'),
  'scope':'Original eight requests in each of two ordered cells; capture success is not model semantic success. No throughput or general tool-schema qualification.'})
 receipt['release_review']={'review_passed':True,'report_sha256':sha(review_path),
  'raw59_credential_candidate_count':release['credential_scan']['candidate_count'],
  'limits':release['credential_scan']['limits']}
 receipt['raw_source_files']=59;receipt['published_raw_files']=55
 receipt['raw_excluded_files']=[r for r in receipt['excluded'] if r['source'].startswith(str(raw)+'/')]
 assert len(receipt['raw_excluded_files'])==4
 # The source audit and peer manifest each retain their original dependency paths.
 for row in receipt['copied']:assert sha(Path(row['source']))==row['sha256']
 write(OUT/'collection-receipt.json',receipt)
 copy_one(Path(__file__),OUT/'stage_completed_evidence.py',{'copied':[]})
 print(json.dumps({'staged':str(OUT),'copied_files':len(receipt['copied']),'raw55_exact':True,'excluded_raw_pyc':4}))
if __name__=='__main__':main()
