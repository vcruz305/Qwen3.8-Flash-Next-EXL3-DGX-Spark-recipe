#!/usr/bin/env python3
"""Read only small model metadata/headers on Spark and compare Oct8 identities."""
from pathlib import Path
import base64, datetime as dt, gzip, hashlib, json, subprocess
BASE=Path("/home/vcruz/src/qwen-overnight-20261008")
OUT=Path("/home/vcruz/src/qwen-followup-20261009/measurement/model-input-audit")
MODELS={
 "305":("flashnext-exl3-3.05bpw","final-305-single"),
 "405":("flashnext-exl3-4.05bpw","final-405-single"),
 "415":("flashnext-exl3-sage-4.15bpw","final-415-single"),
 "cyber387":("CYBER-FROST-3.8-EXL3-SAGE-3.87bpw","final-cyber387-single")}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 remote=r"""
from pathlib import Path
import base64,datetime as dt,glob,gzip,hashlib,json,struct
NAMES=MODEL_NAMES
def sha(raw):return hashlib.sha256(raw).hexdigest()
results={}
for label,name in NAMES.items():
 root=Path('/home/cruzspark/models')/name
 order=glob.glob(str(root/'*.safetensors'))
 identity={'config_sha256':{},'loader_glob_order':[Path(x).name for x in order],
           'weights':[],'full_weight_hashes_computed':False}
 metadata={}
 for name in ('config.json','tokenizer.json','tokenizer_config.json','generation_config.json',
              'special_tokens_map.json','tabby_config.yml','model.safetensors.index.json'):
  path=root/name
  if not path.is_file():continue
  before=path.stat()
  if before.st_size>64*1024**2:raise RuntimeError('Oversize metadata: '+str(path))
  raw=path.read_bytes();after=path.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Metadata changed')
  metadata[name]={'sha256':sha(raw),'bytes':len(raw),'mtime_ns':before.st_mtime_ns,'resolved_path':str(path.resolve())}
  if name!='model.safetensors.index.json':identity['config_sha256'][name]=sha(raw)
 headers=[];seen={};duplicates={}
 for name in order:
  path=Path(name);before=path.stat()
  with path.open('rb') as handle:
   prefix=handle.read(8)
   if len(prefix)!=8:raise RuntimeError('Truncated header')
   size=struct.unpack('<Q',prefix)[0]
   if not 2<=size<=16*1024**2 or 8+size>before.st_size:raise RuntimeError('Invalid header size')
   raw=handle.read(size)
  value=json.loads(raw);after=path.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Weight changed')
  if len(raw)!=size:raise RuntimeError('Truncated header JSON')
  tensors={k:v for k,v in value.items() if k!='__metadata__'}
  counts={};mtp={}
  for key,tensor in tensors.items():
   dtype=tensor['dtype'];counts[dtype]=counts.get(dtype,0)+1
   lo,hi=tensor['data_offsets']
   if not 0<=lo<=hi<=before.st_size-8-size:raise RuntimeError('Invalid tensor offsets')
   if 'mtp' in key.lower():mtp[key]=tensor
   if key in seen:duplicates.setdefault(key,[seen[key]]).append(path.name)
   seen[key]=path.name
  headers.append({'name':path.name,'resolved_path':str(path.resolve()),'bytes':before.st_size,
       'mtime_ns':before.st_mtime_ns,'header_bytes':size,'header_sha256':sha(raw),
       'tensor_count':len(tensors),'dtype_counts':counts,'mtp_tensors':mtp})
  identity['weights'].append({'name':path.name,'path':str(path.resolve()),
             'bytes':before.st_size,'mtime_ns':before.st_mtime_ns})
 identity['weights'].sort(key=lambda x:x['name'])
 if order!=glob.glob(str(root/'*.safetensors')):raise RuntimeError('Loader enumeration changed')
 results[label]={'model_path':str(root),'identity':identity,'metadata':metadata,
    'headers_in_loader_order':headers,'duplicate_keys_in_loader_order':duplicates,
    'weight_payload_bytes_read':0}
print(base64.b64encode(gzip.compress(json.dumps({'captured_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
 'models':results,'scope':'Filesystem metadata and safetensors headers only; no API, CUDA or weight payload reads'}).encode())).decode())
""".replace("MODEL_NAMES",repr({k:v[0] for k,v in MODELS.items()}))
 reply=subprocess.run(['bash',str(BASE/'spark-ssh'),'python3','-'],input=remote,text=True,capture_output=True,timeout=60,check=True)
 raw=gzip.decompress(base64.b64decode(reply.stdout.strip()));current=json.loads(raw)
 (OUT/'current-models.json').write_bytes(raw+b'\n')
 compared={}
 for label,(_,directory) in MODELS.items():
  path=BASE/'recipe/bench/results/2026-10-08/final-api-validation/reports'/directory/'result.json'
  old=json.loads(path.read_text())['model_identity'];now=current['models'][label]['identity']
  original={v['name']:v for v in old['weights']};fresh={v['name']:v for v in now['weights']}
  oldconfig=old['config_sha256'];newconfig=now['config_sha256']
  compared[label]={'historical_result':str(path),'historical_result_sha256':sha(path.read_bytes()),
    'model_path':current['models'][label]['model_path'],'exact_saved_identity_equal':old==now,
    'config_differences':{k:{'old':oldconfig.get(k),'current':newconfig.get(k)} for k in oldconfig.keys()|newconfig.keys() if oldconfig.get(k)!=newconfig.get(k)},
    'added_weight_files':sorted(fresh.keys()-original.keys()),'removed_weight_files':sorted(original.keys()-fresh.keys()),
    'changed_weight_metadata':{k:{'old':original[k],'current':fresh[k]} for k in original.keys()&fresh.keys() if original[k]!=fresh[k]},
    'loader_order_equal':old['loader_glob_order']==now['loader_glob_order'],
    'historical_header_hashes_available':False,
    'qualification':'Matching saved size/mtime/path is not a proof of unchanged weight payload; historical raw header hashes were not in these snapshots.'}
 report={'captured_at_utc':current['captured_at_utc'],'script_sha256':sha(Path(__file__).read_bytes()),'models':compared,
   'all_saved_identities_equal':all(x['exact_saved_identity_equal'] for x in compared.values())}
 (OUT/'comparison.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 (OUT/'audit_model_inputs.py').write_bytes(Path(__file__).read_bytes())
 paths=sorted(x for x in OUT.iterdir() if x.is_file())
 (OUT/'SHA256SUMS').write_text(''.join(sha(x.read_bytes())+'  '+x.name+'\n' for x in paths))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
