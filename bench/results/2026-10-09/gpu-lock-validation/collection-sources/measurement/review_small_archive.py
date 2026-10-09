#!/usr/bin/env python3
"""Bounded byte scan and source-manifest check; no archived module execution."""
import argparse,ast,datetime,hashlib,json,re
from pathlib import Path
PATTERN_SOURCE_SHA='74d8efcd9eb67c332737e0d47d5760b8ab94b321a2a120a2d83c758ba84ec88c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--pattern-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert sha(a.pattern_source)==PATTERN_SOURCE_SHA
 tree=ast.parse(a.pattern_source.read_text())
 assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='patterns' for x in n.targets))
 patterns=ast.literal_eval(assignment.value)
 assert isinstance(patterns,dict) and all(isinstance(k,str) and isinstance(v,bytes) for k,v in patterns.items())
 source=json.loads(a.source_manifest.read_text());expected=source['files'];inventory={};hits=[]
 entries=sorted(a.archive.rglob('*'))
 excluded=[{'path':str(p.relative_to(a.archive)),'reason':'symlink excluded without following','target':str(p.readlink())} for p in entries if p.is_symlink()]
 paths=[p for p in entries if p.is_file() and not p.is_symlink()]
 for path in paths:
  before=path.stat();assert before.st_size<10*1024*1024
  raw=path.read_bytes();after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  name=str(path.relative_to(a.archive));h=hashlib.sha256(raw).hexdigest()
  assert name in expected and h==expected[name]['sha256'] and len(raw)==expected[name]['bytes']
  inventory[name]={'bytes':len(raw),'sha256':h}
  for kind,pattern in patterns.items():
   for match in re.finditer(pattern,raw):hits.append({'path':name,'line':raw[:match.start()].count(b'\n')+1,'byte_offset':match.start(),'type':kind})
 assert set(inventory)==set(expected)
 result={'schema_version':1,'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'scope':'Read-only local file manifest verification plus bounded credential-pattern/key-name scan. No API/model/GPU/lifecycle operations, no saved semantic result changes.',
  'archive':str(a.archive),'source_manifest_sha256':sha(a.source_manifest),'review_source_sha256':sha(Path(__file__)),
  'pattern_source_sha256':PATTERN_SOURCE_SHA,'file_count':len(inventory),'total_bytes':sum(v['bytes'] for v in inventory.values()),
  'source_files':inventory,'excluded_symlinks':excluded,'candidate_count':len(hits),'candidates':hits,'pattern_types':list(patterns),
  'review_passed':not hits,'limits':'Bounded recognizers cannot prove absence of arbitrary unlabeled secrets. Matched values/snippets and value hashes are never emitted. The scan does not infer semantic model success.'}
 with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'output':str(a.output),'sha256':sha(a.output),'files':len(inventory),'bytes':result['total_bytes'],'candidate_count':len(hits),'review_passed':not hits}))
 return 0 if not hits else 1
if __name__=='__main__':raise SystemExit(main())
