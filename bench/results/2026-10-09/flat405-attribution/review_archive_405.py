#!/usr/bin/env python3
"""Bounded credential recognizers and copied-file integrity checks; prints no values."""
import hashlib,json,re,tarfile
from pathlib import Path
B=Path('/home/vcruz/src/qwen-followup-20261009/publication/2026-10-09/flat405-attribution')
receipt=json.loads((B/'collection-receipt.json').read_text())
for row in receipt['files']:
 raw=(B/row['destination']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
patterns={
 'private_key':rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 'bearer_credential':rb'(?i)authorization\s*[:=]\s*[\"\x27]?bearer\s+[A-Za-z0-9_./+\-=]{12,}',
 'known_token_prefix':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,})\b',
 'aws_access_key':rb'\bAKIA[A-Z0-9]{16}\b',
 'jwt':rb'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{20,}\b',
}
sensitive={'api_key','api_token','access_token','refresh_token','password','client_secret','authorization','secret_key','admin_key'}
allowed={None,'','redacted','<redacted>','[REDACTED]'}
findings=[];files=[]
def examine(name,raw):
 files.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 for kind,pattern in patterns.items():
  for match in re.finditer(pattern,raw):findings.append({'path':name,'kind':kind,'offset':match.start()})
 if name.endswith('.json'):
  try:data=json.loads(raw)
  except (ValueError,UnicodeDecodeError):return
  def walk(v,path):
   if isinstance(v,dict):
    for k,x in v.items():
     if str(k).lower() in sensitive and not isinstance(x,(dict,list)) and x not in allowed:findings.append({'path':name,'kind':'nonempty-sensitive-json-field','json_path':path+'.'+str(k)})
     walk(x,path+'.'+str(k))
   elif isinstance(v,list):
    for i,x in enumerate(v):walk(x,path+'['+str(i)+']')
  walk(data,'$')
for p in sorted(B.rglob('*')):
 if not p.is_file() or p.name in ('release-review.json','SHA256SUMS'):continue
 assert not p.is_symlink() and p.stat().st_size<4*1024*1024
 if p.name.endswith('.tar.gz'):
  with tarfile.open(p,'r:gz') as t:
   for m in t.getmembers():
    assert m.isfile() and m.size<4*1024*1024
    examine(str(p.relative_to(B))+'::'+m.name,t.extractfile(m).read())
 else:examine(str(p.relative_to(B)),p.read_bytes())
report={'copied_identities_checked':len(receipt['files']),'scanner_scope':'Recognized credential prefixes, PEM private keys, Authorization bearer values, JWT-shaped strings and selected nonempty JSON field names. All regular archive files including unpacked tar members scanned; no credential stores accessed. No guarantee for arbitrary unlabeled secrets. Source fixtures include the intentional dummy API_KEY="secret" in a fake environment test; it is not an actual credential. Configuration uses local disable_auth=true.', 'recognizers':list(patterns),'sensitive_json_keys':sorted(sensitive),'files':files,'candidate_count':len(findings),'candidates':findings,'passed':not findings}
(B/'release-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'copied_identities_checked':len(receipt['files']),'scanned_entries':len(files),'candidate_count':len(findings),'passed':not findings}))
if findings:raise SystemExit(2)
