#!/usr/bin/env python3
"""Read-only copy of completed small Spark evidence. Never follows symlinks."""
import argparse,base64,datetime,hashlib,json,subprocess
from pathlib import Path
OLD=Path('/home/vcruz/src/qwen-overnight-20261008')
REMOTE=r'''
from pathlib import Path
import base64,datetime,hashlib,json,os,stat
root=Path(SOURCE)
assert root.is_absolute() and not root.is_symlink()
top=json.loads((root/'result.json').read_text())
assert top.get('finished_at_utc') and top.get('state') in ('completed','failed')
files={};omitted=[];total=0
for parent,dirs,names in os.walk(root,followlinks=False):
 for name in list(dirs):
  p=Path(parent)/name
  if p.is_symlink():
   omitted.append({'path':str(p.relative_to(root)),'reason':'symlink excluded without following','target':os.readlink(p)})
   dirs.remove(name)
 for name in sorted(names):
  p=Path(parent)/name;rel=str(p.relative_to(root));st=p.lstat()
  if stat.S_ISLNK(st.st_mode):
   omitted.append({'path':rel,'reason':'symlink excluded without following','target':os.readlink(p)});continue
  assert stat.S_ISREG(st.st_mode),rel
  assert st.st_size<10*1024*1024,rel
  raw=p.read_bytes();after=p.lstat()
  assert (st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  total+=len(raw);assert total<25*1024*1024
  files[rel]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps({'source':str(root),'collected_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'omitted':omitted,'file_count':len(files),'bytes':total,'top_result_state':top['state']}))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert not a.output.exists()
 code=REMOTE.replace('SOURCE',repr(a.source),1)
 r=subprocess.run(['bash',str(OLD/'spark-ssh'),'python3','-'],input=code,text=True,capture_output=True,check=True,timeout=45)
 data=json.loads(r.stdout);out=a.output/'raw';out.mkdir(parents=True)
 for name,info in data['files'].items():
  rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
  raw=base64.b64decode(info.pop('base64'),validate=True)
  assert len(raw)==info['bytes'] and hashlib.sha256(raw).hexdigest()==info['sha256']
  dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
 data['collector_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 manifest=a.output/'remote-source-manifest.json';manifest.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'output':str(a.output),'files':data['file_count'],'bytes':data['bytes'],'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
