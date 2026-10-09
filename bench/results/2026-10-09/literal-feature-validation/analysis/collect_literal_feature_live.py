#!/usr/bin/env python3
"""Copy completed literal-feature evidence; never starts a server or calls its API."""
import argparse,base64,datetime,hashlib,json,shlex,subprocess
from pathlib import Path
NEW=Path('/home/vcruz/src/qwen-followup-20261009')
def digest(raw):return hashlib.sha256(raw).hexdigest()
def main():
 a=argparse.ArgumentParser(description=__doc__)
 a.add_argument('--remote-results',required=True)
 a.add_argument('--remote-launch',required=True)
 a.add_argument('--remote-log',required=True)
 a.add_argument('--output',type=Path,required=True)
 a.add_argument('--aborted-preflight-pid',type=int)
 args=a.parse_args()
 prefix='/home/cruzspark/qwen-followup-20261009/'
 for p in (args.remote_results,args.remote_launch,args.remote_log):
  assert p.startswith(prefix) and '..' not in Path(p).parts
 assert args.output.is_absolute() and not args.output.exists()
 request={'results':args.remote_results,'launch':args.remote_launch,'log':args.remote_log,'aborted_preflight_pid':args.aborted_preflight_pid}
 code=r"""import base64,datetime,hashlib,json,stat
from pathlib import Path
spec=json.loads(SPEC)
root=Path(spec['results']);status=json.loads((root/'result.json').read_text())
if spec.get('aborted_preflight_pid') is not None:
 assert status.get('state')=='preflight' and not Path('/proc/'+str(spec['aborted_preflight_pid'])).exists()
 assert not list(root.rglob('feature.json')) and not list(root.rglob('tools.json')) and not list(root.rglob('attempt-*'))
else:
 assert status.get('finished_at_utc') and status.get('state') in ('completed','failed','interrupted','cleanup_failed')
files=[];exclusions=[]
for p in sorted(root.rglob('*')):
 rel='live/'+str(p.relative_to(root))
 if p.is_symlink():exclusions.append({'source':str(p),'destination':rel,'reason':'symlink never followed'});continue
 if p.is_file():
  if '__pycache__' in p.parts or p.suffix=='.pyc':
   raw=p.read_bytes();exclusions.append({'source':str(p),'destination':rel,'reason':'incidental Python cache','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});continue
  files.append((p,rel))
files.extend([(Path(spec['launch']),'boundaries/launch.json'),(Path(spec['log']),'boundaries/controller.log')])
out=[];total=0
for p,rel in files:
 assert not p.is_symlink();a=p.stat();assert stat.S_ISREG(a.st_mode) and a.st_size<8*1024*1024
 raw=p.read_bytes();z=p.stat();assert (a.st_ino,a.st_size,a.st_mtime_ns)==(z.st_ino,z.st_size,z.st_mtime_ns)
 total+=len(raw);assert total<64*1024*1024
 out.append({'source':str(p),'destination':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'inode':a.st_ino,'mtime_ns':a.st_mtime_ns,'stable_before_after':True,'data':base64.b64encode(raw).decode()})
print(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':out,'exclusions':exclusions,'completed_status':status,'observed_aborted_preflight_pid':spec.get('aborted_preflight_pid')}))
"""
 code=code.replace('SPEC',repr(json.dumps(request)))
 r=subprocess.run(['bash','/home/vcruz/src/qwen-overnight-20261008/spark-ssh','python3 -c '+shlex.quote(code)],text=True,capture_output=True,check=True)
 report=json.loads(r.stdout);args.output.mkdir(parents=True)
 for row in report['files']:
  raw=base64.b64decode(row.pop('data'));assert digest(raw)==row['sha256']
  p=args.output/row['destination'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 report['scope']='Completed bounded regular-file copies only; model symlinks and incidental Python caches omitted with records. No model weights, tensor/library reads, API/GPU/service actions. Completion is not a semantic pass; original status retained. For explicitly aborted preflight, absent controller PID and absent attempt/client artifacts are recorded; no terminal status is fabricated.'
 (args.output/'live-collection-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'output':str(args.output),'files':len(report['files']),'bytes':sum(r['bytes'] for r in report['files']),'state':report['completed_status']['state'],'passed':report['completed_status'].get('passed')}))
if __name__=='__main__':main()
