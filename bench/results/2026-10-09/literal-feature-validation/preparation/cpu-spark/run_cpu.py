import subprocess, pathlib, json, datetime, os, hashlib
out=pathlib.Path(__file__).parent
runtime=pathlib.Path('/home/cruzspark/qwen-followup-20261009/literal-candidate-runtime')
recipe=pathlib.Path('/home/cruzspark/qwen-followup-20261009/recipe-gpu-lock')
py=str(runtime/'venv/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',VENV=str(runtime/'venv'),EXL3_SRC=str(runtime/'exllamav3'),TABBY_DIR=str(runtime/'tabbyAPI'),TABBY_REF='followup/literal-copy-20261009')
record={'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','-C',str(runtime/'tabbyAPI'),'rev-parse','HEAD'],text=True).strip(),'cuda_visible_devices':'','commands':[]}
try:
 for name,args,cwd in [
  ('packages',[py,'-m','pip','list','--format=json'],runtime/'tabbyAPI'),
  ('setup-check',['bash',str(recipe/'exllamav3-tabby/setup.sh'),'--check'],recipe),
  ('pytest',[py,'-m','pytest','-q',*[str(p.relative_to(runtime/'tabbyAPI')) for p in sorted((runtime/'tabbyAPI/tests').glob('test_*.py'))]],runtime/'tabbyAPI')
 ]:
  log=out/(name+'.log')
  with log.open('wb') as handle:
   p=subprocess.run(args,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=600)
  record['commands'].append({'name':name,'argv':args,'cwd':str(cwd),'returncode':p.returncode,'log':log.name,'sha256':hashlib.sha256(log.read_bytes()).hexdigest()})
 record['passed']=all(r['returncode']==0 for r in record['commands'])
except BaseException as e:
 record['passed']=False;record['error']=repr(e)
finally:
 record['end_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True)
