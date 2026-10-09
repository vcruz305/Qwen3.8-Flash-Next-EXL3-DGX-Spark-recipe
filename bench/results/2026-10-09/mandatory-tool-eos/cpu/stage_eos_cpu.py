"""Create only new exact-source runtime views; run CUDA-hidden CPU/check gates."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,traceback
ROOT=Path('/home/cruzspark/qwen-followup-20261009')
BASE=Path('/home/cruzspark/qwen38-exl3-20261008')
BUNDLE=ROOT/'tabby-eos-fd8-combined.bundle'
OUT=ROOT/'results/eos-candidate-cpu'
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest()=='5203304443bb1383297ad5edb8c86ed0b1835e57b28424f72544089e57622a7f'
OUT.mkdir()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
state={'state':'running','started_at_utc':now(),'scope':'New source views plus CUDA-hidden CPU tests and setup --check only; no GPU/API/lifecycle/model operations','results':[]}
def save():
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(OUT/'status.json')
def check(cmd,**kw):return subprocess.check_output(cmd,text=True,**kw).strip()
save()
try:
 assert check(['git','-C',str(BASE/'exllamav3'),'rev-parse','HEAD'])=='24f0dece34f09c8d1e2359d6b3b3f7befef7331b'
 for name,ref,branch in [('eos-candidate-runtime','fd8cbeb1fbb3cec4d2141558d5cfd63a500d50c4','followup/mandatory-tool-eos-20261009'),('literal-eos-candidate-runtime','f4aadf114b0044fa8cbe1b50241dc80ea7d61583','followup/literal-plus-eos-20261009')]:
  runtime=ROOT/name;runtime.mkdir()
  (runtime/'exllamav3').symlink_to(BASE/'exllamav3',target_is_directory=True)
  (runtime/'venv').symlink_to(BASE/'venv',target_is_directory=True)
  with (OUT/(name+'-stage.log')).open('x') as log:
   subprocess.run(['git','clone','--no-hardlinks','--no-checkout',str(BASE/'tabbyAPI'),str(runtime/'tabbyAPI')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
   subprocess.run(['git','-C',str(runtime/'tabbyAPI'),'fetch',str(BUNDLE),'refs/heads/'+branch],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
   subprocess.run(['git','-C',str(runtime/'tabbyAPI'),'checkout','--detach',ref],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
  assert check(['git','-C',str(runtime/'tabbyAPI'),'status','--porcelain'])==''
  row={'runtime':str(runtime),'tabby_commit':ref,'tabby_tree':check(['git','-C',str(runtime/'tabbyAPI'),'rev-parse','HEAD^{tree}']),'engine_commit':'24f0dece34f09c8d1e2359d6b3b3f7befef7331b','started_at_utc':now()};state['results'].append(row);save()
  env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME','TABBY_STRINGS_OBSERVER_CONFIG')}
  env.update(CUDA_VISIBLE_DEVICES='',RECIPE_HOME=str(runtime),VENV=str(runtime/'venv'),EXL3_SRC=str(runtime/'exllamav3'),TABBY_DIR=str(runtime/'tabbyAPI'),TABBY_REF=ref,EXL3_REF=row['engine_commit'])
  tests=sorted(str(p.relative_to(runtime/'tabbyAPI')) for p in (runtime/'tabbyAPI/tests').glob('test_*.py'))
  with (OUT/(name+'-pytest.log')).open('x') as log:
   result=subprocess.run([str(runtime/'venv/bin/python'),'-m','pytest','-q',*tests],cwd=runtime/'tabbyAPI',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
  row['pytest_exit_code']=result.returncode;save();assert result.returncode==0
  with (OUT/(name+'-setup-check.log')).open('x') as log:
   result=subprocess.run(['bash',str(ROOT/'recipe-gpu-lock/exllamav3-tabby/setup.sh'),'--check'],cwd=ROOT/'recipe-gpu-lock',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  row['setup_check_exit_code']=result.returncode;assert result.returncode==0
  assert check(['git','-C',str(runtime/'tabbyAPI'),'status','--porcelain'])==''
  row['finished_at_utc']=now();row['clean']=True;save()
 state['state']='completed';state['passed']=True
except BaseException as e:
 state['state']='failed';state['passed']=False;state['error_type']=type(e).__name__;state['error']=str(e);traceback.print_exc()
finally:
 state['finished_at_utc']=now();save()
