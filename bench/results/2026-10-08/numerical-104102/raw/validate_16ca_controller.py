#!/usr/bin/env python3
from pathlib import Path
import datetime as dt,json,os,re,socket,subprocess,traceback
ROOT=Path('/home/cruzspark/qwen-overnight-20261008');TARGET=Path('/home/cruzspark/qwen38-exl3-20261008');OUT=ROOT/'results/validate-16ca';OUT.mkdir(exist_ok=False)
status={'state':'starting','stages':[]}
def save(state,**kw):
 status.update(state=state,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw);p=OUT/'status.tmp';p.write_text(json.dumps(status,indent=2)+'\n');p.replace(OUT/'status.json')
def run(name,cmd,cwd=ROOT,extra=None,expected=None,allow_failure=False):
 save('running',active=name);env={k:v for k,v in os.environ.items() if not k.startswith('EXL3_') and not k.startswith('EXLLAMAV3_')};env.update(extra or {})
 with (OUT/(name+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
 row={'name':name,'exit_code':r.returncode};status['stages'].append(row);save('checking',active=name)
 if r.returncode and not allow_failure:raise RuntimeError(name+' failed')
 if expected:
  s=(OUT/(name+'.log')).read_text();assert re.search(r'\b'+str(expected)+r' passed\b',s) and 'skipped' not in s,name+' missing required GPU passes'
try:
 assert json.loads((ROOT/'results/setup-gemm-f4-status.json').read_text())['state']=='completed'
 assert (TARGET/'venv/.qwen38-recipe-runtime').read_text().strip()=='16ca20d27c0e4cce15a9bbc131e6d047065395b5'
 with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8899))!=0
 for mode,side in [('1','baseline'),('0','candidate')]:
  env={'PYTHONPATH':str(TARGET/'exllamav3'),'EXL3_GEMM_LEGACY_TILES':mode,'EXL3_INT8_GEMV':'0','SPARK_BC_MODEL':'/home/cruzspark/models/flashnext-exl3-4.05bpw','SPARK_BC_GOLDEN':str(ROOT/'results'/('bc-row-control-405-b532-'+side+'.safetensors')),'SPARK_BC_TUNE_SNAPSHOT':str(ROOT/'results/bc-row-control-cache-20261008T095154Z/coop_autotune_v1.bin')}
  run('gpu-policy-'+mode,[str(TARGET/'venv/bin/python'),'-m','pytest','tests/test_gemm_legacy_tiles_gpu.py','-q'],TARGET/'exllamav3',env,4)
 run('api-f4-gemm',['python3',str(ROOT/'api_f4_gemm_controller.py')],allow_failure=True)
 api=json.loads((ROOT/'results/api-f4-gemm/result.json').read_text())
 assert api['state']=='completed' and api['server_cleanup']['owned_group_empty'] and not api['server_cleanup']['unexpected_exit'],'API run or cleanup incomplete'
 status['api_passed']=api['passed'];save('checking',active='api-f4-gemm')
 for name,script,jobs,output in [('fp16-recovery','spark_quality_matrix.py','quality-16ca-recovery-jobs.json','quality-16ca-recovery'),('q8-allpacks','spark_quality_q8_matrix.py','quality-q8-16ca-candidate-jobs.json','quality-q8-16ca-candidate'),('q8-batch4','spark_quality_q8_matrix.py','quality-q8-16ca-batch4-candidate-jobs.json','quality-q8-16ca-batch4-candidate')]:
  run(name,['python3',str(ROOT/script),'--jobs',str(ROOT/jobs),'--output',str(ROOT/'results'/output)])
  d=json.loads((ROOT/'results'/output/'status.json').read_text());assert d['state']=='completed'
  status['stages'][-1]['assessments']=[{k:j.get(k) for k in ['name','core_passed','investigation_count','assessment_exit']} for j in d['jobs']]
  save('checking',active=name)
  assert all(j.get('core_passed') is True for j in d['jobs']),name+' has core numerical failures'
 save('completed',active=None,passed=status.get('api_passed') is True)
except Exception as exc:save('failed',error=str(exc));traceback.print_exc();raise
