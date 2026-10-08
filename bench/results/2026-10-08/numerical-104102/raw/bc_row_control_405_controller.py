from pathlib import Path
import datetime as dt,hashlib,json,os,socket,subprocess,traceback
ROOT=Path('/home/cruzspark/qwen-overnight-20261008')
STATUS=ROOT/'results/bc-row-control-405-b532-status.json'
with STATUS.open('x') as f:json.dump({'state':'claimed'},f)
record={'state':'starting','jobs':[]}
base=dict(EXL3_INT8_GEMV='0',EXL3_GR_INT8='1',EXL3_MOE_COOP_WIDE='1',EXL3_MTP_HEAD_N='65536',EXL3_DRAFT_CONFIDENCE='0.6')
new=dict(base,EXL3_GDN_PROJ_FP32='1',EXL3_GDN_CONV_TOKEN_MAJOR='0',EXL3_GDN_CONV_BF16_PRODUCT='1',EXL3_ATTN_DECODE_LEGACY_SPLITS='1',EXL3_MOE_COOP_MIXEDK='0',EXL3_MOE_MIXEDK_NOSYNC='1')
def save(state,**kw):
 record.update(state=state,updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw)
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(record,indent=2)+'\n');p.replace(STATUS)
try:
 assert json.loads((ROOT/'results/quality-q8-baseline/status.json').read_text())['state']=='completed'
 with socket.socket() as s:assert s.connect_ex(('127.0.0.1',8899))!=0
 script=ROOT/'spark_bc_row_control.py'
 assert hashlib.sha256(script.read_bytes()).hexdigest()=='5a8b5b898c8dda9cb90752ab3efd36e0c3a30d50a6bd8102615f1896e6453f3a'
 for label,runtime,sha,flags in [('baseline','/home/cruzspark/qwen38-exl3','94ba01d50a13fa9ff672473f2d0eef8b51a71e99',base),('candidate','/home/cruzspark/qwen38-exl3-20261008','b5322c98c5760105de04f7cf7c28bece17fbc78e',new)]:
  engine=runtime+'/exllamav3'
  assert subprocess.check_output(['git','-C',engine,'rev-parse','HEAD'],text=True).strip()==sha
  assert not subprocess.check_output(['git','-C',engine,'status','--porcelain'],text=True).strip()
  stem='bc-row-control-405-b532-'+label
  trace=ROOT/'results'/('prefill-moe-trace-405-b532-'+label)
  env={k:v for k,v in os.environ.items() if not k.startswith('EXL3_') and not k.startswith('EXLLAMAV3_')}
  env.update(flags,PYTHONPATH=engine)
  cmd=[runtime+'/venv/bin/python',str(script),'--model','/home/cruzspark/models/flashnext-exl3-4.05bpw','--trace-report',str(trace.with_suffix('.json')),'--trace-tensors',str(trace.with_suffix('.safetensors')),'--tune-cache-snapshot',str(ROOT/'results/bc-row-control-cache-20261008T095154Z/coop_autotune_v1.bin'),'--output-json',str(ROOT/'results'/(stem+'.json')),'--output-tensors',str(ROOT/'results'/(stem+'.safetensors'))]
  if label=='candidate':cmd+=['--compare-tensors',str(ROOT/'results/bc-row-control-405-b532-baseline.safetensors')]
  save('running',active=label)
  with (ROOT/'logs'/(stem+'.log')).open('x') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
  record['jobs'].append({'label':label,'exit_code':r.returncode});save('checking',active=label)
  assert r.returncode==0,label+' control failed'
  d=json.loads((ROOT/'results'/(stem+'.json')).read_text());assert d['comparison_valid'] is True
 save('completed',active=None)
except Exception as exc:save('failed',error=str(exc));traceback.print_exc();raise
