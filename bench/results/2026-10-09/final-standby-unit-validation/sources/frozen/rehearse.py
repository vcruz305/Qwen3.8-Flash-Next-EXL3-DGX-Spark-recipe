#!/usr/bin/env python3
"""Root-scheduled service failure/rollback rehearsal; never installs or updates sources.

Run only after the final user unit, matching exact sources and measured service.env
are installed. This starts the unit, validates one controlled MainPID failure,
briefly serves the preserved original runtime, and returns to the new unit. Every
artifact directory is exclusive. No sudo, system-wide service, or venv move occurs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME_PATH = Path('/home/cruzspark')
WORK = HOME_PATH / 'qwen-overnight-20261008'
NEW = HOME_PATH / 'qwen38-exl3-20261008'
OLD = HOME_PATH / 'qwen38-exl3'
RECIPE = HOME_PATH / 'qwen-spark-recipe'
UNIT = 'qwen38-exl3.service'
STATE_BASE = HOME_PATH / '.local/state/qwen38-exl3/runs'
ALIAS = 'Qwen3.8-Flash-Next-EXL3'
BASE = 'http://127.0.0.1:8899/v1'
HELPER_SHA = '48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586'
UNIT_PROPS = ('LoadState','ActiveState','SubState','MainPID','InvocationID','NRestarts',
              'ControlGroup','FragmentPath','ExecMainStartTimestamp','ExecMainExitTimestamp','Result')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(command, timeout=30):
    return subprocess.check_output(command, text=True, timeout=timeout).strip()


def ctl(*args, timeout=110):
    return capture(['systemctl','--user',*args], timeout)


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_helpers():
    path = WORK / 'spark_experiment_controller.py'
    if sha(path) != HELPER_SHA:
        raise ValueError('Frozen ownership/identity helper changed')
    module = types.ModuleType('service_frozen_helpers'); module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)  # noqa: S102 - exact-hash frozen local helper
    return module


def read_env(path):
    values = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, separator, value = line.partition('=')
        if not separator or not re.fullmatch(r'[A-Z_][A-Z_0-9]*', key) or key in values:
            raise ValueError('Use unique simple NAME=value assignments in service.env')
        words = shlex.split(value)
        if len(words) > 1 or any(c in value for c in ('$','`')):
            raise ValueError('Use literal service.env values, not shell expressions')
        values[key] = words[0] if words else ''
        if 'PENDING' in values[key]:
            raise ValueError('Measured service.env values are still pending')
    return values


def exact_repo(path, revision, helpers, clean=True):
    info = helpers.git_identity(path)
    if info['commit'] != revision or (clean and info['tracked_changes']):
        raise ValueError('Unexpected or modified source checkout: ' + str(path))
    return info


def unit_status():
    text = ctl('show', UNIT, *['--property=' + name for name in UNIT_PROPS])
    return dict(line.split('=',1) for line in text.splitlines() if '=' in line)


def same_start(a, b):
    return all(a.get(k) == b.get(k) for k in ('MainPID','InvocationID','NRestarts'))


def verify_service_env(settings):
    if sha(settings['__service_env_path']) != settings['__service_env_sha256']:
        raise RuntimeError('Installed service.env changed during the rehearsal')


def check_measured_environment(env, settings, helpers):
    for key, expected in settings.items():
        if helpers.tuning_key(key) or key in ('HOST','PORT','DISABLE_AUTH','SERVED_NAME','PYTHON_BIN','TABBY_REF'):
            if env.get(key) != expected:
                raise RuntimeError('MainPID differs from the measured service setting: ' + key)


def unit_owner(status, settings, helpers):
    pid = int(status.get('MainPID','0'))
    invocation = status.get('InvocationID','')
    if status.get('ActiveState') != 'active' or pid <= 0 or not re.fullmatch(r'[0-9a-f]{32}', invocation):
        raise RuntimeError('Unit has no active, identified MainPID')
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError('MainPID belongs to a different account')
    env = dict(item.decode().split('=',1) for item in (proc/'environ').read_bytes().split(b'\0') if item and b'=' in item)
    verify_service_env(settings)
    check_measured_environment(env,settings,helpers)
    paths = {'QWEN_RECIPE_DIR':str(RECIPE),'RECIPE_HOME':str(NEW),'VENV':str(NEW/'venv'),
             'EXL3_SRC':str(NEW/'exllamav3'),'TABBY_DIR':str(NEW/'tabbyAPI'),
             'STATE_DIR':str(STATE_BASE/invocation),'MODEL_DIR':settings['MODEL_DIR']}
    if env.get('INVOCATION_ID') != invocation or any(env.get(k) != v for k,v in paths.items()):
        raise RuntimeError('MainPID environment does not identify the selected service/runtime/state')
    command = (proc/'cmdline').read_bytes().decode().split('\0')
    if str(NEW/'tabbyAPI/main.py') not in command or str(STATE_BASE/invocation/'config.yml') not in command:
        raise RuntimeError('MainPID is not the selected recipe server')
    groups = [line.split(':',2)[2] for line in (proc/'cgroup').read_text().splitlines()]
    if status.get('ControlGroup') not in groups:
        raise RuntimeError('MainPID is outside the unit control group')
    helpers.require_owned_listener(pid)
    return paths


def basic_api(client, response_model):
    models = client.json_request(BASE, '/models', timeout=5)
    if [row.get('id') for row in models.get('data',[])] != [ALIAS]:
        raise ValueError('Unexpected advertised model set')
    results = []
    for streaming in (False, True):
        payload = {'model':ALIAS,'messages':[{'role':'user','content':'Reply with the single word READY.'}],
                   'max_tokens':32,'temperature':0,'top_k':1,'top_p':1.0,'seed':0,
                   'stream':streaming,'template_vars':{'enable_thinking':False},
                   'chat_template_kwargs':{'enable_thinking':False}}
        if streaming:
            payload['stream_options'] = {'include_usage':True}
        result = client.perform(BASE, None, payload, timeout=90, expected_response_model=response_model)
        if result['finish_reason'] not in ('stop','length') or result['completion_tokens'] <= 0:
            raise ValueError('Basic generation did not produce a completed response')
        if not (result['message'].get('content') or '').strip() or result['message'].get('tool_calls'):
            raise ValueError('Basic generation did not return plain assistant content')
        results.append(client.public_result(result, include_message=True))
    return {'models':models,'requests':results,'scope':'Basic transport/generation check, not a semantic benchmark'}


def ready_new(phase, args, settings, helpers, client, after=None):
    verify_service_env(settings)
    destination = args.output / phase; destination.mkdir()
    start = time.monotonic(); first = None
    while time.monotonic() - start < args.ready_timeout:
        verify_service_env(settings)
        status = unit_status()
        if status.get('ActiveState') in ('failed','inactive'):
            raise RuntimeError('Unit failed or stopped before readiness')
        if int(status.get('MainPID','0')) > 0 and status.get('InvocationID'):
            if first is None:
                first = status
            elif not same_start(first,status):
                raise RuntimeError('Unexpected extra service restart while waiting for readiness')
        try:
            paths = unit_owner(status,settings,helpers)
            models = client.json_request(BASE,'/models',timeout=2)
            if [r.get('id') for r in models.get('data',[])] != [ALIAS]:
                raise ValueError('Service advertised the wrong model')
            break
        except (OSError,ValueError,RuntimeError,client.ApiError):
            time.sleep(1)
    else:
        raise TimeoutError('Service readiness timed out')
    if after is not None:
        if status['InvocationID'] == after['InvocationID'] or status['MainPID'] == after['MainPID']:
            raise RuntimeError('Controlled failure did not produce a fresh MainPID/invocation')
        if int(status['NRestarts']) != int(after['NRestarts']) + 1:
            raise RuntimeError('Expected exactly one automatic restart')
    state = Path(paths['STATE_DIR'])
    deployment = json.loads((state/'deployment.json').read_text())
    for key,expected in (('engine',args.engine_ref),('server',args.tabby_ref),('recipe',args.recipe_ref)):
        if deployment[key]['commit'] != expected or deployment[key].get('tracked_changes'):
            raise ValueError('Deployment source mismatch: ' + key)
    if deployment['model']['resolved_path'] != str(Path(settings['MODEL_DIR']).resolve()):
        raise ValueError('Deployment model path mismatch')
    if json.loads((state/'start.json').read_text())['invocation_id'] != status['InvocationID']:
        raise ValueError('Per-start state belongs to another invocation')
    record = {'phase':phase,'recorded_at_utc':now(),'unit':status,'state_dir':str(state),
              'load_wait_seconds':time.monotonic()-start,'deployment':deployment,
              'api':basic_api(client,ALIAS)}
    final = unit_status()
    if not same_start(status,final):
        raise RuntimeError('Service restarted or exited during the API check')
    unit_owner(final,settings,helpers)
    record['after_api_unit'] = final
    atomic(destination/'record.json',record)
    (destination/'config.yml').write_bytes((state/'config.yml').read_bytes())
    with (destination/'journal.log').open('x') as log:
        subprocess.run(['journalctl','--user','-u',UNIT,'_SYSTEMD_INVOCATION_ID='+status['InvocationID'],
                        '--no-pager','-n','500','-o','short-iso'],stdout=log,stderr=subprocess.STDOUT,timeout=20,check=True)
    return record


def stopped(helpers):
    status = unit_status()
    if status.get('ActiveState') not in ('inactive','failed') or int(status.get('MainPID','0')):
        raise RuntimeError('Unit did not release its MainPID')
    helpers.require_free_port()
    return status


def old_environment(output, plan, helpers):
    # Original serve.sh treats DRY_RUN=0 as a dry run; omit the variable entirely.
    env = {k:v for k,v in os.environ.items() if not helpers.tuning_key(k) and k not in helpers.CONTROLLED
           and k not in ('API_KEY','INVOCATION_ID','JOURNAL_STREAM','NOTIFY_SOCKET','DYNAMIC_DRAFT','QWEN_RECIPE_DIR',
                         'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV')}
    env.update(RECIPE_HOME=str(OLD),VENV=str(OLD/'venv'),EXL3_SRC=str(OLD/'exllamav3'),TABBY_DIR=str(OLD/'tabbyAPI'),
        STATE_DIR=str(output/'old-state'),MODEL_DIR=str(HOME_PATH/'models/flashnext-exl3-3.05bpw'),
        PROFILE='single',NGRAM_RAM='true',MAX_SEQ_LEN='262144',CACHE_SIZE='262144',MAX_BATCH_SIZE='1',
        DRAFT_NUM_TOKENS='5',VISION='false',HOST='127.0.0.1',PORT='8899',DISABLE_AUTH='true',SERVED_NAME=ALIAS,
        EXL3_REF=plan['original_engine_ref'],EXL3_MIN_VERSION='1.5.1',TABBY_REF=plan['original_tabby_ref'],
        EXL3_INT8_GEMV='0',EXL3_GR_INT8='1',EXL3_MOE_COOP_WIDE='1',EXL3_MTP_HEAD_N='65536',EXL3_DRAFT_CONFIDENCE='0.6',
        BIGCORES='5-9,15-19',CUDA_HOME='/usr/local/cuda',TORCH_CUDA_ARCH_LIST='12.1',PYTHONUNBUFFERED='1')
    env.pop('DRY_RUN',None)
    return env


def stop_old(process, token, helpers):
    signals = []
    for sig,limit in ((signal.SIGTERM,90),(signal.SIGKILL,5)):
        if not helpers.group_members(process,token):
            break
        os.killpg(process.pid,sig); signals.append(sig.name)
        deadline=time.monotonic()+limit
        while helpers.group_members(process,token) and time.monotonic()<deadline:
            process.poll();time.sleep(.2)
    if helpers.group_members(process,token):
        raise RuntimeError('Owned original server group did not stop; do not overlap another load')
    return {'exit_code':process.wait(timeout=5),'signals':signals}


def check_source_resolution(env, engine_ref, tabby_ref):
    # Exact checkout/deployment identities are checked independently. The recipe
    # intentionally tracks Tabby main; that label is not a tested commit claim.
    if env.get('EXL3_REF') != engine_ref:
        raise ValueError('Resolved engine ref must match the qualified runtime head')
    if env.get('TABBY_REF') not in ('main', tabby_ref):
        raise ValueError('Use the recipe Tabby main policy or the qualified exact ref')
    return {key: env.get(key) for key in ('EXL3_REF', 'TABBY_REF')}


def preflight(args, helpers):
    plan=json.loads((HERE/'plan.json').read_text())
    if os.getuid()!=1000 or Path.home()!=HOME_PATH:
        raise ValueError('Run this prepared procedure as cruzspark on Spark')
    if OLD.is_symlink() or OLD.resolve()!=OLD or NEW.resolve()!=NEW or RECIPE.resolve()!=RECIPE:
        raise ValueError('Preserve the original canonical runtime and use the explicit dated runtime/final recipe')
    sources={key:exact_repo(path,ref,helpers) for key,path,ref in (
        ('recipe',RECIPE,args.recipe_ref),('engine',NEW/'exllamav3',args.engine_ref),('server',NEW/'tabbyAPI',args.tabby_ref),
        ('old_engine',OLD/'exllamav3',plan['original_engine_ref']),('old_server',OLD/'tabbyAPI',plan['original_tabby_ref']))}
    sources['old_recipe']=helpers.git_identity(args.old_recipe)
    if sources['old_recipe']['commit']!=plan['original_recipe_ref']:
        raise ValueError('Use the preserved original recipe revision')
    for name,expected in plan['original_launcher_sha256'].items():
        if sha(args.old_recipe/name)!=expected:
            raise ValueError('Original launcher/config bytes differ: '+name)
    config=HOME_PATH/'.config/qwen38-exl3/service.env'
    if config.stat().st_mode & 0o077:
        raise ValueError('Protect service.env with mode0600')
    config_sha=sha(config)
    settings=read_env(config)
    if sha(config)!=config_sha:
        raise ValueError('service.env changed while reading the preflight')
    settings.update(__service_env_path=str(config),__service_env_sha256=config_sha)
    sources['service_environment']={'path':str(config),'sha256':config_sha}
    required={'QWEN_RECIPE_DIR':str(RECIPE),'RECIPE_HOME':str(NEW),'VENV':str(NEW/'venv'),
              'EXL3_SRC':str(NEW/'exllamav3'),'TABBY_DIR':str(NEW/'tabbyAPI'),'STATE_DIR':str(STATE_BASE),
              'HOST':'127.0.0.1','PORT':'8899','DISABLE_AUTH':'true','SERVED_NAME':ALIAS}
    if any(settings.get(k)!=v for k,v in required.items()):
        raise ValueError('Installed service.env differs from the selected paths/local service design')
    if settings.get('PROFILE') not in ('single','concurrent') or settings.get('NGRAM_RAM') not in ('true','false'):
        raise ValueError('Use the explicitly measured profile and PLE placement')
    if not Path(settings['MODEL_DIR'],'config.json').is_file():
        raise ValueError('Selected model is missing')
    for name,destination in (
        ('qwen38-exl3.service.example',HOME_PATH/'.config/systemd/user'/UNIT),
        ('10-invocation-state.conf',HOME_PATH/'.config/systemd/user'/f'{UNIT}.d/10-invocation-state.conf'),
        ('qwen38-exl3-start.sh',HOME_PATH/'.local/libexec/qwen38-exl3-start.sh')):
        if sha(HERE/name)!=sha(destination):
            raise ValueError('Installed unit/start wrapper differs from reviewed preparation: '+name)
    env=helpers.resolved_env(RECIPE,NEW,{'env':{},'model_path':settings['MODEL_DIR']})
    sources['recipe_ref_resolution']=check_source_resolution(env,args.engine_ref,args.tabby_ref)
    if 'Linger=yes' not in capture(['loginctl','show-user',str(os.getuid()),'-p','Linger']):
        raise ValueError('Expected existing user lingering is not enabled')
    status=unit_status()
    if status.get('LoadState')!='loaded':
        raise ValueError('Install and daemon-reload the reviewed user unit first')
    if status.get('ActiveState') in ('inactive','failed'):
        helpers.require_free_port()
    return plan,settings,sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('engine-ref','tabby-ref','recipe-ref'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old-recipe',type=Path,default=WORK/'recipe')
    parser.add_argument('--ready-timeout',type=float,default=600)
    args=parser.parse_args()
    if any(not re.fullmatch(r'[0-9a-f]{40}',getattr(args,k)) for k in ('engine_ref','tabby_ref','recipe_ref')):
        parser.error('Pass full exact qualified source SHAs')
    args.output=args.output.expanduser().resolve(); args.output.mkdir(parents=True,exist_ok=False)
    os.chmod(args.output,0o700);os.umask(0o077)
    port_lock=Path(f'/tmp/qwen-experiment-8899-{os.getuid()}.lock').open('a')  # noqa: SIM115 - held for the whole rehearsal
    fcntl.flock(port_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    helpers=load_helpers(); plan,settings,sources=preflight(args,helpers)
    sys.path.insert(0,str(RECIPE/'bench'))
    import api_client as client
    record={'started_at_utc':now(),'sources':sources,'passed':False,'phases':{},'controller_sha256':sha(__file__)}
    def save(): atomic(args.output/'result.json',record)
    def interrupt(signum,_frame): raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    old=None;old_log=None;token=uuid.uuid4().hex;restore=False;cleanup_ok=True
    save()
    try:
        verify_service_env(settings)
        ctl('start',UNIT)
        initial=ready_new('01-initial',args,settings,helpers,client)
        record['phases']['initial']=initial['unit'];save()
        if not same_start(initial['unit'],unit_status()):
            raise RuntimeError('Unit changed before the controlled failure')
        unit_owner(initial['unit'],settings,helpers)
        ctl('kill','--kill-whom=main','--signal=SIGKILL',UNIT)
        # Restart=on-failure imposes a10s delay; do not issue a manual start.
        deadline=time.monotonic()+args.ready_timeout
        while time.monotonic()<deadline:
            status=unit_status()
            if status.get('InvocationID') and status['InvocationID']!=initial['unit']['InvocationID'] and int(status.get('MainPID','0')):
                break
            time.sleep(1)
        else: raise TimeoutError('No automatic restart after the controlled failure')
        restarted=ready_new('02-auto-restarted',args,settings,helpers,client,after=initial['unit'])
        record['phases']['automatic_restart']=restarted['unit'];save()
        verify_service_env(settings)
        restore=True
        ctl('stop',UNIT)
        record['new_unit_stopped_for_rollback']=stopped(helpers);save()
        env=old_environment(args.output,plan,helpers);env['QWEN_EXPERIMENT_OWNER']=token
        import_check=subprocess.run([str(OLD/'venv/bin/python'),'-c',
            'import json,pathlib,exllamav3; print(json.dumps({"module":str(pathlib.Path(exllamav3.__file__).resolve())}))'],
            cwd=args.old_recipe,env=env,capture_output=True,text=True,timeout=90,check=True)
        imported=json.loads(import_check.stdout.strip().splitlines()[-1])
        if Path(imported['module']).parent!=(OLD/'exllamav3/exllamav3').resolve():
            raise ValueError('Original venv imports a different engine installation')
        record['old_import_check']=imported;save()
        old_log=(args.output/'03-original-server.log').open('x')
        old=subprocess.Popen(['bash',str(args.old_recipe/'exllamav3-tabby/serve.sh')],cwd=args.old_recipe,
                             env=env,stdout=old_log,stderr=subprocess.STDOUT,start_new_session=True)
        record['old_process']={'pid':old.pid,'ownership_token':token,'state_dir':env['STATE_DIR'],'started_at_utc':now()};save()
        began=time.monotonic()
        while time.monotonic()-began<args.ready_timeout:
            if old.poll() is not None: raise RuntimeError('Original server exited before readiness')
            try:
                helpers.require_owned_listener(old.pid)
                client.resolve_model(BASE,None,ALIAS)
                break
            except (OSError,RuntimeError,client.ApiError):time.sleep(1)
        else: raise TimeoutError('Original server readiness timed out')
        original={'load_wait_seconds':time.monotonic()-began,'api':basic_api(client,'flashnext-exl3-3.05bpw'),
                  'config':(Path(env['STATE_DIR'])/'config.yml').read_text(),'recorded_at_utc':now()}
        if old.poll() is not None: raise RuntimeError('Original server exited during the basic API gate')
        helpers.require_owned_listener(old.pid)
        atomic(args.output/'03-original-api.json',original)
        record['phases']['original_basic_api']='passed';save()
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record the failed phase and restore service
        record['error']=f'{type(exc).__name__}: {exc}';save()
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if old is not None:
            try:
                record['old_cleanup']=stop_old(old,token,helpers)
                helpers.require_free_port()
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record the failed phase and restore service
                cleanup_ok=False;record['cleanup_error']=f'{type(exc).__name__}: {exc}'
        if old_log:old_log.close()
        if restore and cleanup_ok:
            try:
                verify_service_env(settings)
                ctl('start',UNIT)
                final=ready_new('04-returned-to-final',args,settings,helpers,client)
                record['phases']['returned_to_final']=final['unit']
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record the failed phase and restore service
                record['restore_error']=f'{type(exc).__name__}: {exc}'
        record['finished_at_utc']=now()
        record['passed']=not any(k in record for k in ('error','cleanup_error','restore_error')) and set(record['phases'])=={'initial','automatic_restart','original_basic_api','returned_to_final'}
        save()
    port_lock.close()
    print(json.dumps({'output':str(args.output),'passed':record['passed'],'phases':list(record['phases']),
                      **{k:record[k] for k in ('error','cleanup_error','restore_error') if k in record}}))
    return 0 if record['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
