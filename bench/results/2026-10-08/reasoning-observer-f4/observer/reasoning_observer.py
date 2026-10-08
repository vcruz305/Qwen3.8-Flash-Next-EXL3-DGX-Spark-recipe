"""Opt-in diagnostic observer for one synthetic Qwen reasoning-budget prompt.

No production source edits. Hooks record existing return values and CPU state;
there are no extra awaits or GPU tensor reads in generation. The raw response
is recorded only at the existing backend finish hook and written after the
collector returns. Never enable this observer on a public serving process.
"""
from __future__ import annotations
import asyncio
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

TABBY_HEAD='f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6'
ENGINE_HEAD='16ca20d27c0e4cce15a9bbc131e6d047065395b5'
PROMPT_SHA='881205716cd8b160080728978616516fab726294fd2f84011bc2daf5818b13ec'


def sha(data): return hashlib.sha256(data).hexdigest()


def atomic_new(path, value):
    data=(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
    fd,tmp=tempfile.mkstemp(prefix='.observer-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)  # Atomic publication; refuses existing files and symlinks.
    finally:
        os.unlink(tmp)


def plain_int(value):
    return value if isinstance(value,int) and not isinstance(value,bool) else None


def native_job(container, request_id):
    outer=container.active_job_ids.get(request_id)
    return outer,getattr(outer,'job',None)


def state_snapshot(container, request_id):
    outer,job=native_job(container,request_id)
    if job is None:return None
    forced=getattr(job,'forced_ids',None)
    phases=container.job_phases.get(request_id)
    queue=getattr(outer,'queue',None)
    return {
        'new_tokens':plain_int(getattr(job,'new_tokens',None)),
        'forced_pending':int(forced.shape[-1])-int(job.forced_index) if forced is not None else 0,
        'filters_suspended':bool(getattr(job,'filters_suspended',False)),
        'queued_results':queue.qsize() if queue is not None else None,
        'reported_reasoning_phase':getattr(phases,'reasoning',None),
    }


def pending_cpu_ids(job):
    ids=getattr(job,'forced_ids',None)
    if ids is None:return []
    if getattr(getattr(ids,'device',None),'type',None)!='cpu':
        return None  # Do not add a GPU transfer or synchronization to observe tokens.
    return ids.tolist()[0][int(job.forced_index):]


class Recorder:
    def __init__(self, directory, max_records=12, provenance=None):
        self.directory=Path(directory)
        if not 1<=max_records<=12:raise ValueError('max_records must be 1..12')
        self.directory.mkdir(mode=0o700)
        self.max_records=max_records
        self.active={}
        self.native={}
        self.count=0
        self.provenance=provenance or {}
        atomic_new(self.directory/'observer-manifest.json',{
            'schema_version':1,'diagnostic_only':True,
            'scope':'Exact synthetic prompt only. Raw backend text, existing CPU injection/phase state; no per-token GPU reads, no added generation awaits, no behavioral enforcement.',
            'prompt_sha256':PROMPT_SHA,'max_records':max_records,
            'source':self.provenance,
        })

    def begin(self, request_id, prompt, params, streaming):
        if sha(prompt.encode())!=PROMPT_SHA:return None
        if request_id in self.active:raise RuntimeError('Duplicate observed request id')
        if self.count>=self.max_records:raise RuntimeError('Bounded observer record limit exceeded')
        index=self.count;self.count+=1
        record={
            'schema_version':1,'index':index,'request_id':request_id,
            'prompt_sha256':PROMPT_SHA,'streaming_mode':bool(streaming),
            'reasoning_budget_tokens':getattr(params,'reasoning_budget_tokens',None),
            'reasoning_budget_message':getattr(params,'reasoning_budget_message',None),
            'started_monotonic_ns':time.monotonic_ns(),'events':[],
            'raw_finish':None,'collector_returned_error':False,
        }
        self.active[request_id]=record
        return record

    def event(self, record, kind, **details):
        if record is not None:
            record['events'].append({'kind':kind,'monotonic_ns':time.monotonic_ns(),**details})

    def end(self, record, exception_type=None):
        record['finished_monotonic_ns']=time.monotonic_ns()
        record['raised_exception_type']=exception_type
        self.active.pop(record['request_id'],None)
        self.native={k:v for k,v in self.native.items() if v is not record}
        atomic_new(self.directory/f"request-{record['index']:02d}.json",record)


def install_hooks(collector_module, container_class, job_class, recorder):
    originals={
        'collector':collector_module._chat_stream_collector,
        'constrain':container_class.constrain_generation_output,
        'phase':container_class.set_generation_phase,
        'finish':container_class.handle_finish_chunk,
        'pop':job_class._pop_forced_token,
    }
    signature=inspect.signature(originals['collector'])

    @wraps(originals['collector'])
    async def collector(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        record=recorder.begin(v['request_id'],v['prompt'],v['params'],v['streaming_mode'])
        if record is None:return await originals['collector'](*args,**kwargs)
        raised=None
        try:
            result=await originals['collector'](*args,**kwargs)
            record['collector_returned_error']=isinstance(result,BaseException)
            return result
        except BaseException as error:
            raised=type(error).__name__;raise
        finally:
            recorder.end(record,raised)

    @wraps(originals['constrain'])
    def constrain(self,request_id,text):
        record=recorder.active.get(request_id)
        if record is None:return originals['constrain'](self,request_id,text)
        outer,job=native_job(self,request_id)
        if job is not None:recorder.native[id(job)]=record
        recorder.event(record,'injection_requested',text=text,state=state_snapshot(self,request_id))
        result=originals['constrain'](self,request_id,text)
        recorder.event(record,'injection_result',accepted=result,
                       state=state_snapshot(self,request_id),
                       pending_cpu_ids=pending_cpu_ids(job) if job is not None else None)
        return result

    @wraps(originals['phase'])
    def phase(self,request_id,reasoning):
        record=recorder.active.get(request_id)
        if record is None:return originals['phase'](self,request_id,reasoning)
        before=state_snapshot(self,request_id)
        result=originals['phase'](self,request_id,reasoning)
        recorder.event(record,'phase_result',requested_reasoning=reasoning,accepted=result,
                       before=before,after=state_snapshot(self,request_id))
        return result

    @wraps(originals['finish'])
    def finish(self,result,request_id,full_text,label=None):
        value=originals['finish'](self,result,request_id,full_text,label)
        record=recorder.active.get(request_id)
        if record is not None:
            record['raw_finish']={
                'text':full_text,
                'metrics':{k:value.get(k) for k in ('gen_tokens','finish_reason','eos_reason','stop_str','cached_tokens','draft_accept','draft_reject')},
                'state':state_snapshot(self,request_id),
            }
            recorder.event(record,'backend_finished')
        return value

    @wraps(originals['pop'])
    def pop(self,device):
        record=recorder.native.get(id(self))
        if record is None:return originals['pop'](self,device)
        pending=pending_cpu_ids(self)
        before=plain_int(getattr(self,'new_tokens',None))
        result=originals['pop'](self,device)
        recorder.event(record,'forced_token_sampled',new_tokens_before=before,
                       token_id=pending[0] if pending else None)
        return result

    collector_module._chat_stream_collector=collector
    container_class.constrain_generation_output=constrain
    container_class.set_generation_phase=phase
    container_class.handle_finish_chunk=finish
    job_class._pop_forced_token=pop
    return originals


def verified_source(directory,expected):
    directory=Path(directory).resolve()
    actual=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    changes=subprocess.check_output(['git','-C',str(directory),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if actual!=expected or changes:raise RuntimeError('Observer source revision or clean-state mismatch')
    return {'path':str(directory),'commit':actual,'tracked_changes':changes}


def make_run_wrapper(original_run,target_main,activate):
    """Delay backend imports until main has loaded config and allocator options."""
    @wraps(original_run)
    def run(coro,*args,**kwargs):
        code=getattr(coro,'cr_code',None)
        matching=code is not None and code.co_name=='entrypoint_async' and Path(code.co_filename).resolve()==target_main
        if not matching:return original_run(coro,*args,**kwargs)
        async def traced_startup():
            try:
                activate()
            except BaseException:
                coro.close();raise
            return await coro
        return original_run(traced_startup(),*args,**kwargs)
    return run


def arm_from_environment():
    config_path=Path(os.environ['TABBY_REASONING_OBSERVER_CONFIG']).resolve()
    config=json.loads(config_path.read_text())
    tabby=Path(config['tabby_repo']).resolve();engine=Path(config['engine_repo']).resolve()
    target_main=tabby/'main.py'
    if Path(sys.argv[0]).resolve()!=target_main:
        return False
    if config.get('tabby_commit')!=TABBY_HEAD or config.get('engine_commit')!=ENGINE_HEAD:
        raise RuntimeError('Observer only supports the frozen16ca/f4 diagnostic runtime')
    maximum=int(config.get('max_records',12))
    if not 1<=maximum<=12:raise ValueError('Invalid observer cap')
    activated=False
    original_run=asyncio.run
    def activate():
        nonlocal activated
        if activated:raise RuntimeError('Observer already installed')
        activated=True
        # config and cudaMallocAsync handling have already run in main.entrypoint.
        tabby_source=verified_source(tabby,TABBY_HEAD)
        engine_source=verified_source(engine,ENGINE_HEAD)
        cc=importlib.import_module('endpoints.OAI.utils.chat_completion')
        model=importlib.import_module('backends.exllamav3.model')
        job=importlib.import_module('exllamav3.generator.job')
        if Path(model.__file__).resolve()!=tabby/'backends/exllamav3/model.py':
            raise RuntimeError('Unexpected Tabby import path')
        if Path(job.__file__).resolve()!=engine/'exllamav3/generator/job.py':
            raise RuntimeError('Unexpected engine import path')
        provenance={
            'tabby':tabby_source,'engine':engine_source,
            'observer_sha256':sha(Path(__file__).read_bytes()),
            'sitecustomize_sha256':sha(Path(__file__).with_name('sitecustomize.py').read_bytes()),
            'config_sha256':sha(config_path.read_bytes()),
        }
        recorder=Recorder(config['output_dir'],maximum,provenance)
        install_hooks(cc,model.ExllamaV3Container,job.Job,recorder)
        asyncio.run=original_run  # Only this startup call needed the wrapper.
    asyncio.run=make_run_wrapper(original_run,target_main,activate)
    return True
