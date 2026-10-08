"""Read-only controller review: real parsers/control flow with no network/GPU."""
import argparse
import ast
import contextlib
import io
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import signal
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'recipe/bench'))

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

batch=module('review_final_batch',ROOT/'final_api_batch.py')
wrapper=module('review_final_wrapper',ROOT/'final_api_controller.py')
original={str(p):digest(p) for p in (ROOT/'final_api_batch.py',ROOT/'final_api_controller.py',ROOT/'api_f4_gemm_controller.py',ROOT/'spark_experiment_controller.py')}
results=[]

with tempfile.TemporaryDirectory() as temporary:
    tmp=Path(temporary);model=tmp/'model';model.mkdir();setup=tmp/'setup.json';setup.write_text('{}')
    out=tmp/'commands';out.mkdir();(out/'result.json').write_text('{"server_pid":8123}')
    job={'label':'review', 'engine':'a'*40,'tabby':'b'*40,'model_path':str(model),
         'recipe':str(ROOT/'recipe'),'auto_repeats':3,
         'literal_client':str(ROOT/'reasoning_literal_smoke.py'),
         'concurrency_client':str(ROOT/'reasoning_concurrency_smoke.py'),
         'concurrency_bench':True,'long_context':[32768,240000],
         'env':{'PROFILE':'single','NGRAM_RAM':True,'CACHE_SIZE':262144,'MAX_SEQ_LEN':262144,
                'MAX_BATCH_SIZE':4,'CHUNK_SIZE':2048,'DRAFT_MODE':'mtp','DRAFT_NUM_TOKENS':5,
                'DYNAMIC_DRAFT':True,'EXL3_MOE_COOP_KSPLIT':1,'EXL3_GEMM_LEGACY_TILES':1}}
    jobfile=tmp/'job.json';jobfile.write_text(json.dumps(job))
    parent=wrapper.load_parent(ROOT/'api_f4_gemm_controller.py')
    parent.RUNTIME=ROOT/'tabbyapi-agent';parent.ROOT=ROOT
    wrapper.configure(parent,job,jobfile,out,setup)
    commands=list(parent.commands(True))
    assert [name for name,_ in commands]==['tools','sdk','resilience','auto-1','auto-2','auto-3','literal','literal-unbudgeted','concurrency','concurrency-bench','bench','long-32768','long-240000']
    class Parsed(Exception):pass
    real_parse=argparse.ArgumentParser.parse_args
    parsed=[]
    def stop_after_parse(self,*args,**kwargs):
        parsed.append(real_parse(self,*args,**kwargs));raise Parsed()
    for name,command in commands:
        command=command+['--output',str(tmp/(name+'.json'))]
        with patch.object(argparse.ArgumentParser,'parse_args',stop_after_parse),patch.object(sys,'argv',command[1:]):
            try:runpy.run_path(command[1],run_name='__main__')
            except Parsed:pass
            else:raise AssertionError('client executed beyond parsing')
    assert len(parsed)==13
    assert parent.EXPECTED['literal']==4 and parent.EXPECTED['concurrency']==52 and parent.EXPECTED['literal-unbudgeted']==4
    assert parsed[8].server_pid==8123 and parsed[8].expected_server=='b'*40 and parsed[8].include_unbudgeted
    assert parsed[6].unbudgeted is False and parsed[7].unbudgeted is True
    assert (parsed[9].streams,parsed[9].prompt_tokens,parsed[9].new_tokens,parsed[9].rounds,parsed[9].warmup,parsed[9].run_id)==('1,2,4',1024,256,3,1,'overnight-concurrency-v1')
    results.append({'name':'all_thirteen_generated_commands_parse_without_requests','passed':True})


    # Preserve every old request-producing command; only concurrency gains its opt-in flag.
    old=module('previous_final_wrapper',ROOT/'tabbyapi-diagnostics/final-wrapper-pre-natural-options/final_api_controller.py')
    prior=old.load_parent(ROOT/'api_f4_gemm_controller.py');prior.RUNTIME=parent.RUNTIME;prior.ROOT=parent.ROOT
    old.configure(prior,deepcopy(job),jobfile,out,setup)
    previous_commands=dict(prior.commands(True));new_commands=dict(commands)
    for name,command in previous_commands.items():
        assert new_commands[name]==command+(['--include-unbudgeted'] if name=='concurrency' else []),name
    results.append({'name':'prior_commands_unchanged_except_explicit_concurrency_flag','passed':True})

    # The strong setup guard must still require the exact successful completed five-step build.
    head=parent.subprocess.check_output(['git','-C',str(parent.RECIPE),'rev-parse','HEAD'],text=True).strip()
    valid_setup={'state':'completed','engine':parent.ENGINE,'tabby':parent.TABBY,'runtime':str(parent.RUNTIME),
                 'passed':True,'finished_utc':'now','recipe_commit':head,
                 'commands':[{'name':name,'exit_code':0,'finished_utc':'now'}
                             for name in ('setup','setup-check','engine-budget-cpu','tabby-cpu','tool-tokenizer')]}
    parent.check_setup(valid_setup)
    rejected=[]
    for kind in ('engine','tabby','runtime','passed','finished_utc','recipe_commit','missing_command','duplicate_command','command_exit','unfinished_command','state'):
        bad=deepcopy(valid_setup)
        if kind=='missing_command':bad['commands'].pop()
        elif kind=='duplicate_command':bad['commands'][-1]=deepcopy(bad['commands'][0])
        elif kind=='command_exit':bad['commands'][-1]['exit_code']=1
        elif kind=='unfinished_command':bad['commands'][-1].pop('finished_utc')
        elif kind=='passed':bad[kind]=False
        elif kind=='finished_utc':bad.pop(kind)
        else:bad[kind]='incorrect'
        try:parent.check_setup(bad)
        except ValueError:rejected.append(kind)
        else:raise AssertionError('setup mismatch accepted: '+kind)
    results.append({'name':'exact_setup_gate_positive_and_eleven_rejections','passed':True,'rejected':rejected})

    def cases_report(n,**extra):
        return {'cases':[{'name':str(i),'status':'pass'} for i in range(n)],'summary':{'pass':n},'finished_utc':'now',**extra}
    assert parent.summarize('literal',cases_report(4,unbudgeted=False),0)['passed']
    assert parent.summarize('literal-unbudgeted',cases_report(4,unbudgeted=True),0)['passed']
    assert not parent.summarize('literal-unbudgeted',cases_report(4,unbudgeted=False),0)['passed']
    assert not parent.summarize('literal',cases_report(4,unbudgeted=True),0)['passed']
    concurrency_report=cases_report(52,include_unbudgeted=True,expected_checks=52,client_reports=[{}]*16)
    assert parent.summarize('concurrency',concurrency_report,0)['passed']
    for key,value in [('include_unbudgeted',False),('expected_checks',48),('client_reports',[{}]*15),('cases',concurrency_report['cases'][:-1])]:
        bad=deepcopy(concurrency_report);bad[key]=value
        assert not parent.summarize('concurrency',bad,0)['passed'],key
    assert not parent.summarize('concurrency',concurrency_report,1)['passed']
    results.append({'name':'literal_mode_and_exact_52_16_report_gates','passed':True})

    bench={'completed_at_utc':'now','tag':job['label'],'model':parent.ALIAS,'base_url':parent.BASE,
           'run_id':'overnight-concurrency-v1','errors':[],
           'settings':{'concurrency':[1,2,4],'prompt_tokens_approx':1024,'max_tokens':256,'rounds':3,'warmup':1},
           'results':{str(n):{kind:[{'round_index':i,'complete':True,'errors':[],
                                   'streams':[{'stream_index':k} for k in range(n)]} for i in range(total)]
                           for kind,total in [('warmup',1),('rounds',3)]} for n in (1,2,4)}}
    assert parent.summarize('concurrency-bench',bench,0)['passed']
    rejected=[]
    for kind in ('missing_group','missing_round','extra_round','incomplete','missing_stream','duplicate_stream','wrong_index','stream_error','settings','model','run_id','unfinished','top_error'):
        bad=deepcopy(bench)
        if kind=='missing_group':bad['results'].pop('4')
        elif kind=='missing_round':bad['results']['4']['rounds'].pop()
        elif kind=='extra_round':bad['results']['4']['rounds'].append(deepcopy(bad['results']['4']['rounds'][0]))
        elif kind=='incomplete':bad['results']['4']['rounds'][0]['complete']=False
        elif kind=='missing_stream':bad['results']['4']['rounds'][0]['streams'].pop()
        elif kind=='duplicate_stream':bad['results']['4']['rounds'][0]['streams'][-1]={'stream_index':0}
        elif kind=='wrong_index':bad['results']['4']['rounds'][0]['round_index']=4
        elif kind=='stream_error':bad['results']['4']['rounds'][0]['errors']=['failure']
        elif kind=='settings':bad['settings']['max_tokens']=128
        elif kind=='unfinished':bad.pop('completed_at_utc')
        elif kind=='top_error':bad['errors']=['failure']
        else:bad[kind]='incorrect'
        assert not parent.summarize('concurrency-bench',bad,0)['passed'],kind
        rejected.append(kind)
    assert not parent.summarize('concurrency-bench',bench,1)['passed']
    results.append({'name':'fixed_concurrency_benchmark_gate_and_thirteen_report_rejections','passed':True,'rejected':rejected})

    # Actual wrapper parser runs, but the native lifecycle is replaced before any launch.
    def dummy_parent():return SimpleNamespace(run=lambda args: args.client_timeout)
    with patch.object(wrapper,'load_parent',lambda _:dummy_parent()),patch.object(wrapper,'configure',lambda *a:None):
        baseargs=['final_api_controller.py','--job',str(jobfile),'--setup',str(setup),'--output',str(tmp/'not_created')]
        previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            with patch.object(sys,'argv',baseargs):assert wrapper.main()==2700
            for limit in ('2100','2699'):
                with patch.object(sys,'argv',baseargs+['--client-timeout',limit]),contextlib.redirect_stderr(io.StringIO()):
                    try:wrapper.main()
                    except SystemExit as exc:assert exc.code==2
                    else:raise AssertionError('insufficient concurrent timeout accepted')
        finally:
            for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    results.append({'name':'actual_wrapper_default2700_and_undersized_rejection_without_launch','passed':True})

    # Invalid benchmark selectors/capacity fail during configuration.
    for kind in ('nonboolean','insufficient_slots'):
        bad=deepcopy(job)
        if kind=='nonboolean':bad['concurrency_bench']='true'
        else:bad['env']['MAX_BATCH_SIZE']=1;bad.pop('concurrency_client')
        fresh=wrapper.load_parent(ROOT/'api_f4_gemm_controller.py')
        try:wrapper.configure(fresh,bad,jobfile,out,setup)
        except ValueError:pass
        else:raise AssertionError(kind)
    results.append({'name':'concurrency_benchmark_boolean_and_capacity_checks','passed':True})

    # Existing setup/template/source validation bodies are AST-identical.
    def named_function(source,name):
        return next(node for node in ast.walk(ast.parse(source)) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name)
    current=(ROOT/'final_api_controller.py').read_text()
    previous=(ROOT/'tabbyapi-diagnostics/final-wrapper-pre-natural-options/final_api_controller.py').read_text()
    for name in ('load_parent','check_complete_setup','validate_deployment','load_helpers_with_template','api_json_with_template'):
        assert ast.dump(named_function(current,name),include_attributes=False)==ast.dump(named_function(previous,name),include_attributes=False),name
    current_batch=ast.parse((ROOT/'final_api_batch.py').read_text())
    previous_batch=ast.parse((ROOT/'tabbyapi-diagnostics/final-wrapper-pre-natural-options/final_api_batch.py').read_text())
    main_node=next(node for node in current_batch.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    guards=[node for node in main_node.body if isinstance(node,ast.If) and '3600' in ast.unparse(node.test)]
    assert len(guards)==1;main_node.body.remove(guards[0])
    assert ast.dump(current_batch,include_attributes=False)==ast.dump(previous_batch,include_attributes=False)
    results.append({'name':'setup_template_source_gates_and_entire_batch_signal_lifecycle_ast_unchanged','passed':True})

    # Batch undersize validation happens before any child process/Sampler activity.
    for field in ('concurrency_client','concurrency_bench'):
        jobsfile=tmp/(field+'-jobs.json');jobsfile.write_text(json.dumps([{'label':'bound',field:True}]))
        output=tmp/(field+'-bound')
        with patch.object(batch.subprocess,'Popen',side_effect=AssertionError('must not launch')), \
             patch.object(sys,'argv',['final_api_batch.py','--jobs',str(jobsfile),'--setup',str(setup),'--output',str(output),'--job-timeout','3599']), \
             contextlib.redirect_stderr(io.StringIO()):
            try:batch.main()
            except SystemExit as exc:assert exc.code==2
            else:raise AssertionError('undersized batch accepted')
    results.append({'name':'batch_minimum3600_for_both_concurrent_job_types','passed':True})


def batch_case(kind):
    with tempfile.TemporaryDirectory() as temporary:
        tmp=Path(temporary);jobsfile=tmp/'jobs.json';setup=tmp/'setup.json';setup.write_text('{}')
        jobsfile.write_text(json.dumps([{'label':'one'},{'label':'two'}]))
        spawned=[];samplings=[];now=[0.0]
        class Sampler:
            def __init__(self,path,interval,limit):
                assert (interval,limit)==(10,540);samplings.append(('init',str(path)))
            def tick(self,phase,pid):
                samplings.append(('tick',phase,pid))
                if kind=='cleanup_signal':signal.raise_signal(signal.SIGTERM)
        def exec_helper(code,namespace):
            builtins.exec(code,namespace);namespace['Sampler']=Sampler
        class Process:
            def __init__(self,command,**kwargs):
                assert kwargs['start_new_session'] is True
                self.pid=8400+len(spawned);self.returncode=None if kind in ('timeout','stubborn_timeout','launch_signal','cleanup_signal') else 0
                self.terminated=False;self.termination_calls=0;self.wait_calls=[];spawned.append(self)
                destination=Path(command[command.index('--output')+1]);destination.mkdir()
                value={'state':'completed','passed':True,'server_cleanup':{'owned_group_empty':True},'clients':[]}
                if len(spawned)==1:
                    if kind=='functional_failure':value['passed']=False;self.returncode=1
                    if kind=='server_cleanup':value['server_cleanup']['owned_group_empty']=False
                    if kind.startswith('concurrency'):
                        nested=destination/'concurrency.json'
                        content={'cleanup_failed':kind=='concurrency_failed','owned_client_groups_empty':kind=='concurrency_success'}
                        nested.write_text(json.dumps(content))
                        value['clients']=[{'name':'concurrency','passed':True,'report':str(nested),'report_sha256':digest(nested)}]
                        if kind=='concurrency_missing':nested.unlink()
                (destination/'result.json').write_text(json.dumps(value))
                if kind=='launch_signal' and len(spawned)==1:
                    signal.raise_signal(signal.SIGTERM)
            def poll(self):return self.returncode
            def terminate(self):
                self.terminated=True;self.termination_calls+=1
                if kind=='cleanup_signal':signal.raise_signal(signal.SIGTERM)
                if kind!='stubborn_timeout':self.returncode=-15
            def wait(self,timeout=None):
                self.wait_calls.append(timeout)
                if kind=='stubborn_timeout':raise batch.subprocess.TimeoutExpired('synthetic-child',timeout)
                assert self.returncode is not None
                return self.returncode
        def monotonic():now[0]+=.02;return now[0]
        handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            with patch.object(batch,'exec',exec_helper,create=True),patch.object(batch.subprocess,'Popen',Process), \
                 patch.object(batch.time,'monotonic',side_effect=monotonic),patch.object(batch.time,'sleep',lambda _:None), \
                 patch.object(sys,'argv',['final_api_batch.py','--jobs',str(jobsfile),'--setup',str(setup),'--output',str(tmp/'output'),'--job-timeout','.01']):
                exit_code=batch.main()
                assert {sig:signal.getsignal(sig) for sig in handlers}==handlers,'Signal handlers were not restored'
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
        record=json.loads((tmp/'output/status.json').read_text())
        if kind in ('success','concurrency_success'):
            assert exit_code==0 and len(spawned)==2 and record['passed']
        elif kind=='functional_failure':
            assert exit_code==1 and len(spawned)==2 and not record['passed']
        else:
            assert exit_code==1 and len(spawned)==1 and not record['passed']
        if kind in ('timeout','stubborn_timeout','launch_signal','cleanup_signal'):
            assert spawned[0].termination_calls==1,spawned[0].termination_calls
            assert spawned[0].wait_calls==[210],spawned[0].wait_calls
        if kind=='stubborn_timeout':assert record.get('cleanup_error')
        assert record['finished_utc']
        results.append({'name':'batch_'+kind,'passed':True,'spawned_wrappers':len(spawned)})
for kind in ('success','functional_failure','server_cleanup','concurrency_success','concurrency_failed','concurrency_missing','timeout','stubborn_timeout','launch_signal','cleanup_signal'):
    batch_case(kind)
assert original=={path:digest(path) for path in original},'Review sources changed during execution'
report={'python':sys.version,'test_sha256':digest(__file__),'scope':__doc__,'passed':all(row['passed'] for row in results),'cases':results,'source_hashes':original}
path=ROOT/'review-final-api-natural-results.json'
with path.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
print(json.dumps(report,indent=2))
