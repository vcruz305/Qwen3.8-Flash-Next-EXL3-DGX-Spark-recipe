"""CPU invariants for a diagnostic-only observer; no engine or server import."""
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import reasoning_observer as obs

PROMPT='one synthetic observer test prompt'
DIGEST=hashlib.sha256(PROMPT.encode()).hexdigest()

class Tensor:
    def __init__(self,values,device='cpu'):
        self.values=values;self.device=SimpleNamespace(type=device);self.shape=(1,len(values))
    def tolist(self):
        if self.device.type!='cpu':raise AssertionError('Observer attempted a GPU read')
        return [self.values]


def classes(behavior='normal'):
    token=object();sentinel=object();failure=RuntimeError('original failure identity')
    class Job:
        def __init__(self):
            self.new_tokens=24;self.forced_ids=None;self.forced_index=0;self.filters_suspended=False
        def _pop_forced_token(self,device):
            self.forced_ids=None;self.forced_index=0
            return token
    class Container:
        def __init__(self):
            self.active_job_ids={};self.job_phases={}
        def constrain_generation_output(self,request_id,text):
            job=self.active_job_ids[request_id].job
            job.forced_ids=Tensor([248069]);job.filters_suspended=True
            return True
        def set_generation_phase(self,request_id,reasoning):
            self.job_phases[request_id].reasoning=reasoning
            self.active_job_ids[request_id].job.filters_suspended=False
            return True
        def handle_finish_chunk(self,result,request_id,full_text,label=None):
            return result
    mc=Container();job=Job();queue=asyncio.Queue()
    for i in range(3):queue.put_nowait(i)
    mc.active_job_ids['request-a']=SimpleNamespace(job=job,queue=queue)
    mc.job_phases['request-a']=SimpleNamespace(reasoning=True)
    async def collector(request_id,prompt,params,streaming_mode=True):
        if behavior=='raise':raise failure
        if behavior=='returned_error':return failure
        if behavior=='noop':return sentinel
        assert mc.constrain_generation_output(request_id,'</think>') is True
        assert job._pop_forced_token('unused-device') is token
        job.new_tokens+=1
        assert mc.set_generation_phase(request_id,False) is True
        data={'gen_tokens':30,'finish_reason':'stop','full_text':'unchanged-object'}
        assert mc.handle_finish_chunk(data,request_id,'thought</think>answer') is data
        return sentinel
    return SimpleNamespace(_chat_stream_collector=collector),Container,Job,mc,job,sentinel,failure


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.hashpatch=patch.object(obs,'PROMPT_SHA',DIGEST);self.hashpatch.start()
    def tearDown(self):
        self.hashpatch.stop();self.tmp.cleanup()
    def recorder(self,cap=12):return obs.Recorder(self.root/'traces',cap,{'test':True})
    def params(self):return SimpleNamespace(reasoning_budget_tokens=24,reasoning_budget_message=None)

    def test_observation_preserves_all_return_values_and_records_cpu_boundary(self):
        recorder=self.recorder();module,C,J,mc,job,sentinel,_=classes()
        obs.install_hooks(module,C,J,recorder)
        result=asyncio.run(module._chat_stream_collector('request-a',PROMPT,self.params(),False))
        self.assertIs(result,sentinel)
        report=json.loads((recorder.directory/'request-00.json').read_text())
        self.assertEqual(report['raw_finish']['text'],'thought</think>answer')
        self.assertEqual([x['kind'] for x in report['events']],['injection_requested','injection_result','forced_token_sampled','phase_result','backend_finished'])
        self.assertEqual(report['events'][0]['state']['new_tokens'],24)
        self.assertEqual(report['events'][0]['state']['queued_results'],3)
        self.assertEqual(report['events'][1]['pending_cpu_ids'],[248069])
        self.assertEqual(report['events'][2]['new_tokens_before'],24)
        self.assertEqual(report['events'][2]['token_id'],248069)
        self.assertFalse(report['streaming_mode'])
        self.assertFalse(report['collector_returned_error'])
        self.assertFalse(recorder.active);self.assertFalse(recorder.native)
        self.assertEqual((recorder.directory/'request-00.json').stat().st_mode & 0o777,0o600)

    def test_unmatched_prompt_is_not_recorded_or_exposed(self):
        recorder=self.recorder();module,C,J,mc,job,sentinel,_=classes('noop')
        obs.install_hooks(module,C,J,recorder)
        result=asyncio.run(module._chat_stream_collector('request-a','unrelated private prompt',self.params()))
        self.assertIs(result,sentinel);self.assertEqual(recorder.count,0)
        self.assertEqual(list(recorder.directory.glob('request-*.json')),[])

    def test_original_exception_identity_is_preserved(self):
        recorder=self.recorder();module,C,J,mc,job,_,failure=classes('raise')
        obs.install_hooks(module,C,J,recorder)
        try:asyncio.run(module._chat_stream_collector('request-a',PROMPT,self.params()))
        except RuntimeError as error:self.assertIs(error,failure)
        else:self.fail('Original error was swallowed')
        report=json.loads((recorder.directory/'request-00.json').read_text())
        self.assertEqual(report['raised_exception_type'],'RuntimeError')
        self.assertNotIn('original failure identity',json.dumps(report))

    def test_returned_error_and_missing_raw_finish_remain_visible(self):
        recorder=self.recorder();module,C,J,mc,job,_,failure=classes('returned_error')
        obs.install_hooks(module,C,J,recorder)
        self.assertIs(asyncio.run(module._chat_stream_collector('request-a',PROMPT,self.params())),failure)
        report=json.loads((recorder.directory/'request-00.json').read_text())
        self.assertTrue(report['collector_returned_error']);self.assertIsNone(report['raw_finish'])

    def test_record_count_is_bounded_and_duplicate_ids_rejected(self):
        recorder=self.recorder(1)
        record=recorder.begin('r',PROMPT,self.params(),True)
        with self.assertRaises(RuntimeError):recorder.begin('r',PROMPT,self.params(),True)
        recorder.end(record)
        with self.assertRaises(RuntimeError):recorder.begin('r2',PROMPT,self.params(),True)

    def test_output_claim_and_publication_refuse_overwrite_or_symlink(self):
        recorder=self.recorder()
        with self.assertRaises(FileExistsError):obs.Recorder(recorder.directory)
        p=self.root/'one.json';obs.atomic_new(p,{'x':1})
        with self.assertRaises(FileExistsError):obs.atomic_new(p,{'x':2})
        self.assertEqual(json.loads(p.read_text()),{'x':1})
        dangling=self.root/'dangling';dangling.symlink_to(self.root/'missing')
        with self.assertRaises(FileExistsError):obs.atomic_new(dangling,{'x':3})
        self.assertFalse((self.root/'missing').exists())
        self.assertEqual(list(self.root.glob('.observer-*')),[])

    def test_gpu_tensor_values_are_never_read(self):
        job=SimpleNamespace(forced_ids=Tensor([2],device='cuda'),forced_index=0)
        self.assertIsNone(obs.pending_cpu_ids(job))
        job.forced_ids=Tensor([2,3]);job.forced_index=1
        self.assertEqual(obs.pending_cpu_ids(job),[3])

    def test_no_active_job_has_no_snapshot(self):
        mc=SimpleNamespace(active_job_ids={},job_phases={})
        self.assertIsNone(obs.state_snapshot(mc,'absent'))

    def test_startup_activation_waits_until_the_matching_main_coroutine_runs(self):
        target=self.root/'main.py';events=[]
        namespace={'events':events}
        exec(compile('async def entrypoint_async():\n events.append("body")\n return 42\n',str(target),'exec'),namespace)
        wrapper=obs.make_run_wrapper(asyncio.run,target,lambda:events.append('activate'))
        coro=namespace['entrypoint_async']()
        self.assertEqual(events,[])
        self.assertEqual(wrapper(coro),42)
        self.assertEqual(events,['activate','body'])

    def test_other_coroutines_are_untouched(self):
        events=[]
        async def unrelated():return 7
        wrapper=obs.make_run_wrapper(asyncio.run,self.root/'main.py',lambda:events.append('activate'))
        self.assertEqual(wrapper(unrelated()),7)
        self.assertEqual(events,[])

    def test_failed_activation_closes_original_coroutine(self):
        target=self.root/'main.py';namespace={}
        exec(compile('async def entrypoint_async():\n return 42\n',str(target),'exec'),namespace)
        def fail():raise RuntimeError('activation rejected')
        wrapper=obs.make_run_wrapper(asyncio.run,target,fail);coro=namespace['entrypoint_async']()
        with self.assertRaisesRegex(RuntimeError,'activation rejected'):wrapper(coro)
        self.assertIsNone(coro.cr_frame)

    def test_invalid_caps_are_rejected_without_claiming_output(self):
        for cap in (0,13):
            with self.assertRaises(ValueError):obs.Recorder(self.root/'invalid',cap)
        self.assertFalse((self.root/'invalid').exists())


    def test_real_site_startup_defers_native_imports_until_configured_main_run(self):
        tabby=self.root/'tabby';tabby.mkdir();engine=self.root/'engine';engine.mkdir()
        main=tabby/'main.py';output=self.root/'startup-traces'
        configuration=self.root/'observer.json'
        configuration.write_text(json.dumps({'tabby_repo':str(tabby),'engine_repo':str(engine),
            'tabby_commit':obs.TABBY_HEAD,'engine_commit':obs.ENGINE_HEAD,
            'output_dir':str(output),'max_records':2}))
        main.write_text("""import asyncio,json,os,sys,types
import reasoning_observer as obs
assert 'backends.exllamav3.model' not in sys.modules
assert 'torch' not in sys.modules
assert asyncio.run.__module__ == 'asyncio.runners'  # wraps preserves origin metadata
assert hasattr(asyncio.run,'__wrapped__')
config=json.loads(open(os.environ['TABBY_REASONING_OBSERVER_CONFIG']).read())
# Isolated test doubles: source gates are exercised separately; no real backend import.
obs.verified_source=lambda directory,expected: {'path':str(directory),'commit':expected,'tracked_changes':''}
cc=types.ModuleType('endpoints.OAI.utils.chat_completion')
async def collector(request_id,prompt,params,streaming_mode=True): return None
cc._chat_stream_collector=collector
m=types.ModuleType('backends.exllamav3.model')
m.__file__=config['tabby_repo']+'/backends/exllamav3/model.py'
class C:
 def constrain_generation_output(self,*a):return True
 def set_generation_phase(self,*a):return True
 def handle_finish_chunk(self,*a):return {}
m.ExllamaV3Container=C
j=types.ModuleType('exllamav3.generator.job')
j.__file__=config['engine_repo']+'/exllamav3/generator/job.py'
class J:
 def _pop_forced_token(self,*a):return None
j.Job=J
sys.modules[cc.__name__]=cc;sys.modules[m.__name__]=m;sys.modules[j.__name__]=j
os.environ['OBSERVER_CPU_CONFIG_READY']='1'
async def entrypoint_async():
 assert os.environ['OBSERVER_CPU_CONFIG_READY']=='1'
 assert hasattr(cc._chat_stream_collector,'__wrapped__')
 assert not hasattr(asyncio.run,'__wrapped__')
 assert 'torch' not in sys.modules
 return 17
assert asyncio.run(entrypoint_async())==17
print('configured startup observed without importing torch')
""")
        env=dict(os.environ);env['PYTHONPATH']=str(Path(obs.__file__).parent)
        env['TABBY_REASONING_OBSERVER_CONFIG']=str(configuration)
        result=subprocess.run([sys.executable,str(main)],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('configured startup observed',result.stdout)
        self.assertTrue((output/'observer-manifest.json').exists())

    def test_site_startup_is_inert_for_non_server_commands_even_with_bad_config_path(self):
        env=dict(os.environ);env['PYTHONPATH']=str(Path(obs.__file__).parent)
        env['TABBY_REASONING_OBSERVER_CONFIG']=str(self.root/'missing-config.json')
        result=subprocess.run([sys.executable,'-c','import sys; assert "reasoning_observer" not in sys.modules; print("inert")'],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'inert')

if __name__=='__main__':unittest.main()
