"""Real local signal regressions; harmless CPU children, no API/GPU/Spark work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_matrix as matrix

# Run the real run_job and stop_owned in an isolated Python process. All source,
# model, HTTP and resource probes are replaced by the existing offline fixture.
# Only the children are real; each has its own session and retained owner token.
DRIVER = r'''
import json,os,signal,subprocess,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
from test_run_matrix import ControllerTests,controller
mode=sys.argv[2]; inventory=Path(sys.argv[3]); result_path=Path(sys.argv[4])
f=ControllerTests();f.setUp()
state=controller.InterruptState(); previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
for s in previous:signal.signal(s,state.receive)
original_popen=subprocess.Popen;original_run=controller.run_job;original_stop=controller.stop_owned
launched=[];events=[];triggered=False;cleanup_signals=0
final_line=next(i for i,line in enumerate(Path(controller.__file__).read_text().splitlines(),1) if 'record["server_exit_before_cleanup"] = server.poll()' in line)
def launch(command,**kwargs):
 global triggered
 is_server=command[0]=='bash'
 program='import time; time.sleep(60)' if is_server or mode=='client_registration' else 'pass'
 child=original_popen([sys.executable,'-c',program],**kwargs)
 token=kwargs['env']['QWEN_EXPERIMENT_OWNER'];launched.append((child,token))
 inventory.write_text(json.dumps([{'pid':c.pid,'token':t} for c,t in launched]))
 if is_server:
  f.server=child
  native_poll=child.poll
  def poll():
   global triggered
   caller=sys._getframe(1)
   if mode in ('final_poll','repeated_cleanup') and not triggered and caller.f_code.co_filename==controller.__file__ and caller.f_lineno==final_line:
    triggered=True;events.append('signal_in_final_poll');os.kill(os.getpid(),signal.SIGTERM)
   return native_poll()
  child.poll=poll
 else:f.client=child
 # Write the ordinary synthetic deployment/client reports with existing fixtures.
 ControllerTests.mock_launch(f,command,**kwargs)
 if mode==('server_registration' if is_server else 'client_registration'):
  triggered=True;events.append('signal_before_assignment');os.kill(os.getpid(),signal.SIGTERM)
 return child
f.mock_launch=launch
def stop(child,token):
 global cleanup_signals
 if mode=='repeated_cleanup' and triggered:
  cleanup_signals+=1
  # Real first/repeated signals must never abort owned cleanup.
  os.kill(os.getpid(),signal.SIGINT);os.kill(os.getpid(),signal.SIGTERM)
 return original_stop(child,token)
try:
 with patch.object(controller,'run_job',side_effect=lambda *a,**k:original_run(*a,**k,interrupt=state)):
  try:f.execute(stop_owned=stop)
  except KeyboardInterrupt:events.append('propagated_after_cleanup')
 record=json.loads((f.output/f.job['label']/'result.json').read_text())
 groups=[controller.group_members(c,t) for c,t in launched]
 result_path.write_text(json.dumps({'mode':mode,'events':events,'signal':state.signum,'state':record['state'],'passed':record['passed'],'recorded_signal':record.get('signal'),'finished':record.get('finished_at_utc'),'children':len(launched),'groups':groups,'codes':[c.poll() for c,t in launched],'cleanup_signal_rounds':cleanup_signals,'server_cleanup':record.get('server_cleanup'),'clients':record['clients']}))
finally:
 for sig,handler in previous.items():signal.signal(sig,handler)
 for child,token in reversed(launched):
  if controller.group_members(child,token):os.killpg(child.pid,signal.SIGKILL)
  child.wait(timeout=5)
 f.doCleanups()
'''


@unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc").is_dir(), "requires Linux procfs")
class SignalCleanupTests(unittest.TestCase):
    def run_case(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            inventory = Path(directory) / "children.json"
            result = Path(directory) / "result.json"
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", DRIVER, str(HERE), mode, str(inventory), str(result)],
                    capture_output=True, text=True, timeout=15)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                value = json.loads(result.read_text())
                self.assertEqual(value["state"], "interrupted")
                self.assertFalse(value["passed"])
                self.assertEqual(value["signal"], signal.SIGTERM)
                self.assertEqual(value["recorded_signal"], signal.SIGTERM)
                self.assertIsNone(value["finished"])
                self.assertIn("propagated_after_cleanup", value["events"])
                self.assertTrue(all(not group for group in value["groups"]))
                self.assertTrue(all(code is not None for code in value["codes"]))
                self.assertEqual(value["server_cleanup"]["signals"], ["SIGTERM"])
                return value
            finally:
                # Bounded safety cleanup if this regression ever returns early.
                if inventory.exists():
                    for row in json.loads(inventory.read_text()):
                        child = type("Retained", (), {"pid": row["pid"]})()
                        if matrix.group_members(child, row["token"]):
                            os.killpg(child.pid, signal.SIGKILL)

    def test_real_sigterm_at_server_popen_registration(self):
        value = self.run_case("server_registration")
        self.assertEqual(value["children"], 1)
        self.assertEqual(value["clients"], [])
        self.assertIn("signal_before_assignment", value["events"])

    def test_real_sigterm_at_client_popen_registration(self):
        value = self.run_case("client_registration")
        self.assertEqual(value["children"], 2)
        self.assertEqual(value["clients"][0]["signals"], ["SIGTERM"])

    def test_real_sigterm_during_final_server_poll(self):
        value = self.run_case("final_poll")
        self.assertEqual(value["children"], 2)
        self.assertIn("signal_in_final_poll", value["events"])

    def test_repeated_term_and_int_during_cleanup_cannot_escape(self):
        value = self.run_case("repeated_cleanup")
        self.assertGreaterEqual(value["cleanup_signal_rounds"], 1)
        self.assertEqual(value["children"], 2)

    def test_real_unrelated_group_is_not_signalled(self):
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
            env={**os.environ, "QWEN_EXPERIMENT_OWNER": "unrelated-test-owner"}, start_new_session=True)
        try:
            with self.assertRaisesRegex(RuntimeError, "unverified process group"):
                matrix.stop_owned(child, "different-retained-owner")
            self.assertIsNone(child.poll())
        finally:
            matrix.stop_owned(child, "unrelated-test-owner")


class InterruptStateTests(unittest.TestCase):
    def test_handler_records_without_raising_and_checkpoint_preserves_first_signal(self):
        state = matrix.InterruptState()
        state.receive(signal.SIGTERM, None)
        state.receive(signal.SIGINT, None)
        self.assertEqual(state.signum, signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            state.checkpoint()

    def test_no_signal_checkpoint_is_noop(self):
        self.assertIsNone(matrix.InterruptState().checkpoint())


if __name__ == "__main__":
    unittest.main()
