"""Execute the exact254 signal handler/finally in a CPU-only subprocess."""
import ast,hashlib,json,os,signal
from pathlib import Path
p=Path(__file__).with_name("run_matrix.254.py")
assert hashlib.sha256(p.read_bytes()).hexdigest()=="7e83208d473a124c6a90f8f07de0f338dea7d0510c6cd03f83d63f8cb31e103a"
tree=ast.parse(p.read_text())
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
handler=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=="interrupt")
ns={"signal":signal};exec(compile(ast.Module(body=[handler],type_ignores=[]),str(p),"exec"),ns)
old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};called=[]
class Child:
 def poll(self):os.kill(os.getpid(),signal.SIGTERM);return None
job=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run_job")
block=next(n for n in job.body if isinstance(n,ast.Try))
env={"server":Child(),"record":{},"stop_owned":lambda *a:called.append("stop_owned"),"token":"synthetic","server_log":None,"stamp":lambda:"cpu","save":lambda:None}
try:
 for s in old:signal.signal(s,ns["interrupt"])
 try:exec(compile(ast.Module(body=block.finalbody,type_ignores=[]),str(p),"exec"),env)
 except KeyboardInterrupt as exc:print(json.dumps({"baseline_commit":"254b2b03027f25094845dc31f5f87739f3584d2e","source_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"actual_SIGTERM_received":True,"exception":str(exc),"stop_owned_called":bool(called),"scope":"Exact source handler/finally, fake poll emits real signal; no child, network, API or GPU actions."}))
finally:
 for s,h in old.items():signal.signal(s,h)
