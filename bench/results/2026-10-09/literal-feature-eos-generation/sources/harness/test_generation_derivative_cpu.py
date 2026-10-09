import ast,copy,hashlib,json,tempfile,types,unittest
from pathlib import Path
import controller,input_binding
HERE=Path(__file__).resolve().parent
class GenerationDerivative(unittest.TestCase):
 def test_payloads_preserved_and_only_generation_client_scheduled(self):
  old=json.loads((HERE/'parent-plan.json').read_text());new=json.loads((HERE/'plan.json').read_text())
  self.assertEqual(new['requests'],old['requests'][8:]);self.assertEqual(new['total_chat_requests'],8)
  tree=ast.parse((HERE/'controller.py').read_text());run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run');fn=next(x for x in run.body if isinstance(x,ast.FunctionDef) and x.name=='commands')
  ns={'args':types.SimpleNamespace(observer_dir=Path('/observer'))};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual-generation-commands','exec'),ns)
  commands=list(ns['commands']({},Path('/attempt'),Path('/recipe'),'/python','alias'))
  self.assertEqual(commands,[('generation',['/python','/observer/generation_client.py','--metadata','/attempt/deployment.json','--timeout','120'])])
 def test_owned_lifecycle_and_admission_functions_unchanged(self):
  old=HERE/'frozen/parent_diagnostic_controller.py';self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),'089af7736daab1a1eddd35c60451013ed8fb9562bd93ad977927fce5f82b06ba')
  def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
  a,b=functions(old),functions(HERE/'controller.py')
  for name in ['clean_env','no_hooks','validate_customization','verify_system_site','run_recorded','check_sources','verify_deployment','attach_guard','finalize_owned']:
   with self.subTest(name=name):self.assertEqual(a[name],b[name])
 def binding(self):
  plan=json.loads((HERE/'plan.json').read_text())
  return {'passed':True,'ready_for_live':True,'plan_sha256':input_binding.PLAN_SHA,'tabby_commit':input_binding.TABBY,'engine_commit':input_binding.ENGINE,'requests':[{'name':r['name'],'request_sha256':r['request_sha256'],'expected_status':200,'input_ids':[1,2,3],'original_ids':[1,2],'prompt_tokens':3} for r in plan['requests']]}
 def validate(self,value):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'inputs.json';path.write_text(json.dumps(value));return input_binding.validate_binding(HERE/'plan.json',path,input_binding.sha(path))
 def test_rejects_wrong_source_missing_rows_and_bad_context(self):
  base=self.binding();self.validate(base)
  for mutate in [lambda b:b.update(tabby_commit='0'*40),lambda b:b['requests'].pop(),lambda b:b['requests'][0].update(prompt_tokens=4),lambda b:b['requests'][0].update(expected_status=400),lambda b:b['requests'][0].update(request_sha256='0'*64)]:
   value=copy.deepcopy(base);mutate(value)
   with self.assertRaises(ValueError):self.validate(value)
if __name__=='__main__':unittest.main()
