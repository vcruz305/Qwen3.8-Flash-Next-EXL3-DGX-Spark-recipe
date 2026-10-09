import ast,copy,json,tempfile,unittest
from pathlib import Path
import controller
HERE=Path(__file__).resolve().parent
class ToolRegression(unittest.TestCase):
 def test_real_report_all28_and32(self):
  proof=controller.verify_tool_report(HERE/'fixtures/prior-passed-tools.json');self.assertEqual(proof['checks'],28);self.assertEqual(proof['chat_posts'],32)
 def test_failed_missing_extra_turn_and_feature_request_rejected(self):
  original=json.loads((HERE/'fixtures/prior-passed-tools.json').read_text())
  def missing(d):
   row=next(r for r in d['results'] if 'followup'in r);del row['followup']
  for mutate in [lambda d:d['results'][0].update(passed=False),lambda d:d['results'][0]['request'].update(literal_user_control_tokens=True),missing,lambda d:d.pop('completed_at_utc')]:
   bad=copy.deepcopy(original);mutate(bad)
   with tempfile.TemporaryDirectory() as t:
    f=Path(t)/'report.json';f.write_text(json.dumps(bad))
    with self.assertRaises(ValueError):controller.verify_tool_report(f)
 def test_real_helper_default_tool_command(self):
  h=controller.load(HERE/'frozen/run_matrix.py',controller.HELPER_SHA,'default_command_proof')
  with tempfile.TemporaryDirectory() as d:
   model=Path(d);(model/'config.json').write_text('{}');(model/'fixture.safetensors').write_bytes(b'fixture-only-stat-identity')
   job=h.normalize({'label':'eos-default-tool28','model_path':str(model),'env':{'PROFILE':'concurrent','NGRAM_RAM':'false'},'bench':[],'tools':{}})
  self.assertEqual(list(h.commands(job,Path('/attempt'),Path('/recipe'),'/python','alias')),[('tools',['/python','/recipe/bench/tool_smoke.py','--base-url',h.BASE,'--model','alias'])])
 def test_owned_lifecycle_functions_unchanged(self):
  def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
  a,b=functions(HERE/'frozen/parent_diagnostic_controller.py'),functions(HERE/'controller.py')
  for name in ['clean_env','no_hooks','validate_customization','verify_system_site','run_recorded','check_sources','verify_deployment','attach_guard','finalize_owned']:
   with self.subTest(name=name):self.assertEqual(a[name],b[name])
if __name__=='__main__':unittest.main()
