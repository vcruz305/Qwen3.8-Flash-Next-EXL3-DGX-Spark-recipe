#!/usr/bin/env python3
"""CPU-only contracts for the frozen input preparation, with no Torch/imported server."""
import copy, importlib.util, json, tempfile, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prepared_inputs',HERE/'prepare_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PreparationContracts(unittest.TestCase):
 def setUp(self):self.plan=json.loads((HERE/'plan.json').read_text())
 def test_frozen_plan_payloads_and_count(self):
  self.assertEqual(len(m.validate_plan(self.plan)),61)
 def test_changed_payload_rejected(self):
  self.plan['requests'][0]['request']['temperature']=0.125
  with self.assertRaisesRegex(ValueError,'Payload hash'):m.validate_plan(self.plan)
 def test_duplicate_name_rejected(self):
  self.plan['requests'][1]['name']=self.plan['requests'][0]['name']
  with self.assertRaisesRegex(ValueError,'unique'):m.validate_plan(self.plan)
 def test_tools_before_static_rejected(self):
  self.plan['run_order'].reverse()
  with self.assertRaisesRegex(ValueError,'precede'):m.validate_plan(self.plan)
 def test_check_post_confusion_rejected(self):
  self.plan['standard_default_off_tool_chat_requests']=28
  with self.assertRaisesRegex(ValueError,'count'):m.validate_plan(self.plan)
 def test_no_prior_cache(self):
  self.assertEqual(m.cache_bounds([1,2],[],reject_strict_prefix=True)['cache_max_tokens'],0)
 def test_cache_bounds_all_prior_actual_ids(self):
  original=[{'name':'a','input_ids':[1,2,3,4]},{'name':'b','input_ids':[1,2,7,9]},{'name':'c','input_ids':[1,5]}]
  before=copy.deepcopy(original)
  r=m.cache_bounds([1,2,7,8],original,reject_strict_prefix=True)
  self.assertEqual((r['cache_max_tokens'],r['cache_prior_names']),(3,['b']))
  self.assertEqual(original,before)
 def test_distinct_prior_strict_prefix_rejected(self):
  with self.assertRaisesRegex(ValueError,'strict prefix'):m.cache_bounds([1,2,3],[{'name':'short','input_ids':[1,2]}],reject_strict_prefix=True)
 def test_identical_repeat_allowed(self):
  r=m.cache_bounds([1,2],[{'name':'repeat','input_ids':[1,2]}],reject_strict_prefix=True)
  self.assertEqual(r['cache_max_tokens'],2);self.assertTrue(r['prior_lcp'][0]['same_input'])
 def test_current_prefix_of_prior_is_safe_bound(self):
  r=m.cache_bounds([1,2],[{'name':'long','input_ids':[1,2,3]}],reject_strict_prefix=True)
  self.assertEqual(r['cache_max_tokens'],2)
 def test_unicode_payload_hash_exact(self):
  p={'text':'é<think>','bool':True}
  self.assertEqual(m.digest(p),m.digest({'bool':True,'text':'é<think>'}))
  self.assertNotEqual(m.digest(p),m.digest({'bool':True,'text':'é<think>'}))
if __name__=='__main__':unittest.main(verbosity=2)
