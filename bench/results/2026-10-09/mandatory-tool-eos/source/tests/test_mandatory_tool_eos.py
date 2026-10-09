"""CPU integration for the mandatory-tool reasoning EOS gap.

These tests execute the actual collector and backend methods; only native job
scheduling/allocation and sampler construction are fixtures. The engine's paired
source proof separately runs the actual sampler/Job/MTP selection and handoff.
"""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from backends.exllamav3.reasoning import implicit_reasoning_eos_ids
from tests import test_reasoning_budget as budget_fixture
from tests.test_reasoning_budget import (
    collect, prepared, qwen_parser, recording_backend,
)
from tests.test_natural_reasoning_handoff import natural_plan
from tests.test_tool_delta_streaming import make_request
from tests.test_qwen_tool_contract import xml


PIECES = ["<|im_end|>", "<|endoftext|>", "</think>", "ordinary", "", None]
TOKENIZER = SimpleNamespace(get_id_to_piece_list=lambda _: PIECES)
NAMED = {"type": "function", "function": {"name": "get_weather"}}


class ExplicitStopTests(unittest.TestCase):
    def test_only_implicit_known_eos_is_suppressed(self):
        self.assertEqual(implicit_reasoning_eos_ids(TOKENIZER, [0, 1, 0, 2, 4, 5, 99, -1, True], [], 2), [0, 1])
        self.assertEqual(implicit_reasoning_eos_ids(TOKENIZER, [0, 1], [0], 2), [1])
        self.assertEqual(implicit_reasoning_eos_ids(TOKENIZER, [0, 1], [0, 1], 2), [])

    def test_explicit_text_full_partial_and_spanning_stops_keep_priority(self):
        for stop in ('<|im_end|>', '|im_end|', 'prefix<|im_end|>suffix',
                     'prefix<|im_', 'end|>suffix'):
            with self.subTest(stop=stop):
                self.assertNotIn(0, implicit_reasoning_eos_ids(TOKENIZER, [0, 1], [stop], 2))
        self.assertEqual(implicit_reasoning_eos_ids(TOKENIZER, [0, 1], ['STOP', '\n', ''], 2), [0, 1])

    def test_inputs_are_not_mutated(self):
        ids, stops = [0, 1, 0], ['STOP', 1]
        self.assertEqual(implicit_reasoning_eos_ids(TOKENIZER, ids, stops, 2), [0])
        self.assertEqual((ids, stops), ([0, 1, 0], ['STOP', 1]))


class CollectorPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_forced_choice_with_native_plan_gets_internal_policy(self):
        for stream in (False, True):
            for choice in ('auto', 'none', 'required', NAMED):
                for native in (False, True):
                    with self.subTest(stream=stream, choice=choice, native=native):
                        chunks = ['thought', '</think>', xml('get_weather', [('city', 'Paris')])]
                        if choice == 'none':
                            chunks = ['thought', '</think>', 'answer']
                        mc = recording_backend(chunks)
                        mc.prepare_reasoning_budget = Mock(return_value=natural_plan() if native else None)
                        _, result = await collect(mc, make_request(tool_choice=choice), stream=stream)
                        expected = native and choice in ('required', NAMED)
                        self.assertEqual(mc.options[0].get('mandatory_tool_call', False), expected)
                        router = mc.prepare_reasoning_budget.call_args.kwargs['parser']
                        self.assertEqual(router.tool_calls_in_reasoning, choice not in ('required', NAMED))

    async def test_auto_keeps_in_reasoning_executable_calls_but_required_does_not(self):
        call = xml('get_weather', [('city', 'Paris')])
        for choice in ('auto', 'required', NAMED):
            mc = recording_backend([call, '</think>', call])
            mc.prepare_reasoning_budget = Mock(return_value=natural_plan())
            _, result = await collect(mc, make_request(tool_choice=choice), stream=False)
            self.assertEqual(len(result['tool_calls']), 2 if choice == 'auto' else 1)
            if choice != 'auto':
                self.assertIn(call, result['reasoning_content'])

    async def test_content_start_and_unsupported_fallback_are_unchanged(self):
        for stream in (False, True):
            mc = recording_backend([xml('get_weather', [('city', 'Paris')])])
            mc.prepare_reasoning_budget = Mock(side_effect=AssertionError('not initially reasoning'))
            await collect(mc, make_request(tool_choice='required'), stream=stream, initial=False)
            self.assertNotIn('mandatory_tool_call', mc.options[0])
            self.assertNotIn('reasoning_budget', mc.options[0])


class BackendPolicyTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self):
        mc, _, disconnect, events, jobs, other = budget_fixture.BackendArmingTests().backend()
        ns = mc.generate_gen.__func__.__globals__
        ns['implicit_reasoning_eos_ids'] = implicit_reasoning_eos_ids
        mc.tokenizer.get_id_to_piece_list = lambda _: ['text'] * 9 + ['</think>'] + ['text'] * 88 + ['<|endoftext|>', '<|im_end|>']
        mc.hf_model.eos_tokens = lambda: [99]
        mc.config.eos_token_id_list = [98, 99]
        builds = []
        class Builder:
            settings = []
            @classmethod
            def from_params(cls, params, *args):
                item = cls()
                item.params = params
                return item
            def build(self, greedy):
                result = SimpleNamespace(params=self.params, greedy=greedy)
                builds.append(result)
                return result
        ns['ExllamaV3SamplerBuilder'] = Builder
        ns['_describe_reasoning_settings'] = lambda params: 'fixture'
        params = make_request(temperature=0, max_tokens=10, grammar_string='start: "answer"',
                              banned_tokens=[4], stop=['STOP'])
        return mc, params, disconnect, jobs, builds

    async def run_backend(self, *, plan, mandatory=True, initial=True, stop=None, reasoning=None, handoff=False):
        mc, params, disconnect, jobs, builds = self.fixture()
        if stop is not None:
            params.stop = stop
        if reasoning is not None:
            params.reasoning_override = reasoning
        original = params.model_dump()
        if handoff:
            Job = mc.generate_gen.__func__.__globals__['AsyncJob']
            def set_filters(job, filters):
                job.current_filters = filters
            def set_sampler(job, sampler):
                job.current_sampler = sampler
            Job.set_filters, Job.set_sampler = set_filters, set_sampler
            async def events(job):
                job.budget[2]['on_end'](job.job)
                yield {'stage': 'streaming', 'eos': True, 'text': ''}
            Job.__aiter__ = events
        rows = [row async for row in mc.stream_generate(
            'id', 'prompt', params, disconnect, reasoning_phase=initial,
            reasoning_budget=plan, mandatory_tool_call=mandatory)]
        self.assertEqual(rows, [{'finish_reason': 'stop'}])
        self.assertEqual(params.model_dump(), original)
        self.assertEqual(jobs[0].creation['stop_conditions'], list(set([*params.stop, 98, 99])))
        return jobs[0], builds, params

    async def test_required_sampler_masks_only_implicit_eos_and_preserves_limits(self):
        for budget in (None, 0, 24):
            plan = natural_plan(parser=qwen_parser(tool_calls_in_reasoning=False)) if budget is None else prepared(budget=budget, parser=qwen_parser(tool_calls_in_reasoning=False))
            for stop, expected in ((['STOP'], [4, 99, 98]), ([99], [4, 98]), (['<|im_end|>'], [4, 98]), ([98, 99], [4])):
                with self.subTest(budget=budget, stop=stop):
                    job, builds, _ = await self.run_backend(plan=plan, stop=stop)
                    self.assertEqual(job.creation['sampler'].params.banned_tokens, expected)
                    self.assertEqual(builds[0].params.banned_tokens, [4])
                    self.assertEqual(job.creation['max_new_tokens'], 10)
                    self.assertEqual(job.creation['min_new_tokens'], 0)
                    self.assertEqual(job.creation['filters'], [])

    async def test_natural_and_forced_callbacks_restore_original_sampler(self):
        for plan in (natural_plan(parser=qwen_parser(tool_calls_in_reasoning=False)),
                     prepared(budget=0, parser=qwen_parser(tool_calls_in_reasoning=False))):
            with self.subTest(budget=plan.max_tokens):
                job, builds, _ = await self.run_backend(plan=plan, handoff=True)
                self.assertEqual(job.creation['sampler'].params.banned_tokens, [4, 99, 98])
                self.assertIs(job.current_sampler, builds[0])
                self.assertEqual(len(job.current_filters), 1)

    async def test_restricted_scopes_preserve_original_sampler_identity(self):
        protected = natural_plan(parser=qwen_parser(tool_calls_in_reasoning=False))
        for options in ({'plan': None}, {'plan': replace(protected, parser=None)},
                        {'plan': natural_plan()}, {'plan': protected, 'mandatory': False},
                        {'plan': protected, 'initial': False}):
            with self.subTest(options=options):
                job, builds, _ = await self.run_backend(**options)
                self.assertIs(job.creation['sampler'], builds[0])
                self.assertEqual(len(builds), 1)

    async def test_reusing_one_request_does_not_reclassify_implicit_eos_as_explicit(self):
        mc, params, disconnect, jobs, builds = self.fixture()
        original = params.model_dump()
        for request_id in ('first', 'second'):
            plan = natural_plan(parser=qwen_parser(tool_calls_in_reasoning=False))
            _ = [row async for row in mc.stream_generate(
                request_id, 'prompt', params, disconnect, reasoning_phase=True,
                reasoning_budget=plan, mandatory_tool_call=True)]
        self.assertEqual(params.model_dump(), original)
        self.assertEqual(len(jobs), 2)
        self.assertIsNot(jobs[0].creation['sampler'], jobs[1].creation['sampler'])
        for job in jobs:
            self.assertEqual(job.creation['sampler'].params.banned_tokens, [4, 99, 98])
        self.assertEqual(len(builds), 4)

    async def test_existing_reasoning_overrides_and_token_bans_are_preserved(self):
        job, builds, params = await self.run_backend(
            plan=natural_plan(parser=qwen_parser(tool_calls_in_reasoning=False)),
            reasoning={"temperature": 0.4, "banned_tokens": [7]})
        self.assertEqual(job.creation['sampler'].params.banned_tokens, [7, 99, 98])
        self.assertEqual(job.creation['sampler'].params.temperature, 0.4)
        self.assertEqual(builds[0].params.banned_tokens, [4])


if __name__ == '__main__':
    unittest.main()
