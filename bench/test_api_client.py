"""Protocol regressions for measurements that would otherwise look plausible but be wrong."""
from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from api_client import ApiError, iter_sse, perform, resolve_model, percentile
from tool_smoke import CASES, STRING_ARGS, STRINGS, TYPED, inspect_calls, validate_type


class Response(io.BytesIO):
    def __init__(self, data, content_type="text/event-stream"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


def event(value):
    return "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def chunk(delta=None, finish=None, model="model-a"):
    return {"model": model, "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def usage(prompt=10, completion=3, **extra):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion, **extra}


def stream(*objects, done=True):
    text = ": ping\n\n" + "".join(event(obj) for obj in objects)
    if done:
        text += event("[DONE]")
    return Response(text.encode())


PAYLOAD = {"model": "model-a", "max_tokens": 400, "stream": True,
           "stream_options": {"include_usage": True}}


class ApiIntegrityTests(unittest.TestCase):
    def call(self, response, **kwargs):
        with patch("api_client.open_request", return_value=response):
            return perform("http://test/v1", None, dict(PAYLOAD), **kwargs)

    def test_missing_usage_is_never_replaced_with_max_tokens(self):
        response = stream(chunk({"content": "short answer"}, "stop"))
        with self.assertRaisesRegex(ApiError, "no usage"):
            self.call(response)

    def test_actual_early_eos_token_count_is_preserved(self):
        response = stream(chunk({"content": "Done."}, "stop"),
                          {"usage": usage(), "choices": []})
        result = self.call(response)
        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["sse_events"], 2)

    def test_role_and_ping_are_not_first_generated_token(self):
        response = stream(chunk({"role": "assistant"}), chunk({"content": "answer"}, "stop"),
                          {"usage": usage(), "choices": []})
        with patch("api_client.time.perf_counter", side_effect=[10, 12, 14]):
            result = self.call(response)
        self.assertEqual(result["ttft_s"], 2)
        self.assertEqual(result["wall_s"], 4)
        # Multiple tokens delivered in one SSE event do not give a measurable decode interval.
        self.assertIsNone(result["client_decode_tok_s_estimate"])

    def test_abrupt_stream_end_is_not_success(self):
        response = stream(chunk({"content": "answer"}, "stop"), {"usage": usage()}, done=False)
        with self.assertRaisesRegex(ApiError, "without \\[DONE\\]"):
            self.call(response)

    def test_done_without_finish_reason_is_not_success(self):
        with self.assertRaisesRegex(ApiError, "finish_reason"):
            self.call(stream(chunk({"content": "answer"}), {"usage": usage()}))

    def test_in_band_stream_error_propagates(self):
        with self.assertRaisesRegex(ApiError, "out of memory"):
            self.call(stream({"error": {"message": "out of memory"}}))

    def test_bool_is_not_a_token_count(self):
        bad = usage()
        bad["completion_tokens"] = True
        with self.assertRaisesRegex(ApiError, "nonnegative integer"):
            self.call(stream(chunk({"content": "x"}, "stop"), {"usage": bad}))

    def test_inconsistent_usage_is_rejected(self):
        bad = usage()
        bad["total_tokens"] = 50
        with self.assertRaisesRegex(ApiError, "total_tokens differs"):
            self.call(stream(chunk({"content": "x"}, "stop"), {"usage": bad}))

    def test_alias_requires_explicit_expected_response_model(self):
        def response():
            return stream(chunk({"content": "x"}, "stop", model="canonical-pack"), {"usage": usage()})
        with self.assertRaisesRegex(ApiError, "canonical-pack"):
            self.call(response())
        result = self.call(response(), expected_response_model="canonical-pack")
        self.assertEqual(result["model"], "canonical-pack")
        self.assertEqual(result["requested_model"], "model-a")
        self.assertEqual(result["expected_response_model"], "canonical-pack")

    def test_model_cannot_change_mid_stream(self):
        with self.assertRaisesRegex(ApiError, "changed during"):
            self.call(stream(chunk({"content": "a"}), chunk({"content": "b"}, "stop", "model-b"),
                             {"usage": usage()}))

    def test_server_timings_cache_and_acceptance_are_separate(self):
        response = stream(chunk({"content": "x"}, "stop"),
                          {"usage": usage(prompt=10, completion=3,
                                          completion_tokens_per_sec=72.5,
                                          prompt_tokens_per_sec="1200.0",
                                          prompt_tokens_details={"cached_tokens": 7},
                                          completion_tokens_details={"accepted_prediction_tokens": 2,
                                                                    "rejected_prediction_tokens": 1})})
        result = self.call(response)
        self.assertEqual(result["server_decode_tok_s"], 72.5)
        self.assertEqual(result["server_prefill_tok_s"], 1200)
        self.assertEqual(result["uncached_prompt_tokens"], 3)
        self.assertAlmostEqual(result["draft_acceptance"], 2 / 3)
        self.assertNotEqual(result["client_end_to_end_tok_s"], result["server_decode_tok_s"])

    def test_llama_style_timing_fallback(self):
        response = stream(chunk({"content": "x"}, "stop"),
                          {"usage": usage(), "timings": {"predicted_per_second": 70,
                                                        "prompt_per_second": 1500}})
        result = self.call(response)
        self.assertEqual(result["server_decode_tok_s"], 70)
        self.assertEqual(result["server_prefill_tok_s"], 1500)

    def test_nonstream_response_usage_and_model_are_checked(self):
        obj = {"model": "model-a", "choices": [{"message": {"role": "assistant", "content": "done"},
                                              "finish_reason": "stop"}], "usage": usage()}
        with patch("api_client.open_request", return_value=Response(json.dumps(obj).encode(), "application/json")):
            result = perform("http://test/v1", None, {**PAYLOAD, "stream": False})
        self.assertEqual(result["completion_tokens"], 3)
        self.assertIsNone(result["ttft_s"])

    def test_sse_multiline_comments_no_space_and_eof_event(self):
        raw = io.BytesIO(b": hello\r\n\r\ndata:{\r\ndata: \"value\": 1}\r\n\r\ndata: [DONE]")
        self.assertEqual(list(iter_sse(raw)), ['{\n"value": 1}', "[DONE]"])

    def test_multiple_advertised_models_require_explicit_selection(self):
        with patch("api_client.json_request", return_value={"data": [{"id": "a"}, {"id": "b"}]}):
            with self.assertRaisesRegex(ApiError, "pass --model"):
                resolve_model("http://test/v1", None)
            self.assertEqual(resolve_model("http://test/v1", None, "b"), "b")

    def test_even_sample_median_is_interpolated(self):
        self.assertEqual(percentile([2, 8], 0.5), 5)


class ToolStreamTests(unittest.TestCase):
    def test_fragmented_name_args_and_literal_reasoning_tags_are_preserved(self):
        args = json.dumps(STRING_ARGS)
        chunks = [
            chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                   "function": {"name": "record_", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"name": "strings", "arguments": args[:19]}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[19:]}}]}, "tool_calls"),
            {"usage": usage(completion=100), "choices": []},
        ]
        with patch("api_client.open_request", return_value=stream(*chunks)):
            result = perform("http://test/v1", None, dict(PAYLOAD))
        errors, calls = inspect_calls(result, [STRINGS])
        self.assertEqual(errors, [])
        self.assertEqual(calls, [("record_strings", STRING_ARGS)])
        self.assertEqual(json.loads(result["message"]["tool_calls"][0]["function"]["arguments"])["tag_text"],
                         "<think>literal</think>")

    def test_changing_id_during_call_is_rejected(self):
        response = stream(chunk({"tool_calls": [{"index": 0, "id": "a"}]}),
                          chunk({"tool_calls": [{"index": 0, "id": "b"}]}, "tool_calls"), {"usage": usage()})
        with patch("api_client.open_request", return_value=response):
            with self.assertRaisesRegex(ApiError, "id changed"):
                perform("http://test/v1", None, dict(PAYLOAD))

    def test_noncontiguous_indices_are_rejected(self):
        response = stream(chunk({"tool_calls": [{"index": 1, "id": "a", "type": "function",
                                                 "function": {"name": "ping", "arguments": "{}"}}]}, "tool_calls"),
                          {"usage": usage()})
        with patch("api_client.open_request", return_value=response):
            with self.assertRaisesRegex(ApiError, "not contiguous"):
                perform("http://test/v1", None, dict(PAYLOAD))

    def test_string_looking_like_boolean_must_remain_string(self):
        schema = STRINGS["function"]["parameters"]
        wrong = dict(STRING_ARGS, bool_text=True)
        self.assertTrue(validate_type(wrong, schema, "args"))

    def test_integer_field_rejects_boolean(self):
        self.assertTrue(validate_type(True, {"type": "integer"}, "count"))
        self.assertFalse(validate_type(None, {"type": ["string", "null"]}, "optional"))


if __name__ == "__main__":
    unittest.main()
