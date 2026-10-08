"""Optional real-SDK integration fixtures; no model, GPU or network is used."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdk_smoke
from api_client import ApiError

try:
    import openai
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
except ImportError:
    openai = None


@unittest.skipIf(openai is None, "optional OpenAI SDK not installed")
class SdkIntegrationTests(unittest.TestCase):
    def client(self, *, role=True, model="public-qwen", truncated=False):
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            forced = isinstance(payload.get("tool_choice"), dict)
            followup = payload.get("tool_choice") == "none"
            content = "Tokyo is 21 degrees celsius." if followup else "READY"
            if followup:
                assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
                self.assertEqual(assistant["role"], "assistant")
                result = next(m for m in payload["messages"] if m["role"] == "tool")
                self.assertEqual(result["tool_call_id"], assistant["tool_calls"][0]["id"])
                self.assertEqual(json.loads(result["content"])["source"], "synthetic test fixture")

            arguments = json.dumps({"city": "Tokyo", "unit": "celsius"})
            calls = [{
                "id": "call_fixture",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }] if forced else None
            finish = "length" if truncated else "tool_calls" if forced else "stop"
            usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            if not payload.get("stream"):
                message = {"role": "assistant", "content": None if forced else content}
                if calls:
                    message["tool_calls"] = calls
                return httpx.Response(200, json={
                    "id": "chatcmpl-fixture",
                    "object": "chat.completion",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": usage,
                })

            deltas = [{"role": "assistant"}] if role else []
            if calls:
                call = calls[0]
                deltas += [
                    {"tool_calls": [{
                        "index": 0, "id": call["id"], "type": "function",
                        "function": {"name": "get_weather", "arguments": arguments[:10]},
                    }]},
                    {"tool_calls": [{
                        "index": 0, "function": {"arguments": arguments[10:]},
                    }]},
                ]
            else:
                deltas += [{"content": content[:2]}, {"content": content[2:]}]
            frames = [{
                "id": "chatcmpl-fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            } for delta in deltas]
            frames.append({
                "id": "chatcmpl-fixture", "object": "chat.completion.chunk",
                "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage,
            })
            wire = "".join("data: " + json.dumps(frame) + "\n\n" for frame in frames)
            wire += "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)

        client = openai.OpenAI(
            api_key="synthetic-fixture",
            base_url="http://fixture.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        self.addCleanup(client.close)
        return client, requests

    def test_all_five_cases_and_sdk_message_round_trip(self):
        client, requests = self.client()
        passed = []

        def record(name, payload, streaming, finish_reason, validator):
            self.assertIsNotNone(payload, "fixture must not skip a dependent test")
            response, events = sdk_smoke.sdk_completion(client, payload, streaming)
            message = sdk_smoke.validate_completion(response, "public-qwen", finish_reason)
            validator(message)
            if streaming:
                self.assertIn("chunk", events)
            passed.append(name)
            return response, message

        sdk_smoke.run_cases("public-qwen", 512, record)
        self.assertEqual(passed, [
            "plain_stream", "named_nonstream", "round_trip_nonstream",
            "named_stream", "round_trip_stream",
        ])
        self.assertEqual(len(requests), 5)

    def test_missing_role_in_real_sdk_accumulator_is_rejected(self):
        from pydantic import ValidationError

        client, _ = self.client(role=False)
        response, _ = sdk_smoke.sdk_completion(client, {
            "model": "public-qwen", "messages": [{"role": "user", "content": "READY"}],
        }, True)
        self.assertIsNone(response.choices[0].message.role)
        with self.assertRaises(ValidationError):
            sdk_smoke.validate_completion(response, "public-qwen", "stop")

    def test_wrong_model_is_not_silently_accepted(self):
        client, _ = self.client(model="different-pack")
        response, _ = sdk_smoke.sdk_completion(client, {
            "model": "public-qwen", "messages": [{"role": "user", "content": "READY"}],
        }, True)
        with self.assertRaisesRegex(ApiError, "does not match"):
            sdk_smoke.validate_completion(response, "public-qwen", "stop")

    def test_length_finish_is_not_a_successful_tool_call(self):
        client, _ = self.client(truncated=True)
        with self.assertRaises(openai.LengthFinishReasonError) as caught:
            sdk_smoke.sdk_completion(client, {
                "model": "public-qwen", "messages": sdk_smoke.TOOL_MESSAGES,
                "tools": sdk_smoke.TOOLS, "tool_choice": sdk_smoke.NAMED,
            }, True)
        self.assertEqual(caught.exception.completion.choices[0].finish_reason, "length")


if __name__ == "__main__":
    unittest.main()
