"""Offline protocol checks only; these fixtures never access the network."""

import io
import json
import unittest
from contextlib import redirect_stderr

from api_resilience import CheckError, Client, function
from auto_compatibility import AutoDiagnostic
from test_api_resilience import BASE, PUBLIC, FixtureServer, Response, event_bytes


class AutoFixture(FixtureServer):
    def __init__(self, defect=None):
        super().__init__()
        self.defect = defect

    def __call__(self, request, timeout):
        if not request.full_url.endswith("/chat/completions"):
            return super().__call__(request, timeout)
        payload = json.loads(request.data)
        self.requests.append(("/v1/chat/completions", payload))
        content = '{"status":"READY"}' if payload.get("response_format") else "READY"
        thought = "A tool is unnecessary." if payload.get("enable_thinking") else ""
        message = {
            "role": "assistant",
            "content": content,
            "reasoning_content": thought,
        }
        finish = "stop"
        if self.defect == "length":
            finish = "length"
        elif self.defect == "tool":
            message["tool_calls"] = [
                {
                    "id": "call-leak",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ]
            finish = "tool_calls"
        elif self.defect == "empty":
            message["content"] = ""
        elif self.defect == "no_reasoning":
            message["reasoning_content"] = ""
        elif self.defect == "wrong_json":
            message["content"] = '{"status":"WAIT"}'
        obj = {
            "id": "cmpl-auto-fixture",
            "created": 1,
            "model": PUBLIC,
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        if not payload["stream"]:
            return Response(json.dumps(obj).encode())
        common = {key: obj[key] for key in ("id", "created", "model", "object")}
        frames = [
            {**common, "choices": [{"index": 0, "delta": {"role": "assistant"}}]},
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            key: value
                            for key, value in message.items()
                            if key != "role"
                        },
                    }
                ],
            },
            {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {**common, "choices": [], "usage": obj["usage"]},
        ]
        for frame in frames:
            for choice in frame["choices"]:
                for index, call in enumerate(
                    choice.get("delta", {}).get("tool_calls", [])
                ):
                    call["index"] = index
        return Response(event_bytes(frames), "text/event-stream")


class AutoCompatibilityTests(unittest.TestCase):
    def test_all_fixture_modes_preserve_requests_and_usage(self):
        server = AutoFixture()
        client = Client(BASE, opener=server)
        diagnostic = AutoDiagnostic(client, PUBLIC)
        with redirect_stderr(io.StringIO()):
            self.assertTrue(diagnostic.run(("auto",)))
        self.assertEqual(diagnostic.report["summary"], {"pass": 9})
        requests = [
            trace["request"]
            for trace in client.traces
            if trace["path"] == "/chat/completions"
        ]
        self.assertEqual(len(requests), 8)
        self.assertTrue(
            all(
                request["tools"] and request["tool_choice"] == "auto"
                for request in requests
            )
        )
        self.assertEqual(
            sum(bool(request.get("grammar_string")) for request in requests), 2
        )
        self.assertEqual(
            sum(bool(request.get("response_format")) for request in requests), 2
        )

    def test_incomplete_tool_empty_reasoning_and_wrong_json_do_not_pass(self):
        for defect in ("length", "tool", "empty", "no_reasoning", "wrong_json"):
            for stream in (False, True):
                with self.subTest(defect=defect, stream=stream):
                    client = Client(BASE, opener=AutoFixture(defect))
                    diagnostic = AutoDiagnostic(client, PUBLIC)
                    payload = diagnostic.chat_payload(stream)
                    payload.update(
                        max_tokens=128,
                        tools=[function("ping")],
                        tool_choice="auto",
                        enable_thinking=True,
                    )
                    expected = "READY"
                    if defect == "wrong_json":
                        payload["response_format"] = {"type": "json_object"}
                        expected = {"status": "READY"}
                    with self.assertRaises(CheckError):
                        diagnostic.check_reply(
                            payload, expected, require_reasoning=True
                        )


if __name__ == "__main__":
    unittest.main()
