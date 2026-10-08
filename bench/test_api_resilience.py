"""Offline HTTP/SSE fixtures only: this module must never access the network."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from api_resilience import (
    GROUPS,
    NULLABLE_VALUES,
    CheckError,
    Client,
    Diagnostic,
    ReportWriter,
    assembled,
    function,
    read_metadata,
    sse_events,
)

PUBLIC = "Qwen3.8-Flash-Next-EXL3"
CANONICAL = "flashnext-exl3-3.05bpw"
BASE = "http://offline.invalid:5000/v1"


class Response(io.BytesIO):
    def __init__(self, content, content_type="application/json", status=200):
        super().__init__(content)
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.size = len(content)
        self.closed_at = None

    def close(self):
        if not self.closed:
            self.closed_at = self.tell()
        super().close()


def event_bytes(frames, done=True):
    data = b": keepalive\r\n\r\n"
    for frame in frames:
        data += b"data: " + json.dumps(frame).encode() + b"\n\n"
    return data + (b"data: [DONE]\n\n" if done else b"")


class FixtureServer:
    def __init__(self, unknown="canonical", invalid="reject", token_count=1):
        self.unknown = unknown
        self.invalid = invalid
        self.token_count = token_count
        self.requests = []
        self.responses = []
        self.errors = []

    def response(self, body, stream=False):
        response = Response(
            event_bytes(body) if stream else json.dumps(body).encode(),
            "text/event-stream" if stream else "application/json",
        )
        self.responses.append(response)
        return response

    def error(self, url, status, message):
        body = Response(
            json.dumps({"error": {"message": message}}).encode(), status=status
        )
        self.errors.append(body)
        raise urllib.error.HTTPError(url, status, message, body.headers, body)

    def __call__(self, request, timeout):
        path = urlsplit(request.full_url).path
        payload = json.loads(request.data) if request.data is not None else None
        self.requests.append((path, payload))
        if path == "/v1/model":
            return self.response({"id": CANONICAL})
        if path == "/v1/models":
            return self.response({"object": "list", "data": [{"id": PUBLIC}]})
        if path == "/v1/token/encode":
            return self.response(
                {"tokens": [55] * self.token_count, "length": self.token_count}
            )
        if path == "/health":
            return self.response({"status": "healthy", "issues": []})
        if path not in ("/v1/completions", "/v1/chat/completions"):
            raise AssertionError(f"Unexpected fixture route: {path}")
        model = payload["model"]
        if model not in (PUBLIC, CANONICAL):
            if self.unknown == "reject":
                return self.error(request.full_url, 400, "Unknown model")
            if self.unknown == "canonical":
                model = CANONICAL
        choice = payload.get("tool_choice")
        invalid_name = (
            isinstance(choice, dict) and choice["function"]["name"] == "absent"
        )
        invalid_budget = payload["max_tokens"] < 0
        invalid_grammar = (
            payload.get("grammar_string") == "start: nonexistent_diagnostic_rule"
        )
        if self.invalid == "reject" and (
            invalid_name or invalid_budget or invalid_grammar
        ):
            return self.error(
                request.full_url, 422 if invalid_budget else 400, "Invalid request"
            )
        raw = path == "/v1/completions"
        tokens = max(payload["max_tokens"], 1) if raw else 2
        usage = {"prompt_tokens": 1 if raw else 20, "completion_tokens": tokens}
        usage["total_tokens"] = sum(usage.values())
        name = None
        args = {}
        if payload.get("tools") and not invalid_name:
            if isinstance(choice, dict):
                name = choice["function"]["name"]
            else:
                name = payload["tools"][0]["function"]["name"]
            if name == "echo_nullable":
                args = NULLABLE_VALUES
        reasoning = (
            "I will choose the required tool." if payload.get("enable_thinking") else ""
        )
        message = {"role": "assistant", "content": "READY"}
        if name:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": "call-fixture",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        if reasoning:
            message["reasoning_content"] = reasoning
        finish = "length" if raw else ("tool_calls" if name else "stop")
        obj = {
            "id": f"cmpl-fixture-{len(self.requests)}",
            "created": 1,
            "model": model,
            "object": "text_completion" if raw else "chat.completion",
            "choices": [{"index": 0, "finish_reason": finish}],
            "usage": usage,
        }
        obj["choices"][0].update(
            {"text": "A" * tokens} if raw else {"message": message}
        )
        if not payload.get("stream"):
            return self.response(obj)
        frames = []

        def frame(delta=None, text=None, finish_reason=None, usage_=None, choices=True):
            value = {key: obj[key] for key in ("id", "created", "model", "object")}
            value["choices"] = (
                [{"index": 0, "finish_reason": finish_reason}] if choices else []
            )
            if choices:
                value["choices"][0].update(
                    {"text": text or ""} if raw else {"delta": delta or {}}
                )
            if usage_:
                value["usage"] = usage_
            frames.append(value)

        if raw:
            frame(text="A")
            frame(text="A" * (tokens - 1))
        else:
            frame(delta={"role": "assistant"})
            frame(delta={})  # Empty/progress output must not trigger early disconnect.
            if reasoning:
                frame(delta={"reasoning_content": reasoning})
            if name:
                frame(
                    delta={
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-fixture",
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    }
                )
                arguments = json.dumps(args)
                for offset in range(0, len(arguments), 3):
                    frame(
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": arguments[offset : offset + 3]
                                    },
                                }
                            ]
                        }
                    )
            else:
                frame(delta={"content": "READY"})
        frame(finish_reason=finish)
        frame(usage_=usage, choices=False)
        return self.response(frames, stream=True)


def client(server):
    return Client(BASE, key="offline-secret-must-never-be-recorded", opener=server)


def valid_trace(stream=True, raw=False, tools=False):
    server = FixtureServer()
    diag = Diagnostic(client(server), PUBLIC, settle_seconds=0)
    payload = diag.raw_payload(stream) if raw else diag.chat_payload(stream)
    if tools:
        payload.update(tools=[function("echo_nullable")], tool_choice="required")
    return diag.client.request("/completions" if raw else "/chat/completions", payload)


class OfflineDiagnosticTests(unittest.TestCase):
    def test_full_runner_and_saved_evidence_for_both_unknown_model_policies(self):
        for unknown in ("canonical", "reject"):
            with (
                self.subTest(unknown=unknown),
                tempfile.TemporaryDirectory() as directory,
            ):
                server = FixtureServer(unknown=unknown)
                out = Path(directory) / "result.json"
                diag = Diagnostic(client(server), PUBLIC, output=out, settle_seconds=0)
                with redirect_stderr(io.StringIO()):
                    self.assertTrue(diag.run(GROUPS))
                report = json.loads(out.read_text())
                self.assertEqual(report["summary"], {"pass": 19})
                self.assertEqual(len(report["requests"]), 27)
                self.assertNotIn("offline-secret", out.read_text())
                self.assertFalse(out.with_name(out.name + ".tmp").exists())
                self.assertTrue(
                    all(
                        response.closed for response in server.responses + server.errors
                    )
                )
                token_request = server.requests[2][1]
                self.assertFalse(token_request["add_bos_token"])
                self.assertFalse(token_request["encode_special_tokens"])
                raw_requests = [
                    p for path, p in server.requests if path == "/v1/completions"
                ]
                self.assertTrue(
                    all(
                        p["prompt"] == "X" and p["add_bos_token"] is False
                        for p in raw_requests
                    )
                )
                reasoned = [
                    p for _, p in server.requests if p and p.get("enable_thinking")
                ]
                self.assertEqual(len(reasoned), 4)
                self.assertTrue(
                    all(p["reasoning_budget_tokens"] == 24 for p in reasoned)
                )
                self.assertEqual(
                    report["cases"][-1]["result"]["events_before_close"], 3
                )

    def test_preflight_stops_before_generation_if_prompt_not_one_token(self):
        server = FixtureServer(token_count=2)
        diag = Diagnostic(client(server), PUBLIC)
        with redirect_stderr(io.StringIO()):
            self.assertFalse(diag.run(GROUPS))
        self.assertEqual(len(server.requests), 3)
        self.assertEqual(diag.report["summary"], {"fail": 1})
        self.assertIn("not exactly one token", diag.report["cases"][0]["error"])

    def test_preflight_unadvertised_model_does_not_switch_models(self):
        server = FixtureServer()
        diag = Diagnostic(client(server), "invented")
        with redirect_stderr(io.StringIO()):
            self.assertFalse(diag.run(GROUPS))
        self.assertEqual(len(server.requests), 2)
        self.assertIn("not advertised/current", diag.report["cases"][0]["error"])

    def test_unknown_model_must_not_be_echoed_as_served(self):
        server = FixtureServer(unknown="echo")
        diag = Diagnostic(client(server), PUBLIC)
        with redirect_stderr(io.StringIO()):
            self.assertFalse(diag.run(["aliases"]))
        self.assertEqual(diag.report["summary"], {"pass": 3, "fail": 4})
        errors = [
            case["error"] for case in diag.report["cases"] if case["status"] == "fail"
        ]
        self.assertTrue(all("Standard model field" in error for error in errors))

    def test_invalid_request_still_attempts_recovery_when_unexpectedly_accepted(self):
        server = FixtureServer(invalid="accept")
        diag = Diagnostic(client(server), PUBLIC)
        with redirect_stderr(io.StringIO()):
            self.assertFalse(diag.run(["invalid"]))
        self.assertEqual(diag.report["summary"], {"pass": 1, "fail": 3})
        self.assertEqual(len(server.requests), 9)
        for i in (4, 6, 8):
            path, payload = server.requests[i]
            self.assertEqual(path, "/v1/chat/completions")
            self.assertEqual(
                payload["messages"][0]["content"], "Reply with the word READY."
            )

    def test_disconnect_ignores_role_and_empty_frame_closes_without_draining(self):
        server = FixtureServer()
        cli = client(server)
        diag = Diagnostic(cli, PUBLIC)
        trace = cli.request(
            "/chat/completions", diag.chat_payload(True), abort_after_output=True
        )
        self.assertTrue(trace["client_aborted"])
        self.assertFalse(trace["done"])
        self.assertEqual(len(trace["frames"]), 3)
        self.assertEqual(
            trace["frames"][-1]["choices"][0]["delta"], {"content": "READY"}
        )
        response = server.responses[0]
        self.assertTrue(response.closed)
        self.assertLess(response.closed_at, response.size)

    def test_http_error_closed_and_diagnostic_saved_without_auth(self):
        server = FixtureServer(unknown="reject")
        cli = client(server)
        trace = cli.request(
            "/chat/completions", Diagnostic(cli, "unknown").chat_payload()
        )
        self.assertEqual(trace["status"], 400)
        self.assertEqual(trace["body"], {"error": {"message": "Unknown model"}})
        self.assertTrue(server.errors[0].closed)
        self.assertNotIn("offline-secret", json.dumps(trace))

    def test_recovery_rejects_truncation_tool_state_and_wrong_sentinel(self):
        for failure in ("truncated", "tool_only", "wrong_sentinel", "empty"):
            with self.subTest(failure=failure):
                trace = valid_trace(False, tools=failure == "tool_only")
                choice = trace["body"]["choices"][0]
                if failure == "truncated":
                    choice["finish_reason"] = "length"
                elif failure == "wrong_sentinel":
                    choice["message"]["content"] = "NOT_READY"
                elif failure == "empty":
                    choice["message"]["content"] = ""
                cli = client(FixtureServer())
                cli.request = lambda *args, trace=trace, **kwargs: trace
                with self.assertRaises(CheckError):
                    Diagnostic(cli, PUBLIC).healthy()

    def test_sse_multiline_comments_crlf_and_unterminated_final_event(self):
        response = io.BytesIO(
            b': ignored\r\nid: 1\r\ndata: {"one":\r\ndata: 1}\r\n\r\ndata: [DONE]'
        )
        self.assertEqual(list(sse_events(response)), ['{"one":\n1}', "[DONE]"])

    def test_transport_and_sse_json_errors_are_evidence(self):
        def timeout_opener(request, timeout):
            raise TimeoutError("fixture timeout")

        cli = Client(BASE, opener=timeout_opener)
        trace = cli.request("/chat/completions", Diagnostic(cli, PUBLIC).chat_payload())
        self.assertIn("TimeoutError", trace["transport_or_protocol_error"])
        response = Response(b"data: not-json\n\n", "text/event-stream")
        cli = Client(BASE, opener=lambda request, timeout: response)
        trace = cli.request(
            "/chat/completions", Diagnostic(cli, PUBLIC).chat_payload(True)
        )
        self.assertIn("JSONDecodeError", trace["transport_or_protocol_error"])
        self.assertTrue(response.closed)

    def test_valid_raw_and_chat_assembly(self):
        for stream in (False, True):
            for raw in (False, True):
                with self.subTest(stream=stream, raw=raw):
                    result = assembled(valid_trace(stream, raw), PUBLIC, raw=raw)
                    self.assertEqual(result["model"], PUBLIC)
                    self.assertEqual(
                        result["finish_reason"], "length" if raw else "stop"
                    )
                    self.assertEqual(
                        result["usage"]["completion_tokens"], 8 if raw else 2
                    )

    def test_strict_stream_failures(self):
        mutations = {
            "missing model": lambda t: t["frames"][0].pop("model"),
            "legacy model_name": lambda t: t["frames"][0].update(
                model_name=t["frames"][0].pop("model")
            ),
            "wrong model": lambda t: t["frames"][0].update(model="wrong"),
            "missing ID": lambda t: t["frames"][0].pop("id"),
            "unstable ID": lambda t: t["frames"][-1].update(id="changed"),
            "missing role": lambda t: t["frames"][0]["choices"][0]["delta"].pop("role"),
            "no DONE": lambda t: t.update(done=False),
            "no usage": lambda t: t["frames"][-1].pop("usage"),
            "error event": lambda t: t.update(server_error={"message": "failed"}),
            "false token count": lambda t: t["frames"][-1]["usage"].update(
                prompt_tokens=True
            ),
            "usage sum": lambda t: t["frames"][-1]["usage"].update(total_tokens=999),
            "token budget": lambda t: t["request"].update(max_tokens=1),
            "invalid choice index": lambda t: t["frames"][0]["choices"][0].update(
                index=False
            ),
            "finish missing": lambda t: t["frames"][-2]["choices"][0].update(
                finish_reason=None
            ),
            "output after finish": lambda t: t["frames"].insert(
                -1, copy.deepcopy(t["frames"][2])
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                trace = valid_trace()
                mutate(trace)
                with self.assertRaises(CheckError):
                    assembled(trace, PUBLIC)

    def test_fragmented_nullable_arguments_roundtrip_exact_types_and_whitespace(self):
        for stream in (False, True):
            result = assembled(valid_trace(stream, tools=True), PUBLIC)
            call = result["tool_calls"][0]
            self.assertEqual(call["type"], "function")
            actual = json.loads(call["function"]["arguments"])
            self.assertEqual(actual, NULLABLE_VALUES)
            for key, expected in NULLABLE_VALUES.items():
                self.assertIs(type(actual[key]), type(expected))
            self.assertEqual(actual["code"], "    return 1\n")

    def test_tool_type_index_id_are_required(self):
        for key in ("type", "index", "id"):
            with self.subTest(key=key):
                trace = valid_trace(tools=True)
                trace["frames"][2]["choices"][0]["delta"]["tool_calls"][0].pop(key)
                with self.assertRaises(CheckError):
                    assembled(trace, PUBLIC)

    def test_nullable_check_rejects_bad_types_or_stripped_whitespace(self):
        for field, replacement in (
            ("numeric_text", 123),
            ("boolean_text", True),
            ("json_text", [1, 2]),
            ("code", "return 1"),
            ("actual_null", "null"),
        ):
            with self.subTest(field=field):
                trace = valid_trace(False, tools=True)
                wrong_args = dict(NULLABLE_VALUES, **{field: replacement})
                trace["body"]["choices"][0]["message"]["tool_calls"][0]["function"][
                    "arguments"
                ] = json.dumps(wrong_args)
                cli = Client(
                    BASE,
                    opener=lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError("network")
                    ),
                )
                cli.request = lambda *args, trace=trace, **kwargs: trace
                diag = Diagnostic(cli, PUBLIC)
                with self.assertRaises(CheckError):
                    diag.check_call(
                        trace["request"], {"echo_nullable"}, NULLABLE_VALUES
                    )


class ReportAndCliTests(unittest.TestCase):
    @staticmethod
    def runners():
        from api_resilience import main as resilience_main
        from auto_compatibility import AutoDiagnostic
        from auto_compatibility import main as auto_main
        from test_auto_compatibility import AutoFixture

        return [
            (
                Diagnostic,
                resilience_main,
                FixtureServer,
                GROUPS,
                ["--settle-seconds", "0"],
                19,
                27,
                "66427f4fd50f8e8eb5ea3309f1506c364650308296b710ecc1fa5cf8415e194a",
            ),
            (
                AutoDiagnostic,
                auto_main,
                AutoFixture,
                ("auto",),
                [],
                9,
                11,
                "2a8e116303048edc6bb6e79e019abe6796114faa59aec8e3220bd4c0d48d610f",
            ),
        ]

    def test_existing_file_directory_and_dangling_symlink_refused_before_http(self):
        for cls, _, server_type, groups, *_ in self.runners():
            for kind in ("file", "directory", "dangling_symlink"):
                with (
                    self.subTest(runner=cls.__name__, kind=kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    output = Path(tmp) / "report.json"
                    if kind == "file":
                        output.write_text("previous report")
                    elif kind == "directory":
                        output.mkdir()
                    else:
                        output.symlink_to(Path(tmp) / "absent.json")
                    server = server_type()
                    diagnostic = cls(
                        client(server), PUBLIC, output=output, settle_seconds=0
                    )
                    with self.assertRaises(FileExistsError):
                        diagnostic.run(groups)
                    self.assertEqual(server.requests, [])
                    if kind == "file":
                        self.assertEqual(output.read_text(), "previous report")
                    elif kind == "directory":
                        self.assertTrue(output.is_dir())
                    else:
                        self.assertTrue(output.is_symlink())
                    self.assertEqual(
                        [p.name for p in Path(tmp).iterdir()], ["report.json"]
                    )

    def test_concurrent_claim_and_replaced_report_owner_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            writers = [ReportWriter(output), ReportWriter(output)]

            def publish(index):
                try:
                    writers[index].write({"writer": index})
                    return index
                except FileExistsError:
                    return None

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(publish, range(2)))
            winners = [index for index in results if index is not None]
            self.assertEqual(len(winners), 1)
            winner = winners[0]
            self.assertEqual(json.loads(output.read_text()), {"writer": winner})
            writers[winner].write({"writer": winner, "updated": True})
            replacement = Path(tmp) / "replacement.json"
            replacement.write_text('{"external":true}')
            os.replace(replacement, output)
            with self.assertRaises(FileExistsError):
                writers[winner].write({"writer": winner, "unexpected": True})
            self.assertEqual(json.loads(output.read_text()), {"external": True})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["report.json"])

    def test_checkpoint_failure_keeps_complete_report_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            writer = ReportWriter(output)
            writer.write({"checkpoint": 1})
            with (
                patch(
                    "api_resilience.os.replace",
                    side_effect=OSError("simulated publish failure"),
                ),
                self.assertRaises(OSError),
            ):
                writer.write({"checkpoint": 2})
            self.assertEqual(json.loads(output.read_text()), {"checkpoint": 1})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["report.json"])
            second = ReportWriter(Path(tmp) / "new.json")
            with (
                patch(
                    "api_resilience.os.link",
                    side_effect=OSError("simulated initial failure"),
                ),
                self.assertRaises(OSError),
            ):
                second.write({"checkpoint": 0})
            self.assertFalse(second.path.exists())
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["report.json"])

    def test_initial_checkpoint_precedes_first_http_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            server = FixtureServer()

            def opener(request, timeout):
                if not server.requests:
                    initial = json.loads(output.read_text())
                    self.assertEqual(initial["cases"], [])
                    self.assertEqual(initial["requests"], [])
                    self.assertEqual(initial["selected_groups"], ["raw"])
                    self.assertEqual(initial["summary"], {})
                return server(request, timeout)

            diagnostic = Diagnostic(Client(BASE, opener=opener), PUBLIC, output=output)
            with redirect_stderr(io.StringIO()):
                self.assertTrue(diagnostic.run(("raw",)))
            self.assertEqual(json.loads(output.read_text())["summary"], {"pass": 3})

    def test_metadata_label_and_fingerprint_preserved_without_auth_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "deployment.json"
            value = {
                "engine_commit": "fixture-commit",
                "model": "synthetic-π",
                "config": {"network": {"port": 8899}},
            }
            raw = (json.dumps(value, ensure_ascii=False) + "\n").encode()
            source.write_bytes(raw)
            metadata, origin = read_metadata(source)
            output = Path(tmp) / "report.json"
            diagnostic = Diagnostic(
                Client(BASE, key="fixture-secret-never-record", opener=FixtureServer()),
                PUBLIC,
                output=output,
                metadata=metadata,
                metadata_source=origin,
                label="candidate",
            )
            with redirect_stderr(io.StringIO()):
                self.assertTrue(diagnostic.run(("raw",)))
            report_text = output.read_text()
            report = json.loads(report_text)
            self.assertEqual(report["provenance"], value)
            self.assertEqual(report["label"], "candidate")
            self.assertEqual(
                report["metadata_source"],
                {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()},
            )
            self.assertNotIn("fixture-secret-never-record", report_text)
            self.assertNotIn("Authorization", report_text)

    def test_bad_metadata_and_existing_output_cli_fail_before_http(self):
        for _, main, _, _, extra, *_ in self.runners():
            for kind in (
                "missing_metadata",
                "malformed_metadata",
                "array_metadata",
                "existing_output",
            ):
                with (
                    self.subTest(runner=main.__module__, kind=kind),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    output = Path(tmp) / "report.json"
                    args = [
                        main.__module__,
                        "--base-url",
                        BASE,
                        "--model",
                        PUBLIC,
                        "--output",
                        str(output),
                        *extra,
                    ]
                    if kind == "existing_output":
                        output.write_text("previous report")
                    else:
                        source = Path(tmp) / "deployment.json"
                        if kind == "malformed_metadata":
                            source.write_text("{broken")
                        elif kind == "array_metadata":
                            source.write_text("[]")
                        args += ["--metadata", str(source)]
                    with (
                        patch("sys.argv", args),
                        patch("api_resilience.urllib.request.urlopen") as network,
                        redirect_stderr(io.StringIO()),
                        self.assertRaises(SystemExit) as caught,
                    ):
                        main()
                    self.assertEqual(caught.exception.code, 2)
                    network.assert_not_called()
                    if kind == "existing_output":
                        self.assertEqual(output.read_text(), "previous report")
                    else:
                        self.assertFalse(output.exists())

    def test_cli_preserves_original_fixture_request_fingerprints_and_counts(self):
        # Frozen from the standalone runners before recipe packaging. Report
        # metadata and I/O changes must not quietly change the live fixtures.
        for (
            cls,
            main,
            server_type,
            _,
            extra,
            cases,
            requests,
            expected_hash,
        ) in self.runners():
            with (
                self.subTest(runner=cls.__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                output = Path(tmp) / "report.json"
                source = Path(tmp) / "deployment.json"
                source.write_text('{"fixture":"deployment"}')
                args = [
                    main.__module__,
                    "--base-url",
                    BASE,
                    "--model",
                    PUBLIC,
                    "--output",
                    str(output),
                    "--metadata",
                    str(source),
                    "--label",
                    "offline-candidate",
                    *extra,
                ]
                with (
                    patch("sys.argv", args),
                    patch("api_resilience.urllib.request.urlopen", server_type()),
                    redirect_stderr(io.StringIO()),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(), 0)
                report = json.loads(output.read_text())
                self.assertEqual(report["summary"], {"pass": cases})
                self.assertEqual(len(report["requests"]), requests)
                self.assertEqual(report["provenance"], {"fixture": "deployment"})
                self.assertEqual(report["label"], "offline-candidate")
                self.assertEqual(report["kind"], cls.report_kind)
                fixtures = [
                    {"path": trace["path"], "request": trace["request"]}
                    for trace in report["requests"]
                ]
                raw = json.dumps(
                    fixtures, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash)


if __name__ == "__main__":
    unittest.main()
