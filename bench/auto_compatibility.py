#!/usr/bin/env python3
"""Serial live auto-tool compatibility checks; run only between benchmarks.

Requires adjacent api_resilience.py. Preserves the original resilience fixtures.
Eight short generations cover zero-call replies with reasoning off/on and
explicit client grammar/JSON constraints, in streaming and non-streaming modes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from api_resilience import (
    Client,
    Diagnostic,
    assembled,
    function,
    read_metadata,
    require,
)


class AutoDiagnostic(Diagnostic):
    report_kind = "auto_tool_compatibility"

    def check_reply(self, payload, expected, require_reasoning=False):
        result = assembled(
            self.client.request("/chat/completions", payload), self.model
        )
        require(
            result["finish_reason"] == "stop", "Ordinary reply did not finish with stop"
        )
        require(not result["tool_calls"], "Ordinary reply became a tool call")
        content = result["content"].strip()
        actual = json.loads(content) if isinstance(expected, dict) else content
        require(
            actual == expected,
            f"Content mismatch: actual={actual!r}, expected={expected!r}",
        )
        if require_reasoning:
            require(
                bool(result["reasoning_content"].strip()),
                "Reasoning phase was not exercised",
            )
        return result

    def auto(self):
        for variant in ("plain", "reasoning", "client_grammar", "client_json_schema"):
            for stream in (False, True):
                payload = self.chat_payload(stream)
                payload.update(
                    tools=[function("ping")], tool_choice="auto", max_tokens=128
                )
                payload["messages"] = [
                    {
                        "role": "user",
                        "content": "No tool is needed. Reply only with the plain word READY, without calling ping.",
                    }
                ]
                expected = "READY"
                if variant == "reasoning":
                    payload.update(enable_thinking=True, reasoning_budget_tokens=24)
                    payload["messages"][0]["content"] = (
                        "Think briefly about whether a tool is needed. No tool is needed. "
                        "Then reply only with the plain word READY, without calling ping."
                    )
                elif variant == "client_grammar":
                    payload["grammar_string"] = 'start: "READY"'
                elif variant == "client_json_schema":
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "auto_status",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {"status": {"const": "READY"}},
                                "required": ["status"],
                                "additionalProperties": False,
                            },
                        },
                    }
                    payload["messages"][0]["content"] = (
                        'No tool is needed. Reply with exactly the JSON object {"status":"READY"}, without calling ping.'
                    )
                    expected = {"status": "READY"}
                self.case(
                    f"auto_{variant}_{'stream' if stream else 'nonstream'}",
                    lambda payload=payload, expected=expected, variant=variant: (
                        self.check_reply(
                            payload, expected, require_reasoning=variant == "reasoning"
                        )
                    ),
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8899/v1"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--api-key-env", default="TABBY_API_KEY")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument(
        "--metadata", type=Path, help="JSON deployment/provenance snapshot"
    )
    parser.add_argument("--label", default="auto-compatibility")
    parser.add_argument(
        "--output",
        required=True,
        help="New JSON report path; existing files are refused",
    )
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error("--timeout must be (0,300]")
    try:
        metadata, metadata_source = read_metadata(args.metadata)
        client = Client(args.base_url, os.environ.get(args.api_key_env), args.timeout)
        diagnostic = AutoDiagnostic(
            client,
            args.model,
            output=args.output,
            settle_seconds=0,
            metadata=metadata,
            metadata_source=metadata_source,
            label=args.label,
        )
        okay = diagnostic.run(("auto",))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": args.output, "summary": diagnostic.report["summary"]}))
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
