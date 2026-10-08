#!/usr/bin/env python3
"""Replay the six archived synthetic raw responses through the exact f4 parser.

Pure CPU/standard library. No server, model, network, or live source checkout is
used; this verifies channel routing, not semantic quality or causal attribution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types

PARSER_SHA = "fa222d8e29f2f5ce941137d8f026534bb6a6b8f21cd24c1764842894ae1279c4"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed(entry):
    if not entry["stream"]:
        choice = entry["body"]["choices"][0]
        message = choice["message"]
        return {
            "reasoning": message.get("reasoning_content") or "",
            "content": message.get("content") or "",
            "finish": choice["finish_reason"],
        }
    reasoning, content, finish = [], [], None
    for frame in entry["frames"]:
        for choice in frame.get("choices", []):
            delta = choice.get("delta", {})
            reasoning.append(delta.get("reasoning_content") or "")
            content.append(delta.get("content") or "")
            finish = choice.get("finish_reason") or finish
    if entry.get("done") is not True:
        raise ValueError("Archived SSE did not finish with DONE")
    return {"reasoning": "".join(reasoning), "content": "".join(content), "finish": finish}


def check(root):
    parser_path = root / "source/stream_parser_f4.py"
    if sha(parser_path) != PARSER_SHA:
        raise ValueError("Frozen f4 channel-parser source changed")
    module = types.ModuleType("frozen_f4_stream_parser")
    exec(compile(parser_path.read_bytes(), str(parser_path), "exec"), module.__dict__)
    analysis = json.loads((root / "raw-parser-analysis.json").read_text())
    live = root / "live-observer"
    if sha(live / "result.json") != analysis["controller_result_sha256"]:
        raise ValueError("Observer controller result differs from frozen analysis")
    controller = json.loads((live / "result.json").read_text())
    if controller["state"] != "completed" or controller["observer_traces"]["completed"] != 6:
        raise ValueError("Observer run is incomplete")
    rows, replays = [], 0
    for item in analysis["cases"]:
        path = live / "raw" / item["trace"]
        report_path = live / item["api_report"]
        if sha(path) != item["trace_sha256"] or sha(report_path) != item["api_report_sha256"]:
            raise ValueError("Archived trace/API report differs from frozen analysis")
        trace = json.loads(path.read_text())
        report = json.loads(report_path.read_text())
        case = next(case for case in report["cases"] if case["name"] == item["case"])
        entry = report["requests"][case["request_indices"][0]]
        public = observed(entry)
        raw = trace["raw_finish"]["text"]
        for width in (1, 7, 31, max(1, len(raw))):
            parser = module.Qwen3CoderStreamParser(
                reasoning_start="<think>", reasoning_end="</think>",
                tool_start="<tool_call>", tool_end="</tool_call>",
                start_in_reasoning=True, tool_calls_in_reasoning=True,
            )
            events = []
            for offset in range(0, len(raw), width):
                events.extend(parser.feed(raw[offset:offset + width]))
            events.extend(parser.finish())
            routed = {channel: "" for channel in (module.REASONING, module.CONTENT, module.TOOL)}
            for channel, text in events:
                routed[channel] += text
            if (
                routed[module.REASONING] != public["reasoning"]
                or routed[module.CONTENT] != public["content"]
                or routed[module.TOOL]
            ):
                raise ValueError(f"Channel mismatch in {item['trace']} at chunk width {width}")
            replays += 1
        rows.append({
            "trace": item["trace"], "case": item["case"], "semantic_status": case["status"],
            "forced_positions": [
                event["new_tokens_before"] for event in trace["events"]
                if event["kind"] == "forced_token_sampled"
            ],
            "api_finish_reason": public["finish"],
        })
    if len(rows) != 6 or replays != 24:
        raise ValueError("Expected six traces and 24 parser replays")
    return {
        "scope": __doc__, "parser_sha256": PARSER_SHA,
        "traces": len(rows), "parser_replays": replays,
        "all_channel_matches": True, "semantic_failures_preserved": sum(
            row["semantic_status"] != "pass" for row in rows
        ),
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(check(args.evidence.resolve()), indent=2))


if __name__ == "__main__":
    main()
