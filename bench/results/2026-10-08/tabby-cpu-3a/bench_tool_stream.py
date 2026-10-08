"""CPU-only parser microbenchmark: python -m tests.bench_tool_stream.

This measures tool-delta handling, not GPU generation or end-to-end API speed.
Each run verifies byte-identical arguments against the completed-call parser.
"""

import argparse
import json
import platform
from statistics import median
from time import perf_counter

from endpoints.OAI.utils.toolcall_stream import QwenToolCallDeltaStreamer
from endpoints.OAI.utils.toolcall_formats.qwen3_coder import parse_toolcalls


def benchmark(sizes, chunk_chars=24, repeats=3):
    schema = [
        {
            "function": {
                "name": "write_file",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
            }
        }
    ]
    results = []
    line = '    return "hello world"\n'
    for size in sizes:
        value = (line * (size // len(line) + 1))[:size]
        raw = (
            "<tool_call><function=write_file><parameter=content>\n"
            + value
            + "\n</parameter></function></tool_call>"
        )
        expected = parse_toolcalls(raw, tools=schema)[0].function.arguments
        durations = []
        for _ in range(repeats):
            streamer = QwenToolCallDeltaStreamer(tools=schema)
            fragments = []
            start = perf_counter()
            for i in range(0, len(raw), chunk_chars):
                fragments.extend(
                    delta["function"].get("arguments", "")
                    for delta in streamer.feed(raw[i : i + chunk_chars])
                )
            durations.append(perf_counter() - start)
            if "".join(fragments) != expected or streamer._assembled != [expected]:
                raise AssertionError("streamed argument mismatch")
        results.append(
            {
                "argument_chars": size,
                "chunk_chars": chunk_chars,
                "repeats": repeats,
                "seconds": durations,
                "median_seconds": median(durations),
            }
        )
    return {"python": platform.python_version(), "machine": platform.machine(), "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8192, 65536, 262144, 1048576])
    parser.add_argument("--chunk-chars", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(n <= 0 for n in [*args.sizes, args.chunk_chars, args.repeats]):
        parser.error("sizes, chunk chars and repeats must be positive")
    print(json.dumps(benchmark(args.sizes, args.chunk_chars, args.repeats), indent=2))


if __name__ == "__main__":
    main()
