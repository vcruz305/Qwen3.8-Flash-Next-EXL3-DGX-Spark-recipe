# Synthetic reasoning-budget observer

This is a temporary diagnostic, not a deployment or server feature. The parent controller owns the server, API requests and cleanup. Agents only prepared/reviewed this code on CPU.

## Question and scope

The unchanged `auto_reasoning_stream` fixture at engine `16ca20d27c0e4cce15a9bbc131e6d047065395b5` / Tabby `f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6` produced valid SSE with extra thought-like text in final content after a 24-token reasoning cutoff. The corresponding non-streaming case passed. The exact3a/f4 prompt, grammar, collector and parser comparison is recorded in the adjacent `prompt-and-sse-analysis.json`.

This observer matches only the SHA256 of that synthetic326-token prompt. It records the actual raw backend response before channel parsing, budget injection request/acceptance, native forced-token position and ID, queued event count, and phase switches. It does not force a response, alter a fixture, change sampling/filter settings, copy GPU tensor values to CPU, or add awaits while generation is running. Events stay in memory until the collector finishes; then a private JSON file is published atomically without overwrite. Original return values and exceptions are preserved. Matching requests beyond the configured cap fail clearly instead of recording indefinitely.

## Files and startup

Copy this directory to a new task-local directory on Spark, for example:

`/home/cruzspark/qwen-overnight-20261008/reasoning-observer-f4`

Create a separate private JSON configuration. `output_dir` must not exist; its parent must exist.

```json
{
  "tabby_repo": "/home/cruzspark/qwen38-exl3-20261008/tabbyAPI",
  "engine_repo": "/home/cruzspark/qwen38-exl3-20261008/exllamav3",
  "tabby_commit": "f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6",
  "engine_commit": "16ca20d27c0e4cce15a9bbc131e6d047065395b5",
  "output_dir": "/home/cruzspark/qwen-overnight-20261008/results/reasoning-observer-f4/raw",
  "max_records": 12
}
```

Pass these environment values only to the parent-owned diagnostic server process, preserving its existing runtime/model/profile controls:

```bash
PYTHONPATH=/home/cruzspark/qwen-overnight-20261008/reasoning-observer-f4
TABBY_REASONING_OBSERVER_CONFIG=/home/cruzspark/qwen-overnight-20261008/reasoning-observer-f4/config.json
```

If an existing `PYTHONPATH` is needed, prepend the observer directory. The existing `serve.sh` and its command remain unchanged. The standard `sitecustomize.py` mechanism is inert for all commands except the exact configured `main.py` path. It wraps only that main module's initial `asyncio.run(entrypoint_async())` call. Native/backend imports are delayed until **after** normal config/allocator setup. The ordinary run function is restored during activation. Both source revisions and clean tracked trees are checked before backend imports; actual imported module paths are checked too.

The observer is not active unless `raw/observer-manifest.json` appears with the expected source and observer hashes. A missing manifest or missing matching request files is an incomplete diagnostic, never a passing trace.

## Bounded parent-controlled replay

Keep engine16ca/Tabbyf4 and the same single/RAM/K8V8/MTP5 profile and compatibility flags. After the controller's normal source/model/listener gates pass, run the packaged **unchanged** `bench/auto_compatibility.py` suite three times serially with unique output paths and the normal deployment metadata. Each suite contains9 cases, including the same two reasoning-budget requests; expect **six** matching trace files across the three suites. Keep all successes/failures and actual exits. The observer cap12 allows a separately labelled control only if the first traces leave a concrete question unresolved; it is not a request to run extra cases automatically. Do not use this instrumented run for performance claims.

The controller must stop only its owned server group and retain the original `api-f4-gemm` failure report. A replay pass does not erase that observed failure.

## Interpretation

Pair a trace `request_id` with its `cmpl-` or `chatcmpl-` response ID in the saved API report. Locate the actual `</think>` in `raw_finish.text`. Replay that raw text through the unchanged parser, with `start_in_reasoning=True`, to check that parsed reasoning/content match the saved API channels. The raw string alone is decoded backend output, not an independently captured complete token-ID sequence. The forced-token event records the actual native producer position and the already-CPU forced ID.

If leftover quotation text follows the real forced closing marker and CPU routing reproduces the API channels, the continuation is model behavior after the cutoff. If the marker or reconstructed channels disagree, investigate parsing or event ordering. Compare `forced_token_sampled.new_tokens_before` between matching requests to establish whether the consumer-timed budget cut at different producer token positions. Do not infer causation from protocol-valid SSE alone.

A generic budget-ending message or producer-side budget enforcement would be a separate runtime change requiring independent CPU tests and an unchanged live replay. This observer makes neither change.

## CPU checks

```bash
cd /home/vcruz/src/qwen-overnight-20261008/tabbyapi-diagnostics/reasoning-budget-f4/observer
/home/vcruz/src/qwen-overnight-20261008/tabbyapi-agent/.venv-tools/bin/python -m unittest -v test_reasoning_observer_cpu
```

Fourteen checks cover return/exception identity, prompt isolation, raw/injection/phase metadata, the bounded record cap, non-overwrite/atomic publication, absence of GPU reads, and startup timing after the matching main coroutine is invoked.
