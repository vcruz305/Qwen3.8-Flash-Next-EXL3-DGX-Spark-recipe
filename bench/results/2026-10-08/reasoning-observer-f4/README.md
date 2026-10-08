# Reasoning-budget observer: retained f4 failure and raw parser evidence

This archive records a completed diagnostic with **two 9/9 automatic-tool suite passes and one 8/9 result** on the same loaded 3.05 bpw model. The third suite's `auto_reasoning_stream` response continued an interrupted quotation after the forced reasoning close. Its SSE protocol completed, but the final answer did not meet the unchanged fixture's exact `READY` requirement. The controller correctly retained `passed: false`.

Six narrowly selected raw backend traces explain the channel boundary. **All 24 CPU parser replays match the saved API reasoning and content exactly.** Five successful reasoning requests inserted the requested 24-token cutoff at producer position 24; the failing stream inserted it at position 28. This shows that the consumer-timed cutoff varied in this run and was associated with the failure. It does not prove that every cutoff at 24 will yield a correct answer, or that the marker alone caused every observed difference.

## Runtime and measurement boundary

The live run took place on the DGX Spark from **2026-10-08 10:38:01 to 10:39:22 UTC**. Exact source and deployment identities are recorded in [the controller result](live-observer/result.json), [deployment snapshot](live-observer/deployment.json), and [observer manifest](live-observer/raw/observer-manifest.json).

| Component | Recorded source |
|---|---|
| Engine | `16ca20d27c0e4cce15a9bbc131e6d047065395b5` |
| TabbyAPI | `f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6` |
| Recipe | `218bd438225243e795383f10ef7a886b06d5d839` |
| Pack | `/home/cruzspark/models/flashnext-exl3-3.05bpw` |
| Profile | Single request, n-gram RAM, 262,144-token shared pool, K8/V8 cache |
| Draft settings | MTP depth 5, dynamic confidence 0.6, requested head size 65,536 |
| Chunk size | 2,048 |

The run used the compatibility settings captured in `resolved_tuning`, including GDN projection/conv controls, legacy attention partitions, legacy GEMM tiles, mixed-K cooperative mode off and mixed-K no-sync mode on. It precedes the later explicitly controlled KSPLIT=1 matrix and the producer-budget changes. This archive is not evidence for those later configurations.

The temporary observer matched only the SHA-256 of one **synthetic 326-token prompt**, with a six-record limit. It recorded raw backend text before channel parsing, existing CPU producer positions, the injected native token ID, phase events and queue size. It added no generation await or GPU tensor read. Its hooks still add instrumentation overhead, so these results and timings are **not throughput measurements**. No synthetic tool was executed.

## Outcomes retained without selecting only successful replays

| Suite | Full unchanged suite result | Client exit | Reasoning nonstream cutoff | Reasoning stream cutoff |
|---|---:|---:|---:|---:|
| [auto1.json](live-observer/auto1.json) | 9/9 | 0 | 24 | 24 |
| [auto2.json](live-observer/auto2.json) | 9/9 | 0 | 24 | 24 |
| [auto3.json](live-observer/auto3.json) | 8/9 | 1 | 24 | 28 |

The full suites contain 27 case executions in total. Only their two matching reasoning cases per suite were observed, producing six traces. The **24 replays are six raw responses partitioned four ways**, at character chunk widths 1, 7, 31 and the full response; they are not 24 new model requests or independent quality samples.

In [request-05.json](live-observer/raw/request-05.json), the forced token is `248069` (`</think>`). The raw response contains quotation continuation after that real close and a later second close. The unmodified parser routes the intervening text to content, matching the API response. The trace records zero queued results when injection was requested; it does not capture an independent complete token-ID sequence or establish all scheduling causes. Normal `stop` and stream `[DONE]` indicate protocol completion, not fulfillment of the requested answer.

The server remained alive through the last verification and then exited normally after the controller signaled its verified process group. The recorded group was empty after cleanup. See `server_cleanup` in the result rather than treating an absence of a process as sufficient proof of controlled cleanup.

## What the analyses establish

[raw-parser-analysis.json](raw-parser-analysis.json) retains the original 24 successful routing comparisons, all six cutoff positions, raw strings and paired API report hashes. No routing mismatch was found in this bounded sample.

[prompt-and-sse-analysis.json](prompt-and-sse-analysis.json) compares the earlier 3a and f4 requests, rendered prompt IDs, grammar and relevant source functions. For these particular no-argument ping cases, the nullable-argument guidance introduced in f4 did not apply. The prompt and relevant routing/budget code were unchanged. The earlier [3a report](api-3a-compat-auto.json) and the original [f4 failure report](api-f4-gemm-auto.json) are preserved. Saved SSE alone omitted the raw closing tags and could not establish the injection position; the later observer supplied that missing evidence.

This is a small synthetic regression investigation. It does not measure general reasoning quality, establish a failure rate, validate every tool schema, or show that later candidate fixes have passed. Any new budget enforcement or injection wording needs separate, unchanged live tests.

## Reproduce the CPU routing check

Only the standard Python library is needed:

```bash
python3 bench/results/2026-10-08/reasoning-observer-f4/replay_channels.py
```

The portable checker verifies the frozen f4 parser SHA and the original trace/report hashes before replaying. It prints six trace rows, `parser_replays: 24`, `all_channel_matches: true`, and one preserved semantic failure. It does not connect to a server or write results. Its source is [replay_channels.py](replay_channels.py); the exact upstream parser snapshot and license are under [source/](source/).

The original preparation and analysis scripts are retained under [controller/](controller/), [observer/](observer/) and [historical-analysis/](historical-analysis/). Their original account paths and source guards remain unchanged. They are historical experiment source, not a drop-in deployment entrypoint. In particular, the controller starts a server and sends API requests when executed as a program; do not run it merely to inspect this archive. The old analysis scripts also expect their original checkout layout. Use the portable routing checker above for an offline verification.

## Provenance and verification

[provenance.json](provenance.json) records small-file read-only copies from the completed Spark run and byte comparisons with the retained WSL sources. The controller source matches its execution-time `controller_sha256`; observer, startup hook, config, deployment and six raw trace hashes match their recorded references. Frozen parser and environment source came from the specified Git commits. No model tensor bytes were read during this packaging step.

[packaging-validation.json](packaging-validation.json) records the archive checks, relocated CPU replay and CPU test outcomes. [SHA256SUMS](SHA256SUMS) covers every archived file except the checksum list itself. Original JSON, observer code, controller code, analysis reports and synthetic responses are retained byte-for-byte.
