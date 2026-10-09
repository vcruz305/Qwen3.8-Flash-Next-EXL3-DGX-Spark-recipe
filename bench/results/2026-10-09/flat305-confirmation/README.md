# Flat 3.05 RAM confirmation — October 9, 2026

The fresh original-versus-qualified comparison **did not reproduce the historical 3.05 slowdown**. Both supported stacks completed successfully. The qualified stack measured +0.68% for code and +0.93% for prose, with matching measured answers and aggregate draft work. The experiment supports retaining the qualified engine rather than adding a speculative performance patch; these small differences are not a general speedup or statistical-equivalence claim.

| Cell | Engine / Tabby | Code decode median (tokens/s) | Prose decode median (tokens/s) | Measured requests |
|---|---|---:|---:|---:|
| A | Original 94ba01d / original 816c321 |79.45|53.87|6|
| C | Qualified 24f0dec / published f650bb5 |79.99|54.37|6|

There were **16 requests total, 12 measured**: each fresh server received one warmup plus three measured repetitions for each of the two original workloads, with 400 actual output tokens per request. The owned controller began at `2026-10-09T06:51:33.735691+00:00` and finished at `06:55:15.455502+00:00`. Both servers received SIGTERM, exited 0, and left their owned process groups empty. The [outer result](reports/result.json), [launch record](boundaries/flat305-confirmation-launch.json) and [controller log](boundaries/flat305-confirmation-f6e384ce.log) are retained unchanged.

## Conditions and interpretation

The run used exact public recipe a8c72bdf811646f413fdc03a4e49811a2753d0cf, the original flat3.05 model directory, **RAM PLE placement**, Q8 KV, context/cache 262144, one request slot, chunk 2048, dynamic MTP depth 5/confidence 0.6, head 65536, row budget 0, recorded big-core affinity and the qualified numerical compatibility environment. The unchanged original code/prose payloads use `overnight-v1` salts, thinking false, temperature 0, top-k 1, seed 0 and cold cache. Every request hash and actual prompt/cache/output count matches the frozen original fixture. The retained `--old-tabby` compatibility argument is not a third cell or imported source.

All 12 measured answers match the corresponding repetition in the other cell, and accepted/rejected prediction-token totals match as well. Code's server completion duration decreased by 30ms in each repetition; prose decreased by 50–70ms. Reported durations are rounded to 10ms. The final server emits one additional SSE event per request, consistent with its initial role frame; SSE counts are not GPU verification-window counts. Aggregate draft totals do not prove identical per-window batching or operation order.

This compares complete supported stacks: engine, Tabby and the preserved venv change together. The package differences are ExLlamaV3 1.5.1→1.6.0.post1, llguidance 1.8.0→1.9.1 and added CPU testing packages. It does not isolate a single kernel or server function. The complementary [4.05 three-cell comparison](../flat405-attribution/README.md) provides the separate same-engine server-source comparison. Neither experiment justifies a new performance patch.

Sparse two-second benchmark-phase observations include warmups. A's observed clocks were 2496–2535MHz and temperatures 51–62°C; C's were 2405–2515MHz and 52–65°C. These are sampled ranges, not peaks. Sequential order, three repetitions and ordinary scheduling/thermal variation limit precision. The [summary](analysis/summary.json) retains every measured row, source/package difference and observed sample.

The old stack's October 8 medians were 82.49/55.93 tokens/s. In the new A run, only two of its six measured answers match that historical original, and **none** of its six draft-total pairs match. Requests and actual token counts do match. Therefore, the day-to-day latency difference cannot be treated as fixed-work evidence of an update regression. The [explicit historical comparison](analysis/historical-original-comparison.json) records these distinctions without discarding mismatched trajectories.

The shared autotune snapshots [before](boundaries/autotune-before/identity.json) and [after](boundaries/autotune-after/identity.json) both contain 19,760 bytes, SHA256 `ae8cac3b458863cd2a67254f23af88aaeab1310bc8f3f29750ea081735040466`, and the same mtime. No cache was cleared or tuned. The exact [model audit](model-input-audit/current-models.json) binds current configuration/tokenizer metadata, file identity/order and bounded safetensors headers; no model payload was read or hashed. The separate [audit explanation](../model-input-audit/README.md) limits historical identity claims appropriately.

## Dependency exception, lifecycle and evidence

Original A's actual `pip check` remains **exit 1**, solely for `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`. A fresh read-only package/library/ELF/RECORD check matched the independently established SBSA wheel-tag metadata defect. This is a narrowly admitted historical attribution control, **not a passed full setup check or deployment qualification**. No metadata was repaired. Final C's dependency check passed. Both imports verified exact source paths and no CUDA initialization during preflight.

The new controller f6e384ce is a mechanical derivative of the completed 4.05 controller 73d179bd: only model/audit key, RAM setting, A/C cell list and selected workload changed. Twenty CPU checks passed and were independently repeated, including all original305 payload hashes, actual usage checks, unchanged lifecycle AST proof and real harmless child signal/cleanup checks. Shared GPU flock, inactive supervisor gates, source/package/config/header drift checks, per-cell lock inode checks and owned cleanup remain intact. No engine, Tabby, recipe or venv source was modified by this experiment.

- [reports](reports/) preserves all small request/response reports, logs, resource samples, configuration/deployment snapshots, preflight outputs and cleanup records.
- [frozen bundle](sources/flat305-confirmation-f6e384ce.tar.gz) and [unpacked source](sources/flat305-confirmation/) preserve the exact used preparation, original fixture, parent controller, derivative diff, CPU logs/review and manifest. Bundle SHA256 is `8c21bd4483ec4b91993cf5a8927a4f0b7ecf353202d899ef63b830c7c81ec20f`.
- [collection receipt](collection-receipt.json) maps copied files to source paths, byte counts and hashes. Remote files had unchanged size, inode and mtime around each read. Model symlinks and incidental Python caches were omitted with recorded reasons; no weights, large libraries or credential stores were copied.
- [release review](release-review.json) verifies copied identities and applies bounded credential recognizers to the collected data and frozen-tar contents. Its scope does not guarantee absence of arbitrary unlabeled secrets. The fake CPU-test `API_KEY="secret"` is an intentional static fixture.

A local collection-script preparation typo initially selected a nonexistent results path and copied no evidence; it was corrected before successful collection. This was not a failed live measurement and changed no source result. Earlier preparation READMEs retain their original pending-run wording; this page describes the completed result. Archived scripts retain host-specific paths and must not be automatically executed. Collection, analysis and publication staging performed no inference, API, service or model mutation. Verify with `sha256sum -c SHA256SUMS`.
