# Flat 4.05 supported-stack attribution — October 9, 2026

The fresh controlled comparison **did not reproduce the historical 4.05 slowdown**. All three supported stacks completed their owned lifecycle. The final stack's measured medians were 0.55% higher for code and 1.21% higher for DevOps than the fresh original stack, with the same measured answers and aggregate draft counts. These small differences do not establish a general speed improvement or statistical equivalence. No performance patch was selected from this experiment.

| Cell | Engine / Tabby | Code decode median (tokens/s) | DevOps decode median (tokens/s) | Measured requests |
|---|---|---:|---:|---:|
| A | Original 94ba01d / original 816c321 |75.75|66.31|6|
| B | Qualified 24f0dec / original 816c321 |76.27|66.77|6|
| C | Qualified 24f0dec / published f650bb5 |76.17|67.11|6|

Each cell used a fresh server and sent one warmup plus three measured repetitions for each of code and DevOps: **24 requests total, 18 measured**, each producing 400 output tokens. The run began at `2026-10-09T06:31:22.343260+00:00` and completed at `06:36:22.892463+00:00`. All servers received owned SIGTERM cleanup, exited successfully and left their owned process groups empty. See [outer result](reports/result.json) and the per-cell reports.

## What was held fixed and observed

The unchanged public recipe a8c72bdf811646f413fdc03a4e49811a2753d0cf and benchmark client used the original `overnight-v1` payloads, thinking false, temperature 0, top-k 1 and seed 0. All measured request hashes and actual prompt/cache/output token counts matched the original fixtures and the other cells. The original flat4.05 pack used disk PLE placement, Q8 KV, a 262144-token shared cache/context, one request slot, chunk 2048, dynamic MTP depth 5/confidence 0.6, draft head 65536, row budget 0 and the recorded big-core affinity. The numerical compatibility settings were explicit in every cell; old versions may ignore settings introduced later. No source, venv, model, installed package or autotune cache was changed by the controller.

All **18 measured response hashes** match their corresponding workload/repetition across A/B/C. Corresponding accepted/rejected prediction-token totals also match. Those aggregate counters do not prove identical individual verification-window shapes or ordering. The source review found that these plain, thinking-disabled requests do not install the producer reasoning-budget guard. The native consumer loop is unchanged between the two Tabby revisions; neither finding is a profiler measurement.

A→B changed both engine and preserved venv. Package differences include ExLlamaV3 1.5.1→1.6.0.post1, llguidance 1.8.0→1.9.1, and added CPU testing packages (`iniconfig`, `pluggy`, `pytest`). This is a supported-stack comparison, not an isolated binary-only experiment. B→C retained engine and package versions and changed Tabby source. Actual imports and source paths were checked before serving; the original server's ordinary minimum-version check accepted the final engine. The unsupported original-engine/new-server combination was not run.

A→B median changes were +0.69% in each workload. B→C changes were −0.13% for code and +0.51% for DevOps. The new server emitted one additional SSE event per measured request, consistent with its initial role frame; this does not represent an additional GPU verification window. Reported server durations are rounded to 10ms. Three repeats in a fixed sequential order are insufficient to treat these sub-percent differences as robust speedups.

## Historical and hardware context

The fresh original A stack reproduced all six corresponding original answers and draft totals, while its own medians were below the October 8 original measurements of 77.02/67.76 tokens/s. Therefore, the earlier day-to-day gap cannot be attributed solely to the source update. The [historical analysis](analysis/historical-attribution.md) preserves earlier output/draft comparisons; its absolute reference paths identify the unchanged October 8 evidence, not files created by this experiment.

Sparse resource observations were captured every two seconds during load and benchmark phases. Benchmark-phase observations include warmups: A ranged 2405–2522MHz and 48–64°C; B 2457–2522MHz and 56–65°C; C 2489–2522MHz and 61–66°C. These are observed ranges, not peaks or clock-normalized measurements. Temperature, sequential order and normal scheduling variation remain limitations. The [derived summary](analysis/summary.json) retains every measured row, sparse clock/thermal sample, source identity and package-version difference.

Root captured the shared autotune file at the run boundary. Both [before](boundaries/autotune-before/identity.json) and [after](boundaries/autotune-after/identity.json) copies are 19,760 bytes with SHA256 `ae8cac3b458863cd2a67254f23af88aaeab1310bc8f3f29750ea081735040466` and identical recorded mtime. The controller did not clear or alter the cache. This bounds shared-cache drift between those snapshots, without making a broader claim about other caches.

## Historical dependency exception and excluded attempts

A's actual `pip check` is preserved as **exit 1**, with the sole output `nvidia-cusparselt-cu13 0.8.1 is not supported on this platform`. A read-only verification independently matched the exact official old package, incorrect `manylinux2014_sbsa` WHEEL tag, AArch64 ELF library, RECORD hash and library SHA256 to the prior proof. This narrowly permits a historical attribution control; it is **not a passed full setup check or deployment qualification**. No old wheel metadata was repaired. B and C passed `pip check`. Fresh import checks were CPU-only and verified that CUDA was not initialized.

The first launch failed before an output directory or server was created because the exact model audit file had not yet been staged. Its original [launch record](boundaries/flat405-attribution-launch.json) and [failure log](boundaries/flat405-three-cell-73d179bd.log) are retained. The [relaunch record](boundaries/flat405-attribution-relaunch.json) also records an intervening refusal while another cooperative GPU-lock owner was active. Only the completed A/B/C sequence enters the denominator. No separate full log of that intervening ownership refusal was supplied in the collected paths; its existence is attributed to the root's relaunch record. Neither preparation failure was treated as a model/performance result.

The [model input audit binding](model-input-audit/current-models.json), SHA256 `6dc2117bb602a3f57e798613c005c0c1ec8ad519378168181b00465e40b90840`, supplies exact current metadata/config/tokenizer/loader-order identities and bounded safetensors headers. The controller rechecked that identity and headers. It did not read or hash full model payloads. The separate [audit evidence](../model-input-audit/README.md) explains the limits of comparing current metadata with historical records.

## Evidence and review

- [reports](reports/) preserves every small completed result, request/response report, server/client log, resource sample, deployment/config snapshot and preflight output byte-for-byte. Model symlinks are never followed.
- [frozen bundle](sources/flat405-attribution-73d179bd.tar.gz) and [unpacked source](sources/flat405-attribution/) preserve the exact used controller, helpers, original fixtures, CPU proofs and checksum manifest. Controller SHA256 is `73d179bd97a8b5870421b7f6e6c3757a8e7c620c8dab0757276b6ce19a751a77`.
- Nineteen CPU contract checks passed, independently repeated by the recipe peer. Before the live run, peer review caught and resolved inherited blocked child signals; the final test proves SIGTERM/SIGINT remain unblocked after exec and owned cleanup uses SIGTERM successfully. No earlier controller was used for the completed measurements.
- [collection receipt](collection-receipt.json) maps source paths to copied bytes/hashes and records omitted incidental Python caches and model symlinks. Every remote regular-file read had unchanged inode, size and mtime before/after. No model weights, large installed libraries or credential stores were copied.
- [release review](release-review.json) verifies all copied identities and applies bounded credential recognizers to regular files and all frozen-tar members. It reports zero candidates for its declared patterns and sensitive JSON fields. This is not proof against arbitrary unlabeled secrets. The intentional `API_KEY="secret"` value in a fake CPU-test environment is a static fixture, not a runtime credential.

The archived controllers retain host-specific paths and are source evidence; do not execute them automatically. Frozen preparation READMEs remain unchanged and may describe the then-pending run. The new page and summary describe the completed result. This collection and analysis made no model/API/GPU calls or service/source changes. Verify this directory with `sha256sum -c SHA256SUMS`.
