# Final Q8 numerical checks of cached prefill chunk sizes

The matched **2048 / 2048** experiment passed every unchanged core gate with **zero investigation alerts** on the original and final engines. All 36 paged comparisons and 18 unique complete-prefill comparisons report a maximum absolute logit difference of zero. The recipe retains its 2048-token cached prefill chunk.

The optional **1024 / 4096** comparison passed the core gates but produced three KL investigation triggers across two cases. That setting was not selected for deployment. Its baseline and candidate used different cached prefill chunk sizes, so it does not isolate an engine-source change.

## Sources and method

- Original engine: `94ba01d50a13fa9ff672473f2d0eef8b51a71e99` in the preserved original runtime.
- Final engine: `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`, after the ordinary final Spark setup.
- Pack: the original `flashnext-exl3-3.05bpw` directory, with its PLE/ngram table in RAM in both members of each pair.
- Actual paged storage: K8/V8, with the QSA planes remaining FP16. The adapter records the allocated cache classes, bits, shapes and dtypes.
- Target-only inference: no MTP model is loaded. Prefixes are 255, 1023 and 4095 tokens; query lengths are 1 and 6; each continuation has 48 fixed teacher-forced positions per batch row. The code, tools and multilingual corpora are unchanged.
- The frozen format-2 probe, Q8 adapter, assessor and gates are copied under `raw/`. Candidate reports compare against saved original-engine logits for the same path and exact input IDs. Cache-free complete-prefill references are scored once per corpus, batch size and prefix, separately from the paged cases.

`prefill_chunk` controls cached prompt ingestion. It does not split the separate cache-free complete-prefill reference. The matched 2048 pair runs batch sizes 1 and 4 within each runtime; its candidate points to the newly generated original-2048 report and logits in the same result directory. The older 1024 baseline is used only by the separate 4096 experiment.

These are numerical regression checks on a small correlated corpus. They do not certify broad model capability, tool calling, speculative-generation acceptance, throughput, long-context serving or the final service. The serving/API checks are separate. Vocabulary conventions and the padding caveat are unchanged from [the numerical method](../../../numerical/README.md).

## Results

| Original / final cached prefill chunk | Batch sizes | Paged cases / positions | Core failures | KL investigation triggers | Pooled paired NLL delta | Additional high-confidence disagreements |
|---|---|---:|---:|---:|---:|---:|
| 1024 / 4096 | 1 | 18 / 864 | 0 | 3 across 2 cases | +0.002565432 nats | +1 |
| 2048 / 2048 | 1, 4 | 36 / 4,320 | 0 | 0 | 0 | 0 |

| Experiment | Unique complete-prefill cases / positions | Maximum absolute paired logit difference | Paired NLL delta | Additional high-confidence disagreements |
|---|---:|---:|---:|---:|
| 1024 / 4096 | 9 / 432 | 0 | 0 | 0 |
| 2048 / 2048 | 18 / 2,160 | 0 | 0 | 0 |

For the matched 2048 pair, **every paged comparison also reports maximum absolute difference 0 and KL 0**. Both probes exited successfully and passed their sanity checks; the assessor exited 0. The paired run completed at `2026-10-08T13:27:02.531220+00:00`.

Each runtime has 11 paged high-confidence disagreements when compared with its own cache-free reference in this 2048 cohort. Those counts and the actual paired paged outputs are unchanged between original and final; the added disagreement count is zero. The whole-profile API qualification must still be read separately.

Here, an exact paired result means the reported finite-logit maximum absolute difference is zero. This archive does not claim a new binary comparison or hash of the large tensor artifacts during packaging.

### Retained 4096 investigation

All 12 short-prefix paged comparisons (255 and 1023 tokens) and all nine unique cache-free references report zero maximum absolute paired difference. The three investigation triggers occur at prefix 4095:

| Case | Mean paired KL | p95 paired KL | Paired NLL delta | Added high-confidence disagreements |
|---|---:|---:|---:|---:|
| Code, batch 1, query length 1 | 0.021784963 | 0.053746741 | +0.031686075 nats | 0 |
| Multilingual, batch 1, query length 6 | 0.216099083 | 0.184732080 | -0.019090265 nats | 0 |

The code case triggers the mean-KL investigation threshold. The multilingual case triggers both mean and p95 thresholds while its next-token accuracy remains 48/48 and its NLL improves. A separate tools case at prefix 4095/query length 6 adds one high-confidence disagreement, with NLL delta +0.012007107 nats; both remain within the predeclared core limits. The retained result is an investigation, not a core failure, and 4096 was not promoted.

## Files and integrity

`summary.json` contains the compact derived metrics and links to both full assessments. `prepared-pair-manifest.json` preserves the reviewed two-job file hash, exact command, frozen tool hashes and expected counts. `raw/` contains the original small reports, job inputs, logs, scripts and gate declaration, including the historical original-1024 report needed to audit the 4096 comparison.

The copied reports were checked against the assessor's recorded JSON and gate hashes. The copied jobs and tools match the prepared manifest. `SHA256SUMS` hashes only the files included in this small archive. All `.safetensors` outputs remain at their recorded Spark paths; they were neither copied, opened nor hashed during this collection, while the final API benchmark batch was allowed to proceed.

The frozen matrix runner reports successful experiment completion separately from numerical qualification. Read the assessor's `core_passed` and `investigation_count` fields; a runner exit of 0 alone is not a deployment approval. Repeating a job requires a new owned result directory and corresponding new baseline paths; the preserved evidence must not be overwritten.
