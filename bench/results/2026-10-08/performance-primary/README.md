# Completed primary performance matrix — 8 October 2026

This archive preserves the **17 completed primary jobs** measured from
10:47:46 to 11:46:59 UTC. All performance clients completed and every server was
stopped by its owning controller with SIGTERM and exit code 0. **13 whole jobs
passed; four retained failed tool clients.** Completed measurements and complete
product qualification are different checks here.

The measured sources were:

| Component | Exact identity |
|---|---|
| Engine | `16ca20d27c0e4cce15a9bbc131e6d047065395b5` |
| TabbyAPI | `f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6` |
| Recipe | `218bd438225243e795383f10ef7a886b06d5d839` |
| Frozen experiment controller SHA256 | `48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586` |
| Submitted primary17 JSON SHA256 | `ba8f2d7e9dc24798f13cc7ca4c2b65c96fb7c7fa11290ffc2c85dabe568b1850` |

These results predate the later producer-side reasoning boundary fixes and the
Cyberfrost template override. They do not establish the speed or correctness of
a later engine/server pair. The four model packs kept their original templates.
The Cyberfrost template forced thinking on despite the request's explicit false
flag; the same template and exact requests were used on both sides of its
original-to-primary comparison.

## Results and scope

The [full generated summary](summary/summary.md) contains all 43 single-request
case rows, six concurrency rows, complete client gates and eight controlled
setting comparisons. There are 122 measured single-request requests,
42 measured concurrent requests and 192 tool checks
(182 passed). Warmups are preserved but excluded from the quoted medians.
The [original comparison](comparison-original.json) verifies identical request
SHA256 values, sampling settings, prompt hashes and actual token counts for all
16 original-to-primary comparison rows. It uses the existing
[original API archive](../original-api/README.md).

The baseline-aligned primary decode medians were:

| Pack | PLE table | Code tok/s (change) | DevOps tok/s (change) | Prose tok/s (change) | Tool smoke |
|---|---|---:|---:|---:|---:|
| Flat 3.05 | RAM | 80.05 (-2.96%) | 74.01 (-0.13%) | 53.56 (-4.24%) | 28/28 |
| Flat 4.05 | Disk | 76.18 (-1.09%) | 66.37 (-2.05%) | 47.64 (-1.91%) | 24/28 |
| SAGE 4.15 | Disk | 52.40 (+7.42%) | 49.85 (+7.25%) | 33.76 (+9.79%) | 28/28 |
| Cyberfrost 3.87 | Disk | 44.07 (+7.15%) | 37.24 (+7.54%) | 34.79 (+8.41%) | 26/28 |

The flat packs did not improve across these short suites. The mixed packs did.
On unchanged primary sources, toggling only `EXL3_MOE_MIXEDK_NOSYNC` accounts for
6.38–10.38% higher mixed-pack decode rates in the measured suites. SAGE passed
8/8 selected tool cases with either value; Cyberfrost failed its two exact-string
cases under both settings. Its throughput remains reportable, with those failures
retained.

The 3.05 prefill-only chunk comparison measured 688.79→827.98 tokens/s with
chunk 2,048→4,096 (+20.21%), while median first-output time fell
17.8125→14.8448 seconds (−16.66%) and decode fell 57.785→54.905 tokens/s (−4.98%).
These are two measured requests per setting at 12,157 actual prompt tokens and
256 output tokens. Larger chunks are a latency/throughput tradeoff.

For 3.05 with the PLE table streamed from disk, four batch slots and a shared
1,048,576-token K8/V8 pool, `EXL3_DRAFT_ROW_BUDGET=8` improved aggregate end-to-end
throughput from 34.2972→47.2729 tokens/s at two requests (+37.83%), and
36.8254→50.4184 at four (+36.91%). Each request actually contained 849 prompt
tokens and emitted 256 tokens; every measured request reported zero cached tokens.
This does not test four simultaneous full-context requests. Both settings passed
8/8 selected tool checks. The [validation report](../../../../VALIDATION_2026-10-08.md)
explains the separate prefill, drafting and concurrency comparisons.

The 4.05 control failed both exact-string tests and both adversarial required-tool
tests (an invented no-argument parameter and an unfinished wrapper). Cyberfrost
failed both exact-string tests in its main control and each NOSYNC variant. No
failed tool result was converted into a passing request or removed from the raw
reports. Later fixes require their own retained evidence.

## Contents and integrity

- `reports/<job>/result.json` and its unchanged attempt copy record exact commands,
  exits, model identity, source hashes, resolved tuning and owned cleanup.
- Each attempt retains all client JSON, `deployment.json`, rendered `state/config.yml`,
  `resources.json`, and the raw server log. Configuration and source files are
  historical evidence, including comments that may have been superseded.
- `source-provenance.json` records the source path, size, timestamp and raw SHA256
  of each copied Spark file. It also records hashes and bounded diagnostics for
  35 omitted client stdout logs, which mostly duplicate their retained JSON.
- `server-log-summary.json` records startup duration and bounded warning/error/
  lifecycle lines; full server logs remain available beside the reports.
- `inputs/` preserves the submitted jobs, launch/preflight records, the exact
  controller and the nine recipe source files whose hashes the controller checked.
  Those source files were recovered from the recorded recipe commit and their
  hashes verified against every job; they are not a complete recipe checkout.
- `integrity.json` records post-collection checks. `SHA256SUMS` covers raw bytes
  of every archive file except the checksum list itself. Controller
  `config_sha256` and `deployment_sha256` use canonical JSON serialization and
  intentionally differ from raw-file SHA256 values.

Model metadata hashes, weight file sizes/timestamps and actual loader enumeration
order are retained. **Full weight hashes were not computed** by the matrix or this
collection. No model tensor files are copied. Sparse 10-second resource observations
are not peak-memory or energy guarantees, and startup timings are not controlled
old-versus-new compilation comparisons.

## Recompute without a model or GPU

From this directory, use new output names (the scripts refuse an existing output):

```bash
sha256sum -c SHA256SUMS
python3 summarize_completed_performance.py \
  --reports reports --jobs inputs/primary17.json --output reproduced-summary
python3 compare_original.py \
  --original ../original-api --output reproduced-original-comparison.json
```

The summarizer is frozen with SHA256
`d1292bba6e63d8cddbf25afe553c923aabe2a747d8a9f6d064e520376770e660`.
Both utilities read retained reports only. Generated timestamps and selected
output paths may differ; request hashes, measured values and gate results should
agree. Use the current [matrix runner guide](../../../../bench/matrix.md) for new
experiments. Replaying this historical server lifecycle requires the complete
recorded recipe and engine/server checkouts; running only the copied source
fragments is not a supported installation.
