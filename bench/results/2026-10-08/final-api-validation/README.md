# Final API validation on 8 October 2026

**Final collection: 5 of 5 declared jobs are archived.** 3 completed jobs passed every scheduled client. Original reports retain every failure and actual exit code.

The four single-model jobs use the final engine and Tabby source cohort below. The fifth job tests a four-slot 3.05 configuration, then serial benchmarks and long-context retrieval on that same loaded configuration. Job completion includes the owned server cleanup; the functional, benchmark, and retrieval outcomes remain separate.

## Source and configuration identity

| Component | Exact recorded revision |
| --- | --- |
| ExLlamaV3 | `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` |
| TabbyAPI | `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb` |
| Qualification recipe R1 | `3337bc8d64e4befafa2a1aff342e07abbea242ac` |

The explicit five-job input is [final-api-jobs-selected.json](inputs/final-api-jobs-selected.json), SHA256 `50def7795a0a7401cbf17d585d5f0955e23ed99968de58bda3d92c836335b915`. Each result records its actual source revisions and tracked-file state, resolved tuning, deployment and rendered configuration hashes, model path, configuration hashes, weight sizes/mtimes and natural loader enumeration order. Full weight hashes were not computed. Small model configuration files are copied in `source/models/`; no weights, logits tensors, tokenizer.json, or compiled extension binaries were read or copied by the archiver.

The sources retain Tabby's public branch label `main`; the actual tested commit is separately bound by setup and every job. The [upstream-head read](inputs/final-upstream-heads-prepromotion.json) at 13:42:41 UTC records the still-current official engine and Tabby ancestor heads. That file has SHA256 `ff1b28227f646c436a3a4c8500ef0a4af77b8359d802fe0d2d46c6e62edebce2`.

All jobs use Q8 K/V, chunk size 2048, dynamic MTP depth 5 and confidence 0.6, requested MTP head 65536, and the qualified compatibility controls listed in the input. Single jobs reserve a 262144-token pool and one request slot; 3.05 keeps PLE in RAM, while the larger packs stream it. The concurrent 3.05 job uses disk mode, a shared 1048576-token pool, four slots and draft row budget 8.

Cyber uses the explicit external template `exllamav3-tabby/templates/cyber-frost-3.87bpw-thinking.jinja`, SHA256 `666b82b29f5801f4f546e5724b45bf5f14be7d20b66149df44164626b072ce6d`. The saved deployment identifies the override and the recorded `/model` response retains its loaded content. The underlying pack files remain the original files.

## Actual completed client outcomes

Counts below are functional checks, not HTTP requests; several checks use multiple requests. Repeated auto suites remain distinct repeats of the same nine cases. Benchmark warmups, measured requests, and each three-needle retrieval check are listed separately.

| Job | Tools | SDK | Resilience | Auto | Literal with budget | Literal without budget | Concurrent checks | All scheduled clients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| final-305-single | 28/28 | 5/5 | 19/19 | 27/27 | 0/4 | 0/4 | Not scheduled | Failed |
| final-405-single | 28/28 | 5/5 | 19/19 | 9/9 | Not scheduled | Not scheduled | Not scheduled | Passed |
| final-415-single | 28/28 | 5/5 | 19/19 | 9/9 | Not scheduled | Not scheduled | Not scheduled | Passed |
| final-cyber387-single | 28/28 | 5/5 | 19/19 | 9/9 | Not scheduled | Not scheduled | Not scheduled | Passed |
| final-305-concurrent | Not scheduled | Not scheduled | Not scheduled | Not scheduled | Not scheduled | Not scheduled | 44/52 | Failed |

Independently counted functional totals in the completed reports: **306 passed and 16 failed**.

The 3.05 single job's original budgeted and unbudgeted literal clients each fail all four cases. Their reports preserve the returned value and complete wire responses. Additional raw-observer or thinking-disabled diagnostic runs have separate evidence and never replace these outcomes. The ordinary 28-case tool suite, SDK suite, resilience suite and auto suite use their unchanged fixtures.

The concurrent client retains 16 child reports and the cleanup record for every child. Its independent tally is 44 passed / 8 failed. All owned child groups are empty. The eight failures are the four budgeted and four unbudgeted literal-copy cases; the exact wrong values are preserved in the raw reports and [summary.json](summary.json).

## Short-request throughput

Each cell is the median server-reported decode rate from three measured 400-token outputs, after one excluded warmup, for the fixed code/devops/prose corpus. Sampling is greedy, thinking is disabled in the wire request, the run ID is `overnight-v1`, and every measured prompt reports zero cached tokens. Rates exclude prefill; TTFT and end-to-end rates remain in the raw reports and summary.

| Pack / configuration | Case | Original tok/s | Final tok/s | Observed change | Original → final prompt tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| final-305-single | code | 82.49 | 80.15 | -2.84% | 87 → 87 |
| final-305-single | devops | 74.11 | 70.95 | -4.26% | 63 → 63 |
| final-305-single | prose | 55.93 | 53.32 | -4.67% | 72 → 72 |
| final-405-single | code | 77.02 | 75.85 | -1.52% | 87 → 87 |
| final-405-single | devops | 67.76 | 66.45 | -1.93% | 63 → 63 |
| final-405-single | prose | 48.57 | 47.46 | -2.29% | 72 → 72 |
| final-415-single | code | 48.78 | 52.39 | +7.40% | 87 → 87 |
| final-415-single | devops | 46.48 | 49.73 | +6.99% | 63 → 63 |
| final-415-single | prose | 30.75 | 34.36 | +11.74% | 72 → 72 |
| final-cyber387-single | code | 41.13 | 45.75 | +11.23% | 196 → 160 |
| final-cyber387-single | devops | 34.63 | 44.28 | +27.87% | 172 → 136 |
| final-cyber387-single | prose | 32.09 | 34.36 | +7.07% | 181 → 145 |
| final-305-concurrent | code | — | 79.68 | — | 87 |
| final-305-concurrent | devops | — | 70.51 | — | 63 |
| final-305-concurrent | prose | — | 51.26 | — | 72 |

The original comparison files are retained in [original-api](../original-api/README.md). All nine measured wire request hashes and actual token counts match for each of 3.05, 4.05 and SAGE. Equal counts do not independently prove equal rendered token IDs. Response hashes match 4.05 code/devops, while the other compared cases contain output differences; the per-repeat hashes are exposed in the summary.

Cyber's wire requests match, but its corrected template changes actual prompt counts from 196/172/181 to 160/136/145. Its displayed rate differences therefore cover both the runtime and corrected prompt rendering. They do not isolate the contribution of engine changes. The small deterministic corpus and three repeats support observed rates, without a statistical-significance or broad task-quality claim.

Completed benchmark work contains 45 measured single requests plus 15 excluded warmups.

### Four-slot configuration, simultaneous benchmark requests

These are aggregate end-to-end rates: all completed output tokens in a round divided by the wall time of the whole round. The pool remains configured for four slots even for the one- and two-request measurements. Each level has three measured rounds after one warmup, 1024 approximate prompt tokens and 256 output tokens per request.

| Simultaneous requests | Aggregate end-to-end tok/s, median | Measured requests |
| ---: | ---: | ---: |
| 1 | 38.5724 | 3 |
| 2 | 46.4472 | 6 |
| 4 | 48.5496 | 12 |

Total simultaneous benchmark work: 21 measured requests and 7 excluded warmup requests.

## Long-context retrieval

These synthetic checks place exactly three needles in filler text and validate all three returned values. They report actual API prompt/cache usage. They are retrieval checks on this configuration, with no generalization to long-document reasoning.

| Check | Passed | Requested context | Actual prompt tokens | Cached prompt tokens | Exact needles |
| --- | --- | ---: | ---: | ---: | ---: |
| long-32768 | True | 32768 | 32943 | 0 | 3/3 |
| long-240000 | True | 240000 | 240176 | 0 | 3/3 |

## Lifecycle and resource evidence

The batch ran one owned server at a time on loopback port 8899. Every completed result retains the ownership token, actual PID, load time, client command/exit, final source checks, and server cleanup. The controller refuses an occupied port, and cleanup targets only the verified process group it started. The server was checked alive before cleanup.

| Job | Launch-to-ready seconds | Minimum sampled MemAvailable GiB | Initial → maximum allocated swap GiB | Server cleanup |
| --- | ---: | ---: | ---: | --- |
| final-305-single | 59.08 | 27.249 | 0.124 → 0.135 | SIGTERM; exit 0; owned group empty |
| final-405-single | 53.09 | 42.882 | 0.135 → 0.135 | SIGTERM; exit 0; owned group empty |
| final-415-single | 47.08 | 41.781 | 0.135 → 0.135 | SIGTERM; exit 0; owned group empty |
| final-cyber387-single | 46.10 | 45.522 | 0.135 → 0.135 | SIGTERM; exit 0; owned group empty |
| final-305-concurrent | 50.07 | 38.450 | 0.135 → 0.135 | SIGTERM; exit 0; owned group empty |

Resource samples are spaced about ten seconds apart and include loading and client phases. They are sparse observations, not guaranteed peaks. Allocated swap can be inherited from earlier work; these readings do not measure swap I/O. GB10 GPU-used/free memory fields are reported as unavailable, while system memory is retained.

## Evidence layout and reproduction

- `reports/`: complete original job directories, per-client reports/logs, configuration, deployment, per-job resource samples, launch commands, and final cleanup results.
- `inputs/`: exact selected jobs, final setup status, quality-to-API handoff and its source review, frozen controller/parent/helper/client source, and upstream-head evidence.
- `source/`: exact recipe/client files referenced by their captured hashes and small model configuration files. These source snapshots are evidence, with their recorded original paths.
- `snapshots/`: immutable completed-only collection records while the batch was running.
- `archive-status.json`: every copied remote file's source path, size, mtime and checksum; the final collection flag means the batch has finished and its PID is gone.
- `summary.json`: recomputed counts, per-request comparison hashes, metrics and preserved failure details.

The source snapshot is host-specific. The frozen live controllers expect the retained Spark runtime, setup status and paths; their strict hashes are intentional. Use the public recipe clients for a different installation, with explicit matching model/runtime metadata and fresh output paths. The [controller archive](../final-validation-controllers/README.md) documents the bounded live lifecycle and its CPU review scope.

Recompute only the local summary, without loading a model or making API requests:

```bash
python3 scripts/summarize.py --output /tmp/final-api-summary.json
```

Once collection is final, verify the archived bytes with:

```bash
sha256sum -c SHA256SUMS
```

The collector source is retained as `scripts/collect_completed.py`; it reads bounded, already-completed metadata through the established SSH wrapper and never executes any archived live controller. Importing host-control scripts is unnecessary. Setup evidence is in [final-runtime-setup](../final-runtime-setup/README.md), and the separate matched numerical comparison is in [final-numerical-chunks](../final-numerical-chunks/README.md). Service installation and rollback evidence are separate deployment work.
