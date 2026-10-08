# Completed optional performance jobs

Completed: **16/16** at 2026-10-08T12:58:06.800299+00:00.

Historical engine16ca/Tabbyf4 measurements. A tool failure is retained and prevents treating that whole job as qualified.

| Job | Benchmark/case | Prompt/output/cache tokens, median | Decode tok/s | Prefill tok/s | TTFT s | Draft acceptance | Accepted/rejected draft tokens, median |
|---|---|---:|---:|---:|---:|---:|---:|
| 13-tune-305-static-draft5-device | bench-1/code | 87/400/0 | 79.10 | 140.32 | 0.784 | 63.33% | 304/176 |
| 13-tune-305-static-draft5-device | bench-1/devops | 63/400/0 | 70.39 | 123.53 | 0.670 | 56.19% | 295/230 |
| 13-tune-305-static-draft5-device | bench-1/prose | 72/400/0 | 45.32 | 118.03 | 0.772 | 28.19% | 234/596 |
| 12-control-305-requested-head65536 | bench-1/code | 87/400/0 | 80.15 | 140.32 | 0.744 | 75.91% | 292/93 |
| 12-control-305-requested-head65536 | bench-1/devops | 63/400/0 | 73.88 | 123.53 | 0.634 | 72.99% | 281/104 |
| 12-control-305-requested-head65536 | bench-1/prose | 72/400/0 | 53.78 | 118.03 | 0.724 | 56.15% | 207/162 |
| 12-tune-305-requested-head32768 | bench-1/code | 87/400/0 | 81.85 | 140.32 | 0.750 | 76.04% | 293/92 |
| 12-tune-305-requested-head32768 | bench-1/devops | 63/400/0 | 71.70 | 123.53 | 0.620 | 70.54% | 272/114 |
| 12-tune-305-requested-head32768 | bench-1/prose | 72/400/0 | 53.28 | 118.03 | 0.721 | 56.00% | 201/154 |
| 12-tune-305-requested-head131072 | bench-1/code | 87/400/0 | 82.96 | 140.32 | 0.767 | 79.32% | 303/79 |
| 12-tune-305-requested-head131072 | bench-1/devops | 63/400/0 | 73.83 | 123.53 | 0.639 | 76.55% | 284/87 |
| 12-tune-305-requested-head131072 | bench-1/prose | 72/400/0 | 53.32 | 118.03 | 0.723 | 55.21% | 212/167 |
| 06-control-405-disk | bench-1/code | 87/400/0 | 76.15 | 138.10 | 0.790 | 78.39% | 301/83 |
| 06-control-405-disk | bench-1/devops | 63/400/0 | 66.63 | 121.15 | 0.664 | 73.49% | 281/101 |
| 06-control-405-disk | bench-1/prose | 72/400/0 | 47.61 | 118.03 | 0.749 | 57.10% | 212/157 |
| 06-control-405-disk | bench-2/code | 12157/256/0 | 52.55 | 673.72 | 18.212 | 75.60% | 176/57 |
| 06-tune-405-ngram-ram | bench-1/code | 87/400/0 | 77.41 | 138.10 | 0.789 | 78.39% | 301/83 |
| 06-tune-405-ngram-ram | bench-1/devops | 63/400/0 | 67.49 | 123.53 | 0.659 | 73.49% | 281/101 |
| 06-tune-405-ngram-ram | bench-1/prose | 72/400/0 | 50.05 | 118.03 | 0.736 | 57.10% | 212/157 |
| 06-tune-405-ngram-ram | bench-2/code | 12157/256/0 | 50.17 | 599.72 | 20.519 | 75.60% | 176/57 |
| 07-control-415-disk | bench-1/code | 87/400/0 | 52.29 | 248.57 | 0.546 | 75.69% | 302/97 |
| 07-control-415-disk | bench-1/devops | 63/400/0 | 49.97 | 217.24 | 0.517 | 77.11% | 293/88 |
| 07-control-415-disk | bench-1/prose | 72/400/0 | 33.83 | 211.76 | 0.533 | 57.69% | 210/154 |
| 07-control-415-disk | bench-2/code | 12157/256/0 | 43.14 | 934.15 | 13.238 | 80.42% | 193/47 |
| 07-tune-415-ngram-ram | bench-1/code | 87/400/0 | 52.90 | 255.88 | 0.547 | 75.69% | 302/97 |
| 07-tune-415-ngram-ram | bench-1/devops | 63/400/0 | 50.27 | 217.24 | 0.518 | 77.11% | 293/88 |
| 07-tune-415-ngram-ram | bench-1/prose | 72/400/0 | 35.11 | 211.76 | 0.526 | 57.69% | 210/154 |
| 07-tune-415-ngram-ram | bench-2/code | 12157/256/0 | 42.30 | 915.49 | 13.505 | 80.42% | 193/47 |
| 08-control-cyber387-disk | bench-1/code | 196/400/0 | 44.03 | 337.93 | 0.790 | 73.28% | 274/101 |
| 08-control-cyber387-disk | bench-1/devops | 172/400/0 | 37.14 | 318.52 | 0.729 | 66.41% | 243/119 |
| 08-control-cyber387-disk | bench-1/prose | 181/400/0 | 34.66 | 323.21 | 0.765 | 62.33% | 230/134 |
| 08-control-cyber387-disk | bench-2/code | 12266/256/0 | 35.26 | 877.14 | 14.214 | 68.05% | 172/81 |
| 08-tune-cyber387-ngram-ram | bench-1/code | 196/400/0 | 44.92 | 337.93 | 0.793 | 73.28% | 274/101 |
| 08-tune-cyber387-ngram-ram | bench-1/devops | 172/400/0 | 38.13 | 318.52 | 0.733 | 66.41% | 243/119 |
| 08-tune-cyber387-ngram-ram | bench-1/prose | 181/400/0 | 35.79 | 317.54 | 0.764 | 62.33% | 230/134 |
| 08-tune-cyber387-ngram-ram | bench-2/code | 12266/256/0 | 35.47 | 851.61 | 14.621 | 68.05% | 172/81 |

## Whole-job gates

| Job | All clients pass | Tools pass/total |
|---|---|---:|
| 13-tune-305-static-draft5-device | True | 8/8 |
| 12-control-305-requested-head65536 | True | — |
| 12-tune-305-requested-head32768 | True | — |
| 12-tune-305-requested-head131072 | True | — |
| 09-control-305-concurrent-disk | True | 8/8 |
| 09-tune-305-concurrent-ngram-ram | True | 8/8 |
| 10-control-415-concurrent-disk | True | 8/8 |
| 10-tune-415-concurrent-rowbudget8 | True | 8/8 |
| 11-control-cyber387-concurrent-disk | False | 6/8 |
| 11-tune-cyber387-concurrent-rowbudget8 | False | 5/8 |
| 06-control-405-disk | False | 6/8 |
| 06-tune-405-ngram-ram | False | 6/8 |
| 07-control-415-disk | True | 8/8 |
| 07-tune-415-ngram-ram | True | 8/8 |
| 08-control-cyber387-disk | False | 6/8 |
| 08-tune-cyber387-ngram-ram | False | 6/8 |

## Paired settings

| Control → variant | Only changed tuning | Case | Decode change | Prefill change | Both jobs pass |
|---|---|---|---:|---:|---|
| 12-control-305-requested-head65536 → 12-tune-305-requested-head32768 | EXL3_MTP_HEAD_N: 65536 → 32768 | bench-1/code | 2.12% | 0.00% | True |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head32768 | EXL3_MTP_HEAD_N: 65536 → 32768 | bench-1/devops | -2.95% | 0.00% | True |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head32768 | EXL3_MTP_HEAD_N: 65536 → 32768 | bench-1/prose | -0.93% | 0.00% | True |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head131072 | EXL3_MTP_HEAD_N: 65536 → 131072 | bench-1/code | 3.51% | 0.00% | True |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head131072 | EXL3_MTP_HEAD_N: 65536 → 131072 | bench-1/devops | -0.07% | 0.00% | True |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head131072 | EXL3_MTP_HEAD_N: 65536 → 131072 | bench-1/prose | -0.86% | 0.00% | True |
| 06-control-405-disk → 06-tune-405-ngram-ram | NGRAM_RAM: false → true | bench-1/code | 1.65% | 0.00% | False |
| 06-control-405-disk → 06-tune-405-ngram-ram | NGRAM_RAM: false → true | bench-1/devops | 1.29% | 1.96% | False |
| 06-control-405-disk → 06-tune-405-ngram-ram | NGRAM_RAM: false → true | bench-1/prose | 5.12% | 0.00% | False |
| 06-control-405-disk → 06-tune-405-ngram-ram | NGRAM_RAM: false → true | bench-2/code | -4.53% | -10.98% | False |
| 07-control-415-disk → 07-tune-415-ngram-ram | NGRAM_RAM: false → true | bench-1/code | 1.17% | 2.94% | True |
| 07-control-415-disk → 07-tune-415-ngram-ram | NGRAM_RAM: false → true | bench-1/devops | 0.60% | 0.00% | True |
| 07-control-415-disk → 07-tune-415-ngram-ram | NGRAM_RAM: false → true | bench-1/prose | 3.78% | 0.00% | True |
| 07-control-415-disk → 07-tune-415-ngram-ram | NGRAM_RAM: false → true | bench-2/code | -1.94% | -2.00% | True |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | NGRAM_RAM: false → true | bench-1/code | 2.02% | 0.00% | False |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | NGRAM_RAM: false → true | bench-1/devops | 2.67% | 0.00% | False |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | NGRAM_RAM: false → true | bench-1/prose | 3.26% | -1.75% | False |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | NGRAM_RAM: false → true | bench-2/code | 0.60% | -2.91% | False |

## Completed concurrency batches

| Job | Concurrent requests | Complete measured rounds | Prompt/output/cache per request, median | Aggregate end-to-end tok/s | TTFT per request s | Draft acceptance | Accepted/rejected draft tokens per request, median |
|---|---:|---:|---:|---:|---:|---:|---:|
| 09-control-305-concurrent-disk | 1 | 3/3 | 849/256/0 | 39.76 | 3.289 | 78.66% | 188/51 |
| 09-control-305-concurrent-disk | 2 | 3/3 | 849/256/0 | 31.04 | 6.697 | 72.14% | 192/74 |
| 09-control-305-concurrent-disk | 4 | 3/3 | 849/256/0 | 35.31 | 13.214 | 67.54% | 194/93 |
| 09-tune-305-concurrent-ngram-ram | 1 | 3/3 | 849/256/0 | 41.07 | 3.213 | 78.66% | 188/51 |
| 09-tune-305-concurrent-ngram-ram | 2 | 3/3 | 849/256/0 | 29.53 | 6.863 | 70.33% | 190/81 |
| 09-tune-305-concurrent-ngram-ram | 4 | 3/3 | 849/256/0 | 34.52 | 13.526 | 66.67% | 193/96 |
| 10-control-415-concurrent-disk | 1 | 3/3 | 849/256/0 | 36.28 | 2.492 | 81.93% | 193/43 |
| 10-control-415-concurrent-disk | 2 | 3/3 | 849/256/0 | 48.79 | 4.797 | 80.58% | 195/47 |
| 10-control-415-concurrent-disk | 4 | 3/3 | 849/256/0 | 50.50 | 9.861 | 72.36% | 200/76 |
| 10-tune-415-concurrent-rowbudget8 | 1 | 3/3 | 849/256/0 | 36.99 | 2.330 | 81.93% | 193/43 |
| 10-tune-415-concurrent-rowbudget8 | 2 | 3/3 | 849/256/0 | 43.86 | 4.826 | 84.49% | 182/34 |
| 10-tune-415-concurrent-rowbudget8 | 4 | 3/3 | 849/256/0 | 50.91 | 8.683 | 95.42% | 125/6 |
| 11-control-cyber387-concurrent-disk | 1 | 3/3 | 958/256/0 | 29.74 | 2.748 | 72.06% | 172/69 |
| 11-control-cyber387-concurrent-disk | 2 | 3/3 | 958/256/0 | 31.40 | 5.530 | 61.54% | 174/108 |
| 11-control-cyber387-concurrent-disk | 4 | 3/3 | 958/256/0 | 36.45 | 11.124 | 56.85% | 180/136 |
| 11-tune-cyber387-concurrent-rowbudget8 | 1 | 3/3 | 958/256/0 | 29.21 | 2.892 | 72.06% | 172/69 |
| 11-tune-cyber387-concurrent-rowbudget8 | 2 | 3/3 | 958/256/0 | 33.83 | 5.707 | 64.57% | 158/87 |
| 11-tune-cyber387-concurrent-rowbudget8 | 4 | 3/3 | 958/256/0 | 41.37 | 10.689 | 84.29% | 118/22 |

| Control → variant | Concurrent requests | Control aggregate tok/s | Variant aggregate tok/s | Change | Both jobs pass |
|---|---:|---:|---:|---:|---|
| 09-control-305-concurrent-disk → 09-tune-305-concurrent-ngram-ram | 1 | 39.76 | 41.07 | 3.28% | True |
| 09-control-305-concurrent-disk → 09-tune-305-concurrent-ngram-ram | 2 | 31.04 | 29.53 | -4.87% | True |
| 09-control-305-concurrent-disk → 09-tune-305-concurrent-ngram-ram | 4 | 35.31 | 34.52 | -2.23% | True |
| 10-control-415-concurrent-disk → 10-tune-415-concurrent-rowbudget8 | 1 | 36.28 | 36.99 | 1.95% | True |
| 10-control-415-concurrent-disk → 10-tune-415-concurrent-rowbudget8 | 2 | 48.79 | 43.86 | -10.11% | True |
| 10-control-415-concurrent-disk → 10-tune-415-concurrent-rowbudget8 | 4 | 50.50 | 50.91 | 0.80% | True |
| 11-control-cyber387-concurrent-disk → 11-tune-cyber387-concurrent-rowbudget8 | 1 | 29.74 | 29.21 | -1.81% | False |
| 11-control-cyber387-concurrent-disk → 11-tune-cyber387-concurrent-rowbudget8 | 2 | 31.40 | 33.83 | 7.76% | False |
| 11-control-cyber387-concurrent-disk → 11-tune-cyber387-concurrent-rowbudget8 | 4 | 36.45 | 41.37 | 13.49% | False |

## Matched request and timing scope

| Control → variant | Case or concurrency | Requests per setting | Exact request hashes and counts | Same source, one setting and benchmark contract | Matching response hashes |
|---|---|---:|---|---|---:|
| 12-control-305-requested-head65536 → 12-tune-305-requested-head32768 | bench-1/code | 3/3 | True | True | 3 |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head32768 | bench-1/devops | 3/3 | True | True | 2 |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head32768 | bench-1/prose | 3/3 | True | True | 0 |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head131072 | bench-1/code | 3/3 | True | True | 3 |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head131072 | bench-1/devops | 3/3 | True | True | 3 |
| 12-control-305-requested-head65536 → 12-tune-305-requested-head131072 | bench-1/prose | 3/3 | True | True | 0 |
| 09-control-305-concurrent-disk → 09-tune-305-concurrent-ngram-ram | 1 concurrent | 3/3 | True | True | 3 |
| 09-control-305-concurrent-disk → 09-tune-305-concurrent-ngram-ram | 2 concurrent | 6/6 | True | True | 1 |
| 09-control-305-concurrent-disk → 09-tune-305-concurrent-ngram-ram | 4 concurrent | 12/12 | True | True | 5 |
| 10-control-415-concurrent-disk → 10-tune-415-concurrent-rowbudget8 | 1 concurrent | 3/3 | True | True | 3 |
| 10-control-415-concurrent-disk → 10-tune-415-concurrent-rowbudget8 | 2 concurrent | 6/6 | True | True | 1 |
| 10-control-415-concurrent-disk → 10-tune-415-concurrent-rowbudget8 | 4 concurrent | 12/12 | True | True | 3 |
| 11-control-cyber387-concurrent-disk → 11-tune-cyber387-concurrent-rowbudget8 | 1 concurrent | 3/3 | True | True | 3 |
| 11-control-cyber387-concurrent-disk → 11-tune-cyber387-concurrent-rowbudget8 | 2 concurrent | 6/6 | True | True | 0 |
| 11-control-cyber387-concurrent-disk → 11-tune-cyber387-concurrent-rowbudget8 | 4 concurrent | 12/12 | True | True | 0 |
| 06-control-405-disk → 06-tune-405-ngram-ram | bench-1/code | 3/3 | True | True | 3 |
| 06-control-405-disk → 06-tune-405-ngram-ram | bench-1/devops | 3/3 | True | True | 3 |
| 06-control-405-disk → 06-tune-405-ngram-ram | bench-1/prose | 3/3 | True | True | 3 |
| 06-control-405-disk → 06-tune-405-ngram-ram | bench-2/code | 2/2 | True | True | 2 |
| 07-control-415-disk → 07-tune-415-ngram-ram | bench-1/code | 3/3 | True | True | 3 |
| 07-control-415-disk → 07-tune-415-ngram-ram | bench-1/devops | 3/3 | True | True | 3 |
| 07-control-415-disk → 07-tune-415-ngram-ram | bench-1/prose | 3/3 | True | True | 3 |
| 07-control-415-disk → 07-tune-415-ngram-ram | bench-2/code | 2/2 | True | True | 2 |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | bench-1/code | 3/3 | True | True | 3 |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | bench-1/devops | 3/3 | True | True | 3 |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | bench-1/prose | 3/3 | True | True | 3 |
| 08-control-cyber387-disk → 08-tune-cyber387-ngram-ram | bench-2/code | 2/2 | True | True | 2 |

| Job | Concurrent requests | Measured batch rates, tok/s |
|---|---:|---|
| 09-control-305-concurrent-disk | 1 | 40.8766, 39.7634, 39.3978 |
| 09-control-305-concurrent-disk | 2 | 34.3343, 29.6201, 31.0411 |
| 09-control-305-concurrent-disk | 4 | 37.6534, 35.3070, 33.8269 |
| 09-tune-305-concurrent-ngram-ram | 1 | 42.5428, 40.6175, 41.0686 |
| 09-tune-305-concurrent-ngram-ram | 2 | 28.9067, 36.2449, 29.5293 |
| 09-tune-305-concurrent-ngram-ram | 4 | 37.4458, 34.5213, 32.5270 |
| 10-control-415-concurrent-disk | 1 | 36.1928, 36.9078, 36.2821 |
| 10-control-415-concurrent-disk | 2 | 51.3754, 48.7861, 48.1103 |
| 10-control-415-concurrent-disk | 4 | 50.5047, 51.7021, 50.3823 |
| 10-tune-415-concurrent-rowbudget8 | 1 | 36.8556, 38.3020, 36.9891 |
| 10-tune-415-concurrent-rowbudget8 | 2 | 52.4472, 43.8557, 43.7015 |
| 10-tune-415-concurrent-rowbudget8 | 4 | 50.5584, 51.8150, 50.9098 |
| 11-control-cyber387-concurrent-disk | 1 | 30.3463, 29.7428, 28.9610 |
| 11-control-cyber387-concurrent-disk | 2 | 31.3979, 32.8746, 30.7990 |
| 11-control-cyber387-concurrent-disk | 4 | 36.3356, 36.4482, 36.5468 |
| 11-tune-cyber387-concurrent-rowbudget8 | 1 | 30.1139, 28.9849, 29.2056 |
| 11-tune-cyber387-concurrent-rowbudget8 | 2 | 35.1678, 33.8328, 32.9330 |
| 11-tune-cyber387-concurrent-rowbudget8 | 4 | 41.2825, 41.4260, 41.3654 |

## Scope

- Only state=completed jobs with a final timestamp and verified cleanup enter the summary. A completed job may still fail a tool or concurrency gate.
- Metrics are medians of measured requests, excluding warmup. Server throughput is reported by Tabby; client wall/TTFT are separately retained.
- Concurrency throughput uses only complete measured batches: total output tokens divided by earliest request start to last request finish. Warmup and partial batches are excluded; per-request rates are not summed.
- Each cold repeat has its own deterministic nonce. Different repeat response hashes do not imply nondeterminism, because request hashes differ.
- Actual prompt, output, and cached counts are retained; nominal context size is not substituted for actual prompt tokens.
- This historical optional matrix runs engine16ca/Tabbyf4. Later engine/server revisions and any combined selected profile require separate live qualification.
- Single-knob comparisons use the nearest preceding control for the same matrix block/model, require exactly one changed resolved tuning variable, and compare identical benchmark settings.
- The optional wrapper preserves the frozen primary metric and inclusion code. It adds exact measured request-hash/count comparisons without altering metric values or promoting failed jobs.
- A matched pair records identical source commits, one changed resolved setting, benchmark settings/run ID, and per-repeat or per-round/stream request hashes. Different response hashes and draft acceptance are retained as observations.
- These are sequential small-sample measurements from one machine. Medians and observed round ranges describe these runs; no repeated-matrix confidence interval or universal timing benefit is established.
- The static-drafting job has no preceding control in its own matrix block. It is displayed without an invented matched comparison.
- A RAM-placement comparison includes runtime scheduling and generated-output differences observed under that setting. Matching wire requests alone does not isolate every timing difference to memory-transfer speed.
