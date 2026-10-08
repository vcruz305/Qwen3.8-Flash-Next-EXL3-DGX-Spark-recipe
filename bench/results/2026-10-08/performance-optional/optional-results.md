### Optional placement, draft-head and concurrent checks

The separate optional matrix completed all **16 jobs** between
**2026-10-08T11:53:48.871186+00:00** and **2026-10-08T12:56:46.101871+00:00**.
It measured **102 single requests**, **126 concurrent requests**, and
**104 tool checks: 91 passed and 13 failed**. Ten jobs passed all their scheduled
clients; six retained semantic tool failures. The three MTP-head jobs scheduled
benchmarks only and provide no tool qualification.

These are historical measurements of engine
`16ca20d27c0e4cce15a9bbc131e6d047065395b5`, Tabby
`f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6`, and recipe
`218bd438225243e795383f10ef7a886b06d5d839`. They precede the final producer
reasoning fixes and Cyber template override. The later combined deployment
requires its own qualification. The [optional archive](README.md)
retains every completed report, failed tool result, source identity, resource
sample and shutdown outcome.

#### Requested MTP head size and static drafting

The flat 3.05 RAM runs retained dynamic depth 5 and changed only the requested
`EXL3_MTP_HEAD_N`. Each short case had one warmup and three measured requests
with 400 actual output tokens and zero reused prompt tokens.

| Requested MTP head N | Code tok/s | DevOps tok/s | Prose tok/s | Tool checks |
|---|---:|---:|---:|---|
| 65,536 (control) | 80.15 | 73.88 | 53.78 | Not scheduled |
| 32,768 | 81.85 | 71.70 | 53.28 | Not scheduled |
| 131,072 | 82.96 | 73.83 | 53.32 | Not scheduled |

The startup log confirms the main-model MTP component was used, but does not
report effective loaded head width; N is therefore labeled **requested**.
The recorded source and configuration define eligibility. These small timing
differences were not accompanied by a separate repeated-matrix confidence
estimate or tool checks.

The separately scheduled static-depth-5/device-drafting job recorded
**79.10, 70.39, 45.32 tok/s** for code, DevOps and prose, respectively, and
**8/8** tool checks. The optional summarizer does not assign
that job a matched preceding control; it is an observation, not a claimed
isolated speedup.

#### Concurrent placement and row-budget comparisons

These medians are **whole-batch completed output tokens divided by batch wall
time**, including prefill. Each setting has three measured rounds after a
warmup at 1, 2 and 4 requests. Every request returned 256 actual output tokens,
with zero reused prompt tokens; actual prompts were 849 tokens for 3.05/SAGE
and 958 for Cyber. Each concurrent profile reserved a 1,048,576-token shared
pool, a 262,144-token per-request cap and a maximum batch size of 4.

| Comparison | Concurrent requests | Control tok/s | Variant tok/s | Change | Tool checks, control / variant |
|---|---:|---:|---:|---:|---|
| 3.05 disk → RAM, row budget 0 | 1 | 39.7634 | 41.0686 | +3.28% | 8/8 / 8/8 |
| 3.05 disk → RAM, row budget 0 | 2 | 31.0411 | 29.5293 | -4.87% | 8/8 / 8/8 |
| 3.05 disk → RAM, row budget 0 | 4 | 35.3070 | 34.5213 | -2.23% | 8/8 / 8/8 |
| SAGE disk, row budget 0 → 8 | 1 | 36.2821 | 36.9891 | +1.95% | 8/8 / 8/8 |
| SAGE disk, row budget 0 → 8 | 2 | 48.7861 | 43.8557 | -10.11% | 8/8 / 8/8 |
| SAGE disk, row budget 0 → 8 | 4 | 50.5047 | 50.9098 | +0.80% | 8/8 / 8/8 |
| Cyber disk, row budget 0 → 8 | 1 | 29.7428 | 29.2056 | -1.81% | 6/8 / 5/8 |
| Cyber disk, row budget 0 → 8 | 2 | 31.3979 | 33.8328 | +7.76% | 6/8 / 5/8 |
| Cyber disk, row budget 0 → 8 | 4 | 36.4482 | 41.3654 | +13.49% | 6/8 / 5/8 |

All compared per-round/stream request hashes and actual token counts match.
The archive retains the three individual rates and response-hash comparisons.
The 3.05 RAM medians and SAGE row-budget medians show mixed changes with
overlapping round ranges. Cyber's positive two/four-request rates remain
historical observations under failed tool gates: both jobs failed the two
string fixtures, and row budget 8 also returned string `"True"` where the
streamed typed fixture required a boolean. These rows do not establish a
general mixed-pack row-budget benefit.

#### Larger-pack RAM placement

For each pair, all **11 measured requests** have matching wire-request hashes,
actual prompt/output/cache counts, and response hashes. The short cases have
three measured requests each and 400 output tokens. The longer prompt has two
measured requests and 256 output tokens. Each case has one separate warmup.
The original Cyber template remains in place for these runs.

| Pack | Code disk → RAM tok/s (change) | DevOps disk → RAM tok/s (change) | Prose disk → RAM tok/s (change) | Tool checks, disk / RAM |
|---|---:|---:|---:|---|
| Flat 4.05 | 76.15 → 77.41 (+1.65%) | 66.63 → 67.49 (+1.29%) | 47.61 → 50.05 (+5.12%) | 6/8 / 6/8 |
| SAGE 4.15 | 52.29 → 52.90 (+1.17%) | 49.97 → 50.27 (+0.60%) | 33.83 → 35.11 (+3.78%) | 8/8 / 8/8 |
| Cyberfrost 3.87 | 44.03 → 44.92 (+2.02%) | 37.14 → 38.13 (+2.67%) | 34.66 → 35.79 (+3.26%) | 6/8 / 6/8 |

| Pack | Actual prompt tokens | Prefill disk → RAM tok/s (change) | Decode disk → RAM tok/s (change) | Load disk → RAM, seconds |
|---|---:|---:|---:|---:|
| Flat 4.05 | 12,157 | 673.725 → 599.720 (-10.98%) | 52.550 → 50.170 (-4.53%) | 50.23 → 72.31 |
| SAGE 4.15 | 12,157 | 934.150 → 915.485 (-2.00%) | 43.140 → 42.305 (-1.94%) | 47.20 → 63.26 |
| Cyberfrost 3.87 | 12,266 | 877.145 → 851.610 (-2.91%) | 35.255 → 35.465 (+0.60%) | 46.18 → 63.24 |

RAM placement gave modest short-decode gains in these runs, while longer-prompt
prefill was slower for all three packs and startup took longer. Only SAGE
passed both eight-check tool subsets; neither placement fixed the 4.05 or
Cyber string fixture. These results do not establish a universal RAM speedup.

| Pack | Minimum sampled MemAvailable, disk / RAM GiB | RAM swap used at start / maximum, GiB | Largest observed increase from RAM start, GiB |
|---|---:|---:|---:|
| Flat 4.05 | 42.172 / 7.382 | 0.0001 / 2.2358 | 2.2357 |
| SAGE 4.15 | 41.012 / 4.683 | 0.0745 / 0.0996 | 0.0250 |
| Cyberfrost 3.87 | 44.806 / 8.419 | 0.0996 / 0.0996 | 0.0000 |

Memory figures are sparse ten-second `/proc/meminfo` observations, not true
allocation peaks or swap-I/O measurements. Flat 4.05 RAM accumulated about
2.236 GiB of allocated swap. SAGE and Cyber started with already allocated swap;
their full maximum must not be attributed to the current job. The helper's
10 GiB allowance covers runtime overhead; it is not an additional requirement
to leave 10 GiB of MemAvailable after loading.

The source placement audit explains the large RAM-table increment without a
second full-table CPU copy: flat 4.05 reads directly into the final allocation;
SAGE/Cyber have a roughly 0.284 GiB one-shard transient. The two pinned lookup
sets already exist in disk mode. The archive includes the corrected unloaded
advisories and the separate source audit; neither replaces observed host
resource behavior.
