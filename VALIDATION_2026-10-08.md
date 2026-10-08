# DGX Spark validation — 2026-10-08

The upstream integrations, on-device qualification and persistent-service
rehearsal are complete. ExLlamaV3 PR 21 and TabbyAPI PR 2 are merged, with full
merged trees equal to their qualified heads. The selected 3.05 concurrent/disk
service passes startup, automatic restart after a controlled failure, an actual
original-runtime rollback, and return to the new runtime.

The update improves mixed-K decode and measured 3.05 concurrent throughput;
flat-pack single-request measurements show small regressions. All four packs
pass the final broad tool, SDK, resilience and scheduled automatic suites.
The additional exact native-marker copying task retains 16 failures in the
322-check main matrix. Raw generated values explain those captured failures;
a separate thinking-disabled variant passes 4/4. This report preserves those
limits, failed experiments and diagnostic metadata omissions.

| Identity | Qualified / deployed source |
|---|---|
| Installed ExLlamaV3 pin | `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` |
| ExLlamaV3 PR 21 merge | `678ab0dd1c39cb7ba22d577913132f63d4d242f3`, full tree identical to the pin |
| TabbyAPI qualification head | `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb` |
| Installed TabbyAPI main merge | `f650bb5389e0a273549e47d4d26a765760c013e1`, full tree identical to the qualification head |
| Hardware-run and full-rehearsal recipe | `3337bc8d64e4befafa2a1aff342e07abbea242ac` (R1) |

Official upstream heads were rechecked at **13:42:41 UTC on 2026-10-08**:
ExLlamaV3 `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` and TabbyAPI
`2fd6cc76203a66e13042daf7d76e5898b21c1ad8`, both already integrated.
The recipe's final publication adds documentation and evidence to R1. Its
exact published commit, unchanged runtime/client-byte proof, subsequent
service invocation and live verification are recorded in
[recipe PR 17](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/pull/17).
That record binds the published commit without requiring this file to contain
its own commit hash.

This report separates measured behavior, implementation checks and remaining
limitations. The machine was tested directly; no September engine-only rates
are substituted for the October API results.

## Test machine and starting point

One NVIDIA DGX Spark (GB10, aarch64), 20 CPU cores and approximately 121 GiB of
OS-visible memory. The original machine used NVIDIA driver 580.178.04,
CUDA toolkit 13.0.88, PyTorch 2.13.0+cu130 and Triton 3.7.1. Both sides of the
initial comparison use that same hardware and PyTorch/CUDA stack.

The original engine was
[`94ba01d50a13fa9ff672473f2d0eef8b51a71e99`](https://github.com/vcruz305/exllamav3/commit/94ba01d50a13fa9ff672473f2d0eef8b51a71e99)
(1.5.1), with TabbyAPI
[`816c32195887aaecea1c64528f2921566766259b`](https://github.com/theroyallab/tabbyAPI/commit/816c32195887aaecea1c64528f2921566766259b).
The updated engine incorporates upstream 1.6.0 at
[`151539c77abc7ab7425d30da7a4e8e3c5c154e7b`](https://github.com/turboderp-org/exllamav3/commit/151539c77abc7ab7425d30da7a4e8e3c5c154e7b)
and the Spark fork's additional fixes. The updated Tabby fork incorporates
upstream main and the tool-calling changes.

The upgrade was prepared in a separate runtime and virtual environment.
Existing model weights, the original installation and an unrelated dirty beta
checkout were retained. No model weights were rewritten or requantized.

## Model views

These are the four model directories present on the Spark at the start:

| Report label | Existing directory | Safetensor bytes, GiB | N-gram tensors, GiB | Initial A/B n-gram placement |
|---|---|---:|---:|---|
| Flat 3.05 | `flashnext-exl3-3.05bpw` | 79.150 | 30.40 | RAM |
| Flat 4.05 | `flashnext-exl3-4.05bpw` | 99.941 | 36.36 | Disk |
| SAGE 4.15 | `flashnext-exl3-sage-4.15bpw` | 101.422 | 36.36 | Disk |
| Cyberfrost mixed K | `CYBER-FROST-3.8-EXL3-SAGE-3.87bpw` | 97.585 | 36.36 | Disk |

The Cyberfrost directory says 3.87 bpw. A separate header audit finds an
unweighted average K of 3.77327 over its target expert projections. That
projection-level statistic is not an overall pack bits-per-weight
calculation. The report keeps the actual folder name and does not relabel
or alter the pack.

Both mixed packs include an extra
`mtp_hyper_connection_mixer_patch.safetensors` containing three duplicate
MTP mixer keys. In the original loader order these override keys in the main
shards. Baseline and candidate use the same complete original directories
and the same loader order, including that patch. They do not silently switch
to a view that excludes it.

The model architecture is `Qwen4ExpForConditionalGeneration` / `qwen4_exp`:
hidden size 2,560, expert intermediate size 640, 48 layers, 512 experts and
top-10 routing. Twelve target layers use full attention. Kernel checks include
these actual dimensions as well as synthetic stress shapes.

## API performance method

The single-stream suite uses the recipe's strict `/v1/chat/completions`
client with:

- Code, DevOps and prose prompts.
- Temperature 0, top-k 1, top-p 1, seed 0 and thinking explicitly requested off.
  The original Cyberfrost template overrides that request. The original,
  primary and optional comparisons retain that template on both sides; the
  separately labeled final confirmation uses the corrected template.
- 400 requested output tokens, with the actual emitted token count recorded.
- One warmup and three measured requests for each prompt class.
- Fixed run ID `overnight-v1`, with a distinct early prefix for each request.
- Context and shared KV capacity 262,144; one active job; 8-bit K and V.
- Chunk size 2,048; MTP depth 5, dynamic drafting and confidence 0.6.
- Big-core affinity 5–9,15–19; the same shared base controls and model placement
  on each side. Candidate GDN, attention and MoE compatibility settings are
  listed explicitly with the numerical and performance results.

The primary decode number is the median server-reported decode tokens/second.
The client also records wall time, time to first output, actual prompt tokens,
cache reuse, finish reason and model identity. All quoted throughput samples
must finish with valid JSON or a complete SSE finish and `[DONE]`, contain
actual usage and match the verified model ID.

The original Tabby revision returned the canonical directory name even when
the public alias was requested. The baseline therefore records an explicit,
independently verified response-model alias. The updated server returns the
public requested alias consistently. The initial alias-check failure is
retained as a diagnostic, not counted as a speed sample.

### Original API measurements

| Pack | Code tok/s | DevOps tok/s | Prose tok/s |
|---|---:|---:|---:|
| Flat 3.05 | 82.49 | 74.11 | 55.93 |
| Flat 4.05 | 77.02 | 67.76 | 48.57 |
| SAGE 4.15 | 48.78 | 46.48 | 30.75 |
| Cyberfrost mixed K | 41.13 | 34.63 | 32.09 |

### Final source confirmation: engine `24f0dece3` / TabbyAPI `5a4f3ef`

The independent final replay uses the ordinary installed engine/server,
recipe `3337bc8`, 2,048-token chunks, K8/V8, single-request pools and dynamic
MTP depth 5/confidence 0.6. Placement matches the original for each pack.
Every workload has one warmup followed by three measured responses with
400 actual completion tokens. The table reports median server decode tokens
per second, not client time-to-first-token or aggregate throughput.

| Pack / n-gram placement | Code, original → final | DevOps, original → final | Prose, original → final |
|---|---:|---:|---:|
| Flat 3.05 / RAM | 82.49 → 80.15 | 74.11 → 70.95 | 55.93 → 53.32 |
| Flat 4.05 / disk | 77.02 → 75.85 | 67.76 → 66.45 | 48.57 → 47.46 |
| SAGE 4.15 / disk | 48.78 → 52.39 | 46.48 → 49.73 | 30.75 → 34.36 |
| Cyber Frost 3.87 / disk, template changed | 41.13 → 45.75 | 34.63 → 44.28 | 32.09 → 34.36 |

For flat 3.05, final changes are -2.84%, -4.26% and -4.67%; for flat 4.05,
-1.52%, -1.93% and -2.29%. SAGE changes are +7.40%, +6.99% and +11.74%.
The 3.05 final sample ranges are 79.88–80.38, 66.97–73.36 and 51.72–54.53
respectively. No flat-single speed gain is claimed.

For the first three packs, all nine measured request hashes and actual prompt,
completion and cache-token counts match the original. Generated output hashes
are not universally identical: 3.05 matches 2/3 code, 2/3 DevOps and 0/3 prose
samples; 4.05 matches code/DevOps but differs on prose; SAGE outputs differ.
Thus the observed changes are workload-level API measurements, not a perfect
isolation of source overhead on an identical generated token trajectory.

Cyber uses the reviewed template override so the request's existing
`enable_thinking:false` setting takes effect. All nine client request hashes
still match, but actual prompt counts change from 196/172/181 to 160/136/145
for code/DevOps/prose, and output trajectories change. Its final rate row
must not be attributed entirely to kernel speed. The historical matrix below
retains the matched-original-template Cyber comparison, including the separate
NOSYNC control. The override's exact loaded bytes are verified in the final
runtime metadata.

The selected 3.05 **concurrent/disk** profile separately confirms short-output
server decode medians of **79.68 / 70.51 / 51.26 tokens/s** for
code/DevOps/prose. Its nine request hashes and actual token/cache counts match
the final single/RAM run, giving differences of -0.59%, -0.62% and -3.86%.
Code/prose generated-output hashes differ. This is a small observed profile
tradeoff, not a claim that placement alone caused every difference.

Its separate simultaneous-request corpus has 849 actual prompt tokens, zero
cached prompt tokens and 256 actual completion tokens per request. Each
concurrency level has one warmup round and three measured rounds, totaling
21 measured requests after seven warmup requests.

| Active requests | Final aggregate end-to-end tokens/s | Historical `16ca20d` / `f4` row-8 control | Final per-request server decode tokens/s |
|---|---:|---:|---:|
| 1 | 38.5724 | 40.4456 | 74.85 |
| 2 | 46.4472 | 47.2729 | 57.29 |
| 4 | 48.5496 | 50.4184 | 31.865 |

The final aggregate rates are -4.63%, -1.75% and -3.71% from that historical
row-8 confirmation. All 21 request hashes and actual token/cache counts match;
generated-output hashes differ. The controlled row-0 versus row-8 gains are
reported in the historical tuning section below. No original-94 concurrent
baseline was collected.

**Aggregate end-to-end throughput includes prompt processing and client wall
time. Server decode rate measures the generation interval.** The roughly
80-token/s short decode result and 48.55-token/s four-request aggregate result
come from different metrics and workloads; they are not an 80-to-48 decode
regression. The service retains the measured four-request disk profile with
row budget eight, while the single/RAM profile remains available.


### Long-context retrieval on the selected service profile

Two serial requests run in the selected four-slot 3.05 concurrent/disk
configuration, with the same final sources and 2,048-token chunk. Both recover
all three exact synthetic values placed at separated positions in the prompt.
The request labels are nominal size targets; **the actual API prompt counts
are recorded below** and are not silently rounded down to those labels.

| Nominal target | Actual API prompt tokens | Cached prompt tokens | Actual output tokens | Exact values recovered | Client time to first output |
|---|---:|---:|---:|---:|---:|
| 32,768 | 32,943 | 0 | 71 | 3/3 | 44.47 s |
| 240,000 | 240,176 | 0 | 71 | 3/3 | 302.59 s |

Both finish normally with `stop`, no client/report errors, and pass the strict
retrieval and usage gates. The larger request reports server prefill throughput
of 796.39 tokens/s. Its raw-text token offsets for the three values are 28,828,
117,656 and 211,285; these are raw-text offsets, not asserted chat-token indices.

This demonstrates the tested serial retrieval tasks at approximately 32K and
240K in the allocated four-slot pool. It does not test four simultaneous
240K requests, every position/value combination, all four packs at long
context, or general long-document reasoning. No old-runtime long-context
baseline was collected, so no long-context speedup is claimed.

The [final API archive](bench/results/2026-10-08/final-api-validation/README.md)
retains the complete five-job batch, actual source/config/template identities,
all successful and failed clients, 45 measured short-output samples, 21 measured
concurrent requests, both retrieval reports and owned cleanup evidence. The
batch completed at 13:55:37 UTC. Three jobs pass every scheduled client; the
3.05 single and concurrent jobs retain their targeted literal-copy failures.
Every owned server remained alive until its requested SIGTERM and exited 0;
all recorded client and server groups are empty afterward.

### Completed primary matrix: engine `16ca20d` / TabbyAPI `f4fb6b7`

The 17-job primary matrix ran from **10:47:46 to 11:46:59 UTC** using recipe
`218bd438225243e795383f10ef7a886b06d5d839`, engine
`16ca20d27c0e4cce15a9bbc131e6d047065395b5` and TabbyAPI
`f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6`. Every performance client completed,
and all 17 owned servers shut down with SIGTERM and exit code 0. Thirteen whole
jobs passed; four retained failed tool clients. These measurements predate the
later reasoning-boundary fixes and Cyberfrost template override. Later source
pairs need their own API and performance validation.

The [primary evidence archive](bench/results/2026-10-08/performance-primary/README.md)
retains all reports, actual commands, settings, model metadata, client exits,
resource samples and source hashes. Its
[original comparison](bench/results/2026-10-08/performance-primary/comparison-original.json)
checks that the measured requests and actual token counts match exactly between
the original and primary controls. Parenthesized changes below are relative to
the original table above; every cell is a median of three measured requests.

| Pack | PLE table | Code tok/s (change) | DevOps tok/s (change) | Prose tok/s (change) | Tool smoke |
|---|---|---:|---:|---:|---:|
| Flat 3.05 | RAM | 80.05 (-2.96%) | 74.01 (-0.13%) | 53.56 (-4.24%) | 28/28 |
| Flat 4.05 | Disk | 76.18 (-1.09%) | 66.37 (-2.05%) | 47.64 (-1.91%) | 24/28 |
| SAGE 4.15 | Disk | 52.40 (+7.42%) | 49.85 (+7.25%) | 33.76 (+9.79%) | 28/28 |
| Cyberfrost 3.87 | Disk | 44.07 (+7.15%) | 37.24 (+7.54%) | 34.79 (+8.41%) | 26/28 |

The flat packs show small regressions in these short suites. SAGE and Cyberfrost
show higher measured decode rates. The tool columns are separate acceptance
gates: the 4.05 control failed both exact-string tests and both adversarial
required-tool tests; Cyberfrost failed both exact-string tests. Those failures
remain in the archive and prevent treating those whole jobs as qualified.

The primary controls use the three GDN compatibility flags `1/0/1`,
`EXL3_ATTN_DECODE_LEGACY_SPLITS=1`, `EXL3_GEMM_LEGACY_TILES=1`,
`EXL3_MOE_COOP_MIXEDK=0`, `EXL3_MOE_MIXEDK_NOSYNC=1`,
`EXL3_MOE_COOP_KSPLIT=1`, `EXL3_GR_INT8=1` and `EXL3_INT8_GEMV=0`.
MTP uses depth 5, confidence 0.6 and the 65,536-token shortlist.
`EXL3_DRAFT_ROW_BUDGET=0` is the control. The full resolved environment for each
job is preserved in its final result; the GDN flags and numerical scope are
specified in the numerical section above.

### Mixed-K synchronization comparison

The `02-` jobs change only `EXL3_MOE_MIXEDK_NOSYNC`, using their own immediately
preceding matched control on the same primary engine/server pair. The table is
oriented as the gain from enabling the setting; each value uses three measured
400-token requests. It is independent of the multi-change old-to-new comparison.

| Pack / prompt | NOSYNC=0 tok/s | NOSYNC=1 tok/s | Gain from NOSYNC=1 |
|---|---:|---:|---:|
| SAGE 4.15 / code | 48.66 | 52.39 | +7.67% |
| SAGE 4.15 / devops | 46.64 | 50.01 | +7.23% |
| SAGE 4.15 / prose | 30.65 | 33.83 | +10.38% |
| Cyberfrost 3.87 / code | 41.40 | 44.04 | +6.38% |
| Cyberfrost 3.87 / devops | 34.56 | 37.13 | +7.44% |
| Cyberfrost 3.87 / prose | 32.03 | 34.79 | +8.62% |

Both SAGE jobs passed all eight selected tool checks. Both Cyberfrost jobs passed
6/8, with the same two exact-string failures retained. These results support the
measured speed effect; they do not remove the Cyberfrost tool limitation. All
paired source, environment and request-setting checks are in the
[generated summary](bench/results/2026-10-08/performance-primary/summary/summary.json).

### Dynamic draft depth and confidence

The `03-` group uses flat 3.05 with the PLE table in RAM. Each variant changes one
setting against this group's control; the control was reloaded and measured
separately from the `01-` baseline-aligned table.

| 3.05 dynamic-draft setting | Code tok/s | DevOps tok/s | Prose tok/s |
|---|---:|---:|---:|
| Depth 5, confidence 0.6 (control) | 79.92 | 73.77 | 53.63 |
| Depth 3, confidence 0.6 | 77.60 | 69.46 | 54.38 |
| Depth 5, confidence 0.4 | 81.64 | 75.56 | 52.22 |
| Depth 5, confidence 0.8 | 74.13 | 69.90 | 53.70 |

Confidence 0.4 improves code and DevOps by 2.15% and 2.43%, but lowers prose by
2.63%. Depth 3 improves prose by 1.40% while lowering code and DevOps by 2.90%
and 5.84%. Confidence 0.8 also lowers code and DevOps. All four jobs passed 8/8
selected tool checks. A higher draft-acceptance percentage is therefore not a
sufficient reason to select a setting: accepted work, rejected work and measured
wall time all matter. This group does not establish one faster setting across
all three prompt classes.

### Cold prefill and chunk size

The baseline-aligned medium-context comparison uses one warmup and two measured
requests per pack, run ID `overnight-prefill`, and 256 actual output tokens. Its
construction target is approximately 16,384 tokens, but the serialized requests
contain the actual counts below. Every measured request reports zero cached
prompt tokens. Prefill numbers retain three decimals because a median of two
server timings can have a half-hundredth value.

| Pack | Actual prompt tokens | Original prefill tok/s | Primary prefill tok/s | Change |
|---|---:|---:|---:|---:|
| Flat 3.05 | 12,157 | 684.040 | 688.985 | +0.72% |
| Flat 4.05 | 12,157 | 672.825 | 677.875 | +0.75% |
| SAGE 4.15 | 12,157 | 909.295 | 921.430 | +1.33% |
| Cyberfrost 3.87 | 12,266 | 861.145 | 860.805 | -0.04% |

Cyberfrost adds 109 template tokens to each short request as well: code, DevOps
and prose contain 196/172/181 tokens, compared with 87/63/72 for the other packs.
Its baked template also forces thinking on. The original-to-primary comparison
preserves that behavior and uses identical request hashes per pack; cross-pack
serialized prompts and reasoning behavior are not identical. The later external
template override is a separate change requiring separate measurements.

The `04-` chunk comparison uses flat 3.05 with the PLE table in RAM, 12,157 actual
prompt tokens and 256 output tokens, with all other settings fixed. These are
separate two-request medians from the control pack table above.

| Chunk size | Prefill tok/s | Time to first output, s | Decode tok/s |
|---|---:|---:|---:|
| 1,024 | 547.120 | 22.3846 | 60.220 |
| 2,048 (control) | 688.790 | 17.8125 | 57.785 |
| 4,096 | 827.980 | 14.8448 | 54.905 |

Chunk 4,096 raises measured prefill throughput by **20.21%** and reduces time to
first output by **16.66%** relative to chunk 2,048. It also lowers decode by
**4.98%** in this test. Chunk 1,024 has the opposite tradeoff: prefill is 20.57%
slower while decode is 4.21% faster. These short follow-on generations do not
establish full-window or every-workload behavior for either chunk size.

### Concurrent requests and draft row budget

The `05-` pair uses flat 3.05, `PROFILE=concurrent`, the PLE table streamed from
disk, `MAX_BATCH_SIZE=4`, and one shared 1,048,576-token K8/V8 pool. It changes
only `EXL3_DRAFT_ROW_BUDGET=0` to `8`. Each concurrency level has one warmup
batch and three complete measured batches. Each request contains **849 actual
prompt tokens**, emits **256 actual output tokens**, and reports zero cached
tokens; the nominal prompt-construction target was 1,024.

| Concurrent requests | Row budget 0, aggregate tok/s | Row budget 8, aggregate tok/s | Throughput change | Median TTFT, budget 0 → 8, s |
|---:|---:|---:|---:|---:|
| 1 | 40.5032 | 40.4456 | -0.14% | 3.285 → 3.374 |
| 2 | 34.2972 | 47.2729 | +37.83% | 6.637 → 6.410 |
| 4 | 36.8254 | 50.4184 | +36.91% | 13.425 → 12.636 |

Aggregate throughput is total emitted tokens divided by the interval from the
first request start to the last request finish, including prefill and queueing.
It is not a sum of server-reported per-request decode rates. The row budget
raises this aggregate by **37.83% at two requests** and **36.91% at four**, while
the one-request result changes by −0.14%. Both jobs passed 8/8 selected tool
checks. The budget bounds the draft window more tightly as the active batch
grows; the gain must be assessed with actual accepted/rejected work, not by
assuming that a longer draft window is always faster.

All 42 measured concurrent requests completed. This tests short prompts inside
the reserved shared pool; it does not validate four simultaneous 262,144-token
sessions. The retained per-request latency, draft counts, overlap and errors
provide the limits of the comparison.

### Resource and startup interpretation

Resource samples are sparse observations rather than guaranteed peak
measurements. Unified-memory accounting, resident file pages and GPU allocations
overlap in ways that make simple `memory.used` claims misleading. The recorded
system memory, swap and process information accompany the GPU samples.

Initial source compilation and first-use kernel compilation are separate from
warmed inference. These jobs record startup duration, but do not compare a new
build's first load with an already compiled old installation as though both
had identical startup conditions.

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
requires its own qualification. The [optional archive](bench/results/2026-10-08/performance-optional/README.md)
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

## Tool calling and client behavior

The fork updates cover incremental Qwen parsing, schema-aware strings and
nullable types, call completeness, automatic/named/required choices, parallel-call
limits, streamed assistant roles, aliases and raw completion model identity.
Parser throughput is measured separately on CPU and is not presented as an
equivalent multiplier for model inference.

The live validation includes:

- Streaming and nonstreaming synthetic tool fixtures, including adversarial
  named/required choices, parallel-call limits and deliberate truncation.
- OpenAI SDK stream accumulation, final response revalidation and tool-result
  round trips that reuse the SDK's actual assistant message.
- Raw one-token prompts, model aliases, nullable values, invalid-request
  recovery and an early client disconnect followed by a healthy generation.

No returned tool call is executed. All tool results used in round trips are
synthetic fixtures.

See [tool-calling semantics and limitations](docs/tool-calling.md). XML and
function-name constraints enforce call structure; they do not guarantee full
JSON-schema semantics or the argument values requested in the prompt. A raw
`string|null` parameter containing literal `null` is a separate wire-format
ambiguity from the observed model-generated empty value.

The early flat-3.05 candidate (engine `b5785675`, Tabby `e99ef897`) passed
all five SDK checks and 26 of 28 broad tool checks. Its two string failures were traced to actual incomplete model
output in `auto` mode: the model substituted a chat delimiter for literal
markup and stopped. The same request with named forcing preserved the literal
string correctly. The parser's explicit error was correct; a successful
`tool_calls` finish with incomplete arguments would have hidden the failure.
Streaming errors can follow an HTTP 200 response, so the client also checks
SSE error events and the final completion status.

The recovery suite also exposed a semantic empty-string-versus-null error in
a nullable field. Captured wire output contained an empty parameter. Coercing
all empty strings to null would corrupt valid values, so that was not treated
as a parser fix.

### Parser and tokenizer checks

At exact Tabby commit `3a4d2d5732f0a4de23ffbf2c493d21e9534417aa`,
the WSL CPU suite passed **320 tests and 4,829 subtests**, with two tests
skipped because that CPU environment has no ExLlama GPU runtime. The actual
pack tokenizer passed **27 cases and three prefix-mask checks**. Those skips
are not counted as passes. See the [CPU evidence](bench/results/2026-10-08/tabby-cpu-3a/manifest.json).

A separate fixed-input CPU parser comparison used Python 3.10.12 on x86-64
WSL, 24-character streaming chunks and three repetitions per size. The exact
baseline was `dd0b81b06f21748c232a6b0e52df773a7113b7f9`; the candidate was
`3a4d2d5732f0a4de23ffbf2c493d21e9534417aa`.

| Argument length | Baseline median | Updated median | Baseline / updated |
|---|---:|---:|---:|
| 8 KiB | 4.548 ms | 3.888 ms | 1.17x |
| 256 KiB | 293.797 ms | 111.762 ms | 2.63x |
| 1 MiB | 10.727 s | 0.462 s | 23.21x |

These are parser costs alone. They do not represent a 23x increase in model
or API generation throughput. The [raw comparison](bench/results/2026-10-08/parser-dd0b-3a/comparison.json)
and [source/fixture manifest](bench/results/2026-10-08/parser-dd0b-3a/manifest.json)
identify the measured revisions and preserve every repetition. Earlier
unbound parser experiments remain labeled historical in the CPU archive.

### Final focused string-value checks

The unchanged two-case string smoke tests now pass in both streaming and
nonstreaming mode for flat 4.05 and Cyberfrost at final engine `24f0dec` and
Tabby `5a4f3ef`. Both runs use the single/disk configuration; Cyber uses the
reviewed external thinking-template override.

| Pack | Live requests | Independent raw parameter checks | Actual-parser replay comparisons |
|---|---:|---:|---:|
| Flat 4.05 | 2/2 passed | 8/8 exact | 32/32 match API |
| Cyberfrost with corrected template | 2/2 passed | 8/8 exact | 32/32 match API |

Every raw parameter body contains the requested value exactly, including
`<think>literal</think>` inside the string. Native and accumulated backend
text are byte-identical, and both requests in each pack start outside reasoning.
The raw-body inspection checks physical text independently of the parser.
The replay then exercises native/backend text, four chunk sizes, and both
collector modes. These are four live requests, sixteen raw-body checks and
sixty-four CPU replay comparisons, recorded as separate scopes in the
[final string evidence](bench/results/2026-10-08/final-strings-validation/README.md).
The earlier failures remain historical; their missing raw output does not
support a retrospective cause assignment. The full API qualification follows.

### Final broad API confirmation at `24f0dece3` / `5a4f3ef`

The four final single-request runs use the ordinary installed runtime, clean
recipe `3337bc8`, the selected 2,048-token chunk and each pack's recorded
placement. The Cyber run additionally uses the reviewed template override
that honors `enable_thinking:false`.

| Pack | Broad tool suite | OpenAI SDK | Recovery/resilience | Automatic choice |
|---|---:|---:|---:|---:|
| Flat 3.05, PLE in RAM | 28/28 | 5/5 | 19/19 | 27/27, three rounds |
| Flat 4.05, PLE on disk | 28/28 | 5/5 | 19/19 | 9/9 |
| SAGE 4.15, PLE on disk | 28/28 | 5/5 | 19/19 | 9/9 |
| Cyber Frost 3.87, PLE on disk, corrected template | 28/28 | 5/5 | 19/19 | 9/9 |

These total **262 passing checks** across the four suites: 112 broad tool,
20 SDK, 76 resilience and 54 automatic-choice checks. Repeated automatic
rounds are repeated executions, not additional independent task categories.
They do not include the targeted literal-marker failures or the separate
concurrent suite. The earlier 4.05/Cyber failures at historical source pairs
are retained in their archives; they are not the result of these passing
final requests.

The selected 3.05 concurrent profile's functional client completes **52
checks: 44 pass and eight literal-copy checks fail**. Its four simultaneous
automatic-choice clients pass all 36 checks; its forced-choice clients pass
eight. The four budgeted and four unbudgeted literal-marker cases remain
failures. Six return `thinkaliteralthink`; two unbudgeted streaming cases
return `thinkpliteral</think>` instead of the exact requested string.

All 16 concurrent child sessions finish naturally, with both ownership-empty
flags true and no cleanup signals required. Eight clients exit 0 and the
eight one-case literal clients exit 1. Across the four single-pack runs and
this concurrent functional client, the retained total is **306 passing and
16 failing checks out of 322**, not an all-pass result. The separately scored
benchmark and long-context results follow their own gates.


### Native reasoning-marker copying: retained limitation and tested workaround

The additional fixture asks the 3.05 pack to return exactly
`<think>literal</think>` as a `record_text` string while reasoning is enabled.
The final main matrix retains all **16 failed executions** across single and
concurrent, required/named, streaming/nonstreaming, budgeted/unbudgeted modes.
The calls are structurally valid, but their string values differ from the task.
This does not change the passing broad-suite counts above.

Input tracing finds no lost prompt characters: actual Tabby formatting and
both original/final engine tokenizers preserve the same prompt and all 348
input IDs. The requested marker sequence uses the native IDs for `<think>`,
`literal` and `</think>`.

A separately named raw replay produces one exact value and seven wrong values.
All eight native completion strings equal accumulated backend text. Independent
inspection of each first parameter body, removing only framing line feeds,
exactly matches the API argument; nested opener text is preserved as string
data. **128 actual-parser replays** across native/backend text, four chunk
partitions and streaming/nonstreaming assembly reproduce all API outcomes.
The wrong values are already present in generated output before collector
parsing. This supports a generation-time literal-copy limitation for these
requests, rather than attributing the altered characters to collector loss.

Changing only the existing request setting from `enable_thinking:true` to
`enable_thinking:false` passes **all four separately tested required/named,
streaming/nonstreaming variants**. The message still says “Think briefly”; its
text, expected value and tool schema are unchanged. This is a useful tested
workaround in the 3.05 single/RAM diagnostic, not a guarantee for every pack,
profile, prompt or schema.
It neither changes the server default nor turns the original failures into passes.

The capture instrumentation's own failures are also retained. Its first strict
matcher expected top-p 1.0, while these requests omit top-p and the deployed
sampling override supplies 0.95, so it captured zero traces. A fresh diagnostic
changes only that expectation and captures all eight. Its scalar metadata
recorder then writes null for YAML's `ScalarFloat(0.95)`, because the recorder
accepts only exact primitive types. Both original `capture_valid=false` results
remain unchanged. Separate actual-YAML/config/Pydantic/formatting checks,
source and request hashes, IDs, deployment/config hashes and lifecycle records
reconcile this metadata omission offline. The raw/parser conclusions above are
from that separately qualified review, not a claim that the original capture
assessors passed.

The [literal validation archive](bench/results/2026-10-08/final-literal-validation/README.md)
contains the failed original reports, both diagnostic attempts, eight raw
records, unchanged source snapshots, metadata reconciliation, parser review and
the separately passing thinking-disabled requests. All controller-owned
processes exit and release their groups before service promotion.




## Numerical and kernel checks

### Predeclared comparison rules

Before choosing settings, paired numerical limits were recorded at
2026-10-08 07:40:56 UTC:

| Metric | Rule |
|---|---|
| Pooled increase in next-token NLL | At most 0.02 nats |
| Increase for one fixed-input case | At most 0.05 nats |
| Additional high-confidence disagreements | At most 1 per 48 positions and 2 pooled |
| Mean paired KL above 0.02, or p95 above 0.10 | Investigate; KL alone is not proof of quality loss |

The comparison uses exactly the same saved input IDs, pack view and execution
path on the original and candidate engines. Code, tool-like and multilingual
text are tested with prefixes of 255 and 1,023 tokens, query lengths 1 and 6,
and 48 scored continuation positions. Complete prefill is scored separately
and is not double-counted because two query lengths share that reference.
The initial diagnostic uses **FP16 KV storage**. These short prefixes remain
below the QSA selection threshold; longer-prefix and K8/V8 checks are reported
separately. MTP is disabled in the numerical diagnostic to isolate target-model paths.

An early probe omitted the engine's required `rewind(0)` history commit after
an accepted multi-token forward. That was a harness error, and its artifacts
were excluded. Valid format-2 runs commit recurrent convolution/PLE history,
use a physical page permutation and preserve exact input tensors. CPU
fixtures verify the required commit and saved-tensor behavior.

The saved-logit domain is the complete configured vocabulary of 248,320
columns. The tokenizer knows IDs 0–248,076; the server masks the other 243
columns during sampling. A separate audit of original `94ba01d` versus exploratory candidate `b578`
checks 18 saved tensors / 864 positions per runtime for each of flat 3.05 and
SAGE. No extra column wins argmax in those samples; their maximum per-position
contribution to NLL is 0.0000120 nats for flat 3.05 and 0.0000823 nats for
SAGE. This is not an audit of every later revision or all four packs. That is too small to explain the investigated gate failures. The
paired domain and thresholds were not silently changed after seeing results.

This is a small, correlated numerical regression corpus. It does not measure
GPQA, HLE, coding-benchmark accuracy, broad multilingual capability or quality
relative to an FP8 source model.

### Runtime fixes and source-faithful checks

The integrated engine includes fixes for:

- Cache handling when dynamic MTP accepts no draft rows.
- Missing carry on a one-token prompt and after relevant rewind cases.
- An asynchronous host-to-device race in MTP position buffers, using
  immutable pinned positions for each queued step.
- The fp16 normalization rounding used before int8 GR quantization.
- GPU embedding-mirror behavior across reload.
- Mixed-K cooperative paths and a configurable MTP row budget. The engine
  default remains zero (disabled); any nonzero recipe preset needs its own
  measured concurrency comparison.

Tests exercise mixed-K parity, page/cache behavior, GR paths and the model's
actual 2,560-by-640 expert geometry with top-10 routing. Independent reference
calculations are used where practical rather than comparing a kernel only
against another path that shares its arithmetic.

Early updated full-model probes still exceeded the predeclared limits despite
passing the individual GPU checks. The investigation isolated upstream GDN
projection and convolution precision changes. Changing projection/input
flags alone did not satisfy the gate; those exploratory settings were not
promoted on the strength of speed alone.

The compatibility settings also preserve the original BF16 convolution
product rounding and dense-attention reduction plan. The selected attention
mode retains packed rows and vectorized split work, while restoring the
original split boundaries and combine compiler contract. Eight actual GPU
comparisons against the byte-checked original attention functions produced
bitwise-identical outputs; disabled mode preserved the updated default.

An independent FP64 attention calculation showed similarly small errors in
both original and updated kernels. This is a numerical compatibility measure
for the observed model regression gates; it is not evidence that the upstream
attention formula is wrong. Small rounding differences can change subsequent
mixture-of-experts routing.

The faster cooperative mixed-K path exceeded the SAGE full-model gates even
after attention compatibility was restored. The alternative unified path can
remove redundant host synchronization while keeping its existing arithmetic.
Its SAGE short-context probe matched the original paged logits with no gate or
KL alerts. Final selected settings and broader results follow below.

Python/Triton-only diagnostic commits were tested against the existing native
extension only after verifying that their native C++/CUDA inputs were
unchanged. Those diagnostics do not bypass the serving launcher's full-source
fingerprint requirement; the final installation requires a matching build.

The final native compatibility switch also restores the historical automatic
GEMM tile family and its autotune hash domain. A saved-input control loaded the
actual flat-4.05 first MoE layer. All experts with 1–16 rows matched; every
expert with 17–32 rows differed under the newer automatic plans. For each of
24 sampled row counts from 9 through 32, forcing the preserved historical
shape and SM plan restored the original expert output bit for bit.

The default-off native switch was then checked in two fresh GPU processes.
Four row-group tests with the switch enabled matched the original saved
expert outputs; four with it disabled matched the pre-switch updated outputs.
Each group exercises eager execution, capture and replay with different
expert pointers. Those controls use private copies of the same preserved
autotune cache. They make no inference-speed claim.

### Qualified target-kernel controls at `16ca20d`

The final target-kernel configuration for the primary performance matrix uses
engine `16ca20d27c0e4cce15a9bbc131e6d047065395b5`. It retains GDN projection,
convolution and BF16-product settings `1/0/1`, legacy attention reduction and
GEMM tile settings `1`, mixed-K Coop `0`, and mixed-K NOSYNC `1`. The additional
`EXL3_MOE_COOP_KSPLIT=1` control resolves the remaining flat-pack q1 differences.

The four-request Q8 test first caught a real regression: the 3.05
multilingual/prefix-1023/q1 case produced three additional high-confidence
disagreements, taking the pooled count beyond the predeclared limit of two.
Changing only the shared-expert split-K control to `1` restored zero paired
logit difference in all 12 paged and six unique-prefill cases. The failed run
and successful one-variable control are both retained.

Source inspection explains the measured cause. With the recorded wide tile,
the shared gate/up projection has 10 base blocks for batch1/q1 and 40 for
batch4/q1 on 48 SMs. The upstream occupancy planner chooses split-K `2`, while
the original plan used `1`. Splitting rounds separate FP16 partial results
before their sum. The routed top-10 launch, the down projection and the q6
path do not acquire that same split. This matches the observed q1-only scope;
the isolated control establishes the cause for the measured regression.

The completed same-path Q8 controls against original engine `94ba01d` are:

| Pack and batch | Prefix lengths | Paged cases / positions | Unique-prefill cases / positions | Largest paired absolute logit difference | Core gates / KL alerts |
|---|---|---:|---:|---:|---|
| Flat 3.05, batch1, KSPLIT1 | 255, 1,023, 4,095 | 18 / 864 | 9 / 432 | 0 | pass / 0 |
| Flat 4.05, batch1, KSPLIT1 | 255, 1,023, 4,095 | 18 / 864 | 9 / 432 | 0 | pass / 0 |
| SAGE 4.15, batch1 | 255, 1,023, 4,095 | 18 / 864 | 9 / 432 | 0 | pass / 0 |
| Cyberfrost, batch1, KSPLIT1 | 255, 1,023, 4,095 | 18 / 864 | 9 / 432 | 0 | pass / 0 |
| Flat 3.05, batch4, KSPLIT1 | 255, 1,023 | 12 / 2,304 | 6 / 1,152 | 0 | pass / 0 |
| SAGE 4.15, batch4 | 255, 1,023 | 12 / 2,304 | 6 / 1,152 | 0 | pass / 0 |

These rows also have zero paired NLL change, zero paired KL and zero
additional high-confidence disagreements. The repeated prefixes, q lengths
and batched inputs are correlated; the position counts are not independent
capability trials. The 4,095-token tests exercise sparse-attention selection.
They do not substitute for the separate full-context serving check.

SAGE's two rows use the preceding otherwise-matching control without an
explicit KSPLIT override. All 48 SAGE target MoE layers use the unified
mixed-K path, which does not read that shared-cooperative split control.
Cyberfrost has eight uniform-K target layers, so it received its own explicit
KSPLIT1 rerun. Both mixed packs have a uniform-K MTP layer; target-only probes
do not establish that their draft model is unaffected. Live MTP generation,
tool-call and performance checks cover that separate serving scope.

The [10:41 evidence snapshot](bench/results/2026-10-08/numerical-104102/README.md)
preserves the earlier failures and first successful batch4 control, with full
input/probe/assessment JSON and hashes of the retained tensors. The completed
[batch1 follow-ups](bench/results/2026-10-08/numerical-ksplit1-105304/README.md)
are published as a separate addition. Their full-file hashes were completed
after all measured work and the first literal diagnostic had released the host;
the [final tensor-hash record](bench/results/2026-10-08/final-tensor-hashes/README.md)
contains those three files and the later 4,096/2,048-chunk artifacts. All six
files, totaling 9,012,969,536 bytes, were streamed for SHA256 with unchanged
size, modification time, device and inode before and after. The record completed
at 13:59:55 UTC. No tensor deserialization or measurement overlap occurred.

### Final serving-chunk comparison at `24f0dece3`

After the ordinary final installation, the original and final engines were
compared again with the actual selected **2,048-token cached prefill chunk**.
Both used the 3.05 pack, PLE in RAM, K8/V8 storage and target-only inference.
The frozen inputs span code, tool-like and multilingual text; prefixes of
255, 1,023 and 4,095 tokens; query lengths 1 and 6; and batch sizes 1 and 4.

All **36 paged comparisons / 4,320 positions** and **18 unique complete-prefill
comparisons / 2,160 positions** report maximum absolute paired logit difference
zero. Paired KL, next-token NLL delta and added high-confidence disagreements
are also zero. The unchanged core gates pass with **zero investigation alerts**.
This extends the 3.05 four-request numerical check to prefix 4,095 at the
selected chunk size. It does not extend the other packs' batch-size coverage.
Both engines have the same 11 paged high-confidence disagreements against their
own cache-free references; those are not newly introduced disagreements.

The optional 4,096-token chunk passed the core gates but triggered three KL
investigation thresholds across two prefix-4,095 cases. Its reference used a
1,024-token chunk, so that comparison does not isolate a source-code change.
The multilingual case improved NLL while its paired distribution changed;
the result is retained as an investigation, not relabeled as a core failure.
**The service and recipe retain 2,048**, and the optional 4,096 long-context
job was not selected. These decisions do not erase its measured prefill gain.

The [final numerical archive](bench/results/2026-10-08/final-numerical-chunks/README.md)
contains the unchanged gates, input jobs, source hashes, small reports and full
assessments. Large tensor artifacts remain at their recorded Spark paths;
packaging did not read or hash them while API benchmarking was in progress.

### Reasoning boundaries in the generator and server

The final tool repair spans the producer and the API layer. The earlier
`16ca20d`/`f4fb6b7` pair could enforce a reasoning budget after an accepted
token window had already advanced. In the preserved three-run replay, the
unchanged automatic suite passed 9/9, 9/9 and 8/9. Six raw captures showed five
closures at accepted producer position 24 and one at position 28; the delayed
closure changed the generated continuation. Every capture had an empty
consumer queue at injection. Twenty-four parser replays, using four chunk
sizes for each trace, reproduced the API channels exactly. The evidence
supports a producer timing error and subsequent model continuation, rather
than an SSE ordering or parser-channel leak.

The [raw observer archive](bench/results/2026-10-08/reasoning-observer-f4/README.md)
retains the failed run, every matching trace, source identities and a portable
replay. Its presence is intentional: a later successful run does not erase the
regression that motivated the change.

Engine `f0beceb350d52bfdd62cbce6e6b462b35dd04759` contains the final production
repair. Accepted producer positions own the budget state. The callback changes
the constraint phase before the following token is sampled, while the
incremental tool parser protects open calls, parameter data and partial
delimiters. Rewinds, cancellation, requeue, stop conditions, maximum output
length and callback failures have explicit regression coverage. The active
guard prevents the prebatched verification path from advancing past a boundary
before the corresponding callback can run.

A separate reproduction then confirmed the same class of problem for a
request with **no reasoning budget**: a literal `</think>` inside an argument
activated the raw content-grammar trigger. The final natural-closure guard uses
the producer parser for these eligible requests too, without imposing a
deadline, adding a token limit or forcing closing text. Tabby negotiates this
capability explicitly with the engine. Plain content requests and the existing
policy for calls inside reasoning retain their behavior.

The [final engine CPU archive](bench/results/2026-10-08/engine-cpu-f0/README.md)
records 77 budget/closure tests and the affected selection of 119 tests plus
22 subtests; the 119 includes the 77, so these totals must not be added.
The source-bound comparison from the numerically qualified `16ca20d` engine
to `f0beceb` changes only three Python generator modules and their budget test.
It changes no native kernels, model layers, quantization code or cache
implementation. The final installation still goes through the ordinary
source build and its runtime fingerprint checks.

Tabby
`5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb` passed 382 CPU tests and 5,159
subtests in the independent WSL environment. Two engine-dependent modules
were skipped there and are not counted as passes. Separate tests composed the
actual producer, parser, filter and backend: eight natural-boundary cases and
eleven finite-budget cases passed. These are recorded in the
[final server CPU archive](bench/results/2026-10-08/tabby-cpu-5a/README.md).
The actual Spark installation and live API results are reported separately.

The final engine pin is
`24f0dece34f09c8d1e2359d6b3b3f7befef7331b`. Its follow-ups to `f0beceb`
change only CPU test fixtures and the existing CI workflow. The production
`exllamav3` tree has the same Git tree identity
`b00adae150cef6db7cafaec143c8945e22231042` throughout.
A stale fixture had not initialized the new optional budget state; fixing
that fixture made the existing source-test selection pass **243 tests and
186 subtests** at `455cc4a`. The final `24f0dec` workflow also includes the
77 budget/closure tests in that same CI job. Its exact combined command passed
**320 tests and 186 subtests** locally. Those counts overlap the earlier
selections and must not be added. The workflow now triggers on test changes
without adding jobs or changing permissions. The
[fixture evidence](bench/results/2026-10-08/engine-ci-fixture-455cc4a3/README.md)
and [final combined selection](bench/results/2026-10-08/engine-ci-budget-24f0/README.md)
retain the commands, logs and source identity proof.

The final hosted [engine CI run](https://github.com/vcruz305/exllamav3/actions/runs/37776591337)
passed its CUDA wheel build, installation and import, followed by 17 installed-package
CPU tests and the source selection of 320 tests plus 186 subtests. CI used its normal
synthetic merge checkout; its complete tree is identical to the tested `24f0dec`
head. This was an x86_64 wheel build and CPU validation, separate from the ARM
Spark runtime and GPU experiments. The [release evidence](bench/results/2026-10-08/final-engine-release-evidence/README.md)
retains the full log, metadata and source identities. A fresh upstream check at
12:56:10 UTC found no newer upstream head for either fork.

The ordinary setup on the Spark completed at **13:00:12 UTC** with engine
`24f0dece34f09c8d1e2359d6b3b3f7befef7331b`, Tabby
`5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb`, and qualification recipe
`3337bc8d64e4befafa2a1aff342e07abbea242ac`. The source build for `sm_121`,
runtime import/fingerprint verification and `pip check` succeeded. The actual
Spark engine budget suite passed **77 tests**; the full Tabby suite passed
**398 tests and 5,159 subtests, with no skips**. The actual model-tokenizer check
also passed. These overlap earlier CPU selections and are not additive totals.
The [setup evidence](bench/results/2026-10-08/final-runtime-setup/README.md)
records all five commands, their exits, source identities and build state.


## Setup, provenance and deployment

The setup script preserves local checkouts and rejects dirty sources before
mutating refs or installing packages. A full source/build fingerprint records
the engine commit, canonical Python and source paths, PyTorch, CUDA, compiler
and architecture. The launcher verifies it before loading and writes a
redacted deployment snapshot.

The affected NVIDIA cuSPARSELt wheel has valid aarch64 library bytes but the
internal architecture tag `manylinux2014_sbsa`. The repair is limited to the
exact affected package/version/platform and verifies ELF architecture and the
original RECORD hash before changing the tag to
`manylinux2014_aarch64` and updating its RECORD entry. It keeps the CUDA
library and PyTorch requirement intact, records recovery/audit information
and still requires `pip check` to pass. Later correctly tagged package
versions are not pinned back to this workaround.

The persistent service uses a separate state directory, loopback port 8899
and a user-systemd unit. Restart behavior, logs and rollback commands are
documented in [docs/service.md](docs/service.md).

### Installed service and completed lifecycle rehearsal

The permanent user unit is `qwen38-exl3.service`, enabled with the existing
user-manager lingering setting. Its literal environment file is
`/home/cruzspark/.config/qwen38-exl3/service.env`; its SHA256 is
`9e21fdbed8f1e66cf276f5e947888cd9deb13e580fa6a2533a990d4e8d5c6ad9`.
The installed settings match the selected measured job:

| Setting | Value |
|---|---|
| Pack | `/home/cruzspark/models/flashnext-exl3-3.05bpw` |
| Profile / active jobs | `concurrent` / 4 |
| N-gram placement | Disk (`NGRAM_RAM=false`) |
| Per-request context / shared KV pool | 262,144 / 1,048,576 tokens |
| KV storage / prefill chunk | K8/V8 / 2,048 tokens |
| Drafting | Dynamic MTP, depth 5, confidence 0.6 |
| Draft row budget / head width | 8 / 65,536 |
| CPU affinity | 5–9,15–19 |
| API / advertised model | `http://127.0.0.1:8899/v1` / `Qwen3.8-Flash-Next-EXL3` |

The unit names the final dated runtime directly at
`/home/cruzspark/qwen38-exl3-20261008`. Its launcher uses the canonical recipe
at `/home/cruzspark/qwen-spark-recipe`. Each systemd invocation creates a fresh
private state directory under `/home/cruzspark/.local/state/qwen38-exl3/runs`,
containing its own rendered configuration, start identity and deployment
snapshot. The reviewed unit uses `Restart=on-failure`, a 10-second delay and
control-group cleanup.

The exact R1/24f0/f650 rehearsal ran **14:12:53–14:16:29 UTC** and passed all
four phases:

1. Start the new service; verify source, model, environment, configuration,
   owned MainPID/cgroup and fresh invocation; check both API modes.
2. Send one controlled SIGKILL to the verified MainPID; observe automatic
   recovery with exactly one added restart and a different PID/invocation;
   check both API modes again.
3. Stop the new service, verify port release, and start the original
   `94ba01d` engine / `816c321` Tabby runtime at its original canonical path.
   Verify the original launcher's four recorded runtime-file hashes and check
   streaming/nonstreaming generation. The older recipe checkout also contains
   this task's benchmark-client updates; it is not claimed to be wholly clean.
4. Stop the owned original process group (SIGTERM, exit 0), return to the new
   service with another fresh invocation, and check both API modes.

The successful rehearsal includes six new-runtime and two original-runtime
basic generation requests. These prove availability and transport, not new
performance or semantic benchmarks. At its end, the new unit was enabled and
active/running with MainPID 480312, invocation
`1ddc964e1fcd48d391ffdc6dc6bce217`, and restart counter 0. Those identify the
R1 rehearsal endpoint; publication performs a later ordinary restart whose
identity is retained in PR 17.

The [service rehearsal archive](bench/results/2026-10-08/final-service-rehearsal/README.md)
contains source-bound results, installation and merge proof, invocation
configuration and sanitized lifecycle excerpts. Raw original authentication
logs and credential stores are excluded. The original runtime remains at
`/home/cruzspark/qwen38-exl3` for rollback.

Operations on the Spark:

```bash
systemctl --user status qwen38-exl3.service
journalctl --user -u qwen38-exl3.service -f
systemctl --user restart qwen38-exl3.service
```

From another machine, forward the loopback API with:

```bash
ssh -N -L 8899:127.0.0.1:8899 cruzspark@192.168.8.124
```

The client then uses `http://127.0.0.1:8899/v1`. The single/RAM profile remains
available through the documented launcher for its separately measured workload.


## Reproduction and evidence

Use the same pack view, placement, prompts, sampling, warmup and limits before
comparing two engine revisions. Start each tuning job with a fresh server and
save its deployment snapshot. Change one relevant knob at a time; then repeat
the selected configuration as a separate confirmation run.

The experiment controller retains unique attempt directories, verifies the
owned listener and loaded model, records actual child exits, checks source
and model identity before and after measurements, and stops only its own
process sessions. Interrupted attempts are preserved. A completed failure is
not turned into a successful sample by retrying in place.

| Evidence | Contents |
|---|---|
| [Final API batch](bench/results/2026-10-08/final-api-validation/README.md) | All five final jobs, success/failure counts, actual tokens, concurrent and long-context results, sources and cleanup |
| [Primary tuning matrix](bench/results/2026-10-08/performance-primary/README.md) | 17 measured configurations, original comparisons and controlled row-budget/NOSYNC choices |
| [Optional tuning matrix](bench/results/2026-10-08/performance-optional/README.md) | 16 placement, drafting, head-width and concurrency experiments, including memory/swap observations |
| [Original API measurements](bench/results/2026-10-08/original-api/README.md) | Preserved original-runtime requests and measurements |
| [Final installation](bench/results/2026-10-08/final-runtime-setup/README.md) | Ordinary Spark build, source/fingerprint/package checks and on-device CPU/tokenizer results |
| [Engine release/CI](bench/results/2026-10-08/final-engine-release-evidence/README.md) | Hosted build, installed/source CPU counts, upstream identity and source reviews |
| [Final literal evidence](bench/results/2026-10-08/final-literal-validation/README.md) | Original failures, native output, offline metadata reconciliation, 128 parser replays and thinking-disabled workaround |
| [Final focused string checks](bench/results/2026-10-08/final-strings-validation/README.md) | Live 4.05/Cyber strings, native/backend bodies and 64 separate parser replays |
| [Final chunk numerics](bench/results/2026-10-08/final-numerical-chunks/README.md) | Exact matched 2048 comparison and unselected 4096 investigation |
| [Earlier numerical evidence](bench/results/2026-10-08/numerical-104102/README.md) | Preserved failed and successful source-faithful kernel/model controls |
| [Completed tensor hashes](bench/results/2026-10-08/final-tensor-hashes/README.md) | Deferred full-file integrity of six retained tensors after measurements |
| [Service lifecycle](bench/results/2026-10-08/final-service-rehearsal/README.md) | R1 startup, automatic recovery, original rollback, return to final, and sanitized evidence |

Each archive records its collection scope and SHA256 manifest. Large model
weights and saved numerical tensors stay on the Spark; the repository retains
the relevant identities, small reports and full-file tensor digests. Synthetic
tool calls are never executed during validation.


Related changes:
[ExLlamaV3 PR 21](https://github.com/vcruz305/exllamav3/pull/21),
[TabbyAPI PR 2](https://github.com/vcruz305/tabbyAPI/pull/2).
