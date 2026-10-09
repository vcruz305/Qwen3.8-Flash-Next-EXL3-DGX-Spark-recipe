# DGX Spark follow-up validation — 2026-10-09

This completed follow-up deploys a Tabby fix for premature stopping before a
required tool call, adds a cooperative GPU lock to the recipe, and fixes
interrupted benchmark cleanup. The updated Qwen systemd service passed its
final live checks and is available as a disabled standby. The previously
selected CyberFrost workload was restored through the existing REXL3 manager
and verified ready at 08:42:37 UTC.

It also investigates the exact-marker copying failures and flat-pack
single-request slowdowns reported on October 8. An experimental request-local
literal-input option improved the original synthetic task but failed broader
qualification, so it was kept out of the stable deployment. The selected
engine, numerical compatibility settings, model weights, and historical
October 8 evidence remain unchanged.

The fresh controlled flat-pack comparisons do not reproduce the earlier
slowdowns. Their small timing differences do not justify another kernel patch
or a broad speedup claim. The literal-input diagnostics identify input token
representation as a contributor to one failing task; they do not establish a
universal copying guarantee or repair the model's reasoning boundary.

## Sources and scope

| Component | Source used for this follow-up |
|---|---|
| ExLlamaV3 installed source | `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`, version `1.6.0.post1` |
| Engine fork default branch (`master`) | `678ab0dd1c39cb7ba22d577913132f63d4d242f3`, full tree equal to the installed pin |
| Initial Tabby main / control | `f650bb5389e0a273549e47d4d26a765760c013e1` |
| Promoted Tabby main | `f391c96beea0bcd21cbae3c929f75bb98136ff32`, full tree equal to qualified EOS-only `fd8cbeb1` |
| Literal-input candidate | `a70ae1fa9e457e478c3d96bdc84012a3cb331796`, tree `0f278f289af73204786e2c08d58e056c2564eadf` |
| Isolated live GPU-lock recipe | `254b2b03027f25094845dc31f5f87739f3584d2e` |
| Final operational recipe qualification | `22d673463dba8e1f3a862056348769ab85a3c032`, adding the matrix signal fix |
| Historical control engine / Tabby | `94ba01d50a13fa9ff672473f2d0eef8b51a71e99` / `816c32195887aaecea1c64528f2921566766259b` |

At **08:23:21 UTC on October 9**, the primary GitHub API still reported upstream
ExLlamaV3 `151539c77abc7ab7425d30da7a4e8e3c5c154e7b` and upstream TabbyAPI
`2fd6cc76203a66e13042daf7d76e5898b21c1ad8`. Both are already integrated into
the fork versions used here. TabbyAPI continues to follow the fork's `main`;
the candidate commit above records a validation identity, not a new public pin.

The same DGX Spark, four original pack directories, CUDA/PyTorch stack, and
source-built engine from the [October 8 report](VALIDATION_2026-10-08.md) were
used. A fresh [model-input audit](bench/results/2026-10-09/model-input-audit/README.md)
matched the prior metadata, tokenizer/configuration files, sizes, mtimes, and
loader order. It recorded current bounded tensor-header hashes. Prior full
weight hashes were not available, so this is not a new full-weight integrity
claim. No weights were rewritten or requantized.

## Fresh flat-pack performance checks

The historical runtime was retained at its original path. Measurements used
isolated serving states, no other model owner, the shared GPU lock, exact
source/package snapshots, the original request payloads, one warmup and three
measured requests per workload, and actual API usage. Each measured response
emitted 400 tokens. Context and Q8 KV capacity were 262,144, chunk size 2,048,
batch size one, and MTP depth five with dynamic drafting and confidence 0.6.

The older engine's preserved environment has one known `pip check` failure:
the architecture tag on its existing `nvidia-cusparselt-cu13==0.8.1` package.
The comparison archive records that failure and the installed library,
WHEEL/RECORD, and ELF checks. This control was not described as a fully passing
setup or promoted to deployment. Both current-engine environments passed
`pip check`. The two current-engine 4.05 cells used identical package sets;
the old/new engine comparison also includes the existing llguidance version
change and the recorded test-package differences.

### Flat 4.05: engine and server attribution

N-gram tables were streamed from disk in all three cells. The table contains
median server-reported decode tokens per second from three measured requests
per workload; warmups are excluded.

| Cell | Engine / Tabby | Code | DevOps |
|---|---|---:|---:|
| A | Historical `94ba01d` / `816c321` | 75.75 | 66.31 |
| B | Current `24f0dec` / `816c321` | 76.27 | 66.77 |
| C | Current `24f0dec` / `f650bb5` | 76.17 | 67.11 |

All six corresponding measured response strings and accepted/rejected draft
counts matched across the three cells. B versus A was approximately +0.69%
for each workload; C versus B was -0.13% for code and +0.51% for DevOps.
The complete-stack C/A decode-rate differences were +0.55% and +1.21%.

These are sequential, three-repeat observations with rounded server timings
and sparse clock telemetry. They do not prove statistical equivalence or a
meaningful general performance gain. Even the freshly rerun historical cell
was slower than its October 8 numbers despite matching the corresponding
answers and draft counts. This supports treating the earlier small slowdown
as unconfirmed by the controlled repeat, rather than assigning it to Tabby or
a specific kernel.

See the [complete three-cell archive](bench/results/2026-10-09/flat405-attribution/README.md),
including the failed preparation attempt that made no model requests.

### Flat 3.05: historical/current confirmation

N-gram tables were RAM-resident on both sides. This two-cell check used the
original code and prose workloads, with the same warmup/measurement counts.

| Cell | Engine / Tabby | Code | Prose |
|---|---|---:|---:|
| A | Historical `94ba01d` / `816c321` | 79.45 | 53.87 |
| C | Current `24f0dec` / `f650bb5` | 79.99 | 54.37 |

The differences were +0.68% and +0.93%. All six fresh A/C measured answers
and draft counts matched. By contrast, the previous day's historical samples
matched only two of six answer strings and none of the six draft-count pairs
in this fresh historical run. Those cross-day numbers therefore are not a
fixed-work comparison.

The shared cooperative-kernel autotune file was not reset. Its before/after
snapshots across both controlled experiments had the same 19,760 bytes and
SHA256 `ae8cac3b458863cd2a67254f23af88aaeab1310bc8f3f29750ea081735040466`.
Matching response strings and aggregate draft counts do not prove identical
ordering of every internal verification window.

See the [complete 3.05 confirmation archive](bench/results/2026-10-09/flat305-confirmation/README.md).
The prior mixed-K gains, numerical tests, long-context retrieval, and selected
3.05 concurrency settings remain documented in the October 8 report; this
follow-up does not relabel or rerun that evidence as new results.

## Literal-marker attribution

The original exact-copy request asks a tool to record a value containing
`<think>literal</think>`. Its eight variants exercise
stream/non-stream responses and reasoning controls. These are eight interface
variants of **one synthetic task**, not eight independent held-out tasks.

| Input representation | Drafting | Actual prompt tokens | Exact tool values |
|---|---|---:|---:|
| Native added-token IDs | Disabled | 348 | 0/8 |
| Native added-token IDs | MTP | 348 | 1/8 |
| Explicit ordinary BPE for the two user-literal markers | Disabled | 352 | 8/8 |
| Explicit ordinary BPE for the two user-literal markers | MTP | 352 | 8/8 |

The diagnostic BPE runs preserved the rendered prompt bytes and the original
wire request fields. They changed only the verified user-literal input token
IDs, before context and cache accounting. In the target-only native run, the
wrong spelling was already present before reasoning-budget forcing or content
grammar activation. Disabling drafting therefore did not solve the original
failure. With the BPE input, all eight variants in each drafting mode produced
the exact desired value. The MTP confirmation recorded 268 accepted and 157
rejected draft tokens, establishing that drafting actually ran.

The raw BPE output still emitted a natural reasoning-close marker while its
reasoning text contained an unfinished quotation. The diagnostic success does
**not** establish a reasoning-boundary repair. The parser and grammar still
cannot guarantee that the model selects the application's intended value.

The first BPE attempt aborted after an independently restarted manager was
observed; it sent zero client requests and is retained as an operational
failure. The successful retry and MTP confirmation have separate identities.
The diagnostic observers and input hooks are archived evidence and are not
installed in the production candidate.

See the [native drafting ablation](bench/results/2026-10-09/literal-drafting-ablation/README.md)
and [BPE diagnostic archive](bench/results/2026-10-09/literal-user-bpe-diagnostic/README.md)
for raw outputs, token events, request bindings, cleanup records, and the
independent replay/review results.

## Experimental request-local literal input option

The candidate adds the strict JSON Boolean field
`literal_user_control_tokens`, defaulting to `false`. When enabled on supported
ExLlamaV3 chat requests, it represents the nine documented control markers in
proven user text with ordinary BPE token pieces. Actual system, assistant,
tool-result, and template control tokens retain their native representation.

The implementation proves user-text spans with a detached template render,
requires exact reconstruction and tokenizer byte-roundtrip, and binds an
immutable token plan to the request's prompt and native tokenizer. It applies
the expanded IDs before both context-length checks and generation/cache
bookkeeping. It does not mutate the shared native tokenizer or repair output
strings. Default-off requests retain their original tokenization path.

Strict Boolean validation rejects string, integer, and null values with HTTP
422. Unsupported backends, unsupported list/multimodal content, targeted
continued messages, ambiguous or transformed templates, and non-roundtripping
text fail with HTTP 400 before SSE starts. This tokenizer normalizes some
Unicode input; marker-bearing NFD text can consequently fail the exact
roundtrip contract. The request option deliberately reports that case instead
of silently changing the literal text. The existing default-off behavior is
unchanged.

### CPU and source qualification

- The initial CPU environment passed 403 tests and 5,210 subtests, with two
  existing modules skipped because the engine import was unavailable.
- The final focused feature selection passed 21 tests and 51 subtests.
- The actual Spark environment then passed **419 tests and 5,210 subtests,
  with zero skips**. CUDA was hidden during this CPU selection; the real
  installed engine could be imported. The source-build fingerprint
  `243f414eb2fc` and `setup.sh --check`, including `pip check`, passed.
- Actual Qwen/native-source qualification covered 46 rows across 23
  native/opt-in pairs, including the intended NFD rejection. The token-cache
  proof retained seven unchanged prefix pages before changed IDs and showed
  divergence afterward.
- An independent check exercised four distinct immutable plans with real CPU
  tensors and the installed engine's sequence/cache methods, without GPU
  initialization or native-tokenizer mutation.

### Broad live qualification: original failed gate retained

The source-bound candidate ran on the actual Spark with the selected 3.05
concurrent profile, disk-resident N-gram tables, four jobs, row budget eight,
MTP depth five/confidence 0.6, and a 262,144-token shared pool. There were
61 feature checks followed by the unchanged 28-check standard tool suite:
**89 logical checks over 93 actual chat POSTs**. The candidate passed 84/89.
This remains a failed broad qualification, including after later diagnostics.

| Group | Outcome | What it establishes |
|---|---:|---|
| Existing default-policy tool suite | 28/28 | No observed regression in these standard tool cases |
| Original exact-copy task, opt-in | 8/8 | Improvement across eight interface variants of one task |
| Original exact-copy task, native input | 0/8 exact | Native semantic outcomes remain observations, not opt-in gates |
| Held-out opt-in strings | 10/12 | Exact copying remains imperfect outside the original task |
| Four concurrent opt-in requests | 3/4 | Distinct responses and overlapping HTTP requests; one wrong value |
| Original cache checks | 6/6 | Input accounting/bounds passed; only the 256-token common prefix was reused across policies |

Three failures were substantive: one non-streaming Unicode case exhausted its
256-token limit, one streaming earlier-user-message case stopped before a
required tool call, and one completed concurrent call omitted the requested
`<|endoftext|>` text. Two additional checks had an overstrict assertion in the
client: double-invalid continuation requests correctly returned HTTP 400 from
the existing forced-tool continuation guard, but the client also demanded the
new literal-option name in the error. Those two original failures are not
relabeled as passes. The original raw records, input IDs, and assessor remain
unchanged.

An earlier zero-request attempt stopped in preflight because it rejected the
stock Ubuntu `sitecustomize` module. The next harness admitted only the exact
packaged apport module after its resolved path, SHA256, and dpkg conffile MD5
were verified. No diagnostic observer or user customization was admitted to
the broad live run. Both attempts are retained in the
[literal-feature qualification archive](bench/results/2026-10-09/literal-feature-validation/README.md).

### Separate 16-request diagnostic

A fresh, source-identical diagnostic made eight additional coverage requests,
then the eight unchanged generation requests needed to investigate the three
substantive failures. It instrumented existing native sample/decoder and
reasoning-phase operations without changing input encoding, generated tokens,
request payloads, or output strings. Capture integrity passed for all eight
observed generation requests, with zero observer lookup/observation errors.
This capture result is separate from model semantic success, which was 6/8.

The new coverage checks passed 8/8. Two requests used automatic tool choice to
reach the literal-specific continuation rejection guard. Six cache requests
moved the changed marker beyond the first recurrent checkpoint. Both native
and opt-in starting sequences showed **0 → 2,048 → 4,096** actual cached tokens
for initial, cross-policy, and same-policy repeat requests. Cross-policy token
LCP bounds were 2,323 and 2,324; same-policy bounds were 4,118 and 4,115. Thus
reuse beyond the common template prefix was observed without exceeding the
verified matching token prefixes. Native wrong-value observations remain
separate from these cache gates.

Both Unicode requests produced exact tool values in this run. This does not
turn the earlier length failure into a pass or identify its original cause.
The earlier-user streaming failure reproduced: the native engine processed
`<|im_end|>` at sample 24 while the model was quoting the prior user text, before
a reasoning/content transition or required tool call. No Unicode-recovery
decoder call occurred for that case. The concurrent omission also reproduced:
the model's actual processed output IDs did not contain the requested final
marker, and there were no decoder calls that could have removed it. The raw
engine output matched the backend capture for all eight cases.

These findings identify a premature-stop handling problem and a separate
model value-selection failure. They do not justify repairing tool arguments
by guessing the desired text. The diagnostic sources and raw captures are in
the [16-request archive](bench/results/2026-10-09/literal-feature-diagnostic/README.md).
The literal-input feature remains an experimental branch. It is not included
in the standalone production candidate described next, and the stable recipe
does not advertise this request option as an exact-copy solution.

## Mandatory-tool stop handling

The raw failure motivated a separate server fix at
`fd8cbeb1fbb3cec4d2141558d5cfd63a500d50c4`, based directly on the existing
`f650bb5` main. It has no dependency on the experimental input option. For
required/named choices that begin in reasoning and have the supported producer
handoff, it suppresses implicit model EOS in a separate reasoning sampler.
The existing synchronous phase callback restores the original content sampler
when reasoning closes, before the next content token is sampled. These forced
choices already treat tool examples inside reasoning as reasoning text.

Explicit caller stop IDs and overlapping stop strings keep priority. Existing
reasoning overrides, token bans, automatic tool selection, unsupported handoff
paths, and the overall token limit retain their behavior. The backend now
copies the stop-condition list before adding implicit model EOS; it no longer
mutates the request's original stops and accidentally reclassifies them on a
later choice or reuse.

The standalone candidate passed **409 tests and 5,199 subtests on Spark with
zero skips**. The combined experimental input/EOS build, `f4aadf1`, passed
430 tests and 5,250 subtests with zero skips. Both passed the source-build
fingerprint and setup/dependency checks. A separate engine-source proof passed
eight CPU tests using actual Job/MTP acceptance and unfused sampler code,
covering rejected EOS draft proposals, phase restoration, explicit stops,
limits, and request isolation. It did not execute fused CUDA kernels. These
CPU selections overlap and should not be summed as independent tests.

### Instrumented generation check

The combined build reran the same eight generation payloads and full token-ID
inputs after the source fix. Its earlier coverage/cache requests were omitted,
so cache history differs from the prior 16-request run. It is not a perfectly
isolated numerical A/B comparison. The observed behavior is consistent with
the source and CPU proof; the raw hooks did not directly serialize sampler-ban
contents, so their activation is established through the exact source path and
prepared request/phase state rather than a raw mask dump.

All eight requests returned structurally complete calls and normal terminations.
Both earlier-user-message requests emitted `<|im_start|>` at processed sample
24, reached the budget handoff at accepted count 25, activated the content
grammar, and stopped normally after a complete call at position 56. The prior
premature-EOS failure was absent. The raw capture contained 481 processed
samples and 15 existing decoder calls, with zero observation errors and exact
native/backend text agreement for all eight requests.

**Only five of the eight argument values were exact.** Both earlier-message
calls returned the native wrong value `Remembered <tool_call>user`; the
concurrent ASCII case still omitted `<|endoftext|>`. Both Unicode cases passed.
The fix enforces the required-call stopping contract; it does not establish
semantic recovery or repair tool arguments. These results reinforce keeping
the broader literal-input experiment out of the stable deployment.

The [CPU/source archive](bench/results/2026-10-09/mandatory-tool-eos/README.md)
records the separate standalone and combined identities; the
[instrumented generation archive](bench/results/2026-10-09/literal-feature-eos-generation/README.md)
contains the raw eight-request evidence and both independent analyses. The
live cohort uses a finite reasoning budget of 24 for its six thinking requests;
natural-only handoff and explicit caller stops have separate CPU/source coverage.

### Standalone production-candidate tool gate

The EOS-only build then passed the unchanged **28/28 standard tool checks over
32 actual chat POSTs**. This selected-profile 3.05 run used disk N-gram tables,
four jobs, a 262,144-token shared pool, chunk size 2,048, MTP depth five,
confidence 0.6, and row budget eight. It contained no diagnostic observer or
experimental literal-input field. The source/schema, model/configuration,
package set, and GPU-lock ownership were checked. The server stopped with
SIGTERM/exit 0; the owned processes were gone, the GPU was free, and the same
lock inode could be reacquired. See the
[standalone tool archive](bench/results/2026-10-09/eos-default-tool-validation/README.md).

[TabbyAPI PR #3](https://github.com/vcruz305/tabbyAPI/pull/3) merged at
08:24:22 UTC as `f391c96beea0bcd21cbae3c929f75bb98136ff32`. Its complete tree
is identical to qualified candidate `fd8cbeb1`. The canonical Spark checkout
was advanced to that merged main revision. The input experiment remains on its
separate branches and is absent from this stable server's request schema.

## Cooperative GPU ownership

`GPU_LOCK_FILE=/absolute/shared.lock` adds a process-lifetime flock to the
ordinary launcher. It acquires file descriptor 8 before CUDA/runtime/model
work, preserves the existing file contents, and carries the lock through the
final server exec. The existing per-state-directory lock remains independent.
Preview mode neither opens nor acquires the optional GPU lock.

All cooperating supervisors must use the same persistent regular file. Do not
unlink or replace it while a participant may hold it. A parent controller
already holding the lock must leave the child's option unset; a controller
that delegates ownership to the server must not also hold it itself. This is
cooperation among participating processes, not enforcement against unrelated
programs that ignore the lock.

The benchmark matrix now records the requested lock setting, rejects an older
launcher without this capability, and verifies the server's actual descriptor,
device/inode, and exclusive Linux flock before and after client execution.
Tests cover contention, refusal before runtime verification, inherited lock
ownership, replaced paths, incorrect inodes, unlocked descriptors, and release
on exit.

The recipe CPU selection ran 148 tests successfully on WSL with nine optional
skips. On the Spark client environment it ran 148 successfully with five
optional real-template integration skips; the SDK fixtures were included. A live selected-profile
3.05 server at recipe `254b2b0`, engine `24f0dec`, and Tabby `f650bb5` passed all
28 standard tool checks, making 32 chat POSTs. Its FD8 lock was verified before
and after the clients. It stopped with SIGTERM/exit 0, and the same unchanged
lock inode could be reacquired after the owned server exited.

This is an operational ownership and tool-regression gate; it is not another
throughput or simultaneous-inference measurement. See the
[GPU-lock validation archive](bench/results/2026-10-09/gpu-lock-validation/README.md)
and [service documentation](docs/service.md#shared-gpu-ownership).

## Interrupted benchmark cleanup

The public matrix controller now records TERM/INT signals in a small handler
and raises interruption at explicit checkpoints after child registration.
Cleanup runs without those checkpoints, so a signal arriving between process
creation and registration, during finalization, or repeatedly during shutdown
cannot bypass owned-process cleanup. It does not block signals in child
processes or affect unrelated process groups.

The prior controller's finalization failure was reproduced with a real CPU
child and SIGTERM; its owned cleanup was skipped. Seven focused process tests
now exercise launch handoff, final polling, repeated TERM/INT, and an unrelated
process group. Together with the existing 26 matrix tests, all 33 focused
checks passed. The integrated recipe selection ran 155 tests successfully on
WSL with nine optional skips. The final canonical Spark recipe then ran the
same 155-test selection successfully with five optional real-template skips;
its source-build and dependency checks also passed. These are CPU/process
tests, not GPU or API requests. See the [cleanup archive](bench/results/2026-10-09/public-matrix-signal-cleanup/README.md).

## Publication and final serving state

The canonical recipe was advanced from `a8c72bd` to the qualified operational
code at `22d673463dba8e1f3a862056348769ab85a3c032`; the canonical Tabby checkout
was advanced from `f650bb5` to merged main `f391c96b`. Both checkouts were clean.
Engine `24f0dec` remained unchanged. The actual Git histories confirm the
checked official ExLlamaV3 and Tabby heads are ancestors of these deployed
forks. Recipe PR [#18](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe/pull/18)
adds the final report and evidence to the qualified operational code; those
publication-only additions do not change the tested launchers or benchmark
controller.

The existing private service environment was preserved byte for byte as a
prefix, with only its shared `GPU_LOCK_FILE` assignment appended. A private
byte-identical backup was verified. Unit enablement, model selection, tuning,
and cache settings were retained. The service uses **262,144 tokens per
request and a configured 1,048,576-token shared pool**, unlike the smaller
262,144 pool in the isolated live qualification. This final smoke test is not
a full-capacity or four-full-context admission test.

The actual systemd invocation `55b62ef9cff044bdba68d25f61213408` passed two raw
stream/non-stream transport requests and all **five OpenAI SDK 3.26.0 checks**:
plain streaming, named tools in both modes, and tool-result round trips using
the SDK-assembled assistant messages. These are seven actual chat requests.
Source, environment, model, invocation/cgroup/listener, and FD8 lock ownership
were checked; the same MainPID and invocation remained active with zero
restarts. The unit then stopped successfully and released the unchanged shared
lock. The Qwen unit remains disabled/inactive as the updated standby service.

The existing REXL3 manager was restarted and its previous `CyberFrost3.8`
selection was restored. At **08:42:37 UTC**, the manager reported that exact
request ready with no error, its direct backend advertised the same model ID,
and the active, prefill, and queue metrics were all zero. The manager was
enabled and running, held the original shared-lock inode, and blocked an
independent exclusive-lock attempt. Its private configuration hash was
unchanged. Qwen remained disabled/inactive; the canonical Qwen recipe, engine,
and Tabby checkouts were clean at the qualified revisions above.

The [final service and handback archive](bench/results/2026-10-09/final-standby-unit-validation/README.md)
contains the canonical CPU/setup checks, hash-only service-environment change
receipt, seven-request unit result, owned shutdown, and terminal restoration
proof. The restoration verifier made read-only health/model/metrics requests;
it did not generate additional model output. No private service environment,
key, backup bytes, or full authentication logs are published.

## Evidence conventions

The [October 9 evidence tree](bench/results/2026-10-09/README.md) contains the
completed controller/client sources, exact payload/input bindings, raw JSON
and JSONL records, measured results, source/package/placement snapshots,
cleanup records, and independent release reviews. Each collection includes
SHA256 manifests and a receipt distinguishing copied bytes from generated
summaries. Model symlinks and Python bytecode are omitted from the published
collections and their omissions are recorded.

Credential checks in the release reviews are bounded pattern scans of the
staged artifacts, with the patterns and scan scope recorded. They are not a
general proof that arbitrary data contains no secrets. Private authentication
material and unrelated request histories are not part of this report.

October 8 raw results and conclusions remain in their original files. New
native-policy observations, opt-in feature gates, expected validation errors,
and preparation/operational failures have separate denominators. No failed
semantic response is counted as a successful throughput sample, and no
zero-request attempt is added to a model-quality denominator.
