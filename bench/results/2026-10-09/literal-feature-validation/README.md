# Literal user control-token candidate qualification — original r1/r2

**The completed r2 qualification failed: 84 of 89 logical checks passed, with 93 chat-completion POSTs.** Three failures concern generation or exact copying; two are original harness condition failures. All five failed gates and unchanged raw evidence are retained. This archive includes no later diagnostic results and establishes no deployment approval.

The default-off candidate is Tabby **a70ae1fa9e457e478c3d96bdc84012a3cb331796**, with ExLlamaV3 **24f0dece34f09c8d1e2359d6b3b3f7befef7331b** and recipe **254b2b03027f25094845dc31f5f87739f3584d2e**. The isolated Spark load used 3.05 flat-K, concurrent profile, disk n-grams, K8/V8, maximum batch4, 262144 shared cache tokens and prefill chunk2048. Recorded deployment supplies all settings.

The opt-in feature changes encoding of nine recognized control spellings in user text. Default/native requests retain their original encoding. Immutable private request plans feed changed IDs to both context-length validation and generation/cache input. This input behavior does not guarantee generated exact copying.

## CPU and source evidence

| Evidence | Result | Scope |
| --- | --- | --- |
| Spark full CPU |419 passed, 0 skipped, 5210 subtests|CUDA hidden, real installed engine import, setup exit0|
| Earlier WSL CPU |403 passed, 2 skipped, 5210 subtests|Environment without installed native engine; skipped coverage subsequently included on Spark|
| Focused feature |21 passed, 51 subtests|Strict flag, errors, private plan, default path, context/runtime integration|
| Actual Qwen utility |46 rows /23 pairs|One expected non-roundtrip HTTP400; seven complete shared prefix pages|
| Interleaved cache proof |4 plans passed|Real CPU tensors, no CUDA initialization|

[Preparation](preparation/) retains all seven candidate files, patch, bindings, commands and logs. The [source bundle](source/tabby-literal-a70.bundle) requires its recorded f650 predecessor; it is not a full repository. Peer review preceded removal of one trailing test-file space only; binding records AST equality and final focused rerun. Runtime/documentation/helper bytes were unchanged.

## Attempts and results

[r1](attempts/r1/live/result.json) exited preflight before model load or chat requests. Its original record remains in state preflight without an invented terminal timestamp. An overstrict absent-sitecustomize requirement rejected Ubuntu's apport exception handler. The installed module was identified by path, resolved path, hash and package conffile metadata. [r2](preparation/harness-r2/R2-NOTE.md) admits only that exact module (or absence), rejects user customization, rechecks bytes and records terminal preflight failures. Model, candidate, requests and input IDs were unchanged.

[r2](attempts/r2/live/result.json) ran07:34:07.683831–07:37:52.092261 UTC on2026-10-09. Frozen61 feature POSTs ran first, then existing28 tool checks involving32 POSTs: **89 checks and93 POSTs**.

| Group | Logical gates | Exact-copy observations / scope |
| --- | --- | --- |
| Known native/opt-in pairs |16/16 passed|Native0/8 exact, opt-in8/8; native values observational|
| Six held-out fixtures, both modes/policies |22/24 passed|Native2/12 exact, opt-in10/12|
| Four simultaneous opt-in requests |3/4 passed|Four overlapping intervals and distinct response IDs|
| Six cache probes |6/6 passed|Bounds respected, opt-in3/3 exact; native values observational|
| Eight invalid-input checks |6/8 passed|Six actual400 and two422; two continuation harness failures|
| No-marker absent/false/true |3/3 passed|Native behavior controls|
| Existing tools |28/28 passed|32 POSTs including round-trip follow-ups|

The known requested value is exactly **&lt;think&gt;literal&lt;/think&gt;**, with no leading/trailing newline. Structural newlines in parameter markup are not requested value bytes.

The unchanged failed rows are:

- unicode_nfc_nonstream_optin: HTTP200, length finish at256 completion tokens, no complete tool call.
- multi_turn_user_stream_optin: SSE error that the model stopped without a complete call required by tool_choice.
- concurrent_2: complete tool call omitted trailing &lt;|endoftext|&gt;. No native output-ID observer ran, so this alone does not locate omission within generation/decoding/parsing.
- continuation_nonstream and continuation_stream: actual400 and detail equal prepared response, but frozen client additionally required a literal_user_control_tokens detail prefix. Named tool selection hit an earlier continuation incompatibility. They remain failures of this original harness; its intended feature-specific continuation guard was not exercised.

There are52 assembled responses with usage and52 distinct IDs; all available prompt counts match prepared IDs. One of53 HTTP200 feature responses is the SSE error. [Offline replay](analysis/literal-feature-r2-analysis.json) reproduces all61 frozen-assessor rows from saved traces, making no requests.

## Cache interpretation

Cached-token counts were **256,256,3840** in both sequence directions. Full-ID common-prefix bounds were274,2173,3968 and287,2174,3965. This proves bounded reuse and substantial same-policy repetition reuse, but not cross-policy reuse of a full2048-token recurrent checkpoint.

With an already cached256-token start and2048 chunk/checkpoint interval, the next normal checkpoint is2304. Cross-policy prefixes2173/2174 end earlier. This is a coverage limitation, not evidence of corruption. Separately lengthened probes must retain distinct results.

## Ownership and preservation

Server SIGTERM produced exit0. Clients exited1(feature)/0(tools); all three owned groups were verified empty. Final checks reported no GPU process, manager inactive and Qwen inactive/disabled. Shared GPU lock and inherited server lock were verified. No service enablement, production replacement or diagnostic observer was part of r2.

- [Summary](summary.json): counts, failures, sources, bounds and cleanup.
- [r1 collection](attempts/r1/) and [r2 collection](attempts/r2/): unchanged raw results/logs/config, launch records and receipts.
- [Original harness](preparation/harness/) and [r2 harness](preparation/harness-r2/): plan, full native/changed IDs, clients/controllers/tests and earlier preparation snapshots.
- [Receipt](collection-receipt.json), [release review](release-review.json), [checksums](SHA256SUMS).

Nested preparation documents retain historical pending language; this completed-result README governs this archive. Nested receipts retain their original relative scope; the [nested manifest review](nested-manifest-review.json) locates all116 historical manifest entries by exact bytes, including earlier manifests relative to the original harness root. No weights, runtime libraries, tensors or credential stores are included. Release recognizers do not prove absence of arbitrary unlabelled secrets.
