# Continuation, cache and raw-generation follow-up

**The eight coverage checks passed, and all eight raw generation captures are valid. Exact generation passed six of eight requests.** The original [failed feature qualification](../literal-feature-validation/README.md) remains unchanged: this separate diagnostic does not convert its failed gates into passes.

The run completed on 9 October 2026 at 07:59:11.839913 UTC, using the same isolated Tabby a70, engine 24f0 and recipe 254 sources. The model/profile remains 3.05 flat-K, disk PLE, four request slots, Q8 cache/context 262144, prefill chunk 2048, dynamic MTP depth 5/confidence 0.6 and draft-row budget 8. Exact source commits, deployment/config values, source and input hashes are retained in the reports.

## Coverage checks

The first two requests change only named tool choice to auto in the original continuation fixtures. Both returned JSON HTTP 400 before SSE, with the exact actual-source-prepared message: `literal_user_control_tokens: Literal user control tokens do not support continued messages`. This reaches the intended feature-specific rejection without the earlier named-tool continuation incompatibility. The original two failed harness records are not edited or reassessed.

The six cache probes change only the pre-marker filler from 125 to 135 repetitions; their requested values, schemas, sampling and policy/stream sequence remain unchanged. CPU preparation records complete native/expanded ID arrays and exact prefix bounds. Native engine source proves that a cold start can checkpoint at 2048 and an already reused 256-token prefix can checkpoint at 2304. The new cross-policy LCPs 2323/2324 cover both.

| Request | Actual prompt tokens | Cached tokens | Maximum prior input-ID LCP |
| --- | ---: | ---: | ---: |
| coverage_cache_native_first_0 | 4114 | 0 | 0 |
| coverage_cache_native_first_1 | 4118 | 2048 | 2323 |
| coverage_cache_native_first_2 | 4118 | 4096 | 4118 |
| coverage_cache_optin_first_0 | 4121 | 0 | 287 |
| coverage_cache_optin_first_1 | 4115 | 2048 | 2324 |
| coverage_cache_optin_first_2 | 4115 | 4096 | 4115 |

Both directions started cold in this fresh run, then reused 2048 tokens across the policy change and 4096 tokens on the same-policy repeat. Every reuse stayed within its exact ID-prefix bound. All opt-in cache arguments were exact. Two native cache responses were wrong under their predeclared observation policy; that is separate from cache integrity. The original shorter probes reused only 256 across policies and remain a separate limited result.

## Unchanged generation observations

After coverage, the server received the original Unicode pair, earlier-user-turn pair and four distinct concurrent requests, with unchanged payloads and full input IDs. Both Unicode requests passed in this run; the previous nonstreaming length failure is still preserved and is not explained or fixed by this observation.

The multiturn streaming request again stopped without completing the required call, and concurrent_2 again omitted the trailing endoftext spelling. Thus exact generation was 6/8. The four concurrent request intervals overlap and their response IDs are distinct. No semantics were retried or normalized.

The bounded observer captured all eight requests, with native/backend output text equality 8/8, 470 processed-sample observations, 15 native decoder calls and zero lookup/observation errors. Frozen capture assessment reproduces the stored result exactly in an offline replay. Raw processed IDs retain EOS, healing and rewind state; they must not be treated automatically as final accepted output. These records support a focused downstream investigation, rather than a broad claim that the input-token feature guarantees literal copying.

## Lifecycle and archive

The server received SIGTERM and exited 0. The coverage client, generation client and server groups were empty after cleanup. Root's independent release receipt confirms all four retained PIDs, including the controller, were absent; GPU and port were free; the same cooperative lock inode was reacquired with unchanged contents. REX manager remained enabled/inactive and Qwen remained disabled/inactive. No service enablement or production replacement occurred.

- reports/: exact completed raw responses, SSE frames, observer data, deployment, resources and controller results.
- sources/harness/ and sources/observer/: exact frozen executables, plans, full IDs and CPU evidence.
- analysis/offline-capture-replay.json: independent frozen-assessor replay with no requests.
- collection/: remote file hashes, bounded raw release scan, launch/release and copy receipts.
- summary.json and SHA256SUMS: machine-readable scope/counts and archive hashes.

Nested preparation documents preserve their original pending language; this completed-result README governs the archive. The inherited observer module docstring predates its new native add-on: the actual reviewed scope includes existing CPU token/held-text observation, with no GPU tensor reads, KV access or input/output encoding changes. There is no uninstrumented speed claim.

Do not import archived historical controllers or the pattern-source review script just to inspect them; some historical helpers execute at import. The live controller uses reviewed source subsets and explicit hash-bound modules. The credential recognizer reported no candidates across 25 raw files, but does not prove absence of arbitrary unlabeled secrets. Model symlinks are recorded as exclusions without following them; no weights, tensor artifacts, libraries or credential stores are included.
