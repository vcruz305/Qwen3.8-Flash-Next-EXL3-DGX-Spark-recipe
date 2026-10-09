# Generation replay with the mandatory-EOS fix

**All eight raw captures are valid and all eight API calls completed with `tool_calls`. Five of eight exact-string checks passed.** The two multi-turn calls completed with an incorrect argument, and concurrent_2 still omitted the requested trailing endoftext marker. These failures remain recorded; the experimental literal-input feature is not qualified by this run.

The fresh isolated run completed at 2026-10-09 08:17:41.742352 UTC using combined Tabby f4aadf (a70 literal-input candidate plus the independent mandatory-EOS fix), engine 24f0 and recipe 254. It issued exactly the original Unicode pair, multi-turn pair and four concurrent requests. Actual candidate CPU preparation proves every visible payload, rendered prompt, full original/expanded ID list, context count, replacement count and effective sampling field matches the preceding a70 generation eight. No continuation/cache probes or standard tool cases were repeated here.

Unlike the preceding [combined16 diagnostic](../literal-feature-diagnostic/README.md), this server did not first receive coverage eight. That cache-history difference prevents treating output changes as an isolated numerical A/B. Source/CPU/native evidence for the EOS mechanism is in [mandatory-tool-eos](../mandatory-tool-eos/README.md); unchanged EOS-only default-tool regression is a separate gate. These artifacts do not claim every semantic failure was an EOS defect.

| Request | Actual prompt tokens | Cached tokens | Exact argument |
| --- | ---: | ---: | --- |
| unicode_nfc_nonstream_optin | 356 | 0 | Pass |
| unicode_nfc_stream_optin | 356 | 256 | Pass |
| multi_turn_user_nonstream_optin | 407 | 256 | Fail |
| multi_turn_user_stream_optin | 407 | 256 | Fail |
| concurrent_0 | 325 | 0 | Pass |
| concurrent_1 | 363 | 256 | Pass |
| concurrent_2 | 332 | 0 | Fail |
| concurrent_3 | 361 | 256 | Pass |

Both multi-turn modes now returned a complete call with the value `Remembered <tool_call>user`, which differs from the requested original string. Concurrent_2 again omitted the trailing endoftext marker. The Unicode pair passed, as in the preceding raw diagnostic; this does not erase or explain the earlier qualification's length failure. All four concurrent HTTP intervals overlap and response IDs are distinct.

Independent native review additionally matches physical parameter bodies to API values in all eight cases. The six initial-reasoning cases show budget injection after 24 tokens and phase attachment at 25; the two thinking-off requests have no phase/force events. Each request samples only its final im_end, with no observed rewind. All six reasoning requests here use a finite budget of 24; this eight-case cohort does not add live coverage of explicit caller stops or the natural-only watcher. Those remain source/CPU coverage. The observer does not serialize sampler masks: policy activation is a source-supported inference combined with separate native CPU proof, not a direct mask measurement.

The source-bound observer records native/backend text equality 8/8, 481 processed-sample observations, 15 decoder calls and zero lookup/observation errors. The unchanged assessor exactly reproduces the saved capture result offline. Processed IDs retain EOS, healing and rewind state; they are not automatically emitted tokens. The observer reads existing CPU values and held text, without GPU tensor/KV reads or input/output rewriting. No uninstrumented speed claim is made.

The model/profile remains 3.05 disk PLE, four slots, Q8 context/pool 262144, chunk 2048, dynamic MTP depth 5/confidence 0.6, row budget 8. The server alone holds the shared flock. It received SIGTERM and exited 0; both owned process groups were empty. Root's 08:19:34 release receipt confirms all three retained PIDs absent, GPU/port free and the same lock inode reacquired. REX manager remained enabled/inactive and Qwen disabled/inactive. Canonical source and service policy were not changed by this diagnostic.

reports/ retains exact raw/SSE/observer/deployment/resource evidence. sources/ retains the frozen executable/input/observer bundles; nested preparation documents remain historical. analysis/ contains the exact frozen-assessor replay and input/lifecycle peer proof. collection/ binds remote hashes, file-only staging, root launch/release and the zero-candidate bounded scan. No model files, extensions, tensors, credentials or followed model symlinks are included. The bounded recognizer cannot exclude arbitrary unlabeled secrets. Do not import archived historical helpers merely to inspect them; some are standalone execute-on-import scripts.
