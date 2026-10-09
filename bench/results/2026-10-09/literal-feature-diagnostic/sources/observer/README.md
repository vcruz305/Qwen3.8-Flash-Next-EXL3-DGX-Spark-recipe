# Bounded a70 generation diagnostic

This bundle observes eight unchanged synthetic requests from the completed a70 feature qualification: the Unicode and earlier-user-turn stream/nonstream pairs, followed by the original four concurrent requests together. Their wire payloads, actual rendered prompts and immutable native/expanded token-plan hashes are bound by `observer/expected_requests.json`. The production a70/24 sources and model inputs are unchanged. There is no input-tokenization hook, output rewriting, new generation await, or added model/GPU operation.

The parent controller sends a separately bound eight-request coverage suite first, then invokes:

```bash
python generation_client.py --metadata /absolute/deployment.json --output /fresh/generation.json --timeout 120
```

The generation client records complete collection separately from string-copy semantics. Explicit SSE model errors and length finishes remain observations; transport/incomplete response/usage-provenance errors fail collection. It uses the unchanged reviewed stdlib HTTP implementation. It does not retry or execute tools. Four overlapping HTTP requests do not prove a particular CUDA kernel batch width.

Only the server receives `PYTHONPATH=<this bundle>/observer` and `TABBY_STRINGS_OBSERVER_CONFIG=<exact config.json>`. The config supplies exact `engine_repo`, `tabby_repo`, their 24f0/a70 commits, `output_dir`, `run_label`, and `max_records: 8`. Startup verifies source/import paths and installs hooks after normal configuration and allocator setup. Other requests, including the preceding coverage suite, are not observed. The original collector/finish and producer phase/force hooks retain call ordering and returned objects. The new native hook observes existing `receive_sample` outputs and decoder calls only for matched active jobs; it never moves a tensor to CPU or initiates decoding.

Raw records include rendered input and immutable plan metadata; native full completion and accumulated backend text; phase, guard, force and filter state; processed native IDs, held UTF text/CPU tokens, and existing Unicode-recovery decoder results. **Processed IDs are not automatically final accepted output**: EOS, healing and rewind metadata must be interpreted. Event/text limits and observation errors are explicit and fail the integrity assessment rather than silently truncating evidence. `observer-completion.json` holds the final cumulative native-hook audit.

After owned server cleanup, the controller calls `assess_capture(raw_dir, generation_report, prepared_inputs)` from `assess_capture.py`. It verifies all eight exact request identities, raw/API response IDs, source hashes, input plans/counts, native and available API usage, ordered and complete sample/decoder/phase events, and zero drops or observation errors. Model semantic failures and native/backend text differences remain reported observations and are not relabeled as capture failures or successes.

CPU evidence includes eight actual a70 renderer/native-encoder cases plus48 negative scope guards; seven retained-HTTP/capture tests; and seven native-hook tests covering10 real source-executed Job/SeqTensor/MTP scenarios. The native tests also demonstrate the existing decoder's pad-token removal path without adding a decode call. The retained prior r2 traces are synthetic fixtures, not new live measurements. These counts describe different checks and must not be combined into an inferred model success rate.

The matrix controller is separate and owns source/deployment checks, shared GPU ownership, timeout and process cleanup. This bundle never starts or stops a server itself. `manifest.json` binds every included regular file except itself and bytecode caches; source/runtime files must remain unchanged during the scheduled diagnostic.
