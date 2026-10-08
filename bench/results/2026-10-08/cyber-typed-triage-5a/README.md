# Historical Cyber row8 boolean triage

This archive separates an **experimental historical model response** from a later **CPU-only parser/template investigation**. It is not a final deployed model-run result, and it does not qualify Cyber row8 for serving.

## Historical model evidence

`historical-tools-16ca-f4.json` is the unchanged eight-check tool subset from `11-tune-cyber387-concurrent-rowbudget8`, attempt `1c1db66a44c1`. The runtime was engine `16ca20d27c0e4cce15a9bbc131e6d047065395b5` and Tabby `f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6`, with the Cyber pack's original template, which unconditionally enabled thinking. The experimental concurrent row budget was eight.

The subset passed **5 of 8** checks and failed three: the two existing exact-string cases and the streamed typed-argument case. The typed case passed in non-streaming mode. In streaming mode, `enabled` was returned as the JSON string `"True"`, although the request declared `type: boolean` and supplied `true`. The report preserves the request, response, real usage/timing, and failed status. These timings belong to the historical experiment; this archive introduces no performance comparison.

The API report does **not** contain the raw generated XML parameter. A raw parameter body `True` and a quoted JSON string body `"True"` can both produce the observed API string under the existing parser. This evidence cannot identify which body the model emitted.

## CPU investigation at exact Tabby 5a

`cpu-triage.json` records read-only execution of the actual parser, incremental delta streamer, and Jinja environment at Tabby `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb`. Its source-template paths and SHA-256 hashes are recorded in the report. No model, server, API request, GPU operation, or production code change was involved.

- **Eight template cases:** the original Cyber template and corrected external template each rendered boolean `True`/`False` as canonical lowercase JSON `true`/`false`. String `"True"`/`"False"` values stayed strings with their original capitalization.
- **32 type/raw-value cases:** boolean, nullable boolean, string, and string/boolean union schemas were evaluated against lowercase, capitalized, quoted, and non-keyword values.
- **64 streaming partitions:** one-character and seven-character chunking matched the authoritative completed-call parser for every input.

These are diagnostic fixture counts, not additional live tool successes or a second main test-suite total. The historical request was a single user turn without assistant tool history, so tool-history rendering cannot explain that particular generated value.

## Conclusion and next gate

No renderer capitalization defect or stream-only conversion discrepancy was demonstrated. The historical value remains an unresolved model-output/argument-semantic failure because the raw parameter text was not retained. The parser was not patched to guess a boolean from the saved API string.

The final single-Cyber typed gate uses the corrected external template and the separately frozen final engine/Tabby pair. Its outcome must be reported from that new run. If it fails, a bounded raw capture can distinguish a noncanonical bare boolean from a deliberately quoted value before any further compatibility change is proposed.

`SHA256SUMS` and `manifest.json` bind the copied evidence and this scope note. The original files remain intact.
