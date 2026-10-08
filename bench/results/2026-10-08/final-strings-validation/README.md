# Final literal-string validation: 4.05 and corrected Cyber

Both final diagnostic loads passed the **two unchanged exact-string requests**, one non-streaming and one streaming, at engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` and Tabby `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb`.

| Final model | Original live string cases | Independent raw parameter inspection | Actual 5a collector replay |
| --- | --- | --- | --- |
| Flat K 4.05 | 2/2 passed | All eight parameter bodies exact | 32/32 match the saved API |
| Cyber 3.87 with external thinking-template override | 2/2 passed | All eight parameter bodies exact | 32/32 match the saved API |

The two model loads used the single-request profile with disk PLE. Their complete source, loaded pack, environment, setup, template and request provenance is retained inside each capture directory. Normal owned cleanup and capture integrity were verified by the frozen controller. Exact settings are in the original deployment and lifecycle records.

## What was verified

The original request used automatic tool choice and `enable_thinking=false`. Every function property was a string: `number_text="123"`, `bool_text="true"`, `json_text="{\"nested\": [1, false]}"`, and `tag_text="<think>literal</think>"`. No fixture was changed or replaced by a forced-call variant.

Independent inspection found those exact values inside the native generated parameter bodies, bounded only by the template's single framing LF on each side. It did not call Tabby's parser to infer the raw values. The native full completion and accumulated backend text were byte-identical in each request. The literal `<think>` and `</think>` strings appeared inside the `tag_text` parameter as requested.

The frozen replay then fed each saved native and backend string through the actual 5a collector using one-character, seven-character, 31-character and whole-string partitions, in both collector response modes. All 32 comparisons per model reproduced the saved API content, reasoning channel, decoded argument values and finish reason. Fresh randomized call IDs were excluded from replay comparisons. This replay does not recreate the original SSE timing or claim an independent transport implementation.

The corrected Cyber deployment records template SHA256 `666b82b29f5801f4f546e5724b45bf5f14be7d20b66149df44164626b072ce6d`. Both captured Cyber requests asked for thinking to be disabled and actually began outside reasoning; the rendered prompts include the completed empty reasoning prefix. These two cases validate the disabled-thinking path. The full API suite separately tests enabled thinking.

## Evidence layout and scope

Each `strings-final-*-24f0-5a/` directory retains the original `tools.json`, `strings-capture.json`, `result.json`, `deployment.json`, setup status, raw observer manifest/traces, configuration and logs. All regular evidence files were copied verbatim. Runtime model symlinks and empty lock files are represented in `manifest.json` without following model pointers.

The sibling `*-raw-review.json` files record the direct raw-body inspection. `*-parser-replay.json` files record the actual collector comparisons, with their logs alongside them. `replay-source/` retains the frozen WSL diagnostic script, the exact engine-binding-only proof and the earlier peer-reviewed synthetic self-test result. That self-test correctly records its earlier script hash; the binding proof shows the sole later change was the expected engine constant. It is not another live model run.

The replay script is an archival diagnostic pinned to the original WSL checkout path and exact clean source head. The recorded execution used the Tabby CPU venv with `--capture CAPTURE_DIRECTORY --output NEW.json`. It sends no HTTP requests and loads no model or GPU tensors.

These are **four live string-case results**, **16 raw parameter-body checks**, and **64 CPU replay comparisons**, kept as separate counts. They are not a replacement for the final 28-case-per-pack tool suite, the SDK/resilience/automatic-choice suites, or the batch-four handoff checks. Instrumented observer loads must not be used as throughput measurements.

Earlier 4.05 and Cyber string failures remain historical evidence. They are not carried forward as limitations of these passing final requests. No retrospective causal attribution is made: the earlier runs did not retain equivalent native raw text, and more than one runtime factor changed before the final runs. The earlier experimental Cyber row8 typed failure is preserved in its separate historical triage archive.

`manifest.json` and `SHA256SUMS` bind every copied evidence file and this scope note.
