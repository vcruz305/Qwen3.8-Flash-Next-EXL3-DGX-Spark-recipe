# Final literal-control-marker validation

The current captured failures are already present in the native generated tool argument. The API preserves those argument strings. This is a bounded result for the synthetic exact-copy requests below; it does not establish why the model selected those tokens or explain older runs without raw captures.

All eight captured native completions equal their accumulated backend text. All128 CPU replays through the actual final5a collector reproduce the API exactly: eight requests × native/backend input × four chunk partitions × streaming/nonstreaming collection. Independent raw XML span review finds seven incorrect strings already inside the native parameter body and one exact requested string. The API returns the same value in every case. [Raw/parser review](corrected-diagnostic/raw-parser-review.json)

A separate four-case client, derived from the original unbudgeted requests by changing only enable_thinking to false, passed all four exact-copy checks. It retained the original user message, including “Think briefly,” and the exact requested string. These results remain separate from the failed reasoning-enabled checks. [Thinking-off report](first-diagnostic/live-attempt1/literal-thinking-off.json)

## What was requested

The synthetic function record_text accepts one string named text. The request asks it to preserve exactly:

    <think>literal</think>

The original reasoning-enabled cases cover required and named tool_choice, each in stream and nonstream mode. The budgeted group sets reasoning_budget_tokens to24; the unbudgeted group omits it. Both use temperature0, top_k1, max_tokens256 and parallel_tool_calls=false. Returned tools are never executed by these clients. [Unchanged client](scripts/reasoning_literal_smoke.py)

The server and original request bytes are unchanged between diagnostic attempts. The model is the existing flat3.05 EXL3 pack, using the final single-request RAM profile with2048-token chunks and the verified compatibility settings. Engine24f0dece and Tabby5a4f3efa are the exact observed source commits. [Deployment](corrected-diagnostic/live-capture/deployment.json)

## Outcomes, retained without relabeling

| Evidence group | Budgeted original | Unbudgeted original | Separate thinking-off cases | Raw capture status |
| --- | ---: | ---: | ---: | --- |
| Original final3.05 single job | 0/4 | 0/4 | Not run in that job | No raw observer |
| First diagnostic | 0/4 | 1/4 | 4/4 | Original assessor failed; zero matched records |
| Corrected diagnostic | 0/4 | 1/4 | Not rerun | Eight records retained; original assessor failed on scalar metadata, with separate offline reconciliation |

The original final reports are copied under [input-triage/original-final-305-single](input-triage/original-final-305-single). Both diagnostic assessments are preserved byte-for-byte: [first](first-diagnostic/live-attempt1/literal-capture.json) and [corrected](corrected-diagnostic/live-capture/literal-capture.json). Neither is rewritten to say capture_valid=true.

The broader final API matrix, including its separate concurrent literal failures and its total306/322 results, is reported in the parent validation report. This archive does not add its diagnostic repeats to that primary total.

## Input formatting and encoding are exact

The eight original payload combinations pass through the actual5a Pydantic request model and actual downloaded3.05 chat template without changing the user string. All render the same348-token prompt with SHA25632655bc461bfa9685942882754b89e75f6640a5605004d4a3d609ebfc6076f58, matching the live token count. The actual tokenizer decodes it exactly.

Independent execution of the relevant original94 and final24f0 engine tokenizer methods yields the same348 token IDs. The literal alone encodes as248068,34600,248069, representing the two native reasoning markers around “literal.” These checks rule out suspected input punctuation stripping in the checked formatter and encoder path. They do not imply the model will copy native control-token text correctly. [Render replay](input-triage/render-replay.json) · [Engine tokenizer proof](input-triage/engine-tokenizer-proof.json)

## Why the capture assessments failed

The first observer incorrectly assumed top_p1.0, inherited from the earlier strings fixture, which explicitly supplied that value. This literal fixture omits top_p. The existing recipe default is0.95, so the strict matcher rejected every request. The client results still constitute completed live results; no native output was captured in that first attempt.

The separately named second observer changes only the diagnostic expected default to0.95. All eight requests match and produce full raw records. Its final assessment then rejects the recorded top_p=null field. The source explanation is specific: the real YAML loader preserves the default as a ruamel ScalarFloat(0.95). It passes the numeric matching predicate, but the observer’s exact-type scalar recorder includes only built-in float, so it records null. This affects diagnostic metadata, not the request value or generated text.

The independent proof uses the retained live YAML through actual TabbyConfigModel, global sampling overrides, Pydantic and formatting for all eight variants. It reproduces both successful matching and the null recorded field. The original assessments remain false; a separate sidecar verifies observer/configuration/deployment/source hashes and lifecycle, and records this qualification explicitly. [YAML scalar proof](peer-review/yaml-scalar-proof.json) · [Offline metadata review](corrected-diagnostic/offline-metadata-review.json)

The corrected source differs only in the diagnostic matcher/assessment expectation and corresponding observer hash. Production code, prompts and original clients are unchanged. [Correction manifest](corrected-diagnostic/correction-manifest.json)

## Raw result attribution

In requests00–02, the native parameter body is already “thinkaliteralthink.” Request04 contains the exact requested string. The other four native bodies contain additional punctuation or call-looking markup. Some contain another opening function/parameter tag inside the outer parameter body; that text is preserved as string data by the Qwen XML parser. There is one closing parameter delimiter in each captured output. Independent review of the first parameter’s physical text span, removing only one framing LF at either end, matches the API value in every case.

There are no collector exceptions or rejections in these eight captures. They return completed tool calls with one string argument. The seven failing exact-copy values are present before Tabby’s parsing or SSE assembly; this particular failure is not caused by those steps. The128 replays separately show that fragmentation and streaming mode do not change these captured outcomes. [All retained native records](corrected-diagnostic/live-capture/raw-literal) · [Executed review script](scripts/review_observed_literal_raw.py)

These findings apply to the captured requests. The earlier uncaptured failures are retained as failures without a retrospective causal claim.

## Practical existing-setting workaround

For this kind of exact literal-marker copying request, the separately tested setting is:

    "enable_thinking": false

Only that flag differs from the original unbudgeted four requests. The original message, tool schema, choices and expected argument remain unchanged. The supplemental validator also requires the reasoning field to stay empty. Its four passes demonstrate a practical workaround for this measured task; they do not certify every possible tool value or change the result of the original reasoning-enabled tests. [Supplemental client](scripts/literal_thinking_off_smoke.py)

No production patch, parser coercion, prompt-specific answer rule or default change was made to mask these failures.

## Reproducibility and ownership

The archive retains original reports, raw native/backend text, controller and observer source snapshots, exact source and configuration hashes,14 controller CPU checks, actual-formatting observer checks, the source/type reconciliation and the128 actual5a parser replays. The replay script is CPU-only and makes no API call.

Both diagnostic servers were launched only after the measured final five-job batch had completed and exited, with prior owned cleanup, a free port8899 and no GPU compute owner independently verified. Each diagnostic later exited via its own verified server group after SIGTERM, with exit0 and no surviving owned group. The wrapper’s process ownership, setup/source verification and cleanup implementation were inherited from the frozen reviewed controllers. [First lifecycle](first-diagnostic/live-attempt1/result.json) · [Corrected lifecycle](corrected-diagnostic/live-capture/result.json)

manifest.json and SHA256SUMS cover every archived regular file. Source snapshots and raw results are historical evidence; absolute host paths in these scripts are intentional and are not a general installation interface.
