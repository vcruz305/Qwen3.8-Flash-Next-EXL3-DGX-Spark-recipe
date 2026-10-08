# Bounded literal-tool raw-output observer

This is a separate, opt-in diagnostic for the eight unchanged requests in `reasoning_literal_smoke.py`. It does not patch either production repository or alter the requests, sampling, grammar, model outputs or validators. Root owns any live launch, after the final API benchmark batch has finished.

The observer is derived from the reviewed `77d606e2` strings observer. Its module remains `strings_observer.py`; the original `sitecustomize.py` and `TABBY_STRINGS_OBSERVER_CONFIG` entry point are unchanged so the owned controller can reuse its reviewed startup and cleanup path.

## Strict scope

The configured source pair must be clean engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` and Tabby `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb`, with verified actual import locations. Observation requires the exact retained `record_text` message and schema, thinking enabled, no parallel calls, max_tokens256, temperature0, top_k1, top_p1 and n1. The only variants are required/named choice, streaming/nonstreaming, and an explicit reasoning budget24/omitted budget. Initial reasoning must be active.

The rendered prompt must have SHA256 `32655bc461bfa9685942882754b89e75f6640a5605004d4a3d609ebfc6076f58`. The independent CPU rendering and engine-encoding checks bind that text to348 tokens for this exact3.05 template; the observer hashes the existing string and performs no tokenization or token reads.

Capture is bounded at eight matching requests. It records each variant, native request ID, initial reasoning flag, exact rendered prompt, native `full_completion`, the accumulated backend `full_response`, existing scalar finish metadata and collector completion/error state. The controller must require exactly one capture of each variant and bind those IDs to the eight client responses. Capture completeness and semantic success are separate conclusions.

Named tool choices are converted with the existing `plain()` helper only when serializing diagnostic metadata. The original request object is unchanged. All eight original client payloads and their strict validators are preserved.

## What changed from the prior observer

Only request constants/matching, literal scope metadata, the record cap, exact prompt/initial-phase checks, named-choice serialization and exact source pins change. The collector/finish hooks, startup deferral, source verification, scalar handling and publication functions have identical ASTs to the reviewed predecessor. The site hook is byte-identical.

No extra generation await, per-token hook, CUDA/KV access or response rewriting is introduced. Publication failure leaves the original collector result intact and makes the diagnostic incomplete. The retained raw strings can distinguish parser/transport loss from already incorrect model output; a valid capture by itself does not prove either cause.

## CPU evidence

`check_observer_cpu.py` runs the unchanged eight client payloads through the actual clean5a Pydantic/request formatting path and then a fake native finish/collector on CPU. All eight variants match, serialize their named/required choices correctly, preserve both raw strings and preserve original return identities. Thirteen request-field negative guards plus the cap, initial-phase and exact-prompt guards pass. `observer-cpu-report.json` records the hashes and checks.

The source-derived input-encoding proof is separately retained at `BASE/literal-final-triage-5a/engine-tokenizer-proof.json`. Both old94 and final24f0 source methods preserve the exact348 prompt IDs and decode the literal string unchanged. This finding does not attribute the subsequent generation/parser error. No live model or API was called by these CPU checks.
