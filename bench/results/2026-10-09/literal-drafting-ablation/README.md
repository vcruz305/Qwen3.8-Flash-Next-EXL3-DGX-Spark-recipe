# Original literal requests with drafting disabled and enabled

Both ordered fresh-server cells completed with **valid captures**, while exact literal copying failed in most cases. The original eight visible requests and validators were unchanged; the controlled serving difference was `DRAFT_MODE=disabled` versus `DRAFT_MODE=mtp`.

| Cell | Live requests | Valid raw captures | Exact-value passes | Exact-value failures | API prompt tokens per request | Owned server cleanup |
|---|---:|---:|---:|---:|---:|---|
| Drafting disabled |8|8|0|8|348|SIGTERM, exit0, group empty |
| MTP enabled |8|8|1|7|348|SIGTERM, exit0, group empty |
| Total |16|16|1|15|—|Both complete |

See the derived [summary](summary.json), unchanged [outer result](reports/result.json), [disabled assessment](reports/literal-disabled/literal-capture.json) and [MTP assessment](reports/literal-mtp/literal-capture.json). Capture validity describes complete source-bound evidence, not successful model semantics or general tool-call qualification.

The run began 2026-10-09T06:13:25UTC and finished 06:16:36.782883UTC. It used the original 3.05 flat pack, RAM placement, one request slot, 262144-token cache/context, chunk 2048, the recorded numerical compatibility controls, and exact recipe a8 / engine 24f0 / Tabby f650. MTP used the existing dynamic depth 5 / confidence 0.6 settings; the disabled cell retained those inert settings. The deployment/configuration files and exact job JSONs are retained for both cells.

Each group of eight covers budgeted/unbudgeted reasoning × required/named tool choice × nonstream/stream. The budgeted requests ask for 24 reasoning tokens; max output is 256. The exact expected argument is `<think>literal</think>`. Every actual prompt usage was 348 tokens; actual cached/output counts and finish reasons are preserved per request. In the disabled cell, the four budgeted requests finished with tool calls containing the wrong literal, and the four unbudgeted requests reached length 256 without completing the required tool call. They remain failures. No throughput, TTFT improvement or general accuracy claim is made from this diagnostic.

## What the raw observations establish

The unchanged [phase summary](peer-analysis/phase-summary.json) and per-cell reviews show:

- All eight target-only requests emitted the altered spelling `thinkaliteralthink` by processed position 14, before any phase/forcing event and while no content filter was attached. The retained, non-rewound/non-healing prefix exactly matches the native output prefix.
- Three MTP requests showed that same pre-intervention altered spelling. The remaining five emitted the native closing marker at position 16 while describing an unfinished quoted value, after which the guard admitted closure and the content grammar attached. Four of those five later failed the exact value and one passed.
- Native and backend full text matched in all 16 captures. The phase timing is observable; the model's intended meaning of a closing marker inside a quotation is not established by these traces.

This supports a separately labeled changed-input tokenization diagnostic. It does not justify silently rewriting production user input, attributing every error to the parser, or declaring a general quote/nesting heuristic safe. The changed-input BPE experiment is not included in this archive or denominator.

## Files and independent review

- [reports](reports/) contains all 55 original text/data files from the completed raw source archive, including request/response reports, 16 raw producer captures, jobs, preflight records, configurations, deployment snapshots, logs and clean owned shutdown results.
- [peer-analysis](peer-analysis/) preserves the separate 33-file analysis snapshot and the exact analysis source. Its duplicated 29 raw/result files were checked byte-identical against the source run, and remain so for transparent provenance.
- [sources/measurement/literal-ablation](sources/measurement/literal-ablation/) preserves the frozen two-cell controller, seven exact helper/client/validator modules, input/job snapshots and 14 CPU contract checks. The observer sibling layout is retained so local relative test dependencies remain understandable.
- [sources/literal-ablation-observer-timeline](sources/literal-ablation-observer-timeline/) preserves the actual observer 5127, original payloads, pure timeline validator and their CPU reports. Superseded CPU reports are kept under their original descriptive names and are not final validation.
- [release-review](release-review/) preserves the independent scanner/replay source and result. It replayed 16 unchanged client validations, checked 16 native/backend matches, validated 2,658 complete timeline events with no dropped events, and confirmed both cleanup records.
- [sources/analysis-dependencies](sources/analysis-dependencies/) supplies the exact small API assembler/client sources used by the analysis. Model tokenizer assets and installed Python packages are not vendored; their required identities remain in the original analysis/CPU records.

The raw source archive had 59 files. Four incidental Python cache files are omitted from publication, with names, sizes and hashes recorded in the parent [collection receipt](../collection-receipt.json). The original 59-file archive remains untouched. No model tensors, symlinks or compiled extensions are copied.

The independent credential scan examined all 59 source files and found zero candidates for its declared patterns/key names. It does not guarantee absence of arbitrary unlabeled credentials. Exact raw log bytes are retained after that bounded review; credential stores are not included.

Archived scripts retain their original absolute host paths and are source evidence. Their frozen READMEs describe preparation before the completed run. Do not execute them automatically from the archive: live use requires the parent's cooperative GPU handoff, source/ownership review, unique output paths and the matching installed assets. The stored preflight is a fresh setup check/import check, not a claimed rerun of all historical CPU suites. The 14 controller CPU checks and observer CPU proofs are separate preparation evidence.

No source/runtime/service/model changes, GPU work or new API requests were performed while collecting this archive. The host restoration and any later deployment decision are separate evidence and are not implied here. Verify this directory using `sha256sum -c SHA256SUMS`.
