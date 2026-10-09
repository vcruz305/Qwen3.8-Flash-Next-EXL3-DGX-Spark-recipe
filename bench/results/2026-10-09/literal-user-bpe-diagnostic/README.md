# Changed-input user-span BPE diagnostic — 9 October 2026

This archive records a bounded experiment on the 3.05-bpw model. The visible request bytes are unchanged, but two native added-token IDs inside the known synthetic user text are expanded into ordinary BPE tokens. The actual model input therefore changes from 348 to 352 tokens. Template control IDs remain unchanged.

| Attempt | Drafting | Client requests | Exact literal checks | Capture |
| --- | --- | ---: | ---: | --- |
| First target-only attempt | Disabled | 0 | Not run | Invalid; operational refusal |
| Target-only retry | Disabled | 8 | 8/8 passed | Valid |
| MTP confirmation | Dynamic, maximum depth 5 | 8 | 8/8 passed | Valid |

The eight requests are variants of **one deterministic synthetic prompt**, combining required/named tool choice, streaming/non-streaming, and explicit-budget/unbudgeted behavior. They are not eight diverse examples. The original 348-token experiment is preserved separately in [literal-drafting-ablation](../literal-drafting-ablation/README.md): target-only passed 0/8 and MTP passed 1/8. These results support further evaluation of input-token semantics; they do not establish a general tokenizer policy or broad model-quality improvement.

## What is verified

Each completed request has a native input proof, actual API usage of 352 prompt tokens, unchanged visible wire payload, exact final tool argument, native/backend text equality, and a bounded producer timeline with no dropped records. The altered ID sequence expands only the two known user-span markers; the decoded complete prompt is unchanged. Context length is assessed after this expansion, before generation.

The MTP confirmation records 268 accepted and 157 rejected prediction tokens across eight requests. This demonstrates actual proposal work; it does not identify verification-window counts or establish a throughput benefit. All 16 completed requests have identical native text SHA256 `ecbd48e3c7474ea7532f97ebaca00283cc18bdd9f70d71043e490656ad7c2d68`.

Both completed cells naturally emit the closing reasoning marker at processed position 12 after an unfinished quote, then attach the content grammar. There is no forced output in these traces. Correct final literal copying therefore **does not demonstrate a reasoning-boundary repair** or establish what continuation the model intended. The independent phase analyses retain this distinction.

## Sources and lifecycle

The recorded sources are engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`, Tabby `f650bb5389e0a273549e47d4d26a765760c013e1`, and published recipe `a8c72bdf811646f413fdc03a4e49811a2753d0cf`. The external diagnostic hook does not alter those checkouts or the model files. The target-only controller is `2a0a0e3c…`; the narrowly derived MTP controller is `7c9ca1ac…`. Full hashes, jobs, actual deployments, configs, source checks, observer records, client reports, CPU review artifacts and owned cleanup evidence are included.

The first attempt stopped before clients because the manager-state ownership gate detected a conflict. Its failure and zero-client result remain intact. Its own server, and both completed servers, received SIGTERM, exited 0, and left their owned process groups empty. Controllers held the cooperative GPU lock and checked owner state; they did not enable or disable either service.

The API source checks are live source/import checks, not a claim that the complete CPU suite was rerun on Spark for every cell. The retained local CPU reports describe their own narrower source-execution, request matching, tokenization and lifecycle scopes.

## Archive layout and reproduction boundary

- `reports/`: exact small result files for aborted r1, target-only r2, and MTP confirmation.
- `analysis/`: unchanged independent phase reviews, their source, and the MTP review's matching raw subset.
- `sources/measurement/`: frozen controllers, jobs, model-input audit, exact parent helpers and CPU review evidence.
- `sources/literal-bpe-observer/`: the source-bound input hook, observer, validators, payloads and CPU evidence.
- `collection/`: original remote manifests and bounded release-review receipts.
- `collection-receipt.json`: byte-for-byte source mapping and exclusions.
- `summary.json` and `SHA256SUMS`: machine-readable scope and integrity.

This is evidence for a host-specific diagnostic, not an automatic installer. Do not import arbitrary historical review scripts: some older analysis helpers execute their analysis on import. The collection tool extracts only literal pattern data from its hash-verified pattern source using the Python AST, without executing that source. Controller invocations require the documented Spark paths, published recipe, client environment, exact model metadata, ports and cooperative lock; copied scripts retain those explicit paths and hashes. Read each frozen controller README before any separately authorized run.

Publication omits incidental Python bytecode and model-view symlinks without following them. Original local/remote result files are retained. All regular small text/data copies are exact. The bounded scan found zero credential-pattern/key-name candidates; it cannot guarantee absence of arbitrary unlabeled secrets. No tensors, weight payloads, shared libraries or credential stores were copied or hashed. The scanner's v1 source remains archived for the first receipt; v2 records excluded symlinks explicitly. One initial local MTP analysis assertion assumed zero formatting newlines; the retained direct raw-parameter check documents its correction to one framing newline on each side, without changing raw output or model results.
