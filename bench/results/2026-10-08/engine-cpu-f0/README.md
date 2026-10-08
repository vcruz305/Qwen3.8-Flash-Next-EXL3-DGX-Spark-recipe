# Engine producer-budget CPU evidence — f0beceb3

Exact engine source: [f0beceb350d52bfdd62cbce6e6b462b35dd04759](https://github.com/vcruz305/exllamav3/commit/f0beceb350d52bfdd62cbce6e6b462b35dd04759). The tracked checkout was clean before and after both commands.

These are WSL CPU checks of the optional producer-owned accepted-token budget, natural-only guarded phase ending, and affected generator paths. They are separate from native GPU and full-model numerical tests at engine16ca; the changes after16ca touch only Python generator source and tests. Ordinary final recipe setup is still required to align the installed package, source pin, and runtime fingerprint.

| Gate | Result | Evidence |
|---|---:|---|
| New token-budget cases | 77 passed | token-budget.log / token-budget.xml |
| Token budget plus existing MTP row-budget and batched-sampler cases | 119 passed, 22 subtests passed | affected-generator.log / affected-generator.xml |

The second row includes the first row's77 cases. These are not196 distinct tests. Counts are copied from the preserved pytest output; the XML reports retain the individual case records.

## Exact commands

Working directory was the clean engine checkout recorded in evidence.json. The test selection was:

```bash
/usr/bin/python3 -m pytest --noconftest -q \
  tests/test_token_budget_cpu.py \
  --junitxml=/home/vcruz/src/qwen-overnight-20261008/recipe/bench/results/2026-10-08/engine-cpu-f0/token-budget.xml

/usr/bin/python3 -m pytest --noconftest -q \
  tests/test_token_budget_cpu.py \
  tests/test_mtp_row_budget_cpu.py \
  tests/test_sampler_batch_verify_cpu.py \
  --junitxml=/home/vcruz/src/qwen-overnight-20261008/recipe/bench/results/2026-10-08/engine-cpu-f0/affected-generator.xml
```

`--noconftest` deliberately avoids the repository-wide CUDA cleanup fixture. The selected tests run production source methods with CPU state/tensor shims and do not load a model or use a GPU. Python, Torch and pytest versions, command arguments, timestamps, source-file hashes, log hashes and XML hashes are recorded in evidence.json.

## What the budget checks exercise

The tests execute production Job construction, token acceptance and forced-token handling, banned-string checkpoint/rewind, requeue state, SeqTensor, MTP acceptance/rejection and target-state handoff, actual dense and packed logit-mask preparation, and AsyncJob/AsyncGenerator sequencing.

Coverage includes budget0 and token healing; forcing before tokenN+1; accepted versus rejected speculative tokens; slow-consumer backpressure; natural and forced closing markers; complete forced tails; cancellation, stopping and maximum length; literal markers protected by a synchronous boundary guard; partial banned-marker rollback; requeue while guarded or forced; same-window grammar handoff; callback/guard exceptions contained to the failing job; and prevention of calibration or MTP-carry updates from failed transitions. The existing no-budget batched verification path is also checked.

The natural-only mode uses `max_tokens=None` and no output tensor. It observes accepted native closing markers without encoding, forcing, or changing the job length limit. The tests additionally cover protected literal markers, reversible banned holds, requeue, cancellation, guard/callback errors, and same-window dense/packed mask handoff in this mode. `AsyncJob.supports_natural_token_budget=True` provides explicit capability negotiation.

The generic API starts from an already-known active phase. Tabby supplies the incremental parser guard and content-filter callback; its combined integration and live HTTP validation are recorded separately. No claim about generated answer quality, GPU numerical equivalence, or server throughput follows from this CPU evidence alone.

## Native-source boundary

`native-source-proof.json` records the complete `git diff --name-only` result from16ca to this final source, plus the full diff hash and a clean diff check. Exactly three Python generator modules and the new budget test differ; there are no native kernel, header, build, or model-loader changes. This establishes source scope without claiming that a Python sampling change is already covered by the earlier GPU numerical tests. The predecessor9c archive remains available separately.
