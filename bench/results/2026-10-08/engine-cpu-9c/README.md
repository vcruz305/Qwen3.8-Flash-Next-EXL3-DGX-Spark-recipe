# Engine producer-budget CPU evidence — 9c0bbaaa

Exact engine source: [9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2](https://github.com/vcruz305/exllamav3/commit/9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2). The tracked checkout was clean before and after both commands.

These are WSL CPU checks of the optional producer-owned accepted-token budget and its affected generator paths. They are separate from native GPU and full-model numerical tests at engine16ca; the changes after16ca touch only Python generator source and tests. Ordinary final recipe setup is still required to align the installed package, source pin, and runtime fingerprint.

| Gate | Result | Evidence |
|---|---:|---|
| New token-budget cases | 64 passed | token-budget.log / token-budget.xml |
| Token budget plus existing MTP row-budget and batched-sampler cases | 106 passed, 22 subtests passed | affected-generator.log / affected-generator.xml |

The second row includes the first row's64 cases. These are not170 distinct tests. Counts are copied from the preserved pytest output; the XML reports retain the individual case records.

## Exact commands

Working directory was the clean engine checkout recorded in evidence.json. The test selection was:

```bash
/usr/bin/python3 -m pytest --noconftest -q \
  tests/test_token_budget_cpu.py \
  --junitxml=/home/vcruz/src/qwen-overnight-20261008/recipe/bench/results/2026-10-08/engine-cpu-9c/token-budget.xml

/usr/bin/python3 -m pytest --noconftest -q \
  tests/test_token_budget_cpu.py \
  tests/test_mtp_row_budget_cpu.py \
  tests/test_sampler_batch_verify_cpu.py \
  --junitxml=/home/vcruz/src/qwen-overnight-20261008/recipe/bench/results/2026-10-08/engine-cpu-9c/affected-generator.xml
```

`--noconftest` deliberately avoids the repository-wide CUDA cleanup fixture. The selected tests run production source methods with CPU state/tensor shims and do not load a model or use a GPU. Python, Torch and pytest versions, command arguments, timestamps, source-file hashes, log hashes and XML hashes are recorded in evidence.json.

## What the budget checks exercise

The tests execute production Job construction, token acceptance and forced-token handling, banned-string checkpoint/rewind, requeue state, SeqTensor, MTP acceptance/rejection and target-state handoff, actual dense and packed logit-mask preparation, and AsyncJob/AsyncGenerator sequencing.

Coverage includes budget0 and token healing; forcing before tokenN+1; accepted versus rejected speculative tokens; slow-consumer backpressure; natural and forced closing markers; complete forced tails; cancellation, stopping and maximum length; literal markers protected by a synchronous boundary guard; partial banned-marker rollback; requeue while guarded or forced; same-window grammar handoff; callback/guard exceptions contained to the failing job; and prevention of calibration or MTP-carry updates from failed transitions. The existing no-budget batched verification path is also checked.

The generic API starts from an already-known active phase. Tabby supplies the incremental parser guard and content-filter callback; its combined integration and live HTTP validation are recorded separately. No claim about generated answer quality, GPU numerical equivalence, or server throughput follows from this CPU evidence alone.
