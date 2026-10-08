# Recipe R1 CPU qualification — 8 October 2026

This archive preserves the clean recipe qualification run for commit
**`3337bc8d64e4befafa2a1aff342e07abbea242ac`** and tree
`c78cf27ebeb96dcd94f6bd32f424042e71037cdd`.

The original unittest result is:

```text
Ran 137 tests in 3.888s

OK (skipped=5)
```

The process exited with code 0. The five skip events comprise **one optional
real-template integration class** and **four optional OpenAI SDK tests**. A
class-level skip is not counted in the same way as an individual executed test;
this archive does not derive a passed-test total by subtracting five from 137.
The complete skip messages and individual test outcomes are retained in
[tests.log](tests.log).

## Execution and identity

The [original result](result.json) records:

| Item | Recorded value |
|---|---|
| Start | `2026-10-08T12:28:31.293549+00:00` |
| Finish | `2026-10-08T12:28:35.519571+00:00` |
| Process wall time | 4.226012836 seconds |
| Recipe checkout | `/home/vcruz/src/qwen-overnight-20261008/recipe-qualified-cpu` |
| Clean before / after | `true` / `true` |
| Selected Tabby source | `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb` |
| Exit code | `0` |
| Raw test-log SHA256 | `4709078b8dc2bc09ae078b866bd5ea0f73a1e49a082dba15f15ec488600eb1bb` |

The command was the recipe's full CPU discovery command, using the recorded
Tabby tooling interpreter:

```bash
/home/vcruz/src/qwen-overnight-20261008/tabbyapi-agent/.venv-tools/bin/python \
  -m unittest discover -s bench -p 'test_*.py' -v
```

The result metadata records `TABBY_SOURCE` as the selected natural-end Tabby
checkout. The optional real-template integration class specifically requires
`QWEN_TEST_TABBY_SOURCE` and `QWEN_TEST_TOKENIZER_CONFIG`; its retained skip states
that those integration inputs were not supplied for this invocation. The four
SDK tests report that the optional OpenAI SDK was not installed in this test
interpreter. They are not represented as passing integration tests here.

The [archival identity observation](identity-at-archive.json) independently
confirmed that the qualification checkout still had the exact recorded commit,
tree and an empty porcelain status at archival time. That later observation is
kept distinct from the original run's before/after identity record.

## Shell syntax

The preserved [shell result](shell-syntax.json) records exit code 0 with empty
stdout and stderr for:

```bash
bash -n exllamav3-tabby/setup.sh exllamav3-tabby/env.sh exllamav3-tabby/serve.sh
```

This validates shell syntax. It does not execute an installation or start a
server.

## Evidence boundary and integrity

This is CPU recipe qualification for the exact R1 commit. It does not establish
GPU numerical equivalence, model quality, live API correctness, throughput,
long-context retrieval, or service/rollback behavior. Those checks have separate
source identities and result archives in the
[validation report](../../../../VALIDATION_2026-10-08.md).

`result.json`, `shell-syntax.json` and `tests.log` are unchanged byte copies of
the original artifacts. [Source provenance](source-manifest.json) records their
paths, sizes and hashes. No tests, shell checks, setup commands, GPU/API requests
or source modifications were performed during archival. Verify all archived
bytes from this directory with:

```bash
sha256sum -c SHA256SUMS
```
