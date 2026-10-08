# Final runtime setup on Spark — October 8, 2026

The ordinary recipe setup and its five scheduled checks **completed successfully
at 13:00:12 UTC**, using the exact engine, TabbyAPI and recipe revisions below.
This archive records installation, dependency/build checks and the tests run in
the actual Spark venv. It does not qualify model numerical quality, API behavior,
throughput, persistent service operation or rollback.

## Exact installed sources

| Component | Recorded commit |
|---|---|
| ExLlamaV3 engine | `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` |
| TabbyAPI | `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb` |
| Qualification recipe R1 | `3337bc8d64e4befafa2a1aff342e07abbea242ac` |

The canonical new recipe was
`/home/cruzspark/qwen-spark-recipe`; the candidate runtime remained at
`/home/cruzspark/qwen38-exl3-20261008`. The original runtime's canonical path
was preserved.

The [setup status](setup/status.json) and
[setup snapshot](runtime/setup-snapshot.json) retain the selected Git identities.
The controller checked exact clean source revisions before and after the
scheduled checks. The snapshot also records package versions and relevant
environment settings.

For this qualification, the controller supplied exact `EXL3_REF` and
`TABBY_REF` overrides. This does not change the recipe's normal Tabby `main`
tracking policy. A moving branch label is separate from the exact commit
recorded for this run.

## Actual scheduled checks

| Check | Recorded outcome |
|---|---|
| Ordinary `bash exllamav3-tabby/setup.sh` | Exit 0; source build, Ninja/toolchain and runtime fingerprint verified; `pip check` reported no broken requirements. |
| `setup.sh --check` | Exit 0; engine import/source, build fingerprint, Tabby identity and dependency checks passed. |
| Engine producer-budget tests | **77 passed in 1.28 s**. |
| Tabby test suite | **398 passed and 5,159 subtests passed in 8.89 s; zero skips**. |
| Actual pack tokenizer utility | Exit 0; **37 acceptance/rejection cases and three prefix-mask checks** matched their expected outcomes. |

The reported test durations above come from pytest's own summaries. Their
controller wall times include process startup and are recorded separately.
The engine and Tabby counts are separate suites; this archive does not add
them into a claim of independent capabilities.

Exact raw outputs are retained:

- [Setup log](setup/setup.log) and [check-only log](setup/setup-check.log).
- [77-test engine output](setup/engine-budget-cpu.log).
- [398-test Tabby output](setup/tabby-cpu.log).
- [Tokenizer JSON output](setup/tool-tokenizer.log).

The engine command was
`venv/bin/python -m pytest --noconftest -q tests/test_token_budget_cpu.py`
from the engine checkout. The Tabby command was
`venv/bin/python -m pytest -q -ra` from the Tabby checkout. Each ran with
`PYTHONPATH=.` for that selected source tree. This archive preserves the
commands; no tests were rerun during collection.

The tokenizer utility used the actual flat 3.05 pack tokenizer. Its recorded
vocabulary contains **248,077 entries**, and the tokenizer SHA256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
Acceptance/rejection and mask checks are CPU grammar/tokenizer contracts.
They do not establish that every model will choose the requested function or
copy every requested argument correctly.

## Build state and dependency snapshot

The [stored build state](runtime/build-state.json),
[desired build state](runtime/desired-build-state.json), setup status and
setup snapshot contain the same build fingerprint:

`243f414eb2fc9ac6748f7d3014bef0264ea31e52fa2224d2cf1df314d1fd83bb`

The [small runtime marker](runtime/engine-marker.txt) also names the exact
engine commit. The setup log reports Ninja available from the venv, 12 build
workers and an sm_121 source build. The fingerprint records aarch64, Python
3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0.88, the C++ compiler and intentional build
environment. The full [package list](runtime/packages.json) is retained;
the snapshot includes llguidance 1.9.1, tokenizers 0.23.2 and Triton 3.7.1.

The setup log reports the narrow cuSPARSELt metadata repair as already correct.
The snapshot retains the prior audit, including its original completion time,
before/after metadata and already recorded library identity. This collection
did not reread or hash the CUDA library, compiled extension, model weights or
activation tensors.

A matching source/build fingerprint is installation provenance. It is not
a model-output quality or performance measurement.

## Transition and resolved source review

The [transition launch record](transition/launch.json),
[complete transition status](transition/status.json) and per-command logs retain
the sequence that followed the historical performance matrices. The transition
waited for both measurement-controller PIDs to disappear, required the exact
completed label sets and released port, preserved the original runtime,
created the new canonical recipe checkout and selected the intended source
revisions before invoking setup.

The prerequisite result hashes in the setup status match all **17 primary**
and **16 optional** result files in the corresponding published archives.
This establishes the recorded completion boundary; it does not turn those
matrices' failed functional checks into passes.

| Executed script | SHA256 |
|---|---|
| [qualified_transition.py](scripts/qualified_transition.py) | `a0d818982a95a85d9944f11ecc4da5c2b2b318e38cb05175b392e0a4c23ce2db` |
| [setup_qualified_runtime.py](scripts/setup_qualified_runtime.py) | `fdad4e0244a0a7a45440a74e44507808115785cdaae47b220c7dee2f3a002d5e` |

The [resolved review record](scripts/qualified-transition-source-review.json)
is bound to the executed transition's exact SHA256. It records the correction
that rechecks staged input hashes after the controller wait and again before
setup and quality execution. That was source review, not an additional test
suite or deployment run.

The setup controller's five-command status is separate from the transition's
later quality command. The transition explicitly retains
`quality_gate_review_required: true`. Its own `passed: true` means its
scheduled commands completed successfully; a zero quality-controller exit
does not evaluate or approve the numerical thresholds. The subsequent chunk4096
quality results and any investigation alerts belong in a separate evidence
archive and assessment.

The empty launcher/child summary logs are preserved as original zero-byte
files. Detailed commands and outputs are in their status records and dedicated
logs; no missing output was reconstructed.

## Evidence and reproduction boundaries

[integrity.json](integrity.json) records the archive checks, exact test counts,
build-state agreement, command-log hash matches and prerequisite archive
matches. [source-provenance.json](source-provenance.json) identifies each
original path, size, modification time and raw SHA256. The remote collector
checked each file's size/mtime across its read; the completed status and
command-log hashes were then verified before copying.

To verify the saved bytes from this directory:

```bash
sha256sum -c SHA256SUMS
```

The archived transition is fixed to the historical host, PIDs, source bundles
and unused output paths. It is retained as executed source evidence, not a
general installer to rerun against an active server. Its setup helper has a
guarded entry point. For a new installation, use the ordinary
[current recipe workflow](../../../../README.md), select the intended exact
sources and runtime paths, and keep inference, service and numerical checks
as separate scheduled stages.

Collection was read-only on Spark. It copied only small completed records,
scripts, logs and metadata; it made no API request, setup invocation,
service/process change or new inference/test run.
