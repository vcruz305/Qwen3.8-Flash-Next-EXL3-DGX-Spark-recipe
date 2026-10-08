# Final validation controllers — frozen source archive

This bundle preserves the exact controllers, clients, observer and relevant CPU
review evidence prepared for the final DGX Spark validation. It contains no model
weights and makes no claim that a selected source pair has passed live validation.
Live setup, API, performance and service results belong in their own result
archives and in the [validation report](../../../../VALIDATION_2026-10-08.md).

Every copied source file is byte-identical to its reviewed original.
[Source provenance](source-manifest.json) records the source paths, sizes and
SHA256 values. `SHA256SUMS` covers the archive's raw bytes. No controller, client,
observer or test implementation was edited for publication. No setup, model load,
GPU/API request, or additional optional CPU test was run during this packaging.

## What is frozen

The entry points are in `scripts/`; their required parent/helper files remain
adjacent. Full source hashes are in the manifests. These prefixes identify the
reviewed versions:

| File | SHA256 prefix | Role |
|---|---|---|
| `setup_qualified_runtime.py` | `fdad4e0244a0` | Run ordinary recipe setup and five required setup/CPU/tokenizer commands on an already selected clean source pair. |
| `final_api_controller.py` | `ae9e0965a1ab` | Configure one owned API validation run with explicit source commits, pack and tuning. |
| `final_api_batch.py` | `4820f2515b98` | Run explicit jobs serially, preserve outcomes and sparse resource observations, and stop on failure or uncertain cleanup. |
| `api_f4_gemm_controller.py` | `78e990269f0c` | Exact underlying API process lifecycle and strict client gates. |
| `spark_experiment_controller.py` | `48281d5b51b1` | Exact shared source/model identity, owned-process and resource-sampling helpers. |
| `reasoning_literal_smoke.py` | `8f964990d448` | Required/named calls containing literal closing tags, in both response modes, with an explicit budget or without one. |
| `reasoning_concurrency_smoke.py` | `1e88c70eb82f` | Bounded concurrent compatibility checks against an already-owned four-slot server. |
| `strings_only_controller.py` | `cfc3619894c4` | The two original exact-string requests plus bounded raw observation and separate capture/semantic gates. |

The helper and parent are historical frozen sources. Their original comments,
default paths and source constants are retained. **Use the final wrapper for an
explicit final job; the parent alone selects its historical experiment.** The
wrapper verifies its parent's full hash, supplies the selected source pair and
configuration, and retains the helper's own full-hash check.

The corrected strings observer is preserved under
`scripts/tabbyapi-diagnostics/strings-final-9c-3adc/observer/`:

- Observer SHA256: `77d606e2f2e894f7503a6f41c99a5a94696339b5c15c7b8415b68f8416b945d3`.
- Site hook SHA256: `3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f`.

The strings controller creates a per-run derived observer by changing only the
single expected engine and Tabby source constants. Reversing those replacements
must recover every original byte. The observer retains its actual Git/import-path
gates and exact synthetic request scope. It recognizes the framework-augmented
template variables that reach the real collector. The preceding pre-render-only
matcher is not used by this bundle.

## Recorded host and path boundaries

These controllers target the recorded Linux Spark environment:

| Item | Expected path or value |
|---|---|
| Working root used by the frozen lifecycle | `/home/cruzspark/qwen-overnight-20261008` |
| Candidate runtime | `/home/cruzspark/qwen38-exl3-20261008` |
| Original runtime, preserved at its canonical path | `/home/cruzspark/qwen38-exl3` |
| Default working recipe | `/home/cruzspark/qwen-overnight-20261008/recipe-updated` |
| Final recipe may be supplied through the job | `/home/cruzspark/qwen-spark-recipe` |
| Runtime Python | `/home/cruzspark/qwen38-exl3-20261008/venv/bin/python` |
| OpenAI SDK client Python | `/home/cruzspark/qwen-overnight-20261008/client-venv/bin/python` |
| Listener | `http://127.0.0.1:8899/v1`, IPv4 loopback, no authentication |
| Public model alias | `Qwen3.8-Flash-Next-EXL3` |
| Model tokenizer used by the setup check | `/home/cruzspark/models/flashnext-exl3-3.05bpw/tokenizer.json` |
| CUDA | `/usr/local/cuda`, architecture `12.1` |

The client environment is separate from the model runtime; the SDK requirement
is retained in [requirements-sdk.txt](evidence/requirements-sdk.txt). The scripts
use Linux `/proc`, process groups and the existing CUDA/runtime installation.
They do not accept arbitrary runtime, network or account substitutions. Running
them on another host requires an explicitly reviewed adaptation and new hashes.
The public [matrix guide](../../../../bench/matrix.md) and
[API validation guide](../../../../docs/api-validation.md) cover reusable recipe
clients and current commands.

Keep the original virtual environment at its original path. This bundle neither
moves it nor rehearses service promotion; the [service guide](../../../../docs/service.md)
describes that separate lifecycle.

## Setup gate

`setup_qualified_runtime.py` requires full 40-character source commits and checks
that the engine, server and recipe checkouts are already clean. It refuses an
occupied port 8899. Optional prerequisite matrix/job pairs must contain the exact
expected labels, completed records and certain cleanup; supplied prerequisite
controller PIDs must no longer exist. These are checks, not a polling queue.
The caller still schedules an exclusive GPU window.

A prerequisite matrix may contain failed tool checks while having completed its
measurements and cleanup. The prerequisite gate preserves that distinction; it
does not reinterpret those failures as qualified model behavior.

The setup controller records these five command outcomes:

1. Ordinary `exllamav3-tabby/setup.sh`.
2. `setup.sh --check`.
3. The engine producer-budget CPU suite with `--noconftest`.
4. The TabbyAPI CPU suite.
5. The actual Qwen tokenizer/tool-grammar check.

It records the source tree identities before setup and checks that all three
commits remain unchanged and clean afterward. The runtime marker must identify
the selected engine. The final API wrapper accepts setup evidence only when it
contains the exact runtime/source pair, `passed: true`, a completion timestamp,
the actual recipe commit, and all five distinct successful finished commands.

The setup controller is **source-reviewed**. There is no retained full CPU dry run
of that controller. The 19-group API-controller review tests the setup-status
consumer's positive case and 11 rejection cases; it does not execute the installer.
The scheduled ordinary setup invocation supplies the definitive runtime evidence.

An invocation on the recorded host has this form. Supply real reviewed commits
and new absolute output paths before executing it:

```bash
python3 /absolute/path/to/scripts/setup_qualified_runtime.py \
  --engine "$ENGINE_SHA" --tabby "$TABBY_SHA" \
  --recipe /home/cruzspark/qwen-spark-recipe \
  --output /home/cruzspark/qwen-overnight-20261008/results/new-final-setup
```

Add paired `--after-results` and `--after-jobs` arguments, and `--after-pid` values,
when a known preceding matrix must be checked. The source pair is selected before
this command; the command does not choose or upgrade to an unspecified new head.
The public recipe continues to track the Tabby fork's `main` by policy, while this
individual qualification run records and verifies its exact tested commit.

## Explicit validation jobs

The final wrapper takes a JSON object containing `label`, `engine`, `tabby`,
`model_path`, `env`, and an optional `recipe`. Commits must be full lowercase
40-character hexadecimal values. Paths and the actual files are verified before
launch. The environment must explicitly specify at least:

```text
PROFILE NGRAM_RAM CACHE_SIZE MAX_SEQ_LEN MAX_BATCH_SIZE CHUNK_SIZE
DRAFT_MODE DRAFT_NUM_TOKENS DYNAMIC_DRAFT
EXL3_MOE_COOP_KSPLIT EXL3_GEMM_LEGACY_TILES
```

Supply all other qualified controls as well, including the applicable GDN,
attention, mixed-K, draft-confidence, shortlist and row-budget settings. This
archive does not select final tuning values. `TABBY_REF` is not a job override;
the wrapper separately verifies the exact actual Tabby commit. Jobs cannot inject
`STATE_DIR`, Python-path overrides or authentication secrets.

The default API group and optional additions are distinct:

| Selection | Actual intended checks |
|---|---|
| `api: true` or omitted | Original tool smoke 28, SDK 5, resilience 19, auto compatibility 9 per repetition. |
| `auto_repeats: 1..3` | One to three unchanged auto9 executions, each separately recorded. |
| `literal_client: "/absolute/path/reasoning_literal_smoke.py"` | Four budgeted literal checks plus four separate checks omitting only the budget field. |
| `concurrency_client: "/absolute/path/reasoning_concurrency_smoke.py"` | Opt-in 52-check diagnostic, with 16 total child clients over four serial phases. Requires exactly four server slots. |
| `concurrency_bench: true` | Fixed public throughput benchmark at 1/2/4 requests, nominal 1,024-token prompts, 256 output tokens, three measured batches and one warmup. Requires at least four slots. |
| `--bench` on the single-job wrapper | Original three short suites, 400 output tokens, one warmup and three measured requests, run ID `overnight-v1`. |
| `long_context: [32768, 240000]`, or another supported list | Separate bounded synthetic retrieval checks. Each selected length must be between 32,768 and 240,000. Actual prompt length and zero cached tokens are checked. |

The concurrent compatibility client retains its original **48-check / 12-child**
default. Its explicit `--include-unbudgeted` option adds a fourth phase, reaching
**52 checks / 16 children**. The final wrapper deliberately enables that option.
The first three phases are four auto9 clients, four original forced-tool clients
with both modes, and four budgeted literal requests. Each phase runs four child
clients concurrently. This tests request isolation; child-process overlap does
not establish a particular GPU batch shape or a performance gain.

An optional `PROMPT_TEMPLATE` must be an absolute existing file. The final wrapper
records and verifies both raw-file and loaded text identity, including the actual
`GET /v1/model` template content. A new template is an explicit profile change.
See the [template guide](../../../../docs/prompt-templates.md).

Run a single explicit job against its matching completed setup status:

```bash
python3 /absolute/path/to/scripts/final_api_controller.py \
  --job /absolute/path/to/final-job.json \
  --setup /absolute/path/to/new-final-setup/status.json \
  --output /absolute/path/to/new-final-job-results \
  --bench
```

For a serial list, put the job objects in a JSON array and use:

```bash
python3 /absolute/path/to/scripts/final_api_batch.py \
  --jobs /absolute/path/to/final-jobs.json \
  --setup /absolute/path/to/new-final-setup/status.json \
  --output /absolute/path/to/new-final-batch-results
```

Every output is a new destination; existing outputs are refused. The single-job
client deadline defaults to 2,700 seconds and is bounded at 3,600. A job containing
the 52-check concurrent client requires at least 2,700 seconds. The batch deadline
defaults to 3,600 seconds, is bounded at 5,400, and cannot be set below 3,600 for
jobs with concurrent compatibility or throughput clients. Deadlines include each
client's complete work and must be planned within the scheduled GPU window.

## Ownership, failure and observations

The frozen lifecycle refuses a busy port, uses a unique process ownership token
and a distinct state directory, checks listener ownership and actual loaded model
identity, and verifies the source/configuration inputs after measurement. It
stops only its owned server/client groups. A server that exits before controller
cleanup invalidates the run. Missing, partial, skipped or inconsistent required
client reports also fail the gate.

The serial batch records each actual child exit and sparse resource samples. It
sends one termination request to its own wrapper when interrupted and allows the
wrapper its bounded cleanup interval. Uncertain cleanup prevents the next model
load. Concurrent clients similarly retain process ownership and cleanup evidence
for all their child sessions. Failed model semantics remain failed observations;
there are no automatic retries that replace their results.

The strings-only workflow is described in
[strings-only-controller.md](strings-only-controller.md). It runs exactly the two
original `tool_smoke.py --case strings --mode both` requests, with the observer
injected only into the owned server. Its `capture_valid` and `semantic_passed`
results remain separate. Exit 0 means both pass, exit 1 means a valid capture with
an original semantic failure, and exit 2 means capture integrity/completeness
failed. Observer runs are diagnostics and are not throughput samples.

## Retained CPU evidence and replay boundaries

[Coverage notes](evidence/coverage-notes.json) distinguish the recorded checks:

- The final controller review has **19 review groups**, including 13 actual
  generated argument-parser paths, the setup-status gate, reporting checks and
  cleanup/signal cases. Those numbers overlap and are not API-request counts.
- Literal/concurrent client sources have a retained **26-test** console log,
  covering the unchanged defaults, optional cases and owned child cleanup.
- The strings controller's **12 checks** were reported passing by its author and
  an independent peer. Exact source and binding are retained; no standalone
  console log was preserved, and no new log has been fabricated here.
- The corrected observer has **13 unit checks** plus **eight actual formatting
  fixture checks**, with its corresponding source manifest and retained evidence.
- The exact experiment helper has prior **20-test** publication evidence. That
  publication changed only the adjacent module filename in its test loader;
  the controller bytes remain identical.

Predecessor sources are included only where the frozen tests compare old and new
request construction or lifecycle code. Their historical evidence is not relabeled
as validation of a later source. Old setup controllers that execute a workflow
when imported are excluded. Some review harnesses themselves execute CPU checks
and write their report when imported; run those as deliberate standalone scripts,
not through broad import or discovery of this archive.

For a future offline replay, first verify this archive with
`sha256sum -c SHA256SUMS`. Work from a disposable copy of `scripts/` so report
creation and Python cache files do not change the evidence archive. Link a complete
recipe checkout as `recipe` inside that copy; the exact client input hashes used
by the reviews are listed in [cpu-recipe-inputs.json](evidence/cpu-recipe-inputs.json).
The standalone final review expects that checkout to have its normal Git metadata.
The frozen test fixtures retain their relative directory layout below `scripts/`.

From that disposable CPU workspace, the relevant entry points are:

```bash
python3 -m unittest -q test_spark_experiment_controller_cpu
python3 -m unittest -q test_reasoning_clients_cpu
python3 -m unittest -q test_strings_only_controller_cpu
python3 review_final_api_natural_cpu.py
```

The standalone review creates `review-final-api-natural-results.json` and refuses
an existing report. Its network/generation entry points are intercepted after
argument parsing or replaced by controlled fixtures. Observer rendering checks
also require the exact Tabby source/template inputs documented in the retained
observer README and report; those full external checkouts are not embedded here.
The recorded counts above describe prior runs, not a new relocation or setup
qualification performed during this packaging.
