# Frozen paired numerical checks

These utilities reproduce the format-2 numerical checks used during the
2026-10-08 investigation. They compare a candidate engine with a saved baseline
on the **same model pack and exact input tokens**, and separately compare each
engine's paged execution with its own complete prefill. They do not select a
deployment profile or replace the API, throughput, long-context, or kernel tests.

The probe, assessment script, and original gate declaration are copied
byte-for-byte from the investigation. Verify their hashes from this directory:

```bash
cd bench/numerical
sha256sum -c SHA256SUMS
```

| File | Purpose |
| --- | --- |
| [spark_quality_probe.py](spark_quality_probe.py) | CUDA target-model probe; saves format-2 JSON and optional complete logit tensors. |
| [assess_paired_quality.py](assess_paired_quality.py) | CPU-only assessment of an old/new report pair using the supplied gates. |
| [quality-gates.json](quality-gates.json) | Original declaration, timestamped `2026-10-08T07:40:56.665636+00:00`. |
| [SHA256SUMS](SHA256SUMS) | Frozen SHA256 identities for those three files. |

Current deployment settings and the investigation's results belong in the
[validation report](../../VALIDATION_2026-10-08.md). A file named
`quality-final-...` in an intermediate experiment is not evidence that its
candidate was approved; use the corresponding assessment and report.

## What the probe measures

The fixed code, tool-text, and multilingual corpora are tokenized into
deterministic inputs. Default prefix lengths are 255 and 1,023 tokens; each path
scores the following 48 teacher-forced next-token predictions. Each of the six
corpus/prefix combinations has one complete cache-free prefill and two paged
paths, with query lengths 1 and 6. That is 12 paged cases with 576 scored
positions, plus six unique prefill cases with 288 positions. These positions
share text and labels across paths and are correlated observations.

The probe loads only the target model's text component. It uses FP16 KV cache,
permuted physical pages, and fresh recurrent state for each paged path. For
query lengths greater than one, it commits each fully accepted history block
with `state.rewind(0)`, matching the generator's accepted-history convention.
Saved input-ID tensors have distinct storage. A candidate run with
`--compare-logits` requires a format-2 baseline and checks the actual input
tensors for exact equality before producing paired metrics.

Logits are retained for the complete configured model vocabulary. Metrics
include next-token negative log likelihood (NLL), reference-to-candidate KL
divergence, top-1 agreement, and disagreements where the reference's top-1
probability is at least 0.8. The candidate report contains both comparisons:
candidate paged execution against its own complete prefill, and candidate
execution against the baseline's corresponding path. Full-prefill comparisons
are also saved.

The JSON records the resolved model path, engine import path and Git commit,
Python/Torch/CUDA versions, device, selected environment variables, active
execution paths before and after inference, PLE placement, and cache type. This
is useful provenance, but the recorded Git commit does not establish that the
checkout was clean, and a matching model path does not establish that its files
were unchanged. Preserve the controller's deployment snapshot, model manifest,
source status, exact command, environment controls, exit code, and console log
alongside the probe outputs.

## Run the baseline first

Use two separately preserved, working runtimes. Capture the baseline before
updating its engine or rebuilding its environment. The historical baseline was
the installed working ExLlamaV3 fork; restoring an old stock-upstream recipe is
a different experiment.

Both runs must use the same resolved model view, weights, tokenizer, tensor-file
selection, inputs, PLE placement, and CLI geometry. Keep the same model directory
between runs; do not silently substitute another symlink view or retarget the
view. Preserve the natural `glob.glob("*.safetensors")` enumeration order in
the model manifest: the loader's last file wins when files contain duplicate
tensor keys. Record any intended environment change before running it. Use fresh
processes so import-time engine settings take effect.

The CUDA probe requires the selected runtime's Torch, safetensors, and engine.
It deliberately refuses CPU-only execution. Stop the inference service and
other GPU experiments through their owning controller before running it.
Running concurrent workloads would confound both memory use and the diagnostic
timings.

The following example uses the 3.05 pack with RAM-resident PLE in both runs.
Replace the runtime and recipe paths with the actual preserved locations.
Restore each run's recorded engine environment in its own clean shell; the
commands below select the Python interpreter and engine source, but do not
reconstruct those environment controls. For a disk-PLE comparison, omit
`--ngram-ram` from **both** commands.

First create an exclusively owned output directory. Keep the printed path for
the second run and the assessment:

```bash
set -euo pipefail
NUMERICAL_RECIPE=/absolute/path/to/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe
NUMERICAL_RUN_ROOT="$HOME/qwen-numerical"
mkdir -p "$NUMERICAL_RUN_ROOT"
NUMERICAL_RUN_DIR=$(mktemp -d "$NUMERICAL_RUN_ROOT/run-XXXXXXXX")
printf '%s\n' "$NUMERICAL_RUN_DIR"
```

In the baseline environment:

```bash
BASELINE_RUNTIME=/absolute/path/to/preserved-baseline-runtime
MODEL_VIEW=/home/cruzspark/models/flashnext-exl3-3.05bpw

if PYTHONPATH="$BASELINE_RUNTIME/exllamav3" \
  "$BASELINE_RUNTIME/venv/bin/python" \
  "$NUMERICAL_RECIPE/bench/numerical/spark_quality_probe.py" \
  --model "$MODEL_VIEW" --ngram-ram \
  --contexts 255,1023 --q-lens 1,6 --batch-sizes 1 \
  --steps 48 --prefill-chunk 1024 --cases code,tools,multilingual \
  --output "$NUMERICAL_RUN_DIR/baseline.json" \
  --save-logits "$NUMERICAL_RUN_DIR/baseline.safetensors" \
  >"$NUMERICAL_RUN_DIR/baseline.log" 2>&1; then
  printf '0\n' >"$NUMERICAL_RUN_DIR/baseline.exit"
else
  NUMERICAL_EXIT=$?
  printf '%s\n' "$NUMERICAL_EXIT" >"$NUMERICAL_RUN_DIR/baseline.exit"
  exit "$NUMERICAL_EXIT"
fi
```

In a fresh candidate environment, set `NUMERICAL_RECIPE`,
`NUMERICAL_RUN_DIR`, and `MODEL_VIEW` to the same values. After confirming the
baseline completed successfully:

```bash
CANDIDATE_RUNTIME=/absolute/path/to/candidate-runtime

if PYTHONPATH="$CANDIDATE_RUNTIME/exllamav3" \
  "$CANDIDATE_RUNTIME/venv/bin/python" \
  "$NUMERICAL_RECIPE/bench/numerical/spark_quality_probe.py" \
  --model "$MODEL_VIEW" --ngram-ram \
  --contexts 255,1023 --q-lens 1,6 --batch-sizes 1 \
  --steps 48 --prefill-chunk 1024 --cases code,tools,multilingual \
  --compare-logits "$NUMERICAL_RUN_DIR/baseline.safetensors" \
  --output "$NUMERICAL_RUN_DIR/candidate.json" \
  --save-logits "$NUMERICAL_RUN_DIR/candidate.safetensors" \
  >"$NUMERICAL_RUN_DIR/candidate.log" 2>&1; then
  printf '0\n' >"$NUMERICAL_RUN_DIR/candidate.exit"
else
  NUMERICAL_EXIT=$?
  printf '%s\n' "$NUMERICAL_EXIT" >"$NUMERICAL_RUN_DIR/candidate.exit"
  exit "$NUMERICAL_EXIT"
fi
```

The probe's optional `--max-kl`, `--max-nll-increase`, and `--min-top1`
arguments assess each paged path against **that run's own complete prefill**.
They do not implement the declared old/new paired gates. Leave them unset for
the frozen paired workflow and run the separate assessor. A successful probe
exit without configured thresholds establishes completion and finite-metric
sanity, not acceptance of the candidate.

Assess on any CPU with Python 3; the assessor uses only the standard library:

```bash
python3 "$NUMERICAL_RECIPE/bench/numerical/assess_paired_quality.py" \
  --gates "$NUMERICAL_RECIPE/bench/numerical/quality-gates.json" \
  --baseline "$NUMERICAL_RUN_DIR/baseline.json" \
  --candidate "$NUMERICAL_RUN_DIR/candidate.json" \
  --output "$NUMERICAL_RUN_DIR/assessment.json"
```

## Read the declared gates correctly

The assessor applies the same gates separately to the paged group and the
unique full-prefill group. It scores a full prefill once per corpus, batch, and
prefix; repeated copies in the query-length records must agree exactly.

| Metric | Original declaration |
| --- | --- |
| Pooled paired NLL increase | At most 0.02 nats per scored position. |
| Per-case paired NLL increase | At most 0.05 nats per scored position. |
| Additional high-confidence disagreements per case | At most 1 per 48 scored positions, scaled by the case's position count. |
| Pooled additional high-confidence disagreements | At most 2. |
| Mean paired KL above 0.02 | Investigation trigger. |
| Per-case 95th-percentile paired KL above 0.1 | Investigation trigger. |

A value exactly on a limit passes that limit. Pooled additional disagreements
means candidate total minus baseline total; the sum of positive per-case
increases is a separate diagnostic. For paged paths, each engine's disagreement
count uses its own complete-prefill reference, so the qualifying confidence masks
can differ. Their counts are reported. Full-prefill disagreement compares the
candidate directly with the old prefill, whose self-disagreement baseline is zero.

The assessor rejects inconsistent case sets, input hashes, model paths, cache
types, malformed/nonfinite metrics, and missing paired evidence. It checks that
the paired NLL metrics agree with the supplied old/new reports. It relies on the
successful probe for exact tensor comparison and finite-logit checks; it does
not reload the saved logit tensors or hash model weights itself.

Read the JSON's `status`, `core_passed`, and `investigation_required`.
**Exit 0 covers both `pass` and `investigate`.** A KL-only trigger requires
review before choosing settings even though it does not automatically prove
quality loss. Core failure or invalid evidence returns exit 2. Preserve all
failed and investigated candidates, their original controls, and their
assessment files. A later successful candidate gets new output paths. Do not
edit the gate declaration or score domain to turn an existing failure into a
pass.

## Vocabulary-domain caveat

The original requirement name `finite_unpadded_logits` is a historical
misnomer. The frozen format-2 probe slices logits to
`config.vocab_size == 248320`. The audited tokenizer has 248,077 contiguous
IDs, from 0 through 248,076, leaving 243 additional configured output columns.
Serving masks those additional IDs separately. The frozen NLL and KL
calculations nevertheless include all 248,320 columns in their normalization.
The gate name and metric domain are retained unchanged for comparable evidence.

Two separate audits examined the saved original-`94ba01d` and diagnostic
`b5785675` tensors for the flat 3.05 and SAGE 4.15 packs. Each artifact contained
18 logit tensors, totaling 864 scored positions. All configured columns were
finite, and none of the extra columns was the argmax. The largest per-position
NLL normalization contribution from those columns, taking the larger value
across each old/new pair, was:

| Audited pair | Maximum contribution |
| --- | --- |
| Flat 3.05 | 0.0000119589 nats |
| SAGE 4.15 | 0.0000822894 nats |

These bounds are far below the declared 0.02/0.05-nat NLL limits and cannot
explain NLL changes of that size in these audited runs. They are not an audit of
every subsequent candidate or every pack. The audit artifacts are
`vocab-audit-baseline-b578-305.json`
(SHA256 `29d84880704d6de4ebddcc57e234bc3e5c0b4a6e8f96bdc74ee0a0343311ad03`)
and `vocab-audit-baseline-b578-415.json`
(SHA256 `2a045fe263c346a98be5f8796086b730b2a5d6278f354b0471a069eb506337bb`),
described with the investigation evidence in the
[validation report](../../VALIDATION_2026-10-08.md). A future tokenizer-only
metric should use a separately identified format and a new baseline.

## Output ownership and scope

Both utilities check whether their requested output files already exist.
Those checks are not exclusive file claims and are not safe against competing
writers or dangling symlinks. Allocate one fresh directory per attempt, give
one controller ownership, and never rerun a command against an existing attempt.
The shell examples also redirect logs, so reusing their paths could truncate
earlier logs even when the Python helper refuses to overwrite a report.

The probe writes its final JSON only after a requested logits file has been
serialized successfully. Its writes are not atomic checkpoints. An interrupted
or failed run may leave a partial artifact or only its console log; that is
incomplete evidence. Preserve the actual exit code and failure output. The
assessor's output is likewise a direct write after its existence check.

Default saved logits occupy roughly 409 MiB per run before filesystem overhead.
Keep those tensors in the experiment archive, not in Git. Preserve the much
smaller JSON, command, provenance, and logs for review, plus the tensors whenever
a later paired run or numerical investigation needs them.

This is a small correlated numerical smoke corpus. The default cache stride is
1,280 tokens, not the deployment's full context allocation. The prefixes stay
below the current model's sparse-attention selection threshold. These runs do
not establish 262k-context behavior, 8-bit serving-cache quality, MTP proposal
acceptance, tool-calling correctness, or broad model capability. The query-length
6 path exercises a target verification shape without loading an MTP model.
Diagnostic timings include synchronization, CPU transfers, and metric work and
must not be reported as serving throughput.

The reused assessment fixtures run without CUDA or Torch:

```bash
python3 -m unittest discover -s bench -p test_numerical_assessment.py -v
```

They cover valid pairing, threshold boundaries, separate pooled/per-case and
prefill decisions, investigation-only results, inconsistent evidence, and CLI
failure status. They verify the assessment contract; they do not substitute for
the actual GPU probe or its engine correctness tests.


## Matched K8/V8 cache runs

`spark_quality_q8_probe.py` adapts the frozen probe to the K8/V8 storage used
by the server. Keep it beside the exact base probe and use the same arguments
on both runtimes. It verifies the base hash, changes only cache construction
and cache metadata, and audits the actual allocated packed K/V tensors,
FP16 quantization scales and QSA key planes. Full-prefill references remain
cache-free, and MTP remains disabled. The original numerical arithmetic,
input construction, page permutation and recurrent-history commits are reused.

For example, from this directory, collect a fresh original Q8 run and then a
candidate with the same pack, PLE placement, contexts, query lengths, batch
size, continuation length and chunk size:

```bash
OLD_PYTHON=/path/to/original/venv/bin/python
NEW_PYTHON=/path/to/candidate/venv/bin/python
MODEL=/path/to/the/same/model
"$OLD_PYTHON" spark_quality_q8_probe.py --model "$MODEL" \
  --contexts 255,1023,4095 --q-lens 1,6 --batch-sizes 1 \
  --steps 48 --prefill-chunk 1024 \
  --output original-q8.json --save-logits original-q8.safetensors
"$NEW_PYTHON" spark_quality_q8_probe.py --model "$MODEL" \
  --contexts 255,1023,4095 --q-lens 1,6 --batch-sizes 1 \
  --steps 48 --prefill-chunk 1024 \
  --output candidate-q8.json --save-logits candidate-q8.safetensors \
  --compare-logits original-q8.safetensors
python3 assess_paired_quality_q8.py --gates quality-gates.json \
  --baseline original-q8.json --candidate candidate-q8.json \
  --output assessment-q8.json
```

Supply each runtime's recorded environment separately, as described above;
add `--ngram-ram` to **both** probe commands when that is the tested placement.
Do not compare a Q8 candidate against an FP16 baseline, or relabel an FP16
artifact. The Q8 assessor requires observed cache contracts to match and
reuses the same fixed gate values and scoring arithmetic. Exit zero can still
include investigation alerts: inspect `core_passed` and `investigation_count`.
These checks retain the configured-vocabulary and limited-correlated-corpus
qualifications of the original method. Their diagnostic timings are not API
throughput measurements.
