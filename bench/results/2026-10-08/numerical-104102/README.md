# Numerical evidence snapshot — 2026-10-08 10:41 UTC

This bundle preserves existing numerical evidence through a read-only snapshot selected at **2026-10-08 10:41:02 UTC**, with copying and hashing finished at **10:41:50 UTC**. It contains 237 small files from Spark, eight local historical/CPU/vocabulary supplements and hashes for 44 large tensor artifacts. The tensor bytes remain on Spark and are not included. No model inference, API request, GPU test or source mutation was performed while packaging.

The bundle is a record of the investigation at that time. It does not select a permanent runtime or performance profile. Later assessments and deployment decisions belong in a separate dated addition; do not replace these earlier failed attempts.

## What is in the bundle

| File or directory | Purpose |
| --- | --- |
| `assessment-index.json` | Paths, exact source identities, frozen gate outcomes, explicit job environments and copied scalar summaries for all 30 assessments. |
| `status-index.json` | Controller statuses and actual probe/assessment exits, including failed orchestration and incomplete work. |
| `tensor-manifest.json` | Exact Spark artifact paths, sizes, mtimes and SHA256 hashes for 44 retained tensor files. No tensor data. |
| `native-control-index.json` | BC control validity, exact source/cache identity, own-output checks and preserved row-count diagnostics. |
| `collection.json` | Read-only collection times, original paths and hashes, small-file limits and exclusions. |
| `local-source-artifacts.json` | Provenance for the eight local supplements. |
| `packaging-validation.json` | Transfer/reference verification and the explicit no-inference/no-rescoring record. |
| `raw/results/` | Byte-exact probe, assessment, input, status and selected native records. |
| `raw/logs/` | Byte-exact historical numerical/native test logs, including failed intermediate tests. |
| `raw/wsl/` | Earlier frozen assessments, CPU validation and the two historical vocabulary audits. |
| `SHA256SUMS` | Hashes for the complete small bundle. |

Every copied remote file matches its recorded SHA256. Every baseline, candidate and gate hash referenced by an included assessment is present in the bundle. The original declared gate file is unchanged: SHA256 `84194429b6a0145ab2d08a2cfa09720c965709e3d0633aea6e646827a889a097`.

## Latest completed cases captured here

These rows compare original engine `94ba01d50a13fa9ff672473f2d0eef8b51a71e99` with candidate `16ca20d27c0e4cce15a9bbc131e6d047065395b5`. The actual environment differs between the default candidate and the explicitly labeled KSPLIT1 control. Each link is the original assessment, and its adjacent `input.json` is the execution configuration.

A passed core gate can coexist with a KL investigation trigger. The `status` values and trigger lists remain unchanged in the raw assessment. The following NLL and disagreement values are the frozen assessment's pooled **paged** metrics; complete-prefill comparisons are scored separately.

| Assessment | Scope | Core gates | KL triggers | Pooled paged NLL change (nats) | Additional high-confidence disagreements |
| --- | --- | --- | ---: | ---: | ---: |
| [candidate-qsa-305](raw/results/quality-16ca-recovery/candidate-qsa-305/assessment.json) | FP16 recovery | pass | 4 | -0.000949769232 | 0 |
| [candidate-short-405](raw/results/quality-16ca-recovery/candidate-short-405/assessment.json) | FP16 recovery | pass | 2 | -0.00160755357 | 0 |
| [candidate-batch4-305](raw/results/quality-q8-16ca-batch4-candidate/candidate-batch4-305/assessment.json) | Q8/K8-V8 | **fail** | 1 | 0.00238558931 | 3 |
| [candidate-batch4-415](raw/results/quality-q8-16ca-batch4-candidate/candidate-batch4-415/assessment.json) | Q8/K8-V8 | pass | 0 | 0 | 0 |
| [candidate-batch4-305-ksplit1](raw/results/quality-q8-16ca-batch4-ksplit1/candidate-batch4-305-ksplit1/assessment.json) | Q8/K8-V8 + KSPLIT1 | pass | 0 | 0 | 0 |
| [candidate-305](raw/results/quality-q8-16ca-candidate/candidate-305/assessment.json) | Q8/K8-V8 | pass | 1 | 0.000754565001 | 0 |
| [candidate-405](raw/results/quality-q8-16ca-candidate/candidate-405/assessment.json) | Q8/K8-V8 | pass | 2 | -0.00301395657 | -1 |
| [candidate-415](raw/results/quality-q8-16ca-candidate/candidate-415/assessment.json) | Q8/K8-V8 | pass | 0 | 0 | 0 |
| [candidate-cyber387](raw/results/quality-q8-16ca-candidate/candidate-cyber387/assessment.json) | Q8/K8-V8 | pass | 0 | -1.80672557e-05 | 0 |

The default 3.05 batch4 result failed the pooled additional high-confidence disagreement gate: three additional disagreements exceeded the declared limit of two. Its subsequent KSPLIT1 control passed the unchanged gates, with zero reported logit differences in both paged and unique-prefill comparisons. That result is preserved without claiming qualification for every other pack or serving workload.

All four Q8 batch1 packs passed the core gates. The 3.05 and 4.05 rows still contain KL investigation triggers. SAGE has zero reported paired logit differences in that Q8 batch1 run; Cyber has small nonzero paged differences and should not be described as bitwise identical. The default SAGE Q8 batch4 run passed with zero reported differences. The two FP16 recovery rows passed core gates while retaining their q1 KL investigation triggers.

The index includes 10 `pass`, five `investigate` and 15 `fail` assessments across the full historical sequence. This count describes preserved attempts, not a pass rate for the selected deployment. It includes the rejected Coop-on setting and earlier compatibility failures. Historical native logs likewise include both unsuccessful intermediate candidates and later successful controls.

## Scope and interpretation

The frozen numerical corpus has three fixed technical/code, tool and multilingual inputs. The original dense tests use prefix lengths255 and1023; the separate QSA tests add4095. Q8 batch1 includes all three prefix lengths on all four packs. Q8 batch4 here covers255 and1023 on 3.05 and SAGE. Each case uses48 continuation positions; batching and repeated q lengths produce correlated observations. This is a bounded numerical regression test, not a broad capability benchmark or a240k-context validation.

`q6` is a target-only six-token evaluation window with recurrent-history handling. The probe does not load the MTP model, and these tests do not measure draft acceptance or certify end-to-end speculative generation. The complete-prefill reference uses the base probe's uncached full-prefill path; paged FP16 and K8/V8 results must retain their distinct cache labels. The Q8 adapter verifies actual quantized cache geometry, scale tensors and FP16 QSA planes before scoring.

The assessor evaluates a configured output vocabulary of248320 logits. The legacy gate field remains named `finite_unpadded_logits`, although243 entries lie beyond the tokenizer's248077 entries. That name was not changed after measurements. The two retained vocabulary audits concern the original94 versus b578 comparisons on3.05 andSAGE; they are not a new audit of16ca or every pack. Keep their exact scope if citing them.

NLL and KL are separate signals. The signed pooled additional high-confidence count follows the frozen scorer, including its per-runtime qualifying masks; it should not be reinterpreted as a count of identical-token-level new mistakes. Unique-prefill inputs are aggregated once rather than repeated for every q length. The index only copies values and reports maximum observed scalar differences; it does not recompute logits, alter gates or turn a numeric equality summary into a claim about every tensor byte.

## Configuration and source provenance

The candidate compatibility work introduced controls after the frozen probe's environment whitelist was written. In particular, an omitted `EXL3_GEMM_LEGACY_TILES` or `EXL3_MOE_COOP_KSPLIT` field in a probe's environment object is not proof that the option was unset. Use the adjacent `input.json`, controller status and recorded compiled marker for those actual execution settings. The index preserves both the frozen probe environment and the separately recorded job environment.

The native BC records preserve an actual flat4.05 layer0 fixture, source/input/cache identity, and checks that the diagnostic reproduces its own saved layer output before interpreting old/new differences. Later16ca legacy-tile GPU policy tests report four passes with mode1 and four with mode0; the raw command construction, logs and orchestration status are retained. These controls substantiate the specific kernel/row behavior tested, not general model quality.

The root `validate-16ca/status.json` is deliberately preserved as failed: at that snapshot the combined controller included an API compatibility failure and the default3.05 Q8 batch4 numerical failure. A later successful isolated control does not retroactively change that status.

## Exclusions and later additions

The `quality-q8-16ca-ksplit1-flat-cyber` batch1 matrix was still probing at selection. Its status snapshot is included, but all partial probe/logit/assessment outputs from that directory were excluded. Results completed after selection must be added separately. No outcome is inferred from a planned job, an active status or an empty controller log.

Large tensor files, full activation-trace payloads, model weights and unrelated API/service/performance logs are outside this compact bundle. Native BC reports retain the source-trace hashes needed to identify the larger investigation artifacts, but those full traces are not reproduced here. Raw startup/API authentication logs were not collected. Numerical error records, including the unsuccessful initial save attempt, remain identifiable as historical errors rather than substitute baselines.

The complete tensor manifest hashes25,365,357,416 bytes in44 files. These files are necessary to independently recompute full-vocabulary metrics with the frozen probe/assessor tooling in the recipe's `bench/numerical/` directory. The small bundle alone supports provenance and reported-result review; it does not contain enough data to recompute every logit comparison.

To verify the publication copy, run `sha256sum -c SHA256SUMS` inside this directory. The collection and summary scripts are included for transparency. Their source paths are this particular overnight workspace; they are packaging utilities, not the portable benchmark API or a new inference launcher.
