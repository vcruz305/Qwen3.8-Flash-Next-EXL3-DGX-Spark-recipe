# Bounded user-span BPE diagnostic — prepared, not launched

This is one fresh-server, target-only (`DRAFT_MODE=disabled`) experiment on the original eight visible literal-copy requests. The original completed disabled/MTP pair remains in its separate output directory and is not modified. The new model input deliberately changes: two native added-token IDs wholly inside the one exact synthetic user message expand into ordinary BPE IDs, taking the prompt from 348 to 352 tokens. The visible request JSON and rendered prompt bytes remain unchanged; template control IDs remain native. This is a diagnostic about input representation, not an unchanged-input comparison, a production fix, or a throughput benchmark.

The external hook runs in the tokenizer before context-length validation and cache/generation input bookkeeping. Only the exact hash-bound synthetic prompt may be rewritten. Other prompt text, generated output, forced reasoning tails, parser behavior, sampler settings, tool schemas and client expectations remain unchanged. The original literal client still requires the exact `<think>literal</think>` argument. Both budgeted and unbudgeted requests, required and named tool choice, and stream/nonstream are retained (eight combinations).

The source stack remains recipe `a8c72bdf811646f413fdc03a4e49811a2753d0cf`, engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`, and Tabby `f650bb5389e0a273549e47d4d26a765760c013e1`. This controller is a separately hashed derivative of the frozen first-ablation controller `d9b16b7d0fe079556c37a7a2bc70107c17ce0c2e4dadc5e36a7a1d3e6c405b57`. Its `frozen/` modules preserve the previously reviewed owned lifecycle, source checks, client and raw/timeline assessors byte-for-byte. Only the new input-proof validator is added. The config explicitly enables `diagnostic_user_bpe` and names the audited 3.05 `tokenizer.json`; all three observer Python files and the exact config are retained and checked in deployment-side provenance.

The original model metadata/header audit is reused. It checks config/tokenizer content hashes, literal loader enumeration order, weight filename/size/mtime identities and bounded safetensors header hashes. It does not hash weight payloads or prove that unobserved weight bytes were never modified. All control inputs and model metadata are rechecked before launch and after measurements.

The owner must first stop and drain `rexl3-manager.service` using its documented lifecycle. This script never stops a manager or foreign process, changes a source ref, installs a package, or enables/disables a unit. It acquires `/home/cruzspark/redsnow-gpu.lock` itself for the entire diagnostic, requires the manager inactive and Qwen inactive/disabled, refuses another CUDA owner, and stops only its token/PGID-verified server and clients. Do not invoke it while another cooperating owner holds that lock. The caller owns restoration of the originally enabled REXL3 manager afterward.

The existing canonical recipe/runtime directories are read but not edited. Every launch has fresh state and a unique output root; existing output is refused. Fresh `setup.sh --check`, actual engine/backend source/import and native-capability checks are recorded. This does not claim the prior-day CPU suites were rerun.

Root-scheduled launch, only after observer/controller review and checksum verification:

```bash
python3 /home/cruzspark/qwen-followup-20261009/measurement/literal-user-bpe/run_literal_user_bpe.py \
  --jobs /home/cruzspark/qwen-followup-20261009/measurement/literal-user-bpe/jobs.json \
  --output /home/cruzspark/qwen-followup-20261009/results/literal-user-bpe-disabled-24f0-f650 \
  --runtime /home/cruzspark/qwen38-exl3-20261008 \
  --observer-dir /home/cruzspark/qwen-followup-20261009/literal-bpe-observer/observer \
  --model-inputs /home/cruzspark/qwen-followup-20261009/measurement/literal-user-bpe/model-inputs.json \
  --client-root /home/cruzspark/qwen-overnight-20261008 \
  --manager-unit rexl3-manager.service \
  --gpu-lock /home/cruzspark/redsnow-gpu.lock \
  --ready-timeout 300 --client-timeout 300
```

Expected normal duration is approximately 2–4 minutes; deadlines and bounded owned cleanup remain authoritative. No MTP confirmation cell is authorized by this job list. A separate new experiment may be prepared if this diagnostic warrants one.

The result records `capture_valid` independently of semantic `passed`. Capture validity requires the original eight request combinations/response identities, eight complete native/backend traces, complete bounded producer timelines, exact input-span/token-ID proofs, original348/changed352 counts, and actual API `prompt_tokens=352`. Cached-token counts are retained rather than required to be zero because the original serial fixture sequence reuses prefixes. Generated response hashes and exact client failures are preserved. A valid capture can therefore end with exit1 when literal copying fails; invalid capture/cleanup/interruption ends with exit2. Exit0 requires both valid capture and all eight original semantic checks passing. No evidence should be relabeled as a successful model improvement merely because the capture is complete.

The frozen derived controller passed 17 CPU-only tests on 2026-10-09 (including actual cooperative-lock contention/lifetime and original/duplicate API usage rejection); exact test output and source identities are retained in `cpu-validation.json` and `cpu-validation.log`. An independent peer reviewed the controller diff against the first-ablation source and found no remaining concrete lifecycle or assessment blocker. This is preparation evidence, not a live result.
