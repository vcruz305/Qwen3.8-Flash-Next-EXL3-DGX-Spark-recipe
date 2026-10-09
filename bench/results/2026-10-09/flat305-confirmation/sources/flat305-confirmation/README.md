# Scheduled flat3.05 RAM confirmation

This is a bounded diagnostic preparation, not a deployment or a completed performance result. The root owner alone launches it after a cooperative GPU handoff. It copies the completed, reviewed 4.05 controller 73d179bd into a new directory and changes only the pack/audit key, RAM placement, A/C cell selection and code/prose corpus. The old controller and evidence stay unchanged.

Two fresh sequential server loads use the original 3.05 pack at `/home/cruzspark/models/flashnext-exl3-3.05bpw`:

- A: original engine 94ba01d / original Tabby 816c321 and preserved original venv.
- C: qualified engine 24f0dec / published Tabby f650bb5 and preserved final venv.

There is no B hybrid cell. The retained `--old-tabby` argument is an unused compatibility path from the parent CLI; it must point to an existing isolated path but is not imported or launched by either cell. Both cells use exact public recipe a8c72bd, RAM placement, Q8 KV, context/cache 262144, one request slot, chunk 2048, dynamic MTP depth 5/confidence 0.6, head 65536, row budget 0, big-core affinity and the same numerical compatibility settings as the qualified stack. No package/source installation, environment repair or kernel change occurs.

The unchanged public benchmark client sends the original code and prose payloads: one warmup and three measured repeats per case, 400 actual output tokens, cold cache, `overnight-v1` salts, thinking false, temperature 0, top-k 1 and seed 0. There are eight requests per cell, sixteen total, twelve measured. Original request hashes, prompt/cache/output token counts and finish reasons are enforced. Response hashes and accepted/rejected draft totals remain evidence, not equivalence assumptions. This pair compares complete supported stacks; package, output-trajectory, sequential-order and sparse thermal limits must be reported.

The original305 report is retained byte-for-byte in `testdata/baseline-305.json`, with a minimal selected-case contract in `original-fixture.json`. Twenty CPU checks passed, including reconstruction of all eight original request hashes, actual usage validation, AST identity of inherited lifecycle/source/ownership/import/pip functions and real harmless child signal/cleanup tests. The independent peer reran all twenty checks successfully. These CPU tests make no model/API/GPU calls.

Historical A's actual `pip check` remains exit 1 for the exact known `nvidia-cusparselt-cu13 0.8.1` SBSA wheel-tag metadata defect. It is admitted only as a historical attribution control after a fresh read-only check against the independent library/package proof. It is not a passed full setup check or an eligible deployment. Any different dependency error stops the cell; B/C-style final dependencies must pass. No old metadata is repaired.

Ownership, shared flock, clean source/package/model/header/order checks, child registration, per-cell lock inode checks, signal handling and owned process cleanup are inherited from 73d. REXL3 and Qwen supervisors must remain inactive and the Qwen unit disabled. A failed cell, concurrent owner, changed input or uncertain cleanup stops the sequence. Use a unique output path. The controller must not be run automatically from this evidence directory.
