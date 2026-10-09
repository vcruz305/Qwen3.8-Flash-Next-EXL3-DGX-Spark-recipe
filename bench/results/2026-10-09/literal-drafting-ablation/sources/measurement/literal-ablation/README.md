# Literal drafting ablation, prepared October9

This diagnostic runs the eight unchanged original literal-tool requests twice: first with `DRAFT_MODE=disabled`, then with `DRAFT_MODE=mtp`. Each cell gets a fresh server, unique state directory and original four-case budgeted plus four-case unbudgeted client reports. All other job settings are identical. The exact sources are engine24f0, merged Tabbyf650 and published recipea8c72bd. No engine, server, recipe, model or fixture source is changed.

The controller holds the existing `/home/cruzspark/redsnow-gpu.lock` across both cells. It requires `rexl3-manager.service` inactive with MainPID0, `qwen38-exl3.service` inactive/disabled, and no CUDA process except its own server once loaded. Only the parent operator may perform the incumbent supervisor's documented drain/stop and later restoration. The controller does not stop an unrelated process, change service enablement or select a production configuration.

The current timeline observer records bounded processed-sample/guard/forced-output/phase events as well as native/backend completion strings. Processed samples can subsequently be rewound: event labels and checkpoint state do not claim every observed ID belongs to the final output. The independent validator requires complete paired events, ordered timestamps and zero drops; absence of a phase transition is valid diagnostic evidence. No throughput claims should be made from instrumented timing.

`frozen/` contains seven hash-bound modules. The old literal assessor is reused with explicit ENGINE/TABBY/OBSERVER_SHA global bindings; all other checks remain. Its source and the new controller source are recorded separately. The frozen final wrapper's obsolete prior-day setup-status requirement is replaced with an actual fresh `setup.sh --check`, engine import/capability check and exact clean source checks. These are read-only runtime checks, not a rerun of the previous CPU suites. Every requested body remains byte-identical to the original client; effective top_p remains the existing0.95 YAML default. The timeline observer's numeric metadata recorder correctly preserves ScalarFloat values without converting arbitrary objects/tensors.

Capture completeness and model semantics are separate. Valid semantic failures are retained and the second cell still runs. Invalid capture, signal interruption, input drift or uncertain cleanup stops the sequence. Each output path is claimed exclusively; there are no retries or resume-in-place. A nonzero exit does not mean cleanup was skipped: inspect the retained lifecycle and outer result.

After the parent has stopped/drained the incumbent and verified the host free, transfer this entire directory (excluding `__pycache__`) and the separate `literal-ablation-observer-timeline` directory under `/home/cruzspark/qwen-followup-20261009/`, preserving their relative paths. No previous source files need modification. Then run:

```bash
python3 /home/cruzspark/qwen-followup-20261009/measurement/literal-ablation/run_literal_ablation.py \
  --jobs /home/cruzspark/qwen-followup-20261009/measurement/literal-ablation/jobs.json \
  --output /home/cruzspark/qwen-followup-20261009/results/literal-ablation-24f0-f650 \
  --runtime /home/cruzspark/qwen38-exl3-20261008 \
  --observer-dir /home/cruzspark/qwen-followup-20261009/literal-ablation-observer-timeline/observer \
  --model-inputs /home/cruzspark/qwen-followup-20261009/measurement/literal-ablation/model-inputs.json \
  --client-root /home/cruzspark/qwen-overnight-20261008 \
  --manager-unit rexl3-manager.service \
  --gpu-lock /home/cruzspark/redsnow-gpu.lock \
  --ready-timeout 300 --client-timeout 300
```

Expected ordinary duration is roughly3–6minutes for two loads and16 short requests; it is an estimate, not a result or guarantee. Explicit deadlines bound each load/client; the reviewed lifecycle allows up to90seconds per owned-process cleanup. If the parent imposes a whole-window deadline, send SIGTERM and allow the cleanup path to complete before escalation or supervisor restoration. Never restore the manager while owned cleanup remains uncertain.

The existing Oct8 separate client environment is used only for the frozen lifecycle's client-package inventory; the literal client itself uses the chosen runtime Python. The model input snapshot binds the305 pack's configuration, tokenizer, loader order, file metadata and current header hashes. It does not hash or copy weight payloads.

CPU validation:14 tests, all passed. They exercise actual harmless flock contention/lifetime, owner-state rejection, input/header drift, truthful preflight checks and the difference between semantic failure and capture/cleanup failure. No Spark/API/CUDA execution was part of that test run. Source review also retains the previously reviewed frozen process/session cleanup and server-only observer injection. Verify this bundle with `sha256sum --check SHA256SUMS`; the observer bundle has its own manifest.
