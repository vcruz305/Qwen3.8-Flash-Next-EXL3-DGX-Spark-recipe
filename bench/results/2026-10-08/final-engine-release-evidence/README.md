# Final engine release prerequisites — 24f0/5a

This archive preserves the successful hosted CI result and the final source and
configuration reviews. It does not relabel a pending service configuration as a
live deployment. Final Spark setup, numerical, API/performance and service checks
have separate result archives.

## Successful CI

[GitHub Actions run 37776591337](https://github.com/vcruz305/exllamav3/actions/runs/37776591337)
completed successfully for engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`
on 2026-10-08. The single Linux Python 3.12/CUDA12.8/sm120 job built and installed
the wheel, imported it from a neutral directory, and passed both CPU commands:

- **17 tests** in the existing CPU selection, 0.71 seconds.
- **320 tests plus 186 subtests** in the isolated source selection, 7.25 seconds.
  The 320 includes all 77 finite-budget/natural-ending tests; 77 is not an
  additional count. This command includes `--noconftest` and the UMA, DFlash,
  sampler and producer-budget modules.

The original decoded job log is preserved in [ci-job.log](ci-job.log); the
[short excerpt](ci-relevant.log) copies its relevant trailing lines without
changing their contents. API metadata is retained in `ci-run.json` and
`ci-jobs.json`. GitHub checked out the normal synthetic PR merge
`f405dd28bdcb71b6eff65e0c2a24f836f18010bd`. Its entire Git tree is
`2076ededf18de866cac9fb7b1c0c7215e4af8ff6`, identical to the final 24f0 head.
[Tree proof](ci-merge-tree-proof.json) records both the API and local Git result.

## Source and deployment reviews

The [upstream record](upstream-heads.json) verifies the actual official default
branches at 12:56:10 UTC. ExLlama master remained 151539c and Tabby main remained
2fd6cc7; both are ancestors of the final 24f0/5a source pair. No source update was
needed after this read-only check.

The [transition review](qualified-transition-source-review.json) records the one
staging-integrity finding and its resolution: hashes are rechecked after waiting
for measurements and again before setup/quality execution. This was a source
review, with no additional tests or runtime action by the reviewer.

The [service environment review](service-env-and-405-memory-review.json) compares
`service.env.candidate` with the measured 3.05 concurrent row-budget 8 configuration.
Tuning matches, with the intended engine update to 24f0 and the same default source
repository. CPU affinity/toolchain are explicit; there is no added OMP/thread
tuning or prompt-template override. The file remains a candidate pending final
live qualification and the separate service rehearsal. The combined review also
retains the 4.05 disk/RAM memory observation, with its sparse-sampling and swap-I/O
limits stated explicitly.

The saved PR body and publication readback are the draft snapshot from 12:25:42,
before CI completed; later PR edits are not part of this snapshot. Superseded
workflow statuses are retained separately. All current proof and artifact hashes
are indexed in [evidence.json](evidence.json) and `SHA256SUMS`.
