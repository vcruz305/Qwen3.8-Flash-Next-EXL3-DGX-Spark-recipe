# Final service rehearsal on qualification recipe R1

**All four phases passed.** The rehearsal completed at **2026-10-08 14:16:29.948 UTC**; its controller PID 477438 was absent before collection. This archive contains the measured service installation, one controlled failure and automatic restart, an original-runtime rollback, and the return to the final service. It records **eight successful basic API requests**: two in each of three new service invocations and two against the original runtime. This rehearsal did not run the separate 28-case tool suite or make a new throughput measurement.

## Exact sources and publication boundary

| Component | Revision used by this rehearsal |
|---|---|
| Qualification recipe R1 | `3337bc8d64e4befafa2a1aff342e07abbea242ac` |
| Installed engine | `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` |
| Installed Tabby, actual merged main | `f650bb5389e0a273549e47d4d26a765760c013e1` |
| Original rollback engine | `94ba01d50a13fa9ff672473f2d0eef8b51a71e99` |
| Original rollback Tabby | `816c32195887aaecea1c64528f2921566766259b` |

[promotion/fork-promotion-result.json](promotion/fork-promotion-result.json) records successful fork merges and full-tree equality. Engine merge `678ab0dd1c39cb7ba22d577913132f63d4d242f3` has the exact qualified 24f0 tree; the installed pin remains 24f0. Tabby merge f650 has the exact qualified `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb` tree. The installation result separately records the transition from 5a to actual merged main and verifies the tree equality.

**R1 rehearsal and later published-recipe verification are separate phases.** This result does not claim that a future R3 documentation/evidence commit was already exercised. Publication requires recorded runtime/client byte identity, a normal service restart and verification against the actual published revision.

## Observed lifecycle

| Phase | MainPID | InvocationID | NRestarts | State after API checks |
|---|---:|---|---:|---|
| initial | 477472 | `d09c4b0fee854bfea0ed18f4f74aaca5` | 0 | active/running |
| automatic_restart | 478105 | `aeb016385b5a4de9a2d57d4a847fc485` | 1 | active/running |
| returned_to_final | 480312 | `1ddc964e1fcd48d391ffdc6dc6bce217` | 0 | active/running |

The controller sent one SIGKILL to the identified initial unit MainPID. Automatic restart produced a different PID and invocation, with NRestarts increasing exactly from 0 to 1. Each start created its own recorded configuration, deployment and startup metadata under the invocation directory. The original-runtime phase then stopped the new unit, loaded the preserved original installation, passed two API checks and stopped only its token-owned process group with SIGTERM; the recorded exit was 0. The frozen cleanup helper verifies that the owned group is empty before returning. After rollback, the final unit started again with the fresh third invocation shown above. Its restart counter reset to 0 on this explicit start.

The original virtual environment remained at `/home/cruzspark/qwen38-exl3`; `old_import_check` confirms its module import used that canonical original path. The new service uses `/home/cruzspark/qwen38-exl3-20261008` and the recipe at `/home/cruzspark/qwen-spark-recipe`. The original environment was not relocated.

[installation/result.json](installation/result.json) records successful setup `--check`, exit 0, installed file hashes and unit enablement. [installed/service.env](installed/service.env) records the selected settings: loopback `127.0.0.1:8899`, 3.05 pack, four slots, Q8 cache pool 1,048,576 tokens, per-request limit 262,144, chunk 2048, disk n-gram lookup, dynamic MTP depth 5, confidence 0.6 and row budget 8. Loopback authentication was disabled in this deployment. This archive does not claim LAN exposure or a persistent health check after its recorded end time.

## Evidence and log handling

[reports/result.json](reports/result.json) is the completed four-phase result. The three per-invocation records contain actual API replies, source revisions, environment/configuration identity and before/after unit identity. `invocations/` preserves each start/deployment/configuration record. `launch/` preserves the actual launch record and a filtered controller log. `source/` contains the exact controller/helper and recipe source snapshots; recipe files were read using the recorded commit even if the checkout later moved.

**Raw rollback and authentication logs were never transferred.** All logs were reduced to conservative lifecycle excerpts on Spark before transfer. The narrow allowlist omitted all lines in the collected journals, original-server log and launch/setup logs; therefore these excerpts carry no event details. The structured phase and installation records provide the actual outcomes. [collection.json](collection.json) and [installation-collection.json](installation-collection.json) record original and transformed hashes, byte counts and omission/redaction counts. Credential stores were not opened.

`service-evidence-sanitizer-cpu.json` preserves seven synthetic sanitizer checks on predecessor collector SHA `31b1081c1947d378c40da5f44224ed22b2e280ce868913577d78bcd24bc2e747`; it is not a claim of a second full test run on the current collector. The current collector and supplemental collector are archived verbatim and identified in SHA256SUMS. Collection was read-only; it did not issue API calls, change the service or read/hash model tensors or the compiled extension.

Verify the archived files with `sha256sum --check SHA256SUMS` from this directory. The collectors contain host-specific source paths and are evidence-collection utilities, not portable service installers. Review the repository's service deployment guide before adapting the installation to another machine.
