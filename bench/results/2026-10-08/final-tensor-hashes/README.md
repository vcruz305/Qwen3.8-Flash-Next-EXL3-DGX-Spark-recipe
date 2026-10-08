# Deferred numerical tensor hashes

Six retained numerical `.safetensors` artifacts were hashed after all five final API jobs and their measured clients had exited. The additional bounded literal diagnostic had also finished and released its owned server before these reads began. This archive contains only the small inventory, release records, source, and digests; it does not contain the tensors.

## Timing and measurement isolation

The final API batch finished at **2026-10-08 13:55:37.681129 UTC**. Its PID 430098 was absent, and the source checked the completed result hashes and the finished/absent processes for every short benchmark, the concurrent benchmark, and both long-context clients. Each owned group was empty.

The separate literal diagnostic finished at **13:58:13.466164 UTC**. At **13:59:29.903527 UTC**, its controller 470760, server 470805, and all three client processes were absent; their owned groups were empty. Its server stopped with SIGTERM and exit 0. The diagnostic's semantic and capture outcomes remain in its own report and are not prerequisites for hashing.

The six reads ran from **13:59:46.464057 to 13:59:55.321495 UTC**, after both releases. They streamed exactly **9,012,969,536 bytes (8.393982 GiB)** in sequential 16 MiB reads through `hashlib.sha256`. There was no `safetensors` or Torch import, tensor deserialization, normalization, or model-weight read.

## Retained file identity

The original path, byte size, modification time in nanoseconds, device, and inode were recorded at 13:43:51 UTC in [the inventory](deferred-final-tensor-hash-inventory.json). The hashing helper checked all four metadata values before and after reading each regular, non-symlink file, including the opened file descriptor. Every check passed and every byte count matched the inventory.

| Retained artifact label | Bytes | SHA256 |
| --- | ---: | --- |
| candidate-305-ksplit1 | 643,789,840 | `5857aa90a49f9b1473b38058b4d38610baca937977f81297f2e751e4b80b9ab9` |
| candidate-405-ksplit1 | 643,789,840 | `d1c45e6735197157f380ba14bc6b6c448e3a51157d9c0d805a1d68ca1bcbccbe` |
| candidate-cyber387-ksplit1 | 643,789,848 | `20ef533cd6e2e0bf0f4c2d42ebc3ac0492134710c07f8fe72abe99f5038ec36e` |
| candidate-305-chunk4096-final | 643,789,840 | `fb33b306be169636b2468c5c71919b0540febd0d0d8c66032075d98eb351cfbc` |
| baseline-305-chunk2048-final-pair | 3,218,905,080 | `37ad6f96c6113811646d58ec9e507fedd5a47554c986d35c3226726081793292` |
| candidate-305-chunk2048-final-pair | 3,218,905,088 | `e049518de61c16c6850a3af5323b237688241575890a2903a1f00f11c5d50830` |

Exact remote paths and per-file durations are in [hashes.json](hashes.json). The complete record SHA256 is `8a2582c903dd9011b5a79f23c3bdd25fe3ba3818c4b6d07d9b7ec9f8822c2640`.

A file digest identifies serialized bytes. It does not itself establish numerical equivalence or correct model behavior. The matched 2048 baseline and candidate files have different file sizes; their comparison result is the separately recorded zero-difference numerical assessment, not equal file hashes. The unselected 4096 comparison and the qualified 2048 pair retain their original results in [final-numerical-chunks](../final-numerical-chunks/README.md). The three earlier K-split records remain evidence for their exact 16ca runtime cohort.

## Archive contents

- `hashes.json`: final six-file digests, exact metadata, read counts, timestamps, and completed measurement-process checks.
- `deferred-final-tensor-hash-inventory.json`: immutable pre-read inventory and cohort identity.
- `diagnostic-release.json` and `diagnostic-result.json`: the additional diagnostic cleanup gate and its exact source report.
- `final-api-status.json`: the exact completed five-job batch status referenced by the hash record.
- `hash_deferred_final_tensors.py`: the frozen source used on the Spark. It has hardcoded original artifact identities and ownership gates, and writes only its new result directory. It is a historical host-specific operation, not a general validation client.

No live helper needs to run to verify this small archive. Check its retained bytes with `sha256sum -c SHA256SUMS`. Recomputing a tensor digest requires the original remote artifacts and should be scheduled outside performance measurements. No repository or runtime source was changed by hashing.
