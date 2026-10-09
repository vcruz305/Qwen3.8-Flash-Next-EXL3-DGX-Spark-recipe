# Cooperative GPU lock validation — 9 October 2026

Recipe candidate `254b2b03027f25094845dc31f5f87739f3584d2e` completed one live GPU load with `GPU_LOCK_FILE=/home/cruzspark/redsnow-gpu.lock`, then passed all **28 unchanged tool smoke checks**. The parent matrix controller did not hold the shared GPU lock; the server launcher acquired it and retained descriptor 8 across exec.

The controller checked the actual running server descriptor and Linux `fdinfo` before clients and after measurements. Both checks identify device 66306, inode 2273157 and an advisory write flock. The later release receipt verifies the controller had exited, no GPU owner remained, and the same file could be locked again. Its device/inode/empty contents remained unchanged. The owned server received SIGTERM and exited 0.

The recipe's optional setting defaults to unset. When requested, the launcher refuses a competing cooperative owner before runtime/model/config mutations; the public matrix requires launcher capability and validates actual descriptor ownership, rather than accepting only an environment variable. CPU checks exercise contention, exec lifetime, invalid paths, preview behavior, lock replacement and a selected historical launcher without the capability. The retained independent matrix review records 26 passing checks. The full ARM recipe test output reports **Ran 148 tests; OK (skipped=5)**, and `setup.sh --check` exited 0. We retain the runner's own count wording instead of subtracting skip events into an invented passed count.

The live job used the flat 3.05 model, disk PLE, max batch 4, a 262144-token shared cache, Q8 KV, chunk 2048 and dynamic depth-5 MTP with row budget 8, on engine `24f0dece34f09c8d1e2359d6b3b3f7befef7331b` and Tabby `f650bb5389e0a273549e47d4d26a765760c013e1`. Its 28 client requests were sequential; the profile's four slots do not make this a concurrent throughput experiment. This gate did not replace the canonical published recipe or enable/disable services.

This is a **cooperative ownership mechanism**, not an access-control or GPU-isolation boundary: other software must honor the same persistent lock file. Do not remove/recreate the file while cooperating processes may hold it. The live gate is not a stress test of all possible process races.

## Evidence

- `raw/` contains exact small Spark result/client/resource/config/deployment files, preflight logs, launch record and release proof. Model-view symlinks are recorded as omissions and never followed.
- `source/` contains selected files read directly from Git at the exact tested recipe commit. The controller and recorded runtime/client file hashes match the live result.
- `cpu-review/` preserves earlier local checks and the independent integration review, each with its own source scope.
- `remote-source-manifest.json`, `release-review.json`, `collection-receipt.json` and `SHA256SUMS` bind collection and source identities.
- `summary.json` separates live clients, CPU preflight and release checks.

All copied remote data passed a bounded credential-pattern/key-name scan with zero candidates. The scan does not prove absence of arbitrary unlabeled secrets. CPU source fixtures may contain deliberate fake API-key strings; no credential store was opened. Collection copied only completed small files and performed no API, GPU, service, source or model mutation. Earlier preparation documents remain unchanged, including pending-run wording. Do not automatically execute archived host-specific scripts. Verify file integrity with `sha256sum -c SHA256SUMS`.
