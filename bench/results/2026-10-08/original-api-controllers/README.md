# Historical original API baseline controllers

These files are exact archival copies of the controllers used for the original overnight API baseline. Their bytes match both the preserved WSL files and the remote copies retained on Spark when checked read-only at 2026-10-08 10:53 UTC. `source-provenance.json` records the paths, byte lengths and SHA256 hashes.

| Source | Scope | SHA256 |
| --- | --- | --- |
| `baseline_matrix.py` | Original 4.05, SAGE 4.15, Cyber 3.87 single/disk runs, following the separately launched 3.05 baseline. | `272db7fdbe42d61cbf988d8f40f5463ff304a7afff701c7eb77ca037aa283788` |
| `baseline_305_prefill_controller.py` | Later original 3.05 single/RAM prefill completion with the same prefill payload. | `6f3e227a7e78a83ee3bed5267150c38b7e0c78db39c89e9bb932cc88d2bd74ac` |

The original recipe is `6fbc0a2f1a0a8ee3d94830bb607ec36aa370ea49`, engine `94ba01d50a13fa9ff672473f2d0eef8b51a71e99`, and Tabby `816c32195887aaecea1c64528f2921566766259b`. The controller statuses did not embed a source-file hash at launch. The archive proves the current retained remote/local byte match; it does not retroactively claim that an execution-time source hash existed.

The original statuses are preserved byte-for-byte. The three-pack matrix records successful benchmark/prefill exits and failed tool-smoke exits on all three original servers. Its overall completion timestamp is not a claim that tool checks passed. The separate 3.05 prefill status records a completed client with exit 0 and server cleanup. The original result files remain unchanged in their separate historical result archive.

These scripts are historical execution evidence, not the portable public runner. They contain the original account paths, fixed output names, ambient-environment assumptions and process handling. They execute at module top level: importing them or invoking --help would run their workflow. In particular, the matrix controller waits for the old 3.05 tools file, reads the retained baseline PID and writes fixed historical status/output names. Do not casually rerun them against the active machine or preserved results. They have only been parsed as source during archival; they were not imported or executed.

The modern `bench/run_matrix.py` requires the updated launcher's deployment snapshot and strict client contract. The original 6fbc launcher emits no such snapshot and treats nonempty DRY_RUN values differently, so pointing the modern runner at the old recipe does not reproduce this baseline. Recreating the historical experiment requires an isolated equivalent original environment, matching original recipe/runtime/client bytes and fresh evidence destinations, with any adaptations recorded separately. Keep the archived sources unchanged so their scope and original behavior remain reviewable.
