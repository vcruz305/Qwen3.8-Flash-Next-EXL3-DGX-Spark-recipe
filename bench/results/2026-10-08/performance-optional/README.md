# Historical optional performance matrix — October 8, 2026

This archive contains all **16 completed optional jobs** from the historical
engine16ca / Tabbyf4 / recipe218 matrix. It is not the final deployment record.
The controller reported **10/16** jobs passing all
scheduled clients, with **6** jobs retaining tool
semantic failures. The measurements were not rerun or relabeled during archive
preparation.

## Scope and outcome

| Item | Actual completed scope |
|---|---:|
| Single-request benchmark rows | 36 |
| Measured single requests | 102 |
| Concurrent benchmark rows | 18 |
| Measured concurrent requests | 126 |
| Jobs with an eight-case tool subset | 13 |
| Tool checks passed / total | 91/104 |
| Jobs without a tool client | 3 requested-head jobs |
| Matched case/concurrency comparison contracts | 27 |

The tool subset is `auto,strings,typed,parallel_disabled`, in nonstreaming
and streaming modes. It is eight checks per participating job, not the full
28-check suite. The three requested-head jobs are benchmark-only. The
[failed checks](tool-failures.json) preserve 4.05's repeated-value string error,
Cyber's missing string calls, and one Cyber concurrent typed boolean/string
error. A complete timing report from a failed job remains an observation under
that failed functional gate.

The interval is **2026-10-08T11:53:48.871186+00:00** through
**2026-10-08T12:56:46.101871+00:00**. UTC controller timestamps define this
interval; server log timestamps retain the host's original local formatting.

Read the [optional result tables and limits](optional-results.md), the
[complete generated summary](summary/summary.md), and the
[machine-readable summary](summary/summary.json). The tables preserve small
sequential sample sizes, each setting's three concurrent round rates, matching
request hashes and actual token counts. They make no repeated-matrix
confidence or universal timing claim.

## Exact historical sources

| Component | Recorded identity |
|---|---|
| Engine | `16ca20d27c0e4cce15a9bbc131e6d047065395b5` |
| TabbyAPI | `f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6` |
| Recipe | `218bd438225243e795383f10ef7a886b06d5d839` |
| Owned experiment controller SHA256 | `48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586` |
| Frozen 16-job list SHA256 | `90234a6cdd68b2e2a4b260753d4ed31ab19a0aaa81cf3e00f7270b91c97b7c34` |
| Optional summary wrapper SHA256 | `745e8855fbc53942ffcccc391efd77e2e4a3a4357ddf7c642fb8eb2ab5d84498` |
| Frozen base summarizer SHA256 | `d1292bba6e63d8cddbf25afe553c923aabe2a747d8a9f6d064e520376770e660` |

These identities come from the completed reports and their retained
configuration/deployment snapshots. Collection did not substitute the later
engine24f0/Tabby5a installed in the dated runtime. The historical recipe source
files in `inputs/recipe/` were recovered byte-for-byte from commit218 and
matched against each report's recorded SHA256. The [recipe source manifest](inputs/recipe-source.json)
identifies each file.

The shared controls include the qualified-for-this-matrix GDN 1/0/1 compatibility
flags, legacy attention splits, legacy GEMM tiles, mixed Coop off, mixed NOSYNC
on, and shared-expert KSPLIT 1. All explicit and resolved values are retained.
The original per-pack templates were used, including Cyber's original forced
thinking behavior. Final generator changes, the Cyber override, and a combined
selected profile need separate live qualification.

## Recompute the summary without inference

From this archive directory, choose a new destination:

```bash
python3 summary/summarize_optional_performance.py \
  --reports reports \
  --jobs inputs/optional16.json \
  --output recomputed-summary
sha256sum -c SHA256SUMS
```

The optional wrapper hash-checks the adjacent frozen base summarizer before
using its metric and inclusion logic. Both scripts have guarded entry points.
This command reads the saved reports only and does not load a model, contact an
API, or manipulate a server. The recomputed summary records its new generation
time and local paths; measurement values remain those in the input reports.

The exact experiment controller and job list are retained under `inputs/`.
They are historical host-specific source/configuration evidence, not an
instruction to run sixteen loads against an occupied Spark. To reproduce
inference, review the public [matrix workflow](../../../README.md) and
[service lifecycle](../../../../docs/service.md), select the matching
historical checkouts and model paths, and use new output/state directories.
The later public runner has additional template-provenance handling; its bytes
are intentionally distinguished from this frozen controller.

## Integrity, cleanup and resources

[integrity.json](integrity.json) checks completed top-level and attempt records,
raw/canonical hash distinctions, exact source identities, input checks before
launch/before measurement/after measurement, complete clients, and clean
owned-server termination. Every recorded server was alive before controller
cleanup and exited zero after its controller sent SIGTERM. All 27 recorded
case/concurrency comparisons satisfy same-source, single-setting,
request-hash and actual-count matching contracts. Different response hashes
remain visible where present; no output is silently substituted.

[server-log-summary.json](server-log-summary.json) references each complete
server log and selected diagnostic lines. [source-provenance.json](source-provenance.json)
records original remote paths, file sizes, mtimes and raw SHA256 values;
duplicate client stdout logs were omitted with hashes and bounded diagnostic
summaries because their structured JSON reports are retained. The archive
contains no weight or activation tensors.

[memory-summary.json](memory-summary.json) derives sparse observations from
each attempt's `resources.json`. Initial swap allocation is retained to avoid
attributing earlier jobs' swap to a later load. These are not guaranteed peaks
or swap-I/O measurements. Corrected unloaded advisories are in
`inputs/memory-advisory/`; [the separate placement audit](inputs/ple-ram-transient-review.json)
and its provenance explain table allocations and bounded shard transients.
The first advisory invocation used the wrong helper path and produced no
usable measurements; only the corrected advisories are included here.

Archive preparation initially expected 128 tool checks by assuming every job
had a tool client. Its integrity assertion failed before publishing any
integrity success record. The frozen job list shows the three requested-head
jobs omit that client, so the correct total is **104**. This correction changes
the archive's scope description only: no job, fixture, response or measurement
was rerun or modified. The exact initial and corrected preparation scripts and
the correction record are retained under `preparation/`.
