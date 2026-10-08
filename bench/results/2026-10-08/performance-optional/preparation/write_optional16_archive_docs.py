#!/usr/bin/env python3
"""Write documentation for the completed optional archive; main report is untouched."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil

BASE = Path("/home/vcruz/src/qwen-overnight-20261008")
A = BASE / "recipe/bench/results/2026-10-08/performance-optional"
summary = json.loads((A / "summary/summary.json").read_text())
info = json.loads((A / "integrity.json").read_text())
rows = {(r["label"], r["benchmark"], r["case"]): r for r in summary["bench_rows"]}
concurrency = {(r["label"], r["concurrency"]): r for r in summary["concurrency_rows"]}
jobrows = {r["label"]: r for r in summary["completed_jobs"]}
memory = {r["label"]: r for r in json.loads((A / "memory-summary.json").read_text())["jobs"]}
percent = lambda a, b: f"{(b / a - 1) * 100:+.2f}%"
fmt = lambda x: f"{x:.2f}"
tool = lambda label: ("Not scheduled" if jobrows[label]["tools"] is None else
                      f'{jobrows[label]["tools"]["passed"]}/{jobrows[label]["tools"]["total"]}')
rate = lambda label, case: rows[label, "bench-1", case]["medians"]["server_decode_tok_s"]
short_cases = ("code", "devops", "prose")

head = "| Requested MTP head N | Code tok/s | DevOps tok/s | Prose tok/s | Tool checks |\n|---|---:|---:|---:|---|\n"
for label, title in (
    ("12-control-305-requested-head65536", "65,536 (control)"),
    ("12-tune-305-requested-head32768", "32,768"),
    ("12-tune-305-requested-head131072", "131,072"),
):
    head += f"| {title} | " + " | ".join(fmt(rate(label, c)) for c in short_cases) + f" | {tool(label)} |\n"
static_label = "13-tune-305-static-draft5-device"
static_values = ", ".join(fmt(rate(static_label, c)) for c in short_cases)

concurrent = "| Comparison | Concurrent requests | Control tok/s | Variant tok/s | Change | Tool checks, control / variant |\n|---|---:|---:|---:|---:|---|\n"
for old, new, title in (
    ("09-control-305-concurrent-disk", "09-tune-305-concurrent-ngram-ram", "3.05 disk → RAM, row budget 0"),
    ("10-control-415-concurrent-disk", "10-tune-415-concurrent-rowbudget8", "SAGE disk, row budget 0 → 8"),
    ("11-control-cyber387-concurrent-disk", "11-tune-cyber387-concurrent-rowbudget8", "Cyber disk, row budget 0 → 8"),
):
    for n in (1, 2, 4):
        a = concurrency[old, n]["end_to_end_tok_s_median"]
        b = concurrency[new, n]["end_to_end_tok_s_median"]
        concurrent += f"| {title} | {n} | {a:.4f} | {b:.4f} | {percent(a, b)} | {tool(old)} / {tool(new)} |\n"

ram_short = "| Pack | Code disk → RAM tok/s (change) | DevOps disk → RAM tok/s (change) | Prose disk → RAM tok/s (change) | Tool checks, disk / RAM |\n|---|---:|---:|---:|---|\n"
ram_long = "| Pack | Actual prompt tokens | Prefill disk → RAM tok/s (change) | Decode disk → RAM tok/s (change) | Load disk → RAM, seconds |\n|---|---:|---:|---:|---:|\n"
ram_memory = "| Pack | Minimum sampled MemAvailable, disk / RAM GiB | RAM swap used at start / maximum, GiB | Largest observed increase from RAM start, GiB |\n|---|---:|---:|---:|\n"
for stem, pack in (("06", "405"), ("07", "415"), ("08", "cyber387")):
    title = {"405": "Flat 4.05", "415": "SAGE 4.15", "cyber387": "Cyberfrost 3.87"}[pack]
    old, new = f"{stem}-control-{pack}-disk", f"{stem}-tune-{pack}-ngram-ram"
    cells = []
    for case in short_cases:
        a, b = rate(old, case), rate(new, case)
        cells.append(f"{a:.2f} → {b:.2f} ({percent(a, b)})")
    ram_short += f"| {title} | " + " | ".join(cells) + f" | {tool(old)} / {tool(new)} |\n"
    a, b = rows[old, "bench-2", "code"], rows[new, "bench-2", "code"]
    prompt = sorted(set(a["actual_prompt_tokens"]))
    assert len(prompt) == 1 and prompt == sorted(set(b["actual_prompt_tokens"]))
    cells = []
    for key in ("server_prefill_tok_s", "server_decode_tok_s"):
        x, y = a["medians"][key], b["medians"][key]
        cells.append(f"{x:.3f} → {y:.3f} ({percent(x, y)})")
    ram_long += f"| {title} | {prompt[0]:,} | " + " | ".join(cells) + f' | {jobrows[old]["load_wall_seconds"]:.2f} → {jobrows[new]["load_wall_seconds"]:.2f} |\n'
    a, b = memory[old], memory[new]
    ram_memory += f'| {title} | {a["min_mem_available_gib"]:.3f} / {b["min_mem_available_gib"]:.3f} | {b["initial_swap_used_gib"]:.4f} / {b["max_swap_used_gib"]:.4f} | {b["max_increase_from_initial_swap_gib"]:.4f} |\n'

fragment = f"""### Optional placement, draft-head and concurrent checks

The separate optional matrix completed all **16 jobs** between
**{info['first_job_started_at_utc']}** and **{info['last_job_finished_at_utc']}**.
It measured **102 single requests**, **126 concurrent requests**, and
**104 tool checks: 91 passed and 13 failed**. Ten jobs passed all their scheduled
clients; six retained semantic tool failures. The three MTP-head jobs scheduled
benchmarks only and provide no tool qualification.

These are historical measurements of engine
`16ca20d27c0e4cce15a9bbc131e6d047065395b5`, Tabby
`f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6`, and recipe
`218bd438225243e795383f10ef7a886b06d5d839`. They precede the final producer
reasoning fixes and Cyber template override. The later combined deployment
requires its own qualification. The [optional archive](bench/results/2026-10-08/performance-optional/README.md)
retains every completed report, failed tool result, source identity, resource
sample and shutdown outcome.

#### Requested MTP head size and static drafting

The flat 3.05 RAM runs retained dynamic depth 5 and changed only the requested
`EXL3_MTP_HEAD_N`. Each short case had one warmup and three measured requests
with 400 actual output tokens and zero reused prompt tokens.

{head}
The startup log confirms the main-model MTP component was used, but does not
report effective loaded head width; N is therefore labeled **requested**.
The recorded source and configuration define eligibility. These small timing
differences were not accompanied by a separate repeated-matrix confidence
estimate or tool checks.

The separately scheduled static-depth-5/device-drafting job recorded
**{static_values} tok/s** for code, DevOps and prose, respectively, and
**{tool(static_label)}** tool checks. The optional summarizer does not assign
that job a matched preceding control; it is an observation, not a claimed
isolated speedup.

#### Concurrent placement and row-budget comparisons

These medians are **whole-batch completed output tokens divided by batch wall
time**, including prefill. Each setting has three measured rounds after a
warmup at 1, 2 and 4 requests. Every request returned 256 actual output tokens,
with zero reused prompt tokens; actual prompts were 849 tokens for 3.05/SAGE
and 958 for Cyber. Each concurrent profile reserved a 1,048,576-token shared
pool, a 262,144-token per-request cap and a maximum batch size of 4.

{concurrent}
All compared per-round/stream request hashes and actual token counts match.
The archive retains the three individual rates and response-hash comparisons.
The 3.05 RAM medians and SAGE row-budget medians show mixed changes with
overlapping round ranges. Cyber's positive two/four-request rates remain
historical observations under failed tool gates: both jobs failed the two
string fixtures, and row budget 8 also returned string `"True"` where the
streamed typed fixture required a boolean. These rows do not establish a
general mixed-pack row-budget benefit.

#### Larger-pack RAM placement

For each pair, all **11 measured requests** have matching wire-request hashes,
actual prompt/output/cache counts, and response hashes. The short cases have
three measured requests each and 400 output tokens. The longer prompt has two
measured requests and 256 output tokens. Each case has one separate warmup.
The original Cyber template remains in place for these runs.

{ram_short}
{ram_long}
RAM placement gave modest short-decode gains in these runs, while longer-prompt
prefill was slower for all three packs and startup took longer. Only SAGE
passed both eight-check tool subsets; neither placement fixed the 4.05 or
Cyber string fixture. These results do not establish a universal RAM speedup.

{ram_memory}
Memory figures are sparse ten-second `/proc/meminfo` observations, not true
allocation peaks or swap-I/O measurements. Flat 4.05 RAM accumulated about
2.236 GiB of allocated swap. SAGE and Cyber started with already allocated swap;
their full maximum must not be attributed to the current job. The helper's
10 GiB allowance covers runtime overhead; it is not an additional requirement
to leave 10 GiB of MemAvailable after loading.

The source placement audit explains the large RAM-table increment without a
second full-table CPU copy: flat 4.05 reads directly into the final allocation;
SAGE/Cyber have a roughly 0.284 GiB one-shard transient. The two pinned lookup
sets already exist in disk mode. The archive includes the corrected unloaded
advisories and the separate source audit; neither replaces observed host
resource behavior.
"""
(BASE / "optional16-report-fragment.md").write_text(fragment)
(A / "optional-results.md").write_text(fragment.replace(
    "(bench/results/2026-10-08/performance-optional/README.md)", "(README.md)"))

readme = fr"""# Historical optional performance matrix — October 8, 2026

This archive contains all **16 completed optional jobs** from the historical
engine16ca / Tabbyf4 / recipe218 matrix. It is not the final deployment record.
The controller reported **{info['whole_jobs_passed']}/16** jobs passing all
scheduled clients, with **{info['whole_jobs_failed']}** jobs retaining tool
semantic failures. The measurements were not rerun or relabeled during archive
preparation.

## Scope and outcome

| Item | Actual completed scope |
|---|---:|
| Single-request benchmark rows | {len(summary['bench_rows'])} |
| Measured single requests | {info['single_request_measured_requests']} |
| Concurrent benchmark rows | {len(summary['concurrency_rows'])} |
| Measured concurrent requests | {info['complete_concurrency_measured_requests']} |
| Jobs with an eight-case tool subset | {info['jobs_with_tool_checks']} |
| Tool checks passed / total | {info['tools_passed']}/{info['tools_total']} |
| Jobs without a tool client | 3 requested-head jobs |
| Matched case/concurrency comparison contracts | {info['matched_case_and_concurrency_contracts']} |

The tool subset is `auto,strings,typed,parallel_disabled`, in nonstreaming
and streaming modes. It is eight checks per participating job, not the full
28-check suite. The three requested-head jobs are benchmark-only. The
[failed checks](tool-failures.json) preserve 4.05's repeated-value string error,
Cyber's missing string calls, and one Cyber concurrent typed boolean/string
error. A complete timing report from a failed job remains an observation under
that failed functional gate.

The interval is **{info['first_job_started_at_utc']}** through
**{info['last_job_finished_at_utc']}**. UTC controller timestamps define this
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
"""
(A / "README.md").write_text(readme)
prep = A / "preparation"
prep.mkdir(exist_ok=False)
current = (BASE / "finalize_optional16_integrity.py").read_text()
original = current.replace(
    'assert measured == 102 and concurrent == 126 and tools_total == 104\n    assert sum(bool(j.get("tools")) for j in jobs) == 13',
    'assert measured == 102 and concurrent == 126 and tools_total == 128')
original = original.replace(
    '\n            "jobs_with_tool_checks": sum(bool(j.get("tools")) for j in jobs),\n            "jobs_without_tool_checks": [j["label"] for j in jobs if not j.get("tools")],', '')
assert hashlib.sha256(original.encode()).hexdigest() == "b1b6fd2eb67cd10bd386891b6a30701037b1f39422e8709b2f7b198903869dc6"
(prep / "initial-count-assumption.py").write_text(original)
shutil.copyfile(BASE / "finalize_optional16_integrity.py", prep / "finalize_optional16_integrity.py")
shutil.copyfile(BASE / "archive_optional16.py", prep / "archive_optional16.py")
shutil.copyfile(__file__, prep / "write_optional16_archive_docs.py")
(prep / "count-correction.json").write_text(json.dumps({
    "scope": "Archive-preparation assertion correction only; no experiment rerun",
    "initial_script_sha256": hashlib.sha256(original.encode()).hexdigest(),
    "initial_expected": {"single_measured_requests": 102, "concurrent_measured_requests": 126, "tool_checks": 128},
    "initial_failure": "AssertionError at expected measured=102, concurrent=126, tools_total=128; no integrity success record written",
    "actual_job_derived": {"single_measured_requests": 102, "concurrent_measured_requests": 126,
                          "tool_checks": 104, "tool_passed": 91, "tool_failed": 13, "jobs_with_tool_checks": 13},
    "benchmark_only_jobs": info["jobs_without_tool_checks"],
    "historical_jobs_and_reports_changed": False,
    "matrix_rerun": False
}, indent=2) + "\n")
print(json.dumps({"archive": str(A), "fragment": str(BASE / "optional16-report-fragment.md"),
                  "counts": {k: info[k] for k in ("whole_jobs_passed", "whole_jobs_failed", "tools_passed", "tools_total")}}, indent=2))
