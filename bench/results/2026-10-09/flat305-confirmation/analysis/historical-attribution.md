# Historical flat-K performance attribution

Read-only analysis of the completed Oct8 reports. The data justify a controlled follow-up, not an attribution to a specific patch.

| Pack | Corpus | Original median tok/s | Primary median tok/s | Final median tok/s | Primary same response+draft counts |
|---|---|---:|---:|---:|---:|
| 305 | code | 82.49 | 80.05 | 80.15 | 0/3 |
| 305 | devops | 74.11 | 74.01 | 70.95 | 0/3 |
| 305 | prose | 55.93 | 53.56 | 53.32 | 0/3 |
| 405 | code | 77.02 | 76.18 | 75.85 | 3/3 |
| 405 | devops | 67.76 | 66.37 | 66.45 | 3/3 |
| 405 | prose | 48.57 | 47.64 | 47.46 | 3/3 |

The 4.05 primary control matches all9 original answers and accepted/rejected draft counts. Its server decode time is60–200ms longer per400-token request. Code differences are60/80/60ms; DevOps110/120/130ms; prose200/130/160ms. Server durations have10ms rounding. Thermal conditions were not matched, so these are leads for fresh attribution.

The final4.05 code/DevOps answers still match, but several draft counts change. Most3.05 trajectories or draft counts also differ; their entire timing differences cannot be treated as fixed-work overhead.

Primary SSE counts are one greater for the same4.05 work, consistent with the new initial role event. Do not equate SSE frames with target-verification calls.

Original source94ba01d/816c321; primary16ca20d/f4fb6b7; archived final24f0dec/5a4f3ef. The later publishedf650 server needs its own fresh measurement.

Proposed minimal follow-up:4.05code+DevOps on supported source chain94/816→24/816→24/f650, same pack/Q8/chunk2048/dynamicdepth5/confidence0.6/disk placement and affinity. Preflight the intermediate combination normally. Record actual requests/counts/output hashes/draft work andclock/temperature samples. Only then profile an engine- or server-specific boundary; do not remove numerical compatibility or immutable-pinned-memory fixes.

Exact paths, file hashes, per-repeat values, source commits and sparse primary clock observations are inhistorical-attribution.json. No large artifacts were read.
