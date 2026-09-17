# Native exllamav3 on GB10: tuned launcher and harnesses

Companion to the README section "Tuning the native engine on GB10". Everything here
runs against a source build of [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3)
`master` (upstream + aarch64 guards) at `~/exllamav3` with its venv at `~/exllamav3/.venv`,
and the pack at `~/models/Qwen3.8-Flash-Next-EXL3/`. Adjust the paths at the top of each
script if yours differ.

| File | What |
|---|---|
| `run-qwen38-exl3.sh` | The launcher. Drops the model's page cache, sets `EXL3_INT8_GEMV=0 EXL3_MOE_COOP_WIDE=1`, pins to the ten X925 cores, runs `chat.py -mode qwen35 -mtp -ndt 5 -cs 32768 -tps`. Extra args pass through (`-prompt "..."`, `-cs 262144`). |
| `bench.sh <tag> [chat.py flags]` | One cold 400-token generation through `chat.py`, prints load time and the `-tps` line. `PROMPT="..."` overrides the prompt. Logs to `~/bench_<tag>.log`. The number to quote. |
| `drop-model-cache.sh` | `posix_fadvise(DONTNEED)` on every pack file. On GB10 page cache is GPU-allocatable memory; without this the autosplit loader refuses to load after any large file write. |
| `sweep.py` | In-process greedy sweep over `--ndts`, `--affinity all,big`, `--prompts code,prose,devops`. Reads ~4 tok/s above `bench.sh` (no console streaming, warm). |
| `split_time.py` | Per-round split: target verify forward vs draft forwards vs host, with the q_len histogram. `NDT=k`. |
| `kern_rounds.py` | torch.profiler kernel table normalized per verify round. |
| `bw.py` | Unified-memory copy/read bandwidth (215 GB/s copy measured). |
| `gr_parity.py` | Row-batched vs original GatedResidual kernels against the fp32 torch reference, R=1..8, plus timing over all 97 sites. `EXL3_GR_RB=0/1`. |
| `ab_greedy.py` | Greedy token-id dump for A/B between two env configs; diff the JSON files. |
| `gr-row-batched.patch` | The kernel change as a `git format-patch`, already merged in the fork; here for reference. |

Measure with `-topk 1`. The default temperature-0.8 sampler moves MTP acceptance 58-75%
run to run and the tok/s number with it.
