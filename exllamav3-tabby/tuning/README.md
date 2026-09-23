# Native exllamav3 on GB10: benchmark matrices, harnesses, patches

Research material behind the README section "The native engine, tuned for GB10". **To run the model,
use `../serve.sh` (TabbyAPI) or `../chat.sh` (console); nothing here is needed for that.**

These scripts were run against a source build of [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3)
at `~/exllamav3` with its venv at `~/exllamav3/.venv`, and the pack at `~/models/Qwen3.8-Flash-Next-EXL3/`.
They are kept as run, so the logs stay reproducible; adjust the paths at the top of each script, or
symlink `~/exllamav3` to `$RECIPE_HOME/exllamav3` from `../setup.sh`.

Measure with `-topk 1`. The default temperature-0.8 sampler moves MTP acceptance 58–75% run to
run and the tok/s number with it. Quote `chat.py` numbers, not in-process ones: the harnesses
that drive the generator directly read ~4 tok/s high for kernel changes and ~10 for host-side
ones (the console write per token is itself a host sync).

## Launcher and benchmark

| File | What |
|---|---|
| `../chat.sh` | The console launcher (was `run-qwen38-exl3.sh`). GB10 env, big-core pinning, `chat.py -mode qwen35 -mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs 262144 -tps`. |
| `../serve.sh` | The supported API: latest TabbyAPI on the fork, same knobs. |
| `../beta/serve_openai.sh` | (Beta) the minimal /v1 shim, one job at a time. |
| `bench.sh <tag> [chat.py flags]` | One cold greedy 400-token generation through `chat.py`, prints load time and the `Context:` line with tok/s and acceptance. `PROMPT="..."` overrides the prompt. Logs to `~/bench_<tag>.log`. **The number to quote.** |
| `pubbench.sh` | The 12-cell matrix behind the README table: 3 prompt classes × {host patches on, off} × {`-ndt 5`, `-dds -dc 0.6`}, unattended (`setsid nohup bash pubbench.sh > pubbench.log &`). |
| `i8bench.sh` | int8 mixer on/off × 3 prompt classes with repeats, same path. |
| `ctxsweep.sh` | Load ceiling: cold load at `-cs` 32k…1M (fp16 and `-cq 8,8`), short prompt, reports load result and decode. Everything loads. |
| `ctxfill.py` | The context harness. Builds a prompt to a TOKEN target (`K=240` thousand), plants a needle at a random depth (`SEED`), drives `Generator`/`Job` (chat.py's `-prompt` dies at 128 KiB argv). `TASK=needle NEW=32` for retrieval, `TASK=code NEW=400` for decode-at-depth. `CS=` overrides the cache size, `CQ=8,8` for quantized KV. Reports prompt tokens, prefill and decode t/s, acceptance, hit/miss, peak host memory. |
| `fillsweep.sh`, `fillsweep2.sh` | The needle sweeps: 32k → 480k, both KV precisions, two needle positions at 480k. |
| `depthab.sh` | The decode-at-depth A/B: 400 tokens of code at 4k / 128k / 240k, fp16 vs 8-bit KV. |
| `logs/` | Raw logs and summaries from every matrix above, 2026-09-17; `ctx_runs_summary.txt` is every context run on one page. |
| `../drop-model-cache.sh` | `posix_fadvise(DONTNEED)` on every pack file. On GB10 page cache is GPU-allocatable memory; without this the autosplit loader refuses to load after any large file write. Sleep ≥8 s between back-to-back loads or the second dies to a workspace race. |

## Harnesses

| File | What |
|---|---|
| `mixaudit.py` | Walks every module and buckets parameter bytes by `(class, dtype)`. How the 3.2 GB of fp16 `GatedResidual` weights in a 3-bit pack was found. Run this first on any new pack. |
| `mixq.py` | Simulates mixer storage precision by quantize→dequantize of the loaded `fn_h`/`upx_h` (`MIXBITS=8/6/4`, `0` = fp16 baseline), one variant per process; reports acceptance and an output hash. Decides whether a kernel is worth writing. |
| `gr_int8_parity.py` | `gr_mix_int8` vs `gr_mix` (fp16) on real sites: parity against an fp32 torch reference on the dequantized weights, and per-call timing, R = 1..8. |
| `grtraffic.py` | One-site `gr_mix` timing vs R, fp16 and row-batched. Shows why a single-site microbench reads above spec bandwidth (L2-resident) while the in-situ round does not. |
| `diverge.py` | First diverging token between two `chat.py -basic` logs (e.g. int8 on vs off), per prompt class. |
| `moepath.py` | Hooks `ext.exl3_moe` and histograms `(model, rows)` per call site. Proves a fractional "calls/round" row in the profile is prefill, not decode. |
| `sweep.py` | In-process greedy sweep over `--ndts`, `--affinity all,big`, `--prompts code,prose,devops`. |
| `split_time.py` | Per-round split: target verify forward vs draft forwards vs host, with the q_len histogram. `NDT=k`. |
| `kern_rounds.py` | torch.profiler kernel table normalized per verify round. Remember the prefill forward is in the trace. |
| `accept.py` | Per-position MTP draft acceptance P(pos i accepted \| reached) per prompt and `NDTS`; `DDS=0.6` enables dynamic drafting. The harness that explains prose. |
| `bw.py` | Unified-memory copy/read bandwidth. |
| `gr_parity.py` | Row-batched vs original fp16 GatedResidual kernels against the fp32 reference. `EXL3_GR_RB=0/1`. |
| `ab_greedy.py` | Greedy token-id dump for A/B between two env configs; diff the JSON files. |

## Patches

`patches/` is the fork's [PR #2](https://github.com/vcruz305/exllamav3/pull/2) as `git format-patch`,
three commits, already merged into `master`:

| | Flag | Default |
|---|---|---|
| `0001` int8 GatedResidual mixer kernels | `EXL3_GR_INT8` (`0` disables) | on (since fork [#3](https://github.com/vcruz305/exllamav3/pull/3)) |
| `0002` pruned draft `lm_head` | `EXL3_MTP_HEAD_N=65536` (`0` disables) | on |
| `0003` batched verify, device-resident draft chain, pinned MTP block table | `EXL3_BATCH_VERIFY`, `EXL3_MTP_DEVICE_DRAFT`, `EXL3_EMBED_GPU` | on |
