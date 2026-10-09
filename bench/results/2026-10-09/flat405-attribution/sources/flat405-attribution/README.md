# Flat4.05 source-stack attribution

This is a bounded diagnostic: three fresh server loads, with the original code and DevOps workloads. Each workload has one unmeasured warmup and three measured 400-token responses. The unchanged public client receives the original `overnight-v1` salt, thinking disabled, temperature0, top_k1, seed0, cold prompts, and zero filler. The controller verifies every request hash and actual prompt/cache/completion count against the completed original report. Its CPU checks reconstruct all eight exact payload hashes using byte-identical public a8 client code.

| Cell | Engine / Python environment | Server source |
|---|---|---|
| A | Original 94ba01d, original preserved venv | Original 816c321 |
| B | Qualified 24f0dec, final preserved venv | New isolated checkout of 816c321 |
| C | Qualified 24f0dec, final preserved venv | Published f650bb5 |

The unsupported94/f650 combination is not attempted. B's preliminary CPU import passed using the final venv and isolated old backend; CUDA was hidden and `torch.cuda.is_initialized()` remained false. Its normal old-server minimum-version check passed. That proof does not claim model loading already passed.

All cells use the same original flat4.05 pack, disk PLE, Q8 KV,262144-token shared pool/context, batch 1, chunk 2048, dynamic MTP depth 5/confidence 0.6,65536-row draft head, and big cores 5–9/15–19. The final numerical compatibility settings remain explicit. No kernel, generator, server, weight, venv, or published recipe file is modified by this controller.

The public a8 launcher receives its supported `EXL3_MIN_VERSION=1.5.1` override only for the exact historicalA pair. Its ordinary native import/path/kernel/version/build verification still runs. B/C require 1.6.0.post1. Preflight imports the actual backend from each launch path, records all package versions/native extension path, and checks both real source commits and cleanliness. Preflight hides CUDA and does not construct a model.

The old venv's sole `pip check` failure remains preserved as exit 1. A read-only verifier streams the274 MB NVIDIA library before timing and requires exact equality to the independently proved Oct8 library/WHEEL/RECORD/ELF identity. Only the precise nvidia-cusparselt-cu13 0.8.1 sbsa-tag defect can be classified, and only for A. This is not a successful full `setup.sh --check` or a deployable environment. B/C and all other package failures stop the experiment. Neither venv is repaired.

The operator must schedule this after stopping/draining the current manager through its supported control path. The controller requires the existing regular `/home/cruzspark/redsnow-gpu.lock`, holds it across preflight and all three cells, requires REXL3 manager inactive and Qwen unit inactive/disabled, and rejects another CUDA owner. It does not manage those units itself. Signal handling defers Python interruption until each child is registered, without blocking inherited OS signals, then signals only a group whose PID/PGID/SID and ownership token agree with the frozen lifecycle helper. Unexpected server exit or uncertain cleanup stops the sequence.

Example, **only when root schedules the GPU window**:

```bash
python3 /home/cruzspark/qwen-followup-20261009/measurement/flat405-attribution/run_attribution.py \
  --recipe /path/to/clean/recipe-at-a8c72bd \
  --old-runtime /home/cruzspark/qwen38-exl3 \
  --new-runtime /home/cruzspark/qwen38-exl3-20261008 \
  --old-tabby /home/cruzspark/qwen-followup-20261009/tabby-816-attribution \
  --model-inputs /path/to/current-models.json \
  --output /home/cruzspark/qwen-followup-20261009/results/flat405-three-cell
```

`--preflight-only` runs the same owned-window preflight without launching any server. Existing output paths are rejected; a failed attempt is retained and a rescheduled run needs a new output path.

Each cell preserves source/package/import evidence, deployment/config identity, model metadata and loader ordering, raw client reports and logs, process cleanup, and sparse 2-second GPU clock/temperature/power/resource samples. Source/package/model/config/input drift is checked before load, between workloads and after measurements. Resource observations are not guaranteed peaks. Comparing server decode rate requires checking generated response hashes and accepted/rejected draft counts; identical input hashes alone do not establish identical work. The one added initial SSE role event in newer Tabby is not a verification-window count.

A→B changes the engine and the preserved Python environment together; package-version differences must be considered before attributing the entire difference to engine code. B→C keeps that engine/environment constant. Sequential order, thermal sampling, three repeats and10ms-rounded server duration all limit precision. This diagnostic does not authorize a performance claim, numerical policy change or deployment on its own.

CPU verification: `python3 test_attribution_cpu.py` currently 19 checks, including real harmless child-process cleanup after a pendingSIGTERM, supported source combinations, real original payload reconstruction, exact actual-usage gates and output-trajectory accounting. No GPU or API calls occur in those tests.
