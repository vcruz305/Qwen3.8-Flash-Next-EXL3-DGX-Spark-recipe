# AGENTS.md: instructions for AI agents setting up this recipe

Read this before running anything in this repository.

## The one supported path

The recipe is **the latest vcruz305/tabbyAPI fork on the vcruz305/exllamav3 fork**. Run exactly:

```bash
bash exllamav3-tabby/setup.sh                    # installs the fork runtime + latest TabbyAPI
~/qwen38-exl3/venv/bin/hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 \
  --local-dir ~/models/Qwen3.8-Flash-Next-EXL3
bash exllamav3-tabby/serve.sh                    # OpenAI API on http://127.0.0.1:8899/v1
```

The model id to send in requests is `Qwen3.8-Flash-Next-EXL3`. Check readiness with
`curl -s http://127.0.0.1:8899/v1/models`. The first load takes about a minute.

## Do not

- **Do not `pip install exllamav3`** (PyPI or a TabbyAPI wheel). The runtime is the fork
  built from source by `setup.sh`. A stock wheel lacks the GB10 kernels, and the launchers
  will refuse to start with it.
- **Do not install TabbyAPI's `cu12`/`cu13` extras** or run its `start.sh`. Those pull
  x86_64 exllamav3/torch wheels. `setup.sh` installs TabbyAPI's base package into the
  recipe venv on purpose.
- **Do not pin TabbyAPI** to a commit. The recipe tracks the vcruz305 fork's `main`; re-run `setup.sh` to update.
  Exact deployed commits and dependency versions are recorded in the setup/deployment snapshots.
- **Do not use anything under `vllm-plugin/`** unless the user explicitly asks for vLLM.
  It is a different, slower engine with its own setup, and its `prepare_pack.sh`
  rewrites the pack in place so that exllamav3 can no longer load it.
- **Do not use `exllamav3-tabby/beta/`** as the API. It is a one-request-at-a-time
  shim for A/B testing.
- **Do not use `exllamav3-tabby/legacy/`.** It builds *stock* exllamav3 1.5.0 for a
  historical baseline.
- **Do not raise `MAX_SEQ_LEN` above 262144.** That is the model's trained window, and
  retrieval fails past it.

## Sizing (context and concurrency)

`cache_size` is one KV pool **shared by all concurrent requests**. `serve.sh` picks it by profile:

- `PROFILE=concurrent` (default): 4 concurrent jobs, 262,144 tokens per request,
  1,048,576-token pool, n-gram table streamed from disk.
- `PROFILE=single`: 1 job, 262,144 tokens, `NGRAM_RAM=auto`. The launcher uses pack
  headers and available memory to choose RAM or streaming. For repeatable benchmarks,
  set `NGRAM_RAM=true` or `false` explicitly and keep it constant across A/B runs.

To change sizing, set the environment variables (`CACHE_SIZE`, `MAX_BATCH_SIZE`, `MAX_SEQ_LEN`, `NGRAM_RAM`),
not the rendered config. `CHUNK_SIZE`, `DRAFT_MODE=mtp|disabled`,
`DRAFT_NUM_TOKENS`, and `DYNAMIC_DRAFT=true|false` also have explicit overrides.
`DRY_RUN=1 bash exllamav3-tabby/serve.sh` writes `config.preview.yml` and shows
what it would run without replacing the live config or model view.

These are capacity settings, not fit guarantees for every quant. The header-based
memory estimate is advisory. Validate actual peak memory and context behavior for
each pack. Do not rewrite or requantize the weight files to make them fit.

Each live server needs its own `STATE_DIR`; the launcher holds a lock to prevent
one process from rewriting another process's config and model view.

## Verify you are on the right runtime

```bash
bash exllamav3-tabby/setup.sh --check
```

It must identify the imported runtime under the configured `EXL3_SRC`, confirm the
required fork kernels and version (at least `1.6.0.post1`), verify an existing ABI
fingerprint, and print a TabbyAPI commit. It also runs `pip check`.
If it errors, re-run `bash exllamav3-tabby/setup.sh`. Do not work around it.

## Updates and measurements

Setup refuses dirty source checkouts before changing refs or packages. Preserve
local work with a commit or stash, or select a separate source directory. It never
resets a user's branches. When migrating from the upstream TabbyAPI checkout,
setup preserves `origin` and adds a `recipe` remote for the configured fork.
A changed engine commit or Python/Torch/CUDA/compiler fingerprint triggers a rebuild.

The launcher writes `$STATE_DIR/deployment.json` with source commits, package
versions, resolved model files and a redacted rendered config. Benchmark clients
accept it through `--metadata`. Use `bench/bench_v1.py`, `bench/concurrency.py`
and `bench/tool_smoke.py`; see `bench/README.md` for exact metric definitions.

Run CPU checks before deployment:

```bash
python -m unittest discover -s bench -p 'test_*.py' -v
bash -n exllamav3-tabby/setup.sh exllamav3-tabby/env.sh exllamav3-tabby/serve.sh
```

Do not report requested `max_tokens` as emitted tokens. Preserve actual API usage,
cache counts, server timings, errors and model identity. A failed request or an
incomplete stream is not a successful throughput sample. Record baseline and
candidate with the same pack, prompts, sampling and memory placement.

## If something fails

- `no pack at ...`: the download path differs; set `MODEL_DIR=/path/to/pack`.
- `was prepared for vLLM`: someone ran `vllm-plugin/prepare_pack.sh` on this pack. Run the
  `make_native_view.sh` command the error prints.
- Load refuses for lack of memory with plenty of `MemAvailable`: page cache. `serve.sh` drops the
  pack's pages already. Stop other model servers on the box.
- The build failed: the log is `~/qwen38-exl3/state/exllamav3-build.log`. It needs CUDA 13.x
  `nvcc` at `/usr/local/cuda` (or `CUDA_HOME`).
