# AGENTS.md: instructions for AI agents setting up this recipe

Read this before running anything in this repository.

## The one supported path

The recipe is **the latest TabbyAPI on the vcruz305/exllamav3 fork**. Run exactly:

```bash
bash exllamav3-tabby/setup.sh                    # installs the fork runtime + latest TabbyAPI
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 3.05bpw_h5_ng5 \
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
- **Do not pin TabbyAPI** to a commit. The recipe tracks `main`; re-run `setup.sh` to update.
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
- `PROFILE=single`: 1 job, 262,144 tokens, n-gram table in RAM (the fastest single stream).

To change sizing, set the environment variables (`CACHE_SIZE`, `MAX_BATCH_SIZE`, `MAX_SEQ_LEN`, `NGRAM_RAM`),
not the rendered config. `DRY_RUN=1 bash exllamav3-tabby/serve.sh` shows what it would run.

## Verify you are on the right runtime

```bash
bash exllamav3-tabby/setup.sh --check
```

It must print `exllamav3 1.5.x (fork) at ~/qwen38-exl3/exllamav3/...` and a TabbyAPI commit.
If it errors, re-run `bash exllamav3-tabby/setup.sh`. Do not work around it.

## If something fails

- `no pack at ...`: the download path differs; set `MODEL_DIR=/path/to/pack`.
- `was prepared for vLLM`: someone ran `vllm-plugin/prepare_pack.sh` on this pack. Run the
  `make_native_view.sh` command the error prints.
- Load refuses for lack of memory with plenty of `MemAvailable`: page cache. `serve.sh` drops the
  pack's pages already. Stop other model servers on the box.
- The build failed: the log is `~/qwen38-exl3/state/exllamav3-build.log`. It needs CUDA 13.x
  `nvcc` at `/usr/local/cuda` (or `CUDA_HOME`).
