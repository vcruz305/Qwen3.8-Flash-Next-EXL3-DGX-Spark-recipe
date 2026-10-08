# exllamav3-tabby: the recipe

The latest [vcruz305 TabbyAPI fork](https://github.com/vcruz305/tabbyAPI) on the
[vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) fork runtime. Full instructions are
in the [top-level README](../README.md#quick-start).

| File | Purpose |
|---|---|
| `setup.sh` | Builds the fork at the pinned commit for sm_121 and installs the TabbyAPI fork's `main` into one venv (`~/qwen38-exl3/`). `--check` verifies the runtime path, version, kernels, ABI fingerprint and package consistency |
| `serve.sh` | **The API.** Renders `tabby-config.yml` and starts TabbyAPI. `PROFILE=concurrent` (default) or `single` |
| `tabby-config.yml` | TabbyAPI config template (stock keys only) |
| `chat.sh` | Console chat, the tuned `chat.py` launcher |
| `env.sh` | Shared defaults: fork pin, paths, GB10 kernel knobs, runtime/pack checks |
| `drop-model-cache.sh` | Drops the pack's page cache (a GB10 unified-memory trap) |
| `beta/` | (Beta) minimal one-request-at-a-time OpenAI shim, for A/B only |
| `tools/runtime_state.py` | Records exact source/toolchain/config provenance; verifies the extension build fingerprint |
| `tools/model_memory.py` | Estimates model/ngram and KV memory from headers without loading weights |
| `tools/git_helpers.sh` | Preserves local work and existing remotes during setup updates |
| `../bench/` | Strict API performance clients, tool-call smoke tests and CPU regressions; see [benchmark instructions](../bench/README.md) |
| `tools/make_native_view.sh` | Native view of a pack that `vllm-plugin/prepare_pack.sh` rewrote |
| `bench/`, `tuning/` | GB10 tuning research: harnesses, A/B scripts, logs, patches |
| `legacy/` | Stock exllamav3 1.5.0 build for the historical baseline. Not the runtime |
