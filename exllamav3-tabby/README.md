# exllamav3-tabby: the recipe

The latest [TabbyAPI](https://github.com/theroyallab/tabbyAPI) on the
[vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) fork runtime. Full instructions are
in the [top-level README](../README.md#quick-start).

| File | Purpose |
|---|---|
| `setup.sh` | Builds the fork at the pinned commit for sm_121 and installs TabbyAPI `main` into one venv (`~/qwen38-exl3/`). `--check` only verifies the install |
| `serve.sh` | **The API.** Renders `tabby-config.yml` and starts TabbyAPI. `PROFILE=concurrent` (default) or `single` |
| `tabby-config.yml` | TabbyAPI config template (stock keys only) |
| `chat.sh` | Console chat, the tuned `chat.py` launcher |
| `env.sh` | Shared defaults: fork pin, paths, GB10 kernel knobs, runtime/pack checks |
| `drop-model-cache.sh` | Drops the pack's page cache (a GB10 unified-memory trap) |
| `beta/` | (Beta) minimal one-request-at-a-time OpenAI shim, for A/B only |
| `tools/make_native_view.sh` | Native view of a pack that `vllm-plugin/prepare_pack.sh` rewrote |
| `bench/`, `tuning/` | GB10 tuning research: harnesses, A/B scripts, logs, patches |
| `legacy/` | Stock exllamav3 1.5.0 build for the historical baseline. Not the runtime |
