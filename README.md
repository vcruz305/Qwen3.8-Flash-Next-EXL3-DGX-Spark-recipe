# Qwen3.8-Flash-Next EXL3 on NVIDIA DGX Spark

Run Qwen3.8-Flash-Next on one DGX Spark through the OpenAI-compatible
[vcruz305/tabbyAPI](https://github.com/vcruz305/tabbyAPI) fork and the
[vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) fork, built from source
for GB10 (`sm_121`, aarch64).

This recipe supports flat-K and mixed-K EXL3 packs. The October update brings the
runtime forward to upstream ExLlamaV3 1.6.0, retains the Spark kernels, improves
Qwen tool calling, and adds reproducible API benchmarks and deployment checks.
The exact engine pin lives in [env.sh](exllamav3-tabby/env.sh). TabbyAPI follows
the fork's `main`; setup and each launch record the exact commits they use.

**Start here:** [quick start](#quick-start), [tool calling](docs/tool-calling.md),
[benchmark instructions](bench/README.md), and
[October validation report](VALIDATION_2026-10-08.md).
AI agents should first read [AGENTS.md](AGENTS.md).

## Quick start

Run these commands on the Spark from a clone of this repository:

```bash
# Build the fork for the local CUDA / PyTorch environment and install TabbyAPI.
bash exllamav3-tabby/setup.sh

# Skip the download if you already have a native EXL3 pack.
~/qwen38-exl3/venv/bin/hf download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 3.05bpw_h5_ng5 \
  --local-dir ~/models/Qwen3.8-Flash-Next-EXL3

# One active request, the full trained context window.
PROFILE=single bash exllamav3-tabby/serve.sh
```

For an existing pack, set its actual directory:

```bash
PROFILE=single MODEL_DIR="$HOME/models/flashnext-exl3-3.05bpw" \
  bash exllamav3-tabby/serve.sh
```

The API listens at `http://127.0.0.1:8899/v1`. Check readiness and use the
advertised model name:

```bash
curl --fail-with-body http://127.0.0.1:8899/v1/models

curl --fail-with-body http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next-EXL3",
    "messages": [{"role": "user", "content": "Write a short Python example."}],
    "enable_thinking": false,
    "max_tokens": 256,
    "stream": false
  }'
```

The first build compiles the CUDA extension; the first model load can also
compile kernels. Keep those startup costs separate from warmed inference rates.

## What setup verifies

Everything is installed beneath `~/qwen38-exl3` by default. Override
`RECIPE_HOME` to create a separate installation for an upgrade trial.

- The imported ExLlamaV3 is the fork at the pinned source commit, with the
  required GB10 kernels and a version of at least `1.6.0.post1`.
- The build matches the source SHA, Python, PyTorch, CUDA toolkit, compiler and
  GPU architecture. A changed fingerprint triggers a source rebuild.
- Ninja is available to the build. `MAX_JOBS` controls compilation parallelism.
- TabbyAPI's base dependencies are installed without CUDA extras that would
  replace the Spark runtime with an incompatible wheel.
- `pip check` succeeds. A narrowly scoped helper repairs the architecture tag
  in the known `nvidia-cusparselt-cu13==0.8.1` aarch64 wheel after verifying its
  library architecture and recorded hash. It does not change the library or
  relax dependency checking; the repair has its own audit record.
- Dirty source checkouts are rejected before updates begin. Existing origins,
  branches and local work are preserved.

Check an existing installation without updating it:

```bash
bash exllamav3-tabby/setup.sh --check
```

Re-run `setup.sh` to update TabbyAPI and apply a changed engine pin. For a
reproducible report, retain `$STATE_DIR/deployment.json` and the benchmark JSON,
which include the actual source and runtime identities.

## Context, memory and concurrency

The KV cache is one pool shared by all active requests. A per-request limit of
262,144 tokens does not give every request its own pool of that size.

| Profile | Maximum context per request | Shared KV pool | Active jobs | N-gram placement |
|---|---:|---:|---:|---|
| `single` | 262,144 | 262,144 | 1 | `auto`: estimated from pack headers and available memory |
| `concurrent` (launcher default) | 262,144 | 1,048,576 | 4 | Streamed from disk |

These are capacity settings. Actual memory fit and performance depend on the
pack, current memory use and workload. The launcher estimates n-gram and other
weight sizes from tensor headers, including tables stored in a shared shard,
then prints its memory advisory.

For paired measurements, set `NGRAM_RAM=true` or `false` explicitly and keep
it the same on both sides. Do not compare a RAM-resident candidate against a
disk-streamed baseline and attribute the entire difference to new kernels.

Useful overrides include `MODEL_DIR`, `PROFILE`, `CACHE_SIZE`,
`MAX_BATCH_SIZE`, `MAX_SEQ_LEN`, `NGRAM_RAM`, `CHUNK_SIZE`,
`DRAFT_NUM_TOKENS`, `DYNAMIC_DRAFT`, and `DRAFT_MODE=mtp|disabled`.
Keep `MAX_SEQ_LEN` at or below the trained 262,144-token window. Use a distinct
`STATE_DIR` for each server; the launcher locks it against concurrent rewrites.
For a GPU shared with another supervisor, set `GPU_LOCK_FILE` to the same absolute
lock-file path that supervisor uses. The optional lock stays held for the server
lifetime and refuses a conflicting start before runtime verification or model load.
See [shared GPU ownership](docs/service.md#shared-gpu-ownership).

Preview a configuration without starting the model or changing live state:

```bash
PROFILE=single MODEL_DIR="$HOME/models/flashnext-exl3-4.05bpw" \
  DRY_RUN=1 bash exllamav3-tabby/serve.sh
```

The preview is written to `config.preview.yml`. Edit the environment or source
configuration, rather than a live generated `config.yml`.

## Tool calling

The server uses the `qwen3_5` tool format. The fork adds incremental parsing for
streaming responses, schema-aware argument conversion, complete-call checks,
consistent public model IDs, and valid assistant roles in streamed messages.

Use standard OpenAI `tools` and `tool_choice` fields. A named choice selects
that function; `required` requires a function call; `none` disables calls;
`auto` lets the model decide. Required and named choices, and automatic Qwen requests without explicit output constraints, prefixes or continuations, use a grammar built against the loaded tokenizer for content generation.
Automatic mode permits ordinary answers and zero calls, then enforces complete
call structure once a tool opener appears. The existing policy for tools emitted
inside reasoning is preserved; reasoning is not globally grammar-constrained. `parallel_tool_calls=false` limits a response
to one call. A wrapper unfinished at a normal stop produces an API or SSE error.
Exhausting the token budget retains `finish_reason=length` and can leave partial
arguments; clients must require a complete tool-call finish before execution.

The grammar constrains structure, declared function names and call count. It
does not enforce every JSON Schema property or guarantee that the model copies
the requested values correctly. Validate arguments before executing a tool.

The engine and server coordinate reasoning closure at the producer, before the
next token is sampled. This prevents a literal `</think>` inside tool arguments
from switching the response into its content grammar. The guard applies both
to explicit reasoning budgets and to natural reasoning closure when no budget
is supplied; the latter adds no token limit or forced closing text.

See [the tool-calling guide](docs/tool-calling.md) for examples, supported
schemas, streaming behavior and limitations. Test your actual client and tools:

```bash
python3 bench/tool_smoke.py \
  --model Qwen3.8-Flash-Next-EXL3 --mode both --case all \
  --output results/tool-smoke.json
```

The [API validation guide](docs/api-validation.md) includes checks for invalid
request recovery, early disconnects, automatic zero-call answers, and explicit
client output constraints.

An optional [OpenAI SDK check](bench/sdk_smoke.py) exercises the public stream
accumulator and reuses its assembled assistant message in a tool-result
round trip. Install that client dependency in a separate test environment;
the server does not need it.

### Cyber Frost template override

The inspected Cyber Frost 3.87bpw pack unconditionally enables thinking in its
bundled template, even when a request asks to disable it. Use the reviewed
[template override](docs/prompt-templates.md) to honor the request while
preserving the original pack files and default thinking behavior:

```bash
PROFILE=single \
MODEL_DIR="$HOME/models/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw" \
PROMPT_TEMPLATE="$(pwd)/exllamav3-tabby/templates/cyber-frost-3.87bpw-thinking.jinja" \
  bash exllamav3-tabby/serve.sh
```

The launcher validates the override and records its content hash. This override
is specific to the inspected pack; the server still records the actual template
loaded for each run.

## Measured results

Final source confirmation on the Spark uses engine `24f0dece3` and TabbyAPI
`5a4f3ef`, 2,048-token chunks, K8/V8 storage, dynamic MTP depth 5 and confidence
0.6. These are median **server decode tokens/second**, three measured 400-token
outputs per workload after one warmup:

| Pack / n-gram placement | Code, original → final | DevOps, original → final | Prose, original → final |
|---|---:|---:|---:|
| Flat 3.05 / RAM | 82.49 → 80.15 | 74.11 → 70.95 | 55.93 → 53.32 |
| Flat 4.05 / disk | 77.02 → 75.85 | 67.76 → 66.45 | 48.57 → 47.46 |
| SAGE 4.15 / disk | 48.78 → 52.39 | 46.48 → 49.73 | 30.75 → 34.36 |
| Cyber Frost 3.87 / disk, template changed | 41.13 → 45.75 | 34.63 → 44.28 | 32.09 → 34.36 |

SAGE improves approximately **7–12%** in this confirmation. The flat packs show
small single-request regressions; the update is not a uniform speedup.
Cyber's corrected template honors `enable_thinking:false`, changing its actual
prompt and generated work, so its row is a behavior/configuration comparison.
The report separately preserves matched-template Cyber kernel controls.
All four packs pass the final 28-case tool suite, five SDK checks, 19 recovery
checks and their scheduled automatic-choice rounds. Additional reasoning-enabled requests to copy `<think>literal</think>` exactly
retain 16 failed executions in the single/concurrent matrix. Raw output already
contains the changed values; the four separately tested single/RAM variants
with `enable_thinking:false` pass. The report preserves the original failures and
limits the workaround claim to those tested requests.

For the 3.05 concurrent profile, a controlled row-budget-0 versus row-budget-8
comparison improved aggregate end-to-end throughput by **37.8% at two requests**
and **36.9% at four requests**. This is a tuning comparison within the updated
runtime; an original-runtime concurrent baseline was not measured. The global
row-budget default remains zero. The measured 3.05 service selects eight;
other packs require their own comparison.


The [October report](VALIDATION_2026-10-08.md) records the on-device A/B method,
source commits, tested model views, correctness checks and raw results.

The previous September README is retained as
[historical benchmark documentation](HISTORICAL_BENCHMARKS.md). Its
`examples/chat.py` and historical vLLM measurements use different paths from
the October TabbyAPI suite.

To reproduce the current API measurements:

```bash
python3 bench/bench_v1.py \
  --model Qwen3.8-Flash-Next-EXL3 \
  --suite all --repeat 3 --warmup 1 --run-id comparison-v1 \
  --metadata "$HOME/qwen38-exl3/state/deployment.json" \
  --output results/api-benchmark.json
```

The clients preserve actual completion-token usage, prompt/cache counts,
server decode and prefill rates, client time to first output, finish reasons,
model identity and failures. Requested `max_tokens` and SSE event counts are
never substituted for generated tokens.

For concurrent work and long-context retrieval with recorded token counts, use
[concurrency.py](bench/concurrency.py) and
[long_context.py](bench/long_context.py). Their metric definitions and caveats
are in [bench/README.md](bench/README.md). A throughput run, a synthetic
retrieval test and a numerical regression probe answer different questions;
none certifies general model capability.

## Persistent service and remote access

[The service guide](docs/service.md) provides a user-systemd unit and an
environment file, including restart, logs, upgrades and rollback.

Loopback binding has authentication disabled by default. Access a loopback
server from another machine through an SSH tunnel:

```bash
ssh -N -L 8899:127.0.0.1:8899 YOUR_SPARK_USER@YOUR_SPARK_IP
```

Your client then uses `http://127.0.0.1:8899/v1` on that machine. If you choose
`HOST=0.0.0.0`, the launcher enables authentication by default. Read the
generated `api_tokens.yml` on the Spark and configure a bearer token in your
client. The updated server logs key counts and the file location, without
printing the credentials.

## Troubleshooting

| Symptom | Action |
|---|---|
| Wrong fork, source pin or build fingerprint | Run `setup.sh --check`, then `setup.sh`; retain its build log if it fails. |
| Dirty checkout during setup | Preserve your changes in a commit/stash, or use a separate `RECIPE_HOME`. Setup does not reset your work. |
| Pack missing | Set `MODEL_DIR` to the existing native pack directory. |
| Pack configuration was rewritten for vLLM | Use [make_native_view.sh](exllamav3-tabby/tools/make_native_view.sh) to create a separate native view. |
| Memory estimate or model load fails | Stop competing model workloads, check the selected pack/profile, and try `NGRAM_RAM=false`. |
| Model ID differs from a client configuration | Query `/v1/models` and use its advertised ID; retain the response with your report. |
| Tool response is truncated | Inspect the API error/finish reason and requested token budget; test the same schema with the tool suite. |
| Unexpectedly fast prefill | Check actual cached prompt-token counts before treating it as a cold-prefill result. |

The launcher drops cached pages for the selected pack before loading, which
helps GB10's memory admission checks after downloads and previous loads.
This does not replace an actual memory-fit test.

## Repository layout

| Path | Purpose |
|---|---|
| [exllamav3-tabby/](exllamav3-tabby/) | Supported setup, launchers, configuration and runtime checks |
| [bench/](bench/) | Strict API benchmarks, tool/SDK checks and long-context retrieval |
| [VALIDATION_2026-10-08.md](VALIDATION_2026-10-08.md) | October hardware measurements and validation evidence |
| [HISTORICAL_BENCHMARKS.md](HISTORICAL_BENCHMARKS.md) | Earlier measurements and tuning research |
| [docs/tool-calling.md](docs/tool-calling.md) | Qwen tool semantics and examples |
| [docs/service.md](docs/service.md) | Persistent deployment and rollback |
| `exllamav3-tabby/tuning/`, `exllamav3-tabby/bench/` | Earlier GB10 experiments and source patches |
| `exllamav3-tabby/beta/`, `exllamav3-tabby/legacy/` | Historical experiments |
| [vllm-plugin/](vllm-plugin/README.md) | Separate vLLM route with its own setup |

## Credits

This recipe builds on [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
and [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI), with the
Spark work maintained in the [ExLlamaV3 fork](https://github.com/vcruz305/exllamav3)
and [TabbyAPI fork](https://github.com/vcruz305/tabbyAPI). Model-pack credit and
the earlier experiments remain in the
[historical documentation](HISTORICAL_BENCHMARKS.md#related-repositories).
