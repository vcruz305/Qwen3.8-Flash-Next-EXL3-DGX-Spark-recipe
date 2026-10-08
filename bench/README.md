# API performance and tool calling validation

These clients exercise the deployed TabbyAPI endpoint. They use the Python
standard library and require no access to a GPU themselves. The CPU regression
tests also use PyYAML, already installed by the runtime.

The report format is version 2 for performance and version 1 for tool smoke
tests. Old `decode_tok_s_median`, `window` and `steady` fields are intentionally
replaced by named, documented metrics.

## Before measuring

Keep the baseline and candidate on the same model pack and workload. A speed
difference between a flat-K 3.05 bpw pack and a mixed-K 4.15 bpw pack is a model
comparison; it does not isolate an engine improvement.

Record the model directory, engine and server commits, Torch/CUDA versions, CPU
affinity, serving profile, KV pool, batch limit, n-gram placement and MTP settings.
The launcher saves this in `$STATE_DIR/deployment.json`; pass it with
`--metadata`. The snapshot records weight sizes and configuration-file hashes.
It does not read and hash tens of gigabytes of weights.

Set `NGRAM_RAM=true` or `false` explicitly for an A/B run. The single profile's
`auto` placement depends on current free memory. Keep the same placement,
`CACHE_SIZE`, `MAX_SEQ_LEN`, `MAX_BATCH_SIZE`, `CHUNK_SIZE`,
`DRAFT_NUM_TOKENS` and sampling settings on both sides.

Use a quiet machine with one model server. Complete a benchmark before changing
the server or its environment. Do not run correctness tests at the same time as
a throughput sample.

## Single stream

For each of code, DevOps and prose, run one warmup followed by three measured
requests:

```bash
python bench/bench_v1.py \
  --model Qwen3.8-Flash-Next-EXL3 \
  --suite all --warmup 1 --repeat 3 --max-tokens 400 \
  --label baseline-3.05 --run-id comparison-001 \
  --metadata ~/qwen38-exl3/state/deployment.json \
  --output results/baseline-3.05.json
```

Repeat with the candidate after restarting its model process, using the same
`--run-id` and settings. That gives the same prompt text on both fresh
endpoints. If you run a second benchmark on an endpoint that still has cached
prefixes, use a new run ID or select the warm-cache test deliberately.

By default, each measured request changes an early prompt prefix. The result
records actual `cached_prompt_tokens` so caching remains visible. To measure
reused prompts instead:

```bash
python bench/bench_v1.py --suite code --cache-mode warm \
  --context-tokens 24000 --repeat 3 --output results/warm-24k.json
```

`--context-tokens` is an approximate filler construction target, not a
tokenizer measurement. Compare the actual `prompt_tokens` in the output. The
model is free to stop early; `completion_tokens` always comes from server usage,
and `finish_reason` distinguishes an early stop from the requested length cap.

### Metrics

| Field | Meaning |
|---|---|
| `server_decode_tok_s` | Decode rate reported by TabbyAPI usage, or its llama-compatible timings. Null when absent. |
| `server_prefill_tok_s` | Prompt processing rate reported by the server. Interpret together with actual prompt and cached counts. |
| `client_end_to_end_tok_s` | Actual completion tokens divided by the whole client request time, including queueing, prefill and HTTP overhead. |
| `ttft_s` | Time from starting the HTTP request to the first content, reasoning or tool delta. Role chunks and SSE keepalive comments do not count. |
| `client_decode_tok_s_estimate` | A compatibility estimate using first/last output events and completion count. This is approximate because an SSE event may contain multiple speculative tokens. |
| `draft_acceptance` | Accepted draft tokens divided by accepted plus rejected draft tokens, when the server reports both. |
| `cached_prompt_tokens` | The server-reported cache reuse count, or null if unavailable. |

Use server timings for an engine decode comparison and client wall throughput
for a serving comparison. A lower TTFT from a cached prefix is not a faster cold
prefill. Do not mix the estimated client decode rate with server decode numbers
in the same benchmark column.

Every run preserves the raw usage/timings and hashes of its request and response.
The JSON file is checkpointed after each request, including failed runs.

### Model identity and aliases

With no explicit `--model`, exactly one ID must be advertised by `/v1/models`.
The client never silently selects the first of several unrelated packs. By
default, the response model ID must match the requested ID.

Older TabbyAPI versions can advertise a view alias but respond with the resolved
pack's basename. If you have independently verified that mapping, declare it:

```bash
python bench/bench_v1.py \
  --model Qwen3.8-Flash-Next-EXL3 \
  --response-model flashnext-exl3-3.05bpw \
  --suite all --output results/verified-alias.json
```

Both IDs and the expected response ID are recorded. Any other response ID still
fails the run. The same option exists in the concurrency and tool clients.

## Concurrent serving

Use a server profile with enough batch and cache capacity for the requested
load. The following asks for one, two and four simultaneous requests, with
approximately 3k tokens of prefix and up to 400 generated tokens each:

```bash
python bench/concurrency.py candidate 1,2,4 3000 400 3 \
  --warmup 1 --model Qwen3.8-Flash-Next-EXL3 \
  --metadata ~/qwen38-exl3/state/deployment.json \
  --output results/concurrency.json
```

A barrier starts each group together. The headline `end_to_end_tok_s` is the
sum of actual completion tokens divided by the interval from the earliest
request start to the last response end. This includes prefill and queueing.
Per-stream server timings and TTFT are also preserved.

`all_streams_output_overlap_s` reports how long all output streams overlapped.
It is not used to invent a token rate: OpenAI SSE does not guarantee token counts
per event. The old client inferred a "steady" rate by assuming a constant rate
on every stream, which was especially unreliable with speculative decoding and
short responses.

Any failed stream invalidates that whole batch's throughput sample. Its successful
siblings and error messages remain in the report. A failed batch cannot improve
the displayed median by disappearing.

## Tool calling

```bash
python bench/tool_smoke.py --model Qwen3.8-Flash-Next-EXL3 \
  --mode both --output results/tools.json
```

All tools are synthetic. No shell commands, external requests or real application
actions are executed. The weather tool returns fixed fixture values.

| Case | What it checks |
|---|---|
| `auto` | Ordinary automatic function selection and exact arguments. |
| `no_args` | An empty argument object remains a valid no-argument invocation. |
| `strings` | Strings that look like numbers, booleans, JSON or reasoning markup keep their exact type and content. |
| `typed` | Integer, number, boolean, array, nested object and nullable arguments keep their schema types. |
| `required` | A required tool request produces the requested useful invocation. |
| `named` | Named function selection chooses the function requested by the client. |
| `required_adversarial` | Required calling is enforced despite a system/user prompt demanding plain text. |
| `named_adversarial` | Named selection is enforced despite prompts strongly requesting a different function. |
| `none` | Tool calling is disabled and an ordinary answer is returned. |
| `parallel` | Two calls are exposed with distinct IDs and correct arguments. |
| `parallel_disabled` | A single-call request does not emit multiple invocations. |
| `reasoning_tool` | A thinking-enabled request can finish in a tool invocation. |
| `round_trip` | Reversed tool-result order is matched by ID, then a repeated turn retains the results. |
| `truncated` | A deliberately short generation ends with length and does not expose an incomplete invocation as executable. |

Each case runs both streaming and nonstreaming by default. Stream assembly checks
tool indices, stable IDs, fragmented function names/arguments and the final
finish reason. Tool argument validation distinguishes a boolean from an integer,
and a JSON string from a parsed object.

For targeted diagnosis:

```bash
python bench/tool_smoke.py --list
python bench/tool_smoke.py --case strings,typed,round_trip --mode stream \
  --repeat 3 --output results/tools-targeted.json
```

The report includes the synthetic prompts, assembled responses and exact failures.
A failure can come from model generation, API behavior or parser handling. Use the
captured response and server logs to identify the cause; the pass count is a
functional compatibility result, not a benchmark of model intelligence.

## CPU regression checks

```bash
python -m unittest discover -s bench -p 'test_*.py' -v
bash -n exllamav3-tabby/setup.sh exllamav3-tabby/env.sh exllamav3-tabby/serve.sh
```

These checks cover missing usage, premature stream termination, in-band errors,
model identity, fragmented tool data, typed arguments, safe config rendering,
context limits, memory-header classification and refusal to overwrite local work.
They do not load a model or validate CUDA performance.
