# API resilience and automatic tool selection checks

The recipe includes two serial, standard-library clients for checking a running TabbyAPI server:

- [`bench/api_resilience.py`](../bench/api_resilience.py): raw completions, model identity, forced tools with reasoning, nullable strings, invalid-request recovery, and an early streaming disconnect.
- [`bench/auto_compatibility.py`](../bench/auto_compatibility.py): ordinary responses with tools available, reasoning, and explicit client output constraints.

Run these **between performance measurements**, while the server is otherwise idle. Their wall times describe individual functional checks. Use the [benchmark clients](../bench/README.md) for throughput and concurrency measurements.

Both clients use Python 3.10 or later with no additional packages. Every prompt, function declaration, and requested argument is a synthetic fixture. The clients never execute a returned function or contact an external tool. The separate [OpenAI SDK smoke tests](tool-calling.md) exercise the public SDK rather than these standard-library clients.

## Run the checks

Start the server with the settings and model pack you intend to validate. Supply its launcher-generated `deployment.json` so the report records the source revisions, packages, model, and configuration associated with the run:

```bash
python3 bench/api_resilience.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Qwen3.8-Flash-Next-EXL3 \
  --metadata "$HOME/qwen38-exl3/state/deployment.json" \
  --label candidate \
  --output results/candidate-api-resilience.json

python3 bench/auto_compatibility.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Qwen3.8-Flash-Next-EXL3 \
  --metadata "$HOME/qwen38-exl3/state/deployment.json" \
  --label candidate \
  --output results/candidate-auto-compatibility.json
```

Use the actual `STATE_DIR/deployment.json` if you selected a separate state directory. `--metadata` is optional and accepts a JSON object. The report stores it in `provenance`, together with the source path and SHA-256 of the exact input file in `metadata_source`. This is caller-supplied provenance; it does not independently attest to the server's currently loaded source revision. The observed model ID and all completion model fields are recorded separately.

For authentication, export the credential as `TABBY_API_KEY`; `--api-key-env NAME` selects another environment variable. The clients do not record the authorization header or credential environment variable. The launcher-generated deployment snapshot is already redacted. The clients copy the supplied metadata object into their reports.

The endpoint defaults to `http://127.0.0.1:8899/v1`, or `OPENAI_BASE_URL` when set. `--model` defaults to `OPENAI_MODEL`; if neither is supplied, `/v1/models` must advertise exactly one model. `--timeout` defaults to 45 seconds per HTTP operation. Requests are serial and are never automatically retried.

**Choose a new output filename for each run.** An existing file, directory, or dangling symlink is refused before the first HTTP request. A second process cannot claim the same output path. There is no overwrite option. The report writer publishes complete JSON atomically before preflight, after each check, and at completion. It retains the previous complete checkpoint if publishing a later checkpoint fails. A report without `finished_utc` is incomplete and must not be treated as a finished validation run.

Exit statuses are `0` when every selected check passes, `1` for a failed check, and `2` for invalid CLI input, unreadable/invalid metadata, or report-output errors. Individual case failures are reported while independent remaining cases continue. A failed preflight prevents generation.

## Resilience suite

The full resilience suite contains **19 checks and 27 HTTP requests**, including preflight and recovery requests. It uses TabbyAPI's `/v1/model`, `/v1/models`, `/v1/token/encode`, and `/health` endpoints in addition to both completion endpoints.

| Group | Checks | Required behavior |
| --- | ---: | --- |
| Preflight | 1 | Resolve advertised/current model identity and verify that the raw prompt `X` is exactly one token with BOS disabled. No generation runs if this check fails. |
| `raw` | 2 | Non-streaming and streaming `/v1/completions` use the verified one-token prompt and emit exactly eight tokens, confirmed by actual usage and `finish_reason: length`. This exercises the one-token prefill path relevant to MTP. |
| `aliases` | 6 | Test the canonical model name, unknown names on both completion endpoints in both modes, and loaded-model stability. Requests using the public alias are also checked throughout the suite. |
| `forcing` | 4 | Required and named tool choices with reasoning enabled, in both response modes. The named request must select `ping` even though its prompt asks for `alternate_ping`. A nonempty reasoning phase, one completed call, and exact empty arguments are required. |
| `nullable` | 2 | Required `echo_nullable` call in both modes. Every property has `type: [string, null]`; returned values and JSON types must match the original fixture exactly. |
| `invalid` | 3 | Unknown named tool, negative output budget, and invalid grammar. Each request is immediately followed by a healthy completion, including when the invalid request returns an unexpected status. |
| `disconnect` | 1 | Close SSE after visible generated output and before completion, immediately generate another reply, then check service health and loaded-model stability. |

The nullable fixture requires the strings `123`, `true`, and `[1,2]`; code containing four leading spaces and one trailing newline; and an actual JSON null. Numeric/bool/array coercion, whitespace stripping, or replacing the required null with an empty string fails. The fixture is unchanged from the standalone diagnostic used during the Spark investigation.

Unknown model names have two acceptable documented outcomes: a client error, or a response identifying the actual canonical model when inline model loading is disabled and the server ignores the unknown name. Echoing the invented name as though it were served fails. The case records which behavior occurred.

Recovery requires all three conditions: `finish_reason: stop`, no tool calls, and the exact `READY` sentinel after trimming outer whitespace. This catches a leaked tool grammar or sampler state that produces a syntactically valid response with the wrong behavior.

For a smaller run, select groups explicitly:

```bash
python3 bench/api_resilience.py \
  --cases forcing,nullable \
  --model Qwen3.8-Flash-Next-EXL3 \
  --output results/forced-nullable-check.json
```

Preflight always runs. `--one-token-text` can select a different prompt if another tokenizer does not encode `X` as one token. `--settle-seconds` controls the short delay before the final disconnect health check, after the immediate recovery generation; its default is two seconds.

## Automatic tool selection suite

The auto suite contains **9 checks and 11 HTTP requests**: the same preflight plus eight generations. Every generation declares a `ping` function and keeps `tool_choice: auto`.

| Variant | Non-streaming and streaming requirements |
| --- | --- |
| `plain` | No tool is needed; return plain `READY` with a normal stop and no call. |
| `reasoning` | Produce a nonempty reasoning phase, then return plain `READY` with no call. A 24-token reasoning budget keeps the synthetic check short. |
| `client_grammar` | Preserve the explicit client Lark grammar `start: "READY"`, returning the requested ordinary text with no call. |
| `client_json_schema` | Preserve the standard named JSON response schema and return exactly `{"status":"READY"}`, with no call. |

These checks verify that automatic tool parsing does not turn every request into a forced call or replace the client's own output constraint. Responses must end with `stop`; `length`, empty output, a tool call, a wrong JSON object, or missing required reasoning fails. The payloads are unchanged from the standalone auto diagnostic.

## What the reports establish

The report includes every synthetic request, JSON response or received SSE frame, response ID, model field, finish reason, returned usage, error body, and wall time. The top-level `kind` identifies the runner, `label` names the run, and `summary` counts check results. Reports preserve full failure evidence instead of converting failed requests into successful samples.

Completed streams require `[DONE]`, stable response IDs, assistant role before generated output, valid tool indices/types/IDs, and consistent actual token totals. A requested output budget is never substituted for returned usage. The intentionally aborted stream records its partial frames and `client_aborted` state; it is not passed through the completed-stream validator, because it is deliberately closed before a final usage/finish event.

A successful disconnect case establishes that the client closed a response early and that an immediate subsequent generation, health check, and model check succeeded. Exact server-side cancellation latency requires the corresponding server logs.

Tool argument correctness also depends on the model following the requested content. Structural Qwen tool grammars constrain call syntax, declared function names, and call count; they do not supply full JSON Schema guarantees or make an empty nullable string equivalent to null. These clients retain exact argument checks so a model copying failure remains visible.

The two live suites share preflight and protocol validation. Their 19 and 9 result counts are check counts, not a claim of 28 independent capabilities or a throughput comparison.

## Offline tests

From the recipe root:

```bash
PYTHONPATH=bench python3 -m unittest -v test_api_resilience test_auto_compatibility
```

This runs **24 unit-test methods**: the **17 existing protocol/fixture tests** plus **7 report and CLI tests**. Parameterized subcases are not added to those counts. The HTTP openers are injected in-memory fixtures; the tests never access the network or a GPU.

The original tests cover strict SSE/usage validation, model identity, exact argument types and whitespace, forced reasoning, disconnect behavior, and recovery false-pass guards. The seven packaging tests cover exclusive report ownership, concurrent output claims, interrupted atomic publication, an initial checkpoint before HTTP, metadata/fingerprint handling, CLI failure before network access, and preservation of the original request sequence fingerprints.

The recipe-wide CPU discovery command also includes these tests:

```bash
python3 -m unittest discover -s bench -p 'test_*.py' -v
```
