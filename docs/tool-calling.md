# Tool calling on the Spark recipe

This recipe uses the Qwen pseudo-XML tool format through the
[vcruz305 TabbyAPI fork](https://github.com/vcruz305/tabbyAPI). The server exposes
OpenAI-compatible JSON tool calls; clients do not need to produce or parse the
model's XML. Configure `TOOL_FORMAT=qwen3_coder` when using a pack whose model
metadata does not select that format correctly.

The contracts below describe the implementation and its CPU regression checks.
Run the live validation commands against each deployed pack to verify model
behavior, token budgets and the installed runtime. The SDK checks use synthetic
weather results and execute no external tools.

## Supported request modes

| Request setting | Behavior |
|---|---|
| Omitted `tool_choice`, or `"auto"` | The model can answer with text or call tools. For unconstrained Qwen requests, a content grammar constrains call syntax once a call opener is emitted. |
| `tool_choice: "none"` | Tool declarations are removed from the generation prompt, and generated text is not converted into tool calls. Existing tool history can still be supplied. |
| `tool_choice: "required"` | Qwen XML constrained sampling requires at least one complete call to a declared function. |
| Named `tool_choice` | The grammar permits the selected function, and the template receives its declaration. |
| `parallel_tool_calls: false` with `auto` | The automatic content grammar permits at most one call. Where that grammar is inactive, the API returns at most the first parsed call. |
| `parallel_tool_calls: false` with a forced choice | The grammar restricts generation to one call. |

Tools without parameters are supported. For example,
`{"type":"function","function":{"name":"ping"}}` declares a no-argument function;
a successful call has the JSON argument string `"{}"`.

A named request uses the ordinary OpenAI shape:

```json
{
  "model": "Qwen3.8-Flash-Next-EXL3",
  "messages": [
    {"role": "user", "content": "Get the weather for Tokyo in celsius."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city", "unit"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
  "parallel_tool_calls": false,
  "max_tokens": 1024,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

### What the grammar enforces

Required, named and automatic Qwen modes use llguidance to constrain call
structure, function names and call count. Automatic mode permits ordinary text
and zero calls; after a complete `<tool_call>` or bare `<function=` opener, it
requires a complete call before a normal end of generation. Text before and
after calls remains allowed. A token budget can still interrupt a call.

The grammar uses the actual tokenizer's added token IDs
for native markers such as `<tool_call>`. Hugging Face added tokens marked
`special=false` also need this handling; a grammar containing only their text
can mask the model's normal marker tokens.

When the template starts in a reasoning phase, the grammar applies after that
phase closes. In forced modes, tool examples inside reasoning remain reasoning
text; following reasoning, only calls and bounded layout whitespace are allowed.
Automatic mode preserves the configured `tool_calls_in_reasoning` behavior,
reasoning effort and reasoning budget. Its grammar governs the content phase.

Automatic grammar installation is skipped for another explicit output constraint
(`grammar_string`, `regex_pattern`, `json_schema` or non-text `response_format`),
`response_prefix`, or `continue_final_message`. Those requested settings retain
their existing behavior. It is also skipped for `none`, requests without tools,
and non-Qwen formats. A text `response_format` can be combined with automatic
tool syntax constraints.

Argument parsing is schema-aware, but the XML grammar does **not** implement
full strict JSON Schema enforcement. A declaration containing `strict: true`
does not add full schema validation. Applications should validate required
properties, allowed values, bounds and types before executing their tools.

Qwen grammar construction rejects empty, duplicate or malformed function names
with HTTP 400. A forced request also rejects an undeclared selected function,
unsupported tool formats, and competing output constraints such as
`grammar_string`, `regex_pattern`, `json_schema` or non-text
`response_format`. A forced choice also rejects `response_prefix` and
`continue_final_message`, because its grammar starts at a new call. A missing
or broken llguidance backend reports an error; it does not silently disable
the constraint.

These forced modes are implemented for the `qwen3_coder` format and its
registered aliases. Other formats retain their own parsing behavior and can
use `auto` or `none`.

## Argument types and literal content

Qwen writes parameter values as raw text. The request schema helps distinguish
a string from a JSON number, boolean, object, array or null.

| Parameter schema | Raw model parameter | Returned JSON value |
|---|---|---|
| `{"type":"string"}` | `123` | `"123"` |
| `{"type":"string"}` | `true` | `"true"` |
| `{"type":"string"}` | `{"nested":[1,false]}` | `"{\"nested\":[1,false]}"` |
| `{"type":"integer"}` | `123` | `123` |
| `{"type":"boolean"}` | `true` | `true` |
| `{"type":["string","null"]}` | `123` | `"123"` |
| `{"type":["string","null"]}` | `null` | `null` |
| `{"type":["string","null"]}` | An empty parameter body | `""` |
| `{"type":["string","integer"]}` | `123` | `123` |

For unions containing a string branch, a non-string JSON value is decoded only
when its type is allowed. Otherwise the raw string is preserved. Raw `null`
in a nullable string union deterministically means JSON null. To preserve the
literal text `null` unambiguously, use a string-only parameter schema. An empty
parameter body remains an empty string; it is not coerced to null just because
the schema also permits null.

String values retain quotes, indentation, carriage returns and final blank
lines. Only one surrounding LF inserted by the Qwen template is removed at
each end. Common local schema references, string enums and type unions also
inform this conversion. Schemas without a resolved string branch retain the
legacy JSON-literal conversion behavior.

Literal `<think>`, `</think>`, `</function>`, `<tool_call>` and
`</tool_call>` inside a parameter are kept as argument data. This matters for
tools that write code or files containing markup. The first literal
`</parameter>` still closes that parameter: this delimiter is ambiguous in
the model's wire format. Tools that must transfer arbitrary content containing
it can use an explicitly encoded argument and decode it in the tool
implementation. An unmatched closing wrapper such as `</tool_call>` outside a
call remains ordinary content or reasoning text; a closing tag alone does not
start a call.

## Streaming, history and errors

Each streamed choice starts with `delta.role: "assistant"`. A tool's first
delta includes its index, call ID, type and function name. Later deltas append
argument fragments under the same index. Reasoning, assistant text and tool
deltas preserve their order, including when the inference engine returns
several of them in one chunk.

The official OpenAI Python SDK's public stream accumulator is exercised by
`bench/sdk_smoke.py`. The test validates the final completion object and sends
the SDK-assembled assistant message into a synthetic tool-result followup
without injecting or repairing its role. This catches streams that look
plausible as individual chunks but cannot be reused as conversation history.

For a tool round trip, retain the assistant message containing `tool_calls`,
then append a `role: "tool"` message for each result with the matching
`tool_call_id`. Parallel results can arrive in a different order; Tabby orders
consecutive tool-result messages by their corresponding assistant call IDs for
templates that require positional ordering.

Wait for a successful `finish_reason: "tool_calls"` before executing calls.
The following outcomes require handling by the client:

| Outcome | API behavior |
|---|---|
| Token budget exhausted | `finish_reason: "length"` remains length, including when tool fragments or a completed prefix were generated. |
| Normal stop inside a call | Generation error; an incomplete trailing call is not silently discarded to make the turn appear successful. |
| Malformed or nested function structure | Generation error at normal stop. |
| Empty tool wrapper at normal stop | Generation error instead of a successful empty call. |
| Non-streaming tool parse error | HTTP 502 with an explanation. |
| Streaming tool parse error | An error event; previously emitted partial fragments do not become a successful tool call. |
| Disconnect or missing final stream marker | Incomplete response; the recipe clients fail the validation run. |

The SDK's higher-level parsing interface may raise `LengthFinishReasonError`
on a length-limited response. The SDK smoke test treats it as a failed
functional case and saves the partial completion when the SDK provides one.

Chat and text responses preserve a requested model ID only when it matches the
loaded directory, resolves to that directory through an alias, or is an
explicitly configured dummy alias. If legacy inline-loading behavior ignores
an unknown request name, the response names the actual loaded model. Streaming
text completions also use the standard `model` field.

## Run the validation

Run tests when no throughput benchmark is active. Each result should identify
the pack, recipe/engine/server commits and serving configuration. The launcher
writes this metadata to `$STATE_DIR/deployment.json`.

The dependency-free tool test covers automatic and forced choices, adversarial
forced prompts, nullable and structured arguments, literal markup, reasoning,
parallel calls, reversed results, repeated turns and truncation:

```bash
python3 bench/tool_smoke.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Qwen3.8-Flash-Next-EXL3 \
  --mode both --repeat 1 \
  --metadata ~/qwen38-exl3/state/deployment.json \
  --output results/tools.json
```

Install the optional SDK client in its own environment. The pinned client
version below is the one used for the independent SDK replay; it is separate
from the server's runtime dependencies.

```bash
python3 -m venv .venv-sdk
.venv-sdk/bin/python -m pip install -r bench/requirements-sdk.txt

.venv-sdk/bin/python bench/sdk_smoke.py \
  --base-url http://127.0.0.1:8899/v1 \
  --model Qwen3.8-Flash-Next-EXL3 \
  --metadata ~/qwen38-exl3/state/deployment.json \
  --output results/sdk-smoke.json
```

Set `API_KEY` for an authenticated endpoint. Both clients support an explicit
`--response-model` for a separately verified alias on older server versions;
all other response-model mismatches fail.

The SDK runner makes five checks: a streamed text response, named calls in
both response modes, and tool-result followups in both modes. It records
requests, responses, SDK event counts, wall times and errors as JSON. Wall
times are diagnostic; this runner does not report engine throughput. A failure
or skipped dependent case gives a nonzero exit status.

Its CPU fixtures use the real SDK and an in-memory HTTP transport:

```bash
.venv-sdk/bin/python -m unittest discover -s bench -p 'test_sdk_smoke.py' -v
```

Repeat the live tests after switching each pack into service. Preserve failed
baseline runs as well as candidate runs. Changing a test prompt or expectation
requires a new comparison; do not relabel old failures as passing results.

## Implementation references

- [TabbyAPI tool calling documentation](https://github.com/vcruz305/tabbyAPI/blob/main/docs/10.-Tool-Calling.md)
- [Qwen argument parser](https://github.com/vcruz305/tabbyAPI/blob/main/endpoints/OAI/utils/toolcall_formats/qwen3_coder.py)
- [Automatic and forced tool-choice grammars](https://github.com/vcruz305/tabbyAPI/blob/main/endpoints/OAI/utils/tool_choice.py)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Metric definitions and benchmark commands](../bench/README.md)
