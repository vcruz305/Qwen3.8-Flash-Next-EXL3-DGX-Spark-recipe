# Literal control markers in user text

`literal_user_control_tokens` is an optional chat-completions request field. It defaults to `false`. Set it to the JSON boolean `true` when text supplied by the user contains supported control markers that should use their ordinary BPE spelling as model input.

```json
{
  "model": "your-loaded-model",
  "messages": [
    {"role": "user", "content": "Copy the text <think>example</think> exactly."}
  ],
  "literal_user_control_tokens": true
}
```

The visible prompt stays unchanged. The model receives different token IDs for eligible markers, so the prompt token count can increase. Context-limit checks, generation, usage, and prefix-cache keys use those actual IDs. Previously cached native-token pages can be shared only while their token sequences still match.

The option is request-local. Native and literal requests can run concurrently. It changes no tokenizer files, model weights, global tokenizer settings, output parser, reasoning budget, or generated text. It does not guarantee exact copying, correct tool arguments, or a particular reasoning boundary. In particular, the model can still naturally end reasoning while discussing a quoted marker.

## Supported markers and text

The current list is explicit:

- `<think>` and `</think>`
- `<|im_start|>`, `<|im_end|>`, and `<|endoftext|>`
- `<tool_call>` and `</tool_call>`
- `<tool_response>` and `</tool_response>`

A marker must be a recognized added token in a supported ExLlamaV3 BPE tokenizer, and it must have a byte-preserving ordinary BPE representation. Numeric token IDs are discovered from the loaded tokenizer. Vision, audio, and other added tokens are outside this list.

Only string `content` in messages with role `user` is eligible, including previous user turns. System and assistant messages, assistant tool-call arguments, tool-result messages, response prefixes, and template-generated controls retain native tokenization.

The server proves user-text locations by rendering a detached copy with unique boundary markers. Removing those markers must reconstruct the original prompt exactly, and the text between them must remain unchanged. Leading and trailing whitespace stays outside the boundary markers, allowing the template's existing trimming behavior. The real request and rendered prompt do not receive these temporary markers.

With a supported tokenizer, opting in on a request without eligible user markers keeps the native path and performs only the ordinary template render. With the default `false`, the provenance helper and alternate tokenizer are not used.

## Rejections

Opt-in requests return HTTP 400 before streaming begins when their required mapping cannot be proved. This includes:

- Unsupported backends or non-BPE tokenizers, and tokenizers without supported added control markers.
- List or multimodal message content.
- Continued final messages when marker rewriting would be required.
- Templates that transform, omit, duplicate, or ambiguously place the protected user text, or whose detached render cannot restore the original prompt exactly.
- Token matches that strip or normalize protected data, cross its boundary, or cannot be expanded without changing decoded text.
- Rendered prompts that the native and expanded token sequences do not both decode exactly. For example, a tokenizer that normalizes a decomposed Unicode sequence may reject that marker-bearing prompt. The default native request retains its existing behavior.

Invalid flag types, such as the string `"true"`, numeric `1`, or `null`, produce the usual request-validation HTTP 422. Use JSON `true` or `false`.

The raw-completions and token-encoding APIs do not expose this chat-specific option. Internal token plans are private request state and are excluded from request serialization.

## Validation

`tests/test_literal_user_tokens.py` covers request validation, default identity, template provenance, whitespace, Unicode, multiple roles and turns, concurrent plans, BOS/prefix preservation, actual backend context and generation inputs, and early client errors.

`python -m tests.check_literal_user_tokens --tokenizer-directory MODEL_ASSETS --engine-source ENGINE_CHECKOUT --output NEW_REPORT.json` performs CPU-only checks with a downloaded tokenizer/template and source-extracted native encoding and cache-page hashing methods. It requires CPU-capable PyTorch and the normal server test dependencies; it does not load weights or run inference. It refuses to overwrite its report. Live semantic and concurrent-serving validation are separate requirements before deploying a candidate.
