# Beta: minimal OpenAI shim over the exllamav3 fork

> **Beta.** The supported API is [`../serve.sh`](../serve.sh): the latest TabbyAPI on the
> same fork runtime. Use this shim only for A/B comparisons against TabbyAPI.

```bash
bash exllamav3-tabby/beta/serve_openai.sh          # 127.0.0.1:8899, same port as serve.sh
```

It uses the same venv, fork check, pack check, and GB10 knobs as `../chat.sh`
(`-mtp -ndt 5 -dds -dc 0.6 -cq 8,8 -cs 262144 -topk 1`, n-gram table in RAM).
Run it *instead of* `serve.sh`, not alongside it: both default to port 8899, and two
copies of the model do not fit.

## Known limits

- **One generation at a time.** A global lock around the generator, and the cache is
  built for batch 1. Concurrent clients queue.
- Hand-rolled Qwen prompt format rather than the pack's `chat_template.jinja`.
  Tool definitions go in as prose, and `<function=…><parameter=…>` XML is parsed into
  OpenAI `tool_calls`. `<|im_start|>` is treated as a stop.
- `/v1/chat/completions` and `/v1/models` only. No `/v1/completions` and no structured output.
- If a request omits `max_tokens`, the output cap defaults to 32,768 (was 2,048, which
  silently truncated clients that send no cap; from PR #10).
- Binds `127.0.0.1` by default and has no auth. Only set `HOST=0.0.0.0` on a trusted network.

## Measured

2026-09-20, one GB10, fork `785f206` (before the mixed-K kernel landed in `329e051`):
a greedy 400-token code job ran at **79.5** wall tok/s / **83.8** engine / **74%**
draft acceptance, matching `chat.py`'s 79. A Nous Hermes tool loop at ~80k prompt
survived. Not re-measured on the current pin.
