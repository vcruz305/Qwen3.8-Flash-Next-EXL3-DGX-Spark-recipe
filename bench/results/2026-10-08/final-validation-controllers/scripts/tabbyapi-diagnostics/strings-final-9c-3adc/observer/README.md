# Bounded raw strings observation

This external diagnostic observes the original `strings` case in the recipe's unchanged `bench/tool_smoke.py`, once in each response mode. It changes no repository production source. Use it only on the owned loopback diagnostic server, then remove its environment from the next ordinary server.

The observer requires exact clean engine `9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2` and Tabby `3adc9813f30c4b92d110e7e0235129031dae1c25`. Source and import-path checks run after main has configured its allocator but before the server starts. Other Python commands ignore the startup hook, even when the configuration path is invalid.

## Capture contract

A request matches only when the retained request fields contain the exact original single synthetic user message and four-string `record_strings` function schema, `tool_choice=auto`, parallel calls enabled, `enable_thinking=false`, maximum1024 tokens, temperature0, top_k1, top_p1, n1, and the matching stream mode. The real Tabby request schema discards the wire seed field, so this field is explicitly outside the observer matcher; the unchanged greedy test client still sends seed0.

The collector receives a request copy after template rendering. The matcher permits only the framework additions `messages`, `tools`, `functions`, `add_generation_prompt`, `tool_choice`, `parallel_tool_calls`, and the four special-token keys. Present messages/tools/flags must exactly match the original request; unknown keys or conflicting values reject capture. The actual augmented dictionary is preserved in `matched_request.template_vars`. Special-token values may be strings, integers, or null, as returned by the model adapter.

At most two matched requests are recorded per process. Later matches and unrelated requests continue unchanged without capture. Each record preserves:

- The request ID, response mode, and actual initial reasoning phase.
- The exact rendered prompt and its SHA256.
- The existing engine finish result's `full_completion`, with a presence indicator.
- The backend's separately accumulated `full_response` (the existing finish callback's `full_text` argument).
- Existing scalar finish/token/draft metrics.

Only collector entry/exit and `handle_finish_chunk` are wrapped. The collector is awaited normally; no new waits or token-level hooks are added. The finish callback receives its original objects and returns its original result. No token/KV tensors are inspected, converted, transferred, or synchronized. The raw strings are retained before the final finish chunk returns to the collector; earlier streaming chunks may already have been parsed.

JSON publication occurs after the original collector returns, with private permissions, an exclusively claimed output directory, and no replacement of existing files or symlinks. A publication failure logs only its exception type and preserves the native request return/exception. A missing trace must fail the diagnostic completeness check. This observer provides evidence; it does not repair or normalize any prompt, output, argument, or parser state.

## Per-load configuration

Create a new config JSON and a new output path for each owned model load. The output path's parent must already exist; the observer claims the final directory.

```json
{
  "engine_repo": "/home/cruzspark/qwen38-exl3-20261008/exllamav3",
  "tabby_repo": "/home/cruzspark/qwen38-exl3-20261008/tabbyAPI",
  "engine_commit": "9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2",
  "tabby_commit": "3adc9813f30c4b92d110e7e0235129031dae1c25",
  "run_label": "final405",
  "max_records": 2,
  "output_dir": "/home/cruzspark/qwen-overnight-20261008/results/OWNED-UNIQUE-DIRECTORY/raw-strings"
}
```

Use `run_label=finalcyber` and a different directory for the Cyber load. Set the following variables **only in the server process environment**, retaining any required existing server Python path:

```text
TABBY_STRINGS_OBSERVER_CONFIG=/absolute/path/to/owned-config.json
PYTHONPATH=/absolute/path/to/this/observer
```

The owning controller must require exactly two complete `request-*.json` files with distinct IDs, one stream and one nonstream, the expected run label, both raw strings present, no returned or raised collector error, and a manifest containing the exact pinned sources. Compare each record to its matching original client request/result; do not infer parser corruption from a model that already emitted incorrect raw text. The manifest/config/deployment hashes and loaded model/template identity belong in the enclosing controller evidence.

These are diagnostic runs. Do not use their timings as throughput measurements.

## CPU checks

```bash
python3 -m unittest discover -s observer -p test_strings_observer_cpu.py -q
```

Thirteen cases cover exact scope, stream modes, cap/unmatched delegation, original argument/return/exception identity, native-value read rejection, exclusive files, publication failure, and real deferred startup without importing Torch.

`check_original_fixture_cpu.py` additionally imports the actual Tabby Pydantic request model and original recipe client, then checks both response modes and both saved 4.05/Cyber original requests. The eight actual-fixture checks run without Torch or inference. Two additionally execute the actual request-formatting functions and Jinja renderer through the collector deep-copy handoff; only the loaded-model holder and collector body are CPU stand-ins. They verify that the unchanged observer wrappers recognize and capture the actual augmented request state.
