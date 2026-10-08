# Explicit prompt templates

The launcher accepts `PROMPT_TEMPLATE=/absolute/path/to/template.jinja`. An empty or omitted value keeps the model pack's original template selection. The override uses TabbyAPI's existing `model.prompt_template` setting; it does not modify model weights, tokenizer files, or the model-view directory.

The file must exist, be readable UTF-8 and have the `.jinja` extension. Before replacing the rendered config or starting Tabby, the launcher compiles it with the installed Tabby template environment. An explicit invalid template therefore stops this launch instead of relying on Tabby's fallback template search. This validation also runs with `DRY_RUN=1`, so a template dry run requires the installed Tabby checkout and its base Python dependencies. It does not load a model.

## Cyber-Frost 3.87bpw: honor the request's thinking flag

The Cyber-Frost pack inspected for this recipe begins its chat template with:

```jinja
{%- set enable_thinking = true %}
```

That assignment replaces an explicit request value of `false`. The supplied [Cyber-Frost override](../exllamav3-tabby/templates/cyber-frost-3.87bpw-thinking.jinja) removes only this first line. Its existing undefined/true/false branches already preserve thinking by default and support an empty thinking prefix when disabled. All other pack instructions, baked system text, tool formatting and history handling remain in the file.

The adjacent [provenance record](../exllamav3-tabby/templates/cyber-frost-3.87bpw-thinking.provenance.json) identifies the inspected tokenizer config and the exact change:

| Item | Value |
| --- | --- |
| Original template | 9,005 bytes; SHA256 `ba1946683f7615254fb246f0c0a652fd3aa02066ef8328ed4b8af08219749395` |
| Removed prefix | 34 bytes, including its newline |
| Override | 8,971 bytes; SHA256 `666b82b29f5801f4f546e5724b45bf5f14be7d20b66149df44164626b072ce6d` |

Select it explicitly when serving that pack, with your qualified runtime and memory settings:

```bash
RECIPE_HOME=/absolute/path/to/qualified-runtime \
STATE_DIR=/absolute/path/to/new-server-state \
MODEL_DIR=/absolute/path/to/CYBER-FROST-3.8-EXL3-SAGE-3.87bpw \
PROFILE=single NGRAM_RAM=false \
PROMPT_TEMPLATE=/absolute/path/to/recipe/exllamav3-tabby/templates/cyber-frost-3.87bpw-thinking.jinja \
bash exllamav3-tabby/serve.sh
```

This is an opt-in serving profile for the inspected pack, not an automatic replacement for other models or future pack revisions. If the pack's original template changes, compare it with the provenance record and review the override again.

A chat request can select the existing template branch with top-level `"enable_thinking": false` or `true`. With the OpenAI Python client, pass the Tabby extension through `extra_body`:

```python
response = client.chat.completions.create(
    model="Qwen3.8-Flash-Next-EXL3",
    messages=[{"role": "user", "content": "Reply with READY."}],
    max_tokens=128,
    extra_body={"enable_thinking": False},
)
```

Tabby's normal template-variable precedence still applies: request `template_vars` and model `template_vars_force` can override the flat field. The asset changes prompt construction. Whether the model follows that prompt, produces a useful response, or calls a tool correctly still requires live API checks on the selected model and runtime.

CPU checks exercise Tabby's actual renderer on simple messages, system instructions and tool history. Default/true rendering remains byte-identical in 36 combinations of reasoning effort and generation-prompt settings. For explicit false, the rendered differences are exactly the template's existing reasoning-instruction omission and empty thinking prefix. The tests preserve literal tool arguments and responses as well.

## Verify what the server loaded

The deployment snapshot adds `prompt_template_override` only when an override is selected:

```json
{
  "path": "/absolute/path/to/template.jinja",
  "resolved_path": "/resolved/path/to/template.jinja",
  "bytes": 8971,
  "sha256": "SHA256 of the file bytes",
  "content_sha256": "SHA256 of the UTF-8 text loaded by Tabby"
}
```

The snapshot also records `environment.PROMPT_TEMPLATE` and the rendered `config.values.model.prompt_template`. Symlink aliases retain both their requested path and resolved target. Text-mode reads normalize CRLF and CR newlines to LF, so the file-byte and loaded-text hashes can differ for an external file.

After readiness, `GET /v1/model` exposes the actual template text at `parameters.prompt_template_content`. Hash that value as UTF-8 and compare it with `content_sha256`; the template's short name alone does not prove that the requested content was loaded. The [matrix runner](../bench/matrix.md) performs this check before measurements and rejects missing or fallback content. It fingerprints the external file when normalizing the job and rechecks it before launch, before measurement and after the clients, so changing the file or symlink target invalidates the run and the resume identity.

Keep the file unchanged while the server is running. Changing it on disk does not replace the already loaded template; restart and retain a new deployment record. A permanent service can set the same absolute path in its environment file, following the [service guide](service.md).

## Comparable measurements

Keep the template content hash and request thinking mode explicit in every result. A run on the original Cyber template, which forces thinking, is a different prompt profile from a run that honors explicit false. Retain the original results and label both profiles. Template changes can affect prompt token count, output behavior and speed; they must not be attributed solely to an engine kernel or server parser change.

For a matrix job, add `"PROMPT_TEMPLATE": "/absolute/path/to/template.jinja"` to its `env`. No other runtime, network or state ownership rules change. Frozen historical controllers and results remain unchanged; only the public runner gains this optional provenance check.

## CPU verification

Portable path/provenance checks run in the normal recipe test suite. To include the actual Tabby compilation, lookup, launcher and rendering checks, run with its installed base dependencies:

```bash
TABBY_SOURCE=/absolute/path/to/runtime/tabbyAPI \
/absolute/path/to/runtime/venv/bin/python -m unittest discover \
  -s bench -p test_prompt_templates.py -v
```

All ten tests run without inference, an API server or GPU access. If `TABBY_SOURCE` is omitted, the four checks requiring the actual Tabby renderer are explicitly skipped.
