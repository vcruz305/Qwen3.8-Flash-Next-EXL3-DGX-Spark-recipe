# Mandatory tool-call reasoning EOS fix

This archive qualifies an independent server fix for required or named Qwen tool calls. The request-local literal-input feature is not required by the fix. The baseline is Tabby `f650bb5389e0a273549e47d4d26a765760c013e1`; the EOS-only commit is `fd8cbeb1fbb3cec4d2141558d5cfd63a500d50c4`.

The completed predecessor [raw diagnostic](../literal-feature-diagnostic/README.md) observed a real termination gap: the multi-turn streaming request sampled native `im_end` ID248046 at processed position24 while the response was still reasoning. No forcing, phase change, rewind or decoder call preceded that stop. The required tool-call grammar had not yet been attached. All8 observed native completion strings matched the backend strings. The original a70 qualification and its failures remain unchanged.

Required and named choices already treat tool examples inside reasoning as reasoning text. The fix suppresses only implicit model EOS during that initial phase, using a separate sampler and the existing guarded producer handoff. A verified natural or budget-forced reasoning ending restores the ordinary sampler before content generation. Automatic tools, tool-none requests, requests that start in content and unsupported producer paths retain their previous behavior. Explicit stop IDs and stop strings that overlap the EOS piece keep priority; the request's hard maximum and existing token bans remain in force. The stop list is copied before adding model EOS, preventing reuse of one request from reclassifying implicit EOS as caller stops.

The fix does not rewrite output or promise exact argument values. In the same predecessor capture, concurrent2's native parameter already omitted the requested trailing `endoftext` marker; that token was not sampled and no decoder was called. This generation omission is outside the fix. Both Unicode twins succeeded in this capture, so the earlier one-mode length failure remains unreproduced. Processed native IDs are not automatically emitted output: the final stop token is processed but withheld, and rewinds/healing must be considered. The retained sidecar records these distinctions.

## CPU and source qualification

| Evidence | Tests | Subtests | Scope |
|---|---:|---:|---|
| Spark EOS-only fd8, CUDA hidden |409|5,199|Full repository selection, zero skips, exit0|
| Spark combined f4a, CUDA hidden |430|5,250|Literal-input candidate plus clean EOS cherry-pick, zero skips, exit0|
| WSL EOS-only full selection |393|5,199|Two existing engine-import modules skipped|
| Focused existing/new Tabby phase tests |57|151|Collector policy, stop precedence, backend arming/restoration and reuse|
| Independent new Tabby test replay |11|40|Subset of the focused suite|
| Real native source proof |8|—|Sampler, Job, MTP acceptance, mask restoration and per-job isolation|

These are overlapping suites and repeat runs; their counts must not be added as unique coverage. Both Spark setup checks exit0 with a verified engine build fingerprint and clean package requirements. The actual Spark runs resolve the two WSL import skips.

The native proof executes exact engine24 source on CPU with real Torch tensors and controlled logits/allocation. It covers a proposed EOS draft rejected by target sampling; same-window natural and forced sampler restoration; explicit stop and maximum-token priority; zero-budget forced-tail draining; and interleaved job state. It inspects retention of the EOS ban before the fused greedy tail, but does not run the CUDA fused kernel. This is sequencing and mask coverage, not GPU model-quality or throughput evidence.

The full commands are:

```bash
# From the applicable clean Tabby source directory, with its runtime venv:
CUDA_VISIBLE_DEVICES='' /path/to/venv/bin/python -m pytest -q tests/test_*.py

# Focused integration suite (the WSL command and dependencies are retained in logs/review):
CUDA_VISIBLE_DEVICES='' python -m pytest --noconftest -q \
  tests/test_mandatory_tool_eos.py tests/test_reasoning_budget.py \
  tests/test_natural_reasoning_handoff.py
```

`cpu/stage_eos_cpu.py` records the exact Spark source-view creation and CUDA-hidden commands. It created only new fd8 and f4a runtime views with the existing engine/venv symlinked; it performed no model, API, GPU or service actions. `cpu/spark/status.json` records actual source trees, exits, timestamps and clean worktrees. The five reviewed source files and commit patch are retained under `source/` and `mandatory-tool-eos.patch`. `summary.json` binds the commits and individual source hashes.

The combined diagnostic commit `f4aadf114b0044fa8cbe1b50241dc80ea7d61583` is a clean cherry-pick onto the experimental literal-input a70 branch. Its source equivalence supports a separate unchanged-generation8 live replay, documented in a separate archive. A generation-only fresh server does not reproduce the preceding coverage8 cache history from the prior combined16 run; matching payloads and token IDs alone do not establish an isolated numerical A/B. No live candidate result is claimed by this CPU/source archive.
