# Final Tabby CPU evidence

Exact Tabby source: `3adc9813f30c4b92d110e7e0235129031dae1c25`. The complete CPU suite passed **371 tests and 5,133 subtests** with **two engine-only skips**. The raw log and manifest record the exact command, Python/platform, source hashes, clean tree and exit code.

The actual 248,077-token Qwen tokenizer replay passed **37 grammar cases and three captured-prefix masks**. These checks include closed empty argument objects, native added tokens, literal XML in strings and incomplete call handling. They do not require model weights or CUDA.

The separate cross-project check passed **11 cases** using engine `9c0bbaaa31043f84a62e618d8c3b2e19c45b22c2` and the same Tabby source. It executes native Job/SeqTensor/speculative-acceptance/mask code through the engine's CPU fixture and composes it with Tabby's real parser guard. It checks wrapped and bare calls, literal versus real reasoning-closing tokens, same-window grammar handoff, banned-string rewinds and requeue. Scheduling/model/logit/allocation doubles are explicitly described in the harness and report.

These are separate evidence sets with different scopes; do not add their counts to describe a single suite, and do not interpret CPU control-flow validation as live model accuracy or a throughput benchmark.

## Reproduce

In the exact Tabby checkout with its CPU test dependencies:

```sh
python -m pytest -q
python -m tests.check_qwen_tool_tokenizer /path/to/tokenizer.json
```

With CPU PyTorch/NumPy/pytest, the exact engine checkout and Tabby checkout:

```sh
python check_reasoning_guard_cross_cpu.py --engine /path/to/exllamav3 --tabby /path/to/tabbyAPI --output /new/path/cross-check.json
```

The cross-check refuses an existing report path. Its first run records running state, then publishes the completed report. File hashes are in `manifest.json`; source file hashes are embedded in each detailed manifest/report.
