# Guarded natural reasoning handoff: exact CPU evidence

Final Tabby: `5a4f3efa1c1f60b6966ba0d0d5610f6b953541fb`.
Matching engine: `f0beceb350d52bfdd62cbce6e6b462b35dd04759`.

The complete Tabby suite passed **382 tests / 5,159 subtests**, with **two engine-only module skips**. `manifest.json` records the clean checkout, exact command, Python/platform, raw log and source hashes. The skipped modules contain 16 mocked grammar/recovery checks; they can collect when the installed engine is available.

`native-backend-filter.json` records **8 cases** composing actual Tabby backend methods with actual native Job, base Filter, and MTP acceptance code. The fixture replaces model logits, allocation/scheduling, recovery/metrics plumbing, and the content language. It covers wrapped/bare tool syntax, scalar/MTP delivery, and dense/packed masks. A literal native closing token inside a tool parameter at position4 leaves content filters detached and unjournaled. The real closing marker at8 installs the mask before the next accepted sample. No forced output/deadline is created, and the explicit20-token generation limit stays unchanged.

`native-guard-cross.json` records the separate **11-case** finite-budget/native-guard regression replay at the final pair: protected tags, real endings, same-window handoff, banned rewinds, and requeue. Counts are separate; they are not added to the main suite count or described as live model accuracy.

`predecessor-no-budget-trigger.json` is the preserved clean3adc/9c reproduction. The real collector forwarded no plan when the budget was omitted, the actual backend selected the raw trigger, and native Filter.feed activated it inside an open parameter. The next argument proposal was masked to content-only output. No production edits preceded that reproduction.

## Commands

From the final Tabby checkout:

```sh
python -m pytest -q -ra
```

From the directory containing these harnesses, with the exact source checkouts:

```sh
python check_natural_backend_filter_cpu.py --engine /path/to/exllamav3 --tabby /path/to/tabbyAPI --output /new/path/native-backend-filter.json
python check_reasoning_guard_cross_cpu.py --engine /path/to/exllamav3 --tabby /path/to/tabbyAPI --output /new/path/native-guard-cross.json
```

The backend composition uses Tabby's CPU dependencies plus CPU PyTorch/NumPy/pytest; its recorded WSL run appended the already-installed CPU Torch path to the CPU venv. The predecessor reproduction retains its original fixed sibling checkout names as recorded in its source and report. No harness sends HTTP or loads model weights.
