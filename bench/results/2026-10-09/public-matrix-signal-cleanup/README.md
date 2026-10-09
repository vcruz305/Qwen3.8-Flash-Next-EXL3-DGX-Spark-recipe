# Public matrix cleanup across interruption signals

The baseline matrix controller at recipe `254b2b03027f25094845dc31f5f87739f3584d2e` could raise from its SIGTERM handler during the final `server.poll()` and leave `stop_owned` uncalled. The retained baseline reproduction executes the exact handler/finally source with a fake poll that sends a real SIGTERM; it creates no server or child process. Its failed cleanup observation is preserved in `evidence/matrix-signal-baseline-proof/result.json`.

The corrected controller records the first signal and raises at controlled checkpoints after children are registered. Cleanup runs without asynchronous exceptions from that handler; owned-group identity checks and prior-handler restoration remain in place. New regression tests launch harmless local Python child groups and exercise server/client registration, final polling, repeated TERM/INT during cleanup, and refusal to signal an unrelated group. No GPU or HTTP endpoint is used by those tests.

Source commit `3c693e316547787bca0d45e1fb9aee530b5e3f6d` and its integrated commit `22d673463dba8e1f3a862056348769ab85a3c032` have the same complete Git tree, `e7da8f487248462e1f55fdfbe97e38333e1c22dd`. The collection receipt verifies copied source bytes against the integrated commit. The patch changes only `bench/run_matrix.py` and adds `bench/test_run_matrix_signals.py`; included unchanged matrix tests and env.sh retain their relative paths for CPU reproduction.

Independent targeted validation reported **33 tests, OK**. Root's full recipe runner reported **Ran 155 tests, OK (skipped=9)**; these are the runner's raw counts, without deriving a separate pass total from class-level skip events. Shell syntax and diff checks both exited zero. This archive contains completed CPU evidence, not a new live qualification of the public patch. The separate literal-feature live harness stays bound to recipe254 and its own reviewed outer cleanup protection.

To rerun the bounded offline matrix tests on Linux with procfs and Bash:

```bash
cd source
python3 -m unittest discover -s bench -p 'test_run_matrix*.py' -q
```

The tests use harmless local processes and temporary files. The baseline `reproduce.py` executes at import time and installs signal handlers: run it only as a standalone subprocess if reproducing that original failure; do not import it into another controller. No archived runner or helper needs to be imported merely to inspect the evidence. `SHA256SUMS` covers every regular archive file except itself. Full-recipe tests should be rerun from the corresponding Git checkout, since this source subset deliberately contains only the targeted-test dependencies.
