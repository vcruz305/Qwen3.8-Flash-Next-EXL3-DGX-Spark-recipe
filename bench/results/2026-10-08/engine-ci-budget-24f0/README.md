# Producer budget coverage in ordinary CI

Final source: `24f0dece34f09c8d1e2359d6b3b3f7befef7331b`.

The existing Linux wheel job now includes `tests/test_token_budget_cpu.py` in its
CPU source-test command, alongside the UMA, DFlash reservation and sampler
modules. The command uses `--noconftest` to exclude the shared CUDA fixture.
There are no new jobs, dependencies, permissions or tests in this commit.

The exact combined selection passed **320 tests and 186 subtests in 14.28 seconds**
on WSL CPU. The command and timestamps are in [evidence.json](evidence.json), and
the unchanged console output is in [cpu-combined.log](cpu-combined.log). This count
includes the 77 producer-budget tests; it is not 320 plus another 77.

The test run preceded the workflow-only commit. Evidence proves that its exact
workflow bytes were committed and that the test tree did not change. The entire
production `exllamav3` tree remains `b00adae150cef6db7cafaec143c8945e22231042`,
byte-identical to the previously qualified f0 and455 runtime source. The retained
[diff](workflow.diff) contains only the CI command, title and scope comment.

This is local CPU evidence. The hosted exact-head CI result, final Spark setup and
live API/service validation must be recorded separately. Earlier numerical and
native-GPU evidence retains its original16ca source identity.
