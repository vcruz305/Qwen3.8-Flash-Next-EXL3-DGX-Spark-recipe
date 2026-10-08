# Original Spark API measurements

These are immutable original-runtime observations (engine `94ba01d` and
Tabby `816c321`). Short suites have one warmup and three measured runs per
prompt class; prefill has one warmup and two measured runs. Throughput comes
from the server's actual usage/performance fields, and the client verifies
model identity and response completion.

Flat 3.05 uses RAM PLE; the other three packs use disk PLE. Each pack retains
its own installed chat template. Cyber adds 109 template tokens: its medium
prompt has 12,266 actual tokens versus 12,157 for the other packs. Same-pack
comparisons preserve the original construction.

The initial 3.05 alias-check failure is retained separately and is not a
performance sample. A successful later request used an explicitly verified
canonical response-model alias. Tool report sizes differ because only 3.05
ran the full 24-case original broad suite; check each file's recorded cases
before comparing counts. Later candidate suites add four adversarial checks.
