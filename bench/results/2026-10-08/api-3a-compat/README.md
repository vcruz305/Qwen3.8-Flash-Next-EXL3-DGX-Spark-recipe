# Intermediate b532 / 3a live run

This is an intermediate flat-3.05 run, not the final deployment record.
The controller completed and cleanly stopped its owned server. Its overall
`passed` value is **false**: nullable-value checks still returned an empty
string where the prompt requested null. No failed response was retried or
changed to a pass.

| Suite | Completed checks |
|---|---:|
| Broad tool fixtures | 28/28 pass |
| OpenAI SDK | 5/5 pass |
| Resilience | 17/19 pass; two nullable semantic failures |
| Automatic-choice compatibility | 9/9 pass |

The strict single-stream benchmark completed one warmup and three measured
requests per suite, with 400 actual output tokens and zero reused prompt
tokens in each measured request. Median server decode rates are 81.87 code,
74.28 DevOps and 55.62 prose tokens/s. These rates do not show a material
single-stream speed change versus the original 3.05 run.

The exact source, configuration, package and model identity is in
`deployment.json` and `result.json`. Tool calls are synthetic fixtures and
were not executed. Original artifact bytes and SHA256 checksums are retained.
