# Run a reproducible server experiment matrix

[`run_matrix.py`](run_matrix.py) starts one server at a time, runs the requested strict clients, saves each attempt and stops only its owned processes. Use it on the Linux host that owns the GPU and model files, after setup and the numerical/API qualification for the source/settings you intend to compare. It does not install packages, fetch Git refs, rewrite model packs or decide which setting wins.

The original published controller was a byte-exact copy of the overnight controller, SHA256 `48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586`. This public version adds optional external-template selection and provenance checks described below. Frozen controllers retained with historical measurements still have their original bytes and hashes; this new version must not be substituted when identifying those runs. Its historical docstring still refers to `spark_experiment_controller.py`.

## Prerequisites and ownership

The runner uses Python's standard library, Bash, Git and Linux `/proc` and file locks. The selected runtime must have `venv/`, `exllamav3/` and `tabbyAPI/` directories. Clients run with that runtime's Python. `nvidia-smi` supplies bounded resource observations; a sampling error is retained rather than treated as a trustworthy resource value.

Use a compatible, validated recipe checkout containing the strict clients and the launcher that writes `deployment.json`. Run that recipe's `setup.sh --check` with the intended overrides first. Keep the actual recipe, engine and Tabby commit IDs, dependency versions and model metadata with the results. Use clean, committed source checkouts for a reproducible comparison. The controller records tracked changes and rejects source drift during a matrix; it does not itself decide that a recorded dirty checkout is suitable for publication.

Schedule exclusive access to the GPU. Stop a previously running service through its own supervisor before the experiment. Port8899 must be free; the runner refuses a busy loopback port and never takes over an existing server. It uses a separate session/process group and an unguessable environment token for each job, and verifies membership before signaling. Unknown ownership or failed cleanup halts the matrix before a subsequent model is loaded.

Set `GPU_LOCK_FILE` inside each job's `env` to make its live server cooperate with another supervisor using the same persistent lock file. The path is included in resolved configuration and drift checks; an ambient shell assignment is removed with other tuning overrides. Each server holds the lock through its final process exit, but the runner does not hold it between jobs. Keep the whole experiment window exclusively scheduled. If an outer controller already holds that flock, leave this option unset in the child to avoid competing with your own lock. See [shared GPU ownership](../docs/service.md#shared-gpu-ownership). The selected recipe must expose the `acquire_gpu_lock` launcher capability; a requested lock with an older launcher is rejected before starting the server. Before each client and after measurements, the runner verifies that server FD 8 references the requested device/inode and that Linux reports an exclusive `FLOCK ADVISORY WRITE` on that descriptor. These observations are retained in the result alongside the deployment setting. This checks cooperating process ownership; a process that ignores the lock or replaces it between observations remains outside that protection.

The runner fixes the API at `http://127.0.0.1:8899/v1` with local authentication disabled. Network, component and state paths are controller-owned. This is a local experiment workflow; it does not configure a LAN deployment.

## Define comparable jobs

The job file is a nonempty JSON array. Labels must be unique in that array and contain only letters, digits, dots, hyphens and underscores, beginning with a letter or digit. Use a distinct label or output destination for every independent measurement.

This example compares depth5 with depth3 while retaining the same single-request profile, streamed PLE, shared cache, draft mode and payload. Replace the model paths and add the complete qualified compatibility/tuning settings from your control. The example itself is not a measured recommendation.

```json
[
  {
    "label": "control-depth5",
    "model_path": "/absolute/path/to/model-pack",
    "env": {
      "PROFILE": "single",
      "NGRAM_RAM": false,
      "MAX_SEQ_LEN": 262144,
      "CACHE_SIZE": 262144,
      "MAX_BATCH_SIZE": 1,
      "CHUNK_SIZE": 2048,
      "DRAFT_MODE": "mtp",
      "DRAFT_NUM_TOKENS": 5,
      "DYNAMIC_DRAFT": true,
      "EXL3_MTP_HEAD_N": 65536,
      "EXL3_DRAFT_CONFIDENCE": 0.6
    },
    "bench": [
      {
        "suite": "all",
        "max_tokens": 400,
        "repeat": 3,
        "warmup": 1,
        "run_id": "comparison-v1",
        "cache_mode": "cold",
        "thinking": false
      }
    ]
  },
  {
    "label": "candidate-depth3",
    "model_path": "/absolute/path/to/model-pack",
    "env": {
      "PROFILE": "single",
      "NGRAM_RAM": false,
      "MAX_SEQ_LEN": 262144,
      "CACHE_SIZE": 262144,
      "MAX_BATCH_SIZE": 1,
      "CHUNK_SIZE": 2048,
      "DRAFT_MODE": "mtp",
      "DRAFT_NUM_TOKENS": 3,
      "DYNAMIC_DRAFT": true,
      "EXL3_MTP_HEAD_N": 65536,
      "EXL3_DRAFT_CONFIDENCE": 0.6
    },
    "bench": [
      {
        "suite": "all",
        "max_tokens": 400,
        "repeat": 3,
        "warmup": 1,
        "run_id": "comparison-v1",
        "cache_mode": "cold",
        "thinking": false
      }
    ]
  }
]
```

`PROFILE=single|concurrent` and `NGRAM_RAM=true|false` are required. Use explicit placement for A/B; `auto` would allow available memory to choose different paths. The cache pool is shared between requests, and the per-request cap remains262144. Check the [memory advisory](../docs/memory.md) after unloading the preceding pack before scheduling an optional RAM/concurrency load.

`env` accepts the recipe tuning variables and `EXL3_*` options. It rejects controller-owned runtime, state, model-path and network variables, including `TABBY_REF`. The runner removes ambient tuning overrides before sourcing `env.sh`, so an assignment prefixed to the runner command does not silently replace the recipe default. An exact qualified `EXL3_REF` can be supplied inside each job's `env`. Tabby may retain the recipe's `main` policy: record and verify its actual clean tested commit separately. The runner performs no update operation during measurement.

An optional `PROMPT_TEMPLATE` environment value selects an absolute readable UTF-8 `.jinja` file. The normalized job includes its requested/resolved paths, raw byte count/hash and normalized text hash. The runner rechecks that identity at the existing input gates and requires the deployment snapshot and `/v1/model` actual template text to match before any client runs. A fallback template or file change fails the attempt. See [explicit prompt templates](../docs/prompt-templates.md) for the Cyber-Frost override, API thinking flag and comparisons across prompt profiles.

Keep the same model path/native view, loader file order, payload, `run_id`, sampling, output budget, cache mode, warmup and repeats across the control and candidate. The loader's natural file enumeration matters when an MTP patch duplicates tensor keys; the runner records that order. It hashes model/tokenizer configuration files and records weight paths, sizes and mtimes, but does not hash every large weight shard.

### Measurement fields

| Field | Accepted settings |
| --- | --- |
| `bench` | One object or a list: `suite`, `repeat`, `warmup`, `run_id`, `context_tokens`, `max_tokens`, `cache_mode`, `prompt`, `thinking`, `timeout`. |
| `concurrency` | One object: `streams`, `prompt_tokens`, `new_tokens`, `rounds`, `warmup`, `run_id`, `timeout`. |
| `tools` | One object: `case`, `mode`, `repeat`, `max_tokens`, `timeout`. |
| `tool_cases` | A shorthand list or comma-separated string of case names; use it instead of `tools`. |
| `response_model` | Explicit expected response ID when a known old server advertises an alias but returns a different canonical ID. Omit for strict requested-alias matching. |

For example, a tools-only job can use `"tool_cases": ["auto", "no_args", "strings", "typed", "round_trip"]`. A concurrent job can add `"concurrency": {"streams": [1, 2, 4], "prompt_tokens": 1024, "new_tokens": 256, "rounds": 3, "warmup": 1, "run_id": "concurrent-v1"}`. If a job has tools or concurrency but omits `bench`, no implicit benchmark is added. If it has no measurement fields at all, the runner constructs the default benchmark; write `bench` explicitly for reproducible work.

See the [benchmark client definitions](README.md), [tool-call checks](../docs/tool-calling.md) and [API compatibility diagnostics](../docs/api-validation.md) for the metrics and cases. The matrix runner invokes `bench_v1.py`, `concurrency.py` and `tool_smoke.py`; SDK, resilience, long-context and numerical programs remain separate scheduled checks.

## Execute and retain each attempt

Run with explicit absolute recipe/runtime paths and a new output destination:

```bash
python3 bench/run_matrix.py \
  --recipe /absolute/path/to/qualified-recipe \
  --runtime /absolute/path/to/qualified-runtime \
  --jobs /absolute/path/to/jobs.json \
  --output /absolute/path/to/results/comparison-01
```

The default readiness deadline is600seconds, each client's wall deadline1800seconds, and resource sampling interval10seconds with at most360 samples per job. Adjust `--ready-timeout`, `--client-timeout`, `--sample-interval` and `--max-samples` explicitly when needed; these settings are part of the configuration fingerprint. A client's own request timeout can also be set in its job settings.

Each label has a top-level `result.json` and one or more distinct `attempt-...` directories. Each attempt retains normalized configuration and source identities, a unique server `state/`, rendered configuration, copied `deployment.json`, server/client logs, individual strict-client reports, actual exit codes, wall/load timing and bounded resource samples. Model readiness requires the expected advertised alias, the expected canonical model, an owned listener and matching deployment source/model identity.

The controller checks recipe/runtime identities, package versions, recipe/client hashes, model metadata/loader order and resolved defaults before loading, before measurements and afterward. A source/configuration change invalidates the attempt. It does not accept an unfinished client report or nonzero client exit as a successful sample. A server that has already exited before cleanup also makes the job fail.

Cleanup sends SIGTERM only to the token-verified owned group, waits up to30seconds, then allows a bounded five-second SIGKILL cleanup. Both clients and server have retained ownership tokens. This experiment cleanup deadline differs from the permanent service's90-second stop policy. Sparse memory/power samples are observations and can miss peaks; preserve sampling errors and avoid claiming a guaranteed capacity or maximum power from them.

The controller records TERM/INT signals and delivers interruption at checkpoints
after each child has been registered. A first signal during shutdown, or
repeated signals while cleanup is running, cannot skip the owned groups.
Children inherit no blocked signal mask from this handling. Interruption
remains bounded by the current operation and the ordinary cleanup deadlines;
it does not allow the next model to start before cleanup completes.

### Resume and failure behavior

A finished output is never overwritten silently. `--resume` accepts it only when the normalized job, source/package identities, recipe/client hashes, resolved tuning and controller limits match. It preserves a finished failed attempt too; it does not retry it as if it had never happened. For an independent repeat or a retry of a completed failure, use a new destination or label.

An interrupted attempt without a finish timestamp remains intact. With matching configuration and `--resume`, the runner creates a new attempt directory. Unrecognized existing output or any fingerprint mismatch is rejected. Changing source paths or recipe versions can legitimately change that fingerprint.

Ordinary failed measurements remain in the record and the matrix may proceed to later jobs; process-ownership/cleanup uncertainty stops it. Overall exit0 requires every requested job to finish successfully. Inspect each report's state, exits and `passed` value; an atomic result file or completed orchestration phase alone is not a passing measurement.

## A/B across source revisions and the historical baseline

For source A/B, prepare and validate separate recipe/runtime pairs and use the same job payload/control environment in separate output directories. Run the pairs sequentially. Do not update checkouts while a matrix is active. Preserve actual commits and dependency sets; changing both code and tuning should be identified as a combined treatment rather than attributed to one flag.

The original recipe revision6fbc0a2 used a different launcher contract. It emits no modern deployment snapshot, and its nonempty `DRY_RUN` handling differs. This runner sets `DRY_RUN=0` for the current launcher and therefore cannot simply be pointed at that original recipe. Exact overnight baseline controller sources are archived separately as historical execution evidence, with their original hardcoded paths and outputs. They are not a portable replacement for this runner and should not be imported or casually rerun on retained results.

## CPU verification

```bash
python3 -m unittest discover -s bench -p 'test_run_matrix*.py' -v
```

The 33 matrix regressions cover command routing, strict settings, alias
expectations, model/default/source drift, literal loader order, output/resume
integrity, busy-port refusal, bounded sampling, shared GPU-lock descriptors,
and owned cleanup. The 26 main cases include real CPU file/flock and Bash
checks alongside mocked server/API operations. Seven signal cases use real
local child processes and deliver TERM/INT during launch registration and
finalization, including repeated signals and an unrelated process group.
These CPU tests load no model and make no GPU or live API requests.
`test_prompt_templates.py` separately adds file/config drift, actual loaded
content, and optional real Tabby renderer checks for external templates.

The [October 9 report](../VALIDATION_2026-10-09.md#cooperative-gpu-ownership)
links the actual GPU-lock serving gate and the separately qualified matrix
interruption fix. Completed and interrupted attempts remain distinct evidence.
