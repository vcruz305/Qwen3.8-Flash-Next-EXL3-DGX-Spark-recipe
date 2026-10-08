# Strings-only raw observation controller

`strings_only_controller.py` runs one explicitly selected model through the
frozen owned API lifecycle. It invokes the original recipe
`bench/tool_smoke.py --case strings --mode both --repeat 1 --max-tokens 1024`:
exactly one original request in each response mode. It does not modify the
fixture, prompt, parser, model pack, source checkouts or setup.

The final configure adapter is selected by an explicit reviewed SHA256. It
supplies the exact source/configuration/deployment checks, including any optional
`PROMPT_TEMPLATE` override. The underlying lifecycle stays byte-identical to
`api_f4_gemm_controller.py` SHA256
`78e990269f0cccb51172e524f87193a5607a0027165d20add1b6b6716b42b589`,
which loads the original helper SHA256
`48281d5b51b1b64509385548c6f41f34909c328070274fcb4a06945ddcc44586`.

## Inputs and execution

Prepare a JSON object with only `label`, `engine`, `tabby`, `model_path`,
optional `recipe`, and `env`. Use a simple lowercase label, full 40-character
source commits and the complete qualified control environment. The source
commits are job inputs; this controller does not freeze a model runtime to the
observer's original 9c/3adc source pair.

The `env` contract is the final API adapter's explicit profile, PLE placement,
shared cache, per-request/context/batch/chunk sizes, drafting settings and
qualified EXL3 controls. No extra clients, repetitions or benchmarks are allowed.
The final setup status must satisfy that adapter's complete setup gate.

Run only after the root-owned GPU/port window and source/setup qualification:

```bash
python3 /absolute/path/to/strings_only_controller.py \
  --job /absolute/path/to/strings-job.json \
  --setup /absolute/path/to/completed-setup/status.json \
  --output /absolute/path/to/new-results/strings-pack-profile \
  --observer-dir /absolute/path/to/reviewed-strings-observer \
  --final-wrapper-sha256 ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d
```

The command binds the reviewed final configure adapter at SHA256
`ae9e0965a1abcb1342049039fe59fe21c4e0dc162b03b1f4ddc6923484ae344d`.
It includes the later optional client gates; this strings-only controller still
selects exactly the original two strings requests. A changed adapter requires
an independently reviewed digest.

The controller and its two parent/helper Python files must be siblings. The
selected final configure adapter also lives there as
`final_api_controller.py`. Its runtime/SDK directory conventions are inherited
from the frozen lifecycle. The output parent must already exist. The default
readiness deadline is 600 seconds and the single two-request client deadline
is 450 seconds; bounded overrides are available.

Existing output paths are refused. A sibling
`OUTPUT.observer-inputs/` is exclusively claimed for the observer source,
configuration and derivation manifest. It is retained on failure. Use a new
destination for a repeat; no resume, overwrite or automatic retry is provided.

## Observer source binding

The reviewed observer base is SHA256 `77d606e2f2e894f7503a6f41c99a5a94696339b5c15c7b8415b68f8416b945d3`; its site hook is SHA256 `3d7a620c8cfc39fec68a84f221170b9105fa19260dcdee430d317b869bd1474f`. This base fixes the earlier pre-render-only matcher and has an actual formatting-to-collector CPU check. The frozen observer contains exact expected source constants. For each run the
controller creates a derived copy, replacing only the single `ENGINE_HEAD` and
`TABBY_HEAD` assignments with the job's full commits. It requires the reviewed
base observer/site hashes and verifies that reversing these two replacements
recovers every original byte. The original observer is retained alongside the
derived copy; both hashes, exact replacements and the configuration are recorded.

The unchanged observer still verifies the actual clean Git checkouts and native
import paths. Source binding does not prove compatibility with an arbitrary new
runtime: the chosen pair must retain the reviewed collector and finish-hook
interfaces and pass its own source/API qualification.

Only the server subprocess receives the observer's `PYTHONPATH` and
`TABBY_STRINGS_OBSERVER_CONFIG`. Client and helper Python commands do not receive
the hook. An actual collector request has framework-augmented template variables;
the reviewed matcher checks that state while keeping the message, tool schema,
choice, sampling, thinking flag and response mode within the exact fixture.

## Results and interpretation

The original lifecycle `result.json` and strict `tools.json` remain intact,
including any semantic failure and actual exit codes. After owned cleanup, the
controller adds `strings-capture.json`:

- `capture_valid`: source/configuration identity, completed client/lifecycle,
  owned cleanup, exact fixture payloads and two complete raw traces all verified.
- `semantic_passed`: the original strict strings case passed in both modes.
- `passed`: both conditions are true.
- Trace, rendered-prompt, native-output, backend-output, observer, client and
  lifecycle hashes, plus source derivation and per-mode semantic outcomes.

Exit 0 means valid capture and both semantic checks pass. Exit 1 means the capture
is valid and at least one original semantic check fails. Exit 2 means capture
integrity/completeness failed. A valid diagnostic capture does not turn an
incorrect model response into a passing API fixture.

Exactly two distinct native request IDs, one stream and one nonstream, are
required. Both existing full-completion strings must be present, their equality
or difference is recorded, and collector errors, missing traces, changed files,
wrong payloads, inconsistent outcomes or uncertain cleanup invalidate capture.
The original client summary omits API response IDs, so correlation uses the
unique mode and exact request payload, with distinct IDs verified inside the
observer traces.

No output string is normalized or repaired. These runs include an observer and
must not be used as throughput measurements. Inspect raw native text before
attributing incorrect tool arguments to the API parser.

## CPU checks

```bash
python3 -m unittest -q test_strings_only_controller_cpu
```

The checks use the actual frozen observer and retained original 4.05 request
fixture with temporary synthetic traces. They cover valid semantic failures,
passing results, exact source derivation, missing/duplicate/bad traces, scope and
hash drift, cleanup uncertainty, server-only environment, fixed command
construction and deferred launch-time signals. No API, model, GPU or server is
started.
