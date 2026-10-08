# Run the recipe as a systemd user service

The user service starts `exllamav3-tabby/serve.sh` under the account that owns the model and runtime. It binds to `127.0.0.1:8899`. A small invocation wrapper gives **every start and automatic restart a separate state directory**, retaining its model view, rendered configuration and deployment snapshot. Install it after the chosen sources, model and settings have passed the checks you intend to rely on.

The publication templates are:

| File | Installed purpose |
| --- | --- |
| [`qwen38-exl3.service.example`](../exllamav3-tabby/systemd/qwen38-exl3.service.example) | Base user unit and restart/stop policy. |
| [`10-invocation-state.conf.example`](../exllamav3-tabby/systemd/10-invocation-state.conf.example) | Drop-in that selects the invocation wrapper. |
| [`qwen38-exl3-start.sh`](../exllamav3-tabby/systemd/qwen38-exl3-start.sh) | Creates a private, unique state directory, records start identity and executes the ordinary launcher. |
| [`service.env.example`](../exllamav3-tabby/systemd/service.env.example) | Literal component paths and the measured deployment settings. |

These files do not run setup, fetch/update Git refs, enable lingering or install a system-wide service. The examples describe how to install and verify the arrangement; they are not evidence that a particular deployment or rollback rehearsal has completed.

## Keep both runtime paths intact

The validation-host layout below is an example. Replace the account and model paths for your machine while preserving the same separation:

| Purpose | Example path |
| --- | --- |
| Preserved original runtime | `/home/cruzspark/qwen38-exl3` |
| New runtime | `/home/cruzspark/qwen38-exl3-20261008` |
| Clean recipe checkout for the new unit | `/home/cruzspark/qwen-spark-recipe` |
| Installed service environment | `/home/cruzspark/.config/qwen38-exl3/service.env` |
| Base for all service starts | `/home/cruzspark/.local/state/qwen38-exl3/runs` |

Keep an existing real `~/qwen38-exl3` directory at its original canonical location. The permanent unit can name the dated runtime directly. An optional separate `~/qwen38-exl3-current` symlink may select an installation without moving it; the service does not require that alias.

Do not rename an old virtual environment and replace its original path with the new runtime. Console-script shebangs and editable-install metadata may contain absolute paths, so the moved backup can resolve packages through the new target. The recipe canonicalizes harmless aliases in its build fingerprint, but that does not make a relocated venv portable. See [Python's venv portability guidance](https://docs.python.org/3/library/venv.html#how-venvs-work).

## Configure and install

Run as the account that will serve the model, from the clean recipe checkout you intend to deploy. For a first installation, these commands refuse existing destination files:

```bash
set -euo pipefail

test ! -e "$HOME/.config/qwen38-exl3/service.env"
test ! -e "$HOME/.config/systemd/user/qwen38-exl3.service"
test ! -e "$HOME/.config/systemd/user/qwen38-exl3.service.d/10-invocation-state.conf"
test ! -e "$HOME/.local/libexec/qwen38-exl3-start.sh"

install -d -m 0700 "$HOME/.config/qwen38-exl3"
install -d -m 0755 "$HOME/.config/systemd/user"
install -d -m 0755 "$HOME/.config/systemd/user/qwen38-exl3.service.d"
install -d -m 0700 "$HOME/.local/libexec"
install -d -m 0700 "$HOME/.local/state/qwen38-exl3/runs"

install -m 0600 exllamav3-tabby/systemd/service.env.example \
  "$HOME/.config/qwen38-exl3/service.env"
install -m 0644 exllamav3-tabby/systemd/qwen38-exl3.service.example \
  "$HOME/.config/systemd/user/qwen38-exl3.service"
install -m 0644 exllamav3-tabby/systemd/10-invocation-state.conf.example \
  "$HOME/.config/systemd/user/qwen38-exl3.service.d/10-invocation-state.conf"
install -m 0755 exllamav3-tabby/systemd/qwen38-exl3-start.sh \
  "$HOME/.local/libexec/qwen38-exl3-start.sh"
```

If these files already exist, preserve the working copies and review the changes before updating them. Installing the files does not start the server.

Edit `~/.config/qwen38-exl3/service.env` before starting the unit. The example account paths are not portable placeholders that systemd will resolve automatically. Set all component paths consistently:

| Variable | Meaning |
| --- | --- |
| `QWEN_RECIPE_DIR` | Clean recipe containing the intended `serve.sh`. |
| `RECIPE_HOME` | Validated runtime containing `venv/`, `exllamav3/` and `tabbyAPI/`. |
| `VENV`, `EXL3_SRC`, `TABBY_DIR` | Explicit component paths within that same runtime. |
| `STATE_DIR` | Absolute, non-symlink **base directory** for invocation states. Do not include an invocation ID here. |
| `MODEL_DIR` | Validated pack or its existing native view. |

Copy the complete measured profile, explicit `NGRAM_RAM=true|false`, cache/batch/chunk settings, draft mode/depth, `DYNAMIC_DRAFT`, recurrent allowance and selected `EXL3_*` controls. Put intentional affinity, thread and toolchain overrides here too. The example's single-request/streamed-PLE settings illustrate configuration syntax; they do not certify a different pack's memory fit or performance. Concurrent settings need the corresponding validated shared pool and batch size. Keep `MAX_SEQ_LEN` at or below 262144.

The recipe tracks the Tabby fork's `main` when setup is explicitly run. Keep that tracking policy; record the **actual tested Tabby commit** from setup/deployment snapshots. A branch label is not an immutable tested-source identity. The service itself never runs setup or updates either source checkout. The selected recipe's exact engine pin and normal runtime/fingerprint checks must agree with the installed engine.

The environment file uses systemd's assignment syntax, not shell execution: one `NAME=value` per line, with quotes for spaces. `$HOME`, `~` and references to another variable are not expanded. The user-manager environment is inherited, so specify the measured tuning instead of relying on an interactive shell. See [systemd's execution-environment documentation](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml) and [service command expansion](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml).

Run `setup.sh --check` with the same runtime paths and intentional overrides. For the example layout:

```bash
RECIPE_HOME=/home/cruzspark/qwen38-exl3-20261008 \
  bash /home/cruzspark/qwen-spark-recipe/exllamav3-tabby/setup.sh --check
systemd-analyze --user verify "$HOME/.config/systemd/user/qwen38-exl3.service"
```

Before starting, stop the previous model instance through its own supervisor or the exact retained owned PID. A foreground experiment and this unit must not compete for the GPU and port 8899.

```bash
systemctl --user daemon-reload
systemctl --user enable --now qwen38-exl3.service
systemctl --user status qwen38-exl3.service
journalctl --user -u qwen38-exl3.service -n 100 --no-pager
```

`enable` attaches the unit to the user's default target. Check the account's existing logout/boot policy:

```bash
loginctl show-user "$(id -un)" -p Linger
```

`Linger=yes` needs no change. If it is `no` and unattended boot is required, the host operator can enable lingering for that account. These templates make no host-policy change and require no sudo for user-owned installation. See [loginctl's lingering documentation](https://github.com/systemd/systemd/blob/main/man/loginctl.xml).

## Per-invocation state and readiness

The installed drop-in replaces the base unit's `ExecStart` with the invocation wrapper. The wrapper requires systemd's 32-character `INVOCATION_ID`, appends it to the configured state base, creates that directory exclusively with mode 0700, writes `start.json`, then `exec`s the normal recipe launcher. That preserves the service MainPID across the wrapper transition. It refuses a repeated invocation directory rather than overwriting another start's evidence.

For example, a start with ID `0123456789abcdef0123456789abcdef` uses:

```text
/home/cruzspark/.local/state/qwen38-exl3/runs/0123456789abcdef0123456789abcdef/
```

The base in `service.env` stays unchanged. Each automatic restart and manual stop/start gets a new invocation directory. `start.json` records the start identity and selected paths; the launcher's `deployment.json` records source commits, packages, resolved model files and rendered configuration. Previous directories remain available for comparison.

With `Type=simple`, systemd can report an active process before the model is ready. Wait for the expected alias in the API:

```bash
curl --fail --silent --show-error http://127.0.0.1:8899/v1/models
```

Find the current invocation's evidence, replacing the base if you configured a different one:

```bash
QWEN_STATE_BASE=/home/cruzspark/.local/state/qwen38-exl3/runs
QWEN_INVOCATION=$(systemctl --user show qwen38-exl3.service --property=InvocationID --value)
test -n "$QWEN_INVOCATION"
QWEN_STATE="$QWEN_STATE_BASE/$QWEN_INVOCATION"
test -f "$QWEN_STATE/deployment.json"
systemctl --user show qwen38-exl3.service --property=MainPID --property=InvocationID --property=NRestarts
```

Then run the tool checks and a short strict-client request. Use unused output names if repeating a check within one invocation:

```bash
python3 bench/tool_smoke.py --model Qwen3.8-Flash-Next-EXL3 \
  --case auto,no_args,strings,typed,required,named,none --mode both \
  --output "$QWEN_STATE/service-tools.json"
python3 bench/bench_v1.py --model Qwen3.8-Flash-Next-EXL3 \
  --suite code --repeat 1 --warmup 0 --max-tokens 128 \
  --metadata "$QWEN_STATE/deployment.json" \
  --output "$QWEN_STATE/service-bench.json"
```

Inspect actual source/model identity, usage, finish state and strict-client exits. Keep reports beside the matching deployment snapshot. A successful service start is not itself a tool-calling, numerical-quality or throughput measurement. The broader [API/SDK checks](api-validation.md) and [tool-calling checks](tool-calling.md) remain separately scheduled validation.

## Restart and stop policy

The base unit retains `Restart=on-failure`, a ten-second restart delay, `TimeoutStopSec=90s`, `KillMode=control-group` and `UMask=0077`. It limits starts to five within five minutes. A manual `systemctl --user stop` remains stopped; after correcting repeated startup failures, reset the failed state before starting again.

```bash
systemctl --user stop qwen38-exl3.service
systemctl --user start qwen38-exl3.service
systemctl --user restart qwen38-exl3.service
systemctl --user reset-failed qwen38-exl3.service
journalctl --user -u qwen38-exl3.service -f
```

Editing `service.env` takes effect at the next start or restart. Editing the unit or drop-in also requires `systemctl --user daemon-reload`. Stop the service before running a standalone experiment matrix, then return to it after the experiment has released its owned server. The existing policies are documented upstream under [service restart behavior](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml), [control-group cleanup](https://github.com/systemd/systemd/blob/main/man/systemd.kill.xml) and [execution settings](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml).

## Promotion, controlled failure and rollback evidence

Preserve the current environment, source/dependency identities and invocation snapshot before an update. Build and validate a candidate in a separate canonical directory. Point the service's recipe/runtime/component paths at the matching validated pair, then start it and verify the fresh invocation. Keep the old runtime and its matching recipe at their original paths.

For a controlled failure rehearsal, record the identified active unit's MainPID, InvocationID and NRestarts, confirm its process/config/source ownership, then issue one failure to that unit's main process. A successful automatic-restart gate requires a new MainPID and invocation, exactly one added NRestarts, the unchanged measured environment and successful API checks. A new start record alone does not prove the model loaded or the API recovered.

An exact-source rollback rehearsal stops the new service, requires a released port/GPU, launches the preserved old runtime with its matching original recipe and environment in an owned process group, verifies basic nonstreaming/streaming API behavior, stops only that owned old group, then returns to the new unit and checks another fresh invocation. If ownership or cleanup is uncertain, resolve it before loading another model. Do not use a broad process-name kill.

The historical engine94/Tabby816 baseline has a different launcher contract, including the old nonempty `DRY_RUN` behavior and canonical response-model basename. Recreating it requires its original recipe and recorded environment; leaving newer compatibility flags or editable package mappings in place does not recreate the old runtime. An exact-SHA rehearsal record should retain those identities, actual exits, cleanup and the final healthy invocation. It is evidence after the run succeeds, not a deployment claim made by these templates.

A rollback to an installation kept in place can restore its previous measured service environment and matching recipe/runtime paths. If a venv was already moved, restore its original absolute path or rebuild it at the intended rollback path before relying on it. Recheck source imports, model alias, tool calls and deployment identity after every switch.
