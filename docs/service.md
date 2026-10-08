# Run the validated recipe as a systemd user service

The service starts the existing `exllamav3-tabby/serve.sh` launcher under the
account that owns the model and runtime. It binds to `127.0.0.1:8899` and keeps
its model view, rendered config and deployment snapshot in an explicit
`STATE_DIR`. Install it after the chosen runtime, model and settings have passed
the API, tool-calling and numerical checks you intend to rely on.

The files are
[`qwen38-exl3.service.example`](../exllamav3-tabby/systemd/qwen38-exl3.service.example)
and [`service.env.example`](../exllamav3-tabby/systemd/service.env.example).
The template does not run setup, update Git refs, enable lingering or install a
system-wide service.

## Configure and install

Run these commands as the account that will serve the model, from the recipe
checkout you intend to deploy. For a first installation:

```bash
install -d -m 0700 "$HOME/.config/qwen38-exl3"
install -d -m 0755 "$HOME/.config/systemd/user"
install -m 0600 exllamav3-tabby/systemd/service.env.example \
  "$HOME/.config/qwen38-exl3/service.env"
install -m 0644 exllamav3-tabby/systemd/qwen38-exl3.service.example \
  "$HOME/.config/systemd/user/qwen38-exl3.service"
```

If those files already exist, preserve the working copies and merge the changes
you need instead of repeating the initial copy commands.

Edit `~/.config/qwen38-exl3/service.env` before starting the unit. Replace the
placeholder paths with literal absolute paths:

| Variable | What it selects |
| --- | --- |
| `QWEN_RECIPE_DIR` | The recipe checkout containing `exllamav3-tabby/serve.sh`. |
| `RECIPE_HOME` | The validated runtime directory containing `venv/`, `exllamav3/` and `tabbyAPI/`. |
| `VENV`, `EXL3_SRC`, `TABBY_DIR` | The explicit component paths within that same runtime. Keep these three paths consistent when switching runtimes. |
| `STATE_DIR` | A writable directory dedicated to this service instance. |
| `MODEL_DIR` | The validated EXL3 pack or its existing native view. |

The component paths are explicit so inherited user-manager variables cannot
redirect one component to a different installation.

Keep the profile, explicit `NGRAM_RAM=true|false`, cache, batch, chunk and
`EXL3_*` values from the measurement you want to deploy. The example's
`PROFILE=single` and `NGRAM_RAM=false` describe its configuration; they are
not a fit or performance guarantee for a different pack. A concurrent deployment
needs the matching validated batch and shared-cache settings. The per-request
ceiling remains 262,144 tokens.

The environment file uses systemd's assignment syntax. Use one `NAME=value`
per line, with quotes around a value containing spaces. Write complete paths:
`$HOME`, `~` and references to another variable are not expanded in this
file. The unit uses systemd's explicit `${QWEN_RECIPE_DIR}` argument expansion
to locate the launcher. The process inherits the user manager's environment, so
put the tuning and any required CUDA/compiler overrides in this file rather than
depending on an interactive shell's startup files.
See [the upstream execution-environment documentation](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml)
and [command argument expansion](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml).

Check the selected runtime with the same environment overrides used by the
service. For example, replacing both paths and adding any intentional toolchain
overrides from the environment file:

```bash
RECIPE_HOME=/absolute/path/to/validated-qwen38-runtime \
  bash /absolute/path/to/recipe/exllamav3-tabby/setup.sh --check
systemd-analyze --user verify "$HOME/.config/systemd/user/qwen38-exl3.service"
```

Before the initial start, stop the previous model instance using its own
supervisor or the PID you retained when launching it. A foreground launcher and
this unit must not compete for the GPU and port 8899.

```bash
systemctl --user daemon-reload
systemctl --user enable --now qwen38-exl3.service
systemctl --user status qwen38-exl3.service
journalctl --user -u qwen38-exl3.service -n 100 --no-pager
```

`enable` attaches the unit to the user's default target. To run it after logout
and at boot without an interactive login, check the account's existing setting:

```bash
loginctl show-user "$(id -un)" -p Linger
```

`Linger=yes` needs no change. If it is `no` and unattended boot is required,
the host operator can enable lingering for this account. The template does not
change that host policy. Upstream documents the behavior under
[`loginctl enable-linger`](https://github.com/systemd/systemd/blob/main/man/loginctl.xml).

## Verify the model after every start or restart

With `Type=simple`, systemd can report an active process before the model is
ready. Wait for the API and check that it advertises the configured
`SERVED_NAME`:

```bash
curl --fail --silent --show-error http://127.0.0.1:8899/v1/models
```

Then run the [tool smoke checks](tool-calling.md) and a short benchmark using the
same model ID. Save their results together with
`$STATE_DIR/deployment.json`; the snapshot contains the actual engine/server
commits, packages, model metadata and rendered configuration. The benchmark
clients accept that snapshot with `--metadata`. The tool smoke report can be
stored alongside it. A service restart is not, by itself, an API correctness or
numerical-quality check.

For example, from the chosen recipe checkout, replacing the state path with the
one in `service.env`:

```bash
python3 bench/tool_smoke.py --model Qwen3.8-Flash-Next-EXL3 \
  --case auto,no_args,strings,typed,required,named,none --mode both \
  --output /absolute/path/to/qwen38-service-state/service-tools.json
python3 bench/bench_v1.py --model Qwen3.8-Flash-Next-EXL3 \
  --suite code --repeat 1 --warmup 0 --max-tokens 128 \
  --metadata /absolute/path/to/qwen38-service-state/deployment.json \
  --output /absolute/path/to/qwen38-service-state/service-bench.json
```

The service restarts after a failure with a ten-second delay. A manual
`systemctl --user stop` remains stopped. The template limits starts to five
within five minutes; after correcting a repeated startup failure, use
`systemctl --user reset-failed qwen38-exl3.service` and then start it again.
Stopping sends termination to the service's control group and allows 90 seconds
before forced cleanup. `UMask=0077` restricts access to newly created runtime
files. These behaviors follow
[systemd's service settings](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml),
[process-group cleanup](https://github.com/systemd/systemd/blob/main/man/systemd.kill.xml)
and [execution settings](https://github.com/systemd/systemd/blob/main/man/systemd.exec.xml).

Useful controls:

```bash
systemctl --user stop qwen38-exl3.service
systemctl --user start qwen38-exl3.service
systemctl --user restart qwen38-exl3.service
journalctl --user -u qwen38-exl3.service -f
```

Editing `service.env` takes effect at the next start or restart. Editing the unit
file also requires `systemctl --user daemon-reload`. Stop this service before
running a standalone experiment controller; return to the unit after the
experiment has released its own model process.

## Updates, alias promotion and rollback

Preserve the current environment file and deployment snapshot before an update.
Keep the previous recipe checkout and runtime intact. Build and validate the
candidate in a separate directory, then point `QWEN_RECIPE_DIR`, `RECIPE_HOME` and the three component paths at the
matching validated pair and restart the service. Recheck the
advertised model, tool calls and deployment snapshot after switching.

A stable `~/qwen38-exl3` symlink can select among runtimes that remain at their
original absolute installation paths. Update the alias while the service is
stopped and keep the previous target recorded. The recipe's build fingerprint
canonicalizes the Python executable, so accessing the same installation through
a harmless symlink does not by itself create an ABI mismatch.

**Renaming an existing virtual environment is a different operation.** Console
entry-point shebangs and editable-install metadata can contain absolute paths to
the original venv and source directories. If an old real `~/qwen38-exl3`
directory is moved aside and that path becomes a symlink to a new runtime, a
Python executable inside the moved backup can still resolve editable packages
through the new target. The import-path and fingerprint checks are designed to
catch this mismatch; they do not make a relocated venv portable.
Python documents this limitation in
[the venv portability guidance](https://docs.python.org/3/library/venv.html#how-venvs-work).

Prefer keeping both installations at their original canonical paths. If the old
installation was already moved, rollback requires restoring its original
absolute path or rebuilding it at the intended rollback path with the matching
source/dependency versions. Preserve the original directory until that rollback
has been verified.

To roll back an installation that was kept in place, stop the service, restore
its previous environment file or recorded alias target, select the matching old
recipe, and run that recipe's runtime check. Start the unit and repeat the API
checks. Restoring only an old engine directory while leaving a newer recipe pin,
editable package mapping or dependency set does not recreate the prior runtime.
