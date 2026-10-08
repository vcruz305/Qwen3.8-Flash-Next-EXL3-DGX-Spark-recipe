#!/usr/bin/env bash
# Installed under ~/.local/libexec; invoked only by the qwen38-exl3 user unit.
set -euo pipefail
umask 0077
: "${QWEN_RECIPE_DIR:?Set the selected recipe path in service.env}"
: "${STATE_DIR:?Set an absolute base state path in service.env}"
: "${INVOCATION_ID:?This launcher requires a systemd service invocation}"
[[ "$INVOCATION_ID" =~ ^[0-9a-f]{32}$ ]] || { echo "Invalid systemd invocation ID" >&2; exit 1; }
[[ "$STATE_DIR" == /* && "$STATE_DIR" != / && ! -L "$STATE_DIR" ]] || { echo "STATE_DIR must be an absolute owned directory" >&2; exit 1; }
mkdir -p -- "$STATE_DIR"
export STATE_DIR="${STATE_DIR%/}/$INVOCATION_ID"
mkdir -m 0700 -- "$STATE_DIR"  # Refuse to overwrite a previous invocation.
python3 - "$STATE_DIR/start.json" <<'PY'
from datetime import datetime, timezone
import json, os, pathlib
path=pathlib.Path(__import__('sys').argv[1])
value={"started_at_utc":datetime.now(timezone.utc).isoformat(),
       "invocation_id":os.environ["INVOCATION_ID"],"wrapper_pid":os.getppid(),
       "paths":{k:os.environ.get(k) for k in
                ("QWEN_RECIPE_DIR","RECIPE_HOME","VENV","EXL3_SRC","TABBY_DIR","STATE_DIR","MODEL_DIR")}}
with path.open("x") as file:
    json.dump(value,file,indent=2);file.write("\n")
PY
exec /usr/bin/bash "$QWEN_RECIPE_DIR/exllamav3-tabby/serve.sh"
