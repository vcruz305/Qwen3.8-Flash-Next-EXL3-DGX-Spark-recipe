"""Opt-in diagnostic startup hook, inert for all non-server Python commands."""
import os
import sys
from pathlib import Path
if os.environ.get("TABBY_STRINGS_OBSERVER_CONFIG") and Path(sys.argv[0]).name == "main.py":
    try:
        from strings_observer import arm_from_environment
        arm_from_environment()
    except BaseException as error:
        sys.stderr.write("Strings observer startup rejected: " + type(error).__name__ + "\n")
        os._exit(78)
