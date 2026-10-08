"""Diagnostic-only startup hook, ignored by all non-main.py Python commands."""
import os
import sys
from pathlib import Path
if os.environ.get('TABBY_REASONING_OBSERVER_CONFIG') and Path(sys.argv[0]).name=='main.py':
    try:
        from reasoning_observer import arm_from_environment
        arm_from_environment()
    except BaseException as error:
        sys.stderr.write('Reasoning observer startup rejected: '+type(error).__name__+'\n')
        os._exit(78)
