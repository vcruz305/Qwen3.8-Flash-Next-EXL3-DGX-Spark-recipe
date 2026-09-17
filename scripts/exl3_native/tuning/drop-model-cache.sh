#!/usr/bin/env bash
# Drop the page cache for the model dir (unified memory: cached file pages count against GPU-allocatable RAM
# and exllamav3's autosplit refuses to load if mem_get_info() looks tight). No root needed — fadvise per file.
MODEL="${1:-$HOME/models/Qwen3.8-Flash-Next-EXL3}"
python3 - "$MODEL" <<'EOF'
import os, sys, glob
n = 0
for p in glob.glob(os.path.join(sys.argv[1], "*")):
    if os.path.isfile(p):
        fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
print(f"fadvise DONTNEED on {n} files")
EOF
free -g | sed -n 2p
