#!/usr/bin/env bash
# Drop the page cache for one model directory. On GB10 unified memory, cached file pages count
# against what cudaMemGetInfo reports as free, and exllamav3's autosplit loader refuses to load
# when that looks tight. No root needed: posix_fadvise(DONTNEED) per file (symlinks followed).
#   drop-model-cache.sh [model dir]   (default: $MODEL_DIR, else ~/models/Qwen3.8-Flash-Next-EXL3)
MODEL="${1:-${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-EXL3}}"
python3 - "$MODEL" <<'EOF'
import os, sys, glob
n = 0
for p in glob.glob(os.path.join(sys.argv[1], "*")):
    p = os.path.realpath(p)
    if os.path.isfile(p):
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        n += 1
print(f"fadvise DONTNEED on {n} files")
EOF
free -g | sed -n 2p
