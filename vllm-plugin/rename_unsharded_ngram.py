#!/usr/bin/env python3
"""Rename ngram_embedding.trellis -> ngram_embedding.shard_0.trellis inside the safetensors.

Newer exllamav3 releases write the n-gram table as one unsharded tensor (the 4.05bpw_h6_ng6
revision ships this way). vllm-exl3 0.4.2 and its pack-scan tool only know the shard_N layout,
so the table gets no quant spec, vLLM allocates it dense, and the boot OOMs in seconds. A
one-shard table is the same bytes under the name they expect. Streams the data region, keeps
a .native backup, verifies head and tail hashes. Run before vllm-plugin/prepare_pack.sh."""
import json, struct, os, hashlib, shutil, time, sys
if len(sys.argv) != 2:
    sys.exit("usage: rename_unsharded_ngram.py <pack_dir>")
P = os.path.join(sys.argv[1], "ngram_embedding.safetensors")
BK = P + ".native"; TMP = P + ".renamed"
OLD, NEW = ".ngram_embedding.trellis", ".ngram_embedding.shard_0.trellis"
t0 = time.time()
with open(P, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); data_start = 8 + n
keys = [k for k in hdr if k.endswith(OLD)]
assert len(keys) == 1, keys
k = keys[0]; nk = k[:-len(OLD)] + NEW
hdr[nk] = hdr.pop(k)
assert not any(kk.endswith(".shard_0.trellis") for kk in hdr if kk != nk)
newj = json.dumps(hdr, separators=(",", ":")).encode()
newj += b" " * ((8 - len(newj) % 8) % 8)          # safetensors aligns the header to 8
size = os.path.getsize(P); nbytes = size - data_start
def h(fp, off, ln):
    s = hashlib.sha256(); fp.seek(off); s.update(fp.read(ln)); return s.hexdigest()[:16]
with open(P, "rb") as src:
    head_sha = h(src, data_start, 256 << 20); tail_sha = h(src, size - (256 << 20), 256 << 20)
    src.seek(data_start)
    with open(TMP, "wb") as dst:
        dst.write(struct.pack("<Q", len(newj))); dst.write(newj)
        left = nbytes
        while left:
            b = src.read(min(64 << 20, left)); dst.write(b); left -= len(b)
with open(TMP, "rb") as f:
    n2 = struct.unpack("<Q", f.read(8))[0]; hdr2 = json.loads(f.read(n2)); ds2 = 8 + n2
    assert nk in hdr2 and hdr2[nk]["dtype"] == "I16" and len(hdr2[nk]["shape"]) == 2
    size2 = os.path.getsize(TMP); assert size2 - ds2 == nbytes, (size2 - ds2, nbytes)
    assert h(f, ds2, 256 << 20) == head_sha and h(f, size2 - (256 << 20), 256 << 20) == tail_sha
os.rename(P, BK); os.rename(TMP, P)
print("renamed %s -> %s" % (k.split("layers.")[1], nk.split("layers.")[1]))
print("data region %.2f GB, head/tail sha verified, backup at %s, %.0fs" % (nbytes / 1e9, BK, time.time() - t0))
