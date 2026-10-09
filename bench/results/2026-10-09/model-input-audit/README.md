# Current model input identity audit

At 2026-10-09T05:58:40.407460Z, a read-only audit recorded the four existing model packs and compared their saved identities with the October 8 final API reports. **All four recorded identities matched.** See [summary.json](summary.json) and the unchanged [comparison](raw/comparison.json).

The comparison covers config/tokenizer/generation metadata content hashes, resolved model and weight paths, filenames, sizes, modification times, and the literal unsorted safetensors enumeration order used by the loader. That order matters when an MTP sidecar duplicates tensor keys. The current audit additionally records bounded safetensors header hashes, tensor shapes/dtypes and duplicate-key information; it reads no weight payload bytes.

| Pack label | Resolved directory basename | Saved identity |
|---|---|---|
| Flat 3.05 | flashnext-exl3-3.05bpw | Equal |
| Flat 4.05 | flashnext-exl3-4.05bpw | Equal |
| SAGE 4.15 | flashnext-exl3-sage-4.15bpw | Equal |
| Cyber 3.87 | CYBER-FROST-3.8-EXL3-SAGE-3.87bpw | Equal |

This does **not** prove every weight payload byte was unchanged. The historical identities did not contain raw header hashes or full-weight hashes; matching size/mtime/path is a metadata comparison. New current header hashes can support subsequent bounded drift checks but cannot retroactively establish a historical header comparison.

The exact audit source, current records, comparison and original checksum file are preserved under [raw](raw/). The four small historical API result JSONs used as references are copied byte-for-byte under [historical-reference](historical-reference/); their checksums match the hashes declared by the comparison. The October 8 originals are unchanged.

The audit script retains its original host-specific WSL/SSH/model paths and is evidence of the method, not a general installer. Do not invoke archived host-bound scripts without adapting and reviewing their explicit paths. No CUDA/model load, API call, model mutation or full tensor hashing occurred during this audit or its publication copy.

The collection receipt at the parent directory provides source/destination hashes. Verify this directory with `sha256sum -c SHA256SUMS`; the nested original `raw/SHA256SUMS` is also preserved.
