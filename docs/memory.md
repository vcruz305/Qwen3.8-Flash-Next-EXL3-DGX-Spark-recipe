# Memory advisory and PLE placement

The recipe's [memory helper](../exllamav3-tabby/tools/model_memory.py) reads
safetensors headers and the model configuration without loading weights or
initializing CUDA. It estimates pack storage, the configured K8/V8 cache pool,
and an additional memory allowance. A successful estimate is a reason to try a
controlled load with resource monitoring; it is not a measured peak or a
guarantee that every workload fits.

## What is counted

`need_stream_bytes` contains non-PLE weight bytes, K/V backing storage,
quantization scales, QSA indexer planes when present, and `slack_bytes`.
`need_ram_bytes` adds the PLE table bytes once. These fields and the
`memory.MemAvailable` field remain the interface for experiment controllers.

The named cache components follow the engine's
[quantized-cache layout](https://github.com/vcruz305/exllamav3/blob/b5322c98c5760105de04f7cf7c28bece17fbc78e/exllamav3/cache/quant.py)
and [QSA side planes](https://github.com/vcruz305/exllamav3/blob/b5322c98c5760105de04f7cf7c28bece17fbc78e/exllamav3/cache/qsa.py):

| Component | Storage |
| --- | --- |
| `kv_data_bytes` | One byte for each K value and each V value at K8/V8. |
| `kv_scale_bytes` | One FP16 scale per 32 values, separately for K and V. |
| `qsa_raw_key_bytes` | One FP16 indexer key of `indexer_head_dim` values per token. |
| `qsa_pooled_key_bytes` | One FP16 indexer key per `indexer_compress_ratio` tokens. |
| `qsa_index_bytes` | Raw plus pooled indexer planes. |
| `kv_estimate_bytes` | K/V data, scales, and QSA planes combined. |

There is no additional generic 10% multiplier on those components. Quantization
scales and QSA planes are distinct allocations. In particular, the QSA planes
remain FP16 when the K/V cache is quantized.

For the inspected Qwen packs, all 12 target full-attention layers and the one
MTP layer use two K/V heads of dimension 256. QSA uses one indexer key head of
dimension 128 with compression ratio 4. The full pool is allocated independently
for the target and draft caches; it is shared among concurrent requests.

| Shared token capacity | K/V data | K/V scales | QSA planes | Total cache backing |
| ---: | ---: | ---: | ---: | ---: |
| 262,144 | 3.25000 GiB | 0.203125 GiB | 1.015625 GiB | **4.46875 GiB** |
| 1,048,576 | 13.00000 GiB | 0.812500 GiB | 4.062500 GiB | **17.87500 GiB** |

This replaces the earlier 3.575/14.3 GiB estimates that used a generic allowance
and omitted the QSA planes. Disabling MTP removes its K/V, scale, and indexer
allocations; the target-only 262,144-token total is 4.125 GiB.

The helper reads layer types or the full-attention interval from the
configuration. An ordinary GQA model with neither a hybrid interval nor a
layer-type list is counted as having attention in every hidden layer. Without
QSA configuration, the indexer components are zero. This fallback assumes the
same grouped-query K8/V8 layout; it is not an estimator for MLA or sliding-window
cache designs. Capacities are rounded up to 256-token pages, matching Tabby's
cache sizing. Incomplete or unsupported QSA geometry causes the estimate to
fail; automatic PLE selection then chooses streaming.

## The large PLE table is held once

In both the original `94ba01d` engine and the reviewed `b5322c98` engine, PLE
RAM mode retains a CPU tensor. Each forward gathers the required packed rows,
uploads those rows, and decodes them on the GPU. **There is no full GPU mirror
of the 30–36 GiB PLE table.** MTP borrows the target's ordinary token embedding
and output head; it does not load a second PLE table.

The CPU PLE tensor is an allocation containing a copy of the file data, not a
shared file mapping. File reads can also populate the Linux page cache. Those
clean cached pages are reclaimable; they should not be counted as a second
permanent PLE allocation. Disk mode instead keeps file handles and gathers rows
as needed. Its page-cache residency depends on the workload and available RAM.

The inspected packs use these layouts:

| Pack | PLE tensor layout | PLE data | Extra input-shard staging during RAM assembly |
| --- | --- | ---: | ---: |
| Flat 3.05 | 128 I16 shards, 51 words per row | 30.39851 GiB | At most one 243.19 MiB shard at a time. |
| Flat 4.05 | One I16 tensor, 61 words per row | 36.35901 GiB | No full-table assembly copy with the default loader. |
| SAGE 4.15 | 128 I16 shards, 61 words per row | 36.35901 GiB | At most one 290.87 MiB shard at a time. |
| Cyber 3.87 | 128 I16 shards, 61 words per row | 36.35901 GiB | At most one 290.87 MiB shard at a time. |

The default native `mt_fread` loader reads the unsharded 4.05 tensor directly
into its final CPU destination. The optional Python loader's
`fp.read(...)`/bytearray path can transiently duplicate a large payload; the
recipe's Tabby path does not select it. Sharded RAM loading allocates a single
full slab, copies each input shard into its slice, and releases that shard.

`NGRAM_RAM=true` selects resident PLE storage through Tabby.
`EXL3_NGRAM_LOCK` is a separate engine option, disabled by default, that locks
the same CPU pages against eviction or swapping; it does not create a second
copy. The helper does not change either setting. See the reviewed
[PLE loader](https://github.com/vcruz305/exllamav3/blob/b5322c98c5760105de04f7cf7c28bece17fbc78e/exllamav3/modules/ngram_embedding.py)
and [native file reader](https://github.com/vcruz305/exllamav3/blob/b5322c98c5760105de04f7cf7c28bece17fbc78e/exllamav3/exllamav3_ext/stloader.cpp)
for the placement and staging paths.

## What still consumes the 10 GiB allowance

The default 10 GiB allowance remains a shared budget for allocations outside
the named weight and cache components. It is not 10 GiB of guaranteed free
space after those allocations occur.

The ordinary token embedding is different from PLE. All four inspected packs
have a BF16 token embedding with shape `[248320, 2560]`, or 1.184082 GiB.
The CPU weight is included in the pack estimate. With device-resident MTP token
IDs, `EXL3_EMBED_GPU=1` enables an additional CUDA mirror when the table is
within `EXL3_EMBED_GPU_MAX_MB` (default 4096 MiB). The target and MTP share
that mirror. On the Spark, its CPU and CUDA copies consume the same physical
memory budget. This mirror is created lazily when the CPU embedding receives
CUDA token IDs; see the [embedding implementation](https://github.com/vcruz305/exllamav3/blob/b5322c98c5760105de04f7cf7c28bece17fbc78e/exllamav3/modules/embedding.py).

The recurrent checkpoint reservoir defaults to
`SYSMEM_RECURRENT_CACHE=4096` MiB. It grows with saved checkpoints rather than
being fully allocated at startup. Device recurrent states and speculative
history are additional: with the inspected geometry and five draft tokens,
their backing tensors are approximately 0.6393 GiB for one slot or 2.5570 GiB
for four slots. Increasing `MAX_BATCH_SIZE` or `DRAFT_NUM_TOKENS` increases
that allocation. These quantities are not included in `kv_estimate_bytes`.

Consequently, the possible token-embedding mirror, a filled checkpoint
reservoir, and these recurrent tensors alone use roughly 5.82 GiB of the
default allowance for one slot or 7.74 GiB for four slots. Remaining consumers
include the shard staging above, pinned row buffers, the pruned draft-head
copy, CUDA graphs and kernel workspaces, allocator overhead, and host-process
memory. Long prompts and different chunk sizes can change transient peaks.
The JSON's `slack_covers` and `advisory_note` fields make this boundary explicit.

## Use an unloaded-host snapshot, then measure

Read the advisory after the previous model process has stopped and released its
allocations. Use the actual model directory, draft mode, and shared pool size:

```bash
python3 exllamav3-tabby/tools/model_memory.py \
  --model /home/cruzspark/models/flashnext-exl3-sage-4.15bpw \
  --cache-size 262144 --draft-mode mtp --ngram-ram true \
  --slack-gib 10 --json > memory-advisory.json
```

For a RAM-PLE candidate require `need_ram_bytes <= memory.MemAvailable`;
for streaming use `need_stream_bytes`. `MemAvailable` accounts for
reclaimable memory, whereas `MemFree` excludes the page cache. A snapshot
taken during another model's lifecycle is not an unloaded-host fit decision.

The helper can print a memory warning while exiting successfully; parse the
JSON fields instead of treating exit 0 as proof of a fit. Keep the advisory
beside the deployment configuration and record the actual load and workload
resource samples. A short load or a bounded sample interval does not establish
the worst possible peak, and a full shared cache allocation does not establish
that four simultaneous maximum-context requests meet a latency target.

CPU checks for the corrected geometry and the controller-facing fields:

```bash
python3 -m unittest discover -s bench -p test_model_memory.py -v
```
