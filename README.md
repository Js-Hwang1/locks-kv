# LOCKS

LOCKS selects pages using compact key summaries and attends to the original KV.
This package contains the canonical optimized implementation, with separate
resident GQA, host-backed GQA, and resident MLA paths. Retired experimental
kernels and eager reference selectors are not part of the runtime.

## Installation

The GQA runtime is validated on GH200 with Python 3.10, PyTorch 2.11.0+cu129,
Triton 3.6.0, vLLM 0.24.0, and the CUDA 12.8 compiler. It requires Linux,
an SM90 Hopper GPU, a CUDA compiler, and Ninja. Install into an existing
compatible PyTorch/vLLM environment:

```sh
python -m pip install --no-deps .
```

Compiled extensions are built on first use. Compilation errors are reported;
they do not switch execution to an eager reference implementation.

## GQA configuration

Provide `LOCKS_CONFIG` as a JSON object or a JSON/YAML file. The vLLM plugin
remains inactive when no configuration is supplied. An explicit budget is
required. One process uses one rank and KV mode; configure these before loading
the model or importing selector state.

```sh
export LOCKS_CONFIG='{"variant":"fast","r8_rank":4,"budget_pages":126,"sink_pages":1,"window_pages":1}'
```

Then construct a normal vLLM engine with the desired model, context length,
batch size, BF16 KV, and block size. Alternatively, register programmatically
before constructing the engine:

```python
import locks

locks.register({
    "variant": "fast",
    "r8_rank": 4,
    "budget_pages": 126,
    "sink_pages": 1,
    "window_pages": 1,
})
```

`r8_rank` is the retained compatibility name for rank; supported native ranks
are 2, 4, and 8. The GQA basis uses INT4, token coefficients use INT8, and the
page centroid uses INT8 with BF16 scales. Native selection supports 16- and
32-token pages and head dimensions 64, 128, and 256. The packed host-backed
rank-4 path uses 16-token pages and head dimension 128.

For GQA, `budget_pages` is the **dynamic** allowance. Sink and recent/current
pages are additional reserved slots inside the intended total attended budget.
At page size 16, 126 dynamic + 1 sink + 1 current = 128 pages = 2048 tokens per
KV head. A 128-token total budget instead uses `budget_pages=6`. An incomplete
current page consumes one slot and its nonexistent tokens are masked.

Every GQA group receives one shared set of page IDs. Each query head's page-LSE
scores are normalized over **all visible pages**, including sink/current;
normalized shares are averaged across the group. Mandatory pages are then
reserved and the dynamic allowance is filled by exact top-k. Ties favor lower
page IDs. The current page is scored from its visible original keys, with the
same native query INT8 round-trip as the summary scorer. Sparse attention uses
original BF16 queries, keys, and values. Candidate-only normalization, peak
normalization, and query-head unions are not runtime alternatives.

Resident batch-1 engines automatically use the packed scorer for rank 4,
16-token pages, 128-dimensional heads, eight queries per KV head, and a
configured capacity of 1,024–8,192 pages or at least 16,384 pages. It uses the existing asynchronous
summary prefetch and tensor-core centroid computation. The quantized values,
scores, shared selections, and original-KV attention remain identical to the
separate-array layout. Other resident geometries and the batched offloaded
dispatch keep their existing paths.

The 1,024–8,192-page resident path also fuses current-page scoring with the
first normalization reduction, and combines the remaining normalization with
the GQA vote. Its scorer uses 128 grid partitions unless explicitly configured.
At 2,049–4,096 pages and at most 126 dynamic slots, the existing exact top-k
kernel uses a 4,096-page tile. Single-request metadata refresh uses CPU scalar
state instead of vector tensor operations. These specializations preserve the
all-visible-page normalization and stable page-ID tie break. They do not change
the 128K-and-larger tested context configurations, offloaded execution, or
engines configured for more than one concurrent request.

This layout stores 512 bytes per page/KV head: 490 bytes of selector payload,
16 bytes of tags, a 2-byte coefficient norm bound, and 4 bytes of padding.
The separate-array layout stores the same selector payload and tags in
506 bytes. Use the state's `bytes_per_layer()` for its actual allocation.

## Host-backed GQA

Use `variant="mem-kv"` to keep the complete K/V store in CPU memory, or
`variant="mem-v"` to offload values while keeping keys resident. Set the
capacity explicitly for reproducible measurements, for example with
`hot_slots`, `v_blocks`, `max_pool_gb`, vLLM's context/batch limits, and its
GPU-memory utilization limit.

The full-KV path automatically uses compact engine identity handles, packed
rank-4 summaries, registered CPU huge pages, exact host allocation sizes,
guarded incremental staging, overlapped transfer scheduling, and attention
metadata prefetch. Release/acquire publication and the eviction fence protect
original K/V while transfers overlap decode. The allocator binds its host
mapping to CPU NUMA node 0 by default; the measured platform is GH200.
On a PCIe H200 host, set `LOCKS_HOST_NUMA_NODE` to the CPU NUMA node nearest
the selected GPU. Inspect `nvidia-smi topo -m` and `numactl --hardware` to choose
a CPU node with DRAM, not a GPU memory NUMA node. The setting applies before
allocation and appears in the host-pool description. Linux `libnuma.so.1` and
permission to bind memory to that node are required.

For example, only if the selected GPU is attached to CPU NUMA node 1:

```sh
export LOCKS_HOST_NUMA_NODE=1
```

The same registered-host storage and optimized transport kernels are used on
PCIe. GH200 throughput measurements do not establish H200 PCIe throughput.

The rank-4 selector payload is 490 bytes per page per KV head; the packed
physical record is 512 bytes including tags and padding. Rank-8's logical
payload is 818 bytes. Do not equate payload bytes with total runtime memory.

`rki4_zsplit` controls scorer grid partitioning without changing scores.
The validated Table 3 settings use 9 at 64K/batch 58, 48 at 128K/batch 29,
and 96 at 256K/batch 14, with a 2048-token total attended budget and the
experiment's explicitly configured hot-cache capacities. Numerical parity
does not imply identical performance at arbitrary ranks, batches, or devices.

## MLA / DeepSeek

Resident MLA support is retained through `LocksMLABackend` and its fused
Triton scorer over compact page-local latent factors. This preserves the
repository's existing MLA score: the maximum reconstructed token logit across
query heads, with exact RoPE-key contributions. It is a separate architecture
path, **not** the GQA share-average/r4i4 representation. Its factors are FP32,
and `budget_pages` is the total MLA page budget including sink/recent slots.
These differences are explicit; no GQA paper result should be attributed to
this MLA path.

MLA uses the same optimized exact top-k implementation, selected original
latent KV for attention, and the stock MLA prefill. The model's native DSA
indexer is bypassed only when LOCKS is selected as its MLA backend. MLA
offloading, FP8 latent KV, and DeepSeek V4's distinct attention architecture
are not implemented by this adapter. Its fused scorer uses eager orchestration;
it does not claim the GQA path's full-graph support. MLA integration validation is separate
from the GH200/GLM performance measurements.

## Source layout and validation

- `src/locks/backend/`: vLLM registration, metadata, memory planning, GQA/MLA adapters.
- `src/locks/selection/`: page-summary construction, native scoring, share-average,
  and scalable exact top-k.
- `src/locks/attention/`: resident sparse attention and split reduction.
- `src/locks/tier/`: host storage, staging, cache residency, transfers, and offloaded attention.
- `tests/`: independent reference calculations and regression checks.

The pre-cleanup source remains in Git history at `80de3a9`. Accepted experiment snapshots are
kept separately from this package; historical implementations are not bundled
as alternative runtime paths.

The regression checks require the compatible GPU environment above and an
explicit report destination:

```sh
python tests/check_selection.py --report /tmp/locks_selection.json
python tests/check_shared_average.py --report /tmp/locks_shared_average.json
python tests/check_mla.py --report /tmp/locks_mla.json
python tests/check_batch1_packed.py --report /tmp/locks_batch1_packed.json
python tests/check_host_refresh.py --report /tmp/locks_host_refresh.json
```

`python tests/check_host_numa.py` checks host-node selection and a small Linux
CPU-memory binding without allocating GPU memory. Run the GPU checks on an
idle device; do not run them alongside latency measurements.

The MLA check uses synthetic latent KV and does not establish full-model
DeepSeek correctness or performance. The measured GLM configurations exercise
resident GQA and complete host-backed KV; no new latency claim is made for
the value-only offload mode.
