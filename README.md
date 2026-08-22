# LOCKS

**Page-Local Compact Key Summaries for efficient long-context decoding.** LOCKS
gives each KV *page* a compact, query-independent summary of its keys (the `rki4`
low-rank int-4 page summary) and, at every decode step, attends only the top-`b`
pages that summary ranks highest per (layer, KV-head) -- always keeping the sink
and most-recent pages. Because the summary closely reconstructs each page's attention
mass, a small working set (often 5-13% of the cache) reproduces full-attention
quality: LOCKS stays **within ~1 point of FullKV** at budgets where prior
selectors and evictors lose **7-30+ points**. Prefill is untouched, so
time-to-first-token is unchanged. A single `locks.register()` serves both **GQA**
models (Llama, Qwen, GLM) and **MLA** models (DeepSeek V2/V3/V3.2).

## Highlights

- **Lossless at small budgets.** Within ~1 pt of FullKV on LongBench-v1 and RULER
  once `b >= 512`, and it *tracks the read-every-key oracle to under a point*
  everywhere. It matches or exceeds FullKV at `b >= 512` on LongBench.
- **Large margins over SoTA selectors** at tight budgets, where selection matters
  most (see [Results](#results)).
- **Reasoning-safe.** On MATH-500 / AIME26, LOCKS holds 91-94% where Quest, R-KV,
  TriAttention and LazyEviction collapse below 45% at the *same* budget.
- **MLA / DeepSeek.** Lossless on DeepSeek MLA in vLLM via the same plugin (the
  KV cache is the MLA latent; LOCKS selects over it seamlessly).
- **Faster where it counts.** bs=1 decode is at parity below ~30K context and up
  to **2.0x** at 1M; batched decode reaches **1.8x**. Prefill/TTFT is unchanged.
- **Small footprint.** The only added resident state is the summary, **~9-10% of
  the KV cache** (~768-834 B per page per KV-head). An optional DRAM tier can
  offload V (or K+V).

## Results

Same-engine, identical records, matched budget `b` (tokens per (layer, KV-head);
page 16; sink + recent counted inside `b`). **Oracle** = exact-LSE selection (the
read-every-key ceiling). Full protocol, roster and ablations are in the paper.

**RULER-16K** (Llama-3.1-8B, 13-task mean). FullKV = **94.3**.

| `b` | FullKV | Oracle | **LOCKS** | Quest | ShadowKV | RocketKV |
|----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64   | 94.3 | 80.0 | **78.1** | 39.5 | -    | 69.4 |
| 128  | 94.3 | 87.5 | **87.4** | 57.8 | -    | 85.2 |
| 512  | 94.3 | 91.5 | **91.4** | 79.3 | 84.5 | 89.6 |
| 2048 | 94.3 | 93.7 | **93.7** | 90.3 | 92.5 | 93.1 |

**LongBench-v1** (Llama-3.1-8B, 14-subset mean). FullKV = **47.0**.

| `b` | FullKV | Oracle | **LOCKS** | Quest | ShadowKV | RocketKV |
|----:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64   | 47.0 | 45.9 | **45.7** | 31.6 | -    | 41.6 |
| 128  | 47.0 | 46.6 | **46.7** | 39.8 | -    | 44.1 |
| 512  | 47.0 | 46.8 | **47.1** | 45.9 | 46.3 | 46.6 |
| 2048 | 47.0 | 47.1 | **47.2** | 46.9 | 46.9 | 46.9 |

**Reasoning: MATH-500** (Qwen3-4B, thinking on, avg@4). FullKV = **94.0**.

| `b` | FullKV | Oracle | **LOCKS** | Quest | R-KV | TriAttention | LazyEviction |
|----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 64   | 94.0 | 56.0 | **54.5** | 1.5  | 5.0  | 11.5 | 0.5  |
| 256  | 94.0 | 92.0 | **91.0** | 9.0  | 31.5 | 44.5 | 19.5 |
| 1024 | 94.0 | 94.0 | **94.0** | 68.5 | 77.0 | 81.5 | 75.0 |
| 2048 | 94.0 | 93.5 | **93.5** | 87.5 | 90.5 | 91.5 | 92.0 |

Where value-blind evictors and score-based selectors discard the reasoning
working set, LOCKS keeps it: at `b = 256` it scores **91.0** vs the best baseline's
44.5. (AIME26 shows the same pattern; InfiniteBench at 100K+ context on
GLM-4-9B-Chat-1M has LOCKS the highest measured mean of any deployable selector.)

**MLA / DeepSeek-V2-Lite, RULER-16K** (vLLM 0.24, `b = 2048` = **12.5% attended**):

| arm | attended | avg | vs FullKV |
|---|:---:|:---:|:---:|
| FullKV | 100% | 85.07 | - |
| **LOCKS @2048** | 12.5% | **85.36** | **+0.29** |

Lossless: LOCKS on the MLA latent is within noise of full attention.

## Efficiency

bs=1 decode TPOT, GLM-4-9B-Chat-1M, one H200 NVL, `b = 2048`, vs the *faster* of
FA3 / FlashInfer:

| context | dense (ms) | **LOCKS (ms)** | speedup |
|--------:|:---:|:---:|:---:|
| 16K  | 6.93  | 7.08  | 0.98x (parity) |
| 64K  | 7.97  | 7.38  | 1.08x |
| 128K | 8.74  | 8.27  | 1.06x |
| 256K | 11.63 | 8.96  | **1.30x** |
| 512K | 16.17 | 10.47 | **1.54x** |
| 1M   | 26.19 | 12.95 | **2.02x** |

Parity below ~30K, crossover ~30K, up to 2.02x at 1M (the hot path is
bandwidth-bound and pulls ahead once the dense KV read dominates). Batched decode
amplifies the gap: up to **1.80x at 256K / batch 4** on GLM. TTFT is at parity
(prefill is byte-identical); the only one-time cost is the summary build
(0.5s @ 16K -> 3.7s @ 1M).

## Install

```bash
pip install locks-kv          # import name is `locks`
pip install "locks-kv[vllm]"  # + a compatible vLLM engine
pip install "locks-kv[all]"   # + transformers, pyyaml (YAML configs)
```

Requires Python >= 3.10 and an NVIDIA GPU. Hot-path CUDA kernels compile once at
first use via `torch.utils.cpp_extension` (Hopper `sm_90a`); reference Triton
kernels are the automatic fallback.

## Quickstart

LOCKS installs as a vLLM **general plugin** and auto-registers at engine startup
(entry point `locks = locks.backend.register:register`). It is **inert until you
set `LOCKS_CONFIG`**, so a stock vLLM is unaffected until you opt in. Everything
is one `LocksConfig`, given via the `LOCKS_CONFIG` env var (inline JSON, or a path
to a JSON/YAML file):

```bash
# GQA model (Llama/Qwen/GLM): fixed per-(layer,kv-head) page budget (the main rki4 path).
# budget_pages = selected pages; 128 pages = a 2048-token budget at page size 16.
export LOCKS_CONFIG='{"variant": "fast", "budget_pages": 128}'
vllm serve meta-llama/Llama-3.1-8B-Instruct

# MLA model (DeepSeek V2/V3/V3.2): same plugin, fixed page budget.
export LOCKS_CONFIG='{"variant": "fast", "budget_pages": 128, "window_pages": 10}'
vllm serve deepseek-ai/DeepSeek-V2-Lite-Chat --trust-remote-code

# The memory play: offload V to DRAM, keep K resident (GQA).
export LOCKS_CONFIG='{"variant": "mem-v", "budget_pages": 128}'

# Turn LOCKS off (stock attention / FullKV reference line):
export LOCKS_DISABLE=1
```

The same `locks.register()` handles GQA and MLA: it patches vLLM's backend
selector so that wherever vLLM would pick FlashAttention (GQA) it gets the LOCKS
backend, and wherever it picks an MLA backend (DeepSeek latent attention) it gets
the LOCKS MLA backend. You can also register explicitly from Python:

```python
import locks
locks.register()               # or locks.register({"variant": "fast", "budget_pages": 128})
```

### Configuration (`LocksConfig`)

| field | meaning | default |
|---|---|---|
| `variant` | `fast` (K+V resident), `mem-v` (V in DRAM), `mem-kv` (K+V in DRAM) | `fast` |
| `budget_pages` | fixed **absolute** SELECTED pages per (layer,kv-head), EXCLUDING the always-attended sink/recent (total attended = `budget_pages` + `sink_pages` + `window_pages`); the main knob | none |
| `budget` | fixed selected-page **fraction** of the selectable region (alternative to `budget_pages`) | none |
| `coverage` | **legacy**; the main path is fixed-budget. If set alone (no `budget`/`budget_pages`) selection falls back to a **fixed 0.1 fraction** -- it is NOT adaptive | `0.95` |
| `sink_pages` / `window_pages` | always-kept first / recent pages | `1` / `1` |
| `r8_rank` | rank of the page summary | `8` |
| `quad_combine` | GQA group combine: `nrm` (peak-normalized mass sum) or `max` | `nrm` |
| `use_cuda` | hand-CUDA hot path vs Triton reference | `true` |

Note: MLA currently supports the `fast` variant with a fixed budget
(`budget_pages` / `budget`); the DRAM tier is GQA-only.

## How it works

- **Selection (Stage A).** Each page carries a query-independent **rki4** summary
  built when the page finalizes: an int-4 column-major basis + int8 coefficients +
  an int8 page centroid, from the page-gram eigendecomposition. A query scores
  every page directly (a per-head page-mass estimate) and a static top-`b` keeps
  the highest-scoring pages per (layer, KV-head); the GQA group is combined by a
  peak-normalized mass sum (`nrm`). No second exact pass. The summary is the only
  added resident state, ~9-10% of the KV cache.
- **Sparse decode.** A paged-attention kernel reads only the selected pages plus
  the always-kept sink and recent window. Prefill is untouched (TTFT parity).
- **MLA.** For DeepSeek latent attention the same idea applies to the latent
  cache: a local low-rank summary over each page's `c_KV` block plus the exact
  decoupled RoPE term, max-unioned over the query heads, top-`b` selected. Only
  the decode seam is overridden, so prefill stays stock.
- **Optional DRAM tier (Stage B).** `mem-v` / `mem-kv` offload V (or K+V) to a
  pinned host-DRAM pool, keeping only the summary and the pages called each step
  resident. Measured realized saving ~38% peak VRAM for the K-only engine cache;
  the tier is opt-in and off by default.

## Dependencies

| dependency | why | how declared |
|---|---|---|
| `torch` | tensors + runtime CUDA (`cpp_extension`) kernels | hard (`>=2.4`) |
| `triton` | reference / fallback kernels | hard (`>=3.0`) |
| `vllm` | plugin host + attention backend | extra `[vllm]` (`>=0.8`) |
| `transformers` | tokenizers for bench/eval (lazy) | extra `[bench]` |
| `pyyaml` | YAML `LOCKS_CONFIG` files (JSON needs no dep) | extra `[yaml]` |

`numpy` is not a direct dependency (it arrives transitively via torch). vLLM is
an extra, not a hard floor, because it is a heavy CUDA-ABI-sensitive wheel that
deployments pin themselves.

## License

Apache-2.0. See `LICENSE` for the full text.
