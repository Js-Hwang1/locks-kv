"""r8 build / refresh — the query-INDEPENDENT low-rank int8 page summary.

Runs OUTSIDE the CUDA graph (torch + ``linalg.eigh``, data-dependent shapes),
once per page-finalize, exactly like the reference ``dynkv_plugin.py:185-213``
and following the content-tag invalidation of ``vllm/kernels/selection_state.py``
(rebuild ONLY finalized pages whose stored tag no longer matches the live K).

For a FINALIZED page every page token is valid (``(p+1)*page <= seq_len``), so
no per-token validity mask is needed here (the partial tail page is never built
- it sits inside the always-attended window). Per (block, kv-head):

    mu = mean_t K            (d,)                    page centroid
    dc = K - mu              (page, d)               centered keys
    Gm = dc @ dcᵀ            (page, page)            page-gram (cheap eigh)
    S², U = eigh(Gm)                                 ascending eigenpairs
    S8 = sqrt(clamp(S²[-r:]))                        (r,)   top-r singular values
    U8 = U[..., -r:]         (page, r)
    c  = U8 * S8             (page, r)               coords
    Vk = dcᵀ @ U8 / S8       (d, r)                  right singular axes

Then int8 fake-quant at the configured grain (default: per-column Vk, per-page
c, per-(block,head) mu -- matching the R8State scale layouts) and scatter into
the layer's slab; the page-final key's leading TAGW channels become the new tag.
"""
from __future__ import annotations

import os

import torch

from .. import arch as _arch


# --------------------------------------------------------------------------- #
# Custom 16x16 cyclic-Jacobi eigensolver (Piece C).                            #
#                                                                              #
# ``torch.linalg.eigh`` on the batched page-grams routes to cusolver           #
# ``syevBatched``, which FAILS (CUSOLVER_STATUS_INTERNAL_ERROR) at >= 65536    #
# grams -- i.e. bs16 @ 16K, bs1 @ 131K, bs4 @ 64K -- a correctness blocker for #
# long-ctx / high-batch (measured: scratch_audit/build_probe.py).  The page is #
# 16 tokens, so every gram is a symmetric 16x16.  This one-warp-per-gram        #
# cyclic-Jacobi solver has NO batch-size ceiling and removes the ~33 ms         #
# cusolver overhead.  Eigenvalues are sorted ASCENDING in torch to match the    #
# ``torch.linalg.eigh`` contract the build epilogue expects.                    #
# --------------------------------------------------------------------------- #
_JAC_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef JAC_WARPS
#define JAC_WARPS 8
#endif
#ifndef JAC_SWEEPS
#define JAC_SWEEPS 16
#endif
#define JN 16              // page = 16 -> symmetric 16x16 gram
#define JPAD 17            // smem row pad -> conflict-free

// TWO kernels, dispatched by gram count M (measured, interleaved A/B under
// contention, scratch_deepopt3/jacobi_ab.py):
//   * PARALLEL-ORDER (round-robin) Jacobi -- 15 rounds of 8 DISJOINT pairs
//     per sweep, all 8 two-sided rotations applied concurrently (they commute
//     exactly).  ~7x less serial depth for ~+14% work (JAC_SWEEPS 16 vs 14
//     for the weaker per-sweep convergence; accuracy vs torch eigh unchanged
//     at ev_rel ~5e-6 / proj2 ~1e-5).  WINS when depth-bound (small M = the
//     per-fire TAIL path): M=256 562 vs 687 us (1.22x).
//   * CYCLIC Jacobi (the original): 120 sequential rotations/sweep.  WINS
//     when throughput-bound (bulk builds): M=4096 1134 vs 1291, M=65536
//     13.5 vs 17.6 ms (parallel pays its +14% work there).
// Dispatch: parallel for M <= JAC_PAR_MAX (1024), cyclic above.
// Round-robin schedule (fix 15, rotate 0..14): round r pairs
//   (15, r) and ((r+i) mod 15, (r-i) mod 15) for i = 1..7.
__device__ __forceinline__ void rr_pair(int r, int i, int* p, int* q) {
    int a, b;
    if (i == 0) { a = 15; b = r; }
    else {
        a = (r + i) % 15;
        b = (r - i + 15) % 15;
    }
    *p = a < b ? a : b;
    *q = a < b ? b : a;
}

__global__ void jacobi_eigh16_par(const float* __restrict__ Gm, // (M,16,16)
                                  float* __restrict__ W,         // (M,16)
                                  float* __restrict__ Vout,      // (M,16,16)
                                  const int M) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * JAC_WARPS + warp;
    if (g >= M) return;
    __shared__ float As[JAC_WARPS][JN][JPAD];
    __shared__ float Vs[JAC_WARPS][JN][JPAD];
    __shared__ float Cs[JAC_WARPS][8], Ss[JAC_WARPS][8];
    __shared__ int Pp[JAC_WARPS][8], Qq[JAC_WARPS][8];
    float (*A)[JPAD] = As[warp];
    float (*V)[JPAD] = Vs[warp];
    const float* Gg = Gm + (long)g * JN * JN;
    for (int i = lane; i < JN * JN; i += 32) {
        int r = i / JN, c = i % JN;
        A[r][c] = Gg[i];
        V[r][c] = (r == c) ? 1.f : 0.f;
    }
    __syncwarp();
    for (int sweep = 0; sweep < JAC_SWEEPS; ++sweep) {
        for (int rnd = 0; rnd < JN - 1; ++rnd) {
            // 1. lanes 0-7: pair + angle of this round (round-start A). The
            //    %15 pair math runs ONCE here (8 lanes) -- recomputing it in
            //    the phases (16 lanes x 8 pairs x 2 phases) was measured 5x
            //    slower than the whole rotation math (no HW modulo).
            if (lane < 8) {
                int p, q;
                rr_pair(rnd, lane, &p, &q);
                Pp[warp][lane] = p;
                Qq[warp][lane] = q;
                float apq = A[p][q];
                float cc = 1.f, ss = 0.f;
                if (fabsf(apq) > 1e-20f) {
                    float theta = (A[q][q] - A[p][p]) / (2.f * apq);
                    float t = (theta >= 0.f ? 1.f : -1.f)
                              / (fabsf(theta) + sqrtf(theta * theta + 1.f));
                    cc = rsqrtf(t * t + 1.f);
                    ss = t * cc;
                }
                Cs[warp][lane] = cc;
                Ss[warp][lane] = ss;
            }
            __syncwarp();
            // 2. column phase: A <- A J, V <- V J (8 disjoint column pairs).
            if (lane < JN) {
                int k = lane;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int p = Pp[warp][i], q = Qq[warp][i];
                    float cc = Cs[warp][i], ss = Ss[warp][i];
                    float akp = A[k][p], akq = A[k][q];
                    A[k][p] = cc * akp - ss * akq;
                    A[k][q] = ss * akp + cc * akq;
                    float vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = cc * vkp - ss * vkq;
                    V[k][q] = ss * vkp + cc * vkq;
                }
            }
            __syncwarp();
            // 3. row phase: A <- J^T A (8 disjoint row pairs).
            if (lane < JN) {
                int k = lane;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int p = Pp[warp][i], q = Qq[warp][i];
                    float cc = Cs[warp][i], ss = Ss[warp][i];
                    float apk = A[p][k], aqk = A[q][k];
                    A[p][k] = cc * apk - ss * aqk;
                    A[q][k] = ss * apk + cc * aqk;
                }
            }
            __syncwarp();
        }
    }
    // write eigenvalues (diag) + eigenvector columns.
    if (lane < JN) W[(long)g * JN + lane] = A[lane][lane];
    for (int i = lane; i < JN * JN; i += 32)
        Vout[(long)g * JN * JN + i] = V[i / JN][i % JN];
}

#define JAC_SWEEPS_CYC 14
// CYCLIC Jacobi (the original kernel): best THROUGHPUT at bulk M.
__global__ void jacobi_eigh16_cyc(const float* __restrict__ Gm,
                                  float* __restrict__ W,
                                  float* __restrict__ Vout,
                                  const int M) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = blockIdx.x * JAC_WARPS + warp;
    if (g >= M) return;
    __shared__ float As[JAC_WARPS][JN][JPAD];
    __shared__ float Vs[JAC_WARPS][JN][JPAD];
    float (*A)[JPAD] = As[warp];
    float (*V)[JPAD] = Vs[warp];
    const float* Gg = Gm + (long)g * JN * JN;
    for (int i = lane; i < JN * JN; i += 32) {
        int r = i / JN, c = i % JN;
        A[r][c] = Gg[i];
        V[r][c] = (r == c) ? 1.f : 0.f;
    }
    __syncwarp();
    for (int sweep = 0; sweep < JAC_SWEEPS_CYC; ++sweep) {
        for (int p = 0; p < JN - 1; ++p) {
            for (int q = p + 1; q < JN; ++q) {
                float apq = A[p][q];
                if (fabsf(apq) > 1e-20f) {
                    float app = A[p][p], aqq = A[q][q];
                    float theta = (aqq - app) / (2.f * apq);
                    float t = (theta >= 0.f ? 1.f : -1.f)
                              / (fabsf(theta) + sqrtf(theta * theta + 1.f));
                    float cc = rsqrtf(t * t + 1.f);
                    float ss = t * cc;
                    if (lane < JN) {
                        int k = lane;
                        float akp = A[k][p], akq = A[k][q];
                        A[k][p] = cc * akp - ss * akq;
                        A[k][q] = ss * akp + cc * akq;
                    }
                    __syncwarp();
                    if (lane < JN) {
                        int k = lane;
                        float apk = A[p][k], aqk = A[q][k];
                        A[p][k] = cc * apk - ss * aqk;
                        A[q][k] = ss * apk + cc * aqk;
                    }
                    if (lane < JN) {
                        int k = lane;
                        float vkp = V[k][p], vkq = V[k][q];
                        V[k][p] = cc * vkp - ss * vkq;
                        V[k][q] = ss * vkp + cc * vkq;
                    }
                    __syncwarp();
                }
            }
        }
    }
    if (lane < JN) W[(long)g * JN + lane] = A[lane][lane];
    for (int i = lane; i < JN * JN; i += 32)
        Vout[(long)g * JN * JN + i] = V[i / JN][i % JN];
}

#define JAC_PAR_MAX 1024   // measured crossover: depth-bound below, thru above

void jacobi_eigh16_launch(torch::Tensor Gm, torch::Tensor W, torch::Tensor V,
                          int64_t M) {
    const int block = JAC_WARPS * 32;
    const int grid = (int)((M + JAC_WARPS - 1) / JAC_WARPS);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (M <= JAC_PAR_MAX)
        jacobi_eigh16_par<<<grid, block, 0, stream>>>(
            Gm.data_ptr<float>(), W.data_ptr<float>(), V.data_ptr<float>(),
            (int)M);
    else
        jacobi_eigh16_cyc<<<grid, block, 0, stream>>>(
            Gm.data_ptr<float>(), W.data_ptr<float>(), V.data_ptr<float>(),
            (int)M);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("jacobi_eigh16_launch", &jacobi_eigh16_launch, "16x16 Jacobi eigh");
}
"""

_JAC_MOD = None
_JAC_TRIED = False


def _jac_mod():
    global _JAC_MOD, _JAC_TRIED
    if _JAC_TRIED:
        return _JAC_MOD
    _JAC_TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        _JAC_MOD = load_inline(
            name="locks_jacobi_eigh16", cpp_sources="", cuda_sources=_JAC_SRC,
            extra_cuda_cflags=["-O3", _arch.arch_flag()],
            verbose=False)
    except Exception:
        _JAC_MOD = None
    return _JAC_MOD


def _eigh_jacobi(Gm: torch.Tensor):
    """Drop-in for ``torch.linalg.eigh`` on batched symmetric 16x16 grams.
    Returns (S2, U) with S2 ASCENDING and U the eigenvector columns, matching
    the ``torch.linalg.eigh`` contract.  Falls back to torch.linalg.eigh when
    the extension is unavailable or the page is not 16 (kernel is 16x16-only)."""
    mod = _jac_mod()
    if mod is None or Gm.shape[-1] != 16 or Gm.shape[-2] != 16:
        return torch.linalg.eigh(Gm)
    *lead, n, _ = Gm.shape
    A = Gm.reshape(-1, 16, 16).contiguous().to(torch.float32)
    M = A.shape[0]
    W = torch.empty(M, 16, device=A.device, dtype=torch.float32)
    V = torch.empty(M, 16, 16, device=A.device, dtype=torch.float32)
    mod.jacobi_eigh16_launch(A, W, V, M)
    # sort ascending (eigh contract) + reorder eigenvector columns to match.
    order = W.argsort(dim=-1)                                  # (M,16)
    S2 = torch.gather(W, -1, order)
    U = torch.gather(V, -1, order.unsqueeze(-2).expand(M, 16, 16))
    return (S2.reshape(*lead, 16).to(Gm.dtype),
            U.reshape(*lead, 16, 16).to(Gm.dtype))


# --------------------------------------------------------------------------- #
# int8/int4 fake-quant helpers.  code = round(x / s), symmetric levels          #
# +/-(2^(bits-1)-1) (int8: +/-127, int4: +/-7), s = amax(|x|, grain)/lvl.  int4  #
# packs two signed nibbles per int8 byte along the LAST (rank) axis -> the code  #
# tensor's last dim halves; the fp16 scale layout is bit-width-independent.      #
# --------------------------------------------------------------------------- #
def _quant(x: torch.Tensor, dim, bits=8):
    lvl = float((1 << (bits - 1)) - 1)                    # 127 (int8) / 7 (int4)
    s = (x.abs().amax(dim=dim, keepdim=True) / lvl).clamp_min(1e-8)
    code = (x / s).round().clamp_(-lvl, lvl).to(torch.int8)
    return code, s


def _pack_i4(code: torch.Tensor) -> torch.Tensor:
    """Pack an int8-typed int4 code (..., r) into (..., r//2) int8: byte =
    lo_nibble | (hi_nibble << 4), each nibble a 4-bit two's-complement value."""
    r = code.shape[-1]
    assert r % 2 == 0
    nib = (code.to(torch.int32) & 0xF)                    # 4-bit two's complement
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8).view(torch.int8).contiguous()


def _quant_mu(mu):
    # mu (..., d) -> scale per (block, head): amax over d.  mu STAYS int8.
    code, s = _quant(mu, dim=-1, bits=8)
    return code, s.squeeze(-1).to(torch.float16)          # (..., )


def _quant_vk(Vk, grain, bits=8):
    # Vk (..., d, r).  "col" (default) = per-column r (amax over d);
    # "tensor" = one scale per (block, head), broadcast into the r slots.
    if grain == "col":
        code, s = _quant(Vk, dim=-2, bits=bits)           # s (...,1,r)
        sc = s.squeeze(-2).to(torch.float16)              # (..., r)
    else:
        code, s = _quant(Vk, dim=(-2, -1), bits=bits)     # s (...,1,1)
        sc = s.squeeze(-2).expand(*Vk.shape[:-2], Vk.shape[-1])
        sc = sc.contiguous().to(torch.float16)            # (..., r)
    if bits == 4:
        code = _pack_i4(code)                             # (..., d, r//2)
    return code, sc


def _quant_c(c, grain, bits=8):
    # c (..., page, r).  "page" (default) = per-page-token (amax over r);
    # "tensor" = one scale per (block, head), broadcast into the page slots.
    if grain == "page":
        code, s = _quant(c, dim=-1, bits=bits)            # s (...,page,1)
        sc = s.squeeze(-1).to(torch.float16)              # (..., page)
    else:
        code, s = _quant(c, dim=(-2, -1), bits=bits)      # s (...,1,1)
        sc = s.squeeze(-1).expand(*c.shape[:-2], c.shape[-2])
        sc = sc.contiguous().to(torch.float16)            # (..., page)
    if bits == 4:
        code = _pack_i4(code)                             # (..., page, r//2)
    return code, sc


# --------------------------------------------------------------------------- #
# Steady-state page-finalize TAIL rebuild (scratch_deepopt3 iteration 2).      #
#                                                                              #
# The full ``*_build_refresh`` is a per-layer loop with device syncs           #
# (``.item()``, ``unique``, boolean gathers): measured ~99 ms per finalize     #
# step at bs1/16K under contention (~2100 launches + 32 serial 1-gram jacobi   #
# calls), i.e. ~6 ms/step amortized -- the dominant LOCKS decode overhead.     #
# In the steady decode state the ONLY page that can change is the one that     #
# just finalized: block_table[r, seq//page - 1].  These tail builders rebuild  #
# exactly that page for EVERY request, for ALL layers, in one batched,         #
# sync-free op chain (~55 launches total, one jacobi launch).  Rebuilding is   #
# IDEMPOTENT (the summary is a pure function of the page's K bytes), so        #
# unconditionally rebuilding every request's last finalized page -- including  #
# rows that did not just cross a boundary, padded rows (stale-but-valid block  #
# ids), and rows whose page was already built -- is safe.  Short rows          #
# (seq < page) clamp to page 0: its (partial-K) summary is never scored        #
# before the page finalizes (window >= 1) and is rebuilt at that step.        #
# The builder gate calls these ONLY on steady finalize steps; slot changes /  #
# first decode take the full tag-scan refresh above.                          #
# --------------------------------------------------------------------------- #
def _tail_blocks(block_table, seq_lens, n_req: int, page: int, rows):
    """Physical block of each (selected) request's LAST finalized page
    (fin-1, clamped to 0).  ``rows`` None = all first n_req rows; else a
    device int64 row-index tensor (the rows that just crossed a boundary,
    computed on the HOST from the gate's CPU lens).  Sync-free."""
    if rows is None:
        sl = seq_lens[:n_req].to(torch.int64)
        bt = block_table[:n_req]
    else:
        sl = seq_lens[rows].to(torch.int64)
        bt = block_table[rows]
    fin1 = (sl // page - 1).clamp_(min=0)                     # (n,)
    return bt.to(torch.int64).gather(1, fin1[:, None]).squeeze(1)


def _scatter_index(L: int, l0: int, blocks: torch.Tensor):
    """(lidx, bidx) flat (L*n,) advanced-index pair for the L-major slabs,
    covering layers [l0, l0+L) x blocks."""
    n = blocks.shape[0]
    lidx = torch.arange(l0, l0 + L, device=blocks.device,
                        dtype=torch.int64).repeat_interleave(n)
    bidx = blocks.repeat(L)
    return lidx, bidx


def _page_eigh(Kp_flat: torch.Tensor, r: int):
    """Shared eigh core: (N, page, n_kv, d) bf16 K pages ->
    (mu, U8, S8, S2r, Vk) fp32, the exact op sequence of the per-layer
    refreshes (mean / center / page-gram / eigh / top-r factors)."""
    Kp = Kp_flat.permute(0, 2, 1, 3).float()                  # (N,n_kv,page,d)
    mu = Kp.mean(dim=2)
    dc = Kp - mu[:, :, None, :]
    Gm = torch.matmul(dc, dc.transpose(-1, -2))
    if Gm.shape[-1] == 16 and os.environ.get("LOCKS_EIGH", "jacobi") != "torch":
        S2, U = _eigh_jacobi(Gm)
    else:
        S2, U = torch.linalg.eigh(Gm)
    S2r = S2[..., -r:].clamp_min(1e-10)
    S8 = S2r.sqrt()
    U8 = U[..., -r:]
    Vk = torch.matmul(dc.transpose(-1, -2), U8) / S8[..., None, :]
    return mu, U8, S8, S2r, Vk


def _bulk_blocks(block_table, seq_lens, n_req: int, page: int, max_fin: int):
    """All finalized (request, page) physical blocks, flattened to (P,) with
    invalid slots clamped to block 0 (idempotent rebuild; never scored before
    its own finalize step).  Fixed shapes, zero syncs."""
    device = block_table.device
    sl = seq_lens[:n_req].to(torch.int64)
    n_fin = sl // page                                        # (R,)
    pidx = torch.arange(max_fin, device=device)
    valid = pidx[None, :] < n_fin[:, None]                    # (R, F)
    return block_table[:n_req, :max_fin].to(torch.int64) \
        .masked_fill_(~valid, 0).reshape(-1)                  # (P,)


def _build_blocks(write_fn, st, K_layers, blocks: torch.Tensor,
                  chunk_pairs: int) -> int:
    """Run ``write_fn`` (a ``*_write``) over ``blocks`` for every layer, in
    layer groups (small P) or pair slices (large P). ``chunk_pairs`` bounds
    the fp32 transient to ~1 GiB."""
    P = blocks.shape[0]
    if P == 0:
        return 0
    L = len(K_layers)
    page, n_kv, d = st.page, st.n_kv, st.d
    if P <= chunk_pairs:                     # group layers per math chain
        lg = max(1, chunk_pairs // P)
        for l0 in range(0, L, lg):
            Ks = K_layers[l0:l0 + lg]
            Kp_raw = torch.stack([K[blocks] for K in Ks])
            lidx, bidx = _scatter_index(len(Ks), l0, blocks)
            write_fn(st, Kp_raw.reshape(len(Ks) * P, page, n_kv, d),
                     lidx, bidx)
    else:                                    # slice pairs within each layer
        for l, K in enumerate(K_layers):
            for s0 in range(0, P, chunk_pairs):
                bs_ = blocks[s0:s0 + chunk_pairs]
                lidx, bidx = _scatter_index(1, l, bs_)
                write_fn(st, K[bs_], lidx, bidx)
    return L * P


# --------------------------------------------------------------------------- #
# DELTA rebuild on batch-composition changes (scratch_deepopt3 iteration 6).  #
#                                                                              #
# A mixed(prefill+decode)->pure-decode transition, a request arrival, an      #
# end-of-generate DRAIN step (vLLM repeats padded decode steps with frozen    #
# seq lens while finished rows condense/shuffle), or a preemption resume all  #
# break the steady +1 gate.  Rebuilding everything (bulk) on each such step   #
# measured ~800 ms at bs16/16K, and ROW-identity heuristics mis-fire on the   #
# drain shuffles (summaries are keyed by PHYSICAL BLOCK -- a row shuffle      #
# invalidates nothing).  The precise selector is the system's original        #
# content mechanism, batched: gather EVERY finalized page's content tag       #
# across ALL layers in one pass (~64 MB / ~30 us GPU at bs16/16K, ~35        #
# launches), compare against the stored tags, rebuild ONLY stale blocks      #
# (any-layer stale -> rebuild the block in all layers; NaN-init tags make    #
# never-built pages stale by construction).  Same-ids-same-content reuse is  #
# correctly skipped; same-ids-different-content slot reuse is caught at the  #
# same tag-collision trust level the original per-layer scan accepted.       #
# ONE device sync (the data-dependent stale count) on this rare path.        #
# --------------------------------------------------------------------------- #
def _delta_select(tag_slab, K_layers, block_table, seq_lens, n_req: int,
                  page: int, max_fin: int, tagw: int) -> torch.Tensor:
    """(blocks,) whose stored tag mismatches the live K in ANY layer."""
    device = block_table.device
    sl = seq_lens[:n_req].to(torch.int64)
    fin = sl // page                                       # (R,)
    pidx = torch.arange(max_fin, device=device)
    valid = pidx[None, :] < fin[:, None]                   # (R, F)
    blk = block_table[:n_req, :max_fin].to(torch.int64) \
        .masked_fill_(~valid, 0).reshape(-1)               # (P,)
    tag_cur = torch.stack([K[blk, page - 1, :, :tagw] for K in K_layers])
    stale = (tag_cur != tag_slab[:, blk]).any(-1).any(-1)  # (L, P)
    stale = stale.any(dim=0) & valid.reshape(-1)           # (P,)
    return blk[stale]                                      # (n,) SYNC


# --------------------------------------------------------------------------- #