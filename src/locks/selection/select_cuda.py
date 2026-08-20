"""topb_select_cuda — hand-CUDA (Hopper sm_90a) static top-b page selector.

Bitwise-identical drop-in for the golden Triton ``topb_select`` in
``select.py``: one CTA per (request, kv-head), STATIC per-request budget ``b``
(from ``st.b_fix``, identical across heads/layers), select the top-b pages by
``st.score (R, n_kv, MP) fp32`` over the SELECTABLE region
``[sink_pages, n_sel_hi)``, ALWAYS-keep sinks + recent window + partial tail,
and write ``st.page_table (R, n_kv, MP) int32`` (ascending-compacted, -1 padded)
+ ``st.page_cnt (R, n_kv) int32``.  The GQA group union is already folded into
the score by ``r8_score`` (group-max), so this is one top-b per (req, kv).

WHY radix-select (not sort).  Selection only needs ONE order statistic --
``tau`` = the b-th LARGEST selectable score -- yet the previous kernel ran a full
O(n log^2 n) in-smem BITONIC SORT of all selectable pages to extract it (72us at
bs1/64K = a third of the decode step; ~6% SM occupancy at bs1).  A sort is
wasted work.  This kernel finds ``tau`` in O(n) with a 4-pass MSD radix-select
over the fp32 scores reinterpreted as order-preserving uint32 (``f2u``): each
pass histograms one 8-bit digit (256 shared bins) of the elements whose already
fixed high bits match, scans the buckets high->low to locate the one holding
rank ``b``, fixes that digit, and narrows.  After 4 digits every bit is fixed and
the 32-bit prefix IS the order-uint of ``tau`` (``u2f`` back to fp32).  The final
``score >= tau`` keep test is a PURE FLOAT compare, byte-identical to Triton's
``score >= tl.sort(sv,desc)[b-1]``: radix returns the exact b-th-largest VALUE
(ties at ``tau`` all kept, order-statistic exact under duplicates), so the kept
SET and its ascending compaction are bytewise the reference's.

Semantics matched to the reference kernel (``select.py:_topb_select_kernel``),
EXACTLY:
  * tau = the b-th LARGEST score over the selectable slice.  Radix-select finds
    the value ``v`` with ``count(score>v) < b <= count(score>=v)`` == srt(desc)
    [b-1], the same value ``tl.sort`` yields.  ``b`` is clamped to ``[1,n_sel]``;
    the -inf pad Triton sorts to the tail is never the b-th largest (b<=n_sel) so
    it is simply not scanned here.
  * keep(page): DETERMINISTIC EXACT-B (user ruling 2026-07-20; the old
    threshold-tie rule "ties at tau all kept, count can exceed b" is
    ABOLISHED).  Selectable kept = the g pages STRICTLY above tau plus
    exactly quota = b - g pages EQUAL to tau, lowest page index first
    (count-and-demote in the smem cache; global scores untouched), so the
    selectable count == b BY CONSTRUCTION, every step, plus always =
    (p < sink) | (p >= n_sel_hi) and the keep_all short-circuit.
    Identical rule in the Triton golden (eqrank <= quota form, same set).
  * compaction: ascending page-id order via an in-smem inclusive prefix sum,
    ids in [0, cnt), -1 in [cnt, mp) -- bytewise identical to the Triton
    ``cumsum`` scatter.

Static-``b`` specialisation: ``b`` is a single per-request scalar (block-uniform,
also uniform across the kv-head grid dim) -> zero warp divergence in the
histogram/keep loops; the radix passes are ``b``-independent, ``b`` only steers
the bucket scan.  Loop bounds / smem sizes come from ``P_PAD`` (a stable launch
constant for a given state), so the launch config is fixed.

Fixed launch shapes (grid = (n_req, n_kv), block = 512 by default, override with
LOCKS_TOPB_BLK), host-sync-free, allocation-free -> FULL CUDA-graph safe (verified
by a capture/replay test in ``scratch_topb/``).  Self-contained ``load_inline``
build (sm_90a, hash-cached).
"""
from __future__ import annotations

import os

import triton  # only for next_power_of_2 (identical P_PAD to select.py)
import torch

from .. import arch as _arch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <math_constants.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// fp32 -> order-preserving uint32 and back (radix key).  Monotone in float
// order for all finite values and +-inf; +0/-0 map to distinct keys but the
// downstream keep test is a pure float compare so the kept set is unaffected.
__device__ __forceinline__ unsigned f2u(float f) {
    unsigned u = __float_as_uint(f);
    unsigned mask = (unsigned)(-(int)(u >> 31)) | 0x80000000u;
    return u ^ mask;
}
__device__ __forceinline__ float u2f(unsigned x) {
    unsigned mask = ((x >> 31) - 1u) | 0x80000000u;
    return __uint_as_float(x ^ mask);
}

// ======================================================================== //
// EXACT-B TIE CUTOFF (2026-07-22 fix, ctx>=128K split/TAUC route).
//
// The page-parallel finish (sel_tau|sel_tauc_* -> sel_keepscan) had NO tie
// resolution: keepscan applied a bare `score >= tau`, so an exact fp32 tie AT
// tau kept a strict SUPERSET of the top-b (measured page_cnt 8193 vs the
// invariant 128).  Every OTHER route resolves the tie by count-and-demote
// (topb_select_kernel, nrmtopb fused, nrmtopb_select_v2, sel_finish, and the
// never-dispatched sel_finish_global's ebm_eb bitmap): keep every page with
// score > tau, then fill the remaining `quota = b - #(>tau)` slots with the
// LOWEST-PAGE-INDEX equals.
//
// The demote CANNOT be carried the way sel_finish carries it here: keepscan
// is page-parallel (one CTA per chunk), so no single CTA owns the score
// array, and the score buffer is GLOBAL -- demoting in it would mutate
// st.score, which is a PUBLISHED output (the PTAB gates, the r8 screen and
// the field instruments all read it), and would make the selection depend on
// inter-CTA write visibility.  sel_finish_global's ebm_eb bitmap is also not
// portable here: it is 2048 SMEM WORDS, i.e. it silently caps at P_PAD <=
// 65536, so it would corrupt smem at ctx >= 512K (P_PAD 131072) -- the reason
// it is ported as a CUTOFF INDEX instead of a bitmap.
//
// Carrier: ONE int per (request, kv-head), `gcut`, written by the same kernel
// that writes gtau (no new launch, no new stream edge, kernel boundary = the
// visibility fence), consumed read-only by keepscan.  Nothing is mutated:
// st.score is untouched.
//     keep(i) = always | keep_all | score[i] > tau
//                                 | (score[i] == tau && i <= gcut)
// gcut = INT_MAX  -> the old `>= tau` exactly (no tie overflow; also the
//                    kill-switch state, LOCKS_SEL_EXACTB=0)
// gcut = -1       -> quota 0, no equal survives
// otherwise       -> page index of the quota-th equal in ascending order.
//
// This routine finds that index: an ascending, block-parallel, coalesced scan
// with a running count of equals; it exits at the tile that contains the
// quota-th equal.  Deterministic (no atomics, no warp-order dependence): the
// rank of an equal is its position in ASCENDING PAGE ORDER, exactly the
// documented tie rule.  All threads of the block must call it (barriers).
// ======================================================================== //
#define LOCKS_NO_CUT 0x7fffffff
__device__ int sel_tie_cutoff(const float* __restrict__ score, const long sbase,
                              const int lo, const int hi, const float tauf,
                              const int quota, const int tid, const int BLK,
                              int* s_run, int* s_cut, int* wsum)
{
    if (quota <= 0) return -1;
    const int lane = tid & 31, wid = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;
    if (tid == 0) { *s_run = 0; *s_cut = LOCKS_NO_CUT; }
    __syncthreads();
    for (int base = lo; base < hi; base += BLK) {
        const int i = base + tid;
        const bool eq = (i < hi) && (score[sbase + i] == tauf);
        const unsigned m = __ballot_sync(0xffffffffu, eq);
        const int wpref = __popc(m & ((1u << lane) - 1u));
        if (lane == 0) wsum[wid] = __popc(m);
        __syncthreads();
        int tot = 0, wex = 0;
        for (int w = 0; w < nwarp; ++w) { if (w == wid) wex = tot; tot += wsum[w]; }
        if (eq && (*s_run + wex + wpref) == quota - 1) *s_cut = i;
        __syncthreads();
        if (tid == 0) *s_run += tot;
        __syncthreads();
        if (*s_cut != LOCKS_NO_CUT) break;
    }
    return *s_cut;
}

// One CTA per (request, kv-head).  blockIdx.x = request, blockIdx.y = kv-head.
// Dynamic smem carves: sc[P_PAD] fp32 (page-score cache: ONE global read shared
// by radix + keep) | hist[256] i32 (radix bins) | kbuf[P_PAD] i32 (keep flags,
// scanned in-place into the compaction rank).  All loop bounds strided over
// blockDim so any (P_PAD, BLK) is correct.  Static smem holds the small scan
// scratch (thread totals + per-warp offsets + the radix bucket/rank broadcast).
#define MAXBLK 1024
__global__ void topb_select_kernel(
    const float* __restrict__ score,     // (R, n_kv, MP)  fp32
    const int*   __restrict__ npg,       // (R,)  n_pages
    const int*   __restrict__ nsh,       // (R,)  n_sel_hi
    const int*   __restrict__ bfix,      // (R,)  static b
    int*         __restrict__ page_table,// (R, n_kv, MP)  int32
    int*         __restrict__ page_cnt,  // (R, n_kv)      int32
    const int  n_sink,
    const long s_sr, const long s_sh,    // score strides (row, head)
    const int  mp,
    const long t_sr, const long t_sh,    // page_table strides (row, head)
    const long c_sr,                     // page_cnt row stride
    const int  P_PAD, const int prof_mode)
{
    const int r   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;

    const int n_pages  = npg[r];
    const int n_sel_hi = nsh[r];
    int       b        = bfix[r];
    const int n_sel    = n_sel_hi - n_sink;      // selectable page count
    const bool keep_all = (n_sel <= 1);          // too short -> attend all

    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;

    extern __shared__ unsigned char smem[];
    float* sc   = reinterpret_cast<float*>(smem);            // P_PAD
    int*   hist = reinterpret_cast<int*>(sc + P_PAD);        // 256
    int*   kbuf = hist + 256;                                // P_PAD
    // (a 4KB per-thread-totals array declared here was DEAD -- the scan uses
    // warp shuffles + warp_ex only; removed, freeing static smem)
    __shared__ int   warp_ex[32];                            // per-warp offsets
    __shared__ int   s_sel, s_k, s_cb;                       // radix bucket/rank/count

    // PDL: when launched with the programmatic-stream-serialization
    // attribute (LOCKS_PDL=1), this grid starts while the SCORE kernel
    // drains; everything above is score-independent prologue and the wait
    // (no-op without an armed dependency) fences before the score read.
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- cache all page scores into smem (one coalesced global read) ----
    for (int i = tid; i < P_PAD; i += BLK)
        sc[i] = (i < n_pages) ? score[sbase + i] : -CUDART_INF_F;
    __syncthreads();

    // ---- tau = b-th largest selectable score via 4-pass MSD radix-select ----
    // Order-uint keys; ``prefix`` accumulates fixed high bits, each pass
    // histograms the next 8-bit digit of the keys still matching ``prefix`` and
    // one warp scans buckets HIGH->low (via shuffle) to the one holding rank k.
    float tau = -CUDART_INF_F;
    if (!keep_all && prof_mode != 2) {   // prof_mode 2 = compaction-only (tau=-inf)
        if (b < 1)     b = 1;
        if (b > n_sel) b = n_sel;
        unsigned prefix = 0u;    // fixed high bits of tau's key
        unsigned kmask  = 0u;    // which high bits are fixed
        int      k      = b;     // rank sought within the matching set
        // NOTE (deep-opt audit): a warp-aggregated histogram (__match_any +
        // leader-only atomicAdd) was tried against digit clustering and
        // MEASURED SLOWER (radix 7.0->8.5us bs1/16K, 8.6->12.3 bs16/64K):
        // Hopper's smem atomics absorb the hot-bin bursts cheaper than the
        // ballot/match/ffs/popc overhead per element.  Plain atomics stay.
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            for (int i = tid; i < 256; i += BLK) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < n_sel; i += BLK) {
                unsigned u = f2u(sc[n_sink + i]);
                if ((u & kmask) == prefix)
                    atomicAdd(&hist[(u >> shift) & 0xFF], 1);
            }
            __syncthreads();
            // warp 0 locates the crossing bucket: each lane owns 8 bins, a
            // shuffle suffix-scan finds the segment holding rank k, then the
            // crossing lane linearly scans its 8 bins (all warp-synchronous).
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;                       // inclusive suffix over lanes
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;               // elements in higher lanes
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;             // crossing lane
                if (lane == cl) {
                    int acc = above, sel = cl * 8;
                    int cb  = 0;                     // crossing-bucket count
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; cb = c; break; }
                        acc += c;
                    }
                    s_sel = sel;
                    s_k   = k - acc;                 // residual rank in the bucket
                    s_cb  = cb;
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift;
            k       = s_k;
            // EARLY EXIT (bitwise-safe): when the residual rank equals the
            // crossing-bucket count, tau_ref is the SMALLEST element carrying
            // this prefix, so keep(score >= tau_ref) == keep(key >= prefix
            // with all-zero low bits): no element with the prefix sits below
            // tau_ref (it would be in the bucket), and every higher element
            // differs in the already-fixed bits.  tau = u2f(prefix||0...0)
            // yields the IDENTICAL kept set through the unchanged float
            // compare, so the remaining radix passes cannot change anything.
            // On clustered attention scores this fires at pass 2-3 for most
            // steps (the radix was 4.3us of topb's 8.8us at bs1/16K;
            // scratch_deepopt2/topb_probe.py).
            if (k == s_cb)
                break;
            kmask  |= 0xFFu << shift;
            __syncthreads();
        }
        tau = u2f(prefix);                                // b-th largest value
    }
    if (prof_mode == 1) {                // prof: radix-only, skip keep/compact
        if (tid == 0) page_cnt[(long)r * c_sr + kh] = (int)(tau > -CUDART_INF_F);
        return;
    }

    // DETERMINISTIC EXACT-B (user ruling 2026-07-20: threshold-tie
    // semantics abolished).  g = #(selectable > tau), E = #(== tau);
    // post-radix E >= b-g always and E == b-g unless true fp32 collisions
    // at tau (measure-zero), so the common path is byte-identical; on
    // overflow the highest-page-index equals beyond quota are demoted to
    // -inf IN THE SMEM CACHE ONLY (ascending serial walk = deterministic;
    // the global score output is untouched) and the unchanged keep test
    // below then yields EXACTLY b selectable pages, always.
    __shared__ int s_gt_eb, s_eq_eb;
    if (tid == 0) { s_gt_eb = 0; s_eq_eb = 0; }
    __syncthreads();
    if (!keep_all) {
        int gt_l = 0, eq_l = 0;
        for (int i = n_sink + tid; i < n_sel_hi; i += BLK) {
            const float v = sc[i];
            if (v > tau) ++gt_l;
            else if (v == tau) ++eq_l;
        }
        if (gt_l) atomicAdd(&s_gt_eb, gt_l);
        if (eq_l) atomicAdd(&s_eq_eb, eq_l);
    }
    __syncthreads();
    if (tid == 0 && !keep_all && s_eq_eb > b - s_gt_eb) {
        int left = b - s_gt_eb;
        for (int i = n_sink; i < n_sel_hi; ++i)
            if (sc[i] == tau) {
                if (left > 0) --left; else sc[i] = -CUDART_INF_F;
            }
    }
    __syncthreads();

    // ---- keep flag per page into kbuf (reads the cached score) ----
    for (int i = tid; i < P_PAD; i += BLK) {
        int keep = 0;
        if (i < n_pages) {
            bool always = (i < n_sink) || (i >= n_sel_hi);   // sinks+window+tail
            keep = (always || keep_all || sc[i] >= tau) ? 1 : 0;
        }
        kbuf[i] = keep;
    }
    __syncthreads();

    // ---- inclusive prefix sum of keep flags -> compaction rank, in place ----
    // Blocked 2-phase scan (each thread owns a contiguous ELS chunk): O(P_PAD)
    // work in ~2 passes + a single BLK-wide block scan, vs the log(P_PAD)-pass
    // Hillis-Steele.  Integer adds -> the incl[] (hence page_table) is bytewise
    // the reference's regardless of scan order.
    const int ELS  = (P_PAD + BLK - 1) / BLK;
    const int base = tid * ELS;
    int ttot = 0;
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) { int idx = base + j; if (idx < P_PAD) ttot += kbuf[idx]; }
    // block exclusive scan of the per-thread totals (warp scan + warp offsets)
    int incl_w = ttot;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, incl_w, d);
        if (lane >= d) incl_w += up;
    }
    if (lane == 31) warp_ex[wid] = incl_w;   // per-warp total
    __syncthreads();
    if (wid == 0) {
        int w  = (tid < nwarp) ? warp_ex[tid] : 0;
        int wi = w;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, wi, d);
            if (lane >= d) wi += up;
        }
        if (tid < nwarp) warp_ex[tid] = wi - w;   // exclusive per-warp offset
    }
    __syncthreads();
    int off = warp_ex[wid] + (incl_w - ttot);     // exclusive prefix for this thread
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) {
        int idx = base + j;
        if (idx < P_PAD) { off += kbuf[idx]; kbuf[idx] = off; }  // inclusive rank
    }
    __syncthreads();
    const int cnt = kbuf[P_PAD - 1];             // total kept pages

    if (tid == 0) page_cnt[(long)r * c_sr + kh] = cnt;

    // ---- -1 pad [cnt, mp) then ascending scatter of kept ids into [0, cnt) ----
    for (int i = cnt + tid; i < mp; i += BLK)
        page_table[tbase + i] = -1;
    for (int i = tid; i < n_pages; i += BLK) {
        int prev = (i > 0) ? kbuf[i - 1] : 0;
        if (kbuf[i] - prev == 1)                 // this page is kept
            page_table[tbase + (kbuf[i] - 1)] = i;
    }
    // PDL: release a dependent decode kernel as this grid drains (no-op
    // when none is armed).
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

void topb_select_cuda(torch::Tensor score, torch::Tensor npg,
                      torch::Tensor nsh, torch::Tensor bfix,
                      torch::Tensor page_table, torch::Tensor page_cnt,
                      int64_t n_req, int64_t n_kv, int64_t n_sink,
                      int64_t mp, int64_t P_PAD, int64_t blk,
                      int64_t prof_mode, int64_t pdl) {
    TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
    TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table int32");
    TORCH_CHECK(page_cnt.scalar_type() == at::kInt, "page_cnt int32");
    TORCH_CHECK(npg.scalar_type() == at::kInt && nsh.scalar_type() == at::kInt
                && bfix.scalar_type() == at::kInt, "params int32");
    const int BLK = (int)blk;
    // sc[P_PAD] fp32 + hist[256] i32 + kbuf[P_PAD] i32  (thtot/warp_ex are static)
    const size_t smem = 256ul * sizeof(int)
                        + (size_t)P_PAD * sizeof(float)
                        + (size_t)P_PAD * sizeof(int);
    // opt in to >48KB dynamic smem once (host-side attr set; NOT a stream op,
    // so it is never captured into a CUDA graph -- safe under capture).
    static int recorded_max = 0;
    if ((int)smem > recorded_max) {
        cudaFuncSetAttribute(topb_select_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        recorded_max = (int)smem;
    }
    dim3 grid((unsigned)n_req, (unsigned)n_kv);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (pdl) {
        // programmatic dependent launch: overlap this grid's prologue with
        // the tail of the preceding (score) kernel on the stream.  Captured
        // into CUDA graphs as a programmatic edge (CUDA >= 12.0).
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = dim3((unsigned)BLK);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, topb_select_kernel,
            score.data_ptr<float>(), npg.data_ptr<int>(), nsh.data_ptr<int>(),
            bfix.data_ptr<int>(), page_table.data_ptr<int>(),
            page_cnt.data_ptr<int>(), (int)n_sink,
            (long)score.stride(0), (long)score.stride(1), (int)mp,
            (long)page_table.stride(0), (long)page_table.stride(1),
            (long)page_cnt.stride(0), (int)P_PAD, (int)prof_mode);
        return;
    }
    topb_select_kernel<<<grid, BLK, smem, stream>>>(
        score.data_ptr<float>(), npg.data_ptr<int>(), nsh.data_ptr<int>(),
        bfix.data_ptr<int>(), page_table.data_ptr<int>(),
        page_cnt.data_ptr<int>(), (int)n_sink,
        score.stride(0), score.stride(1), (int)mp,
        page_table.stride(0), page_table.stride(1), page_cnt.stride(0),
        (int)P_PAD, (int)prof_mode);
}

// ======================================================================== //
// TAIL-OPT fused nrm-combine + top-b: ONE kernel replaces the (nrm kernel +
// topb kernel) pair of the production nrm chain.  Same grid (req, kv-head).
// The nrm head computes score[p] = sum_g exp(score_h[g,p] - hmax[g]) in ONE
// pass -- hmax comes from the SCORE kernel's epilogue (the hmax handshake:
// order-preserving-uint atomicMax = exact fp32 page max; has_hm==0 falls
// back to an in-kernel phase-1 sweep) -- writes st.score (contract kept)
// and fills the radix smem cache directly, then runs the BYTE-IDENTICAL
// radix-select + keep + compaction of topb_select_kernel above.  Removes
// one launch + one full read of score_h from the critical chain.  The exp
// sum order (g ascending) and the fp32 max are identical to the separate
// nrm kernel -> score bitwise equal, page_table/page_cnt byte-equal (gated).
// hmax slots are RESET to f2u(-inf) after reading (graph-replay invariant).
// ======================================================================== //
// P13 vector width helpers: VEC in {1,2,4} floats, host-selected by the MP /
// stride alignment (score_h rows are MP apart; MP=1030 at a 16K engine only
// admits VEC=2).  Loads/stores over CONSECUTIVE pages within one (g) row.
template <int VEC>
__device__ __forceinline__ void ldvec(float* dst, const float* src) {
    if constexpr (VEC == 4) {
        const float4 t = *reinterpret_cast<const float4*>(src);
        dst[0] = t.x; dst[1] = t.y; dst[2] = t.z; dst[3] = t.w;
    } else if constexpr (VEC == 2) {
        const float2 t = *reinterpret_cast<const float2*>(src);
        dst[0] = t.x; dst[1] = t.y;
    } else {
        dst[0] = *src;
    }
}
template <int VEC>
__device__ __forceinline__ void stvec(float* dst, const float* src) {
    if constexpr (VEC == 4) {
        *reinterpret_cast<float4*>(dst) = make_float4(src[0], src[1],
                                                      src[2], src[3]);
    } else if constexpr (VEC == 2) {
        *reinterpret_cast<float2*>(dst) = make_float2(src[0], src[1]);
    } else {
        *dst = src[0];
    }
}

template <int VEC, int GT>
__global__ __launch_bounds__(MAXBLK) void nrmtopb_select_kernel(
    const float* __restrict__ score_h,   // (R, n_kv, G, MP) per-head S
    unsigned*    __restrict__ hmax,      // (R, n_kv, 8) f2u page maxima
    const int*   __restrict__ npg,       // (R,)  n_pages
    const int*   __restrict__ nsh,       // (R,)  n_sel_hi
    const int*   __restrict__ bfix,      // (R,)  static b
    float*       __restrict__ score,     // (R, n_kv, MP)  written [0, nsh)
    int*         __restrict__ page_table,// (R, n_kv, MP)  int32
    int*         __restrict__ page_cnt,  // (R, n_kv)      int32
    const int  n_sink, const int n_kv, const int G,
    const long sh_sr, const long sh_sh, const long sh_sg,
    const long s_sr, const long s_sh,    // score strides (row, head)
    const int  mp,
    const long t_sr, const long t_sh,    // page_table strides (row, head)
    const long c_sr,                     // page_cnt row stride
    const int  P_PAD, const int has_hm)
{
    const int r   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;

    const int n_pages  = npg[r];
    const int n_sel_hi = nsh[r];
    int       b        = bfix[r];
    const int n_sel    = n_sel_hi - n_sink;
    const bool keep_all = (n_sel <= 1);

    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;

    extern __shared__ unsigned char smem[];
    float* sc   = reinterpret_cast<float*>(smem);            // P_PAD
    int*   hist = reinterpret_cast<int*>(sc + P_PAD);        // 256
    int*   kbuf = hist + 256;                                // P_PAD
    __shared__ int   warp_ex[32];
    __shared__ int   s_sel, s_k, s_cb;
    __shared__ float hm_sh[8];
    __shared__ float wmax[32][8];        // has_hm==0 phase-1 scratch

    // PDL: start while the SCORE grid drains; fence before reading its
    // outputs (score_h + hmax).
    asm volatile("griddepcontrol.wait;" ::: "memory");

    const float* shp = score_h + (long)r * sh_sr + (long)kh * sh_sh;

    // P13: zero the radix pass-0 bins UP FRONT; the nrm loop below counts
    // each page's digit as it computes the score (order-free integer adds ->
    // identical 256 counts), so the radix never re-scans for digit 0.
    for (int i = tid; i < 256; i += BLK) hist[i] = 0;

    // ---- per-group page max: handshake read (+reset) or in-kernel sweep ----
    if (has_hm) {
        if (tid < 8) {
            const long hidx = ((long)r * n_kv + kh) * 8 + tid;
            hm_sh[tid] = u2f(hmax[hidx]);
            hmax[hidx] = 0x007FFFFFu;    // f2u(-inf): rest-state invariant
        }
        __syncthreads();
    } else {
        // P13 sweep: VEC pages per thread, loads HOISTED past the g<G
        // predicate (the OPT1 lesson applied to the sweep -- the old form
        // serialized G dependent gmem round-trips per page).  fp32 max is
        // order-free, so any page grouping gives the bit-exact hm[g].
        float lmax[GT];
        #pragma unroll
        for (int g = 0; g < GT; ++g) lmax[g] = -CUDART_INF_F;
        const int nfull = (n_sel_hi / VEC) * VEC;
        for (int i = tid * VEC; i < nfull; i += BLK * VEC) {
            float sv[GT][VEC];
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                if (g < G)
                    ldvec<VEC>(&sv[g][0], shp + (long)g * sh_sg + i);
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                if (g < G) {
                    #pragma unroll
                    for (int v = 0; v < VEC; ++v)
                        lmax[g] = fmaxf(lmax[g], sv[g][v]);
                }
        }
        for (int i = nfull + tid; i < n_sel_hi; i += BLK) {
            float sv[GT];
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                sv[g] = (g < G) ? shp[(long)g * sh_sg + i] : -CUDART_INF_F;
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                if (g < G)
                    lmax[g] = fmaxf(lmax[g], sv[g]);
        }
        #pragma unroll
        for (int g = 0; g < GT; ++g) {
            #pragma unroll
            for (int o = 16; o >= 1; o >>= 1)
                lmax[g] = fmaxf(lmax[g], __shfl_xor_sync(~0u, lmax[g], o));
            if (lane == 0) wmax[wid][g] = lmax[g];
        }
        __syncthreads();
        if (wid == 0 && lane < GT) {
            float m = -CUDART_INF_F;
            for (int w = 0; w < nwarp; ++w) m = fmaxf(m, wmax[w][lane]);
            hm_sh[lane] = m;
        }
        __syncthreads();
    }
    float hm[GT];
    #pragma unroll
    for (int g = 0; g < GT; ++g) hm[g] = hm_sh[g];

    // ---- single pass: nrm exp-sum -> score (contract) + radix smem cache ---
    // sc[i] for i >= n_sel_hi is never consumed (keep is forced by `always`
    // there and the radix scans [n_sink, n_sel_hi) only) -> -inf, no read of
    // the stale global score (byte-identical outputs to the two-kernel path).
    // P13: VEC pages per thread (hoisted row loads, vector score/sc stores);
    // each page keeps its g-ascending expf/add order -> scores bitwise-equal.
    // The pass-0 histogram add rides the same loop (counts order-free).
    {
        const int nfull = (n_sel_hi / VEC) * VEC;
        for (int i = tid * VEC; i < nfull; i += BLK * VEC) {
            float sv[GT][VEC];
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                if (g < G)
                    ldvec<VEC>(&sv[g][0], shp + (long)g * sh_sg + i);
            float out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float acc = 0.f;
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    if (g < G)
                        acc += __expf(sv[g][v] - hm[g]);
                out[v] = acc;
            }
            stvec<VEC>(score + sbase + i, out);
            stvec<VEC>(sc + i, out);
            #pragma unroll
            for (int v = 0; v < VEC; ++v)
                if (i + v >= n_sink)
                    atomicAdd(&hist[f2u(out[v]) >> 24], 1);
        }
        for (int i = nfull + tid; i < P_PAD; i += BLK) {
            float v = -CUDART_INF_F;
            if (i < n_sel_hi) {
                float sv[GT];
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    sv[g] = (g < G) ? shp[(long)g * sh_sg + i] : 0.f;
                float acc = 0.f;
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    if (g < G)
                        acc += __expf(sv[g] - hm[g]);
                score[sbase + i] = acc;
                v = acc;
                if (i >= n_sink)
                    atomicAdd(&hist[f2u(acc) >> 24], 1);
            }
            sc[i] = v;
        }
    }
    __syncthreads();

    // ---- tau radix-select: BYTE-IDENTICAL to topb_select_kernel ------------
    // (P13: digit 0 uses the histogram the nrm loop already accumulated --
    // same per-page u32 bins, integer adds, order-free -> identical counts.)
    float tau = -CUDART_INF_F;
    if (!keep_all) {
        if (b < 1)     b = 1;
        if (b > n_sel) b = n_sel;
        unsigned prefix = 0u;
        unsigned kmask  = 0u;
        int      k      = b;
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            if (digit > 0) {
                for (int i = tid; i < 256; i += BLK) hist[i] = 0;
                __syncthreads();
                // (P13 note: VEC-vectorizing this smem scan measured a WASH
                // at 128K and -1.7% at 16K; a pass-0 boundary-bin candidate
                // shortcut (warp-exact k-th over <=32 stashed values) measured
                // -1.0us @128K / -0.2 @16K despite firing -- the phase is
                // crossing-scan + barrier bound and extra control structure
                // deoptimizes it.  Kept scalar and straight-line.)
                for (int i = tid; i < n_sel; i += BLK) {
                    unsigned u = f2u(sc[n_sink + i]);
                    if ((u & kmask) == prefix)
                        atomicAdd(&hist[(u >> shift) & 0xFF], 1);
                }
                __syncthreads();
            }
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;
                if (lane == cl) {
                    int acc = above, sel = cl * 8;
                    int cb  = 0;
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; cb = c; break; }
                        acc += c;
                    }
                    s_sel = sel;
                    s_k   = k - acc;
                    s_cb  = cb;
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift;
            k       = s_k;
            if (k == s_cb)
                break;
            kmask  |= 0xFFu << shift;
            __syncthreads();
        }
        tau = u2f(prefix);
    }

    // DETERMINISTIC EXACT-B (see topb_select_kernel: same count-and-demote
    // on the smem cache; keep test below unchanged).
    __shared__ int s_gt_eb, s_eq_eb;
    if (tid == 0) { s_gt_eb = 0; s_eq_eb = 0; }
    __syncthreads();
    if (!keep_all) {
        int gt_l = 0, eq_l = 0;
        for (int i = n_sink + tid; i < n_sel_hi; i += BLK) {
            const float v = sc[i];
            if (v > tau) ++gt_l;
            else if (v == tau) ++eq_l;
        }
        if (gt_l) atomicAdd(&s_gt_eb, gt_l);
        if (eq_l) atomicAdd(&s_eq_eb, eq_l);
    }
    __syncthreads();
    if (tid == 0 && !keep_all && s_eq_eb > b - s_gt_eb) {
        int left = b - s_gt_eb;
        for (int i = n_sink; i < n_sel_hi; ++i)
            if (sc[i] == tau) {
                if (left > 0) --left; else sc[i] = -CUDART_INF_F;
            }
    }
    __syncthreads();

    // ---- keep + scan + compact: same integer prefix sums / final bytes as
    // topb_select_kernel; P13 stores kbuf PADDED (KB: one skip word per 32,
    // kills the stride-8 8-way bank conflicts of the per-thread segments)
    // and prefills the WHOLE page_table row with -1 in VEC chunks BEFORE the
    // compaction overwrites [0, cnt) -- each cell's FINAL value is unchanged.
#define KB(i) kbuf[(i) + ((i) >> 5)]
    for (int i = tid; i < P_PAD; i += BLK) {
        int keep = 0;
        if (i < n_pages) {
            bool always = (i < n_sink) || (i >= n_sel_hi);
            keep = (always || keep_all || sc[i] >= tau) ? 1 : 0;
        }
        KB(i) = keep;
    }
    // -1 prefill of the full row rides BEFORE the barrier (independent smem/
    // gmem targets); the compaction below overwrites [0, cnt).
    {
        int i = tid * VEC;
        const int mpf = (mp / VEC) * VEC;
        for (; i < mpf; i += BLK * VEC) {
            if constexpr (VEC == 4)
                *reinterpret_cast<int4*>(page_table + tbase + i) =
                    make_int4(-1, -1, -1, -1);
            else if constexpr (VEC == 2)
                *reinterpret_cast<int2*>(page_table + tbase + i) =
                    make_int2(-1, -1);
            else
                page_table[tbase + i] = -1;
        }
        for (i = mpf + tid; i < mp; i += BLK)
            page_table[tbase + i] = -1;
    }
    __syncthreads();

    const int ELS  = (P_PAD + BLK - 1) / BLK;
    const int base = tid * ELS;
    int ttot = 0;
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) { int idx = base + j; if (idx < P_PAD) ttot += KB(idx); }
    int incl_w = ttot;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, incl_w, d);
        if (lane >= d) incl_w += up;
    }
    if (lane == 31) warp_ex[wid] = incl_w;
    __syncthreads();
    if (wid == 0) {
        int w  = (tid < nwarp) ? warp_ex[tid] : 0;
        int wi = w;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, wi, d);
            if (lane >= d) wi += up;
        }
        if (tid < nwarp) warp_ex[tid] = wi - w;
    }
    __syncthreads();
    int off = warp_ex[wid] + (incl_w - ttot);
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) {
        int idx = base + j;
        if (idx < P_PAD) { off += KB(idx); KB(idx) = off; }
    }
    __syncthreads();
    const int cnt = KB(P_PAD - 1);

    if (tid == 0) page_cnt[(long)r * c_sr + kh] = cnt;

    for (int i = tid; i < n_pages; i += BLK) {
        int prev = (i > 0) ? KB(i - 1) : 0;
        const int ki = KB(i);
        if (ki - prev == 1)
            page_table[tbase + (ki - 1)] = i;
    }
#undef KB
    // PDL: release the dependent decode kernel as this grid drains.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

void nrmtopb_select_cuda(torch::Tensor score_h, torch::Tensor hmax,
                         torch::Tensor npg, torch::Tensor nsh,
                         torch::Tensor bfix, torch::Tensor score,
                         torch::Tensor page_table, torch::Tensor page_cnt,
                         int64_t n_req, int64_t n_kv, int64_t G,
                         int64_t n_sink, int64_t mp, int64_t P_PAD,
                         int64_t blk, int64_t has_hm, int64_t pdl) {
    TORCH_CHECK(score_h.scalar_type() == at::kFloat, "score_h must be fp32");
    TORCH_CHECK(hmax.scalar_type() == at::kInt, "hmax int32");
    TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
    TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table int32");
    TORCH_CHECK(page_cnt.scalar_type() == at::kInt, "page_cnt int32");
    TORCH_CHECK(G <= 8, "nrmtopb: G must be <= 8");
    const int BLK = (int)blk;
    // P13: kbuf is bank-conflict PADDED (one skip word per 32).
    const size_t smem = 256ul * sizeof(int)
                        + (size_t)P_PAD * sizeof(float)
                        + ((size_t)P_PAD + (size_t)P_PAD / 32 + 1)
                          * sizeof(int);
    // P13 vector width: consecutive-page float4/float2 loads (score_h rows,
    // score row, page_table -1 prefill) need every row stride VEC-aligned;
    // torch base pointers are 256 B aligned.  MP=1030 admits VEC=2.
    const long strides[7] = {score_h.stride(0), score_h.stride(1),
                             score_h.stride(2), score.stride(0),
                             score.stride(1), page_table.stride(0),
                             page_table.stride(1)};
    auto okvec = [&](int v) {
        for (int i = 0; i < 7; ++i) if (strides[i] % v) return false;
        return true;
    };
    const int vec = okvec(4) ? 4 : (okvec(2) ? 2 : 1);
    const int gt = (G <= 4) ? 4 : 8;   // template group bound (regs)
    void* kfn =
        (vec == 4) ? (gt == 4 ? (void*)nrmtopb_select_kernel<4, 4>
                              : (void*)nrmtopb_select_kernel<4, 8>)
      : (vec == 2) ? (gt == 4 ? (void*)nrmtopb_select_kernel<2, 4>
                              : (void*)nrmtopb_select_kernel<2, 8>)
                   : (gt == 4 ? (void*)nrmtopb_select_kernel<1, 4>
                              : (void*)nrmtopb_select_kernel<1, 8>);
    static int recorded_max[6] = {0, 0, 0, 0, 0, 0};
    const int ri = ((vec == 4) ? 2 : (vec == 2) ? 1 : 0) * 2 + (gt == 4 ? 0 : 1);
    if ((int)smem > recorded_max[ri]) {
        cudaFuncSetAttribute(kfn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        recorded_max[ri] = (int)smem;
    }
    dim3 grid((unsigned)n_req, (unsigned)n_kv);
    auto stream = at::cuda::getCurrentCUDAStream();
    unsigned* hm_ptr = reinterpret_cast<unsigned*>(hmax.data_ptr<int>());
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = dim3((unsigned)BLK);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        auto launch_pdl = [&](auto* fn) {
            cudaLaunchKernelEx(&cfg, fn,
                score_h.data_ptr<float>(), hm_ptr,
                npg.data_ptr<int>(), nsh.data_ptr<int>(), bfix.data_ptr<int>(),
                score.data_ptr<float>(), page_table.data_ptr<int>(),
                page_cnt.data_ptr<int>(),
                (int)n_sink, (int)n_kv, (int)G,
                (long)score_h.stride(0), (long)score_h.stride(1),
                (long)score_h.stride(2),
                (long)score.stride(0), (long)score.stride(1), (int)mp,
                (long)page_table.stride(0), (long)page_table.stride(1),
                (long)page_cnt.stride(0), (int)P_PAD, (int)has_hm);
        };
        if (vec == 4 && gt == 4)      launch_pdl(nrmtopb_select_kernel<4, 4>);
        else if (vec == 4)            launch_pdl(nrmtopb_select_kernel<4, 8>);
        else if (vec == 2 && gt == 4) launch_pdl(nrmtopb_select_kernel<2, 4>);
        else if (vec == 2)            launch_pdl(nrmtopb_select_kernel<2, 8>);
        else if (gt == 4)             launch_pdl(nrmtopb_select_kernel<1, 4>);
        else                          launch_pdl(nrmtopb_select_kernel<1, 8>);
    } else {
        auto launch_plain = [&](auto* fn) {
            fn<<<grid, BLK, smem, stream>>>(
                score_h.data_ptr<float>(), hm_ptr,
                npg.data_ptr<int>(), nsh.data_ptr<int>(), bfix.data_ptr<int>(),
                score.data_ptr<float>(), page_table.data_ptr<int>(),
                page_cnt.data_ptr<int>(),
                (int)n_sink, (int)n_kv, (int)G,
                (long)score_h.stride(0), (long)score_h.stride(1),
                (long)score_h.stride(2),
                (long)score.stride(0), (long)score.stride(1), (int)mp,
                (long)page_table.stride(0), (long)page_table.stride(1),
                (long)page_cnt.stride(0), (int)P_PAD, (int)has_hm);
        };
        if (vec == 4 && gt == 4)      launch_plain(nrmtopb_select_kernel<4, 4>);
        else if (vec == 4)            launch_plain(nrmtopb_select_kernel<4, 8>);
        else if (vec == 2 && gt == 4) launch_plain(nrmtopb_select_kernel<2, 4>);
        else if (vec == 2)            launch_plain(nrmtopb_select_kernel<2, 8>);
        else if (gt == 4)             launch_plain(nrmtopb_select_kernel<1, 4>);
        else                          launch_plain(nrmtopb_select_kernel<1, 8>);
    }
    cudaError_t le_ = cudaGetLastError();
    TORCH_CHECK(le_ == cudaSuccess, "nrmtopb launch failed: ",
                cudaGetErrorString(le_), " vec=", vec, " grid=(", grid.x,
                ",", grid.y, ") blk=", BLK, " smem=", smem);
}

// ======================================================================== //
// SELECT-V2 (lever 2, 2026-07-21): small-P bs=1 redesign of the fused
// nrm-combine + top-b.  HMAX-HANDSHAKE ONLY (has_hm=1 semantics: consume +
// reset st._nrmtopb_hmax).  Same grid (req, kv-head), one CTA per group,
// but the 4-digit full radix (3 extra full sc re-scans + ~20 barriers) is
// replaced by:
//   * a TOP-13-BIT histogram (256-bin byte level + 8192-bin fine level),
//     BOTH accumulated inside the combine loop (order-free integer adds);
//   * warp-0 crossing scans (byte, then the byte's 32 fine sub-bins) fix
//     13 bits in ONE pass with NO extra sweep; the 128-boundary bin at 13
//     bits (sign+exp+4 mantissa bits) holds few pages for real draws;
//   * the boundary bin resolved EXACTLY by one warp-ballot pass over the
//     collected members (<=32): full 32-bit keys, no more digit rounds;
//   * heavy-tie fallback (bin > 32): the deployed digit-1..3 radix rounds
//     VERBATIM over the sc cache (fires on adversarial tie bands only);
//   * DETERMINISTIC EXACT-B count-and-demote byte-identical to the
//     deployed kernel (ascending page-index demote in the smem cache,
//     quota = b - #(> tau); n_sink/tail always-keep; keep_all
//     short-circuit).  The ballot path derives #(>tau)/#(==tau)
//     analytically (equal fp32 values share every histogram bin, so all
//     equals live in the resolved bin); padded-tau early exits keep
//     exactly b by construction so the demote provably never fires there
//     (same as deployed); the deep path re-counts (deployed code).
//   * compaction via per-32-page warp ballots + one warp-0 chunk scan
//     (ascending page order preserved) instead of the smem ELS scan.
// Output contract: page_table AND page_cnt AND score byte-identical to
// nrmtopb_select_kernel has_hm=1 (gated over random draws + tie bands +
// all-equal + underflow-to-+0 at 16K/32K shapes:
// scratch_qgemv/probe_sel_v2.py).  Assumes NaN-free score_h (deployed
// assumption).  Env: LOCKS_SEL_V2=1 routes (default 0 = deployed kernel),
// LOCKS_SEL_V2_BLK block size (default 256).
// ======================================================================== //
#define V2_HB13 8192          // 13-bit fine histogram bins (f2u >> 19)
#define V2_BALLOT_MAX 32      // boundary-bin size the one-warp resolve takes

template <int VEC, int GT>
__global__ __launch_bounds__(MAXBLK) void nrmtopb_select_v2_kernel(
    const float* __restrict__ score_h,   // (R, n_kv, G, MP) per-head S
    unsigned*    __restrict__ hmax,      // (R, n_kv, 8) f2u page maxima
    const int*   __restrict__ npg,       // (R,)  n_pages
    const int*   __restrict__ nsh,       // (R,)  n_sel_hi
    const int*   __restrict__ bfix,      // (R,)  static b
    float*       __restrict__ score,     // (R, n_kv, MP)  written [0, nsh)
    int*         __restrict__ page_table,// (R, n_kv, MP)  int32
    int*         __restrict__ page_cnt,  // (R, n_kv)      int32
    const int  n_sink, const int n_kv, const int G,
    const long sh_sr, const long sh_sh, const long sh_sg,
    const long s_sr, const long s_sh,    // score strides (row, head)
    const int  mp,
    const long t_sr, const long t_sh,    // page_table strides (row, head)
    const long c_sr,                     // page_cnt row stride
    const int  P_PAD)
{
    const int r   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;

    const int n_pages  = npg[r];
    const int n_sel_hi = nsh[r];
    int       b        = bfix[r];
    const int n_sel    = n_sel_hi - n_sink;
    const bool keep_all = (n_sel <= 1);

    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;

    extern __shared__ unsigned char smem[];
    float*    sc     = reinterpret_cast<float*>(smem);              // P_PAD
    int*      hist   = reinterpret_cast<int*>(sc + P_PAD);          // 256
    int*      hist13 = hist + 256;                                  // V2_HB13
    unsigned* kmk    = reinterpret_cast<unsigned*>(hist13 + V2_HB13);
    int*      choff  = reinterpret_cast<int*>(kmk + (P_PAD >> 5));
    unsigned* cbuf   = reinterpret_cast<unsigned*>(choff + (P_PAD >> 5));
    __shared__ float hm_sh[8];
    __shared__ int s_sel, s_k, s_cb;          // crossing-scan broadcast
    __shared__ int s_path;                    // 0 PAD | 1 BALLOT | 2 DEEP
    __shared__ unsigned s_pfx, s_mmk, s_tau_u;
    __shared__ int s_bk, s_bcb;               // ballot rank / bin count
    __shared__ int s_mcnt, s_quota, s_demote;
    __shared__ int s_gt_eb, s_eq_eb;

    // PDL: start while the SCORE grid drains; fence before reading its
    // outputs (score_h + hmax).
    asm volatile("griddepcontrol.wait;" ::: "memory");
#ifdef LOCKS_FIXC
    // FIX-C (session-P lever 1): release PSS dependents at ENTRY.  The
    // deferred [kv_gemv_nf -> rope_and_cache_fixc] pair launches under
    // this 4-CTA kernel's window (128 idle SMs; session-P probe 4).  They
    // consume NOTHING this kernel writes; the plain decode split behind
    // them still waits on full completion of all three.  The deployed
    // drain-site trigger at the kernel tail becomes a no-op second fire.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

    const float* shp = score_h + (long)r * sh_sr + (long)kh * sh_sh;

    // hist + hist13 are adjacent int carves: one vectorized zero sweep
    // ((256 + V2_HB13) / 4 int4 stores; smem base and carve offsets are
    // 16B-aligned for pow2 P_PAD).
#ifdef LOCKS_SEL_PMERGE
    // ---- PMERGE (lever 3, 2026-07-22): barrier B1 DELETED via a hardware
    // producer/consumer named barrier (bar 1, count = BLK).  Warps 0-1
    // zero hist+hist13, consume+reset hmax into hm_sh, and init the
    // surviving s_* atomics, then bar.arrive (no wait); combine warps
    // (2..) bar.sync ONCE before touching hm_sh / the histograms, hidden
    // under their own score_h gmem-load latency.  Outputs bitwise: the
    // page->thread partition changes but per-page arithmetic (g-ascending
    // __expf adds), store addresses, and the order-free integer histogram
    // atomics are unchanged.  hm_sh + hmax read/reset stay on the SAME
    // threads as deployed (tid<8 = warp 0); no other warp touches hmax.
    // bar.arrive/bar.sync is the hardware split barrier, not a flag spin
    // (the 27r E4 race class is a different mechanism).  Host asserts
    // BLK >= 96 (2 producer warps + >= 1 combine warp).
    if (wid < 2) {
        int4* hz = reinterpret_cast<int4*>(hist);
        const int nz4 = (256 + V2_HB13) >> 2;
        const int4 z4 = make_int4(0, 0, 0, 0);
        for (int i = tid; i < nz4; i += 64) hz[i] = z4;
        if (tid < 8) {
            const long hidx = ((long)r * n_kv + kh) * 8 + tid;
            hm_sh[tid] = u2f(hmax[hidx]);
            hmax[hidx] = 0x007FFFFFu;    // f2u(-inf): rest-state invariant
        }
        if (tid == 0) { s_mcnt = 0; s_gt_eb = 0; s_eq_eb = 0; }
        __threadfence_block();
        asm volatile("bar.arrive 1, %0;" :: "r"(BLK) : "memory");
    } else {
        asm volatile("bar.sync 1, %0;" :: "r"(BLK) : "memory");
        float hm[GT];
        #pragma unroll
        for (int g = 0; g < GT; ++g) hm[g] = hm_sh[g];
        const int ct = tid - 64;
        const int CBLK = BLK - 64;
        const int nfull = (n_sel_hi / VEC) * VEC;
        for (int i = ct * VEC; i < nfull; i += CBLK * VEC) {
            float sv[GT][VEC];
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                if (g < G)
                    ldvec<VEC>(&sv[g][0], shp + (long)g * sh_sg + i);
            float out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float acc = 0.f;
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    if (g < G)
                        acc += __expf(sv[g][v] - hm[g]);
                out[v] = acc;
            }
            stvec<VEC>(score + sbase + i, out);
            stvec<VEC>(sc + i, out);
            #pragma unroll
            for (int v = 0; v < VEC; ++v)
                if (i + v >= n_sink) {
                    const unsigned u = f2u(out[v]);
                    atomicAdd(&hist[u >> 24], 1);
                    atomicAdd(&hist13[u >> 19], 1);
                }
        }
        for (int i = nfull + ct; i < P_PAD; i += CBLK) {
            float v = -CUDART_INF_F;
            if (i < n_sel_hi) {
                float sv[GT];
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    sv[g] = (g < G) ? shp[(long)g * sh_sg + i] : 0.f;
                float acc = 0.f;
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    if (g < G)
                        acc += __expf(sv[g] - hm[g]);
                score[sbase + i] = acc;
                v = acc;
                if (i >= n_sink) {
                    const unsigned u = f2u(acc);
                    atomicAdd(&hist[u >> 24], 1);
                    atomicAdd(&hist13[u >> 19], 1);
                }
            }
            sc[i] = v;
        }
    }
    __syncthreads();                                              // B2
#else
    {
        int4* hz = reinterpret_cast<int4*>(hist);
        const int nz4 = (256 + V2_HB13) >> 2;
        const int4 z4 = make_int4(0, 0, 0, 0);
        for (int i = tid; i < nz4; i += BLK) hz[i] = z4;
    }
    if (tid < 8) {
        const long hidx = ((long)r * n_kv + kh) * 8 + tid;
        hm_sh[tid] = u2f(hmax[hidx]);
        hmax[hidx] = 0x007FFFFFu;    // f2u(-inf): rest-state invariant
    }
    if (tid == 0) {
        s_path = 0; s_tau_u = 0u; s_mcnt = 0; s_demote = 0;
        s_gt_eb = 0; s_eq_eb = 0;
    }
    __syncthreads();                                              // B1

    float hm[GT];
    #pragma unroll
    for (int g = 0; g < GT; ++g) hm[g] = hm_sh[g];

    // ---- combine: nrm exp-sum -> score + sc cache (EXACT deployed
    // expressions: g-ascending __expf adds -> score bitwise-equal) + BOTH
    // histogram levels riding the same loop (order-free integer adds) -----
    {
        const int nfull = (n_sel_hi / VEC) * VEC;
        for (int i = tid * VEC; i < nfull; i += BLK * VEC) {
            float sv[GT][VEC];
            #pragma unroll
            for (int g = 0; g < GT; ++g)
                if (g < G)
                    ldvec<VEC>(&sv[g][0], shp + (long)g * sh_sg + i);
            float out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; ++v) {
                float acc = 0.f;
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    if (g < G)
                        acc += __expf(sv[g][v] - hm[g]);
                out[v] = acc;
            }
            stvec<VEC>(score + sbase + i, out);
            stvec<VEC>(sc + i, out);
            #pragma unroll
            for (int v = 0; v < VEC; ++v)
                if (i + v >= n_sink) {
                    const unsigned u = f2u(out[v]);
                    atomicAdd(&hist[u >> 24], 1);
                    atomicAdd(&hist13[u >> 19], 1);
                }
        }
        for (int i = nfull + tid; i < P_PAD; i += BLK) {
            float v = -CUDART_INF_F;
            if (i < n_sel_hi) {
                float sv[GT];
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    sv[g] = (g < G) ? shp[(long)g * sh_sg + i] : 0.f;
                float acc = 0.f;
                #pragma unroll
                for (int g = 0; g < GT; ++g)
                    if (g < G)
                        acc += __expf(sv[g] - hm[g]);
                score[sbase + i] = acc;
                v = acc;
                if (i >= n_sink) {
                    const unsigned u = f2u(acc);
                    atomicAdd(&hist[u >> 24], 1);
                    atomicAdd(&hist13[u >> 19], 1);
                }
            }
            sc[i] = v;
        }
    }
    __syncthreads();                                              // B2
#endif  // LOCKS_SEL_PMERGE (B1 site)

    if (b < 1)     b = 1;
    if (b > n_sel) b = n_sel;

#ifdef LOCKS_SEL_PMERGE
    // ---- PMERGE: barrier B3 DELETED.  EVERY warp runs the crossing scan
    // redundantly on the shared (stable, post-B2) histograms and derives
    // path/tau/pfx/mmk/bk/bcb in REGISTERS -- warp-local shfl/ballot code
    // identical to the deployed warp-0 scan, plus two extra shfl
    // broadcasts ((fabove, fcnt) at the crossing lane) so every lane
    // holds k1/cb1.  Deterministic: all warps read identical smem ->
    // identical registers; nothing is published, so the publish barrier
    // goes away.  The page_table -1 prefill (warps 1..) is unchanged and
    // now runs after the scan; B6 still orders it before the scatter.
    int      path = 0, sel0g = 0, k0g = 0, bk = 0, bcb = 0;
    unsigned tau_u = 0u, mmk = 0u, pfx = 0u;
    if (!keep_all) {
        int seg = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
        int suf = seg;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_down_sync(0xffffffffu, suf, d);
            if (lane + d < 32) suf += up;
        }
        int above = suf - seg;
        bool cross = (above < b) && (b <= suf);
        unsigned bal = __ballot_sync(0xffffffffu, cross);
        int cl = __ffs(bal) - 1;
        int sel0 = 0, k0 = 0, cb0 = 0;
        if (lane == cl) {
            int acc = above, sel = cl * 8;
            int cb  = 0;
            #pragma unroll
            for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                int c = hist[d];
                if (acc + c >= b) { sel = d; cb = c; break; }
                acc += c;
            }
            sel0 = sel; k0 = b - acc; cb0 = cb;
        }
        sel0 = __shfl_sync(0xffffffffu, sel0, cl);
        k0   = __shfl_sync(0xffffffffu, k0,   cl);
        cb0  = __shfl_sync(0xffffffffu, cb0,  cl);
        if (k0 == cb0) {
            // whole-bin boundary: byte-padded tau (deployed digit-0 early
            // exit; keep >= tau holds exactly b selectable, no demote).
            tau_u = ((unsigned)sel0) << 24;               // path 0
        } else {
            const int fcnt = hist13[sel0 * 32 + lane];
            int fsuf = fcnt;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int up = __shfl_down_sync(0xffffffffu, fsuf, d);
                if (lane + d < 32) fsuf += up;
            }
            const int fabove = fsuf - fcnt;
            const bool fcross = (fabove < k0) && (k0 <= fsuf);
            const unsigned fbal = __ballot_sync(0xffffffffu, fcross);
            const int fl = __ffs(fbal) - 1;
            const int fab = __shfl_sync(0xffffffffu, fabove, fl);
            const int cb1 = __shfl_sync(0xffffffffu, fcnt,   fl);
            const int k1  = k0 - fab;
            const unsigned pfx13 = ((unsigned)(sel0 * 32 + fl)) << 19;
            if (k1 == cb1) {              // 13-bit-padded tau: exact-b set
                tau_u = pfx13;                            // path 0
            } else if (cb1 <= V2_BALLOT_MAX) {
                path = 1; pfx = pfx13; mmk = 0xFFF80000u;
                bk = k1; bcb = cb1;
            } else {                      // heavy bin: deployed radix rounds
                path = 2; sel0g = sel0; k0g = k0;
            }
        }
    }
#else
    // ---- warp-0: byte crossing (deployed digit-0 scan) then fine crossing
    // over the byte's 32 sub-bins; ONE barrier publishes the path ----------
    if (wid == 0 && !keep_all) {
        int seg = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
        int suf = seg;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_down_sync(0xffffffffu, suf, d);
            if (lane + d < 32) suf += up;
        }
        int above = suf - seg;
        bool cross = (above < b) && (b <= suf);
        unsigned bal = __ballot_sync(0xffffffffu, cross);
        int cl = __ffs(bal) - 1;
        int sel0 = 0, k0 = 0, cb0 = 0;
        if (lane == cl) {
            int acc = above, sel = cl * 8;
            int cb  = 0;
            #pragma unroll
            for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                int c = hist[d];
                if (acc + c >= b) { sel = d; cb = c; break; }
                acc += c;
            }
            sel0 = sel; k0 = b - acc; cb0 = cb;
        }
        sel0 = __shfl_sync(0xffffffffu, sel0, cl);
        k0   = __shfl_sync(0xffffffffu, k0,   cl);
        cb0  = __shfl_sync(0xffffffffu, cb0,  cl);
        if (k0 == cb0) {
            // whole-bin boundary: byte-padded tau (deployed digit-0 early
            // exit; keep >= tau holds exactly b selectable, no demote).
            if (lane == 0) { s_path = 0; s_tau_u = ((unsigned)sel0) << 24; }
        } else {
            const int fcnt = hist13[sel0 * 32 + lane];
            int fsuf = fcnt;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int up = __shfl_down_sync(0xffffffffu, fsuf, d);
                if (lane + d < 32) fsuf += up;
            }
            const int fabove = fsuf - fcnt;
            const bool fcross = (fabove < k0) && (k0 <= fsuf);
            const unsigned fbal = __ballot_sync(0xffffffffu, fcross);
            const int fl = __ffs(fbal) - 1;
            if (lane == fl) {
                const int k1 = k0 - fabove, cb1 = fcnt;
                const unsigned pfx = ((unsigned)(sel0 * 32 + fl)) << 19;
                if (k1 == cb1) {          // 13-bit-padded tau: exact-b set
                    s_path = 0; s_tau_u = pfx;
                } else if (cb1 <= V2_BALLOT_MAX) {
                    s_path = 1; s_pfx = pfx; s_mmk = 0xFFF80000u;
                    s_bk = k1; s_bcb = cb1;
                } else {                  // heavy bin: deployed radix rounds
                    s_path = 2; s_sel = sel0; s_k = k0;
                }
            }
        }
    }
#endif  // LOCKS_SEL_PMERGE (B3 scan site)
    // -1 prefill of the full page_table row fills the warp-0-scan shadow
    // (all other warps are otherwise idle B2->B3; independent gmem target;
    // the scatter overwrites [0, cnt) after B7 -- final bytes unchanged).
    if (wid > 0 || keep_all) {
        const int t2 = tid - 32;                  // warps 1.. carry the fill
        const int B2K = BLK - 32;
        if (t2 >= 0) {
            int i = t2 * VEC;
            const int mpf = (mp / VEC) * VEC;
            for (; i < mpf; i += B2K * VEC) {
                if constexpr (VEC == 4)
                    *reinterpret_cast<int4*>(page_table + tbase + i) =
                        make_int4(-1, -1, -1, -1);
                else if constexpr (VEC == 2)
                    *reinterpret_cast<int2*>(page_table + tbase + i) =
                        make_int2(-1, -1);
                else
                    page_table[tbase + i] = -1;
            }
            for (i = mpf + t2; i < mp; i += B2K)
                page_table[tbase + i] = -1;
        }
    }
#ifdef LOCKS_SEL_PMERGE
    // PMERGE: no B3; tau from the redundant scan's register (u2f(0u) on
    // paths 1/2 exactly as the deployed s_tau_u=0 init; overwritten there).
    float tau = keep_all ? -CUDART_INF_F : u2f(tau_u);
#else
    __syncthreads();                                              // B3

    int      path = keep_all ? 0 : s_path;
    float    tau  = keep_all ? -CUDART_INF_F : u2f(s_tau_u);
    unsigned mmk = s_mmk, pfx = s_pfx;
    int      bk = s_bk, bcb = s_bcb;
#endif  // LOCKS_SEL_PMERGE (B3 site)

    // ---- heavy-tie fallback: deployed digit-1..3 rounds VERBATIM over the
    // sc cache; hands off to the ballot resolve when the candidate bin
    // shrinks under the warp size, or lands on the exact 32-bit tau --------
    if (path == 2) {
#ifdef LOCKS_SEL_PMERGE
        // PMERGE: seeds live in registers (every warp computed them); the
        // digit rounds below keep their deployed smem publishes+barriers.
        unsigned prefix = ((unsigned)sel0g) << 24;
        unsigned kmask  = 0xFF000000u;
        int      k      = k0g;
#else
        unsigned prefix = ((unsigned)s_sel) << 24;
        unsigned kmask  = 0xFF000000u;
        int      k      = s_k;
#endif  // LOCKS_SEL_PMERGE (path-2 seeds)
        bool     landed = true;
        for (int digit = 1; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            for (int i = tid; i < 256; i += BLK) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < n_sel; i += BLK) {
                unsigned u = f2u(sc[n_sink + i]);
                if ((u & kmask) == prefix)
                    atomicAdd(&hist[(u >> shift) & 0xFF], 1);
            }
            __syncthreads();
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;
                if (lane == cl) {
                    int acc = above, sel = cl * 8;
                    int cb  = 0;
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; cb = c; break; }
                        acc += c;
                    }
                    s_sel = sel;
                    s_k   = k - acc;
                    s_cb  = cb;
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift;
            k       = s_k;
            kmask  |= 0xFFu << shift;
            if (k == s_cb) {          // padded tau: exact-b, no demote
                landed = false; path = 0; tau = u2f(prefix);
                break;
            }
            if (s_cb <= V2_BALLOT_MAX && shift > 0) {
                landed = false; path = 1;
                pfx = prefix; mmk = kmask; bk = k; bcb = s_cb;
                break;
            }
            __syncthreads();
        }
        if (landed) {
            // all 32 bits fixed: exact tau; equal values may spill past
            // the quota -> deployed count-and-demote below.
            tau  = u2f(prefix);
            path = 3;
        }
    }

    // ---- boundary-bin ballot resolve: exact k-th largest among the <=32
    // members (full 32-bit keys); demote counts derived analytically -------
    if (path == 1) {
        for (int i = tid; i < n_sel; i += BLK) {
            unsigned u = f2u(sc[n_sink + i]);
            if ((u & mmk) == pfx) {
                int p = atomicAdd(&s_mcnt, 1);
                if (p < V2_BALLOT_MAX) cbuf[p] = u;
            }
        }
        __syncthreads();                                          // B4
#ifdef LOCKS_SEL_PMERGE
        // PMERGE: barrier B5 DELETED.  Every warp resolves the <=32-member
        // bin redundantly from cbuf (stable post-B4); tau/quota/demote are
        // functions of the member VALUE MULTISET only (cgt/ceq are value-
        // comparison counts, and equal-valued hit lanes carry the same
        // value), so any cbuf fill order gives identical results -- the
        // deployed fill order via atomicAdd was already nondeterministic.
        int quota_r, demote_r;
        {
            const unsigned ul = (lane < bcb) ? cbuf[lane] : 0u;
            int cgt = 0, ceq = 0;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const unsigned v = __shfl_sync(0xffffffffu, ul, j);
                if (j < bcb) { cgt += (v > ul); ceq += (v == ul); }
            }
            const bool hit = (lane < bcb) && (cgt < bk) && (bk <= cgt + ceq);
            const unsigned bal = __ballot_sync(0xffffffffu, hit);
            const int cl = __ffs(bal) - 1;
            tau_u = __shfl_sync(0xffffffffu, ul, cl);
            // #(score > tau) global = (b - bk) above the bin + cgt in
            // it; equals all live in the bin -> quota = bk - cgt.
            const int cgt_c = __shfl_sync(0xffffffffu, cgt, cl);
            const int ceq_c = __shfl_sync(0xffffffffu, ceq, cl);
            quota_r  = bk - cgt_c;
            demote_r = (ceq_c > quota_r) ? 1 : 0;
        }
        tau = u2f(tau_u);
        if (demote_r) {
            // DETERMINISTIC EXACT-B: ascending page-index demote of equals
            // beyond the quota, smem cache only (deployed walk verbatim;
            // demote_r/quota_r are warp-uniform AND warp-invariant, so the
            // branch and its barrier stay block-uniform).
            if (tid == 0) {
                int left = quota_r;
                for (int i = n_sink; i < n_sel_hi; ++i)
                    if (sc[i] == tau) {
                        if (left > 0) --left; else sc[i] = -CUDART_INF_F;
                    }
            }
            __syncthreads();
        }
#else
        if (wid == 0) {
            const unsigned ul = (lane < bcb) ? cbuf[lane] : 0u;
            int cgt = 0, ceq = 0;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                const unsigned v = __shfl_sync(0xffffffffu, ul, j);
                if (j < bcb) { cgt += (v > ul); ceq += (v == ul); }
            }
            const bool hit = (lane < bcb) && (cgt < bk) && (bk <= cgt + ceq);
            const unsigned bal = __ballot_sync(0xffffffffu, hit);
            const int cl = __ffs(bal) - 1;
            if (lane == cl) {
                s_tau_u = ul;
                // #(score > tau) global = (b - bk) above the bin + cgt in
                // it; equals all live in the bin -> quota = bk - cgt.
                const int quota = bk - cgt;
                s_quota  = quota;
                s_demote = (ceq > quota) ? 1 : 0;
            }
        }
        __syncthreads();                                          // B5
        tau = u2f(s_tau_u);
        if (s_demote) {
            // DETERMINISTIC EXACT-B: ascending page-index demote of equals
            // beyond the quota, smem cache only (deployed walk verbatim).
            if (tid == 0) {
                int left = s_quota;
                for (int i = n_sink; i < n_sel_hi; ++i)
                    if (sc[i] == tau) {
                        if (left > 0) --left; else sc[i] = -CUDART_INF_F;
                    }
            }
            __syncthreads();
        }
#endif  // LOCKS_SEL_PMERGE (B5 site)
    } else if (path == 3) {
        // deep landing: deployed count-and-demote VERBATIM.
        int gt_l = 0, eq_l = 0;
        for (int i = n_sink + tid; i < n_sel_hi; i += BLK) {
            const float v = sc[i];
            if (v > tau) ++gt_l;
            else if (v == tau) ++eq_l;
        }
        if (gt_l) atomicAdd(&s_gt_eb, gt_l);
        if (eq_l) atomicAdd(&s_eq_eb, eq_l);
        __syncthreads();
        if (tid == 0 && s_eq_eb > b - s_gt_eb) {
            int left = b - s_gt_eb;
            for (int i = n_sink; i < n_sel_hi; ++i)
                if (sc[i] == tau) {
                    if (left > 0) --left; else sc[i] = -CUDART_INF_F;
                }
        }
        __syncthreads();
    }

    // ---- keep + ballot compaction: per-32-page masks, one warp-0 chunk
    // scan, direct scatter (ascending page order == deployed bytes) --------
    const int nchunk = P_PAD >> 5;
    for (int c = wid; c < nchunk; c += nwarp) {
        const int p = (c << 5) + lane;
        bool keep = false;
        if (p < n_pages) {
            const bool always = (p < n_sink) || (p >= n_sel_hi);
            keep = always || keep_all || (sc[p] >= tau);
        }
        const unsigned m = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) kmk[c] = m;
    }
    __syncthreads();                                              // B6
#ifdef LOCKS_SEL_PMERGE
    // PMERGE: barrier B7 DELETED.  Every warp runs the deployed chunk scan
    // redundantly (identical kmk inputs post-B6 -> identical run/incl on
    // every warp) and stores choff for ALL chunks.  The choff stores are
    // duplicate aligned WORD writes of bit-identical values (final bytes
    // deterministic under any interleaving); each warp's scatter reads
    // back only values its own lanes wrote, ordered by __syncwarp -- no
    // cross-warp read-after-write remains, so no block barrier is needed.
    // page_cnt is written by warp 0 only (deployed line).
    {
        const int segc  = (nchunk + 31) >> 5;   // contiguous chunks/lane
        const int cbase = lane * segc;
        int tot = 0;
        for (int j = 0; j < segc; ++j) {
            const int c = cbase + j;
            if (c < nchunk) tot += __popc(kmk[c]);
        }
        int incl = tot;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += up;
        }
        int run = incl - tot;                   // exclusive base for lane
        for (int j = 0; j < segc; ++j) {
            const int c = cbase + j;
            if (c < nchunk) { choff[c] = run; run += __popc(kmk[c]); }
        }
        if (wid == 0 && lane == 31) page_cnt[(long)r * c_sr + kh] = incl;
    }
    __syncwarp();
#else
    if (wid == 0) {
        const int segc  = (nchunk + 31) >> 5;   // contiguous chunks/lane
        const int cbase = lane * segc;
        int tot = 0;
        for (int j = 0; j < segc; ++j) {
            const int c = cbase + j;
            if (c < nchunk) tot += __popc(kmk[c]);
        }
        int incl = tot;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, incl, d);
            if (lane >= d) incl += up;
        }
        int run = incl - tot;                   // exclusive base for lane
        for (int j = 0; j < segc; ++j) {
            const int c = cbase + j;
            if (c < nchunk) { choff[c] = run; run += __popc(kmk[c]); }
        }
        if (lane == 31) page_cnt[(long)r * c_sr + kh] = incl;
    }
    __syncthreads();                                              // B7
#endif  // LOCKS_SEL_PMERGE (B7 site)
    for (int c = wid; c < nchunk; c += nwarp) {
        const unsigned m = kmk[c];
        if ((m >> lane) & 1u) {
            const int pos = choff[c] + __popc(m & ((1u << lane) - 1u));
            page_table[tbase + pos] = (c << 5) + lane;
        }
    }
    // PDL: release the dependent decode kernel as this grid drains.
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

void nrmtopb_select_v2_cuda(torch::Tensor score_h, torch::Tensor hmax,
                            torch::Tensor npg, torch::Tensor nsh,
                            torch::Tensor bfix, torch::Tensor score,
                            torch::Tensor page_table, torch::Tensor page_cnt,
                            int64_t n_req, int64_t n_kv, int64_t G,
                            int64_t n_sink, int64_t mp, int64_t P_PAD,
                            int64_t blk, int64_t pdl) {
    TORCH_CHECK(score_h.scalar_type() == at::kFloat, "score_h must be fp32");
    TORCH_CHECK(hmax.scalar_type() == at::kInt, "hmax int32");
    TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
    TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table int32");
    TORCH_CHECK(page_cnt.scalar_type() == at::kInt, "page_cnt int32");
    TORCH_CHECK(G <= 8, "nrmtopb_v2: G must be <= 8");
    TORCH_CHECK(P_PAD >= 64 && (P_PAD & (P_PAD - 1)) == 0,
                "nrmtopb_v2: P_PAD must be a pow2 >= 64");
    const int BLK = (int)blk;
#ifdef LOCKS_SEL_PMERGE
    // PMERGE precondition (no-fallback rule): 2 producer warps + >= 1
    // combine warp.  Deployed blk is 512/1024; 96 is the hard floor.
    TORCH_CHECK(BLK >= 96, "LOCKS_SEL_PMERGE requires blk >= 96, got ", BLK);
#endif
    const size_t smem = (size_t)P_PAD * sizeof(float)             // sc
                        + 256ul * sizeof(int)                     // hist
                        + (size_t)V2_HB13 * sizeof(int)           // hist13
                        + 2ul * (size_t)(P_PAD >> 5) * sizeof(int)
                        + 32ul * sizeof(unsigned);                // cbuf
    const long strides[7] = {score_h.stride(0), score_h.stride(1),
                             score_h.stride(2), score.stride(0),
                             score.stride(1), page_table.stride(0),
                             page_table.stride(1)};
    auto okvec = [&](int v) {
        for (int i = 0; i < 7; ++i) if (strides[i] % v) return false;
        return true;
    };
    const int vec = okvec(4) ? 4 : (okvec(2) ? 2 : 1);
    const int gt = (G <= 4) ? 4 : 8;
    void* kfn =
        (vec == 4) ? (gt == 4 ? (void*)nrmtopb_select_v2_kernel<4, 4>
                              : (void*)nrmtopb_select_v2_kernel<4, 8>)
      : (vec == 2) ? (gt == 4 ? (void*)nrmtopb_select_v2_kernel<2, 4>
                              : (void*)nrmtopb_select_v2_kernel<2, 8>)
                   : (gt == 4 ? (void*)nrmtopb_select_v2_kernel<1, 4>
                              : (void*)nrmtopb_select_v2_kernel<1, 8>);
    static int recorded_max_v2[6] = {0, 0, 0, 0, 0, 0};
    const int ri = ((vec == 4) ? 2 : (vec == 2) ? 1 : 0) * 2 + (gt == 4 ? 0 : 1);
    if ((int)smem > recorded_max_v2[ri]) {
        cudaFuncSetAttribute(kfn,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        recorded_max_v2[ri] = (int)smem;
    }
    dim3 grid((unsigned)n_req, (unsigned)n_kv);
    auto stream = at::cuda::getCurrentCUDAStream();
    unsigned* hm_ptr = reinterpret_cast<unsigned*>(hmax.data_ptr<int>());
    if (pdl) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = dim3((unsigned)BLK);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        auto launch_pdl = [&](auto* fn) {
            cudaLaunchKernelEx(&cfg, fn,
                score_h.data_ptr<float>(), hm_ptr,
                npg.data_ptr<int>(), nsh.data_ptr<int>(), bfix.data_ptr<int>(),
                score.data_ptr<float>(), page_table.data_ptr<int>(),
                page_cnt.data_ptr<int>(),
                (int)n_sink, (int)n_kv, (int)G,
                (long)score_h.stride(0), (long)score_h.stride(1),
                (long)score_h.stride(2),
                (long)score.stride(0), (long)score.stride(1), (int)mp,
                (long)page_table.stride(0), (long)page_table.stride(1),
                (long)page_cnt.stride(0), (int)P_PAD);
        };
        if (vec == 4 && gt == 4)      launch_pdl(nrmtopb_select_v2_kernel<4, 4>);
        else if (vec == 4)            launch_pdl(nrmtopb_select_v2_kernel<4, 8>);
        else if (vec == 2 && gt == 4) launch_pdl(nrmtopb_select_v2_kernel<2, 4>);
        else if (vec == 2)            launch_pdl(nrmtopb_select_v2_kernel<2, 8>);
        else if (gt == 4)             launch_pdl(nrmtopb_select_v2_kernel<1, 4>);
        else                          launch_pdl(nrmtopb_select_v2_kernel<1, 8>);
    } else {
        auto launch_plain = [&](auto* fn) {
            fn<<<grid, BLK, smem, stream>>>(
                score_h.data_ptr<float>(), hm_ptr,
                npg.data_ptr<int>(), nsh.data_ptr<int>(), bfix.data_ptr<int>(),
                score.data_ptr<float>(), page_table.data_ptr<int>(),
                page_cnt.data_ptr<int>(),
                (int)n_sink, (int)n_kv, (int)G,
                (long)score_h.stride(0), (long)score_h.stride(1),
                (long)score_h.stride(2),
                (long)score.stride(0), (long)score.stride(1), (int)mp,
                (long)page_table.stride(0), (long)page_table.stride(1),
                (long)page_cnt.stride(0), (int)P_PAD);
        };
        if (vec == 4 && gt == 4)      launch_plain(nrmtopb_select_v2_kernel<4, 4>);
        else if (vec == 4)            launch_plain(nrmtopb_select_v2_kernel<4, 8>);
        else if (vec == 2 && gt == 4) launch_plain(nrmtopb_select_v2_kernel<2, 4>);
        else if (vec == 2)            launch_plain(nrmtopb_select_v2_kernel<2, 8>);
        else if (gt == 4)             launch_plain(nrmtopb_select_v2_kernel<1, 4>);
        else                          launch_plain(nrmtopb_select_v2_kernel<1, 8>);
    }
    cudaError_t le_ = cudaGetLastError();
    TORCH_CHECK(le_ == cudaSuccess, "nrmtopb_v2 launch failed: ",
                cudaGetErrorString(le_), " vec=", vec, " grid=(", grid.x,
                ",", grid.y, ") blk=", BLK, " smem=", smem);
}

// (nrmtopb_pp_kernel + launch + maxblocks, the sm_120-only cooperative
// page-parallel selector, removed 2026-07-21 with the sm_120 lane; see
// ours_doc/REFUTED_ARMS_INDEX.md, tag pre-cleanup-2026-07-21.)


// ======================================================================== //
// P11 SELECT-SPLIT (H200): 3-kernel non-cooperative split of the fused
// nrmtopb chain for LONG-CONTEXT low-batch, where the fused kernel's O(P)
// phases (has_hm=0 max sweep + nrm exp-sum + radix pass-0 scan) serialize
// inside 1 CTA/(req,kv-head) = 8 CTAs at bs1 (6% of an H200).  The sm_120
// cooperative pp kernel loses to ~11 grid.sync x ~1.3us (KERNEL_OPT_HANDOFF
// section 3); kernel BOUNDARIES are the sync here instead (2 extra launches,
// ~0.5-0.8us each in-graph).  The score kernel is UNTOUCHED (the hmax
// atomicMax handshake is a measured H200 pipe regression; the per-(g) maxima
// come from a page-parallel partial-max kernel instead).
//
// BITWISE CONTRACT (outputs identical to nrmtopb_select_kernel, has_hm=0):
//  * hm[g]: fp32 max is associative/commutative over any partition -> the
//    Z-partial reduce reproduces the sweep's exact per-(kv-head, group) max.
//  * score[i] = sum_g exp(score_h[g,i]-hm[g]), g ascending, per page in ONE
//    thread -> identical float sequence to the fused nrm.  sc in the finish
//    kernel re-reads the fp32 store (lossless round-trip).
//  * radix pass-0 histogram: per-page u32 bin counts accumulated by integer
//    atomicAdd (order-free) over the same page set [n_sink, n_sel_hi) ->
//    identical 256 counts; passes 1-3 + crossing scan + keep + compaction are
//    the fused kernel's text verbatim -> identical tau, page_table, page_cnt.
//  * hmax is neither read nor reset (the caller gates the split to the
//    has_hm=0 path, where the fused kernel does not touch it either).
// ======================================================================== //
__global__ void sel_pmax_kernel(
    const float* __restrict__ score_h,   // (R, n_kv, G, MP)
    const int*   __restrict__ nsh,       // (R,)
    float*       __restrict__ gpmax,     // (R*n_kv, 8, Z) chunk partial maxima
    int*         __restrict__ ghist,     // (R*n_kv, 256) pass-0 bins (zeroed)
    const int n_kv, const int G, const int Z, const int cw,
    const long sh_sr, const long sh_sh, const long sh_sg, const int P_PAD)
{
    const int z  = blockIdx.x;
    const int kh = blockIdx.y;
    const int r  = blockIdx.z;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;
    const int n_sel_hi = nsh[r];
    const int group = r * n_kv + kh;

    // zero this group's pass-0 histogram (striped over the group's Z CTAs)
    for (int i = z * BLK + tid; i < 256; i += Z * BLK)
        ghist[(long)group * 256 + i] = 0;

    const int c0 = z * cw, c1 = min(c0 + cw, P_PAD);
    const float* shp = score_h + (long)r * sh_sr + (long)kh * sh_sh;
    __shared__ float wmax[32][8];
    float lmax[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) lmax[g] = -CUDART_INF_F;
    for (int i = c0 + tid; i < c1; i += BLK) {
        if (i < n_sel_hi) {
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                if (g < G)
                    lmax[g] = fmaxf(lmax[g], shp[(long)g * sh_sg + i]);
        }
    }
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            lmax[g] = fmaxf(lmax[g], __shfl_xor_sync(~0u, lmax[g], o));
        if (lane == 0) wmax[wid][g] = lmax[g];
    }
    __syncthreads();
    if (wid == 0 && lane < 8) {
        float m = -CUDART_INF_F;
        for (int w = 0; w < nwarp; ++w) m = fmaxf(m, wmax[w][lane]);
        gpmax[((long)group * 8 + lane) * Z + z] = m;
    }
}

__global__ void sel_nrm_hist_kernel(
    const float* __restrict__ score_h,   // (R, n_kv, G, MP)
    const int*   __restrict__ nsh,       // (R,)
    const float* __restrict__ gpmax,     // (R*n_kv, 8, Zg) chunk partials
    float*       __restrict__ score,     // (R, n_kv, MP) written [0, nsh)
    int*         __restrict__ ghist,     // (R*n_kv, 256) pass-0 bins
    const int Zg,                        // gpmax slots (== Z unless PMAX FOLD
                                         //  wrote them at the score grid's z)
    const int n_sink, const int n_kv, const int G, const int Z, const int cw,
    const long sh_sr, const long sh_sh, const long sh_sg,
    const long s_sr, const long s_sh, const int P_PAD)
{
    const int z  = blockIdx.x;
    const int kh = blockIdx.y;
    const int r  = blockIdx.z;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int n_sel_hi = nsh[r];
    const int group = r * n_kv + kh;

    // reduce the Z chunk maxima -> hm[g] (warp w owns group g=w; order-free)
    __shared__ float hm_sh[8];
    if (wid < 8) {
        float m = -CUDART_INF_F;
        const float* gp = gpmax + ((long)group * 8 + wid) * Zg;
        for (int zz = lane; zz < Zg; zz += 32) m = fmaxf(m, gp[zz]);
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
        if (lane == 0) hm_sh[wid] = m;
    }
    __syncthreads();
    float hm[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) hm[g] = hm_sh[g];

    const long  sbase = (long)r * s_sr + (long)kh * s_sh;
    const float* shp = score_h + (long)r * sh_sr + (long)kh * sh_sh;
    int* gh = ghist + (long)group * 256;
    const int c0 = z * cw, c1 = min(c0 + cw, P_PAD);
    // Pass-2 Lever 2a: accumulate the pass-0 radix histogram in a PER-CTA smem
    // bank, then merge once (256 global atomics/CTA).  The old form did one
    // GLOBAL atomicAdd per page into a 4-group histogram -> hot-bin contention
    // (ncu: 67us at 2.7% DRAM / 5.4% SM, latency-bound).  Integer counts sum
    // order-free, so the merged gh[] is byte-identical (gated vs fused).
    __shared__ int lh[256];
    for (int i = tid; i < 256; i += BLK) lh[i] = 0;
    __syncthreads();
    for (int i = c0 + tid; i < c1; i += BLK) {
        if (i < n_sel_hi) {
            // identical float sequence to the fused nrm (OPT1 hoisted loads)
            float sv[8];
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                sv[g] = (g < G) ? shp[(long)g * sh_sg + i] : 0.f;
            float acc = 0.f;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                if (g < G)
                    acc += __expf(sv[g] - hm[g]);
            score[sbase + i] = acc;
            if (i >= n_sink)
                atomicAdd(&lh[f2u(acc) >> 24], 1);     // smem, per-CTA
        }
    }
    __syncthreads();
    for (int b = tid; b < 256; b += BLK)
        if (lh[b]) atomicAdd(&gh[b], lh[b]);           // one merge/CTA/bin
}

// FLAT-MASS combine (doc 18): score_h holds raw per-head MASSES (shifted
// exps) instead of LSE values; the exact combine weight is 1/pmax_g (the
// identity e^{K-hm} = 1/pmax). This kernel is sel_nrm_hist with the exp
// replaced by one FMA per element; hist semantics unchanged. gflag gets a
// nonzero on non-finite/non-positive pmax (K too small = overflow, or
// total underflow) -- checked host-side at run end (in-graph no branch).
__global__ void sel_nrm_flat_kernel(
    const float* __restrict__ score_h,
    const int*   __restrict__ nsh,
    const float* __restrict__ gpmax,
    float*       __restrict__ score,
    int*         __restrict__ ghist,
    int*         __restrict__ gflag,
    const int Zg,
    const int n_sink, const int n_kv, const int G, const int Z, const int cw,
    const long sh_sr, const long sh_sh, const long sh_sg,
    const long s_sr, const long s_sh, const int P_PAD)
{
    const int z  = blockIdx.x;
    const int kh = blockIdx.y;
    const int r  = blockIdx.z;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int n_sel_hi = nsh[r];
    const int group = r * n_kv + kh;

    __shared__ float hm_sh[8];
    if (wid < 8) {
        float m = -CUDART_INF_F;
        const float* gp = gpmax + ((long)group * 8 + wid) * Zg;
        for (int zz = lane; zz < Zg; zz += 32) m = fmaxf(m, gp[zz]);
        #pragma unroll
        for (int o = 16; o >= 1; o >>= 1)
            m = fmaxf(m, __shfl_xor_sync(~0u, m, o));
        if (lane == 0) {
            hm_sh[wid] = (m > 0.f && isfinite(m)) ? (1.f / m) : 0.f;
            if (!(m > 0.f && isfinite(m)) && wid < G && z == 0)
                atomicExch(gflag, 1);
        }
    }
    __syncthreads();
    float rp[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) rp[g] = hm_sh[g];

    const long  sbase = (long)r * s_sr + (long)kh * s_sh;
    const float* shp = score_h + (long)r * sh_sr + (long)kh * sh_sh;
    int* gh = ghist + (long)group * 256;
    __shared__ int lh[256];
    for (int i = tid; i < 256; i += BLK) lh[i] = 0;
    __syncthreads();
    const int c0 = z * cw, c1 = min(c0 + cw, P_PAD);
    for (int i = c0 + tid; i < c1; i += BLK) {
        if (i < n_sel_hi) {
            float sv[8];
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                sv[g] = (g < G) ? shp[(long)g * sh_sg + i] : 0.f;
            float acc = 0.f;
            #pragma unroll
            for (int g = 0; g < 8; ++g)
                if (g < G)
                    acc += sv[g] * rp[g];
            score[sbase + i] = acc;
            if (i >= n_sink)
                atomicAdd(&lh[f2u(acc) >> 24], 1);
        }
    }
    __syncthreads();
    for (int b = tid; b < 256; b += BLK)
        if (lh[b]) atomicAdd(&gh[b], lh[b]);
}

// FLAT K update (doc 18d): K'[i] = K[i] + log(pmax_i) + SLACK, from the
// gpmax buffer sel_pmax just filled. One tiny launch per layer per step.
__global__ void flat_k_update_kernel(
    float* __restrict__ K, const float* __restrict__ gpmax,
    const int Zg, const int n, const float slack)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float m = -CUDART_INF_F;
        const float* gp = gpmax + (long)i * Zg;
        for (int z = 0; z < Zg; ++z) m = fmaxf(m, gp[z]);
        if (m > 0.f && isfinite(m))
            K[i] = K[i] + __logf(m) + slack;
    }
}

__global__ void sel_finish_kernel(
    const int*   __restrict__ ghist,     // (R*n_kv, 256) pass-0 bins
    const int*   __restrict__ npg,       // (R,)
    const int*   __restrict__ nsh,       // (R,)
    const int*   __restrict__ bfix,      // (R,)
    const float* __restrict__ score,     // (R, n_kv, MP)
    int*         __restrict__ page_table,// (R, n_kv, MP)
    int*         __restrict__ page_cnt,  // (R, n_kv)
    const int  n_sink, const int n_kv,
    const long s_sr, const long s_sh, const int mp,
    const long t_sr, const long t_sh, const long c_sr, const int P_PAD)
{
    const int r   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;

    const int n_pages  = npg[r];
    const int n_sel_hi = nsh[r];
    int       b        = bfix[r];
    const int n_sel    = n_sel_hi - n_sink;
    const bool keep_all = (n_sel <= 1);

    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;

    extern __shared__ unsigned char smem[];
    float* sc   = reinterpret_cast<float*>(smem);            // P_PAD
    int*   hist = reinterpret_cast<int*>(sc + P_PAD);        // 256
    int*   kbuf = hist + 256;                                // P_PAD
    __shared__ int   warp_ex[32];
    __shared__ int   s_sel, s_k, s_cb;

    asm volatile("griddepcontrol.wait;" ::: "memory");

    // sc mirrors the fused kernel's cache: score for [0, n_sel_hi), else -inf
    // (fp32 store->load round-trip of the nrm acc is lossless).
    for (int i = tid; i < P_PAD; i += BLK)
        sc[i] = (i < n_sel_hi) ? score[sbase + i] : -CUDART_INF_F;
    __syncthreads();

    // ---- tau radix-select: fused text, pass 0 histogram PRECOMPUTED --------
    float tau = -CUDART_INF_F;
    if (!keep_all) {
        if (b < 1)     b = 1;
        if (b > n_sel) b = n_sel;
        unsigned prefix = 0u;
        unsigned kmask  = 0u;
        int      k      = b;
        const int* gh = ghist + ((long)r * n_kv + kh) * 256;
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            if (digit == 0) {
                for (int i = tid; i < 256; i += BLK) hist[i] = gh[i];
                __syncthreads();
            } else {
                for (int i = tid; i < 256; i += BLK) hist[i] = 0;
                __syncthreads();
                for (int i = tid; i < n_sel; i += BLK) {
                    unsigned u = f2u(sc[n_sink + i]);
                    if ((u & kmask) == prefix)
                        atomicAdd(&hist[(u >> shift) & 0xFF], 1);
                }
                __syncthreads();
            }
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;
                if (lane == cl) {
                    int acc = above, sel = cl * 8;
                    int cb  = 0;
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; cb = c; break; }
                        acc += c;
                    }
                    s_sel = sel;
                    s_k   = k - acc;
                    s_cb  = cb;
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift;
            k       = s_k;
            if (k == s_cb)
                break;
            kmask  |= 0xFFu << shift;
            __syncthreads();
        }
        tau = u2f(prefix);
    }

    // DETERMINISTIC EXACT-B (see topb_select_kernel: same count-and-demote
    // on the smem cache; keep test below unchanged).
    __shared__ int s_gt_eb, s_eq_eb;
    if (tid == 0) { s_gt_eb = 0; s_eq_eb = 0; }
    __syncthreads();
    if (!keep_all) {
        int gt_l = 0, eq_l = 0;
        for (int i = n_sink + tid; i < n_sel_hi; i += BLK) {
            const float v = sc[i];
            if (v > tau) ++gt_l;
            else if (v == tau) ++eq_l;
        }
        if (gt_l) atomicAdd(&s_gt_eb, gt_l);
        if (eq_l) atomicAdd(&s_eq_eb, eq_l);
    }
    __syncthreads();
    if (tid == 0 && !keep_all && s_eq_eb > b - s_gt_eb) {
        int left = b - s_gt_eb;
        for (int i = n_sink; i < n_sel_hi; ++i)
            if (sc[i] == tau) {
                if (left > 0) --left; else sc[i] = -CUDART_INF_F;
            }
    }
    __syncthreads();

    // ---- keep + scan + compact: fused kernel text VERBATIM -----------------
    for (int i = tid; i < P_PAD; i += BLK) {
        int keep = 0;
        if (i < n_pages) {
            bool always = (i < n_sink) || (i >= n_sel_hi);
            keep = (always || keep_all || sc[i] >= tau) ? 1 : 0;
        }
        kbuf[i] = keep;
    }
    __syncthreads();

    const int ELS  = (P_PAD + BLK - 1) / BLK;
    const int base = tid * ELS;
    int ttot = 0;
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) { int idx = base + j; if (idx < P_PAD) ttot += kbuf[idx]; }
    int incl_w = ttot;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, incl_w, d);
        if (lane >= d) incl_w += up;
    }
    if (lane == 31) warp_ex[wid] = incl_w;
    __syncthreads();
    if (wid == 0) {
        int w  = (tid < nwarp) ? warp_ex[tid] : 0;
        int wi = w;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, wi, d);
            if (lane >= d) wi += up;
        }
        if (tid < nwarp) warp_ex[tid] = wi - w;
    }
    __syncthreads();
    int off = warp_ex[wid] + (incl_w - ttot);
    #pragma unroll 4
    for (int j = 0; j < ELS; ++j) {
        int idx = base + j;
        if (idx < P_PAD) { off += kbuf[idx]; kbuf[idx] = off; }
    }
    __syncthreads();
    const int cnt = kbuf[P_PAD - 1];

    if (tid == 0) page_cnt[(long)r * c_sr + kh] = cnt;

    for (int i = cnt + tid; i < mp; i += BLK)
        page_table[tbase + i] = -1;
    for (int i = tid; i < n_pages; i += BLK) {
        int prev = (i > 0) ? kbuf[i - 1] : 0;
        if (kbuf[i] - prev == 1)
            page_table[tbase + (kbuf[i] - 1)] = i;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// ======================================================================== //
// GLOBAL-MEMORY FINISH (long-ctx >448K): sel_finish_kernel caches sc[P_PAD]
// + kbuf[P_PAD] in smem (1024 + 8*P_PAD bytes), which BLOWS the 227KB Hopper
// opt-in cap at P_PAD>=32768 (ctx>=256K) -- the fused nrmtopb kernel dies the
// same way ("nrmtopb launch failed: invalid argument smem=267268").  This
// finish keeps NOTHING page-indexed in smem: the radix re-reads `score` from
// GLOBAL (already written by sel_nrm_hist), and the keep flags / compaction
// ranks live in a GLOBAL kbuf scratch (R*n_kv*P_PAD int32).  smem is just
// hist[256] + warp_ex[32] + scalars (~1.3KB), so it launches at ANY P_PAD
// (validated to 1M).  Single CTA per (req,kv-head), same as sel_finish.
//
// BITWISE CONTRACT (identical page_table/page_cnt to sel_finish_kernel):
//  * tau: pass-0 uses the SAME precomputed ghist; passes 1-3 histogram the
//    SAME order-uint digits over the SAME selectable score range, just read
//    from global `score` (lossless fp32) instead of the smem sc mirror
//    (sc[j]==score[sbase+j] for j<n_sel_hi by construction) -> identical tau.
//  * keep(i) = always | keep_all | score[i]>=tau : same float compare.
//  * compaction: the SAME blocked 2-phase integer inclusive scan, kbuf in
//    global memory (order-free integer adds) -> identical ranks, ascending
//    page ids, -1 pad.  Grid/launch identical; only the memory SPACE moved.
// ======================================================================== //
__global__ void sel_finish_global_kernel(
    const int*   __restrict__ ghist,     // (R*n_kv, 256) pass-0 bins
    const int*   __restrict__ npg,       // (R,)
    const int*   __restrict__ nsh,       // (R,)
    const int*   __restrict__ bfix,      // (R,)
    const float* __restrict__ score,     // (R, n_kv, MP)  written [0, nsh)
    int*         __restrict__ gkbuf,     // (R*n_kv, P_PAD) global keep/rank
    int*         __restrict__ page_table,// (R, n_kv, MP)
    int*         __restrict__ page_cnt,  // (R, n_kv)
    const int  n_sink, const int n_kv,
    const long s_sr, const long s_sh, const int mp,
    const long t_sr, const long t_sh, const long c_sr, const int P_PAD)
{
    const int r   = blockIdx.x;
    const int kh  = blockIdx.y;
    const int tid = threadIdx.x;
    const int BLK = blockDim.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    const int nwarp = (BLK + 31) >> 5;

    const int n_pages  = npg[r];
    const int n_sel_hi = nsh[r];
    int       b        = bfix[r];
    const int n_sel    = n_sel_hi - n_sink;
    const bool keep_all = (n_sel <= 1);

    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;
    int* kb = gkbuf + ((long)r * n_kv + kh) * (long)P_PAD;

    __shared__ int hist[256];
    __shared__ int warp_ex[32];
    __shared__ int s_sel, s_k, s_cb;

    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- tau radix: pass0 from ghist(global), passes1-3 re-scan score(global)
    float tau = -CUDART_INF_F;
    if (!keep_all) {
        if (b < 1)     b = 1;
        if (b > n_sel) b = n_sel;
        unsigned prefix = 0u;
        unsigned kmask  = 0u;
        int      k      = b;
        const int* gh = ghist + ((long)r * n_kv + kh) * 256;
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            if (digit == 0) {
                for (int i = tid; i < 256; i += BLK) hist[i] = gh[i];
                __syncthreads();
            } else {
                for (int i = tid; i < 256; i += BLK) hist[i] = 0;
                __syncthreads();
                for (int i = tid; i < n_sel; i += BLK) {
                    unsigned u = f2u(score[sbase + n_sink + i]);
                    if ((u & kmask) == prefix)
                        atomicAdd(&hist[(u >> shift) & 0xFF], 1);
                }
                __syncthreads();
            }
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;
                if (lane == cl) {
                    int acc = above, sel = cl * 8;
                    int cb  = 0;
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; cb = c; break; }
                        acc += c;
                    }
                    s_sel = sel;
                    s_k   = k - acc;
                    s_cb  = cb;
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift;
            k       = s_k;
            if (k == s_cb)
                break;
            kmask  |= 0xFFu << shift;
            __syncthreads();
        }
        tau = u2f(prefix);
    }

    // DETERMINISTIC EXACT-B (large-P variant: score is GLOBAL, no smem
    // cache to demote -- use a demotion BITMAP instead, populated only on
    // the measure-zero fp32-collision overflow; the common path adds one
    // counting sweep and nothing else.  8KB static smem, P_PAD <= 65536.)
    __shared__ int s_gt_eb, s_eq_eb, s_ovf_eb;
    __shared__ unsigned ebm_eb[2048];
    if (tid == 0) { s_gt_eb = 0; s_eq_eb = 0; s_ovf_eb = 0; }
    __syncthreads();
    if (!keep_all) {
        int gt_l = 0, eq_l = 0;
        for (int i = n_sink + tid; i < n_sel_hi; i += BLK) {
            const float v = score[sbase + i];
            if (v > tau) ++gt_l;
            else if (v == tau) ++eq_l;
        }
        if (gt_l) atomicAdd(&s_gt_eb, gt_l);
        if (eq_l) atomicAdd(&s_eq_eb, eq_l);
    }
    __syncthreads();
    if (tid == 0 && !keep_all && s_eq_eb > b - s_gt_eb) s_ovf_eb = 1;
    __syncthreads();
    if (s_ovf_eb) {
        for (int w = tid; w < 2048; w += BLK) ebm_eb[w] = 0u;
        __syncthreads();
        if (tid == 0) {
            int left = b - s_gt_eb;
            for (int i = n_sink; i < n_sel_hi; ++i)
                if (score[sbase + i] == tau) {
                    if (left > 0) --left;
                    else ebm_eb[i >> 5] |= (1u << (i & 31));
                }
        }
        __syncthreads();
    }

    // ---- keep flags into global kb (reads global score) --------------------
    for (int i = tid; i < P_PAD; i += BLK) {
        int keep = 0;
        if (i < n_pages) {
            bool always = (i < n_sink) || (i >= n_sel_hi);
            float scv = (i < n_sel_hi) ? score[sbase + i] : -CUDART_INF_F;
            keep = (always || keep_all || scv >= tau) ? 1 : 0;
            if (s_ovf_eb && keep && !always && scv == tau
                && (ebm_eb[i >> 5] & (1u << (i & 31))))
                keep = 0;
        }
        kb[i] = keep;
    }
    __syncthreads();

    // ---- blocked 2-phase inclusive scan over global kb (== sel_finish) -----
    const int ELS  = (P_PAD + BLK - 1) / BLK;
    const int base = tid * ELS;
    int ttot = 0;
    for (int j = 0; j < ELS; ++j) { int idx = base + j; if (idx < P_PAD) ttot += kb[idx]; }
    int incl_w = ttot;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, incl_w, d);
        if (lane >= d) incl_w += up;
    }
    if (lane == 31) warp_ex[wid] = incl_w;
    __syncthreads();
    if (wid == 0) {
        int w  = (tid < nwarp) ? warp_ex[tid] : 0;
        int wi = w;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, wi, d);
            if (lane >= d) wi += up;
        }
        if (tid < nwarp) warp_ex[tid] = wi - w;
    }
    __syncthreads();
    int off = warp_ex[wid] + (incl_w - ttot);
    for (int j = 0; j < ELS; ++j) {
        int idx = base + j;
        if (idx < P_PAD) { off += kb[idx]; kb[idx] = off; }
    }
    __syncthreads();
    const int cnt = kb[P_PAD - 1];

    if (tid == 0) page_cnt[(long)r * c_sr + kh] = cnt;

    for (int i = cnt + tid; i < mp; i += BLK)
        page_table[tbase + i] = -1;
    for (int i = tid; i < n_pages; i += BLK) {
        int prev = (i > 0) ? kb[i - 1] : 0;
        if (kb[i] - prev == 1)
            page_table[tbase + (kb[i] - 1)] = i;
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// ======================================================================== //
// PAGE-PARALLEL FINISH (long-ctx, fills the machine at low batch).  The
// single-CTA sel_finish_global is CORRECT but grid-starved: 1 CTA/(req,kv) =
// 4 CTAs at bs1 serially scanning P_PAD=131072 pages (measured 1M sel 442us,
// 51% of the chain -> the bs=1 win SHRINKS with ctx).  This spreads the finish
// over Z chunk-CTAs (grid (Z, n_kv, R)) with KERNEL BOUNDARIES as the sync
// (no cooperative grid.sync -> graph-captures as plain launches on H200):
//   tau (1 CTA/grp, radix) -> keepscan (page-parallel: keep + PER-CHUNK
//   inclusive rank in smem + chunk kept-count + -1 prefill) -> offset (1
//   CTA/grp, exclusive scan of chunk counts) -> scatter (page-parallel).
// BITWISE: tau is the same radix; keep(i)=score[i]>=tau; the compaction is
// ascending (chunks contiguous & ascending, within-chunk ascending, chunk
// bases = exclusive prefix of per-chunk counts in page order) -> identical
// page_table/page_cnt to sel_finish (gated vs fused at 64K/128K).
// ======================================================================== //
__global__ void sel_tau_kernel(
    const int*   __restrict__ ghist,     // (R*n_kv, 256) pass-0 bins
    const int*   __restrict__ npg, const int* __restrict__ nsh,
    const int*   __restrict__ bfix,
    const float* __restrict__ score,     // (R, n_kv, MP)
    float*       __restrict__ gtau,      // (R*n_kv) out
    int*         __restrict__ gcut,      // (R*n_kv) out: exact-b tie cutoff
    const int n_sink, const int n_kv,
    const long s_sr, const long s_sh, const int P_PAD, const int exactb)
{
    const int r = blockIdx.x, kh = blockIdx.y, tid = threadIdx.x;
    const int BLK = blockDim.x, lane = tid & 31, wid = tid >> 5;
    const int n_sel_hi = nsh[r];
    int b = bfix[r];
    const int n_sel = n_sel_hi - n_sink;
    const bool keep_all = (n_sel <= 1);
    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    __shared__ int hist[256];
    __shared__ int s_sel, s_k, s_cb;
    __shared__ int s_run_eb, s_cut_eb, wsum_eb[32];
    int qeq = 0;              // >0: equals allowed at tau; 0: set below; -1: none
    float tau = -CUDART_INF_F;
    if (!keep_all) {
        if (b < 1) b = 1; if (b > n_sel) b = n_sel;
        unsigned prefix = 0u, kmask = 0u; int k = b;
        const int* gh = ghist + ((long)r * n_kv + kh) * 256;
        #pragma unroll
        for (int digit = 0; digit < 4; ++digit) {
            const int shift = 24 - 8 * digit;
            if (digit == 0) {
                for (int i = tid; i < 256; i += BLK) hist[i] = gh[i];
                __syncthreads();
            } else {
                for (int i = tid; i < 256; i += BLK) hist[i] = 0;
                __syncthreads();
                for (int i = tid; i < n_sel; i += BLK) {
                    unsigned u = f2u(score[sbase + n_sink + i]);
                    if ((u & kmask) == prefix)
                        atomicAdd(&hist[(u >> shift) & 0xFF], 1);
                }
                __syncthreads();
            }
            if (wid == 0) {
                int seg = 0;
                #pragma unroll
                for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
                int suf = seg;
                #pragma unroll
                for (int d = 1; d < 32; d <<= 1) {
                    int up = __shfl_down_sync(0xffffffffu, suf, d);
                    if (lane + d < 32) suf += up;
                }
                int above = suf - seg;
                bool cross = (above < k) && (k <= suf);
                unsigned bal = __ballot_sync(0xffffffffu, cross);
                int cl = __ffs(bal) - 1;
                if (lane == cl) {
                    int acc = above, sel = cl * 8, cb = 0;
                    #pragma unroll
                    for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                        int c = hist[d];
                        if (acc + c >= k) { sel = d; cb = c; break; }
                        acc += c;
                    }
                    s_sel = sel; s_k = k - acc; s_cb = cb;
                }
            }
            __syncthreads();
            prefix |= ((unsigned)s_sel) << shift; k = s_k;
            // EXACT-B: an early break means the boundary bin is fully taken
            // (count(>=tau) == b exactly), so no tie can overspend.  Falling
            // out of digit 3 means the b-th largest VALUE is tau and only
            // `k` of the equals may be kept -> tie cutoff below.
            if (k == s_cb) { qeq = -1; break; }
            kmask |= 0xFFu << shift; __syncthreads();
        }
        tau = u2f(prefix);
        if (qeq == 0) qeq = k;
    }
    int cut = LOCKS_NO_CUT;
    if (exactb && !keep_all && qeq > 0)
        cut = sel_tie_cutoff(score, sbase, n_sink, n_sel_hi, tau, qeq,
                             tid, BLK, &s_run_eb, &s_cut_eb, wsum_eb);
    if (tid == 0) {
        gtau[(long)r * n_kv + kh] = tau;
        gcut[(long)r * n_kv + kh] = cut;
    }
}

__global__ void sel_keepscan_kernel(
    const int*   __restrict__ npg, const int* __restrict__ nsh,
    const float* __restrict__ score, const float* __restrict__ gtau,
    const int*   __restrict__ gcut,      // (R*n_kv) exact-b tie cutoff
    int*         __restrict__ gkbuf,     // (R*n_kv, P_PAD) per-chunk ranks
    int*         __restrict__ gcnt,      // (R*n_kv, Z) per-chunk kept count
    int*         __restrict__ page_table,
    const int n_sink, const int n_kv, const int Z, const int cw,
    const long s_sr, const long s_sh, const int mp,
    const long t_sr, const long t_sh, const int P_PAD)
{
    const int z = blockIdx.x, kh = blockIdx.y, r = blockIdx.z;
    const int group = r * n_kv + kh;
    const int tid = threadIdx.x, BLK = blockDim.x;
    const int lane = tid & 31, wid = tid >> 5, nwarp = (BLK + 31) >> 5;
    const int n_pages = npg[r], n_sel_hi = nsh[r];
    const int n_sel = n_sel_hi - n_sink;
    const bool keep_all = (n_sel <= 1);
    const float tau = gtau[group];
    // EXACT-B: `>= tau` alone keeps a SUPERSET when scores tie at tau (the
    // 2026-07-22 defect).  gcut is the page index of the last equal that the
    // quota admits (LOCKS_NO_CUT = no overflow / kill switch, -1 = quota 0).
    const int cut = gcut[group];
    const int c0 = z * cw, c1 = min(c0 + cw, P_PAD);
    const int clen = c1 - c0;
    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;
    int* kb = gkbuf + (long)group * (long)P_PAD;
    extern __shared__ int ksm[];         // cw ints
    __shared__ int warp_ex[32];
    for (int li = tid; li < cw; li += BLK) {
        int keep = 0;
        const int i = c0 + li;
        if (li < clen && i < n_pages) {
            bool always = (i < n_sink) || (i >= n_sel_hi);
            float scv = (i < n_sel_hi) ? score[sbase + i] : -CUDART_INF_F;
            keep = (always || keep_all || scv > tau
                    || (scv == tau && i <= cut)) ? 1 : 0;
        }
        ksm[li] = keep;
        if (li < clen && i < mp) page_table[tbase + i] = -1;   // -1 prefill
    }
    __syncthreads();
    // blocked 2-phase inclusive scan over ksm[cw] (per-chunk, starts at 0)
    const int ELS = (cw + BLK - 1) / BLK, base = tid * ELS;
    int ttot = 0;
    for (int j = 0; j < ELS; ++j) { int idx = base + j; if (idx < cw) ttot += ksm[idx]; }
    int incl_w = ttot;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_up_sync(0xffffffffu, incl_w, d);
        if (lane >= d) incl_w += up;
    }
    if (lane == 31) warp_ex[wid] = incl_w;
    __syncthreads();
    if (wid == 0) {
        int w = (tid < nwarp) ? warp_ex[tid] : 0, wi = w;
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            int up = __shfl_up_sync(0xffffffffu, wi, d);
            if (lane >= d) wi += up;
        }
        if (tid < nwarp) warp_ex[tid] = wi - w;
    }
    __syncthreads();
    int off = warp_ex[wid] + (incl_w - ttot);
    for (int j = 0; j < ELS; ++j) {
        int idx = base + j;
        if (idx < cw) { off += ksm[idx]; ksm[idx] = off; }   // inclusive rank
    }
    __syncthreads();
    for (int li = tid; li < clen; li += BLK) kb[c0 + li] = ksm[li];
    if (tid == 0) gcnt[group * Z + z] = (clen > 0) ? ksm[clen - 1] : 0;
}

__global__ void sel_offset_kernel(
    const int* __restrict__ gcnt, int* __restrict__ goff,
    int* __restrict__ page_cnt, const int n_kv, const int Z, const long c_sr)
{
    const int r = blockIdx.x, kh = blockIdx.y, group = r * n_kv + kh;
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int zz = 0; zz < Z; ++zz) {
            goff[group * Z + zz] = acc;
            acc += gcnt[group * Z + zz];
        }
        page_cnt[(long)r * c_sr + kh] = acc;
    }
}

__global__ void sel_scatter_kernel(
    const int* __restrict__ npg, const int* __restrict__ gkbuf,
    const int* __restrict__ goff, int* __restrict__ page_table,
    const int n_kv, const int Z, const int cw,
    const long t_sr, const long t_sh, const int P_PAD)
{
    const int z = blockIdx.x, kh = blockIdx.y, r = blockIdx.z;
    const int group = r * n_kv + kh;
    const int tid = threadIdx.x, BLK = blockDim.x;
    const int n_pages = npg[r];
    const int c0 = z * cw, c1 = min(c0 + cw, P_PAD), clen = c1 - c0;
    const long tbase = (long)r * t_sr + (long)kh * t_sh;
    const int* kb = gkbuf + (long)group * (long)P_PAD;
    const int cbase = goff[group * Z + z];
    for (int li = tid; li < clen; li += BLK) {
        const int i = c0 + li;
        if (i < n_pages) {
            int prev = (li > 0) ? kb[c0 + li - 1] : 0;   // per-chunk local rank
            int rank = kb[c0 + li];
            if (rank - prev == 1)
                page_table[tbase + cbase + (rank - 1)] = i;
        }
    }
}

// (The PARALLEL sel_tau (tau_pp), WF1 (sel_wf) and WF2 (sel_wf2) fused arms
// were removed 2026-07-21, cleanup Wave 3a: tau_pp measured 2.3x worse than
// ship at 256K; WF1/WF2 kernel-win did not survive E2E.  Mechanisms in
// ours_doc/REFUTED_ARMS_INDEX.md; recover from tag pre-cleanup-2026-07-21.)
// ======================================================================== //
// TAUC (opt-in LOCKS_SEL_TAUC=1): tau from the EXISTING pass-0 ghist + ONE
// page-parallel candidate compaction + a tiny single-CTA finish.
// The shipped single-CTA sel_tau re-scans ALL of score[] from global up to
// 3x on a (R, n_kv) grid (ncu h8selncu 2026-07-16 @1M: 46.9us of the 83.5us
// select chain, grid=4 CTAs, dram 0.5%, long_scoreboard 20.8 -- the TOP
// artificial bottleneck of the whole decode chain). TAUC: digit 0 needs NO
// scan (ghist IS the digit-0 histogram); ONE page-parallel scan compacts the
// boundary-bin candidates (avg n_sel/256) into gkbuf, whose keepscan
// lifetime starts strictly later in-stream; digits 1-3 refine over the
// compacted values only. BYTE-IDENTICAL tau: same radix, integer bin counts
// are order-free, and the compact buffer ORDER is irrelevant (only counts
// are consumed). gstate[group*4] = {prefix, cand_count(atomic), k, done}.
// FINFIX pick0: verbatim pick0 plus the ghist2 zeroing prologue (256 ints
// per group over the warp) so the compact_fx digit-1 histogram needs no
// separate memset launch.
__global__ void sel_tauc_pick0_fx_kernel(
    const int* __restrict__ nsh, const int* __restrict__ bfix,
    const int* __restrict__ ghist, int* __restrict__ gstate,
    float* __restrict__ gtau, int* __restrict__ gcut,
    int* __restrict__ ghist2,
    int n_sink, int n_kv) {
    const int r = blockIdx.x, kh = blockIdx.y, lane = threadIdx.x & 31;
    const int group = r * n_kv + kh;
    for (int i = lane; i < 256; i += 32) ghist2[(long)group * 256 + i] = 0;
    const int n_sel = nsh[r] - n_sink;
    if (n_sel <= 1) {
        if (lane == 0) { gtau[group] = -CUDART_INF_F;
                         gcut[group] = LOCKS_NO_CUT;
                         gstate[group * 4 + 3] = 1; }
        return;
    }
    int b = bfix[r]; if (b < 1) b = 1; if (b > n_sel) b = n_sel;
    const int* h = ghist + (long)group * 256;
    int seg = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) seg += h[lane * 8 + j];
    int suf = seg;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_down_sync(0xffffffffu, suf, d);
        if (lane + d < 32) suf += up;
    }
    const int above = suf - seg;
    const bool cross = (above < b) && (b <= suf);
    const unsigned bal = __ballot_sync(0xffffffffu, cross);
    const int cl = __ffs(bal) - 1;
    int sel_l = 0, nk_l = 0, cb_l = 0;
    if (lane == cl) {
        int acc = above, sel = cl * 8, cb = 0;
        #pragma unroll
        for (int d = cl * 8 + 7; d >= cl * 8; --d) {
            const int c = h[d];
            if (acc + c >= b) { sel = d; cb = c; break; }
            acc += c;
        }
        sel_l = sel; nk_l = b - acc; cb_l = cb;
    }
    const int fsel = __shfl_sync(0xffffffffu, sel_l, cl);
    const int fnk  = __shfl_sync(0xffffffffu, nk_l, cl);
    const int fcb  = __shfl_sync(0xffffffffu, cb_l, cl);
    if (lane == 0) {
        const unsigned pref = (unsigned)fsel << 24;
        gstate[group * 4 + 0] = (int)pref;
        gstate[group * 4 + 1] = 0;
        gstate[group * 4 + 2] = fnk;
        const int done = (fnk == fcb) ? 1 : 0;
        gstate[group * 4 + 3] = done;
        // EXACT-B: a digit-0 stop takes the whole boundary byte-bin, so
        // count(>=tau) == b exactly and no tie can overspend.
        if (done) { gtau[group] = u2f(pref); gcut[group] = LOCKS_NO_CUT; }
    }
}
__global__ void sel_tauc_pick0_kernel(
    const int* __restrict__ nsh, const int* __restrict__ bfix,
    const int* __restrict__ ghist, int* __restrict__ gstate,
    float* __restrict__ gtau, int* __restrict__ gcut, int n_sink, int n_kv) {
    const int r = blockIdx.x, kh = blockIdx.y, lane = threadIdx.x & 31;
    const int group = r * n_kv + kh;
    const int n_sel = nsh[r] - n_sink;
    if (n_sel <= 1) {
        if (lane == 0) { gtau[group] = -CUDART_INF_F;
                         gcut[group] = LOCKS_NO_CUT;
                         gstate[group * 4 + 3] = 1; }
        return;
    }
    int b = bfix[r]; if (b < 1) b = 1; if (b > n_sel) b = n_sel;
    const int* h = ghist + (long)group * 256;
    int seg = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) seg += h[lane * 8 + j];
    int suf = seg;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int up = __shfl_down_sync(0xffffffffu, suf, d);
        if (lane + d < 32) suf += up;
    }
    const int above = suf - seg;
    const bool cross = (above < b) && (b <= suf);
    const unsigned bal = __ballot_sync(0xffffffffu, cross);
    const int cl = __ffs(bal) - 1;
    int sel_l = 0, nk_l = 0, cb_l = 0;
    if (lane == cl) {
        int acc = above, sel = cl * 8, cb = 0;
        #pragma unroll
        for (int d = cl * 8 + 7; d >= cl * 8; --d) {
            const int c = h[d];
            if (acc + c >= b) { sel = d; cb = c; break; }
            acc += c;
        }
        sel_l = sel; nk_l = b - acc; cb_l = cb;
    }
    const int fsel = __shfl_sync(0xffffffffu, sel_l, cl);
    const int fnk  = __shfl_sync(0xffffffffu, nk_l, cl);
    const int fcb  = __shfl_sync(0xffffffffu, cb_l, cl);
    if (lane == 0) {
        const unsigned pref = (unsigned)fsel << 24;
        gstate[group * 4 + 0] = (int)pref;
        gstate[group * 4 + 1] = 0;             // candidate counter
        gstate[group * 4 + 2] = fnk;
        const int done = (fnk == fcb) ? 1 : 0;
        gstate[group * 4 + 3] = done;
        if (done) { gtau[group] = u2f(pref); gcut[group] = LOCKS_NO_CUT; }
    }
}

// --- FINFIX (doc 27, 2026-07-18): de-serialized tauc tail ------------------
// compact_fx: verbatim candidate SET semantics; the per-hit global atomic
// becomes a warp-aggregated claim (gbuf order was already nondeterministic
// and is consumed as a set), and the accepted candidates' digit-1 bytes are
// histogrammed per-CTA in smem and merged into ghist2 (free scratch in this
// route) so finish skips its O(cnt) digit-1 pass entirely.
__global__ void sel_tauc_compact_fx_kernel(
    const int* __restrict__ nsh, const float* __restrict__ score,
    int* __restrict__ gstate, unsigned* __restrict__ gbuf,
    int* __restrict__ ghist2,
    int n_sink, int n_kv, int Z, int cw, long s_sr, long s_sh, int P_PAD) {
    const int z = blockIdx.x, kh = blockIdx.y, r = blockIdx.z;
    const int group = r * n_kv + kh;
    const int tid = threadIdx.x, BLK = blockDim.x;
    const int lane = tid & 31;
    const int n_sel = nsh[r] - n_sink;
    if (n_sel <= 1) return;
    if (gstate[group * 4 + 3]) return;                 // done at digit 0
    __shared__ int h1[256];
    for (int i = tid; i < 256; i += BLK) h1[i] = 0;
    __syncthreads();
    const unsigned top = (unsigned)gstate[group * 4 + 0] >> 24;
    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    unsigned* gb = gbuf + (long)group * P_PAD;
    const int c0 = z * cw, c1 = min(c0 + cw, n_sel);
    for (int ib = c0; ib < c1; ib += BLK) {
        const int i = ib + tid;
        unsigned u = 0u; bool hit = false;
        if (i < c1) {
            u = f2u(score[sbase + n_sink + i]);
            hit = ((u >> 24) == top);
        }
        const unsigned m = __ballot_sync(0xffffffffu, hit);
        if (m) {
            const int leader = __ffs(m) - 1;
            int base = 0;
            if (lane == leader)
                base = atomicAdd(&gstate[group * 4 + 1], __popc(m));
            base = __shfl_sync(0xffffffffu, base, leader);
            if (hit) {
                const int off = __popc(m & ((1u << lane) - 1u));
                gb[base + off] = u;
                atomicAdd(&h1[(u >> 16) & 0xFF], 1);
            }
        }
    }
    __syncthreads();
    for (int i = tid; i < 256; i += BLK)
        if (h1[i]) atomicAdd(&ghist2[(long)group * 256 + i], h1[i]);
}
// finish_fx: digit 1 reads the prebuilt ghist2 (the digit-1 filter
// (u & 0xFF000000) == prefix is a no-op over gbuf -- compact only admitted
// that top byte -- so ghist2's integer counts are EXACTLY the old pass-1
// histogram). Digits 2-3 keep the O(cnt) scans but the same-bin smem
// atomics are warp-aggregated (__match_any_sync leader-add): the L18/L22
// bin-concentration pathology serialized ~cnt adds in one CTA; aggregation
// divides the conflict chain by up to 32. Integer counts identical.
__global__ void sel_tauc_finish_fx_kernel(
    const int* __restrict__ nsh, const int* __restrict__ gstate,
    const unsigned* __restrict__ gbuf, const int* __restrict__ ghist2,
    float* __restrict__ gtau, int* __restrict__ gcut,
    const float* __restrict__ score, const long s_sr, const long s_sh,
    int n_sink, int n_kv, int P_PAD, int exactb) {
    const int r = blockIdx.x, kh = blockIdx.y;
    const int tid = threadIdx.x, BLK = blockDim.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int group = r * n_kv + kh;
    const int n_sel = nsh[r] - n_sink;
    if (n_sel <= 1) return;
    if (gstate[group * 4 + 3]) return;
    const int n_sel_hi = nsh[r];
    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const unsigned* gb = gbuf + (long)group * P_PAD;
    const int cnt = gstate[group * 4 + 1];
    unsigned prefix = (unsigned)gstate[group * 4 + 0];
    unsigned kmask  = 0xFF000000u;
    int k = gstate[group * 4 + 2];
    int qeq = 0;
    __shared__ int hist[256];
    __shared__ int s_sel, s_k, s_cb;
    __shared__ int s_run_eb, s_cut_eb, wsum_eb[32];
    #pragma unroll
    for (int digit = 1; digit < 4; ++digit) {
        const int shift = 24 - 8 * digit;
        if (digit == 1) {
            for (int i = tid; i < 256; i += BLK)
                hist[i] = ghist2[(long)group * 256 + i];
            __syncthreads();
        } else {
            for (int i = tid; i < 256; i += BLK) hist[i] = 0;
            __syncthreads();
            for (int i = tid; i < cnt; i += BLK) {
                const unsigned u = gb[i];
                if ((u & kmask) == prefix) {
                    const int d = (int)((u >> shift) & 0xFF);
                    const unsigned pm = __match_any_sync(__activemask(), d);
                    if (lane == (__ffs(pm) - 1))
                        atomicAdd(&hist[d], __popc(pm));
                }
            }
            __syncthreads();
        }
        if (wid == 0) {
            int seg = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
            int suf = seg;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int up = __shfl_down_sync(0xffffffffu, suf, d);
                if (lane + d < 32) suf += up;
            }
            const int above = suf - seg;
            const bool cross = (above < k) && (k <= suf);
            const unsigned bal = __ballot_sync(0xffffffffu, cross);
            const int cl = __ffs(bal) - 1;
            if (lane == cl) {
                int acc = above, sel = cl * 8, cb = 0;
                #pragma unroll
                for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                    const int c = hist[d];
                    if (acc + c >= k) { sel = d; cb = c; break; }
                    acc += c;
                }
                s_sel = sel; s_k = k - acc; s_cb = cb;
            }
        }
        __syncthreads();
        prefix |= ((unsigned)s_sel) << shift;
        k = s_k;
        if (k == s_cb) { qeq = -1; break; }
        kmask |= 0xFFu << shift;
        __syncthreads();
    }
    const float tauf = u2f(prefix);
    if (qeq == 0) qeq = k;
    int cut = LOCKS_NO_CUT;
    if (exactb && qeq > 0)
        cut = sel_tie_cutoff(score, sbase, n_sink, n_sel_hi, tauf, qeq,
                             tid, BLK, &s_run_eb, &s_cut_eb, wsum_eb);
    if (tid == 0) { gtau[group] = tauf; gcut[group] = cut; }
}
// flat_k_update_fx: identical max/log/slack values; the ONE-WARP serial
// Zg-deep sweep becomes one BLOCK per (group,g) item with a tree reduce
// (fp max is order-exact) -- the profile priced the old shape at 7.7us of
// pure single-warp latency for 24KB of reads.
__global__ void flat_k_update_fx_kernel(
    float* __restrict__ K, const float* __restrict__ gpmax,
    const int Zg, const int n, const float slack)
{
    const int i = blockIdx.x;
    if (i >= n) return;
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    __shared__ float wred[32];
    const float* gp = gpmax + (long)i * Zg;
    float m = -CUDART_INF_F;
    for (int z = tid; z < Zg; z += blockDim.x) m = fmaxf(m, gp[z]);
    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) wred[wid] = m;
    __syncthreads();
    if (tid == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        float mm = wred[0];
        for (int w = 1; w < nw; ++w) mm = fmaxf(mm, wred[w]);
        if (mm > 0.f && isfinite(mm)) K[i] = K[i] + __logf(mm) + slack;
    }
}

__global__ void sel_tauc_compact_kernel(
    const int* __restrict__ nsh, const float* __restrict__ score,
    int* __restrict__ gstate, unsigned* __restrict__ gbuf,
    int n_sink, int n_kv, int Z, int cw, long s_sr, long s_sh, int P_PAD) {
    const int z = blockIdx.x, kh = blockIdx.y, r = blockIdx.z;
    const int group = r * n_kv + kh;
    const int tid = threadIdx.x, BLK = blockDim.x;
    const int n_sel = nsh[r] - n_sink;
    if (n_sel <= 1) return;
    if (gstate[group * 4 + 3]) return;                 // done at digit 0
    const unsigned top = (unsigned)gstate[group * 4 + 0] >> 24;
    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    unsigned* gb = gbuf + (long)group * P_PAD;
    const int c0 = z * cw, c1 = min(c0 + cw, n_sel);
    for (int i = c0 + tid; i < c1; i += BLK) {
        const unsigned u = f2u(score[sbase + n_sink + i]);
        if ((u >> 24) == top)
            gb[atomicAdd(&gstate[group * 4 + 1], 1)] = u;
    }
}
__global__ void sel_tauc_finish_kernel(
    const int* __restrict__ nsh, const int* __restrict__ gstate,
    const unsigned* __restrict__ gbuf, float* __restrict__ gtau,
    int* __restrict__ gcut,
    const float* __restrict__ score, const long s_sr, const long s_sh,
    int n_sink, int n_kv, int P_PAD, int exactb) {
    const int r = blockIdx.x, kh = blockIdx.y;
    const int tid = threadIdx.x, BLK = blockDim.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int group = r * n_kv + kh;
    const int n_sel = nsh[r] - n_sink;
    if (n_sel <= 1) return;
    if (gstate[group * 4 + 3]) return;
    const int n_sel_hi = nsh[r];
    const long sbase = (long)r * s_sr + (long)kh * s_sh;
    const unsigned* gb = gbuf + (long)group * P_PAD;
    const int cnt = gstate[group * 4 + 1];             // == digit-0 bin count
    unsigned prefix = (unsigned)gstate[group * 4 + 0];
    unsigned kmask  = 0xFF000000u;
    int k = gstate[group * 4 + 2];
    int qeq = 0;
    __shared__ int hist[256];
    __shared__ int s_sel, s_k, s_cb;
    __shared__ int s_run_eb, s_cut_eb, wsum_eb[32];
    #pragma unroll
    for (int digit = 1; digit < 4; ++digit) {
        const int shift = 24 - 8 * digit;
        for (int i = tid; i < 256; i += BLK) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < cnt; i += BLK) {
            const unsigned u = gb[i];
            if ((u & kmask) == prefix)
                atomicAdd(&hist[(u >> shift) & 0xFF], 1);
        }
        __syncthreads();
        if (wid == 0) {
            int seg = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) seg += hist[lane * 8 + j];
            int suf = seg;
            #pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                int up = __shfl_down_sync(0xffffffffu, suf, d);
                if (lane + d < 32) suf += up;
            }
            const int above = suf - seg;
            const bool cross = (above < k) && (k <= suf);
            const unsigned bal = __ballot_sync(0xffffffffu, cross);
            const int cl = __ffs(bal) - 1;
            if (lane == cl) {
                int acc = above, sel = cl * 8, cb = 0;
                #pragma unroll
                for (int d = cl * 8 + 7; d >= cl * 8; --d) {
                    const int c = hist[d];
                    if (acc + c >= k) { sel = d; cb = c; break; }
                    acc += c;
                }
                s_sel = sel; s_k = k - acc; s_cb = cb;
            }
        }
        __syncthreads();
        prefix |= ((unsigned)s_sel) << shift;
        k = s_k;
        if (k == s_cb) { qeq = -1; break; }
        kmask |= 0xFFu << shift;
        __syncthreads();
    }
    const float tauf = u2f(prefix);
    if (qeq == 0) qeq = k;
    int cut = LOCKS_NO_CUT;
    if (exactb && qeq > 0)
        cut = sel_tie_cutoff(score, sbase, n_sink, n_sel_hi, tauf, qeq,
                             tid, BLK, &s_run_eb, &s_cut_eb, wsum_eb);
    if (tid == 0) { gtau[group] = tauf; gcut[group] = cut; }
}

// --- FIELD AUDIT (instrument, LOCKS_CNT_AUDIT=1; no launch when off) ------ //
// Accumulates the page_cnt-vs-invariant distribution over every (step, layer,
// request, kv-head) unit the graph replays.  acc slots:
//   0 units | 1 violations (cnt > inv) | 2 max cnt | 3 sum cnt | 4 sum inv
//   5 max excess | 6 sum excess | 7..11 excess histogram {0, 1-8, 9-64,
//   65-512, >512} | 12 keep_all units skipped (cudagraph-warmup dummies,
//   n_sel<=1, where page_cnt == n_pages BY DESIGN).  Read host-side by
//   _runtime.cnt_audit_step.
__global__ void sel_cnt_audit_kernel(
    const int* __restrict__ npg, const int* __restrict__ nsh,
    const int* __restrict__ bfix, const int* __restrict__ page_cnt,
    unsigned long long* __restrict__ acc,
    const int n_sink, const int n_kv, const long c_sr)
{
    if (threadIdx.x != 0) return;
    const int r = blockIdx.x, kh = blockIdx.y;
    const int n_pages = npg[r], n_sel_hi = nsh[r];
    const int n_sel = n_sel_hi - n_sink;
    int b = bfix[r];
    if (n_sel <= 1) {          // keep_all (cudagraph-warmup dummy states)
        atomicAdd(&acc[12], 1ull);
        return;
    }
    if (b < 1) b = 1;
    if (b > n_sel) b = n_sel;
    const int inv = min(n_sink, n_pages) + b + (n_pages - n_sel_hi);
    const int c = page_cnt[(long)r * c_sr + kh];
    const int e = c - inv;
    atomicAdd(&acc[0], 1ull);
    atomicAdd(&acc[1], (unsigned long long)(e > 0 ? 1 : 0));
    atomicMax(&acc[2], (unsigned long long)max(c, 0));
    atomicAdd(&acc[3], (unsigned long long)max(c, 0));
    atomicAdd(&acc[4], (unsigned long long)inv);
    atomicMax(&acc[5], (unsigned long long)max(e, 0));
    atomicAdd(&acc[6], (unsigned long long)max(e, 0));
    const int bkt = (e <= 0) ? 7 : (e <= 8) ? 8 : (e <= 64) ? 9
                                : (e <= 512) ? 10 : 11;
    atomicAdd(&acc[bkt], 1ull);
}

void sel_cnt_audit(torch::Tensor npg, torch::Tensor nsh, torch::Tensor bfix,
                   torch::Tensor page_cnt, torch::Tensor acc,
                   int64_t n_req, int64_t n_kv, int64_t n_sink) {
    TORCH_CHECK(acc.scalar_type() == at::kLong && acc.numel() >= 16,
                "sel_cnt_audit: acc must be int64[>=16]");
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g((unsigned)n_req, (unsigned)n_kv);
    sel_cnt_audit_kernel<<<g, 32, 0, stream>>>(
        npg.data_ptr<int>(), nsh.data_ptr<int>(), bfix.data_ptr<int>(),
        page_cnt.data_ptr<int>(),
        reinterpret_cast<unsigned long long*>(acc.data_ptr<int64_t>()),
        (int)n_sink, (int)n_kv, (long)page_cnt.stride(0));
}

void sel_split_cuda(torch::Tensor score_h, torch::Tensor npg,
                    torch::Tensor nsh, torch::Tensor bfix,
                    torch::Tensor score, torch::Tensor page_table,
                    torch::Tensor page_cnt, torch::Tensor gpmax,
                    torch::Tensor ghist, torch::Tensor gkbuf,
                    torch::Tensor gtau, torch::Tensor gcut,
                    torch::Tensor gcnt, torch::Tensor goff,
                    int64_t n_req, int64_t n_kv, int64_t G, int64_t n_sink,
                    int64_t mp, int64_t P_PAD, int64_t Z, int64_t wblk,
                    int64_t nblk, int64_t global_finish, int64_t Z_fin,
                    torch::Tensor gstate, torch::Tensor ghist2,
                    int64_t tau_pp,
                    int64_t pmax_done, int64_t Zg,
                    int64_t flat, torch::Tensor gflag, int64_t finfix,
                    int64_t exactb) {
    TORCH_CHECK(score_h.scalar_type() == at::kFloat, "score_h must be fp32");
    TORCH_CHECK(gcut.scalar_type() == at::kInt
                && gcut.numel() >= n_req * n_kv, "gcut int32 scratch small");
    TORCH_CHECK(score.scalar_type() == at::kFloat, "score must be fp32");
    TORCH_CHECK(page_table.scalar_type() == at::kInt, "page_table int32");
    TORCH_CHECK(page_cnt.scalar_type() == at::kInt, "page_cnt int32");
    TORCH_CHECK(G <= 8, "sel_split: G must be <= 8");
    TORCH_CHECK(wblk >= 256, "sel_split: wide BLK must cover 8 hm warps");
    TORCH_CHECK(gpmax.numel() >= n_req * n_kv * 8 * Z, "gpmax scratch small");
    TORCH_CHECK(ghist.numel() >= n_req * n_kv * 256, "ghist scratch small");
    const int cw = (int)((P_PAD + Z - 1) / Z);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 wgrid((unsigned)Z, (unsigned)n_kv, (unsigned)n_req);
    dim3 wblock((unsigned)wblk);
    // PMAX FOLD: the score kernel already wrote gpmax (at its own grid z =
    // Zg) and zeroed ghist; sel_pmax is skipped entirely.
    if (!pmax_done)
    sel_pmax_kernel<<<wgrid, wblock, 0, stream>>>(
        score_h.data_ptr<float>(), nsh.data_ptr<int>(),
        gpmax.data_ptr<float>(), ghist.data_ptr<int>(),
        (int)n_kv, (int)G, (int)Z, cw,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2), (int)P_PAD);
    if (flat) {
    // FLAT combine (doc 18): masses + 1/pmax weights; the exp pass is gone.
    sel_nrm_flat_kernel<<<wgrid, wblock, 0, stream>>>(
        score_h.data_ptr<float>(), nsh.data_ptr<int>(),
        gpmax.data_ptr<float>(), score.data_ptr<float>(),
        ghist.data_ptr<int>(), gflag.data_ptr<int>(),
        (int)(pmax_done ? Zg : Z),
        (int)n_sink, (int)n_kv, (int)G, (int)Z, cw,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2),
        (long)score.stride(0), (long)score.stride(1), (int)P_PAD);
    } else {
    sel_nrm_hist_kernel<<<wgrid, wblock, 0, stream>>>(
        score_h.data_ptr<float>(), nsh.data_ptr<int>(),
        gpmax.data_ptr<float>(), score.data_ptr<float>(),
        ghist.data_ptr<int>(),
        (int)(pmax_done ? Zg : Z),
        (int)n_sink, (int)n_kv, (int)G, (int)Z, cw,
        (long)score_h.stride(0), (long)score_h.stride(1),
        (long)score_h.stride(2),
        (long)score.stride(0), (long)score.stride(1), (int)P_PAD);
    }
    dim3 ngrid((unsigned)n_req, (unsigned)n_kv);
    if (global_finish) {
        // long-ctx PAGE-PARALLEL finish: the smem sc/kbuf caches would blow the
        // cap AND the single-CTA global finish is grid-starved.  Spread over
        // Z_fin chunk-CTAs (grid (Z_fin,n_kv,R)); kernel boundaries are the
        // sync.  gkbuf holds per-chunk ranks (P_PAD ints/group), gcnt/goff the
        // per-chunk counts/offsets (Z_fin ints/group).
        TORCH_CHECK(gkbuf.numel() >= n_req * n_kv * (long)P_PAD,
                    "sel_split: gkbuf scratch too small");
        const int cwf = (int)((P_PAD + Z_fin - 1) / Z_fin);
        const size_t ksmem = (size_t)cwf * sizeof(int);
        static size_t rec_ks = 0;
        if (ksmem > 48ul * 1024 && ksmem > rec_ks) {
            cudaFuncSetAttribute(sel_keepscan_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ksmem);
            rec_ks = ksmem;
        }
        dim3 fgrid((unsigned)Z_fin, (unsigned)n_kv, (unsigned)n_req);
        if (tau_pp == 2) {
            // TAUC: digit-0 pick from the pass-0 ghist (no scan) -> ONE
            // page-parallel candidate compaction into gkbuf (its keepscan
            // lifetime starts strictly later in-stream) -> tiny finish over
            // the compacted boundary-bin values. Byte-identical tau.
            if (finfix)
                sel_tauc_pick0_fx_kernel<<<ngrid, 32, 0, stream>>>(
                    nsh.data_ptr<int>(), bfix.data_ptr<int>(),
                    ghist.data_ptr<int>(), gstate.data_ptr<int>(),
                    gtau.data_ptr<float>(), gcut.data_ptr<int>(),
                    ghist2.data_ptr<int>(),
                    (int)n_sink, (int)n_kv);
            else
                sel_tauc_pick0_kernel<<<ngrid, 32, 0, stream>>>(
                    nsh.data_ptr<int>(), bfix.data_ptr<int>(),
                    ghist.data_ptr<int>(), gstate.data_ptr<int>(),
                    gtau.data_ptr<float>(), gcut.data_ptr<int>(),
                    (int)n_sink, (int)n_kv);
            if (finfix) {
                // FINFIX route (doc 27): pick0_fx zeroed ghist2; compact
                // builds the digit-1 histogram there; finish digit-1 O(256).
                sel_tauc_compact_fx_kernel<<<fgrid, (unsigned)nblk, 0, stream>>>(
                    nsh.data_ptr<int>(), score.data_ptr<float>(),
                    gstate.data_ptr<int>(),
                    reinterpret_cast<unsigned*>(gkbuf.data_ptr<int>()),
                    ghist2.data_ptr<int>(),
                    (int)n_sink, (int)n_kv, (int)Z_fin, cwf,
                    (long)score.stride(0), (long)score.stride(1), (int)P_PAD);
                sel_tauc_finish_fx_kernel<<<ngrid, 256, 0, stream>>>(
                    nsh.data_ptr<int>(), gstate.data_ptr<int>(),
                    reinterpret_cast<const unsigned*>(gkbuf.data_ptr<int>()),
                    ghist2.data_ptr<int>(),
                    gtau.data_ptr<float>(), gcut.data_ptr<int>(),
                    score.data_ptr<float>(),
                    (long)score.stride(0), (long)score.stride(1),
                    (int)n_sink, (int)n_kv, (int)P_PAD, (int)exactb);
            } else {
                sel_tauc_compact_kernel<<<fgrid, (unsigned)nblk, 0, stream>>>(
                    nsh.data_ptr<int>(), score.data_ptr<float>(),
                    gstate.data_ptr<int>(),
                    reinterpret_cast<unsigned*>(gkbuf.data_ptr<int>()),
                    (int)n_sink, (int)n_kv, (int)Z_fin, cwf,
                    (long)score.stride(0), (long)score.stride(1), (int)P_PAD);
                sel_tauc_finish_kernel<<<ngrid, 256, 0, stream>>>(
                    nsh.data_ptr<int>(), gstate.data_ptr<int>(),
                    reinterpret_cast<const unsigned*>(gkbuf.data_ptr<int>()),
                    gtau.data_ptr<float>(), gcut.data_ptr<int>(),
                    score.data_ptr<float>(),
                    (long)score.stride(0), (long)score.stride(1),
                    (int)n_sink, (int)n_kv, (int)P_PAD, (int)exactb);
            }
        } else {
            sel_tau_kernel<<<ngrid, (unsigned)nblk, 0, stream>>>(
                ghist.data_ptr<int>(), npg.data_ptr<int>(), nsh.data_ptr<int>(),
                bfix.data_ptr<int>(), score.data_ptr<float>(), gtau.data_ptr<float>(),
                gcut.data_ptr<int>(), (int)n_sink, (int)n_kv,
                (long)score.stride(0), (long)score.stride(1), (int)P_PAD,
                (int)exactb);
        }
        sel_keepscan_kernel<<<fgrid, (unsigned)nblk, ksmem, stream>>>(
            npg.data_ptr<int>(), nsh.data_ptr<int>(), score.data_ptr<float>(),
            gtau.data_ptr<float>(), gcut.data_ptr<int>(),
            gkbuf.data_ptr<int>(), gcnt.data_ptr<int>(),
            page_table.data_ptr<int>(), (int)n_sink, (int)n_kv, (int)Z_fin, cwf,
            (long)score.stride(0), (long)score.stride(1), (int)mp,
            (long)page_table.stride(0), (long)page_table.stride(1), (int)P_PAD);
        sel_offset_kernel<<<ngrid, 32, 0, stream>>>(
            gcnt.data_ptr<int>(), goff.data_ptr<int>(), page_cnt.data_ptr<int>(),
            (int)n_kv, (int)Z_fin, (long)page_cnt.stride(0));
        sel_scatter_kernel<<<fgrid, (unsigned)nblk, 0, stream>>>(
            npg.data_ptr<int>(), gkbuf.data_ptr<int>(), goff.data_ptr<int>(),
            page_table.data_ptr<int>(), (int)n_kv, (int)Z_fin, cwf,
            (long)page_table.stride(0), (long)page_table.stride(1), (int)P_PAD);
    } else {
        const size_t smem = 256ul * sizeof(int)
                            + (size_t)P_PAD * sizeof(float)
                            + (size_t)P_PAD * sizeof(int);
        static int recorded_max = 0;
        if ((int)smem > recorded_max) {
            cudaFuncSetAttribute(sel_finish_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            recorded_max = (int)smem;
        }
        sel_finish_kernel<<<ngrid, (unsigned)nblk, smem, stream>>>(
            ghist.data_ptr<int>(), npg.data_ptr<int>(), nsh.data_ptr<int>(),
            bfix.data_ptr<int>(), score.data_ptr<float>(),
            page_table.data_ptr<int>(), page_cnt.data_ptr<int>(),
            (int)n_sink, (int)n_kv,
            (long)score.stride(0), (long)score.stride(1), (int)mp,
            (long)page_table.stride(0), (long)page_table.stride(1),
            (long)page_cnt.stride(0), (int)P_PAD);
    }
    cudaError_t le_ = cudaGetLastError();
    TORCH_CHECK(le_ == cudaSuccess, "sel_split launch failed: ",
                cudaGetErrorString(le_), " global_finish=", global_finish,
                " P_PAD=", P_PAD);
}

void flat_k_update(torch::Tensor K, torch::Tensor gpmax, int64_t Zg,
                   int64_t n, double slack, int64_t finfix) {
    TORCH_CHECK(K.scalar_type() == at::kFloat && K.is_contiguous(), "K");
    TORCH_CHECK(gpmax.scalar_type() == at::kFloat, "gpmax");
    auto stream = at::cuda::getCurrentCUDAStream();
    if (finfix) {
        // FINFIX (doc 27): block-per-item tree reduce, value-identical
        // (fp max is order-exact); the one-warp shape was 7.7us of latency.
        flat_k_update_fx_kernel<<<(unsigned)n, 256, 0, stream>>>(
            K.data_ptr<float>(), gpmax.data_ptr<float>(), (int)Zg, (int)n,
            (float)slack);
    } else {
        const unsigned nb = (unsigned)((n + 31) / 32);
        dim3 kgrid(nb);
        flat_k_update_kernel<<<kgrid, 32, 0, stream>>>(
            K.data_ptr<float>(), gpmax.data_ptr<float>(), (int)Zg, (int)n,
            (float)slack);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flat_k_update", &flat_k_update,
          "FLAT K shift update from gpmax (doc 18d)");
    m.def("topb_select_cuda", &topb_select_cuda,
          "static top-b page selection radix-select (Hopper sm_90a)");
    m.def("nrmtopb_select_cuda", &nrmtopb_select_cuda,
          "fused nrm GQA-combine + radix top-b (Hopper sm_90a)");
    m.def("nrmtopb_select_v2_cuda", &nrmtopb_select_v2_cuda,
          "SELECT-V2 small-P fused nrm+top-b, hmax handshake only (lever 2)");
    m.def("sel_split_cuda", &sel_split_cuda,
          "P11 3-kernel page-parallel nrm+topb split (H200 long-ctx)");
    m.def("sel_cnt_audit", &sel_cnt_audit,
          "LOCKS_CNT_AUDIT instrument: page_cnt-vs-invariant accumulator");
}
"""

_MOD = None
_TRIED = False


def _get():
    """Build (once, hash-cached) and return the extension."""
    global _MOD, _TRIED
    if _TRIED:
        return _MOD
    _TRIED = True
    from torch.utils.cpp_extension import load_inline
    _cf = ["-O3", _arch.arch_flag()]
    # FIX-C opt-in (LOCKS_QFIRST_FIXC=1): compile the select_v2 ENTRY
    # trigger (griddepcontrol.launch_dependents at kernel entry) so the
    # deferred kv/rope PSS dependents run under the select window.  Flag
    # unset -> define off, TU byte-identical (SASS-identity gated).
    if os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1":
        _cf.append("-DLOCKS_FIXC")
    _name = "locks_topb_select_cuda"
    # SEL phase-merge opt-in (LOCKS_SEL_PMERGE=1, 2026-07-22, audit lever
    # 3; session-P probe 7 left the lane OPEN: standalone 6.71us @16K,
    # barrier stall 48%).  Compiles the select_v2 kernel with barriers
    # B1/B3/B5/B7 merged away (named producer/consumer barrier for the
    # zero/init phase; redundant warp-uniform recompute of the crossing
    # scan / ballot resolve / chunk scan).  Selection-identical by
    # construction; gated bitwise by probe_sel_v2.py's full battery with
    # the flag on.  Distinct extension name keeps the shared torch-
    # extensions cache collision-free per flag vector.  Flag unset ->
    # define off, TU byte-identical (SASS-identity gated).
    if os.environ.get("LOCKS_SEL_PMERGE", "0") == "1":
        _cf.append("-DLOCKS_SEL_PMERGE")
        _name = "locks_topb_select_pmerge"
    _MOD = load_inline(
        name=_name,
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=_cf,
        verbose=False)
    print("[topb_select_cuda] hand-CUDA radix-select top-b ACTIVE"
          + (" (SEL_PMERGE)" if "-DLOCKS_SEL_PMERGE" in _cf else ""),
          flush=True)
    return _MOD


def fused_available() -> bool:
    """True when the module builds and exposes the fused nrm+topb kernel.

    NO-FALLBACK RULE (2026-07-22): a BUILD FAILURE is no longer swallowed
    into False.  It used to route the deployed rki4 chain to the Triton
    nrm passes (a different implementation, ~84 us/layer at 128K) with no
    log line, so the cell measured a configuration nobody asked for.  The
    only legal way to leave the fused consumer is the explicit
    LOCKS_FUSE_NRMTOPB=0 / LOCKS_TOPB_TRITON=1 escape (see
    quad_score_cuda._fuse_nrmtopb_ready)."""
    mod = _get()
    return mod is not None and hasattr(mod, "nrmtopb_select_cuda")


def _blk_for(P_PAD: int) -> int:
    """Adaptive block (shared by both selectors; benched on H200: 512@16K,
    1024@64K).  Env override always wins."""
    env = os.environ.get("LOCKS_TOPB_BLK")
    blk = int(env) if env else (1024 if P_PAD > 1024 else 512)
    return max(32, min(1024, (blk // 32) * 32))


# --- LEVER-2 SELECT-V2 (2026-07-21): small-P redesign, hmax-handshake only.
# Opt-in (LOCKS_SEL_V2=1, default 0 = the deployed radix kernel; the old path
# is NEVER removed).  Gated bitwise vs the deployed has_hm=1 kernel by
# scratch_qgemv/probe_sel_v2.py (random draws + tie bands + all-equal).
_SEL_V2_MAXP = 8192   # smem plan sized for P_PAD <= 8192 (ctx <= ~64K)
_SEL_V2_SAID = False  # one-shot engagement sentinel (gate cells verify it)
_SEL_V2_INERT_SAID = False   # one-shot: flag on but precondition unmet
_SPLIT_SAID = False          # one-shot: split-chain engagement + its reason
_GFIN_SAID = False           # one-shot: split GLOBAL-kbuf finish engagement


def _sel_v2_enabled() -> bool:
    return os.environ.get("LOCKS_SEL_V2", "0") == "1"


def _sel_v2_blk() -> int:
    # GH200 sweep 2026-07-21 (probe_sel_v2, in-graph 16K/32K): 1024 best at
    # both deployed shapes (5.48/7.02us vs 256: 7.13/11.09 -- the wide block
    # shortens every strided sweep; barriers are cheap at 4 CTAs/GPU).
    blk = int(os.environ.get("LOCKS_SEL_V2_BLK", "1024"))
    return max(64, min(1024, (blk // 32) * 32))


# (The page-parallel cooperative selector helpers (_PP_ZMAX, _pp_enabled,
# _ensure_pp_scratch, _pp_choose_Z) were removed 2026-07-21 with the
# sm_120 lane; see ours_doc/REFUTED_ARMS_INDEX.md.)


# --- P11 SELECT-SPLIT (H200 long-ctx, has_hm=0 path only) ------------------ //
# Pass-2 Lever 1: raised 64->256 so the pmax/nrm_hist wide kernels can fill the
# machine at LOW batch (bs1 Gact=4: Z was capped at 64 = 256 CTAs; now up to
# 256 = 1024 CTAs).  gpmax scratch scales with _SPLIT_ZMAX (Gtot*8*ZMAX ints);
# order-free fp32 max/exp-sum => bitwise-invariant to Z.
_SPLIT_ZMAX = 256


def _split_enabled() -> bool:
    """3-kernel page-parallel split: opt-in (A/B via LOCKS_SEL_SPLIT=1) and
    only for shapes with enough pages that the fused kernel's O(P) phases
    dominate its ~20-barrier chain (LOCKS_SPLIT_MINP, default 4096)."""
    return os.environ.get("LOCKS_SEL_SPLIT", "0") == "1"


def _split_minp() -> int:
    return int(os.environ.get("LOCKS_SPLIT_MINP", "4096"))


def _split_Z(P_PAD: int, Gact: int) -> int:
    """Wide-kernel grid.z: fill the machine at low batch (target CTAs env
    LOCKS_SPLIT_TARGET), chunk >= 128 pages, capped by scratch.
    Pass-2 Lever 1: default target 768 (~6 CTAs/SM on H200) so bs1 (Gact=4)
    launches Z=192 = 768 CTAs instead of the starved 48; measured pmax+nrm_hist
    drop (bs1/1M SEL 145->73us).  High batch (Gact=64) naturally gets small Z
    (768/64=12) since the group dim already fills the machine."""
    if env := os.environ.get("LOCKS_SPLIT_Z"):
        return max(1, min(_SPLIT_ZMAX, int(env)))
    target = int(os.environ.get("LOCKS_SPLIT_TARGET", "768"))
    Z = max(1, target // max(1, Gact))
    return max(1, min(Z, _SPLIT_ZMAX, P_PAD // 128))


def _split_global_finish(P_PAD: int) -> bool:
    """The smem sel_finish caches sc[P_PAD]+kbuf[P_PAD] (1024 + 8*P_PAD B); when
    that exceeds the device opt-in cap (H200 227KB @ P_PAD>=32768 / ctx>=256K)
    it fails to launch.  Route to the GLOBAL-kbuf finish there (works to 1M).
    A 4KB margin keeps a safety gap below the hard cap."""
    if os.environ.get("LOCKS_SPLIT_FORCE_GLOBAL", "0") == "1":
        return True   # test hook: exercise the global finish at small P_PAD
    smem = 1024 + 8 * int(P_PAD)
    gfin = smem > (_arch.smem_cap() - 4096)
    # LEGAL capability dispatch made EXPLICIT (no-fallback rule, 2026-07-22).
    # Invariant: the global-kbuf finish computes the SAME selection as the
    # smem finish (same scan, kbuf lives in global instead of shared), so
    # only the memory space differs; the smem variant cannot LAUNCH here.
    if gfin:
        global _GFIN_SAID
        if not _GFIN_SAID:
            _GFIN_SAID = True
            print(f"[locks] SELECT-SPLIT global-kbuf finish (smem finish "
                  f"needs {smem}B > cap {_arch.smem_cap()}B - 4096)",
                  flush=True)
    return gfin


def _split_lowbatch(n_req: int, P_PAD: int) -> bool:
    """Pass-2 Lever 1b: at LOW batch the single-CTA-per-group fused selector is
    grid-starved (bs1 = 4 CTAs).  For P_PAD large enough that the split's
    per-launch overhead is amortized, route low batch to the page-parallel split
    even where the fused smem cache still fits (measured: bs1/128K fused 27.0us
    -> split+PP-finish 24.7us).  Below the threshold (64K: P_PAD=8192) the fused
    single kernel wins, so the floor is P_PAD>=16384 (~128K)."""
    if os.environ.get("LOCKS_SPLIT_LOWBATCH", "1") == "0":
        return False
    maxreq = int(os.environ.get("LOCKS_SPLIT_LB_MAXREQ", "4"))
    minpad = int(os.environ.get("LOCKS_SPLIT_LB_MINPAD", "16384"))
    return int(n_req) <= maxreq and int(P_PAD) >= minpad


# Page-parallel finish: keep the local per-chunk scan smem (cw_fin ints) small
# by picking Z_fin so cw_fin ~= _SPLIT_CWF; capped so gcnt/goff scratch fits.
_SPLIT_CWF = 2048          # target per-chunk pages (8 KB smem local scan)
_SPLIT_ZFIN_MAX = 256


def _split_Zfin(P_PAD: int) -> int:
    if env := os.environ.get("LOCKS_SPLIT_ZFIN"):
        return max(1, min(_SPLIT_ZFIN_MAX, int(env)))
    cwf = int(os.environ.get("LOCKS_SPLIT_CWF", str(_SPLIT_CWF)))
    return max(1, min(_SPLIT_ZFIN_MAX, -(-int(P_PAD) // max(1, cwf))))


_EXACTB_SAID = False


def _exactb_enabled() -> bool:
    """EXACT-B tie resolution on the ctx>=128K page-parallel finish.

    DEFAULT ON: this is a CORRECTNESS fix, not an experiment.  The kill
    switch LOCKS_SEL_EXACTB=0 restores the pre-2026-07-22 `score >= tau`
    keep rule (which overspends the budget on exact fp32 ties, up to the
    WHOLE context: measured page_cnt 8193 vs the invariant 128) and exists
    only so the fix can be A/B-priced against the shipped vintage."""
    return os.environ.get("LOCKS_SEL_EXACTB", "1") != "0"


def _cnt_audit_enabled() -> bool:
    return os.environ.get("LOCKS_CNT_AUDIT", "0") == "1"


def _ensure_split_scratch(st, n_kv: int, P_PAD: int, want_pp: bool = False):
    R = int(getattr(st, "max_reqs", 0) or st.score.shape[0])
    Gtot = R * int(n_kv)
    if getattr(st, "_sp_gtot", 0) < Gtot:
        dev = st.score.device
        # PMAX FOLD writes gpmax at the SCORE kernel's grid z (up to ~4096
        # at auto_zsplit's cap), not the select Z; size for the max of both.
        _zmax = (4096 if os.environ.get("LOCKS_PMAX_FOLD", "0") == "1"
                 else _SPLIT_ZMAX)
        st._sp_gpmax = torch.zeros(Gtot * 8 * _zmax,
                                   dtype=torch.float32, device=dev)
        st._sp_ghist = torch.zeros(Gtot * 256, dtype=torch.int32, device=dev)
        # FINFIX digit-1 histogram scratch (ghist2) + per-group tau state
        # {prefix, kmask, k, done} for the TAUC pick/finish chain.
        st._sp_ghist2 = torch.zeros(Gtot * 256, dtype=torch.int32, device=dev)
        st._sp_gstate = torch.zeros(Gtot * 4, dtype=torch.int32, device=dev)
        # FLAT guard flag (doc 18): nonzero = K shift violated (checked at
        # run end by the harness; in-graph there is no host branch).
        st._sp_gflag = torch.zeros(1, dtype=torch.int32, device=dev)
        st._sp_gtau = torch.zeros(Gtot, dtype=torch.float32, device=dev)
        # EXACT-B tie cutoff, one int per (request, kv-head): the page index
        # of the last equal-to-tau page the quota admits (LOCKS_NO_CUT =
        # 0x7fffffff = no tie overflow).  Written by the same kernel that
        # writes gtau, read-only in keepscan; st.score is NEVER mutated.
        st._sp_gcut = torch.full((Gtot,), 0x7fffffff, dtype=torch.int32,
                                 device=dev)
        # LOCKS_CNT_AUDIT instrument buffer (12 int64 counters; unallocated
        # and never launched when the env is off).
        if _cnt_audit_enabled():
            st._sp_cnt_audit = torch.zeros(16, dtype=torch.int64, device=dev)
        st._sp_gcnt = torch.zeros(Gtot * _SPLIT_ZFIN_MAX, dtype=torch.int32,
                                  device=dev)
        st._sp_goff = torch.zeros(Gtot * _SPLIT_ZFIN_MAX, dtype=torch.int32,
                                  device=dev)
        st._sp_gtot = Gtot
        st._sp_gkbuf_pad = 0
    # page-parallel finish kbuf scratch (R*n_kv*P_PAD int32): grow with P_PAD.
    # Allocated on the warmup call (before graph capture), only when the
    # page-parallel finish is actually reached (smem-cap OR low-batch route),
    # so no in-graph allocation.
    if want_pp and getattr(st, "_sp_gkbuf_pad", 0) < P_PAD:
        st._sp_gkbuf = torch.empty(Gtot * int(P_PAD), dtype=torch.int32,
                                   device=st.score.device)
        st._sp_gkbuf_pad = int(P_PAD)


def sel_split_select_cuda(st, n_req: int) -> None:
    """Page-parallel nrm+topb: pmax (wide) -> nrm+pass-0 hist (wide) -> finish.
    Finish is the smem radix (P_PAD fits) or, at long ctx, the PAGE-PARALLEL
    tau/keepscan/offset/scatter (fills the machine, no smem P_PAD cache).
    Outputs byte-identical to the fused has_hm=0 kernel (CUDA-side contract).
    hmax is neither read nor reset, so callers gate this to the has_hm=0 path."""
    mod = _get()
    n_kv, MP = st.n_kv, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    # page-parallel finish when the smem finish can't launch (smem-cap) OR the
    # low-batch route asks for it (fills the machine at bs<=4); either way the
    # gkbuf scratch must exist.
    gfin = _split_global_finish(P_PAD) or _split_lowbatch(n_req, P_PAD)
    _ensure_split_scratch(st, n_kv, P_PAD, want_pp=gfin)
    Gact = int(n_req) * int(n_kv)
    wblk = int(os.environ.get("LOCKS_SPLIT_BLK", "256"))
    Z = _split_Z(P_PAD, Gact)
    st._split_last_Z = Z
    Zfin = _split_Zfin(P_PAD) if gfin else 1
    st._split_last_Zfin = Zfin
    gkbuf = st._sp_gkbuf if gfin else st._sp_ghist   # placeholder when unused
    # TAUC: candidate-compaction tau (digit-0 from the pass-0 ghist + ONE
    # page-parallel compact + tiny finish). DEFAULT-ON in the global-finish
    # route since 2026-07-16 (bitwise gate PASS via gate_sel_glm.py at
    # 64K-1M incl. bs4; chain @1M sel 50.9 -> 33.1us; SELECT_KERNEL_
    # CAMPAIGN.md section 2). LOCKS_SEL_TAUC=0 is the explicit kill switch
    # (tau_pp=0 compiles the legacy single-CTA sel_tau route).
    # gkbuf doubles as the candidate buffer.
    tau_pp = 2 if (gfin
                   and os.environ.get("LOCKS_SEL_TAUC", "1") != "0") else 0
    # PMAX FOLD (LOCKS_PMAX_FOLD): the score kernel emitted gpmax at its own
    # grid z (st._fold_Zs) and zeroed ghist; skip sel_pmax and point the
    # nrm_hist gpmax-reduce at the folded slot count.
    fold_Zg = (int(getattr(st, "_fold_Zs", 0))
               if os.environ.get("LOCKS_PMAX_FOLD", "0") == "1" else 0)
    # FLAT (doc 18): score_h holds masses; route the linear combine.
    _flat_on = os.environ.get("LOCKS_FLAT", "0") == "1"
    # FINFIX (doc 27, agent-audited): de-serialized tauc tail (compact_fx
    # digit-1 histogram into ghist2 + finish_fx O(256)/match-aggregated +
    # block-per-item flat_k_update). TAUC route only. Opt-in until gates+
    # matched pair pass; LOCKS_SEL_FINFIX=1 arms it.
    # DEFAULT REVERTED 2026-07-18 (whole-ladder rule): the 1M pair won
    # (-1.71%) but 256K REGRESSED +4.87% (tauc-tail tax without the FLAT
    # flat_k win; mechanism post-mortem pending, doc 27e). Deploy arms
    # enable LOCKS_SEL_FINFIX=1 at ctx >= 1M only, where the pair is green.
    _finfix = int(tau_pp == 2
                  and os.environ.get("LOCKS_SEL_FINFIX", "0") == "1")
    # EXACT-B (2026-07-22, default ON): the page-parallel finish resolves
    # ties at tau by the same count-and-demote rule as every other route
    # (strict > plus an ascending-page-index fill capped at quota = b -
    # #(>tau)), carried as a per-group cutoff index in st._sp_gcut.
    _exactb = 1 if _exactb_enabled() else 0
    global _EXACTB_SAID
    if gfin and not _EXACTB_SAID:
        _EXACTB_SAID = True
        _st = ("ON" if _exactb else
               "OFF (LOCKS_SEL_EXACTB=0 kill switch = the >=tau vintage)")
        print(f"[locks] SELECT-SPLIT EXACT-B tie resolution {_st}", flush=True)
    mod.sel_split_cuda(
        st.score_h, st.n_pages, st.n_sel_hi, st.b_fix,
        st.score, st.page_table, st.page_cnt,
        st._sp_gpmax, st._sp_ghist, gkbuf,
        st._sp_gtau, st._sp_gcut, st._sp_gcnt, st._sp_goff,
        int(n_req), int(n_kv), int(st.G), int(st.sink_pages), int(MP),
        int(P_PAD), int(Z), int(wblk), _blk_for(P_PAD),
        1 if gfin else 0, int(Zfin),
        st._sp_gstate, st._sp_ghist2, tau_pp,
        1 if fold_Zg else 0, fold_Zg,
        1 if _flat_on else 0, st._sp_gflag, _finfix, _exactb)
    if _cnt_audit_enabled() and getattr(st, "_sp_cnt_audit", None) is not None:
        mod.sel_cnt_audit(st.n_pages, st.n_sel_hi, st.b_fix, st.page_cnt,
                          st._sp_cnt_audit, int(n_req), int(n_kv),
                          int(st.sink_pages))
    if _flat_on and getattr(st, "_flat_K_cur", None) is not None:
        # FLAT K update (doc 18d): tighten this layer's shifts from the
        # per-head mass maxima sel_pmax just wrote. One tiny launch.
        # under PMAX FOLD the gpmax slots come from the score kernel's own
        # grid z (st._fold_Zs), not the select Z (doc 27c latent-bug fix)
        _kzg = int(fold_Zg) if fold_Zg else int(Z)
        mod.flat_k_update(st._flat_K_cur, st._sp_gpmax,
                          _kzg, int(n_kv) * int(st.G),
                          float(os.environ.get("LOCKS_FLAT_SLACK", "2.0")),
                          _finfix)
    st._topb_done = True


def nrmtopb_select_cuda(st, n_req: int, has_hmax: bool = True) -> None:
    """TAIL-OPT fused Stage-A.2 for the nrm-combine chain: ONE kernel does the
    single-pass nrm GQA-combine (st.score_h -> st.score, using the score
    kernel's hmax handshake when ``has_hmax``) AND the byte-identical radix
    top-b (st.page_table / st.page_cnt).  Callers that invoke
    ``topb_select_cuda`` afterwards get a no-op for this step (the
    ``_topb_done`` flag), so every existing score->topb call sequence keeps
    its contract with one fewer launch."""
    # (The PAGE-PARALLEL cooperative selector (LOCKS_PP_SELECT, sm_120-only)
    # was removed 2026-07-21 with the sm_120 lane, user ruling; see
    # ours_doc/REFUTED_ARMS_INDEX.md, recover from tag pre-cleanup-2026-07-21.
    # Its structural idea -- spread pages over grid.z at low batch -- lives on
    # in the P11 split below, which is the H200-refuted-then-relegalized form.)
    mod = _get()
    n_kv, MP = st.n_kv, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    # P11 SELECT-SPLIT: page-parallel path for long-ctx low-batch on the
    # has_hm=0 chain (H200 default).  hmax-path excluded: the split never
    # reads/resets the handshake buffer, so it must not shadow a score kernel
    # that published one.  Two triggers (deterministic per (shape, env) ->
    # graph-capture stable):
    #   * opt-in A/B via LOCKS_SEL_SPLIT=1 for P_PAD >= _split_minp (as before);
    #   * AUTO when the fused nrmtopb smem cache (1028 + 8.125*P_PAD B) exceeds
    #     the device opt-in cap (H200 227KB @ P_PAD>=32768 / ctx>=256K) -- there
    #     the fused kernel FAILS TO LAUNCH ("invalid argument"), so the split
    #     (page-parallel GLOBAL finish, no P_PAD smem cache) is the only path
    #     that runs.  This makes 256K..1M work with NO env flag.
    fused_fits = (1028 + 8.125 * P_PAD) <= (_arch.smem_cap() - 4096)
    if (not has_hmax) and (
            (_split_enabled() and P_PAD >= _split_minp())
            or not fused_fits
            or _split_lowbatch(n_req, P_PAD)):
        # Capability/regime dispatch, EXPLICIT (no-fallback rule, 2026-07-22).
        # !! HISTORY, kept because the claim it corrects was load-bearing: an
        # earlier version of this comment asserted the split chain is
        # "selection-IDENTICAL to the fused kernel ... same exact-b
        # count-and-demote".  THAT WAS FALSE.  sel_keepscan_kernel (applied at
        # the keep test) kept `scv >= tau` with NO tie demote anywhere in the
        # TAUC chain, so on an exact-fp32 tie AT tau this route kept a strict
        # SUPERSET of the top-b and OVERSPENT the budget: 29/52 adversarial
        # cases, worst case page_cnt 8193 against an invariant of 128 (the
        # whole context at a 128-page budget).  Trigger: any page more than
        # ~88 nats below the head max gives expf -> +0.0, so tau = +0 whenever
        # fewer than b pages carry nonzero mass for a (layer, KV head).
        # !! FIXED 2026-07-22 (user GO), DEFAULT ON: the tau-stage kernel now
        # publishes an EXACT-B tie cutoff per (request, kv-head) -- gcut, the
        # page index of the last equal-to-tau page the quota admits -- and
        # keepscan applies `> tau || (== tau && i <= gcut)`, i.e. the same
        # strict-> plus ascending-page-index fill capped at quota = b - #(>tau)
        # that the ctx<=64K route uses.  Nothing is mutated (st.score stays
        # pristine), no new launch, fixed grids -> graph-safe.  Battery
        # scratch_qgemv/run_exactb_v2.sh split: 29/52 FAIL -> 52/52 PASS; the
        # other four route batteries unchanged (504/104/52/52, 0 fails).
        # Kill switch LOCKS_SEL_EXACTB=0 restores the overspending vintage
        # (A/B pricing only).  FIELD RATE on real GLM-4-9B data, 128k and
        # 256k bs=1, in-graph audit over 28,000 (step, layer, kv-head) units
        # per ctx: 0 violations, max page_cnt 128 = the invariant, i.e. the
        # defect was ADVERSARIAL-ONLY and the >=128K budget numbers already
        # published are not contaminated.  Cost: matched A/B/A pairs at 128k
        # and 256k are NULL (see ours_doc/EXACTB_128K_FIX.md).
        # Evidence: ours_doc/EXACT_TOPB_VERIFICATION.md (the defect),
        # ours_doc/EXACTB_128K_FIX.md (the fix + gates).
        # Reasons for the dispatch, in the order tested: the manual A/B force, the
        # smem CAPACITY boundary (the fused kernel caches 1028 + 8.125*P_PAD
        # B and simply FAILS TO LAUNCH above the device opt-in cap), and the
        # low-batch grid-starvation row.
        global _SPLIT_SAID
        if not _SPLIT_SAID:
            _SPLIT_SAID = True
            why = ("LOCKS_SEL_SPLIT=1" if (_split_enabled()
                                           and P_PAD >= _split_minp())
                   else "smem CAPACITY" if not fused_fits
                   else "low-batch grid starvation")
            print(f"[locks] SELECT-SPLIT ACTIVE ({why}): P_PAD={P_PAD} "
                  f"n_req={n_req} fused_smem={int(1028 + 8.125 * P_PAD)}B "
                  f"cap={_arch.smem_cap()}B", flush=True)
        sel_split_select_cuda(st, n_req)
        return
    if _sel_v2_enabled() and not (has_hmax and 64 <= P_PAD <= _SEL_V2_MAXP):
        # LOCKS_SEL_V2=1 was REQUESTED but its precondition does not hold
        # (the v2 smem plan caps P_PAD at 8192 = ctx ~64K, and it consumes +
        # resets the hmax handshake, so has_hmax is mandatory).  The deployed
        # radix kernel below is selection-identical, but the cell is NOT
        # measuring SELECT-V2 -- say so once instead of silently swapping the
        # implementation under the flag.
        global _SEL_V2_INERT_SAID
        if not _SEL_V2_INERT_SAID:
            _SEL_V2_INERT_SAID = True
            # capturing= answers the question this sentinel exists for: an
            # INERT launch OUTSIDE capture is a warmup artifact; one INSIDE
            # capture means the REPLAYED graph does not carry SELECT-V2.
            print(f"[locks] SELECT-V2 INERT (has_hmax={has_hmax} "
                  f"P_PAD={P_PAD} not in [64, {_SEL_V2_MAXP}], "
                  f"capturing={torch.cuda.is_current_stream_capturing()}) "
                  "-> deployed radix selector", flush=True)
    pdl = int(os.environ.get("LOCKS_PDL", "1"))
    # LEVER-2 SELECT-V2 routing (P11 precedent): hmax-handshake shapes only
    # (the v2 kernel consumes + resets the handshake buffer; has_hm=0 and
    # long-ctx P_PAD keep the deployed kernel / split).
    if has_hmax and _sel_v2_enabled() and 64 <= P_PAD <= _SEL_V2_MAXP:
        global _SEL_V2_SAID
        if not _SEL_V2_SAID:
            _SEL_V2_SAID = True
            print(f"[locks] SELECT-V2 ACTIVE (P_PAD={P_PAD} "
                  f"blk={_sel_v2_blk()})", flush=True)
        mod.nrmtopb_select_v2_cuda(
            st.score_h, st._nrmtopb_hmax, st.n_pages, st.n_sel_hi, st.b_fix,
            st.score, st.page_table, st.page_cnt,
            int(n_req), int(n_kv), int(st.G), int(st.sink_pages), int(MP),
            int(P_PAD), _sel_v2_blk(), pdl)
        st._topb_done = True
        return
    mod.nrmtopb_select_cuda(
        st.score_h, st._nrmtopb_hmax, st.n_pages, st.n_sel_hi, st.b_fix,
        st.score, st.page_table, st.page_cnt,
        int(n_req), int(n_kv), int(st.G), int(st.sink_pages), int(MP),
        int(P_PAD), _blk_for(P_PAD), 1 if has_hmax else 0, pdl)
    st._topb_done = True


def topb_select_cuda(st, n_req: int) -> None:
    """HOT Stage-A.2 (hand-CUDA): write st.page_table / st.page_cnt from
    st.score using the static per-request budget st.b_fix.  Bitwise drop-in for
    ``locks.selection.topb_select`` (identical selected sets, compaction order,
    page_cnt).  Requires ``derive_page_params`` already run this step.
    """
    if getattr(st, "_topb_done", False):
        # the fused nrm+topb consumer already selected this step (launched by
        # the score entry); consume the flag so the next step re-selects.
        st._topb_done = False
        return
    mod = _get()
    n_kv, MP = st.n_kv, st.max_pages
    P_PAD = triton.next_power_of_2(MP)
    # Adaptive block: long context (big P_PAD) wins with more threads (shorter
    # per-thread compaction chunk); short context prefers 512 (less scan
    # overhead).  Env override always wins.  Benched on H200: 512@16K, 1024@64K.
    env = os.environ.get("LOCKS_TOPB_BLK")
    blk = int(env) if env else (1024 if P_PAD > 1024 else 512)
    blk = max(32, min(1024, (blk // 32) * 32))           # 32..1024, multiple of 32
    prof = int(os.environ.get("LOCKS_TOPB_PROF", "0"))   # 0 full,1 radix,2 compact
    pdl = int(os.environ.get("LOCKS_PDL", "1"))          # programmatic launch
    mod.topb_select_cuda(
        st.score, st.n_pages, st.n_sel_hi, st.b_fix,
        st.page_table, st.page_cnt,
        int(n_req), int(n_kv), int(st.sink_pages), int(MP),
        int(P_PAD), blk, prof, pdl)
