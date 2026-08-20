"""rki4_score_cuda -- the S4 register-pipelined CUDA scorer for rki4.

Mainline port of ``LOCKS-test2/kernel/r8score_cuda.py`` (ours_doc/
RKI4_KERNELS.md sections 4-5), GEOMETRY-GENERALIZED: nothing model-specific is
hardcoded.  Per (page, kv-head) resident summary: V (d x 8 basis, int4,
column-major packed, per-column bf16 scales), C (page x 8 coeffs, int8,
per-token bf16 scales), mu (d, int8, bf16 scale).  Score per (g, page):

  qt[r]   = sum_d V[d][r] * q_g[d] * colscale[r]
  mudot_g = sum_d mu[d] * q_g[d] * mus
  tok[t]  = s * ( cs[t] * sum_r C[t][r] * qt[r] + mudot_g )
  S       = LSE_t tok[t]

Lane map: 32 lanes = 4 octets x (rr = lane&7); rr owns basis column rr
(contiguous packed int4, vsub4+dp4a vs int8-staged q), a d/8 slice of mu, and
tokens {rr, rr+8, ...}; LSE via 3 xor-shfls within each octet.  2-deep
register software pipeline: page p+1's loads are issued into a register
B-bank before computing page p (the S4 verdict: p16 9.82/28.33/54.08 us at
16/64/128K cold z=64 on H200 at the flagship (8, 4, 128) geometry).

GENERALITY (one load_inline, 24 instantiations):

  * ``template <int PGT, int DHEAD, bool BT, bool G4>``: PGT in {16, 32} x
    DHEAD in {64, 128, 256}; BT=false is the dense packed-tensor addressing
    (the gate battery / standalone contract), BT=true the in-situ block-table
    variant (page row = bt[r, p] * n_kv + kh over the (NB, n_kv, ...) slabs,
    loop bound = n_sel_hi[r], output stride MP).  Identical float arithmetic
    and op order in both.  DHEAD=256 doubles the per-lane pipeline registers,
    so its instantiations use a 1-deep (no-pipeline) page loop -- a
    compile-time property of the template, not a runtime branch.
  * G (query heads per kv head) and n_kv are RUNTIME parameters: the fixed
    g = lane>>3 map becomes a g-tile loop (4 heads per sweep, summary bytes
    loaded ONCE per page and reused across tiles); q staging is warp-strided
    over the G heads (no NW == G assumption), smem sized for G <= MAXG = 16.
    G4=true compiles the G <= 4 single-pass tile (94 regs at <32,128> vs the
    runtime loop's 158, which cost a resident block/SM and 1.7x at p32 long
    ctx); the host picks it from the runtime G at launch.

CALL CONTRACT (refresh-period-reuse ready): every entry takes the CALLER's
query tensor, launches on ``at::cuda::getCurrentCUDAStream()`` AT CALL TIME,
and touches no module-global workspace or step-keyed state (the only module
globals are the compiled-extension handle and the device SM count).  Output /
scratch tensors are the caller's (``S`` / ``st.score_h``).  The int8 q
staging happens inside the kernel per call, so a stale or side-stream query
needs no extra preparation.

Geometry is TORCH_CHECKed at launch against the supported set (d in {64, 128,
256}, page in {16, 32}, G <= 16, rank 8); unsupported geometry raises --
there is no fallback scorer for rki4 (no Triton reference exists).
"""
from __future__ import annotations

import os

import torch

from .. import arch as _arch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <math_constants.h>

#ifndef NW
#define NW 4                 // warps/CTA; override -DNW=8 (floor sweep, doc 21)
#endif
#define TB (NW * 32)
#ifndef RNK
#define RNK 8                // summary rank; -DRNK=4|2 via LOCKS_RKI4_RANK
#endif
#if RNK != 8 && RNK != 4 && RNK != 2
#error "RNK must be 8, 4, or 2"
#endif
// RNK<8 support scope (rank campaign v2, 2026-07-27, user ruling "r4/r2
// must never be worse than r8 at ANY context"): the six-slab flagship lane
// is now rank-parametric too -- coeff loads take the RNK-width pattern the
// AOS reader already used, V/vs column reads clamp to the RNK live columns,
// and the mma epilogue gates dead output columns (verbatim the AOS branch's
// gating). rank<8 therefore rides r8's EXACT lane set at every ctx
// (six-slab+HMAX/SEL_V2/CO at <512K, AOS+PFR at >=512K) with strictly
// fewer bytes. v1's "MMA+AOS entries only" #error is retired.
#if RNK != 8 && (defined(RKI4_MMA_S1PROBE) || defined(RKI4_CP_NODT) || defined(RKI4_CP_NOEXP))
#error "RNK != 8 excludes V8/probe builds (rank campaign v1 scope)"
#endif
#define MAXG 16

// ---- rope geometry (MODEL-AGNOSTIC, 2026-07-22) --------------------------- //
// The NOROPE/QFIRST_CO staging prelude ropes q in-kernel.  Its geometry used to
// be HARDCODED to GLM-4 (rotary_dim 64, interleaved/non-neox pairing), which is
// wrong for every half-split (neox) arch -- Llama, Qwen, ... -- and for any
// other partial-rotary fraction.  It is now a COMPILE-TIME pair supplied by
// _runtime.rope_cflags() from the loaded model's own rotary module.  The
// defaults reproduce GLM exactly, and rope_cflags() emits NO flags on GLM
// geometry, so the deployed GLM TU is textually and byte-identical to the
// pre-2026-07-22 build (a runtime branch here would have put registers and a
// predicate in the flagship's staging prelude for nothing).
#ifndef LOCKS_ROT_DIM
#define LOCKS_ROT_DIM 64
#endif
#ifndef LOCKS_ROT_NEOX
#define LOCKS_ROT_NEOX 0
#endif
#define LOCKS_ROT_HALF (LOCKS_ROT_DIM / 2)

// One roped output element.  ``x`` = this head's row base, ``crow`` = the
// cos|sin cache row for this position (cos at [0, HALF), sin at [HALF,
// ROT_DIM)), ``dd`` < LOCKS_ROT_DIM.  Math is the inductor rope kernel's,
// verbatim: bf16 loads upcast to fp32, SEPARATE __fmul_rn/__fsub_rn/__fadd_rn
// in the dump's operand order (the triton kernel does NOT fma-contract; nvcc
// would), caller does the bf16 round-trip.
__device__ __forceinline__ float locks_rope_elem(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ crow, const int dd) {
#if LOCKS_ROT_NEOX
    // half-split: out[i] = x[i]*cos - x[i+H]*sin ; out[i+H] = x[i+H]*cos + x[i]*sin
    const int lo = (dd < LOCKS_ROT_HALF) ? dd : dd - LOCKS_ROT_HALF;
    const float c = __bfloat162float(__ldg(crow + lo));
    const float s = __bfloat162float(__ldg(crow + LOCKS_ROT_HALF + lo));
    const float e = __bfloat162float(x[lo]);
    const float o = __bfloat162float(x[lo + LOCKS_ROT_HALF]);
    return (dd < LOCKS_ROT_HALF)
        ? __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s))
        : __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s));
#else
    // interleaved (GLM): pair (2p, 2p+1) shares cos/sin index p.
    const int pr = dd >> 1;
    const float c = __bfloat162float(__ldg(crow + pr));
    const float s = __bfloat162float(__ldg(crow + LOCKS_ROT_HALF + pr));
    const float e = __bfloat162float(x[2 * pr]);
    const float o = __bfloat162float(x[2 * pr + 1]);
    return (dd & 1) ? __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s))
                    : __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s));
#endif
}

// int4-unpack ILP: full unroll issues all 4 V-words' vsub4 lo/hi in
// parallel -> ~20 live regs (the dominant consumer at G=8, ptxas differential).

// cp.async 16B helper: used by the DEPLOYED PFR record ring (>=512K arm
// set).  Loads only -> math/op-order unchanged -> score_h bitwise-invariant.
// cp.async.cg.shared.global is sm_80+ (valid on sm_90a).
#ifdef RKI4_MMA
#define RKI4_RING 3
__device__ __forceinline__ void cpasync16(void* dst_smem, const void* src) {
    unsigned s = (unsigned)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"(s), "l"(src));
}
__device__ __forceinline__ void cpasync_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cpasync_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
#endif

// RKI4_MMA: replace the dp4a projection (qt = V^T q, the DOMINANT ~2048-dp4a
// term) with ONE int8 tensor-core matmul over the whole GQA group. A(16x32)=q
// (rows 0-7 = the 8 query heads, single int8 limb; rows 8-15 = 0), B(32x8)=one
// page's 8 basis columns, C(16x8)=qt[head][basis]. The K=128 contraction is
// ORDERED [even-d (64) | odd-d (64)] so B's two K-halves are exactly the __vsub4
// lo/hi of the int4 basis (no nibble-scatter) and A's two halves are the
// already-staged q8e/q8o words. int8*int8->int32 is exact + associative, so C
// is BITWISE-identical to the dp4a acci (same integer products, same sum). Only
// the G8 d128 flagship instantiation activates it (if constexpr); the shipped
// dp4a path is untouched when RKI4_MMA is not defined.
#ifdef RKI4_MMA
__device__ __forceinline__ void r8_mma_s8(int d[4], const unsigned a[4],
                                          const unsigned b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// signed int4 -> int8x4: low / high nibble of each byte, biased (n^8)-8. These
// are the SAME expansions the dp4a path applies before dp4a, so the mma sees
// bit-identical V bytes.
__device__ __forceinline__ unsigned r8_lo(unsigned w) {
    return __vsub4((w & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
}
__device__ __forceinline__ unsigned r8_hi(unsigned w) {
    return __vsub4(((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
}
#endif

// Host-side record size: same formula as the kernel RECB (p16 d128 AOS
// flagship geometry; 64B-aligned). r8: 832, r4: 512, r2: 384.
#define RKI4_RECB_HOST (((RNK * 64 + 128 + 16 * RNK + 32 + 2 * RNK + 2) + 63) / 64 * 64)

// RKI4_MMA_SRED: strength-reduced addressing for the G8 d128 mma page loop.
// DEFAULT-ON since 2026-07-16 (host plumbs the define unless RKI4_MMA_SRED=0;
// byte-gated + event-timed, ours_doc/SCORE_KERNEL_ISSUE_BOUND.md section 7).
// Composes with RKI4_MMA_BIAS only; the CPA ring and the DEDUP tile carry
// their own loop bodies and are separate A/B arms (set RKI4_MMA_SRED=0).
// (The STAGE and VCP staging arms were removed 2026-07-21; wave 3b also
// removed WIDE2, PTRX and V8. Mechanisms and verdicts live in
// ours_doc/REFUTED_ARMS_INDEX.md; recover from tag pre-cleanup-2026-07-21.)

// RKI4_MMA_AOS: single record per (row) replacing the six summary arrays:
// [v4 512 | mu 128 | c8 8*PGT | cs 2*PGT | vs 16 | mus 2 | pad], 64B-aligned
// (p16: 832B). STRUCTURAL escape from the measured 64-reg/8-CTA equilibrium:
// ONE pointer + ONE stride frees the ~10 pointer registers (9 CTAs become
// reachable WITHOUT launch_bounds or spill) and each page's bytes are one
// contiguous L2-friendly block. Same bytes at new addresses -> bitwise gate.
// Host entry: r8score_aos (rec passed via the v4 kernel arg; dense only).
#ifdef RKI4_MMA_AOS
#if !defined(RKI4_MMA) || !defined(RKI4_MMA_SRED)
#error "RKI4_MMA_AOS requires RKI4_MMA + RKI4_MMA_SRED"
#endif
// RKI4_MMA_PFR: cp.async record ring; the staged bytes ARE the AOS record.
#if defined(RKI4_MMA_PFR) && !defined(RKI4_MMA_AOS)
#error "RKI4_MMA_PFR requires RKI4_MMA_AOS"
#endif
// RKI4_MMA_MUC (doc 25): mu rides the mma's dead basis columns at RNK<8;
// integer-exact vs the IDP chain -> bitwise. Stage 1 scope guard:
#if defined(RKI4_MMA_MUC) && (RNK == 8 || !defined(RKI4_MMA_AOS) || !defined(RKI4_MMA_BIAS) || defined(RKI4_MMA_S1PROBE))
#error "RKI4_MMA_MUC stage 1 covers RNK<8 AOS+BIAS builds only"
#endif
// ALLG (general tensor-core projection, G in {4, 8}): scoped to the AOS+BIAS
// build (the six-slab loop stays G8-flagship). MUC is excluded: mu rides the
// mma's dead basis columns in a G8-fragment order the G4 tile does not read
// (and PK2 wants those columns for page packing). FOLDR is excluded in v1:
// its epilogue publishes the b-octet fold rows unconditionally (G4 models cap
// at 128K ctx; FOLDR is a >=512K arm, so nothing in the zoo needs the combo).
#if defined(RKI4_MMA_ALLG) && (!defined(RKI4_MMA_AOS) || !defined(RKI4_MMA_BIAS))
#error "RKI4_MMA_ALLG requires the RKI4_MMA_AOS + RKI4_MMA_BIAS build"
#endif
#if defined(RKI4_MMA_ALLG) && defined(RKI4_MMA_MUC)
#error "RKI4_MMA_ALLG excludes RKI4_MMA_MUC (set LOCKS_RKI4_MUC=0)"
#endif
#if defined(RKI4_MMA_ALLG) && defined(RKI4_MMA_FOLDR)
#error "RKI4_MMA_ALLG v1 excludes RKI4_MMA_FOLDR (b-octet fold rows)"
#endif
// RKI4_MMA_SCREEN (doc 26): certified-screen kernel. Needs the AOS record
// (+ the nrmC pad field) and the BIAS int-mma; FLAT-domain semantics.
#if defined(RKI4_MMA_SCREEN) && (!defined(RKI4_MMA_AOS) || !defined(RKI4_MMA_BIAS))
#error "RKI4_MMA_SCREEN requires RKI4_MMA_AOS + RKI4_MMA_BIAS"
#endif
// RKI4_MMA_FOLDR (doc 27d, red-team-approved): in-register pmax fold at the
// S-write site; FLAT+AOS only, excludes the refuted/probe loop arms.
#if defined(RKI4_MMA_FOLDR) && (!defined(RKI4_MMA_FLAT) || !defined(RKI4_MMA_AOS) || defined(RKI4_MMA_S1PROBE) || defined(RKI4_CP_NODT) || defined(RKI4_CP_NOEXP))
#error "RKI4_MMA_FOLDR requires FLAT+AOS; excludes probe arms"
#endif
#endif

// PGT: page tokens (16|32).  DHEAD: head dim (64|128|256).  BT: block-table
// addressing (in-situ) vs dense page-major packed tensors (gate battery).
// G4: compile-time specialization for G <= 4 (one g-tile pass, g = lane>>3;
// the S4 register shape: 94 regs vs the runtime loop's 158 at <32,128>,
// which cost a resident block per SM and 1.7x at p32 long ctx).  Host
// dispatch picks it from the runtime G; the g-tile LOOP remains the general
// path for G > 4 -- template specialization, not a runtime fallback.
// OCCUPANCY (L4, measured): the flagship page-16 G4 d=128 kernel is
// register/occupancy-bound at 16K+ (the 3-deep pipeline REGRESSED,
// register-bound; ours_doc/RKI4_KERNELS.md L1 NEG).  A min-blocks=6 launch
// bound trades a few registers for one extra resident CTA/SM -> a
// bitwise-safe ~5% score win (16K 10.76 -> 10.23us, 64K 29.5 -> 28.4) with
// NO result change (launch_bounds is math-invariant, G1 re-certified).
// Applied ONLY to the page-16 d=128 G4 instantiation (the paper flagship):
// the page-32 kernel has a HEAVIER per-page register profile and min-blocks
// 6 REGRESSED it (measured p32 128K 30.6 -> 32.9), so PGT==16 is the exact
// guard.  Every other geometry keeps min-blocks=0 (unconstrained), so it is
// byte- AND perf-identical there.
#define RKI4_G8_MINBLK 0
// CO-KERNEL SEAM (P2b, 2026-07-20): the DEPLOYED build (no define)
// compiles the original __global__ kernel VERBATIM -- byte-identical
// binary (gate: cuobjdump REG/SMEM anchor under the deployed -D set,
// RKI4_MMA=1).  A co-kernel TU compiles this same source with
// -DRKI4_DEVICE_BODY: the head becomes a __device__ function
// (rki4_score_dev) callable from a fat dispatcher kernel that runs the
// score CONCURRENTLY with independent work (no streams, no events, no
// barriers -- the halves share nothing).
#ifdef RKI4_DEVICE_BODY
// __device__ form: grid coordinates become parameters (co_r/co_kh/co_z =
// the logical (blockIdx.x, .y, .z); co_gz = the logical gridDim.z) -- the
// co-kernel TU string-replaces the body's blockIdx/gridDim reads with
// these names in ITS COPY of the source; the deployed text is untouched.
template <int PGT, int DHEAD, bool BT, bool G4, bool G8 = false, bool PF = false>
__device__ __noinline__
void rki4_score_dev(
    const int co_r, const int co_kh, const int co_z, const int co_gz,
    const int co_gy,
#else
template <int PGT, int DHEAD, bool BT, bool G4, bool G8 = false, bool PF = false>
__global__ __launch_bounds__(TB, (G4 && DHEAD == 128 && PGT == 16) ? 6
                                 : RKI4_G8_MINBLK)
void rki4_score_kernel(
#endif
    const __nv_bfloat16* __restrict__ q,   // (R, n_kv * G, d)
    const uint8_t* __restrict__ v4,        // dense (n_kv, P, RNK, d/2)
    const __nv_bfloat16* __restrict__ vs,  //   |  BT (NB, n_kv, RNK, d/2)
    const int8_t* __restrict__ c8,         //   |  etc. per the same swap
    const __nv_bfloat16* __restrict__ cs,
    const int8_t* __restrict__ mu8,
    const __nv_bfloat16* __restrict__ mus,
    const int* __restrict__ bt,            // BT only: (R, MB) page -> block
    const int* __restrict__ nsh,           // BT only: (R,) n_sel_hi
    float* __restrict__ S,                 // (R, n_kv, G, MP)
    const int P_dense, const int MP, const int G, const float sm_scale,
    const long q_sr, const long q_sh, const long bt_sr,
    float* __restrict__ gpmax_out,         // PMAX FOLD: (group, 8, gridDim.z)
    int*   __restrict__ ghist_out,         //   chunk maxima + ghist zeroing;
                                           //   nullptr = fold off (dead code)
    const float* __restrict__ Kflat,       // FLAT (doc 18): per-(kh,g) shift
                                           //   constants; nullptr = LSE path
    const int* __restrict__ sl_rope,       // NOROPE (round A): seq_lens
    const __nv_bfloat16* __restrict__ csc_rope,  // (1M, 64) bf16 cos|sin
    __nv_bfloat16* __restrict__ qout_rope, // roped-q publish (R, n_kv*G, D);
                                           // qout_rope==nullptr = legacy
                                           // (q arrives ALREADY roped)
    unsigned* __restrict__ hmax_pub)       // HMAX HANDSHAKE (lever 1): f2u
                                           // per-(r,kh,g) page maxima out,
                                           // (R, n_kv, 8) int32 buffer;
                                           // nullptr = no publish (legacy)
{
    constexpr int VW4 = DHEAD / 32;        // uint4 words per packed V column
    constexpr int MUW = DHEAD / 32;        // uint32 mu words per lane
    // 2-deep page pipeline: G4 only.  For G>4 the g-tile LOOP already issues
    // 2+ independent compute tiles per page (intra-page ILP), so the extra
    // register-doubling B-bank buys little latency hiding while the 128-reg
    // footprint caps occupancy at 4 CTAs/SM (ncu: Block Limit Registers=4,
    // 25% theoretical).  Dropping it for G>4 frees registers -> more resident
    // CTAs.  Compile-time (not a runtime branch); math + op order unchanged
    // (prefetch only) -> score_h bitwise-invariant (GLM_KERNEL_OPT gate).
    // NOTE: the register-doubling 2-deep b-bank prefetch was TESTED for G8 and
    // REGRESSED (regs 96->125 -> occupancy 28->18% -> less total MLP: score
    // 94->127us @256K). The register-free MLP path (cp.async staging) is used
    // instead (below). So PIPE stays G4-only.
#ifdef RKI4_MMA
#ifdef RKI4_MMA_ALLG
    // ALLG (rank campaign v2, 2026-07-26): the int8 tensor-core projection for
    // EVERY GQA group size at d128, not just the G8 flagship -- G<8 models
    // (Llama/Qwen G4 class) ride the same m16n8k32 chain with the dead A rows
    // (heads >= G) zeroed at staging and the b-octet tile elided at G <= 4.
    // The mma is integer-exact, so score_h is BITWISE equal to the dp4a
    // score_tile path it replaces (gate: tests/gate_rank_mma.py). The G8
    // instantiation compiles VERBATIM the non-ALLG source (guards live only
    // in the G4-template branch) -> flagship codegen untouched.
    constexpr bool USE_MMA = (G8 || G4) && (DHEAD == 128);
#else
    constexpr bool USE_MMA = G8 && (DHEAD == 128);
#endif
#else
    constexpr bool USE_MMA = false;
#endif
    constexpr bool PIPE = (DHEAD < 256) && G4 && !USE_MMA;
    // HMAX HANDSHAKE (lever 1) fold coverage: paths whose S-store call
    // sites fold hm_a/hm_b in REGISTERS (zero-latency publish at the tail;
    // the generic kernel-end re-scan pays ~1us of unhidden load->reduce->
    // atomic latency, measured invariant to fence/barrier structure).
    // Folded: the SRED non-STAGE non-AOS mma loop (deployed scorer), the
    // !SRED shipped loop BOTH variants (the co TU compiles this one -- its
    // build mirrors only RKI4_MMA), and the dp4a G8 kill-switch path; every
    // other arm keeps the re-scan (correct for all).
#ifndef RKI4_MMA_AOS
    constexpr bool HM_FOLD_MMA = true;
#else
    constexpr bool HM_FOLD_MMA = false;
#endif
    constexpr bool HM_FOLD = (USE_MMA && HM_FOLD_MMA) || (!USE_MMA && G8);
    float hm_a = -CUDART_INF_F, hm_b = -CUDART_INF_F;   // heads gl / gl+4
    const int r    = blockIdx.x;
    const int kh   = blockIdx.y;
    const int zsp  = blockIdx.z;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nworker = gridDim.z * NW;
    const int wid = zsp * NW + warp;
    const int gl  = lane >> 3;             // slot within the 4-head g-tile
    const int rr  = lane & 7;              // basis column / token pair id
    const int n_kv = gridDim.y;
    const int P = BT ? __ldg(nsh + r) : P_dense;   // pages to score

    __shared__ float qsh[MAXG * DHEAD];
    // NOTE: q8e/q8o/q8d per-head rows DO carry a residual 2-/4-way bank
    // conflict on the per-page dp4a reads, but +1-word padding to break it was
    // MEASURED a NET REGRESSION (short_scoreboard 0.73->0.47 but score 86->89us
    // @256K, shared-load wavefronts 3.97M->6.01M): the odd stride split the
    // 128-bit-vectorized q8 loads into scalar loads, adding LSU traffic that
    // outweighed the conflict win. Kept UNPADDED (the measured-faster layout).
    // NOTE (2026-07-16): the 19.6% smem-load bank-conflict rate here was
    // A/B'd with +4-word row pads (bitwise PASS): DEAD FLAT at 32768/65536
    // (72.99 vs 72.96 / 129.09 vs 129.12 us). Conflicts hide under global
    // latency; pads reverted. Do not re-try without new evidence.
    __shared__ uint32_t q8e[MAXG][DHEAD / 8], q8o[MAXG][DHEAD / 8];
    __shared__ uint32_t q8d[MAXG][DHEAD / 4];
    __shared__ float qsc[MAXG];
    // HMAX HANDSHAKE (lever 1) CTA-stage scratch: rest-state init HERE so
    // the staging prelude's existing __syncthreads covers it (zero added
    // barriers before the epilogue's single join).
    __shared__ unsigned s_hm[8];
    if (hmax_pub != nullptr && tid < 8) s_hm[tid] = 0x007FFFFFu;
    if (qout_rope) {
        // NOROPE (round A): q arrives UNROPED -- the forward's fused rope
        // kernel (triton_poi_fused_2) is deleted.  Replicate its EXACT math
        // here (locks_rope_elem: bf16 loads upcast to fp32, cos/sin from the
        // bf16 cache row of pos = sl[r]-1, SEPARATE __fmul_rn/__fsub_rn/
        // __fadd_rn in the dump's operand order), plus a bf16 ROUND-TRIP so
        // staging quantizes exactly the bytes the deleted kernel used to
        // store.  Dims >= LOCKS_ROT_DIM pass through (partial rotary; the
        // bf16 round-trip is the identity there).  The pairing (interleaved
        // vs half-split) and rotary_dim are the loaded model's, compiled in
        // -- see the LOCKS_ROT_DIM/LOCKS_ROT_NEOX block at the top.  The
        // roped bf16 q is ALSO published to qout_rope: the split kernels' q
        // source under NOROPE (P2) -- zsplit CTAs write identical bytes
        // (benign redundancy, deterministic).
        const long pos = (long)__ldg(sl_rope + r) - 1;
        const __nv_bfloat16* crow = csc_rope + (long)LOCKS_ROT_DIM * pos;
        for (int i = tid; i < G * DHEAD; i += TB) {
            const int h = i / DHEAD, d = i % DHEAD;
            const long qb = (long)r * q_sr + (long)(kh * G + h) * q_sh;
            float v;
            if (d < LOCKS_ROT_DIM) {
                v = locks_rope_elem(q + qb, crow, d);
            } else {
                v = __bfloat162float(q[qb + d]);
            }
            const __nv_bfloat16 vb = __float2bfloat16(v);
            qout_rope[((long)r * n_kv * G + (long)kh * G + h) * DHEAD + d]
                = vb;
            qsh[i] = __bfloat162float(vb);
        }
    } else {
        for (int i = tid; i < G * DHEAD; i += TB)
            qsh[i] = __bfloat162float(
                q[(long)r * q_sr + (long)(kh * G + i / DHEAD) * q_sh
                  + (i % DHEAD)]);
    }
    __syncthreads();
    // stage q as int8, de-interleaved into even/odd-d packed words; warp w
    // stages heads w, w+NW, ... (no NW == G assumption)
    for (int gg = warp; gg < G; gg += NW) {
        const float* qw = qsh + gg * DHEAD;
        float am = 0.f;
        for (int j = lane; j < DHEAD; j += 32) am = fmaxf(am, fabsf(qw[j]));
        #pragma unroll
        for (int st = 16; st >= 1; st >>= 1)
            am = fmaxf(am, __shfl_xor_sync(~0u, am, st));
        const float qs = fmaxf(am, 1e-8f) / 127.f;
        if (lane == 0) qsc[gg] = qs;
        if (lane < DHEAD / 8) {
            uint32_t we = 0, wo = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int pe = (int)rintf(qw[8 * lane + 2 * b] / qs);
                const int po = (int)rintf(qw[8 * lane + 2 * b + 1] / qs);
                we |= (uint32_t)(uint8_t)(int8_t)pe << (8 * b);
                wo |= (uint32_t)(uint8_t)(int8_t)po << (8 * b);
            }
            q8e[gg][lane] = we; q8o[gg][lane] = wo;
        }
        // d-order packed words for the mu dp4a (DHEAD/4 words; > 32 at d=256)
        for (int j = lane; j < DHEAD / 4; j += 32) {
            uint32_t wd = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int pd = (int)rintf(qw[4 * j + b] / qs);
                wd |= (uint32_t)(uint8_t)(int8_t)pd << (8 * b);
            }
            q8d[gg][j] = wd;
        }
    }
    __syncthreads();

#ifdef RKI4_MMA
    // Hoisted A fragments = the query in int8 (CONSTANT across pages). Lane map
    // for the mma: gidm = lane>>2 = head (M-row), t4m = lane&3 = K-offset.
    // Even-d limb (K 0-63) uses q8e words {t4m,4+t4m,8+t4m,12+t4m}, odd-d limb
    // (K 64-127) uses q8o; A rows 8-15 (a1,a3) are zero.
    unsigned Af[4][4];
    int qsum_mma = 0;                              // BIAS: sum_d q_int8[head]
    __shared__ float qt_sh[NW][8][8];              // per-warp qt[head][basis]
    if constexpr (USE_MMA) {
        const int gidm = lane >> 2, t4m = lane & 3;
        Af[0][0] = q8e[gidm][t4m];      Af[0][1] = 0u;
        Af[0][2] = q8e[gidm][4 + t4m];  Af[0][3] = 0u;
        Af[1][0] = q8e[gidm][8 + t4m];  Af[1][1] = 0u;
        Af[1][2] = q8e[gidm][12 + t4m]; Af[1][3] = 0u;
        Af[2][0] = q8o[gidm][t4m];      Af[2][1] = 0u;
        Af[2][2] = q8o[gidm][4 + t4m];  Af[2][3] = 0u;
        Af[3][0] = q8o[gidm][8 + t4m];  Af[3][1] = 0u;
        Af[3][2] = q8o[gidm][12 + t4m]; Af[3][3] = 0u;
#ifdef RKI4_MMA_ALLG
        // ALLG: heads >= G have UNINITIALIZED q8e/q8o rows (staging loops run
        // gg < G only). Zero their A fragments so the dead C rows are exact
        // zeros, not garbage. Compiled ONLY into the G4-template instantiation
        // (G8 promises G == 8): the flagship SASS stays verbatim.
        if constexpr (G4) {
            if (gidm >= G) {
                #pragma unroll
                for (int f = 0; f < 4; ++f) { Af[f][0] = 0u; Af[f][2] = 0u; }
            }
        }
#endif
#ifdef RKI4_MMA_BIAS
        // Biased-nibble mma: feed the raw nibble (V+8) to drop the 8 __vsub4/
        // page (ALU), fold the +8 plane out exactly in the epilogue via
        // C = C' - 8*qsum, qsum = sum over all 128 d of q_int8[head] (hoisted,
        // per head, in a register). int32-exact -> byte-identical to vsub4.
        #pragma unroll
        for (int j = 0; j < DHEAD / 4 / 4; ++j)     // 8 words of the 32 per head
            qsum_mma = __dp4a((int)q8d[gidm][8 * t4m + j], 0x01010101, qsum_mma);
        qsum_mma += __shfl_xor_sync(~0u, qsum_mma, 1);
        qsum_mma += __shfl_xor_sync(~0u, qsum_mma, 2);
        qsum_mma *= 8;                              // pre-scale the fold-out
#ifdef RKI4_MMA_ALLG
        // ALLG: q8d rows >= G are uninitialized -> force the dead heads'
        // fold-out to 0 (their acc rows are exact 0 from the zeroed A frags,
        // and 0 - 0 keeps the dead qt slots exactly 0). G4-template only.
        if constexpr (G4) { if (gidm >= G) qsum_mma = 0; }
#endif
#endif
    }
#endif

#ifdef RKI4_MMA_FOLDR
    float fold_a = -CUDART_INF_F, fold_b = -CUDART_INF_F;
#endif
    const long pb_v = (long)kh * (long)P_dense;    // dense: page-major/head
    const int* btr = BT ? (bt + (long)r * bt_sr) : nullptr;
    auto row_of = [&](int pp) -> long {
        if constexpr (BT) return (long)__ldg(btr + pp) * n_kv + kh;
        else              return pb_v + pp;
    };
    auto load_page = [&](int pp, uint4* vv, uint32_t* mm, uint2* cc,
                         __nv_bfloat16& sv, __nv_bfloat16& smu,
                         __nv_bfloat16* sc) {
        const long row = row_of(pp);
#ifdef RKI4_MMA_AOS
        // ---- G4 (non-MMA) AOS record reader ---------------------------- //
        // The six summary components live in ONE 64B-aligned record per
        // (page, kv-head); the AOS launch passes ONLY that record (via v4)
        // and nullptr for vs/c8/cs/mu8/mus, so the six-array reads below
        // (the #else) faulted.  Read every field from the record at the
        // compile-time RO_* offsets (mirroring the G8 mma loop / the
        // rki4_state layout) with a per-lane stride that stays naturally
        // aligned AND in-bounds for RNK in {2, 4, 8}: the pre-AOS uint2 C
        // read was BOTH null and (at RNK<8, whose token stride is RNK<8
        // bytes) misaligned.  The basis columns exist only for rr < RNK, so
        // the V column and the vs scale are read at a clamped column index
        // (in-bounds) and compute_page forces qt = 0 for lanes rr >= RNK,
        // which makes those clamped bytes inert in the token sum.
        constexpr int  RO_MU_L  = RNK * (DHEAD / 2);
        constexpr int  RO_C8_L  = RO_MU_L + DHEAD;
        constexpr int  RO_CS_L  = RO_C8_L + RNK * PGT;
        constexpr int  RO_VS_L  = RO_CS_L + 2 * PGT;
        constexpr int  RO_MUS_L = RO_VS_L + 2 * RNK;
        constexpr long RECB_L   = ((RO_MUS_L + 2) + 63) / 64 * 64;
        const uint8_t* p_rec = v4 + row * RECB_L;
        const int vc = (rr < RNK) ? rr : (RNK - 1);        // clamp to V region
        const uint4* colv = reinterpret_cast<const uint4*>(
            p_rec + (long)vc * (DHEAD / 2));
        #pragma unroll
        for (int c = 0; c < VW4; ++c) vv[c] = __ldg(colv + c);
        // mu slice for this lane (full-d field, 8-way split over rr, RNK-
        // independent); d128 -> MUW == 4 -> one aligned uint4 per lane.
        if constexpr (MUW % 4 == 0) {
            const uint4* mup = reinterpret_cast<const uint4*>(
                p_rec + RO_MU_L) + rr * (MUW / 4);
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(mup + j);
                mm[4 * j + 0] = m4.x; mm[4 * j + 1] = m4.y;
                mm[4 * j + 2] = m4.z; mm[4 * j + 3] = m4.w;
            }
        } else {
            const uint32_t* mup = reinterpret_cast<const uint32_t*>(
                p_rec + RO_MU_L) + rr * MUW;
            #pragma unroll
            for (int j = 0; j < MUW; ++j) mm[j] = __ldg(mup + j);
        }
        // per-token coeff C[token, 0:RNK]: RNK contiguous bytes, read with an
        // access width that divides the RNK-byte token stride (aligned for
        // RNK in {2,4,8}), zero-padded to the uint2 the token sum consumes.
        #pragma unroll
        for (int tt = 0; tt < PGT / 8; ++tt) {
            const uint8_t* pc = p_rec + RO_C8_L + (long)(rr + 8 * tt) * RNK;
#if RNK == 8
            cc[tt] = __ldg(reinterpret_cast<const uint2*>(pc));
#elif RNK == 4
            cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(pc));
            cc[tt].y = 0u;
#else
            cc[tt].x = (unsigned)__ldg(
                reinterpret_cast<const unsigned short*>(pc));
            cc[tt].y = 0u;
#endif
            sc[tt] = *reinterpret_cast<const __nv_bfloat16*>(
                p_rec + RO_CS_L + (long)(rr + 8 * tt) * 2);
        }
        sv  = *reinterpret_cast<const __nv_bfloat16*>(
            p_rec + RO_VS_L + (long)vc * 2);
        smu = *reinterpret_cast<const __nv_bfloat16*>(p_rec + RO_MUS_L);
#else
        // rank campaign v2: the six-slab reader is rank-parametric like the
        // AOS one -- V/vs columns exist only for rr < RNK (clamp; score_tile
        // zeroes qt for rr >= RNK), and the per-token coeff stride is RNK
        // bytes (uint2 @r8, unsigned @r4, ushort @r2). RNK==8 compiles
        // verbatim the original loads.
        const int vcn = (rr < RNK) ? rr : (RNK - 1);
        const uint4* colv = reinterpret_cast<const uint4*>(
            v4 + (row * RNK + vcn) * (DHEAD / 2));
        #pragma unroll
        for (int c = 0; c < VW4; ++c) vv[c] = __ldg(colv + c);
        if constexpr (MUW % 4 == 0) {
            const uint4* mup = reinterpret_cast<const uint4*>(
                mu8 + row * DHEAD) + rr * (MUW / 4);
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(mup + j);
                mm[4 * j + 0] = m4.x; mm[4 * j + 1] = m4.y;
                mm[4 * j + 2] = m4.z; mm[4 * j + 3] = m4.w;
            }
        } else {
            const uint32_t* mup = reinterpret_cast<const uint32_t*>(
                mu8 + row * DHEAD) + rr * MUW;
            #pragma unroll
            for (int j = 0; j < MUW; ++j) mm[j] = __ldg(mup + j);
        }
        #pragma unroll
        for (int tt = 0; tt < PGT / 8; ++tt) {
            const int8_t* pc = c8 + (long)(row * PGT + rr + 8 * tt) * RNK;
#if RNK == 8
            cc[tt] = __ldg(reinterpret_cast<const uint2*>(pc));
#elif RNK == 4
            cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(pc));
            cc[tt].y = 0u;
#else
            cc[tt].x = (unsigned)__ldg(
                reinterpret_cast<const unsigned short*>(pc));
            cc[tt].y = 0u;
#endif
            sc[tt] = cs[row * PGT + rr + 8 * tt];
        }
        sv = vs[row * RNK + vcn];
        smu = mus[row];
#endif
    };
    // Per-page scoring: bytes are loaded ONCE (the register bank) and swept
    // by the g-tile loop (4 heads per sweep; the G4 instantiation compiles
    // the single pass).  The octet-uniform `active` mask only gates the S
    // write; all shfls stay warp-converged (the g-tile loop bound is
    // warp-uniform).
    auto compute_page = [&](int p, const uint4* vv, const uint32_t* mm,
                            const uint2* cc, __nv_bfloat16 sv,
                            __nv_bfloat16 smu, const __nv_bfloat16* sc) {
        const int ob = lane & 24;          // octet base lane (qt shfl source)
        auto score_tile = [&](int g, bool active) {
            // ---- phase 1: qt = <V[:,rr], q_g> via vsub4 + dp4a -------- //
            // phase 1 accumulation.
            int acci = 0;
            #pragma unroll
            for (int c = 0; c < VW4; ++c) {
                const uint32_t ws[4] = {vv[c].x, vv[c].y, vv[c].z, vv[c].w};
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    // 8 signed int4 -> two int8x4 words: (n^8)-8 per byte
                    const uint32_t lo = __vsub4(
                        (ws[k] & 0x0F0F0F0Fu) ^ 0x08080808u, 0x08080808u);
                    const uint32_t hi = __vsub4(
                        ((ws[k] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u,
                        0x08080808u);
                    acci = __dp4a((int)lo, (int)q8e[g][c * 4 + k], acci);
                    acci = __dp4a((int)hi, (int)q8o[g][c * 4 + k], acci);
                }
            }
            // RNK<8: lanes rr >= RNK hold no basis column (load_page clamped
            // their V/vs reads to a valid in-region column), so force their
            // projection to exactly 0 -- the token sum over the octet then
            // spans only the RNK live basis lanes, matching the oracle.  For
            // RNK==8 (rr in [0,8)) the predicate is compile-time always-true
            // -> folded away, r8 SASS verbatim.  Was AOS-only; now covers the
            // rank-parametric six-slab lane too (rank campaign v2).
            float qt = (rr < RNK) ? ((float)acci * qsc[g] * __bfloat162float(sv))
                                  : 0.f;
            // ---- phase 1b: mu dot, MUW dp4a + 3 xor-shfl -------------- //
            int mui = 0;
            #pragma unroll
            for (int j = 0; j < MUW; ++j)
                mui = __dp4a((int)mm[j], (int)q8d[g][rr * MUW + j], mui);
            float md = (float)mui;
            md += __shfl_xor_sync(~0u, md, 1);
            md += __shfl_xor_sync(~0u, md, 2);
            md += __shfl_xor_sync(~0u, md, 4);
            const float mud = md * qsc[g] * __bfloat162float(smu);
            // ---- phase 2: PGT/8 tokens per lane, qt via octet shfl ---- //
            float mx = -CUDART_INF_F, es = 0.f;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                float dt = 0.f;
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                        * __shfl_sync(~0u, qt, ob + u);
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                        * __shfl_sync(~0u, qt, ob + 4 + u);
                const float tok = sm_scale * (dt * __bfloat162float(sc[tt])
                    + mud);
                const float nm = fmaxf(mx, tok);
                es = es * __expf(mx - nm) + __expf(tok - nm);
                mx = nm;
            }
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1) {
                const float om = __shfl_xor_sync(~0u, mx, st);
                const float oe = __shfl_xor_sync(~0u, es, st);
                const float nm = fmaxf(mx, om);
                es = es * __expf(mx - nm) + oe * __expf(om - nm);
                mx = nm;
            }
            {
                const float osv = mx + __logf(es);
                if (rr == 0 && active)
                    S[(((long)r * n_kv + kh) * G + g) * MP + p] = osv;
                return osv;   // HMAX: call sites fold (or discard)
            }
        };
        if constexpr (G4) {
            score_tile(gl, gl < G);
        } else if constexpr (G8) {
            // compile-time G=8: exactly 2 full g-tiles, both ALWAYS active
            // (heads 0-3 and 4-7). No runtime loop counter, no active mask, no
            // variable bound -> fewer live registers -> 6 CTAs/SM (like G4).
            // Math + op order identical to the runtime loop (2 tiles) below ->
            // score_h bitwise-invariant.
            {   // HMAX (lever 1): fold the stored values in SSA registers
                const float hsa = score_tile(gl, true);
                const float hsb = score_tile(gl + 4, true);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa);
                    hm_b = fmaxf(hm_b, hsb);
                }
            }
        } else {
            for (int g0 = 0; g0 < G; g0 += 4)
                score_tile(g0 + gl, g0 + gl < G);
        }
    };
    int p = wid;
    if constexpr (USE_MMA) {
#ifdef RKI4_MMA
        // ----- int8 tensor-core projection loop (G8 d128 flagship) --------- //
        // Per page: (1) load this lane's mu/C/cs (rr=lane&7, for phase-1b + the
        // token reconstruction), (2) mma the whole GQA-group projection into
        // qt_sh, (3) run phase-1b (mu.q) + phase-2 (16-token LSE) for all 8
        // heads as two octet tiles -- byte-identical arithmetic to the dp4a
        // score_tile (same qt floats, same sum + LSE order).
        const int gidm = lane >> 2, t4m = lane & 3;   // mma lane map (proj)
        uint32_t mm[MUW]; uint2 cc[PGT / 8];
        __nv_bfloat16 smu, sc[PGT / 8];
#ifdef RKI4_MMA_SRED
        // SRED (strength-reduced addressing): dense rows advance by EXACTLY
        // nworker per iteration (row = pb_v + pp), so every summary pointer
        // advances by a constant byte stride, and the S store slot advances
        // by nworker floats in BOTH addressing modes (MP is fixed). Hoist
        // per-lane pointers once and increment per page, replacing the
        // per-page 64-bit index re-derivation (ncu/SASS 2026-07-16: the two
        // S stores alone re-evaluated (((r*n_kv+kh)*G+g)*MP+pp in ~11-inst
        // 64-bit chains EACH page). Addresses are identical by construction
        // -> same loads, same float op order -> score_h byte-identical
        // (gate: scratch_mma_score/sred_gate.py). BT keeps the per-page row
        // derivation for the summary loads (block-table gather is not affine
        // in pp) but still gets the S-store + smem-pointer reduction.
        float* qtw = &qt_sh[warp][gidm][2 * t4m];
        // ALLG: at the G4-template instantiation the b-octet heads (gl+4)
        // do not exist -- their qsc/q8d rows are uninitialized and their S
        // rows belong to the NEXT kv-head. BOFF=0 aliases the b-side decls
        // onto the (live) a-side so every read is safe; the b tile calls are
        // compile-time elided below. G8: BOFF=4 reproduces the source
        // verbatim -> flagship SASS unchanged.
        constexpr int BOFF = G8 ? 4 : 0;
        const float* qt_a = &qt_sh[warp][gl][0];
        const float* qt_b = &qt_sh[warp][gl + BOFF][0];
        const uint32_t* q8d_a = &q8d[gl][rr * MUW];
        const uint32_t* q8d_b = &q8d[gl + BOFF][rr * MUW];
        const float qsc_m = qsc[gidm];
        const float qsc_a = qsc[gl], qsc_b = qsc[gl + BOFF];
#ifdef RKI4_MMA_FLAT
        // FLAT (doc 18): per-head shift constants for the mass exps.
        // (BOFF clamps the b-read at the G4-template; b is elided there.)
        const float K_a = Kflat[kh * G + gl], K_b = Kflat[kh * G + gl + BOFF];
#else
        constexpr float K_a = 0.f, K_b = 0.f;
#endif
        float* Sp_a = S + ((((long)r * n_kv + kh) * G + gl) * MP) + wid;
        float* Sp_b = Sp_a + (long)BOFF * MP;
        const long row0 = BT ? 0L : (pb_v + wid);
        const uint4* p_mu = reinterpret_cast<const uint4*>(
            mu8 + row0 * DHEAD) + rr * (MUW / 4);
        const int8_t* p_c8 = c8 + (row0 * PGT + rr) * RNK;
        const __nv_bfloat16* p_cs = cs + row0 * PGT + rr;
        const __nv_bfloat16* p_mus = mus + row0;
        // rank v2: basis columns exist only for index < RNK -- clamp the dead
        // lanes' pointers in-bounds (their mma output cols are gated to 0 by
        // the #if RNK==8/#else epilogues). RNK==8: clamps fold, verbatim SASS.
        const int gvc = (gidm < RNK) ? gidm : (RNK - 1);
        const int svc = (2 * t4m < RNK) ? 2 * t4m : (RNK - 2 < 0 ? 0 : RNK - 2);
        const unsigned* p_v4 = reinterpret_cast<const unsigned*>(
            v4 + (row0 * RNK + gvc) * (DHEAD / 2)) + t4m;
        const __nv_bfloat16* p_vs = vs + row0 * RNK + svc;
        // tile: byte-identical arithmetic and op order to the shipped tile(g)
        // lambda below; only the qt/q8d/qsc/S accesses go through hoisted
        // per-lane pointers (same addresses, same values).
        auto tile = [&](const float* qtg, const uint32_t* q8dg,
                        const float qscg, float* Sp, const float Kg) {
#ifdef RKI4_MMA_MUC
            // MUC: md arrives via the mu mma column (integer-exact twin of
            // the IDP chain), broadcast through the dead qt slot.
            const float mud = qtg[RNK] * qscg * __bfloat162float(smu);
#else
            int mui = 0;
            #pragma unroll
            for (int j = 0; j < MUW; ++j)
                mui = __dp4a((int)mm[j], (int)q8dg[j], mui);
            float md = (float)mui;
            md += __shfl_xor_sync(~0u, md, 1);
            md += __shfl_xor_sync(~0u, md, 2);
            md += __shfl_xor_sync(~0u, md, 4);
            const float mud = md * qscg * __bfloat162float(smu);
#endif
#ifdef RKI4_MMA_FLAT
            // FLAT-MASS (doc 18): mass = sum_t exp(l_t - Kg). No per-page
            // max pass, no rescale chain, no log; Kg validity is guarded
            // select-side via pmax finiteness. tokv math is VERBATIM the
            // LSE2 pass-1 form; only the reduction after it changes.
            float esf = 0.f;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#ifdef RKI4_CP_NODT
                // CRITICAL-PATH PROBE (timing-only): kill the 8-FMA
                // reconstruct chain; one dependent op keeps the cc load live.
                float dt = (float)(int)cc[tt].x + qtg[0];
#else
                float dt = 0.f;
                #pragma unroll
                for (int u = 0; u < (RNK < 4 ? RNK : 4); ++u)
                    dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                        * qtg[u];
#if RNK == 8
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                        * qtg[4 + u];
#endif
#endif
                const float tok = sm_scale * (dt * __bfloat162float(sc[tt])
                    + mud);
#ifdef RKI4_CP_NOEXP
                // CRITICAL-PATH PROBE (timing-only, doc 19 pre-check):
                // exp -> add; keeps the tok dependence, removes the XU op.
                esf += (tok - Kg);
#else
                esf += __expf(tok - Kg);
#endif
            }
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1)
                esf += __shfl_xor_sync(~0u, esf, st);
            if (rr == 0) *Sp = esf;
            // FOLDR/HMAX: hand the final mass back so call sites fold the
            // per-group max in a plain SSA register (red-team ruling: no
            // captured-pointer accumulators -- a removed-arm precedent, see REFUTED_ARMS_INDEX.md).
            return esf;
#elif defined(RKI4_MMA_LSE2)
            // Two-pass LSE: pass 1 = pure max (FMNMX only), pass 2 = ONE exp
            // per token + plain butterfly sum. Drops the online-rescale exps
            // (16 -> 4 EX2/page) and the serial es-rescale FMUL/FFMA chains
            // (the dominant register-dependency 'wait' stall). The reduction
            // ORDER changes -> scores are value-equal (~ulp), NOT bitwise;
            // the contract is the page RANKING: gated by selection-identity
            // + value tolerance (sred_gate.py cmpv), not the byte gate.
            float tokv[PGT / 8];
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                float dt = 0.f;
                #pragma unroll
                for (int u = 0; u < (RNK < 4 ? RNK : 4); ++u)
                    dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                        * qtg[u];
#if RNK == 8
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                        * qtg[4 + u];
#endif
                tokv[tt] = sm_scale * (dt * __bfloat162float(sc[tt]) + mud);
            }
            float mx = tokv[0];
            #pragma unroll
            for (int tt = 1; tt < PGT / 8; ++tt) mx = fmaxf(mx, tokv[tt]);
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, st));
            float es = 0.f;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt)
                es += __expf(tokv[tt] - mx);
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1)
                es += __shfl_xor_sync(~0u, es, st);
            {   // HMAX: store from a temp and return it (es/mx uniform
                // post-butterfly -> identical bytes; log on all lanes).
                const float osv = mx + __logf(es);
                if (rr == 0) *Sp = osv;
                return osv;
            }
#else
            float mx = -CUDART_INF_F, es = 0.f;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                float dt = 0.f;
                #pragma unroll
                for (int u = 0; u < (RNK < 4 ? RNK : 4); ++u)
                    dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                        * qtg[u];
#if RNK == 8
                #pragma unroll
                for (int u = 0; u < 4; ++u)
                    dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                        * qtg[4 + u];
#endif
                const float tok = sm_scale * (dt * __bfloat162float(sc[tt])
                    + mud);
#ifdef RKI4_MMA_PEEL
                // Peel tt=0: from (mx=-inf, es=0) the update is EXACTLY
                // mx=tok, es = 0*expf(-inf)+expf(0) = 1.0 (each step exact
                // in fp32) -> skip 2 exps/tile. Resolved at compile time by
                // the unroll; byte-identical (same gate as SRED/BIAS).
                if (tt == 0) {
                    mx = tok; es = 1.f;
                } else
#endif
                {
                    const float nm = fmaxf(mx, tok);
                    es = es * __expf(mx - nm) + __expf(tok - nm);
                    mx = nm;
                }
            }
            #pragma unroll
            for (int st = 1; st < 8; st <<= 1) {
                const float om = __shfl_xor_sync(~0u, mx, st);
                const float oe = __shfl_xor_sync(~0u, es, st);
                const float nm = fmaxf(mx, om);
                es = es * __expf(mx - nm) + oe * __expf(om - nm);
                mx = nm;
            }
            {
                const float osv = mx + __logf(es);
                if (rr == 0) *Sp = osv;
                return osv;
            }
#endif  // RKI4_MMA_LSE2
        };
#ifdef RKI4_MMA_AOS
        // AOS record loop: identical bytes, identical float order; only the
        // ADDRESSES change (one record base + compile-time field offsets).
        {
        constexpr int  RO_MU  = RNK * (DHEAD / 2);
        // Field offsets + record size are FORMULAS of (RNK, DHEAD, PGT),
        // 64B-aligned; MUST mirror rki4_state._rec_bytes_aos / the view
        // slicing (r8: 832B at p16 d128, r4: 512B, r2: 384B).
        constexpr int  RO_C8  = RO_MU + DHEAD;
        constexpr int  RO_CS  = RO_C8 + RNK * PGT;
        constexpr int  RO_VS  = RO_CS + 2 * PGT;
        constexpr int  RO_MUS = RO_VS + 2 * RNK;
        constexpr long RECB   = ((RO_MUS + 2) + 63) / 64 * 64;
        const uint8_t* p_rec = v4 + (BT ? 0L : (pb_v + wid)) * RECB;
        // BT row clamp: the engine's memory-profile pass scores a placeholder
        // 2-block state against a full-length fake block table. The six-slab
        // path silently read mapped garbage there (outputs discarded); the
        // compact record tensor FAULTS on the same OOB rows. Clamp to the
        // record slab (host passes NB via the dense-only P_dense slot);
        // every VALID row is untouched -> bitwise gates unaffected.
        const long rmax_rec = BT ? ((long)P_dense * n_kv - 1) : 0L;
#ifdef RKI4_MMA_PFR
        // PFR (2026-07-17, ncu at the z-wave operating point: long_scoreboard
        // 4.35 dominant, warps_active 44%): 2-deep cp.async ring of the FULL
        // record per (warp, slot). Every per-lane record read below hits smem
        // at the SAME field offsets -> identical bytes, bitwise-equal scores;
        // each page's record gains a full page-body of load cover. smem cost
        // NW*2*RECB (6.7KB at 832B) keeps 8 CTAs/SM resident (one wave).
        {
        constexpr int RC16 = (int)(RECB / 16);
        __shared__ uint4 recring[NW][2][RC16];
        auto issue_rec = [&](int pv, int slot) {
            if (pv < P) {
                long rowb;
                if constexpr (BT) {
                    rowb = row_of(pv);
                    rowb = rowb < 0 ? 0 : (rowb > rmax_rec ? rmax_rec : rowb);
                } else {
                    rowb = pb_v + pv;
                }
                const uint4* r16 = reinterpret_cast<const uint4*>(
                    v4 + rowb * RECB);
                for (int c = lane; c < RC16; c += 32)
                    cpasync16(&recring[warp][slot][c], r16 + c);
            }
            cpasync_commit();
        };
        issue_rec(wid, 0);
        issue_rec(wid + nworker, 1);
        int pslot = 0;
        for (int pp = wid; pp < P; pp += nworker) {
            cpasync_wait<1>();
            __syncwarp();
            const uint8_t* p_smem = reinterpret_cast<const uint8_t*>(
                &recring[warp][pslot][0]);
            // (1) mu / C / cs / mus from the staged record (plain smem loads;
            // same offsets and bytes as the global path)
#ifndef RKI4_MMA_MUC
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = *(reinterpret_cast<const uint4*>(
                    p_smem + RO_MU) + rr * (MUW / 4) + j);
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
#endif
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                cc[tt] = *(reinterpret_cast<const uint2*>(
                    p_smem + RO_C8 + (rr + 8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = *(reinterpret_cast<const unsigned*>(
                    p_smem + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)*(reinterpret_cast<const unsigned short*>(
                    p_smem + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = *reinterpret_cast<const __nv_bfloat16*>(
                    p_smem + RO_CS + (rr + 8 * tt) * 2);
            }
            smu = *reinterpret_cast<const __nv_bfloat16*>(p_smem + RO_MUS);
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_MUC
            const bool ismu = (gidm == RNK);
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_smem + (ismu ? RO_MU : gidm * (DHEAD / 2)));
#else
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_smem + gidm * (DHEAD / 2));
#endif
            const unsigned cw0 = cwp[t4m],      cw1 = cwp[4 + t4m],
                           cw2 = cwp[8 + t4m],  cw3 = cwp[12 + t4m];
#ifdef RKI4_MMA_MUC
            const unsigned cm4 = ismu ? cwp[16 + t4m] : 0u,
                           cm5 = ismu ? cwp[20 + t4m] : 0u,
                           cm6 = ismu ? cwp[24 + t4m] : 0u,
                           cm7 = ismu ? cwp[28 + t4m] : 0u;
#endif
#ifdef RKI4_MMA_BIAS
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
#else
            b[0] = r8_lo(cw0); b[1] = r8_lo(cw1); r8_mma_s8(acc, Af[0], b);
            b[0] = r8_lo(cw2); b[1] = r8_lo(cw3); r8_mma_s8(acc, Af[1], b);
            b[0] = r8_hi(cw0); b[1] = r8_hi(cw1); r8_mma_s8(acc, Af[2], b);
            b[0] = r8_hi(cw2); b[1] = r8_hi(cw3); r8_mma_s8(acc, Af[3], b);
#if RNK == 8
            const int c0 = acc[0], c1 = acc[1];
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] : 0;
#endif
#endif
#ifdef RKI4_MMA_S1PROBE
            // TIMING PROBE ONLY (two-stage split S1 floor): store the raw
            // int32 mma coefficients instead of running LSE/tiles. Output is
            // NOT score_h semantics -- never gate, never ship; the probe
            // prices stage-1 of the proposed split (doc sec 17).
            Sp_a[0] = (float)c0;
            if constexpr (G8) Sp_b[0] = (float)c1;   // ALLG G4: no b rows
            (void)qsc_m;
            issue_rec(pp + 2 * nworker, pslot);
#else
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                p_smem + RO_VS + 4 * t4m);
#ifdef RKI4_MMA_MUC
            qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                     : (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                     : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
            __syncwarp();
#ifdef RKI4_MMA_FOLDR
            fold_a = fmaxf(fold_a, tile(qt_a, q8d_a, qsc_a, Sp_a, K_a));
            if constexpr (G8)
                fold_b = fmaxf(fold_b, tile(qt_b, q8d_b, qsc_b, Sp_b, K_b));
#else
            tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
            if constexpr (G8)          // ALLG G4: b-octet heads do not exist
                tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
#endif
            issue_rec(pp + 2 * nworker, pslot);
#endif  // RKI4_MMA_S1PROBE
            pslot ^= 1;
            Sp_a += nworker; Sp_b += nworker;
            __syncwarp();      // qt_sh reuse next page
        }
        cpasync_wait<0>();
        }
#else
        for (int pp = wid; pp < P; pp += nworker) {
            if constexpr (BT) {
                long rowb = row_of(pp);
                rowb = rowb < 0 ? 0 : (rowb > rmax_rec ? rmax_rec : rowb);
                p_rec = v4 + rowb * RECB;
            }
            // (1) mu / C / cs / mus from the record
#ifndef RKI4_MMA_MUC
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(reinterpret_cast<const uint4*>(
                    p_rec + RO_MU) + rr * (MUW / 4) + j);
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
#endif
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                    p_rec + RO_C8 + (rr + 8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                    p_rec + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)__ldg(reinterpret_cast<const unsigned short*>(
                    p_rec + RO_C8 + (rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = *reinterpret_cast<const __nv_bfloat16*>(
                    p_rec + RO_CS + (rr + 8 * tt) * 2);
            }
            smu = *reinterpret_cast<const __nv_bfloat16*>(p_rec + RO_MUS);
            // (2) mma projection from the record's basis block
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_MUC
            const bool ismu = (gidm == RNK);
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_rec + (ismu ? RO_MU : gidm * (DHEAD / 2)));
#else
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                p_rec + gidm * (DHEAD / 2));
#endif
            const unsigned cw0 = __ldg(cwp + t4m),      cw1 = __ldg(cwp + 4 + t4m),
                           cw2 = __ldg(cwp + 8 + t4m),  cw3 = __ldg(cwp + 12 + t4m);
#ifdef RKI4_MMA_MUC
            const unsigned cm4 = ismu ? __ldg(cwp + 16 + t4m) : 0u,
                           cm5 = ismu ? __ldg(cwp + 20 + t4m) : 0u,
                           cm6 = ismu ? __ldg(cwp + 24 + t4m) : 0u,
                           cm7 = ismu ? __ldg(cwp + 28 + t4m) : 0u;
#endif
#ifdef RKI4_MMA_BIAS
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
#else
            b[0] = r8_lo(cw0); b[1] = r8_lo(cw1); r8_mma_s8(acc, Af[0], b);
            b[0] = r8_lo(cw2); b[1] = r8_lo(cw3); r8_mma_s8(acc, Af[1], b);
            b[0] = r8_hi(cw0); b[1] = r8_hi(cw1); r8_mma_s8(acc, Af[2], b);
            b[0] = r8_hi(cw2); b[1] = r8_hi(cw3); r8_mma_s8(acc, Af[3], b);
#if RNK == 8
            const int c0 = acc[0], c1 = acc[1];
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] : 0;
#endif
#endif
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                p_rec + RO_VS + 4 * t4m);
#ifdef RKI4_MMA_MUC
            qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                     : (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                     : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
            __syncwarp();
            // (3) tiles (identical)
#ifdef RKI4_MMA_FOLDR
            fold_a = fmaxf(fold_a, tile(qt_a, q8d_a, qsc_a, Sp_a, K_a));
            if constexpr (G8)
                fold_b = fmaxf(fold_b, tile(qt_b, q8d_b, qsc_b, Sp_b, K_b));
#else
            tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
            if constexpr (G8)          // ALLG G4: b-octet heads do not exist
                tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
#endif
            if constexpr (!BT) p_rec += (long)nworker * RECB;
            Sp_a += nworker; Sp_b += nworker;
            __syncwarp();      // qt_sh reuse next page
        }
#endif  // RKI4_MMA_PFR
        }
#else   // !RKI4_MMA_AOS: six-array addressing
#if defined(LOCKS_RANKMAP) && RNK < 8
        // ================= LEVER B: rank-tuned MLP map ==================== //
        // Confirmed lever (PF probe): v4->mma is the EXPOSED critical-path load
        // (long_scoreboard 0.42; prefetch hides it). Unroll-and-jam KJ pages/
        // warp: HOIST all KJ pages' v4 loads together (latencies overlap), then
        // mma+tile each page serially (reuse qt_sh). Page 0's v4 gates its mma;
        // pages 1..KJ-1 see their v4 latency HIDDEN under the earlier pages'
        // mma+tile -> memory-level parallelism depth KJ. Lower rank -> deeper KJ
        // (fewer real basis => more pages affordable). Fold in the dead-lane v4
        // SKIP: only gidm<RNK issue the v4 load (dead mma cols are epilogue-
        // masked to 0), cutting r2's v4 requests to 2/8. Each page's mma+tile is
        // byte-identical to the deployed loop, just interleaved -> score_h
        // BITWISE (r8_lo == BIAS acc-qsum). KJ tunable via LOCKS_RANKMAP_KJ.
        (void)Sp_a; (void)Sp_b; (void)p_mu; (void)p_c8; (void)p_cs;
        (void)p_mus; (void)p_v4; (void)p_vs; (void)qtw; (void)PF;
#ifdef LOCKS_RANKMAP_KJ
        constexpr int KJ = LOCKS_RANKMAP_KJ;           // grind override (-D)
#else
        // KJ=1 the measured sweet spot (in-situ 256k h03, r2): the entire win
        // is the dead-lane v4 SKIP (r2 -6.96%, 48 regs -> 10 blocks/SM), NOT the
        // unroll-jam MLP -- at 256k (many waves) v4 latency is already hidden
        // inter-warp so MLP is redundant, and deeper KJ only costs registers
        // (KJ=2 1.2663 @9 blocks, KJ>=4 REGRESSES +24% @7 blocks). r4's 4/8 skip
        // is too small to beat the loop overhead (RANKMAP not beneficial at r4).
        constexpr int KJ = 1;                          // default pages/iter
#endif
        for (int pp = wid; pp < P; pp += KJ * nworker) {
            long rowk[KJ];
            unsigned cwk[KJ][4];
            bool hask[KJ];
            #pragma unroll
            for (int k = 0; k < KJ; ++k) {
                const int ppk = pp + k * nworker;
                hask[k] = (ppk < P);                   // warp-uniform
                rowk[k] = hask[k] ? row_of(ppk) : 0L;
                // dead-lane v4 SKIP + hoisted load (MLP): only gidm<RNK load.
                if (gidm < RNK && hask[k]) {
                    const unsigned* pv = reinterpret_cast<const unsigned*>(
                        v4 + (rowk[k] * RNK + gidm) * (DHEAD / 2)) + t4m;
                    cwk[k][0] = __ldg(pv);      cwk[k][1] = __ldg(pv + 4);
                    cwk[k][2] = __ldg(pv + 8);  cwk[k][3] = __ldg(pv + 12);
                } else {
                    cwk[k][0] = 0u; cwk[k][1] = 0u; cwk[k][2] = 0u; cwk[k][3] = 0u;
                }
            }
            const long Sbase = (((long)r * n_kv + kh) * G + gl) * MP;
            #pragma unroll
            for (int k = 0; k < KJ; ++k) {
                if (!hask[k]) continue;                 // warp-uniform tail
                const long row = rowk[k];
                const uint4* q_mu = reinterpret_cast<const uint4*>(
                    mu8 + row * DHEAD) + rr * (MUW / 4);
                #pragma unroll
                for (int j = 0; j < MUW / 4; ++j) {
                    const uint4 m4 = __ldg(q_mu + j);
                    mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
                }
                const int8_t* q_c8 = c8 + (row * PGT + rr) * RNK;
                const __nv_bfloat16* q_cs = cs + row * PGT + rr;
                #pragma unroll
                for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 4
                    cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                        q_c8 + (8 * tt) * RNK));
#else
                    cc[tt].x = (unsigned)__ldg(reinterpret_cast<
                        const unsigned short*>(q_c8 + (8 * tt) * RNK));
#endif
                    cc[tt].y = 0u;
                    sc[tt] = q_cs[8 * tt];
                }
                smu = mus[row];
                int acc[4] = {0, 0, 0, 0};
                unsigned b[2];
                b[0]=r8_lo(cwk[k][0]); b[1]=r8_lo(cwk[k][1]); r8_mma_s8(acc, Af[0], b);
                b[0]=r8_lo(cwk[k][2]); b[1]=r8_lo(cwk[k][3]); r8_mma_s8(acc, Af[1], b);
                b[0]=r8_hi(cwk[k][0]); b[1]=r8_hi(cwk[k][1]); r8_mma_s8(acc, Af[2], b);
                b[0]=r8_hi(cwk[k][2]); b[1]=r8_hi(cwk[k][3]); r8_mma_s8(acc, Af[3], b);
                const int c0 = (2 * t4m + 0 < RNK) ? acc[0] : 0;
                const int c1 = (2 * t4m + 1 < RNK) ? acc[1] : 0;
                const int svc2 = (2 * t4m < RNK) ? 2 * t4m : 0;
                const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                    vs + row * RNK + svc2);
                qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
                qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
                __syncwarp();
                float* Spa = S + Sbase + (pp + k * nworker);
                const float hsa = tile(qt_a, q8d_a, qsc_a, Spa, K_a);
                float hsb = -CUDART_INF_F;
                if constexpr (G8)
                    hsb = tile(qt_b, q8d_b, qsc_b, Spa + (long)BOFF * MP, K_b);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa); hm_b = fmaxf(hm_b, hsb);
                }
                __syncwarp();
            }
        }
#elif defined(RKI4_PACK)
        // ============ r4 PAGE-PACK: 2 pages/warp fill all 8 mma cols ======= //
        // The m16n8k32 mma has n=8 basis columns FIXED by the instruction, so a
        // rank-4 summary leaves 4 DEAD columns and issues the SAME mma as rank
        // 8 -- the rank-independent compute wall that makes r4 no faster than
        // r8. PACK loads page0's 4 basis into cols 0-3 (gidm<RNK) and page1's 4
        // basis into cols 4-7 (gidm>=RNK); the shared query A-fragment (Af,
        // rows=heads) projects BOTH pages in ONE mma. Output col j = <q_head,
        // basis-col-j> lands at qt_sh[head][j] verbatim -- cols 0-3 -> page0,
        // 4-7 -> page1 -- so the write pointer (qtw = &qt_sh[warp][gidm][2*t4m])
        // is unchanged; the tile reads page0 at qt base +0 and page1 at +RNK.
        // int8*int8 mma is exact and column-order-invariant, so each page's
        // score_h is BITWISE the rank-4 single-page value (gate_pack_mma.py).
        // Minimal six-slab variant only (no BIAS/PEEL/LSE2/MUC/FLAT/AOS/PFR).
#if RNK != 4
#error "RKI4_PACK: the 2-page pack requires RNK==4 (8/RNK==2 pages/warp)"
#endif
        (void)Sp_a; (void)Sp_b; (void)p_mu; (void)p_c8; (void)p_cs;
        (void)p_mus; (void)p_v4; (void)p_vs; (void)gvc; (void)svc; (void)PF;
        for (int pp = wid; pp < P; pp += 2 * nworker) {
            const int pp0 = pp, pp1 = pp + nworker;
            const bool has1 = (pp1 < P);          // warp-uniform (pp,P uniform)
            const long row0 = row_of(pp0);
            const long row1 = has1 ? row_of(pp1) : row0;
            // (2) mma projection: page0 basis -> cols 0-3, page1 -> cols 4-7.
            const long rowB  = (gidm < RNK) ? row0 : row1;
            const int  bgidm = (gidm < RNK) ? gidm : (gidm - RNK);
            const unsigned* pB = reinterpret_cast<const unsigned*>(
                v4 + (rowB * RNK + bgidm) * (DHEAD / 2)) + t4m;
            const unsigned cw0 = __ldg(pB),     cw1 = __ldg(pB + 4),
                           cw2 = __ldg(pB + 8), cw3 = __ldg(pB + 12);
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
            b[0] = r8_lo(cw0); b[1] = r8_lo(cw1); r8_mma_s8(acc, Af[0], b);
            b[0] = r8_lo(cw2); b[1] = r8_lo(cw3); r8_mma_s8(acc, Af[1], b);
            b[0] = r8_hi(cw0); b[1] = r8_hi(cw1); r8_mma_s8(acc, Af[2], b);
            b[0] = r8_hi(cw2); b[1] = r8_hi(cw3); r8_mma_s8(acc, Af[3], b);
            const int c0 = acc[0], c1 = acc[1];   // all 8 cols REAL -> no mask
            // V-scale + qt col: cols 0-3 (t4m<2) page0, cols 4-7 page1.
            const bool colP1 = (2 * t4m >= RNK);
            const long rowV  = colP1 ? row1 : row0;
            const int  bcol  = colP1 ? (2 * t4m - RNK) : (2 * t4m);
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                vs + rowV * RNK + bcol);
            // output col = 2*t4m (0-7): natural layout, cols 0-3 in [head][0..3]
            // (page0), 4-7 in [head][4..7] (page1). qtw is unchanged.
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
            __syncwarp();
            const long Sbase = (((long)r * n_kv + kh) * G + gl) * MP;
            // (3a) page0 tokens (row0) + tile -> S[pp0]
            {
                const uint4* q_mu = reinterpret_cast<const uint4*>(
                    mu8 + row0 * DHEAD) + rr * (MUW / 4);
                #pragma unroll
                for (int j = 0; j < MUW / 4; ++j) {
                    const uint4 m4 = __ldg(q_mu + j);
                    mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
                }
                const int8_t* q_c8 = c8 + (row0 * PGT + rr) * RNK;
                const __nv_bfloat16* q_cs = cs + row0 * PGT + rr;
                #pragma unroll
                for (int tt = 0; tt < PGT / 8; ++tt) {
                    cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                        q_c8 + (8 * tt) * RNK));
                    cc[tt].y = 0u;
                    sc[tt] = q_cs[8 * tt];
                }
                smu = mus[row0];
                float* S0a = S + Sbase + pp0;
                const float hsa = tile(qt_a, q8d_a, qsc_a, S0a, K_a);
                float hsb = -CUDART_INF_F;
                if constexpr (G8)
                    hsb = tile(qt_b, q8d_b, qsc_b, S0a + (long)BOFF * MP, K_b);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa); hm_b = fmaxf(hm_b, hsb);
                }
            }
            // (3b) page1 tokens (row1) + tile -> S[pp1]  (skip the odd tail)
            if (has1) {
                const uint4* q_mu = reinterpret_cast<const uint4*>(
                    mu8 + row1 * DHEAD) + rr * (MUW / 4);
                #pragma unroll
                for (int j = 0; j < MUW / 4; ++j) {
                    const uint4 m4 = __ldg(q_mu + j);
                    mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
                }
                const int8_t* q_c8 = c8 + (row1 * PGT + rr) * RNK;
                const __nv_bfloat16* q_cs = cs + row1 * PGT + rr;
                #pragma unroll
                for (int tt = 0; tt < PGT / 8; ++tt) {
                    cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                        q_c8 + (8 * tt) * RNK));
                    cc[tt].y = 0u;
                    sc[tt] = q_cs[8 * tt];
                }
                smu = mus[row1];
                float* S1a = S + Sbase + pp1;
                const float hsa = tile(qt_a + RNK, q8d_a, qsc_a, S1a, K_a);
                float hsb = -CUDART_INF_F;
                if constexpr (G8)
                    hsb = tile(qt_b + RNK, q8d_b, qsc_b,
                               S1a + (long)BOFF * MP, K_b);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa); hm_b = fmaxf(hm_b, hsb);
                }
            }
            __syncwarp();      // qt_sh reuse next iteration
        }
#else
        // ---- Fix 1: depth-1 software pipeline of the dominant v4 load ----- //
        // ncu (DEPLOYED kernel, 256K/bs4): long_scoreboard 6.22 (~4x the next
        // stall), issue_active 63%, DRAM 27% SOL, occupancy 47% @ 1 wave,
        // 64 regs -> 8 blocks/SM. GLOBAL-LOAD-LATENCY bound with occupancy
        // headroom; the stall carrier is the per-page v4 read (cw0..cw3).
        // Double-buffer THAT ONE load: prefetch page (pp+nworker)'s v4 into a
        // register bank while page pp's mma + LSE runs, hiding the ~470-cycle
        // latency under compute. int8*int8 mma is exact + associative, so
        // moving WHEN the load happens leaves the score VALUE (hence the page
        // RANKING) byte-identical -- the ONLY correctness risk is the bank
        // rotation, handled by consuming page pp's bank (written last iter /
        // in the prologue) and never prefetching past P. The +4 bank registers
        // are FUNDED by spilling the hoisted Af[4][4] query fragment back to
        // per-page reads of the smem-resident q8e/q8o (Af is loop-invariant;
        // smem reads are cheap and not long_scoreboard). Guarded to the
        // DEPLOYED instantiation only (BT && PGT==16, within USE_MMA = G8 &&
        // d128); every other instantiation compiles the ORIGINAL loop below,
        // byte- and perf-identical.
        // LAUNCH-TIME GATE (PF): the host instantiates BOTH PF=false (this
        // kernel == the byte-identical deployed loop, taken by every regime
        // where the prefetch does NOT strictly win -- all bs=1 ctx and the
        // bs<=4 heavy-zsplit regime) and PF=true (the prefetch, dispatched only
        // for n_req >= LOCKS_FIX1_PF_MINREQ where it wins). PF is a non-type
        // template param, so PF=false register/SASS == deployed exactly ->
        // zero regression there BY CONSTRUCTION.
        if constexpr (BT && PGT == 16 && PF) {
            unsigned nv0 = 0u, nv1 = 0u, nv2 = 0u, nv3 = 0u;   // v4 prefetch bank
#ifdef LOCKS_FIX1_CARRYROW
            // Fix1b (flag, default OFF): CARRY the BT row forward -- gather
            // row_of(pp) ONCE and reuse for this page's pointers, removing the
            // duplicate per-page bt gather. MEASURED to cost registers
            // (56->69) -> 7 blocks/SM, so it is a knob, not the default.
            long cur_row = (wid < P) ? row_of(wid) : 0L;
            if (wid < P) {
                const unsigned* pf = reinterpret_cast<const unsigned*>(
                    v4 + (cur_row * RNK + ((gidm < RNK) ? gidm : (RNK - 1)))
                        * (DHEAD / 2)) + t4m;  // rank v2 clamp
                nv0 = __ldg(pf);      nv1 = __ldg(pf + 4);
                nv2 = __ldg(pf + 8);  nv3 = __ldg(pf + 12);
            }
#else
            if (wid < P) {                    // prologue: prefetch page = wid
                const long rowp = row_of(wid);
                const unsigned* pf = reinterpret_cast<const unsigned*>(
                    v4 + (rowp * RNK + gidm) * (DHEAD / 2)) + t4m;
                nv0 = __ldg(pf);      nv1 = __ldg(pf + 4);
                nv2 = __ldg(pf + 8);  nv3 = __ldg(pf + 12);
            }
#endif
            for (int pp = wid; pp < P; pp += nworker) {
#ifdef LOCKS_FIX1_CARRYROW
                const long row = cur_row;      // carried (no per-page bt gather)
#else
                const long row = row_of(pp);   // BT gather: not affine in pp
#endif
                p_mu = reinterpret_cast<const uint4*>(
                    mu8 + row * DHEAD) + rr * (MUW / 4);
                p_c8 = c8 + (row * PGT + rr) * RNK;
                p_cs = cs + row * PGT + rr;
                p_mus = mus + row;
                p_vs = vs + row * RNK + svc;               // rank v2 clamp
                // (1) per-(lane=rr) mu / C / cs / mus loads (same addresses)
                #pragma unroll
                for (int j = 0; j < MUW / 4; ++j) {
                    const uint4 m4 = __ldg(p_mu + j);
                    mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
                }
                #pragma unroll
                for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                    cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                        p_c8 + (8 * tt) * RNK));
#elif RNK == 4
                    cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                        p_c8 + (8 * tt) * RNK));
                    cc[tt].y = 0u;
#else
                    cc[tt].x = (unsigned)__ldg(reinterpret_cast<
                        const unsigned short*>(p_c8 + (8 * tt) * RNK));
                    cc[tt].y = 0u;
#endif
                    sc[tt] = p_cs[8 * tt];
                }
                smu = *p_mus;
                // (2) consume THIS page's v4 from the bank, THEN issue the NEXT
                //     page's v4 into the bank so its latency overlaps this
                //     page's mma + LSE. Snapshot precedes the overwrite (WAR):
                //     cw0..cw3 are page pp's bytes, written for pp last
                //     iteration (or the prologue at pp==wid). ppn<P guards the
                //     final page: it consumes without an out-of-range prefetch.
                const unsigned cw0 = nv0, cw1 = nv1, cw2 = nv2, cw3 = nv3;
                const int ppn = pp + nworker;
                if (ppn < P) {
#ifdef LOCKS_FIX1_CARRYROW
                    cur_row = row_of(ppn);     // the ONE bt gather this iteration
                    const unsigned* pf = reinterpret_cast<const unsigned*>(
                        v4 + (cur_row * RNK + ((gidm < RNK) ? gidm : (RNK - 1)))
                        * (DHEAD / 2)) + t4m;  // rank v2 clamp
#else
                    const long rown = row_of(ppn);
                    const unsigned* pf = reinterpret_cast<const unsigned*>(
                        v4 + (rown * RNK + gidm) * (DHEAD / 2)) + t4m;
#endif
                    nv0 = __ldg(pf);      nv1 = __ldg(pf + 4);
                    nv2 = __ldg(pf + 8);  nv3 = __ldg(pf + 12);
                }
                // Af source -- register/occupancy knob (measured per ctx; the
                // 256K/bs4 grid is ~1 wave, 128K/bs8 ~0.5 wave, so the tradeoff
                // differs by ctx). Both branches feed the mma bit-identical
                // A-fragments (== the hoisted Af[0..3]).
#ifdef LOCKS_FIX1_AF_HOIST
                // variant A: HOISTED Af, no per-page smem re-read (less O(P)
                // work) at the cost of registers (may drop to 7 blocks/SM).
                const unsigned (&A0)[4] = Af[0];
                const unsigned (&A1)[4] = Af[1];
                const unsigned (&A2)[4] = Af[2];
                const unsigned (&A3)[4] = Af[3];
#else
                // variant B: SPILL Af to per-page smem reads (frees registers ->
                // 9 blocks/SM), small O(P) smem re-read per page.
                unsigned A0[4], A1[4], A2[4], A3[4];
                A0[0]=q8e[gidm][t4m];      A0[1]=0u; A0[2]=q8e[gidm][4 + t4m];  A0[3]=0u;
                A1[0]=q8e[gidm][8 + t4m];  A1[1]=0u; A1[2]=q8e[gidm][12 + t4m]; A1[3]=0u;
                A2[0]=q8o[gidm][t4m];      A2[1]=0u; A2[2]=q8o[gidm][4 + t4m];  A2[3]=0u;
                A3[0]=q8o[gidm][8 + t4m];  A3[1]=0u; A3[2]=q8o[gidm][12 + t4m]; A3[3]=0u;
#endif
                int acc[4] = {0, 0, 0, 0};
                unsigned b[2];
#ifdef RKI4_MMA_BIAS
#ifdef RKI4_MMA_MUC
                b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A0, b);
                b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A1, b);
                b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A2, b);
                b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
                b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, A3, b);
#else
                b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A0, b);
                b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A1, b);
                b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A2, b);
                b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
                b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, A3, b);
#endif
#if RNK == 8
                const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
                const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                               : (2 * t4m + 0 == RNK ? acc[0] : 0);
                const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                               : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
                const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
                const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
#else
                b[0] = r8_lo(cw0); b[1] = r8_lo(cw1); r8_mma_s8(acc, A0, b);
                b[0] = r8_lo(cw2); b[1] = r8_lo(cw3); r8_mma_s8(acc, A1, b);
                b[0] = r8_hi(cw0); b[1] = r8_hi(cw1); r8_mma_s8(acc, A2, b);
                b[0] = r8_hi(cw2); b[1] = r8_hi(cw3); r8_mma_s8(acc, A3, b);
#if RNK == 8
                const int c0 = acc[0], c1 = acc[1];
#else
                const int c0 = (2 * t4m + 0 < RNK) ? acc[0] : 0;
                const int c1 = (2 * t4m + 1 < RNK) ? acc[1] : 0;
#endif
#endif
                const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(p_vs);
#ifdef RKI4_MMA_MUC
                qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                         : (float)c0 * qsc_m * __bfloat162float(sv2.x);
                qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                         : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
                qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
                qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
                __syncwarp();
                // (3) phase-1b (mu.q) + phase-2 (LSE) per octet head
                {   // HMAX (lever 1): fold the stored values in SSA registers
                    const float hsa = tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
                    const float hsb = tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
                    if (hmax_pub != nullptr) {
                        hm_a = fmaxf(hm_a, hsa);
                        hm_b = fmaxf(hm_b, hsb);
                    }
                }
                Sp_a += nworker; Sp_b += nworker;
                __syncwarp();      // qt_sh reuse next page
            }
        } else {
        for (int pp = wid; pp < P; pp += nworker) {
            if constexpr (BT) {
                const long row = row_of(pp);
                p_mu = reinterpret_cast<const uint4*>(
                    mu8 + row * DHEAD) + rr * (MUW / 4);
                p_c8 = c8 + (row * PGT + rr) * RNK;
                p_cs = cs + row * PGT + rr;
                p_mus = mus + row;
                p_v4 = reinterpret_cast<const unsigned*>(
                    v4 + (row * RNK + gvc) * (DHEAD / 2)) + t4m;   // rank v2 clamp
                p_vs = vs + row * RNK + svc;
            }
            // (1) per-(lane=rr) mu / C / cs / mus loads (same addresses)
            // PROBE (diagnostic, default-OFF): LOCKS_PROBE_SKIP_{MU,C8,V4}
            // replace one slab's global loads with 0 so ncu t_requests reveals
            // that slab's share of the (rank-independent) load-instruction
            // count. RNK==8 SASS byte-identical when no probe flag is set.
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
#ifdef LOCKS_PROBE_SKIP_MU
                const uint4 m4 = make_uint4(0u, 0u, 0u, 0u);
#else
                const uint4 m4 = __ldg(p_mu + j);
#endif
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
#if defined(LOCKS_C8RELAY) && RNK == 2
            // C' (RNK==2): the build (rki4_build _write_post, gated
            // LOCKS_C8RELAY) relaid c8 so lane rr's two tokens (rr, rr+8) sit
            // ADJACENTLY at positions 2rr, 2rr+1. ONE aligned 4-byte load
            // (p_c8 + rr*RNK == c8 row + (2rr)*RNK) fetches BOTH -> HALVES the
            // c8 requests (the count-cut lever, not width). tt=0 -> low 16 bits
            // (token rr), tt=1 -> high (token rr+8). Same codes/order = bitwise.
            const unsigned c8relw = __ldg(
                reinterpret_cast<const unsigned*>(p_c8 + rr * RNK));
#endif
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#ifdef LOCKS_PROBE_SKIP_C8
                cc[tt].x = 0u; cc[tt].y = 0u;
#elif RNK == 8
                cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                    p_c8 + (8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                    p_c8 + (8 * tt) * RNK));
                cc[tt].y = 0u;
#elif defined(LOCKS_C8RELAY)
                cc[tt].x = (c8relw >> (tt * 16)) & 0xFFFFu;
                cc[tt].y = 0u;
#elif defined(LOCKS_C8WIDE)
                // LEVER C (RNK==2): widen the 2-byte sub-word c8 code load to
                // an ALIGNED 4-byte word + extract this token's 2 codes (low if
                // the token index is even, high if odd). Removes the LDG.U16
                // sub-word penalty (r2 = 17.6% of kernel time, top slab). No
                // overread -- token 15 is odd so its word = tokens 14,15,
                // in-page. Same codes, same lane->token, same LSE order ->
                // BITWISE identical (gate_pack_mma --flag LOCKS_C8WIDE).
                {
                    const int tok = rr + 8 * tt;
                    const unsigned w = __ldg(reinterpret_cast<const unsigned*>(
                        p_c8 + (8 * tt) * RNK - (tok & 1) * RNK));
                    cc[tt].x = (w >> ((tok & 1) * 16)) & 0xFFFFu;
                }
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)__ldg(reinterpret_cast<
                    const unsigned short*>(p_c8 + (8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = p_cs[8 * tt];
            }
            smu = *p_mus;
            // (2) mma projection: qt[head][basis] for all 8x8 -> qt_sh
#ifdef LOCKS_PROBE_SKIP_V4
            const unsigned cw0 = 0u, cw1 = 0u, cw2 = 0u, cw3 = 0u;
#else
            const unsigned cw0 = __ldg(p_v4),      cw1 = __ldg(p_v4 + 4),
                           cw2 = __ldg(p_v4 + 8),  cw3 = __ldg(p_v4 + 12);
#endif
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_BIAS
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif
#else
            b[0] = r8_lo(cw0); b[1] = r8_lo(cw1); r8_mma_s8(acc, Af[0], b);
            b[0] = r8_lo(cw2); b[1] = r8_lo(cw3); r8_mma_s8(acc, Af[1], b);
            b[0] = r8_hi(cw0); b[1] = r8_hi(cw1); r8_mma_s8(acc, Af[2], b);
            b[0] = r8_hi(cw2); b[1] = r8_hi(cw3); r8_mma_s8(acc, Af[3], b);
#if RNK == 8
            const int c0 = acc[0], c1 = acc[1];
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] : 0;
#endif
#endif
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(p_vs);
#ifdef RKI4_MMA_MUC
            qtw[0] = (2 * t4m + 0 == RNK) ? (float)c0
                     : (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (2 * t4m + 1 == RNK) ? (float)c1
                     : (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
            qtw[0] = (float)c0 * qsc_m * __bfloat162float(sv2.x);
            qtw[1] = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#endif
            __syncwarp();
            // (3) phase-1b (mu.q) + phase-2 (LSE) per octet head
            {   // HMAX (lever 1): fold the stored values in SSA registers
                const float hsa = tile(qt_a, q8d_a, qsc_a, Sp_a, K_a);
                const float hsb = tile(qt_b, q8d_b, qsc_b, Sp_b, K_b);
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa);
                    hm_b = fmaxf(hm_b, hsb);
                }
            }
            if constexpr (!BT) {
                p_mu  += (long)nworker * (DHEAD / 16);
                p_c8  += (long)nworker * PGT * RNK;
                p_cs  += (long)nworker * PGT;
                p_mus += nworker;
                p_v4  += (long)nworker * (RNK * DHEAD / 8);
                p_vs  += (long)nworker * RNK;
            }
            Sp_a += nworker; Sp_b += nworker;
            __syncwarp();      // qt_sh reuse next page
        }
        }   // end !BT / PGT!=16 original six-array loop (else of Fix 1)
#endif  // RKI4_PACK
#endif  // RKI4_MMA_AOS
#else   // !RKI4_MMA_SRED: the shipped per-page addressing loop
        for (int pp = wid; pp < P; pp += nworker) {
            const long row = row_of(pp);
            // (1) per-(lane=rr) mu / C / cs / mus loads (d128 -> MUW=4)
            const uint4* mup = reinterpret_cast<const uint4*>(
                mu8 + row * DHEAD) + rr * (MUW / 4);
            #pragma unroll
            for (int j = 0; j < MUW / 4; ++j) {
                const uint4 m4 = __ldg(mup + j);
                mm[4*j+0]=m4.x; mm[4*j+1]=m4.y; mm[4*j+2]=m4.z; mm[4*j+3]=m4.w;
            }
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
#if RNK == 8
                cc[tt] = __ldg(reinterpret_cast<const uint2*>(
                    c8 + (row * PGT + rr + 8 * tt) * RNK));
#elif RNK == 4
                cc[tt].x = __ldg(reinterpret_cast<const unsigned*>(
                    c8 + (row * PGT + rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#else
                cc[tt].x = (unsigned)__ldg(reinterpret_cast<const unsigned short*>(
                    c8 + (row * PGT + rr + 8 * tt) * RNK));
                cc[tt].y = 0u;
#endif
                sc[tt] = cs[row * PGT + rr + 8 * tt];
            }
            smu = mus[row];
            // (2) mma projection: qt[head][basis] for all 8x8 -> qt_sh
            // (rank v2: dead basis cols gidm >= RNK clamp in-bounds; their
            // mma output cols are zero-gated in the epilogue)
            const unsigned* cwp = reinterpret_cast<const unsigned*>(
                v4 + (row * RNK + ((gidm < RNK) ? gidm : (RNK - 1)))
                     * (DHEAD / 2));                       // basis column gidm
            const unsigned cw0 = __ldg(cwp + t4m),      cw1 = __ldg(cwp + 4 + t4m),
                           cw2 = __ldg(cwp + 8 + t4m),  cw3 = __ldg(cwp + 12 + t4m);
#ifdef RKI4_MMA_MUC
            const unsigned cm4 = ismu ? __ldg(cwp + 16 + t4m) : 0u,
                           cm5 = ismu ? __ldg(cwp + 20 + t4m) : 0u,
                           cm6 = ismu ? __ldg(cwp + 24 + t4m) : 0u,
                           cm7 = ismu ? __ldg(cwp + 28 + t4m) : 0u;
#endif
            int acc[4] = {0, 0, 0, 0};
            unsigned b[2];
#ifdef RKI4_MMA_BIAS
            // Biased nibble b'' = (nib ^ 8) = V + 8 (the int4 is two's-complement,
            // so V = (nib^8) - 8). One LOP3 (AND+XOR fused) per limb, DROPPING
            // the __vsub4 vs r8_lo/r8_hi (-8 ALU/page). The +8 plane folds out
            // exactly: C = C'' - 8*qsum (qsum_mma already carries the *8).
#ifdef RKI4_MMA_MUC
            b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
            b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
            b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
            b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
            b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
#else
            b[0] = (cw0 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw1 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[0], b);
            b[0] = (cw2 & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = (cw3 & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[1], b);
            b[0] = ((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[2], b);
            b[0] = ((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
            b[1] = ((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u; r8_mma_s8(acc, Af[3], b);
#endif
#if RNK == 8
            const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
#ifdef RKI4_MMA_MUC
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma
                           : (2 * t4m + 0 == RNK ? acc[0] : 0);
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma
                           : (2 * t4m + 1 == RNK ? acc[1] : 0);
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
#endif  // fold -8*qsum
#else
            b[0] = r8_lo(cw0); b[1] = r8_lo(cw1); r8_mma_s8(acc, Af[0], b);
            b[0] = r8_lo(cw2); b[1] = r8_lo(cw3); r8_mma_s8(acc, Af[1], b);
            b[0] = r8_hi(cw0); b[1] = r8_hi(cw1); r8_mma_s8(acc, Af[2], b);
            b[0] = r8_hi(cw2); b[1] = r8_hi(cw3); r8_mma_s8(acc, Af[3], b);
#if RNK == 8
            const int c0 = acc[0], c1 = acc[1];
#else
            const int c0 = (2 * t4m + 0 < RNK) ? acc[0] : 0;
            const int c1 = (2 * t4m + 1 < RNK) ? acc[1] : 0;
#endif
#endif
            // c0/c1 = qt raw for head=gidm, basis {2*t4m, 2*t4m+1}
            const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
                vs + row * RNK + ((2 * t4m < RNK) ? 2 * t4m
                 : (RNK - 2 < 0 ? 0 : RNK - 2)));  // rank v2 clamp
            const float qscg = qsc[gidm];
            qt_sh[warp][gidm][2*t4m]     = (float)c0 * qscg
                                          * __bfloat162float(sv2.x);
            qt_sh[warp][gidm][2*t4m + 1] = (float)c1 * qscg
                                          * __bfloat162float(sv2.y);
            __syncwarp();
            // (3) phase-1b (mu.q) + phase-2 (LSE) per octet head
            auto tile = [&](int g) {
                int mui = 0;
                #pragma unroll
                for (int j = 0; j < MUW; ++j)
                    mui = __dp4a((int)mm[j], (int)q8d[g][rr * MUW + j], mui);
                float md = (float)mui;
                md += __shfl_xor_sync(~0u, md, 1);
                md += __shfl_xor_sync(~0u, md, 2);
                md += __shfl_xor_sync(~0u, md, 4);
                const float mud = md * qsc[g] * __bfloat162float(smu);
                float mx = -CUDART_INF_F, es = 0.f;
                #pragma unroll
                for (int tt = 0; tt < PGT / 8; ++tt) {
                    float dt = 0.f;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u)
                        dt += (float)((int)(cc[tt].x << (24 - 8 * u)) >> 24)
                            * qt_sh[warp][g][u];
                    #pragma unroll
                    for (int u = 0; u < 4; ++u)
                        dt += (float)((int)(cc[tt].y << (24 - 8 * u)) >> 24)
                            * qt_sh[warp][g][4 + u];
                    const float tok = sm_scale * (dt * __bfloat162float(sc[tt])
                        + mud);
                    const float nm = fmaxf(mx, tok);
                    es = es * __expf(mx - nm) + __expf(tok - nm);
                    mx = nm;
                }
                #pragma unroll
                for (int st = 1; st < 8; st <<= 1) {
                    const float om = __shfl_xor_sync(~0u, mx, st);
                    const float oe = __shfl_xor_sync(~0u, es, st);
                    const float nm = fmaxf(mx, om);
                    es = es * __expf(mx - nm) + oe * __expf(om - nm);
                    mx = nm;
                }
                {
                    const float osv = mx + __logf(es);
                    if (rr == 0)
                        S[(((long)r * n_kv + kh) * G + g) * MP + pp] = osv;
                    return osv;   // HMAX: call sites fold
                }
            };
            {   // HMAX (lever 1): fold the stored values in SSA registers
                const float hsa = tile(gl);      // octet o=gl -> heads 0-3
                const float hsb = tile(gl + 4);  //             heads 4-7
                if (hmax_pub != nullptr) {
                    hm_a = fmaxf(hm_a, hsa);
                    hm_b = fmaxf(hm_b, hsb);
                }
            }
            __syncwarp();      // qt_sh reuse next page
        }
#endif  // RKI4_MMA_SRED (else = shipped per-page addressing loop)
#endif
    } else if constexpr (PIPE) {
        // 2-deep software pipeline: issue page p+1's loads into a register
        // B-bank before computing page p (the S4 structure).
        uint4 va[VW4]; uint32_t mua[MUW]; uint2 ca[PGT / 8];
        __nv_bfloat16 vsa, musa, csa[PGT / 8];
        if (p < P) load_page(p, va, mua, ca, vsa, musa, csa);
        while (p < P) {
            const int pn = p + nworker;
            uint4 vb[VW4]; uint32_t mub[MUW]; uint2 cb[PGT / 8];
            __nv_bfloat16 vsb, musb, csb[PGT / 8];
            if (pn < P) load_page(pn, vb, mub, cb, vsb, musb, csb);
            compute_page(p, va, mua, ca, vsa, musa, csa);
            p = pn;
            #pragma unroll
            for (int c = 0; c < VW4; ++c) va[c] = vb[c];
            #pragma unroll
            for (int j = 0; j < MUW; ++j) mua[j] = mub[j];
            vsa = vsb; musa = musb;
            #pragma unroll
            for (int tt = 0; tt < PGT / 8; ++tt) {
                ca[tt] = cb[tt]; csa[tt] = csb[tt];
            }
        }
    } else {
        // DHEAD=256: the doubled V/mu register bank would spill; 1-deep loop
        // (a compile-time property of this instantiation, never a runtime
        // branch).
        uint4 va[VW4]; uint32_t mua[MUW]; uint2 ca[PGT / 8];
        __nv_bfloat16 vsa, musa, csa[PGT / 8];
        for (; p < P; p += nworker) {
            load_page(p, va, mua, ca, vsa, musa, csa);
            compute_page(p, va, mua, ca, vsa, musa, csa);
        }
    }
    // ---- PMAX FOLD epilogue (SELECT_KERNEL_CAMPAIGN.md 8): emit per-CTA
    // per-g maxima of the score_h entries THIS CTA just wrote, replacing the
    // sel_pmax kernel's full re-read sweep, and zero this group's pass-0
    // ghist (which sel_pmax owned). hm downstream becomes a max over the
    // SAME value multiset under a different partition; fmax is order-free
    // => bitwise-identical selection. Block-scope fence makes the CTA's own
    // global writes visible for the re-read; empty coverage emits -inf
    // exactly as sel_pmax did. No cross-CTA dependency (the sync to the
    // consumer stays the existing score->select kernel boundary).
    if (gpmax_out != nullptr) {
        __threadfence_block();
        __syncthreads();
        const int fgroup = r * n_kv + kh;
        for (int i = zsp * (int)blockDim.x + tid; i < 256;
             i += (int)gridDim.z * (int)blockDim.x)
            ghist_out[(long)fgroup * 256 + i] = 0;
        __shared__ float foldm[NW][8];
#ifdef RKI4_MMA_FOLDR
        if constexpr (USE_MMA) {
            // FOLDR (doc 27d): the per-group maxima were folded in registers
            // at the S-write sites; publish the warp's two group slots and
            // skip the 8.39MB S re-scan entirely. Same page sets, fp max is
            // order-exact -> gpmax bitwise-equal to the scan (gated).
            if (rr == 0) {
                foldm[warp][gl] = fold_a;
                foldm[warp][gl + 4] = fold_b;
            }
        } else {
            // non-flagship instantiations (no register fold): keep the scan
            // (compile-correct; the fold host entries assert flagship-only)
            const int fg = lane >> 2, fs = lane & 3;
            float fm = -CUDART_INF_F;
            if (fg < G) {
                const float* Sg = S + (((long)r * n_kv + kh) * G + fg) * MP;
                for (long pp = (long)wid + (long)fs * nworker; pp < P;
                     pp += 4L * nworker)
                    fm = fmaxf(fm, Sg[pp]);
            }
            fm = fmaxf(fm, __shfl_xor_sync(~0u, fm, 1));
            fm = fmaxf(fm, __shfl_xor_sync(~0u, fm, 2));
            if (fs == 0) foldm[warp][fg] = fm;
        }
#else
        const int fg = lane >> 2, fs = lane & 3;   // 8 g-slots x 4 page-slots
        float fm = -CUDART_INF_F;
        if (fg < G) {
            const float* Sg = S + (((long)r * n_kv + kh) * G + fg) * MP;
            for (long pp = (long)wid + (long)fs * nworker; pp < P;
                 pp += 4L * nworker)
                fm = fmaxf(fm, Sg[pp]);
        }
        fm = fmaxf(fm, __shfl_xor_sync(~0u, fm, 1));
        fm = fmaxf(fm, __shfl_xor_sync(~0u, fm, 2));
        if (fs == 0) foldm[warp][fg] = fm;
#endif
        __syncthreads();
        if (tid < 8) {
            float m2 = -CUDART_INF_F;
            const int nw = ((int)blockDim.x + 31) >> 5;
            for (int w = 0; w < nw; ++w) m2 = fmaxf(m2, foldm[w][tid]);
            gpmax_out[((long)fgroup * 8 + tid) * gridDim.z + zsp] = m2;
        }
    }
    // ---- HMAX HANDSHAKE epilogue (lever 1, 2026-07-21): publish the
    // per-(r, kh, g) page maxima of the score_h entries THIS WARP just
    // wrote, so the fused nrm+topb consumer (select_cuda, has_hm=1) can
    // skip its phase-1 sweep.  HYBRID form (both simpler shapes measured
    // at the deployed 16K z=128 point: warp-local direct-global +1.14us
    // -- z*NW-deep same-address atomic chains; 3-barrier CTA-staged
    // +0.88us -- convergence tax): WARP-LOCAL re-read (every page-loop
    // path above is warp-strided (wid), so each warp re-reads ONLY its
    // own S stores; fence.cta orders each lane's stores and the warp
    // barrier is the sync-with edge that makes them visible to the
    // scanning lanes -- PTX model), smem-staged CTA fold (s_hm rest-init
    // rides the staging prelude barrier), ONE __syncthreads join, then
    // <= 8 global atomics/CTA (z-deep chains through 32 disjoint L2
    // slots).  fp32 fmax is order-free over the same value
    // multiset, so the published max is BIT-EXACT vs the consumer's
    // in-kernel sweep for EVERY score variant (mma / dp4a / probe arms;
    // zero store-site edits).  f2u is the order-preserving uint map (uint
    // atomicMax == exact fp32 max, order/partition-free); 0x007FFFFF ==
    // f2u(-inf) is the buffer's rest state (the consumer resets after
    // reading -> graph-replay invariant; empty coverage publishes
    // nothing).  Assumes scores are NaN-free (LSE of finite bf16 content;
    // f2u(NaN) would outrank +inf while the consumer's fmaxf sweep drops
    // NaN).  hmax_pub == nullptr (legacy) = dead code.
    if (hmax_pub != nullptr) {
        if constexpr (HM_FOLD) {
            // Store-site register fold (deployed mma SRED loop + dp4a G8):
            // hm_a/hm_b hold the exact per-(warp, head gl / gl+4) maxima of
            // the STORED values, uniform across each rr-group (sv is post-
            // butterfly uniform; every lane folded every page).  No S
            // re-read, no fence: values never left registers.
            if (rr == 0) {
                unsigned ua = __float_as_uint(hm_a);
                ua ^= (unsigned)(-(int)(ua >> 31)) | 0x80000000u;
                if (ua != 0x007FFFFFu && gl < G) atomicMax(&s_hm[gl], ua);
                unsigned ub = __float_as_uint(hm_b);
                ub ^= (unsigned)(-(int)(ub >> 31)) | 0x80000000u;
                if (ub != 0x007FFFFFu && gl + 4 < G)
                    atomicMax(&s_hm[gl + 4], ub);
            }
            __syncthreads();
            if (tid < 8 && s_hm[tid] != 0x007FFFFFu)
                atomicMax(&hmax_pub[((long)r * n_kv + kh) * 8 + tid],
                          s_hm[tid]);
        } else {
            __threadfence_block();
            __syncwarp();
            const int hg = lane >> 2, hs = lane & 3;   // 8 g x 4 page-slots
            float hf = -CUDART_INF_F;
            if (hg < G) {
                const float* Sg = S + (((long)r * n_kv + kh) * G + hg) * MP;
                for (long pp = (long)wid + (long)hs * nworker; pp < P;
                     pp += 4L * nworker)
                    hf = fmaxf(hf, Sg[pp]);
            }
            hf = fmaxf(hf, __shfl_xor_sync(~0u, hf, 1));
            hf = fmaxf(hf, __shfl_xor_sync(~0u, hf, 2));
            if (hs == 0 && hg < G) {
                unsigned u = __float_as_uint(hf);
                u ^= (unsigned)(-(int)(u >> 31)) | 0x80000000u;
                if (u != 0x007FFFFFu) atomicMax(&s_hm[hg], u);
            }
            __syncthreads();
            if (tid < 8 && s_hm[tid] != 0x007FFFFFu)
                atomicMax(&hmax_pub[((long)r * n_kv + kh) * 8 + tid],
                          s_hm[tid]);
        }
    }
}

using KFn = void (*)(const __nv_bfloat16*, const uint8_t*,
                     const __nv_bfloat16*, const int8_t*,
                     const __nv_bfloat16*, const int8_t*,
                     const __nv_bfloat16*, const int*, const int*, float*,
                     const int, const int, const int, const float,
                     const long, const long, const long, float*, int*,
                     const float*, const int*, const __nv_bfloat16*,
                     __nv_bfloat16*, unsigned*);

static KFn pick_kernel(int PGT, int DHEAD, bool bt_mode, bool g4, bool g8,
                       bool pf = false) {
    // Fix1 prefetch gate: ONLY the deployed <16,128,BT,g8> instantiation has a
    // PF=true (v4-prefetch) variant; the host requests it (pf=true) only in the
    // regimes where it strictly wins. Every other geometry/regime falls through
    // to PF=false (the byte-identical deployed loop) below.
    if (pf && bt_mode && g8 && PGT == 16 && DHEAD == 128)
        return rki4_score_kernel<16, 128, true, false, true, /*PF=*/true>;
#define RKI4_CASE(PGT_, DH_) if (PGT == PGT_ && DHEAD == DH_) { if (bt_mode) { if (g4) return rki4_score_kernel<PGT_, DH_, true, true, false>; if (g8) return rki4_score_kernel<PGT_, DH_, true, false, true>; return rki4_score_kernel<PGT_, DH_, true, false, false>; } if (g4) return rki4_score_kernel<PGT_, DH_, false, true, false>; if (g8) return rki4_score_kernel<PGT_, DH_, false, false, true>; return rki4_score_kernel<PGT_, DH_, false, false, false>; }
    RKI4_CASE(16, 64) RKI4_CASE(16, 128) RKI4_CASE(16, 256)
    RKI4_CASE(32, 64) RKI4_CASE(32, 128) RKI4_CASE(32, 256)
#undef RKI4_CASE
    TORCH_CHECK(false, "rki4: unsupported geometry page=", PGT,
                " d=", DHEAD, " (supported: page {16,32} x d {64,128,256})");
    return nullptr;
}

// G8 compile-time specialization is default ON for G==8; LOCKS_SCORE_NO_G8
// forces the runtime g-tile path (for the bitwise gate: the two must produce
// score_h bitwise-equal, since G8 is the same 2 tiles unrolled).
static inline bool g8_enabled() {
    // read per-call (host-side, negligible) so the bitwise gate can toggle
    // G8-specialized vs runtime g-tile within one process.
    return std::getenv("LOCKS_SCORE_NO_G8") == nullptr;
}

static void check_common(const torch::Tensor& q, const torch::Tensor& v4,
                         const torch::Tensor& vs, const torch::Tensor& c8,
                         const torch::Tensor& cs, const torch::Tensor& mu8,
                         const torch::Tensor& mus, const torch::Tensor& S,
                         int n_kv, int PGT, int DHEAD, int G) {
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(q.stride(2) == 1, "q innermost dim must be contiguous");
    TORCH_CHECK(q.size(1) == (long)n_kv * G, "q heads != n_kv * G");
    TORCH_CHECK(q.size(2) == DHEAD, "q head dim != summary d");
    TORCH_CHECK(G >= 1 && G <= MAXG, "G must be in [1, ", MAXG, "], got ", G);
    TORCH_CHECK(v4.size(2) == RNK && v4.size(3) == DHEAD / 2, "v4 layout");
    TORCH_CHECK(c8.size(2) == PGT && c8.size(3) == RNK, "c8 layout");
    TORCH_CHECK(v4.is_contiguous() && vs.is_contiguous() && c8.is_contiguous()
                && cs.is_contiguous() && mu8.is_contiguous()
                && mus.is_contiguous(), "summary tensors must be contiguous");
    TORCH_CHECK(S.is_contiguous() && S.dim() == 4 && S.size(1) == n_kv
                && S.size(2) == G, "S must be contiguous (R, n_kv, G, MP)");
    TORCH_CHECK(S.scalar_type() == torch::kFloat, "S must be fp32");
}

void r8score(torch::Tensor q, torch::Tensor v4, torch::Tensor vs,
             torch::Tensor c8, torch::Tensor cs, torch::Tensor mu8,
             torch::Tensor mus, torch::Tensor S, double sm_scale,
             int64_t zsplit)
{
#ifdef RKI4_MMA_FLAT
    TORCH_CHECK(false, "RKI4_MMA_FLAT build: use rki4_score_bt_aos_flat "
                       "(base entries pass Kflat=nullptr)");
#endif

    const int R = q.size(0);
    const int n_kv = v4.size(0);
    const int P = v4.size(1);
    const int PGT = c8.size(2);
    const int DHEAD = v4.size(3) * 2;
    const int G = S.size(2);
    check_common(q, v4, vs, c8, cs, mu8, mus, S, n_kv, PGT, DHEAD, G);
    TORCH_CHECK(S.size(0) == R && S.size(3) == P, "S shape vs (R, P)");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535,
                "zsplit must be in [1, 65535] (CUDA gridDim.z limit)");
    KFn kfn = pick_kernel(PGT, DHEAD, /*bt=*/false, /*g4=*/G <= 4, /*g8=*/(G == 8) && g8_enabled());
    dim3 grid((unsigned)R, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        v4.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr()),
        c8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(cs.data_ptr()),
        mu8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(mus.data_ptr()),
        nullptr, nullptr,
        S.data_ptr<float>(), P, /*MP=*/P, G, (float)sm_scale,
        q.stride(0), q.stride(1), 0L, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "r8score launch: ", cudaGetErrorString(e));
}

void r8score_aos(torch::Tensor q, torch::Tensor rec, torch::Tensor S,
                 double sm_scale, int64_t zsplit)
{
#ifdef RKI4_MMA_FLAT
    TORCH_CHECK(false, "RKI4_MMA_FLAT build: use rki4_score_bt_aos_flat "
                       "(base entries pass Kflat=nullptr)");
#endif

#ifndef RKI4_MMA_AOS
    (void)q; (void)rec; (void)S; (void)sm_scale; (void)zsplit;
    TORCH_CHECK(false, "r8score_aos requires an RKI4_MMA_AOS build");
#else
    const int R = q.size(0);
    const int n_kv = rec.size(0);
    const int P = rec.size(1);
    const int G = S.size(2);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST,
                "rec must be contiguous (n_kv, P, 832) uint8");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(0) == R && S.size(1) == n_kv && S.size(3) == P, "S");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "r8score_aos covers the G in {4,8} d128 p16 records only");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535, "zsplit range");
    KFn kfn = pick_kernel(16, 128, /*bt=*/false, /*g4=*/(G <= 4), /*g8=*/(G == 8));
    dim3 grid((unsigned)R, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr,
        S.data_ptr<float>(), P, /*MP=*/P, G, (float)sm_scale,
        q.stride(0), q.stride(1), 0L, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "r8score_aos launch: ",
                cudaGetErrorString(e));
#endif
}

void rki4_score_bt_aos(torch::Tensor q, torch::Tensor rec, torch::Tensor bt,
                       torch::Tensor nsh, torch::Tensor S, double sm_scale,
                       int64_t zsplit, int64_t n_req)
{
#ifdef RKI4_MMA_FLAT
    TORCH_CHECK(false, "RKI4_MMA_FLAT build: use rki4_score_bt_aos_flat "
                       "(base entries pass Kflat=nullptr)");
#endif

#ifndef RKI4_MMA_AOS
    (void)q; (void)rec; (void)bt; (void)nsh; (void)S; (void)sm_scale;
    (void)zsplit; (void)n_req;
    TORCH_CHECK(false, "rki4_score_bt_aos requires an RKI4_MMA_AOS build");
#else
    const int n_kv = rec.size(1);
    const int G = S.size(2);
    const int MP = S.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST,
                "rec must be contiguous (NB, n_kv, 832) uint8");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(1) == n_kv, "S must be (R, n_kv, G, MP) fp32");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "rki4_score_bt_aos covers G in {4,8} d128 p16 records only");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1,
                "block table must be int32, row-contiguous");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(),
                "n_sel_hi must be contiguous int32");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && n_req <= bt.size(0)
                && n_req <= nsh.size(0) && n_req <= q.size(0),
                "n_req out of range");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535, "zsplit range");
    KFn kfn = pick_kernel(16, 128, /*bt=*/true, /*g4=*/(G <= 4), /*g8=*/(G == 8));
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        // P_dense carries NB for the BT record-row clamp (dense-only slot)
        S.data_ptr<float>(), (int)rec.size(0), MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0), nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "rki4_score_bt_aos launch: ",
                cudaGetErrorString(e));
#endif
}

// PMAX FOLD variant: identical to rki4_score_bt_aos plus the epilogue outputs
// (per-CTA chunk maxima into gpmax at (group, 8, zsplit) and the pass-0 ghist
// zeroing). Separate entry so every existing caller keeps its ABI; the base
// entry passes nullptrs (the epilogue is dead code there).
void rki4_score_bt_aos_fold(torch::Tensor q, torch::Tensor rec,
                            torch::Tensor bt, torch::Tensor nsh,
                            torch::Tensor S, torch::Tensor gpmax,
                            torch::Tensor ghist, double sm_scale,
                            int64_t zsplit, int64_t n_req)
{
#ifndef RKI4_MMA_AOS
    (void)q; (void)rec; (void)bt; (void)nsh; (void)S; (void)gpmax;
    (void)ghist; (void)sm_scale; (void)zsplit; (void)n_req;
    TORCH_CHECK(false, "rki4_score_bt_aos_fold requires an RKI4_MMA_AOS build");
#else
    const int n_kv = rec.size(1);
    const int G = S.size(2);
    const int MP = S.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST,
                "rec must be contiguous (NB, n_kv, RECB) uint8");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(1) == n_kv, "S must be (R, n_kv, G, MP) fp32");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "rki4_score_bt_aos_fold covers G in {4,8} d128 p16 records only");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1,
                "block table must be int32, row-contiguous");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(),
                "n_sel_hi must be contiguous int32");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && n_req <= bt.size(0)
                && n_req <= nsh.size(0) && n_req <= q.size(0),
                "n_req out of range");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535, "zsplit range");
    TORCH_CHECK(gpmax.scalar_type() == torch::kFloat
                && gpmax.numel() >= (long)n_req * n_kv * 8 * zsplit,
                "gpmax scratch too small for (group, 8, zsplit)");
    TORCH_CHECK(ghist.scalar_type() == torch::kInt
                && ghist.numel() >= (long)n_req * n_kv * 256,
                "ghist scratch too small");
    KFn kfn = pick_kernel(16, 128, /*bt=*/true, /*g4=*/(G <= 4), /*g8=*/(G == 8));
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        S.data_ptr<float>(), (int)rec.size(0), MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0),
        gpmax.data_ptr<float>(), ghist.data_ptr<int>(), nullptr,
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "rki4_score_bt_aos_fold launch: ",
                cudaGetErrorString(e));
#endif
}

// FLAT-MASS entry (doc 18): identical checks/launch to rki4_score_bt_aos
// plus the per-(kh,g) shift tensor. Compiled meaningfully only under
// RKI4_MMA_FLAT (otherwise the kernel ignores Kflat and computes LSE --
// callers must not route here without the flag; the python side gates).
void rki4_score_bt_aos_flat(torch::Tensor q, torch::Tensor rec,
                            torch::Tensor bt, torch::Tensor nsh,
                            torch::Tensor S, torch::Tensor Kflat,
                            double sm_scale, int64_t zsplit, int64_t n_req)
{
#ifndef RKI4_MMA_AOS
    (void)q; (void)rec; (void)bt; (void)nsh; (void)S; (void)Kflat;
    (void)sm_scale; (void)zsplit; (void)n_req;
    TORCH_CHECK(false, "rki4_score_bt_aos_flat requires an RKI4_MMA_AOS build");
#else
    const int n_kv = rec.size(1);
    const int G = S.size(2);
    const int MP = S.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST,
                "rec must be contiguous (NB, n_kv, RECB) uint8");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(1) == n_kv, "S must be (R, n_kv, G, MP) fp32");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "rki4_score_bt_aos_flat covers G in {4,8} d128 p16 records only");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1,
                "block table must be int32, row-contiguous");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(),
                "n_sel_hi must be contiguous int32");
    TORCH_CHECK(Kflat.scalar_type() == torch::kFloat && Kflat.is_contiguous()
                && Kflat.numel() >= (long)n_kv * G,
                "Kflat must be contiguous fp32 (n_kv*G,)");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && n_req <= bt.size(0)
                && n_req <= nsh.size(0) && n_req <= q.size(0),
                "n_req out of range");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535, "zsplit range");
    KFn kfn = pick_kernel(16, 128, /*bt=*/true, /*g4=*/(G <= 4), /*g8=*/(G == 8));
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        S.data_ptr<float>(), (int)rec.size(0), MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0),
        nullptr, nullptr, Kflat.data_ptr<float>(),
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "rki4_score_bt_aos_flat launch: ",
                cudaGetErrorString(e));
#endif
}

// FLAT+FOLD combined entry (doc 27c GO): the FLAT mass score with the PMAX
// FOLD epilogue armed (gpmax/ghist non-null). The fold epilogue re-reads
// the S values this launch wrote, so it is score-domain-agnostic; under
// FLAT it emits per-chunk MASS maxima -- exactly what sel_pmax computed --
// letting the select chain skip its 8.39MB score_h pass (pmax_done route).
void rki4_score_bt_aos_flat_fold(torch::Tensor q, torch::Tensor rec,
                                 torch::Tensor bt, torch::Tensor nsh,
                                 torch::Tensor S, torch::Tensor gpmax,
                                 torch::Tensor ghist, torch::Tensor Kflat,
                                 double sm_scale, int64_t zsplit,
                                 int64_t n_req)
{
#ifndef RKI4_MMA_FLAT
    (void)q; (void)rec; (void)bt; (void)nsh; (void)S; (void)gpmax;
    (void)ghist; (void)Kflat; (void)sm_scale; (void)zsplit; (void)n_req;
    TORCH_CHECK(false, "flat_fold requires an RKI4_MMA_FLAT build");
#else
    const int n_kv = rec.size(1);
    const int G = S.size(2);
    const int MP = S.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST, "rec");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat
                && S.size(1) == n_kv, "S must be (R, n_kv, G, MP) fp32");
    TORCH_CHECK(q.size(1) == (long)n_kv * G && q.size(2) == 128
                && (G == 4 || G == 8) && (G != 8 || g8_enabled()),
                "flat_fold covers G in {4,8} d128 p16 records");
    TORCH_CHECK(Kflat.is_contiguous() && Kflat.scalar_type() == torch::kFloat
                && Kflat.numel() == (long)n_kv * G, "Kflat (n_kv*G) fp32");
    TORCH_CHECK(gpmax.scalar_type() == torch::kFloat
                && gpmax.numel() >= (long)n_req * n_kv * 8 * zsplit,
                "gpmax scratch too small");
    TORCH_CHECK(ghist.scalar_type() == torch::kInt
                && ghist.numel() >= (long)n_req * n_kv * 256,
                "ghist scratch too small");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1, "bt");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(), "nsh");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && zsplit >= 1
                && zsplit <= 65535, "shape");
    KFn kfn = pick_kernel(16, 128, /*bt=*/true, /*g4=*/(G <= 4), /*g8=*/(G == 8));
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(),
        nullptr, nullptr, nullptr, nullptr, nullptr,
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        S.data_ptr<float>(), (int)rec.size(0), MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0),
        gpmax.data_ptr<float>(), ghist.data_ptr<int>(),
        Kflat.data_ptr<float>(),
        nullptr, nullptr, nullptr, nullptr);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "flat_fold launch: ",
                cudaGetErrorString(e));
#endif
}

void rki4_score_bt(torch::Tensor q, torch::Tensor v4, torch::Tensor vs,
                   torch::Tensor c8, torch::Tensor cs, torch::Tensor mu8,
                   torch::Tensor mus, torch::Tensor bt, torch::Tensor nsh,
                   torch::Tensor S, double sm_scale, int64_t zsplit,
                   int64_t n_req,
                   c10::optional<torch::Tensor> sl_rope,
                   c10::optional<torch::Tensor> csc_rope,
                   c10::optional<torch::Tensor> qout_rope,
                   c10::optional<torch::Tensor> hmax_pub)
{
    // NOROPE (round A): when qout_rope is given, q arrives UNROPED and the
    // kernel ropes it in the staging prelude (pos = sl-1, bf16 cos|sin
    // cache) and publishes the roped bf16 bytes to qout_rope.  All three
    // absent = legacy behavior, bit-for-bit.
    const int* slp = sl_rope ? sl_rope->data_ptr<int>() : nullptr;
    const __nv_bfloat16* cscp = csc_rope
        ? reinterpret_cast<const __nv_bfloat16*>(csc_rope->data_ptr())
        : nullptr;
    __nv_bfloat16* qoutp = qout_rope
        ? reinterpret_cast<__nv_bfloat16*>(qout_rope->data_ptr())
        : nullptr;
    TORCH_CHECK(!qoutp || (slp && cscp),
                "norope: qout_rope requires sl_rope + csc_rope");
    // HMAX HANDSHAKE (lever 1): optional publish buffer for the kernel-end
    // epilogue; absent = legacy (dead code, zero new work).
    unsigned* hmp = nullptr;
    if (hmax_pub) {
        TORCH_CHECK(hmax_pub->scalar_type() == torch::kInt
                    && hmax_pub->is_contiguous()
                    && hmax_pub->numel() >= (long)n_req * v4.size(1) * 8,
                    "hmax_pub must be contiguous int32 (R, n_kv, 8)");
        hmp = reinterpret_cast<unsigned*>(hmax_pub->data_ptr<int>());
    }
    if (qoutp) {
        TORCH_CHECK(sl_rope->scalar_type() == torch::kInt
                    && csc_rope->scalar_type() == torch::kBFloat16
                    && csc_rope->stride(1) == 1
                    && csc_rope->size(1) == LOCKS_ROT_DIM
                    && qout_rope->scalar_type() == torch::kBFloat16
                    && qout_rope->is_contiguous(),
                    "norope: sl int32, csc (P,LOCKS_ROT_DIM) bf16 row-contig, "
                    "qout contiguous bf16");
    }
#ifdef RKI4_MMA_FLAT
    TORCH_CHECK(false, "RKI4_MMA_FLAT build: use rki4_score_bt_aos_flat "
                       "(base entries pass Kflat=nullptr)");
#endif

    const int n_kv = v4.size(1);           // per-layer slab (NB, n_kv, ...)
    const int PGT = c8.size(2);
    const int DHEAD = v4.size(3) * 2;
    const int G = S.size(2);
    const int MP = S.size(3);
    check_common(q, v4, vs, c8, cs, mu8, mus, S, n_kv, PGT, DHEAD, G);
    TORCH_CHECK(c8.size(1) == n_kv, "c8 layout (NB, n_kv, PGT, RNK)");
    TORCH_CHECK(bt.scalar_type() == torch::kInt && bt.stride(1) == 1,
                "block table must be int32, row-contiguous");
    TORCH_CHECK(nsh.scalar_type() == torch::kInt && nsh.is_contiguous(),
                "n_sel_hi must be contiguous int32");
    TORCH_CHECK(n_req >= 1 && n_req <= S.size(0) && n_req <= bt.size(0)
                && n_req <= nsh.size(0) && n_req <= q.size(0),
                "n_req out of range");
    TORCH_CHECK(zsplit >= 1 && zsplit <= 65535,
                "zsplit must be in [1, 65535] (CUDA gridDim.z limit)");
    // Fix1 launch gate: enable the v4-prefetch kernel ONLY for n_req >=
    // LOCKS_FIX1_PF_MINREQ (default 8 -- the bs>=8 frontier where the prefetch
    // strictly wins). bs<=4 and bs=1 take the byte-identical deployed loop, so
    // the shipped bs=1 headline and the 256K/bs4 point are unchanged BY
    // CONSTRUCTION. Host-side env read (negligible); LOCKS_FIX1_PF_OFF=1 forces
    // the deployed kernel everywhere (A/B baseline).
    // env reads are cached ONCE (static) -- getenv in the per-(layer,step)
    // launch path would be host overhead on the bs=1 hot path.
    static const int _pf_minreq = [](){
        const char* e = std::getenv("LOCKS_FIX1_PF_MINREQ"); return e ? atoi(e) : 8; }();
    static const bool _pf_off = std::getenv("LOCKS_FIX1_PF_OFF") != nullptr;
    const bool g8 = (G == 8) && g8_enabled();
    const bool pf_gate = !_pf_off && g8 && (n_req >= _pf_minreq);
    static bool _pf_announced = false;
    if (!_pf_announced) {
        _pf_announced = true;
        fprintf(stderr, "[locks] FIX1_PF gate: n_req=%d minreq=%d -> PF=%s\n",
                n_req, _pf_minreq, pf_gate ? "true(prefetch)" : "false(deployed)");
    }
    KFn kfn = pick_kernel(PGT, DHEAD, /*bt=*/true, /*g4=*/G <= 4,
                          /*g8=*/g8, /*pf=*/pf_gate);
    dim3 grid((unsigned)n_req, (unsigned)n_kv, (unsigned)zsplit);
    kfn<<<grid, TB, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        v4.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(vs.data_ptr()),
        c8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(cs.data_ptr()),
        mu8.data_ptr<int8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(mus.data_ptr()),
        bt.data_ptr<int>(), nsh.data_ptr<int>(),
        S.data_ptr<float>(), 0, MP, G, (float)sm_scale,
        q.stride(0), q.stride(1), bt.stride(0), nullptr, nullptr, nullptr,
        slp, cscp, qoutp, hmp);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "rki4_score_bt launch: ",
                cudaGetErrorString(e));
}

#ifdef RKI4_MMA_SCREEN
// ---------------------------------------------------------------------------
// Certified-screen pass (doc 26).  Per (page, kv-head): the EXACT int-mma
// q.V projection (verbatim BIAS chain), then dt for ALL 16 tokens x 8 heads
// in ONE m16n8k8 bf16 HMMA with fp32 accumulation (c8 is EXACT in bf16;
// the only error source is qt's bf16 rounding), exact exps, and a rigorous
// per-(page,kvh) log-domain radius
//     eps = sm * nrmC * max|qt| * 2^-8 + EPS_FP   (nrmC build-stored,
// rounded UP; EPS_FP covers the epilogue's own fp32 rounding).  OUTPUT:
// score_h = LOWER bounds mass' * e^-eps (so the UNMODIFIED nrm+topb select
// gives the certificate's tau-set), rad = eps.  The exact top-b is then
// {p: upper >= tau} resolved by the exact kernel on the ambiguous band
// (python flow); selection identity is guaranteed by the interval algebra.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void scr_hmma_bf16(float* d, const unsigned* a,
                                              const unsigned b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

template <int PGT, int DHEAD, bool BT>
__global__ __launch_bounds__(TB)
void rki4_screen_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ v4,        // AOS records (BT: (NB,n_kv,RECB))
    const int* __restrict__ bt,
    const int* __restrict__ nsh,
    float* __restrict__ S,                 // (R, n_kv, G, MP) LOWER bounds
    float* __restrict__ rad,               // (R, n_kv, MP) radius eps
    const float* __restrict__ Kflat,       // (n_kv*G) FLAT shifts
    const float sm_scale, const int n_kv, const int G, const int MP,
    const int zsp_n, const long q_sr, const long q_sh, const long bt_sr,
    const long P_dense) {
    static_assert(PGT == 16 && DHEAD == 128, "screen covers the flagship");
    const int r = blockIdx.y, kh = blockIdx.x;
    const int zsp = blockIdx.z, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int P = BT ? __ldg(nsh + r) : (int)P_dense;
    const int nworker = zsp_n * NW;
    const int wid = zsp * NW + warp;

    // ---- q staging: int8 even/odd words + scales (prelude as the AOS path)
    __shared__ float qsh[MAXG * DHEAD];
    __shared__ uint32_t q8e[MAXG][DHEAD / 8], q8o[MAXG][DHEAD / 8];
    __shared__ uint32_t q8d[MAXG][DHEAD / 4];
    __shared__ float qsc[MAXG];
    for (int i = tid; i < G * DHEAD; i += TB)
        qsh[i] = __bfloat162float(
            q[(long)r * q_sr + (long)(kh * G + i / DHEAD) * q_sh
              + (i % DHEAD)]);
    __syncthreads();
    for (int gg = warp; gg < G; gg += NW) {
        const float* qw = qsh + gg * DHEAD;
        float am = 0.f;
        for (int j = lane; j < DHEAD; j += 32) am = fmaxf(am, fabsf(qw[j]));
        #pragma unroll
        for (int st = 16; st >= 1; st >>= 1)
            am = fmaxf(am, __shfl_xor_sync(~0u, am, st));
        const float qs = fmaxf(am, 1e-8f) / 127.f;
        if (lane == 0) qsc[gg] = qs;
        if (lane < DHEAD / 8) {
            uint32_t we = 0, wo = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 4; ++b2) {
                const int pe = (int)rintf(qw[8 * lane + 2 * b2] / qs);
                const int po = (int)rintf(qw[8 * lane + 2 * b2 + 1] / qs);
                we |= (uint32_t)(uint8_t)(int8_t)pe << (8 * b2);
                wo |= (uint32_t)(uint8_t)(int8_t)po << (8 * b2);
            }
            q8e[gg][lane] = we; q8o[gg][lane] = wo;
        }
        for (int j = lane; j < DHEAD / 4; j += 32) {
            uint32_t wd = 0;
            #pragma unroll
            for (int b2 = 0; b2 < 4; ++b2) {
                const int pd = (int)rintf(qw[4 * j + b2] / qs);
                wd |= (uint32_t)(uint8_t)(int8_t)pd << (8 * b2);
            }
            q8d[gg][j] = wd;
        }
    }
    __syncthreads();

    constexpr int MUW = DHEAD / 32;
    const int gidm = lane >> 2, t4m = lane & 3;
    unsigned Af[4][4];
    int qsum_mma = 0;
    {
        Af[0][0] = q8e[gidm][t4m];      Af[0][1] = 0u;
        Af[0][2] = q8e[gidm][4 + t4m];  Af[0][3] = 0u;
        Af[1][0] = q8e[gidm][8 + t4m];  Af[1][1] = 0u;
        Af[1][2] = q8e[gidm][12 + t4m]; Af[1][3] = 0u;
        Af[2][0] = q8o[gidm][t4m];      Af[2][1] = 0u;
        Af[2][2] = q8o[gidm][4 + t4m];  Af[2][3] = 0u;
        Af[3][0] = q8o[gidm][8 + t4m];  Af[3][1] = 0u;
        Af[3][2] = q8o[gidm][12 + t4m]; Af[3][3] = 0u;
        #pragma unroll
        for (int j = 0; j < DHEAD / 4 / 4; ++j)
            qsum_mma = __dp4a((int)q8d[gidm][8 * t4m + j], 0x01010101,
                              qsum_mma);
        qsum_mma += __shfl_xor_sync(~0u, qsum_mma, 1);
        qsum_mma += __shfl_xor_sync(~0u, qsum_mma, 2);
        qsum_mma *= 8;
    }
    // mud octet map (as the exact tile): rr slices d, gl = lane>>3
    const int rr = lane & 7, gl = lane >> 3;
    constexpr int RO_MU  = RNK * (DHEAD / 2);
    constexpr int RO_C8  = RO_MU + DHEAD;
    constexpr int RO_CS  = RO_C8 + RNK * PGT;
    constexpr int RO_VS  = RO_CS + 2 * PGT;
    constexpr int RO_MUS = RO_VS + 2 * RNK;
    constexpr long RECB  = ((RO_MUS + 2) + 63) / 64 * 64;
    const long pb_v = (long)kh * P_dense;
    const int* btr = BT ? (bt + (long)r * bt_sr) : nullptr;
    auto row_of = [&](int pp) -> long {
        return BT ? ((long)__ldg(btr + pp) * n_kv + kh) : (pb_v + pp);
    };
    const long rmax_rec = BT ? ((long)P_dense * n_kv - 1) : 0L;
    float* Sp = S + (((long)r * n_kv + kh) * G) * MP;
    float* Rp = rad + ((long)r * n_kv + kh) * MP;

    constexpr int RC16 = (int)(RECB / 16);
    __shared__ uint4 scrring[NW][2][RC16];
    auto issue_rec = [&](int pv, int slot) {
        if (pv < P) {
            long rowb = row_of(pv);
            rowb = rowb < 0 ? 0 : (rowb > rmax_rec ? rmax_rec : rowb);
            const uint4* r16 = reinterpret_cast<const uint4*>(v4 + rowb * RECB);
            for (int c = lane; c < RC16; c += 32)
                cpasync16(&scrring[warp][slot][c], r16 + c);
        }
        cpasync_commit();
    };
    issue_rec(wid, 0);
    issue_rec(wid + nworker, 1);
    int pslot = 0;
    for (int pp = wid; pp < P; pp += nworker) {
        cpasync_wait<1>();
        __syncwarp();
        const uint8_t* p_rec = reinterpret_cast<const uint8_t*>(
            &scrring[warp][pslot][0]);
        // (1) exact int-mma q.V (BIAS chain, verbatim values)
        int acc[4] = {0, 0, 0, 0};
        unsigned b[2];
        // RNK<8: dead basis columns are predicated (OOB past the ring slot
        // otherwise); column RNK carries the MUC-PERMUTED int8 mu (records
        // at rank<8 are MUC builds), fed RAW -- its integer output IS the
        // exact mu.q, fetched below for the mud factor.
        const bool blive = (gidm < RNK);
        const bool ismu = (RNK < 8) && (gidm == RNK);
        const unsigned* cwp = reinterpret_cast<const unsigned*>(
            p_rec + (ismu ? RO_MU : gidm * (DHEAD / 2)));
        const unsigned cw0 = (blive || ismu) ? cwp[t4m] : 0u,
                       cw1 = (blive || ismu) ? cwp[4 + t4m] : 0u,
                       cw2 = (blive || ismu) ? cwp[8 + t4m] : 0u,
                       cw3 = (blive || ismu) ? cwp[12 + t4m] : 0u;
        const unsigned cm4 = ismu ? cwp[16 + t4m] : 0u,
                       cm5 = ismu ? cwp[20 + t4m] : 0u,
                       cm6 = ismu ? cwp[24 + t4m] : 0u,
                       cm7 = ismu ? cwp[28 + t4m] : 0u;
        b[0] = ismu ? cw0 : ((cw0 & 0x0F0F0F0Fu) ^ 0x08080808u);
        b[1] = ismu ? cw1 : ((cw1 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[0], b);
        b[0] = ismu ? cw2 : ((cw2 & 0x0F0F0F0Fu) ^ 0x08080808u);
        b[1] = ismu ? cw3 : ((cw3 & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[1], b);
        b[0] = ismu ? cm4 : (((cw0 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
        b[1] = ismu ? cm5 : (((cw1 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[2], b);
        b[0] = ismu ? cm6 : (((cw2 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u);
        b[1] = ismu ? cm7 : (((cw3 >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u); r8_mma_s8(acc, Af[3], b);
        const int c0raw = acc[0];   // mu column output (lane t4m == RNK/2)
#if RNK == 8
        const int c0 = acc[0] - qsum_mma, c1 = acc[1] - qsum_mma;
#else
        const int c0 = (2 * t4m + 0 < RNK) ? acc[0] - qsum_mma : 0;
        const int c1 = (2 * t4m + 1 < RNK) ? acc[1] - qsum_mma : 0;
#endif
        const __nv_bfloat162 sv2 = *reinterpret_cast<const __nv_bfloat162*>(
            p_rec + RO_VS + 4 * t4m);
        const float qsc_m = qsc[gidm];
#if RNK == 8
        const float qtv0 = (float)c0 * qsc_m * __bfloat162float(sv2.x);
        const float qtv1 = (float)c1 * qsc_m * __bfloat162float(sv2.y);
#else
        const float qtv0 = (2 * t4m + 0 < RNK)
            ? (float)c0 * qsc_m * __bfloat162float(sv2.x) : 0.f;
        const float qtv1 = (2 * t4m + 1 < RNK)
            ? (float)c1 * qsc_m * __bfloat162float(sv2.y) : 0.f;
#endif
        // (2) B fragment = this lane's own qt pair in bf16 (no smem transit)
        const __nv_bfloat162 bq2 = __floats2bfloat162_rn(qtv0, qtv1);
        const unsigned bq = *reinterpret_cast<const unsigned*>(&bq2);
        // (3) A fragment: c8 tokens {gidm, gidm+8} x ranks {2t4m, 2t4m+1}
        //     (int8 pairs are contiguous; bf16 conversion is EXACT)
        unsigned afr[2];
        {
            const int8_t* c8p = reinterpret_cast<const int8_t*>(p_rec + RO_C8);
            const int t0 = gidm, t1 = gidm + 8;
            const bool alive = (2 * t4m < RNK);
            const short w0 = alive ? *reinterpret_cast<const short*>(
                c8p + t0 * RNK + 2 * t4m) : (short)0;
            const short w1 = alive ? *reinterpret_cast<const short*>(
                c8p + t1 * RNK + 2 * t4m) : (short)0;
            const __nv_bfloat162 a0 = __floats2bfloat162_rn(
                (float)(int)(int8_t)(w0 & 0xFF), (float)(int)(int8_t)(w0 >> 8));
            const __nv_bfloat162 a1 = __floats2bfloat162_rn(
                (float)(int)(int8_t)(w1 & 0xFF), (float)(int)(int8_t)(w1 >> 8));
            afr[0] = *reinterpret_cast<const unsigned*>(&a0);
            afr[1] = *reinterpret_cast<const unsigned*>(&a1);
        }
        // (4) ONE HMMA: dt' for tokens {gidm, gidm+8} x heads {2t4m, 2t4m+1}
        float dt4[4] = {0.f, 0.f, 0.f, 0.f};
        scr_hmma_bf16(dt4, afr, bq);
        // (5) epilogue: e = exp(sm * dt' * cs[tok] - K[head]); the mud term
        //     factors out of the token sum (exact in R; fp round -> radius)
        const float cs0 = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(p_rec + RO_CS + 2 * gidm));
        const float cs1 = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(
                p_rec + RO_CS + 2 * (gidm + 8)));
        const float Kh0 = Kflat[kh * G + 2 * t4m];
        const float Kh1 = Kflat[kh * G + 2 * t4m + 1];
        float e00 = __expf(sm_scale * dt4[0] * cs0 - Kh0);
        float e01 = __expf(sm_scale * dt4[1] * cs0 - Kh1);
        float e10 = __expf(sm_scale * dt4[2] * cs1 - Kh0);
        float e11 = __expf(sm_scale * dt4[3] * cs1 - Kh1);
        float s0 = e00 + e10, s1 = e01 + e11;   // this lane's 2-token partial
        #pragma unroll
        for (int st = 4; st < 32; st <<= 1) {   // sum the 8 gidm-lanes
            s0 += __shfl_xor_sync(~0u, s0, st);
            s1 += __shfl_xor_sync(~0u, s1, st);
        }
        // (6) mud factor per head + publish
        const __nv_bfloat16 smu = *reinterpret_cast<const __nv_bfloat16*>(
            p_rec + RO_MUS);
        const float smuf = __bfloat162float(smu);
#if RNK == 8
        const uint4* mup = reinterpret_cast<const uint4*>(p_rec + RO_MU);
        uint32_t mm4[MUW];
        #pragma unroll
        for (int j = 0; j < MUW / 4; ++j) {
            const uint4 m4 = *(mup + rr * (MUW / 4) + j);
            mm4[4*j+0]=m4.x; mm4[4*j+1]=m4.y; mm4[4*j+2]=m4.z; mm4[4*j+3]=m4.w;
        }
        int mia = 0, mib = 0;
        #pragma unroll
        for (int j = 0; j < MUW; ++j) {
            mia = __dp4a((int)mm4[j], (int)q8d[gl][rr * MUW + j], mia);
            mib = __dp4a((int)mm4[j], (int)q8d[gl + 4][rr * MUW + j], mib);
        }
        float mda = (float)mia, mdb = (float)mib;
        #pragma unroll
        for (int st = 1; st < 8; st <<= 1) {
            mda += __shfl_xor_sync(~0u, mda, st);
            mdb += __shfl_xor_sync(~0u, mdb, st);
        }
        const float muda = mda * qsc[gl] * smuf;      // heads gl, gl+4
        const float mudb = mdb * qsc[gl + 4] * smuf;
#else
        // MUC column: (row g, col RNK) lives at lane (gidm=g, t4m=RNK/2), c0
        const float mda = (float)__shfl_sync(~0u, c0raw, 4 * gl + RNK / 2);
        const float mdb = (float)__shfl_sync(~0u, c0raw, 4 * (gl + 4) + RNK / 2);
        const float muda = mda * qsc[gl] * smuf;
        const float mudb = mdb * qsc[gl + 4] * smuf;
#endif
        // (7) radius: eps = sm * nrmC * max|qt| * 2^-8 + fp slack
        float qtm = fmaxf(fabsf(qtv0), fabsf(qtv1));
        #pragma unroll
        for (int st = 1; st < 32; st <<= 1)
            qtm = fmaxf(qtm, __shfl_xor_sync(~0u, qtm, st));
        const float nrmC = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(p_rec + RO_MUS + 2));
        const float eps = sm_scale * nrmC * qtm * 0.00390625f + 1e-6f;
        const float elo = __expf(-eps);
        // (8) publish: score_h = LOWER bound mass' * e^{sm*mud} * e^{-eps};
        //     head h's token sum lives in lanes with 2*t4m==h(-1): gather by
        //     shfl from the owning column, write from the gl-octet leaders.
        //     s0/s1 already warp-summed across gidm-lanes (all lanes hold
        //     their column pair's totals). Lane (rr==0) of each octet writes
        //     its 2 heads: sums for head gl live in column t4m = gl>>1 ...
        const int col_a = gl >> 1, sel_a = gl & 1;         // head gl
        const int col_b = (gl + 4) >> 1, sel_b = (gl + 4) & 1;
        const float sa0 = __shfl_sync(~0u, s0, col_a);
        const float sa1 = __shfl_sync(~0u, s1, col_a);
        const float sb0 = __shfl_sync(~0u, s0, col_b);
        const float sb1 = __shfl_sync(~0u, s1, col_b);
        const float sa = sel_a ? sa1 : sa0;
        const float sb = sel_b ? sb1 : sb0;
        if (rr == 0) {
            Sp[(long)gl * MP + pp] =
                sa * __expf(sm_scale * muda) * elo;
            Sp[(long)(gl + 4) * MP + pp] =
                sb * __expf(sm_scale * mudb) * elo;
            if (gl == 0) Rp[pp] = eps;
        }
        issue_rec(pp + 2 * nworker, pslot);
        pslot ^= 1;
        __syncwarp();
    }
    cpasync_wait<0>();
}

void rki4_score_bt_aos_screen(
    torch::Tensor q, torch::Tensor rec, torch::Tensor btt, torch::Tensor nsh,
    torch::Tensor S, torch::Tensor rad, torch::Tensor Kflat, double scale,
    int64_t zsplit, int64_t n_req) {
    const int n_kv = rec.size(1);
    const int G = S.size(2);
    const int MP = S.size(3);
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && q.stride(2) == 1, "q");
    TORCH_CHECK(rec.scalar_type() == torch::kUInt8 && rec.is_contiguous()
                && rec.dim() == 3 && rec.size(2) == RKI4_RECB_HOST, "rec");
    TORCH_CHECK(S.is_contiguous() && S.scalar_type() == torch::kFloat, "S");
    TORCH_CHECK(rad.is_contiguous() && rad.scalar_type() == torch::kFloat
                && rad.size(-1) == MP, "rad");
    TORCH_CHECK(Kflat.is_contiguous() && Kflat.scalar_type() == torch::kFloat
                && Kflat.numel() == (long)n_kv * G, "Kflat");
    TORCH_CHECK(q.size(2) == 128 && G == 8,
                "screen covers the G8 d128 p16 flagship");
    dim3 grid(n_kv, (unsigned)n_req, (unsigned)zsplit);
    auto stream = at::cuda::getCurrentCUDAStream();
    rki4_screen_kernel<16, 128, true><<<grid, TB, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        rec.data_ptr<uint8_t>(), btt.data_ptr<int>(), nsh.data_ptr<int>(),
        S.data_ptr<float>(), rad.data_ptr<float>(), Kflat.data_ptr<float>(),
        (float)scale, n_kv, G, MP, (int)zsplit,
        q.stride(0), q.stride(1), btt.stride(0), (long)rec.size(0));
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "screen launch: ", cudaGetErrorString(e));
}
#endif  // RKI4_MMA_SCREEN

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m_) {
    m_.def("r8score", &r8score);
    m_.def("r8score_aos", &r8score_aos);
    m_.def("rki4_score_bt", &rki4_score_bt,
           py::arg("q"), py::arg("v4"), py::arg("vs"), py::arg("c8"),
           py::arg("cs"), py::arg("mu8"), py::arg("mus"), py::arg("bt"),
           py::arg("nsh"), py::arg("S"), py::arg("sm_scale"),
           py::arg("zsplit"), py::arg("n_req"),
           py::arg("sl_rope") = py::none(),
           py::arg("csc_rope") = py::none(),
           py::arg("qout_rope") = py::none(),
           py::arg("hmax_pub") = py::none());
    m_.def("rki4_score_bt_aos", &rki4_score_bt_aos);
    m_.def("rki4_score_bt_aos_fold", &rki4_score_bt_aos_fold);
    m_.def("rki4_score_bt_aos_flat", &rki4_score_bt_aos_flat);
    m_.def("rki4_score_bt_aos_flat_fold", &rki4_score_bt_aos_flat_fold);
#ifdef RKI4_MMA_SCREEN
    m_.def("rki4_score_bt_aos_screen", &rki4_score_bt_aos_screen);
#endif
}
"""

_MOD = None
_SMS = None


# NOROPE (round A): scorer-side rope absorption -- the python flag only
# controls whether rki4_score_only passes the (sl, csc, qout) triple; the
# kernel branches on qout != nullptr at runtime (one binary, both paths).
_NOROPE = (os.environ.get("LOCKS_NOROPE", "0") == "1"
           or os.environ.get("LOCKS_QFIRST_CO", "0") == "1")

# HMAX HANDSHAKE (lever 1, 2026-07-21): when on, the score launch passes
# st._nrmtopb_hmax so the kernel-end epilogue publishes per-(r,kh,g) page
# maxima (f2u atomicMax), and the paired select consumes them (has_hm=1
# skips its phase-1 sweep; the kernel resets the buffer after reading --
# rest-state invariant, graph-replay safe).  The publish and the consume are
# paired per launch via st._hmax_armed (set by the score entry that actually
# published -- six-slab BT + coqkv co-kernel; the AOS entries and the
# memory-profile zero_() path do NOT publish and leave the flag off).
_HMAX_ON = os.environ.get("LOCKS_HMAX", "0") == "1"

# ---- hmax handshake (owned here since the quad/clse variants were deleted) -- #
GMX = 8                     # max GQA group supported (mirrors the CUDA constant)
_F2U_NEGINF = 0x007FFFFF    # order-preserving-uint encoding of -inf


def _ensure_hmax(st):
    """Per-state hmax handshake buffer (max_reqs, n_kv, GMX) int32, rest state =
    f2u(-inf).  Allocated once at the first EAGER call (before any graph
    capture); the score kernel atomicMax-es into it (write_hm=1) and the fused
    nrm+topb consumer reads + resets it, so the rest-state invariant holds at
    every kernel entry (graph-replay safe)."""
    hm = getattr(st, "_nrmtopb_hmax", None)
    R = int(getattr(st, "max_reqs", 0) or st.score.shape[0])
    if hm is None or hm.shape[0] < R:
        hm = torch.full((R, int(st.n_kv), GMX), _F2U_NEGINF,
                        dtype=torch.int32, device=st.score.device)
        st._nrmtopb_hmax = hm
    return hm
_HMAX_ANNOUNCED = False
_SM120_SAID = False       # one-shot: arch dispatch to the plain-torch twin
_NARROWBT_SAID = False    # one-shot: profile-pass score_h zeroing


def _hmax_buf(st):
    """The handshake publish buffer when the lever is on, else None (the
    kernel's hmax_pub == nullptr legacy path).  Allocation is lazy-eager
    (first call precedes any graph capture; graph-safe on hit)."""
    if not _HMAX_ON:
        return None
    return _ensure_hmax(st)


def _get(verbose: bool = False):
    global _MOD
    if _MOD is None:
        # i-axis guard (knob contract P2): this TU's V unpack is int4-
        # hardcoded (nibble PRMT path). A non-i4 config must never reach a
        # CUDA build -- loud here, no fallback; the sm_120 torch twin and
        # the P4 i-axis template cases are the only other lanes.
        from .rki4_state import IBITS as _ibits
        assert _ibits == 4, (
            f"locks rki4 CUDA score TU is int4-only, LOCKS_RKI4_IBITS="
            f"{_ibits}: the i-axis CUDA case is deferred (P4); non-i4 runs "
            "the sm_120 torch twin lane only")
        from torch.utils.cpp_extension import load_inline
        import os as _os
        _cf = ["-O3", "--use_fast_math", _arch.arch_flag()]
        if _os.environ.get("LOCKS_PTXAS_V"):       # register/occupancy report
            _cf += ["-Xptxas=-v"]
        if _os.environ.get("RKI4_MMA"): _cf += ["-DRKI4_MMA"]
        # RKI4_PACK (rank campaign v2, r4/r2 systems opt): pack 8/RNK pages per
        # warp into the m16n8k32 mma's 8 fixed basis columns so a rank-4 summary
        # fills all 8 columns (page0 -> cols 0-3, page1 -> cols 4-7) and issues
        # HALF the (rank-independent) mma per page. Six-slab default variant
        # only (RNK==4); the #ifdef leaves the RNK==8 SASS untouched.
        if _os.environ.get("RKI4_PACK") or _os.environ.get("LOCKS_RKI4_PACK"):
            _cf += ["-DRKI4_PACK"]
        # PROBE flags (diagnostic, default-OFF): skip one slab's global loads so
        # ncu t_requests attributes the rank-independent load-instruction count
        # to slabs. Not a deployment path (scores are wrong); RNK==8 SASS
        # byte-identical when unset.
        for _pk in ("LOCKS_PROBE_SKIP_MU", "LOCKS_PROBE_SKIP_C8",
                    "LOCKS_PROBE_SKIP_V4"):
            if _os.environ.get(_pk):
                _cf += [f"-D{_pk}"]
        # LEVER C (RNK==2 only): aligned full-word c8 code load (kills the
        # 2-byte sub-word penalty). RNK==8/4 paths + SASS byte-identical (the
        # define is inert outside the RNK==2 branch).
        if _os.environ.get("LOCKS_C8WIDE"):
            _cf += ["-DLOCKS_C8WIDE"]
        # C' (RNK==2 only): relaid c8 layout -- one aligned 4-byte load fetches
        # lane rr's two tokens (rr, rr+8), HALVING c8 requests. MUST pair with
        # the build's LOCKS_C8RELAY relay (rki4_build); r4/r8 + SASS unchanged
        # (the define lives only in the RNK==2 elif branch).
        if _os.environ.get("LOCKS_C8RELAY"):
            _cf += ["-DLOCKS_C8RELAY"]
        # LEVER B (RNK<8 only): rank-tuned MLP map -- unroll-and-jam KJ pages/
        # warp with hoisted v4 loads + dead-lane skip, hiding the exposed v4
        # latency. RNK==8 SASS byte-identical (the branch is RNK<8-gated).
        # LOCKS_RANKMAP_KJ overrides the per-rank pages/iter for register tuning.
        if _os.environ.get("LOCKS_RANKMAP"):
            _cf += ["-DLOCKS_RANKMAP"]
        if _kj := _os.environ.get("LOCKS_RANKMAP_KJ"):
            _cf += [f"-DLOCKS_RANKMAP_KJ={int(_kj)}"]
        # BIAS landed DEFAULT-ON 2026-07-16 (byte-gate PASS + event-time win
        # -5.3%/-8.4% at 256K/1M on top of SRED: it shortens the dependent
        # chain off the stall-carrying cw load; SCORE_KERNEL_ISSUE_BOUND.md
        # section 9). RKI4_MMA_BIAS=0 is the explicit kill switch.
        if _os.environ.get("RKI4_MMA_BIAS", "1") != "0":
            _cf += ["-DRKI4_MMA_BIAS"]
        # PEEL landed DEFAULT-ON 2026-07-16 (byte-gate PASS, -3.4%/-4.1% at
        # 256K/1M on SRED+BIAS; exact tt=0 closed form skips 2 exps/tile).
        # RKI4_MMA_PEEL=0 is the explicit kill switch.
        if _os.environ.get("RKI4_MMA_PEEL", "1") != "0":
            _cf += ["-DRKI4_MMA_PEEL"]
        # SRED landed DEFAULT-ON 2026-07-16 (byte-gate PASS + event-time win,
        # ours_doc/SCORE_KERNEL_ISSUE_BOUND.md section 7). RKI4_MMA_SRED=0 is
        # the explicit kill switch (compiles the pre-SRED loop).
        if _os.environ.get("RKI4_MMA_SRED", "1") != "0":
            _cf += ["-DRKI4_MMA_SRED"]
        # LSE2 landed DEFAULT-ON 2026-07-16 night (BT-shape re-profile: kernel
        # ISSUE-bound, LSE2 -7.3%/-7.2% free-clock at 512K/1M shapes; cmpv
        # selection-identity gate PASS; token-equality PASS; E2E 512K +2.7%,
        # 1M +1.8%, 256K neutral -- SCORE_KERNEL_ISSUE_BOUND.md section 13).
        # The old dense-1M refutation was build-vintage-specific (dense now
        # also faster). RKI4_MMA_LSE2=0 is the explicit kill switch.
        if _os.environ.get("RKI4_MMA_LSE2", "1") != "0":
            _cf += ["-DRKI4_MMA_LSE2"]
        # Fix1 A-fragment occupancy knob (candidate experiment): default B
        # (Af spilled to smem, 9 blocks/SM); LOCKS_FIX1_AF_HOIST=1 = variant A
        # (Af hoisted in registers, ~7 blocks/SM). Only affects the deployed
        # BT/PGT16/G8 pipelined loop.
        if _os.environ.get("LOCKS_FIX1_AF_HOIST"): _cf += ["-DLOCKS_FIX1_AF_HOIST"]
        if _os.environ.get("LOCKS_FIX1_CARRYROW"): _cf += ["-DLOCKS_FIX1_CARRYROW"]
        if _os.environ.get("RKI4_MMA_AOS"): _cf += ["-DRKI4_MMA_AOS"]
        if _os.environ.get("RKI4_MMA_ALLG"):
            # ALLG (rank campaign v2): tensor-core projection for G4 models.
            # Requires the AOS build; excludes MUC (kernel #errors enforce).
            _cf += ["-DRKI4_MMA_ALLG"]
        if _os.environ.get("RKI4_MMA_PFR"): _cf += ["-DRKI4_MMA_PFR"]
        if _os.environ.get("RKI4_MMA_S1PROBE"): _cf += ["-DRKI4_MMA_S1PROBE"]
        if _os.environ.get("RKI4_MMA_FLAT"): _cf += ["-DRKI4_MMA_FLAT"]
        if _os.environ.get("RKI4_CP_NOEXP"): _cf += ["-DRKI4_CP_NOEXP"]
        if _os.environ.get("RKI4_CP_NODT"): _cf += ["-DRKI4_CP_NODT"]
        from .rki4_state import MUC as _muc
        # rank v2: MUC is AOS-only (kernel #error enforces AOS+BIAS);
        # never emit the define on a six-slab build.
        if _muc and _os.environ.get("RKI4_MMA_AOS"):
            _cf += ["-DRKI4_MMA_MUC"]
        if _os.environ.get("LOCKS_SCREEN"): _cf += ["-DRKI4_MMA_SCREEN"]
        if _os.environ.get("LOCKS_PMAX_FOLDR"): _cf += ["-DRKI4_MMA_FOLDR"]
        # Removed opt-in arms (2026-07-21 cleanup, Wave 3a; each REFUTED with
        # its mechanism in ours_doc/REFUTED_ARMS_INDEX.md; recover from git
        # tag pre-cleanup-2026-07-21): RKI4_NOUNPACK, RKI4_CUNROLL_N,
        # RKI4_CPASYNC, RKI4_RING, RKI4_MMA_CPA, RKI4_MMA_DEDUP,
        # RKI4_MMA_QREG, RKI4_MMA_CPRMT, RKI4_MMA_SVFOLD, RKI4_MMA_OCC10,
        # RKI4_MMA_STAGE, RKI4_MMA_VCP, RKI4_MMA_OCC9, RKI4_MMA_ILV2,
        # RKI4_MMA_ACC2, RKI4_MMA_DT2, LOCKS_BTPF, RKI4_NW.
        # Rank knob (doc 19/21): ONE env drives the python layout
        # (rki4_state.RNK) and the kernel -DRNK; they assert-match at import.
        if _rk := _os.environ.get("LOCKS_RKI4_RANK"): _cf += [f"-DRNK={int(_rk)}"]
        # ROPE GEOMETRY (model-agnostic, 2026-07-22): the NOROPE/QFIRST_CO
        # staging prelude ropes q in-kernel, so the arch's (rotary_dim,
        # neox) pair is compiled in.  GLM geometry -> NO flags and NO name
        # suffix, i.e. the deployed TU + its cached binary are unchanged;
        # any other arch gets its own defines AND its own extension name, so
        # a warm torch-extensions cache can never hand Llama GLM's rope.
        from ..backend import _runtime as _rt
        _cf += _rt.rope_cflags()
        _MOD = load_inline(name="locks_rki4_score" + _rt.rope_tag(),
                           cpp_sources="",
                           cuda_sources=_SRC, extra_cuda_cflags=_cf,
                           verbose=bool(verbose or _os.environ.get("LOCKS_PTXAS_V")))
    return _MOD


def _sm_count() -> int:
    """Device SM count (a device constant; cached, not step-keyed state)."""
    global _SMS
    if _SMS is None:
        _SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count
    return _SMS


def auto_zsplit(n_req: int, n_kv: int, max_pages: int, mult=None) -> int:
    """grid.z page-split: fill ~5 blocks per SM at (n_req x n_kv) base CTAs,
    bounded below the point where a split owns < ~8 pages.  Pure host integer
    math on launch-shape constants -> stable under CUDA-graph capture.
    Rationale: low-n_kv models (e.g. n_kv=2 at TP=1) would otherwise launch
    only n_kv x z blocks and starve the device.

    TARGET = 5 * SMs for LOW BATCH (n_req <= 4), else 4 * SMs.  After the G>4
    pipeline drop (PIPE = ... && G4) the d128/p16 g-tile kernel holds 5 CTAs/SM
    (ncu: Block Limit Registers 4->5, 95 regs).  At low batch the grid is < 1
    wave, so filling the 5th slot is a measured bs=1 GLM score win of ~7%
    (128K 41.4->38.5us, 1M 302->280us; z=6*SMs over-splits and regresses).
    At high batch the base (n_req*n_kv) already fills the machine and 5*SMs only
    pushes the grid past a whole-wave boundary (partial-wave tail), so the
    proven 4*SMs is kept.  z-split partitions pages across CTAs writing DISJOINT
    score_h slots -> bitwise-invariant to the target; a pure launch-shape (perf)
    knob (scratch_rki4_port/ztune.py)."""
    if mult is None:
        mult = 5 if n_req <= 4 else 4
    target = mult * _sm_count()
    z = -(-target // max(n_req * n_kv, 1))          # ceil div
    return max(1, min(z, max(1, max_pages // 8), 65535))


def _lowbatch_mult(n_req: int, max_pages: int, G: int, d: int):
    """Low-batch (n_req <= 4) CTAs/SM target for auto_zsplit, mma-path-aware.

    Returns None for every path except the RKI4_MMA G8-d128 flagship, so the
    dp4a kernel and all non-flagship geometries keep the shipped 5*SM EXACTLY
    (zsplit is a bitwise-invariant launch knob, but this keeps the default
    untouched).  The mma kernel frees registers (96 -> 56) to hold 9 CTAs/SM,
    and its now-dominant stall is the un-pipelined V-load long_scoreboard --
    hidden by MORE resident warps.  The optimal split grows with the working
    set (MEASURED, h200x4-04 bs1, cuda-event median): 64K wants 5*SM (L2-
    resident: more splits only add per-split fixed overhead), 128K ~9, 256K
    ~16, 1M ~32.  mult = clamp(max_pages/1024, 5, 32) fits all five measured
    ctx within ~0.5% of their per-ctx optimum, is monotone in ctx, and leaves
    <= 64K UNCHANGED at 5 (no short-ctx regression).  LOCKS_RKI4_MMA_ZMULT
    forces a fixed mult (A/B tuning)."""
    # mma path activates only for the G8 instantiation (g8_enabled(): the C++
    # host picks g8 unless LOCKS_SCORE_NO_G8 forces the runtime g-tile path).
    if (n_req <= 4 and os.environ.get("RKI4_MMA") and G == 8 and d == 128
            and os.environ.get("LOCKS_SCORE_NO_G8") is None):
        ov = os.environ.get("LOCKS_RKI4_MMA_ZMULT")
        # 2026-07-17 re-tune on h200x8-03 (post-AOS/LSE2; the 32-cap dated
        # from h200x4-04 pre-AOS): z-sweep at the BT shapes shows a clean
        # bowl at ONE EXACT RESIDENT WAVE (8 CTAs/SM at this kernel's
        # occupancy; grid 1056 = 132 SMs x 8): mult 8 beats 32 by -9.4%
        # @P=32768 (78.7->71.3us) and -4.0% @P=65536 (139.0->133.6),
        # monotone worse toward more waves (4224 CTAs = +21%). 128K keeps
        # mult 8 (old formula already 8); <=64K keeps 5 (unmeasured today,
        # old measured optimum). Bitwise-invariant launch knob; matched-pair
        # E2E gated before this landed (SCORE_KERNEL_ISSUE_BOUND.md 14).
        return int(ov) if ov else max(5, min(8, max_pages // 1024))
    return None


_AOS_CACHE = {}


def _pack_aos(v4, vs, c8, cs, mu8, mus):
    """Repack the six dense summary tensors into (n_kv, P, 832) uint8 records
    ([v4 512 | mu 128 | c8 128 | cs 32 | vs 16 | mus 2 | pad 14], p16 d128).
    Cached by (data_ptr, shape): pack once per summary (graph-safe on hit)."""
    key = (v4.data_ptr(), tuple(v4.shape))
    rec = _AOS_CACHE.get(key)
    if rec is not None:
        return rec
    n_kv, P, rnk, dh = v4.shape
    assert rnk == 8 and dh == 64 and c8.shape[2] == 16, \
        "AOS pack covers the p16 d128 flagship only"
    rec = torch.zeros(n_kv, P, 832, dtype=torch.uint8, device=v4.device)
    rec[..., 0:512]   = v4.reshape(n_kv, P, 512)
    rec[..., 512:640] = mu8.view(torch.uint8).reshape(n_kv, P, 128)
    rec[..., 640:768] = c8.view(torch.uint8).reshape(n_kv, P, 128)
    rec[..., 768:800] = cs.view(torch.uint8).reshape(n_kv, P, 32)
    rec[..., 800:816] = vs.view(torch.uint8).reshape(n_kv, P, 16)
    rec[..., 816:818] = mus.view(torch.uint8).reshape(n_kv, P, 2)
    _AOS_CACHE[key] = rec
    return rec


def rki4_score(q, v4, vs, c8, cs, mu8, mus, S, sm_scale, zsplit=None):
    """Packed-tensor entry (the standalone contract; gate battery + timing).

    q (R, n_kv*G, d) bf16; v4/vs/c8/cs/mu8/mus in the (n_kv, P, ...) packed
    layout of ``build_summary``; S (R, n_kv, G, P) fp32 out (caller-owned).
    ``zsplit`` None = auto-size from the device SM count.  RKI4_MMA_AOS
    builds repack into the 832B record (cached; identical bytes) and launch
    the record kernel through ``r8score_aos``."""
    if zsplit is None:
        n_req = q.shape[0]
        mult = _lowbatch_mult(n_req, v4.shape[1], S.shape[2], v4.shape[3] * 2)
        zsplit = auto_zsplit(n_req, v4.shape[0], v4.shape[1], mult=mult)
    if os.environ.get("RKI4_MMA_AOS"):
        rec = _pack_aos(v4, vs, c8, cs, mu8, mus)
        _get().r8score_aos(q, rec, S, float(sm_scale), int(zsplit))
        return
    _get().r8score(q, v4, vs, c8, cs, mu8, mus, S, float(sm_scale),
                   int(zsplit))


def rki4_score_only(st, layer: int, q, block_table, seq_lens, n_req: int,
                    scale: float) -> None:
    """HOT Stage-A.1 (rki4): write ``st.score`` (R, n_kv, MP) fp32 for the
    selectable region from the layer's packed slabs.

    The BT kernel writes the per-head page LSE into ``st.score_h``
    (R, n_kv, G, MP) for pages [0, n_sel_hi); the GQA combine then reduces it
    into ``st.score`` exactly where the quad/clse path combines ("nrm" = the
    fused CUDA nrm+topb consumer when available (G <= 8), else the shared
    ``_nrm_launch`` passes; "max" = fixed-shape amax).  ``st.n_sel_hi``
    must have been refreshed this step (``derive_page_params``).  seq_lens is
    unused: every scored page is finalized by construction (window >= 1), so
    no per-token validity mask exists.  q is the CALLER's query tensor and the
    launch stream is resolved at call time, so a future refresh-period /
    side-stream scheduler can drive this entry unchanged.  Fixed launch
    shapes, no host sync, no allocation -> full-CUDA-graph safe."""
    # sm_120 (Blackwell): the hand-CUDA rki4 score kernel's sm_120 dispatch
    # lane was removed (commit 2404b45). Route to the plain-torch core-logic
    # twin, which fills st.score_h identically (byte-faithful to the G1 torch
    # reference incl. the i8-q round-trip). EXPLICIT arch gate -- like FA's
    # per-arch kernels, NOT a silent wrong-arch fallback: Hopper falls through
    # to the CUDA kernel below. Downstream combine/top-b run on Triton
    # (LOCKS_FUSE_NRMTOPB=0 / LOCKS_TOPB_TRITON=1); decode on Triton
    # (LOCKS_DECODE_TRITON=1).
    from .. import arch as _arch
    # LOCKS_FORCE_SCORE_TORCH=1: EXPLICIT opt-in to the plain-torch score twin
    # on ANY arch (not a silent fallback -- loud one-shot log below, same
    # discipline as the sm_120 gate).  Its one use is page-geometry validation
    # on Hopper: the deployed hand-CUDA scorer is page {16,32}-only, so page 64
    # (an appendix ps-knob study) would TORCH_CHECK-abort in the CUDA launch;
    # the torch twin is page-parametric and byte-faithful to the G1 torch
    # reference, so it yields the SAME score_h (hence page rankings) that the
    # sm_120 b40x4 lane already produces for page 64.  NOT a deployment path.
    _force_torch = os.environ.get("LOCKS_FORCE_SCORE_TORCH", "0") == "1"
    if _arch.is_blackwell_sm120() or _force_torch:
        # EXPLICIT arch dispatch, now also LOUD (no-fallback rule,
        # 2026-07-22): this is a DIFFERENT implementation (plain torch, not
        # the hand-CUDA scorer) and a different performance class, so a
        # sm_120 boot must never look like the deployed Hopper config in a
        # log.  (Consistency note for the driver: the decode lane REFUSES
        # sm_120 outright at _runtime._decode_backend; see the audit's
        # class-D item.)
        global _SM120_SAID
        if not _SM120_SAID:
            _SM120_SAID = True
            _why = ("sm_120 (Blackwell)" if _arch.is_blackwell_sm120()
                    else "LOCKS_FORCE_SCORE_TORCH=1 (page-geometry validation "
                         "lane -- e.g. page 64 on Hopper, which the {16,32}-"
                         "only hand-CUDA scorer rejects)")
            print(f"[locks] FALLBACK rki4 score: {_why} -- the hand-CUDA "
                  "scorer lane is skipped; running the plain-torch twin "
                  "(score_h identical, NOT the deployed kernel path)",
                  flush=True)
        from . import rki4_score_torch as _r8t
        _r8t.rki4_score_torch(st, layer, q, block_table, seq_lens, n_req, scale)
        st._hmax_armed = False
        return
    z = st.zsplit if st.zsplit else auto_zsplit(
        n_req, st.n_kv, st.max_pages,
        mult=_lowbatch_mult(n_req, st.max_pages, st.G, st.d))
    # HMAX (lever 1): default = not armed for this launch; the publishing
    # branch below arms it.  A still-armed flag HERE means a previous publish
    # was never consumed (e.g. co-kernel published but the consume declined
    # and we re-score): wipe to rest state first, so pending maxima can never
    # pair with THIS launch's score_h bytes (atomicMax only grows; the two
    # compilations differ in the characterized FMA-residual class).
    if getattr(st, "_hmax_armed", False):
        st._nrmtopb_hmax.fill_(_F2U_NEGINF)
        st._hmax_armed = False
    if int(block_table.shape[1]) < int(st.max_pages):
        # Engine memory-profile pass: the placeholder block table is narrower
        # than the selectable range, so the kernel's __ldg(bt + pp) would read
        # PAST the table (nsh comes from the fake full-length request). The
        # six-array path survived this OOB by allocator luck; the record
        # tensor faults on it. Outputs are discarded in this pass: zero the
        # per-head scores and let the normal combine/select run deterministic.
        # Real tables are allocated full-width (max_blocks >= max_pages), so
        # this shape check never fires on a live step (graph-safe: host-only).
        # Made LOUD once (no-fallback rule, 2026-07-22): the invariant that
        # keeps it safe is "real block tables are allocated full width
        # (max_blocks >= max_pages)", so a line here on a LIVE step would
        # mean the selection ran on zeroed scores.
        global _NARROWBT_SAID
        if not _NARROWBT_SAID:
            _NARROWBT_SAID = True
            print(f"[locks] score: narrow block table "
                  f"({int(block_table.shape[1])} < max_pages "
                  f"{int(st.max_pages)}) -> score_h zeroed for this pass "
                  "(engine memory-profile pass only; outputs discarded)",
                  flush=True)
        st.score_h.zero_()
    elif os.environ.get("RKI4_MMA_AOS") and getattr(st, "pp_rec", None) is not None:
        # NOROPE covers the deployed six-slab entry only (round A scope);
        # the AOS entries stage q un-roped and would silently mis-score.
        assert not _NOROPE, \
            "LOCKS_NOROPE=1 is not wired for the RKI4_MMA_AOS entries"
        # BT-AOS: read the 832B records the build wrote through the state's
        # strided views (one pointer per page instead of six re-derivations;
        # nsys TAUCCHK: the six-array BT kernel was 89.9us/layer = 2.2x dense).
        if (os.environ.get("LOCKS_FLAT", "0") == "1"
                and os.environ.get("LOCKS_PMAX_FOLD", "0") == "1"):
            # FLAT+FOLD (doc 27c GO): mass score + folded pmax epilogue;
            # the select chain runs pmax_done (sel_pmax deleted).
            if getattr(st, "_flat_K", None) is None:
                st._flat_K = torch.full(
                    (int(st.L), int(st.n_kv) * int(st.G)),
                    float(os.environ.get("LOCKS_FLAT_K0", "30.0")),
                    dtype=torch.float32, device=q.device)
            st._flat_K_cur = st._flat_K[layer]
            from . import select_cuda as _selc
            _selc._ensure_split_scratch(st, st.n_kv, st.max_pages,
                                        want_pp=False)
            st._fold_Zs = int(z)
            _get().rki4_score_bt_aos_flat_fold(
                q, st.pp_rec[layer], block_table, st.n_sel_hi, st.score_h,
                st._sp_gpmax, st._sp_ghist, st._flat_K_cur, float(scale),
                int(z), int(n_req))
        elif os.environ.get("LOCKS_FLAT", "0") == "1":
            # FLAT-MASS (doc 18): route to the _flat entry with this layer's
            # per-head shifts. Seed = constant (doc 18d: any K is rank-exact;
            # the per-step update self-tightens); scratch allocates on the
            # eager seed pass (pre-capture).
            if getattr(st, "_flat_K", None) is None:
                st._flat_K = torch.full(
                    (int(st.L), int(st.n_kv) * int(st.G)),
                    float(os.environ.get("LOCKS_FLAT_K0", "30.0")),
                    dtype=torch.float32, device=q.device)
            st._flat_K_cur = st._flat_K[layer]
            _get().rki4_score_bt_aos_flat(q, st.pp_rec[layer], block_table,
                                          st.n_sel_hi, st.score_h,
                                          st._flat_K_cur, float(scale),
                                          int(z), int(n_req))
        elif os.environ.get("LOCKS_PMAX_FOLD", "0") == "1":
            # PMAX FOLD: the score epilogue emits per-CTA chunk maxima +
            # ghist zeros; sel_pmax is skipped downstream (select reads
            # gpmax at Zg = this launch's z). Scratch allocates on the
            # eager seed pass (pre-capture).
            from . import select_cuda as _selc
            _selc._ensure_split_scratch(st, st.n_kv, st.max_pages,
                                        want_pp=False)
            st._fold_Zs = int(z)
            _get().rki4_score_bt_aos_fold(q, st.pp_rec[layer], block_table,
                                          st.n_sel_hi, st.score_h,
                                          st._sp_gpmax, st._sp_ghist,
                                          float(scale), int(z), int(n_req))
        else:
            _get().rki4_score_bt_aos(q, st.pp_rec[layer], block_table,
                                     st.n_sel_hi, st.score_h, float(scale),
                                     int(z), int(n_req))
    else:
        # rank v2 (2026-07-27): the six-slab lane is rank-parametric (RNK-width
        # coeff loads + clamped dead-column reads + the pre-existing epilogue
        # gating), so RNK<8 rides the FULL flagship lane set here -- the v1
        # "AOS entries only" assert is retired (user ruling: r4/r2 must never
        # be worse than r8 at any context).
        v4, vs, c8, cs, mu8, mus, _tag = st.layer_state(layer)
        # ``st._q_preroped`` (set by the attention impl before EVERY select, and
        # by the co op's own absorbed branches): the rope-absorbing lanes do not
        # absorb at every token count.  LOCKS_QFIRST_CO absorbs at nt==1 only;
        # at nt>1 its op ran the stock fused F.linear + in-place vllm rotary, so
        # q arrives ALREADY roped and the staging prelude must not rope it a
        # second time.  Default False = the pre-2026-07-22 behaviour.
        if _NOROPE and not getattr(st, "_q_preroped", False):
            # NOROPE (round A): the forward's rope kernel is deleted; the
            # scorer ropes q in its staging prelude (pos = sl-1 from the
            # metadata seq_lens) and publishes the roped bf16 bytes to
            # st.q_rope for the split kernels.  _csc_cache is wired by
            # register._wire_norope; a missing cache here is a wiring bug.
            csc = getattr(st, "_csc_cache", None)
            if csc is None:
                from ..backend import _runtime as _rt
                csc = st._csc_cache = _rt._NOROPE_CSC
            assert csc is not None, \
                "LOCKS_NOROPE=1 but the cos_sin_cache is not wired " \
                "(register._wire_norope did not run)"
            _get().rki4_score_bt(q, v4, vs, c8, cs, mu8, mus, block_table,
                                 st.n_sel_hi, st.score_h, float(scale),
                                 int(z), int(n_req), sl_rope=seq_lens,
                                 csc_rope=csc, qout_rope=st.q_rope,
                                 hmax_pub=_hmax_buf(st))
        else:
            _get().rki4_score_bt(q, v4, vs, c8, cs, mu8, mus, block_table,
                                 st.n_sel_hi, st.score_h, float(scale),
                                 int(z), int(n_req),
                                 hmax_pub=_hmax_buf(st))
        st._hmax_armed = _HMAX_ON

def rki4_select_only(st, n_req: int) -> None:
    """Stage-A.2: GQA-combine st.score_h -> st.score and run page selection
    (fused nrm+topb when available).  Split from the score launch so a
    two-stream prep pipeline (LOCKS_STALE_PIPE) can overlap select(L) with
    score(L+1); calling score_only then select_only back-to-back on one
    stream is bitwise-identical to the old monolithic entry."""
    # HMAX (lever 1): consume the per-launch arming flag set by the score
    # entry that published (six-slab BT / coqkv).  has_hmax=True lets the
    # fused consumer skip its phase-1 sweep and read the published maxima
    # (bit-exact vs the sweep: same fp32 multiset, order-free max), then
    # reset the buffer (rest-state invariant).  Non-consuming paths restore
    # the rest state explicitly so a publish can never leak across steps.
    hm_armed = bool(getattr(st, "_hmax_armed", False))
    st._hmax_armed = False
    if st.combine == "nrm":
        # Fused single-pass nrm+topb consumer (the quad/clse TAIL-OPT): ONE
        # CUDA kernel combines st.score_h -> st.score AND runs the radix
        # top-b, then sets st._topb_done so the adapter's topb call no-ops.
        # Selection is byte-equal to the separate passes (tests/
        # test_tailopt.py::test_fused_nrmtopb_bitwise_eq_unfused + GATE A of
        # the rki4 port); the generic Triton combine grows with MP (~84 us/
        # layer extra at 128K bs1).  has_hmax follows the handshake arming
        # (LOCKS_HMAX=1 + a publishing score launch; default off).  G <= 8 is
        # the fused kernel's compiled bound.  NO-FALLBACK RULE (2026-07-22):
        # the fused consumer is the ONLY nrm path.  The old
        # LOCKS_FUSE_NRMTOPB=0 / LOCKS_TOPB_TRITON=1 escapes routed to the
        # deleted quad Triton combine; a build failure or an out-of-range G is
        # now a loud error, never a silent slower path.
        from . import select_cuda
        assert st.G <= 8, (
            f"locks rki4: fused nrm+topb supports G <= 8, got G={st.G}")
        assert select_cuda.fused_available(), (
            "locks rki4: the fused nrm+topb selector failed to build "
            "(LOCKS_CUDA_VERBOSE=1 prints the nvcc log); there is no "
            "fallback combine")
        _ensure_hmax(st)
        if hm_armed:
            global _HMAX_ANNOUNCED
            if not _HMAX_ANNOUNCED:
                _HMAX_ANNOUNCED = True
                print("[locks] HMAX handshake ACTIVE: score-kernel "
                      "publish -> select has_hm=1 (phase-1 sweep "
                      "skipped)", flush=True)
        select_cuda.nrmtopb_select_cuda(st, int(n_req), has_hmax=hm_armed)
    else:
        if hm_armed:
            st._nrmtopb_hmax.fill_(_F2U_NEGINF)       # no fused consumer
        torch.amax(st.score_h, dim=2, out=st.score)


def rki4_score_state(st, layer: int, q, block_table, seq_lens, n_req: int,
                     scale: float) -> None:
    """Monolithic prep entry (score + combine/select), unchanged contract."""
    rki4_score_only(st, layer, q, block_table, seq_lens, n_req, scale)
    rki4_select_only(st, int(n_req))
