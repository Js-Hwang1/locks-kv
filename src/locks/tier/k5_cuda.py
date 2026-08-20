"""K10 (2026-07-19): hand-CUDA fused graph-head evict+promote (+K8 clock tick).

The true-exposure profile (graphprof, --cuda-graph-trace=graph) showed the two
Triton scalar walkers (_k5_evict + _k5_promote) cost 27-64us EACH per step on
the in-graph critical path at bs=1 -- 4-6 dependent scalar loads per entry per
single-warp program, plus Triton's strong-atomic lowering.  This CUDA port
fuses both phases into ONE launch with a grid-stride loop, one thread per
entry, plain device atomics (no membar chains: every consumer is ordered by a
kernel boundary or a recorded event, so relaxed device atomicity suffices --
the same argument as the A+C "relaxed" land, sec 19).

Phase disjointness (why no barrier between evict and promote): evict rows hold
THIS prologue's owners, promote rows the fenced batch's owners (owner CAS at
assign, released only by promote); evict CASes p2s[old] for RESIDENT pages,
promote CASes p2s[blk] for PENDING (-2) pages, and one (l,kv,blk) cell cannot
be both.  Residency policy only -- bitwise-free by design (readers treat
-1/-2 alike; bytes install strictly before their mapping publishes).

Semantics are copied VERBATIM from k5.py::_k5_evict_kernel /
_k5_promote_kernel; TICK folds the K8 once-per-step clock advance
(k5.py::_step_setup at the head).  Kill switch: LOCKS_K5_HEADV=0.
"""
from __future__ import annotations

import os

import torch

from .. import arch as _arch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// ---- rope geometry (MODEL-AGNOSTIC, 2026-07-22) --------------------------- //
// rope_and_cache[_fixc] rope the decode token's K in-kernel (the forward's rope
// is deleted under NOROPE / QFIRST_CO).  The geometry used to be HARDCODED to
// GLM-4 (rotary_dim 64, interleaved pairing), silently wrong for half-split
// (neox) archs.  It is now a COMPILE-TIME pair from _runtime.rope_cflags(),
// which emits nothing on GLM geometry -- the deployed GLM TU is byte-identical
// to the pre-2026-07-22 build.  See src/locks/selection/rki4_score_cuda.py for
// the twin definition (the two kernels must agree element for element: the
// score kernel ropes q, this one ropes k).
#ifndef LOCKS_ROT_DIM
#define LOCKS_ROT_DIM 64
#endif
#ifndef LOCKS_ROT_NEOX
#define LOCKS_ROT_NEOX 0
#endif
#define LOCKS_ROT_HALF (LOCKS_ROT_DIM / 2)

__device__ __forceinline__ float locks_rope_elem(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ crow, const int dd) {
#if LOCKS_ROT_NEOX
  const int lo = (dd < LOCKS_ROT_HALF) ? dd : dd - LOCKS_ROT_HALF;
  const float c = __bfloat162float(crow[lo]);
  const float s = __bfloat162float(crow[LOCKS_ROT_HALF + lo]);
  const float e = __bfloat162float(x[lo]);
  const float o = __bfloat162float(x[lo + LOCKS_ROT_HALF]);
  return (dd < LOCKS_ROT_HALF)
      ? __fsub_rn(__fmul_rn(e, c), __fmul_rn(o, s))
      : __fadd_rn(__fmul_rn(o, c), __fmul_rn(e, s));
#else
  const int pr = dd >> 1;
  const float c = __bfloat162float(crow[pr]);
  const float s = __bfloat162float(crow[LOCKS_ROT_HALF + pr]);
  const float me = __bfloat162float(x[dd]);
  const float pa = __bfloat162float(x[dd ^ 1]);
  return (dd & 1) ? __fadd_rn(__fmul_rn(me, c), __fmul_rn(pa, s))
                  : __fsub_rn(__fmul_rn(me, c), __fmul_rn(pa, s));
#endif
}

#define ENTW 6   // entry words: (l, kv, blk, slot, gen, pad)

__global__ void k5_headv_kernel(
    const int* __restrict__ elist, const int* __restrict__ ecnt,
    const int* __restrict__ plist, const int* __restrict__ pcnt,
    int* __restrict__ p2s, int* __restrict__ s2p,
    int* __restrict__ owner, const int* __restrict__ gen,
    int* __restrict__ clock_w, int* __restrict__ fetch_w,
    const long NB, const long S, const int NKV, const int tick) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  if (tick && tid == 0) { *clock_w += 1; *fetch_w = 0; }
  // ---- phase 1: evict (this prologue's snapshotted batch) --------------- //
  const int en = *ecnt;
  for (int i = tid; i < en; i += stride) {
    const int h = elist[i * ENTW + 3];
    if (h < 0) continue;                       // DEAD row (assign give-up)
    const int l   = elist[i * ENTW + 0];
    const int kv  = elist[i * ENTW + 1];
    const int blk = elist[i * ENTW + 2];
    const long base = (long)(l * NKV + kv);
    const int old = s2p[base * S + h];
    if (old >= 0 && old != blk)               // unmap iff still mapped HERE
      atomicCAS(&p2s[base * NB + old], h, -1);
    s2p[base * S + h] = blk;
  }
  // ---- phase 2: promote (the fenced batch; disjoint from phase 1) ------- //
  const int pn = *pcnt;
  for (int i = tid; i < pn; i += stride) {
    const int h = plist[i * ENTW + 3];
    if (h < 0) continue;
    const int l   = plist[i * ENTW + 0];
    const int kv  = plist[i * ENTW + 1];
    const int blk = plist[i * ENTW + 2];
    const int g   = plist[i * ENTW + 4];
    const long base = (long)(l * NKV + kv);
    int installed = 0;
    if (gen[blk] == g && atomicCAS(&p2s[base * NB + blk], -2, h) == -2)
      installed = 1;
    if (!installed) s2p[base * S + h] = -1;   // stale: abandon the slot
    owner[base * S + h] = 0;                  // always release
  }
}

// S3 (2026-07-19): per-(layer, kv, logical-page) resolved table
// {slot, vb, blk, 0} built ONCE per step at the graph head, AFTER the
// evict/promote image mutations (the maps are step-stable from here; the
// in-step k5c -1->-2 claims are deliberately invisible -- readers treat
// them identically).  The decode split kernel then replaces its 3-deep
// dependent load chain (pt -> bt[pt] -> p2s/vbo[blk]) with one 16B line.
// bs=1 (request row 0).
__global__ void s3_build_kernel(
    const int* __restrict__ bt, const int* __restrict__ sl,
    const int* __restrict__ p2s, const int* __restrict__ vbo,
    int* __restrict__ tab,
    const long NB, const long MP, const int PAGE) {
  const int l = blockIdx.x, kv = blockIdx.y;
  const int NKV = gridDim.y;
  const int npg = (sl[0] + PAGE - 1) / PAGE;   // incl. the partial tail
  const long base = (long)(l * NKV + kv);
  int4* trow = reinterpret_cast<int4*>(tab + base * MP * 4);
  for (int pt = threadIdx.x; pt < npg; pt += blockDim.x) {
    const int blk = bt[pt];
    int4 e;
    e.x = p2s[base * NB + blk];
    e.y = vbo[blk];
    e.z = blk;
    e.w = 0;
    trow[pt] = e;
  }
}

void s3_build(torch::Tensor bt, torch::Tensor sl, torch::Tensor p2s,
              torch::Tensor vbo, torch::Tensor tab,
              int64_t L, int64_t nkv, int64_t NB, int64_t MP, int64_t page) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)L, (unsigned)nkv);
  s3_build_kernel<<<grid, 256, 0, stream>>>(
      bt.data_ptr<int>(), sl.data_ptr<int>(), p2s.data_ptr<int>(),
      vbo.data_ptr<int>(), tab.data_ptr<int>(),
      (long)NB, (long)MP, (int)page);
}

// Z1 (2026-07-19): mailbox snapshot + per-unit BUCKETIZE in one pass.
// The hr12/hr6 localization pair named _k5_assign's full-mailbox scan as
// the offload tax's kernel home (+0.029 of +0.042 busy at dm~7/unit):
// every (l,kv) program scanned the WHOLE snapshot.  This kernel replaces
// the 1.5MiB snap memcpy + the clamp launch and, while copying, scatters
// each PENDING entry's index into its unit's bucket, so assign reads only
// its own entries.  Bucket overflow takes assign's give-up path at source
// (revert CAS -2 -> -1, entry stays slot=-1 = DEAD to every walker, ovf
// latch).  Single block: the copy, the counter publish and the mail_cnt
// zero are program-ordered (the zev record follows on the same stream).
__global__ void mail_snap_bucketize_kernel(
    const int* __restrict__ mail, const int* __restrict__ mail_cnt,
    int* __restrict__ snap, int* __restrict__ snap_cnt,
    int* __restrict__ bcnt, int* __restrict__ bidx,
    int* __restrict__ p2s, int* __restrict__ ovf,
    const long NB, const int NKV, const int NU, const int MCAP,
    const int CAP, const int zero_src) {
  const int tid = threadIdx.x;
  const int TB = blockDim.x;
  const int c = min(*mail_cnt, MCAP);
  for (int u = tid; u < NU; u += TB) bcnt[u] = 0;
  __syncthreads();
  for (int i = tid; i < c; i += TB) {
    const int l = mail[i * 6 + 0], kv = mail[i * 6 + 1];
    const int blk = mail[i * 6 + 2], sl = mail[i * 6 + 3];
    const int g = mail[i * 6 + 4];
    snap[i * 6 + 0] = l; snap[i * 6 + 1] = kv; snap[i * 6 + 2] = blk;
    snap[i * 6 + 3] = sl; snap[i * 6 + 4] = g; snap[i * 6 + 5] = 0;
    if (sl == -1) {                       // PENDING: assign's work list
      const int u = l * NKV + kv;
      const int p = atomicAdd(&bcnt[u], 1);
      if (p < CAP) bidx[(long)u * CAP + p] = i;
      else {                              // give-up at source (bitwise-free)
        atomicCAS(&p2s[(long)u * NB + blk], -2, -1);
        snap[i * 6 + 3] = -1;             // stays DEAD to every walker
        *ovf = 1;
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
    *snap_cnt = c;
    if (zero_src) *((int*)mail_cnt) = 0;
  }
}

void mail_snap_bucketize(torch::Tensor mail, torch::Tensor mail_cnt,
                         torch::Tensor snap, torch::Tensor snap_cnt,
                         torch::Tensor bcnt, torch::Tensor bidx,
                         torch::Tensor p2s, torch::Tensor ovf,
                         int64_t NB, int64_t nkv, int64_t nu, int64_t mcap,
                         int64_t cap, bool zero_src) {
  auto stream = at::cuda::getCurrentCUDAStream();
  mail_snap_bucketize_kernel<<<1, 1024, 0, stream>>>(
      mail.data_ptr<int>(), mail_cnt.data_ptr<int>(),
      snap.data_ptr<int>(), snap_cnt.data_ptr<int>(),
      bcnt.data_ptr<int>(), bidx.data_ptr<int>(),
      p2s.data_ptr<int>(), ovf.data_ptr<int>(),
      (long)NB, (int)nkv, (int)nu, (int)mcap, (int)cap, zero_src);
}

// FA1 (2026-07-19, user design): fetch-ALL-selected per layer into a
// selection-indexed landing buffer, racy-published with per-page epoch
// flags (tag = clk*L + lidx, unique per (step, layer) -- stale entries
// never match, no clear pass).  Runs on a side stream forked after
// select(L); the split races it and falls back to pinned-in-place reads
// (identical bytes, I2) for pages that have not landed.  No residency
// protocol: this replaces the hot-cache machinery at bs=1.
__global__ void fa1_fetch_kernel(
    const int* __restrict__ tab,          // page_table row r=0: (n_kv, MP)
    const int* __restrict__ cnt,          // page_cnt row r=0: (n_kv,)
    const int* __restrict__ bt,           // block_table row 0
    const int* __restrict__ sl,           // seq_lens
    const int8_t* __restrict__ pool_valid,  // (L*NKV, NB) -- layer slice base
    const int16_t* __restrict__ poolv,    // pinned V (UVA)
    const int16_t* __restrict__ poolk,    // pinned K (UVA)
    int16_t* __restrict__ land_v, int16_t* __restrict__ land_k,
    int* __restrict__ flags,              // (NKV * LCAP)
    const int* __restrict__ clock_w,
    const long NB, const long MP, const int NKV, const int L,
    const int lidx, const int PAGE, const int D, const int LCAP,
    const int has_k) {
  const int PE = PAGE * D;                 // int16 elems per (page, kv) tile
  const int seq = sl[0];
  const int ep = clock_w[0] * L + lidx;
  const int tid = threadIdx.x, TB = blockDim.x;
  for (int w = blockIdx.x; ; w += gridDim.x) {
    const int kh = w / LCAP, j = w % LCAP;
    if (kh >= NKV) return;
    const int c = cnt[kh];
    if (j >= c || j >= LCAP) continue;
    const int pt = tab[(long)kh * MP + j];
    if ((pt + 1) * PAGE > seq) continue;   // partial tail: staged serves it
    const long blk = bt[pt];
    if (!pool_valid[((long)lidx * NKV + kh) * NB + blk]) continue;
    const long src = (((long)lidx * NB + blk) * NKV + kh) * (long)PE;
    const long dst = ((long)kh * LCAP + j) * (long)PE;
    for (int e = tid; e < PE; e += TB) land_v[dst + e] = poolv[src + e];
    if (has_k)
      for (int e = tid; e < PE; e += TB) land_k[dst + e] = poolk[src + e];
    __threadfence();                       // bytes before the flag (release)
    __syncthreads();
    if (tid == 0)
      atomicExch(&flags[(long)kh * LCAP + j], ep);
  }
}

// FA1-v3.2 STABLE-SLOT assignment (fixes v3's rank-shift amplification:
// the ascending-compacted table shifts every rank past a churn insertion,
// so slot==rank tags mismatched on ~half the table while content was ~90%
// unchanged).  Per (layer, kh): selected blk -> PERSISTENT landing slot
// while it stays selected.  Phase A (this kernel, one CTA per kh):
//   want[j] = (gen[bt[tab[j]]]<<32)|blk for landable selections
//   match  : want j matches slot s iff tag[s] == want[j]  (content-keyed;
//            a re-selected blk finds its old bytes wherever they sit)
//   claim  : unmatched wants take free slots (deterministic serial scan);
//            copy list (slot, pt) emitted for phase B
//   slot_of[j] = the slot; -1 = no landing (tail / not pool_valid).
// slot_of is RACY-SAFE by construction: the split re-verifies content via
// the tag at that slot, so any stale value can only miss -> pinned read.
__global__ void fa1_assign_v3_kernel(
    const int* __restrict__ tab, const int* __restrict__ cnt,
    const int* __restrict__ bt, const int* __restrict__ sl,
    const int8_t* __restrict__ pool_valid,
    const long long* __restrict__ flags, const int* __restrict__ gen,
    int* __restrict__ slot_of,            // (NKV * LCAP)
    int* __restrict__ copy_list,          // (NKV * LCAP * 2): slot, pt
    int* __restrict__ copy_cnt,           // (NKV)
    const long NB, const long MP, const int NKV, const int lidx,
    const int PAGE) {
  const int kh = blockIdx.x;
  const int t = threadIdx.x;              // 0..LCAP-1 (one thread per rank)
  const int LC = blockDim.x;
  extern __shared__ long long smem_ll[];
  long long* tag = smem_ll;               // [LC]
  long long* want = smem_ll + LC;         // [LC]
  int* used = reinterpret_cast<int*>(want + LC);        // [LC]
  int* mine = used + LC;                  // [LC] matched slot per rank
  const int seq = sl[0];
  const int c = cnt[kh];
  tag[t] = flags[(long)kh * LC + t];
  used[t] = 0;
  long long w = -1;
  int pt = -1;
  if (t < c) {
    pt = tab[(long)kh * MP + t];
    if ((pt + 1) * PAGE <= seq) {
      const long blk = bt[pt];
      if (pool_valid[(long)kh * NB + blk])
        w = ((long long)gen[blk] << 32) | (long long)blk;
    }
  }
  want[t] = w;
  __syncthreads();
  int ms = -1;
  if (w >= 0)
    for (int s = 0; s < LC; ++s)
      if (tag[s] == w) { ms = s; break; }
  mine[t] = ms;
  __syncthreads();
  if (ms >= 0) used[ms] = 1;              // unique wants -> no write race
  __syncthreads();
  if (t == 0) {                            // deterministic serial claim
    int n = 0, free_s = 0;
    for (int j = 0; j < LC; ++j) {
      if (want[j] < 0) { slot_of[(long)kh * LC + j] = -1; continue; }
      if (mine[j] >= 0) { slot_of[(long)kh * LC + j] = mine[j]; continue; }
      while (free_s < LC && used[free_s]) ++free_s;
      if (free_s >= LC) { slot_of[(long)kh * LC + j] = -1; continue; }
      used[free_s] = 1;
      slot_of[(long)kh * LC + j] = free_s;
      copy_list[((long)kh * LC + n) * 2 + 0] = free_s;
      copy_list[((long)kh * LC + n) * 2 + 1] = tab[(long)kh * MP + j];
      ++n;
    }
    copy_cnt[kh] = n;
  }
}

// Phase B: copy ONLY the claimed slots' pages pinned -> landing, publish
// the {blk,gen} tag after the bytes (release).  Grid-stride over the copy
// lists of all kh.
__global__ void fa1_copy_v3_kernel(
    const int* __restrict__ bt, const int8_t* __restrict__ pool_valid,
    const int16_t* __restrict__ poolv, const int16_t* __restrict__ poolk,
    int16_t* __restrict__ land_v, int16_t* __restrict__ land_k,
    long long* __restrict__ flags, const int* __restrict__ gen,
    const int* __restrict__ copy_list, const int* __restrict__ copy_cnt,
    const long NB, const int NKV, const int lidx,
    const int PAGE, const int D, const int LCAP, const int has_k) {
  const int PE = PAGE * D;
  const int tid = threadIdx.x, TB = blockDim.x;
  for (int w = blockIdx.x; ; w += gridDim.x) {
    const int kh = w / LCAP, i = w % LCAP;
    if (kh >= NKV) return;
    if (i >= copy_cnt[kh]) continue;
    const int s = copy_list[((long)kh * LCAP + i) * 2 + 0];
    const int pt = copy_list[((long)kh * LCAP + i) * 2 + 1];
    const long blk = bt[pt];
    const long long want = ((long long)gen[blk] << 32) | (long long)blk;
    const long slot = (long)kh * LCAP + s;
    const long src = (((long)lidx * NB + blk) * NKV + kh) * (long)PE;
    const long dst = slot * (long)PE;
    for (int e = tid; e < PE; e += TB) land_v[dst + e] = poolv[src + e];
    if (has_k)
      for (int e = tid; e < PE; e += TB) land_k[dst + e] = poolk[src + e];
    __threadfence();
    __syncthreads();
    if (tid == 0)
      atomicExch(reinterpret_cast<unsigned long long*>(&flags[slot]),
                 (unsigned long long)want);
  }
}

void fa1_step_v32(torch::Tensor tab, torch::Tensor cnt, torch::Tensor bt,
                  torch::Tensor sl, torch::Tensor pool_valid_l,
                  int64_t poolv_ptr, int64_t poolk_ptr,
                  torch::Tensor land_v, torch::Tensor land_k,
                  torch::Tensor flags, torch::Tensor gen,
                  torch::Tensor slot_of, torch::Tensor copy_list,
                  torch::Tensor copy_cnt,
                  int64_t NB, int64_t MP, int64_t nkv, int64_t lidx,
                  int64_t page, int64_t d, int64_t lcap, bool has_k) {
  TORCH_CHECK(flags.scalar_type() == torch::kInt64
              && slot_of.scalar_type() == torch::kInt32
              && lcap <= 1024, "fa1 v3.2 arg contract");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int LC = (int)lcap;
  const size_t smem = (size_t)LC * (8 + 8 + 4 + 4);
  fa1_assign_v3_kernel<<<(int)nkv, LC, smem, stream>>>(
      tab.data_ptr<int>(), cnt.data_ptr<int>(), bt.data_ptr<int>(),
      sl.data_ptr<int>(), pool_valid_l.data_ptr<int8_t>(),
      reinterpret_cast<const long long*>(flags.data_ptr<int64_t>()),
      gen.data_ptr<int>(), slot_of.data_ptr<int>(),
      copy_list.data_ptr<int>(), copy_cnt.data_ptr<int>(),
      (long)NB, (long)MP, (int)nkv, (int)lidx, (int)page);
  fa1_copy_v3_kernel<<<64, 256, 0, stream>>>(
      bt.data_ptr<int>(), pool_valid_l.data_ptr<int8_t>(),
      reinterpret_cast<const int16_t*>(poolv_ptr),
      reinterpret_cast<const int16_t*>(poolk_ptr),
      land_v.data_ptr<int16_t>(), land_k.data_ptr<int16_t>(),
      reinterpret_cast<long long*>(flags.data_ptr<int64_t>()),
      gen.data_ptr<int>(), copy_list.data_ptr<int>(),
      copy_cnt.data_ptr<int>(),
      (long)NB, (int)nkv, (int)lidx, (int)page, (int)d, (int)lcap,
      has_k ? 1 : 0);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "fa1 v3.2 launch: ", cudaGetErrorString(e));
}

// FA1-v3: the tagged fetch.  Per selected slot (kh, j) with block blk:
//   want = (gen[blk] << 32) | blk
//   flags[slot] == want  ->  SKIP (bytes already landed and unchanged;
//                            gen bumps on rewrite auto-invalidate)
//   else copy pinned V(+K) -> landing, __threadfence, publish want.
// Per-layer landings sized LCAP3 == the R1 budget (128): the buffer IS the
// attended set -- no admission/eviction, size-exact (user R2 ruling).
// Non-pool_valid pages (fresh staged) are neither copied nor tagged: the
// split's staged path serves them.  Racy by design: a consumer that reads
// a stale flag falls back to the pinned bytes (equal by I2).
__global__ void fa1_fetch_v3_kernel(
    const int* __restrict__ tab, const int* __restrict__ cnt,
    const int* __restrict__ bt, const int* __restrict__ sl,
    const int8_t* __restrict__ pool_valid,
    const int16_t* __restrict__ poolv, const int16_t* __restrict__ poolk,
    int16_t* __restrict__ land_v, int16_t* __restrict__ land_k,
    long long* __restrict__ flags, const int* __restrict__ gen,
    const long NB, const long MP, const int NKV, const int L,
    const int lidx, const int PAGE, const int D, const int LCAP,
    const int has_k) {
  const int PE = PAGE * D;
  const int seq = sl[0];
  const int tid = threadIdx.x, TB = blockDim.x;
  for (int w = blockIdx.x; ; w += gridDim.x) {
    const int kh = w / LCAP, j = w % LCAP;
    if (kh >= NKV) return;
    const int c = cnt[kh];
    if (j >= c || j >= LCAP) continue;
    const int pt = tab[(long)kh * MP + j];
    if ((pt + 1) * PAGE > seq) continue;   // partial tail: staged serves it
    const long blk = bt[pt];
    if (!pool_valid[((long)lidx * NKV + kh) * NB + blk]) continue;
    const long long want = ((long long)gen[blk] << 32) | (long long)blk;
    const long slot = (long)kh * LCAP + j;
    if (flags[slot] == want) continue;     // landed + unchanged: SKIP
    const long src = (((long)lidx * NB + blk) * NKV + kh) * (long)PE;
    const long dst = slot * (long)PE;
    for (int e = tid; e < PE; e += TB) land_v[dst + e] = poolv[src + e];
    if (has_k)
      for (int e = tid; e < PE; e += TB) land_k[dst + e] = poolk[src + e];
    __threadfence();                       // bytes before the tag (release)
    __syncthreads();
    if (tid == 0)
      atomicExch(reinterpret_cast<unsigned long long*>(&flags[slot]),
                 (unsigned long long)want);
  }
}

void fa1_fetch_v3(torch::Tensor tab, torch::Tensor cnt, torch::Tensor bt,
                  torch::Tensor sl, torch::Tensor pool_valid,
                  int64_t poolv_ptr, int64_t poolk_ptr,
                  torch::Tensor land_v, torch::Tensor land_k,
                  torch::Tensor flags, torch::Tensor gen,
                  int64_t NB, int64_t MP, int64_t nkv, int64_t L,
                  int64_t lidx, int64_t page, int64_t d, int64_t lcap,
                  bool has_k) {
  TORCH_CHECK(flags.scalar_type() == torch::kInt64
              && gen.scalar_type() == torch::kInt32,
              "fa1_v3: flags int64, gen int32");
  auto stream = at::cuda::getCurrentCUDAStream();
  fa1_fetch_v3_kernel<<<64, 256, 0, stream>>>(
      tab.data_ptr<int>(), cnt.data_ptr<int>(), bt.data_ptr<int>(),
      sl.data_ptr<int>(), pool_valid.data_ptr<int8_t>(),
      reinterpret_cast<const int16_t*>(poolv_ptr),
      reinterpret_cast<const int16_t*>(poolk_ptr),
      land_v.data_ptr<int16_t>(), land_k.data_ptr<int16_t>(),
      reinterpret_cast<long long*>(flags.data_ptr<int64_t>()),
      gen.data_ptr<int>(),
      (long)NB, (long)MP, (int)nkv, (int)L, (int)lidx,
      (int)page, (int)d, (int)lcap, has_k ? 1 : 0);
}

void fa1_fetch(torch::Tensor tab, torch::Tensor cnt, torch::Tensor bt,
               torch::Tensor sl, torch::Tensor pool_valid,
               int64_t poolv_ptr, int64_t poolk_ptr,
               torch::Tensor land_v, torch::Tensor land_k,
               torch::Tensor flags, torch::Tensor clock_w,
               int64_t NB, int64_t MP, int64_t nkv, int64_t L, int64_t lidx,
               int64_t page, int64_t d, int64_t lcap, bool has_k) {
  auto stream = at::cuda::getCurrentCUDAStream();
  fa1_fetch_kernel<<<64, 256, 0, stream>>>(
      tab.data_ptr<int>(), cnt.data_ptr<int>(), bt.data_ptr<int>(),
      sl.data_ptr<int>(), pool_valid.data_ptr<int8_t>(),
      reinterpret_cast<const int16_t*>(poolv_ptr),
      reinterpret_cast<const int16_t*>(poolk_ptr),
      land_v.data_ptr<int16_t>(), land_k.data_ptr<int16_t>(),
      flags.data_ptr<int>(), clock_w.data_ptr<int>(),
      (long)NB, (long)MP, (int)nkv, (int)L, (int)lidx,
      (int)page, (int)d, (int)lcap, has_k ? 1 : 0);
}

void k5_headv(torch::Tensor elist, torch::Tensor ecnt,
              torch::Tensor plist, torch::Tensor pcnt,
              torch::Tensor p2s, torch::Tensor s2p,
              torch::Tensor owner, torch::Tensor gen,
              torch::Tensor clock_w, torch::Tensor fetch_w,
              int64_t NB, int64_t S, int64_t nkv, bool tick) {
  TORCH_CHECK(elist.is_cuda() && elist.scalar_type() == torch::kInt32);
  TORCH_CHECK(p2s.is_contiguous() && s2p.is_contiguous()
              && owner.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream();
  k5_headv_kernel<<<16, 256, 0, stream>>>(
      elist.data_ptr<int>(), ecnt.data_ptr<int>(),
      plist.data_ptr<int>(), pcnt.data_ptr<int>(),
      p2s.data_ptr<int>(), s2p.data_ptr<int>(),
      owner.data_ptr<int>(), gen.data_ptr<int>(),
      clock_w.data_ptr<int>(), fetch_w.data_ptr<int>(),
      (long)NB, (long)S, (int)nkv, (int)tick);
}

// INV (2026-07-19 late): one-launch CUDA port of residency.py::
// _invalidate_kernel.  The Triton walker spends 24.9us/step EXPOSED on the
// main stream (graphprof nsys_gt_tg) for O(1) real work at bs=1 decode (the
// single rewritten tail page): grid (L, n_req) single-warp programs, each
// re-deriving p0/p1 and looping pages.  Here: ONE CTA per request, one
// thread per (l, kv) row (L*NKV = 160 at the flagship), identical stores.
// Semantics VERBATIM (valid=0 always; p2s -1 iff slot != -1 i.e. resident
// OR pending; s2p -1 iff slot >= 0; gen +1 once per (req, page) by thread
// 0).  Pure int stores -- state effects identical by construction; the
// bitwise gate is the PTAB battery + suite.  Kill switch: LOCKS_INV_CUDA=0.
__global__ void inv_written_kernel(
    const int* __restrict__ bt, const int* __restrict__ sl,
    const int* __restrict__ qsl,
    int* __restrict__ p2s, int* __restrict__ s2p,
    signed char* __restrict__ valid, int* __restrict__ gen,
    const long NB, const long S, const long bt_stride,
    const int PAGE, const int L, const int NKV) {
  const int r = blockIdx.x;
  const int t = threadIdx.x;
  const int sl_r = sl[r];
  if (sl_r <= 0 || t >= L * NKV) return;
  const int ql = qsl[r + 1] - qsl[r];
  const int p0 = (sl_r - ql) / PAGE;
  const int p1 = (sl_r - 1) / PAGE;
  const long base = (long)t;                  // == l * NKV + kv
  for (int p = p0; p <= p1; ++p) {
    const int blk = bt[(long)r * bt_stride + p];
    if (t == 0) atomicAdd(&gen[blk], 1);
    valid[base * NB + blk] = 0;
    const int slot = p2s[base * NB + blk];
    if (slot != -1) p2s[base * NB + blk] = -1;  // resident OR pending
    if (slot >= 0)  s2p[base * S + slot] = -1;
  }
}

void inv_written(torch::Tensor bt, torch::Tensor sl, torch::Tensor qsl,
                 torch::Tensor p2s, torch::Tensor s2p, torch::Tensor valid,
                 torch::Tensor gen, int64_t NB, int64_t S, int64_t page,
                 int64_t L, int64_t NKV) {
  TORCH_CHECK(bt.is_cuda() && bt.scalar_type() == torch::kInt32);
  TORCH_CHECK(p2s.is_contiguous() && s2p.is_contiguous()
              && valid.is_contiguous());
  TORCH_CHECK(L * NKV <= 1024, "inv_written: L*NKV must fit one CTA");
  const int n_req = sl.size(0);
  const int thr = (int)(((L * NKV) + 31) / 32) * 32;
  auto stream = at::cuda::getCurrentCUDAStream();
  inv_written_kernel<<<n_req, thr, 0, stream>>>(
      bt.data_ptr<int>(), sl.data_ptr<int>(), qsl.data_ptr<int>(),
      p2s.data_ptr<int>(), s2p.data_ptr<int>(),
      (signed char*)valid.data_ptr(), gen.data_ptr<int>(),
      (long)NB, (long)S, (long)bt.stride(0), (int)page, (int)L, (int)NKV);
}

// NOROPE (round A): the fast arm's nt==1 KV write.  The forward's rope
// kernel is deleted at decode, so `key` arrives UNROPED; the stock
// reshape_and_cache would cache it unroped.  Same-launch-count CUDA
// replacement: rope dims < 64 with the inductor kernel's exact math
// (bf16 loads -> fp32, separate __fmul_rn/__fsub_rn/__fadd_rn, bf16
// round; partner = dd^1, cos/sin at csc[64*(sl-1) + dd/2 (+32)]) and
// scatter K+V to the paged cache at slot_mapping[0].  V is a raw byte
// copy; k dims >= 64 raw (== the deleted kernel's fp32 round-trip
// identity).  slot < 0 honors vLLM's per-element no-write contract.
// Graph-safe (fixed shapes; slot/sl read at replay).  d == 128 flagship.
__global__ void rope_and_cache_kernel(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ val,
    __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
    const long* __restrict__ slot, const int* __restrict__ sl,
    const __nv_bfloat16* __restrict__ csc,
    const long ksh, const long vsh,
    const long cb, const long ct, const long ch,
    const int page, const int n_kv, const int d) {
  const long s = slot[0];
  if (s < 0) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_kv * d) return;
  const int kv = i / d, dd = i % d;
  const long dst = (s / page) * cb + (s % page) * ct + (long)kv * ch + dd;
  vc[dst] = val[(long)kv * vsh + dd];
  if (dd < LOCKS_ROT_DIM) {
    const long pos = (long)sl[0] - 1;
    kc[dst] = __float2bfloat16(locks_rope_elem(
        key + (long)kv * ksh, csc + (long)LOCKS_ROT_DIM * pos, dd));
  } else {
    kc[dst] = key[(long)kv * ksh + dd];
  }
}

void rope_and_cache(torch::Tensor key, torch::Tensor val, torch::Tensor kc,
                    torch::Tensor vc, torch::Tensor slot, torch::Tensor sl,
                    torch::Tensor csc, int64_t page) {
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16 && key.size(0) == 1
              && key.size(2) == 128 && key.stride(2) == 1
              && val.stride(2) == 1, "rope_and_cache: key/val (1, n_kv, 128)");
  TORCH_CHECK(slot.scalar_type() == torch::kLong
              && sl.scalar_type() == torch::kInt
              && csc.scalar_type() == torch::kBFloat16
              && csc.stride(1) == 1 && csc.size(1) == LOCKS_ROT_DIM,
              "rope_and_cache: slot int64, sl int32, csc (P,ROT) bf16");
  TORCH_CHECK(kc.dim() == 4 && kc.stride(3) == 1 && vc.stride(3) == 1,
              "rope_and_cache: caches (NB, page, n_kv, d), inner-contig");
  const int n_kv = key.size(1), d = key.size(2);
  const int n = n_kv * d;
  auto stream = at::cuda::getCurrentCUDAStream();
  rope_and_cache_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(val.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(kc.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(vc.data_ptr()),
      slot.data_ptr<long>(), sl.data_ptr<int>(),
      reinterpret_cast<const __nv_bfloat16*>(csc.data_ptr()),
      key.stride(1), val.stride(1),
      kc.stride(0), kc.stride(1), kc.stride(2),
      (int)page, n_kv, d);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "rope_and_cache launch: ",
              cudaGetErrorString(e));
}

#ifdef LOCKS_FIXC
// FIX-C (session-P lever 1, 2026-07-22): rope_and_cache as the PSS
// dependent of kv_gemv_nf.  Same math, same bytes as rope_and_cache_kernel
// (byte-gated standalone); the ONLY additions are (a) the entry
// cudaGridDependencySynchronize -- it waits for the kv-GEMV grid (this
// launch's PSS primary, which fires per-CTA exit triggers) to COMPLETE and
// makes its out[] stores visible before key/val are read -- and (b) the
// PSS-attr launch below, so both grids run under the select_v2 cover
// window (select's outputs are not read here; the plain decode split
// behind this kernel closes every remaining hazard by stream order).
//
// UNIFIED nt >= 1 (2026-07-22, ours_doc/BATCHED_CO_DESIGN 2.5).  grid.y = nt,
// r = blockIdx.y, s = slot[r], key/val indexed by their own row strides.  The
// ONE genuine single-row dependency was `pos = sl[0] - 1`, a DERIVED rope
// position that is only valid for a decode row; it dissolves by READING
// `positions[r]`, which vLLM supplies per row and which is valid for prefill
// and decode alike, so there is no split-quality KV write.  On a decode row
// positions[r] == sl[r] - 1 IDENTICALLY, which is why nt == 1 stays bitwise
// (and is checked host-side under LOCKS_FIXC_POSCHK=1).
__global__ void rope_and_cache_fixc_kernel(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ val,
    __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
    const long* __restrict__ slot, const long* __restrict__ pos_in,
    const __nv_bfloat16* __restrict__ csc,
    const long krow, const long vrow, const long ksh, const long vsh,
    const long cb, const long ct, const long ch,
    const int page, const int n_kv, const int d) {
#if __CUDA_ARCH__ >= 900
  cudaGridDependencySynchronize();
#endif
  const int r = blockIdx.y;
  const long s = slot[r];
  if (s < 0) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_kv * d) return;
  const int kv = i / d, dd = i % d;
  const __nv_bfloat16* krow_p = key + (long)r * krow + (long)kv * ksh;
  const __nv_bfloat16* vrow_p = val + (long)r * vrow + (long)kv * vsh;
  const long dst = (s / page) * cb + (s % page) * ct + (long)kv * ch + dd;
  vc[dst] = vrow_p[dd];
  if (dd < LOCKS_ROT_DIM) {
    const long pos = pos_in[r];
    kc[dst] = __float2bfloat16(locks_rope_elem(
        krow_p, csc + (long)LOCKS_ROT_DIM * pos, dd));
  } else {
    kc[dst] = krow_p[dd];
  }
}

void rope_and_cache_fixc(torch::Tensor key, torch::Tensor val,
                         torch::Tensor kc, torch::Tensor vc,
                         torch::Tensor slot, torch::Tensor positions,
                         torch::Tensor csc, int64_t page) {
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16
              && key.size(2) == 128 && key.stride(2) == 1
              && val.stride(2) == 1 && val.size(0) == key.size(0),
              "rope_and_cache_fixc: key/val (nt, n_kv, 128)");
  TORCH_CHECK(slot.scalar_type() == torch::kLong
              && positions.scalar_type() == torch::kLong
              && positions.stride(0) == 1
              && csc.scalar_type() == torch::kBFloat16
              && csc.stride(1) == 1 && csc.size(1) == LOCKS_ROT_DIM,
              "rope_and_cache_fixc: slot/positions int64, csc (P,ROT) bf16");
  const int nt = (int)key.size(0);
  TORCH_CHECK(slot.size(0) >= nt && positions.size(0) >= nt,
              "rope_and_cache_fixc: slot/positions shorter than nt");
  TORCH_CHECK(kc.dim() == 4 && kc.stride(3) == 1 && vc.stride(3) == 1,
              "rope_and_cache_fixc: caches (NB, page, n_kv, d)");
  const int n_kv = key.size(1), d = key.size(2);
  const int n = n_kv * d;
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t lcfg = {};
  lcfg.gridDim = dim3((n + 255) / 256, nt);
  lcfg.blockDim = dim3(256);
  lcfg.dynamicSmemBytes = 0;
  lcfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  lcfg.attrs = attr;
  lcfg.numAttrs = 1;
  cudaLaunchKernelEx(&lcfg, rope_and_cache_fixc_kernel,
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(val.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(kc.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(vc.data_ptr()),
      slot.data_ptr<long>(), positions.data_ptr<long>(),
      reinterpret_cast<const __nv_bfloat16*>(csc.data_ptr()),
      key.stride(0), val.stride(0), key.stride(1), val.stride(1),
      kc.stride(0), kc.stride(1), kc.stride(2),
      (int)page, n_kv, d);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "rope_and_cache_fixc launch: ",
              cudaGetErrorString(e));
}
#endif  // LOCKS_FIXC

// E3 (mail diet, round B-compose): cnt-BOUNDED mailbox copy.  service()'s
// three full-buffer D2D copies (mail->snap, snap->plist, snap->elist) move
// MCAP*ENTW words of ~empty buffers every step; every consumer walks
// [0, cnt) only, so copying [0, min(cnt, cap)*entw) is read-equal by
// construction.  cnt is read ON DEVICE (sync-free; the min replicates the
// overflow clamp -- entries past MCAP were never written).
__global__ void mail_bcopy_kernel(const int* __restrict__ src,
                                  int* __restrict__ dst,
                                  const int* __restrict__ cnt_ptr,
                                  const int cap, const int entw) {
  const int c = *cnt_ptr;
  const int n = (c < cap ? c : cap) * entw;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    dst[i] = src[i];
}

void mail_bcopy(torch::Tensor src, torch::Tensor dst, torch::Tensor cnt,
                int64_t cap, int64_t entw) {
  TORCH_CHECK(src.scalar_type() == torch::kInt32 && src.is_contiguous()
              && dst.is_contiguous() && cnt.scalar_type() == torch::kInt32,
              "mail_bcopy: int32 contiguous src/dst, int32 cnt");
  auto stream = at::cuda::getCurrentCUDAStream();
  mail_bcopy_kernel<<<16, 256, 0, stream>>>(
      src.data_ptr<int>(), dst.data_ptr<int>(), cnt.data_ptr<int>(),
      (int)cap, (int)entw);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "mail_bcopy launch: ", cudaGetErrorString(e));
}

// NOROPE v3 (capture surgery): delete named kernel nodes from a CAPTURED
// cuda graph, rewiring each victim's dependencies onto its dependents.
// Triton kernels are DRIVER-launched (cuLaunchKernel), so the driver graph
// API is the one that can read their function names.  Used to prune the
// 39 per-layer rope kernels (triton_poi_fused_2) from the decode full
// graph after capture_end and before instantiation: the trace stays 100%
// STOCK (no custom op -> no functionalization clones, the v2 killer; no
// cache tag -> prefill artifact bit-identical), and the locks consumers
// rope in-kernel from the never-roped QKV buffer (P1/P2, bitwise-gated).
// The pruned nodes' outputs (the roped q/k buffers) become unwritten
// garbage that nothing reads under LOCKS_NOROPE.  Duplicate-edge adds
// return CUDA_ERROR_INVALID_VALUE and are tolerated per pair.
int64_t graph_prune_kernels(int64_t graph_handle, std::string prefix,
                            int64_t min_count) {
  CUgraph g = reinterpret_cast<CUgraph>(graph_handle);
  size_t n = 0;
  CUresult rr = cuGraphGetNodes(g, nullptr, &n);
  TORCH_CHECK(rr == CUDA_SUCCESS, "prune: cuGraphGetNodes count ", (int)rr);
  std::vector<CUgraphNode> nodes(n);
  rr = cuGraphGetNodes(g, nodes.data(), &n);
  TORCH_CHECK(rr == CUDA_SUCCESS, "prune: cuGraphGetNodes fill ", (int)rr);
  // pass 1: collect matches.  min_count gates the whole prune: the FULL
  // decode graph holds ~L rope nodes; PIECEWISE per-layer pieces hold one
  // each AND are replayed for nt>1 (prefill) batches -- pruning those
  // would corrupt prefill, so a graph below the threshold is left intact.
  std::vector<CUgraphNode> vict;
  for (size_t i = 0; i < n; ++i) {
    CUgraphNodeType ty;
    if (cuGraphNodeGetType(nodes[i], &ty) != CUDA_SUCCESS
        || ty != CU_GRAPH_NODE_TYPE_KERNEL)
      continue;
    CUDA_KERNEL_NODE_PARAMS p;
    if (cuGraphKernelNodeGetParams(nodes[i], &p) != CUDA_SUCCESS)
      continue;
    const char* name = nullptr;
    if (p.func) cuFuncGetName(&name, p.func);
#if CUDA_VERSION >= 12000
    if (!name && p.kern) cuKernelGetName(&name, p.kern);
#endif
    if (!name || std::strncmp(name, prefix.c_str(), prefix.size()) != 0)
      continue;
    vict.push_back(nodes[i]);
  }
  if ((int64_t)vict.size() < min_count) return 0;
  for (CUgraphNode v : vict) {
    size_t nd = 0, nu = 0;
    // CUDA 13 removed the 3-arg dependency-query/add driver APIs: the
    // edge-data variants became the only signatures (CUgraphEdgeData*).
    // Null edge data == default edges, so both branches are semantically
    // identical; measured break: nvcc 13.0 on the cluster NGC 25.08
    // container (2026-07-21), builds fine on the GH200 venv nvcc 12.8.
#if CUDA_VERSION >= 13000
    cuGraphNodeGetDependencies(v, nullptr, nullptr, &nd);
    cuGraphNodeGetDependentNodes(v, nullptr, nullptr, &nu);
    std::vector<CUgraphNode> deps(nd), users(nu);
    if (nd) cuGraphNodeGetDependencies(v, deps.data(), nullptr, &nd);
    if (nu) cuGraphNodeGetDependentNodes(v, users.data(), nullptr, &nu);
    for (size_t u = 0; u < nu; ++u)
      for (size_t d = 0; d < nd; ++d) {
        CUresult ar = cuGraphAddDependencies(g, &deps[d], &users[u],
                                             nullptr, 1);
        TORCH_CHECK(ar == CUDA_SUCCESS || ar == CUDA_ERROR_INVALID_VALUE,
                    "prune: cuGraphAddDependencies ", (int)ar);
      }
#else
    cuGraphNodeGetDependencies(v, nullptr, &nd);
    cuGraphNodeGetDependentNodes(v, nullptr, &nu);
    std::vector<CUgraphNode> deps(nd), users(nu);
    if (nd) cuGraphNodeGetDependencies(v, deps.data(), &nd);
    if (nu) cuGraphNodeGetDependentNodes(v, users.data(), &nu);
    for (size_t u = 0; u < nu; ++u)
      for (size_t d = 0; d < nd; ++d) {
        CUresult ar = cuGraphAddDependencies(g, &deps[d], &users[u], 1);
        TORCH_CHECK(ar == CUDA_SUCCESS || ar == CUDA_ERROR_INVALID_VALUE,
                    "prune: cuGraphAddDependencies ", (int)ar);
      }
#endif
    rr = cuGraphDestroyNode(v);
    TORCH_CHECK(rr == CUDA_SUCCESS, "prune: cuGraphDestroyNode ", (int)rr);
  }
  return (int64_t)vict.size();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k5_headv", &k5_headv, "K10 fused graph-head evict+promote(+tick)");
  m.def("graph_prune_kernels", &graph_prune_kernels,
        "NOROPE v3: delete named kernel nodes from a captured graph");
  m.def("s3_build", &s3_build, "S3 resolved (slot,vb,blk) step table");
  m.def("mail_snap_bucketize", &mail_snap_bucketize,
        "Z1 snapshot+clamp+per-unit bucketize");
  m.def("fa1_fetch", &fa1_fetch, "FA1 fetch-all-selected into landing");
  m.def("fa1_fetch_v3", &fa1_fetch_v3,
        "FA1-v3 tagged fetch (skip landed+unchanged; per-layer landing)");
  m.def("fa1_step_v32", &fa1_step_v32,
        "FA1-v3.2 stable-slot assign + churn-only copy");
  m.def("inv_written", &inv_written, "INV one-launch invalidate_written");
  m.def("rope_and_cache", &rope_and_cache,
        "NOROPE fast-arm nt==1 KV write (rope k + scatter)");
#ifdef LOCKS_FIXC
  m.def("rope_and_cache_fixc", &rope_and_cache_fixc,
        "FIX-C PSS rope+cache (gdc fence on kv_gemv_nf; select-cover)");
#endif
  m.def("mail_bcopy", &mail_bcopy, "E3 cnt-bounded mailbox copy");
}
"""

_MOD = None
_TRIED = False


def get_mod():
    """Build once; None if the extension cannot build.

    NOTE (no-fallback rule, 2026-07-22): returning None is a REPORT, not a
    licence -- every fast-path consumer (attn._norope_fast_write,
    attn._fixc_defer_write, _runtime.fixc_flush, register's NOROPE prune)
    asserts on it.  Do not add an `if mod is None: <other implementation>`
    branch; the build log (LOCKS_CUDA_VERBOSE=1) is the fix path."""
    global _MOD, _TRIED
    if _TRIED:
        return _MOD
    _TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        _cf = ["-O3", _arch.arch_flag()]
        # FIX-C opt-in (LOCKS_QFIRST_FIXC=1): compile the PSS rope+cache
        # variant.  Flag unset -> define off, TU byte-identical
        # (SASS-identity gated).
        if os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1":
            _cf.append("-DLOCKS_FIXC")
        # ROPE GEOMETRY (model-agnostic, 2026-07-22): rope_and_cache[_fixc]
        # rope K in-kernel, so the loaded arch's (rotary_dim, neox) pair is
        # compiled in.  GLM geometry -> no flags, no name suffix (byte-
        # identical deployed TU); any other arch gets its own name so a warm
        # extensions cache cannot serve it GLM's binary.
        from ..backend import _runtime as _rt
        _cf += _rt.rope_cflags()
        _MOD = load_inline(
            name="locks_k5_headv" + _rt.rope_tag(),
            cpp_sources="", cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=_cf,
            extra_ldflags=["-lcuda"],
            verbose=bool(int(os.environ.get("LOCKS_CUDA_VERBOSE", "0"))))
        print("[locks_k5_cuda] K10 fused head (CUDA) ACTIVE", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[locks_k5_cuda] build failed ({type(e).__name__}: {e})",
              flush=True)
        _MOD = None
    return _MOD
