"""P1 (2026-07-19): hand-CUDA fused staging prologue.

The true-exposure profile (graphprof) put the OUT-OF-GRAPH tier prologue at
+123..134us/step vs dense (14 extra plain launches; the Triton chain
vsm-fill + _vflush_clear + fcnt-zero + _flush_collect + _flush_copy +
_valloc + _vslot costs ~37us of kernel time plus ~2-5us Grace launch gap per
plain launch, and _flush_copy's (MAXF, L) idle wave burns ~15us even with an
EMPTY flush list).  This module replaces the chain with TWO kernels:

  staging_lifecycle  grid (1,), 256 threads -- the whole per-step lifecycle
    in I4 program order: vsm-fill; vflush_clear; [flush_cnt=0 + collect]
    gated on PAGE COMPLETION (np_ctx changed since the last scan, tracked in
    a device np_prev[r] -- the O(ctx/page) settled-prefix scan becomes
    O(1)-amortized at decode: 1 scan per 16 steps; chunked prefill (ql>1)
    always scans); valloc (thread-0 serial pop, duplicate-claim CAS kept);
    vslot.
  staging_flush_copy  grid (4, L), 256 threads -- entry-STRIDED copy (each
    block strides the flush list, ~2us when empty vs the fixed idle wave);
    bit-exact int16 copies v_pool[l,vb](+k_pool) -> pinned pool(s) + sets
    pool_valid, exactly like _flush_copy_kernel.

SEMANTICS: identical state evolution to the Triton chain except flush
COLLECTION timing -- a page becomes collectable at the first lifecycle call
after it completes (same step it completes at bs=1: np_ctx changes exactly
then), and MAXF-overflow retries wait for the next scan instead of the next
step (pages stay STAGED-readable in place meanwhile; bytes identical, I1-I4
preserved: the free still happens only after the flush claim, the copy still
reads before valloc's reuse materializes -- the in-graph scatter that would
clobber a reused vb runs stream-ordered after staging_flush_copy).
Kill switch: LOCKS_STAGING_CUDA=0.
"""
from __future__ import annotations

import os

import torch

from .. import arch as _arch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// ------------------------- lifecycle (grid (1,)) ------------------------- //
__global__ void staging_lifecycle_kernel(
    const int* __restrict__ bt, const int* __restrict__ sl,
    const int* __restrict__ qsl,
    long long* __restrict__ vsm,          // v_slot_mapping (int64)
    int* __restrict__ vfl,                // v_flushed
    int* __restrict__ vbo,
    int* __restrict__ flist, int* __restrict__ fcnt,
    int* __restrict__ stack, int* __restrict__ top,
    int* __restrict__ ovf,
    int* __restrict__ np_prev,            // (R,) last-scanned np_ctx
    const long bt_sr, const int n_req, const int n_tokens,
    const int PAGE, const int MAXF) {
  const int tid = threadIdx.x;
  const int TB = blockDim.x;
  // 1. reset this step's v_slot_mapping
  for (int t = tid; t < n_tokens; t += TB) vsm[t] = -1;
  __syncthreads();
  if (n_req == 0) return;                 // capture dummy batch: no mutation
  // 2. vflush_clear: rewritten pages lose the stale flushed bit
  for (int r = 0; r < n_req; ++r) {
    const int s = sl[r], q = qsl[r + 1] - qsl[r];
    if (s <= 0 || q <= 0) continue;
    const int p0 = (s - q) / PAGE, p1 = (s - 1) / PAGE;
    for (int p = p0 + tid; p <= p1; p += TB) vfl[bt[r * bt_sr + p]] = 0;
  }
  __syncthreads();
  // 3. collect (claim + free), gated on page completion / prefill chunks
  bool need = false;
  for (int r = 0; r < n_req; ++r) {
    const int s = sl[r], q = qsl[r + 1] - qsl[r];
    if (s <= 0 || q <= 0) continue;
    if (q > 1 || ((s - q) / PAGE) != np_prev[r]) { need = true; break; }
  }
  if (!need) {
    // LOAD-BEARING: the flush-copy kernel runs every step reading *fcnt; a
    // stale count would RE-COPY freed (and possibly re-scattered) staging
    // slots over the pinned pool (wrong bytes -- caught by matched_mns).
    if (tid == 0) *fcnt = 0;
  }
  if (need) {
    if (tid == 0) *fcnt = 0;
    __syncthreads();
    for (int r = 0; r < n_req; ++r) {     // SERIALIZED over requests (races)
      const int s = sl[r], q = qsl[r + 1] - qsl[r];
      if (s <= 0 || q <= 0) continue;
      const int np_ctx = (s - q) / PAGE;
      if (tid == 0) np_prev[r] = np_ctx;
      for (int p = tid; p < np_ctx; p += TB) {
        const int blk = bt[r * bt_sr + p];
        const int vb = vbo[blk];
        if (vb < 0) continue;
        if (atomicExch(&vfl[blk], 1) != 0) continue;   // claim (latch)
        const int i = atomicAdd(fcnt, 1);
        if (i < MAXF) {
          flist[i * 2 + 0] = blk;
          flist[i * 2 + 1] = vb;
          // free ONLY after the flush claim (I1); copy reads via flist
          const int old = atomicExch(&vbo[blk], -1);
          if (old >= 0) stack[atomicAdd(top, 1)] = old;
        } else {
          atomicSub(fcnt, 1);
          vfl[blk] = 0;                    // over: retry at the next scan
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
  // 5. valloc: thread-0 serial pop (matches the Triton grid-(1,) semantics)
  if (tid == 0) {
    for (int r = 0; r < n_req; ++r) {
      const int s = sl[r], q = qsl[r + 1] - qsl[r];
      if (s <= 0 || q <= 0) continue;
      const int p0 = (s - q) / PAGE, p1 = (s - 1) / PAGE;
      for (int p = p0; p <= p1; ++p) {
        const int blk = bt[r * bt_sr + p];
        if (vbo[blk] >= 0) continue;
        const int idx = atomicAdd(top, -1) - 1;
        if (idx < 0) { atomicAdd(top, 1); *ovf = 1; }
        else {
          const int nv = stack[idx];
          const int old = atomicCAS(&vbo[blk], -1, nv);
          if (old != -1) stack[atomicAdd(top, 1)] = nv;   // dup claim: return
        }
      }
    }
  }
  __syncthreads();
  // 6. vslot: fill this step's token -> staging-slot mapping
  for (int r = 0; r < n_req; ++r) {
    const int s = sl[r], q0 = qsl[r], q = qsl[r + 1] - q0;
    if (s <= 0 || q <= 0) continue;
    for (int i = tid; i < q; i += TB) {
      const int pos = s - q + i;
      const int blk = bt[r * bt_sr + pos / PAGE];
      const int vb = vbo[blk];
      vsm[q0 + i] = (vb >= 0) ? (long long)vb * PAGE + pos % PAGE : -1;
    }
  }
}

// --------------------- flush copy (grid (CB, L)) ------------------------- //
__global__ void staging_flush_copy_kernel(
    const __nv_bfloat16* __restrict__ vp,   // (L, NV, page, n_kv, d)
    int16_t* __restrict__ pool,             // pinned UVA (L, NB, kv, page, d)
    int8_t* __restrict__ valid,             // (L*NKV, NB)
    const __nv_bfloat16* __restrict__ kp,
    int16_t* __restrict__ poolk,
    const int* __restrict__ flist, const int* __restrict__ fcnt,
    const long NB, const long svl, const long svb, const long svt,
    const long svh,
    const int NKV, const int PAGE, const int D, const int MAXF,
    const int has_k) {
  const int l = blockIdx.y;
  const int cnt = min(*fcnt, MAXF);
  const int tid = threadIdx.x, TB = blockDim.x;
  for (int i = blockIdx.x; i < cnt; i += gridDim.x) {
    const long blk = flist[i * 2 + 0];
    const long vb = flist[i * 2 + 1];
    for (int kv = 0; kv < NKV; ++kv) {
      const long src0 = (long)l * svl + vb * svb + (long)kv * svh;
      const long dst0 = (((long)l * NB + blk) * NKV + kv) * (long)(PAGE * D);
      for (int e = tid; e < PAGE * D; e += TB) {
        const int t = e / D, dd = e % D;
        const long s = src0 + (long)t * svt + dd;
        pool[dst0 + e] = *reinterpret_cast<const int16_t*>(&vp[s]);
        if (has_k)
          poolk[dst0 + e] = *reinterpret_cast<const int16_t*>(&kp[s]);
      }
      if (tid == 0) valid[((long)l * NKV + kv) * NB + blk] = 1;
    }
  }
}

void staging_lifecycle(torch::Tensor bt, torch::Tensor sl, torch::Tensor qsl,
                       torch::Tensor vsm, torch::Tensor vfl,
                       torch::Tensor vbo, torch::Tensor flist,
                       torch::Tensor fcnt, torch::Tensor stack,
                       torch::Tensor top, torch::Tensor ovf,
                       torch::Tensor np_prev,
                       int64_t n_req, int64_t n_tokens, int64_t page,
                       int64_t maxf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  staging_lifecycle_kernel<<<1, 256, 0, stream>>>(
      bt.data_ptr<int>(), sl.data_ptr<int>(), qsl.data_ptr<int>(),
      reinterpret_cast<long long*>(vsm.data_ptr<int64_t>()),
      vfl.data_ptr<int>(), vbo.data_ptr<int>(),
      flist.data_ptr<int>(), fcnt.data_ptr<int>(), stack.data_ptr<int>(),
      top.data_ptr<int>(), ovf.data_ptr<int>(), np_prev.data_ptr<int>(),
      (long)bt.stride(0), (int)n_req, (int)n_tokens, (int)page, (int)maxf);
}

void staging_flush_copy(torch::Tensor vp, int64_t pool_ptr,
                        torch::Tensor valid, torch::Tensor kp,
                        int64_t poolk_ptr, torch::Tensor flist,
                        torch::Tensor fcnt,
                        int64_t L, int64_t NB, int64_t nkv, int64_t page,
                        int64_t d, int64_t maxf, bool has_k) {
  // pool/poolk arrive as raw UVA device addresses (MappedHostVPool.dev_view
  // is a duck-typed pointer wrapper, not a torch.Tensor -- pool.py:82).
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(4, (unsigned)L);
  staging_flush_copy_kernel<<<grid, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(vp.data_ptr()),
      reinterpret_cast<int16_t*>(pool_ptr), valid.data_ptr<int8_t>(),
      reinterpret_cast<const __nv_bfloat16*>(kp.data_ptr()),
      reinterpret_cast<int16_t*>(poolk_ptr),
      flist.data_ptr<int>(), fcnt.data_ptr<int>(),
      (long)NB, (long)vp.stride(0), (long)vp.stride(1), (long)vp.stride(2),
      (long)vp.stride(3),
      (int)nkv, (int)page, (int)d, (int)maxf, has_k ? 1 : 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("staging_lifecycle", &staging_lifecycle, "P1 fused staging prologue");
  m.def("staging_flush_copy", &staging_flush_copy, "P1 strided flush copy");
}
"""

_MOD = None
_TRIED = False


def get_mod():
    global _MOD, _TRIED
    if _TRIED:
        return _MOD
    _TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        _MOD = load_inline(
            name="locks_staging_prologue",
            cpp_sources="", cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=["-O3", _arch.arch_flag()],
            verbose=bool(int(os.environ.get("LOCKS_CUDA_VERBOSE", "0"))))
        print("[locks_staging_cuda] P1 fused prologue (CUDA) ACTIVE",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[locks_staging_cuda] build failed ({type(e).__name__}: {e})",
              flush=True)
        _MOD = None
    return _MOD
