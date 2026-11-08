from __future__ import annotations
import os
from .. import arch as _arch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ bool consecutive_decode(
    const int* bt, const int* gen, const int* guard,
    const long stride, const int r, const int s, const int q,
    const int page, const int previous_pages, const bool enabled) {
  if (!enabled || q != 1 || s <= 1 || guard[r * 6 + 1] != 1 ||
      guard[r * 6] != s - q || previous_pages < 0 ||
      previous_pages > (s - q) / page) return false;
  const int first = bt[r * stride];
  const int previous_last = bt[r * stride + (s - q - 1) / page];
  const int written_previous_last = ((s - q) % page) != 0;
  return first == guard[r * 6 + 2] && gen[first] == guard[r * 6 + 3] &&
         previous_last == guard[r * 6 + 4] &&
         gen[previous_last] == guard[r * 6 + 5] + written_previous_last;
}


__global__ void staging_lifecycle_kernel(
    const int* __restrict__ bt, const int* __restrict__ sl,
    const int* __restrict__ qsl,
    long long* __restrict__ vsm,
    int* __restrict__ vfl,
    int* __restrict__ vbo,
    int* __restrict__ flist, int* __restrict__ fcnt,
    int* __restrict__ stack, int* __restrict__ top,
    int* __restrict__ ovf,
    int* __restrict__ np_prev,
    int* __restrict__ guard, const int* __restrict__ gen,
    const bool frontier,
    const long bt_sr, const int n_req, const int n_tokens,
    const int PAGE, const int MAXF) {
  const int tid = threadIdx.x;
  const int TB = blockDim.x;
  __shared__ int all_flushed;
  if (tid == 0) all_flushed = 1;

  for (int t = tid; t < n_tokens; t += TB) vsm[t] = -1;
  __syncthreads();
  if (n_req == 0) return;

  for (int r = 0; r < n_req; ++r) {
    const int s = sl[r], q = qsl[r + 1] - qsl[r];
    if (s <= 0 || q <= 0) continue;
    const int p0 = (s - q) / PAGE, p1 = (s - 1) / PAGE;
    for (int p = p0 + tid; p <= p1; p += TB) vfl[bt[r * bt_sr + p]] = 0;
  }
  __syncthreads();

  bool need = false;
  for (int r = 0; r < n_req; ++r) {
    const int s = sl[r], q = qsl[r + 1] - qsl[r];
    if (s <= 0 || q <= 0) continue;
    if (q > 1 || ((s - q) / PAGE) != np_prev[r] ||
        (frontier && !consecutive_decode(bt, gen, guard, bt_sr, r, s, q,
                                        PAGE, np_prev[r], frontier))) {
      need = true; break;
    }
  }
  if (!need) {


    if (tid == 0) *fcnt = 0;
  }
  if (need) {
    if (tid == 0) *fcnt = 0;
    __syncthreads();
    for (int r = 0; r < n_req; ++r) {
      const int s = sl[r], q = qsl[r + 1] - qsl[r];
      if (s <= 0 || q <= 0) continue;
      const int np_ctx = (s - q) / PAGE;
      const int first_page = consecutive_decode(bt, gen, guard, bt_sr, r, s, q,
                                                PAGE, np_prev[r], frontier)
                             ? np_prev[r] : 0;
      for (int p = first_page + tid; p < np_ctx; p += TB) {
        const int blk = bt[r * bt_sr + p];
        const int vb = vbo[blk];
        if (vb < 0) continue;
        if (atomicExch(&vfl[blk], 1) != 0) continue;
        const int i = atomicAdd(fcnt, 1);
        if (i < MAXF) {
          flist[i * 2 + 0] = blk;
          flist[i * 2 + 1] = vb;

          const int old = atomicExch(&vbo[blk], -1);
          if (old >= 0) stack[atomicAdd(top, 1)] = old;
        } else {
          atomicSub(fcnt, 1);
          vfl[blk] = 0;
          atomicExch(&all_flushed, 0);
        }
      }
      __syncthreads();
      if (tid == 0) np_prev[r] = np_ctx;
      __syncthreads();
    }
  }
  __syncthreads();

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
          if (old != -1) stack[atomicAdd(top, 1)] = nv;
        }
      }
    }
  }
  __syncthreads();

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
  __syncthreads();
  if (frontier && tid == 0) {
    for (int r = 0; r < n_req; ++r) {
      const int s = sl[r], q = qsl[r + 1] - qsl[r];
      if (s <= 0 || q <= 0 || !all_flushed) {
        guard[r * 6 + 1] = -1;
        continue;
      }
      const int first = bt[r * bt_sr];
      const int last = bt[r * bt_sr + (s - 1) / PAGE];
      guard[r * 6] = s;
      guard[r * 6 + 1] = q;
      guard[r * 6 + 2] = first;
      guard[r * 6 + 3] = gen[first];
      guard[r * 6 + 4] = last;
      guard[r * 6 + 5] = gen[last];
    }
  }
}


__global__ void staging_flush_copy_kernel(
    const __nv_bfloat16* __restrict__ vp,
    int16_t* __restrict__ pool,
    int8_t* __restrict__ valid,
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
                       torch::Tensor guard, torch::Tensor gen, bool frontier,
                       int64_t n_req, int64_t n_tokens, int64_t page,
                       int64_t maxf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  staging_lifecycle_kernel<<<1, 256, 0, stream>>>(
      bt.data_ptr<int>(), sl.data_ptr<int>(), qsl.data_ptr<int>(),
      reinterpret_cast<long long*>(vsm.data_ptr<int64_t>()),
      vfl.data_ptr<int>(), vbo.data_ptr<int>(),
      flist.data_ptr<int>(), fcnt.data_ptr<int>(), stack.data_ptr<int>(),
      top.data_ptr<int>(), ovf.data_ptr<int>(), np_prev.data_ptr<int>(),
      guard.data_ptr<int>(), gen.data_ptr<int>(), frontier,
      (long)bt.stride(0), (int)n_req, (int)n_tokens, (int)page, (int)maxf);
}

void staging_flush_copy(torch::Tensor vp, int64_t pool_ptr,
                        torch::Tensor valid, torch::Tensor kp,
                        int64_t poolk_ptr, torch::Tensor flist,
                        torch::Tensor fcnt,
                        int64_t L, int64_t NB, int64_t nkv, int64_t page,
                        int64_t d, int64_t maxf, bool has_k) {


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
  m.def("staging_lifecycle", &staging_lifecycle);
  m.def("staging_flush_copy", &staging_flush_copy);
}
"""
_MOD = None
_TRIED = False


def get_mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load_inline

    _MOD = load_inline(
        name="locks_staging_prologue",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=["-O3", _arch.arch_flag()],
        verbose=bool(int(os.environ.get("LOCKS_CUDA_VERBOSE", "0"))),
    )
    print("[locks_staging_cuda] P1 fused prologue (CUDA) ACTIVE", flush=True)
    return _MOD
