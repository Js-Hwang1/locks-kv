from __future__ import annotations
import os
from .. import arch as _arch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <vector>
#include <ATen/cuda/CUDAContext.h>


#ifndef LOCKS_ROT_DIM
#define LOCKS_ROT_DIM 64
#endif
#ifndef LOCKS_ROT_NEOX
#define LOCKS_ROT_NEOX 0
#endif
#define LOCKS_ROT_HALF (LOCKS_ROT_DIM / 2)


#define ENTW 6

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

  const int en = *ecnt;
  for (int i = tid; i < en; i += stride) {
    const int h = elist[i * ENTW + 3];
    if (h < 0) continue;
    const int l   = elist[i * ENTW + 0];
    const int kv  = elist[i * ENTW + 1];
    const int blk = elist[i * ENTW + 2];
    const long base = (long)(l * NKV + kv);
    const int old = s2p[base * S + h];
    if (old >= 0 && old != blk)
      atomicCAS(&p2s[base * NB + old], h, -1);
    s2p[base * S + h] = blk;
  }

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
    if (gen[blk] == g) {
      const int old = atomicCAS(&p2s[base * NB + blk], -2, h);
      if (old == -2 || old == h) installed = 1;
    }
    if (!installed) s2p[base * S + h] = -1;
    owner[base * S + h] = 0;
  }
}


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
    if (sl == -1) {
      const int u = l * NKV + kv;
      const int p = atomicAdd(&bcnt[u], 1);
      if (p < CAP) bidx[(long)u * CAP + p] = i;
      else {
        atomicCAS(&p2s[(long)u * NB + blk], -2, -1);
        snap[i * 6 + 3] = -1;
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
  const long base = (long)t;
  for (int p = p0; p <= p1; ++p) {
    const int blk = bt[(long)r * bt_stride + p];
    if (t == 0) atomicAdd(&gen[blk], 1);
    valid[base * NB + blk] = 0;
    const int slot = p2s[base * NB + blk];
    if (slot != -1) p2s[base * NB + blk] = -1;
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


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("k5_headv", &k5_headv);

  m.def("mail_snap_bucketize", &mail_snap_bucketize);

  m.def("inv_written", &inv_written);

  m.def("mail_bcopy", &mail_bcopy);
}
"""
_MOD = None
_TRIED = False


def get_mod():
    global _MOD
    if _MOD is not None:
        return _MOD
    from torch.utils.cpp_extension import load_inline

    _cf = ["-O3", _arch.arch_flag()]
    _MOD = load_inline(
        name="locks_k5_headv",
        cpp_sources="",
        cuda_sources=_CUDA_SRC,
        extra_cuda_cflags=_cf,
        extra_ldflags=["-lcuda"],
        verbose=bool(int(os.environ.get("LOCKS_CUDA_VERBOSE", "0"))),
    )
    print("[locks_k5_cuda] K10 fused head (CUDA) ACTIVE", flush=True)
    return _MOD
