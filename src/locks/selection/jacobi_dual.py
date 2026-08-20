"""Fused batched 16x16 symmetric eigensolver -- DUAL-gram parallel Jacobi.

Drop-in for ``torch.linalg.eigh`` on the rki4 page-grams (16x16), 2.80x over
cusolver at the 16K point (4096 grams) and 2.2-2.9x across 4K-262K grams
(H200). Two grams per warp (lanes 0-15 gram A, 16-31 gram B) so every lane is
busy in the column/row phases -- the shipped ``build._eigh_jacobi`` wastes
16/32 lanes and is only 1.01x at 16K. No per-matrix cusolver launch, no batch
ceiling.

Gauge note: only the TOP-8 eigenpairs feed the rki4 summary, and
``C V^T = U8 U8^T dc`` (the projection onto the top-8 eigenspace) is invariant
to eigenvector sign/rotation inside that subspace, so a different-but-equivalent
factorization reconstructs the SAME per-token page-LSE => same selection => same
K3. This is why the swap is gated on RECONSTRUCTION-EQUIVALENCE (page-LSE within
the int4/int8 quant floor; recall/mass identical to <=1e-4 on the e28 oracle,
Llama-3.1-8B real 16K keys) and NOT on bitwise int4 codes (which flip ~49% on
the harmless sign/rotation ambiguity). ``ours_doc/TTFT_16K_KERNEL.md``.

8 Jacobi sweeps suffice (recon_rel 8.4e-6, top-8-projector maxabs 4.1e-5 vs
cusolver). NOT graph-safe (runs on page-finalize off the hot path), like the
cusolver path it replaces.
"""
from __future__ import annotations

import os

import torch

from .. import arch as _arch

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef NW
#define NW 4
#endif
#ifndef SWEEPS
#define SWEEPS 8
#endif
#define JN 16
#define JPAD 17

__device__ __forceinline__ void rr_pair(int r, int i, int* p, int* q) {
    int a, b;
    if (i == 0) { a = 15; b = r; }
    else { a = (r + i) % 15; b = (r - i + 15) % 15; }
    *p = a < b ? a : b;
    *q = a < b ? b : a;
}

__global__ void jacobi_eigh16_dual(const float* __restrict__ Gm,
                                   float* __restrict__ W,
                                   float* __restrict__ Vout,
                                   const int M) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;
    const int s    = lane & 15;
    const int g = ((blockIdx.x * NW) + warp) * 2 + half;
    const bool act = g < M;
    __shared__ float As[NW][2][JN][JPAD];
    __shared__ float Vs[NW][2][JN][JPAD];
    __shared__ float Cs[NW][2][8], Ss[NW][2][8];
    __shared__ int   Pp[NW][2][8], Qq[NW][2][8];
    float (*A)[JPAD] = As[warp][half];
    float (*V)[JPAD] = Vs[warp][half];
    if (act) {
        const float* Gg = Gm + (long)g * JN * JN;
        #pragma unroll
        for (int c = 0; c < JN; ++c) {
            A[s][c] = Gg[s * JN + c];
            V[s][c] = (s == c) ? 1.f : 0.f;
        }
    }
    __syncwarp();
    for (int sweep = 0; sweep < SWEEPS; ++sweep) {
        for (int rnd = 0; rnd < JN - 1; ++rnd) {
            if (act && s < 8) {
                int p, q; rr_pair(rnd, s, &p, &q);
                Pp[warp][half][s] = p; Qq[warp][half][s] = q;
                float apq = A[p][q];
                float cc = 1.f, ss = 0.f;
                if (fabsf(apq) > 1e-20f) {
                    float th = (A[q][q] - A[p][p]) / (2.f * apq);
                    float t = (th >= 0.f ? 1.f : -1.f)
                              / (fabsf(th) + sqrtf(th * th + 1.f));
                    cc = rsqrtf(t * t + 1.f); ss = t * cc;
                }
                Cs[warp][half][s] = cc; Ss[warp][half][s] = ss;
            }
            __syncwarp();
            if (act) {
                int k = s;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int p = Pp[warp][half][i], q = Qq[warp][half][i];
                    float cc = Cs[warp][half][i], ss = Ss[warp][half][i];
                    float akp = A[k][p], akq = A[k][q];
                    A[k][p] = cc * akp - ss * akq;
                    A[k][q] = ss * akp + cc * akq;
                    float vkp = V[k][p], vkq = V[k][q];
                    V[k][p] = cc * vkp - ss * vkq;
                    V[k][q] = ss * vkp + cc * vkq;
                }
            }
            __syncwarp();
            if (act) {
                int k = s;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int p = Pp[warp][half][i], q = Qq[warp][half][i];
                    float cc = Cs[warp][half][i], ss = Ss[warp][half][i];
                    float apk = A[p][k], aqk = A[q][k];
                    A[p][k] = cc * apk - ss * aqk;
                    A[q][k] = ss * apk + cc * aqk;
                }
            }
            __syncwarp();
        }
    }
    if (act) {
        W[(long)g * JN + s] = A[s][s];
        #pragma unroll
        for (int c = 0; c < JN; ++c)
            Vout[(long)g * JN * JN + s * JN + c] = V[s][c];
    }
}

void launch_dual(torch::Tensor Gm, torch::Tensor W, torch::Tensor V, int64_t M) {
    const int block = NW * 32;
    const int grid = (int)((M + 2 * NW - 1) / (2 * NW));
    auto stream = at::cuda::getCurrentCUDAStream();
    jacobi_eigh16_dual<<<grid, block, 0, stream>>>(
        Gm.data_ptr<float>(), W.data_ptr<float>(), V.data_ptr<float>(), (int)M);
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "jacobi_dual: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_dual", &launch_dual, "dual-gram 16x16 Jacobi eigh");
}
"""

_MOD = None
_TRIED = False
_SWEEPS = int(os.environ.get("LOCKS_RKI4_EIGH_SWEEPS", "8"))


def _mod():
    global _MOD, _TRIED
    if _TRIED:
        return _MOD
    _TRIED = True
    try:
        from torch.utils.cpp_extension import load_inline
        _MOD = load_inline(
            name=f"locks_jacobi_dual_sw{_SWEEPS}", cpp_sources="",
            cuda_sources=_SRC,
            extra_cuda_cflags=["-O3", f"-DSWEEPS={_SWEEPS}", _arch.arch_flag()],
            verbose=False)
    except Exception:
        _MOD = None
    return _MOD


def eigh_dual(Gm: torch.Tensor):
    """(M, 16, 16) fp32 symmetric grams -> (S2 ascending, U columns), the
    ``torch.linalg.eigh`` contract. Falls back to cusolver if the extension is
    unavailable or the shape is not 16x16."""
    mod = _mod()
    if mod is None or Gm.shape[-1] != 16 or Gm.shape[-2] != 16:
        return torch.linalg.eigh(Gm)
    *lead, n, _ = Gm.shape
    A = Gm.reshape(-1, 16, 16).contiguous().to(torch.float32)
    M = A.shape[0]
    W = torch.empty(M, 16, device=A.device, dtype=torch.float32)
    V = torch.empty(M, 16, 16, device=A.device, dtype=torch.float32)
    mod.launch_dual(A, W, V, M)
    order = W.argsort(dim=-1)
    S2 = torch.gather(W, -1, order)
    U = torch.gather(V, -1, order.unsqueeze(-2).expand(M, 16, 16))
    return (S2.reshape(*lead, 16).to(Gm.dtype),
            U.reshape(*lead, 16, 16).to(Gm.dtype))
