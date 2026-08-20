"""Deterministic sliced-QKV GEMV (the sanctioned Q-first lane, P1 wire).

Two hand kernels replace the fused cuBLAS QKV projection: a row-per-CTA
bf16 GEMV with fused bias (BLK=128, 16B vectorized loads, fp32 accumulate,
fixed two-stage reduction => BITWISE-DETERMINISTIC across runs -- P0 gates:
q-slice 11.60us, kv-slice 4.79us, relerr at the bf16 class bound,
determinism 200/200).  Both write into ONE (nt, 5120) output buffer (q rows
then kv rows), so the caller's fused-layout split is unchanged and no concat
copy exists.

UNIFIED nt >= 1 (2026-07-22, ours_doc/BATCHED_CO_DESIGN 2.4).  One CTA per
OUTPUT ROW at every nt (grid.x = rows, unchanged); the token axis is a
compile-time accumulator tile ``TT`` on grid.y, with a CLAMPED x row index so
the hot loop carries no guard and the weight row is read ONCE for all TT
tokens (the read-once property is why the hidden mass is batch-invariant).

  * BITWISE BY CONSTRUCTION at every nt, including nt == 1.  The reduction
    tree for output (m, t) is the same ``fmaf`` sequence over the same ``i``
    order followed by the same two-stage ``__shfl_down_sync`` pair; it does
    not depend on TT or on nt.  nt == 1 is the n = 1 instance, not a case.
  * TT is a LAUNCH-SHAPE knob, the class of ``auto_zsplit``'s ``mult``: one
    kernel body, one result, bitwise-invariant to the knob.  ``tt_for(nt)``
    is monotone in nt.  It is NOT a quality split (there is no second
    implementation for it to race).

Exposed as the OPAQUE op ``locks_gv.qkv`` (pure, out-of-place):
  * nt == 1  -> the two GEMV kernels (deterministic sliced path).
  * nt  > 1  -> F.linear on the fused weight (STOCK bytes: prefill is
    byte-identical to the deployed arm; the branch lives INSIDE the op
    body, which executes per call -- a python shape branch in the traced
    forward would be baked to one side, the NOROPE-v1 postmortem).
No mutation -> no functionalization clones (the NOROPE-v2 postmortem).
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from . import _runtime
from .. import arch as _arch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

// ROW GEOMETRY (unified nt >= 1).  grid.x = output row m; grid.y = token
// tile t0 = blockIdx.y * TT.  ``xs`` / ``ys`` are the x / y ROW strides in
// elements (0 is legal at nt == 1: the clamp pins every lane to row 0).
template <int BLK, int TT>
__device__ __forceinline__ void gemv_bias_rows(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y,
    const int d, const int nt, const long xs, const long ys) {
  const long m = blockIdx.x;
  const int t0 = (int)blockIdx.y * TT;
  const uint4* Wv = reinterpret_cast<const uint4*>(W + m * (long)d);
  const uint4* xv[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t)
    xv[t] = reinterpret_cast<const uint4*>(x + (long)min(t0 + t, nt - 1) * xs);
  const int nv = d >> 3;
  float acc[TT];
#pragma unroll
  for (int t = 0; t < TT; ++t) acc[t] = 0.f;
  for (int i = threadIdx.x; i < nv; i += BLK) {
    const uint4 wv = Wv[i];               // weight read ONCE for all TT
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      const uint4 qv = xv[t][i];
      const __nv_bfloat162* q2 = reinterpret_cast<const __nv_bfloat162*>(&qv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 wf = __bfloat1622float2(w2[j]);
        const float2 qf = __bfloat1622float2(q2[j]);
        acc[t] = fmaf(wf.x, qf.x, acc[t]);
        acc[t] = fmaf(wf.y, qf.y, acc[t]);
      }
    }
  }
  __shared__ float red[TT][BLK / 32];
#pragma unroll
  for (int t = 0; t < TT; ++t) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      acc[t] += __shfl_down_sync(~0u, acc[t], o);
    if ((threadIdx.x & 31) == 0) red[t][threadIdx.x >> 5] = acc[t];
  }
  __syncthreads();
  if (threadIdx.x < 32) {
#pragma unroll
    for (int t = 0; t < TT; ++t) {
      float v = (threadIdx.x < BLK / 32) ? red[t][threadIdx.x] : 0.f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
      if (threadIdx.x == 0 && t0 + t < nt)
        y[(long)(t0 + t) * ys + m] =
            __float2bfloat16(v + __bfloat162float(b[m]));
    }
  }
#if __CUDA_ARCH__ >= 900
  // Fire per-CTA completion as the last statement: the y store above is
  // issued first, so the trigger's flush covers it; the dependent (coqkv /
  // rope_and_cache_fixc) gdc-syncs before reading y.  No-op when launched
  // plain.
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// The DEPLOYED entry: PDL entry fence (x is the predecessor's output).
template <int BLK, int TT>
__global__ void __launch_bounds__(BLK) gemv_bias_row(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y,
    const int d, const int nt, const long xs, const long ys) {
#if __CUDA_ARCH__ >= 900
  // PDL: launched with programmatic serialization this grid comes up while
  // the predecessor (rms_norm / q-GEMV) drains; x is predecessor output, so
  // wait before the first x load.  No-op under a plain launch.
  cudaGridDependencySynchronize();
#endif
  gemv_bias_rows<BLK, TT>(W, x, b, y, d, nt, xs, ys);
}

// PDL launch (deployed select idiom): the grid launches under the
// predecessor's tail, hiding launch latency + teardown serialization;
// the kernel's own gdc_sync provides the data fence.  LOCKS_PDL=0 kills.
static bool pdl_on() {
  static int v = -1;
  if (v < 0) {
    const char* e = getenv("LOCKS_PDL");
    v = (e != nullptr && e[0] == '0') ? 0 : 1;
  }
  return v == 1;
}

// TILE POLICY (launch-shape knob, NOT a code-path split).  Monotone in nt;
// nt == 1 -> TT == 1, i.e. literally the pre-unification instruction stream.
// LOCKS_GEMV_TT pins it for the standalone tile sweep (gate only).
static int tt_for(int nt) {
  static int forced = -2;
  if (forced == -2) {
    const char* e = getenv("LOCKS_GEMV_TT");
    forced = (e != nullptr && e[0] != '\0') ? atoi(e) : -1;
  }
  if (forced > 0) return forced;
  if (nt <= 1) return 1;
  if (nt <= 2) return 2;
  if (nt <= 4) return 4;
  return 8;
}

#define LOCKS_GEMV_LAUNCH(TT)                                                 \
  do {                                                                        \
    const int ntile = (nt + (TT) - 1) / (TT);                                 \
    if (pdl_on() || force_pss) {                                              \
      cudaLaunchConfig_t cfg = {};                                            \
      cfg.gridDim = dim3(rows, ntile);                                        \
      cfg.blockDim = dim3(128);                                               \
      cfg.dynamicSmemBytes = 0;                                               \
      cfg.stream = stream;                                                    \
      cudaLaunchAttribute attr[1];                                            \
      attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;        \
      attr[0].val.programmaticStreamSerializationAllowed = 1;                 \
      cfg.attrs = attr;                                                       \
      cfg.numAttrs = 1;                                                       \
      cudaLaunchKernelEx(&cfg, gemv_bias_row<128, TT>, w, x, b, y, d, nt,     \
                         xs, ys);                                             \
    } else {                                                                  \
      gemv_bias_row<128, TT><<<dim3(rows, ntile), 128, 0, stream>>>(          \
          w, x, b, y, d, nt, xs, ys);                                         \
    }                                                                         \
  } while (0)

static void launch_gemv(int rows, int nt, long xs, long ys,
                        cudaStream_t stream,
                        const __nv_bfloat16* w, const __nv_bfloat16* x,
                        const __nv_bfloat16* b, __nv_bfloat16* y,
                        const int d, bool force_pss = false) {
  switch (tt_for(nt)) {
    case 1: LOCKS_GEMV_LAUNCH(1); break;
    case 2: LOCKS_GEMV_LAUNCH(2); break;
    case 4: LOCKS_GEMV_LAUNCH(4); break;
    default: LOCKS_GEMV_LAUNCH(8); break;
  }
}

// ROW-SHAPE HELPER: a 1-D tensor is the nt == 1 instance (rows = 1, stride
// unobservable); a 2-D tensor carries (nt, cols) with its own row stride.
static void rowshape(const torch::Tensor& t, int& n, long& s) {
  if (t.dim() == 1) { n = 1; s = 0; }
  else { n = (int)t.size(0); s = t.stride(0); }
}

// out(nt, M_total) <- [Wq | Wkv] @ x + b, two launches writing disjoint row
// ranges of ONE buffer (the caller's fused split sees stock layout).
void qkv_gemv(torch::Tensor x, torch::Tensor W, torch::Tensor b,
              torch::Tensor out, int64_t q_size) {
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16
              && W.is_contiguous() && b.is_contiguous()
              && out.is_contiguous(), "qkv_gemv: contiguous bf16");
  const int M = W.size(0), d = W.size(1), QS = (int)q_size;
  int nt, ntq; long xs, ys;
  rowshape(x, nt, xs);
  rowshape(out, ntq, ys);
  if (out.dim() == 1) ys = M;
  TORCH_CHECK(nt == ntq && out.numel() == (long)nt * M && b.numel() == M
              && QS < M && (xs % 8) == 0, "qkv_gemv: shapes/alignment");
  auto stream = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(W.data_ptr());
  const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const __nv_bfloat16* bp = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  launch_gemv(QS, nt, xs, ys, stream, w, xp, bp, yp, d);
  launch_gemv(M - QS, nt, xs, ys, stream, w + (long)QS * d, xp, bp + QS,
              yp + QS, d);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "qkv_gemv launch: ", cudaGetErrorString(e));
}

#ifdef LOCKS_FIXC
// FIX-C (session-P lever 1, 2026-07-22): NO-FENCE kv-slice GEMV for the
// select-cover schedule.  Launched with the PSS attribute directly behind
// nrmtopb_select_v2 (which fires griddepcontrol.launch_dependents at
// ENTRY under -DLOCKS_FIXC), so this grid comes up while the 4-CTA select
// runs on an otherwise idle machine (session-P probe 4: joint 0.68..0.75x
// serial).  SAFETY ARGUMENT (audit D-4, the transitive chain): this
// kernel reads ONLY x = the rms output and the static W/bias slabs.  The
// chain rms -> q_gemv -> rki4_score_bt -> [hmax refill] -> select_v2 is
// PLAIN-launched under the FIX-C construction (LOCKS_PDL=0 on H200);
// every plain launch is a full completion + visibility point, so by the
// time select_v2 BEGINS execution (the earliest instant its entry trigger
// can fire), the rms store is complete and visible device-wide.  This
// grid launches strictly after that trigger, hence NO
// cudaGridDependencySynchronize is needed.  Under LOCKS_PDL=1 the same
// holds transitively: score_bt's plain launch orders rms before it, and
// a PSS select_v2 cannot begin before score_bt completes (score_bt fires
// no early trigger).  The deployed gemv_bias_row's entry fence would
// instead spin until select COMPLETES, destroying the cover (audit D-2).
// The per-CTA exit trigger STAYS: rope_and_cache_fixc is this grid's PSS
// dependent and gdc-syncs on it before reading y.
// SAME BODY as the deployed entry (one ``gemv_bias_rows`` instantiation);
// the ONLY difference is the absent entry fence, which is why they cannot
// drift.  No runtime branch exists: the two __global__ wrappers are
// separate compiled entries.
template <int BLK, int TT>
__global__ void __launch_bounds__(BLK) gemv_bias_row_nf(
    const __nv_bfloat16* __restrict__ W, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y,
    const int d, const int nt, const long xs, const long ys) {
  gemv_bias_rows<BLK, TT>(W, x, b, y, d, nt, xs, ys);
}

#define LOCKS_GEMV_NF_LAUNCH(TT)                                              \
  cudaLaunchKernelEx(&cfg, gemv_bias_row_nf<128, TT>, wp, xp, bp, yp, d,      \
                     nt, xs, ys)

// FIX-C kv-slice launch: PSS attr ALWAYS (the FIX-C local chain is
// independent of the LOCKS_PDL global-chain switch, which is 0 on H200).
void kv_gemv_nf(torch::Tensor x, torch::Tensor W, torch::Tensor b,
                torch::Tensor out, int64_t q_size) {
  const int M = W.size(0), d = W.size(1), QS = (int)q_size;
  int nt, nty; long xs, ys;
  rowshape(x, nt, xs);
  rowshape(out, nty, ys);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && nt == nty
              && out.size(-1) == M - QS && (xs % 8) == 0
              && x.stride(-1) == 1 && out.stride(-1) == 1,
              "kv_gemv_nf: bf16 kv slice (nt, M-QS), rows inner-contiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  const int rows = M - QS;
  const int TT = tt_for(nt);
  const int ntile = (nt + TT - 1) / TT;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(rows, ntile);
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const __nv_bfloat16* wp =
      reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()) + (long)QS * d;
  const __nv_bfloat16* xp =
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const __nv_bfloat16* bp =
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()) + QS;
  __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  switch (TT) {
    case 1: LOCKS_GEMV_NF_LAUNCH(1); break;
    case 2: LOCKS_GEMV_NF_LAUNCH(2); break;
    case 4: LOCKS_GEMV_NF_LAUNCH(4); break;
    default: LOCKS_GEMV_NF_LAUNCH(8); break;
  }
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "kv_gemv_nf: ", cudaGetErrorString(e));
}
#endif  // LOCKS_FIXC

// single-slice entries (P2: the fork sits between them)
void q_gemv(torch::Tensor x, torch::Tensor W, torch::Tensor b,
            torch::Tensor out, int64_t q_size) {
  const int d = W.size(1), QS = (int)q_size;
  int nt, nty; long xs, ys;
  rowshape(x, nt, xs);
  rowshape(out, nty, ys);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && nt == nty
              && out.size(-1) == QS && (xs % 8) == 0
              && x.stride(-1) == 1 && out.stride(-1) == 1,
              "q_gemv: bf16 q slice (nt, QS), rows inner-contiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gemv(QS, nt, xs, ys, stream,
              reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), d);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "q_gemv: ", cudaGetErrorString(e));
}

void kv_gemv(torch::Tensor x, torch::Tensor W, torch::Tensor b,
             torch::Tensor out, int64_t q_size) {
  const int M = W.size(0), d = W.size(1), QS = (int)q_size;
  int nt, nty; long xs, ys;
  rowshape(x, nt, xs);
  rowshape(out, nty, ys);
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && nt == nty
              && out.size(-1) == M - QS && (xs % 8) == 0
              && x.stride(-1) == 1 && out.stride(-1) == 1,
              "kv_gemv: bf16 kv slice (nt, M-QS), rows inner-contiguous");
  auto stream = at::cuda::getCurrentCUDAStream();
  launch_gemv(M - QS, nt, xs, ys, stream,
              reinterpret_cast<const __nv_bfloat16*>(W.data_ptr())
                  + (long)QS * d,
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()) + QS,
              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), d);
  cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "kv_gemv: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv_gemv", &qkv_gemv, "deterministic sliced QKV GEMV (+bias)");
  m.def("q_gemv", &q_gemv, "q-slice GEMV (+bias)");
  m.def("kv_gemv", &kv_gemv, "kv-slice GEMV (+bias)");
#ifdef LOCKS_FIXC
  m.def("kv_gemv_nf", &kv_gemv_nf,
        "FIX-C no-fence kv-slice GEMV (PSS launch, select-cover)");
#endif
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
        _cf = ["-O3", _arch.arch_flag()]
        # FIX-C opt-in (LOCKS_QFIRST_FIXC=1): compile the no-fence kv GEMV
        # variant.  Flag unset -> the define is off, the preprocessed TU is
        # byte-identical to the deployed one (SASS-identity gated).
        if os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1":
            _cf.append("-DLOCKS_FIXC")
        _MOD = load_inline(
            name="locks_gemv", cpp_sources="", cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=_cf,
            verbose=bool(int(os.environ.get("LOCKS_CUDA_VERBOSE", "0"))))
        print("[locks_gemv] deterministic sliced-QKV GEMV ACTIVE", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[locks_gemv] build failed ({type(e).__name__}: {e})",
              flush=True)
        _MOD = None
    return _MOD


_OP_READY = False


def ensure_qkv_op() -> None:
    """Register torch.ops.locks_gv.qkv once (opaque, PURE): the nt branch
    runs in the op body per call; outputs are freshly allocated (no
    mutation, no clones).  PEP-563 trap: pin real annotations."""
    global _OP_READY
    if _OP_READY:
        return
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = Library("locks_gv", "FRAGMENT")

    # BIAS-FREE ARCHS (model-agnostic, 2026-07-22): Llama/Qwen build the fused
    # QKV with bias=False, so ``bias`` is None there.  The GEMV kernels fuse
    # the bias add unconditionally (no `if (has_bias)` branch -- the
    # no-fallback rule keeps kernels precondition-asserting), so the DECODE
    # path is handed a real all-zeros row (`v + 0.0f == v`, bitwise).  PREFILL
    # keeps passing bias=None into F.linear, so prefill bytes stay stock.
    def _b(bias, weight):
        return bias if bias is not None else _runtime.zero_bias(weight)

    def qkv(hidden: torch.Tensor, weight: torch.Tensor,
            bias: Optional[torch.Tensor], q_size: int) -> torch.Tensor:
        if hidden.shape[0] == 1:
            # nt == 1 is the WIRE's reason to exist; a failed build must not
            # silently restore the stock fused linear (R1 pattern, coqkv_cuda
            # .py:281).  nt > 1 (prefill) is a shape branch, not a fallback.
            mod = get_mod()
            assert mod is not None, (
                "LOCKS_QKVGEMV: gemv extension build FAILED (get_mod None). "
                "Prebuild offline (LOCKS_CUDA_VERBOSE=1 shows the compile "
                "error) -- no silent stock-linear fallback.")
            out = torch.empty(1, weight.shape[0], dtype=hidden.dtype,
                              device=hidden.device)
            mod.qkv_gemv(hidden.reshape(-1), weight, _b(bias, weight), out,
                         int(q_size))
            return out
        return torch.nn.functional.linear(hidden, weight, bias)

    def qkv_fake(hidden: torch.Tensor, weight: torch.Tensor,
                 bias: Optional[torch.Tensor], q_size: int) -> torch.Tensor:
        return hidden.new_empty(hidden.shape[0], weight.shape[0])

    # ---- P2 split ops: q first (fork point), kv second ------------------- #
    # nt > 1 (prefill): the q op computes the STOCK fused linear ONCE and
    # stashes the kv half for the kv op (prefill bytes stock, single
    # compute); nt == 1: each op runs its GEMV.  The stash lives in op-
    # body python (opaque -> executes per call; keyed by tensor identity,
    # consumed exactly once -- the qf2 mark/nonce precedent).
    _kv_stash: dict = {}

    def q_first(hidden: torch.Tensor, weight: torch.Tensor,
                bias: Optional[torch.Tensor], q_size: int) -> torch.Tensor:
        qs = int(q_size)
        if hidden.shape[0] == 1:
            mod = get_mod()
            assert mod is not None, (
                "LOCKS_QFIRST_GV: gemv extension build FAILED (get_mod "
                "None). Prebuild offline (LOCKS_CUDA_VERBOSE=1) -- no silent "
                "stock-linear fallback.")
            out = torch.empty(1, qs, dtype=hidden.dtype,
                              device=hidden.device)
            mod.q_gemv(hidden.reshape(-1), weight, _b(bias, weight), out, qs)
            return out
        full = torch.nn.functional.linear(hidden, weight, bias)
        _kv_stash[(hidden.data_ptr(), hidden.shape[0])] = \
            full[:, qs:].contiguous()
        return full[:, :qs].contiguous()

    def q_first_fake(hidden: torch.Tensor, weight: torch.Tensor,
                     bias: Optional[torch.Tensor],
                     q_size: int) -> torch.Tensor:
        return hidden.new_empty(hidden.shape[0], int(q_size))

    def kv_second(hidden: torch.Tensor, weight: torch.Tensor,
                  bias: Optional[torch.Tensor], q_size: int) -> torch.Tensor:
        qs = int(q_size)
        kvn = weight.shape[0] - qs
        if hidden.shape[0] == 1:
            mod = get_mod()
            assert mod is not None, (
                "LOCKS_QFIRST_GV: gemv extension build FAILED (get_mod "
                "None). Prebuild offline (LOCKS_CUDA_VERBOSE=1) -- no silent "
                "stock-linear fallback.")
            out = torch.empty(1, kvn, dtype=hidden.dtype,
                              device=hidden.device)
            mod.kv_gemv(hidden.reshape(-1), weight, _b(bias, weight), out, qs)
            return out
        st = _kv_stash.pop((hidden.data_ptr(), hidden.shape[0]), None)
        if st is not None:
            return st
        return torch.nn.functional.linear(
            hidden, weight[qs:], bias[qs:] if bias is not None else None)

    def kv_second_fake(hidden: torch.Tensor, weight: torch.Tensor,
                       bias: Optional[torch.Tensor],
                       q_size: int) -> torch.Tensor:
        return hidden.new_empty(hidden.shape[0],
                                weight.shape[0] - int(q_size))

    _ann = {"hidden": torch.Tensor, "weight": torch.Tensor,
            "bias": Optional[torch.Tensor], "q_size": int,
            "return": torch.Tensor}
    for f in (qkv, qkv_fake, q_first, q_first_fake, kv_second,
              kv_second_fake):
        f.__annotations__ = dict(_ann)
    direct_register_custom_op(op_name="qkv", op_func=qkv, mutates_args=[],
                              fake_impl=qkv_fake, target_lib=lib)
    direct_register_custom_op(op_name="q_first", op_func=q_first,
                              mutates_args=[], fake_impl=q_first_fake,
                              target_lib=lib)
    direct_register_custom_op(op_name="kv_second", op_func=kv_second,
                              mutates_args=[], fake_impl=kv_second_fake,
                              target_lib=lib)
    ensure_qkv_op._lib = lib
    _OP_READY = True
