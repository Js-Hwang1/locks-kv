"""Direct cusolver ``Xsyevd`` (batched) with a PERSISTENT workspace.

Drop-in for ``torch.linalg.eigh`` on the rki4 page-grams (batched 16x16 fp32),
BITWISE-identical to it -- same solver, same math, same selected pages -- but
with the ~2.4 GiB cusolver workspace allocated ONCE and reused every build,
instead of re-mapped per call (the cuMem* churn that dominates the summary
build's TTFT cost).  This is NOT jacobi: ``torch.linalg.eigh`` on these batched
CUDA grams dispatches to ``cusolverDnXsyevBatched`` (uplo=LOWER, jobz=VECTOR,
CUDA_R_32F -- empirically determined, workspace 2.398 GiB at batch 8192 which
matches the term P1 measured), and this module calls that SAME routine with the
SAME parameters, so its eigenvalues (W) and eigenvectors (U = the column-major
output transposed) are byte-identical to torch's (gate: S2/U torch.equal, PTAB
selection-identity EXACT).

Why it removes the churn AND (potentially) the host sync:
  * The workspace tensor is held by a module-level cache reference, so the
    caching allocator NEVER frees or unmaps it -- every build after the first
    reuses the SAME device pointer, paying ZERO workspace cuMem*.  (The shipped
    ``torch.linalg.eigh`` frees its workspace back to the pool each call, and
    under expandable_segments the pool re-maps the 2.4 GiB block on the next
    build -> the per-build cuMem* churn.)
  * ``cusolverDnXsyevBatched`` is issued purely stream-ordered here and the
    per-matrix convergence ``info`` is NOT read back on the hot path, so this
    call does not host-sync the serving thread (unlike ATen's eigh, which reads
    info + can throw -- an 83.6 ms host stall behind queued work, measured).

torch does not export ``getCurrentCUDASolverDnHandle``, so a PRIVATE cusolver
handle is kept (created once) and bound to torch's CURRENT stream on each call
(stream binding preserves ordering; the syevd numerics are handle-independent,
so this stays bitwise -- proven by the gate).  Workspace size is queried ONCE
via ``*_bufferSize`` at allocation and never re-queried on the hot path.

NOT graph-safe by design (eager cusolver, like the ``torch.linalg.eigh`` it
replaces); runs on page-finalize off the hot path.
"""
from __future__ import annotations

import torch

from .. import arch as _arch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>

// PRIVATE persistent handle + params (torch does not export its cusolver
// handle).  Bound to torch's CURRENT stream on every call so ordering is
// preserved; the eigendecomposition is handle-independent -> bitwise.
static cusolverDnHandle_t g_handle = nullptr;
static cusolverDnParams_t g_params = nullptr;

static cusolverDnHandle_t handle() {
  if (!g_handle)
    TORCH_CHECK(cusolverDnCreate(&g_handle) == CUSOLVER_STATUS_SUCCESS,
                "cusolverDnCreate failed");
  cusolverDnSetStream(g_handle, at::cuda::getCurrentCUDAStream());
  return g_handle;
}
static cusolverDnParams_t params() {
  if (!g_params)
    TORCH_CHECK(cusolverDnCreateParams(&g_params) == CUSOLVER_STATUS_SUCCESS,
                "cusolverDnCreateParams failed");
  return g_params;
}

// device workspace bytes for XsyevBatched at (n, batch), fp32 VECTORS LOWER.
// Queried ONCE at allocation; host workspace is 0 for this routine (asserted).
int64_t xsyevb_dev_bytes(int64_t n, int64_t batch) {
  size_t dev = 0, host = 0;
  auto st = cusolverDnXsyevBatched_bufferSize(
      handle(), params(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, CUDA_R_32F, nullptr, n, CUDA_R_32F, nullptr, CUDA_R_32F,
      &dev, &host, batch);
  TORCH_CHECK(st == CUSOLVER_STATUS_SUCCESS, "XsyevBatched bufferSize st=", (int)st);
  TORCH_CHECK(host == 0, "XsyevBatched host workspace unexpectedly non-zero: ", host);
  return (int64_t)dev;
}

// A (batch,n,n) fp32: OVERWRITTEN with eigenvectors (cusolver column-major).
// W (batch,n): eigenvalues ascending.  ws (uint8, >= dev bytes), info (batch
// int32).  jobz=VECTOR, uplo=LOWER (== torch.linalg.eigh UPLO='L').  Purely
// stream-ordered: info is NOT read here (no host sync).
void xsyevb_run(torch::Tensor A, torch::Tensor W, torch::Tensor ws,
                torch::Tensor info, int64_t batch) {
  int64_t n = A.size(-1);
  auto st = cusolverDnXsyevBatched(
      handle(), params(), CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      n, CUDA_R_32F, A.data_ptr(), n, CUDA_R_32F, W.data_ptr(), CUDA_R_32F,
      ws.data_ptr(), (size_t)ws.numel(), nullptr, (size_t)0,
      info.data_ptr<int>(), batch);
  TORCH_CHECK(st == CUSOLVER_STATUS_SUCCESS, "XsyevBatched st=", (int)st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("xsyevb_dev_bytes", &xsyevb_dev_bytes, "XsyevBatched device workspace bytes");
  m.def("xsyevb_run", &xsyevb_run, "batched Xsyevd with a caller-owned workspace");
}
"""

_MOD = None


def _mod():
    """Build (once) the cusolver Xsyevd extension.  No silent fallback: if the
    opt-in mode is selected the extension must load or the run fails loudly."""
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        _MOD = load_inline(
            name="locks_rki4_syevd", cpp_sources="", cuda_sources=_SRC,
            extra_cuda_cflags=["-O3", _arch.arch_flag()],
            extra_ldflags=["-lcusolver"], with_cuda=True, verbose=False)
    return _MOD


# Persistent workspace cache, keyed by (device_index, n, cap_batch).  Each
# entry holds the device workspace (uint8) and the per-matrix info buffer; the
# Python references keep them off the caching allocator's free list forever, so
# the 2.4 GiB block is mapped ONCE and never re-churned.
_WS: dict[tuple[int, int, int], dict] = {}


def _ws(device: torch.device, n: int, cap_batch: int) -> dict:
    key = (device.index if device.index is not None else
           torch.cuda.current_device(), n, cap_batch)
    e = _WS.get(key)
    if e is None:
        nbytes = _mod().xsyevb_dev_bytes(n, cap_batch)      # ONE-TIME query
        e = _WS[key] = dict(
            ws=torch.empty(nbytes, device=device, dtype=torch.uint8),
            info=torch.empty(cap_batch, device=device, dtype=torch.int32),
            nbytes=nbytes)
    return e


def eigh_syevd(Gm: torch.Tensor, cap_batch: int):
    """(M, n, n) fp32 symmetric grams -> (S2 ascending, U), the
    ``torch.linalg.eigh`` contract, via ``cusolverDnXsyevBatched`` with a
    PERSISTENT workspace.  Chunked at ``cap_batch`` (the workspace is sized for
    it, sufficient for any batch <= cap_batch).  BITWISE-identical to
    ``torch.linalg.eigh`` (gate-verified)."""
    assert Gm.dtype == torch.float32 and Gm.dim() == 3 and Gm.shape[-1] == Gm.shape[-2]
    M, n = Gm.shape[0], Gm.shape[-1]
    e = _ws(Gm.device, n, cap_batch)
    ws, info = e["ws"], e["info"]
    S2 = torch.empty(M, n, device=Gm.device, dtype=torch.float32)
    U = torch.empty_like(Gm)
    for i in range(0, M, cap_batch):
        m = min(cap_batch, M - i)
        A = Gm[i:i + m].contiguous()            # cusolver overwrites -> keep Gm
        W = S2[i:i + m]
        _mod().xsyevb_run(A, W, ws, info[:m], m)
        # cusolver writes eigenvectors column-major; row-major read = U^T.
        U[i:i + m] = A.transpose(-1, -2)
    return S2, U
