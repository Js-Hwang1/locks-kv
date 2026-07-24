"""Stage B, launch 2 of 2: the split-K merge.

The split kernel (``attention/decode.py``) writes one online-softmax partial
``(m, l, acc)`` per (request, kv-head, split). This kernel recombines the active
splits of each (request, query-head) into the final attention output with the
standard flash-decode rescale, and writes ``out`` in place.

Active-split count is recomputed on device from ``page_cnt`` (the same
``pps = max(cdiv(cnt, SPLIT), PPS_MIN)`` rule the split kernel used), so inactive
splits that stored nothing are masked out and never read. Launch shape is fixed
(``(n_req, n_kv*G)``); every dynamic quantity is a device-side value -> the whole
Stage B stays inside a FULL CUDA graph. Padded request rows (``page_cnt == 0``
everywhere) write zeros, never NaN.
"""
from __future__ import annotations

import os

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

# ORDER2 (2026-07-19, FMERGE-v2): the GOLDEN adjacent-pair halving order for
# the two fp sums.  Built from LOGICAL ops only (reshape / trans / split /
# elementwise add), so its value sequence is independent of the Triton
# layout a given kernel context picks -- the property tl.sum lacks (K8a
# refutation: identical source shapes still diverged by 1 bf16 ulp inside a
# different kernel).  The fused in-kernel merge in tier/decode_mem.py uses
# the IDENTICAL op sequence => bitwise-equal across kernels by construction.
# m_max stays tl.max (fp max is order-free).  Decode-numerics class vs the
# legacy order (PTAB step-1 binding, token shas advisory).  Enabled per
# process by LOCKS_MERGE_ORDER2=1 (the mem arm's deployed env); every
# launcher passes the flag so resident references and the mem path always
# agree inside one process.
_ORDER2 = os.environ.get("LOCKS_MERGE_ORDER2", "0") == "1"


@triton.jit
def _fold_vec(x, n: tl.constexpr):
    a, b = tl.split(tl.reshape(x, (n // 2, 2)))
    return a + b


@triton.jit
def _fold_rows(y, n: tl.constexpr, D: tl.constexpr):
    a, b = tl.split(tl.trans(tl.reshape(y, (n // 2, 2, D)), (0, 2, 1)))
    return a + b


@triton.jit
def _tree_sum_vec(x, N: tl.constexpr):
    """Adjacent-pair halving sum of a [N] tensor -> [1] (N power of 2,
    N <= 1024).  Logical ops only; trace-time unrolled with direct constexpr
    expressions (reassigning a constexpr demotes it to a tensor)."""
    if N >= 2:
        x = _fold_vec(x, N)
    if N >= 4:
        x = _fold_vec(x, N // 2)
    if N >= 8:
        x = _fold_vec(x, N // 4)
    if N >= 16:
        x = _fold_vec(x, N // 8)
    if N >= 32:
        x = _fold_vec(x, N // 16)
    if N >= 64:
        x = _fold_vec(x, N // 32)
    if N >= 128:
        x = _fold_vec(x, N // 64)
    if N >= 256:
        x = _fold_vec(x, N // 128)
    if N >= 512:
        x = _fold_vec(x, N // 256)
    if N >= 1024:
        x = _fold_vec(x, N // 512)
    return x


@triton.jit
def _tree_sum_rows(y, N: tl.constexpr, D: tl.constexpr):
    """Adjacent-pair halving sum over axis 0 of a [N, D] tensor -> [D]
    (N power of 2, N <= 1024).  Same trace-time unrolling as above."""
    if N >= 2:
        y = _fold_rows(y, N, D)
    if N >= 4:
        y = _fold_rows(y, N // 2, D)
    if N >= 8:
        y = _fold_rows(y, N // 4, D)
    if N >= 16:
        y = _fold_rows(y, N // 8, D)
    if N >= 32:
        y = _fold_rows(y, N // 16, D)
    if N >= 64:
        y = _fold_rows(y, N // 32, D)
    if N >= 128:
        y = _fold_rows(y, N // 64, D)
    if N >= 256:
        y = _fold_rows(y, N // 128, D)
    if N >= 512:
        y = _fold_rows(y, N // 256, D)
    if N >= 1024:
        y = _fold_rows(y, N // 512, D)
    return tl.reshape(y, (D,))


@triton.jit
def merge_splits_kernel(m_ptr, l_ptr, acc_ptr, cnt_ptr, o_ptr,
                        stride_ot, stride_oh, stride_pr, stride_cntr,
                        G: tl.constexpr, D: tl.constexpr,
                        SPLIT: tl.constexpr, SPLIT_PAD: tl.constexpr,
                        PPS_MIN: tl.constexpr, PDL: tl.constexpr = False,
                        ORDER2: tl.constexpr = False,
                        PW: tl.constexpr = True):
    r = tl.program_id(0)
    h = tl.program_id(1)
    if PDL and not PW:
        # K9-v1 placement (measurement ctl): wait before ANY load.
        gdc_wait()
        gdc_launch_dependents()
    kh = h // G
    g = h % G
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
    n_act = tl.cdiv(cnt, pps)              # splits that actually stored a partial
    offs_s = tl.arange(0, SPLIT_PAD)
    offs_d = tl.arange(0, D)
    smask = offs_s < n_act
    idx = r * stride_pr + (kh * SPLIT + offs_s) * G + g
    if PDL and PW:
        # K9-v2: everything above is legal PRE-WAIT work -- page_cnt is the
        # SELECT kernel's output, and select completed before our
        # predecessor (the split) started.  Only the partials (the split's
        # own stores) need the wait.  Measured wait window ~7.6us/layer
        # (nsys_r3: duration 10.9 vs 3.3us true work).
        gdc_wait()
        gdc_launch_dependents()
    m = tl.load(m_ptr + idx, mask=smask, other=float("-inf"))
    l = tl.load(l_ptr + idx, mask=smask, other=0.0)
    acc = tl.load(acc_ptr + idx[:, None] * D + offs_d[None, :],
                  mask=smask[:, None], other=0.0)
    m_max = tl.max(m, axis=0)
    # empty splits carry (m=-inf, l=0): weight 0 (the `where` guards -inf - -inf)
    w = tl.where(l > 0, tl.exp(m - m_max), 0.0)
    if ORDER2:
        l_tot = tl.reshape(_tree_sum_vec(l * w, SPLIT_PAD), ())
        o = _tree_sum_rows(acc * w[:, None], SPLIT_PAD, D)
    else:
        l_tot = tl.sum(l * w, axis=0)
        o = tl.sum(acc * w[:, None], axis=0)
    # padded request rows (cnt = 0 everywhere): write 0, never NaN
    o = tl.where(l_tot > 0, o / l_tot, 0.0)
    tl.store(o_ptr + r * stride_ot + h * stride_oh + offs_d,
             o.to(o_ptr.dtype.element_ty))
