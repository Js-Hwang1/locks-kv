from __future__ import annotations
import triton
import triton.language as tl

_ORDER2 = False


@triton.jit
def merge_splits_kernel(
    m_ptr,
    l_ptr,
    acc_ptr,
    cnt_ptr,
    o_ptr,
    stride_ot,
    stride_oh,
    stride_pr,
    stride_cntr,
    G: tl.constexpr,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
    SPLIT_PAD: tl.constexpr,
    PPS_MIN: tl.constexpr,
    PDL: tl.constexpr = False,
    ORDER2: tl.constexpr = False,
    PW: tl.constexpr = True,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    kh = h // G
    g = h % G
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
    n_act = tl.cdiv(cnt, pps)
    offs_s = tl.arange(0, SPLIT_PAD)
    offs_d = tl.arange(0, D)
    smask = offs_s < n_act
    idx = r * stride_pr + (kh * SPLIT + offs_s) * G + g
    m = tl.load(m_ptr + idx, mask=smask, other=float("-inf"))
    l = tl.load(l_ptr + idx, mask=smask, other=0.0)
    acc = tl.load(
        acc_ptr + idx[:, None] * D + offs_d[None, :], mask=smask[:, None], other=0.0
    )
    m_max = tl.max(m, axis=0)
    w = tl.where(l > 0, tl.exp(m - m_max), 0.0)
    l_tot = tl.sum(l * w, axis=0)
    o = tl.sum(acc * w[:, None], axis=0)
    o = tl.where(l_tot > 0, o / l_tot, 0.0)
    tl.store(
        o_ptr + r * stride_ot + h * stride_oh + offs_d, o.to(o_ptr.dtype.element_ty)
    )
