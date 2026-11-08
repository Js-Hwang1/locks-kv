from __future__ import annotations
import torch
import triton
import triton.language as tl

NEG = -1e30


@triton.jit
def _mla_fac_batched_kernel(
    qc_ptr,
    qr_ptr,
    lf_ptr,
    rf_ptr,
    kr_ptr,
    valid_ptr,
    row_ptr,
    out_ptr,
    scale,
    H: tl.constexpr,
    D: tl.constexpr,
    Dr: tl.constexpr,
    PAGE: tl.constexpr,
    R: tl.constexpr,
    BLK_H: tl.constexpr,
    PREC: tl.constexpr,
):
    pp = tl.program_id(0)
    b = tl.load(row_ptr + pp)
    d = tl.arange(0, D)
    dr = tl.arange(0, Dr)
    t = tl.arange(0, PAGE)
    rr = tl.arange(0, R)
    rf = tl.load(rf_ptr + pp * R * D + rr[:, None] * D + d[None, :])
    lft = tl.load(lf_ptr + pp * PAGE * R + t[:, None] * R + rr[None, :])
    krt = tl.load(kr_ptr + pp * PAGE * Dr + dr[:, None] + t[None, :] * Dr)
    valid_p = tl.load(valid_ptr + pp)
    tmask = t < valid_p
    page_max = tl.full((), -1e30, tl.float32)
    for h0 in tl.static_range(0, H, BLK_H):
        hh = h0 + tl.arange(0, BLK_H)
        qc = tl.load(qc_ptr + b * H * D + hh[:, None] * D + d[None, :])
        qr = tl.load(qr_ptr + b * H * Dr + hh[:, None] * Dr + dr[None, :])
        qR = tl.dot(qc, tl.trans(rf), input_precision=PREC)
        s_nope = tl.dot(qR, tl.trans(lft), input_precision=PREC)
        s_rope = tl.dot(qr, krt, input_precision=PREC)
        s = tl.where(tmask[None, :], (s_nope + s_rope) * scale, -1e30)
        page_max = tl.maximum(page_max, tl.max(s))
    tl.store(out_ptr + pp, page_max)


def mla_page_score_factored_batched(
    q_c,
    q_rope,
    Lf_all,
    Rf_all,
    k_rope_all,
    valid_all,
    row_all,
    scale,
    blk_h=16,
    prec="tf32",
):
    (totalP, page, R) = Lf_all.shape
    D = Rf_all.shape[2]
    (B, H, _) = q_c.shape
    Dr = q_rope.shape[2]
    assert H % blk_h == 0, f"H={H} must be divisible by BLK_H={blk_h}"
    out = torch.empty(totalP, device=Lf_all.device, dtype=torch.float32)
    if totalP == 0:
        return out
    _mla_fac_batched_kernel[totalP,](
        q_c.contiguous().float(),
        q_rope.contiguous().float(),
        Lf_all.contiguous().float(),
        Rf_all.contiguous().float(),
        k_rope_all.contiguous().float(),
        valid_all.contiguous().to(torch.int32),
        row_all.contiguous().to(torch.int32),
        out,
        float(scale),
        H,
        D,
        Dr,
        page,
        R,
        blk_h,
        prec,
        num_warps=8,
        num_stages=3,
    )
    return out
