"""Fused Triton page-score kernel for LOCKS on MLA models.

Computes, per page p (grid over pages), the LOCKS page score
    page_score[p] = max_h max_{t<valid[p]} (q_c[h].C8[p,t] + q_rope[h].k_rope[p,t]) * scale
without ever materializing the [H, P, page] logit tensor the eager einsum builds
(the decode-time cost the kernel removes). One latent KV head shared by all H
query heads => the head combine is a max-union. Parametric in (H, D, Dr, page)
so one kernel serves V2-Lite (16,512,64,16), V3 (128,...), V4-Flash (64,...).

Gated ranking-identical to `mla_score_torch.mla_page_score_ref`
(see `ours_doc/MLA_KERNEL_DESIGN.md`, gate `scratch_mla/gate_mla_score.py`).

This is the V-recon variant: it consumes the rank-8 *reconstruction* C8, exactly
what the validated smoke selects on. The V-factored (int4 basis) variant is a
documented follow-up.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

NEG = -1e30


@triton.jit
def _mla_page_score_kernel(
    qc_ptr, qr_ptr, c8_ptr, kr_ptr, valid_ptr, out_ptr,
    scale,
    H: tl.constexpr, D: tl.constexpr, Dr: tl.constexpr,
    PAGE: tl.constexpr, BLK_H: tl.constexpr, PREC: tl.constexpr,
):
    p = tl.program_id(0)
    d = tl.arange(0, D)
    dr = tl.arange(0, Dr)
    t = tl.arange(0, PAGE)

    # page-local blocks loaded ONCE, transposed straight from HBM to [feat, PAGE]
    # (avoids holding both C8 and its SRAM transpose -> halves shared memory).
    c8t = tl.load(c8_ptr + p * PAGE * D + d[:, None] + t[None, :] * D)      # [D, PAGE]
    krt = tl.load(kr_ptr + p * PAGE * Dr + dr[:, None] + t[None, :] * Dr)   # [Dr, PAGE]
    valid_p = tl.load(valid_ptr + p)
    tmask = t < valid_p

    page_max = tl.full((), -1e30, tl.float32)
    for h0 in tl.static_range(0, H, BLK_H):
        hh = h0 + tl.arange(0, BLK_H)
        qc = tl.load(qc_ptr + hh[:, None] * D + d[None, :])     # [BLK_H, D]
        qr = tl.load(qr_ptr + hh[:, None] * Dr + dr[None, :])   # [BLK_H, Dr]
        # PREC="tf32": tensor-core dots. The eager reference einsum ALSO runs
        # TF32 (torch allow_tf32 default True on this stack), so this is
        # ranking-identical to it, not an approximation -- gated kept_diff==0
        # (max|score delta| ~6.5e-3, sub-tie). "ieee" (full fp32) is ~60x slower
        # (fp32 has no tensor-core path) and was the original kernel's mistake.
        s = tl.dot(qc, c8t, input_precision=PREC) + \
            tl.dot(qr, krt, input_precision=PREC)               # [BLK_H, PAGE]
        s = tl.where(tmask[None, :], s * scale, -1e30)
        page_max = tl.maximum(page_max, tl.max(s))
    tl.store(out_ptr + p, page_max)


def mla_page_score_triton(q_c, q_rope, C8, k_rope, valid, scale, blk_h=None,
                          prec="tf32"):
    """Return page_score [P] fp32. Fused, tensor-core (tf32) page scorer;
    7.7x over the eager einsum at V3.2 shapes, ranking-identical (gate:
    scratch_mla/mla_fast_kernel.py, kept_diff==0). See MLA_KERNEL_DESIGN.md."""
    assert C8.is_cuda, "MLA score kernel is CUDA-only"
    P, page, D = C8.shape
    H = q_c.shape[0]
    Dr = q_rope.shape[1]
    assert q_c.shape == (H, D)
    assert q_rope.shape == (H, Dr)
    assert k_rope.shape == (P, page, Dr)
    assert valid.shape == (P,)

    q_c = q_c.contiguous().float()
    q_rope = q_rope.contiguous().float()
    C8 = C8.contiguous().float()
    k_rope = k_rope.contiguous().float()
    valid = valid.contiguous().to(torch.int32)

    if blk_h is None:
        # head tile so qc[BLK_H, D] + c8t[D, PAGE] fit shared memory on both
        # sm_90 (H200) and sm_120: 16 (fp32) keeps it ~160KB.
        blk_h = min(H, 16)
    assert H % blk_h == 0, f"H={H} must be divisible by BLK_H={blk_h}"

    out = torch.empty(P, device=C8.device, dtype=torch.float32)
    _mla_page_score_kernel[(P,)](
        q_c, q_rope, C8, k_rope, valid, out,
        float(scale), H, D, Dr, page, blk_h, prec,
        num_warps=8, num_stages=3,
    )
    return out


@triton.jit
def _mla_fac_batched_kernel(
    qc_ptr, qr_ptr, lf_ptr, rf_ptr, kr_ptr, valid_ptr, row_ptr, out_ptr,
    scale, H: tl.constexpr, D: tl.constexpr, Dr: tl.constexpr,
    PAGE: tl.constexpr, R: tl.constexpr, BLK_H: tl.constexpr, PREC: tl.constexpr,
):
    # ONE launch over ALL rows' cached pages (grid = total_pages). row_ptr[pp]
    # maps each global page to its batch row so q is fetched per-row. Reads the
    # COMPACT factors Rf[pp][R,D], Lf[pp][page,R] (score = (q_c@Rf^T)@Lf), never
    # the full C8. This is the launch-overhead fix: 1 kernel/layer, not B.
    pp = tl.program_id(0)
    b = tl.load(row_ptr + pp)
    d = tl.arange(0, D); dr = tl.arange(0, Dr); t = tl.arange(0, PAGE)
    rr = tl.arange(0, R)
    rf = tl.load(rf_ptr + pp * R * D + rr[:, None] * D + d[None, :])         # [R,D]
    lft = tl.load(lf_ptr + pp * PAGE * R + t[:, None] * R + rr[None, :])     # [page,R]
    krt = tl.load(kr_ptr + pp * PAGE * Dr + dr[:, None] + t[None, :] * Dr)   # [Dr,page]
    valid_p = tl.load(valid_ptr + pp); tmask = t < valid_p
    page_max = tl.full((), -1e30, tl.float32)
    for h0 in tl.static_range(0, H, BLK_H):
        hh = h0 + tl.arange(0, BLK_H)
        qc = tl.load(qc_ptr + b * H * D + hh[:, None] * D + d[None, :])      # q_c[b,hh]
        qr = tl.load(qr_ptr + b * H * Dr + hh[:, None] * Dr + dr[None, :])
        qR = tl.dot(qc, tl.trans(rf), input_precision=PREC)                 # [BLK_H,R]
        s_nope = tl.dot(qR, tl.trans(lft), input_precision=PREC)            # [BLK_H,page]
        s_rope = tl.dot(qr, krt, input_precision=PREC)                      # [BLK_H,page]
        s = tl.where(tmask[None, :], (s_nope + s_rope) * scale, -1e30)
        page_max = tl.maximum(page_max, tl.max(s))
    tl.store(out_ptr + pp, page_max)


def mla_page_score_factored_batched(q_c, q_rope, Lf_all, Rf_all, k_rope_all,
                                    valid_all, row_all, scale, blk_h=16,
                                    prec="tf32"):
    """Batched factored page score. q_c/q_rope: [B,H,*]. Lf_all[totalP,page,R],
    Rf_all[totalP,R,Lkv], k_rope_all[totalP,page,Dr], valid_all/row_all[totalP].
    Returns page_score[totalP] fp32 (one score per cached page, in row-concat
    order). See MLA_KERNEL_DESIGN.md; gated ranking vs the eager ref."""
    totalP, page, R = Lf_all.shape
    D = Rf_all.shape[2]
    B, H, _ = q_c.shape
    Dr = q_rope.shape[2]
    assert H % blk_h == 0, f"H={H} must be divisible by BLK_H={blk_h}"
    out = torch.empty(totalP, device=Lf_all.device, dtype=torch.float32)
    if totalP == 0:
        return out
    _mla_fac_batched_kernel[(totalP,)](
        q_c.contiguous().float(), q_rope.contiguous().float(),
        Lf_all.contiguous().float(), Rf_all.contiguous().float(),
        k_rope_all.contiguous().float(), valid_all.contiguous().to(torch.int32),
        row_all.contiguous().to(torch.int32), out,
        float(scale), H, D, Dr, page, R, blk_h, prec,
        num_warps=8, num_stages=3,
    )
    return out
