"""Torch reference for LOCKS page scoring on MLA models (DeepSeek V2-Lite/V3/V4).

This is the *oracle* that the Triton kernel (`mla_score_triton.py`) is gated
against. It is the scoring of `scratch_mla/mla_smoke.py::_r8_keep`, isolated and
made parametric / G-agnostic. See `ours_doc/MLA_KERNEL_DESIGN.md`.

Shapes (per decode step, one layer):
  q_c    : [H, D]         absorbed query in the latent space (q_nope @ W_UK[h])
  q_rope : [H, Dr]        RoPE'd query pe part (per head)
  C8     : [P, page, D]   rank-8 reconstruction of each page's c_KV block
  k_rope : [P, page, Dr]  exact RoPE keys per page (kept exact)
  valid  : [P]            real-token count per page (last page may be short)
  scale  : float          MLA softmax_scale

MLA keeps ONE latent KV head shared by all H query heads, so the page score is a
max-union over ALL heads (not a GQA group combine):
  logit[h,p,t]  = (q_c[h].C8[p,t] + q_rope[h].k_rope[p,t]) * scale
  page_score[p] = max_h max_{t<valid[p]} logit[h,p,t]
"""
from __future__ import annotations

import torch

NEG = -1e30


def mla_page_score_ref(q_c, q_rope, C8, k_rope, valid, scale):
    """Return page_score [P] (fp32). Pure-torch reference; see module docstring."""
    q_c = q_c.float()
    q_rope = q_rope.float()
    C8 = C8.float()
    k_rope = k_rope.float()
    P, page, D = C8.shape
    H = q_c.shape[0]
    Dr = q_rope.shape[1]
    assert q_c.shape == (H, D)
    assert k_rope.shape == (P, page, Dr)
    assert valid.shape == (P,)

    s_nope = torch.einsum("hD,ptD->hpt", q_c, C8)          # [H,P,page]
    s_rope = torch.einsum("hd,ptd->hpt", q_rope, k_rope)   # [H,P,page]
    approx = (s_nope + s_rope) * scale                     # [H,P,page]

    tok = torch.arange(page, device=C8.device)
    vmask = tok[None, :] < valid[:, None]                  # [P,page]
    approx = approx.masked_fill(~vmask[None], NEG)
    head = approx.amax(dim=2)                              # [H,P]
    return head.amax(dim=0)                                # [P]


def topb_middle(page_score, budget_pages, n_sink, n_recent):
    """Kept-page bool mask [P]: sink + recent forced, top-scored middle fills
    the rest. Mirrors mla_smoke._kept_pages so the kernel gate can compare the
    full selection, not just the raw score."""
    P = page_score.shape[0]
    keep = torch.zeros(P, dtype=torch.bool, device=page_score.device)
    if P <= budget_pages:
        keep[:] = True
        return keep
    forced = set(range(min(n_sink, P))) | set(range(max(0, P - n_recent), P))
    for p in forced:
        keep[p] = True
    n_mid = budget_pages - len(forced)
    if n_mid > 0:
        cand = torch.ones(P, dtype=torch.bool, device=page_score.device)
        for p in forced:
            cand[p] = False
        idx = cand.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            k = min(n_mid, idx.numel())
            top = page_score[idx].topk(k).indices
            keep[idx[top]] = True
    return keep
