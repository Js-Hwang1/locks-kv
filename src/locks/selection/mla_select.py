"""MLA page selection -- the single entry the LOCKS MLA backend forwards to.

MLA (DeepSeek V2/V3/V3.2) keeps ONE latent KV head; the page score is a
max-union over ALL query heads of the exact logit
``(q_c . C8 + q_rope . k_rope) * scale`` over a LOCAL rank-``rank`` summary of the
page's latent block (``mla_score_torch`` / ``mla_score_triton``). This mirrors the
GQA ``select_pages`` adapter (score -> top-b keep mask) but on the latent cache.
"""
from __future__ import annotations

import torch

from .mla_score_torch import mla_page_score_ref, topb_middle

try:  # the Triton kernel imports triton at module load; keep import-safe
    from .mla_score_triton import mla_page_score_triton
except Exception:  # pragma: no cover
    mla_page_score_triton = None

NEG = float("-inf")


_BUILD_ANNOUNCED = False


def _page_basis(blk, r):
    """Top-``r`` left singular basis of each page from the SMALL page x page
    Gram, NOT a tall-skinny SVD.

    This is the construction the method is defined by (eigendecompose the page
    Gram $D D^\\top \\in R^{B x B}$) and the one the GQA path already uses
    (``r8i4_syevd`` -> ``cusolverDnXsyevBatched``). The MLA path used to call
    ``torch.linalg.svd`` on the [n, page, Lkv] = [n, 64, 512] blocks instead,
    which dispatches to a Jacobi-type cuSOLVER routine: on 2026-08-13 that
    WEDGED for 44 h on 8 H200s with every TP rank parked in this one call, so
    the Gram route is a correctness requirement here, not an optimization.

    Returns ``(W, sig)``: ``W [n, page, r]`` orthonormal columns and
    ``sig [n, r]`` singular values, descending. Modes below the spectral floor
    are returned with ``sig == 0`` so the caller can drop them; the method's
    build specifies exactly that for zero modes, and it keeps the ``Vh = W^T
    blk / sig`` form from amplifying noise directions.
    """
    G = torch.matmul(blk, blk.transpose(1, 2))          # [n, page, page], PSD
    lam, W = torch.linalg.eigh(G)                       # ascending eigenvalues
    lam = lam[:, -r:].flip(-1)                          # descending, top r
    W = W[:, :, -r:].flip(-1)                           # [n, page, r]
    sig = lam.clamp_min(0).sqrt()
    tol = sig[:, :1] * (torch.finfo(sig.dtype).eps ** 0.5)
    return W, torch.where(sig > tol, sig, sig.new_zeros(()))


def build_page_summary(kv_c, n_cache, page, rank):
    """Rank-``rank`` local reconstruction ``C8 [n_cache, page, Lkv]`` of the
    prompt's latent block. ``n_cache`` = the number of FULL prompt pages;
    generated/partial pages are scored as the recent window."""
    Lkv = kv_c.shape[1]
    if n_cache <= 0:
        return kv_c.new_zeros(0, page, Lkv)
    blk = kv_c[:n_cache * page].view(n_cache, page, Lkv)
    r = min(rank, page, Lkv)
    W, sig = _page_basis(blk, r)
    # W W^T blk IS the truncated-SVD reconstruction, with dropped modes zeroed
    # out of the projector; Vh never has to be formed.
    W = W * (sig > 0).to(blk.dtype)[:, None, :]
    return torch.matmul(W, torch.matmul(W.transpose(1, 2), blk))


def build_page_factors(kv_c, n_cache, page, rank, rpad=16):
    """Per-page rank-``rank`` SVD FACTORS (Lf[n_cache,page,rpad], Rf[n_cache,
    rpad,Lkv]) instead of the full reconstruction C8=Lf@Rf. Two wins over
    build_page_summary: the cache is ~``Lkv/rpad``x smaller (the OOM fix -- a page
    is stored as rpad*Lkv + page*rpad, not page*Lkv), and the factored score
    reads O(P*rpad*Lkv) not O(P*page*Lkv). ``rpad`` (>=16) pads the rank so the
    Triton score kernel's tl.dot has contraction/output dims >=16; the extra
    ranks are ZERO so the score is unchanged (C8 identical to build_page_summary
    on the first r columns)."""
    Lkv = kv_c.shape[1]
    if n_cache <= 0:
        return (kv_c.new_zeros(0, page, rpad), kv_c.new_zeros(0, rpad, Lkv))
    blk = kv_c[:n_cache * page].view(n_cache, page, Lkv)
    r = min(rank, page, Lkv, rpad)
    global _BUILD_ANNOUNCED
    t0 = None
    if not _BUILD_ANNOUNCED:                 # one-shot: a stalled build used to
        import time                          # be invisible (44 h, no log line)
        if blk.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
    W, sig = _page_basis(blk, r)
    # Same Lf, Rf the SVD produced (U = W, Vh = W^T blk / sig), so the factored
    # score and therefore the page ranking are unchanged. Dropped modes get an
    # exactly-zero Rf row rather than a 1/sig blow-up, which would otherwise
    # meet a zero Lf column as 0 * inf = NaN.
    Lf = kv_c.new_zeros(n_cache, page, rpad)
    Rf = kv_c.new_zeros(n_cache, rpad, Lkv)
    Lf[:, :, :r] = W * sig[:, None, :]
    inv = torch.where(sig > 0, 1.0 / torch.clamp_min(sig, torch.finfo(sig.dtype).tiny),
                      sig.new_zeros(()))
    Rf[:, :r, :] = torch.matmul(W.transpose(1, 2), blk) * inv[:, :, None]
    if t0 is not None:
        if blk.is_cuda:
            torch.cuda.synchronize()
        _BUILD_ANNOUNCED = True
        print(f"[locks] MLA factor build: pages={n_cache} page={page} "
              f"Lkv={Lkv} rank={r} gram-eigh {1e3 * (time.perf_counter() - t0):.1f} ms",
              flush=True)
    return (Lf, Rf)


def mla_page_score(q_c, q_rope, C8, k_rope, valid, scale, use_triton=False):
    """Per-page LOCKS score over the cached pages. Torch ref by default (the
    eager decode is already O(L); the fused Triton kernel is the latency path
    and is gated ranking-identical to the ref)."""
    if use_triton and mla_page_score_triton is not None:
        return mla_page_score_triton(q_c, q_rope, C8, k_rope, valid, scale)
    return mla_page_score_ref(q_c, q_rope, C8, k_rope, valid, scale)


def mla_select_pages(q_c, q_rope, C8, k_rope, valid, scale, n_pages,
                     budget_pages, n_sink, n_recent, use_triton=False):
    """Return keep bool[n_pages]. Cached prompt pages are scored via the local
    summary; uncached (recent/generated) pages get -inf and are kept by the
    recent-window force in ``topb_middle``."""
    n_cache = C8.shape[0]
    page_score = q_c.new_full((n_pages,), NEG)
    if n_cache > 0:
        page_score[:n_cache] = mla_page_score(
            q_c, q_rope, C8, k_rope, valid, scale, use_triton=use_triton)
    return topb_middle(page_score, budget_pages, n_sink, n_recent)
