from __future__ import annotations
import torch
from types import SimpleNamespace
from .topk_scalable import select


def topb_middle(page_score, budget_pages, n_sink, n_recent):
    pages = page_score.numel()
    if budget_pages < n_sink + n_recent:
        raise ValueError("MLA total page budget must include every sink/recent page")
    device = page_score.device
    state = SimpleNamespace(
        score=page_score.reshape(1, 1, pages),
        max_pages=pages,
        n_kv=1,
        sink_pages=n_sink,
        budget_abs=budget_pages - n_sink - n_recent,
        budget=0.0,
        n_pages=torch.tensor([pages], dtype=torch.int32, device=device),
        n_sel_hi=torch.tensor(
            [max(0, pages - n_recent)], dtype=torch.int32, device=device
        ),
        b_fix=torch.tensor(
            [min(budget_pages - n_sink - n_recent, max(0, pages - n_sink - n_recent))],
            dtype=torch.int32,
            device=device,
        ),
        page_table=torch.full((1, 1, pages), -1, dtype=torch.int32, device=device),
        page_cnt=torch.zeros((1, 1), dtype=torch.int32, device=device),
    )
    select(state, 1)
    keep = torch.zeros(pages + 1, dtype=torch.bool, device=device)
    keep.scatter_(
        0, state.page_table[0, 0].long().clamp_min(-1).remainder(pages + 1), True
    )
    return keep[:pages]


_BUILD_ANNOUNCED = False


def _page_basis(blk, r):
    G = torch.matmul(blk, blk.transpose(1, 2))
    (lam, W) = torch.linalg.eigh(G)
    lam = lam[:, -r:].flip(-1)
    W = W[:, :, -r:].flip(-1)
    sig = lam.clamp_min(0).sqrt()
    tol = sig[:, :1] * torch.finfo(sig.dtype).eps ** 0.5
    return (W, torch.where(sig > tol, sig, sig.new_zeros(())))


def build_page_factors(kv_c, n_cache, page, rank, rpad=16):
    Lkv = kv_c.shape[1]
    if n_cache <= 0:
        return (kv_c.new_zeros(0, page, rpad), kv_c.new_zeros(0, rpad, Lkv))
    blk = kv_c[: n_cache * page].view(n_cache, page, Lkv)
    r = min(rank, page, Lkv, rpad)
    global _BUILD_ANNOUNCED
    t0 = None
    if not _BUILD_ANNOUNCED:
        import time

        if blk.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
    (W, sig) = _page_basis(blk, r)
    Lf = kv_c.new_zeros(n_cache, page, rpad)
    Rf = kv_c.new_zeros(n_cache, rpad, Lkv)
    Lf[:, :, :r] = W * sig[:, None, :]
    inv = torch.where(
        sig > 0,
        1.0 / torch.clamp_min(sig, torch.finfo(sig.dtype).tiny),
        sig.new_zeros(()),
    )
    Rf[:, :r, :] = torch.matmul(W.transpose(1, 2), blk) * inv[:, :, None]
    if t0 is not None:
        if blk.is_cuda:
            torch.cuda.synchronize()
        _BUILD_ANNOUNCED = True
        print(
            f"[locks] MLA factor build: pages={n_cache} page={page} Lkv={Lkv} rank={r} gram-eigh {1000.0 * (time.perf_counter() - t0):.1f} ms",
            flush=True,
        )
    return (Lf, Rf)
