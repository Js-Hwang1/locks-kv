from __future__ import annotations
import torch


def _tail_blocks(block_table, seq_lens, n_req: int, page: int, rows):
    if rows is None:
        sl = seq_lens[:n_req].to(torch.int64)
        bt = block_table[:n_req]
    else:
        sl = seq_lens[rows].to(torch.int64)
        bt = block_table[rows]
    fin1 = (sl // page - 1).clamp_(min=0)
    return bt.to(torch.int64).gather(1, fin1[:, None]).squeeze(1)


def _scatter_index(L: int, l0: int, blocks: torch.Tensor):
    n = blocks.shape[0]
    lidx = torch.arange(
        l0, l0 + L, device=blocks.device, dtype=torch.int64
    ).repeat_interleave(n)
    bidx = blocks.repeat(L)
    return (lidx, bidx)


def _bulk_blocks(block_table, seq_lens, n_req: int, page: int, max_fin: int):
    device = block_table.device
    sl = seq_lens[:n_req].to(torch.int64)
    n_fin = sl // page
    pidx = torch.arange(max_fin, device=device)
    valid = pidx[None, :] < n_fin[:, None]
    return (
        block_table[:n_req, :max_fin]
        .to(torch.int64)
        .masked_fill_(~valid, 0)
        .reshape(-1)
    )


def _stack_key_pages(K_layers, blocks, *, token=None, width=None):
    first = K_layers[0]
    tier = getattr(first, "_tier", None)
    if tier is not None and all(
        (
            getattr(key, "_tier", None) is tier
            and getattr(key, "_lidx", -1) == first._lidx + index
            for (index, key) in enumerate(K_layers)
        )
    ):
        from ..tier.gather import gather_layers

        return gather_layers(
            tier, first._lidx, len(K_layers), blocks, token=token, width=width
        )
    if token is None:
        return torch.stack([key[blocks] for key in K_layers])
    return torch.stack([key[blocks, token, :, :width] for key in K_layers])


def _build_blocks(
    write_fn, st, K_layers, blocks: torch.Tensor, chunk_pairs: int
) -> int:
    P = blocks.shape[0]
    if P == 0:
        return 0
    L = len(K_layers)
    (page, n_kv, d) = (st.page, st.n_kv, st.d)
    if P <= chunk_pairs:
        lg = max(1, chunk_pairs // P)
        for l0 in range(0, L, lg):
            Ks = K_layers[l0 : l0 + lg]
            Kp_raw = _stack_key_pages(Ks, blocks)
            (lidx, bidx) = _scatter_index(len(Ks), l0, blocks)
            write_fn(st, Kp_raw.reshape(len(Ks) * P, page, n_kv, d), lidx, bidx)
    else:
        for l, K in enumerate(K_layers):
            for s0 in range(0, P, chunk_pairs):
                bs_ = blocks[s0 : s0 + chunk_pairs]
                (lidx, bidx) = _scatter_index(1, l, bs_)
                write_fn(st, K[bs_], lidx, bidx)
    return L * P


def _delta_select(
    tag_slab,
    K_layers,
    block_table,
    seq_lens,
    n_req: int,
    page: int,
    max_fin: int,
    tagw: int,
) -> torch.Tensor:
    device = block_table.device
    sl = seq_lens[:n_req].to(torch.int64)
    fin = sl // page
    pidx = torch.arange(max_fin, device=device)
    valid = pidx[None, :] < fin[:, None]
    blk = (
        block_table[:n_req, :max_fin]
        .to(torch.int64)
        .masked_fill_(~valid, 0)
        .reshape(-1)
    )
    tag_cur = _stack_key_pages(K_layers, blk, token=page - 1, width=tagw)
    stale = (tag_cur != tag_slab[:, blk]).any(-1).any(-1)
    stale = stale.any(dim=0) & valid.reshape(-1)
    return blk[stale]
