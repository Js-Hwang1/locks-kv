from __future__ import annotations
import os
import torch
from .build import (
    _build_blocks,
    _bulk_blocks,
    _delta_select,
    _scatter_index,
    _tail_blocks,
    _stack_key_pages,
)
from .rki4_state import IBITS, QMAX, QMIN, RNK, Rki4State
from .. import quant as _kvq
from .rki4_state import MUC as _MUC

_EIGH_CHUNK = int(os.environ.get("LOCKS_EIGH_CHUNK", "8192"))
_RKI4_EIGH = "cusolver".strip().lower()
_EIGH_MS = 0.0
_EIGH_N = 0
_EIGH_GRAMS = 0


def _eigh_chunked(Gf: torch.Tensor):
    _bp = os.environ.get("LOCKS_BUILD_TIME") == "1"
    if _bp:
        global _EIGH_MS, _EIGH_N, _EIGH_GRAMS
        _be0 = torch.cuda.Event(enable_timing=True)
        _be1 = torch.cuda.Event(enable_timing=True)
        _be0.record()
    n = Gf.shape[-1]
    S2 = torch.empty(Gf.shape[0], n, device=Gf.device)
    U = torch.empty_like(Gf)
    for i in range(0, Gf.shape[0], _EIGH_CHUNK):
        (S2[i : i + _EIGH_CHUNK], U[i : i + _EIGH_CHUNK]) = torch.linalg.eigh(
            Gf[i : i + _EIGH_CHUNK]
        )
    res = (S2, U)
    if _bp:
        _be1.record()
        torch.cuda.synchronize()
        _EIGH_MS += _be0.elapsed_time(_be1)
        _EIGH_N += 1
        _EIGH_GRAMS += int(Gf.shape[0])
        print(
            f"[locks] eigh BUILD PROFILE: {_EIGH_N} calls, {_EIGH_GRAMS} grams, {_EIGH_MS / 1000:.2f}s total",
            flush=True,
        )
    return res


def _mask_blocks(st: Rki4State, K: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    nb_state = (
        st.pp_rec.shape[1]
        if getattr(st, "pp_rec", None) is not None
        else st.pp_v4.shape[1]
    )
    nb = min(int(nb_state), int(K.shape[0]))
    return blocks[(blocks >= 0) & (blocks < nb)]


def _write_pre(st: Rki4State, Kp_flat: torch.Tensor):
    (page, tagw) = (st.page, st.tagw)
    tag_new = Kp_flat[:, page - 1, :, :tagw]
    K = Kp_flat.permute(0, 2, 1, 3).float().contiguous()
    mu = K.mean(2)
    dc = K - mu[:, :, None]
    Gm = dc @ dc.transpose(-1, -2)
    return (tag_new, mu, dc, Gm)


def _pack_ibit(Vc: torch.Tensor) -> torch.Tensor:
    if IBITS == 8:
        return Vc.view(torch.uint8).contiguous()
    ppb = 8 // IBITS
    mask = (1 << IBITS) - 1
    out = (Vc[..., 0::ppb] & mask).to(torch.uint8)
    for k in range(1, ppb):
        out = out | (Vc[..., k::ppb] & mask).to(torch.uint8) << k * IBITS
    return out.contiguous()


def _write_post(st: Rki4State, mu, dc, S2, U, tag_new, lidx, bidx) -> None:
    page = st.page
    (N, n_kv) = (mu.shape[0], mu.shape[1])
    S2 = S2.reshape(N, n_kv, page)
    U = U.reshape(N, n_kv, page, page)
    S8 = S2[..., -RNK:].clamp_min(1e-10).sqrt()
    U8 = U[..., -RNK:]
    C = U8 * S8[..., None, :]
    V = dc.transpose(-1, -2) @ U8 / S8[..., None, :]
    vsc = (V.abs().amax(2) / float(QMAX)).clamp_min(1e-08)
    Vq = torch.round(V / vsc[:, :, None]).clamp(QMIN, QMAX)
    Vc = Vq.permute(0, 1, 3, 2).contiguous().to(torch.int8)
    v4 = _pack_ibit(Vc)
    csc = (C.abs().amax(-1) / 127.0).clamp_min(1e-08)
    Cq = torch.round(C / csc[..., None]).clamp(-127, 127)
    msc = (mu.abs().amax(-1) / 127.0).clamp_min(1e-08)
    muq = torch.round(mu / msc[..., None]).clamp(-127, 127)
    st.pp_v4[lidx, bidx] = v4
    st.pp_vs[lidx, bidx] = vsc.to(torch.bfloat16)
    c8q = Cq.to(torch.int8)
    st.pp_c8[lidx, bidx] = c8q
    st.pp_cs[lidx, bidx] = csc.to(torch.bfloat16)
    if _MUC and st.G == 8:
        muq = torch.cat([muq[..., 0::2], muq[..., 1::2]], dim=-1)
    st.pp_mu8[lidx, bidx] = muq.to(torch.int8)
    st.pp_mus[lidx, bidx] = msc.to(torch.bfloat16)
    if getattr(st, "pp_nrmC", None) is not None:
        nrmC = (Cq.abs().sum(-1) * csc).amax(-1) * (1.0 + 2.0 ** (-7))
        st.pp_nrmC[lidx, bidx] = nrmC.to(torch.bfloat16)
    st.pp_tag[lidx, bidx] = tag_new.to(st.pp_tag.dtype)


def _rki4_write(
    st: Rki4State, Kp_flat: torch.Tensor, lidx: torch.Tensor, bidx: torch.Tensor
) -> None:
    nb = (
        st.pp_v4.shape[1] if getattr(st, "pp_rec", None) is None else st.pp_rec.shape[1]
    )
    ok = (bidx >= 0) & (bidx < nb)
    (Kp_flat, lidx, bidx) = (Kp_flat[ok], lidx[ok], bidx[ok])
    tag_override = None
    if _kvq.KVQ is not None:
        hooked = _kvq.build_hook(st, Kp_flat, lidx, bidx)
        if hooked is None:
            return
        (Kp_flat, tag_override, lidx, bidx) = hooked
    (tag_new, mu, dc, Gm) = _write_pre(st, Kp_flat)
    if tag_override is not None:
        tag_new = tag_override
    (S2, U) = _eigh_chunked(Gm.reshape(-1, st.page, st.page))
    _write_post(st, mu, dc, S2, U, tag_new, lidx, bidx)


def rki4_build_tail(
    st: Rki4State,
    K_layers,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    n_req: int,
    rows=None,
) -> int:
    if n_req == 0:
        return 0
    from .rki4_tailgraph import tail_graph_run

    r = tail_graph_run(st, K_layers, block_table, seq_lens, n_req, rows)
    if r is not None:
        return r
    L = len(K_layers)
    blocks = _tail_blocks(block_table, seq_lens, n_req, st.page, rows)
    blocks = _mask_blocks(st, K_layers[0], blocks)
    if blocks.numel() == 0:
        return 0
    n = blocks.shape[0]
    Kp_raw = _stack_key_pages(K_layers, blocks)
    (lidx, bidx) = _scatter_index(L, 0, blocks)
    _rki4_write(st, Kp_raw.reshape(L * n, st.page, st.n_kv, st.d), lidx, bidx)
    return L * n


def rki4_build_bulk(
    st: Rki4State,
    K_layers,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    n_req: int,
    max_fin: int,
    chunk_pairs: int = 8192,
) -> int:
    if n_req == 0 or max_fin <= 0:
        return 0
    blocks = _bulk_blocks(block_table, seq_lens, n_req, page=st.page, max_fin=max_fin)
    blocks = _mask_blocks(st, K_layers[0], blocks)
    return _build_blocks(_rki4_write, st, K_layers, blocks, chunk_pairs)


def rki4_build_delta(
    st: Rki4State,
    K_layers,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    n_req: int,
    max_fin: int,
    chunk_pairs: int = 8192,
) -> int:
    if n_req == 0 or max_fin <= 0:
        return 0
    blocks = _delta_select(
        st.pp_tag, K_layers, block_table, seq_lens, n_req, st.page, max_fin, st.tagw
    )
    blocks = _mask_blocks(st, K_layers[0], blocks)
    return _build_blocks(_rki4_write, st, K_layers, blocks, chunk_pairs)


def rki4_build_refresh(
    st: Rki4State,
    layer: int,
    K: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    n_req: int,
    *,
    force: bool = False,
) -> int:
    device = K.device
    (page, tagw) = (st.page, st.tagw)
    if n_req == 0:
        return 0
    sl = seq_lens[:n_req].to(torch.int64)
    n_fin = sl // page
    max_fin = int(n_fin.max().item())
    if max_fin == 0:
        return 0
    pidx = torch.arange(max_fin, device=device)
    fin_mask = pidx[None, :] < n_fin[:, None]
    bt = block_table[:n_req, :max_fin].to(torch.int64)
    blocks_flat = bt[fin_mask]
    blocks_flat = _mask_blocks(st, K, blocks_flat)
    if blocks_flat.numel() == 0:
        return 0
    if force:
        stale = torch.ones(blocks_flat.numel(), dtype=torch.bool, device=device)
    else:
        tag_cur = K[blocks_flat, page - 1, :, :tagw].float()
        tag_old = st.pp_tag[layer, blocks_flat].float()
        stale = (tag_cur != tag_old).any(dim=(1, 2))
    sb = torch.unique(blocks_flat[stale])
    if sb.numel() == 0:
        return 0
    lidx = torch.full((sb.numel(),), layer, device=device, dtype=torch.int64)
    _rki4_write(st, K[sb], lidx, sb)
    return int(sb.numel())
