from __future__ import annotations
from typing import Protocol, Tuple
import torch
import triton
import triton.language as tl
from .merge import merge_splits_kernel

_DECODE_NS = 3


class VSource(Protocol):
    SRC_ID: int

    def args(self) -> Tuple[torch.Tensor, int, int, int]: ...


def _split_kv(kv_cache: torch.Tensor):
    if kv_cache.shape[0] == 2:
        return (kv_cache[0], kv_cache[1])
    if kv_cache.shape[1] == 1:
        return (kv_cache.select(1, 0), None)
    return (kv_cache.select(1, 0), kv_cache.select(1, 1))


class ResidentVSource:
    SRC_ID = 0

    def __init__(self, kv_cache: torch.Tensor):
        (_, v) = _split_kv(kv_cache)
        if v is None:
            raise ValueError(
                "ResidentVSource requires a resident V half; this cache is K-only (mem variant) -- use a tier VSource instead."
            )
        self._v = v

    def args(self) -> Tuple[torch.Tensor, int, int, int]:
        v = self._v
        return (v, v.stride(0), v.stride(1), v.stride(2))

    def tier_ptrs(self):
        v = self._v
        return (v, v, v, v, 0, 0, 0)


def _adaptive_split(max_reqs: int, max_pages: int, budget: float, hard_cap: int) -> int:
    if int(max_reqs) > 4:
        return hard_cap
    pps_min = _pps_min(int(max_reqs))
    import math as _m

    est_active = _m.ceil(1.5 * float(budget) * int(max_pages) / max(1, pps_min))
    return int(min(hard_cap, max(16, triton.next_power_of_2(max(1, est_active)))))


def ensure_stage_b_buffers(st, device, split: int) -> None:
    if getattr(st, "m_part", None) is not None:
        st.split = getattr(st, "split", split)
        return
    (R, n_kv, G, D) = (st.max_reqs, st.n_kv, st.G, st.D)
    f32 = torch.float32
    split = _adaptive_split(
        R,
        int(getattr(st, "max_pages", 1 << 20)),
        float(getattr(st, "budget", 0.1) or 0.1),
        split,
    )
    if getattr(st, "low_context", False) and R == 1 and n_kv == 8 and G == 4 and D == 128 and st.max_pages <= 4096 and getattr(st, "budget_abs", 0):
        slots = st.budget_abs + st.sink_pages + st.window_pages
        split = min(split, triton.next_power_of_2(max(1, triton.cdiv(slots, 2))))
    st.split = split
    st.m_part = torch.empty(R, n_kv, split, G, device=device, dtype=f32)
    st.l_part = torch.empty(R, n_kv, split, G, device=device, dtype=f32)
    st.acc_part = torch.empty(R, n_kv, split, G, D, device=device, dtype=f32)


_PPS_TP_ANNOUNCED = False


def _pps_min(n_req: int, n_kv: int = 8) -> int:
    from .. import arch as _arch

    if _arch.is_blackwell_sm120():
        return 4 if n_req == 1 else 16
    if n_kv == 1 and 4 < n_req <= 16:
        global _PPS_TP_ANNOUNCED
        if not _PPS_TP_ANNOUNCED:
            _PPS_TP_ANNOUNCED = True
            print(
                f"[locks] decode pps_min=4 (rank-local n_kv=1, n_req={n_req}; tensor-parallel configuration)",
                flush=True,
            )
        return 4
    if n_req == 1:
        return 2
    return 8 if n_req < 16 else 16


@triton.jit
def sparse_decode_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bt_ptr,
    tab_ptr,
    cnt_ptr,
    sl_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    sm_scale,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_vb,
    stride_vt,
    stride_vh,
    hotv_ptr,
    p2s_ptr,
    vbo_ptr,
    poolv_ptr,
    hotk_ptr,
    poolk_ptr,
    stride_btr,
    stride_tabr,
    stride_tabh,
    stride_cntr,
    stride_pr,
    t_lidx,
    t_NB,
    t_S,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
    PPS_MIN: tl.constexpr,
    V_SRC: tl.constexpr,
    K_SRC: tl.constexpr,
    NKV: tl.constexpr,
    NS: tl.constexpr,
):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    cnt = tl.load(cnt_ptr + r * stride_cntr + kh)
    pps = tl.maximum(tl.cdiv(cnt, SPLIT), PPS_MIN)
    j0 = sp * pps
    if j0 >= cnt:
        return
    j1 = tl.minimum(j0 + pps, cnt)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, PAGE)
    gmask = offs_g < G
    seq_len = tl.load(sl_ptr + r)
    q = tl.load(
        q_ptr
        + r * stride_qt
        + (kh * G + offs_g)[:, None] * stride_qh
        + offs_d[None, :],
        mask=gmask[:, None],
        other=0.0,
    )
    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, D], tl.float32)
    for j in tl.range(j0, j1, num_stages=NS):
        pt = tl.load(tab_ptr + r * stride_tabr + kh * stride_tabh + j).to(tl.int64)
        blk = tl.load(bt_ptr + r * stride_btr + pt).to(tl.int64)
        tok = pt * PAGE + offs_t
        tmask = tok < seq_len
        k = tl.load(
            k_ptr
            + blk * stride_kb
            + kh * stride_kh
            + offs_t[:, None] * stride_kt
            + offs_d[None, :],
            mask=tmask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr
            + blk * stride_vb
            + kh * stride_vh
            + offs_t[:, None] * stride_vt
            + offs_d[None, :],
            mask=tmask[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc=acc * alpha[:, None])
        m_i = m_new
    base = r * stride_pr + (kh * SPLIT + sp) * G
    tl.store(m_ptr + base + offs_g, m_i, mask=gmask)
    tl.store(l_ptr + base + offs_g, l_i, mask=gmask)
    tl.store(
        acc_ptr + (base + offs_g)[:, None] * D + offs_d[None, :],
        acc,
        mask=gmask[:, None],
    )


def sparse_paged_decode_batched(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    st,
    out: torch.Tensor,
    vsource: VSource,
    *,
    scale: float = None,
) -> None:
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    G = st.G
    split = st.split
    pps_min = _pps_min(n_req, st.n_kv)
    (v_ptr, svb, svt, svh) = vsource.args()
    (hot_ptr, p2s_ptr, vbo_ptr, pool_ptr, t_lidx, t_NB, t_S) = vsource.tier_ptrs()
    (k_ptr, _) = _split_kv(kv_cache)
    (skb, skt, skh) = (k_ptr.stride(0), k_ptr.stride(1), k_ptr.stride(2))
    (hotk_ptr, poolk_ptr) = (v_ptr, v_ptr)
    (_, page, n_kv, d) = k_ptr.shape
    slots = st.budget_abs + st.sink_pages + st.window_pages if getattr(st, "budget_abs", 0) else 0
    if getattr(st, "low_context", False) and n_req == 1 and n_kv == 8 and G == 4 and d == 128 and page == 16 and st.max_pages <= 4096 and vsource.SRC_ID == 0 and slots in (4, 8, 16):
        from .compact import compact_selected_attention
        compact_selected_attention[(32,)](
            q, k_ptr, v_ptr, block_table, st.page_table, st.page_cnt, seq_lens, out,
            skb, skt, skh, svb, svt, svh, st.page_table.stride(1), True, True,
            slots * 16, float(scale), num_warps=4 if slots <= 8 else 8,
        )
        return
    sparse_decode_split_kernel[n_req, n_kv, split](
        q,
        k_ptr,
        v_ptr,
        block_table,
        st.page_table,
        st.page_cnt,
        seq_lens,
        st.m_part,
        st.l_part,
        st.acc_part,
        scale,
        q.stride(0),
        q.stride(1),
        skb,
        skt,
        skh,
        svb,
        svt,
        svh,
        hot_ptr,
        p2s_ptr,
        vbo_ptr,
        pool_ptr,
        hotk_ptr,
        poolk_ptr,
        block_table.stride(0),
        st.page_table.stride(0),
        st.page_table.stride(1),
        st.page_cnt.stride(0),
        st.m_part.stride(0),
        t_lidx,
        t_NB,
        t_S,
        G=G,
        G_PAD=max(16, triton.next_power_of_2(G)),
        PAGE=page,
        D=d,
        SPLIT=split,
        PPS_MIN=pps_min,
        V_SRC=0,
        K_SRC=0,
        NKV=n_kv,
        NS=_DECODE_NS,
        num_warps=4,
    )
    from .merge import _ORDER2

    merge_splits_kernel[n_req, n_kv * G](
        st.m_part,
        st.l_part,
        st.acc_part,
        st.page_cnt,
        out,
        out.stride(0),
        out.stride(1),
        st.m_part.stride(0),
        st.page_cnt.stride(0),
        G=G,
        D=d,
        SPLIT=split,
        SPLIT_PAD=triton.next_power_of_2(split),
        PPS_MIN=pps_min,
        ORDER2=_ORDER2,
    )
