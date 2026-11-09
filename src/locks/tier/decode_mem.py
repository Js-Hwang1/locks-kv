from __future__ import annotations
from typing import Optional
import torch
import triton
import triton.language as tl
from ..attention.decode import _pps_min
from ..attention.merge import _ORDER2, merge_splits_kernel

_NOROPE = False
_PIPELINE_STAGES = int("4")
_NUM_WARPS = int("4")
_PUBLISH_ON_COPY = True


@triton.jit
def _mem_decode_split_kernel(
    q_ptr,
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
    kres_ptr,
    stride_kb,
    stride_kt,
    stride_kh,
    hotv_ptr,
    hotk_ptr,
    stgv_ptr,
    stgk_ptr,
    svb,
    svt,
    svh,
    poolv_ptr,
    poolk_ptr,
    p2s_ptr,
    vbo_ptr,
    stride_btr,
    stride_tabr,
    stride_tabh,
    stride_cntr,
    stride_pr,
    t_lidx,
    t_NB,
    t_S,
    epoch_ptr,
    clock_ptr,
    valid_ptr,
    gen_ptr,
    mail_ptr,
    mail_cnt_ptr,
    ovf_ptr,
    t_MCAP,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
    PPS_MIN: tl.constexpr,
    K_TIER: tl.constexpr,
    NKV: tl.constexpr,
    K5C: tl.constexpr,
    LOAD_CACHE: tl.constexpr = "",
    PIPELINE_STAGES: tl.constexpr = 3,
    PUBLISH_COPY: tl.constexpr = False,
    PREFETCH_META: tl.constexpr = False,
    META_PAD: tl.constexpr = 16,
    FILL_CACHE: tl.constexpr = False,
    CACHE_SLOT_BITS: tl.constexpr = 0,
    CACHE_DISPATCH: tl.constexpr = 0,
):
    r = tl.program_id(0)
    kh = tl.program_id(1)
    sp = tl.program_id(2)
    if CACHE_DISPATCH:
        # Separate kernels keep cached heads on the original fast path.
        # GPU counters select the path without a host synchronization.
        misses = tl.load(mail_cnt_ptr + kh)
        if CACHE_DISPATCH == 1:
            if misses > 0:
                return
        else:
            if misses == 0:
                return
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
    if K5C:
        k5_clk = tl.load(clock_ptr)
    if PREFETCH_META:
        tl.static_assert(True)
        mj = j0 + tl.arange(0, META_PAD)
        valid_meta = mj < j1
        meta_pt = tl.load(
            tab_ptr + r * stride_tabr + kh * stride_tabh + mj, mask=valid_meta, other=0
        )
        meta_blk = tl.load(bt_ptr + r * stride_btr + meta_pt, mask=valid_meta, other=0)
        if PUBLISH_COPY:
            meta_slot = tl.atomic_add(
                p2s_ptr + kh * t_NB + meta_blk, 0, mask=valid_meta, sem="acquire"
            )
        else:
            meta_slot = tl.load(
                p2s_ptr + kh * t_NB + meta_blk, mask=valid_meta, other=-1
            )
        meta_vb = tl.load(vbo_ptr + meta_blk, mask=valid_meta, other=-1)
    for j in tl.range(j0, j1, num_stages=PIPELINE_STAGES):
        if PREFETCH_META:
            mi = tl.full((1,), j - j0, tl.int32)
            pt32 = tl.reshape(tl.gather(meta_pt, mi, axis=0), ())
        else:
            pt32 = tl.load(tab_ptr + r * stride_tabr + kh * stride_tabh + j)
        pt = pt32.to(tl.int64)
        if PREFETCH_META:
            blk32 = tl.reshape(tl.gather(meta_blk, mi, axis=0), ())
            blk = blk32.to(tl.int64)
        else:
            blk32 = tl.load(bt_ptr + r * stride_btr + pt)
            blk = blk32.to(tl.int64)
        tok = pt * PAGE + offs_t
        tmask = tok < seq_len
        complete = (pt + 1) * PAGE <= seq_len
        if PREFETCH_META:
            slot = tl.reshape(tl.gather(meta_slot, mi, axis=0), ())
            vb = tl.reshape(tl.gather(meta_vb, mi, axis=0), ())
        else:
            if PUBLISH_COPY:
                slot = tl.atomic_add(p2s_ptr + kh * t_NB + blk, 0, sem="acquire")
            else:
                slot = tl.load(p2s_ptr + kh * t_NB + blk)
            vb = tl.load(vbo_ptr + blk)
        use_hot = complete & (slot >= 0)
        use_stg = (use_hot == 0) & (vb >= 0)
        if K5C:
            if use_hot:
                tl.store(epoch_ptr + kh * t_S + slot, k5_clk)
            if complete & (slot == -1):
                oldp = tl.atomic_cas(p2s_ptr + kh * t_NB + blk, -1, -2)
                if oldp == -1:
                    vld = tl.load(valid_ptr + kh * t_NB + blk)
                    if (vb < 0) & (vld == 0):
                        tl.store(p2s_ptr + kh * t_NB + blk, -1)
                    elif vb >= 0:
                        tl.store(p2s_ptr + kh * t_NB + blk, -1)
                    else:
                        pos = tl.atomic_add(mail_cnt_ptr, 1, sem="relaxed")
                        if pos < t_MCAP:
                            gval = tl.load(gen_ptr + blk)
                            tl.store(mail_ptr + pos * 6 + 0, t_lidx)
                            tl.store(mail_ptr + pos * 6 + 1, kh)
                            tl.store(mail_ptr + pos * 6 + 2, blk32)
                            tl.store(mail_ptr + pos * 6 + 3, -1)
                            tl.store(mail_ptr + pos * 6 + 4, gval)
                        else:
                            tl.store(p2s_ptr + kh * t_NB + blk, -1)
                            tl.store(ovf_ptr, 1)
        cache_slot = tl.maximum(slot, 0)
        if FILL_CACHE:
            encoded = -slot - 3
            cache_slot = tl.where(
                slot < -2, encoded & ((1 << CACHE_SLOT_BITS) - 1), cache_slot
            )
            copy_page = (slot < -2) & ((encoded >> CACHE_SLOT_BITS) == r)
        hot_off = (
            (kh * t_S + cache_slot).to(tl.int64) * (PAGE * D)
            + offs_t[:, None] * D
            + offs_d[None, :]
        )
        stg_off = (
            tl.maximum(vb, 0).to(tl.int64) * svb
            + kh * svh
            + offs_t[:, None] * svt
            + offs_d[None, :]
        )
        pin_off = (
            ((t_lidx * t_NB + blk) * NKV + kh).to(tl.int64) * (PAGE * D)
            + offs_t[:, None] * D
            + offs_d[None, :]
        )
        if K_TIER == 0:
            k = tl.load(
                kres_ptr
                + blk * stride_kb
                + kh * stride_kh
                + offs_t[:, None] * stride_kt
                + offs_d[None, :],
                mask=tmask[:, None],
                other=0.0,
            )
        else:
            kp = tl.where(
                use_hot,
                hotk_ptr + hot_off,
                tl.where(use_stg, stgk_ptr + stg_off, poolk_ptr + pin_off),
            )
            k = tl.load(kp, mask=tmask[:, None], other=0, cache_modifier=LOAD_CACHE).to(
                tl.bfloat16, bitcast=True
            )
        vp = tl.where(
            use_hot,
            hotv_ptr + hot_off,
            tl.where(use_stg, stgv_ptr + stg_off, poolv_ptr + pin_off),
        )
        v = tl.load(vp, mask=tmask[:, None], other=0, cache_modifier=LOAD_CACHE).to(
            tl.bfloat16, bitcast=True
        )
        if FILL_CACHE:
            # The owning request stores the exact KV it already loaded for
            # attention. Publication happens after this entire kernel ends.
            if copy_page:
                tl.store(hotk_ptr + hot_off, k.to(tl.int16, bitcast=True), mask=tmask[:, None])
                tl.store(hotv_ptr + hot_off, v.to(tl.int16, bitcast=True), mask=tmask[:, None])
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


def mem_decode_ptrsel(
    q: torch.Tensor,
    kv_cache: Optional[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    st,
    out: torch.Tensor,
    tier,
    lidx: int,
    *,
    scale: float = None,
    k5c: bool = False,
    fused_fill: bool = False,
) -> None:
    if scale is None:
        scale = st.scale
    n_req = seq_lens.shape[0]
    G = st.G
    split = st.split
    pps_min = _pps_min(n_req)
    r5 = tier.residency
    stg = tier.staging
    vp16 = stg.v_pool[lidx].view(torch.int16)
    has_k = tier.offload_k
    if has_k:
        kres = vp16
        skb = skt = skh = 0
        kp16 = stg.k_pool[lidx].view(torch.int16)
        hotk = r5.hot_k_i16[lidx]
        poolk = tier.pool_k.dev_view
        n_kv = r5.n_kv
    else:
        from ..attention.decode import _split_kv

        (kres, _) = _split_kv(kv_cache)
        (skb, skt, skh) = (kres.stride(0), kres.stride(1), kres.stride(2))
        n_kv = kres.shape[2]
        kp16 = vp16
        hotk = r5.hot_i16[lidx]
        poolk = tier.pool.dev_view
    d = r5.d
    if k5c:
        from .k5 import _MCAP

        k5 = tier._k5_state()
        (_epoch, _clock) = (r5.lru_clock[lidx], r5.clock)
        (_valid, _gen) = (r5.pool_valid[lidx], k5.gen)
        (_mail, _mcnt, _ovf, _mcap) = (k5.mail, k5.mail_cnt, r5.overflow, _MCAP)
    else:
        (_epoch, _clock) = (r5.clock, r5.clock)
        (_valid, _gen) = (r5.clock, r5.clock)
        (_mail, _mcnt, _ovf, _mcap) = (r5.clock, r5.clock, r5.clock, 0)
    if fused_fill:
        # The asynchronous K5 mailbox is inactive for PCIe fills; reuse this
        # kernel argument for the per-head miss counts instead.
        _mcnt = tier.pcie_cache.counts
    max_selected = st.page_table.shape[2]
    if getattr(st, "budget_abs", 0) > 0:
        max_selected = min(
            max_selected, st.budget_abs + st.sink_pages + st.window_pages
        )
    # Keep the scratch strides and reduction order, but do not launch blocks
    # that cannot own any selected pages under the configured budget.
    grid_split = min(split, triton.cdiv(max_selected, pps_min))
    for fill_mode in ((1, 2) if fused_fill else (0,)):
        _mem_decode_split_kernel[n_req, n_kv, grid_split](
            q,
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
            kres,
            skb,
            skt,
            skh,
            r5.hot_i16[lidx],
            hotk,
            vp16,
            kp16,
            vp16.stride(0),
            vp16.stride(1),
            vp16.stride(2),
            tier.pool.dev_view,
            poolk,
            r5.page2slot[lidx],
            stg.vbo,
            block_table.stride(0),
            st.page_table.stride(0),
            st.page_table.stride(1),
            st.page_cnt.stride(0),
            st.m_part.stride(0),
            lidx,
            r5.NB,
            r5.S,
            _epoch,
            _clock,
            _valid,
            _gen,
            _mail,
            _mcnt,
            _ovf,
            _mcap,
            G=G,
            G_PAD=max(16, triton.next_power_of_2(G)),
            PAGE=r5.page,
            D=d,
            SPLIT=split,
            PPS_MIN=pps_min,
            K_TIER=1 if has_k else 0,
            NKV=n_kv,
            K5C=bool(k5c),
            LOAD_CACHE="",
            PIPELINE_STAGES=1 if fill_mode == 2 else 4,
            PUBLISH_COPY=not fused_fill,
            FILL_CACHE=fill_mode == 2,
            CACHE_DISPATCH=fill_mode,
            CACHE_SLOT_BITS=(r5.S - 1).bit_length(),
            PREFETCH_META=True,
            META_PAD=triton.next_power_of_2(max(pps_min, triton.cdiv(max_selected, split))),
            launch_pdl=bool(False),
            num_warps=4,
        )
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
        PDL=bool(False),
        ORDER2=_ORDER2,
        PW=False,
        launch_pdl=bool(False),
    )


def mem_dynamic_decode(
    q,
    kv_cache,
    block_table,
    seq_lens,
    st,
    out,
    tier,
    lidx,
    *,
    scale=None,
    fetch="kernel",
    impl="ptrsel",
    tier_stepped=False,
):
    if fetch != "kernel" or impl not in ("auto", "ptrsel"):
        raise ValueError(
            "Only the optimized pointer-selection attention path is available"
        )
    if tier.pcie_cache is not None:
        tier.pcie_cache.prepare(lidx, st.page_table, st.page_cnt, block_table, seq_lens)
        mem_decode_ptrsel(q, kv_cache, block_table, seq_lens, st, out, tier, lidx,
                          scale=scale, k5c=False, fused_fill=tier.pcie_cache.fused)
        tier.pcie_cache.finish(lidx)
        return
    from .k5 import _K5_STATS

    fused = seq_lens.shape[0] == 1 and (not _K5_STATS) and (not tier_stepped)
    forked = False
    if fused:
        if lidx == 0:
            tier.k8_head()
    elif not tier_stepped:
        tier.side_step(
            lidx, st.page_table, st.page_cnt, block_table, seq_lens, seq_lens.shape[0]
        )
        forked = True
    mem_decode_ptrsel(
        q, kv_cache, block_table, seq_lens, st, out, tier, lidx, scale=scale, k5c=fused
    )
    if forked:
        tier.side_join(lidx)
