from __future__ import annotations
import os
import torch
import triton
import triton.language as tl

_K5_STATS = os.environ.get("LOCKS_K5_STATS") not in (None, "", "0") or os.environ.get(
    "LOCKS_TIER_ASSERT"
) not in (None, "", "0")
_MCAP = 65536
_NEVER = tl.constexpr(2147483647)
_V7B_ANNOUNCED = False
_K7C_ANNOUNCED = False


def _mcap_for(max_reqs: int, n_layers: int, n_kv: int) -> int:
    if max_reqs <= 1 or False:
        return 65536
    need = max_reqs * n_layers * n_kv * 128
    cap = 65536
    while cap < need:
        cap *= 2
    return min(cap, 2**21)


_K5_PROMO1 = True
_PUBLISH_ON_COPY = True
_PROMO1_ANNOUNCED = False


@triton.jit
def _masked_cas(ptr, compare, value, active):
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; mov.u32 $0, 0; @p atom.acq_rel.gpu.global.cas.b32 $0, [$1], $2, $3; }",
        constraints="=r,l,r,r,r",
        args=[ptr, compare, value, active.to(tl.int32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def _bcopy(src, dst, cnt, entw: int = 6, cap: int | None = None):
    from .k5_cuda import get_mod

    mod = get_mod()
    mod.mail_bcopy(
        src.view(-1), dst.view(-1), cnt, 65536 if cap is None else int(cap), entw
    )
    return


@triton.jit
def _k5c_kernel(
    tab_ptr,
    cnt_ptr,
    bt_ptr,
    sl_ptr,
    p2s_ptr,
    epoch_ptr,
    clock_ptr,
    vbo_ptr,
    valid_ptr,
    scratch_ptr,
    gen_ptr,
    mail_ptr,
    mail_cnt_ptr,
    ovf_ptr,
    stats_ptr,
    lidx,
    NB,
    S,
    MP,
    MCAP,
    stride_btr,
    stride_tr,
    stride_cr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
    STATS: tl.constexpr,
):
    r = tl.program_id(0)
    kv = tl.program_id(1)
    n = tl.load(cnt_ptr + r * stride_cr + kv)
    sl = tl.load(sl_ptr + r)
    clk = tl.load(clock_ptr)
    n_sel = clk * 0
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        valid = idx < n
        pt = tl.load(tab_ptr + r * stride_tr + kv * MP + idx, mask=valid, other=0)
        blk = tl.load(bt_ptr + r * stride_btr + pt, mask=valid, other=0)
        complete = (pt + 1) * PAGE <= sl
        act = valid & complete
        n_sel += tl.sum(act.to(tl.int32), axis=0)
        slot = tl.load(p2s_ptr + kv * NB + blk, mask=valid, other=-1)
        is_hit = act & (slot >= 0)
        tl.store(epoch_ptr + kv * S + slot, clk, mask=is_hit)
        cand = act & (slot == -1)
        cas_ptr = tl.where(cand, p2s_ptr + kv * NB + blk, scratch_ptr)
        cas_cmp = tl.where(cand, -1, _NEVER)
        old = _masked_cas(cas_ptr, cas_cmp, tl.full([BLOCK], -2, tl.int32), cand)
        won = cand & (old == -1)
        vb = tl.load(vbo_ptr + blk, mask=won, other=-1)
        vld = tl.load(valid_ptr + kv * NB + blk, mask=won, other=0)
        lost_v = won & (vb < 0) & (vld == 0)
        tl.store(p2s_ptr + kv * NB + blk, tl.full([BLOCK], -1, tl.int32), mask=lost_v)
        won = won & (lost_v == 0)
        staged = won & (vb >= 0)
        tl.store(p2s_ptr + kv * NB + blk, tl.full([BLOCK], -1, tl.int32), mask=staged)
        pin = won & (vb < 0)
        m32 = pin.to(tl.int32)
        n_pin = tl.sum(m32, axis=0)
        base = tl.atomic_add(mail_cnt_ptr, n_pin, sem="relaxed")
        pos = base + tl.cumsum(m32, axis=0) - m32
        ok = pin & (pos < MCAP)
        g = tl.load(gen_ptr + blk, mask=ok, other=0)
        tl.store(mail_ptr + pos * 6 + 0, tl.full([BLOCK], 0, tl.int32) + lidx, mask=ok)
        tl.store(mail_ptr + pos * 6 + 1, tl.full([BLOCK], 0, tl.int32) + kv, mask=ok)
        tl.store(mail_ptr + pos * 6 + 2, blk, mask=ok)
        tl.store(mail_ptr + pos * 6 + 3, tl.full([BLOCK], -1, tl.int32), mask=ok)
        tl.store(mail_ptr + pos * 6 + 4, g, mask=ok)
        over = pin & (pos >= MCAP)
        n_over = tl.sum(over.to(tl.int32), axis=0)
        if n_over > 0:
            tl.store(p2s_ptr + kv * NB + blk, tl.full([BLOCK], -1, tl.int32), mask=over)
            tl.store(ovf_ptr, 1)
        if STATS:
            n_lost = tl.sum(lost_v.to(tl.int32), axis=0)
            if n_lost > 0:
                tl.atomic_add(stats_ptr + 4, n_lost)
            tl.atomic_add(stats_ptr + 1, n_pin - n_over)
            tl.atomic_add(stats_ptr + 2, tl.sum(staged.to(tl.int32), axis=0))
            if n_over > 0:
                tl.atomic_add(stats_ptr + 3, n_over)
    if STATS:
        tl.atomic_add(stats_ptr + 0, n_sel)


@triton.jit
def _k5_assign_kernel(
    mail_ptr,
    mcnt_ptr,
    p2s_ptr,
    epoch_ptr,
    owner_ptr,
    hand_ptr,
    clock_ptr,
    scratch_ptr,
    ovf_ptr,
    stats_ptr,
    bcnt_ptr,
    bidx_ptr,
    CAP,
    fscr_ptr,
    NB,
    S,
    L,
    NKV: tl.constexpr,
    BLOCK: tl.constexpr,
    STATS: tl.constexpr,
    BUCKETED: tl.constexpr = False,
    WIDE: tl.constexpr = False,
):
    l = tl.program_id(0)
    kv = tl.program_id(1)
    n = tl.load(mcnt_ptr)
    clk = tl.load(clock_ptr)
    unit = l * NKV + kv
    tag = kv + 1
    cur = tl.load(hand_ptr + unit)
    spent = clk * 0
    n_asn = clk * 0
    if BUCKETED:
        n = tl.load(bcnt_ptr + unit)
    for off in range(0, n, BLOCK):
        pos = off + tl.arange(0, BLOCK)
        inb = pos < n
        if BUCKETED:
            idx = tl.load(bidx_ptr + unit * CAP + pos, mask=inb, other=0)
            es = tl.load(mail_ptr + idx * 6 + 3, mask=inb, other=0)
            need = inb & (es == -1)
        else:
            idx = pos
            el = tl.load(mail_ptr + idx * 6 + 0, mask=inb, other=-1)
            ek = tl.load(mail_ptr + idx * 6 + 1, mask=inb, other=-1)
            es = tl.load(mail_ptr + idx * 6 + 3, mask=inb, other=0)
            need = inb & (el == l) & (ek == kv) & (es == -1)
        n_need = tl.sum(need.to(tl.int32), axis=0)
        if WIDE:
            while (n_need > 0) & (spent < 2 * S):
                lane = tl.arange(0, BLOCK)
                c = (cur + lane) % S
                ep = tl.load(epoch_ptr + unit * S + c)
                own = tl.load(owner_ptr + unit * S + c)
                isf = (ep <= clk - L) & (own == 0)
                f32 = isf.to(tl.int32)
                rkf = tl.cumsum(f32, axis=0) - f32
                nfree = tl.sum(f32, axis=0)
                tl.store(fscr_ptr + unit * BLOCK + rkf, c, mask=isf)
                tl.debug_barrier()
                m32 = need.to(tl.int32)
                rkn = tl.cumsum(m32, axis=0) - m32
                take = need & (rkn < nfree)
                cs = tl.load(fscr_ptr + unit * BLOCK + rkn, mask=take, other=0)
                cas_ptr = tl.where(take, owner_ptr + unit * S + cs, scratch_ptr)
                cas_cmp = tl.where(take, 0, _NEVER)
                old = _masked_cas(
                    cas_ptr, cas_cmp, tl.zeros([BLOCK], tl.int32) + tag, take
                )
                won = take & (old == 0)
                tl.store(mail_ptr + idx * 6 + 3, cs, mask=won)
                tl.store(epoch_ptr + unit * S + cs, clk, mask=won)
                n_asn += tl.sum(won.to(tl.int32), axis=0)
                need = need & (won == 0)
                n_need = tl.sum(need.to(tl.int32), axis=0)
                cur += BLOCK
                spent += BLOCK
                tl.debug_barrier()
        else:
            while (n_need > 0) & (spent < S):
                m32 = need.to(tl.int32)
                rk = tl.cumsum(m32, axis=0) - m32
                issued = tl.minimum(n_need, S - spent)
                live = need & (rk < issued)
                c = (cur + rk) % S
                ep = tl.load(epoch_ptr + unit * S + c, mask=live, other=0)
                own = tl.load(owner_ptr + unit * S + c, mask=live, other=1)
                free = live & (ep <= clk - L) & (own == 0)
                cas_ptr = tl.where(free, owner_ptr + unit * S + c, scratch_ptr)
                cas_cmp = tl.where(free, 0, _NEVER)
                old = _masked_cas(
                    cas_ptr, cas_cmp, tl.zeros([BLOCK], tl.int32) + tag, free
                )
                won = free & (old == 0)
                tl.store(mail_ptr + idx * 6 + 3, c, mask=won)
                tl.store(epoch_ptr + unit * S + c, clk, mask=won)
                n_asn += tl.sum(won.to(tl.int32), axis=0)
                need = live & (won == 0) | need & (rk >= issued)
                n_need = tl.sum(need.to(tl.int32), axis=0)
                cur += issued
                spent += issued
        n_left = tl.sum(need.to(tl.int32), axis=0)
        if n_left > 0:
            blkl = tl.load(mail_ptr + idx * 6 + 2, mask=need, other=0)
            tl.store(
                p2s_ptr + unit * NB + blkl, tl.full([BLOCK], -1, tl.int32), mask=need
            )
            tl.store(ovf_ptr, 1)
            if STATS:
                tl.atomic_add(stats_ptr + 3, n_left)
    tl.store(hand_ptr + unit, cur)
    if STATS:
        tl.atomic_add(stats_ptr + 5, n_asn)


@triton.jit
def _k7c_bucketize_kernel(
    snap_ptr,
    scnt_ptr,
    p2s_ptr,
    bcnt_ptr,
    bidx_ptr,
    ovf_ptr,
    CAP,
    NB,
    NKV: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.load(scnt_ptr)
    one = tl.full([BLOCK], 1, tl.int32)
    for i0 in range(pid * BLOCK, n, GRID * BLOCK):
        idx = i0 + tl.arange(0, BLOCK)
        inb = idx < n
        el = tl.load(snap_ptr + idx * 6 + 0, mask=inb, other=0)
        ek = tl.load(snap_ptr + idx * 6 + 1, mask=inb, other=0)
        es = tl.load(snap_ptr + idx * 6 + 3, mask=inb, other=0)
        pend = inb & (es == -1)
        unit = el * NKV + ek
        pos = tl.atomic_add(bcnt_ptr + unit, one, mask=pend)
        ok = pend & (pos < CAP)
        tl.store(bidx_ptr + unit * CAP + pos, idx, mask=ok)
        over = pend & (pos >= CAP)
        n_over = tl.sum(over.to(tl.int32), axis=0)
        if n_over > 0:
            blk = tl.load(snap_ptr + idx * 6 + 2, mask=over, other=0)
            tl.store(
                p2s_ptr + unit * NB + blk, tl.full([BLOCK], -1, tl.int32), mask=over
            )
            tl.store(ovf_ptr, 1)


@triton.jit
def _k5_evict_kernel(
    elist_ptr, ecnt_ptr, p2s_ptr, s2p_ptr, NB, S, NKV: tl.constexpr, GRID: tl.constexpr
):
    pid = tl.program_id(0)
    n = tl.load(ecnt_ptr)
    for i in range(pid, n, GRID):
        h = tl.load(elist_ptr + i * 6 + 3)
        if h >= 0:
            l = tl.load(elist_ptr + i * 6 + 0)
            kv = tl.load(elist_ptr + i * 6 + 1)
            blk = tl.load(elist_ptr + i * 6 + 2)
            base = (l * NKV + kv).to(tl.int64)
            old = tl.load(s2p_ptr + base * S + h)
            if old >= 0:
                if old != blk:
                    tl.atomic_cas(p2s_ptr + base * NB + old, h, -1)
            tl.store(s2p_ptr + base * S + h, blk)


@triton.jit
def _k5_promote_kernel(
    plist_ptr,
    pcnt_ptr,
    p2s_ptr,
    s2p_ptr,
    owner_ptr,
    gen_ptr,
    NB,
    S,
    NKV: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.load(pcnt_ptr)
    for i in range(pid, n, GRID):
        h = tl.load(plist_ptr + i * 6 + 3)
        if h >= 0:
            l = tl.load(plist_ptr + i * 6 + 0)
            kv = tl.load(plist_ptr + i * 6 + 1)
            blk = tl.load(plist_ptr + i * 6 + 2)
            g = tl.load(plist_ptr + i * 6 + 4)
            base = (l * NKV + kv).to(tl.int64)
            cur_g = tl.load(gen_ptr + blk)
            installed = 0
            if cur_g == g:
                old = tl.atomic_cas(p2s_ptr + base * NB + blk, -2, h)
                if (old == -2) | (old == h):
                    installed = 1
            if installed == 0:
                tl.store(s2p_ptr + base * S + h, -1)
            tl.store(owner_ptr + base * S + h, 0)


def k5_headv(res, k5, tick: bool) -> None:
    from .k5_cuda import get_mod

    mod = get_mod()
    mod.k5_headv(
        k5.elist_dev,
        k5.ecnt_dev,
        k5.plist_dev,
        k5.pcnt_dev,
        res.page2slot,
        res.slot2page,
        k5.owner,
        k5.gen,
        res.clock,
        res.fetch_cnt,
        res.NB,
        res.S,
        res.n_kv,
        bool(tick),
    )
    return


class K5State:
    def __init__(self, residency, device):
        r = residency
        i32 = torch.int32
        dev = torch.device(device)
        self.owner = r.owner
        self.hand = r.hand
        self.gen = r.gen
        self.msrc = torch.zeros((r.R, r.n_kv, r.MP), dtype=torch.int8, device=dev)
        L = r.page2slot.shape[0]
        self.mcap = _mcap_for(r.R, L, r.n_kv)
        self.mail = torch.zeros((self.mcap, 6), dtype=i32, device=dev)
        self.mail_cnt = torch.zeros((1,), dtype=i32, device=dev)
        self.elist_dev = torch.zeros((self.mcap, 6), dtype=i32, device=dev)
        self.ecnt_dev = torch.zeros((1,), dtype=i32, device=dev)
        self.scratch = torch.zeros((1,), dtype=i32, device=dev)
        self.zcnt = torch.zeros((1,), dtype=i32, device=dev)
        self.stats = torch.zeros((8,), dtype=i32, device=dev)
        self.plist_host = torch.zeros((self.mcap, 6), dtype=i32, pin_memory=True)
        self.plist_dev = torch.zeros((self.mcap, 6), dtype=i32, device=dev)
        self.pcnt_dev = torch.zeros((1,), dtype=i32, device=dev)
        self.snap_mail = [
            torch.zeros((self.mcap, 6), dtype=i32, device=dev) for _ in range(2)
        ]
        self.snap_cnt = [torch.zeros((1,), dtype=i32, device=dev) for _ in range(2)]
        self.bkt_cap = max(2048, 128 * r.R)
        self.bkt_cnt = torch.zeros((L * r.n_kv,), dtype=i32, device=dev)
        self.bkt_idx = torch.zeros((L * r.n_kv * self.bkt_cap,), dtype=i32, device=dev)
        self.fscr = torch.zeros((L * r.n_kv * 256,), dtype=i32, device=dev)
        self.bytes_gib = (
            self.owner.numel() * 4
            + self.hand.numel() * 4
            + self.gen.numel() * 4
            + self.msrc.numel()
            + self.mail.numel() * 4
            + self.stats.numel() * 4
        ) / 2**30


def k5_step(
    res, k5, staging, lidx, page_table, page_cnt, block_table, seq_lens, n_req
) -> None:
    from .residency import _step_setup_kernel

    (n_kv, S, NB, MP) = (res.n_kv, res.S, res.NB, res.MP)
    _step_setup_kernel[1,](res.clock, res.fetch_cnt, num_warps=1)
    if n_req > 1:
        global _V7B_ANNOUNCED
        if not _V7B_ANNOUNCED:
            _V7B_ANNOUNCED = True
            print(
                f"[locks] K7B ACTIVE: batch k5c protocol (n_req={n_req}, per-step assign on the service stream)",
                flush=True,
            )
    _k5c_kernel[n_req, n_kv](
        page_table,
        page_cnt,
        block_table,
        seq_lens,
        res.page2slot[lidx],
        res.lru_clock[lidx],
        res.clock,
        staging.vbo,
        res.pool_valid[lidx],
        k5.scratch,
        k5.gen,
        k5.mail,
        k5.mail_cnt,
        res.overflow,
        k5.stats,
        lidx,
        NB,
        S,
        MP,
        getattr(k5, "mcap", 65536),
        block_table.stride(0),
        page_table.stride(0),
        page_cnt.stride(0),
        PAGE=res.page,
        BLOCK=256,
        STATS=_K5_STATS,
        num_warps=4,
    )
    return


_TR_GRID = int("264")
_TR_WARPS = int("4")


@triton.jit
def _k5_transport_kernel(
    mail_ptr,
    mcnt_ptr,
    poolv_ptr,
    poolk_ptr,
    hotv_ptr,
    hotk_ptr,
    NB,
    S,
    stride_hl,
    GRID: tl.constexpr,
    NKV: tl.constexpr,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    HAS_K: tl.constexpr,
    p2s_ptr=None,
    gen_ptr=None,
    PUBLISH: tl.constexpr = False,
):
    pid = tl.program_id(0)
    n = tl.load(mcnt_ptr)
    offs_t = tl.arange(0, PAGE)
    offs_d = tl.arange(0, D)
    tile = offs_t[:, None] * D + offs_d[None, :]
    for i in range(pid, n, GRID):
        s32 = tl.load(mail_ptr + i * 6 + 3)
        if s32 >= 0:
            l = tl.load(mail_ptr + i * 6 + 0).to(tl.int64)
            kv = tl.load(mail_ptr + i * 6 + 1).to(tl.int64)
            blk = tl.load(mail_ptr + i * 6 + 2).to(tl.int64)
            slot = s32.to(tl.int64)
            page_off = ((l * NB + blk) * NKV + kv) * (PAGE * D) + tile
            hot_off = l * stride_hl + (kv * S + slot) * (PAGE * D) + tile
            tl.store(hotv_ptr + hot_off, tl.load(poolv_ptr + page_off))
            if HAS_K:
                tl.store(hotk_ptr + hot_off, tl.load(poolk_ptr + page_off))
            if PUBLISH:
                tl.debug_barrier()
                expected_gen = tl.load(mail_ptr + i * 6 + 4)
                current_gen = tl.load(gen_ptr + blk)
                if current_gen == expected_gen:
                    tl.atomic_cas(
                        p2s_ptr + (l * NKV + kv) * NB + blk, -2, s32, sem="release"
                    )


class K5Service:
    def __init__(self, tier, k5):
        self._tier = tier
        self._k5 = k5
        self._stream = torch.cuda.Stream(device=tier.device)
        self._inflight = []
        self._par = 0
        self._pub_pending = False
        self._elist_batch = None
        self.sel = 0
        self.pinned_miss = 0
        self.staged_miss = 0
        self.dropped = 0
        self.lost_v = 0
        self.assigned = 0
        self.ce_bytes = 0
        self.fetch_us = 0.0
        self.steps = 0
        self._tile_bytes = (
            tier.page * tier.d * 2 * (2 if tier.pool_k is not None else 1)
        )
        self._timing = []

    def service(self, defer_zev: bool = False) -> None:
        (t, k5) = (self._tier, self._k5)
        res = t.residency
        self.steps += 1
        k5.ecnt_dev.zero_()
        published = False
        _prologue_owns = int(res.R) > 1
        if not _prologue_owns:
            still = []
            for fence_ev, b in self._inflight:
                if not published and fence_ev.query():
                    _bcopy(k5.snap_mail[b], k5.plist_dev, k5.snap_cnt[b], cap=k5.mcap)
                    k5.pcnt_dev.copy_(k5.snap_cnt[b], non_blocking=True)
                    published = True
                    self._pub_pending = True
                else:
                    still.append((fence_ev, b))
            self._inflight = still
        if not published:
            k5.pcnt_dev.zero_()
        if _K5_STATS:
            stats = k5.stats.cpu()
            d_sel = int(stats[0])
            d_pin = int(stats[1])
            d_stg = int(stats[2])
            d_drop = int(stats[3])
            d_lost = int(stats[4])
            d_asn = int(stats[5])
            self.sel += d_sel
            self.pinned_miss += d_pin
            self.staged_miss += d_stg
            self.dropped += d_drop
            self.lost_v += d_lost
            self.assigned += d_asn
            self.ce_bytes += d_asn * self._tile_bytes
            k5.stats.zero_()
            units = max(1, t.L * t.n_kv)
            occ = int((res.page2slot >= 0).sum())
            clk_v = int(res.clock.item())
            win = 1 if int(res.R) == 1 else t.L
            ep = res.lru_clock.view(-1, res.S)
            ow = k5.owner.view(-1, res.S)
            n_fresh = int((ep > clk_v - win).sum())
            n_owned = int((ow != 0).sum())
            n_free = int(((ep <= clk_v - win) & (ow == 0)).sum())
            n_ow_stale = int(((ow != 0) & (ep <= clk_v - 4 * win)).sum())
            print(
                f"[k5census] step={self.steps} clk={clk_v} win={win} slots={ep.numel()} fresh={n_fresh} owned={n_owned} ow_stale={n_ow_stale} free={n_free} inflight={len(self._inflight)} pubpend={int(self._pub_pending)}",
                flush=True,
            )
            print(
                f"[k5stats] step={self.steps} L={t.L} nkv={t.n_kv} units={units} S={res.S} NB={t.NB} sel={d_sel} sel/u={d_sel / units:.2f} pinned_miss={d_pin} staged_miss={d_stg} asn={d_asn} miss/u={(d_pin + d_stg) / units:.3f} pin/u={d_pin / units:.3f} stg/u={d_stg / units:.3f} dropped={d_drop} drop/u={d_drop / units:.3f} lost_v={d_lost} resident={occ} res/u={occ / units:.1f} fill={occ / (units * max(1, res.S)):.3f}",
                flush=True,
            )
            if self.steps % 16 == 0:
                print(self.stats_line(), flush=True)
            still_t = []
            for t0, fence in self._timing:
                if fence.query():
                    self.fetch_us += t0.elapsed_time(fence) * 1000.0
                else:
                    still_t.append((t0, fence))
            self._timing = still_t
        b = self._par
        busy = any((bi == b for (_, bi) in self._inflight))
        if busy:
            return
        self._par ^= 1
        has_k = t.pool_k is not None
        poolv = t.pool.dev_view
        poolk = t.pool_k.dev_view if has_k else poolv
        hotv = res.hot_i16
        hotk = res.hot_k_i16 if has_k else hotv
        main = torch.cuda.current_stream(device=t.device)
        gate = torch.cuda.Event()
        gate.record(main)
        tr = self._stream
        tr.wait_event(gate)
        with torch.cuda.stream(tr):
            _bcopy(k5.mail, k5.snap_mail[b], k5.mail_cnt, cap=k5.mcap)
            torch.clamp(k5.mail_cnt, max=k5.mcap, out=k5.snap_cnt[b])
            _nreq = int(res.R)
            use_k7c = _nreq > 1
            if use_k7c:
                global _K7C_ANNOUNCED
                if not _K7C_ANNOUNCED:
                    _K7C_ANNOUNCED = True
                    print(
                        f"[locks] K7C ACTIVE: batch service capacity (mcap={k5.mcap} bkt_cap={k5.bkt_cap} wide-draw assign, n_req={_nreq})",
                        flush=True,
                    )
                k5.bkt_cnt.zero_()
                _k7c_bucketize_kernel[64,](
                    k5.snap_mail[b],
                    k5.snap_cnt[b],
                    res.page2slot,
                    k5.bkt_cnt,
                    k5.bkt_idx,
                    res.overflow,
                    k5.bkt_cap,
                    t.NB,
                    NKV=t.n_kv,
                    BLOCK=256,
                    GRID=64,
                    num_warps=4,
                )
            _k5_assign_kernel[t.L, t.n_kv](
                k5.snap_mail[b],
                k5.snap_cnt[b],
                res.page2slot,
                res.lru_clock,
                k5.owner,
                k5.hand,
                res.clock,
                k5.scratch,
                res.overflow,
                k5.stats,
                k5.bkt_cnt,
                k5.bkt_idx,
                k5.bkt_cap if use_k7c else 2048,
                k5.fscr,
                t.NB,
                res.S,
                1 if _nreq == 1 else t.L,
                NKV=t.n_kv,
                BLOCK=256,
                STATS=_K5_STATS,
                BUCKETED=use_k7c,
                WIDE=use_k7c,
                num_warps=4,
            )
            if _prologue_owns:
                from .k5_cuda import get_mod

                mod = get_mod()
                if mod is not None:
                    mod.k5_headv(
                        k5.snap_mail[b],
                        k5.snap_cnt[b],
                        k5.snap_mail[b],
                        k5.zcnt,
                        res.page2slot,
                        res.slot2page,
                        k5.owner,
                        k5.gen,
                        res.clock,
                        res.fetch_cnt,
                        res.NB,
                        res.S,
                        res.n_kv,
                        False,
                    )
                else:
                    _k5_evict_kernel[64,](
                        k5.snap_mail[b],
                        k5.snap_cnt[b],
                        res.page2slot,
                        res.slot2page,
                        res.NB,
                        res.S,
                        NKV=res.n_kv,
                        GRID=64,
                        num_warps=1,
                    )
            if not _prologue_owns:
                _bcopy(k5.snap_mail[b], k5.elist_dev, k5.snap_cnt[b], cap=k5.mcap)
                k5.ecnt_dev.copy_(k5.snap_cnt[b], non_blocking=True)
                self._elist_batch = b
            k5.mail_cnt.zero_()
        zev = torch.cuda.Event()
        zev.record(tr)
        if defer_zev:
            self._zev_pending = zev
        else:
            main.wait_event(zev)
        t0 = None
        if _K5_STATS:
            t0 = torch.cuda.Event(enable_timing=True)
            t0.record(tr)
        with torch.cuda.stream(tr):
            publish_now = _prologue_owns and t._plan_deferred
            _k5_transport_kernel[264,](
                k5.snap_mail[b],
                k5.snap_cnt[b],
                poolv,
                poolk,
                hotv,
                hotk,
                t.NB,
                res.S,
                hotv.stride(0),
                GRID=264,
                NKV=t.n_kv,
                PAGE=t.page,
                D=t.d,
                HAS_K=has_k,
                p2s_ptr=res.page2slot,
                gen_ptr=k5.gen,
                PUBLISH=publish_now,
                num_warps=4,
            )
        fence = torch.cuda.Event()
        fence.record(tr)
        self._inflight.append((fence, b))
        if _K5_STATS:
            t1 = torch.cuda.Event(enable_timing=True)
            t1.record(tr)
            self._timing.append((t0, t1))

    def consume_fenced_prologue(self) -> bool:
        (t, k5) = (self._tier, self._k5)
        res = t.residency
        if not self._inflight:
            return False
        (fence_ev, b) = self._inflight[0]
        if not fence_ev.query():
            return False
        self._inflight.pop(0)
        main = torch.cuda.current_stream(device=t.device)
        main.wait_event(fence_ev)
        from .k5_cuda import get_mod

        mod = get_mod()
        if mod is not None:
            mod.k5_headv(
                k5.snap_mail[b],
                k5.snap_cnt[b],
                k5.snap_mail[b],
                k5.zcnt,
                res.page2slot,
                res.slot2page,
                k5.owner,
                k5.gen,
                res.clock,
                res.fetch_cnt,
                res.NB,
                res.S,
                res.n_kv,
                False,
            )
            mod.k5_headv(
                k5.snap_mail[b],
                k5.zcnt,
                k5.snap_mail[b],
                k5.snap_cnt[b],
                res.page2slot,
                res.slot2page,
                k5.owner,
                k5.gen,
                res.clock,
                res.fetch_cnt,
                res.NB,
                res.S,
                res.n_kv,
                False,
            )
        else:
            _k5_evict_kernel[64,](
                k5.snap_mail[b],
                k5.snap_cnt[b],
                res.page2slot,
                res.slot2page,
                res.NB,
                res.S,
                NKV=res.n_kv,
                GRID=64,
                num_warps=1,
            )
            _k5_promote_kernel[64,](
                k5.snap_mail[b],
                k5.snap_cnt[b],
                res.page2slot,
                res.slot2page,
                k5.owner,
                k5.gen,
                res.NB,
                res.S,
                NKV=res.n_kv,
                GRID=64,
                num_warps=1,
            )
        global _PROMO1_ANNOUNCED
        if not _PROMO1_ANNOUNCED:
            _PROMO1_ANNOUNCED = True
            print(
                "[locks] PROMO1 ACTIVE: fenced batch applied at the prologue (snapshot-direct, install latency 2 -> 1 step)",
                flush=True,
            )
        return True

    def stats_line(self) -> str:
        sel = max(self.sel, 1)
        steps = max(self.steps, 1)
        return f"[locks k5] steps={self.steps} sel={self.sel} pinned_miss={self.pinned_miss} staged_miss={self.staged_miss} assigned={self.assigned} dropped={self.dropped} lost_v={self.lost_v} hit={1.0 - (self.pinned_miss + self.staged_miss) / sel:.4f} fetched={self.ce_bytes / 2**20:.1f}MiB ({self.ce_bytes / steps / 2**20:.3f}MiB/step) fetch_gpu={self.fetch_us / 1000.0:.1f}ms ({self.fetch_us / steps:.1f}us/step side-stream)"
