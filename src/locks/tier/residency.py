from __future__ import annotations
import torch
import triton
import triton.language as tl

_I32_MAX = tl.constexpr(2147483647)


@triton.jit
def _step_setup_kernel(clock_ptr, fcnt_ptr):
    clk = tl.load(clock_ptr)
    tl.store(clock_ptr, clk + 1)
    tl.store(fcnt_ptr, 0)


@triton.jit
def _invalidate_kernel(
    bt_ptr,
    sl_ptr,
    qsl_ptr,
    p2s_ptr,
    s2p_ptr,
    valid_ptr,
    gen_ptr,
    NB,
    S,
    stride_btr,
    PAGE: tl.constexpr,
    NKV: tl.constexpr,
    KV_PAD: tl.constexpr,
):
    l = tl.program_id(0)
    r = tl.program_id(1)
    sl = tl.load(sl_ptr + r)
    if sl > 0:
        ql = tl.load(qsl_ptr + r + 1) - tl.load(qsl_ptr + r)
        p0 = (sl - ql) // PAGE
        p1 = (sl - 1) // PAGE
        offs_kv = tl.arange(0, KV_PAD)
        kmask = offs_kv < NKV
        for p in range(p0, p1 + 1):
            blk = tl.load(bt_ptr + r * stride_btr + p)
            if l == 0:
                tl.atomic_add(gen_ptr + blk, 1)
            rows = (l * NKV + offs_kv) * NB + blk
            tl.store(valid_ptr + rows, tl.zeros([KV_PAD], tl.int8), mask=kmask)
            slot = tl.load(p2s_ptr + rows, mask=kmask, other=-1)
            res = kmask & (slot >= 0)
            gone = kmask & (slot != -1)
            tl.store(p2s_ptr + rows, tl.full([KV_PAD], -1, tl.int32), mask=gone)
            tl.store(
                s2p_ptr + (l * NKV + offs_kv) * S + slot,
                tl.full([KV_PAD], -1, tl.int32),
                mask=res,
            )


class Residency:
    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        n_kv: int,
        hot_slots: int,
        page: int,
        d: int,
        max_reqs: int,
        max_pages: int,
        dtype: torch.dtype,
        device,
        offload_k: bool = False,
    ):
        (self.L, self.NB, self.n_kv) = (num_layers, num_blocks, n_kv)
        (self.S, self.page, self.d) = (hot_slots, page, d)
        (self.R, self.MP) = (max_reqs, max_pages)
        self.dtype = dtype
        self.offload_k = offload_k
        dev = torch.device(device)
        self.device = dev
        (i32, i8) = (torch.int32, torch.int8)
        self.hot = torch.zeros(
            (num_layers, n_kv, hot_slots, page, d), dtype=dtype, device=dev
        )
        self.hot_i16 = self.hot.view(torch.int16)
        self.hot_k = (
            torch.zeros((num_layers, n_kv, hot_slots, page, d), dtype=dtype, device=dev)
            if offload_k
            else None
        )
        self.hot_k_i16 = self.hot_k.view(torch.int16) if offload_k else None
        self.page2slot = torch.full(
            (num_layers, n_kv, num_blocks), -1, dtype=i32, device=dev
        )
        self.slot2page = torch.full(
            (num_layers, n_kv, hot_slots), -1, dtype=i32, device=dev
        )
        self.lru_clock = torch.zeros(
            (num_layers, n_kv, hot_slots), dtype=i32, device=dev
        )
        self.pool_valid = torch.zeros(
            (num_layers, n_kv, num_blocks), dtype=i8, device=dev
        )
        self.gen = torch.zeros((num_blocks,), dtype=i32, device=dev)
        self.owner = torch.zeros((num_layers, n_kv, hot_slots), dtype=i32, device=dev)
        self.hand = torch.zeros((num_layers, n_kv), dtype=i32, device=dev)
        self.clock = torch.ones((1,), dtype=i32, device=dev)
        self.overflow = torch.zeros((1,), dtype=i32, device=dev)
        self.miss_pages = torch.zeros(
            (max_reqs, n_kv, max_pages), dtype=i32, device=dev
        )
        self.miss_cnt = torch.zeros((max_reqs, n_kv), dtype=i32, device=dev)
        self.victim_slots = torch.zeros(
            (max_reqs, n_kv, max_pages), dtype=i32, device=dev
        )
        self.FCAP = max_reqs * n_kv * max_pages
        self.fetch_list = torch.zeros((self.FCAP, 3), dtype=i32, device=dev)
        self.fetch_cnt = torch.zeros((1,), dtype=i32, device=dev)
        self.fetch_grid = 1024
        elem = torch.empty(0, dtype=dtype).element_size()
        hot_n = self.hot.numel() + (self.hot_k.numel() if offload_k else 0)
        self.hot_gib = hot_n * elem / 2**30
        self.maps_gib = (
            sum(
                (
                    t.numel() * t.element_size()
                    for t in (
                        self.page2slot,
                        self.slot2page,
                        self.lru_clock,
                        self.pool_valid,
                        self.miss_pages,
                        self.miss_cnt,
                        self.victim_slots,
                        self.fetch_list,
                        self.fetch_cnt,
                    )
                )
            )
            / 2**30
        )

    def invalidate_written(self, block_table, seq_lens, query_start_loc) -> None:
        n_req = seq_lens.shape[0]
        kv_pad = max(1, triton.next_power_of_2(self.n_kv))
        _invalidate_kernel[self.L, n_req](
            block_table,
            seq_lens,
            query_start_loc,
            self.page2slot,
            self.slot2page,
            self.pool_valid,
            self.gen,
            self.NB,
            self.S,
            block_table.stride(0),
            PAGE=self.page,
            NKV=self.n_kv,
            KV_PAD=kv_pad,
            num_warps=1,
        )

    def assert_maps_inverse(
        self,
        layer: int | None = None,
        pending_evict: dict | None = None,
        pending_incoming: list | None = None,
        verified_published: set | None = None,
    ) -> None:
        layers = list(range(self.L)) if layer is None else [layer]
        pend = pending_evict if pending_evict is not None else {}
        published = verified_published if verified_published is not None else set()
        for pl, pkv, ph, pblk in published:
            assert (pl, pkv, ph) in pend and pend[pl, pkv, ph][0] == pblk
        inc = (
            pending_incoming
            if pending_incoming is not None
            else [(k[0], k[1], v[0]) for (k, v) in pend.items()]
        )
        for l in layers:
            for kv in range(self.n_kv):
                s2p = self.slot2page[l, kv]
                p2s = self.page2slot[l, kv]
                own = self.owner[l, kv]
                occ = (s2p >= 0) & (own == 0)
                pages = s2p[occ].long()
                if pages.numel() != torch.unique(pages).numel():
                    raise AssertionError(
                        f"I3 residency: layer {l} kv {kv} has two slots holding the same page (slot2page not injective)"
                    )
                slots = torch.nonzero(occ, as_tuple=False).flatten()
                back = p2s[pages]
                if not torch.equal(back, slots.to(back.dtype)):
                    raise AssertionError(
                        f"I2 residency: layer {l} kv {kv} page2slot/slot2page are not mutual inverses (install broke single-owner)"
                    )
                adv = torch.nonzero(p2s >= 0, as_tuple=False).flatten()
                if adv.numel():
                    hs = p2s[adv].long()
                    if bool(((hs < 0) | (hs >= self.S)).any()):
                        raise AssertionError(
                            f"I2 residency: layer {l} kv {kv} page2slot holds an out-of-range slot id (S={self.S})"
                        )
                    bad = s2p[hs] != adv.to(s2p.dtype)
                    if bool(bad.any()):
                        b0 = int(adv[bad][0])
                        h0 = int(p2s[b0])
                        raise AssertionError(
                            f"I2 residency: layer {l} kv {kv} page {b0} is advertised at slot {h0}, but slot2page[{h0}]={int(s2p[h0])} -- the slot's BYTES belong to a different page than page2slot claims"
                        )
                fly = (s2p >= 0) & (own != 0)
                if bool(fly.any()):
                    fsl = torch.nonzero(fly, as_tuple=False).flatten()
                    fb = p2s[s2p[fly].long()]
                    outgoing = fb == fsl.to(fb.dtype)
                    if bool((~((fb < 0) | outgoing)).any()):
                        raise AssertionError(
                            f"I2 residency: layer {l} kv {kv} has an in-flight slot whose page is already resident elsewhere"
                        )
                    for h in fsl[outgoing].tolist():
                        if (l, kv, int(h)) not in pend:
                            raise AssertionError(
                                f"I2 residency: layer {l} kv {kv} slot {h} is CLAIMED while still advertising its old page, and no graph-head evict is queued for it (a deferred eviction was LOST)"
                            )
        want = set(layers)
        for pl, pkv, pblk in inc:
            if pl not in want:
                continue
            cur = int(self.page2slot[pl, pkv, pblk])
            if cur >= 0 and (pl, pkv, cur, pblk) not in published:
                raise AssertionError(
                    f"I2 residency: layer {pl} kv {pkv} page {pblk} is queued for transport but is ALREADY advertised as resident at slot {cur} (in-flight bytes readable)"
                )
        for (pl, pkv, ph), (pblk, pgen) in pend.items():
            if pl not in want:
                continue
            if int(self.gen[pblk]) == int(pgen) and int(self.owner[pl, pkv, ph]) == 0:
                raise AssertionError(
                    f"I2 residency: layer {pl} kv {pkv} slot {ph} has an ASSIGNED in-flight transfer of page {pblk} (current generation) but its owner claim is FREE -- the claim was lost; a later batch could assign this slot"
                )

    def bytes_report(self) -> str:
        return f"hot={self.hot_gib:.3f}GiB (S={self.S} slots/kv, working-set bound) maps={self.maps_gib:.3f}GiB (NB={self.NB} int8/int32 bookkeeping)"
