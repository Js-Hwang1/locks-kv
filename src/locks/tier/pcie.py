"""Layer-local cache fills for PCIe hosts.

All phases run on the attention stream: protect selected hot pages, assign
unprotected slots, attend while filling misses, then publish the new mappings.
Kernel boundaries fence eviction and publication. Original KV is transferred
once and the attention kernel also stores its unchanged bytes in the hot cache.
"""
import torch
import triton
import triton.language as tl
from .k5 import _masked_cas


@triton.jit
def _mark_selected(TAB, CNT, BT, SL, P2S, EPOCH, CLOCK, VBO, VALID,
                   BLOCKS, SLOTS, COUNTS, NB, S, LAYER,
                   STRIDE_TR, STRIDE_TH, STRIDE_CR, STRIDE_BT,
                   NK: tl.constexpr, PAGE: tl.constexpr,
                   CAP: tl.constexpr, BLOCK: tl.constexpr):
    r, h = tl.program_id(0), tl.program_id(1)
    count = tl.load(CNT + r * STRIDE_CR + h)
    length = tl.load(SL + r)
    clock = tl.load(CLOCK)
    unit = LAYER * NK + h
    for off in range(0, count, BLOCK):
        j = off + tl.arange(0, BLOCK)
        active = j < count
        page = tl.load(TAB + r * STRIDE_TR + h * STRIDE_TH + j,
                       mask=active, other=0)
        block = tl.load(BT + r * STRIDE_BT + page, mask=active, other=0)
        complete = active & ((page + 1) * PAGE <= length)
        slot = tl.load(P2S + unit * NB + block, mask=active, other=-1)
        hit = complete & (slot >= 0)
        tl.store(EPOCH + unit * S + slot, clock, mask=hit)
        staged = tl.load(VBO + block, mask=active, other=-1) >= 0
        candidate = complete & (slot == -1) & ~staged
        valid = tl.load(VALID + unit * NB + block, mask=candidate, other=1)
        tl.device_assert(~candidate | (valid != 0), "PCIe cache: original KV is unavailable")
        # A shared prefix may appear in several requests. Claim it once.
        old = _masked_cas(P2S + unit * NB + block,
                          tl.full((BLOCK,), -1, tl.int32),
                          tl.full((BLOCK,), -2, tl.int32) - r, candidate)
        claimed = candidate & (old == -1)
        flags = claimed.to(tl.int32)
        n = tl.sum(flags, 0)
        base = tl.atomic_add(COUNTS + h, n, sem="relaxed")
        pos = base + tl.cumsum(flags, 0) - flags
        tl.device_assert(base + n <= CAP, "PCIe cache: selected-page capacity exceeded")
        tl.store(BLOCKS + h * CAP + pos, block, mask=claimed)
        tl.store(SLOTS + h * CAP + pos, -1, mask=claimed)


@triton.jit
def _assign_slots(BLOCKS, SLOTS, COUNTS, P2S, S2P, EPOCH, HAND, CLOCK,
                  SCRATCH, NB, S, LAYER, NK: tl.constexpr,
                  CAP: tl.constexpr, BLOCK: tl.constexpr,
                  REUSE_STEPS: tl.constexpr = 8, SLOT_BITS: tl.constexpr = 0,
                  FUSED: tl.constexpr = False):
    h = tl.program_id(0)
    unit = LAYER * NK + h
    count = tl.load(COUNTS + h)
    clock = tl.load(CLOCK)
    cursor = tl.load(HAND + unit) % S
    spent = 0
    lanes = tl.arange(0, BLOCK)
    for off in range(0, count, BLOCK):
        index = off + lanes
        need = index < count
        block = tl.load(BLOCKS + h * CAP + index, mask=need, other=0)
        while (tl.sum(need.to(tl.int32), 0) > 0) & (spent < 2 * S):
            # Prefer pages unused over several decode steps. A second pass
            # relaxes this preference, but never evicts a current selection.
            first_pass = spent < S
            limit = tl.where(first_pass, S, 2 * S)
            advance = tl.minimum(BLOCK, limit - spent)
            scanned = lanes < advance
            slot = (cursor + lanes) % S
            epoch = tl.load(EPOCH + unit * S + slot, mask=scanned, other=clock)
            cutoff = clock - tl.where(first_pass, REUSE_STEPS, 0)
            free = scanned & ((epoch == 0) | (epoch < cutoff))
            free_i = free.to(tl.int32)
            free_rank = tl.cumsum(free_i, 0) - free_i
            nfree = tl.sum(free_i, 0)
            tl.store(SCRATCH + h * BLOCK + free_rank, slot, mask=free)
            tl.debug_barrier()
            rank = tl.cumsum(need.to(tl.int32), 0) - need.to(tl.int32)
            take = need & (rank < nfree)
            chosen = tl.load(SCRATCH + h * BLOCK + rank, mask=take, other=0)
            previous = tl.load(S2P + unit * S + chosen, mask=take, other=-1)
            evict = take & (previous >= 0)
            _masked_cas(P2S + unit * NB + previous, chosen,
                        tl.full((BLOCK,), -1, tl.int32), evict)
            tl.store(S2P + unit * S + chosen, block, mask=take)
            tl.store(EPOCH + unit * S + chosen, clock, mask=take)
            tl.store(SLOTS + h * CAP + index, chosen, mask=take)
            if FUSED:
                # Encode the sole copying request and its slot. Other requests
                # sharing this page may read host KV but cannot race the copy.
                owner = -tl.load(P2S + unit * NB + block, mask=take, other=-2) - 2
                tl.store(P2S + unit * NB + block,
                         -3 - chosen - (owner << SLOT_BITS), mask=take)
            need = need & ~take
            cursor = (cursor + advance) % S
            spent += advance
            tl.debug_barrier()
        # Insufficient hot capacity is a cache miss, not a dropped KV page.
        # Unassigned pages continue to use the original host-backed attention.
        tl.store(P2S + unit * NB + block, -1, mask=need)
    tl.store(HAND + unit, cursor)


@triton.jit
def _fill_cache(BLOCKS, SLOTS, COUNTS, HOST_K, HOST_V, HOT_K, HOT_V,
                P2S, NB, S, LAYER, NK: tl.constexpr, PAGE: tl.constexpr,
                D: tl.constexpr, CAP: tl.constexpr, GRID: tl.constexpr):
    worker, h = tl.program_id(0), tl.program_id(1)
    count = tl.load(COUNTS + h)
    offsets = tl.arange(0, PAGE)[:, None] * D + tl.arange(0, D)[None, :]
    for index in range(worker, count, GRID):
        slot = tl.load(SLOTS + h * CAP + index)
        if slot >= 0:
            block = tl.load(BLOCKS + h * CAP + index).to(tl.int64)
            host = ((LAYER * NB + block) * NK + h) * (PAGE * D) + offsets
            unit = (LAYER * NK + h).to(tl.int64)
            hot = (unit * S + slot) * (PAGE * D) + offsets
            tl.store(HOT_K + hot, tl.load(HOST_K + host))
            tl.store(HOT_V + hot, tl.load(HOST_V + host))
            tl.debug_barrier()
            tl.store(P2S + (LAYER * NK + h) * NB + block, slot)


class PcieCache:
    """One workspace reused across layers; no host synchronization in decode."""
    @staticmethod
    def workspace_bytes(n_kv, capacity):
        return 4 * n_kv * (2 * capacity + 1 + 256)

    def __init__(self, tier, selected_page_capacity):
        self.slot_bits = (tier.residency.S - 1).bit_length()
        # Very large geometries retain the explicit fill path rather than
        # overflowing the int32 owner/slot encoding.
        self.fused = tier.residency.R * (1 << self.slot_bits) < 2**31 - 3
        self.tier = tier
        self.capacity = tier.residency.R * selected_page_capacity
        self.blocks = torch.empty((tier.n_kv, self.capacity), dtype=torch.int32, device=tier.device)
        self.slots = torch.empty_like(self.blocks)
        self.counts = torch.zeros(tier.n_kv, dtype=torch.int32, device=tier.device)
        self.scratch = torch.empty((tier.n_kv, 256), dtype=torch.int32, device=tier.device)

    def prepare(self, layer, table, counts, block_table, seq_lens):
        t = self.tier
        res = t.residency
        if layer == 0:
            res.clock.add_(1)
        self.counts.zero_()
        _mark_selected[seq_lens.shape[0], t.n_kv](
            table, counts, block_table, seq_lens, res.page2slot,
            res.lru_clock, res.clock, t.staging.vbo, res.pool_valid,
            self.blocks, self.slots, self.counts, t.NB, res.S, layer,
            table.stride(0), table.stride(1), counts.stride(0), block_table.stride(0),
            NK=t.n_kv, PAGE=t.page, CAP=self.capacity, BLOCK=128,
            num_warps=4, debug=True)
        _assign_slots[(t.n_kv,)](
            self.blocks, self.slots, self.counts, res.page2slot, res.slot2page,
            res.lru_clock, res.hand, res.clock, self.scratch, t.NB, res.S, layer,
            NK=t.n_kv, CAP=self.capacity, BLOCK=256, SLOT_BITS=self.slot_bits,
            FUSED=self.fused, num_warps=4)
        if not self.fused:
            grid = min(256, self.capacity)
            _fill_cache[grid, t.n_kv](
                self.blocks, self.slots, self.counts, t.pool_k.dev_view, t.pool.dev_view,
                res.hot_k_i16, res.hot_i16, res.page2slot, t.NB, res.S, layer,
                NK=t.n_kv, PAGE=t.page, D=t.d, CAP=self.capacity, GRID=grid, num_warps=4)

    def finish(self, layer):
        if not self.fused:
            return
        t = self.tier
        _publish_cache[(t.n_kv,)](self.blocks, self.slots, self.counts,
            t.residency.page2slot, t.NB, layer, NK=t.n_kv, CAP=self.capacity,
            BLOCK=128, num_warps=4)


@triton.jit
def _publish_cache(BLOCKS, SLOTS, COUNTS, P2S, NB, LAYER,
                   NK: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)
    count = tl.load(COUNTS + h)
    for off in range(0, count, BLOCK):
        index = off + tl.arange(0, BLOCK)
        slot = tl.load(SLOTS + h * CAP + index, mask=index < count, other=-1)
        block = tl.load(BLOCKS + h * CAP + index, mask=slot >= 0, other=0)
        tl.store(P2S + (LAYER * NK + h) * NB + block, slot, mask=slot >= 0)
