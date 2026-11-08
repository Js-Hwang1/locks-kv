from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_kv_kernel(
    key_ptr,
    val_ptr,
    kp_ptr,
    vp_ptr,
    vslot_ptr,
    skt,
    skh,
    svt,
    svh,
    vpb,
    vpt,
    vph,
    NKV: tl.constexpr,
    KV_PAD: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_K: tl.constexpr,
):
    t = tl.program_id(0)
    offs_kv = tl.arange(0, KV_PAD)
    offs_d = tl.arange(0, D)
    kmask = offs_kv < NKV
    vs = tl.load(vslot_ptr + t)
    if vs >= 0:
        vb = (vs // PAGE).to(tl.int64)
        voff = vs % PAGE
        dst = vb * vpb + voff * vpt + offs_kv[:, None] * vph + offs_d[None, :]
        v = tl.load(
            val_ptr + t * svt + offs_kv[:, None] * svh + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        tl.store(vp_ptr + dst, v, mask=kmask[:, None])
        if HAS_K:
            k = tl.load(
                key_ptr + t * skt + offs_kv[:, None] * skh + offs_d[None, :],
                mask=kmask[:, None],
                other=0.0,
            )
            tl.store(kp_ptr + dst, k, mask=kmask[:, None])


@triton.jit
def _scatter_kv_engine_kernel(
    key_ptr,
    val_ptr,
    kc_ptr,
    vp_ptr,
    kp_ptr,
    slot_ptr,
    vslot_ptr,
    skt,
    skh,
    svt,
    svh,
    kcb,
    kct,
    kch,
    vpb,
    vpt,
    vph,
    NKV: tl.constexpr,
    KV_PAD: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_K: tl.constexpr,
):
    t = tl.program_id(0)
    offs_kv = tl.arange(0, KV_PAD)
    offs_d = tl.arange(0, D)
    kmask = offs_kv < NKV
    slot = tl.load(slot_ptr + t)
    if slot >= 0:
        k = tl.load(
            key_ptr + t * skt + offs_kv[:, None] * skh + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        blk = (slot // PAGE).to(tl.int64)
        off = slot % PAGE
        tl.store(
            kc_ptr + blk * kcb + off * kct + offs_kv[:, None] * kch + offs_d[None, :],
            k,
            mask=kmask[:, None],
        )
    vs = tl.load(vslot_ptr + t)
    if vs >= 0:
        v = tl.load(
            val_ptr + t * svt + offs_kv[:, None] * svh + offs_d[None, :],
            mask=kmask[:, None],
            other=0.0,
        )
        vb = (vs // PAGE).to(tl.int64)
        voff = vs % PAGE
        dst = vb * vpb + voff * vpt + offs_kv[:, None] * vph + offs_d[None, :]
        tl.store(vp_ptr + dst, v, mask=kmask[:, None])
        if HAS_K:
            ks = tl.load(
                key_ptr + t * skt + offs_kv[:, None] * skh + offs_d[None, :],
                mask=kmask[:, None],
                other=0.0,
            )
            tl.store(kp_ptr + dst, ks, mask=kmask[:, None])


class Staging:
    @staticmethod
    def default_v_blocks(chunk_pages: int, max_reqs: int) -> int:
        return chunk_pages + chunk_pages // 4 + 4 * max_reqs + 64

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        n_kv: int,
        page: int,
        d: int,
        max_reqs: int,
        max_pages: int,
        max_tokens: int,
        dtype: torch.dtype,
        device,
        v_blocks: int | None = None,
        max_flush: int | None = None,
        offload_k: bool = False,
    ):
        (self.L, self.NB, self.n_kv) = (num_layers, num_blocks, n_kv)
        (self.page, self.d) = (page, d)
        (self.R, self.MP) = (max_reqs, max_pages)
        self.max_tokens = max_tokens
        self.offload_k = offload_k
        dev = torch.device(device)
        self.device = dev
        (i32, i64) = (torch.int32, torch.int64)
        chunk_pages = (max_tokens + page - 1) // page
        if v_blocks is None or v_blocks <= 0:
            v_blocks = Staging.default_v_blocks(chunk_pages, max_reqs)
        if max_flush is None or max_flush <= 0:
            max_flush = chunk_pages + 2 * max_reqs + 64
        self.NV = v_blocks
        self.MAXF = max_flush
        self.chunk_pages = chunk_pages
        self.v_pool = torch.zeros(
            (num_layers, v_blocks, page, n_kv, d), dtype=dtype, device=dev
        )
        self.k_pool = (
            torch.zeros((num_layers, v_blocks, page, n_kv, d), dtype=dtype, device=dev)
            if offload_k
            else None
        )
        self.vbo = torch.full((num_blocks,), -1, dtype=i32, device=dev)
        self.v_flushed = torch.zeros((num_blocks,), dtype=i32, device=dev)
        self.v_free_stack = torch.arange(v_blocks, dtype=i32, device=dev)
        self.v_free_top = torch.full((1,), v_blocks, dtype=i32, device=dev)
        self.v_slot_mapping = torch.full((max_tokens,), -1, dtype=i64, device=dev)
        self.flush_list = torch.zeros((max_flush, 2), dtype=i32, device=dev)
        self.flush_cnt = torch.zeros((1,), dtype=i32, device=dev)
        self.overflow = torch.zeros((1,), dtype=i32, device=dev)
        self._np_prev = torch.full((max_reqs,), -1, dtype=i32, device=dev)
        self._decode_guard = torch.full((max_reqs, 6), -1, dtype=i32, device=dev)
        self._frontier = True
        elem = torch.empty(0, dtype=dtype).element_size()
        pools = self.v_pool.numel() + (self.k_pool.numel() if offload_k else 0)
        self.vpool_gib = pools * elem / 2**30
        self.maps_gib = (
            sum(
                (
                    t.numel() * t.element_size()
                    for t in (
                        self.vbo,
                        self.v_flushed,
                        self.v_free_stack,
                        self.v_slot_mapping,
                        self.flush_list,
                    )
                )
            )
            / 2**30
        )

    def begin_step(
        self,
        block_table,
        seq_lens,
        query_start_loc,
        max_query_len: int,
        n_tokens: int,
        pool,
        pool_valid,
        capture: bool = False,
        poolk=None,
        generation=None,
    ) -> None:
        if n_tokens > self.max_tokens:
            raise RuntimeError(
                f"tier begin_step: {n_tokens} tokens > v_slot_mapping capacity {self.max_tokens}; raise max_tokens sizing"
            )
        self.v_slot_mapping.fill_(-1)
        from .staging_cuda import get_mod

        mod = get_mod()
        n_req = 0 if capture else seq_lens.shape[0]
        mod.staging_lifecycle(
            block_table,
            seq_lens,
            query_start_loc,
            self.v_slot_mapping,
            self.v_flushed,
            self.vbo,
            self.flush_list,
            self.flush_cnt,
            self.v_free_stack,
            self.v_free_top,
            self.overflow,
            self._np_prev,
            self._decode_guard,
            generation if generation is not None else self._np_prev,
            self._frontier and generation is not None,
            n_req,
            n_tokens,
            self.page,
            self.MAXF,
        )
        if not capture:
            has_k = self.k_pool is not None
            vp = self.v_pool
            mod.staging_flush_copy(
                vp,
                pool.data_ptr(),
                pool_valid,
                self.k_pool if has_k else vp,
                poolk.data_ptr() if has_k and poolk is not None else pool.data_ptr(),
                self.flush_list,
                self.flush_cnt,
                self.L,
                self.NB,
                self.n_kv,
                self.page,
                self.d,
                self.MAXF,
                has_k,
            )
        return

    def scatter_kv(self, lidx: int, key, value: torch.Tensor, n_tokens: int) -> None:
        vp = self.v_pool[lidx]
        kv_pad = max(1, triton.next_power_of_2(self.n_kv))
        has_k = self.k_pool is not None
        kp = self.k_pool[lidx] if has_k else vp
        skt = key.stride(0) if has_k else 0
        skh = key.stride(1) if has_k else 0
        _scatter_kv_kernel[n_tokens,](
            key if has_k else value,
            value,
            kp,
            vp,
            self.v_slot_mapping,
            skt,
            skh,
            value.stride(0),
            value.stride(1),
            vp.stride(0),
            vp.stride(1),
            vp.stride(2),
            NKV=self.n_kv,
            KV_PAD=kv_pad,
            D=self.d,
            PAGE=self.page,
            HAS_K=has_k,
        )

    def scatter_kv_engine(
        self, lidx: int, key, value: torch.Tensor, kv_cache: torch.Tensor, slot_mapping
    ) -> None:
        n = slot_mapping.shape[0]
        if n == 0:
            return
        K = kv_cache.select(1, 0)
        vp = self.v_pool[lidx]
        has_k = self.k_pool is not None
        kp = self.k_pool[lidx] if has_k else vp
        _scatter_kv_engine_kernel[n,](
            key,
            value,
            K,
            vp,
            kp,
            slot_mapping,
            self.v_slot_mapping,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            vp.stride(0),
            vp.stride(1),
            vp.stride(2),
            NKV=self.n_kv,
            KV_PAD=max(1, triton.next_power_of_2(self.n_kv)),
            D=self.d,
            PAGE=self.page,
            HAS_K=has_k,
        )

    def _written_pages(self, block_table, seq_lens, query_start_loc):
        n_req = seq_lens.shape[0]
        page = self.page
        sl = seq_lens.tolist()
        ql = (query_start_loc[1 : n_req + 1] - query_start_loc[:n_req]).tolist()
        for r in range(n_req):
            if sl[r] <= 0 or ql[r] <= 0:
                continue
            ctx = sl[r] - ql[r]
            np_ctx = ctx // page
            complete = block_table[r, :np_ctx].tolist()
            p0 = ctx // page
            p1 = (sl[r] - 1) // page
            written = block_table[r, p0 : p1 + 1].tolist()
            yield (r, ql[r] == 1, complete, written)

    def assert_free_partition(self) -> None:
        top = int(self.v_free_top.item())
        if not 0 <= top <= self.NV:
            raise AssertionError(f"I3 staging: free_top={top} out of [0,{self.NV}]")
        assigned = self.vbo[self.vbo >= 0]
        n_assigned = int(assigned.numel())
        n_unique = int(torch.unique(assigned).numel())
        if n_assigned != n_unique:
            raise AssertionError(
                f"I3 staging: a staging block is assigned to two engine blocks (assigned={n_assigned} unique={n_unique})"
            )
        if n_assigned + top != self.NV:
            raise AssertionError(
                f"I3 staging: assigned({n_assigned}) + free({top}) != NV({self.NV}); the free-stack leaked or double-counted a block"
            )

    def assert_no_lost_v(
        self, block_table, seq_lens, query_start_loc, pool_valid
    ) -> None:
        for r, _dec, complete, _written in self._written_pages(
            block_table, seq_lens, query_start_loc
        ):
            for blk in complete:
                staged = int(self.vbo[blk].item()) >= 0
                flushed = bool((pool_valid[:, :, blk] == 1).all())
                if not (staged or flushed):
                    raise AssertionError(
                        f"I1: request {r} block {blk} is a complete written page with neither staged (vbo=-1) nor flushed (pool_valid=0) V -> would read zeros"
                    )

    def assert_capacity(self) -> None:
        floor = self.chunk_pages + self.R
        if self.NV < floor:
            raise AssertionError(
                f"I1 capacity: v_blocks NV={self.NV} < floor {floor} (chunk_pages={self.chunk_pages} + max_reqs={self.R}); one prefill chunk could exhaust staging and lose V."
            )

    def bytes_report(self) -> str:
        return f"v_pool={self.vpool_gib:.3f}GiB (NV={self.NV} blocks, working-set bound) maps={self.maps_gib:.3f}GiB"
