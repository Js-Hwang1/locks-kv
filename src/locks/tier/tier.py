from __future__ import annotations
import math
import os
import torch
from .pool import MappedHostVPool
from .residency import Residency
from .staging import Staging

_ASSERT = os.environ.get("LOCKS_TIER_ASSERT") not in (None, "", "0")


class Tier:
    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        n_kv: int,
        page: int,
        d: int,
        hot_slots: int,
        max_reqs: int,
        max_pages: int,
        max_tokens: int,
        dtype: torch.dtype,
        device,
        v_blocks: int | None = None,
        max_pool_gb: float = 256.0,
        offload_k: bool = False,
        summary_cache: bool = False,
        mem_transport: str = "async",
        selected_page_capacity: int | None = None,
    ):
        (self.L, self.NB, self.n_kv) = (num_layers, num_blocks, n_kv)
        (self.page, self.d) = (page, d)
        self.device = torch.device(device)
        self.dtype = dtype
        self.offload_k = offload_k
        self.summary_cache = bool(summary_cache)
        self.mem_transport = mem_transport
        self.pcie_cache = None
        self.pool = MappedHostVPool(
            num_layers=num_layers,
            num_blocks=num_blocks,
            n_kv=n_kv,
            page=page,
            d=d,
            dtype=dtype,
            device=device,
            max_pool_gb=max_pool_gb,
        )
        self.pool_k = (
            MappedHostVPool(
                num_layers=num_layers,
                num_blocks=num_blocks,
                n_kv=n_kv,
                page=page,
                d=d,
                dtype=dtype,
                device=device,
                max_pool_gb=max_pool_gb,
            )
            if offload_k
            else None
        )
        self.residency = Residency(
            num_layers=num_layers,
            num_blocks=num_blocks,
            n_kv=n_kv,
            hot_slots=hot_slots,
            page=page,
            d=d,
            max_reqs=max_reqs,
            max_pages=max_pages,
            dtype=dtype,
            device=device,
            offload_k=offload_k,
        )
        self.staging = Staging(
            num_layers=num_layers,
            num_blocks=num_blocks,
            n_kv=n_kv,
            page=page,
            d=d,
            max_reqs=max_reqs,
            max_pages=max_pages,
            max_tokens=max_tokens,
            dtype=dtype,
            device=device,
            v_blocks=v_blocks,
            offload_k=offload_k,
        )
        self.staging.assert_capacity()
        self.fm_ctr = torch.zeros(
            (max_reqs * n_kv,), dtype=torch.int32, device=self.device
        )
        if mem_transport == "pcie":
            if not offload_k:
                raise ValueError("PCIe cache fills require full-KV offloading")
            from .pcie import PcieCache

            self.pcie_cache = PcieCache(self, selected_page_capacity or max_pages)

    @staticmethod
    def recommend_hot_slots(
        budget_frac: float,
        max_pages: int,
        *,
        target_ratio: float = 6.0,
        sink: int = 1,
        win: int = 1,
        floor: int = 64,
        n_seqs: int = 1,
        budget_pages: int | None = None,
        cap: int | None = None,
    ) -> int:
        selectable = max(1, int(max_pages) - sink - win)
        b = max(1, math.ceil(float(budget_frac) * selectable))
        ns = max(1, int(n_seqs))
        if ns > 1 and True:
            bp = (
                b
                if budget_pages is None or int(budget_pages) <= 0
                else min(int(budget_pages), selectable)
            )
            s = (bp + sink + win) * ns
        else:
            s = (math.ceil(target_ratio * b) + sink + win) * ns
        s = max(floor, s)
        return int(min(s, max(int(max_pages) * ns, floor)))

    @classmethod
    def from_config(
        cls,
        cfg,
        *,
        num_layers,
        num_blocks,
        n_kv,
        page,
        d,
        max_reqs,
        max_pages,
        max_tokens,
        dtype,
        device,
    ) -> "Tier":
        if not cfg.is_mem:
            raise ValueError(f"Tier requires a mem variant, got {cfg.variant!r}")
        hot_slots = cfg.hot_slots
        if hot_slots is None or hot_slots <= 0:
            eff_budget = cfg.budget if cfg.budget is not None else 0.15
            hot_slots = cls.recommend_hot_slots(
                eff_budget,
                max_pages,
                n_seqs=max_reqs,
                target_ratio=float(getattr(cfg, "hot_ratio", 6.0)),
                sink=cfg.sink_pages,
                win=cfg.window_pages,
                budget_pages=getattr(cfg, "budget_pages", None),
            )
        return cls(
            num_layers=num_layers,
            num_blocks=num_blocks,
            n_kv=n_kv,
            page=page,
            d=d,
            hot_slots=hot_slots,
            max_reqs=max_reqs,
            max_pages=max_pages,
            max_tokens=max_tokens,
            dtype=dtype,
            device=device,
            v_blocks=cfg.v_blocks,
            max_pool_gb=cfg.max_pool_gb,
            offload_k=cfg.offload_k,
            summary_cache=bool(getattr(cfg, "mem_summary_cache", False)),
            mem_transport=cfg.mem_transport,
            selected_page_capacity=(
                min(max_pages, cfg.budget_pages + cfg.sink_pages + cfg.window_pages)
                if cfg.budget_pages else max_pages
            ),
        )

    def begin_step(
        self,
        block_table,
        seq_lens,
        query_start_loc,
        max_query_len: int,
        n_tokens: int,
        capture: bool = False,
    ) -> None:
        self._cur_mql = int(max_query_len)
        self._cur_nreq = int(seq_lens.shape[0])
        if not capture:
            if self.pcie_cache is None:
                if getattr(self, "_early_serviced", False):
                    self._early_serviced = False
                    svc = self._k5_service()
                    zp = getattr(svc, "_zev_pending", None)
                    if zp is not None:
                        torch.cuda.current_stream(self.device).wait_event(zp)
                        svc._zev_pending = None
                else:
                    self._k5_service().service()
            self.residency.invalidate_written(block_table, seq_lens, query_start_loc)
            if self.pcie_cache is None and self.residency.R > 1:
                from .k5 import _K5_PROMO1

                if _K5_PROMO1:
                    self._k5_service().consume_fenced_prologue()
        self.staging.begin_step(
            block_table,
            seq_lens,
            query_start_loc,
            max_query_len,
            n_tokens,
            pool=self.pool.dev_view,
            pool_valid=self.residency.pool_valid,
            capture=capture,
            poolk=self.pool_k.dev_view if self.pool_k is not None else None,
            generation=self.residency.gen,
        )
        if _ASSERT and (not capture):
            self.assert_invariants(block_table, seq_lens, query_start_loc)

    def assert_invariants(self, block_table, seq_lens, query_start_loc) -> None:
        torch.cuda.synchronize(self.device)
        if int(self.staging.overflow.item()) != 0:
            raise AssertionError(
                "I1 capacity: staging free-stack UNDERFLOWED (a written page got no staging block -> would lose V). Raise v_blocks to cover the concurrent-prefill peak (see Staging.assert_capacity)."
            )
        self.staging.assert_free_partition()
        (pend, incoming) = self._pending_evict()
        verified = set()
        from .k5 import _PUBLISH_ON_COPY

        if _PUBLISH_ON_COPY and pend:
            for (layer, head, slot), (block, generation) in pend.items():
                if int(self.residency.page2slot[layer, head, block]) != slot:
                    continue
                assert int(self.residency.gen[block]) == generation
                assert int(self.residency.slot2page[layer, head, slot]) == block
                assert int(self.residency.pool_valid[layer, head, block]) != 0
                assert torch.equal(
                    self.residency.hot_i16[layer, head, slot].cpu(),
                    self.pool.host[layer, block, head].view(torch.int16),
                )
                if self.offload_k:
                    assert torch.equal(
                        self.residency.hot_k_i16[layer, head, slot].cpu(),
                        self.pool_k.host[layer, block, head].view(torch.int16),
                    )
                verified.add((layer, head, slot, block))
        self.residency.assert_maps_inverse(
            pending_evict=pend, pending_incoming=incoming, verified_published=verified
        )
        self.staging.assert_no_lost_v(
            block_table, seq_lens, query_start_loc, self.residency.pool_valid
        )

    def _pending_evict(self) -> tuple:
        if not self._plan_deferred:
            return (None, None)
        k5 = getattr(self, "_k5", None)
        if k5 is None:
            return (None, None)
        cap = int(k5.mail.shape[0])
        pend: dict = {}
        incoming: list = []
        srcs = [(k5.ecnt_dev, k5.elist_dev), (k5.mail_cnt, k5.mail)]
        svc = getattr(self, "_k5_svc", None)
        if svc is not None:
            for _fence, b in svc._inflight:
                srcs.append((k5.snap_cnt[b], k5.snap_mail[b]))
        for cnt, buf in srcs:
            n = min(int(cnt.item()), cap)
            if n <= 0:
                continue
            for l, kv, blk, h, g in buf[:n, :5].to("cpu").tolist():
                incoming.append((int(l), int(kv), int(blk)))
                if h >= 0:
                    pend[int(l), int(kv), int(h)] = (int(blk), int(g))
        return (pend, incoming)

    def write_kv(self, lidx, value, n_tokens, key=None) -> None:
        self.staging.scatter_kv(lidx, key, value, n_tokens)

    def write_kv_konly(self, lidx, key, value, kv_cache, slot_mapping) -> None:
        if self.summary_cache:
            self.staging.scatter_kv(lidx, key, value, slot_mapping.shape[0])
        else:
            self.staging.scatter_kv_engine(lidx, key, value, kv_cache, slot_mapping)

    def gather_k_blocks(self, lidx: int, blocks: torch.Tensor) -> torch.Tensor:
        from .gather import gather

        return gather(self, lidx, blocks)

    def gather_k_tags(
        self, lidx: int, blocks: torch.Tensor, token: int, width: int
    ) -> torch.Tensor:
        from .gather import gather

        return gather(self, lidx, blocks, token=token, width=width)

    def side_step(
        self, lidx, page_table, page_cnt, block_table, seq_lens, n_req, main_stream=None
    ) -> None:
        from .k5 import k5_headv, k5_step

        k5 = self._k5_state()
        ms = main_stream or torch.cuda.current_stream(device=self.device)
        self._plan_deferred = True
        if lidx == 0:
            k5_headv(self.residency, k5, tick=False)
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=self.device)
            n_layers = len(self.residency.hot_i16)
            self._ev_fork = [torch.cuda.Event() for _ in range(n_layers)]
            self._ev_tail = [torch.cuda.Event() for _ in range(n_layers)]
        self._ev_fork[lidx].record(ms)
        side = self._side_stream
        with torch.cuda.stream(side):
            side.wait_event(self._ev_fork[lidx])
            k5_step(
                self.residency,
                k5,
                self.staging,
                lidx,
                page_table,
                page_cnt,
                block_table,
                seq_lens,
                n_req,
            )
            self._ev_tail[lidx].record(side)

    def k8_head(self):
        from .k5 import k5_headv

        self._plan_deferred = True
        k5_headv(self.residency, self._k5_state(), tick=True)

    def side_join(self, lidx, main_stream=None):
        if self._ev_tail is not None:
            stream = main_stream or torch.cuda.current_stream(device=self.device)
            stream.wait_event(self._ev_tail[lidx])

    _side_stream = None
    _ev_fork = None
    _ev_tail = None
    _plan_deferred = False

    def _k5_state(self):
        k5 = getattr(self, "_k5", None)
        if k5 is None:
            from .k5 import K5State

            k5 = K5State(self.residency, self.device)
            self._k5 = k5
        return k5

    def _k5_service(self):
        svc = getattr(self, "_k5_svc", None)
        if svc is None:
            from .k5 import K5Service

            svc = K5Service(self, self._k5_state())
            self._k5_svc = svc
        return svc

    def early_service(self) -> None:
        if self.pcie_cache is not None:
            return
        if not getattr(self, "_early_announced", False):
            self._early_announced = True
            print("[locks] K5_EARLY: first early service fired", flush=True)
        self._k5_service().service(defer_zev=True)
        self._early_serviced = True

    def bytes_report(self) -> str:
        tag = "mem-kv" if self.offload_k else "mem-v"
        kpool = (
            f" + K-pool {self.pool_k.pool_gib:.2f}GiB DRAM"
            if self.pool_k is not None
            else ""
        )
        return f"[locks tier {tag}] {self.pool.vram_note()}{kpool} | {self.residency.bytes_report()} | {self.staging.bytes_report()}"
