"""Tier -- the DRAM offload facade + the ``TierVSource`` Stage-B seam.

Composes the three owned pieces into one object the backend drives:

  pool       MappedHostVPool  -- pinned host V (keyed by physical block, DRAM)
  residency  Residency        -- hot buffer + exact-LRU + gather (I2/I3)
  staging    Staging          -- v_pool + per-step flush/alloc lifecycle (I1/I4)

``begin_step`` is the ONE per-step host entry (called by the builder OUTSIDE the
graph, like the r8 refresh): it runs the residency invalidation then the staging
stage/flush/alloc lifecycle in the fixed I4 order, and (in debug) checks all four
invariants. ``TierVSource`` is what Stage B consumes as the ``V_SRC==1`` source,
mirroring ``ResidentVSource`` -- it carries the tier's pointer union WITHOUT the
decode kernel knowing any tier internals (LOCKS_DESIGN.md 8.1).

SCAFFOLD: the tier ALLOCATES and its lifecycle bookkeeping RUNS (invariants
exercised). The actual V byte movement (write-path scatter, flush copy, residency
gather) and the ``V_SRC==1`` tiered decode load are the documented FOLLOW-UP; see
tier/README.md. mem-v therefore runs correct-by-fallback (stock FA over resident
V) today while the tier machinery is validated in shadow.
"""
from __future__ import annotations

import math
import os
from typing import Tuple

import torch

from .pool import MappedHostVPool
from .residency import Residency
from .staging import Staging

# Debug latch: run the (host-syncing) invariant asserts inside begin_step.
_ASSERT = os.environ.get("LOCKS_TIER_ASSERT") not in (None, "", "0")


class Tier:
    """DRAM V-offload tier (mem-v). Facade over pool + residency + staging."""

    def __init__(self, *, num_layers: int, num_blocks: int, n_kv: int,
                 page: int, d: int, hot_slots: int, max_reqs: int,
                 max_pages: int, max_tokens: int, dtype: torch.dtype, device,
                 v_blocks: int | None = None, max_pool_gb: float = 256.0,
                 offload_k: bool = False, summary_cache: bool = False):
        self.L, self.NB, self.n_kv = num_layers, num_blocks, n_kv
        self.page, self.d = page, d
        self.device = torch.device(device)
        self.dtype = dtype
        self.offload_k = offload_k
        # P5 spec v2: the engine page holds summary RECORDS, not K.  K leaves
        # VRAM; the summary build reads K from the tier (gather_k_blocks) and
        # the write path never touches the engine page.
        self.summary_cache = bool(summary_cache)

        # V pinned host pool (mem-v + mem-kv); K pinned host pool (mem-kv only).
        self.pool = MappedHostVPool(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, dtype=dtype, device=device, max_pool_gb=max_pool_gb)
        self.pool_k = (MappedHostVPool(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, dtype=dtype, device=device, max_pool_gb=max_pool_gb)
            if offload_k else None)
        self.residency = Residency(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv,
            hot_slots=hot_slots, page=page, d=d, max_reqs=max_reqs,
            max_pages=max_pages, dtype=dtype, device=device, offload_k=offload_k)
        self.staging = Staging(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, max_reqs=max_reqs, max_pages=max_pages, max_tokens=max_tokens,
            dtype=dtype, device=device, v_blocks=v_blocks, offload_k=offload_k)
        # capacity is a static property of the sizing -> check once at build.
        self.staging.assert_capacity()
        # K8a fused-merge per-(request, kv-head) ticket counters (decode_mem).
        # Eagerly allocated: the first decode may run inside graph capture,
        # where a lazy torch alloc is illegal.  Zero-invariant: the merging
        # CTA resets its unit's counter every launch.
        self.fm_ctr = torch.zeros((max_reqs * n_kv,), dtype=torch.int32,
                                  device=self.device)
        # S3 resolved step table {slot, vb, blk, 0} per (layer, kv, logical
        # page); built once per step at the graph head (k8_head), consumed by
        # the decode split kernel (bs=1 K8 path).  Eager alloc (capture).
        self.s3_tab = torch.zeros((num_layers, n_kv, max_pages, 4),
                                  dtype=torch.int32, device=self.device)
        # FA1 landing (fetch-all design, user 2026-07-19): the per-layer
        # attention working set, selection-indexed -- (n_kv, LCAP) pages of
        # K and V (int16 bit copies) + per-page epoch flags.  ~4MB each.
        self.FA1_LCAP = 256
        _le = n_kv * self.FA1_LCAP * page * d
        # v2: parity double-buffer (layer lidx uses lidx & 1); epoch
        # flags make cross-parity reuse self-invalidating, and the ONE
        # side stream serializes fetches, so no clobber is possible.
        self.fa1_land_v = torch.zeros((2, _le), dtype=torch.int16,
                                      device=self.device)
        self.fa1_land_k = torch.zeros((2, _le), dtype=torch.int16,
                                      device=self.device)
        self.fa1_flags = torch.full((2, n_kv * self.FA1_LCAP), -1,
                                    dtype=torch.int32, device=self.device)
        # FA1-v3 (R2-EXACT): PER-LAYER landings, one slot per attended page
        # (LCAP3 = the R1 budget 128 -- NO padding, NO extra slots: the
        # buffer IS the attended set, ~168MB K+V total at the flagship) +
        # one {blk,gen} int64 tag per slot.  The tag lets the fetch SKIP
        # bytes already landed and unchanged (traffic -> true churn); a
        # rewritten block's gen bump auto-invalidates.  NOT a cache: no
        # admission, no eviction, size == attended set exactly; the split
        # falls back to the racy pinned read on any tag miss (I2).
        # Allocated only under the flag (168MB).
        if os.environ.get("LOCKS_MEM_FA1_V3", "0") == "1":
            self.FA1_LCAP3 = 128
            _le3 = n_kv * self.FA1_LCAP3 * page * d
            self.fa1_land_v3 = torch.zeros((num_layers, _le3),
                                           dtype=torch.int16,
                                           device=self.device)
            self.fa1_land_k3 = torch.zeros((num_layers, _le3),
                                           dtype=torch.int16,
                                           device=self.device)
            self.fa1_flags3 = torch.full(
                (num_layers, n_kv * self.FA1_LCAP3), -1,
                dtype=torch.int64, device=self.device)
            # v3.2 stable-slot bookkeeping (~40KB int32 arrays; NOT extra
            # page slots): slot_of rank->slot indirection (racy-safe: the
            # split re-verifies content via the tag), per-layer copy list
            # + count scratch for the churn-only phase-B copies.
            self.fa1_slot_of = torch.full(
                (num_layers, n_kv * self.FA1_LCAP3), -1,
                dtype=torch.int32, device=self.device)
            self.fa1_copy_list = torch.zeros(
                (num_layers, n_kv * self.FA1_LCAP3 * 2),
                dtype=torch.int32, device=self.device)
            self.fa1_copy_cnt = torch.zeros((num_layers, n_kv),
                                            dtype=torch.int32,
                                            device=self.device)
        else:
            self.fa1_land_v3 = self.fa1_land_k3 = self.fa1_flags3 = None
            self.fa1_slot_of = self.fa1_copy_list = self.fa1_copy_cnt = None
            self.FA1_LCAP3 = 0

    @staticmethod
    def recommend_hot_slots(budget_frac: float, max_pages: int, *,
                            target_ratio: float = 6.0, sink: int = 1,
                            win: int = 1, floor: int = 64,
                            n_seqs: int = 1,
                            budget_pages: int | None = None,
                            cap: int | None = None) -> int:
        """Working-set-proportional hot-buffer sizing (measured; scratch_memopt3).

        The hot buffer's ONLY job is to keep the REUSED selected pages resident so
        a decode step re-reads them for free instead of re-fetching over PCIe.
        The hot-hit rate is a stable function of ``S / b`` (slots per per-unit
        budget ``b``), NOT of ``S`` absolutely -- measured on the shipped exact-LRU
        gather over real Llama-3.1-8B RULER selections at 16/32/64/128K:

            S/b   1.0   1.5   2.0   3.0   4.0   6.0
            hit   .65   .78   .83   .91   .95   .98

        So sizing ``S = target_ratio * b`` fixes BOTH the hit rate (~.98 at r=6,
        ~.95 at r=4) AND the resident-V fraction (= r * budget) independent of
        context.  A FIXED ``S`` (e.g. 2048) instead gives a context-varying
        resident fraction -- >100% of V below 32K (the offload is then a net LOSS)
        and only 25% at 128K -- and a hit rate that swings .91..>.99.  Sizing to
        the working set is the memory win: resident-V = r*budget of the full V, so
        the saving SCALES as 1/budget (16-25x at 1%, 1.7-2.5x at 10%).  hot_slots
        does NOT affect output bytes (a miss re-reads identical V from the pinned
        pool) -- it is a pure VRAM<->fetch-cost knob, so any value stays bitwise.

        ``budget_frac`` = per-step selected page fraction (1-cr); ``max_pages`` =
        the longest per-unit page count (max_model_len/page). ``n_seqs`` =
        concurrent requests SHARING each (layer, kv-head) slot pool: the hot
        working set is per-REQUEST-unit, so S must scale with the batch
        (measured: S sized for one request at bs8 thrashes -- every step
        misses hundreds of pages per unit and the bounded CLOCK probe burns
        its whole budget; the v2 section-5 hot column at bs51 is likewise
        ~51x the bs1 cell). Caps at ``n_seqs * max_pages`` (never size past
        every context) and floors at ``floor``.

        BATCH (n_seqs > 1) IS R2-EXACT (bs>1 campaign, 2026-07-21): per-seq
        slots = the ATTENDED SET exactly (``budget_pages`` selectable pages +
        sink + win = budget_tokens/page under the inclusive-budget ruling;
        fraction-derived b when ``budget_pages`` is unset), NOT target_ratio*b.
        The hr-scaled formula at batch reserved working-set MULTIPLES per
        sequence and clamped to n_seqs*max_pages = bs x full-ctx KV -- measured
        hot=81.29 GiB at 32K/bs32 and 121.02 GiB at 128K/bs32 -> vLLM boot
        death "No available memory for the cache blocks"
        (results/{bmap,batch32/baseline}/logs/locks-mem_tp1_ctx32768.log) --
        AND was a hot cache larger than the attended set, the structure the
        R2 memory-exact ruling forbids.  Misses stay bitwise (the pinned-pool
        read path serves identical bytes); pool size is a pure VRAM<->fetch
        knob, so this is residency sizing only, no numerics surface.  bs=1
        (n_seqs == 1) takes the legacy branch untouched -- bit-inert by
        construction.  Kill switch LOCKS_MEM_BATCH_HOT=0 restores the legacy
        batch sizing (boot-infeasible at >=32K/bs32 on 96 GiB; kept for A/B).
        memplan.build_plan calls THIS function with THESE args -- the plan
        and the Residency alloc stay mirrored by construction."""
        selectable = max(1, int(max_pages) - sink - win)
        b = max(1, math.ceil(float(budget_frac) * selectable))
        ns = max(1, int(n_seqs))
        if ns > 1 and os.environ.get("LOCKS_MEM_BATCH_HOT", "1") == "1":
            bp = b if (budget_pages is None or int(budget_pages) <= 0) \
                else min(int(budget_pages), selectable)
            s = (bp + sink + win) * ns
        else:
            s = (math.ceil(target_ratio * b) + sink + win) * ns
        s = max(floor, s)
        return int(min(s, max(int(max_pages) * ns, floor)))

    @classmethod
    def from_config(cls, cfg, *, num_layers, num_blocks, n_kv, page, d,
                    max_reqs, max_pages, max_tokens, dtype, device) -> "Tier":
        """Build from the single LocksConfig (mem variants only).

        ``cfg.hot_slots <= 0`` requests AUTO working-set-proportional sizing
        (:meth:`recommend_hot_slots`) from the configured budget + ``max_pages``;
        any positive ``hot_slots`` is honored verbatim (the default 2048 path is
        unchanged, so every gate is byte-for-byte identical)."""
        if not cfg.is_mem:
            raise ValueError(f"Tier requires a mem variant, got {cfg.variant!r}")
        hot_slots = cfg.hot_slots
        if hot_slots is None or hot_slots <= 0:
            # coverage mode has no fixed page fraction; the measured attended
            # fraction at the flagship coverages is ~.10-.15, so size generously
            # (over-provisioning only costs a bit of VRAM, never correctness).
            eff_budget = cfg.budget if cfg.budget is not None else 0.15
            hot_slots = cls.recommend_hot_slots(
                eff_budget, max_pages, n_seqs=max_reqs,
                target_ratio=float(getattr(cfg, "hot_ratio", 6.0)),
                sink=cfg.sink_pages, win=cfg.window_pages,
                budget_pages=getattr(cfg, "budget_pages", None))
        return cls(
            num_layers=num_layers, num_blocks=num_blocks, n_kv=n_kv, page=page,
            d=d, hot_slots=hot_slots, max_reqs=max_reqs, max_pages=max_pages,
            max_tokens=max_tokens, dtype=dtype, device=device,
            v_blocks=cfg.v_blocks, max_pool_gb=cfg.max_pool_gb,
            offload_k=cfg.offload_k,
            summary_cache=bool(getattr(cfg, "mem_summary_cache", False)))

    # --------------------------------------------------------------------- #
    # Per-step host entry (OUTSIDE the graph; builder drives it).           #
    # --------------------------------------------------------------------- #
    def begin_step(self, block_table, seq_lens, query_start_loc,
                   max_query_len: int, n_tokens: int,
                   capture: bool = False) -> None:
        """Residency invalidation + staging stage/flush/alloc lifecycle. Fixed
        order: invalidate (drops stale residency/validity) THEN staging (whose
        internal order is the I4 mechanism). ``capture`` (cudagraph-capture
        dummy batches) skips BOTH mutations: the dummy block table carries
        arbitrary real block ids, so invalidating from it would drop live
        residency/validity for pages that were never rewritten."""
        # K8d: the write hook's decode proof (mirrors _forward_mem's
        # max_query_len==1 split exactly; set every step, capture included --
        # FULL-capture forwards must take the stash path so the captured
        # graph contains the SCAT-fused kernel).  _k8d_armed distinguishes
        # builder-driven steps from the PIECEWISE capture group's KV-write-
        # only passes (no metadata, no begin_step, no forward to consume a
        # stash): armed here, disarmed by the last layer's hook.
        self._cur_mql = int(max_query_len)
        self._cur_nreq = int(seq_lens.shape[0])
        self._k8d_armed = True
        if not capture:
            # K5 transport service FIRST: promote-list <- fenced CE batches,
            # then drain the mailbox and issue this round of CE transfers.
            # (Before invalidate: the promote list is consumed by the in-graph
            # promote AFTER this prologue's invalidate, and the generation
            # guard makes any interleaving safe.)
            # LOCKS_K5_EARLY: the compute_logits pre-hook already ran this
            # step's service at the POST-GRAPH host point (clock-warm, under
            # the LM-head span, zev pre-completed; register._wire_k5_early)
            # -- consume the mark instead of double-servicing.  Un-hooked
            # steps (boot, transitions) fall back to servicing here.
            if getattr(self, "_early_serviced", False):
                self._early_serviced = False
                svc = self._k5_service()
                zp = getattr(svc, "_zev_pending", None)
                if zp is not None:
                    # the deferred mailbox-zero gate: the NEXT graph's K5b
                    # must not append before the zero (same semantics as
                    # the in-service wait, taken at the true pre-graph
                    # point instead of in front of the LM head).
                    torch.cuda.current_stream(self.device).wait_event(zp)
                    svc._zev_pending = None
            else:
                self._k5_service().service()
            self.residency.invalidate_written(block_table, seq_lens,
                                              query_start_loc)
            # PROMO1 (bs>1 phase 3): apply the oldest FENCED transport batch
            # NOW (main, pre-graph), snapshot-direct -- install latency
            # 2 -> 1 step, and (running at EVERY step type of a batch
            # engine, prefill/drain included) no window-boundary batch is
            # ever left unapplied (the deployed publish path leaked its
            # owners and lost its evict there; k5.py comments).  ENGINE-
            # level gate: R == 1 keeps the deployed schedule verbatim.
            if self.residency.R > 1:
                from .k5 import _K5_PROMO1
                if _K5_PROMO1:
                    self._k5_service().consume_fenced_prologue()
        self.staging.begin_step(
            block_table, seq_lens, query_start_loc, max_query_len, n_tokens,
            pool=self.pool.dev_view, pool_valid=self.residency.pool_valid,
            capture=capture,
            poolk=self.pool_k.dev_view if self.pool_k is not None else None)
        if _ASSERT and not capture:
            self.assert_invariants(block_table, seq_lens, query_start_loc)

    def assert_invariants(self, block_table, seq_lens, query_start_loc) -> None:
        """Check I1-I4's structurally-checkable halves (host-syncing; debug)."""
        # K7: the assign kernel mutates owner/epoch/snapshot ASYNCHRONOUSLY on
        # the K5Service transfer stream (issued by this prologue's service()).
        # Host .cpu() reads below do not order against it -- drain the device
        # first so the observation point is truly settled (red-team change 3;
        # debug path only, this method is host-syncing by design).
        torch.cuda.synchronize(self.device)
        # Graph-safe valloc cannot raise; the exhaustion latch makes I1's loud
        # half a debug check (build-time assert_capacity is the static guard).
        if int(self.staging.overflow.item()) != 0:
            raise AssertionError(
                "I1 capacity: staging free-stack UNDERFLOWED (a written page got "
                "no staging block -> would lose V). Raise v_blocks to cover the "
                "concurrent-prefill peak (see Staging.assert_capacity).")
        self.staging.assert_free_partition()                       # I3 staging
        pend, incoming = self._pending_evict()
        self.residency.assert_maps_inverse(                        # I2/I3 resid
            pending_evict=pend, pending_incoming=incoming)
        self.staging.assert_no_lost_v(block_table, seq_lens,       # I1
                                      query_start_loc,
                                      self.residency.pool_valid)
        # I4 is enforced by begin_step's call order (not a state predicate).

    def _pending_evict(self) -> tuple:
        """The PENDING-EVICT set the residency checker needs: ``(l, kv, slot) ->
        blk`` for every K5 claim whose GRAPH-HEAD evict has not run yet.

        Two sources, both live at a host-synchronized observation point:
          * ``k5.elist_dev[:ecnt_dev]`` -- the batch this prologue published;
            ``_k5_evict_kernel`` applies it at the graph head, so between the
            prologue and the head it is queued-but-unapplied;
          * ``k5.mail[:mail_cnt]``     -- claims k5b appended that no prologue
            has snapshotted yet (the normal state at the END of a step, and the
            state that persists for several steps down K5Service's ``busy``
            early-return path).
        The union is exactly "an evict is pending for this slot".

        Returns ``(None, None)`` unless the DEFERRED-EVICT schedule is actually
        in use (``side_step`` ran at least once).  Under the legacy in-stream
        claim the checker therefore keeps its historical strictness verbatim:
        k5b re-points slot2page at claim time there, so no slot may ever be
        claimed while still advertising its old page.

        K7 split (sec 22c change 4): rows are returned as TWO structures --
          * ``pend``: {(l, kv, h): (blk, gen)} for rows with an ASSIGNED slot
            (h >= 0) only.  These key R4's OUTGOING exemption and the new R6
            (an assigned, current-gen row must still hold its owner claim).
            Under K7, raw-mail rows have h == -1 and hold NO claim yet, so
            they must NOT enable the R4 exemption.
          * ``incoming``: [(l, kv, blk), ...] for ALL rows (assigned or not):
            R5 (the incoming page is never advertised) applies from the
            moment of the append, because the append is what set p2s to -2.
        """
        if not self._plan_deferred:
            return None, None
        k5 = getattr(self, "_k5", None)
        if k5 is None:
            return None, None
        cap = int(k5.mail.shape[0])
        pend: dict = {}
        incoming: list = []
        srcs = [(k5.ecnt_dev, k5.elist_dev), (k5.mail_cnt, k5.mail)]
        # PROMO1 batch engines: pending rows live in the IN-FLIGHT
        # snapshots (no elist is ever published there); walk them too so
        # R4/R5/R6 see the same pending set the apply will.
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
                    pend[(int(l), int(kv), int(h))] = (int(blk), int(g))
        return pend, incoming

    # --------------------------------------------------------------------- #
    # FOLLOW-UP hooks (documented; not wired in the scaffold).              #
    # --------------------------------------------------------------------- #
    def write_kv(self, lidx, value, n_tokens, key=None) -> None:
        """In-graph write path (K+V-resident scaffold): scatter this step's V
        (and K, mem-kv) into the staging pool(s) via ``staging.v_slot_mapping``.
        Runs AFTER begin_step so the scatter targets the (re)allocated staging
        slots. The stock reshape_and_cache already wrote K (and V) resident; this
        gives the tier a byte-identical V copy to serve the tiered decode."""
        self.staging.scatter_kv(lidx, key, value, n_tokens)

    def write_kv_konly(self, lidx, key, value, kv_cache, slot_mapping) -> None:
        """In-graph K-only write path (cfg.mem_k_only): K -> engine K-only cache
        via ``slot_mapping``, V -> staging via ``v_slot_mapping`` (+ K -> k_pool
        for mem-kv), in one launch. Replaces reshape_and_cache_flash (there is no
        resident V half to write). Runs AFTER begin_step (v_slot_mapping ready).

        SUMMARY CACHE (spec v2): the engine page holds RECORDS, so K is NEVER
        written to it.  This step's K+V go to the tier staging pools ONLY
        (scatter_kv); the page's summary record is written INTO the aliased
        engine page by the finalize build (host prologue), not per token."""
        if self.summary_cache:
            self.staging.scatter_kv(lidx, key, value, slot_mapping.shape[0])
        else:
            self.staging.scatter_kv_engine(lidx, key, value, kv_cache,
                                           slot_mapping)

    def gather_k_blocks(self, lidx: int, blocks: torch.Tensor) -> torch.Tensor:
        """Summary-cache build K source (spec v2): gather COMPLETE settled
        pages' K for physical ``blocks`` from the tier, byte-identical to the
        K that used to live in the engine cache.  Off-graph (host prologue).

        Per block: STAGED (vbo>=0) -> k_pool[lidx][vbo] (page-major, direct);
        else FLUSHED -> the pinned host K pool (kv-major, transposed back).  A
        complete settled page is one or the other by I1; anything else latches
        loudly (never a silent wrong summary).  Returns (n, page, n_kv, d)."""
        assert self.offload_k and self.pool_k is not None, \
            "gather_k_blocks requires mem-kv (K in the tier)"
        dev, page, n_kv, d = self.device, self.page, self.n_kv, self.d
        blocks = blocks.to(device=dev, dtype=torch.long).reshape(-1)
        n = int(blocks.numel())
        out = torch.empty(n, page, n_kv, d, dtype=self.dtype, device=dev)
        if n == 0:
            return out
        vb = self.staging.vbo.index_select(0, blocks)          # (n,) staging slot
        staged = vb >= 0
        if bool(staged.any()):
            out[staged] = self.staging.k_pool[lidx].index_select(
                0, vb[staged].to(torch.long))
        pin = ~staged
        if bool(pin.any()):
            pb = blocks[pin].to("cpu")
            # pinned host K (L, NB, n_kv, page, d) -> (m, page, n_kv, d)
            kh = self.pool_k.host[lidx].index_select(0, pb).permute(0, 2, 1, 3)
            out[pin] = kh.contiguous().to(dev)
        return out

    def step(self, lidx, page_table, page_cnt, block_table, seq_lens,
             n_req, eager_pin: bool = False) -> None:
        """Per-layer K5 control plane feeding Stage B's hot buffer (redesign
        4.5): K5a classify+claim (atomicCAS, cross-request dedup) then K5b
        victim+install (bounded CLOCK probe, owner claim-then-verify; STAGED
        misses copied in-kernel, PINNED misses posted to the DMA mailbox for
        the copy engine; the decode reads not-yet-cached pages from
        staged/pinned IN PLACE, so every arm is bitwise-free). At layer 0 the
        promote installs the previous step's FENCED copy-engine pages.

        Replaces the deleted ``_plan_fused`` chain (serial per-kv-head program
        with a load-bearing tl.debug_barrier; 45.5 us mean / 231.8 us max per
        layer-step measured on sm_120) and ``_fetch_flat``'s SM-holding UVA
        transport of pinned pages (the F2/F3 red-team fixes)."""
        from .k5 import k5_promote, k5_step
        k5 = self._k5_state()
        if lidx == 0:
            k5_promote(self.residency, k5)
        # P10: on the lookahead side fork (eager_pin) K5b copies PINNED misses
        # host->hot in kernel a layer early, so feed it the pinned pool views.
        poolv = self.pool.dev_view if eager_pin else None
        poolk = (self.pool_k.dev_view
                 if (eager_pin and self.pool_k is not None) else None)
        k5_step(self.residency, k5, self.staging, lidx, page_table, page_cnt,
                block_table, seq_lens, n_req, poolv=poolv, poolk=poolk,
                eager_pin=eager_pin)

    def side_step(self, lidx, page_table, page_cnt, block_table, seq_lens,
                  n_req, main_stream=None) -> None:
        """P15 overlap schedule, FORK half: the K5 control plane (classify +
        PLAN) runs on a persistent SIDE stream, forked after this layer's
        selection on the main stream; the decode does NOT wait it (the plan
        mutates only k5-private state -- owner/epoch/mailbox -- plus the
        page2slot -1 -> -2 PENDING CAS, which no reader distinguishes from -1,
        so any concurrent read of page2slot/hot bytes is of step-stable,
        fully-written data).  Image mutations happen ONLY at the graph head:
        phase-1 evict (this prologue's fresh batch) then phase-2 install (the
        fenced batch).

        The caller MUST close the fork with :meth:`side_join` at the END of the
        SAME layer, after the decode that does not depend on it (decode_mem
        does this).  Fork and join being inside one layer makes the schedule
        CAPTURE-TOPOLOGY-INDEPENDENT: a per-layer capture, a piecewise capture
        and the deployed one-graph-per-step capture all see a balanced
        fork/join pair and no unjoined side work at any layer boundary.  The
        earlier shape (fork every layer, join only at the last) left non-final
        layers unjoined, which is legal in ONE full-step graph and
        ``cudaErrorStreamCaptureUnjoined`` in any finer capture.

        Overlap is preserved because the join is ordered AFTER the layer's own
        decode+merge: the side work of layer L hides under the main work of
        layer L, which is ~an order of magnitude larger per layer (the win
        never needed a multi-layer-deep pipeline, only side < main per layer).
        """
        from .k5 import _HEADV, k5_evict, k5_headv, k5_promote, k5_step
        k5 = self._k5_state()
        ms = main_stream or torch.cuda.current_stream(device=self.device)
        self._plan_deferred = True
        if lidx == 0:
            if _HEADV:                        # K10: one vectorized launch
                k5_headv(self.residency, k5, tick=False)
            else:
                k5_evict(self.residency, k5)   # phase-1 (main, graph head)
                k5_promote(self.residency, k5)  # phase-2 (existing install)
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=self.device)
            n_layers = len(self.residency.hot_i16)
            self._ev_fork = [torch.cuda.Event() for _ in range(n_layers)]
            self._ev_tail = [torch.cuda.Event() for _ in range(n_layers)]
        self._ev_fork[lidx].record(ms)
        side = self._side_stream
        with torch.cuda.stream(side):
            side.wait_event(self._ev_fork[lidx])
            k5_step(self.residency, k5, self.staging, lidx, page_table,
                    page_cnt, block_table, seq_lens, n_req, plan=True)
            self._ev_tail[lidx].record(side)

    def fa1_step(self, lidx, page_table, page_cnt, block_table, seq_lens,
                 main_stream=None) -> None:
        """FA1 fork: fetch-all-selected on the side stream, forked after
        select(L) on main; the decode RACES it (landing-or-pinned, bytes
        equal).  Join at layer end via fa1_join (capture-balanced pair, the
        proven side_step topology)."""
        from .k5_cuda import get_mod
        ms = main_stream or torch.cuda.current_stream(device=self.device)
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(device=self.device)
            n_layers = len(self.residency.hot_i16)
            self._ev_fork = [torch.cuda.Event() for _ in range(n_layers)]
            self._ev_tail = [torch.cuda.Event() for _ in range(n_layers)]
            # v3.1: fetch STREAM POOL.  fetch(L) depends only on select(L)
            # (no cross-layer edge), but one side stream SERIALIZES the 40
            # launches at the ~14us UVA launch floor (G0a) = ~0.56ms/step
            # exposed at the deep join -- the v3 smoke residual.  Round-
            # robin across NSTREAMS side streams so the floors overlap;
            # the deep join waits every tail event, order-free.
            self._fa1_streams = [torch.cuda.Stream(device=self.device)
                                 for _ in range(8)]
        self._ev_fork[lidx].record(ms)
        side = (self._fa1_streams[lidx & 7]
                if self.fa1_flags3 is not None else self._side_stream)
        r5 = self.residency
        with torch.cuda.stream(side):
            side.wait_event(self._ev_fork[lidx])
            if self.fa1_flags3 is not None:
                # v3.2: stable-slot assign (content-keyed reuse) + churn-
                # only copies into the per-layer exact-size landing.
                get_mod().fa1_step_v32(
                    page_table[0], page_cnt[0], block_table[0], seq_lens,
                    r5.pool_valid[lidx], self.pool.dev_view.data_ptr(),
                    (self.pool_k.dev_view.data_ptr()
                     if self.pool_k is not None
                     else self.pool.dev_view.data_ptr()),
                    self.fa1_land_v3[lidx], self.fa1_land_k3[lidx],
                    self.fa1_flags3[lidx], r5.gen,
                    self.fa1_slot_of[lidx], self.fa1_copy_list[lidx],
                    self.fa1_copy_cnt[lidx],
                    self.NB, r5.MP, self.n_kv, lidx, self.page,
                    self.d, self.FA1_LCAP3, self.pool_k is not None)
            else:
                _p = lidx & 1
                get_mod().fa1_fetch(
                    page_table[0], page_cnt[0], block_table[0], seq_lens,
                    r5.pool_valid, self.pool.dev_view.data_ptr(),
                    (self.pool_k.dev_view.data_ptr()
                     if self.pool_k is not None
                     else self.pool.dev_view.data_ptr()),
                    self.fa1_land_v[_p], self.fa1_land_k[_p],
                    self.fa1_flags[_p], r5.clock,
                    self.NB, r5.MP, self.n_kv, self.L, lidx, self.page,
                    self.d, self.FA1_LCAP, self.pool_k is not None)
            self._ev_tail[lidx].record(side)

    def k8_head(self, block_table=None, seq_lens=None,
                s3: bool = False) -> None:
        """K8 graph-head launch (once per step, before layer 0's decode; see
        decode_mem.py K8b): clock tick (ONE per step -- the assign recency
        window in k5.py rescales L -> 1 with it), phase-1 evict of this
        prologue's snapshotted batch, phase-2 promote of the fenced batch.
        The per-layer control plane is absorbed into the decode split kernel;
        no side stream exists under K8.  PLAN deferred-evict semantics apply
        (claims stay -2 PENDING until a later head), so the checker's R4/R5
        exemption is licensed exactly as under side_step."""
        from .k5 import _HEADV, k5_evict, k5_headv, k5_promote
        from .residency import _step_setup_kernel
        k5 = self._k5_state()
        self._plan_deferred = True
        if _HEADV:                        # K10: 3 launches -> 1 (tick folded)
            k5_headv(self.residency, k5, tick=True)
        else:
            _step_setup_kernel[(1,)](self.residency.clock,
                                     self.residency.fetch_cnt, num_warps=1)
            k5_evict(self.residency, k5)   # phase-1 (main, graph head)
            k5_promote(self.residency, k5)  # phase-2 (existing install)
        if s3:
            # S3: build the resolved step table AFTER the image mutations
            # above (they are the last p2s writers before the in-step k5c
            # claims, which the table deliberately does not see).
            from .k5_cuda import get_mod
            mod = get_mod()
            r5 = self.residency
            mod.s3_build(block_table[0], seq_lens, r5.page2slot,
                         self.staging.vbo, self.s3_tab,
                         self.L, self.n_kv, self.NB, r5.MP, self.page)

    def side_join(self, lidx, main_stream=None) -> None:
        """P15 overlap schedule, JOIN half (see :meth:`side_step`): close layer
        ``lidx``'s fork on the main stream.  Call at the END of the layer, after
        the decode+merge, so the side work overlapped it.  A no-op if this layer
        never forked."""
        if self._ev_tail is None:
            return
        ms = main_stream or torch.cuda.current_stream(device=self.device)
        if (getattr(self, "_fa1_streams", None) is not None
                and self.fa1_flags3 is not None and lidx == self.L - 1):
            # v3.1 stream pool + deep join: one tail event per POOL STREAM
            # must be waited (the single-stream transitive ordering is
            # gone).  The last fork on stream s is layer L-8+s; waiting
            # the last 8 tails covers every stream.
            for li in range(max(0, self.L - 8), self.L):
                ms.wait_event(self._ev_tail[li])
            return
        ms.wait_event(self._ev_tail[lidx])

    _side_stream = None
    _ev_fork = None
    _ev_tail = None
    _plan_deferred = False

    def _k5_state(self):
        """Lazy K5 device/pinned state (owner word, hand, mailbox, promote)."""
        k5 = getattr(self, "_k5", None)
        if k5 is None:
            from .k5 import K5State
            k5 = K5State(self.residency, self.device)
            self._k5 = k5
        return k5

    def _k5_service(self):
        """Lazy host side of the K5 transport split (mailbox -> copy engine)."""
        svc = getattr(self, "_k5_svc", None)
        if svc is None:
            from .k5 import K5Service
            svc = K5Service(self, self._k5_state())
            self._k5_svc = svc
        return svc

    def early_service(self) -> None:
        """LOCKS_K5_EARLY: run this step's K5 service at the POST-GRAPH host
        point (the compute_logits pre-hook) instead of the next build's
        prologue.  Same call, same mailbox, same event ordering (the gate
        records main AFTER the just-enqueued graph; zev's wait lands before
        the LM head, microseconds of mailbox-zero) -- only the HOST issue
        time moves, so the tr-stream kernels execute right at graph end:
        CLOCK-WARM (the droop law: the late prologue runs them ~5x slow at
        the SM floor) and OVERLAPPED under the LM-head span, with the next
        graph's zev wait pre-completed.  begin_step consumes the mark."""
        if not getattr(self, "_early_announced", False):
            self._early_announced = True
            print("[locks] K5_EARLY: first early service fired", flush=True)
        self._k5_service().service(defer_zev=True)
        self._early_serviced = True

    @property
    def copy_engine(self):
        """Lazy copy-engine DMA (double-buffered side streams)."""
        ce = getattr(self, "_copy_engine", None)
        if ce is None:
            from .dma import CopyEngine
            ce = CopyEngine()
            self._copy_engine = ce
        return ce

    def step_dma(self, lidx, page_table, page_cnt, block_table, seq_lens,
                 n_req, wait: bool = True):
        """Copy-engine variant of ``step``: plan the misses on device, then fetch
        the FLUSHED miss pages through the hardware COPY ENGINE (not the SM-holding
        UVA kernel), so the PCIe transfer overlaps decode. Returns the side-stream
        event; if ``wait`` the caller's stream waits on it (in-layer, dependent),
        else the caller overlaps it (prefetch). STAGED misses go to the kernel."""
        poolk = self.pool_k.dev_view if self.pool_k is not None else None
        self.residency.gather_plan(lidx, page_table, page_cnt, block_table,
                                   seq_lens, n_req)
        ev = self.residency.gather_fetch_dma(
            lidx, self.staging, self.pool.dev_view, n_req, self.copy_engine,
            poolk=poolk)
        if wait:
            torch.cuda.current_stream().wait_event(ev)
        return ev

    def vsource(self, lidx: int) -> "TierVSource":
        """The Stage-B KV source for layer ``lidx`` (V_SRC==1, and K_SRC==1 for
        mem-kv)."""
        return TierVSource(self, lidx)

    def bytes_report(self) -> str:
        tag = "mem-kv" if self.offload_k else "mem-v"
        kpool = (f" + K-pool {self.pool_k.pool_gib:.2f}GiB DRAM"
                 if self.pool_k is not None else "")
        return (f"[locks tier {tag}] {self.pool.vram_note()}{kpool} | "
                f"{self.residency.bytes_report()} | "
                f"{self.staging.bytes_report()}")


class TierVSource:
    """LOCKS-mem V source (mirrors ``ResidentVSource``): the ``V_SRC==1`` seam.

    Stage B consumes this exactly like the fast path's ResidentVSource -- same
    ``SRC_ID`` + ``args()`` protocol -- so ``attention/decode.py`` stays tier-
    blind. The tier's 3-way V read (hot buffer | staged v_pool | pinned pool) is
    a compile-time ``V_SRC==1`` branch inside the decode kernel; this object
    supplies the pointer UNION that branch needs.

    SCAFFOLD. ``args()`` returns the staging ``v_pool`` layer view as the primary
    base so the shared decode wrapper's ``v_ptr, svb, svt, svh = vsource.args()``
    line is satisfied and addresses valid memory (the current V_SRC==1 arm is a
    documented resident-shaped fallback). The FULL union the follow-up kernel
    consumes is exposed by ``union()``; wiring the extra pointers into the
    kernel's V_SRC==1 arm is the follow-up. Until then mem-v decode delegates to
    stock FA (backend/attn.py), so ``args()`` is never on a live tiered path.
    """

    SRC_ID = 1

    def __init__(self, tier: Tier, lidx: int):
        self._tier = tier
        self._lidx = lidx
        self._vp = tier.staging.v_pool[lidx]        # (NV, page, n_kv, d)
        # mem-kv: K is offloaded too -> the decode kernel's K_SRC==1 arm reads K
        # from the tier (staged k_pool | hot_k | pinned K pool). mem-v: K_SRC=0.
        self.K_SRC = 1 if tier.offload_k else 0

    def args(self) -> Tuple[torch.Tensor, int, int, int]:
        # Primary base = staging v_pool layer view (the STAGED source and the
        # decode kernel's ``v_ptr``); block/token/head strides match resident V.
        vp = self._vp
        return vp, vp.stride(0), vp.stride(1), vp.stride(2)

    def k_args(self) -> Tuple[torch.Tensor, int, int, int]:
        """mem-kv K base = the staging k_pool layer view (the STAGED K source and
        the decode kernel's ``k_ptr`` under K_SRC==1); same paged layout as V."""
        kp = self._tier.staging.k_pool[self._lidx]
        return kp, kp.stride(0), kp.stride(1), kp.stride(2)

    def k_tier_ptrs(self):
        """The K-side (hot_k, pinned-K-pool) pointers the K_SRC==1 arm consumes
        (page2slot/vbo/pool_valid are SHARED with V -> already in tier_ptrs)."""
        t = self._tier
        return t.residency.hot_k[self._lidx], t.pool_k.dev_view

    def tier_ptrs(self):
        """The extra pointers the ``V_SRC==1`` decode arm consumes beyond
        ``args()``: (hot, page2slot, vbo, pinned-pool, lidx, NB, S).

          * hot        r.hot[lidx]        bf16 (n_kv, S, page, d)  -- resident hits
          * page2slot  r.page2slot[lidx]  int32 (n_kv, NB)         -- hot addressing
          * vbo        s.vbo              int32 (NB,)              -- staged addressing
          * pinned     pool.dev_view      int16 UVA device ptr     -- flushed, zero-copy
        The kernel keys the 3-way select on ``page2slot>=0`` (+ page complete),
        ``vbo>=0``, else the pinned pool at ``page_offset(lidx, blk, kv)``."""
        t = self._tier
        l = self._lidx
        return (t.residency.hot[l], t.residency.page2slot[l], t.staging.vbo,
                t.pool.dev_view, l, t.NB, t.residency.S)

    def union(self) -> dict:
        """Full pointer union for the FOLLOW-UP V_SRC==1 decode load.

        The kernel selects, per (block, kv-head, token), among:
          * hot buffer     hot[lidx]      via page2slot[lidx][blk]  (complete+resident)
          * staged v_pool  v_pool[lidx]   via vbo[blk]              (partial/unflushed)
          * pinned pool    pool.dev_view  at pool.page_offset(...)  (flushed, zero-copy)
        keyed by ``pool_valid[lidx]`` + ``vbo``. See _p2_sparse_decode_split_kernel
        in vllm/kernels/dram_tier.py for the exact selection + masking.
        """
        t = self._tier
        l = self._lidx
        r, s = t.residency, t.staging
        return dict(
            lidx=l, NB=t.NB, S=r.S, n_kv=t.n_kv, page=t.page, d=t.d,
            hot=r.hot[l], hot_i16=r.hot_i16[l],
            page2slot=r.page2slot[l], pool_valid=r.pool_valid[l],
            v_pool=s.v_pool[l], vbo=s.vbo,
            pool=t.pool.dev_view, pool_base=t.pool.page_offset(l, 0, 0),
        )
