"""LocksMetadataBuilder -- FlashAttentionMetadataBuilder + persistent LOCKS state.

Owns the r8 Stage-A lifecycle:

  * Allocates ``R8State`` ONCE (so every buffer address is stable across
    CUDA-graph replays), LAZILY at the first build() where the KV cache is
    allocated -- ``R8State`` is sized from ``cache_config.num_gpu_blocks``, which
    is unknown at builder __init__ (same reason the DRAM tier defers to
    ``_ensure_dram`` at first build). At that first build it also keys a
    ``kv_cache.data_ptr() -> physical layer index`` map from
    ``compilation_config.static_forward_context`` (the exact tensor object the
    impl.forward receives), so the impl's layer-index-free select call can find
    the right L-major slab.

  * Runs the eigh ``r8_build_refresh`` OUTSIDE the graph, in build(), per layer,
    over that layer's resident K half -- tag-gated so only newly-finalized pages
    rebuild (steady-state cost = one tag compare/layer). Skipped on cudagraph
    capture dummy batches (``build_for_cudagraph_capture`` sets ``_capturing``),
    mirroring the DRAM tier's ``begin_step(capture=...)``.

  * Refreshes the per-step derived selection params (``derive_page_params``) once
    per step (graph-safe, one launch) and attaches the state as ``md.locks`` for
    the impl to consume.

  * PREFILL-OVERLAPPED SUMMARY BUILD (:class:`_PrefillOverlap`). ``build()`` is a
    PRE-forward hook, so at prefill time the KV does not exist yet and nothing
    could be summarized; the whole prompt therefore used to be built on the
    FIRST DECODE STEP, on the main stream, inside TTFT (the measured 737 ms
    first-decode sync-storm; ~45 ms @8K / ~224 ms @64K for clse even after the
    batched bulk rewrite). But ``summary(layer L)`` depends ONLY on layer L's K,
    which exists the instant layer L's prefill attention completes, and NOTHING
    during prefill ever reads a summary (they are consumed only by the decode
    score kernel). So the build is fully hideable under prefill compute (212 ms
    @8K ... 10.9 s @128K). ``build()`` now plans, per prefill/mixed step, exactly
    the pages that step finalizes (host-side, sync-free, from the CPU seq lens +
    query_start_loc), and a per-layer post-attention hook issues that layer's
    build on a SIDE CUDA STREAM. The next decode ``build()`` fences
    (``current_stream().wait_event``) before anything reads a summary tag or
    code. See :class:`_PrefillOverlap` for the full ordering argument.

CUDA-graph support: with the real graph-safe Stage A the pure-decode pipeline is
fixed-shape / allocation-free / host-sync-free, so the builder declares
``UNIFORM_SINGLE_TOKEN_DECODE`` -> pure-decode batches replay as FULL graphs
(select + decode INSIDE the graph, zero python per step). The eigh build never
runs inside the graph; only the persistent r8/derived buffers it refreshes
outside the graph are read at replay. The eager validation bridge (torch topk ->
not capturable) forces ``NEVER`` -> attention runs eagerly between graph pieces.
The side-stream build is likewise outside the graph: the fence is an event wait
issued on the main stream BEFORE the replay, and no plan is ever attached on a
capture batch, so capture is never polluted by a cross-stream dependency.
"""
from __future__ import annotations

import os
import traceback

import torch

# Profiling-only NVTX ranges (LOCKS_NVTX=1): attribute the per-step host
# prologue (FA build / refresh gate / derive) on an nsys timeline. Zero-cost
# when off (module-level constant, no per-step env read).
_NVTX = os.environ.get("LOCKS_NVTX", "0") == "1"
_nvtx_push = torch.cuda.nvtx.range_push
_nvtx_pop = torch.cuda.nvtx.range_pop

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder

from ..attention.decode import _split_kv, ensure_stage_b_buffers
from . import _runtime


class _TierKLayer:
    """Duck-typed per-layer K view for the P5 summary-only engine cache.

    Under ``mem_summary_cache`` the engine page holds summary RECORDS, so the
    build can no longer read a resident K half.  This proxy makes the UNCHANGED
    build code (``K[blocks]`` and ``K[blocks, page-1, :, :tagw]``) gather K from
    the tier (staged k_pool | pinned pool_k) instead, byte-identical to the old
    engine K.  Off-graph (host prologue), only on finalize/refresh steps."""

    __slots__ = ("_tier", "_lidx")

    def __init__(self, tier, lidx: int):
        self._tier, self._lidx = tier, lidx

    @property
    def shape(self):
        """Engine-K-shaped (NB, page, n_kv, d) — the build's bound checks
        (``_mask_blocks``) and gathers see the same geometry the resident
        half used to expose."""
        t = self._tier
        return (t.NB, t.page, t.n_kv, t.d)

    @property
    def device(self):
        return self._tier.device

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            pages = self._tier.gather_k_blocks(self._lidx, idx[0])
            return pages[(slice(None),) + tuple(idx[1:])]
        return self._tier.gather_k_blocks(self._lidx, idx)


def _build_record_arena(layer_kv, L: int):
    """Return the single contiguous engine buffer spanning all L layers as a
    flat bf16 tensor in LAYER-MAJOR order, for QuadState.rec to alias (spec v2)
    -- OR ``None`` when the engine layer tensors are SEPARATE storages (this
    vLLM stack: each layer is its own allocation, only adjacent in memory), in
    which case QuadState keeps a separate arena (K still leaves VRAM; only the
    ~9 GiB fold-in is deferred).  Never corrupts: it only aliases when the L
    tensors provably share ONE storage laid out layer-major."""
    t0 = layer_kv[0]
    per = t0.numel()
    per_bytes = per * t0.element_size()
    base = t0.data_ptr()
    storage = t0.untyped_storage()
    ok = (storage.nbytes() >= L * per_bytes)
    for i, t in enumerate(layer_kv):
        if (t.numel() != per or not t.is_contiguous()
                or t.data_ptr() != base + i * per_bytes
                or t.untyped_storage().data_ptr() != storage.data_ptr()):
            ok = False
            break
    if not ok:
        print("[locks] summary_cache: engine KV layers are SEPARATE storages "
              "-> QuadState keeps a separate record arena (aliasing fold-in "
              "deferred; K still leaves VRAM)", flush=True)
        return None
    full = t0.new_empty(0)
    full.set_(storage, t0.storage_offset(), (L * per,))
    return full


def _seq_lens_host(cam):
    """Per-step HOST (CPU) seq lens, sync-free, WITHOUT the deprecated
    ``seq_lens_cpu`` property.

    vLLM 0.24 decorates ``CommonAttentionMetadata.seq_lens_cpu`` with
    ``@typing_extensions.deprecated``; inside the engine that access pays the
    warnings machinery EVERY step -- measured ~6.9 ms/decode-step (nsys +
    perf_counter, scratch_deepopt3), the single largest LOCKS host CPU cost.
    The runner populates the same pinned CPU tensor in the plain (non-property)
    field ``seq_lens_cpu_upper_bound`` (exact outside async spec decode, which
    never reaches the pure-decode LOCKS path); ``_seq_lens_cpu`` is the
    deprecated-path backing store. Read those directly; fall back to None (gate
    then refreshes every step, correct but slow)."""
    sl = getattr(cam, "seq_lens_cpu_upper_bound", None)
    if sl is None:
        sl = getattr(cam, "_seq_lens_cpu", None)
    return sl

# split=128: measured Stage-B bandwidth knee at bs=1 (2064 vs 1674 GB/s @
# split64) with no bs=4 regression (pps floors at PPS_MIN so the active-split
# count is unchanged there).
_SPLIT = 128


# --------------------------------------------------------------------------- #
# Prefill-overlapped summary build.  DEFAULT OFF -- and the measurement says so. #
# --------------------------------------------------------------------------- #
# The summary build was expected to be "fully hideable" under prefill compute.
# It is not, and the reason is that prefill compute is BUSY, not IDLE.  Measured
# (Llama-3.1-8B, bs=1, clse, budget .1, full cudagraphs, sequential same-GPU runs
# on an RTX PRO 6000 Blackwell; scratch_prefill_overlap/OVERLAP_REPORT.md).  Note
# t(1) is PREFILL ONLY -- vLLM samples token 1 off the prefill forward -- so the
# serialized build lives in t(2)-t(1), the FIRST DECODE STEP:
#
#                          t(1) TTFT      t(2)-t(1)        t(2)
#   8K   1 chunk   ser      383.8          42.6           425.4
#   8K   1 chunk   overlap  408.8 (+25.0)  19.2 (-23.4)   427.7 (+2.2)
#   8K   4 chunks  ser      390.4          37.6           427.8
#   8K   4 chunks  overlap  412.4 (+21.9)  12.9 (-24.7)   425.4 (-2.4)
#   32K  default   ser     2244.3         110.1          2357.1
#   32K  default   overlap 2334.8 (+90.5)  25.0 (-85.1)  2359.5 (+2.4)
#
# The overlap removes the build from the first decode step exactly as designed
# (with chunked prefill, completely: 12.9 ms == the dense first-decode step).
# But the prefill forward absorbs ~NONE of it: Delta(t(1)) == -Delta(first decode)
# to within 2.4 ms at every point, so t(2) -- the user-visible total -- is
# CONSERVED.  A saturated GPU cannot hide work; concurrency just time-slices it.
#
# So this is an INTER-TOKEN-LATENCY lever, not a total-latency one: it converts a
# per-request 25 ms (8K) / 97 ms (32K) stall at token 2 into an equal addition to
# TTFT.  Since LOCKS's headline claim is "prefill is untouched, TTFT = parity",
# the default stays OFF, which is byte-for-byte the legacy path.  Set
# LOCKS_PREFILL_OVERLAP=1 when p99 inter-token latency matters more than TTFT.
#
# LOCKS_PREFILL_OVERLAP=1   enable the side-stream prefill build
# LOCKS_OVERLAP_DELAY_MS=X  inject an X-ms GPU stall at the head of every
#                           side-stream fire.  RACE TEST ONLY: with the fence in
#                           place the decode still reads complete summaries (just
#                           later); it exists so the fence can be SHOWN to be
#                           load-bearing.
# LOCKS_OVERLAP_NO_FENCE=1  drop the event wait.  RACE TEST ONLY -- the decode is
#                           then allowed to read a half-written summary.  Never
#                           set this outside scratch_prefill_overlap.
_OVERLAP = os.environ.get("LOCKS_PREFILL_OVERLAP", "0") == "1"
# LOCKS_FORCE_FULL_REFRESH=1 -- DIAGNOSTIC ONLY (2026-07-22). Bypasses the
# refresh-skip gate entirely and rebuilds every settled page of every row from
# the live K on every step: the ``src/locks`` stateless-summary reference,
# realized inside this plugin so a lifecycle A/B holds the estimator, the
# kernels, the selection chain and the GQA combine constant. Not a shipping
# path (it reinstates the sync storm the gate exists to remove); it exists so
# "is the incremental lifecycle guilty?" is a one-flag experiment.
_FORCE_FULL = os.environ.get("LOCKS_FORCE_FULL_REFRESH", "0") == "1"
_OV_DELAY_MS = float(os.environ.get("LOCKS_OVERLAP_DELAY_MS", "0") or 0.0)
_OV_NO_FENCE = os.environ.get("LOCKS_OVERLAP_NO_FENCE", "0") == "1"
# torch.cuda._sleep() counts SM clock cycles; 1.5 GHz is a safe under-estimate
# on every GPU we run (so the requested delay is a LOWER bound, which is all a
# race test needs).
_OV_DELAY_CYCLES = int(_OV_DELAY_MS * 1e-3 * 1.5e9)
_OV_CHUNK_PAIRS = 8192          # == selection.build._build_blocks' default


class _PrefillOverlap:
    """Issue each layer's page-summary build on a side stream, during prefill.

    ORDERING ARGUMENT (why this is safe):

    * WRITE-AFTER-WRITE (K) -- ``summary(L)`` reads only layer L's resident K
      half, and only at physical blocks whose pages are ALREADY FINALIZED as of
      this step (``(p+1)*page <= seq_len``).  Later prefill chunks append tokens
      to strictly later slots, so the main stream never rewrites a page the side
      stream is summarizing.  Slot/block REUSE (preemption, request eviction)
      can, and is caught by the content-tag rebuild (see below).
    * READ-AFTER-WRITE (K) -- the side stream must not read layer L's K before
      the main stream wrote it.  ``_layer_done`` runs after ``impl.forward``
      returned for layer L, i.e. after the KV-cache update and the attention of
      that layer were ENQUEUED on the main stream; ``side.wait_stream(main)``
      turns that program order into a device-side dependency.
    * READ-AFTER-WRITE (summary) -- NOTHING in a prefill/mixed forward reads a
      summary (attn.py delegates every non-pure-decode batch to stock FA).  The
      first reader is the next decode ``build()``: the delta tag scan, then
      ``derive`` + the in-graph score kernel.  ``fence()`` at the top of that
      build issues ``current_stream().wait_event(ev)``, so every side-stream
      write is visible to everything the main stream does afterwards, including
      a CUDA-graph replay (the wait is a main-stream node enqueued before the
      replay, never inside it).
    * ALLOCATOR -- the physical-block index tensor is allocated on the MAIN
      stream (it must observe this step's block-table H2D) and consumed on the
      side stream, so it is ``record_stream``-ed.  Everything else the build
      allocates is born and freed on the side stream; the summary slabs are
      persistent (``R8State`` / ``QuadState`` alloc-once) and never freed.

    COVERAGE.  The overlap covers exactly the pages a prefill/mixed step
    finalizes.  Whatever it misses -- a prefix-cache hit (those pages were never
    written by this engine step), a preemption resume, block reuse, a layer whose
    hook did not fire -- keeps its stale/NaN content tag, so the first decode
    step's ``_delta_refresh`` tag scan rebuilds it.  The happy path degenerates
    to a pure tag pass with zero rebuilt blocks.
    """

    __slots__ = ("_dev", "_stream", "_event", "_pending", "_plan", "_seen",
                 "_hooked", "_impls", "_write_fn", "_layer_k", "_state",
                 "_lg", "_grouped", "n_steps", "n_fires", "n_pages_built",
                 "n_rebound")

    def __init__(self, device):
        self._dev = device
        self._stream = torch.cuda.Stream(device=device)
        self._event = torch.cuda.Event()
        self._pending = False
        self._plan = None          # blocks int64 (P,) device tensor
        self._seen = None          # per-layer "already fired this step" flags
        self._hooked = False
        self._impls = []           # (impl, original bound forward) for teardown
        self._write_fn = None
        self._layer_k = None
        self._state = None
        self._lg = 1               # layers per fire (== _build_blocks' `lg`)
        self._grouped = True
        self.n_steps = 0
        self.n_fires = 0
        self.n_pages_built = 0
        self.n_rebound = 0         # impls re-targeted from a dead overlap

    # ---- wiring --------------------------------------------------------- #
    def bind(self, state, layer_k, write_fn):
        self._state, self._layer_k, self._write_fn = state, layer_k, write_fn
        self._seen = [False] * len(layer_k)

    def install(self, sfc, layer_names) -> int:
        """Wrap every LOCKS layer's ``impl.forward`` with a post-attention hook.

        The hook has to fire per LAYER, and ``build()`` (the only metadata hook
        vLLM offers) fires once per STEP -- so the callback is installed on the
        impl object itself.  ``impl.forward`` is invoked from inside the
        ``unified_attention_with_output`` CUSTOM OP (``self.impl.forward(self,
        q, k, v, kv_cache, md, output=...)``), which is opaque to dynamo: a
        per-instance wrapper is therefore invisible to torch.compile and cannot
        graph-break.  (An ``nn.Module`` forward hook on the ``Attention`` module
        would NOT be: dynamo inlines module hooks into the traced graph.)

        REBIND-ON-REINSTALL (2026-07-17, approved fix; SELECT_KERNEL_
        CAMPAIGN.md section 9): vLLM constructs the metadata builder TWICE
        (profiling-phase cache config, then the real one).  The wrapper
        therefore resolves its target overlap through ``impl._locks_ov_target``
        AT CALL TIME instead of closing over the installing instance; a later
        builder's install() re-points that attribute at ITSELF (the overlap
        bound to the LIVE state/slabs) and counts the impl as installed.
        Before this, the second install() found ``_locks_ov_wrapped`` and
        skipped -> returned 0 -> the second builder kept ``_ov=None`` (the
        "NO layer hooks installed" branch) while the live hooks pointed at
        the dead first-builder overlap whose plan never arms: the opt-in
        overlap silently degraded to the serialized first-decode build."""
        if self._hooked:
            return len(self._impls) + self.n_rebound
        for lidx, name in enumerate(layer_names):
            mod = sfc.get(name)
            impl = getattr(mod, "impl", None)
            if impl is None:
                continue
            if getattr(impl, "_locks_ov_wrapped", False):
                impl._locks_ov_target = self       # re-target the live overlap
                self.n_rebound += 1
                continue
            orig = impl.forward

            def _wrapped(*args, _orig=orig, _l=lidx, _impl=impl, **kw):
                out = _orig(*args, **kw)
                tgt = getattr(_impl, "_locks_ov_target", None)
                if tgt is not None:
                    tgt._layer_done(_l)
                return out

            impl.forward = _wrapped
            impl._locks_ov_wrapped = True
            impl._locks_ov_target = self
            self._impls.append((impl, orig))
        self._hooked = True
        return len(self._impls) + self.n_rebound

    def uninstall(self) -> None:
        """Restore the unwrapped impls (tests / teardown; never in serving).
        Restores and clears targets only for impls THIS instance originally
        wrapped (rebound impls are owned by the first wrapper's records)."""
        for impl, orig in self._impls:
            try:
                del impl.forward
            except AttributeError:
                impl.forward = orig
            impl._locks_ov_wrapped = False
            impl._locks_ov_target = None
        self._impls = []
        self._hooked = False

    # ---- per-step plan --------------------------------------------------- #
    def plan(self, blocks) -> None:
        """Arm the hooks with THIS step's newly-finalized physical blocks.

        The layers are fired in the SAME groups ``selection.build._build_blocks``
        uses (``lg = chunk_pairs // P`` layers per math chain when the block
        count is small, single-layer chunk slices when it is large), for two
        reasons.  (1) BITWISE: the eigh / gemm batch shape is then identical to
        the serialized bulk build, so the summaries are byte-identical rather
        than merely equivalent.  (2) HOST COST: one fire per group instead of one
        per layer.  Measured on this geometry (L=32, clse, 8K), a per-layer fire
        costs 29.2 ms of HOST launch time against the grouped build's 1.6 ms --
        which is pure added latency on the prefill critical path, since the host
        must launch it between the layers it is meant to hide under.
        """
        self._plan = blocks
        P = int(blocks.shape[0])
        self._grouped = P <= _OV_CHUNK_PAIRS
        self._lg = max(1, _OV_CHUNK_PAIRS // P) if self._grouped else 1
        if self._seen is not None:
            for i in range(len(self._seen)):
                self._seen[i] = False
        self.n_steps += 1

    def disarm(self) -> None:
        self._plan = None

    @property
    def armed(self) -> bool:
        return self._plan is not None

    # ---- the side-stream build ------------------------------------------ #
    def _layer_done(self, lidx: int) -> None:
        """Post-attention hook for layer ``lidx``: fire this layer's GROUP once
        its last member's K has been written."""
        blocks = self._plan
        if blocks is None:                       # decode / capture / no work
            return
        seen = self._seen
        L = len(seen)
        if lidx >= L or seen[lidx]:              # defensive: one fire/step/layer
            return
        if torch.cuda.is_current_stream_capturing():
            return                               # never inside a graph capture
        seen[lidx] = True
        lg = self._lg
        l0 = (lidx // lg) * lg
        end = min(l0 + lg, L)
        if lidx != end - 1:
            return                               # group not complete yet
        if _NVTX:
            _nvtx_push("locks.ov_build")
        side = self._stream
        # RAW on K: everything the main stream enqueued up to here (this group's
        # KV-cache updates + attentions) must complete before we read its K.
        side.wait_stream(torch.cuda.current_stream())
        from ..selection.build import _scatter_index
        st, write_fn = self._state, self._write_fn
        P = blocks.shape[0]
        with torch.cuda.stream(side):
            if _OV_DELAY_CYCLES:
                torch.cuda._sleep(_OV_DELAY_CYCLES)
            # Both arms mirror selection.build._build_blocks EXACTLY (same
            # grouping, same slicing, same write_fn, same fp32 op chain) -- the
            # only difference from the serialized path is the stream it runs on,
            # which is why the summaries come out byte-identical.
            if self._grouped:
                Ks = self._layer_k[l0:end]
                Kp = torch.stack([K[blocks] for K in Ks])
                li, bi = _scatter_index(len(Ks), l0, blocks)
                write_fn(st, Kp.reshape(len(Ks) * P, st.page, st.n_kv, st.d),
                         li, bi)
            else:
                K = self._layer_k[l0]
                for s0 in range(0, P, _OV_CHUNK_PAIRS):
                    bs_ = blocks[s0:s0 + _OV_CHUNK_PAIRS]
                    li, bi = _scatter_index(1, l0, bs_)
                    write_fn(st, K[bs_], li, bi)
        # `blocks` was born on the main stream; the caching allocator must not
        # hand its memory back until this side-stream work has retired.
        blocks.record_stream(side)
        self._event.record(side)
        self._pending = True
        self.n_fires += 1
        self.n_pages_built += P * (end - l0)
        if _NVTX:
            _nvtx_pop()

    # ---- the fence ------------------------------------------------------- #
    def fence(self) -> bool:
        """Main stream waits for every side-stream summary write issued so far.

        Returns True when a fence was actually issued.  A single wait on the
        (re-recorded) event covers ALL prior builds: the side stream is in-order,
        so the last record dominates.  Never issued while the current stream is
        capturing -- a capture batch never arms a plan, so nothing can be
        pending there; the assert makes that invariant loud rather than silent.
        """
        if not self._pending:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "locks prefill-overlap: side-stream build pending while the "
                "current stream is CAPTURING -- a plan leaked into a cudagraph "
                "capture batch. Refusing to pollute the graph.")
        self._pending = False
        if _OV_NO_FENCE:               # race test: deliberately unsafe
            return False
        torch.cuda.current_stream().wait_event(self._event)
        return True

    def drain(self) -> None:
        """Hard host-side drain (teardown / before capture)."""
        if self._pending and not torch.cuda.is_current_stream_capturing():
            self._stream.synchronize()
            self._pending = False


def _host_prev_cur(cam, n_req):
    """(prev, cur) HOST int64 token counts per row: computed-before-this-step and
    computed-after.  Sync-free -- every source is a CPU tensor the runner already
    maintains.  Returns (None, None) when no host view exists (the builder then
    keeps the legacy serialized first-decode build)."""
    qs = getattr(cam, "query_start_loc_cpu", None)
    if qs is None or qs.shape[0] < n_req + 1:
        return None, None
    qlen = (qs[1:n_req + 1] - qs[:n_req]).to(torch.int64)
    sl = _seq_lens_host(cam)
    if sl is not None and sl.shape[0] >= n_req:
        cur = sl[:n_req].to(torch.int64)
        return (cur - qlen).clamp_min(0), cur
    nct = getattr(cam, "_num_computed_tokens_cpu", None)
    if nct is not None and nct.shape[0] >= n_req:
        prev = nct[:n_req].to(torch.int64)
        return prev, prev + qlen
    return None, None


def _finalized_block_plan(block_table, prev, cur, n_req, page, device):
    """Physical blocks whose page FINALIZES on this (prefill/mixed) step.

    Pure host arithmetic -> ONE small H2D + one gather; no device sync, so the
    prefill forward's launch pipeline is never stalled.  For a row with ``prev``
    computed tokens before this step, the pages that become fully written now are
    ``[prev // page, cur // page)``: ``prev // page`` is the page that was PARTIAL
    before the step (or the first fresh one when ``prev`` is page-aligned), and
    ``cur // page`` is the first page still partial after it.  Decode rows in a
    mixed batch fall out of the same formula (a one-page range exactly when
    ``cur % page == 0``).  Nothing outside ``[0, cur // page)`` is ever emitted,
    so only FINALIZED pages -- every token of which is now in the cache -- are
    summarized, which is the same precondition the serialized bulk build has.

    Pages the engine did NOT write this step (prefix-cache hits) are correctly
    excluded here and are picked up by the first decode step's tag scan.

    The gather runs on the caller's (main) stream, so it observes this step's
    block-table H2D; the result is a PRIVATE tensor, which is what lets the side
    stream read it later without racing the next step's block-table upload.
    """
    lo = prev // page
    hi = cur // page
    cnt = (hi - lo).clamp_min_(0)
    total = int(cnt.sum())
    if total == 0:
        return None
    rows = torch.repeat_interleave(torch.arange(n_req, dtype=torch.int64), cnt)
    # offset of each (row, page) pair inside its own row's range
    starts = torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    pages = torch.repeat_interleave(lo, cnt) + (torch.arange(total) - starts)
    stride = block_table.shape[1]
    flat = (rows * stride + pages).to(device, non_blocking=True)
    return block_table.reshape(-1)[flat].to(torch.int64)


class LocksMetadataBuilder(FlashAttentionMetadataBuilder):
    """FA builder + persistent LOCKS decode state + per-step param refresh."""

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        cfg = _runtime.get_config()
        _State, _derive, _select, graph_safe = _runtime.resolve_selection()
        # Both fast and mem pure-decode pipelines are fixed-shape, allocation-
        # free and host-sync-free inside the graph: fast = select + decode; mem =
        # select + tier.step gather (fetch="kernel", graph-safe) + tiered decode +
        # the in-graph V write scatter. All persistent buffers are refreshed
        # OUTSIDE the graph (the r8/quad summary in build(), the tier lifecycle in
        # begin_step). test_mem_tier_graph_capture proves the mem gather+decode
        # replays bitwise. So mem claims FULL graphs too (was PIECEWISE purely by
        # the variant=="fast" policy gate).
        if cfg is not None and graph_safe and not cfg.force_eager \
                and (cfg.variant == "fast" or cfg.is_mem):
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._cfg = _runtime.get_config()
        self._device = device
        self._vllm_config = vllm_config
        self._layer_names = list(layer_names)
        self._state = None
        self._derive = None
        self._r8 = False          # True => real graph-safe r8 path
        self._capturing = False   # set during cudagraph-capture dummy builds
        self._layer_kv = None     # per-layer resident kv_cache tensor views
        self._layer_k = None      # per-layer resident K-half views (tail build)
        self._prev_seq_lens = None  # host seq_lens snapshot (refresh-skip gate)
        self._prev_settled = None   # host SETTLED (seq_lens - query_len) snapshot
        self._built_once = False  # first full build -> bulk; later -> tag delta
        self._tier = None         # locks.tier.Tier (mem variants; lazy)
        self._tier_tried = False
        self._ov = None           # _PrefillOverlap (lazy, with the state)
        self._lease = None        # PrefillLease (mem variants; transient)
        # P6: pinned+device ping-pong pair for the tail-refresh row indices
        # (lazy; see _rows_to_device).
        self._rows_pin = None
        self._rows_dev = None
        self._rows_par = 0
        # mem + leased chunked prefill assumes at most ONE mid-prefill request
        # (one lease) and no prefix-cache resurrections. Assert at init, loudly.
        if self._cfg is not None and self._cfg.is_mem:
            sched = vllm_config.scheduler_config
            mpp = int(getattr(sched, "max_num_partial_prefills", 1) or 1)
            if mpp != 1:
                raise RuntimeError(
                    f"locks mem: max_num_partial_prefills={mpp} unsupported "
                    "(the prefill lease covers exactly one mid-prefill request)")
            if bool(getattr(vllm_config.cache_config,
                            "enable_prefix_caching", False)):
                raise RuntimeError(
                    "locks mem: enable_prefix_caching is unsupported (a cache "
                    "hit resumes with prior context the lease never saw)")
            # mns>1 FENCE LIFTED (2026-07-20, GH200 live-engine resolution):
            # the historical "V-corruption" was a PROTOCOL ARTIFACT of the
            # fast-vs-mem matched-mns gate -- at MIXED prefill+decode steps
            # the FAST arm delegates the whole batch to stock DENSE
            # attention (impl guard max_query_len != 1) while mem batch-
            # splits and stays SPARSE, so the two arms compute different
            # attended sets exactly at those windows and their tokens
            # legitimately diverge (first decode of the first finisher =
            # the observed req-0/position-1 signature).  Mem itself is
            # PROVEN consistent by three independent identities on this
            # box: mem-mns1 vs mem-mns4 TOKEN-IDENTICAL (the corrected
            # self-consistency gate), budget=1.0 fast-vs-mem TOKEN-
            # IDENTICAL (tier bytes exact), and I1-I4 asserts green under
            # LOCKS_TIER_ASSERT.  The control confirms the mechanism:
            # fast-mns1 vs fast-mns4 diverges the same way (fast changed
            # function, not mem).  Regression gate going forward =
            # scratch_mem_r8i4/sys_eq_amass.py compared MEM-vs-MEM across
            # mns (self-consistency), not fast-vs-mem.  The fast arm's
            # mixed-window dense delegation stands recorded as a FAST-arm
            # semantic inconsistency (paper note).
            pass
        # mem-v/mem-kv: the DRAM tier is built lazily at first build (needs
        # num_gpu_blocks) and its per-step stage/flush lifecycle runs OUTSIDE the
        # graph in build(), exactly like the r8 refresh. The mem path runs the
        # SAME graph-safe Stage-A selection state as fast (the tiered decode
        # consumes st.page_table/page_cnt identically); it is set up below, and
        # the tier is set up lazily in _ensure_tier. mem-v (K resident) builds
        # the quad/clse summary from the resident K half exactly like fast.
        if self._cfg is None:
            return
        if self._cfg.is_mem:
            print("[locks] mem tier WIRED; Tier + Stage-A state deferred to "
                  "first build (num_gpu_blocks not set yet)", flush=True)
            # fall through to the shared Stage-A setup.

        State, derive, _select, graph_safe = _runtime.resolve_selection()
        self._r8 = graph_safe
        self._derive = derive
        if self._r8:
            # R8State is sized from num_gpu_blocks (unknown until the cache is
            # allocated) and needs the per-layer kv_cache tensors -> defer the
            # whole construction to the first build() (see _ensure_state).
            print("[locks] r8 selection WIRED (graph-safe); R8State + layer map "
                  "deferred to first build (num_gpu_blocks not set yet)",
                  flush=True)
            return

        # --- bridge fallback: no num_blocks needed, construct now -------------
        try:
            max_reqs = vllm_config.scheduler_config.max_num_seqs
            max_model_len = vllm_config.model_config.max_model_len
            bs = self.block_size
            max_pages = (max_model_len + bs - 1) // bs
            n_kv = self.num_heads_kv
            G = self.num_heads_q // n_kv
            self._state = State(device, self._cfg, max_reqs, n_kv, G,
                                self.headdim, max_pages, _SPLIT)
            ensure_stage_b_buffers(self._state, device, _SPLIT)
            print(f"[locks] BRIDGE buffers: reqs={max_reqs} n_kv={n_kv} G={G} "
                  f"d={self.headdim} max_pages={max_pages} split={_SPLIT} "
                  f"graph_safe={graph_safe}", flush=True)
        except Exception as e:
            # NO-FALLBACK RULE: _state = None here degrades every decode step
            # to stock FA under the locks tag (see _ensure_state).
            print("[locks] bridge buffer alloc FAILED "
                  "-- refusing the silent stock-FA/FullKV delegation:",
                  flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise RuntimeError(
                "locks: bridge Stage-A buffer allocation failed "
                f"({type(e).__name__}: {e}). No FullKV fallback under the "
                "locks tag -- run LOCKS_DISABLE=1 for the reference line."
            ) from e

    # --------------------------------------------------------------------- #
    # Lazy R8State construction (num_gpu_blocks known only at first build).  #
    # --------------------------------------------------------------------- #
    def _ensure_state(self):
        if self._state is not None or not self._r8:
            return self._state
        # Same window as _ensure_tier: vLLM's cudagraph-memory profiling runs
        # against a THROWAWAY minimal cache with num_gpu_blocks SET (=2), so the
        # `not nb` test below cannot see it. Building here would key this
        # state's ptr->layer map to pointers freed moments later AND latch it
        # (the `self._state is not None` early-return above), reproducing the
        # stale-map failure that the tier just had. Skip the window, build for
        # the real cache.
        if _runtime.in_profiling():
            return None
        nb = self._vllm_config.cache_config.num_gpu_blocks
        if not nb:                       # cache not allocated yet
            return None
        # Resolve the per-layer resident kv_cache tensors (the SAME tensor object
        # the impl.forward receives: attention.get_attention_context returns
        # attn_layer.kv_cache directly) and key the ptr -> layer-index map.
        sfc = self._vllm_config.compilation_config.static_forward_context
        layer_kv = []
        ptr2layer = {}
        for lidx, name in enumerate(self._layer_names):
            mod = sfc.get(name)
            kvc = getattr(mod, "kv_cache", None) if mod is not None else None
            if not isinstance(kvc, torch.Tensor) or kvc.numel() == 0:
                return None              # not bound yet -> retry next build
            layer_kv.append(kvc)
            ptr2layer[kvc.data_ptr()] = lidx

        # r8i4 geometry contract: validate BEFORE the blanket alloc
        # try/except below, which exists to degrade genuine OOM to the FullKV
        # delegation -- it must NOT swallow the supported-geometry asserts
        # (an unsupported model would silently serve FullKV while tagged
        # r8i4). quad/clse/r8 keep the existing semantics unchanged.
        if True:   # r8i4 geometry contract: validate BEFORE the alloc guard
            from ..selection.r8i4_state import R8i4State as _R8i4Geom
            _R8i4Geom.validate_geometry(
                head_dim=self.headdim,
                G=self.num_heads_q // self.num_heads_kv,
                page=self.block_size)

        try:
            cfg = self._cfg
            # ONE summary state: r8i4 (the quad / clse / r8 variants were
            # deleted 2026-07-22).
            from ..selection import R8i4State as _State
            from ..selection.r8i4_state import RNK as rank  # LOCKS_R8I4_RANK (2|4|8)
            n_kv = self.num_heads_kv
            G = self.num_heads_q // n_kv
            max_reqs = self._vllm_config.scheduler_config.max_num_seqs
            max_model_len = self._vllm_config.model_config.max_model_len
            page = self.block_size
            max_pages = (max_model_len + page - 1) // page
            # STATIC selection is budget-driven; if only a coverage was given,
            # fall back to a fixed fraction (no adaptive per-head b).
            budget = cfg.budget if cfg.budget is not None else 0.1
            # The GQA combine (nrm default) + the score-kernel zsplit, which is
            # resolved ONCE here (config field / LOCKS_R8I4_Z, read in
            # LocksConfig.load -- never per call; None = auto-size from the
            # device SM count at launch).
            extra_state = {}
            extra_state["combine"] = cfg.quad_combine
            extra_state["zsplit"] = getattr(cfg, "r8i4_zsplit", None)
            st = _State(
                self._device, num_layers=len(self._layer_names),
                num_blocks=int(nb), n_kv=n_kv, G=G, head_dim=self.headdim,
                page=page, max_reqs=max_reqs, max_pages=max_pages,
                rank=rank, budget=budget, budget_pages=cfg.budget_pages,
                sink_pages=cfg.sink_pages, window_pages=cfg.window_pages,
                **extra_state)
            # Stage-B ownership lives in attention/; it reads st.D / st.max_reqs /
            # st.n_kv / st.G. R8State exposes .d (lowercase) -> alias it, and give
            # it a .scale slot (unused: the impl passes scale explicitly).
            st.D = st.d
            st.scale = None
            ensure_stage_b_buffers(st, self._device, _SPLIT)
            st._layer_of = ptr2layer
            self._layer_kv = layer_kv
            # Per-layer resident K-half views (stable storage): the steady-state
            # tail rebuild gathers from every layer each finalize step.  Under
            # spec v2 the engine holds RECORDS (no K half) -> the build gathers
            # K from the tier (_k_layers -> _TierKLayer); leave _layer_k None so
            # a resident-K read is impossible.
            if getattr(cfg, "mem_summary_cache", False):
                self._layer_k = None
            else:
                self._layer_k = [_split_kv(kvc)[0] for kvc in layer_kv]
            self._state = st
            if not getattr(cfg, "mem_summary_cache", False):
                self._ensure_overlap(sfc)
            print(f"[locks] {_State.__name__} ALLOCATED (score=r8i4): layers="
                  f"{st.L} num_blocks={st.NB} reqs={max_reqs} n_kv={n_kv} G={G} "
                  f"d={self.headdim} page={page} max_pages={max_pages} "
                  f"rank={st.r} v_bits={getattr(st, 'v_bits', '-')} "
                  f"combine={getattr(st, 'combine', '-')} "
                  f"coords={getattr(st, 'coords', '-')} "
                  f"rec={getattr(st, 'rec_bytes', 0) if getattr(st, 'rec', None) is not None else 0} "
                  f"budget={budget} sink={cfg.sink_pages} "
                  f"window={cfg.window_pages} split={_SPLIT} "
                  f"selector_MiB={st.bytes_per_layer() * st.L / 2**20:.1f}",
                  flush=True)
        except Exception as e:
            # NO-FALLBACK RULE (2026-07-22): this used to swallow the failure
            # and leave self._state = None, which makes attn.forward delegate
            # every decode step to stock FA -- i.e. the cell serves and
            # MEASURES FullKV while the log still prints "[locks] ACTIVE"
            # (the sentinel only proves the backend class was selected).  A
            # state allocation failure is now fatal: fix the cause (VRAM
            # headroom / geometry / a wiring bug) or run the reference line
            # explicitly with LOCKS_DISABLE=1.
            print("[locks] R8State alloc FAILED -- refusing the silent "
                  "stock-FA/FullKV delegation:", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise RuntimeError(
                "locks: Stage-A state allocation failed "
                f"({type(e).__name__}: {e}). There is no FullKV fallback "
                "under the locks tag -- lower gpu_memory_utilization / fix "
                "the geometry, or run the reference line with "
                "LOCKS_DISABLE=1.") from e
        return self._state

    # --------------------------------------------------------------------- #
    # Prefill-overlapped summary build (see _PrefillOverlap).               #
    # --------------------------------------------------------------------- #
    def _ensure_overlap(self, sfc) -> None:
        """Create the side stream + install the per-layer post-attention hooks.

        Called once, from ``_ensure_state``, i.e. the moment the summary slabs
        and the per-layer K views exist.  ``LOCKS_PREFILL_OVERLAP=0`` leaves the
        builder on the legacy serialized path (first-decode ``_bulk_refresh``)."""
        if not _OVERLAP or self._ov is not None:
            return
        try:
            from ..selection.r8i4_build import _r8i4_write as write_fn
            ov = _PrefillOverlap(self._device)
            ov.bind(self._state, self._layer_k, write_fn)
            n = ov.install(sfc, self._layer_names)
            if n == 0:
                print("[locks] prefill overlap: NO layer hooks installed -> "
                      "serialized first-decode build", flush=True)
                return
            self._ov = ov
            print(f"[locks] prefill overlap ACTIVE: side-stream summary build "
                  f"hooked on {n}/{len(self._layer_names)} layers "
                  f"(rebound={ov.n_rebound}, "
                  f"delay_ms={_OV_DELAY_MS} fence={'OFF (RACE TEST)' if _OV_NO_FENCE else 'on'})",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            print("[locks] prefill overlap install FAILED (serialized build):",
                  flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            self._ov = None

    def _ov_schedule(self, md, cam, st) -> None:
        """Prefill / mixed step: arm the per-layer hooks with this step's
        newly-finalized physical blocks (host arithmetic, one gather)."""
        ov = self._ov
        if ov is None or st is None or self._capturing:
            if ov is not None:
                ov.disarm()
            return
        n_req = int(getattr(cam, "num_reqs", 0) or 0)
        n_req = min(n_req, md.block_table.shape[0])
        if n_req <= 0:
            ov.disarm()
            return
        prev, cur = _host_prev_cur(cam, n_req)
        if prev is None:
            ov.disarm()               # no host lens -> legacy first-decode build
            return
        try:
            blocks = _finalized_block_plan(md.block_table, prev, cur, n_req,
                                           self.block_size, self._device)
        except Exception as e:  # noqa: BLE001
            print("[locks] prefill overlap plan FAILED (this step falls back "
                  "to the first-decode build):", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            ov.disarm()
            return
        if blocks is None:
            ov.disarm()
            return
        ov.plan(blocks)
        # Pages built during prefill make the first decode step's FULL path a
        # tag DELTA (a no-op scan in the happy path) instead of a bulk rebuild.
        self._built_once = True

    # --------------------------------------------------------------------- #
    # Lazy Tier construction (mem variants; num_gpu_blocks known at build).  #
    # --------------------------------------------------------------------- #
    def _ensure_tier(self):
        if self._tier is not None or self._tier_tried:
            return self._tier
        # vLLM 0.24 runs a cudagraph-memory profiling phase against a THROWAWAY
        # minimal KV cache and SETS num_gpu_blocks to that minimal count (=2)
        # before the dummy forwards (gpu_model_runner._init_minimal_kv_cache_
        # for_profiling), then frees the cache and resets num_gpu_blocks
        # (_cleanup_profiling_kv_cache). So `nb` is TRUTHY inside the window and
        # cannot gate it: building here would key _layer_of to pointers that are
        # freed moments later, latch _tier_tried, and publish that dead map on
        # the runtime singleton -- which is exactly how the write hook came to
        # raise "kv_cache pointer not in the tier layer map". Skip the window
        # explicitly and do NOT latch, so the real cache still gets a tier.
        if _runtime.in_profiling():
            return None
        nb = self._vllm_config.cache_config.num_gpu_blocks
        if not nb:                        # cache not allocated yet
            return None
        # Resolve the per-layer resident kv_cache tensors (dtype/device source);
        # bind only once every layer's cache tensor is materialised. Key a
        # kv_cache.data_ptr() -> physical layer index map (the SAME tensor object
        # impl.do_kv_cache_update/forward receive), so the write hook -- which
        # gets no attn_metadata -- can find this layer's tier slab.
        sfc = self._vllm_config.compilation_config.static_forward_context
        layer_kv = []
        ptr2layer = {}
        for lidx, name in enumerate(self._layer_names):
            mod = sfc.get(name)
            kvc = getattr(mod, "kv_cache", None) if mod is not None else None
            if not isinstance(kvc, torch.Tensor) or kvc.numel() == 0:
                return None               # not bound yet -> retry next build
            layer_kv.append(kvc)
            ptr2layer[kvc.data_ptr()] = lidx
        self._tier_tried = True
        try:
            from ..tier import Tier
            cfg = self._cfg
            sched = self._vllm_config.scheduler_config
            comp = self._vllm_config.compilation_config
            max_reqs = sched.max_num_seqs
            max_model_len = self._vllm_config.model_config.max_model_len
            page = self.block_size
            max_pages = (max_model_len + page - 1) // page
            max_tokens = max(
                int(getattr(sched, "max_num_batched_tokens", 0) or 0),
                int(getattr(comp, "max_cudagraph_capture_size", 0) or 0),
                max_reqs) + 8
            dtype = layer_kv[0].dtype
            self._tier = Tier.from_config(
                cfg, num_layers=len(self._layer_names), num_blocks=int(nb),
                n_kv=self.num_heads_kv, page=page, d=self.headdim,
                max_reqs=max_reqs, max_pages=max_pages, max_tokens=max_tokens,
                dtype=dtype, device=self._device)
            self._tier._layer_of = ptr2layer          # write-hook layer lookup
            self._layer_kv = layer_kv
            # Publish the tier on the runtime singleton so the impl's write hook
            # (do_kv_cache_update, no attn_metadata) and forward can reach it.
            _runtime.set_tier(self._tier)
            print(f"[locks] Tier ALLOCATED ({cfg.variant}): "
                  f"{self._tier.bytes_report()}", flush=True)
        except Exception as e:
            print("[locks] Tier alloc FAILED (mem-v -> stock FA):", flush=True)
            traceback.print_exception(type(e), e, e.__traceback__)
            self._tier = None
        return self._tier

    # --------------------------------------------------------------------- #
    # Prefill lease lifecycle (mem variants; leased chunked prefill, 3.4).   #
    # --------------------------------------------------------------------- #
    def _ensure_lease(self):
        if self._lease is not None:
            return self._lease
        from ..tier.lease import PrefillLease
        page = self.block_size
        max_model_len = self._vllm_config.model_config.max_model_len
        t = ((max_model_len + page - 1) // page) * page
        self._lease = PrefillLease(self._tier, t)
        print(f"[locks] prefill lease ALLOCATED: {self._lease.gib:.2f} GiB "
              f"(K+V, {t} tokens, transient)", flush=True)
        return self._lease

    def _lease_plan(self, md, cam, n_req) -> None:
        """Choose + arm THIS step's leased prefill row.

        At most ONE mid-prefill request exists (max_num_partial_prefills == 1,
        asserted at init).  A row with prior context (ctx > 0) IS the
        continuation and takes the lease (a watermark gap triggers the pinned
        recovery fill inside the lease).  Otherwise the largest fresh prefill
        row is leased (it is the only candidate that can continue next step);
        fresh rows that complete within the step never need the lease again and
        the next plan reclaims it.  Host arithmetic only; the per-layer fill is
        issued by the impl's batch-split forward."""
        md.locks_lease = None
        md.locks_lease_row = -1
        qs = getattr(cam, "query_start_loc_cpu", None)
        sl = _seq_lens_host(cam)
        if qs is None or qs.shape[0] < n_req + 1 or sl is None \
                or sl.shape[0] < n_req:
            # no host view (foreign runner): one guarded sync on a prefill step.
            qs = md.query_start_loc[:n_req + 1].cpu()
            sl = md.seq_lens[:n_req].cpu()
        md.locks_qsl_cpu = qs
        md.locks_sl_cpu = sl
        ql = (qs[1:n_req + 1] - qs[:n_req]).tolist()
        sls = sl[:n_req].tolist()
        cont = [r for r in range(n_req) if ql[r] > 1 and sls[r] - ql[r] > 0]
        fresh = [r for r in range(n_req) if ql[r] > 1 and sls[r] - ql[r] == 0]
        if len(cont) > 1:
            raise RuntimeError(
                f"locks mem: {len(cont)} continuation prefill rows in one step "
                "(max_num_partial_prefills must be 1)")
        row = cont[0] if cont else (
            max(fresh, key=lambda r: ql[r]) if fresh else -1)
        if row < 0:
            return
        lease = self._ensure_lease()
        if not cont:
            lease.wm = 0                     # fresh claim resets the watermark
        lease.plan_step(sls[row] - ql[row], sls[row])
        md.locks_lease = lease
        md.locks_lease_row = row

    # --------------------------------------------------------------------- #
    def build_for_cudagraph_capture(self, common_attn_metadata):
        # Capture/warmup dummy batches carry dummy block tables / seq lens: the
        # eigh r8_build_refresh must NOT run on them (it would write garbage r8
        # codes for dummy blocks). The captured graph reads the persistent r8 /
        # derived buffers the REAL build() refreshes; only their addresses are
        # baked at capture, so skipping the eager build here is correct.
        # The side-stream prefill build must also be quiet here: no plan is armed
        # on a capture batch (``_ov_schedule`` bails on ``_capturing``), and any
        # build still in flight from an earlier real step is drained on the HOST
        # before capture begins -- an event wait recorded inside a capture would
        # bake a cross-stream dependency on a non-captured stream into the graph.
        if self._ov is not None:
            self._ov.disarm()
            self._ov.drain()
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        if _NVTX:
            _nvtx_push("locks.fa_build")
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        if _NVTX:
            _nvtx_pop()
        # mem variants: run the tier stage/flush/alloc lifecycle OUTSIDE the
        # graph (host), every step (staging happens during prefill AND decode).
        # Skipped on cudagraph-capture dummy batches (capture=True). Attach the
        # tier so the impl serves V (and K for mem-kv) from it. Then FALL THROUGH
        # to the shared Stage-A setup below: the mem tiered decode consumes the
        # SAME st.page_table/page_cnt as fast, so mem needs both md.locks_tier
        # AND md.locks.
        if self._cfg is not None and self._cfg.is_mem:
            tier = self._ensure_tier()
            if tier is not None:
                tier.begin_step(
                    md.block_table, md.seq_lens, md.query_start_loc,
                    md.max_query_len, md.slot_mapping.shape[0],
                    capture=self._capturing)
                md.locks_tier = tier
        st = self._ensure_state()
        # LOCKS_CUPROF_*: worker-side cudaProfilerApi bracket (instrumentation;
        # no-op when unset). Must run BEFORE the fence/refresh below so the
        # first-decode-step host prologue is inside the nsys capture window.
        _runtime.cuprof_step(st is not None and md.max_query_len == 1,
                             self._capturing)
        # LOCKS_DUMP_PTAB: empirical page-ranking gate artifact (no-op unset).
        _runtime.ptab_dump_step(st, st is not None and md.max_query_len == 1,
                                self._capturing, seq_lens=md.seq_lens)
        # LOCKS_CNT_AUDIT: exact-b field-rate instrument (no-op unset).
        _runtime.cnt_audit_step(st, st is not None and md.max_query_len == 1,
                                self._capturing)
        if st is None or md.max_query_len != 1:
            # PREFILL / MIXED / no-state. Nothing here reads a summary, so no
            # fence: instead arm this step's per-layer side-stream builds. They
            # run under the prefill compute that immediately follows this build.
            # FAST MIXED STEP (2026-07-22): a chunked-prefill step can carry
            # DECODE rows, which the impl now serves sparsely via the batch-split
            # forward instead of bailing the whole step to stock FA (that bail
            # measured those rows as FullKV under the locks tag: 12.50% activation
            # at 125K/mns=4). Those rows READ summaries, so this step must do what
            # the pure-decode path does before a read -- fence the side-stream
            # builds and refresh newly-finalized pages -- and stash the host views
            # + state the split forward needs. Only when decode rows are actually
            # present, so pure-prefill steps keep the un-fenced overlap.
            if (st is not None and not self._capturing
                    and self._cfg is not None and not self._cfg.is_mem):
                n_req_x = md.seq_lens.shape[0]
                qs_x = getattr(common_attn_metadata, "query_start_loc_cpu", None)
                sl_x = _seq_lens_host(common_attn_metadata)
                if (qs_x is None or qs_x.shape[0] < n_req_x + 1
                        or sl_x is None or sl_x.shape[0] < n_req_x):
                    qs_x = md.query_start_loc[:n_req_x + 1].cpu()
                    sl_x = md.seq_lens[:n_req_x].cpu()
                ql_x = qs_x[1:n_req_x + 1] - qs_x[:n_req_x]
                if bool((ql_x == 1).any()):
                    md.locks_qsl_cpu = qs_x
                    md.locks_sl_cpu = sl_x
                    if self._ov is not None:
                        self._ov.disarm()
                        self._ov.fence()
                    if self._r8:
                        self._maybe_refresh(st, md, common_attn_metadata,
                                            n_req_x)
                    md.locks = st
            self._ov_schedule(md, common_attn_metadata, st)
            if (st is not None and self._cfg is not None and self._cfg.is_mem
                    and getattr(md, "locks_tier", None) is not None
                    and not self._capturing):
                # mem prefill/mixed step (P4 leased chunked prefill, 3.4):
                # (a) refresh the summaries of pages SETTLED so far (tokens of
                #     PRIOR steps only: this chunk's K is written during the
                #     forward AFTER this build) -- spreads the mem-kv in-prefill
                #     build over the chunks AND keeps the mixed-step decode
                #     rows' selection fresh;
                # (b) plan the prefill lease for the (single) mid-prefill row;
                # (c) stash host views + the Stage-A state for the impl's
                #     batch-split forward.
                n_req = md.seq_lens.shape[0]
                # A/B knob (G6 decomposition): LOCKS_MEM_PREFILL_REFRESH=0
                # defers the whole summary build to the first decode step's
                # delta (the pre-P4 timing; mixed-step decode rows then select
                # on stale tails until that delta runs). Default ON.
                if os.environ.get("LOCKS_MEM_PREFILL_REFRESH", "1") != "0":
                    self._refresh_prefill(st, md, common_attn_metadata, n_req)
                self._lease_plan(md, common_attn_metadata, n_req)
                md.locks = st
            return md
        # PURE DECODE. Everything below (the delta tag scan, derive, and the
        # in-graph score kernel) READS the summaries -> fence first. Disarm the
        # hooks so a decode-step impl.forward (eager / piecewise / capture)
        # cannot re-issue a build.
        if self._ov is not None:
            self._ov.disarm()
            self._ov.fence()
        # mem: the leased request (if any) has left prefill -> free the lease
        # (host holds every complete page; the staged tail serves the decode).
        if self._lease is not None and not self._capturing:
            self._lease.release()
            self._lease = None
        n_req = md.seq_lens.shape[0]
        if self._r8:
            # 1. (Re)build the r8 summary of newly-finalized pages OUTSIDE the
            #    graph, per layer. Tag-gated: steady state rebuilds nothing.
            #    At the first decode build every prompt page's K is committed
            #    (the current decode token is not yet in the cache, but it lives
            #    in the always-attended window, so it is never scored), so the
            #    whole prompt is summarized correctly here in one shot; scored
            #    pages are ALWAYS a subset of finalized pages (window >= 1).
            need_derive = True
            if not self._capturing:
                if _NVTX:
                    _nvtx_push("locks.refresh")
                need_derive = self._maybe_refresh(st, md,
                                                  common_attn_metadata, n_req)
                if _NVTX:
                    _nvtx_pop()
            # 2. Per-step derived selection params (graph-safe, one launch).
            #    All derive outputs are pure functions of ceil(seq/page)
            #    (select.py::_derive_params_kernel), which only changes on
            #    steps where an advanced row hits seq % page == 1 -> skip the
            #    launch otherwise (the persistent buffers already hold the
            #    identical values). Capture steps keep the unconditional
            #    (dummy) derive, matching the pre-gating behaviour.
            if need_derive:
                from ..selection import derive_page_params
                if _NVTX:
                    _nvtx_push("locks.derive")
                derive_page_params(st, md.seq_lens, n_req)
                if _NVTX:
                    _nvtx_pop()
        else:
            self._derive(md.seq_lens, st, self._cfg)   # bridge no-op
        md.locks = st
        return md

    def _maybe_refresh(self, st, md, cam, n_req):
        """Refresh-skip gate. The summary codes only change when a page
        FINALIZES or a request slot is reused; both are detectable on the HOST
        with no device sync from the runner's CPU seq lens:

          * steady step, no boundary crossed  -> NOTHING to do (skip).
          * steady step, some rows crossed a page boundary (cur % page == 0)
            -> ``*_build_tail`` for exactly THOSE rows: batched, sync-free
            rebuild of each one's last finalized page (idempotent; ~55
            launches vs the full refresh's measured ~99 ms sync-storm).
          * anything else (first step, shape change, slot reuse, preemption)
            -> ``*_build_delta`` once anything has been built (tag scan, rebuild
            only stale blocks), else ``*_build_bulk``: batched zero-sync rebuild
            of ALL finalized pages of the batch (idempotent; covers slot/block
            reuse WITHOUT the tag scan; the measured 737 ms first-decode
            sync-storm becomes a few dozen bandwidth-bound launches).

            With the prefill overlap on (the default), the prompt's pages were
            already summarized on the side stream DURING prefill, so
            ``_built_once`` is already True at the first decode step and this
            branch is the DELTA tag pass -- which finds nothing stale and
            rebuilds ZERO blocks. ``_bulk_refresh`` remains the correctness
            fallback for the overlap-off / hooks-not-installed / no-host-lens
            paths, and the delta covers everything the overlap could not reach
            (prefix-cache hits, preemption, block reuse) by content tag.
          * no host lens at all -> legacy per-layer tag-scan refresh.

        Padded uniform-decode rows (full-cudagraph batches pad n_req up to the
        capture size) sit at ``cur == prev == 0`` and are treated as steady;
        they never cross a boundary so the tail skips them. (The pre-iter-2
        gate treated padded batches as never steady, silently running the full
        refresh EVERY step whenever ``n_req < capture size``.)

        Returns ``need_derive``: whether the derive params can have changed
        this step (ceil(seq/page) increments on ``cur % page == 1`` rows)."""
        page = self.block_size
        if _FORCE_FULL:
            # DIAGNOSTIC REFERENCE ARM (LOCKS_FORCE_FULL_REFRESH=1). Rebuilds
            # EVERY settled page of EVERY row from the live K on EVERY step --
            # the ``src/locks`` stateless-summary semantics expressed inside
            # this plugin, so an A/B against the shipped gate varies ONLY the
            # build LIFECYCLE (estimator, kernels, selection, combine all held
            # constant). Never a shipping path: it reinstates exactly the
            # sync-storm the gate exists to remove.
            self._prev_seq_lens = None
            self._prev_settled = None
            settled = self._settled_dev(md, n_req)
            max_fin = int(settled.max().item()) // page
            if max_fin > 0:
                self._bulk_refresh(st, md, n_req, max_fin, settled=settled)
                self._built_once = True
            return True
        sl_cpu = _seq_lens_host(cam)
        qs_cpu = getattr(cam, "query_start_loc_cpu", None)
        if (sl_cpu is None or qs_cpu is None
                or qs_cpu.shape[0] < n_req + 1):   # no host views -> conservative
            self._prev_seq_lens = None
            self._prev_settled = None
            self._r8_refresh(st, md, n_req)
            return True
        cur = sl_cpu[:n_req]
        # SETTLED tokens = seq_lens - query_len: the tokens whose K PRIOR steps
        # already wrote. seq_lens counts this step's in-flight query tokens too,
        # and their K is written by the forward that runs AFTER this prologue.
        # On a pure-decode step query_len == 1 everywhere and cur_st == cur - 1
        # (the value this gate used before); on a MIXED chunked-prefill step
        # (max_num_seqs > 1) a prefilling row's query_len is the whole chunk,
        # and cur - 1 over-counts it by chunk-1 tokens -- hundreds of pages
        # whose K does not exist yet. Both the gate predicate and the refresh
        # bound below are therefore expressed in SETTLED tokens.
        cur_st = (cur - (qs_cpu[1:n_req + 1] - qs_cpu[:n_req]).to(cur.dtype))
        cur_st = cur_st.clamp_(min=0)
        prev = self._prev_seq_lens
        prev_st = self._prev_settled
        full = True
        crossed = None
        if (prev is not None and prev.shape == cur.shape
                and prev_st is not None and prev_st.shape == cur_st.shape):
            # A row is STEADY only if it decoded exactly one token BOTH this
            # step (cur == prev + 1 <=> query_len == 1) AND last step
            # (cur_st == prev_st + 1 <=> the PREVIOUS step's query_len was 1:
            # settled advances by the previous step's query length). The second
            # conjunct is what makes the prefill->decode transition a FULL step:
            # a row leaving chunked prefill advances its seq_len by exactly one
            # (so the old single-conjunct `adv` called it steady) while its
            # settled length jumps by the whole final chunk, leaving that
            # chunk's pages unsummarized forever behind a steady gate.
            adv = (cur == prev + 1) & (cur_st == prev_st + 1)
            pad = (cur == 0) & (prev == 0)     # padded uniform-decode rows
            # frozen rows (vLLM end-of-generate drain steps replaying padded
            # decodes with discarded tokens) change nothing -- in BOTH views.
            frozen = (cur == prev) & (cur_st == prev_st)
            if bool(torch.all(adv | pad | frozen)):   # steady batch
                full = False
                # Fire the tail ONE STEP AFTER the boundary: vLLM seq_lens
                # includes the token decoded THIS step, whose K is written
                # inside the forward AFTER this host prologue, so at
                # cur % page == 0 the finalized page's last row is not yet
                # in the cache and its summary would carry one stale row
                # forever (confirmed 6/6, scratch_offbyone/). At
                # cur_st % page == 0 (== cur % page == 1 on this branch, where
                # every row has query_len 1), fin1 = sl//page - 1 addresses
                # the same page, now settled, on the same step it first
                # becomes scoreable (n_sel_hi crosses it here too).
                crossed = (cur_st % page == 0) & adv
        if os.environ.get("LOCKS_DEBUG_GATE", "0") == "1":
            print(f"[locks gate] full={full} "
                  f"crossed={crossed.sum().item() if crossed is not None else '-'} "
                  f"n_req={n_req} cur={cur[:4].tolist()} "
                  f"prev={prev[:4].tolist() if prev is not None else None} "
                  f"cur_st={cur_st[:4].tolist()} "
                  f"prev_st={prev_st[:4].tolist() if prev_st is not None else None}",
                  flush=True)
        self._prev_seq_lens = cur.clone()
        self._prev_settled = cur_st.clone()
        need_derive = True
        if full:
            # Bound and per-row lengths both in SETTLED tokens (see above): the
            # refresh's own n_fin is sl // page, so passing anything that counts
            # an in-flight token admits a page the forward has not written yet.
            # A never-built page's tag is NaN, so the delta rebuilds it FROM
            # BYTES THAT ARE NOT ITS KEYS and then stores a tag derived from
            # those bytes -- self-sealing corruption, not staleness.
            max_fin = int(cur_st.max().item()) // page      # CPU tensor: free
            settled = self._settled_dev(md, n_req)
            # First full build: zero-sync bulk (everything is stale anyway).
            # Every later composition change (arrival, drain shuffle, slot
            # reuse, preemption resume): batched TAG-SCAN delta -- rebuild
            # only blocks whose content tag went stale (the measured ~800 ms
            # bulk per transition at bs16/16K becomes ~tag-pass + new work).
            if self._built_once:
                self._delta_refresh(st, md, n_req, max_fin, settled=settled)
            else:
                self._bulk_refresh(st, md, n_req, max_fin, settled=settled)
                self._built_once = True
        else:
            fired = bool(crossed.any())
            if fired:
                rows = self._rows_to_device(torch.nonzero(crossed).squeeze(-1))
                self._tail_refresh(st, md, n_req, rows)
            # derive params change only when ceil(sl/page) does (% page == 1),
            # and `crossed` IS that predicate on this branch -- reuse it
            # instead of recomputing the same expression (P6).
            need_derive = fired
        return need_derive

    def _rows_to_device(self, idx):
        """HOST tail-row indices -> device, sync-free.

        The old ``.to(self._device)`` on the pageable nonzero() result issued
        a blocking cudaMemcpy whose stream sync stalled the host until every
        in-flight graph replay drained: a measured 4.9 ms pipeline break on
        EVERY page-crossing step at 16K bs1 (nsys P6HT_locks_ctx16384_b1,
        cudaStreamSynchronize inside locks.refresh; the engine runs the host
        prologue of step N+1 under the GPU of step N, and the sync forfeited
        that overlap).  A preallocated pinned+device pair with a non_blocking
        copy is stream-ordered before the tail's gathers, so the device sees
        identical values (bitwise-inert) and the host never waits.  Ping-pong
        parity: at bs>1 consecutive steps can both cross, and the async copy
        of step N may still be queued when step N+1's host writes the pinned
        buffer -- two pairs make the reuse window >= 2 steps."""
        k = idx.shape[0]
        pin = self._rows_pin
        if pin is None or pin[0].shape[0] < k:
            cap = max(k, int(self._vllm_config.scheduler_config.max_num_seqs))
            self._rows_pin = pin = [
                torch.empty(cap, dtype=torch.int64, pin_memory=True)
                for _ in range(2)]
            self._rows_dev = [
                torch.empty(cap, dtype=torch.int64, device=self._device)
                for _ in range(2)]
        p = self._rows_par = self._rows_par ^ 1
        pin[p][:k].copy_(idx)
        self._rows_dev[p][:k].copy_(pin[p][:k], non_blocking=True)
        return self._rows_dev[p][:k]

    def _r8_refresh(self, st, md, n_req):
        """Run r8_build_refresh for every layer over its resident K half.

        NOT graph-safe (eigh + boolean gather) -> only ever called here, on the
        host, before graph replay. First decode step / slot changes only; the
        steady-state finalize path is ``_tail_refresh``."""
        if getattr(self._cfg, "mem_summary_cache", False):
            # This conservative no-host-lens path reads _split_kv(engine) as K,
            # but under spec v2 the engine page holds RECORDS -- summarizing it
            # would poison every selection silently. The vLLM runner always
            # provides host lens; anything else is a wiring bug. Refuse.
            raise RuntimeError(
                "locks mem_summary_cache: refresh fallback without host seq "
                "lens reached -- the engine page holds records, not K; cannot "
                "build summaries from it (LOCKS_MEM_R8I4.md B1)")
        from ..selection import r8i4_build_refresh as _refresh
        bt = md.block_table
        # settled lengths (see _maybe_refresh): n_fin inside the refresh is
        # sl // page and must not count the in-flight tokens' pages (== -1 on
        # a pure-decode step, the whole chunk on a mixed one).
        sl = self._settled_dev(md, n_req)
        for lidx, kvc in enumerate(self._layer_kv):
            K, _ = _split_kv(kvc)          # (NB, page, n_kv, d) resident K half
            _refresh(st, lidx, K, bt, sl, n_req)

    def _k_layers(self):
        """Per-layer K source for the summary build.  Flag-OFF: the resident
        engine K-half views (``self._layer_k``).  Summary cache (spec v2): the
        engine holds records, so gather K from the tier via ``_TierKLayer``
        proxies (byte-identical to the old engine K, off-graph)."""
        cfg = self._cfg
        if getattr(cfg, "mem_summary_cache", False):
            tier = _runtime.get_tier()
            if tier is None:
                raise RuntimeError(
                    "mem_summary_cache: summary build reached before the tier "
                    "was built (no K source)")
            return [_TierKLayer(tier, l) for l in range(len(self._layer_names))]
        return self._layer_k

    def _tail_refresh(self, st, md, n_req, rows):
        """Steady-state finalize step: batched all-layer sync-free rebuild of
        the crossing rows' last finalized page (selection.build.r8_build_tail)."""
        from ..selection import r8i4_build_tail as _tail
        # settled lengths: no-op for fin1 under the % page == 1 gate
        # ((sl-1)//page == sl//page there), kept for the uniform invariant.
        _tail(st, self._k_layers(), md.block_table,
              (md.seq_lens - 1).clamp(min=0), n_req, rows)

    def _settled_dev(self, md, n_req):
        """(n_req,) device SETTLED token counts = ``seq_lens - query_len``.

        The K of the tokens this step computes is written by the forward that
        runs AFTER this host prologue, so a page is complete-and-summarizable
        only strictly below this bound. On a pure-decode step query_len == 1
        for every row and this is bitwise ``seq_lens - 1`` (the value the
        builder used before); on a prefill/mixed step it excludes the WHOLE
        in-flight chunk, which ``seq_lens - 1`` did not. Sync-free (two small
        device ops on the already-resident query_start_loc)."""
        qsl = md.query_start_loc
        ql = qsl[1:n_req + 1] - qsl[:n_req]
        return (md.seq_lens[:n_req] - ql).clamp(min=0)

    def _bulk_refresh(self, st, md, n_req, max_fin, settled=None):
        """First decode step / slot change: batched zero-sync rebuild of ALL
        finalized pages, all layers (selection.build.r8_build_bulk)."""
        from ..selection import r8i4_build_bulk as _bulk
        if settled is None:
            settled = (md.seq_lens - 1).clamp(min=0)
        _bulk(st, self._k_layers(), md.block_table, settled, n_req, max_fin)

    def _delta_refresh(self, st, md, n_req, max_fin, settled=None):
        """Batch-composition change: batched all-layer tag scan, rebuild only
        stale blocks (selection.build.r8_build_delta)."""
        from ..selection import r8i4_build_delta as _delta
        if settled is None:
            settled = (md.seq_lens - 1).clamp(min=0)
        _delta(st, self._k_layers(), md.block_table, settled, n_req, max_fin)

    def _refresh_prefill(self, st, md, cam, n_req):
        """Summary refresh on a PREFILL/MIXED step (mem variants, P4).

        Settled tokens here are the tokens computed by PRIOR steps
        (= seq_lens - query_lens): the whole current chunk is in flight and its
        K is written per layer during the forward AFTER this build, so it must
        never be summarized this step (the same off-by-one family as A0, one
        chunk wide). First refresh is a bulk build, later ones the tag-scan
        delta; whatever a step misses keeps a stale tag and the next refresh
        (or the first decode step's delta) rebuilds it."""
        prev, _cur = _host_prev_cur(cam, n_req)
        if prev is None:
            return                      # no host lens: first-decode bulk covers
        max_fin = int(prev.max().item()) // self.block_size
        if max_fin <= 0:
            return                      # nothing settled yet (first chunk)
        qsl = md.query_start_loc
        ql_dev = qsl[1:n_req + 1] - qsl[:n_req]
        settled = (md.seq_lens[:n_req] - ql_dev).clamp(min=0)
        from ..selection import (r8i4_build_bulk as _bulk,
                                 r8i4_build_delta as _delta)
        if self._built_once:
            _delta(st, self._k_layers(), md.block_table, settled, n_req, max_fin)
        else:
            _bulk(st, self._k_layers(), md.block_table, settled, n_req, max_fin)
            self._built_once = True
