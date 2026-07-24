"""LocksAttentionImpl -- FlashAttentionImpl + LOCKS decode-time page selection.

Only a plain pure single-token decode step on a standard causal decoder layer
runs the LOCKS Stage-A + Stage-B pipeline; every other path (prefill, mixed
batch, cascade, encoder, profiling, quantized cache, sliding window, ...)
inherits stock FlashAttention unchanged (this is a FlashAttentionImpl subclass).

FAST variant (this file): pure decode does
    select_pages_r8(...)                 -> st.page_table / st.page_cnt  (Stage A)
    sparse_paged_decode_batched(...,     -> out                          (Stage B)
        vsource=ResidentVSource(kv_cache))
K is read from the resident cache; V through the ResidentVSource seam (the engine
V half). The mem variants swap ONLY the VSource (a later agent) -- this call site
does not change. Non-fast variants delegate to stock FA here until the tier lands.
"""
from __future__ import annotations

from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from ..attention.decode import ResidentVSource
from . import _runtime

# NOROPE (round A): the forward's rope kernel is deleted at nt==1; the
# scorer ropes q (publishing st.q_rope, the decode kernels' q source below)
# and the write paths rope k (K8d SCAT on mem, _norope_fast_write on fast).
_NOROPE = __import__("os").environ.get("LOCKS_NOROPE", "0") == "1"
# P2b co-kernel lane shares every NOROPE consumption path (q from
# st.q_rope, k roped in the write paths): one gate for both.
_Q_ABSORBED = (_NOROPE or __import__("os").environ.get(
    "LOCKS_QFIRST_CO", "0") == "1")


_QCO = __import__("os").environ.get("LOCKS_QFIRST_CO", "0") == "1"


def _absorbed_now(max_query_len: int) -> bool:
    """Does the fused QKV buffer hold UNROPED q for THIS call?

    A FUNCTION OF METADATA, not of a process flag, a shape, or an op receipt.

      * ``LOCKS_NOROPE`` -- the trace stays 100% stock and the rope kernels
        write SEPARATE output buffers (NOROPE v3 deletes those nodes from
        captured decode graphs).  The fused QKV buffer is NEVER roped in place,
        at any step type: absorbed always.
      * ``LOCKS_QFIRST_CO`` -- the ``locks_co.qkv`` op absorbs (returns unroped
        q, defers the KV write) on EVERY pure-decode step and ropes in place on
        mixed / prefill steps.  Its decision IS ``max_query_len == 1``, so this
        reads the same ``max_query_len`` and agrees BY CONSTRUCTION.

    Why not the op's receipt (which would seem more direct): under vLLM's
    piecewise CUDA graphs the op is captured and its python does not re-run on a
    replayed decode step, while this impl runs eagerly every step -- so a
    receipt goes stale and the impl reads a prefill's stamp.  ``max_query_len``
    is in the impl's own metadata every step, captured or eager, so there is no
    staleness.  Why not the shape ``nt == 1`` (the previous fix): the unified op
    absorbs at nt > 1 pure decode too, so ``nt == 1`` would UNDER-count and a
    multi-row decode step would read roped-as-unroped.  ``max_query_len`` is the
    op's real condition, not a proxy, which is why this does not reintroduce the
    5.48 double-rope."""
    return _NOROPE or (_QCO and max_query_len == 1)


def _norope_qk_views(impl, value):
    """NOROPE v3: reconstruct the UNROPED q/k as views of the fused QKV row
    from the v slice (v = qkv[..., (nh+nkv)*hs :] in the STOCK trace).  The
    captured graph's rope nodes are PRUNED, so the forward's q/k args are
    unwritten garbage; the QKV buffer itself is never roped in place --
    these views are the true unroped bytes.  Pure metadata (as_strided on
    the same storage), zero kernels, capture-safe.  Loud on any non-fused
    layout (a different model/trace shape means NOROPE is unsupported).

    T==1 STRIDE EXEMPTION (2026-07-22, found taking the CO/FIX-C lane EAGER
    for the first time -- the accuracy lane runs force_eager=True): vLLM
    reshapes the v slice with ``view(-1, n_kv, head_size)``.  Under
    torch.compile the leading dim is SYMBOLIC, so view cannot collapse and
    keeps the fused row stride; in eager at T==1 the size-1 leading dim makes
    its stride unobservable and view picks the packed value (n_kv*hs).  The
    row stride is genuinely dead information for a single row, so it is
    checked only when there is more than one row.  Everything that actually
    locates the bytes -- the inner strides, the head count, and the v-slice
    storage offset -- is still asserted."""
    import torch
    nh, nkv, hs = impl.num_heads, impl.num_kv_heads, impl.head_size
    row = (nh + 2 * nkv) * hs
    so = value.storage_offset()
    t = value.shape[0] if value.dim() == 3 else -1
    assert (value.dim() == 3 and value.shape[1] == nkv and value.shape[2] == hs
            and value.stride(2) == 1 and value.stride(1) == hs
            and (value.stride(0) == row or t == 1)
            and so >= (nh + nkv) * hs
            and (so - (nh + nkv) * hs) % row == 0), \
        f"NOROPE: value is not the fused-QKV v slice (shape " \
        f"{tuple(value.shape)}, strides {tuple(value.stride())}, off {so}, " \
        f"expected inner ({nkv},{hs}) row {row})"
    t = value.shape[0]
    base = so - (nh + nkv) * hs
    q = torch.as_strided(value, (t, nh, hs), (row, hs, 1), base)
    k = torch.as_strided(value, (t, nkv, hs), (row, hs, 1), base + nh * hs)
    return q, k


def _norope_fast_write(impl, key, value, kv_cache, slot_mapping) -> None:
    """Fast-arm nt==1 KV write under NOROPE: rope k dims<64 + scatter, one
    launch, byte-identical to [triton rope -> stock reshape_and_cache]
    (k5_cuda.rope_and_cache; formula U3-locked).  v3: the ``key`` ARG is
    the pruned rope node's unwritten output -- the true unroped k is
    reconstructed from the QKV buffer via ``_norope_qk_views``.  Raises on
    a wiring miss rather than silently caching wrong k; a dummy pass (all
    slots < 0, no metadata) is skipped per vLLM's no-write contract."""
    from vllm.forward_context import get_forward_context

    from ..attention.decode import _split_kv
    from ..tier import k5_cuda
    csc = _runtime._NOROPE_CSC
    md = getattr(get_forward_context(), "attn_metadata", None)
    if isinstance(md, dict):
        md = next(iter(md.values()), None)
    sl = getattr(md, "seq_lens", None)
    if sl is None or csc is None:
        if int(slot_mapping.max().item()) < 0:
            return                      # dummy pass: nothing to write
        raise RuntimeError(
            "LOCKS_NOROPE: fast KV write without metadata/csc wiring")
    k_un = _norope_qk_views(impl, value)[1]
    kres, vres = _split_kv(kv_cache)
    mod = k5_cuda.get_mod()
    assert mod is not None, "LOCKS_NOROPE: k5_cuda build required"
    mod.rope_and_cache(k_un, value, kres, vres, slot_mapping, sl, csc,
                       int(kres.shape[1]))


def _fixc_defer_write(impl, value, kv_cache, slot_mapping) -> None:
    """FIX-C: resolve the NOROPE fast-write arguments NOW, defer the launch
    to _runtime.fixc_flush (directly behind the select launch, where the
    kv+rope pair rides select_v2's entry-trigger cover window).  Mirrors
    _norope_fast_write's wiring resolution.  Reachable only when the co
    op's FIX-C branch armed the projection stash this step, which is
    metadata-gated -- so a missing metadata/csc here is a wiring bug, loud
    per the no-fallback rule."""
    from vllm.forward_context import get_forward_context

    from ..attention.decode import _split_kv
    from ..tier import k5_cuda
    csc = _runtime._NOROPE_CSC
    md = getattr(get_forward_context(), "attn_metadata", None)
    if isinstance(md, dict):
        md = next(iter(md.values()), None)
    sl = getattr(md, "seq_lens", None)
    if sl is None or csc is None:
        raise RuntimeError(
            "LOCKS FIXC: deferred KV write without metadata/csc wiring")
    k_un = _norope_qk_views(impl, value)[1]
    kres, vres = _split_kv(kv_cache)
    mod = k5_cuda.get_mod()
    assert mod is not None and hasattr(mod, "rope_and_cache_fixc"), (
        "LOCKS FIXC: k5_cuda extension lacks rope_and_cache_fixc (built "
        "without -DLOCKS_FIXC?). Prebuild offline -- no silent fallback.")
    _runtime.fixc_arm_write(k_un, value, kres, vres, slot_mapping, sl, csc,
                            int(kres.shape[1]))


_ANNOUNCED = False
_DELEG_SAID = False      # one-shot: pure-decode step delegated with no state

# ---------------------------------------------------------------------------- #
# SPARSE-ACTIVATION ACCOUNTING (always on; python ints, no device sync on the
# sparse path).  Companion to the NO-SILENT-FULLKV detector below, closing the
# case that detector deliberately does NOT cover.
#
# The `m.max_query_len != 1` disjunct is treated as a step-TYPE split, which is
# correct for pure prefill.  It is NOT correct under vLLM V1 chunked prefill with
# max_num_seqs > 1: there a DECODE row is co-scheduled with another sequence's
# prefill chunk, max_query_len becomes the chunk size, and those decode rows are
# served by stock FlashAttention -- i.e. measured as FullKV under the locks tag.
# Same failure mode found in src/quest on 2026-07-22 (there it silently disabled
# Quest entirely at max_num_seqs=4).  Counting DECODE ROWS, not steps, so a
# partially-dense cell is quantified rather than merely suspected.
# ---------------------------------------------------------------------------- #
#
# ROPE-ABSORPTION ACCOUNTING (2026-07-22).  `decode_rows_stock_proj` counts
# decode rows whose step ran the STOCK fused projection + rope.  Since the
# nt >= 1 unification the op no longer has a single-row precondition, so this
# is now only the residue: a step whose select-adapter refs stash did not match
# this call (the batch composition moved), or a step with no live metadata.
# Those rows are CORRECT and fully sparse -- the counter is not an error rate,
# it is the answer to "did the CO/FIX-C kernels actually run in this cell".  It
# was 99.50% at max_num_seqs=4 before the unification.
_ACT = {"decode_rows_sparse": 0, "decode_rows_dense": 0, "dense_reasons": {},
        "decode_rows_stock_proj": 0}


def get_activation():
    """{sparse, dense, frac, dense_reasons} over DECODE ROWS.  frac < 1.0 means
    part of the cell was served as FullKV and the number is INFLATED."""
    s = _ACT["decode_rows_sparse"]
    d = _ACT["decode_rows_dense"]
    if s + d == 0:
        return None
    return {"sparse": s, "dense": d, "frac": s / (s + d),
            "dense_reasons": dict(_ACT["dense_reasons"]),
            "stock_proj": _ACT["decode_rows_stock_proj"]}


def _locks_activation_report():
    a = get_activation()
    if a is None:
        return
    ok = a["frac"] >= 1.0
    print(f"[locks] SPARSE ACTIVATION {a['sparse']}/{a['sparse'] + a['dense']} "
          f"decode rows = {100 * a['frac']:.2f}%  "
          f"[{'OK' if ok else 'INVALID -- part of this cell is FullKV'}]",
          flush=True)
    if a["dense_reasons"]:
        print(f"[locks]   decode rows served DENSE by gate: {a['dense_reasons']}",
              flush=True)
    if a["stock_proj"]:
        print(f"[locks]   ROPE-ABSORBING lane INACTIVE for {a['stock_proj']}/"
              f"{a['sparse']} sparse decode rows (their step had nt>1, so the "
              "single-row co op ran the stock fused projection + rope). "
              "Correct, and fully sparse -- but those rows did NOT exercise "
              "the CO/FIX-C kernels.", flush=True)


__import__("atexit").register(_locks_activation_report)


class LocksAttentionImpl(FlashAttentionImpl):
    """FlashAttentionImpl + LOCKS-fast pure-decode page selection."""

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        m = attn_metadata
        cfg = _runtime.get_config()
        # mem variants: the tiered decode is the ONLY mem decode path (shipped
        # _MEM_STAGEB_WIRED=True). No feature-off latency fallback -- assert the
        # precondition rather than branch to stock FA (the no-fallback rule).
        # LOCKS_DISABLE=1 (register inert -> cfg is None) is the stock-FA/FullKV
        # reference line; a prefill / mixed / non-standard step is a step-TYPE
        # split handled INSIDE _forward_mem (dense stock FA over the engine
        # cache), not a latency fallback.
        if cfg is not None and cfg.is_mem:
            assert _runtime.mem_stageb_wired(), (
                "locks mem: _MEM_STAGEB_WIRED is False -- the mem variant has no "
                "feature-off decode path (set it True, or run LOCKS_DISABLE=1 "
                "for the stock-FA/FullKV line)")
            return self._forward_mem(layer, query, key, value, kv_cache, m,
                                     output, output_scale, output_block_scale)
        st = getattr(m, "locks", None) if m is not None else None
        # Delegate everything that is not a plain pure-decode step on a standard
        # causal decoder layer with an unquantized resident cache to stock FA.
        _reasons = (
            ("no_cfg", cfg is None), ("variant!=fast", cfg is not None and cfg.variant != "fast"),
            ("no_state", st is None), ("no_metadata", m is None),
            ("output_is_None", output is None),
            ("max_query_len!=1", m is not None and m.max_query_len != 1),
            ("kv_cache_empty", kv_cache.numel() == 0),
            ("use_cascade", bool(getattr(m, "use_cascade", False))),
            ("not_causal", m is not None and m.causal is not True),
            ("attn_type!=DECODER", self.attn_type != AttentionType.DECODER),
            ("sliding_window", self.sliding_window != (-1, -1)),
            ("alibi", self.alibi_slopes is not None),
            ("sinks", self.sinks is not None),
            ("dcp>1", getattr(self, "dcp_world_size", 1) > 1),
            ("quantized_kv", bool(is_quantized_kv_cache(self.kv_cache_dtype))),
        )
        if any(v for _, v in _reasons):
            if (st is None and cfg is not None and cfg.variant == "fast"
                    and m is not None and output is not None
                    and m.max_query_len == 1 and kv_cache.numel() != 0):
                # NO-SILENT-FULLKV DETECTOR (no-fallback rule, 2026-07-22).
                # Every other disjunct above is a step-TYPE / layer-TYPE
                # split; `st is None` on a PURE DECODE step with a live cache
                # is not -- it means the builder never attached Stage-A state
                # and this step is served (and measured) as FullKV under the
                # locks tag.  The "[locks] ACTIVE" sentinel does not catch it
                # (it only proves the backend class was selected), so make it
                # greppable.  Boot phases cannot reach here: they either have
                # no metadata (m is None) or an empty cache.
                global _DELEG_SAID
                if not _DELEG_SAID:
                    _DELEG_SAID = True
                    print("[locks] FALLBACK no-state: pure-decode step with "
                          "NO Stage-A state -> stock FA (FullKV) for this "
                          "step. If this line appears after boot the cell is "
                          "NOT measuring locks.", flush=True)
            # BATCH-SPLIT (2026-07-22).  `max_query_len != 1` alone does NOT mean
            # "no decode work": under vLLM V1 chunked prefill a DECODE row rides
            # along with another sequence's prefill chunk.  Bailing the whole step
            # to stock FA served those rows as FullKV under the locks tag
            # (measured 12.50% activation at 125K/mns=4).  When that is the ONLY
            # tripped condition we now split the batch instead: prefill rows keep
            # stock FA (correct for them), decode rows get the identical Stage-A +
            # Stage-B path on a compacted sub-batch.  Mirrors the mem variant's
            # _forward_mem_batchsplit, which has always done this.
            import torch as _t
            _only_mql = (
                st is not None and m is not None and output is not None
                and kv_cache.numel() != 0 and cfg is not None
                and cfg.variant == "fast" and m.max_query_len != 1
                and all(not v for n, v in _reasons if n != "max_query_len!=1")
                and not _t.cuda.is_current_stream_capturing())
            if _only_mql:
                return self._forward_fast_batchsplit(
                    layer, query, key, value, kv_cache, m, output, st, cfg,
                    output_scale, output_block_scale)
            # Count DECODE ROWS that this delegation sends to stock FA. Derived
            # from query_start_loc (rows with query_len == 1). Pure prefill steps
            # contribute 0 and are correctly not counted as a loss.
            if m is not None and kv_cache.numel() != 0 and output is not None:
                qsl = getattr(m, "query_start_loc", None)
                if qsl is not None:
                    try:
                        n_dec = int((__import__("torch").diff(qsl) == 1).sum())
                    except Exception:                   # noqa: BLE001
                        n_dec = 0
                    if n_dec:
                        _ACT["decode_rows_dense"] += n_dec
                        key = ",".join(n for n, v in _reasons if v)
                        _ACT["dense_reasons"][key] = \
                            _ACT["dense_reasons"].get(key, 0) + n_dec
            if _runtime._FIXC and _runtime.fixc_pending():
                # FIX-C safety net: a deferred kv/rope pair must be consumed
                # by the locks pure-decode path THIS step; delegating with a
                # pending stash would leave the KV cache unwritten.
                raise RuntimeError(
                    "locks FIXC: deferred kv/rope pending on a delegated "
                    "(non-locks) attention path -- wiring bug")
            return super().forward(layer, query, key, value, kv_cache, m,
                                   output, output_scale=output_scale,
                                   output_block_scale=output_block_scale)

        # --- pure single-token decode: Stage A + Stage B (in-graph when the
        # real graph-safe selection is installed; eager/piecewise on the bridge).
        # KV of the new token is already in the paged cache (0.24 writes it via a
        # separate op before attention). ---------------------------------------
        _State, _derive, select_pages, _graph_safe = _runtime.resolve_selection()
        n_req = m.seq_lens.shape[0]
        # every row on this path is a decode row served SPARSELY (max_query_len==1)
        _ACT["decode_rows_sparse"] += n_req
        out3 = (output if output.dim() == 3
                else output.view(output.shape[0], self.num_heads, -1))

        st_use = st
        # Q SOURCE (2026-07-22).  Neither `_Q_ABSORBED` (process state) nor a
        # shape is the right test: consume the op's RECEIPT.  See
        # _absorbed_now.  A step the op served with the stock projection
        # carries ROPED q in the fused buffer and must be served exactly like
        # the stock lane -- same rows, same kernels, same selection; only the
        # pointer differs.
        if _Q_ABSORBED:
            assert not (_NOROPE and n_req > 1), (
                "LOCKS_NOROPE: multi-row pure decode is unsupported -- the "
                "rope nodes are pruned from EVERY captured uniform-decode "
                "bucket, but do_kv_cache_update only re-ropes k at nt==1, so "
                "rows>1 would cache garbage K. Run max_num_seqs=1, or use "
                "LOCKS_QFIRST_CO (whose op keeps stock rope + stock KV write "
                "at nt>1).")
            # This IS a pure-decode step (max_query_len == 1 -- it is why we are
            # here), so _absorbed_now reduces to the module constant _Q_ABSORBED:
            # the CO op absorbed on this step, NOROPE always does.  Inlined (not
            # _absorbed_now(1)) -- bs=1 is launch-bound and a module-constant
            # read is the cheapest possible signal, cheaper than the old nt==1.
            absorbed = _Q_ABSORBED
            # tells the score kernel whether its staging prelude must rope q
            # (and publish st.q_rope) or whether q arrives already roped.
            st_use._q_preroped = not absorbed
        else:
            absorbed = False
        select_pages((_norope_qk_views(self, value)[0] if absorbed
                      else query),
                     kv_cache, m.block_table, m.seq_lens, st_use,
                     n_req, self.scale, cfg)
        _runtime.decode_dispatch(
            (st_use.q_rope[:n_req] if absorbed else query),
            kv_cache, m.block_table, m.seq_lens, st_use, out3,
            ResidentVSource(kv_cache), self.scale, cfg)

        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(f"[locks] fast decode path ACTIVE (n_req={n_req} "
                  f"graph_safe={_graph_safe} "
                  f"budget={cfg.budget} coverage={cfg.coverage})", flush=True)
        return output

    # ---- mem-v write hook: K -> engine cache, V -> DRAM tier ------------- #
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Per-layer, in-graph KV write. For mem the new token's V (and K for
        mem-kv) is scattered into the DRAM tier's staging pool at this step's
        ``v_slot_mapping`` (filled by the builder's begin_step, OUTSIDE the
        graph). K stays in the engine cache.

          * K-only engine cache (spec patched, cfg.mem_k_only): stock
            reshape_and_cache_flash cannot run (no V half) -> K is scattered into
            the K-only cache via ``slot_mapping`` by the SAME tier kernel.
          * K+V engine cache (scaffold, mem_k_only off): stock write runs first
            (K+V resident) and the tier gets a byte-identical V copy to serve the
            tiered decode -- validates the pipeline without the spec patch.

        Non-mem / feature-off falls through to the stock write."""
        cfg = _runtime.get_config()
        if cfg is None or not cfg.is_mem or not _runtime.mem_stageb_wired():
            if _Q_ABSORBED and cfg is not None and kv_cache.numel() != 0:
                # WHICH WRITE.  On a CO pure-decode step the op deferred the KV
                # projection (fixc_arm_proj), so the stash is armed and the
                # write rides fixc_flush behind the select -- the op absorbs on
                # EVERY pure-decode step, so on those steps this is always the
                # path.  ``fixc_proj_armed()`` is set by, and only by, that
                # branch; a mixed step (op roped in place) leaves it empty and
                # the stock write below is the right write for the roped k.
                if _runtime._FIXC and _runtime.fixc_proj_armed():
                    _fixc_defer_write(self, value, kv_cache, slot_mapping)
                    return
                if _NOROPE and key.shape[0] == 1:
                    # NOROPE (round A) only: the trace's rope nodes are pruned
                    # for EVERY decode bucket, so k arrives unroped with no op
                    # to defer it.  T > 1 keeps rope in the forward.
                    _norope_fast_write(self, key, value, kv_cache,
                                       slot_mapping)
                    return
            return super().do_kv_cache_update(layer, key, value, kv_cache,
                                              slot_mapping)
        n_tokens = slot_mapping.shape[0]
        if n_tokens == 0 or kv_cache.numel() == 0:
            return
        tier = _runtime.get_tier()
        if tier is None:
            # vLLM's cudagraph memory profiler runs dummy forwards against a
            # throwaway minimal KV cache before the real pool exists, so the
            # tier cannot be built yet and that cache is never read. Skip the
            # write for the duration of that window ONLY (see
            # _runtime.in_profiling); a missing tier at any other time is a
            # hard error, as before.
            if _runtime.in_profiling():
                return
            # Round 6 diagnosis (GH200): the PIECEWISE capture group runs
            # BEFORE FULL, builds NO attention metadata (so no builder call
            # can construct the tier), and marks its dummy KV writes
            # skippable via slot_mapping filled with -1 -- vLLM's own
            # per-element no-write contract (gpu_model_runner.py:5833-5837),
            # which stock concat_and_cache AND the tier scatter kernels
            # ("vs >= 0" guard) already honor. Honor the whole-batch form
            # here: an all--1 mapping is a dummy run and needs no tier. The
            # host sync is safe by construction -- this branch is reachable
            # only BEFORE the tier exists, i.e. eager boot phases; FULL
            # capture builds the tier first (bfcc -> build() -> _ensure_tier)
            # and takes the tier path below, whose kernel records the
            # per-element guard into the graph.
            if int(slot_mapping.max().item()) < 0:
                return
            raise RuntimeError("locks mem: KV write before the tier was built")
        lidx = _runtime.tier_layer_index(tier, kv_cache)
        # K8d: an unconsumed stash from a previous layer means the decode
        # proof failed and that layer attended WITHOUT its KV write -- loud.
        if getattr(tier, "_k8d_stash", None) is not None:
            raise RuntimeError(
                "locks mem K8d: stale KV-write stash (layer %d) -- the "
                "pure-decode proof failed; refusing silent corruption"
                % tier._k8d_stash[0])
        if cfg.mem_k_only:
            # K8d (2026-07-19): at bs=1 pure decode under the K8 schedule the
            # scatter is absorbed into the decode split kernel's tail-page
            # CTA (decode_mem.py SCAT phase).  Stash; the ptrsel launch this
            # same layer consumes it.  Condition mirrors _forward_mem's
            # max_query_len==1 dispatch exactly (tier._cur_mql, set by
            # begin_step) -- if the forward takes any other path the stash
            # is never consumed and the check above raises next layer.
            from ..tier.decode_mem import k8d_active
            # in_profiling: _forward_mem returns output untouched in that
            # window (nothing is read), so a stash would never be consumed.
            armed = getattr(tier, "_k8d_armed", False)
            if lidx == tier.L - 1:
                # disarm: passes with no begin_step (piecewise capture's
                # KV-write-only group) must never stash.
                tier._k8d_armed = False
            if (armed and tier.offload_k
                    and k8d_active()
                    and not _runtime.in_profiling()
                    and getattr(tier, "_cur_mql", 0) == 1
                    and getattr(tier, "_cur_nreq", 0) == 1):
                k_stash = (_norope_qk_views(self, value)[1]
                           if _Q_ABSORBED else key)
                if tier.summary_cache:
                    # RECORDS (spec v2): there is no engine K half -- the
                    # engine page holds summary records, and the SCAT
                    # engine-K store would corrupt them.  That store is
                    # guarded by the kernel's own ``slot >= 0`` test, so a
                    # constant -1 slot mapping disables EXACTLY it; the
                    # staging stores (vsm_ptr path -- the same
                    # destinations write_kv_konly's scatter_kv hits)
                    # proceed unchanged.  (Records-round fix for the
                    # +0.05 un-absorbed-write regression.)
                    neg = getattr(tier, "_k8d_neg_slot", None)
                    if neg is None:
                        neg = tier._k8d_neg_slot = slot_mapping.new_full(
                            (1,), -1)
                    tier._k8d_stash = (lidx, k_stash, value,
                                       kv_cache.select(1, 0), neg)
                else:
                    tier._k8d_stash = (lidx, k_stash, value,
                                       kv_cache.select(1, 0), slot_mapping)
                return
            # K -> engine K-only cache (slot_mapping); V -> staging (v_slot).
            tier.write_kv_konly(lidx, key, value, kv_cache, slot_mapping)
        else:
            # Scaffold: K+V resident (stock write) + V copy into the tier.
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            tier.write_kv(lidx, value, n_tokens,
                          key=(key if cfg.offload_k else None))

    # ---- mem-v decode (the memory play) ---------------------------------- #
    def _forward_fast_batchsplit(self, layer, query, key, value, kv_cache, m,
                                 output, st, cfg, output_scale,
                                 output_block_scale):
        """BATCH-SPLIT forward for FAST prefill/mixed steps (2026-07-22).

        Chunked prefill co-schedules a decode row with another sequence's prefill
        chunk, so ``max_query_len != 1`` for the step even though genuine decode
        work is present.  Bailing the step to stock FA measured those rows as
        FullKV under the locks tag.  Here:

          * the stock-FA pass runs over the WHOLE batch first -- that is exactly
            right for the prefill rows, which need dense causal attention over
            their chunk plus resident prefix;
          * the decode rows are then recomputed on a compacted sub-batch with the
            SAME derive -> select -> decode chain the pure-decode path uses, and
            written over their slots.

        Overwriting is what keeps the two paths bit-identical: the sub-batch sees
        identical q / block_table / seq_lens, so selection and Stage-B are the
        same functions of the same inputs as at max_num_seqs=1.  The dense pass
        over decode rows is redundant work on mixed steps only; correctness of the
        measurement, not mixed-step latency, is the point.

        Eager by construction: vLLM full-graph-captures only uniform decode
        buckets, and the caller asserts we are not capturing.

        Q-ABSORBED LANES (2026-07-22).  This path used to assert
        ``not _NOROPE and not _Q_ABSORBED``.  It does not any more: the q source
        is READ FROM THE OP'S RECEIPT (``_absorbed_now``), never re-derived.
        This is NOT a fallback -- no row is served by a different attention
        method.  Every row of a mixed step goes through ONE projection and ONE
        rope (the op's stock branch, or NOROPE's untouched trace), and every
        decode row then goes through the SAME Stage-A + Stage-B chain as at
        max_num_seqs=1.  A mixed step takes the stock projection because a
        prefill row is not selected over at all, so there is no score kernel
        for its q rope to be absorbed into -- a difference in the WORK, not two
        implementations of one computation.
        """
        import torch
        assert not torch.cuda.is_current_stream_capturing(), \
            "locks fast: batch-split forward under CUDA-graph capture"
        # 1. dense pass for every row (prefill rows are final after this).
        super().forward(layer, query, key, value, kv_cache, m, output,
                        output_scale=output_scale,
                        output_block_scale=output_block_scale)
        n_req = m.seq_lens.shape[0]
        qsl_cpu = getattr(m, "locks_qsl_cpu", None)
        sl_cpu = getattr(m, "locks_sl_cpu", None)
        if qsl_cpu is None or sl_cpu is None:
            qsl_cpu = m.query_start_loc[:n_req + 1].cpu()
            sl_cpu = m.seq_lens[:n_req].cpu()
        ql = (qsl_cpu[1:n_req + 1] - qsl_cpu[:n_req]).tolist()
        sl = sl_cpu[:n_req].tolist()
        q0s = qsl_cpu[:n_req].tolist()
        dec_rows = [r for r in range(n_req) if ql[r] == 1 and sl[r] > 0]
        if not dec_rows:
            return output
        dev = query.device
        H = self.num_heads
        nt = m.num_actual_tokens
        # Q SOURCE under the rope-absorbing lanes (2026-07-22; this used to be
        # `assert not _NOROPE and not _Q_ABSORBED`).  A mixed step is served by
        # the op's stock branch, so the receipt says NOT absorbed and `query`
        # IS the roped q -- the same tensor the non-absorbed lanes read.
        # (`absorbed` is True here only for NOROPE, whose fused buffer is never
        # roped in place.)  Either way q
        # is gathered by TOKEN offset (tok_dev = each decode row's
        # query_start_loc) while st.q_rope below is indexed by the COMPACTED
        # row (the score kernel writes row r of the sub-batch it was handed).
        absorbed = _Q_ABSORBED and _absorbed_now(m.max_query_len)
        qsrc = _norope_qk_views(self, value)[0] if absorbed else query
        q3 = qsrc[:nt] if qsrc.dim() == 3 else qsrc[:nt].view(nt, H, -1)
        out3 = output[:nt] if output.dim() == 3 else output[:nt].view(nt, H, -1)
        # index tensors are step constants; cache them so the L layers of a step
        # do not each pay a blocking H2D (same trick as _forward_mem_batchsplit).
        _bs = getattr(m, "_locks_fast_bs_idx", None)
        if _bs is None:
            _bs = {
                "rows": torch.tensor(dec_rows, dtype=torch.long, device=dev),
                "tok": torch.tensor([q0s[r] for r in dec_rows],
                                    dtype=torch.long, device=dev),
            }
            m._locks_fast_bs_idx = _bs
        rows_dev, tok_dev = _bs["rows"], _bs["tok"]
        n_dec = len(dec_rows)
        q_dec = q3.index_select(0, tok_dev)
        bt_dec = m.block_table.index_select(0, rows_dev)
        sl_dec = m.seq_lens.index_select(0, rows_dev)
        out_dec = torch.empty_like(q_dec)
        from ..selection import derive_page_params
        _State, _derive, select_pages, _gs = _runtime.resolve_selection()
        derive_page_params(st, sl_dec, n_dec)
        st._q_preroped = not absorbed
        select_pages(q_dec, kv_cache, bt_dec, sl_dec, st, n_dec, self.scale, cfg)
        # The Q-first ops read (st, block_table, seq_lens, scale) stashed by the
        # select adapter on the PREVIOUS call and rely on the metadata tensors
        # being refreshed IN PLACE.  bt_dec/sl_dec are per-step index_select
        # COPIES, so leaving them stashed would let the next nt==1 step's layer-0
        # op score against a dead snapshot (wrong seq_len => wrong rope position
        # and wrong page count).  Drop them: with no stash the op takes its stock
        # nt==1 branch, which returns UNROPED q -- exactly the absorbed contract
        # the impl and the KV-write hook already implement.
        _runtime._qf2_refs = None
        _runtime.decode_dispatch(st.q_rope[:n_dec] if absorbed else q_dec,
                                 kv_cache, bt_dec, sl_dec, st, out_dec,
                                 ResidentVSource(kv_cache), self.scale, cfg)
        out3.index_copy_(0, tok_dev, out_dec)
        _ACT["decode_rows_sparse"] += n_dec
        if _Q_ABSORBED and not absorbed:
            _ACT["decode_rows_stock_proj"] += n_dec
        return output

    def _forward_mem(self, layer, query, key, value, kv_cache, m, output,
                     output_scale, output_block_scale):
        """mem attention. Pure single-token decode runs the SAME Stage-A select
        as fast, then the tier residency gather (miss pages -> hot buffer) and
        the tiered Stage-B decode (V from hot | staged | pinned; K resident for
        mem-v). Prefill / mixed steps take the BATCH-SPLIT forward (P4, redesign
        3.4.3): prefill rows -> leased-KV stock-FA varlen; decode rows -> the
        same Stage-A + tiered Stage-B on a compacted sub-batch. Mixed steps are
        always eager (vLLM full-graph-captures only uniform decode buckets).

        No fast/slow branch on the decode path: every pure decode step takes the
        tiered path (a single path); the split below is a step-TYPE split
        (decode vs prefill rows), not a latency fallback."""
        # forward() routes here only for mem variants, and the flag is the shipped
        # master switch (True). No feature-off decode fallback: assert, don't
        # branch (per the no-fallback rule). LOCKS_DISABLE=1 is the stock-FA line.
        assert _runtime.mem_stageb_wired(), "locks mem path reached with the flag off"
        cfg = _runtime.get_config()
        st = getattr(m, "locks", None) if m is not None else None
        tier = getattr(m, "locks_tier", None) if m is not None else None
        # vLLM 0.24 profiles cudagraph memory with a THROWAWAY minimal cache
        # that is non-empty (num_gpu_blocks=2), so numel()==0 below no longer
        # detects the profiling dummy run, and _dummy_run executes a FULL
        # forward, not just KV writes. Return the preallocated output untouched:
        # _dummy_run discards it, and delegating to stock FA is IMPOSSIBLE for
        # mem_k_only -- the spec patch pins the cache's K/V axis to 1
        # (register.py, head_size_v=0) while flash_attn does kv_cache.unbind(1)
        # expecting 2 ("not enough values to unpack"). No tier, no state, no
        # attention: nothing in this window is read.
        if _runtime.in_profiling():
            return output
        if (m is None or output is None or kv_cache.numel() == 0
                or getattr(m, "use_cascade", False)
                or m.causal is not True
                or self.attn_type != AttentionType.DECODER
                or self.sliding_window != (-1, -1)
                or self.alibi_slopes is not None
                or self.sinks is not None
                or getattr(self, "dcp_world_size", 1) > 1
                or is_quantized_kv_cache(self.kv_cache_dtype)):
            # profiling (no cache yet) / non-standard layer: stock FA.
            return super().forward(layer, query, key, value, kv_cache, m,
                                   output, output_scale=output_scale,
                                   output_block_scale=output_block_scale)
        if st is None or tier is None:
            raise RuntimeError(
                "locks mem: forward reached with locks state/tier missing "
                "(st=%r tier=%r) -- the boot-time VRAM plan (memplan) or the "
                "builder failed; refusing a silent dense fallback"
                % (st is not None, tier is not None))
        if m.max_query_len != 1:
            # prefill / mixed step: batch-split forward (eager by construction).
            return self._forward_mem_batchsplit(
                layer, query, key, value, kv_cache, m, output, st, tier, cfg)
        # --- pure single-token decode: Stage A + tier gather + tiered Stage B --
        from ..tier import mem_dynamic_decode
        lidx = _runtime.tier_layer_index(tier, kv_cache)
        n_req = m.seq_lens.shape[0]
        out3 = (output if output.dim() == 3
                else output.view(output.shape[0], self.num_heads, -1))
        _State, _derive, select_pages, _gs = _runtime.resolve_selection()
        import os as _os
        _dbg = _os.environ.get("LOCKS_MEM_DEBUG") == "1"
        side_done = False
        st_use = st
        # same q-source rule as the fast arm (2026-07-22).  The two gates here
        # used to disagree (_NOROPE for select, _Q_ABSORBED for decode); they
        # agreed only because at nt==1 the CO op's `query` IS the unroped view.
        absorbed = _Q_ABSORBED and _absorbed_now(m.max_query_len)
        st_use._q_preroped = not absorbed
        select_pages((_norope_qk_views(self, value)[0] if absorbed
                      else query),
                     kv_cache, m.block_table, m.seq_lens, st_use,
                     n_req, self.scale, cfg)
        if _dbg:
            import torch as _t
            _t.cuda.synchronize()
            print(f"[dbg] L{lidx} select OK cnt0={st_use.page_cnt[0].tolist()} "
                  f"sl={m.seq_lens[:2].tolist()} nreq={n_req}", flush=True)
        mem_dynamic_decode((st_use.q_rope[:n_req] if absorbed else query),
                           kv_cache, m.block_table, m.seq_lens, st_use,
                           out3, tier, lidx, scale=self.scale, fetch="kernel",
                           impl="auto", tier_stepped=side_done)
        if _dbg:
            import torch as _t
            _t.cuda.synchronize()
            print(f"[dbg] L{lidx} mem_decode OK", flush=True)
        global _ANNOUNCED
        if not _ANNOUNCED:
            _ANNOUNCED = True
            print(f"[locks] MEM decode path ACTIVE (variant={cfg.variant} "
                  f"n_req={n_req} k_only={cfg.mem_k_only} budget={cfg.budget} "
                  f"coverage={cfg.coverage})", flush=True)
        return output

    def _forward_mem_batchsplit(self, layer, query, key, value, kv_cache, m,
                                output, st, tier, cfg):
        """BATCH-SPLIT forward for mem prefill / mixed steps (redesign 3.4.3).

        Partition rows by query length (host tensors the builder stashed on the
        metadata, sync-free):

          * prefill rows -> stock flash_attn_varlen over the LEASE (the
            continuation / just-leased request) or over the direct q/k/v
            tensors (a prefill fully contained in this step: no prior context).
          * decode rows  -> compacted Stage-A selection + tiered Stage-B decode
            (identical math to the pure-decode path; page/derive params are
            re-derived on the compacted rows).

        Replaces ``_prefill_konly`` (single-shot-only, ``.item()`` under
        capture: the measured cudaErrorStreamCaptureInvalidated at
        nsys_H200_AMM_*_p16_b1.log:234) and kills the mnbt >= ctx constraint.
        Always eager: vLLM only full-graph-captures uniform decode buckets."""
        import torch
        assert not torch.cuda.is_current_stream_capturing(), \
            "locks mem: batch-split (prefill/mixed) forward under CUDA-graph " \
            "capture -- capture batches must be uniform single-token decode"
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        qsl_cpu = getattr(m, "locks_qsl_cpu", None)
        sl_cpu = getattr(m, "locks_sl_cpu", None)
        n_req = m.seq_lens.shape[0]
        if qsl_cpu is None or sl_cpu is None:
            # no host view stashed (foreign builder path): one guarded sync.
            qsl_cpu = m.query_start_loc[:n_req + 1].cpu()
            sl_cpu = m.seq_lens[:n_req].cpu()
        ql = (qsl_cpu[1:n_req + 1] - qsl_cpu[:n_req]).tolist()
        sl = sl_cpu[:n_req].tolist()
        q0s = qsl_cpu[:n_req].tolist()
        lidx = _runtime.tier_layer_index(tier, kv_cache)
        H = self.num_heads
        nt = m.num_actual_tokens
        # q source: identical rule to the fast batch-split (2026-07-22).  The
        # PREFILL rows below need the ROPED q for stock FA, so the roped
        # `query` is used for them unconditionally; only the decode-row gather
        # follows `absorbed`.
        absorbed = _Q_ABSORBED and _absorbed_now(m.max_query_len)
        q3 = (query[:nt] if query.dim() == 3
              else query[:nt].view(nt, H, -1))
        q3_dec_src = (_norope_qk_views(self, value)[0][:nt] if absorbed
                      else q3)
        out3 = (output[:nt] if output.dim() == 3
                else output[:nt].view(nt, H, -1))
        dev = query.device
        lease = getattr(m, "locks_lease", None)
        lease_row = int(getattr(m, "locks_lease_row", -1))

        dec_rows = [r for r in range(n_req) if ql[r] == 1 and sl[r] > 0]
        pre_rows = [r for r in range(n_req) if ql[r] > 1]

        # The per-step device index tensors below (per-row cu_seqlens, and the
        # decode-row gather indices) are IDENTICAL across all L layers of a step
        # -- they are pure functions of ql / sl / q0s / dec_rows, which are step
        # constants. Building them fresh in every layer costs ~L blocking
        # torch.tensor H2D syncs per step (profiled at 16K/bs8: 0.34 s/step,
        # ~90% of a mixed step at low decode count). Build them ONCE and cache on
        # the step metadata (a fresh object per step -> the cache is naturally
        # step-scoped; the batch-split path is eager-only, asserted above, so
        # there is no cudagraph aliasing). The values are bitwise-identical to
        # the per-layer construction, so mem==fast is unaffected.
        _bs = getattr(m, "_locks_bs_idx", None)
        if _bs is None:
            _bs = {
                "cu_q": {r: torch.tensor([0, ql[r]], dtype=torch.int32,
                                         device=dev) for r in pre_rows},
                "cu_k": {r: torch.tensor([0, sl[r]], dtype=torch.int32,
                                         device=dev)
                         for r in pre_rows if r == lease_row},
                "rows_dev": (torch.tensor(dec_rows, dtype=torch.long, device=dev)
                             if dec_rows else None),
                "tok_dev": (torch.tensor([q0s[r] for r in dec_rows],
                                         dtype=torch.long, device=dev)
                            if dec_rows else None),
            }
            m._locks_bs_idx = _bs

        # ---- prefill rows ---------------------------------------------------
        for r in pre_rows:
            t0, t1 = sl[r] - ql[r], sl[r]
            q0, q1 = q0s[r], q0s[r] + ql[r]
            cu_q = _bs["cu_q"][r]
            if r == lease_row:
                # leased request: fill this layer's lease slice from the tier
                # pools (staged for this chunk; pinned only on recovery), then
                # stock FA over the contiguous lease prefix.
                lease.fill_layer(lidx, m.block_table[r])
                cu_k = _bs["cu_k"][r]
                flash_attn_varlen_func(
                    q=q3[q0:q1], k=lease.k[lidx][:t1], v=lease.v[lidx][:t1],
                    out=out3[q0:q1], cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=ql[r], max_seqlen_k=t1,
                    softmax_scale=self.scale, causal=True,
                    fa_version=self.vllm_flash_attn_version)
            elif t0 == 0:
                # prefill fully contained in this step: direct q/k/v.
                flash_attn_varlen_func(
                    q=q3[q0:q1], k=key[q0:q1], v=value[q0:q1],
                    out=out3[q0:q1], cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
                    max_seqlen_q=ql[r], max_seqlen_k=ql[r],
                    softmax_scale=self.scale, causal=True,
                    fa_version=self.vllm_flash_attn_version)
            else:
                raise RuntimeError(
                    f"locks mem: continuation prefill row {r} (ctx={t0}) has no "
                    f"lease (lease_row={lease_row}) -- builder lease plan bug "
                    "or max_num_partial_prefills > 1")

        # ---- decode rows (compacted sub-batch) ------------------------------
        if dec_rows:
            from ..selection import derive_page_params
            from ..tier import mem_dynamic_decode
            rows_dev = _bs["rows_dev"]
            tok_dev = _bs["tok_dev"]
            n_dec = len(dec_rows)
            q_dec = q3_dec_src.index_select(0, tok_dev)
            bt_dec = m.block_table.index_select(0, rows_dev)
            sl_dec = m.seq_lens.index_select(0, rows_dev)
            out_dec = torch.empty_like(q_dec)
            _State, _derive, select_pages, _gs = _runtime.resolve_selection()
            derive_page_params(st, sl_dec, n_dec)
            st._q_preroped = not absorbed
            select_pages(q_dec, kv_cache, bt_dec, sl_dec, st, n_dec,
                         self.scale, cfg)
            _runtime._qf2_refs = None      # see the fast batch-split
            mem_dynamic_decode((st.q_rope[:n_dec] if absorbed else q_dec),
                               kv_cache, bt_dec, sl_dec, st, out_dec,
                               tier, lidx, scale=self.scale, fetch="kernel",
                               impl="auto")
            out3.index_copy_(0, tok_dev, out_dec)
        return output
