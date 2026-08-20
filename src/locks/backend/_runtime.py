"""Backend runtime singletons: the one LocksConfig and the Stage-A resolver.

register() builds the single :class:`LocksConfig` and stashes it here; the impl
and builder read it (avoids a register<->attn import cycle). ``resolve_selection``
returns the Stage-A entry points, preferring the REAL graph-safe ``locks.selection``
(the STATIC r8-ranked pipeline) and falling back to the eager validation bridge in
``_selection_bridge.py`` when it is not available (or ``LOCKS_FORCE_BRIDGE=1``).

The real Stage-A (``locks.selection``, LOCKS_R8_STATIC_SPEC.md) is the STATIC
r8-ranked pipeline: ``R8State`` (r8 codes keyed by physical block, L-major),
``derive_page_params(st, seq_lens, n_req)`` (per-step, one launch), and
``select_pages_r8(st, layer, q, block_table, seq_lens, n_req, scale)`` which reads
the per-layer r8 codes -- codes that must first be BUILT from finalized pages via
``r8_build_refresh(st, layer, K, block_table, seq_lens, n_req)`` (eigh-based, NOT
graph-safe, tag-gated, runs on page-finalize off the hot path).

WIRING (this agent). The lifecycle the earlier scaffold flagged is now built in
``builder.py``:

  * R8State needs ``num_gpu_blocks`` (unknown at builder __init__) -> the builder
    constructs it LAZILY at the first build() where the cache is allocated, and
    keys a ``kv_cache.data_ptr() -> layer index`` map from
    ``compilation_config.static_forward_context`` (the same tensor object the
    impl.forward receives -- verified in vLLM 0.24 attention.get_attention_context).
  * the eigh build/refresh runs in build() OUTSIDE the graph (host, before graph
    replay), per layer, over the layer's resident K half, tag-gated so only
    newly-finalized pages rebuild. Skipped on cudagraph-capture dummy batches
    (``build_for_cudagraph_capture`` sets a capture flag), exactly like the DRAM
    tier's ``begin_step(capture=...)``.
  * ``derive_page_params`` runs once per step in build() (graph-safe, one launch),
    so the per-layer select kernels inside the graph just read the refreshed
    persistent buffers.

The impl (``attn.py``) calls the select fn returned here with the BRIDGE-compatible
signature ``(q, kv_cache, block_table, seq_lens, st, n_req, scale, cfg)`` and never
sees a layer index, so :func:`_r8_select_adapter` recovers the layer from the
``st._layer_of`` map (populated by the builder) and calls ``select_pages_r8`` with
``derive=False`` (the builder already derived this step). Stage-B decode is
selector-agnostic: it consumes ``st.page_table`` / ``st.page_cnt`` plus the Stage-B
partials, so swapping the bridge for the r8 path does not touch attention/.
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import LocksConfig

_config: Optional[LocksConfig] = None

# The real graph-safe r8 Stage A is now wired (builder constructs R8State at first
# build, runs the r8_build_refresh page-finalize hook outside the graph, and
# derives per-step params). Set LOCKS_FORCE_BRIDGE=1 to force the eager validation
# bridge instead (kept as a fallback / A-B reference; not CUDA-graph capturable).
# (_R8_WIRED, constant True since the r8 land, folded out 2026-07-21.)

# mem Stage-B readiness. TRUE (shipped 2026-07): the bitwise gate PASSED end to
# end on sm_120 -- mem-kv greedy decode is token-identical to the fast resident
# sparse decode (scratch_mem_e2e: [11,18327,11,4400,13,362,17136,18640] under
# BOTH eager and FULL-cudagraph replay), and the kernel-level torch.equal gates
# (tests/test_mem_tier.py, test_mem_cuda.py) hold for the tiered vs resident
# decode. All four legs are live: (a) the V_SRC==1 3-way tiered decode kernel
# arm; (b) the per-step KV write scatter (attn.do_kv_cache_update ->
# tier.write_kv_konly, K into pool_k for mem-kv); (c) the residency gather
# (tier.step, inside mem_dynamic_decode(fetch="kernel") from _forward_mem before
# Stage-B -- steady-state h/s/p=1016/8/0, ZERO pinned reads); (d) the K-only
# KV-cache-spec patch (register._patch_kv_cache_spec_k_only, num_gpu_blocks 2x).
# There is NO feature-off decode fallback: the mem forward asserts this flag (the
# no-fallback rule). LOCKS_DISABLE=1 (register inert) is the stock-FA/FullKV line.
# (LOCKS_MEM_FORCE, an OR-latch over a hardcoded True, removed 2026-07-21 --
# cleanup Wave 2; it could never change the result.)
def mem_stageb_wired() -> bool:
    return True


def set_config(cfg: LocksConfig) -> None:
    global _config
    _config = cfg


def get_config() -> Optional[LocksConfig]:
    return _config


# The single DRAM Tier (mem variants). The builder publishes it here at first
# build so the impl's write hook (``do_kv_cache_update``, which receives no
# attn_metadata) and forward can reach it without threading it through vLLM.
_tier = None


def set_tier(tier) -> None:
    global _tier
    _tier = tier


def get_tier():
    return _tier


# vLLM 0.24 profiles cudagraph memory (gpu_worker.profile_cudagraph_memory)
# by allocating a THROWAWAY minimal KV cache
# (gpu_model_runner._init_minimal_kv_cache_for_profiling) and running dummy
# forwards against it. In that window the real block pool does not exist, so
# the DRAM tier legitimately cannot be built, and the dummy cache's contents
# are discarded (_cleanup_profiling_kv_cache). register() brackets that call
# to mark the window; the mem write hook uses it to SKIP the write instead of
# raising. This is PHASE DETECTION, not a fast/slow fallback: outside the
# window a missing tier stays a hard error. It is a flag rather than a
# cache_config probe because _init_minimal_kv_cache_for_profiling SETS
# num_gpu_blocks (to the minimal count) before the dummy run, so that field
# cannot distinguish the phase.
_in_profiling = False


def set_profiling(on: bool) -> None:
    global _in_profiling
    _in_profiling = bool(on)


def in_profiling() -> bool:
    return _in_profiling


def tier_layer_index(tier, kv_cache) -> int:
    """Physical layer index for ``kv_cache`` from the tier's ptr map (built by
    the builder from static_forward_context, the exact tensor the impl sees)."""
    layer_of = getattr(tier, "_layer_of", None)
    if layer_of is None:
        raise RuntimeError("locks mem: tier has no _layer_of map (builder wiring)")
    lidx = layer_of.get(kv_cache.data_ptr())
    if lidx is None:
        raise RuntimeError(
            "locks mem: kv_cache pointer not in the tier layer map "
            f"(ptr={kv_cache.data_ptr():#x}); check static_forward_context wiring")
    return lidx


def rki4_selection_available() -> bool:
    """True when locks.selection exposes the rki4 Stage-A surface.
    (Import errors propagate -- see r8_selection_available.)"""
    from .. import selection as sel
    return all(getattr(sel, n, None) is not None for n in
               ("Rki4State", "derive_page_params", "rki4_build_refresh"))




def _force_bridge() -> bool:
    return os.environ.get("LOCKS_FORCE_BRIDGE") not in (None, "", "0")


_BRIDGE_SAID = False      # one-shot sentinel for the explicit bridge escape


# Kernel-backend latches: None = untried, True = CUDA in use, False = Triton
# fallback (a build failed on the first eager call; never re-tried).
_SEL_CUDA = None
_DEC_CUDA = None


def _layer_index(st, kv_cache) -> int:
    layer_of = getattr(st, "_layer_of", None)
    if layer_of is None:
        raise RuntimeError(
            "locks r8: R8State has no _layer_of map (builder did not wire it)")
    lidx = layer_of.get(kv_cache.data_ptr())
    if lidx is None:
        raise RuntimeError(
            "locks r8: kv_cache pointer not in the builder's layer map "
            f"(ptr={kv_cache.data_ptr():#x}); the r8 state is L-major and needs "
            "the physical layer index. Check static_forward_context wiring.")
    return lidx


_PPSEL_ANNOUNCED = False

# --- LOCKS_CUPROF: worker-side cudaProfilerApi bracket (instrumentation) --- #
# nsys --capture-range=cudaProfilerApi needs cudaProfilerStart() from INSIDE
# every TP worker process: the driver-side hook (bench_lat BENCH_PROFILER_HOOK)
# never reaches the 4 TP workers, and delayed attach misses graph-node
# registration, so TP>1 decode kernels were uncapturable. The builder calls
# ``cuprof_step(is_decode, capturing)`` from build() -- the per-step host
# prologue that runs in EVERY worker -- and this module decides when to
# bracket. Two addressing modes (both count PURE-DECODE builds; capture dummy
# builds are excluded by the ``capturing`` flag):
#   LOCKS_CUPROF_START_STEP=n / LOCKS_CUPROF_STOP_STEP=m
#       absolute pure-decode build counters (start at the n-th, stop at the
#       m-th; the stop fires in the m-th step's prologue, so steps n..m-1 are
#       fully inside the window).
#   LOCKS_CUPROF_TRANS=k [LOCKS_CUPROF_STEPS=K, default 12]
#       arm at the k-th prefill->decode TRANSITION (a build whose batch is
#       pure decode after a non-pure-decode build): capture the FIRST K
#       pure-decode steps of that phase. This self-locates the first decode
#       step after a prefill (where the serialized summary refresh runs)
#       without counting warmup steps, which vary (wait_steady iterates).
# Every step also drops an NVTX mark "locks.step d=<n>" while the bracket is
# armed, so per-step boundaries are recoverable from the trace. Pure
# instrumentation: default-off, zero work when unset.
_CUP_START = int(os.environ.get("LOCKS_CUPROF_START_STEP", "-1"))
_CUP_STOP = int(os.environ.get("LOCKS_CUPROF_STOP_STEP", "-1"))
_CUP_TRANS = int(os.environ.get("LOCKS_CUPROF_TRANS", "-1"))
_CUP_STEPS = int(os.environ.get("LOCKS_CUPROF_STEPS", "12"))
_CUP_ENABLED = _CUP_START >= 0 or _CUP_TRANS >= 0
_cup_dec = 0          # pure-decode builds seen (1-based after increment)
_cup_trans = 0        # prefill->decode transitions seen
_cup_prev_dec = False
_cup_on = False
_cup_done = False
_cup_stop_at = -1


# (The sm_120 dispatch rows were removed 2026-07-21 with the sm_120 lane,
#   user ruling: Blackwell is no longer a paper claim.  The lane's history
#   (CUDA-whenever-cfg.use_cuda, the never-run in-situ race) is recorded in
#   ours_doc/REFUTED_ARMS_INDEX.md; recover from tag pre-cleanup-2026-07-21.)
_DECODE_TABLE = {
    ("sm90", "bs1"): "triton",
    ("sm90", "bs2_7"): "triton",
    ("sm90", "bs8+"): "triton",
}
_DEC_RESOLVED: dict = {}



def cuprof_step(is_decode: bool, capturing: bool = False) -> None:
    """Per-step worker-side profiler bracket; called from builder.build()."""
    global _cup_dec, _cup_trans, _cup_prev_dec, _cup_on, _cup_done, _cup_stop_at
    if not _CUP_ENABLED or _cup_done or capturing:
        return
    import torch
    if is_decode and not _cup_prev_dec:
        _cup_trans += 1
    _cup_prev_dec = is_decode
    if not is_decode:
        return
    _cup_dec += 1
    if not _cup_on:
        hit = (_cup_dec == _CUP_START) if _CUP_TRANS < 0 else \
              (_cup_trans == _CUP_TRANS and _cup_stop_at < 0)
        if not hit:
            return
        _cup_stop_at = _CUP_STOP if _CUP_TRANS < 0 else _cup_dec + _CUP_STEPS
        try:
            torch.cuda.profiler.start()
            _cup_on = True
            print(f"[locks cuprof] START pid={os.getpid()} "
                  f"dec_step={_cup_dec} trans={_cup_trans} "
                  f"stop_at={_cup_stop_at}", flush=True)
        except Exception as e:  # noqa: BLE001
            _cup_done = True
            print(f"[locks cuprof] start FAILED: {e}", flush=True)
            return
    # in-window: per-step boundary mark (includes the start step itself)
    torch.cuda.nvtx.mark(f"locks.step d={_cup_dec}")
    if _cup_stop_at >= 0 and _cup_dec >= _cup_stop_at:
        try:
            torch.cuda.synchronize()   # flush the window's tail into capture
            torch.cuda.profiler.stop()
            print(f"[locks cuprof] STOP pid={os.getpid()} "
                  f"dec_step={_cup_dec}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[locks cuprof] stop FAILED: {e}", flush=True)
        _cup_on = False
        _cup_done = True


# --- LOCKS_DUMP_PTAB: empirical page-ranking gate artifact ----------------- #
# User ruling 2026-07-17: decode-numerics arms (pps/split class) are gated by
# an EMPIRICAL selection-identity artifact instead of token shas (which are
# advisory for that class): at pure-decode step LOCKS_DUMP_PTAB_STEP (counted
# like cuprof, capture builds excluded) every worker dumps the selection
# tables (stale mode: st.pp_ptab/pp_pcnt, the per-layer persistent tables;
# else st.page_table/page_cnt) to LOCKS_DUMP_PTAB.pid<pid>.pt. Two arms'
# dumps must be EXACTLY equal (selection precedes decode). Gate-run-only
# instrumentation: default-off, zero work when unset.
_DUMP_PTAB = os.environ.get("LOCKS_DUMP_PTAB", "")
_DUMP_STEP = int(os.environ.get("LOCKS_DUMP_PTAB_STEP", "15"))
_dump_dec = 0
_dump_done = False


def ptab_dump_step(st, is_decode: bool, capturing: bool = False,
                   seq_lens=None) -> None:
    global _dump_dec, _dump_done
    # Dummy decode builds (profiling AND cudagraph/compile warmup: tiny
    # fake seqs -> cnt 0/1) must not advance the step counter, or low
    # DUMP_STEP values bind to dummy state instead of real selections
    # (found via the step-2 gate).  Real cells are >=16K tokens; every
    # dummy class is far below 1024.  Gate-run-only, sync acceptable.
    if (not _DUMP_PTAB or _dump_done or capturing or not is_decode
            or st is None or in_profiling()):
        return
    if seq_lens is not None and int(seq_lens.max()) < 1024:
        return
    # Bind the counter to REAL selection states, not build ordinals: after
    # boot, page_cnt holds init/warmup leftovers (0/1 — dummy selects run
    # during cudagraph warmup, and the fast arm has an extra large-seq
    # build).  A state qualifies once a real top-b landed (cnt >= 64).
    # DUMP_STEP=N therefore means: the state after the N-th REAL decode
    # step, identically on both arms — step 1 is the cross-arm gate.
    if int(st.page_cnt.max()) < 64:
        return
    _dump_dec += 1
    if _dump_dec != _DUMP_STEP:
        return
    _dump_done = True
    import torch
    try:
        if getattr(st, "pp_ptab", None) is not None:
            blob = {"pp_ptab": st.pp_ptab.cpu(), "pp_pcnt": st.pp_pcnt.cpu()}
        else:
            blob = {"page_table": st.page_table.cpu(),
                    "page_cnt": st.page_cnt.cpu()}
        # Certified-screening design data (doc 26): the per-head and combined
        # scores of the LAST prep before this step -- real-data boundary/tie
        # density for the screen's error-radius sizing. Gate-run-only.
        for k in ("score_h", "score"):
            v = getattr(st, k, None)
            if v is not None:
                blob[k] = v.cpu()
        try:                       # stable cross-run identity: TP rank
            import torch.distributed as dist
            rid = dist.get_rank() if dist.is_initialized() else os.getpid()
        except Exception:  # noqa: BLE001
            rid = os.getpid()
        path = f"{_DUMP_PTAB}.r{rid}.pt"
        torch.save(blob, path)
        print(f"[locks] PTAB DUMP at pure-decode step {_dump_dec} -> {path} "
              f"({','.join(f'{k}:{tuple(v.shape)}' for k, v in blob.items())})",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[locks] PTAB DUMP FAILED: {e}", flush=True)


# --- LOCKS_CNT_AUDIT=1: page_cnt-vs-invariant FIELD RATE instrument -------- #
# The ctx>=128K page-parallel finish keeps `score >= tau` with no tie
# resolution (defect, 2026-07-22), so an exact fp32 tie AT tau keeps a
# SUPERSET of the top-b.  Adversarial cases overspend by up to the whole
# context; this instrument answers whether it fires on REAL data.  The
# accumulator kernel (select_cuda.sel_cnt_audit) runs INSIDE the captured
# decode graph, one launch per (layer, step), so it covers every (step,
# layer, request, kv-head) unit; this reader prints the running totals every
# LOCKS_CNT_AUDIT_EVERY decode steps (host sync -- gate runs only, never a
# timing cell).  No launch and no allocation when the env is unset.
_CNT_AUDIT = os.environ.get("LOCKS_CNT_AUDIT", "0") == "1"
_CNT_EVERY = int(os.environ.get("LOCKS_CNT_AUDIT_EVERY", "16"))
_cnt_dec = 0


def cnt_audit_step(st, is_decode: bool, capturing: bool = False) -> None:
    global _cnt_dec
    if not _CNT_AUDIT or st is None or not is_decode or capturing:
        return
    acc = getattr(st, "_sp_cnt_audit", None)
    if acc is None:
        return
    _cnt_dec += 1
    if _cnt_dec % _CNT_EVERY:
        return
    v = [int(x) for x in acc.cpu().tolist()]
    u = max(v[0], 1)
    print(f"[locks] CNTAUDIT dec_step={_cnt_dec} units={v[0]} viol={v[1]} "
          f"viol_frac={v[1] / u:.6f} max_cnt={v[2]} mean_cnt={v[3] / u:.3f} "
          f"mean_inv={v[4] / u:.3f} max_excess={v[5]} sum_excess={v[6]} "
          f"hist[eq={v[7]},1-8={v[8]},9-64={v[9]},65-512={v[10]},"
          f">512={v[11]}] keepall_skipped={v[12]}", flush=True)


# --- LOCKS_SEG_TIMING=1: real-schedule segment walls via CUDA events ------- #
# Events recorded inside the captured decode graph REPLAY with it, so their
# elapsed times measure the REAL schedule. nsys --cuda-graph-trace=node
# SERIALIZES graph execution (measured: hidden-by-concurrency 0.01ms/step in
# the trace vs an ~83% differential hide rate un-traced), so traces cannot
# attribute the step; these brackets can (SELECT_KERNEL_CAMPAIGN.md 4c).
# The python-side dict state does not advance during graph replay, so every
# replay rewrites the SAME event pair per (segment, layer): the atexit dump
# reads the LAST step's real per-layer walls after a final synchronize.
_SEG_ON = os.environ.get("LOCKS_SEG_TIMING", "0") == "1"
_SEG_EV: dict = {}
_SEG_REGISTERED = False


def _seg_register():
    global _SEG_REGISTERED
    if _SEG_REGISTERED:
        return
    _SEG_REGISTERED = True
    import atexit

    def _dump():
        import torch
        try:
            torch.cuda.synchronize()
        except Exception:
            return
        tot: dict = {}
        n: dict = {}
        for (seg, lidx), (a, b) in sorted(_SEG_EV.items()):
            try:
                ms = a.elapsed_time(b)
            except Exception:
                continue
            tot[seg] = tot.get(seg, 0.0) + ms
            n[seg] = n.get(seg, 0) + 1
        print("[locks] SEG_TIMING last-step REAL walls: "
              + "  ".join(f"{k}={v:.3f}ms/{n[k]}L" for k, v in sorted(tot.items())),
              flush=True)
    atexit.register(_dump)


def _seg_pair(key):
    import torch
    p = _SEG_EV.get(key)
    if p is None:
        # external=True (cudaEventRecordExternal): required for events whose
        # record lands inside a captured graph to stay timeable across
        # replays (plain graph-node events reject cudaEventElapsedTime).
        try:
            mk = lambda: torch.cuda.Event(enable_timing=True, external=True)  # noqa: E731
            p = (mk(), mk())
        except TypeError:                      # older torch: no external flag
            p = (torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True))
        _SEG_EV[key] = p
        _seg_register()
    return p


# --------------------------------------------------------------------------- #
# QF2 (2026-07-19, user lane): score OFF the critical path.  The patched GLM
# forward (register.py) projects Q first, ropes it, and calls
# torch.ops.locks_qf.prescore(q, lidx) BEFORE the KV projection; the op forks
# the score (rki4_score_only, the same kernel/bytes) onto a side stream so it
# overlaps KV-GEMM + rope-k + kv_write.  The select adapter below consumes it
# (event wait + select_only) instead of running the serial score.  Selection
# math untouched: same q bytes (QF1 gate: scores BITWISE via the int8
# quantizer), same kernels, scheduling only.  Step identity: a nonce bumped
# in the derive wrapper (per-step) guards against stale prescores; under
# graph REPLAY none of this python runs -- the captured graph froze the side
# schedule at capture time (the side_step/lookahead precedent).
_QF2 = (os.environ.get("LOCKS_QFIRST2", "0") == "1"
        or os.environ.get("LOCKS_QFIRST_GV", "0") == "1")
# P2b co-kernel lane: score computed INSIDE the locks_co.qkv op
# (main-stream, no events); the adapter consumes via _co_mark.
_QCO = os.environ.get("LOCKS_QFIRST_CO", "0") == "1"
_co_mark: dict = {}

# ---------------------------------------------------------------------- #
# CO RECEIPT (2026-07-22, BATCHED_CO_DESIGN 9.1).  The trap that produced
# LongBench qasper 5.48 vs 47.44 with every detector green: ``_Q_ABSORBED``
# is a process-wide env flag, but whether the fused QKV buffer holds UNROPED
# q is a property of the CALL.  The impl used to INFER the latter from the
# former, then from a shape (``nt == 1``) -- a RE-DERIVATION of the op's
# decision, which is correct only for as long as the two formulas agree.
#
# HOW THE IMPL KNOWS THE Q SOURCE (2026-07-22).  Not a process flag, not a
# consume-once RECEIPT (both fail), but the ONE metadata quantity the op's
# absorbed decision is a function of: ``max_query_len``.
#
# A receipt (the op stamps what it did, the impl reads it) was tried and is
# WRONG under vLLM's piecewise CUDA graphs: the projection op is captured, so
# its python (the stamp) does NOT re-run on a replayed decode step, while the
# attention impl runs EAGERLY every step and reads a stale stamp -- a prefill's,
# for example (`want nt=4, got (8192, False)`).  The eager accuracy lane, which
# pairs op->impl 1:1, never hit it; the graph efficiency lane crashed on it.
#
# The op absorbs (returns UNROPED q, defers the KV write) on EVERY pure-decode
# step and ropes in place on mixed/prefill steps -- i.e. its decision is exactly
# ``max_query_len == 1``.  The impl reads the SAME ``max_query_len`` from its own
# metadata, so op and impl AGREE BY CONSTRUCTION, with no per-step op state for
# graph replay to drop.  This is a re-derivation from the op's REAL condition,
# not from a proxy (the shape ``nt==1``) that diverged from it -- which is why
# it does not reintroduce the 5.48 double-rope.  attn._absorbed_now is the impl
# side; coqkv_cuda.ensure_co_op is the op side.

# FIX-C (session-P lever 1, opt-in 2026-07-22, ours_doc/
# H200_SESSION_P_RESULTS.md route decision): union-kernel retirement +
# select-cover schedule at nt==1 under the CO wire.  The locks_co.qkv op
# runs q_gemv + the DEPLOYED-TU score (rki4_score_only: bt numerics class
# = the CO-OFF/>=128K reference class, NOROPE prelude ropes q into
# st.q_rope, hmax published) and DEFERS the kv projection; the write hook
# defers rope+cache; the select adapter flushes both DIRECTLY BEHIND the
# select launch, where select_v2's -DLOCKS_FIXC ENTRY trigger releases
# them as PSS dependents under its 4-CTA window (kv_gemv_nf: no gdc
# fence, inputs predate the chain -- safety argument in gemv_cuda.py;
# rope_and_cache_fixc: gdc fence on kv).  The decode split launches PLAIN
# behind them and closes every hazard.  Requires the CO wire (+HMAX+
# SEL_V2, asserted in register.py); inert without LOCKS_QFIRST_CO=1, so
# the >=128K rungs (CO ctx-gated off) are untouched by construction.
_FIXC = (os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1") and _QCO
_FIXC_POSCHK = os.environ.get("LOCKS_FIXC_POSCHK", "0") == "1"
_fixc_proj = None    # (step, lidx, h, W, bias, out_kv, qs, positions)
_fixc_write = None   # (step, lidx, k_un, value, kres, vres, slot, sl, csc, pg)


def fixc_arm_proj(lidx: int, h, W, bias, out_kv, qs: int, positions) -> None:
    """Arm the deferred kv projection (co-op body, FIX-C branch).

    ``positions`` is the op's own per-row rope position vector; it travels
    with the stash because the KV-write hook (do_kv_cache_update) is not
    handed positions by vLLM, and deriving the position from ``seq_lens - 1``
    is exactly the single-row assumption the unified rope+cache kernel
    removes (BATCHED_CO_DESIGN 2.5)."""
    global _fixc_proj
    assert _fixc_proj is None, (
        "locks FIXC: un-consumed deferred kv projection "
        f"(lidx {_fixc_proj[1]}) while arming lidx {lidx}")
    _fixc_proj = (_qf2_step, lidx, h, W, bias, out_kv, qs, positions)


def fixc_proj_armed() -> bool:
    return _fixc_proj is not None


def fixc_arm_write(k_un, value, kres, vres, slot, sl, csc, page: int) -> None:
    """Arm the deferred KV write (attn.do_kv_cache_update, FIX-C branch)."""
    global _fixc_write
    assert _fixc_proj is not None, (
        "locks FIXC: KV-write deferral without a projection stash")
    assert _fixc_write is None, "locks FIXC: un-consumed deferred KV write"
    _fixc_write = (_qf2_step, _fixc_proj[1], k_un, value, kres, vres, slot,
                   sl, csc, page)


def fixc_pending() -> bool:
    return _fixc_proj is not None or _fixc_write is not None


def fixc_flush(lidx: int) -> None:
    """FIX-C: launch the deferred [kv_gemv_nf -> rope_and_cache_fixc] pair
    DIRECTLY BEHIND the select launch (same stream).  select_v2 fires
    launch_dependents at ENTRY (-DLOCKS_FIXC build), kv_gemv_nf is its PSS
    dependent (NO gdc fence: its inputs -- rms output + static weights --
    are complete and visible before select begins, see the kernel's safety
    comment), rope_and_cache_fixc is kv's PSS dependent (gdc fence: it
    reads kv's output; kv fires per-CTA exit triggers).  The decode split
    launches PLAIN afterwards and waits select+kv+rope by stream order."""
    global _fixc_proj, _fixc_write
    from ..tier import k5_cuda
    from . import gemv_cuda
    p, w = _fixc_proj, _fixc_write
    assert p is not None and w is not None, (
        f"locks FIXC: flush at lidx {lidx} without armed stashes "
        f"(proj={'set' if p else 'None'} write={'set' if w else 'None'})")
    pstep, plidx, h, W, bias, out_kv, qs, pos = p
    wstep, wlidx, k_un, value, kres, vres, slot, sl, csc, page = w
    assert pstep == _qf2_step == wstep and plidx == lidx == wlidx, (
        f"locks FIXC: stale/mispaired stash (proj step {pstep} lidx {plidx},"
        f" write step {wstep} lidx {wlidx}, now step {_qf2_step} lidx {lidx})")
    _fixc_proj = _fixc_write = None
    gv = gemv_cuda.get_mod()
    k5 = k5_cuda.get_mod()
    assert (gv is not None and hasattr(gv, "kv_gemv_nf")
            and k5 is not None and hasattr(k5, "rope_and_cache_fixc")), (
        "LOCKS_QFIRST_FIXC: gemv/k5 extension missing the -DLOCKS_FIXC "
        "entries (kv_gemv_nf / rope_and_cache_fixc). Prebuild offline with "
        "LOCKS_QFIRST_FIXC=1 -- no silent fallback.")
    nt = int(k_un.shape[0])
    pos = pos[:nt]
    if _FIXC_POSCHK:
        # FREE METADATA GATE (design 2.5): on a decode row the engine's own
        # `positions[r]` must equal `seq_lens[r] - 1`.  Any disagreement is a
        # metadata bug, and it is the only way the unified KV write could rope
        # k at the wrong offset.  Debug-only (host sync).
        import torch as _t
        bad = int((pos != (sl[:nt].to(pos.dtype) - 1)).sum().item())
        assert bad == 0, (
            f"locks FIXC POSCHK: positions[r] != seq_lens[r]-1 on {bad}/{nt} "
            f"rows (lidx {lidx}) -- metadata bug, the KV write would rope k "
            f"at the wrong offset.  pos={pos[:8].tolist()} "
            f"sl={sl[:8].tolist()}")
        del _t
    gv.kv_gemv_nf(h, W, bias, out_kv, qs)
    k5.rope_and_cache_fixc(k_un, value, kres, vres, slot, pos, csc, page)


def _co_consume(lidx: int) -> bool:
    """Consume the co-kernel score mark: NO event wait -- the score ran
    on the MAIN stream inside the op, so stream order is the sync."""
    if not _QCO or _co_mark.get(lidx) != _qf2_step:
        return False
    _co_mark[lidx] = -1
    return True

_qf2_step = 0
_qf2_mark: dict = {}
_qf2_ev: dict = {}
_qf2_stream = None
_qf2_refs = None          # (st, block_table, seq_lens, scale) stashed per call


def qf2_derive_tick() -> None:
    global _qf2_step
    _qf2_step += 1


def qf2_prescore(q2d, lidx: int) -> None:
    """Custom-op body: fork the layer's score onto the side stream (eager /
    capture time; replay reuses the captured schedule)."""
    global _qf2_stream
    refs = _qf2_refs
    _dbg = os.environ.get("LOCKS_QF2_DEBUG") == "1"
    if refs is None or q2d.shape[0] != 1:
        if _dbg:
            print(f"[qf2] SKIP lidx={lidx} refs={'None' if refs is None else 'set'} "
                  f"nt={q2d.shape[0]}", flush=True)
        return                        # step 0 / prefill: adapter runs full
    # P2 capture fix (2026-07-20): the PIECEWISE capture's KV-write-only
    # group runs the MODEL forward with NO attention metadata -- the impl
    # delegates (no adapter, no consume, no join), so a fork here leaves
    # captured side-stream work unjoined and capture ABORTS
    # (StreamCaptureUnjoined; probe: FORK 120 / CONSUME 79 = one 40-layer
    # pass unconsumed).  Fork only when the attention impl will run.
    try:
        from vllm.forward_context import get_forward_context
        md = getattr(get_forward_context(), "attn_metadata", None)
    except Exception:  # noqa: BLE001
        md = None
    if md is None:
        if _dbg:
            print(f"[qf2] SKIP lidx={lidx} no-metadata pass", flush=True)
        return
    st, bt, sl, scale = refs
    if sl.shape[0] != 1:
        if _dbg:
            print(f"[qf2] SKIP lidx={lidx} sl={sl.shape[0]}", flush=True)
        return
    if _dbg:
        print(f"[qf2] FORK lidx={lidx} step={_qf2_step}", flush=True)
    import torch as _t
    if _qf2_stream is None:
        _qf2_stream = _t.cuda.Stream(device=q2d.device)
    ev_f = _qf2_ev.setdefault(("f", lidx), _t.cuda.Event())
    ev_d = _qf2_ev.setdefault(("d", lidx), _t.cuda.Event())
    cur = _t.cuda.current_stream(device=q2d.device)
    ev_f.record(cur)
    q3 = q2d.view(1, st.n_kv * st.G, st.d)
    from ..selection import rki4_score_cuda
    with _t.cuda.stream(_qf2_stream):
        _qf2_stream.wait_event(ev_f)
        rki4_score_cuda.rki4_score_only(st, int(lidx), q3, bt, sl, 1,
                                        float(scale))
        ev_d.record(_qf2_stream)
    _qf2_mark[lidx] = _qf2_step


def _qf2_consume(st, lidx: int) -> bool:
    if not _QF2 or _qf2_mark.get(lidx) != _qf2_step:
        if os.environ.get("LOCKS_QF2_DEBUG") == "1":
            print(f"[qf2] MISS lidx={lidx} mark={_qf2_mark.get(lidx)} "
                  f"step={_qf2_step}", flush=True)
        return False
    _qf2_mark[lidx] = -1
    import torch as _t
    _t.cuda.current_stream().wait_event(_qf2_ev[("d", lidx)])
    if os.environ.get("LOCKS_QF2_DEBUG") == "1":
        print(f"[qf2] CONSUME lidx={lidx}", flush=True)
    return True


def _rki4_select_adapter(q, kv_cache, block_table, seq_lens, st, n_req, scale,
                         cfg) -> None:
    """Bridge-signature Stage-A entry for the rki4 score mode.

    The S4 register-pipelined CUDA kernel scores every selectable page from
    the packed summary slabs into ``st.score_h`` (per-head LSE); the GQA
    combine reduces it into ``st.score`` exactly where the quad/clse path
    combines (nrm via the shared ``_nrm_launch`` passes, max via amax), and
    the score-agnostic radix top-b consumes ``st.score`` unchanged.  rki4 is
    CUDA-only BY DESIGN (there is no Triton reference): a kernel-build
    failure raises loudly instead of latching a fallback (the no-fallback
    rule).  ``derive=False`` semantics: the builder derived the per-step page
    params already."""
    from ..selection import rki4_score_cuda

    lidx = _layer_index(st, kv_cache)
    _pb = None
    if True:
        if _SEG_ON:
            _pa, _pb = _seg_pair(("prep", int(lidx)))
            _pa.record()
        if _QF2 or _QCO:
            # stash the per-step refs the NEXT step's prescore needs (the
            # buffers are persistent under the v1 runner; content refreshed
            # by the prologue before the prescore fires)
            global _qf2_refs
            _qf2_refs = (st, block_table, seq_lens, scale)
            # scale is a scalar (softmax 1/sqrt(d)) that never goes stale;
            # persist it on the state so the CO op can read it WITHOUT the
            # stash (which the batch-split drops), i.e. the op can pull fresh
            # st/bt/sl from the current metadata and still know the scale.
            st._scale = scale
        if _QCO and _co_consume(int(lidx)):
            # P2b: score_h computed by the co-kernel this layer, main
            # stream -- combine+select only.  FIX-C: the score ran in the
            # op via the DEPLOYED TU (bt class) instead; flush the deferred
            # [kv_gemv_nf -> rope_and_cache_fixc] pair directly behind the
            # select launch so they ride its entry-trigger cover window.
            rki4_score_cuda.rki4_select_only(st, n_req)
            if _FIXC:
                fixc_flush(int(lidx))
        elif _QF2 and _qf2_consume(st, int(lidx)):
            # QF2: score_h already computed on the side stream from the SAME
            # roped q bytes -- combine+select only.
            rki4_score_cuda.rki4_select_only(st, n_req)
        else:
            rki4_score_cuda.rki4_score_state(st, lidx, q, block_table,
                                             seq_lens, n_req, scale)
    global _PPSEL_ANNOUNCED
    if not _PPSEL_ANNOUNCED:
        _PPSEL_ANNOUNCED = True
        print(f"[locks] selection kernels: CUDA rki4 score (S4, zsplit="
              f"{st.zsplit or 'auto'}, combine={st.combine}) + "
              f"{'Triton' if os.environ.get('LOCKS_TOPB_TRITON', '0') == '1' else 'CUDA radix'}"
              " top-b", flush=True)
    if os.environ.get("LOCKS_TOPB_TRITON", "0") == "1":
        from ..selection.select import topb_select
        topb_select(st, n_req)
    else:
        from ..selection import select_cuda
        select_cuda.topb_select_cuda(st, n_req)
    if _SEG_ON and _pb is not None:
        _pb.record()


def _decode_backend(n_req: int) -> str:
    """Resolve the decode backend for this (arch, bs-bucket): table + cache.
    LOCKS_DECODE_TRITON=1 (the LOCKS_TOPB_TRITON analog) forces Triton for
    A/B races; there is no CUDA-forcing knob (CUDA is reached by the table)."""
    if os.environ.get("LOCKS_DECODE_TRITON", "0") == "1":
        return "triton"
    from .. import arch as _arch
    assert not _arch.is_blackwell_sm120(), (
        "locks fast decode: the sm_120 dispatch lane was removed 2026-07-21 "
        "(no-fallback rule: no silent wrong-arch route). See "
        "ours_doc/REFUTED_ARMS_INDEX.md; recover from tag "
        "pre-cleanup-2026-07-21.")
    bucket = "bs1" if n_req == 1 else ("bs2_7" if n_req < 8 else "bs8+")
    key = ("sm90", bucket)
    hit = _DEC_RESOLVED.get(key)
    if hit is None:
        assert key in _DECODE_TABLE, f"no decode dispatch row for {key}"
        hit = _DEC_RESOLVED[key] = _DECODE_TABLE[key]
        print(f"[locks] decode dispatch {key} -> {hit}", flush=True)
    return hit


# NOROPE (round A): the rotary bf16 cos|sin cache (P, rotary_dim) + geometry,
# published by attnwire.publish_rope at model load (MODEL-AGNOSTIC since
# 2026-07-22; it used to be a GLM-only walk in register._wire_norope); read by
# attn._norope_fast_write (the QFIRST_CO/NOROPE v3 wire; the v2
# locks_nr.rope_gate op was removed 2026-07-21 -- see register.py tombstone).
# The DEFAULTS are GLM-4's geometry (partial rotary 64, interleaved/non-neox)
# so an unpublished process compiles byte-identical kernels to the pre-2026-07-22
# tree; publish_rope overwrites them from the loaded model's own rotary module.
_NOROPE_CSC = None
_NOROPE_HEAD = 128
_NOROPE_ROT = 64                 # rotary_dim (<= head_size; partial rotary)
_NOROPE_NEOX = False             # False = interleaved (GLM), True = half-split


def rope_cflags() -> list:
    """-D flags that specialize the rope-absorbing kernels (score prelude,
    rope_and_cache[_fixc]) to THIS model's rotary geometry.

    Compile-time, not runtime: the GLM geometry (64, non-neox) emits NO flags
    at all, so the deployed TU stays byte-identical to the pre-model-agnostic
    tree and the GLM SASS/latency cannot move.  Any other geometry gets its own
    define set AND its own extension name (rope_tag) so a warm build cache can
    never hand one architecture the other's binary."""
    if int(_NOROPE_ROT) == 64 and not _NOROPE_NEOX:
        return []
    return [f"-DLOCKS_ROT_DIM={int(_NOROPE_ROT)}",
            f"-DLOCKS_ROT_NEOX={1 if _NOROPE_NEOX else 0}"]


def rope_tag() -> str:
    """Extension-name suffix pairing with rope_cflags ("" on GLM geometry)."""
    if int(_NOROPE_ROT) == 64 and not _NOROPE_NEOX:
        return ""
    return f"_rot{int(_NOROPE_ROT)}{'n' if _NOROPE_NEOX else 'i'}"


# ---- bias-free QKV support (Llama/Qwen attention_bias=False) --------------- #
# The hand GEMV kernels fuse the bias add unconditionally (`y[m] = v + b[m]`),
# a design the no-fallback rule keeps: the kernel asserts its precondition
# rather than branching.  An arch WITHOUT a QKV bias is served a real
# all-zeros bias row, which is bitwise identical (`v + 0.0f == v` for every
# finite/inf v, and bf16(-0.0 + 0.0) == +0.0 only where the GEMM already
# produced +-0).  PREFILL still passes bias=None into F.linear, so the stock
# prefill bytes are untouched.  Cached per (weight storage, rows) and warmed
# at wire time, so no allocation ever happens inside a captured region.
_ZERO_BIAS: dict = {}


def zero_bias(weight):
    import torch
    key = (weight.data_ptr(), int(weight.shape[0]), str(weight.device))
    b = _ZERO_BIAS.get(key)
    if b is None:
        b = _ZERO_BIAS[key] = torch.zeros(int(weight.shape[0]),
                                          dtype=weight.dtype,
                                          device=weight.device)
    return b


def decode_dispatch(q, kv_cache, block_table, seq_lens, st, out, vsource,
                    scale, cfg) -> None:
    """Stage-B decode: routed by the measured-regime table above (hand-CUDA
    Hopper kernel or the Triton kernel), with a one-shot latched fallback to
    Triton on CUDA build failure."""
    from ..attention.decode import sparse_paged_decode_batched
    global _DEC_CUDA
    if _SEG_ON:
        _da, _db = _seg_pair(("decode", int(_layer_index(st, kv_cache))))
        _da.record()
    use_cuda = (cfg.use_cuda and _DEC_CUDA is not False
                and _decode_backend(seq_lens.shape[0]) == "cuda")
    if use_cuda:
        try:
            from ..attention import decode_cuda
            decode_cuda.sparse_paged_decode_batched_cuda(
                q, kv_cache, block_table, seq_lens, st, out, vsource,
                scale=scale)
            if _DEC_CUDA is None:
                _DEC_CUDA = True
                print("[locks] decode kernel: CUDA (Hopper)", flush=True)
            if _SEG_ON:
                _db.record()
            return
        except Exception as e:  # noqa: BLE001
            _DEC_CUDA = False
            print(f"[locks] CUDA decode unavailable "
                  f"({type(e).__name__}: {e}) -> Triton", flush=True)
    sparse_paged_decode_batched(q, kv_cache, block_table, seq_lens, st, out,
                                vsource, scale=scale)
    if _SEG_ON:
        _db.record()


def resolve_selection():
    """(StateClass, derive_fn, select_fn, graph_safe).

    Returns the real graph-safe r8 path when it is wired and available;
    the eager validation bridge ONLY under the explicit LOCKS_FORCE_BRIDGE
    escape. ``select_fn`` always has the bridge-compatible call signature the
    impl uses.

    NO-FALLBACK RULE (2026-07-22): the configured score mode is now a
    PRECONDITION, not a preference.  The old code silently downgraded
    rki4 -> quad -> r8 -> bridge whenever a surface was missing, which
    serves and MEASURES a different selector under the same tag."""
    if not _force_bridge():
        from .. import selection as sel
        _check_selection_surface()          # one-shot (hot path: per layer)
        _derive = sel.derive_page_params
        if _QF2 or _QCO:
            def _derive(st_, sl_, nr_, __d=sel.derive_page_params):
                qf2_derive_tick()          # per-step nonce for the prescore
                return __d(st_, sl_, nr_)
        return (sel.Rki4State, _derive, _rki4_select_adapter, True)
    global _BRIDGE_SAID
    if not _BRIDGE_SAID:
        _BRIDGE_SAID = True
        print("[locks] LOCKS_FORCE_BRIDGE=1: eager validation BRIDGE "
              "selection (NOT the deployed graph-safe path)", flush=True)
    from ._selection_bridge import BridgeState, bridge_derive, bridge_select
    return BridgeState, bridge_derive, bridge_select, False


_SURFACE_CHECKED = False


def _check_selection_surface() -> None:
    """One-shot PRECONDITION check for the configured score mode.

    NO-FALLBACK RULE (2026-07-22): resolve_selection used to silently
    downgrade rki4 -> quad -> r8 -> eager bridge whenever a surface was
    missing (and swallowed the ImportError that made it missing), i.e. it
    served and MEASURED a different selector under the same tag.  The mode
    is now a precondition; the only legal way to the bridge is the explicit
    LOCKS_FORCE_BRIDGE escape.  Checked once because resolve_selection is a
    per-layer host call."""
    global _SURFACE_CHECKED
    if _SURFACE_CHECKED:
        return
    assert rki4_selection_available(), (
        "locks: the rki4 Stage-A surface is incomplete (locks.selection is "
        "missing Rki4State / derive_page_params / rki4_build_refresh) -- no "
        "silent bridge fallback. Fix the import, or set LOCKS_FORCE_BRIDGE=1 "
        "to ask for the eager validation bridge explicitly.")
    _SURFACE_CHECKED = True
