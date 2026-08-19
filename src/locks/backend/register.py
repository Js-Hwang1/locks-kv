"""register() -- the vLLM general-plugin entry point.

Builds the single :class:`LocksConfig` (from an optional ``LOCKS_CONFIG`` json/
yaml plus any explicit overrides) and patches ``CudaPlatform.get_attn_backend_cls``
so that wherever vLLM would pick the FlashAttention backend it gets
:class:`LocksAttentionBackend` instead -- a FlashAttentionBackend subclass, so
every non-decode path inherits stock FA.

``LOCKS_DISABLE=1`` makes register() inert (the FullKV / stock-FA reference line).
This is the ONE place besides ``LocksConfig.load`` that reads the environment.
"""
from __future__ import annotations

import os

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from ..config import LocksConfig
from . import _runtime
from .attn import LocksAttentionImpl
from .builder import LocksMetadataBuilder

_APPLIED = False


class LocksAttentionBackend(FlashAttentionBackend):
    """FlashAttentionBackend with the LOCKS impl + builder swapped in.

    FAST variant keeps the stock FA KV-cache shape (K+V resident). mem variants
    with ``cfg.mem_k_only`` (the default) get a K-ONLY engine block pool: the
    spec is patched to head_size_v=0 and the shape drops the V half, so
    num_gpu_blocks ~doubles at the same gpu_memory_utilization (the VRAM saving).

    NOTE: get_name() is deliberately inherited (-> "FLASH_ATTN"): vLLM 0.24
    looks the name up in AttentionBackendEnum, and only the registered FA name
    resolves. The impl/builder still come from THIS subclass."""

    @staticmethod
    def get_impl_cls() -> type[LocksAttentionImpl]:
        return LocksAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[LocksMetadataBuilder]:
        return LocksMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        cfg = _runtime.get_config()
        if (cfg is not None and cfg.is_mem and cfg.mem_k_only
                and _runtime.mem_stageb_wired()):
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            if getattr(cfg, "mem_summary_cache", False):
                # P5 spec v2: the engine page HOLDS THE RECORDS, not K.
                # The RE-KEY happens ONCE, in the spec patch
                # (_patch_kv_cache_spec_k_only: head_size -> rec_bytes /
                # (block*elem)); vLLM 0.24 then passes that spec.head_size
                # INTO this hook (gpu/attn_utils.py:303), so re-deriving
                # here double-applies (records-round smoke: alloc at hs=27
                # vs reshape at hs=11 -> boot crash).  Pass head_size
                # through; only the K/V-axis-1 SHAPE is this hook's job.
                return (num_blocks, 1, block_size, num_kv_heads, head_size)
            # K-only engine cache (spec patched head_size_v=0): keep the K/V axis
            # at size 1 so _split_kv/select(1, 0) still applies and the flat
            # buffer bytes (num_blocks * page_bytes) match the patched page size.
            return (num_blocks, 1, block_size, num_kv_heads, head_size)
        return FlashAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size,
            cache_dtype_str=cache_dtype_str)


def _summary_head_size(cfg, block_size: int, head_size: int = 128) -> int:
    """P5 spec v2: the re-keyed engine head_size so a page holds exactly
    ``n_kv x rec_bytes`` summary records (elem=2 bf16).  Single source of
    truth: ``r8i4_state.record_bytes_padded`` (padded up so the record tiles
    the page)."""
    elem = 2
    from ..selection.r8i4_state import record_bytes_padded
    rec_bytes = record_bytes_padded(d=head_size, page=block_size, elem=elem)
    if rec_bytes % (block_size * elem) != 0:
        raise ValueError(
            f"summary cache: rec_bytes {rec_bytes} not a multiple of "
            f"block*elem {block_size * elem} -- cannot tile the engine page")
    return rec_bytes // (block_size * elem)


_FA_PATH = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
_OUR_PATH = f"{__name__}.LocksAttentionBackend"
# MLA (DeepSeek V2/V3/V3.2): vLLM resolves an MLA backend (path contains
# ".mla.") for latent archs (num_key_value_heads==1 + kv_lora_rank). We swap it
# for the LOCKS MLA impl -- the SAME seam vLLM uses to separate MLA from GQA.
_MLA_PATH = __name__.rsplit(".", 1)[0] + ".mla_attn.LocksMLABackend"
_ANNOUNCED = False
_ANNOUNCED_MLA = False


def _neuter_dsa_indexer() -> None:
    """No-op DeepSeek-V3.2's DSA lightning indexer (LOCKS ignores it). Its
    ``forward`` normally ends in ``self.indexer_op(...)`` which writes the shared
    topk buffer (that LOCKS never reads) and runs ``fp8_fp4_mqa_logits`` -- the
    per-step allocation that OOMs at mns>1 and a large wasted-compute tax. We
    replace ``Indexer.forward`` with a no-op; the caller discards its return."""
    try:
        from vllm.model_executor.models import deepseek_v2 as _dsv2
    except Exception as exc:                                # pragma: no cover
        print(f"[locks] DSA-indexer neuter SKIPPED ({exc})", flush=True)
        return
    Indexer = getattr(_dsv2, "Indexer", None)
    if Indexer is None or getattr(Indexer.forward, "_locks_neutered", False):
        return

    def forward(self, *args, **kwargs):
        return None

    forward._locks_neutered = True
    Indexer.forward = forward
    print("[locks] DSA indexer NEUTERED (no-op forward; LOCKS does its own "
          "selection) -- frees the per-step fp8_fp4_mqa_logits alloc + compute",
          flush=True)


def register(overrides: dict | None = None) -> None:
    """vLLM general-plugin entry point (runs in every engine process)."""
    global _APPLIED
    if _APPLIED or os.environ.get("LOCKS_DISABLE"):
        return
    # R2 (reliability, 2026-07-21): with the dist-info materialized, vLLM
    # auto-loads this plugin in EVERY process whose VLLM_PLUGINS is unset;
    # the config default then fired mem patches on unsuspecting stock
    # engines (measured: the h200x8-04 smoke OOM, 2026-07-21). Without an
    # explicit LOCKS_CONFIG (or overrides), the plugin is INERT.
    if overrides is None and not os.environ.get("LOCKS_CONFIG"):
        print("[locks] register: no LOCKS_CONFIG and no overrides -> plugin "
              "INERT (stock vLLM). Set LOCKS_CONFIG to activate.", flush=True)
        return

    cfg = LocksConfig.load(overrides)
    _runtime.set_config(cfg)

    # DeepSeek-V3.2 (DSA): LOCKS does its OWN page selection and IGNORES the
    # trained lightning indexer. Left running, the indexer executes every decode
    # step (fp8_fp4_mqa_logits over the full context) -- pure wasted compute AND
    # a ~520MB/step allocation that OOMs at mns>1 (measured h200x8, 2026-07-26).
    # No-op its forward so LOCKS pays only for its own selection + attention.
    # Harmless on V2/V3 (no Indexer class instances).
    _neuter_dsa_indexer()

    from vllm.platforms import cuda
    # 0.24: get_attn_backend_cls is defined on CudaPlatformBase (CudaPlatform is
    # the Nvml/NonNvml alias). Patch the class in the MRO that defines it.
    target = next(k for k in cuda.CudaPlatform.__mro__
                  if "get_attn_backend_cls" in k.__dict__)
    orig = target.__dict__["get_attn_backend_cls"].__func__

    def get_attn_backend_cls(cls, *args, **kwargs):
        path = orig(cls, *args, **kwargs)
        if path == _FA_PATH:                          # GQA -- unchanged
            global _ANNOUNCED
            if not _ANNOUNCED:
                _ANNOUNCED = True
                print(f"[locks] ACTIVE variant={cfg.variant} "
                      f"budget={cfg.budget} coverage={cfg.coverage} "
                      f"sink={cfg.sink_pages} window={cfg.window_pages} "
                      f"-> {_OUR_PATH}", flush=True)
            return _OUR_PATH
        if isinstance(path, str) and ".mla." in path and path != _MLA_PATH:
            # MLA (DeepSeek). The mem tier + adaptive coverage are GQA-only; the
            # MLA impl requires a fixed page budget (asserted at first decode).
            global _ANNOUNCED_MLA
            if not _ANNOUNCED_MLA:
                _ANNOUNCED_MLA = True
                print(f"[locks] ACTIVE (MLA) variant={cfg.variant} "
                      f"budget={cfg.budget} budget_pages={cfg.budget_pages} "
                      f"sink={cfg.sink_pages} window={cfg.window_pages} "
                      f"{path} -> {_MLA_PATH}", flush=True)
            return _MLA_PATH
        return path

    target.get_attn_backend_cls = classmethod(get_attn_backend_cls)
    _APPLIED = True
    print("[locks] registered (patched CudaPlatform.get_attn_backend_cls)",
          flush=True)
    # QFIRST (2026-07-19, user-sanctioned LSE2-class lane): split the model's
    # fused QKV projection into Q-GEMM then KV-GEMM (row slices of the SAME
    # fused weight/bias) so q exists first -- milestone 1 of the
    # score-off-critical-path restructure.  NOT bitwise vs the fused GEMM
    # (measured max|dq|~1e-3: cuBLAS picks shape-dependent reduction
    # orders), therefore gated by the SELECTION-IDENTITY battery (PTAB
    # step-1, multi-seed) before any scheduling work lands on top.
    # Compile-cache segregation for the same reason as lookahead: the
    # patched forward is traced code the disk cache cannot see.
    if (os.environ.get("LOCKS_QFIRST", "0") == "1"
            or os.environ.get("LOCKS_QFIRST2", "0") == "1"
            or os.environ.get("LOCKS_QFIRST_LITE", "0") == "1"):
        _tag = ("locks_qf3"
                if os.environ.get("LOCKS_QFIRST_LITE", "0") == "1"
                else "locks_qf2"
                if os.environ.get("LOCKS_QFIRST2", "0") == "1"
                else "locks_qf1")
        _root = os.environ.get(
            "VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(_root, _tag)
        _wire_qfirst()
    # QFIRST_CO (P2b, the co-kernel lane; user "make it work" order
    # 2026-07-20): ONE opaque op = q-GEMV -> [kv-GEMV || r8i4 score] fat
    # kernel (no streams/events/joins -- main-stream order is the sync);
    # prefill ropes IN the op body (stock bytes, no view-mutating ops in
    # the trace -> no clones).  Consumers ride the NOROPE machinery
    # (_Q_ABSORBED): splits read st.q_rope, write paths rope k.  The
    # co-kernel's compilation-context score wobble is CHARACTERIZED
    # zero-selection-flip (0/512) under the exact-128 selector; the
    # flip-class ruling covers the residual.
    if (os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1"
            and os.environ.get("LOCKS_QFIRST_CO", "0") != "1"):
        # FIX-C rides the CO wire and inherits its ctx<=64K gate: with CO
        # off (>=128K cells) the flag is INERT by construction.  Loud note
        # so a cell log always shows the flag state.
        print("[locks] FIXC INERT (LOCKS_QFIRST_CO off -- ctx gate)",
              flush=True)
    if os.environ.get("LOCKS_QFIRST_CO", "0") == "1":
        if cfg.is_mem:
            from ..tier.decode_mem import k8d_active
            assert k8d_active(), \
                "LOCKS_QFIRST_CO on mem requires K8d (SCAT ropes k)"
        _root = os.environ.get(
            "VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(_root, "locks_co1")
        _wire_qfirst_co()
        # FIX-C (session-P lever 1, opt-in LOCKS_QFIRST_FIXC=1): union-
        # kernel retirement + select-cover schedule.  Same traced forward
        # (the locks_co.qkv op), different op-body branch: q_gemv +
        # deployed-TU score (bt class), kv+rope deferred behind the
        # select_v2 entry trigger.  Requires the full deployed ctx<=64K
        # set: select_v2 carries the -DLOCKS_FIXC entry trigger and the
        # hmax handshake pairs the score publish with it.
        if os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1":
            assert not cfg.is_mem, \
                "LOCKS_QFIRST_FIXC is fast-arm only (this round)"
            assert (os.environ.get("LOCKS_HMAX") == "1"
                    and os.environ.get("LOCKS_SEL_V2") == "1"), \
                "LOCKS_QFIRST_FIXC requires LOCKS_HMAX=1 + LOCKS_SEL_V2=1 " \
                "(select_v2 carries the entry trigger)"
            print("[locks] FIXC ACTIVE: union-kernel retirement -- "
                  "score_bt exposed, [kv_gemv_nf -> rope_and_cache_fixc] "
                  "under the select_v2 entry-trigger window", flush=True)
    # QFIRST_GV (P2 of the sanctioned Q-first lane; user flip-class ruling
    # 2026-07-20): q-GEMV -> rope-q (dummy-k split, bitwise) -> prescore
    # fork (side-stream score under the kv-GEMV+rope-k+preamble cover) ->
    # kv-GEMV -> rope-k -> attn.  Prefill = ONE stock fused linear inside
    # the q op (kv half stashed for the kv op) -- stock bytes.  The
    # selection flip class vs the fused reference is USER-SANCTIONED
    # (deterministic per-arm; boundary near-ties only; accuracy re-gate
    # on the cluster stands before paper claims).
    if os.environ.get("LOCKS_QFIRST_GV", "0") == "1":
        _root = os.environ.get(
            "VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(_root, "locks_gv2")
        _wire_qfirst_gv()
    # QKVGEMV (P1 of the sanctioned Q-first lane, 2026-07-20): swap ONLY
    # the QKV projection at nt==1 for the deterministic sliced GEMV pair
    # (gemv_cuda) -- rope/score/select/decode all stock.  Purpose:
    # ISOLATE the numerics question (does sliced-GEMV q flip selection?)
    # for the step-2 identity battery, before any scheduling work (P2).
    # Cache tag segregation as for qfirst (the patched forward is traced).
    if os.environ.get("LOCKS_QKVGEMV", "0") == "1":
        _root = os.environ.get(
            "VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(_root, "locks_gv1")
        _wire_qkvgemv()
    # NOROPE (round A, 2026-07-20): delete the fused rope kernel from the
    # nt==1 (decode) forward -- the scorer ropes q in its staging prelude
    # (publishing st.q_rope, the decode kernels' q source) and the write
    # paths rope k (K8d SCAT on mem, rope_and_cache on fast).  Bitwise
    # BY CONSTRUCTION (gate_norope_p1/p2: replicated inductor math, bf16
    # round-trips); binding gates = same-arm off/on PTAB + token equality.
    # Prefill (nt > 1) keeps stock rope: the skip is a shape branch, so
    # inductor compiles the two variants separately.  Compile-cache
    # segregation as for qfirst (the patched forward is traced code).
    if os.environ.get("LOCKS_NOROPE", "0") == "1":
        assert not any(os.environ.get(f, "0") == "1" for f in
                       ("LOCKS_QFIRST", "LOCKS_QFIRST2",
                        "LOCKS_QFIRST_LITE")), \
            "LOCKS_NOROPE is exclusive with the QFIRST lanes"
        if cfg.is_mem:
            from ..tier.decode_mem import k8d_active
            assert k8d_active(), \
                "LOCKS_NOROPE on mem requires K8d (the split ropes the " \
                "stashed k); enable LOCKS_MEM_K8D"
        # v3: STOCK trace (no forward patch, NO cache tag -- the compiled
        # artifacts are bit-identical to the baseline's); the rope kernels
        # are pruned from the CAPTURED decode graph instead (_wire_norope).
        _wire_norope()
    # K5_EARLY (round B): mem-only (the hook no-ops without a tier); the
    # _K5_STATS diagnostic mode keeps the prologue service (host-syncing).
    if (os.environ.get("LOCKS_K5_EARLY", "0") == "1"
            and os.environ.get("LOCKS_K5_STATS", "0") != "1"):
        _wire_k5_early()
    # mem variants + mem_k_only (default): patch the KV-cache spec to
    # head_size_v=0 so the engine block pool holds K only (V lives in the tier)
    # and num_gpu_blocks ~doubles at the same gpu_memory_utilization. This is the
    # realized-memory half of the mem play; it is served by the impl's K-only
    # write hook + tiered decode (V_SRC==1).
    if cfg.is_mem and cfg.mem_k_only and _runtime.mem_stageb_wired():
        _patch_kv_cache_spec_k_only()
    # mem variants: reserve the tier+summary+lease VRAM out of vLLM's
    # profiling-based KV budget (AC-M1; fixes the boot OOM at util 0.9 --
    # nsys_H200_AMM_ctx16384_p16_b1.log:146).
    if cfg.is_mem:
        from .memplan import patch_available_memory
        patch_available_memory()
        _patch_profiling_window()
    # P6: FULL-bucket coverage to bsMAX. vLLM's FULL (uniform-decode) graph
    # buckets are the capture sizes <= max_num_seqs (cudagraph_dispatcher
    # initialize_cudagraph_keys), so an off-grid max_num_seqs (e.g. 51) leaves
    # the largest FULL bucket BELOW it (48) and a bsMAX decode routes
    # PIECEWISE: per-partition replay, eager attention between pieces, and the
    # lookahead side schedule disengages by design (PLA_REPORT.md section 9:
    # measured +4.6% with zero overlap benefit). Rounding max_num_seqs UP to
    # the bucket grid makes every reachable decode batch pad into a FULL
    # graph -- the exact protocol PLA validated (nt=51 into the bucket-56
    # graph: tokens bitwise, pad cost ~1%: sync 370.45 vs 366.87 us/layer).
    # Admission-cap increase only; KV-capacity limits still bind. Env escape:
    # LOCKS_BSMAX_PAD=0 restores the unpadded config.
    if os.environ.get("LOCKS_BSMAX_PAD", "1") != "0" and not cfg.force_eager:
        _patch_bsmax_full_bucket()


def _patch_profiling_window() -> None:
    """Mark vLLM's cudagraph-memory-profiling window for the mem write hook.

    vllm 0.24 gpu_worker.profile_cudagraph_memory allocates a throwaway minimal
    KV cache (gpu_model_runner._init_minimal_kv_cache_for_profiling) and runs
    dummy forwards against it BEFORE the real block pool exists. The DRAM tier
    cannot be built then (builder._ensure_tier: num_gpu_blocks unset at that
    point), so the mem write hook would raise on a cache whose contents are
    discarded moments later (_cleanup_profiling_kv_cache). Bracketing the call
    lets the hook skip exactly that window and keep raising everywhere else --
    phase detection, not a fallback. No-op (with a printed note) if vLLM ever
    renames the method, in which case the old hard error returns."""
    # Defined on the model runner (gpu_model_runner.GPUModelRunner); the worker
    # only CALLS it (gpu_worker.py: self.model_runner.profile_cudagraph_memory()).
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _MR
    except Exception as exc:                       # pragma: no cover
        print(f"[locks] mem: profiling-window patch SKIPPED ({exc})", flush=True)
        return
    orig = getattr(_MR, "profile_cudagraph_memory", None)
    if orig is None or getattr(orig, "_locks_wrapped", False):
        if orig is None:
            print("[locks] mem: profiling-window patch SKIPPED "
                  "(GPUModelRunner.profile_cudagraph_memory absent)", flush=True)
        return

    def profile_cudagraph_memory(self, *args, **kwargs):
        _runtime.set_profiling(True)
        try:
            return orig(self, *args, **kwargs)
        finally:
            _runtime.set_profiling(False)

    profile_cudagraph_memory._locks_wrapped = True
    _MR.profile_cudagraph_memory = profile_cudagraph_memory
    print("[locks] mem: patched GPUModelRunner.profile_cudagraph_memory "
          "(tier-less dummy-run window)", flush=True)

    # The window above is NOT the only tier-less dummy-run phase: vLLM 0.24's
    # kernel_warmup (flashinfer_autotune + the FlashInfer mixed-batch warmup)
    # calls _dummy_run(is_profile=True) AFTER profile_cudagraph_memory returns
    # and BEFORE the first real forward builds the tier (measured: the round-4
    # crash at attn.py:145 from kernel_warmup.py:137). Key the flag on vLLM's
    # own is_profile argument instead of chasing call sites: is_profile=True
    # forces eager (gpu_model_runner.py:5782 force_eager=is_profile), so an
    # is_profile run can NEVER be a graph-capture run and skipping the tier
    # write there cannot poison a captured graph; the capture-path dummy runs
    # (gpu_model_runner.py:6675/6686) pass no is_profile and still execute the
    # real path. Every observed caller passes is_profile as a KEYWORD.
    orig_dummy = getattr(_MR, "_dummy_run", None)
    if orig_dummy is not None and not getattr(orig_dummy, "_locks_wrapped", False):
        def _dummy_run(self, *args, **kwargs):
            if not kwargs.get("is_profile"):
                return orig_dummy(self, *args, **kwargs)
            _runtime.set_profiling(True)
            try:
                return orig_dummy(self, *args, **kwargs)
            finally:
                _runtime.set_profiling(False)
        _dummy_run._locks_wrapped = True
        _MR._dummy_run = _dummy_run
        print("[locks] mem: patched GPUModelRunner._dummy_run "
              "(is_profile => tier-less window)", flush=True)
    elif orig_dummy is None:
        print("[locks] mem: _dummy_run patch SKIPPED (method absent)", flush=True)


def _cudagraph_bucket_up(n: int) -> int:
    """Smallest vLLM cudagraph capture size >= n (the documented grid
    [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16), vllm 0.24
    config/compilation.py); n > 512 is beyond the capture cap and returns n
    unchanged."""
    if n <= 1:
        return 1
    if n <= 2:
        return 2
    if n <= 4:
        return 4
    if n <= 248:
        return (n + 7) // 8 * 8
    if n <= 512:
        return (n + 15) // 16 * 16
    return n


def _patch_bsmax_full_bucket() -> None:
    """Round ``SchedulerConfig.max_num_seqs`` up to the cudagraph bucket grid
    so the largest FULL uniform-decode bucket covers bsMAX (see register()).
    Patched at SchedulerConfig.__post_init__ so every downstream consumer
    (capture-size generation, dispatcher key filter, dummy runs, persistent
    buffer sizing) sees the padded value consistently."""
    from vllm.config.scheduler import SchedulerConfig

    orig = SchedulerConfig.__post_init__

    def __post_init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        mns = int(self.max_num_seqs)
        pad = _cudagraph_bucket_up(mns)
        if pad != mns and pad <= int(self.max_num_batched_tokens):
            self.max_num_seqs = pad
            print(f"[locks] FULL-bucket coverage to bsMAX: max_num_seqs "
                  f"{mns} -> {pad} (cudagraph bucket grid; "
                  f"LOCKS_BSMAX_PAD=0 restores)", flush=True)

    SchedulerConfig.__post_init__ = __post_init__
    print("[locks] bsMAX FULL-bucket padding wired "
          "(SchedulerConfig.__post_init__)", flush=True)


_QF_OP_READY = False


def _ensure_qf_op() -> None:
    """torch.ops.locks_qf.prescore(q, lidx): dynamo-opaque side-stream score
    fork (the locks_la.capture registration pattern)."""
    global _QF_OP_READY
    if _QF_OP_READY:
        return
    import torch
    from torch.library import Library
    from vllm.utils.torch_utils import direct_register_custom_op

    lib = Library("locks_qf", "FRAGMENT")

    # NOTE: this module has `from __future__ import annotations`, which
    # stringifies signature annotations; torch's infer_schema eval's them
    # without `torch` in scope.  Real-type __annotations__ sidestep it.
    def prescore(q, lidx):
        _runtime.qf2_prescore(q, lidx)

    def prescore_fake(q, lidx):
        return None

    _ann = {"q": torch.Tensor, "lidx": int, "return": None}
    prescore.__annotations__ = dict(_ann)
    prescore_fake.__annotations__ = dict(_ann)

    # mutates_args non-empty is LOAD-BEARING: a no-output, no-mutation
    # op is DCE'd by inductor (measured: zero op-body executions, all
    # 800 scores serial on stream 7).  Declaring q mutated keeps the op
    # alive AND pins q-consumer ordering across it (semantically safe;
    # the lookahead op's arena declaration is the precedent).
    direct_register_custom_op(op_name="prescore", op_func=prescore,
                              mutates_args=["q"], fake_impl=prescore_fake,
                              target_lib=lib)
    _ensure_qf_op._lib = lib
    _QF_OP_READY = True


def _wire_qfirst() -> None:
    """QFIRST milestone 1: patch the attention forward so Q is projected
    before KV (two F.linear calls over row slices of the fused weight;
    identical math per output element, reduction ORDER differs by cuBLAS
    shape choice -- the reason this lane is selection-identity-gated).
    Rope and everything downstream stay verbatim.  Unquantized weights only
    (sliced views of packed quant layouts would be wrong); refuses loudly
    otherwise.

    MODEL-AGNOSTIC since 2026-07-22: the target used to be the hardcoded
    ``vllm.model_executor.models.chatglm.GLMAttention`` class, which made the
    lane SILENTLY INERT on every non-GLM model.  ``attnwire`` discovers the
    fused-QKV attention modules of whatever arch is loaded and patches those
    instances (via a per-model subclass); an arch without the surface raises."""
    from . import attnwire

    lite = os.environ.get("LOCKS_QFIRST_LITE", "0") == "1"
    qf2 = os.environ.get("LOCKS_QFIRST2", "0") == "1"
    if qf2 or lite:
        _ensure_qf_op()
    attnwire.wire("QFIRST_LITE" if lite else "QFIRST2" if qf2 else "QFIRST")


# torch.ops.locks_nr.rope_gate (_ensure_nr_op, NOROPE-v2's dynamo-opaque
# prefill rope) removed 2026-07-21: zero callers since the v3 wire moved the
# rope into k5_cuda.rope_and_cache (attn.py _norope_fast_write).  The v2
# mechanism + the P3-v1 double-rope postmortem stay recorded in
# ours_doc/REFUTED_ARMS_INDEX.md; recover from tag pre-cleanup-2026-07-21.


def _wire_k5_early() -> None:
    """LOCKS_K5_EARLY (round B): issue the mem arm's K5 service at the
    post-graph host point.  The steady-window exposure map (nsys_gt_tg)
    shows transport (33.8us) + assign (31.6us) executing AFTER the sampler
    sync at drooped clocks, outside the LM-head span -- because the HOST
    only reaches the prologue there.  service() is host-read-free on the
    steady path and every ordering is carried by events (gate/zev), so the
    issue point can move to a compute_logits PRE-hook: after the decode
    graph is enqueued, before the LM head -- the kernels then run at graph
    end, clock-warm, under the LM head, and the next graph's zev wait
    finds assign complete.  _K5_STATS keeps the prologue path (it
    host-syncs by design).  Kill switch: LOCKS_K5_EARLY=0."""
    from . import attnwire

    def _cb(model, vllm_config):
        orig_cl = model.compute_logits

        def compute_logits(*a, **kw):
            tier = _runtime.get_tier()
            if tier is not None and not _runtime.in_profiling():
                tier.early_service()
            return orig_cl(*a, **kw)

        model.compute_logits = compute_logits
        print("[locks] K5_EARLY ACTIVE: service issued at the post-graph "
              "host point (compute_logits pre-hook)", flush=True)

    attnwire.on_model_loaded(_cb)


def _co_preflight(sites, arch: str) -> None:
    """CO/FIX-C preconditions that depend on the loaded model, checked ONCE at
    wire time so a mismatch is a loud boot failure instead of a silently
    mis-scored decode.

    The union co-kernel is RETIRED (2026-07-22, BATCHED_CO_DESIGN 2.7): it was
    never launched in the deployed set (FIX-C routes the score through the
    G-dispatching deployed TU), and generalizing it to nt > 1 was refuted
    pre-build on registers.  ``LOCKS_QFIRST_CO=1`` without
    ``LOCKS_QFIRST_FIXC=1`` therefore has no kernel behind it any more.  It
    must FAIL, not degrade: ``_runtime._FIXC`` is False in that combination, so
    the op would take its stock branch on every step while the cell still
    printed ``QFIRST_CO ACTIVE`` -- a measurement of the wrong configuration,
    the exact silent-inert class this lane's model-agnostic round removed."""
    if os.environ.get("LOCKS_QFIRST_FIXC", "0") == "1":
        return
    raise RuntimeError(
        f"LOCKS_QFIRST_CO=1 without LOCKS_QFIRST_FIXC=1 (arch {arch!r}): the "
        "union co-kernel this combination used to launch was RETIRED on "
        "2026-07-22 (ours_doc/BATCHED_CO_DESIGN_2026-07-22.md 2.7 -- not in "
        "the deployed set, and reg-bound at 4 CTA/SM vs the score's 8). Set "
        "LOCKS_QFIRST_FIXC=1 (the deployed ctx<=64K set), or unset "
        "LOCKS_QFIRST_CO. No silently-inert lane.")


def _wire_qfirst_co() -> None:
    """P2b forward: the locks_co.qkv op replaces [QKV GEMM + rotary]
    wholesale; everything downstream is stock.  lidx stamping + rotary
    cache/geometry publish happen in ``attnwire`` at load_model.

    MODEL-AGNOSTIC since 2026-07-22 (was a hardcoded GLMAttention class
    patch + a ``model.transformer.encoder.layers`` walk, i.e. silently inert
    on every other architecture)."""
    from . import attnwire
    from .coqkv_cuda import ensure_co_op
    ensure_co_op()
    attnwire.wire("QFIRST_CO", extra_check=_co_preflight)


def _wire_qfirst_gv() -> None:
    """P2: the full Q-first forward on the deterministic GEMV pair.  Score
    forks to the side stream the moment roped q exists; the kv-GEMV +
    rope-k + attention preamble are its cover; the select adapter consumes
    the prescored score_h via the qf2 mark (machinery proven; _QF2 is
    true under LOCKS_QFIRST_GV)."""
    from . import attnwire
    from .gemv_cuda import ensure_qkv_op
    ensure_qkv_op()
    _ensure_qf_op()
    attnwire.wire("QFIRST_GV")


def _wire_qkvgemv() -> None:
    """P1 numerics wire: the attention forward with the QKV projection
    routed through the opaque pure op ``locks_gv.qkv`` (deterministic
    sliced GEMV at nt==1, stock F.linear inside the op at nt>1 -- prefill
    bytes stock).  Everything after the projection is VERBATIM stock."""
    from . import attnwire
    from .gemv_cuda import ensure_qkv_op
    ensure_qkv_op()
    attnwire.wire("QKVGEMV")


def _wire_norope() -> None:
    """NOROPE (round A, v3 -- CAPTURE SURGERY): the compiled trace stays
    100% STOCK (v1 postmortem: a shape branch cannot remove the decode
    rope, vLLM traces one dynamic-nt artifact; v2 postmortem: a mutating
    custom op on the qkv views costs functionalization CLONES per layer,
    +0.08..+0.11 e2e, refuted).  Instead the 39 per-layer rope kernels
    (``triton_poi_fused_2``) are DELETED from the captured decode graph by
    a ``capture_end`` post-hook (k5_cuda.graph_prune_kernels: driver-API
    node walk, dependency rewiring, name-prefix match).  The pruned
    kernels' outputs (the roped q/k buffers) become unwritten garbage that
    nothing reads under LOCKS_NOROPE: the impls reconstruct the UNROPED
    q/k as views of the never-roped QKV buffer (attn._norope_qk_views) and
    the P1/P2 kernels rope in-kernel, bitwise (gate_norope_p1/p2).
    Prefill runs the stock artifact uncaptured -- rope intact, bytes
    stock.  Layer-0's rope is fused into the step-head kernel and stays
    (harmless: the QKV buffer is never roped in place, so the uniform
    in-kernel rope is correct for all layers; 39/40 of the launch prize).
    The rotary cache/geometry are published to ``_runtime`` at load."""
    import torch

    # raw_cuda_graph() is only legal on graphs built with keep_graph=True
    # (torch keeps the cudaGraph_t after instantiation); vLLM constructs
    # its capture objects without it -> force the kwarg for every graph
    # while the flag is on (a few KB of driver bookkeeping per graph).
    orig_init = torch.cuda.CUDAGraph.__init__

    def graph_init(self, *a, **kw):
        kw["keep_graph"] = True
        return orig_init(self, *a, **kw)

    torch.cuda.CUDAGraph.__init__ = graph_init

    orig_ce = torch.cuda.CUDAGraph.capture_end

    def capture_end(self, *a, **kw):
        r = orig_ce(self, *a, **kw)
        try:
            from ..tier import k5_cuda
            mod = k5_cuda.get_mod()
            assert mod is not None, "NOROPE v3 requires the k5_cuda build"
            npr = mod.graph_prune_kernels(self.raw_cuda_graph(),
                                          "triton_poi_fused_2", 30)
            if npr:
                print(f"[locks] NOROPE v3: pruned {npr} rope nodes from "
                      "the captured decode graph", flush=True)
        except Exception:
            import traceback
            print("[locks] NOROPE v3 prune FAILED (graph left stock -> "
                  "double-rope hazard):", flush=True)
            traceback.print_exc()
            raise
        return r

    torch.cuda.CUDAGraph.capture_end = capture_end

    # MODEL-AGNOSTIC since 2026-07-22: the cos/sin publish was a GLM-only
    # ``model.transformer.encoder.layers`` walk with a hardcoded
    # ``csc.shape[1] == 64`` assert.  attnwire discovers the arch's own rotary
    # and validates the geometry it publishes.
    from . import attnwire
    attnwire.wire_rope_only("NOROPE")


def _patch_kv_cache_spec_k_only() -> None:
    """mem + mem_k_only: patch ``Attention.get_kv_cache_spec`` to
    ``head_size_v=0`` so the engine block pool holds K only and num_gpu_blocks
    ~doubles at the same gpu_memory_utilization. vLLM 0.24's FullAttentionSpec is
    natively asymmetric (page_bytes = block * n_kv * (head_size + head_size_v) *
    dtype), so head_size_v=0 halves the page. Rejects asymmetric / non-Full
    layers loudly (no silent wrong V). Ports the prototype
    ``dynkv_backend_v1._patch_kv_cache_spec``."""
    import dataclasses

    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # 0.24: get_kv_cache_spec lives on the Attention layer module.
    try:
        from vllm.attention.layer import Attention
    except Exception:  # noqa: BLE001
        from vllm.model_executor.layers.attention.attention import Attention

    orig = Attention.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig(self, vllm_config)
        if spec is None:
            return spec
        if type(spec) is not FullAttentionSpec:
            raise RuntimeError(
                "locks mem_k_only requires plain FullAttentionSpec layers; got "
                f"{type(spec).__name__} for {getattr(self, 'layer_name', '?')!r}")
        if spec.head_size_v != spec.head_size:
            raise RuntimeError(
                "locks mem_k_only: layer has asymmetric head_size_v "
                f"({spec.head_size_v} != {spec.head_size}) -- unsupported")
        cfg = _runtime.get_config()
        if cfg is not None and getattr(cfg, "mem_summary_cache", False):
            # P5 spec v2: engine page = summary records (head_size = hs,
            # head_size_v=0).  K leaves VRAM entirely; the ~10.7x-smaller page
            # is the contract-#2 capacity win.
            hs = _summary_head_size(cfg, spec.block_size, spec.head_size)
            return dataclasses.replace(spec, head_size=hs, head_size_v=0)
        return dataclasses.replace(spec, head_size_v=0)

    Attention.get_kv_cache_spec = get_kv_cache_spec
    print("[locks] mem_k_only: patched Attention.get_kv_cache_spec "
          "(head_size_v=0; summary_cache -> head_size=records)", flush=True)
