"""memplan -- boot-time VRAM plan for the mem variants (AC-M1).

THE BUG THIS FIXES (measured, benchmarks/efficiency_v2/logs/
nsys_H200_AMM_ctx16384_p16_b1.log:137-166): vLLM sizes ``num_gpu_blocks`` from
the free memory left at ``gpu_memory_utilization`` DURING ITS PROFILING RUN,
which happens BEFORE any LOCKS allocation exists (Tier + QuadState are lazily
built at first build()).  With the mem-kv K-only spec the page halves, so NB
DOUBLES (111,445 blocks = 108.8 GiB at 16K/util 0.9) and the engine cache eats
the whole budget; the tier (hot+staging ~8 GiB) then squeezes in above the
utilization target and the NB-keyed QuadState (~9 GiB at that NB) OOMs.  The
downstream casualty is CUDA-graph capture: the OOM leaves ``st=None``, the mem
forward falls to the dense prefill path under FULL capture, and its ``.item()``
invalidates the graph (cudaErrorStreamCaptureInvalidated at
torch/cuda/graphs.py capture_end).

THE FIX (LOCKS_KERNEL_REDESIGN.md v2, 3.4 "assert_capacity is restated" +
3.6.4 + AC-M1): make vLLM's profiling-based sizing SEE the LOCKS plan.  We
patch ``Worker.determine_available_memory`` to return

    avail' = (avail - fixed_bytes) / (1 + f)

where ``fixed_bytes`` is every NB-independent LOCKS allocation (hot buffer,
staging pools, one prefill lease of max_model_len tokens, Stage-B partials,
slack) and ``f`` is the NB-PROPORTIONAL overhead per engine-cache byte
(summary records + residency/staging maps per block).  vLLM then computes
NB' = avail' / page_bytes, and by construction

    engine(NB') + f * engine(NB') + fixed  <=  avail,

so the lazily-allocated Tier + QuadState always fit inside the utilization
budget.  Deterministic, no placeholder allocations, no chicken-and-egg (the
per-block overhead is a closed-form function of the model geometry + config).

Host-DRAM bound (3.6.4): the NB-keyed pinned pools are capped loudly by
``max_pool_gb`` (pool.py:126-129); ``plan()`` reports the projected host bytes
so the raise never surprises.
"""
from __future__ import annotations

import dataclasses
import math

from ..config import LocksConfig

# Mirror of builder._SPLIT (Stage-B split count) -- import cycle keeps it here.
_SPLIT = 128
_SLACK_BYTES = 512 << 20        # allocator fragmentation + FA workspace + jit


@dataclasses.dataclass
class MemPlan:
    fixed_bytes: int            # NB-independent LOCKS VRAM
    overhead_per_block: int     # NB-proportional LOCKS VRAM (bytes / engine blk)
    engine_block_bytes: int     # engine cache bytes / block (all layers)
    host_bytes_per_block: int   # pinned host pool bytes / block (all layers)
    detail: dict

    @property
    def f(self) -> float:
        return self.overhead_per_block / self.engine_block_bytes

    def shrink(self, avail: int) -> int:
        """avail -> avail' such that engine(NB') + overhead(NB') + fixed <= avail."""
        return max(0, int((avail - self.fixed_bytes) / (1.0 + self.f)))


def build_plan(vllm_config, cfg: LocksConfig) -> MemPlan:
    mc = vllm_config.model_config
    pc = vllm_config.parallel_config
    sched = vllm_config.scheduler_config
    comp = vllm_config.compilation_config
    cache = vllm_config.cache_config

    L = mc.get_num_layers(pc)
    n_kv = mc.get_num_kv_heads(pc)
    d = mc.get_head_size()
    page = cache.block_size
    elem = 2                                     # bf16/fp16 KV
    max_model_len = mc.max_model_len
    max_reqs = sched.max_num_seqs
    max_pages = (max_model_len + page - 1) // page
    # mirrors builder._ensure_tier's max_tokens
    max_tokens = max(int(getattr(sched, "max_num_batched_tokens", 0) or 0),
                     int(getattr(comp, "max_cudagraph_capture_size", 0) or 0),
                     max_reqs) + 8
    chunk_pages = (max_tokens + page - 1) // page
    kv_mult = 2 if cfg.offload_k else 1          # K+V pools vs V-only

    # ---- summary record size (single source of truth) --------------------- #
    # rki4 packed slab per (page, kv-head), incl. tag; the engine page under
    # the summary cache is the PADDED size (register._summary_head_size), the
    # NB-keyed Rki4State slabs the exact one.
    from ..selection.rki4_state import record_bytes, record_bytes_padded
    rec_bytes = record_bytes(d=d, page=page)
    rec_bytes_engine = record_bytes_padded(d=d, page=page, elem=elem)

    # ---- engine cache bytes per block (what vLLM divides avail by) -------- #
    summary_cache = bool(getattr(cfg, "mem_summary_cache", False))
    if summary_cache:
        # P5 spec v2: the engine page HOLDS THE RECORDS (head_size re-keyed so
        # page_bytes == n_kv x rec_bytes).  The K-only K page (~10.7x bigger) is
        # gone AND the separate summary arena folds INTO the engine page.
        per_layer_block = n_kv * rec_bytes_engine
    else:
        # mem_k_only halves the page (head_size_v=0).
        per_layer_block = page * n_kv * d * elem * (1 if cfg.mem_k_only else 2)
    engine_block = per_layer_block * L

    # ---- NB-proportional LOCKS overhead per block -------------------------- #
    # summary records: a SEPARATE (L,NB,384) arena.  Spec v2's design FOLDS this
    # into the engine page (aliased -> 0 extra), but that aliasing needs the L
    # engine layer tensors to share ONE storage; on this vLLM stack they are
    # SEPARATE per-layer allocations (measured: untyped_storage = 1 layer), so
    # the arena is kept and counted CONSERVATIVELY (K still leaves VRAM; only
    # the ~9 GiB fold-in is deferred).  If a future stack coalesces the engine
    # tensors, QuadState aliases and this over-reserves harmlessly.
    summary_per_block = L * n_kv * rec_bytes
    # residency maps: page2slot i32 + pool_valid i8 per (l, kv, blk)
    resid_per_block = L * n_kv * (4 + 1)
    # staging maps: vbo i32 + v_flushed i32 per blk
    staging_per_block = 8
    overhead_per_block = summary_per_block + resid_per_block + staging_per_block

    # ---- fixed (NB-independent) LOCKS VRAM --------------------------------- #
    from ..tier.staging import Staging
    from ..tier.tier import Tier
    hot_slots = cfg.hot_slots
    if hot_slots is None or hot_slots <= 0:
        eff_budget = cfg.budget if cfg.budget is not None else 0.15
        # MUST mirror Tier.from_config exactly (same function, same args) or
        # the plan diverges from the Residency alloc.  At n_seqs > 1 this is
        # the R2-exact per-sequence attended-set sizing (see the batch note in
        # recommend_hot_slots; kill LOCKS_MEM_BATCH_HOT=0) -- the fix for the
        # 2026-07-20 batch boot failures (hot=81.29 GiB at 32K/bs32 -> NB~0 ->
        # "No available memory for the cache blocks").
        hot_slots = Tier.recommend_hot_slots(
            eff_budget, max_pages, n_seqs=max_reqs,
            target_ratio=float(getattr(cfg, "hot_ratio", 6.0)),
            sink=cfg.sink_pages, win=cfg.window_pages,
            budget_pages=getattr(cfg, "budget_pages", None))
    hot = L * n_kv * hot_slots * page * d * elem * kv_mult
    nv = cfg.v_blocks if (cfg.v_blocks or 0) > 0 else \
        Staging.default_v_blocks(chunk_pages, max_reqs)
    staging = L * nv * page * n_kv * d * elem * kv_mult
    # ONE prefill lease of max_model_len tokens, K+V (leased prefill, 3.4).
    # O(one request) BY READ CONTRACT: chunk-k prefill attention reads the
    # FULL prefix K_lease[:sl] via FA varlen (lease.py), and at most one lease
    # exists (max_num_partial_prefills == 1) -- so this term does NOT scale
    # with bs, and it CANNOT shrink to max_num_batched_tokens without breaking
    # leased-prefill correctness (the banked 2026-07-20 "lease -> mnbt" note
    # predates the fence resolution; the bs-scaling term of the batch boot
    # failures was `hot`, fixed above).  Mirrors builder._ensure_lease.
    lease = 2 * L * max_pages * page * n_kv * d * elem
    # Stage-B split partials: m/l (f32) + acc (f32, D) per (R, n_kv*SPLIT*G)
    G = (mc.get_num_attention_heads(pc) // max(n_kv, 1)) or 1
    parts = max_reqs * n_kv * _SPLIT * G * (4 + 4 + d * 4)
    # request-keyed scratch (score/page_table/score_h/miss/victim/fetch lists)
    scratch = max_reqs * n_kv * max_pages * (4 + 4 + 4 * G + 4 * 3 + 12) + (1 << 20)
    fixed = hot + staging + lease + parts + scratch + _SLACK_BYTES

    # host pinned pool holds FULL K (and V for mem-kv) per token, NOT records:
    # bytes/block = full KV page (page*n_kv*d*elem) x L x (K+V mult).  Under
    # spec v2 per_layer_block is record-sized, so price the host pool directly.
    full_kv_layer_block = page * n_kv * d * elem
    host_per_block = full_kv_layer_block * L * kv_mult
    return MemPlan(
        fixed_bytes=int(fixed), overhead_per_block=int(overhead_per_block),
        engine_block_bytes=int(engine_block),
        host_bytes_per_block=int(host_per_block),
        detail=dict(hot=hot, staging=staging, lease=lease, parts=parts,
                    scratch=scratch, slack=_SLACK_BYTES,
                    rec_bytes=rec_bytes, hot_slots=hot_slots, nv=nv,
                    chunk_pages=chunk_pages, L=L, n_kv=n_kv, d=d, page=page,
                    max_pages=max_pages, max_reqs=max_reqs))


def _mem_available_bytes() -> int:
    """Host RAM currently available for a (non-swappable) pinned allocation."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


# leave this much host RAM free of the pinned pools (other procs + page cache)
_DRAM_SLACK_BYTES = 32 << 30


def host_capped_num_blocks(plan: "MemPlan", cfg: LocksConfig,
                           nb_vram: int) -> tuple[int, dict]:
    """Cap NB so the pinned host pools fit BOTH the per-pool ``max_pool_gb`` cap
    (pool.py checks each MappedHostVPool separately) AND available host DRAM.

    Under the summary cache the engine page is record-sized, so vLLM's VRAM
    division yields a huge NB (6-12x) and the FULL-K+V host pool would be 500+
    GiB -> pool.py RAISE at boot.  This bounds NB to the largest batch host DRAM
    can back (num_gpu_blocks_override), the honest capacity ceiling.

    DEMAND CAP (LOCKS_MEM_RKI4.md B1): every NB-keyed structure (pinned pools,
    summary slabs, residency maps) scales with NB, and under the mem scope
    (prefix caching refused at builder init) at most ``max_reqs x max_pages``
    blocks can ever be mapped at once -- any NB above that is pure host/VRAM
    waste. Cap to demand (+1% watermark + 64 margin)."""
    gib = 1 << 30
    host_pb = max(1, plan.host_bytes_per_block)          # both pools, all layers
    kv_mult = 2 if cfg.offload_k else 1
    per_pool_pb = max(1, host_pb // kv_mult)              # ONE MappedHostVPool
    nb_pool_cap = int((cfg.max_pool_gb * gib) // per_pool_pb)
    dram_budget = max(0, _mem_available_bytes() - _DRAM_SLACK_BYTES)
    nb_dram_cap = int(dram_budget // host_pb) if dram_budget else nb_vram
    d = plan.detail
    nb_demand = int(d["max_reqs"]) * int(d["max_pages"])
    nb_demand += nb_demand // 100 + 64
    nb_cap = min(nb_pool_cap, nb_dram_cap, nb_demand)
    detail = dict(nb_pool_cap=nb_pool_cap, nb_dram_cap=nb_dram_cap,
                  nb_demand=nb_demand,
                  dram_avail_gib=_mem_available_bytes() / gib,
                  host_pb=host_pb, capped=nb_cap < nb_vram)
    return max(1, min(nb_vram, nb_cap)), detail


_PATCHED = False


def patch_available_memory() -> None:
    """Patch ``Worker.determine_available_memory`` (vLLM 0.24, v1 GPU worker)
    to subtract the LOCKS mem-variant VRAM plan from the KV budget."""
    global _PATCHED
    if _PATCHED:
        return
    from vllm.v1.worker.gpu_worker import Worker
    from . import _runtime

    orig = Worker.determine_available_memory

    def determine_available_memory(self):
        avail = orig(self)
        cfg = _runtime.get_config()
        if cfg is None or not cfg.is_mem:
            return avail
        plan = build_plan(self.vllm_config, cfg)
        shrunk = plan.shrink(avail)
        nb = shrunk // max(plan.engine_block_bytes, 1)
        gib = 1 << 30
        d = plan.detail
        print(f"[locks memplan] avail={avail / gib:.2f}GiB -> {shrunk / gib:.2f}GiB "
              f"(fixed={plan.fixed_bytes / gib:.2f}GiB: hot={d['hot'] / gib:.2f} "
              f"staging={d['staging'] / gib:.2f} lease={d['lease'] / gib:.2f} "
              f"parts+scratch+slack={(d['parts'] + d['scratch'] + d['slack']) / gib:.2f}; "
              f"per-block +{plan.f * 100:.1f}% [rec={d['rec_bytes']}B x {d['n_kv']}kv "
              f"x {d['L']}L]); projected NB~{nb} "
              f"host~{nb * plan.host_bytes_per_block / gib:.1f}GiB", flush=True)
        # P7: cap NB so the pinned host pools fit max_pool_gb AND host DRAM (the
        # summary cache balloons NB -> 500+ GiB host pool -> pool.py boot RAISE).
        nb_cap, cd = host_capped_num_blocks(plan, cfg, int(nb))
        if cd["capped"]:
            self.vllm_config.cache_config.num_gpu_blocks_override = nb_cap
            print(f"[locks memplan] host cap: NB {nb} -> {nb_cap} "
                  f"(num_gpu_blocks_override; host pool "
                  f"{nb_cap * plan.host_bytes_per_block / gib:.0f}GiB total, "
                  f"per-pool cap {cd['nb_pool_cap']}, dram cap {cd['nb_dram_cap']}, "
                  f"demand cap {cd['nb_demand']}, "
                  f"MemAvailable {cd['dram_avail_gib']:.0f}GiB)", flush=True)
        return shrunk

    Worker.determine_available_memory = determine_available_memory
    _PATCHED = True
    print("[locks] memplan: patched Worker.determine_available_memory "
          "(tier+summary+lease reserved out of the KV budget)", flush=True)
