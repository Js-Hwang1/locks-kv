from __future__ import annotations
import dataclasses
from ..config import LocksConfig

_SPLIT = 128
_SLACK_BYTES = 512 << 20


@dataclasses.dataclass
class MemPlan:
    fixed_bytes: int
    overhead_per_block: int
    engine_block_bytes: int
    host_bytes_per_block: int
    detail: dict

    @property
    def f(self) -> float:
        return self.overhead_per_block / self.engine_block_bytes

    def shrink(self, avail: int) -> int:
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
    elem = 2
    max_model_len = mc.max_model_len
    max_reqs = sched.max_num_seqs
    max_pages = (max_model_len + page - 1) // page
    max_tokens = (
        max(
            int(getattr(sched, "max_num_batched_tokens", 0) or 0),
            int(getattr(comp, "max_cudagraph_capture_size", 0) or 0),
            max_reqs,
        )
        + 8
    )
    chunk_pages = (max_tokens + page - 1) // page
    kv_mult = 2 if cfg.offload_k else 1
    from ..selection.rki4_state import record_bytes, record_bytes_padded

    rec_bytes = record_bytes(d=d, page=page)
    rec_bytes_engine = record_bytes_padded(d=d, page=page, elem=elem)
    summary_cache = bool(getattr(cfg, "mem_summary_cache", False))
    if summary_cache:
        if cfg.offload_k and cfg.mem_k_only:
            if not (cfg.mem_k_only and cfg.offload_k):
                raise ValueError("Minimal engine handles require host-backed K and V")
            per_layer_block = page * elem
        else:
            per_layer_block = n_kv * rec_bytes_engine
    else:
        per_layer_block = page * n_kv * d * elem * (1 if cfg.mem_k_only else 2)
    engine_block = per_layer_block * L
    summary_per_block = L * n_kv * rec_bytes
    resid_per_block = L * n_kv * (4 + 1)
    staging_per_block = 8
    overhead_per_block = summary_per_block + resid_per_block + staging_per_block
    from ..tier.staging import Staging
    from ..tier.tier import Tier

    hot_slots = cfg.hot_slots
    if hot_slots is None or hot_slots <= 0:
        eff_budget = cfg.budget if cfg.budget is not None else 0.15
        hot_slots = Tier.recommend_hot_slots(
            eff_budget,
            max_pages,
            n_seqs=max_reqs,
            target_ratio=float(getattr(cfg, "hot_ratio", 6.0)),
            sink=cfg.sink_pages,
            win=cfg.window_pages,
            budget_pages=getattr(cfg, "budget_pages", None),
        )
    hot = L * n_kv * hot_slots * page * d * elem * kv_mult
    nv = (
        cfg.v_blocks
        if (cfg.v_blocks or 0) > 0
        else Staging.default_v_blocks(chunk_pages, max_reqs)
    )
    staging = L * nv * page * n_kv * d * elem * kv_mult
    lease = 2 * L * max_pages * page * n_kv * d * elem
    G = mc.get_num_attention_heads(pc) // max(n_kv, 1) or 1
    parts = max_reqs * n_kv * 128 * G * (4 + 4 + d * 4)
    scratch = max_reqs * n_kv * max_pages * (4 + 4 + 4 * G + 4 * 3 + 12) + (1 << 20)
    fixed = hot + staging + lease + parts + scratch + _SLACK_BYTES
    full_kv_layer_block = page * n_kv * d * elem
    host_per_block = full_kv_layer_block * L * kv_mult
    return MemPlan(
        fixed_bytes=int(fixed),
        overhead_per_block=int(overhead_per_block),
        engine_block_bytes=int(engine_block),
        host_bytes_per_block=int(host_per_block),
        detail=dict(
            hot=hot,
            staging=staging,
            lease=lease,
            parts=parts,
            scratch=scratch,
            slack=_SLACK_BYTES,
            rec_bytes=rec_bytes,
            hot_slots=hot_slots,
            nv=nv,
            chunk_pages=chunk_pages,
            L=L,
            n_kv=n_kv,
            d=d,
            page=page,
            max_pages=max_pages,
            max_reqs=max_reqs,
        ),
    )


def _mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


_DRAM_SLACK_BYTES = 32 << 30


def host_capped_num_blocks(
    plan: "MemPlan", cfg: LocksConfig, nb_vram: int
) -> tuple[int, dict]:
    gib = 1 << 30
    host_pb = max(1, plan.host_bytes_per_block)
    kv_mult = 2 if cfg.offload_k else 1
    per_pool_pb = max(1, host_pb // kv_mult)
    nb_pool_cap = int(cfg.max_pool_gb * gib // per_pool_pb)
    dram_budget = max(0, _mem_available_bytes() - _DRAM_SLACK_BYTES)
    nb_dram_cap = int(dram_budget // host_pb) if dram_budget else nb_vram
    d = plan.detail
    nb_demand = int(d["max_reqs"]) * int(d["max_pages"])
    nb_demand += 0 + 64
    nb_cap = min(nb_pool_cap, nb_dram_cap, nb_demand)
    detail = dict(
        nb_pool_cap=nb_pool_cap,
        nb_dram_cap=nb_dram_cap,
        nb_demand=nb_demand,
        dram_avail_gib=_mem_available_bytes() / gib,
        host_pb=host_pb,
        capped=nb_cap < nb_vram,
    )
    return (max(1, min(nb_vram, nb_cap)), detail)


_PATCHED = False


def patch_available_memory() -> None:
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
        print(
            f"[locks memplan] avail={avail / gib:.2f}GiB -> {shrunk / gib:.2f}GiB (fixed={plan.fixed_bytes / gib:.2f}GiB: hot={d['hot'] / gib:.2f} staging={d['staging'] / gib:.2f} lease={d['lease'] / gib:.2f} parts+scratch+slack={(d['parts'] + d['scratch'] + d['slack']) / gib:.2f}; per-block +{plan.f * 100:.1f}% [rec={d['rec_bytes']}B x {d['n_kv']}kv x {d['L']}L]); projected NB~{nb} host~{nb * plan.host_bytes_per_block / gib:.1f}GiB",
            flush=True,
        )
        (nb_cap, cd) = host_capped_num_blocks(plan, cfg, int(nb))
        if cd["capped"]:
            self.vllm_config.cache_config.num_gpu_blocks_override = nb_cap
            print(
                f"[locks memplan] host cap: NB {nb} -> {nb_cap} (num_gpu_blocks_override; host pool {nb_cap * plan.host_bytes_per_block / gib:.0f}GiB total, per-pool cap {cd['nb_pool_cap']}, dram cap {cd['nb_dram_cap']}, demand cap {cd['nb_demand']}, MemAvailable {cd['dram_avail_gib']:.0f}GiB)",
                flush=True,
            )
        return shrunk

    Worker.determine_available_memory = determine_available_memory
    _PATCHED = True
    print(
        "[locks] memplan: patched Worker.determine_available_memory (tier+summary+lease reserved out of the KV budget)",
        flush=True,
    )
