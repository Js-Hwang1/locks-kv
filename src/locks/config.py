"""LocksConfig — the single source of truth for all LOCKS behaviour.

Replaces the ~50 scattered ``KVCOMP_*`` environment variables of the research
prototype. Built ONCE at plugin registration from the vLLM config plus an
optional ``LOCKS_CONFIG`` (a JSON/YAML file path or an inline JSON string), and
threaded explicitly to every component. Kernels receive plain scalars/constexprs
derived from this object and NEVER read the environment themselves.

Two build targets, one algorithm core (see ours_doc/LOCKS_DESIGN.md):

  * ``fast``   — K+V resident; decode-time page selection + sparse attention.
  * ``mem-v``  — V offloaded to a DRAM tier (default; the memory play).
  * ``mem-kv`` — K+V offloaded (adds the low-rank r8 selector). [P3]
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Literal, Optional

Variant = Literal["fast", "mem-v", "mem-kv"]
Table = Literal["kv-union", "q-head", "kv-shared"]


@dataclasses.dataclass(frozen=True)
class LocksConfig:
    # ---- variant --------------------------------------------------------- #
    # Default flipped mem-v -> fast 2026-07-21 (R2): the flagship arm is the
    # sane default when a config names no variant; mem is opt-in.
    variant: Variant = "fast"

    # ---- Stage A: decode-time page selection ----------------------------- #
    # Adaptive per-head nucleus coverage over EXACT page masses (flagship).
    coverage: float = 0.95
    # Alternative to coverage: a fixed per-step selected-page FRACTION (1-cr).
    # When set, overrides `coverage`.
    budget: Optional[float] = None
    # Fixed ABSOLUTE per-(layer,kv-head) selected-page budget b (in pages), to
    # match the baselines' fixed-B axis (B tokens = budget_pages * page). When set
    # (>0) it OVERRIDES `budget`/`coverage`: derive_page_params uses the static
    # b = min(budget_pages, n_selectable) instead of ceil(fraction * n_selectable).
    budget_pages: Optional[int] = None
    # Budget ALLOCATION across the (layer, kv-head) units: every unit gets
    # exactly b = ceil(budget * n_selectable) pages (the per-unit top-b radix of
    # select.py). This STATIC allocation is the only budget path. (A conserved-
    # total "dynamic" per-unit budget was measured to give ~0 task headroom over
    # static -- kept as a paper negative, code removed; see task #44.)
    table: Table = "kv-union"
    sink_pages: int = 1
    window_pages: int = 1
    # Use the r8 low-rank screen for selection instead of the full bf16 K scan.
    # Required for mem-kv (K is not resident); optional for the others.
    r8_screen: bool = False
    r8_rank: int = 8
    # ---- selection summary: rki4, the ONLY score ------------------------- #
    # The packed rank-8 int-4 page-projection summary scored by the S4
    # register-pipelined hand-CUDA kernel (ours_doc/RKI4_KERNELS.md): int4
    # COLUMN-major basis V (per-column bf16 scales) + int8 coeffs C (per-token
    # bf16 scales) + int8 mu -> exact per-head page-LSE estimate, streamed at
    # 5d + 10*page + 18 B per (page, kv-head): ~820 B at (d=128, page 16) =
    # ~9.4% of the 8 KB bf16 K+V page; ~978 B = ~6.0% at page 32.  CUDA-only
    # (no Triton reference exists).  Supported geometry (TORCH_CHECKed at
    # wiring): d in {64, 128, 256}, page in {16, 32, 64} (64 = six-slab
    # lane only; CUDA score/decode + AOS records stay {16,32}), rank any
    # integer in [1, page-1] (LOCKS_RKI4_RANK; CUDA kernel stays {2,4,8}),
    # V-basis bits in {2, 4, 8} (LOCKS_RKI4_IBITS; CUDA + AOS stay i4),
    # G <= 8 query heads per
    # kv head (the fused nrm+topb selector's bound; validate_geometry enforces
    # it at wiring time -- G and n_kv are runtime parameters).
    #
    # The quad / clse / r8 score modes and the `score` selector field were
    # DELETED 2026-07-22 (user directive: one maintained mainline, no dead code
    # behind flags).  There is nothing to choose between.
    # GQA combine over the group's per-head page scores S[g]:
    #   "nrm" (DEFAULT) = per-head-normalized mass sum, sum_g exp(S[g] -
    #     max_page S[g]).  The confirmed win (+5.8pt AIME, LongBench-safe):
    #     group-max was the WORST combine.  Graph-safe (fixed extra passes).
    #   "max" = the plain GQA group max (kept reachable for ablation).
    quad_combine: Literal["max", "nrm"] = "nrm"
    # rki4 score-kernel page-split (grid.z of the S4 scorer).  None (default)
    # = auto-size from the device SM count at launch (fills ~4 blocks/SM at
    # the step's (n_req x n_kv) base grid).  Resolved ONCE at plugin init
    # (this field, or the LOCKS_RKI4_Z env read in load()); the kernel never
    # reads the environment.
    rki4_zsplit: Optional[int] = None

    # ---- Stage B / tier (mem variants only) ------------------------------ #
    # Hot-buffer slots per (layer, kv-head). <= 0 (default) = AUTO working-set
    # sizing (Tier.recommend_hot_slots: S = hot_ratio x per-unit budget).
    # A fixed positive value (the old default 2048 = 4 GiB at Llama-8B geometry)
    # is honored verbatim. Pure VRAM<->fetch knob: ANY value stays bitwise
    # (a miss re-reads identical bytes from the pinned pool). AUTO is what makes
    # the C3 steady-state resident claim scale with the budget (AC-M2).
    hot_slots: int = 0
    # AUTO hot sizing ratio S/b (measured hit curve: 3.0 -> ~.91 cold-start /
    # ~.98 steady RULER reuse, 6.0 -> ~.98 cold-start). Default 6.0 = shipped
    # behavior; the rki4-MEM design point (LOCKS_MEM_RKI4.md) runs 3.0.
    # Pure VRAM<->fetch knob, bitwise-free. Ignored when hot_slots > 0.
    hot_ratio: float = 6.0
    v_blocks: Optional[int] = None  # V-staging pool blocks (None = auto-size)
    max_pool_gb: float = 256.0     # pinned host-pool cap
    # Realize the VRAM saving: patch the KV-cache spec to head_size_v=0 so the
    # engine block pool holds K ONLY (num_gpu_blocks ~doubles) and V lives in the
    # tier. Requires the K-only prefill path (single-shot; direct q/k/v). When
    # False the engine keeps K+V resident and the tier gets a byte-identical V
    # copy -- a scaffold that validates the tiered decode without the spec patch
    # (no memory saving). mem variants only.
    mem_k_only: bool = True

    # SUMMARY-ONLY ENGINE CACHE (P5 / spec patch v2, LOCKS_KERNEL_REDESIGN.md
    # 3.6): make K truly leave VRAM.  The engine block pool is re-keyed to
    # head_size = REC_HS so a page holds exactly ``n_kv`` x 384 B summary
    # RECORDS (not K); QuadState.rec ALIASES that engine buffer (the separate
    # (L,NB,384) arena disappears); the decode already reads K/V from the tier
    # (mem-kv); the summary BUILD reads K from the tier (staged k_pool | pinned
    # pool_k | lease) instead of a resident K half.  Net: the ~96 GiB K-only
    # engine cache collapses to a ~9 GiB record arena (~10.7x smaller engine
    # page -> NB expands, contract #2).  mem-kv + arena + Hopper geometry only;
    # requires offload_k (K in the tier).  Default OFF: the flag-OFF path is
    # byte-identical to P4.  Env: LOCKS_MEM_SUMMARY_CACHE=1 forces it on.
    mem_summary_cache: bool = False

    # ---- engine / kernels ------------------------------------------------ #
    # Use the hand-CUDA Hopper (sm_90a) kernels for the hot path (r8_score,
    # topb_select, sparse decode) instead of the Triton reference. The CUDA
    # r8_score is ~7x faster; the kernels build once via cpp_extension at first
    # eager warmup (before graph capture) and are bitwise-matched to the Triton
    # reference. Falls back to Triton per-kernel if a build fails.
    use_cuda: bool = True
    force_eager: bool = False      # documented escape hatch (never CUDA graphs)

    # --------------------------------------------------------------------- #
    @property
    def is_mem(self) -> bool:
        return self.variant in ("mem-v", "mem-kv")

    @property
    def offload_k(self) -> bool:
        return self.variant == "mem-kv"

    def __post_init__(self) -> None:
        if self.variant not in ("fast", "mem-v", "mem-kv"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.offload_k and not self.r8_screen:
            # mem-kv cannot scan resident K -> must use the r8 screen
            object.__setattr__(self, "r8_screen", True)
        if not (0.0 < self.coverage <= 1.0):
            raise ValueError(f"coverage must be in (0,1], got {self.coverage}")
        if self.budget is not None and not (0.0 < self.budget <= 1.0):
            raise ValueError(f"budget must be in (0,1], got {self.budget}")
        if self.rki4_zsplit is not None and not (
                1 <= self.rki4_zsplit <= 65535):
            raise ValueError(
                f"rki4_zsplit must be None (auto) or in [1, 65535] (CUDA "
                f"gridDim.z limit), got {self.rki4_zsplit}")
        if self.quad_combine not in ("max", "nrm"):
            raise ValueError(
                f"quad_combine must be 'max' or 'nrm', got {self.quad_combine!r}")
        if not (0.0 < float(self.hot_ratio) <= 64.0):
            raise ValueError(
                f"hot_ratio must be in (0, 64], got {self.hot_ratio}")
        if self.mem_summary_cache:
            # spec v2 needs: mem-kv (K in the tier, so nothing reads engine K),
            # the K-only spec path (the engine page is ours to re-key), and a
            # summary record arena (the records ARE the engine page content).
            if self.variant != "mem-kv":
                raise ValueError(
                    "mem_summary_cache requires variant='mem-kv' (K must live "
                    f"in the tier); got {self.variant!r}")
            if not self.mem_k_only:
                raise ValueError(
                    "mem_summary_cache requires mem_k_only=True (the engine "
                    "page is re-keyed to hold summary records)")

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def load(cls, overrides: Optional[dict] = None) -> "LocksConfig":
        """Build from the optional ``LOCKS_CONFIG`` (file path or inline JSON)
        plus explicit ``overrides``. This is the ONLY place LOCKS reads the
        environment for configuration."""
        data: dict = {}
        spec = os.environ.get("LOCKS_CONFIG")
        if spec:
            if os.path.isfile(spec):
                text = open(spec).read()
                if spec.endswith((".yaml", ".yml")):
                    import yaml
                    data = yaml.safe_load(text) or {}
                else:
                    data = json.loads(text)
            else:
                data = json.loads(spec)
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        # Single-knob env escape for the P5 summary-only engine cache (spec v2),
        # so the correctness gates + coordinator sweep can flip it without a JSON
        # rebuild (mirrors the LOCKS_REC arena escape).  Explicit LOCKS_CONFIG /
        # overrides win.
        _sc = os.environ.get("LOCKS_MEM_SUMMARY_CACHE")
        if _sc is not None and "mem_summary_cache" not in data:
            data["mem_summary_cache"] = _sc not in ("", "0", "false", "False")
        # rki4 zsplit escape (read ONCE here, at plugin init -- never per
        # call).  Explicit LOCKS_CONFIG / overrides win.
        _z = os.environ.get("LOCKS_RKI4_Z")
        if _z is not None and "rki4_zsplit" not in data:
            data["rki4_zsplit"] = int(_z)
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown LocksConfig keys: {sorted(unknown)}")
        return cls(**data)
