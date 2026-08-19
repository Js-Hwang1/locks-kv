"""R8i4State -- persistent buffers for the packed r8i4 page-projection score.

The r8i4 score mode (ours_doc/R8I4_KERNELS.md; ours_doc/PAPER_R8I4.md section
4) stores, per (page, kv-head), the rank-8 page-projection summary in the
PACKED layout the S4 register-pipelined CUDA kernel streams directly:

  pp_v4  (L, NB, n_kv, RNK, d*i/8) uint8  IBITS-bit basis V, COLUMN-major
                                        packed at 8//i values/byte (i4
                                        default: d/2 bytes/column, lo nibble
                                        = even d row), per-column
  pp_vs  (L, NB, n_kv, 8)        bf16   scales (absmax/7)
  pp_c8  (L, NB, n_kv, page, 8)  int8   coeffs C = U8*S8, per-token
  pp_cs  (L, NB, n_kv, page)     bf16   scales (absmax/127)
  pp_mu8 (L, NB, n_kv, d)        int8   page centroid mu, per-page
  pp_mus (L, NB, n_kv)           bf16   scale (absmax/127)
  pp_tag (L, NB, n_kv, TAGW)     bf16   content-tag invalidation, NaN-init

Footprint per (page, kv-head) at the r8/i4 flagship: 4d (v4) + 16 (vs) +
8*page (c8) + 2*page (cs) + d (mu8) + 2 (mus) = 5d + 10*page + 18 B; at the
d=128 flagship that is 818 B at page 16 (~9.4% of the 8 KB bf16 K+V page) /
978 B at page 32 (~6.0%), plus the 16 B tag. In general only the V term
moves with the (r, i) knobs: RNK*d*i/8 B (record_bytes is authoritative).

R8i4State MIRRORS :class:`R8State` (alloc-once / graph-safe / physical-block
keyed, one L-major slab per layer).  The per-step outputs / scratch (score,
page_table, page_cnt, n_pages, n_sel_hi, b_fix, budget64) are BYTE-IDENTICAL
to R8State's, so ``derive_page_params`` and ``topb_select`` are reused
unchanged.  The score kernel writes the PER-HEAD page LSE into ``score_h``
(R, n_kv, G, MP) -- the same tensor the quad/clse nrm-combine path consumes --
and the GQA combine (``_nrm_launch`` | group max) reduces it into ``score``
exactly where the quad/clse path does it.

Kernel geometry contract: d in {64, 128, 256} (template instantiations),
page in {16, 32, 64} (multiple-of-16 only -- vLLM FlashAttention requires it;
64 = six-slab lane only, the Hopper CUDA score/decode kernels and the AOS
record stay {16, 32}), G <= 8 (the fused
nrm+topb selector's asserted bound -- the score kernel's g-tile loop would
take G <= 16 but the only combine path does not), rank any integer in
[1, page-1] (knob contract; the Hopper CUDA kernel stays {2, 4, 8} via its
own #error).  n_kv, G, d and page are READ
from the engine geometry by the builder; the supported SET is asserted HERE,
at wiring time, so an unsupported model fails loudly at state alloc rather
than inside a kernel.
"""
from __future__ import annotations

import os

import torch

TAGW = 8   # bf16 leading key channels of the page-final token = content tag

# Summary rank knob (USER DIRECTIVE 2026-07-17, doc 19): all ranks share one
# build (top-RNK of the SAME page-gram eigh) and one kernel family. Env so
# the module constants (record layout below) resolve at import, matching
# _AOS; the kernel build adds -DRNK to match (r8i4_score_cuda._get).
# Domain = any integer >= 1 (knob contract, ruling 2026-07-30); the upper
# bound RNK < page (centered-gram rank ceiling) is enforced at geometry
# validation, where page is known. r=1 is the allowed near-degenerate edge
# (single basis column; scales lean on the clamp_min floors). The Hopper
# CUDA score kernel keeps its own RNK in {2,4,8} #error (loud, no fallback).
RNK = int(os.environ.get("LOCKS_R8I4_RANK", "8"))
assert RNK >= 1, f"LOCKS_R8I4_RANK must be a positive integer, got {RNK}"

# V-basis quant bit-width knob (knob contract P2, ruling 2026-07-30): the
# i-axis quantizes the V BASIS ONLY -- C (pp_c8), mu (pp_mu8) and the cs/mus/
# vs scales are i-invariant. Module-import env symmetric with LOCKS_R8I4_RANK
# so the record layout below resolves at import. i4 is the shipped path and
# must stay byte-identical (the regression anchor); i8 = no packing, i2 = 4
# values/byte. The Hopper CUDA score TU is int4-hardcoded and asserts i4 at
# build time (r8i4_score_cuda._get) -- loud, never a silent wrong-width read.
IBITS = int(os.environ.get("LOCKS_R8I4_IBITS", "4"))
assert IBITS in (2, 4, 8), f"LOCKS_R8I4_IBITS must be 2|4|8, got {IBITS}"
# Per-column quant range: QMAX = 2^(i-1)-1, QMIN = -2^(i-1) (asymmetric
# signed, the shipped i4 -8..7 generalized). Single source for build + twin.
QMAX = (1 << (IBITS - 1)) - 1
QMIN = -(1 << (IBITS - 1))

# MUC (doc 25/25b): mu rides the mma's dead basis columns at RNK<8.
# Default ON for rank<8 (4/4 bitwise gates + e2e composition verified,
# 2026-07-17); LOCKS_R8I4_MUC=0 is the kill switch. r8 stage 2 pending.
# rank v2 (2026-07-27): MUC is AOS-only machinery (the kernel #error always
# enforced AOS+BIAS); now that rank<8 also rides the SIX-SLAB lane (<512K),
# the default is ON only when the AOS build is requested -- otherwise the
# builder would permute mu for a kernel with no MUC read path (G8), the
# exact bug class of MUC_G4_RANK_BUG.
MUC = os.environ.get(
    "LOCKS_R8I4_MUC",
    "1" if (RNK < 8 and os.environ.get("R8I4_MMA_AOS")) else "0") == "1"
assert not (MUC and RNK >= 8), "MUC stage 1 covers RNK<8 only " \
    "(mu rides the mma's dead basis columns; none exist at RNK >= 8)"

# AoS record mode (R8I4_MMA_AOS): the six score-read components live in ONE
# 64B-aligned record per (page, kv-head) -- [v4 RNK*d/2 | mu 128 | c8
# page*RNK | cs 2*page | vs 2*RNK | mus 2 | pad] (832B at r8/p16/d128,
# 512B r4, 384B r2) -- exposed to the build as STRIDED VIEWS (the
# index_put writes land in record fields directly), and read whole by the
# BT-AOS score kernel (one pointer per page instead of six re-derivations;
# nsys TAUCCHK 2026-07-16: the six-array BT kernel is 89.9us/layer = 2.2x
# the dense kernel at the same page count). The tag keeps its OWN slab
# (build-only; it is never read by the score kernel). Offsets MUST mirror
# the kernel's RO_* formulas (r8i4_score_cuda).
_AOS = bool(os.environ.get("R8I4_MMA_AOS"))


def _rec_bytes_aos(d: int, page: int) -> int:
    """AOS record footprint: raw fields rounded UP to 64B (kernel RECB)."""
    raw = RNK * d // 2 + d + page * RNK + 2 * page + 2 * RNK + 2
    return (raw + 63) // 64 * 64


_REC_BYTES = _rec_bytes_aos(128, 16)


def _v_bytes(d: int) -> int:
    """V-basis slab bytes: RNK columns of d IBITS-bit values, packed at
    PPB = 8//IBITS values/byte. IBITS=4 reproduces the shipped RNK*d/2
    byte-for-byte (the i4 regression anchor); i8 -> RNK*d, i2 -> RNK*d/4.
    d*IBITS is byte-aligned for every supported (d, i) pair (asserted at
    alloc), so the ceil never rounds."""
    return RNK * ((d * IBITS + 7) // 8)


def record_bytes(d: int, page: int, tagw: int = TAGW) -> int:
    """Per-(page, kv-head) r8i4 slab footprint in bytes, INCLUDING the tag.

    Single source of truth for the mem spec-v2 engine page sizing
    (backend/register._summary_head_size) and the memplan NB-overhead
    accounting. Mirrors the R8i4State allocation exactly:
      vb RNK*d*i/8 (u8) + vs RNK*2 (bf16) + c8 page*RNK (i8) + cs page*2
      (bf16) + mu8 d (i8) + mus 2 (bf16) + tag tagw*2 (bf16)
    = 5d + 10*page + 18 + 2*tagw at the i4 default (834 B at d=128, page 16,
    tagw 8); only the V term moves with i.
    R8I4_MMA_AOS: the six components pad to one 832B record (+ the tag
    slab), so the footprint is 832 + 2*tagw (848 at tagw 8)."""
    if _AOS:
        assert d == 128 and page == 16, \
            "AOS records cover the p16 d128 flagship only"
        assert IBITS == 4, "AOS records cover the i4 flagship only"
        return _REC_BYTES + tagw * 2
    return _v_bytes(d) + RNK * 2 + page * RNK + page * 2 + d + 2 + tagw * 2


def record_bytes_padded(d: int, page: int, elem: int = 2,
                        tagw: int = TAGW) -> int:
    """``record_bytes`` rounded UP to a multiple of ``page * elem`` so the
    engine page (head_size = padded/(page*elem)) tiles it exactly. The engine
    record page is a PLACEHOLDER on stacks whose per-layer engine tensors are
    separate storages (this one): the live slabs stay in R8i4State; the
    fold-in (record-major aliasing) is the documented follow-up
    (LOCKS_MEM_R8I4.md section 1)."""
    rb = record_bytes(d, page, tagw)
    q = page * elem
    return (rb + q - 1) // q * q


class R8i4State:
    """Persistent r8i4 selection state + per-step derived params.

    Allocated once (metadata-builder init) from engine maxima; every tensor
    address is stable across decode steps (CUDA-graph capture requirement).
    ``zsplit`` is the score kernel's grid.z page-split, resolved ONCE from the
    config (LOCKS_R8I4_Z / r8i4_zsplit); ``None`` (default) auto-sizes from
    the device SM count at launch -- the kernel never reads the environment.
    """

    def __init__(self, device, *, num_layers: int, num_blocks: int,
                 n_kv: int, G: int, head_dim: int, page: int,
                 max_reqs: int, max_pages: int,
                 rank: int = RNK, budget: float = 0.1, budget_pages=None,
                 sink_pages: int = 1, window_pages: int = 1,
                 tagw: int = TAGW, combine: str = "nrm", zsplit=None):
        assert window_pages >= 1, "window_pages>=1: the partial tail must live " \
            "inside the always-attended window so the selectable region is exact"
        # ---- kernel geometry contract (the SUPPORTED set; see module doc) - #
        # Single source of truth = validate_geometry (also called by the
        # builder BEFORE its blanket alloc try/except, so an unsupported model
        # raises loudly at init instead of silently delegating to FullKV).
        self.validate_geometry(head_dim=head_dim, G=G, page=page, rank=rank)
        assert combine in ("max", "nrm")
        assert zsplit is None or zsplit >= 1, \
            f"zsplit must be None (auto) or >= 1, got {zsplit}"
        L, NB, D, MP, R = num_layers, num_blocks, head_dim, max_pages, max_reqs
        self.L, self.NB, self.n_kv, self.G, self.d = L, NB, n_kv, G, D
        self.page, self.r, self.tagw = page, RNK, tagw
        self.v_bits = IBITS      # V-basis quant width (the builder banner)
        self.max_reqs, self.max_pages = R, MP
        self.budget = float(budget)
        self.budget_abs = int(budget_pages or 0)   # >0 => fixed-B static budget
        self.sink_pages, self.window_pages = int(sink_pages), int(window_pages)
        self.combine = combine
        self.zsplit = int(zsplit) if zsplit else None   # None = auto-size
        self.device = device
        u8, i8, bf16 = torch.uint8, torch.int8, torch.bfloat16
        i32, f32, f64 = torch.int32, torch.float32, torch.float64

        # ---- packed r8i4 selector state (per layer, physical-block keyed) #
        if _AOS:
            # ONE record per (block, kv-head); the six components are
            # STRIDED VIEWS into it, so the build's index_put writes land in
            # record fields directly and the BT-AOS score kernel reads whole
            # records. Field offsets MUST mirror the kernel's RO_* formulas
            # (r8i4_score_cuda: MU = RNK*d/2, then C8/CS/VS/MUS packed).
            assert D == 128 and page == 16, \
                "AOS records cover the p16 d128 flagship only"
            assert IBITS == 4, "AOS records cover the i4 flagship only"
            o_mu = RNK * D // 2
            o_c8 = o_mu + D
            o_cs = o_c8 + page * RNK
            o_vs = o_cs + 2 * page
            o_mus = o_vs + 2 * RNK
            self.pp_rec = torch.zeros(L, NB, n_kv, _REC_BYTES, device=device,
                                      dtype=u8)
            r_ = self.pp_rec
            self.pp_v4 = r_[..., 0:o_mu].unflatten(-1, (RNK, D // 2))
            self.pp_mu8 = r_[..., o_mu:o_c8].view(i8)
            self.pp_c8 = r_[..., o_c8:o_cs].view(i8).unflatten(-1, (page, RNK))
            self.pp_cs = r_[..., o_cs:o_vs].view(bf16)
            self.pp_vs = r_[..., o_vs:o_mus].view(bf16)
            self.pp_mus = r_[..., o_mus:o_mus + 2].view(bf16).squeeze(-1)
            # SCREEN (doc 26): per-(page,kvh) coefficient norm
            # max_t(cs_t*||c8[t,:]||_1) in the pad bytes -- the certified
            # screen's error-radius ingredient. Zero on non-screen builds.
            self.pp_nrmC = r_[..., o_mus + 2:o_mus + 4].view(bf16).squeeze(-1)
        else:
            self.pp_rec = None
            # V basis at IBITS bits/value: per-column packed width d*i/8
            # (i4 -> d/2, the shipped layout; i8 -> d; i2 -> d/4).
            assert D * IBITS % 8 == 0, \
                f"V column d={D} x {IBITS} bits must be byte-aligned"
            self.pp_v4 = torch.zeros(L, NB, n_kv, RNK, D * IBITS // 8,
                                     device=device, dtype=u8)
            self.pp_vs = torch.zeros(L, NB, n_kv, RNK, device=device,
                                     dtype=bf16)
            self.pp_c8 = torch.zeros(L, NB, n_kv, page, RNK, device=device,
                                     dtype=i8)
            self.pp_cs = torch.zeros(L, NB, n_kv, page, device=device,
                                     dtype=bf16)
            self.pp_mu8 = torch.zeros(L, NB, n_kv, D, device=device, dtype=i8)
            self.pp_mus = torch.zeros(L, NB, n_kv, device=device, dtype=bf16)
        # NaN-init tag: NaN != anything -> every page builds on first touch.
        self.pp_tag = torch.full((L, NB, n_kv, tagw), float("nan"),
                                 device=device, dtype=bf16)

        # ---- per-step outputs / scratch (request-row keyed) -------------- #
        # IDENTICAL to R8State so derive_page_params + topb_select are reused.
        self.score = torch.empty(R, n_kv, MP, device=device, dtype=f32)
        self.page_table = torch.empty(R, n_kv, MP, device=device, dtype=i32)
        self.page_cnt = torch.zeros(R, n_kv, device=device, dtype=i32)
        # (STALE_PREP per-layer tables removed 2026-07-21, wave 3b; see
        # ours_doc/REFUTED_ARMS_INDEX.md.)

        # ---- per-head kernel output + nrm-combine scratch ---------------- #
        # The S4 kernel always writes the per-head LSE here; combine="nrm"
        # reduces via the shared quad/clse _nrm_launch passes, combine="max"
        # via a fixed-shape amax into st.score.
        self.score_h = torch.empty(R, n_kv, G, MP, device=device, dtype=f32)
        self.hmax = (torch.zeros(R, n_kv, G, device=device, dtype=f32)
                     if combine == "nrm" else None)
        # NOROPE (round A): roped-q publish target -- the split kernels' q
        # source when the forward's rope kernel is deleted (8 KB at the
        # flagship; allocated unconditionally, written only under the flag).
        # _csc_cache = the rotary bf16 cos|sin table, wired by
        # register._wire_norope; None = legacy (q arrives roped).
        self.q_rope = torch.zeros(R, n_kv * G, D, device=device,
                                  dtype=torch.bfloat16)
        self._csc_cache = None

        # ---- per-request derived params (BatchedDecodeState-style) ------- #
        self.n_pages = torch.zeros(R, device=device, dtype=i32)
        self.n_sel_hi = torch.zeros(R, device=device, dtype=i32)
        self.b_fix = torch.ones(R, device=device, dtype=i32)
        self.budget64 = torch.tensor([self.budget], device=device, dtype=f64)

        # ---- fit invariant (checked, not assumed) ------------------------- #
        # The memplan boot reservation prices the slabs at record_bytes(); if
        # the allocation above ever drifts from that formula the mem-variant
        # boot silently under-reserves and the state alloc OOMs downstream
        # (the triple-confirmed memplan pricing bug). Verify equality here.
        planned = record_bytes(D, page, tagw) * n_kv * NB
        actual = self.bytes_per_layer()
        assert actual == planned, (
            f"r8i4 slab bytes/layer {actual} != record_bytes() plan "
            f"{planned} (d={D} page={page} tagw={tagw} n_kv={n_kv} NB={NB}) "
            "-- update r8i4_state.record_bytes to match the allocation")

    # --------------------------------------------------------------------- #
    @staticmethod
    def validate_geometry(*, head_dim: int, G: int, page: int,
                          rank: int = RNK) -> None:
        """The kernel geometry contract, callable WITHOUT allocating (the
        builder validates before its try/except so geometry violations are
        never swallowed into a silent FullKV delegation)."""
        assert head_dim in (64, 128, 256), \
            f"r8i4 supports d in {{64, 128, 256}}, model has d={head_dim}"
        # G <= 8, NOT 16 (2026-07-22, BATCHED_CO_DESIGN 7.6 / RENDER D3).  The
        # shipped fused nrm+topb selector asserts `st.G <= 8`
        # (r8i4_score_cuda.py: "fused nrm+topb supports G <= 8"; the G-generic
        # Triton `_nrm_launch` fallback was deleted with the quad removal at
        # 6052517).  Advertising G <= 16 let a G=16 model (Qwen3-235B 64q/4kv,
        # Llama-3.1-405B 128q/8kv at TP1) pass the boot contract, allocate
        # state, burn a full prefill, and assert on the FIRST decode step.
        # Fail at WIRING time instead, naming the real bound and the model.
        assert 1 <= G <= 8, (
            f"r8i4 supports 1 <= G <= 8 query heads per kv head, got G={G}. "
            "The fused nrm+topb selector (the only combine path) asserts "
            "G <= 8; a G>8 model boots and dies on the first decode step. "
            "Split the KV heads across more TP ranks to bring G <= 8, or "
            "extend the selector.")
        # page domain = multiples of 16 {16,32,64} (ruling 2026-07-30). page 64
        # rides the SIX-SLAB lane only: torch score twin (sm_120), the build
        # (page x page gram + eigh, page-parametric), the six-slab record
        # (page*RNK C-slab), the Triton decode (PAGE constexpr, tl.arange(0,PAGE)
        # -- 64 is pow2) and the nrm+topb select (page-count, not page-size).
        # The AOS record (r8i4_state.py:102/167) + Hopper CUDA score/decode
        # kernels still assert page in {16,32}; page=64 hits those loudly (never
        # a silent wrong path) until they gain a p64 case. page=8 is BLOCKED
        # UPSTREAM by vLLM FlashAttention (block_size must be a multiple of 16;
        # flash_attn.py:148) -- the engine cannot init at block_size 8, so it is
        # out of the supported domain regardless of the LOCKS lanes.
        assert page in (16, 32, 64), \
            f"r8i4 supports page in {{16, 32, 64}} (vLLM FlashAttention needs " \
            f"block_size a multiple of 16), engine block_size={page}"
        # Rank upper bound (knob contract, ruling 2026-07-30): the centered
        # page-gram dc@dc^T has rank <= page-1, so RNK >= page is DEGENERATE
        # (top-RNK includes ~0 eigenvalues; no compression). RNK is an
        # import-time constant but page is only known here -- the domain
        # check completes at geometry validation: 1 <= RNK < page.
        assert RNK < page, (
            f"r8i4 rank must satisfy 1 <= RNK < page (centered-gram rank "
            f"ceiling is page-1), got LOCKS_R8I4_RANK={RNK} at page={page}")
        assert rank == RNK, \
            f"r8i4 rank is fixed at import by LOCKS_R8I4_RANK={RNK} " \
            f"(the packed layout + kernel -DRNK), got rank={rank}"

    @classmethod
    def record_bytes(cls, d: int, page: int, tagw: int = TAGW) -> int:
        """Classmethod alias of :func:`record_bytes` (memplan's accessor)."""
        return record_bytes(d, page, tagw)

    def bytes_per_layer(self) -> int:
        if self.pp_rec is not None:
            return self.pp_rec[0].numel() + self.pp_tag[0].numel() * 2
        return (self.pp_v4[0].numel() + self.pp_vs[0].numel() * 2
                + self.pp_c8[0].numel() + self.pp_cs[0].numel() * 2
                + self.pp_mu8[0].numel() + self.pp_mus[0].numel() * 2
                + self.pp_tag[0].numel() * 2)

    def layer_state(self, layer: int):
        """Return the per-layer (v4, vs, c8, cs, mu8, mus, tag) views for
        ``layer`` (contiguous slabs; in-place writes propagate)."""
        return (self.pp_v4[layer], self.pp_vs[layer], self.pp_c8[layer],
                self.pp_cs[layer], self.pp_mu8[layer], self.pp_mus[layer],
                self.pp_tag[layer])
